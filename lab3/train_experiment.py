import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import time
import hashlib
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

class IntelDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, part, transform=None, data_dir="."):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["part"] == part].reset_index(drop=True)
        self.transform = transform
        self.data_dir = Path(data_dir)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = "seg_train" if row["part"] == "train" else "seg_test"
        
        base_folder = self.data_dir / folder
        if (base_folder / folder).is_dir(): 
            base_folder = base_folder / folder
            
        img_path = base_folder / row["label"] / row["file"]
        return self.transform(Image.open(img_path).convert("RGB")), CLASSES.index(row["label"])

def build_model(mode, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if mode == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, 6)
    
    if mode == "frozen":
        for name, param in model.named_parameters():
            if "fc" not in name: param.requires_grad = False
        opt = torch.optim.AdamW(model.fc.parameters(), lr=3e-3, weight_decay=1e-4)
    elif mode == "unfreeze":
        for name, param in model.named_parameters():
            if not ("fc" in name or "layer4" in name): param.requires_grad = False
        opt = torch.optim.AdamW([
            {"params": model.fc.parameters(), "lr": 3e-3},
            {"params": model.layer4.parameters(), "lr": 3e-4}
        ], weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    return model, opt

def set_train_mode(model, mode):
    model.train()
    if mode == "frozen":
        for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]: m.eval()
    elif mode == "unfreeze":
        for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]: m.eval()

def get_git_info():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
        return commit, bool(status)
    except Exception:
        return None, None

def run_training(mode, seed, data_dir, device):
    print(f"\n=== Запуск: {mode.upper()}, Seed: {seed} ===")
    
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_t = T.Compose([T.RandomResizedCrop(160, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    val_t = T.Compose([T.Resize((160, 160)), T.ToTensor(), T.Normalize(mean, std)])
    
    torch.manual_seed(1)
    train_loader = torch.utils.data.DataLoader(IntelDataset("split.csv", "train", train_t, data_dir), batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(IntelDataset("split.csv", "validation", val_t, data_dir), batch_size=64, shuffle=False, num_workers=0)
    
    model, opt = build_model(mode, seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    run_dir = Path(f"runs/{mode}_seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    history = {"train_loss": [], "val_macro_f1": []}
    start_time = time.time()
    
    for epoch in range(8):
        train_loss_sum, total_samples = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            set_train_mode(model, mode)
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * images.size(0)
            total_samples += images.size(0)
            
        epoch_loss = train_loss_sum / total_samples
        history["train_loss"].append(epoch_loss)
        
        model.eval()
        val_preds, val_targets, val_logits = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                val_logits.append(outputs.cpu().numpy())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels.numpy())
                
        val_logits = np.concatenate(val_logits)
        macro_f1 = f1_score(val_targets, val_preds, average="macro")
        history["val_macro_f1"].append(macro_f1)
        print(f"Епоха {epoch+1}/8 | Train Loss: {epoch_loss:.4f} | Val Macro-F1: {macro_f1:.4f}")
    
    train_time = time.time() - start_time
    np.save(run_dir / "val_logits.npy", val_logits)
    
    commit, uncommitted = get_git_info()
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    
    eval_modules = []
    if mode == "frozen": eval_modules = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    elif mode == "unfreeze": eval_modules = ["conv1", "bn1", "layer1", "layer2", "layer3"]
    
    run_info = {
        "mode": mode, "seed": seed,
        "commit": commit, "uncommitted_changes": uncommitted,
        "split_sha256": hashlib.sha256(Path("split.csv").read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "pytorch": torch.__version__, "torchvision": torchvision.__version__,
        "device": str(device), "device_name": device_name,
        "settings": {
            "train_transform": "RandomResizedCrop(160, scale=(0.7,1.0)), RandomHorizontalFlip, ToTensor, Normalize(ImageNet)",
            "eval_transform": "Resize((160,160)), ToTensor, Normalize(ImageNet)",
            "mean": mean, "std": std,
            "train_loader": {"batch_size": 32, "shuffle": True, "drop_last": False, "num_workers": 0},
            "eval_loader": {"batch_size": 64, "shuffle": False, "num_workers": 0},
            "epochs": 8, "loss": "CrossEntropyLoss(mean)", "optimizer": "AdamW", "weight_decay": 1e-4,
            "optimizer_defaults": list(opt.defaults.items()) if isinstance(opt.defaults, dict) else opt.defaults,
            "param_groups": [{"n_params": sum(p.numel() for p in g["params"]), "lr": g["lr"]} for g in opt.param_groups],
            "bn_momentum": model.bn1.momentum, "bn_eps": model.bn1.eps,
            "weights": None if mode == "scratch" else "IMAGENET1K_V1",
            "eval_modules": eval_modules
        },
        "train_loss": history["train_loss"], "val_macro_f1": history["val_macro_f1"],
        "time_seconds": train_time
    }
    with open(run_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4, ensure_ascii=False)
        
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    Path("weights_local").mkdir(exist_ok=True)
    torch.save(cpu_state, f"weights_local/{mode}_seed{seed}.pt")
    
    return macro_f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=".")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    experiments = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3), 
                   ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
    
    results = {}
    for mode, seed in experiments:
        results[(mode, seed)] = run_training(mode, seed, args.data, device)
        
    means = {m: np.mean([f for (mm, s), f in results.items() if mm == m]) for m in ["scratch", "frozen", "unfreeze"]}
    best_mode = max(means, key=means.get)
    best_seed = max([s for (m, s) in results.keys() if m == best_mode], key=lambda s: results[(best_mode, s)])
    
    state = torch.load(f"weights_local/{best_mode}_seed{best_seed}.pt", map_location="cpu", weights_only=True)
    torch.save({
        "state_dict": state, "regime": best_mode, "seed": best_seed,
        "classes": CLASSES, "img_size": 160
    }, "lab3_model.pt")
    
    print(f"\nЕксперименти завершено. Найкращий режим: {best_mode} (seed {best_seed})")

if __name__ == "__main__":
    main()