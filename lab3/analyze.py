import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

class IntelTestDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_dir="."):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["part"] == "test"].reset_index(drop=True)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transform = T.Compose([T.Resize((160, 160)), T.ToTensor(), T.Normalize(mean, std)])
        self.data_dir = Path(data_dir)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        base_folder = self.data_dir / "seg_test"
        if (base_folder / "seg_test").is_dir(): 
            base_folder = base_folder / "seg_test"
            
        img_path = base_folder / row["label"] / row["file"]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), CLASSES.index(row["label"]), row["file"], str(img_path)

def verify_hash_and_get_runs():
    current_hash = hashlib.sha256(Path("split.csv").read_bytes()).hexdigest()
    runs_dir = Path("runs")
    for run_folder in sorted(runs_dir.iterdir()):
        if not run_folder.is_dir(): continue
        with open(run_folder / "run.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            if data["split_sha256"] != current_hash:
                print(f"Хеш split.csv у {run_folder.name} не збігається")
                sys.exit(1)
    print("Перевірка хешу успішна")

def plot_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"scratch": "red", "frozen": "blue", "unfreeze": "green"}
    linestyles = {1: "-", 2: "--", 3: ":"}
    
    runs_dir = Path("runs")
    for run_folder in sorted(runs_dir.iterdir()):
        if not run_folder.is_dir(): continue
        with open(run_folder / "run.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        mode, seed = data["mode"], data["seed"]
        label = f"{mode}_{seed}"
        axes[0].plot(range(1, 9), data["train_loss"], color=colors[mode], linestyle=linestyles.get(seed, "-"), alpha=0.7, label=label)
        axes[1].plot(range(1, 9), data["val_macro_f1"], color=colors[mode], linestyle=linestyles.get(seed, "-"), alpha=0.7, label=label)
        
    axes[0].set_title("Втрата на Train"); axes[0].set_xlabel("Епоха")
    axes[1].set_title("Macro-F1 на Validation"); axes[1].set_xlabel("Епоха")
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.savefig("curves.png")

def calculate_effects():
    runs_dir = Path("runs")
    results = {"scratch": [], "frozen": [], "unfreeze": []}
    scratch_curve = None

    print("\nТаблиця семи запусків:")
    print(f"{'#':<2} | {'Режим':<9} | {'Зерно':<5} | {'Втрата train, ep8':<18} | {'macro-F1 val, ep8':<18} | Час, с")
    n = 0
    for mode in ["scratch", "frozen", "unfreeze"]:
        for seed in ([1] if mode == "scratch" else [1, 2, 3]):
            n += 1
            with open(runs_dir / f"{mode}_seed{seed}" / "run.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            f1 = data["val_macro_f1"][-1]
            results[mode].append(f1)
            if mode == "scratch": scratch_curve = data["val_macro_f1"]
            print(f"{n:<2} | {mode:<9} | {seed:<5} | {data['train_loss'][-1]:<18.4f} | {f1:<18.4f} | {data['time_seconds']:.1f}")

    f_scratch1 = results["scratch"][0]
    f_fr, f_un = np.mean(results["frozen"]), np.mean(results["unfreeze"])
    r_fr = max(results["frozen"]) - min(results["frozen"])
    r_un = max(results["unfreeze"]) - min(results["unfreeze"])
    R = max(r_fr, r_un)
    delta_pre, delta_unf = f_fr - f_scratch1, f_un - f_fr

    def verdict(d):
        return "не відрізняється від розкиду (результату немає)" if abs(d) <= R else "ефект вартий перевірки, але не доведений"

    print("\nРозрахунок ефектів:")
    print(f"F_scratch,1 = {f_scratch1:.4f}")
    print(f"F_frozen = {f_fr:.4f}, R_frozen = {r_fr:.4f}")
    print(f"F_unfreeze = {f_un:.4f}, R_unfreeze = {r_un:.4f}")
    print(f"R = {R:.4f}")
    print(f"Delta_pre = {delta_pre:+.4f}: {verdict(delta_pre)}")
    print(f"Delta_unf = {delta_unf:+.4f}: {verdict(delta_unf)}")
    print(f"Delta_pre + Delta_unf = {delta_pre + delta_unf:+.4f}; F_unfreeze - F_scratch,1 = {f_un - f_scratch1:+.4f}")
    print("Крива scratch (macro-F1 val за епохами):", [round(v, 4) for v in scratch_curve])

    means = {m: np.mean(v) for m, v in results.items()}
    best_mode = max(means, key=means.get)
    best_seed = [1, 2, 3][int(np.argmax(results[best_mode]))] if best_mode != "scratch" else 1
    ckpt = torch.load("lab3_model.pt", map_location="cpu", weights_only=True)
    print(f"Середні F: { {m: round(float(v), 4) for m, v in means.items()} }")
    print(f"Вибір збігається з правилом: {(ckpt['regime'], ckpt['seed']) == (best_mode, best_seed)}")
    if abs(delta_unf) <= R:
        print("Увага: |Delta_unf| <= R, вибір між frozen і unfreeze визначив шум")

def evaluate_test(data_dir):
    checkpoint = torch.load("lab3_model.pt", map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    dataset = IntelTestDataset("split.csv", data_dir=data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    all_preds, all_labels, all_probs, all_files, all_paths = [], [], [], [], []
    with torch.no_grad():
        for images, labels, files, paths in loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.numpy()); all_labels.extend(labels.numpy())
            all_files.extend(files); all_paths.extend(paths); all_probs.extend(probs.numpy())
            
    test_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    
    print("\nМатриця плутанини:")
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(6)))
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.tight_layout(); plt.savefig("confusion_matrix.png")
    
    n = len(all_labels)
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    with open(f"runs/{checkpoint['regime']}_seed{checkpoint['seed']}/run.json", "r", encoding="utf-8") as f:
        val_f1 = json.load(f)["val_macro_f1"][-1]
    print(f"\nmacro-F1 val (вибраний) = {val_f1:.4f}, test = {test_f1:.4f}, різниця = {val_f1 - test_f1:+.4f}; SE = {se:.4f}")
    
    errors = [{"true": all_labels[i], "pred": all_preds[i], "prob": all_probs[i][all_preds[i]], "file": all_files[i], "path": all_paths[i]} 
              for i in range(len(all_labels)) if all_labels[i] != all_preds[i]]
    pair_dict = defaultdict(list)
    for e in errors: pair_dict[(e["true"], e["pred"])].append(e)
    for k in pair_dict: pair_dict[k].sort(key=lambda x: str(x["file"]))
        
    sorted_pairs = sorted(pair_dict.keys(), key=lambda k: (-len(pair_dict[k]), k[0], k[1]))
    top_5_errors = []
    idx = 0
    while len(top_5_errors) < 5 and sorted_pairs and idx < 1000:
        for p in sorted_pairs:
            if idx < len(pair_dict[p]): top_5_errors.append(pair_dict[p][idx])
            if len(top_5_errors) == 5: break
        idx += 1
        
    print("\n5 характерних помилок:")
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, err in enumerate(top_5_errors):
        true_name, pred_name = CLASSES[err["true"]], CLASSES[err["pred"]]
        count = len(pair_dict[(err["true"], err["pred"])])
        print(f"{err['file']} | {true_name} -> {pred_name} | p={err['prob']:.4f} | таких помилок у test: {count}")
        axes[i].imshow(Image.open(err["path"]).convert("RGB"))
        axes[i].set_title(f"T: {true_name}\nP: {pred_name}\np={err['prob']:.2f}", fontsize=9)
        axes[i].axis("off")
    plt.tight_layout(); plt.savefig("errors.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=".")
    args = parser.parse_args()
    
    verify_hash_and_get_runs()
    plot_curves()
    calculate_effects()
    evaluate_test(args.data)