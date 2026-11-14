import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
import copy
import pandas as pd
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

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

def build_model(mode, seed=1):
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

def print_params_info(mode, model, opt):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt_params_set = set(p for group in opt.param_groups for p in group['params'])
    frozen_in_opt = any(p in opt_params_set for p in model.parameters() if not p.requires_grad)
    
    print(f"\n[{mode.upper()}] Параметри:")
    print(f"Всього: {total} | Навчаються: {trainable}")
    print(f"Заморожені параметри в оптимізаторі: {'ТАК (ПОМИЛКА)' if frozen_in_opt else 'Ні (Правильно)'}")
    for i, group in enumerate(opt.param_groups):
        num_p = sum(p.numel() for p in group['params'])
        print(f" Група {i+1}: параметрів={num_p}, lr={group['lr']}")

def check_module_changes(prefix, old_state, new_state, mode):
    params_changed = False
    bn_changed = False
    has_bn = False
    for key in old_state.keys():
        if key.startswith(prefix + ".") or key == prefix:
            if "running" in key or "num_batches_tracked" in key:
                has_bn = True
                if not torch.equal(old_state[key], new_state[key]): bn_changed = True
            else:
                if not torch.equal(old_state[key], new_state[key]): params_changed = True
                
    bn_str = "так" if bn_changed else "ні"
    if not has_bn: bn_str = "немає BN"
    
    frozen_modules = []
    if mode == "frozen": frozen_modules = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    elif mode == "unfreeze": frozen_modules = ["conv1", "bn1", "layer1", "layer2", "layer3"]
    
    expected_params = "ні" if prefix in frozen_modules else "так"
    expected_bn = ("ні" if has_bn else "немає BN") if prefix in frozen_modules else ("так" if has_bn else "немає BN")
    
    p_ch = "так" if params_changed else "ні"
    match = "так" if (p_ch == expected_params and bn_str == expected_bn) else "ні"
    
    return p_ch, bn_str, match

def main(data_dir="."):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    t = T.Compose([T.RandomResizedCrop(160, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    
    for mode in ["scratch", "frozen", "unfreeze"]:
        m, o = build_model(mode)
        print_params_info(mode, m, o)

    torch.manual_seed(1)
    loader = torch.utils.data.DataLoader(IntelDataset("split.csv", "train", t, data_dir=data_dir), batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    modules_to_check = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]
    
    for mode in ["frozen", "unfreeze"]:
        print(f"\n--- Перевірка змін модулів у {mode.upper()} ---")
        model, opt = build_model(mode)
        old_state = copy.deepcopy(model.state_dict())
        
        for i, (images, labels) in enumerate(loader):
            if i >= 5: break
            set_train_mode(model, mode)
            opt.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            opt.step()
            
        new_state = model.state_dict()
        print(f"{'Модуль':<10} | {'Параметри':<15} | {'Ковзні BN':<15} | Збігається з очікуваним")
        for mod in modules_to_check:
            p_ch, bn_ch, match = check_module_changes(mod, old_state, new_state, mode)
            print(f"{mod:<10} | {p_ch:<15} | {bn_ch:<15} | {match}")

    print("\n--- Дослід з помилкою (FROZEN з model.train()) ---")
    model, opt = build_model("frozen")
    old_state = copy.deepcopy(model.state_dict())
    
    for i, (images, labels) in enumerate(loader):
        if i >= 5: break
        model.train() 
        opt.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        opt.step()
        
    new_state = model.state_dict()
    print(f"{'Модуль':<10} | {'Параметри':<15} | {'Ковзні BN':<15} | Збігається з очікуваним")
    for mod in modules_to_check:
        p_ch, bn_ch, match = check_module_changes(mod, old_state, new_state, "frozen")
        print(f"{mod:<10} | {p_ch:<15} | {bn_ch:<15} | {match}")

    max_mean = torch.max(torch.abs(old_state['layer1.0.bn1.running_mean'] - new_state['layer1.0.bn1.running_mean'])).item()
    max_var = torch.max(torch.abs(old_state['layer1.0.bn1.running_var'] - new_state['layer1.0.bn1.running_var'])).item()
    print(f"Макс. зміна running_mean: {max_mean}")
    print(f"Макс. зміна running_var: {max_var}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=".")
    args = parser.parse_args()
    main(args.data)