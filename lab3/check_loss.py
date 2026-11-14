import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import numpy as np
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
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, CLASSES.index(row["label"])

def set_train_mode(model, mode):
    model.train()
    if mode == "frozen":
        for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3, model.layer4]: m.eval()
    elif mode == "unfreeze":
        for m in [model.conv1, model.bn1, model.layer1, model.layer2, model.layer3]: m.eval()

def build_model(mode, seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if mode == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, 6)
    if mode == "frozen":
        for name, param in model.named_parameters():
            if "fc" not in name: param.requires_grad = False
    elif mode == "unfreeze":
        for name, param in model.named_parameters():
            if not ("fc" in name or "layer4" in name): param.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=1e-4)
    return model, opt

def main(data_dir="."):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_t = T.Compose([T.RandomResizedCrop(160, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    
    criterion = nn.CrossEntropyLoss()
    expected_L0 = np.log(6)
    print(f"Очікувана втрата ln(6) ≈ {expected_L0:.4f}\n")
    
    for mode in ["scratch", "frozen"]:
        model, _ = build_model(mode, seed=1)
        set_train_mode(model, mode)
        
        dataset = IntelDataset("split.csv", part="train", transform=train_t, data_dir=data_dir)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
        images, labels = next(iter(loader))
        
        with torch.no_grad():
            loss = criterion(model(images), labels)
            
        print(f"Режим {mode.upper()}:")
        print(f"Фактична втрата L0 = {loss.item():.4f}")
        print(f"Відхилення = {abs(loss.item() - expected_L0):.4f}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=".", help="Шлях до набору даних")
    args = parser.parse_args()
    main(args.data)