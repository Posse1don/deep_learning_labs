import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

torch.manual_seed(1)
np.random.seed(1)
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
        return image, CLASSES.index(row["label"]), row["label"]

def main(data_dir="."):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_transform = T.Compose([
        T.RandomResizedCrop(160, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    
    train_dataset = IntelDataset("split.csv", part="train", transform=train_transform, data_dir=data_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=0)
    
    images, labels, str_labels = next(iter(train_loader))
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < 16:
            img = images[i].numpy().transpose((1, 2, 0))
            img = img * np.array(std) + np.array(mean)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(str_labels[i])
            ax.axis("off")
            
    plt.tight_layout()
    plt.savefig("batch_check.png")
    print("batch_check.png збережено")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=".", help="Шлях до набору даних")
    args = parser.parse_args()
    main(args.data)