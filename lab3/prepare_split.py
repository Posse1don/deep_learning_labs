import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

def extract_number(filename):
    return int(filename.split('.')[0])

def main(data_dir="."):
    base_path = Path(data_dir)
    records = []
    
    train_files, val_files, test_files = set(), set(), set()

    train_dir = base_path / "seg_train"
    if (train_dir / "seg_train").is_dir(): train_dir = train_dir / "seg_train"

    rng_train = np.random.default_rng(0)
    for cls in CLASSES:
        cls_dir = train_dir / cls
        files = sorted([f.name for f in cls_dir.iterdir() if f.is_file()])
        idx = np.arange(len(files))
        rng_train.shuffle(idx)
        
        for i in idx[:200]:
            records.append({"file": files[i], "label": cls, "part": "train"})
            train_files.add(files[i])

    test_dir = base_path / "seg_test"
    if (test_dir / "seg_test").is_dir(): test_dir = test_dir / "seg_test"

    rng_val_test = np.random.default_rng(0)
    for cls in CLASSES:
        cls_dir = test_dir / cls
        files = sorted([f.name for f in cls_dir.iterdir() if f.is_file()])
        idx = np.arange(len(files))
        rng_val_test.shuffle(idx)
        
        cut = len(files) // 2
        for i in idx[:cut]:
            records.append({"file": files[i], "label": cls, "part": "validation"})
            val_files.add(files[i])
        for i in idx[cut:]:
            records.append({"file": files[i], "label": cls, "part": "test"})
            test_files.add(files[i])

    df = pd.DataFrame(records)
    df.to_csv("split.csv", index=False, lineterminator='\n')
    
    print("Контроль перетинів (має бути 0):")
    print(f"Train & Validation: {len(train_files & val_files)}")
    print(f"Train & Test: {len(train_files & test_files)}")
    print(f"Validation & Test: {len(val_files & test_files)}\n")

    print(f"{'Клас':<12} | {'train':<15} | {'validation':<15} | {'test':<15}")
    print("-" * 65)
    
    total_counts = {"train": 0, "validation": 0, "test": 0}
    total_sums = {"train": 0, "validation": 0, "test": 0}
    
    for cls in CLASSES:
        row_str = f"{cls:<12} | "
        cls_df = df[df["label"] == cls]
        for part in ["train", "validation", "test"]:
            part_df = cls_df[cls_df["part"] == part]
            count = len(part_df)
            num_sum = sum(extract_number(f) for f in part_df["file"])
            total_counts[part] += count
            total_sums[part] += num_sum
            row_str += f"{count}/{num_sum:<10} | "
        print(row_str)
        
    print("-" * 65)
    print(f"{'Разом':<12} | {total_counts['train']}/{total_sums['train']:<10} | "
          f"{total_counts['validation']}/{total_sums['validation']:<10} | "
          f"{total_counts['test']}/{total_sums['test']:<10}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=".", help="Шлях до набору даних")
    args = parser.parse_args()
    main(args.data)