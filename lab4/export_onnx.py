import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms as T
import onnx
import onnxruntime as ort
from sklearn.metrics import f1_score

EXPECTED_CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def extract_number(filename: str) -> int:
    return int(Path(filename).stem)


def resolve_part_dir(data_dir: Path, folder: str) -> Path:
    base = data_dir / folder
    if (base / folder).is_dir():
        base = base / folder
    return base


def verify_split(split_path: str) -> pd.DataFrame:
    df = pd.read_csv(split_path)
    print("=== Перевірка split.csv ===")
    for part, exp_count, exp_sum in [
        ("train", 1200, 11993084),
        ("test", 1502, 33351444),
    ]:
        sub = df[df["part"] == part]
        count = len(sub)
        num_sum = sum(extract_number(f) for f in sub["file"])
        ok = (count == exp_count) and (num_sum == exp_sum)
        print(f"{part:<5} | Кількість: {count} | Сума: {num_sum} | OK: {ok}")
        if not ok:
            raise ValueError(f"Розбіжність поділу для {part}!")
    return df


class IntelEvalDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, img_size: int, classes: list[str]):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.classes = classes
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = "seg_train" if row["part"] == "train" else "seg_test"
        base_dir = resolve_part_dir(self.data_dir, folder)
        img_path = base_dir / row["label"] / row["file"]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), self.classes.index(row["label"])


def create_ort_session(onnx_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])


def run_export_and_check(model_path: str, split_path: str, data_dir: str, out_onnx: str = "lab4_fp32.onnx") -> dict:
    df = verify_split(split_path)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = list(ckpt["classes"])
    assert classes == EXPECTED_CLASSES, f"Неправильний порядок класів: {classes}"
    img_size = int(ckpt["img_size"])

    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 6)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(f"\n=== Експорт у {out_onnx} ===")
    dummy_input = torch.randn(1, 3, img_size, img_size)
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        dummy_input,
        out_onnx,
        dynamo=False,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    export_time = time.perf_counter() - t0
    print(f"Час експорту: {export_time:.2f} с")

    onnx_model = onnx.load(out_onnx)
    n_conv = sum(1 for n in onnx_model.graph.node if n.op_type == "Conv")
    n_bn = sum(1 for n in onnx_model.graph.node if n.op_type == "BatchNormalization")
    print(f"Вузлів Conv: {n_conv} (очік. 20) | BatchNormalization: {n_bn} (очік. 0)")

    test_df = df[df["part"] == "test"].reset_index(drop=True)
    test_ds = IntelEvalDataset(test_df, data_dir, img_size, classes)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    sess = create_ort_session(out_onnx)
    pt_logits_list, ort_logits_list, labels_list = [], [], []

    print("\n=== Перевірка відповідності на test ===")
    with torch.no_grad():
        for images, labels in test_loader:
            pt_out = model(images).numpy()
            ort_out = sess.run(["logits"], {"input": images.numpy()})[0]
            pt_logits_list.append(pt_out)
            ort_logits_list.append(ort_out)
            labels_list.append(labels.numpy())

    pt_logits = np.concatenate(pt_logits_list, axis=0)
    ort_logits = np.concatenate(ort_logits_list, axis=0)
    y_true = np.concatenate(labels_list, axis=0)

    max_diff = float(np.max(np.abs(ort_logits - pt_logits)))
    pt_preds = pt_logits.argmax(axis=1)
    ort_preds = ort_logits.argmax(axis=1)
    agreement = float(np.mean(pt_preds == ort_preds))
    pt_f1 = float(f1_score(y_true, pt_preds, average="macro"))
    ort_f1 = float(f1_score(y_true, ort_preds, average="macro"))

    print(f"max |z_ONNX - z_PyTorch|: {max_diff:.6e} (має бути < 1e-3)")
    print(f"Збіг класів (argmax):     {agreement:.4f} (має бути 1.0000)")
    print(f"macro-F1 PyTorch:         {pt_f1:.4f}")
    print(f"macro-F1 ONNX FP32:       {ort_f1:.4f}")

    return {
        "export_time_sec": export_time,
        "n_conv": n_conv,
        "n_bn": n_bn,
        "max_logit_diff": max_diff,
        "argmax_agreement": agreement,
        "pytorch_macro_f1": pt_f1,
        "onnx_fp32_macro_f1": ort_f1,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    run_export_and_check(args.model, args.split, args.data)