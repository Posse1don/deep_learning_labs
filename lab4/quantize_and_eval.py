import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.metrics import f1_score
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from export_onnx import (
    EXPECTED_CLASSES,
    MEAN,
    STD,
    IntelEvalDataset,
    create_ort_session,
    extract_number,
    resolve_part_dir,
)

EXPECTED_CALIB_SUM = 1921470
EXPECTED_CALIB_COUNTS = {
    "buildings": 40,
    "forest": 35,
    "glacier": 36,
    "mountain": 33,
    "sea": 29,
    "street": 27,
}


def build_calibration_df(split_path: str) -> pd.DataFrame:
    df = pd.read_csv(split_path)
    train_df = df[df["part"] == "train"].sort_values("file", kind="mergesort").reset_index(drop=True)
    idx = np.random.default_rng(0).permutation(len(train_df))[:200]
    calib_df = train_df.iloc[idx].reset_index(drop=True)

    num_sum = sum(extract_number(f) for f in calib_df["file"])
    counts = calib_df["label"].value_counts().to_dict()
    counts_ordered = {c: int(counts.get(c, 0)) for c in EXPECTED_CLASSES}

    print("=== Контроль калібрувальної вибірки ===")
    print(f"Кількість: {len(calib_df)} (очік. 200) | Сума імен: {num_sum} (очік. {EXPECTED_CALIB_SUM})")
    print(f"По класах: {counts_ordered}")
    assert len(calib_df) == 200 and num_sum == EXPECTED_CALIB_SUM, "Помилка контрольної суми калібрування!"
    assert counts_ordered == EXPECTED_CALIB_COUNTS, f"Помилка розподілу класів: {counts_ordered}"
    return calib_df


class IntelCalibrationReader(CalibrationDataReader):
    def __init__(self, calib_df: pd.DataFrame, data_dir: str, img_size: int = 160, input_name: str = "input"):
        self.calib_df = calib_df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.input_name = input_name
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        self._iter = None
        self.rewind()

    def rewind(self):
        self._iter = iter(range(len(self.calib_df)))

    def get_next(self):
        idx = next(self._iter, None)
        if idx is None:
            return None
        row = self.calib_df.iloc[idx]
        base_dir = resolve_part_dir(self.data_dir, "seg_train")
        img = Image.open(base_dir / row["label"] / row["file"]).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).numpy().astype(np.float32)
        return {self.input_name: tensor}


def quantize_model(
    fp32_onnx: str,
    prep_onnx: str,
    out_int8_onnx: str,
    calib_df: pd.DataFrame,
    data_dir: str,
    per_channel: bool = True,
    img_size: int = 160,
) -> float:
    t0 = time.perf_counter()
    quant_pre_process(fp32_onnx, prep_onnx, skip_symbolic_shape=True)
    reader = IntelCalibrationReader(calib_df, data_dir, img_size=img_size, input_name="input")
    quantize_static(
        model_input=prep_onnx,
        model_output=out_int8_onnx,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )
    return time.perf_counter() - t0


def predict_onnx(onnx_path: str, test_loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
    sess = create_ort_session(onnx_path)
    preds_list, labels_list = [], []
    for images, labels in test_loader:
        logits = sess.run(["logits"], {"input": images.numpy()})[0]
        preds_list.append(logits.argmax(axis=1))
        labels_list.append(labels.numpy())
    return np.concatenate(preds_list), np.concatenate(labels_list)


def evaluate_variant(name: str, preds: np.ndarray, y_true: np.ndarray, fp32_preds: np.ndarray) -> dict:
    n = len(y_true)
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    acc = float(np.mean(preds == y_true))
    agreement = float(np.mean(preds == fp32_preds))

    b = int(np.sum((preds == y_true) & (fp32_preds != y_true)))
    c = int(np.sum((preds != y_true) & (fp32_preds == y_true)))
    delta_acc = (b - c) / n
    se_delta = float(np.sqrt(b + c) / n)
    significant = abs(b - c) > 2 * np.sqrt(b + c)

    if name == "FP32":
        verdict = "базовий варіант"
    elif not significant:
        verdict = "не відрізняється від шуму"
    else:
        direction = "краще за FP32" if b > c else "гірше за FP32"
        verdict = f"відрізняється від шуму ({direction})"

    print(
        f"{name:<16} | macro-F1: {macro_f1:.4f} | acc: {acc:.4f} | "
        f"згода: {agreement:.4f} | b={b}, c={c} | delta_acc={delta_acc:+.4f} +/- {se_delta:.4f} | {verdict}"
    )
    return {
        "variant": name,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "agreement_with_fp32": agreement,
        "b": b,
        "c": c,
        "delta_acc": delta_acc,
        "se_delta_acc": se_delta,
        "significant": bool(significant),
        "verdict": verdict,
    }


def main(split_path: str, data_dir: str, fp32_onnx: str = "lab4_fp32.onnx"):
    calib_df = build_calibration_df(split_path)

    print("\n=== Квантизація в INT8 (per_channel=True) ===")
    quant_time = quantize_model(
        fp32_onnx=fp32_onnx,
        prep_onnx="lab4_prep.onnx",
        out_int8_onnx="lab4_int8.onnx",
        calib_df=calib_df,
        data_dir=data_dir,
        per_channel=True,
    )
    print(f"Час підготовки та квантизації: {quant_time:.2f} с")

    df = pd.read_csv(split_path)
    test_df = df[df["part"] == "test"].reset_index(drop=True)
    test_ds = IntelEvalDataset(test_df, data_dir, img_size=160, classes=EXPECTED_CLASSES)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print("\n=== Оцінка якості на test 1502 зображення ===")
    fp32_preds, y_true = predict_onnx(fp32_onnx, test_loader)
    int8_preds, _ = predict_onnx("lab4_int8.onnx", test_loader)

    evaluate_variant("FP32", fp32_preds, y_true, fp32_preds)
    evaluate_variant("INT8", int8_preds, y_true, fp32_preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    main(args.split, args.data)