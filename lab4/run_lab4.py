import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import argparse
from pathlib import Path

import pandas as pd
import torch

from export_onnx import EXPECTED_CLASSES, IntelEvalDataset, run_export_and_check
from quantize_and_eval import (
    build_calibration_df,
    evaluate_variant,
    predict_onnx,
    quantize_model,
)
from per_tensor_exp import run_per_tensor_experiment
from benchmark_latency import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="ЛР4")
    parser.add_argument("--model", type=str, required=True, help="Шлях до lab3_model.pt")
    parser.add_argument("--split", type=str, required=True, help="Шлях до split.csv")
    parser.add_argument("--data", type=str, required=True, help="Шлях до теки з seg_train/seg_test")
    parser.add_argument("--out", type=str, default="lab4_results.json", help="Файл результатів JSON")
    args = parser.parse_args()

    fp32_onnx = "lab4_fp32.onnx"
    prep_onnx = "lab4_prep.onnx"
    int8_onnx = "lab4_int8.onnx"
    int8_pt_onnx = "lab4_int8_per_tensor.onnx"

    step1 = run_export_and_check(args.model, args.split, args.data, out_onnx=fp32_onnx)

    calib_df = build_calibration_df(args.split)
    calib_counts = {c: int((calib_df["label"] == c).sum()) for c in EXPECTED_CLASSES}
    quant_time = quantize_model(
        fp32_onnx=fp32_onnx,
        prep_onnx=prep_onnx,
        out_int8_onnx=int8_onnx,
        calib_df=calib_df,
        data_dir=args.data,
        per_channel=True,
    )
    print(f"Час підготовки та квантизації (per-channel): {quant_time:.2f} с")

    df = pd.read_csv(args.split)
    test_df = df[df["part"] == "test"].reset_index(drop=True)
    test_ds = IntelEvalDataset(test_df, args.data, img_size=160, classes=EXPECTED_CLASSES)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print("\n=== Оцінка якості на test 1502 зображення ===")
    fp32_preds, y_true = predict_onnx(fp32_onnx, test_loader)
    int8_preds, _ = predict_onnx(int8_onnx, test_loader)
    eval_fp32 = evaluate_variant("FP32", fp32_preds, y_true, fp32_preds)
    eval_int8 = evaluate_variant("INT8", int8_preds, y_true, fp32_preds)

    step4 = run_per_tensor_experiment(
        split_path=args.split,
        data_dir=args.data,
        fp32_onnx=fp32_onnx,
        prep_onnx=prep_onnx,
        out_onnx=int8_pt_onnx,
    )

    step5 = run_benchmark(
        split_path=args.split,
        data_dir=args.data,
        fp32_onnx=fp32_onnx,
        int8_onnx=int8_onnx,
    )

    results = {
        "step1_export_and_verify": step1,
        "step2_calibration_and_quantization": {
            "calib_count": len(calib_df),
            "calib_sum": int(sum(int(Path(f).stem) for f in calib_df["file"])),
            "calib_counts_by_class": calib_counts,
            "quant_time_sec": quant_time,
        },
        "step3_quality": {
            "FP32": eval_fp32,
            "INT8": eval_int8,
        },
        "step4_per_tensor_experiment": step4,
        "step5_latency": step5,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nУсі результати збережено у {args.out}")


if __name__ == "__main__":
    main()