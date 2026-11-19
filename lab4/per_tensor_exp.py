import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import numpy as np
import pandas as pd
import torch
import onnx
from onnx import numpy_helper

from export_onnx import EXPECTED_CLASSES, IntelEvalDataset
from quantize_and_eval import (
    build_calibration_df,
    quantize_model,
    predict_onnx,
    evaluate_variant,
)


def analyze_conv_ranges(fp32_onnx: str) -> dict:
    model = onnx.load(fp32_onnx)
    initializers = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}

    conv_stats = []
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        weight_name = node.input[1]
        w = initializers[weight_name]  # (C_out, C_in, kH, kW)
        alpha_c = np.max(np.abs(w), axis=(1, 2, 3))
        max_alpha = float(np.max(alpha_c))
        med_alpha = float(np.median(alpha_c))
        r = max_alpha / med_alpha
        bits = float(np.log2(r))
        conv_stats.append({
            "node": node.name,
            "weight": weight_name,
            "shape": list(w.shape),
            "r": float(r),
            "bits_lost_median": bits,
        })

    conv_stats_sorted = sorted(conv_stats, key=lambda x: x["r"], reverse=True)
    median_r = float(np.median([x["r"] for x in conv_stats]))
    median_bits = float(np.log2(median_r))

    print("=== Діапазони каналів Conv у lab4_fp32.onnx ===")
    print(f"Усього згорток: {len(conv_stats)}")
    print("Топ-3 згортки за r = max(alpha_c) / median(alpha_c):")
    for item in conv_stats_sorted[:3]:
        print(f"  {item['node']:<45} | r = {item['r']:.2f} | log2(r) = {item['bits_lost_median']:.2f} біта")
    print(f"Медіана r за всіма {len(conv_stats)} згортками: {median_r:.2f} (log2(r) = {median_bits:.2f} біта)")

    return {
        "n_conv": len(conv_stats),
        "top3": conv_stats_sorted[:3],
        "median_r": median_r,
        "median_bits_lost": median_bits,
        "all_convs": conv_stats_sorted,
    }


def run_per_tensor_experiment(
    split_path: str,
    data_dir: str,
    fp32_onnx: str = "lab4_fp32.onnx",
    prep_onnx: str = "lab4_prep.onnx",
    out_onnx: str = "lab4_int8_per_tensor.onnx",
) -> dict:
    ranges_info = analyze_conv_ranges(fp32_onnx)

    calib_df = build_calibration_df(split_path)
    print("\n=== Квантизація в INT8 (per_channel=False) ===")
    quant_time = quantize_model(
        fp32_onnx=fp32_onnx,
        prep_onnx=prep_onnx,
        out_int8_onnx=out_onnx,
        calib_df=calib_df,
        data_dir=data_dir,
        per_channel=False,
    )
    print(f"Час квантизації (per-tensor): {quant_time:.2f} с")

    df = pd.read_csv(split_path)
    test_df = df[df["part"] == "test"].reset_index(drop=True)
    test_ds = IntelEvalDataset(test_df, data_dir, img_size=160, classes=EXPECTED_CLASSES)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print("\n=== Оцінка якості INT8 (per_channel=False) на test ===")
    fp32_preds, y_true = predict_onnx(fp32_onnx, test_loader)
    pt_int8_preds, _ = predict_onnx(out_onnx, test_loader)
    eval_res = evaluate_variant("INT8_per_tensor", pt_int8_preds, y_true, fp32_preds)

    return {
        "conv_ranges": ranges_info,
        "quant_time_sec": quant_time,
        "evaluation": eval_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    run_per_tensor_experiment(args.split, args.data)