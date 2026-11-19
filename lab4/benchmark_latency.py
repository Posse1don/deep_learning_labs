import sys
sys.stdout.reconfigure(encoding="utf-8")

import time
import platform
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
import onnx
import onnxruntime as ort

from export_onnx import MEAN, STD, create_ort_session, resolve_part_dir


def get_cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return name.strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def get_env_info() -> dict:
    return {
        "cpu": get_cpu_name(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "threads": 1,
        "batch_size": 1,
    }


def prepare_single_input(split_path: str, data_dir: str, img_size: int = 160) -> np.ndarray:
    df = pd.read_csv(split_path)
    first_test = df[df["part"] == "test"].iloc[0]
    base_dir = resolve_part_dir(Path(data_dir), "seg_test")
    img_path = base_dir / first_test["label"] / first_test["file"]
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0).numpy().astype(np.float32)


def run_benchmark(
    split_path: str,
    data_dir: str,
    fp32_onnx: str = "lab4_fp32.onnx",
    int8_onnx: str = "lab4_int8.onnx",
    rounds: int = 3,
    warmup: int = 20,
    iters: int = 100,
) -> dict:
    x = prepare_single_input(split_path, data_dir, img_size=160)
    sess_fp32 = create_ort_session(fp32_onnx)
    sess_int8 = create_ort_session(int8_onnx)

    timings_ms = {"FP32": [], "INT8": []}
    round_p50 = {"FP32": [], "INT8": []}

    feed = {"input": x}
    outs = ["logits"]

    print(f"=== Замір затримки ({rounds} раунди x {iters} вимірів, прогрів {warmup}, B=1, 1 потік) ===")
    for r in range(1, rounds + 1):
        for name, sess in [("FP32", sess_fp32), ("INT8", sess_int8)]:
            for _ in range(warmup):
                sess.run(outs, feed)

            round_times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                sess.run(outs, feed)
                t1 = time.perf_counter()
                round_times.append((t1 - t0) * 1000.0)

            timings_ms[name].extend(round_times)
            r_p50 = float(np.median(round_times))
            round_p50[name].append(r_p50)
            print(f"Раунд {r} | {name:<4} | p50 = {r_p50:.2f} мс")

    fp32_size_mb = Path(fp32_onnx).stat().st_size / 1e6
    int8_size_mb = Path(int8_onnx).stat().st_size / 1e6

    fp32_p50 = float(np.percentile(timings_ms["FP32"], 50))
    fp32_p95 = float(np.percentile(timings_ms["FP32"], 95))
    int8_p50 = float(np.percentile(timings_ms["INT8"], 50))
    int8_p95 = float(np.percentile(timings_ms["INT8"], 95))

    speedup_p50 = fp32_p50 / int8_p50
    fp32_min, fp32_max = min(round_p50["FP32"]), max(round_p50["FP32"])
    int8_min, int8_max = min(round_p50["INT8"]), max(round_p50["INT8"])
    non_overlapping = (int8_max < fp32_min) or (fp32_max < int8_min)

    env = get_env_info()
    print("\n=== Підсумок затримки ===")
    print(f"FP32 | {fp32_size_mb:.2f} МБ | p50: {fp32_p50:.2f} мс | p95: {fp32_p95:.2f} мс | раунди: {[round(v, 2) for v in round_p50['FP32']]}")
    print(f"INT8 | {int8_size_mb:.2f} МБ | p50: {int8_p50:.2f} мс | p95: {int8_p95:.2f} мс | раунди: {[round(v, 2) for v in round_p50['INT8']]}")
    print(f"Прискорення (p50_FP32 / p50_INT8): {speedup_p50:.2f}x | Відрізки не перетинаються: {non_overlapping}")
    print(f"CPU: {env['cpu']} | OS: {env['os']}")

    return {
        "environment": env,
        "FP32": {
            "file_size_mb": fp32_size_mb,
            "p50_ms": fp32_p50,
            "p95_ms": fp32_p95,
            "round_p50_ms": round_p50["FP32"],
            "all_ms": timings_ms["FP32"],
        },
        "INT8": {
            "file_size_mb": int8_size_mb,
            "p50_ms": int8_p50,
            "p95_ms": int8_p95,
            "round_p50_ms": round_p50["INT8"],
            "all_ms": timings_ms["INT8"],
        },
        "speedup_p50": speedup_p50,
        "latency_change_measured": bool(non_overlapping),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()
    run_benchmark(args.split, args.data)