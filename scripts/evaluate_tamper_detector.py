"""Evaluate the local image-tampering detector on two labelled directories.

The script deliberately keeps the threshold fixed: use a separate development
set for threshold tuning, and reserve the supplied acceptance set for reporting.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tamper_detector import tamper_zero_model_analysis


def _analyse(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        # GPU fallback messages are operational logs, not benchmark results.
        with contextlib.redirect_stdout(io.StringIO()):
            result = tamper_zero_model_analysis(str(path))
        records.append(result)
    return records


def _summary(records: list[dict]) -> dict:
    def values(key: str) -> np.ndarray:
        return np.array([float(item.get(key, 0.0) or 0.0) for item in records])

    return {
        "count": len(records),
        "mean_tamper_score": round(float(values("tamper_score").mean()), 4),
        "median_tamper_score": round(float(np.median(values("tamper_score"))), 4),
        "fake_likely": sum(item.get("verdict") == "fake_likely" for item in records),
        "uncertain": sum(item.get("verdict") == "uncertain" for item in records),
        "copy_move_alerts": sum(
            item.get("copy_move", {}).get("verdict") == "fake_likely" for item in records
        ),
        "block_noise_alerts": sum(
            item.get("block_noise", {}).get("verdict") == "fake_likely" for item in records
        ),
        "multi_dim_alerts": sum(
            item.get("multi_dim", {}).get("verdict") == "fake_likely" for item in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--tampered", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all images")
    args = parser.parse_args()

    real_paths = sorted(args.real.glob("*.*"))
    tampered_paths = sorted(args.tampered.glob("*.*"))
    if args.limit:
        real_paths, tampered_paths = real_paths[: args.limit], tampered_paths[: args.limit]
    if not real_paths or not tampered_paths:
        raise SystemExit("Both input directories must contain images.")

    started = time.perf_counter()
    real, tampered = _analyse(real_paths), _analyse(tampered_paths)
    elapsed = time.perf_counter() - started

    # Current production threshold: tamper_score >= .60.
    fp = sum(item.get("tamper_score", 0.0) >= 0.60 for item in real)
    tp = sum(item.get("tamper_score", 0.0) >= 0.60 for item in tampered)
    result = {
        "threshold": 0.60,
        "real": _summary(real),
        "tampered": _summary(tampered),
        "false_positive_rate": round(fp / len(real), 4),
        "true_positive_rate": round(tp / len(tampered), 4),
        "balanced_accuracy": round(((len(real) - fp) / len(real) + tp / len(tampered)) / 2, 4),
        "elapsed_seconds": round(elapsed, 2),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
