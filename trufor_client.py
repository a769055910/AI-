"""Client for the TruFor inference endpoint on the GPU service.

The endpoint is deliberately separate from the legacy hand-crafted detectors:
an unavailable model must never be converted into a "real image" verdict.
"""
from __future__ import annotations

import os
from typing import Any

import requests


TRUFOR_API = os.environ.get("TRUFOR_API", os.environ.get("GPU_API", "http://127.0.0.1:6006").rstrip("/") + "/trufor_analyze")


def analyze_trufor(image_path: str, timeout: int = 180) -> dict[str, Any]:
    unavailable = {
        "available": False,
        "score": None,
        "reliability": None,
        "map_png": None,
        "confidence_png": None,
        "detail": "TruFor 篡改定位模型不可用；传统物理线索仅供辅助研判。",
        "method": "TruFor",
    }
    try:
        with open(image_path, "rb") as image_file:
            response = requests.post(
                TRUFOR_API,
                files={"file": (os.path.basename(image_path), image_file, "application/octet-stream")},
                timeout=timeout,
            )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", payload)
        if not isinstance(data, dict) or not data.get("available"):
            unavailable["detail"] = str((data or {}).get("detail", unavailable["detail"]))[:300]
            return unavailable
        score = float(data.get("score"))
        reliability = float(data.get("reliability", 0.0))
        return {
            "available": True,
            "score": round(max(0.0, min(1.0, score)), 4),
            "reliability": round(max(0.0, min(1.0, reliability)), 4),
            "map_png": data.get("map_png"),
            "confidence_png": data.get("confidence_png"),
            "map_coverage": float(data.get("map_coverage", 0.0) or 0.0),
            "detail": str(data.get("detail", "TruFor 推理完成。")),
            "method": "TruFor (RGB + Noiseprint++)",
        }
    except Exception as exc:
        unavailable["detail"] = f"TruFor 调用失败：{type(exc).__name__}"
        return unavailable
