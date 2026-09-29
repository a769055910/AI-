"""Measure the user-facing single-image SSE detection flow."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "测试数据" / "真实图片1jpg.jpg"
REPORT = ROOT / "output" / "image_flow_benchmark.json"


def save(report):
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    started = time.perf_counter()
    report = {"image": str(IMAGE.relative_to(ROOT)), "events": []}
    with IMAGE.open("rb") as image_file:
        response = requests.post(
            "http://127.0.0.1:5000/api/detect/image/stream",
            files={"file": ("benchmark.jpg", image_file, "image/jpeg")},
            data=[("models", "ai_generated"), ("models", "deepfake"), ("models", "tamper")],
            stream=True,
            timeout=600,
        )
        response.raise_for_status()
        event_type = "message"
        for raw_line in response.iter_lines(decode_unicode=True):
            line = raw_line or ""
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                report["events"].append({
                    "at_seconds": round(time.perf_counter() - started, 3),
                    "event": event_type,
                    "step": payload.get("stepName"),
                    "percent": payload.get("percent"),
                    "message": payload.get("message"),
                    "label": payload.get("label"),
                })
                save(report)
    report["total_seconds"] = round(time.perf_counter() - started, 3)
    save(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
