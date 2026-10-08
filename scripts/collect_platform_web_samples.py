"""Collect curated public AI outputs without altering their image bytes.

Run from any directory; the source list includes provenance and expected hashes.
Only the project's upload quality rules determine acceptance, never detector scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from image_quality import MAX_IMAGE_BYTES, validate_image_bytes

EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "test_data/ai_platform_web_20261008/sources.json")
    parser.add_argument("--output", type=Path, default=ROOT / "test_data/ai_platform_web_20261008")
    parser.add_argument("--cache", type=Path, help="Optional local download cache")
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)

    def collect(item):
        item = dict(item)
        try:
            stem = item["sample_id"]
            candidates = list((args.output / item["platform"]).glob(stem + ".*"))
            if args.cache:
                candidates.extend(args.cache.glob(item["candidate_id"] + ".*"))
            data = next((p.read_bytes() for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == item["expected_sha256"]), None)
            if data is None:
                with requests.get(item["image_url"], stream=True, timeout=(15, 45)) as response:
                    response.raise_for_status()
                    data = bytearray()
                    for chunk in response.iter_content(65536):
                        data.extend(chunk)
                        if len(data) > MAX_IMAGE_BYTES:
                            raise ValueError("Image exceeds the project's 20 MB limit")
                data = bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            if digest != item["expected_sha256"]:
                raise ValueError("Source bytes have changed; review provenance before updating the hash")
            quality = validate_image_bytes(data)
            if not quality["valid"]:
                raise ValueError(quality["reason"])
            with Image.open(io.BytesIO(data)) as image:
                ext = EXTENSIONS.get(image.format)
                if ext is None:
                    raise ValueError("Unsupported upload format")
            relative = Path(item["platform"]) / (stem + ext)
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing output differs; refusing to overwrite it")
            if not destination.exists():
                destination.write_bytes(data)
            return {
                "file": relative.as_posix(), "sample_id": stem,
                "expected_label": "ai_generated", "task_type": item["task_type"],
                "platform": item["platform"], "generator": item["generator"],
                "subject": item["subject"], "provenance": item["provenance"],
                "source_page": item["source_page"], "image_url": item["image_url"],
                "evidence_url": item.get("evidence_url", item["source_page"]),
                "collected_date": item["collected_date"],
                "width": quality["width"], "height": quality["height"],
                "short_edge": quality["short_edge"], "bytes": len(data),
                "sharpness": quality["sharpness"], "sha256": digest,
                "quality_valid": True, "visual_review": "generated_output",
                "watermark_ground_truth": "unknown",
            }
        except Exception as exc:
            return {"sample_id": item["sample_id"], "error": str(exc)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(collect, sources))
    accepted = [r for r in results if "error" not in r]
    rejected = [r for r in results if "error" in r]
    if accepted:
        with (args.output / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(accepted[0]))
            writer.writeheader()
            writer.writerows(accepted)
    (args.output / "collection_report.json").write_text(json.dumps({
        "accepted": len(accepted), "by_platform": dict(Counter(r["platform"] for r in accepted)),
        "total_bytes": sum(r["bytes"] for r in accepted), "rejected": rejected,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "by_platform": dict(Counter(r["platform"] for r in accepted)), "rejected": rejected}, ensure_ascii=False))
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
