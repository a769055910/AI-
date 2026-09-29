"""构建并评估 AI 全图生成检测的可复现小型基准集。

来源：
- CIFAKE：项目内已有的官方 test split（Stable Diffusion v1.4 / CIFAR-10）。
- Tiny-GenImage：GenImage 的公开 Hugging Face 小型镜像，仅用于补充跨生成器、
  较高分辨率的评测样本；原始 GenImage 由其官方项目维护。

该脚本不用于训练或调阈值。检测阶段复用 app.py 的 NPR、相机物证与融合逻辑，
并请求现有 GPU /detect 接口，因此结果对应页面中的“AI 全图生成”专项。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DATASET = "TheKernel01/Tiny-GenImage"
ROWS_API = "https://datasets-server.huggingface.co/rows"
GENERATOR_NAMES = {
    1: "ADM", 2: "BigGAN", 3: "GLIDE", 4: "Midjourney",
    5: "SD14", 6: "SD15", 7: "VQDM", 8: "Wukong",
}


def _extension(url: str) -> str:
    path = url.split("?", 1)[0].lower()
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        if path.endswith(suffix):
            return suffix
    return ".jpg"


def collect_tiny_genimage(output: Path, real_count: int, fake_per_generator: int, seed: int) -> None:
    """抽取平衡的真实图片和 8 种生成器的全图生成图片。"""
    if output.exists():
        if any(output.iterdir()):
            raise SystemExit(f"输出目录非空：{output}。请使用新的目录，避免覆盖既有样本。")
    else:
        output.mkdir(parents=True)
    quotas = {0: real_count, **{key: fake_per_generator for key in GENERATOR_NAMES}}
    selected: dict[int, list[dict]] = {key: [] for key in quotas}
    # 公开镜像的验证集按真实/各生成器交错组织；顺序读取优先避免预览接口限流。
    # seed 保留在命令行中，作为评测配置与后续扩展的可追溯信息。
    page_offsets = list(range(0, 7000, 100))

    for offset in page_offsets:
        params = {"dataset": DATASET, "config": "default", "split": "validation", "offset": offset, "length": 100}
        for attempt in range(3):
            response = requests.get(ROWS_API, params=params, timeout=40)
            if response.status_code != 429:
                response.raise_for_status()
                break
            if attempt == 2:
                response.raise_for_status()
            time.sleep(5 * (attempt + 1))
        for item in response.json().get("rows", []):
            row = item.get("row", {})
            label, generator = row.get("label"), row.get("generator")
            bucket = 0 if label == 0 else generator
            if bucket not in quotas or len(selected[bucket]) >= quotas[bucket]:
                continue
            image = row.get("image") or {}
            if image.get("src"):
                selected[bucket].append({"row_idx": item.get("row_idx"), "image": image, "label": label, "generator": generator})
        if all(len(selected[key]) >= quota for key, quota in quotas.items()):
            break

    missing = {key: quotas[key] - len(selected[key]) for key in quotas if len(selected[key]) < quotas[key]}
    if missing:
        raise RuntimeError(f"无法在公开预览接口中收集足量样本：{missing}")

    rows: list[dict[str, str]] = []
    for bucket, items in selected.items():
        category = "real" if bucket == 0 else "ai_generated"
        generator_name = "Real" if bucket == 0 else GENERATOR_NAMES[bucket]
        destination = output / category / generator_name
        destination.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(items, 1):
            url = item["image"]["src"]
            filename = f"{index:03d}{_extension(url)}"
            image_response = requests.get(url, timeout=60)
            image_response.raise_for_status()
            (destination / filename).write_bytes(image_response.content)
            rows.append({
                "split": str(destination.relative_to(output)),
                "filename": filename,
                "expected_label": "真实图片" if bucket == 0 else "AI生成图片",
                "generator": generator_name,
                "source": "Tiny-GenImage (public mirror of GenImage), validation split",
                "source_row": str(item["row_idx"]),
            })
            print(f"已收集 {category}/{generator_name}/{filename}")

    with (output / "labels.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "README.md").write_text(
        "# AI 全图生成补充评测集\n\n"
        "本集从 Tiny-GenImage 的公开镜像按固定随机种子抽取，不用于训练或调阈值。"
        "原始 GenImage 官方项目：https://github.com/GenImage-Dataset/GenImage\n",
        encoding="utf-8",
    )
    print(f"完成：{len(rows)} 张，标签清单：{output / 'labels.csv'}")


def read_samples(manifests: list[Path]) -> list[dict[str, str]]:
    samples = []
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                # 项目验收清单同时含换脸与篡改样本；AI 全图生成评测不能混入它们。
                if row.get("detector_focus") and row["detector_focus"] != "ai_generated":
                    continue
                image_path = manifest.parent / row["split"] / row["filename"]
                if image_path.is_file():
                    row["image_path"] = str(image_path)
                    row["manifest"] = str(manifest)
                    samples.append(row)
                else:
                    print(f"跳过不存在的样本：{image_path}")
    return samples


def prepare_evaluation_image(image_path: Path, cache_dir: Path, max_side: int) -> Path:
    """Keep the source untouched and cap only the evaluator's working copy.

    Full-resolution phone images are common in production, but several local
    forensic routines scale super-linearly on them.  A 1280px working copy is
    still well above the public upload quality floor (short edge >= 256px),
    while making this small regression benchmark reproducible.
    """
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        if max(width, height) <= max_side:
            return image_path
        scale = max_side / max(width, height)
        target = (round(width * scale), round(height * scale))
        image.thumbnail(target, Image.Resampling.LANCZOS)
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = cache_dir / f"{image_path.parent.name}_{image_path.stem}_{max_side}.jpg"
        if not destination.exists():
            image.convert("RGB").save(destination, "JPEG", quality=95, subsampling=0)
        return destination


def evaluate(manifests: list[Path], output: Path, gpu_api: str, max_side: int) -> None:
    # Import here so collection does not require the application's ML dependencies.
    from app import compute_combined_verdict, fuse_npr_with_physical_evidence
    from camera_forensics import analyze_camera_imaging
    from npr_detector import npr_analyze

    samples = read_samples(manifests)
    if not samples:
        raise SystemExit("没有可评测的图片样本。")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output.parent / f"{output.stem}_normalized_inputs"
    results = []
    for index, sample in enumerate(samples, 1):
        image_path = Path(sample["image_path"])
        evaluation_path = prepare_evaluation_image(image_path, cache_dir, max_side)
        try:
            with evaluation_path.open("rb") as image_file:
                response = requests.post(
                    f"{gpu_api.rstrip('/')}/detect",
                    files={"file": (evaluation_path.name, image_file, "application/octet-stream")},
                    data={"models": "ai_generated"},
                    timeout=300,
                )
            response.raise_for_status()
            payload = response.json()
            if payload.get("code") != 200:
                raise RuntimeError(payload.get("msg", str(payload)))
            npr = fuse_npr_with_physical_evidence(npr_analyze(str(evaluation_path)), analyze_camera_imaging(str(evaluation_path)))
            combined = compute_combined_verdict(npr, payload["data"].get("specialized_models"))
            result = {
                **sample,
                "evaluated_max_side": str(max_side),
                "final_score": f"{combined['final_score']:.4f}",
                "final_verdict": combined["final_verdict"],
                "fusion_mode": combined["fusion_mode"],
                "npr_score": f"{combined['npr_score']:.4f}",
                "univfd_score": combined["univfd_ai_score"],
                "aide_score": combined["aide_ai_score"],
                "dear_r_score": combined["dear_r_ai_score"],
                "probe_score": combined["probe_ai_score"],
                "error": "",
            }
        except Exception as exc:  # Preserve failed rows; do not silently inflate metrics.
            result = {**sample, "evaluated_max_side": str(max_side), "final_score": "", "final_verdict": "检测失败", "fusion_mode": "", "npr_score": "", "univfd_score": "", "aide_score": "", "dear_r_score": "", "probe_score": "", "error": str(exc)[:300]}
        results.append(result)
        if index == 1 or index == len(samples) or index % 10 == 0 or result["error"]:
            print(f"[{index}/{len(samples)}] {image_path.name}: {result['final_verdict']} {result['final_score']} {result['error']}")

    fields = list(results[0])
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    usable = [row for row in results if row["final_score"]]
    report = {"total": len(results), "usable": len(usable), "failed": len(results) - len(usable), "thresholds": {}}
    for threshold, name in ((0.60, "生产阈值（AI 全图生成）"),):
        tp = fp = tn = fn = 0
        for row in usable:
            predicted_ai = float(row["final_score"]) >= threshold
            actual_ai = row["expected_label"] == "AI生成图片"
            if predicted_ai and actual_ai: tp += 1
            elif predicted_ai: fp += 1
            elif actual_ai: fn += 1
            else: tn += 1
        total = tp + fp + tn + fn
        report["thresholds"][name] = {
            "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round((tp + tn) / total, 4) if total else None,
            "ai_recall": round(tp / (tp + fn), 4) if tp + fn else None,
            "ai_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "real_specificity": round(tn / (tn + fp), 4) if tn + fp else None,
        }
    report["by_source"] = dict(Counter(row.get("source", "") for row in usable))
    json_path = output.with_suffix(".summary.json")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"逐样本结果：{output}\n汇总：{json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect-tiny-genimage")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--real-count", type=int, default=48)
    collect.add_argument("--fake-per-generator", type=int, default=6)
    collect.add_argument("--seed", type=int, default=20260923)
    run = subparsers.add_parser("evaluate")
    run.add_argument("--manifest", type=Path, action="append", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--gpu-api", default="http://127.0.0.1:6006")
    run.add_argument("--max-side", type=int, default=1280, help="最长边超过该值时只缩放评测副本（默认：1280）")
    args = parser.parse_args()
    if args.command == "collect-tiny-genimage":
        collect_tiny_genimage(args.output, args.real_count, args.fake_per_generator, args.seed)
    else:
        evaluate(args.manifest, args.output, args.gpu_api, args.max_side)


if __name__ == "__main__":
    main()
