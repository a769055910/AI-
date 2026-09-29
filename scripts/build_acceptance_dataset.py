"""构建每组 100 张图像的批量检测验收集。

默认从 CIFAKE 官方测试集抽取 AI 真实 / AI 生成两组；其余数据在取得
FaceForensics++、Columbia 等官方数据集的使用授权后，可通过对应参数补齐。
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import pandas as pd


SEED = 20260819
SPLITS = {
    "ai_real": ("01_AI真实_CIFAKE", "真实图片", "CIFAKE", "ai_generated"),
    "ai_fake": ("02_AI生成_CIFAKE", "AI生成图片", "CIFAKE", "ai_generated"),
    "deepfake_real": ("03_换脸真实_DeepfakeFace", "真实人脸", "Deepfake Face Image Classification", "deepfake"),
    "deepfake_fake": ("04_换脸伪造_DeepfakeFace", "深度换脸", "Deepfake Face Image Classification", "deepfake"),
    "tamper_real": ("05_篡改真实_CASIA1", "真实图片", "CASIA 1.0", "tamper"),
    "tamper_fake": ("06_篡改伪造_CASIA1", "图像篡改", "CASIA 1.0", "tamper"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def clear_images(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES:
            file.unlink()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        file for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
    )


def copy_sample(source: Path, destination: Path, label: str, source_name: str, detector: str) -> list[dict[str, str]]:
    files = image_files(source)
    if len(files) < 100:
        raise ValueError(f"{source} 仅有 {len(files)} 张图像，至少需要 100 张。")
    selected = random.Random(SEED).sample(files, 100)
    clear_images(destination)
    rows = []
    for index, file in enumerate(selected, 1):
        filename = f"{index:03d}{file.suffix.lower()}"
        shutil.copy2(file, destination / filename)
        rows.append({
            "split": destination.name,
            "filename": filename,
            "expected_label": label,
            "detector_focus": detector,
            "source": source_name,
        })
    return rows


def export_cifake(parquet_path: Path, real_folder: Path, fake_folder: Path) -> list[dict[str, str]]:
    data = pd.read_parquet(parquet_path)
    rows = []
    # CIFAKE 的 ClassLabel 为 0=FAKE、1=REAL。
    for label_id, folder, label in ((1, real_folder, "真实图片"), (0, fake_folder, "AI生成图片")):
        subset = data[data["label"] == label_id]
        if len(subset) < 100:
            raise ValueError(f"CIFAKE 标签 {label_id} 样本不足 100 张。")
        selected = subset.sample(n=100, random_state=SEED).reset_index(drop=True)
        clear_images(folder)
        for index, item in selected.iterrows():
            filename = f"{index + 1:03d}.jpg"
            (folder / filename).write_bytes(item["image"]["bytes"])
            rows.append({
                "split": folder.name,
                "filename": filename,
                "expected_label": label,
                "detector_focus": "ai_generated",
                "source": "CIFAKE test split (Stable Diffusion v1.4 / CIFAR-10)",
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifake", type=Path, required=True, help="CIFAKE test parquet 文件")
    parser.add_argument("--output", type=Path, default=Path("验收测试集"))
    parser.add_argument("--deepfake-real", type=Path)
    parser.add_argument("--deepfake-fake", type=Path)
    parser.add_argument("--tamper-real", type=Path)
    parser.add_argument("--tamper-fake", type=Path)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    folders = {key: output / spec[0] for key, spec in SPLITS.items()}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    rows = export_cifake(args.cifake, folders["ai_real"], folders["ai_fake"])
    optional_sources = {
        "deepfake_real": args.deepfake_real,
        "deepfake_fake": args.deepfake_fake,
        "tamper_real": args.tamper_real,
        "tamper_fake": args.tamper_fake,
    }
    for key, source in optional_sources.items():
        if source:
            _, label, source_name, detector = SPLITS[key]
            rows.extend(copy_sample(source, folders[key], label, source_name, detector))

    with (output / "labels.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "filename", "expected_label", "detector_focus", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"已建立 {len(rows)} 张图像：{output.resolve()}")


if __name__ == "__main__":
    main()
