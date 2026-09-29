"""从项目内已有、标注明确的高质量图片建立 AI 全图生成回归集。"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_quality import validate_image_bytes


SOURCE_DIR = ROOT / "测试数据"
OUTPUT_DIR = ROOT / "评测数据" / "AI全图生成_高质量本地样本_20260923"

SAMPLES = [
    ("千问AI生成.png", "AI生成图片", "通义千问生成图（项目本地样本）"),
    ("即梦AI生成.png", "AI生成图片", "即梦生成图（项目本地样本）"),
    ("文心AI生成.jpeg", "AI生成图片", "文心生成图（项目本地样本）"),
    ("混元AI生成.jpeg", "AI生成图片", "混元生成图（项目本地样本）"),
    ("讯飞AI生成.jpg3cp07", "AI生成图片", "讯飞生成图（项目本地样本）"),
    ("豆包AI生成.png", "AI生成图片", "豆包生成图（项目本地样本）"),
    ("生成小狗吃面图.png", "AI生成图片", "生成式 AI 图片（项目本地样本）"),
    ("真实图片1jpg.jpg", "真实图片", "真实相机图片（项目本地样本）"),
    ("真实图片2jpg.jpg", "真实图片", "真实相机图片（项目本地样本）"),
    ("真实图片3jpg.jpg", "真实图片", "真实相机图片（项目本地样本）"),
    ("真实图片4.jpg", "真实图片", "真实相机图片（项目本地样本）"),
]


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise SystemExit(f"输出目录非空：{OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counters = {"AI生成图片": 0, "真实图片": 0}
    rows = []
    for source_name, label, source in SAMPLES:
        source_path = SOURCE_DIR / source_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        quality = validate_image_bytes(source_path.read_bytes())
        if not quality["valid"]:
            raise ValueError(f"{source_name} 未通过质量校验：{quality['reason']}")
        counters[label] += 1
        folder = OUTPUT_DIR / label
        folder.mkdir(exist_ok=True)
        # 少数历史样本扩展名不规范；复制到评测集时统一使用可识别的扩展名。
        suffix = source_path.suffix.lower() if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else ".jpg"
        filename = f"{counters[label]:02d}{suffix}"
        shutil.copy2(source_path, folder / filename)
        rows.append({
            "split": label,
            "filename": filename,
            "expected_label": label,
            "source": source,
            "width": quality["width"],
            "height": quality["height"],
            "sharpness": quality["sharpness"],
        })

    with (OUTPUT_DIR / "labels.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"已建立 {len(rows)} 张高质量样本：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
