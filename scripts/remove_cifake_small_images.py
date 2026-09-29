"""移除已弃用的 CIFAKE 32×32 AI 全图生成验收样本，并同步更新标签清单。"""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_DIR = ROOT / "验收测试集"
REMOVED_SPLITS = {"01_AI真实_CIFAKE", "02_AI生成_CIFAKE"}
REMOVED_PATHS = [*(ACCEPTANCE_DIR / split for split in REMOVED_SPLITS), ROOT / "CIFAKE_test.parquet"]


def main() -> None:
    for path in REMOVED_PATHS:
        resolved = path.resolve()
        if ROOT not in resolved.parents:
            raise RuntimeError(f"拒绝处理工作区外路径：{resolved}")
        if not path.exists():
            raise FileNotFoundError(path)

    labels_path = ACCEPTANCE_DIR / "labels.csv"
    with labels_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
        fields = list(rows[0]) if rows else []
    kept_rows = [row for row in rows if row.get("split") not in REMOVED_SPLITS]
    removed_count = len(rows) - len(kept_rows)
    if removed_count != 200:
        raise RuntimeError(f"标签清单中预期移除 200 条 CIFAKE 记录，实际为 {removed_count} 条。")

    temp_path = labels_path.with_suffix(".csv.tmp")
    with temp_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept_rows)
    os.replace(temp_path, labels_path)

    for path in REMOVED_PATHS:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    print(f"已删除 {removed_count} 条 CIFAKE 标签、2 个图片目录和 CIFAKE_test.parquet。")


if __name__ == "__main__":
    main()
