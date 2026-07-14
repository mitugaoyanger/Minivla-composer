"""文件读写辅助函数。"""

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    """用 UTF-8 保存 JSON 文件。"""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    """将字典列表保存为 CSV。"""
    rows = list(rows)
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
