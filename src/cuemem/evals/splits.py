"""开发集 / 测试集划分：按（题型，是否拒答）分层抽样，划分结果以题目 ID 列表的形式提交进仓库。"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from cuemem.config import PROJECT_ROOT
from cuemem.evals.datasets import Sample

SPLITS_DIR = PROJECT_ROOT / "evals" / "splits"


def stratified_ids(samples: Sequence[Sample], size: int, *, seed: int = 0) -> list[str]:
    """按（题型，是否拒答）分层，名额按比例分配（最大余数法），层内用固定种子随机抽取。"""
    if not 0 < size <= len(samples):
        raise ValueError(f"size 必须在 1 到 {len(samples)} 之间")
    groups: dict[tuple[str, bool], list[str]] = defaultdict(list)
    for s in samples:
        groups[(s.question_type, s.is_abstention)].append(s.sample_id)
    exact = {key: size * len(ids) / len(samples) for key, ids in groups.items()}
    quota = {key: int(value) for key, value in exact.items()}
    leftover = size - sum(quota.values())
    for key in sorted(exact, key=lambda k: (quota[k] - exact[k], k))[:leftover]:
        quota[key] += 1
    rng = random.Random(seed)
    chosen: list[str] = []
    for key in sorted(groups):
        ids = sorted(groups[key])
        rng.shuffle(ids)
        chosen.extend(ids[: quota[key]])
    return sorted(chosen)


def split_namespace(dataset: str) -> str:
    """划分按题目定义：MemLens 各档长度是同一批题，共用一套划分文件。"""
    return "memlens" if dataset.startswith("memlens") else dataset


def split_path(dataset: str, name: str) -> Path:
    return SPLITS_DIR / f"{split_namespace(dataset)}_{name}.json"


def save_split(dataset: str, name: str, ids: Sequence[str], *, seed: int, note: str = "") -> Path:
    path = split_path(dataset, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": split_namespace(dataset),
        "name": name,
        "seed": seed,
        "size": len(ids),
        "created": datetime.now().strftime("%Y-%m-%d"),
        "note": note,
        "ids": sorted(ids),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_split(dataset: str, name: str) -> list[str]:
    path = split_path(dataset, name)
    if not path.exists():
        raise FileNotFoundError(f"找不到划分文件 {path}，请先运行 cuemem eval split")
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["ids"])
