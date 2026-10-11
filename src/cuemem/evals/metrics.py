"""评测指标：置信区间、分位数、检索召回与 NDCG。"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import numpy as np

from cuemem.core.models import Hit


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """对均值做 bootstrap，返回 (1 - alpha) 置信区间。固定种子，结果可复现。"""
    if not values:
        return float("nan"), float("nan")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = rng.choice(data, size=(n_resamples, data.size), replace=True).mean(axis=1)
    low, high = np.quantile(resampled, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def hit_turns(hit: Hit) -> set[tuple[str, int]]:
    """一条命中（一轮）覆盖的发言：（会话 ID，发言下标）。"""
    indices = hit.unit.meta.get("turn_indices", [])
    return {(hit.unit.session_id, int(i)) for i in indices}


def retrieval_metrics(
    hits: Sequence[Hit],
    *,
    evidence_sessions: Collection[str],
    evidence_turns: Collection[tuple[str, int]],
    relevant_units: int,
) -> dict[str, float]:
    """在给定命中列表上计算检索指标（命中列表本身就是前 k 条）。

    - session_recall_any：是否至少命中一个证据会话；
    - session_recall_all：证据会话被命中的比例；
    - turn_recall_any：是否至少有一条命中包含证据发言；
    - turn_ndcg：以"包含证据发言的轮"为相关项的 NDCG，relevant_units 为理想情况下的相关轮数。
    """
    sessions = set(evidence_sessions)
    turns = set(evidence_turns)
    hit_sessions = {h.unit.session_id for h in hits}
    relevant = [1.0 if hit_turns(h) & turns else 0.0 for h in hits]
    metrics: dict[str, float] = {}
    if sessions:
        metrics["session_recall_any"] = float(bool(hit_sessions & sessions))
        metrics["session_recall_all"] = len(hit_sessions & sessions) / len(sessions)
    if turns and relevant_units > 0:
        dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevant, start=1))
        ideal = sum(
            1 / math.log2(rank + 1) for rank in range(1, min(relevant_units, len(hits)) + 1)
        )
        metrics["turn_recall_any"] = float(any(relevant))
        metrics["turn_ndcg"] = dcg / ideal if ideal else 0.0
    return metrics
