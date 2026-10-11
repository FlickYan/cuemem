"""核心记忆（用户画像）：从事实表生成，不调用 LLM。

画像是"重要且当前有效的事实"的物化视图：事实一变就重新生成，因此画像永远不会和事实表矛盾，
也不会出现 LLM 改写时编造的内容。代价是行文不如 LLM 重写的段落流畅，但对 reader 来说，
逐条的事实陈述同样好用，而且每条都能追溯到来源（D-022）。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from cuemem.core.models import CoreMemory, Entity, Fact


def build_core_memory(
    space_id: str,
    facts: Sequence[Fact],
    people: Sequence[Entity],
    *,
    max_chars: int = 1600,
    min_importance: int = 4,
    now: datetime | None = None,
) -> CoreMemory:
    """选出参与者（用户本人或各位说话人）的重要、当前有效的事实，按重要度和新近程度排序，
    截断到 max_chars（约 400 tokens）。多位说话人时按人分组。"""
    person_ids = [p.id for p in people]
    names = {p.id: p.name for p in people}
    chosen = [
        f
        for f in facts
        if f.is_current and f.subject_id in names and f.importance >= min_importance
    ]
    chosen.sort(key=lambda f: (-f.importance, -f.recorded_at.timestamp(), f.id))
    lines: list[str] = []
    used: list[str] = []
    size = 0
    grouped = len({f.subject_id for f in chosen}) > 1
    by_person: dict[str, list[Fact]] = {pid: [] for pid in person_ids}
    for fact in chosen:
        line = f"- {fact.statement}"
        if size + len(line) + 1 > max_chars:
            continue
        size += len(line) + 1
        by_person[fact.subject_id].append(fact)
        used.append(fact.id)
    for pid in person_ids:
        if not by_person[pid]:
            continue
        if grouped:
            lines.append(f"{names[pid]}:")
        lines.extend(f"- {f.statement}" for f in by_person[pid])
    updated_at = now or max((f.recorded_at for f in chosen), default=datetime.now())
    return CoreMemory(
        space_id=space_id, text="\n".join(lines), fact_ids=used, updated_at=updated_at
    )
