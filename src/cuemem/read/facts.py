"""事实增强检索（N3 的读取侧，即评测中的 B2F）：检索算法仍是 B2 的 dense top-k，
区别只在写入路径产出的结构被用上了：

1. **事实增强键**（LongMemEval 论文的 key expansion）：事实和"轮"一起参与向量检索，
   事实命中时把分数传给它的来源轮——"键"是事实，"值"仍是原始对话。一句被大段对话稀释的
   个人信息（"顺便说一下，我每周练两次瑜伽"）因此更容易被找到；
2. **带有效期的事实**：命中的事实连同时间与有效状态放进上下文，并沿版本链把新旧版本互相
   带出——问"现在"时看到最新值，问"以前"时也看得到被取代的旧值；
3. **核心记忆**：重要且当前有效的事实，常驻上下文。

N4 会在此基础上加入稀疏检索、RRF 融合、时间感知与重排。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from cuemem.core.models import CoreMemory, Fact, FactHit, Hit, MediaItem
from cuemem.core.timeutil import day, partial_date_start
from cuemem.core.trace import span
from cuemem.embed.base import TextEmbedder
from cuemem.read.query import TimeRange
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore

Validity = Literal["full", "current", "none"]
ImageMode = Literal["context", "isolated", "none"]

# 事实是压缩过的摘要：明确告诉 reader 细节以原话为准，避免它因为事实措辞与问题不完全一致而拒答
FACTS_HEADER = (
    "Notes from memory: short facts summarized from earlier conversations, with the date each "
    "was said. They may leave out details; the conversation excerpts below keep the original "
    'wording. "current" = still true; "no longer true" = changed in a later conversation; '
    '"corrected" = the user later corrected or refined it.'
)
FACTS_HEADER_PLAIN = (
    "Notes from memory: short facts summarized from earlier conversations, with the date each "
    "was said. They may leave out details; the conversation excerpts below keep the original "
    "wording."
)


@dataclass
class MemoryView:
    """一次检索给 reader 的全部材料。"""

    rounds: list[Hit]
    facts: list[FactHit]
    core: CoreMemory | None
    # 图片（N6）：装入的轮里的图片；按哪种描述展示（上下文感知 / 孤立 / 不展示描述）；
    # 要附上原图的图片 ID（再感知）
    media: dict[str, MediaItem] = field(default_factory=dict)
    image_mode: ImageMode = "context"
    attach: list[str] = field(default_factory=list)


class FactAugmentedRetriever:
    def __init__(self, db: MemoryDB, vectors: VectorStore, embedder: TextEmbedder) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder

    def search(
        self,
        space_id: str,
        query: str,
        *,
        k: int = 10,
        fact_k: int = 10,
        pool: int = 50,
        fact_keys: bool = True,
        current_only: bool = False,
        chain: bool = True,
    ) -> MemoryView:
        with span("retrieve", route="fact_augmented", k=k, fact_k=fact_k) as record:
            query_vec = self.embedder.embed([query], "query").dense[0]
            hits = self.vectors.search_dense(
                space_id, query_vec, max(pool, k + fact_k), kinds=["round", "fact"]
            )
            units = {u.id: u for u in self.db.get_units(space_id, [h.unit_id for h in hits])}
            round_score: dict[str, float] = {}
            fact_score: dict[str, float] = {}
            for h in hits:
                unit = units.get(h.unit_id)
                if unit is None:
                    continue
                target = round_score if unit.kind == "round" else fact_score
                target[unit.id] = max(target.get(unit.id, float("-inf")), h.score)
            facts = {f.id: f for f in self.db.get_facts(space_id, ids=list(fact_score))}
            round_units = {uid: u for uid, u in units.items() if u.kind == "round"}
            via_fact: set[str] = set()
            if fact_keys and facts:
                sessions = sorted({f.session_id for f in facts.values()})
                for unit in self.db.list_units(space_id, kinds=["round"], session_ids=sessions):
                    round_units.setdefault(unit.id, unit)
                turn_to_round = {t: u.id for u in round_units.values() for t in u.source_turn_ids}
                for fact in facts.values():
                    for turn_id in fact.source_turn_ids:
                        rid = turn_to_round.get(turn_id)
                        if rid is not None and fact_score[fact.id] > round_score.get(
                            rid, float("-inf")
                        ):
                            round_score[rid] = fact_score[fact.id]
                            via_fact.add(rid)
            ranked = sorted(round_score.items(), key=lambda item: (-item[1], item[0]))[:k]
            rounds = [
                Hit(
                    unit=round_units[rid],
                    score=score,
                    route="fact_key" if rid in via_fact else "dense",
                    rank=rank,
                )
                for rank, (rid, score) in enumerate(ranked, start=1)
            ]
            chosen = self._choose_facts(
                space_id, facts, fact_score, fact_k, current_only=current_only, chain=chain
            )
            record["hits"] = [[h.unit.id, round(h.score, 4), h.route] for h in rounds]
            record["facts"] = [[fh.fact.id, round(fh.score, 4), fh.via_chain] for fh in chosen]
        return MemoryView(rounds=rounds, facts=chosen, core=self.db.get_core_memory(space_id))

    def _choose_facts(
        self,
        space_id: str,
        facts: dict[str, Fact],
        scores: dict[str, float],
        fact_k: int,
        *,
        current_only: bool,
        chain: bool,
    ) -> list[FactHit]:
        return choose_facts(
            self.db, space_id, facts, scores, fact_k, current_only=current_only, chain=chain
        )


def choose_facts(
    db: MemoryDB,
    space_id: str,
    facts: dict[str, Fact],
    scores: dict[str, float],
    fact_k: int,
    *,
    current_only: bool,
    chain: bool,
) -> list[FactHit]:
    """按分数取前 fact_k 条事实。

    chain=True 时沿版本链补齐：命中旧版本带出取代它的新版本，命中新版本带出它取代的旧版本。
    """
    ordered = sorted(scores, key=lambda fid: (-scores[fid], fid))
    chosen = [
        FactHit(fact=facts[fid], score=scores[fid])
        for fid in ordered
        if fid in facts and (facts[fid].is_current or not current_only)
    ][:fact_k]
    if not chain or current_only:
        return chosen
    present = {fh.fact.id for fh in chosen}
    successors = [fh.fact.superseded_by for fh in chosen if fh.fact.superseded_by]
    linked = db.get_facts(space_id, ids=[i for i in successors if i not in present])
    linked += db.get_facts(space_id, superseded_by=sorted(present))
    for fact in linked:
        if fact.id not in present:
            present.add(fact.id)
            chosen.append(FactHit(fact=fact, score=scores.get(fact.id, 0.0), via_chain=True))
    return chosen


def format_facts(
    facts: Sequence[FactHit],
    *,
    validity: Validity = "full",
    question_time: datetime | None = None,
    highlight: TimeRange | None = None,
) -> tuple[str, dict[str, str]]:
    """按主语和时间排列事实并编号（f1、f2……），返回（上下文文本，编号 → 事实 ID）。

    validity="full" 标注有效状态并指向取代它的事实；"current" 只给当前事实（调用方已过滤），
    与 "none" 一样只标日期——这两种用于消融"有效期"机制的作用。
    highlight（L2）：事件时间（没有就用说出的时间）落在问题所指区间内的事实标上 ★。
    """
    ordered = sorted(facts, key=lambda fh: (fh.fact.subject, fh.fact.recorded_at, fh.fact.id))
    refs = {fh.fact.id: f"f{i}" for i, fh in enumerate(ordered, start=1)}
    lines = []
    for fh in ordered:
        fact = fh.fact
        notes = [f"said on {day(fact.recorded_at)}"]
        if question_time is not None:
            days = (question_time.date() - fact.recorded_at.date()).days
            notes[0] += f", {days} days before the question"
        if fact.valid_from and fact.valid_from != day(fact.recorded_at):
            notes.append(("happened " if fact.category == "event" else "since ") + fact.valid_from)
        if validity == "full":
            notes.append(fact_status(fact, refs))
        mark = " ★" if highlight is not None and highlight.contains(_moment(fact)) else ""
        lines.append(f"[{refs[fact.id]}] ({'; '.join(notes)}){mark} {fact.statement}")
    header = FACTS_HEADER if validity == "full" else FACTS_HEADER_PLAIN
    body = "\n".join(lines)
    return f"{header}\n{body}", {ref: fid for fid, ref in refs.items()}


def _moment(fact: Fact) -> datetime:
    """事实发生的时间：有事件时间用它的起点，否则用说出的时间。"""
    if fact.valid_from:
        try:
            return partial_date_start(fact.valid_from)
        except ValueError:
            pass
    return fact.recorded_at.replace(tzinfo=None)


def fact_status(fact: Fact, refs: dict[str, str]) -> str:
    """事实的有效状态：current，或何时起不再成立 / 何时被更正（refs 里有新版本时指向它）。"""
    if fact.is_current:
        return "current"
    successor = refs.get(fact.superseded_by or "")
    by = f", see [{successor}]" if successor else ""
    if fact.supersede_reason == "corrected":
        assert fact.invalidated_at is not None
        return f"corrected on {day(fact.invalidated_at)}{by}"
    since = f" from {fact.valid_to}" if fact.valid_to else ""
    return f"no longer true{since}{by}"
