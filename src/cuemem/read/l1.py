"""L1 快速召回（不调用 LLM）：多粒度混合检索 + 时间感知 + 打分重排（设计文档 §6.4）。

一次查询向量化之后，所有步骤都在本地完成：

1. **多粒度、多路召回**：轮、事实、会话摘要三种单元，各走 dense 与 sparse 两路（sparse 可选
   text-embedding-v4 的词项权重或 BM25），每路取前 pool 条；
2. **加权 RRF 融合到"轮"**：轮自己的两路排名直接参与；事实的两路排名按"键—值"映射成它的来源轮
   （事实增强键，N3）；会话摘要给它所在会话的轮加一个会话先验；
3. **时间感知**：问题里有时间表达时（"两周前""上周六"），落在该区间的轮（或来源事实的事件时间
   落在区间内的轮）加分；问"最近 / 现在"时新的优先，问"第一次 / 最早"时旧的优先；
4. 按 token 预算从高分到低分装入轮（不设预算时取前 k 轮）；事实单独按两路 RRF 排序，
   沿版本链补齐。

各项权重在开发集上用纯检索指标（不调用 reader）调好，见 scripts/tune_l1.py。
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from cuemem.core.models import CoreMemory, Fact, FactHit, Hit, MemoryUnit
from cuemem.core.timeutil import partial_date_start
from cuemem.core.trace import span
from cuemem.embed.base import TextEmbedder
from cuemem.read.bm25 import BM25
from cuemem.read.facts import choose_facts
from cuemem.read.query import QueryInfo, analyze
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore

SparseRoute = Literal["v4", "bm25", "none"]
KINDS = ("round", "fact", "summary")


@dataclass(frozen=True)
class L1Config:
    k: int = 10  # 放进上下文的轮数（设了 budget_tokens 时是上限）
    # 按 token 预算打包：按分数从高到低取轮，直到预算用完（至少 1 轮，最多 max_rounds 轮）。
    # 一"轮"的长度因数据集而异（LongMemEval 约 490 tokens，LoCoMo 的一句话约 40 tokens），
    # 固定条数会让不同数据集拿到的上下文量差十几倍；为空表示按条数（k）
    budget_tokens: int | None = 4000  # dev100 与 LoCoMo 上最稳的预算（D-029）
    max_rounds: int = 40
    fact_k: int = 10  # 放进上下文的事实条数（不含沿版本链补进来的）
    pool: int = 30  # 每一路召回的候选数
    sparse: SparseRoute = "bm25"
    # modality：先在 dense、sparse 各自内部按分数把轮与事实键取最大值，再在两种模态之间做 RRF；
    # kind：轮、事实、摘要各自排名后全部做 RRF（对照用：丢掉分数大小，效果明显更差）
    fusion: Literal["modality", "kind"] = "modality"
    rrf_k: int = 60
    sparse_weight: float = 0.5  # modality 融合时 sparse 模态的权重（dev100 网格搜索）
    round_weight: float = 1.0  # 轮自己的排名在融合里的权重
    fact_weight: float = 1.0  # 事实增强键的权重
    summary_weight: float = 0.0  # 会话摘要给所在会话的先验（dev100 上有害，默认关闭）
    time_weight: float = 1.0  # 落在问题时间区间内的加分（以"某一路排第 1"的分数为单位）
    order_weight: float = 0.0  # 问"最近 / 最早"时按新旧加分（dev100 上有害，默认关闭）
    chain: bool = True  # 事实沿版本链补齐


@dataclass
class L1Result:
    query: QueryInfo
    rounds: list[Hit]
    facts: list[FactHit]
    core: CoreMemory | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, object] = field(default_factory=dict)


class L1Retriever:
    def __init__(self, db: MemoryDB, vectors: VectorStore, embedder: TextEmbedder) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        # BM25 索引按空间缓存；单元数变化（写入了新内容）时重建
        self._bm25: dict[tuple[str, str], tuple[int, BM25, list[MemoryUnit]]] = {}

    def search(
        self, space_id: str, question: str, question_time: datetime, cfg: L1Config
    ) -> L1Result:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        query = analyze(question, question_time)
        with span("retrieve", route="l1", k=cfg.k, sparse=cfg.sparse) as record:
            batch = self.embedder.embed([question], "query")
            timings["embed"] = (time.perf_counter() - started) * 1000
            dense_vec = batch.dense[0]
            sparse_vec = batch.sparse[0] if batch.sparse is not None else None
            routes: dict[tuple[str, str], list[tuple[str, float]]] = {}
            for kind in KINDS:
                hits = self.vectors.search_dense(space_id, dense_vec, cfg.pool, kinds=[kind])
                routes[(kind, "dense")] = [(h.unit_id, h.score) for h in hits]
                if cfg.sparse == "v4" and sparse_vec is not None and len(sparse_vec) > 0:
                    hits = self.vectors.search_sparse(space_id, sparse_vec, cfg.pool, kinds=[kind])
                    routes[(kind, "sparse")] = [(h.unit_id, h.score) for h in hits]
                elif cfg.sparse == "bm25":
                    routes[(kind, "sparse")] = self._bm25_search(space_id, kind, question, cfg.pool)
            timings["recall"] = (time.perf_counter() - started) * 1000 - timings["embed"]

            rounds, facts, debug = self._fuse(space_id, query, routes, cfg)
            record["hits"] = [[h.unit.id, round(h.score, 5)] for h in rounds]
            record["facts"] = [[fh.fact.id, round(fh.score, 5)] for fh in facts]
            record["time_range"] = query.time_range.expr if query.time_range else None
        timings["total"] = (time.perf_counter() - started) * 1000
        return L1Result(
            query=query,
            rounds=rounds,
            facts=facts,
            core=self.db.get_core_memory(space_id),
            timings_ms=timings,
            debug=debug,
        )

    # ---------- 融合与重排 ----------

    def _fuse(
        self,
        space_id: str,
        query: QueryInfo,
        routes: dict[tuple[str, str], list[tuple[str, float]]],
        cfg: L1Config,
    ) -> tuple[list[Hit], list[FactHit], dict[str, object]]:
        unit_ids = list(dict.fromkeys(uid for hits in routes.values() for uid, _ in hits))
        units = {u.id: u for u in self.db.get_units(space_id, unit_ids)}
        fact_ids = [uid for uid in unit_ids if units.get(uid) and units[uid].kind == "fact"]
        facts = {f.id: f for f in self.db.get_facts(space_id, ids=fact_ids)}

        # 事实 → 来源轮：载入这些事实所在会话的全部轮
        sessions = sorted({f.session_id for f in facts.values()})
        round_units = {uid: u for uid, u in units.items() if u.kind == "round"}
        for unit in self.db.list_units(space_id, kinds=["round"], session_ids=sessions):
            round_units.setdefault(unit.id, unit)
        turn_to_round = {t: u.id for u in round_units.values() for t in u.source_turn_ids}
        fact_rounds = {
            fid: list(
                dict.fromkeys(turn_to_round[t] for t in f.source_turn_ids if t in turn_to_round)
            )
            for fid, f in facts.items()
        }

        unit_rank = 1.0 / (cfg.rrf_k + 1)  # 某一路排第 1 的分数，作为各项加分的单位
        round_score: dict[str, float] = defaultdict(float)
        for modality in ("dense", "sparse"):
            round_hits = [
                (u, x) for u, x in routes.get(("round", modality), []) if u in round_units
            ]
            fact_hits = routes.get(("fact", modality), [])
            if cfg.fusion == "kind":
                # 对照方案：轮与事实键各自排名，全部做 RRF
                for rank, (uid, _) in enumerate(round_hits, start=1):
                    round_score[uid] += cfg.round_weight / (cfg.rrf_k + rank)
                mapped: list[str] = []
                for fid, _ in fact_hits:
                    for rid in fact_rounds.get(fid, []):
                        if rid not in mapped:
                            mapped.append(rid)
                for rank, rid in enumerate(mapped, start=1):
                    round_score[rid] += cfg.fact_weight / (cfg.rrf_k + rank)
                continue
            # 默认方案：同一模态内按分数取最大值（轮自己的分数 vs 它来源事实的分数），
            # 保留"一条很强的直接命中"；sparse 分数在轮与事实两个索引里尺度不同，先各自归一化
            best: dict[str, float] = {}
            for uid, score in _normalized(round_hits, modality):
                best[uid] = max(best.get(uid, float("-inf")), cfg.round_weight * score)
            for fid, score in _normalized(fact_hits, modality):
                for rid in fact_rounds.get(fid, []):
                    best[rid] = max(best.get(rid, float("-inf")), cfg.fact_weight * score)
            weight = 1.0 if modality == "dense" else cfg.sparse_weight
            ranked_ids = sorted(best, key=lambda rid: (-best[rid], rid))
            for rank, rid in enumerate(ranked_ids, start=1):
                round_score[rid] += weight / (cfg.rrf_k + rank)

        # 会话摘要：给所在会话的轮加会话先验（只给已经进入候选的轮）
        session_score: dict[str, float] = defaultdict(float)
        for route in ("dense", "sparse"):
            for rank, (uid, _) in enumerate(routes.get(("summary", route), []), start=1):
                summary = units.get(uid)
                if summary is not None:
                    session_score[summary.session_id] += 1.0 / (cfg.rrf_k + rank)
        for rid in list(round_score):
            round_score[rid] += cfg.summary_weight * session_score.get(
                round_units[rid].session_id, 0.0
            )

        # 时间感知：落在问题时间区间内的轮加分；来源事实的事件时间落在区间内也算
        in_range: set[str] = set()
        if query.time_range is not None:
            for rid in round_score:
                if query.time_range.contains(round_units[rid].ts):
                    in_range.add(rid)
            for fid, fact in facts.items():
                moment = _fact_moment(fact)
                if moment is not None and query.time_range.contains(moment):
                    in_range.update(rid for rid in fact_rounds.get(fid, []) if rid in round_score)
            # 按区间宽度打折："上周六"全额加分，"过去两个月""去年"这种宽区间几乎不加——
            # 宽区间会覆盖大半历史，加分只会打乱排序
            specificity = min(1.0, 14 / max(1.0, query.time_range.days))
            for rid in in_range:
                round_score[rid] += cfg.time_weight * specificity * unit_rank

        # 问"最近 / 现在"时新的优先，问"第一次 / 最早"时旧的优先
        if (query.wants_latest or query.wants_first) and round_score:
            stamps = {rid: round_units[rid].ts.timestamp() for rid in round_score}
            low, high = min(stamps.values()), max(stamps.values())
            span_s = (high - low) or 1.0
            for rid, stamp in stamps.items():
                position = (stamp - low) / span_s
                bonus = position if query.wants_latest else 1.0 - position
                round_score[rid] += cfg.order_weight * unit_rank * bonus

        ranked_all = sorted(round_score.items(), key=lambda item: (-item[1], item[0]))
        if cfg.budget_tokens is None:
            ranked = ranked_all[: cfg.k]
        else:
            ranked, used = [], 0
            for rid, score in ranked_all[: cfg.max_rounds]:
                cost = estimate_tokens(round_units[rid].text)
                if ranked and used + cost > cfg.budget_tokens:
                    continue  # 放不下就跳过，后面更短的轮可能还放得下
                ranked.append((rid, score))
                used += cost
        rounds = [
            Hit(unit=round_units[rid], score=score, route="l1", rank=rank)
            for rank, (rid, score) in enumerate(ranked, start=1)
        ]

        # 事实：自己的两路 RRF，时间区间内的加分
        fact_score: dict[str, float] = defaultdict(float)
        for route in ("dense", "sparse"):
            for rank, (fid, _) in enumerate(routes.get(("fact", route), []), start=1):
                if fid in facts:
                    fact_score[fid] += 1.0 / (cfg.rrf_k + rank)
        if query.time_range is not None:
            specificity = min(1.0, 14 / max(1.0, query.time_range.days))
            for fid, fact in facts.items():
                moment = _fact_moment(fact)
                if moment is not None and query.time_range.contains(moment):
                    fact_score[fid] += cfg.time_weight * specificity * unit_rank
        chosen = choose_facts(
            self.db,
            space_id,
            facts,
            dict(fact_score),
            cfg.fact_k,
            current_only=False,
            chain=cfg.chain,
        )
        debug: dict[str, object] = {"in_time_range": len(in_range), "candidates": len(round_score)}
        return rounds, chosen, debug

    def _bm25_search(self, space_id: str, kind: str, query: str, k: int) -> list[tuple[str, float]]:
        units = self.db.list_units(space_id, kinds=[kind])
        cached = self._bm25.get((space_id, kind))
        if cached is None or cached[0] != len(units):
            cached = (len(units), BM25([u.text for u in units]), units)
            self._bm25[(space_id, kind)] = cached
        _, index, indexed = cached
        return [(indexed[i].id, score) for i, score in index.search(query, k)]


def _normalized(hits: list[tuple[str, float]], modality: str) -> list[tuple[str, float]]:
    """dense 的余弦相似度在各类单元间本来可比，原样返回；sparse 分数按本列表最大值归一化。"""
    if modality == "dense" or not hits:
        return hits
    top = max(score for _, score in hits) or 1.0
    return [(uid, score / top) for uid, score in hits]


def _fact_moment(fact: Fact) -> datetime | None:
    """事实发生的时间：有事件时间用事件时间的起点，否则没有（说出的时间由所在轮的时间代表）。"""
    if fact.valid_from:
        try:
            return partial_date_start(fact.valid_from)
        except ValueError:
            return None
    return None


def estimate_tokens(text: str) -> int:
    """粗略估计 token 数：汉字约 1 个 token 一个字，其他文字约 4 个字符一个 token。"""
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    return cjk + (len(text) - cjk) // 4 + 1
