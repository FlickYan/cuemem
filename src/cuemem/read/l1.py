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

图片（N6）：图片描述也是一种"键"，命中时把分数交给图片所在的轮（与事实增强键相同）；
可选一路"以文搜图"（问题与图片在同一个图文向量空间里比较）。图片按相关度排序，供回答时
挑几张原图"再看一眼"。
打分（candidates）与打包（pack_rounds）分开：L2 对多个子查询分别打分、融合后再打包（read/l2.py）。

exclude（N8）：Agent 的工作记忆（当前会话最近几轮）已经原样放在上下文里，检索时排除这些轮与其中的
图片，召回的名额留给上下文里没有的记忆。
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from cuemem.core.models import CoreMemory, Fact, FactHit, Hit, MediaItem, MemoryUnit
from cuemem.core.timeutil import partial_date_start
from cuemem.core.trace import span
from cuemem.embed.base import FloatArray, TextEmbedder
from cuemem.embed.multimodal import ImageEmbedder
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
    # N5 复议后默认不用稀疏路（D-026 修订）：换用 qwen3.7 后 BM25 在 4k / 6k 预算下都降低召回，
    # 重排也救不回来；N4 的 B4 仍用 BM25（评测方法里显式指定）
    sparse: SparseRoute = "none"
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
    # 图片（N6）：用哪一种描述作检索键与证据——上下文感知（image）、孤立（image_iso），
    # 或不用描述（None：只把图片当作 "[image]" 占位，即纯文本记忆）
    image_kind: Literal["image", "image_iso"] | None = "image"
    image_weight: float = 1.0  # 图片描述键的权重
    # 以文搜图这一路在模态间 RRF 里的权重（0 = 不用）。MemLens dev100 上让关键图片所在轮的召回
    # 56% → 85%（D-036）；空间里没有图片时这一路自动跳过，不增加延迟与成本
    visual_weight: float = 0.5
    attach_images: int = 0  # 回答时附上几张原图（再感知）；0 = 只用描述


@dataclass
class L1Result:
    query: QueryInfo
    rounds: list[Hit]
    facts: list[FactHit]
    core: CoreMemory | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, object] = field(default_factory=dict)
    media: dict[str, MediaItem] = field(default_factory=dict)  # 装入的轮里的图片
    attach: list[str] = field(default_factory=list)  # 要附上原图的图片 ID（按相关度）


@dataclass
class L1Candidates:
    """一次查询的全部候选（打包之前）：按分数排好的轮，以及事实的分数。"""

    query: QueryInfo
    ranked: list[tuple[str, float]]  # 轮 ID 与分数，从高到低
    round_units: dict[str, MemoryUnit]
    facts: dict[str, Fact]
    fact_score: dict[str, float]
    fact_rounds: dict[str, list[str]]  # 事实 → 来源轮
    timings_ms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, object] = field(default_factory=dict)
    media_score: dict[str, float] = field(default_factory=dict)  # 图片 ID → 相关度（RRF 分）
    # 各路召回的原始结果（单元 ID 与分数，从高到低），键是（单元类型，模态）；探针从这里取分数分布
    routes: dict[tuple[str, str], list[tuple[str, float]]] = field(default_factory=dict)


class L1Retriever:
    def __init__(
        self,
        db: MemoryDB,
        vectors: VectorStore,
        embedder: TextEmbedder,
        *,
        image_embedder: ImageEmbedder | None = None,
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.image_embedder = image_embedder  # 以文搜图用；为空时没有这一路
        # BM25 索引按空间缓存；单元数变化（写入了新内容）时重建
        self._bm25: dict[tuple[str, str], tuple[int, BM25, list[MemoryUnit]]] = {}

    def search(
        self,
        space_id: str,
        question: str,
        question_time: datetime,
        cfg: L1Config,
        *,
        exclude: Collection[str] = (),
    ) -> L1Result:
        cands = self.candidates(space_id, question, question_time, cfg, exclude=exclude)
        return self.pack(space_id, cands, cfg)

    def pack(self, space_id: str, cands: L1Candidates, cfg: L1Config) -> L1Result:
        """把一次打分的候选按预算装入。控制器先用探针看过候选、决定走 L1 时，直接接着打包。"""
        started = time.perf_counter()
        with span("pack", route="l1", budget=cfg.budget_tokens) as record:
            rounds = pack_rounds(cands.ranked, cands.round_units, cfg)
            facts = choose_facts(
                self.db,
                space_id,
                cands.facts,
                cands.fact_score,
                cfg.fact_k,
                current_only=False,
                chain=cfg.chain,
            )
            record["hits"] = [[h.unit.id, round(h.score, 5)] for h in rounds]
            record["facts"] = [[fh.fact.id, round(fh.score, 5)] for fh in facts]
            media, attach = collect_media(
                self.db, space_id, rounds, cands.media_score, cfg.attach_images
            )
            if attach:
                record["attach"] = attach
        timings = dict(cands.timings_ms)
        timings["total"] += (time.perf_counter() - started) * 1000
        return L1Result(
            query=cands.query,
            rounds=rounds,
            facts=facts,
            core=self.db.get_core_memory(space_id),
            timings_ms=timings,
            debug=cands.debug,
            media=media,
            attach=attach,
        )

    def candidates(
        self,
        space_id: str,
        question: str,
        question_time: datetime,
        cfg: L1Config,
        *,
        query: QueryInfo | None = None,
        dense_vec: FloatArray | None = None,
        exclude: Collection[str] = (),
    ) -> L1Candidates:
        """一次查询的候选与分数（不打包）。

        query 可以传入改写过的问题分析（L2 用查询规划给出的时间区间替换规则解析的结果）；
        dense_vec 是已经算好的查询向量（L2 把几个子查询一次向量化）；
        exclude 是不参与检索的轮与图片的 ID（已经在上下文里的）。
        """
        timings: dict[str, float] = {}
        started = time.perf_counter()
        query = query or analyze(question, question_time)
        with span("retrieve", route="l1", k=cfg.k, sparse=cfg.sparse) as record:
            sparse_vec = None
            if dense_vec is None:
                batch = self.embedder.embed([question], "query")
                dense_vec = batch.dense[0]
                sparse_vec = batch.sparse[0] if batch.sparse is not None else None
            timings["embed"] = (time.perf_counter() - started) * 1000
            routes: dict[tuple[str, str], list[tuple[str, float]]] = {}
            kinds = [*KINDS, cfg.image_kind] if cfg.image_kind else list(KINDS)
            for kind in kinds:
                hits = self.vectors.search_dense(space_id, dense_vec, cfg.pool, kinds=[kind])
                routes[(kind, "dense")] = [(h.unit_id, h.score) for h in hits]
                if cfg.sparse == "v4" and sparse_vec is not None and len(sparse_vec) > 0:
                    hits = self.vectors.search_sparse(space_id, sparse_vec, cfg.pool, kinds=[kind])
                    routes[(kind, "sparse")] = [(h.unit_id, h.score) for h in hits]
                elif cfg.sparse == "bm25":
                    routes[(kind, "sparse")] = self._bm25_search(space_id, kind, question, cfg.pool)
            if (
                cfg.visual_weight > 0
                and self.image_embedder is not None
                and self._has_images(space_id)
            ):
                vector = self.image_embedder.embed_texts([question])[0]
                hits = self.vectors.search_images(space_id, vector, cfg.pool)
                routes[("media", "visual")] = [(h.unit_id, h.score) for h in hits]
            timings["recall"] = (time.perf_counter() - started) * 1000 - timings["embed"]
            skip = set(exclude)
            if skip:  # 打分前从各路召回里去掉；事实、图片映射回来的分数在打分后再去一遍
                routes = {
                    key: [(uid, score) for uid, score in hits if uid not in skip]
                    for key, hits in routes.items()
                }
                record["excluded"] = len(skip)

            cands = self._score(space_id, query, routes, cfg)
            cands.routes = routes
            if skip:
                cands.ranked = [(rid, score) for rid, score in cands.ranked if rid not in skip]
                cands.media_score = {
                    mid: score for mid, score in cands.media_score.items() if mid not in skip
                }
            record["candidates"] = len(cands.ranked)
            record["time_range"] = query.time_range.expr if query.time_range else None
        timings["total"] = (time.perf_counter() - started) * 1000
        cands.timings_ms = timings
        return cands

    # ---------- 融合与重排 ----------

    def _score(
        self,
        space_id: str,
        query: QueryInfo,
        routes: dict[tuple[str, str], list[tuple[str, float]]],
        cfg: L1Config,
    ) -> L1Candidates:
        unit_ids = list(
            dict.fromkeys(
                uid for (kind, _), hits in routes.items() if kind != "media" for uid, _ in hits
            )
        )
        units = {u.id: u for u in self.db.get_units(space_id, unit_ids)}
        fact_ids = [uid for uid in unit_ids if units.get(uid) and units[uid].kind == "fact"]
        facts = {f.id: f for f in self.db.get_facts(space_id, ids=fact_ids)}
        image_units = {uid: u for uid, u in units.items() if u.kind in ("image", "image_iso")}
        visual = routes.get(("media", "visual"), [])
        visual_media = {m.id: m for m in self.db.get_media(space_id, ids=[i for i, _ in visual])}

        # 事实、图片 → 来源轮：载入它们所在会话的全部轮
        sessions = sorted(
            {f.session_id for f in facts.values()} | {u.session_id for u in image_units.values()}
        )
        round_units = {uid: u for uid, u in units.items() if u.kind == "round"}
        for unit in self.db.list_units(space_id, kinds=["round"], session_ids=sessions):
            round_units.setdefault(unit.id, unit)
        if visual_media:  # 以文搜图命中的图片可能在任何会话里：按发言找到所在的轮
            wanted = sorted({m.source_turn_id for m in visual_media.values()})
            # 只取这几句发言（以前载入整个空间的原始日志，1.6 万轮时约 250 毫秒）
            turn_sessions = {t.id: t.session_id for t in self.db.get_turns(space_id, ids=wanted)}
            extra = sorted(set(turn_sessions.values()) - set(sessions))
            for unit in self.db.list_units(space_id, kinds=["round"], session_ids=extra):
                round_units.setdefault(unit.id, unit)
        turn_to_round = {t: u.id for u in round_units.values() for t in u.source_turn_ids}
        fact_rounds = {
            fid: list(
                dict.fromkeys(turn_to_round[t] for t in f.source_turn_ids if t in turn_to_round)
            )
            for fid, f in facts.items()
        }
        image_rounds = {
            uid: [turn_to_round[t] for t in u.source_turn_ids if t in turn_to_round]
            for uid, u in image_units.items()
        }

        unit_rank = 1.0 / (cfg.rrf_k + 1)  # 某一路排第 1 的分数，作为各项加分的单位
        round_score: dict[str, float] = defaultdict(float)
        for modality in ("dense", "sparse"):
            round_hits = [
                (u, x) for u, x in routes.get(("round", modality), []) if u in round_units
            ]
            fact_hits = routes.get(("fact", modality), [])
            image_hits = routes.get((cfg.image_kind, modality), []) if cfg.image_kind else []
            if cfg.fusion == "kind":
                # 对照方案：轮与事实键各自排名，全部做 RRF
                for rank, (uid, _) in enumerate(round_hits, start=1):
                    round_score[uid] += cfg.round_weight / (cfg.rrf_k + rank)
                for keys, key_rounds, weight in (
                    (fact_hits, fact_rounds, cfg.fact_weight),
                    (image_hits, image_rounds, cfg.image_weight),
                ):
                    mapped: list[str] = []
                    for key, _ in keys:
                        for rid in key_rounds.get(key, []):
                            if rid not in mapped:
                                mapped.append(rid)
                    for rank, rid in enumerate(mapped, start=1):
                        round_score[rid] += weight / (cfg.rrf_k + rank)
                continue
            # 默认方案：同一模态内按分数取最大值（轮自己的分数 vs 它来源事实、图片描述的分数），
            # 保留"一条很强的直接命中"；sparse 分数在各个索引里尺度不同，先各自归一化
            best: dict[str, float] = {}
            for uid, score in _normalized(round_hits, modality):
                best[uid] = max(best.get(uid, float("-inf")), cfg.round_weight * score)
            for fid, score in _normalized(fact_hits, modality):
                for rid in fact_rounds.get(fid, []):
                    best[rid] = max(best.get(rid, float("-inf")), cfg.fact_weight * score)
            for iid, score in _normalized(image_hits, modality):
                for rid in image_rounds.get(iid, []):
                    best[rid] = max(best.get(rid, float("-inf")), cfg.image_weight * score)
            weight = 1.0 if modality == "dense" else cfg.sparse_weight
            ranked_ids = sorted(best, key=lambda rid: (-best[rid], rid))
            for rank, rid in enumerate(ranked_ids, start=1):
                round_score[rid] += weight / (cfg.rrf_k + rank)

        # 以文搜图：第三种模态，按图片所在轮的名次参与 RRF
        media_score: dict[str, float] = defaultdict(float)
        if visual:
            visual_rounds: list[str] = []
            for rank, (media_id, _) in enumerate(visual, start=1):
                media_score[media_id] += 1.0 / (cfg.rrf_k + rank)
                item = visual_media.get(media_id)
                owner = turn_to_round.get(item.source_turn_id) if item else None
                if owner is not None and owner not in visual_rounds:
                    visual_rounds.append(owner)
            for rank, owner in enumerate(visual_rounds, start=1):
                round_score[owner] += cfg.visual_weight / (cfg.rrf_k + rank)
        # 图片描述键的名次也算进图片的相关度（挑原图时用）
        for modality in ("dense", "sparse"):
            hits = routes.get((cfg.image_kind, modality), []) if cfg.image_kind else []
            for rank, (iid, _) in enumerate(hits, start=1):
                linked = image_units[iid].meta.get("media_id") if iid in image_units else None
                if linked:
                    media_score[str(linked)] += 1.0 / (cfg.rrf_k + rank)

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

        ranked = sorted(round_score.items(), key=lambda item: (-item[1], item[0]))
        return L1Candidates(
            query=query,
            ranked=ranked,
            round_units=round_units,
            facts=facts,
            fact_score=dict(fact_score),
            fact_rounds=fact_rounds,
            debug={"in_time_range": len(in_range), "candidates": len(round_score)},
            media_score=dict(media_score),
        )

    def _has_images(self, space_id: str) -> bool:
        return self.db.has_media(space_id)

    def _bm25_search(self, space_id: str, kind: str, query: str, k: int) -> list[tuple[str, float]]:
        units = self.db.list_units(space_id, kinds=[kind])
        cached = self._bm25.get((space_id, kind))
        if cached is None or cached[0] != len(units):
            cached = (len(units), BM25([u.text for u in units]), units)
            self._bm25[(space_id, kind)] = cached
        _, index, indexed = cached
        return [(indexed[i].id, score) for i, score in index.search(query, k)]


def collect_media(
    db: MemoryDB,
    space_id: str,
    rounds: Sequence[Hit],
    media_score: dict[str, float],
    attach: int,
) -> tuple[dict[str, MediaItem], list[str]]:
    """装入的轮里的图片（回答时把描述填回原位），以及要附上原图的前 attach 张。

    附图优先选相关度高的（描述键或以文搜图命中过的），不够时按轮的名次补。
    """
    ids = [m for h in rounds for m in h.unit.meta.get("media_ids", [])]
    if not ids:
        return {}, []
    media = {m.id: m for m in db.get_media(space_id, ids=ids)}
    if attach <= 0:
        return media, []
    order = {media_id: i for i, media_id in enumerate(ids)}
    ranked = sorted(media, key=lambda m: (-media_score.get(m, 0.0), order[m]))
    return media, ranked[:attach]


def pack_rounds(
    ranked: list[tuple[str, float]],
    round_units: dict[str, MemoryUnit],
    cfg: L1Config,
    *,
    route: str = "l1",
) -> list[Hit]:
    """按分数从高到低装入轮：设了 token 预算时装到预算用完（至少 1 轮，最多 max_rounds 轮），
    否则取前 k 轮。放不下的轮跳过，后面更短的轮可能还放得下。"""
    if cfg.budget_tokens is None:
        chosen = ranked[: cfg.k]
    else:
        chosen, used = [], 0
        for rid, score in ranked[: cfg.max_rounds]:
            cost = estimate_tokens(round_units[rid].text)
            if chosen and used + cost > cfg.budget_tokens:
                continue
            chosen.append((rid, score))
            used += cost
    return [
        Hit(unit=round_units[rid], score=score, route=route, rank=rank)
        for rank, (rid, score) in enumerate(chosen, start=1)
    ]


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
