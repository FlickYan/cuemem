"""L2 深度检索（调用 LLM，设计文档 §6.5）。

    问题 ──查询规划（1 次便宜模型调用）──▶ 子查询、时间区间、目标实体、答案类型
         ──多路召回──▶ 原问题和每个子查询各做一遍 L1 打分（几个查询一次批量向量化）
                      + 记忆图上的个性化 PageRank（种子：点名的实体、排在前面的事实与轮）
         ──融合──▶ 各查询的排名按"取最好名次"融合；（可选）重排模型对前几十条重新打分
         ──打包──▶ 按 token 预算装入轮；计数、列举类问题多给事实
         ──充分性检查（每次 1 次便宜模型调用）──▶ 缺信息就按它给的查询补检索，最多 R 次

L1 不调用 LLM、约 100 ms；L2 多花 2–4 次便宜模型调用和一两秒，换取更完整的证据。
什么时候值得花这笔钱，由检索控制器决定（N5 先给规则版，N7 做可学习版）。
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from cuemem.core.models import CoreMemory, Fact, FactHit, Hit, MemoryUnit
from cuemem.core.trace import record_event, span
from cuemem.embed.base import TextEmbedder
from cuemem.errors import ContentFilteredError, StructuredOutputError
from cuemem.llm.client import LLMClient
from cuemem.read.facts import choose_facts
from cuemem.read.graph import GraphConfig, MemoryGraph
from cuemem.read.l1 import L1Candidates, L1Config, L1Retriever, estimate_tokens, pack_rounds
from cuemem.read.plan import QueryPlan, QueryPlanner, rule_plan
from cuemem.read.rerank import Reranker
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore

CHECK_SYSTEM = """You check whether memories retrieved from a personal assistant's past conversations with the user are enough to answer a question. Do not answer the question.

Set "sufficient" to true if the memories contain the information the question needs, even when the answer still has to be worked out from them (adding up amounts, comparing dates, computing an age or a time difference). Conflicting or outdated values are not a reason to search further: the most recent one usually applies. For questions that count, add up or list things, the items are often spread over several conversations: set "sufficient" to false if some items are probably missing, for example when a memory refers to another occasion that is not itself among the memories, or the memories cover only part of the period asked about.
If something is missing, say what in "missing" (one short sentence) and write up to 3 new search queries that would find it, phrased the way the user would have mentioned it in a conversation. Do not repeat the queries already tried."""  # noqa: E501

CHECK_ROUND_CHARS = 700  # 充分性检查只看每轮的开头，控制成本
MAX_FOLLOWUP_QUERIES = 3


@dataclass(frozen=True)
class L2Config:
    # 每个查询的打分沿用 L1；打包预算 6k tokens、最多 60 轮（L1 是 4k、40 轮）
    l1: L1Config = field(default_factory=lambda: L1Config(budget_tokens=6000, max_rounds=60))
    plan: bool = True  # False：不调用规划模型，用规则版计划（只有原问题一个查询）
    fusion: Literal["max", "rrf"] = "max"  # 多个查询的排名怎么合：取最好名次 / 名次倒数求和
    sub_query_weight: float = 1.0
    # 图扩散与重排都实现了，但在 dev100 的纯检索上没有提高"预算内的召回"，默认关闭（D-030）
    ppr: bool = False
    ppr_weight: float = 0.5  # 图扩散这一路在融合里的权重
    graph: GraphConfig = field(default_factory=GraphConfig)
    seed_rounds: int = 5  # 图扩散的种子：融合后排在前面的轮与事实
    seed_facts: int = 10
    rerank: bool = False
    rerank_pool: int = 30
    iterations: int = 2  # 充分性检查的最多次数（0 = 不检查、不补检索）
    followup_budget: int = 1000  # 每次补检索最多再装入多少 tokens 的轮
    followup_facts: int = 5
    fact_k: int = 10
    aggregate_fact_k: int = 25  # 计数、求和、列举类问题多给事实（事实很短，一条约 30 tokens）


@dataclass
class CheckRecord:
    iteration: int
    sufficient: bool
    missing: str
    queries: list[str]
    added_rounds: int = 0
    added_facts: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "sufficient": self.sufficient,
            "missing": self.missing,
            "queries": self.queries,
            "added_rounds": self.added_rounds,
            "added_facts": self.added_facts,
            "error": self.error,
        }


@dataclass
class L2Result:
    plan: QueryPlan
    rounds: list[Hit]
    facts: list[FactHit]
    core: CoreMemory | None
    checks: list[CheckRecord] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def details(self) -> dict[str, Any]:
        """评测结果里记录的 L2 过程：计划、每次充分性检查、图扩散与重排的情况。"""
        return {
            "plan": self.plan.as_dict(),
            "checks": [c.as_dict() for c in self.checks],
            **self.debug,
        }


class CheckOutput(BaseModel):
    sufficient: bool = True
    missing: str = ""
    queries: list[str] = []

    @field_validator("queries", mode="before")
    @classmethod
    def _queries(cls, value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [str(v).strip() for v in items if str(v or "").strip()]


class SufficiencyChecker:
    def __init__(self, llm: LLMClient, *, role: str = "planner") -> None:
        self.llm = llm
        self.role = role

    def check(
        self,
        question: str,
        question_time: datetime,
        rounds: Sequence[Hit],
        facts: Sequence[FactHit],
        tried: Sequence[str],
        *,
        cache_scope: str | None = None,
    ) -> CheckOutput:
        lines = [f"Question date: {question_time:%Y-%m-%d (%A)}", f"Question: {question}"]
        lines.append("Queries already tried: " + "; ".join(tried))
        lines.append("\nMemories:")
        ordered_facts = sorted(facts, key=lambda fh: (fh.fact.recorded_at, fh.fact.id))
        for i, fh in enumerate(ordered_facts, start=1):
            when = fh.fact.valid_from or f"{fh.fact.recorded_at:%Y-%m-%d}"
            lines.append(f"[f{i}] ({when}) {fh.fact.statement}")
        for i, hit in enumerate(sorted(rounds, key=lambda h: (h.unit.ts, h.unit.id)), start=1):
            text = hit.unit.text
            if len(text) > CHECK_ROUND_CHARS:
                text = text[:CHECK_ROUND_CHARS] + " …"
            lines.append(f"[m{i}] {hit.unit.ts:%Y-%m-%d (%a)}\n{text}")
        messages = [
            {"role": "system", "content": CHECK_SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ]
        return self.llm.structured(self.role, messages, CheckOutput, cache_scope=cache_scope).value


class L2Retriever:
    def __init__(
        self,
        db: MemoryDB,
        vectors: VectorStore,
        embedder: TextEmbedder,
        llm: LLMClient,
        l1: L1Retriever,
        *,
        reranker: Reranker | None = None,
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.l1 = l1
        self.planner = QueryPlanner(llm)
        self.checker = SufficiencyChecker(llm)
        self.reranker = reranker
        # 记忆图按空间缓存；事实、实体或轮的数量变化（写入、巩固、遗忘之后）时重建
        self._graphs: dict[tuple[str, GraphConfig], tuple[tuple[int, int, int], MemoryGraph]] = {}
        self._lock = threading.Lock()

    def search(
        self,
        space_id: str,
        question: str,
        question_time: datetime,
        cfg: L2Config,
        *,
        cache_scope: str | None = None,
    ) -> L2Result:
        timings: dict[str, float] = {}
        started = time.perf_counter()
        plan = (
            self.planner.plan(question, question_time, cache_scope=cache_scope)
            if cfg.plan
            else rule_plan(question, question_time)
        )
        timings["plan"] = _since(started)
        debug: dict[str, Any] = {}

        with span("retrieve", route="l2", queries=len(plan.queries)) as record:
            pool = self._recall(space_id, plan, plan.queries, question_time, cfg, debug)
            timings["recall"] = _since(started) - timings["plan"]
            ranked = pool.ranked
            if cfg.rerank and self.reranker is not None:
                ranked = self._rerank(question, ranked, pool.round_units, cfg, cache_scope, debug)
            rounds = pack_rounds(ranked, pool.round_units, cfg.l1, route="l2")
            fact_k = cfg.aggregate_fact_k if plan.aggregate else cfg.fact_k
            facts = choose_facts(
                self.db,
                space_id,
                pool.facts,
                pool.fact_score,
                fact_k,
                current_only=False,
                chain=cfg.l1.chain,
            )
            record["hits"] = [[h.unit.id, round(h.score, 5)] for h in rounds]
            record["facts"] = [[fh.fact.id, round(fh.score, 5)] for fh in facts]
        timings["first_pass"] = _since(started)

        checks: list[CheckRecord] = []
        tried = list(plan.queries)
        for iteration in range(1, cfg.iterations + 1):
            with span("check", iteration=iteration) as record:
                outcome = self._check(question, question_time, rounds, facts, tried, cache_scope)
                outcome.iteration = iteration
                checks.append(outcome)
                record.update(outcome.as_dict())
                new_queries = [q for q in outcome.queries if q.casefold() not in _folded(tried)]
                if outcome.sufficient or not new_queries:
                    break
                new_queries = new_queries[:MAX_FOLLOWUP_QUERIES]
                tried.extend(new_queries)
                extra = self._recall(space_id, plan, new_queries, question_time, cfg, None)
                added_rounds = _add_rounds(rounds, extra, cfg)
                added_facts = self._add_facts(space_id, facts, extra, cfg.followup_facts)
                outcome.added_rounds, outcome.added_facts = len(added_rounds), len(added_facts)
                rounds.extend(added_rounds)
                facts.extend(added_facts)
        timings["total"] = _since(started)
        debug["evidence_tokens"] = sum(estimate_tokens(h.unit.text) for h in rounds)
        return L2Result(
            plan=plan,
            rounds=rounds,
            facts=facts,
            core=self.db.get_core_memory(space_id),
            checks=checks,
            timings_ms=timings,
            debug=debug,
        )

    # ---------- 多路召回与融合 ----------

    def _recall(
        self,
        space_id: str,
        plan: QueryPlan,
        queries: Sequence[str],
        question_time: datetime,
        cfg: L2Config,
        debug: dict[str, Any] | None,
    ) -> L1Candidates:
        """对每个查询做一遍 L1 打分，按查询融合，再（第一轮时）加上图扩散这一路。"""
        vectors = self.embedder.embed(list(queries), "query").dense
        per_query = [
            self.l1.candidates(
                space_id,
                text,
                question_time,
                cfg.l1,
                query=replace(plan.query, text=text),
                dense_vec=vectors[i],
            )
            for i, text in enumerate(queries)
        ]
        weights = [1.0 if text == plan.question else cfg.sub_query_weight for text in queries]
        fused = _fuse_queries(per_query, weights, cfg)
        if cfg.ppr and debug is not None:
            self._diffuse(space_id, plan, fused, cfg, debug)
        return fused

    def _diffuse(
        self,
        space_id: str,
        plan: QueryPlan,
        fused: L1Candidates,
        cfg: L2Config,
        debug: dict[str, Any],
    ) -> None:
        """个性化 PageRank：图上离种子近的轮与事实获得一路额外的排名。"""
        graph = self._graph(space_id, cfg.graph)
        entity_seeds = graph.link_entities([plan.question, *plan.entities])
        seeds: dict[str, float] = {}
        _add_seeds(seeds, entity_seeds, 0.4)
        top_facts = sorted(fused.fact_score, key=lambda f: (-fused.fact_score[f], f))
        _add_seeds(seeds, {f: fused.fact_score[f] for f in top_facts[: cfg.seed_facts]}, 0.3)
        _add_seeds(seeds, dict(fused.ranked[: cfg.seed_rounds]), 0.3)
        scores = graph.personalized_pagerank(seeds, cfg.graph)
        rrf_k = cfg.l1.rrf_k
        round_score = dict(fused.ranked)
        new_rounds = [rid for rid, _ in graph.ranked(scores, "round") if rid not in round_score]
        if new_rounds:  # 图上找到的新候选轮：载入它们的内容
            for unit in self.db.get_units(space_id, new_rounds):
                fused.round_units[unit.id] = unit
        for rank, (rid, _) in enumerate(graph.ranked(scores, "round"), start=1):
            if rid in fused.round_units:
                round_score[rid] = round_score.get(rid, 0.0) + cfg.ppr_weight / (rrf_k + rank)
        new_facts = [fid for fid, _ in graph.ranked(scores, "fact") if fid not in fused.facts]
        for fact in self.db.get_facts(space_id, ids=new_facts):
            fused.facts[fact.id] = fact
        for rank, (fid, _) in enumerate(graph.ranked(scores, "fact"), start=1):
            if fid in fused.facts:
                fused.fact_score[fid] = fused.fact_score.get(fid, 0.0) + cfg.ppr_weight / (
                    rrf_k + rank
                )
        fused.ranked = sorted(round_score.items(), key=lambda item: (-item[1], item[0]))
        debug["ppr"] = {
            "linked_entities": len(entity_seeds),
            "new_rounds": len(new_rounds),
            "new_facts": len(new_facts),
            "graph": graph.stats,
        }

    def _graph(self, space_id: str, cfg: GraphConfig) -> MemoryGraph:
        size = (
            len(self.db.get_facts(space_id)),
            len(self.db.get_entities(space_id)),
            len(self.db.list_units(space_id, kinds=["round"])),
        )
        with self._lock:
            cached = self._graphs.get((space_id, cfg))
            if cached is not None and cached[0] == size:
                return cached[1]
        with span("graph_build"):
            graph = MemoryGraph.build(self.db, self.vectors, space_id, cfg)
        with self._lock:
            self._graphs[(space_id, cfg)] = (size, graph)
        return graph

    def _rerank(
        self,
        question: str,
        ranked: list[tuple[str, float]],
        units: dict[str, MemoryUnit],
        cfg: L2Config,
        cache_scope: str | None,
        debug: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """重排模型给前 rerank_pool 条打分；最终名次 = 融合名次与重排名次的 RRF。"""
        assert self.reranker is not None
        pool = ranked[: cfg.rerank_pool]
        with span("rerank", docs=len(pool)):
            scores = self.reranker.scores(
                question, [units[rid].text for rid, _ in pool], cache_scope=cache_scope
            )
        by_rerank = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        rrf_k = cfg.l1.rrf_k
        combined = {rid: 1.0 / (rrf_k + rank) for rank, (rid, _) in enumerate(pool, start=1)}
        for rank, i in enumerate(by_rerank, start=1):
            combined[pool[i][0]] += 1.0 / (rrf_k + rank)
        reordered = sorted(combined.items(), key=lambda item: (-item[1], item[0]))
        debug["rerank"] = {"pool": len(pool), "top_score": round(max(scores, default=0.0), 4)}
        return reordered + ranked[cfg.rerank_pool :]

    # ---------- 充分性检查与补检索 ----------

    def _check(
        self,
        question: str,
        question_time: datetime,
        rounds: Sequence[Hit],
        facts: Sequence[FactHit],
        tried: Sequence[str],
        cache_scope: str | None,
    ) -> CheckRecord:
        try:
            output = self.checker.check(
                question, question_time, rounds, facts, tried, cache_scope=cache_scope
            )
        except (StructuredOutputError, ContentFilteredError) as exc:
            # 检查失败就不再补检索：用第一轮的证据照常回答
            record_event("check_failed", error=type(exc).__name__)
            return CheckRecord(0, True, "", [], error=type(exc).__name__)
        return CheckRecord(0, output.sufficient, output.missing, output.queries)

    def _add_facts(
        self, space_id: str, facts: list[FactHit], extra: L1Candidates, limit: int
    ) -> list[FactHit]:
        present = {fh.fact.id for fh in facts}
        fresh = {fid: f for fid, f in extra.facts.items() if fid not in present}
        scores = {fid: s for fid, s in extra.fact_score.items() if fid in fresh}
        added = choose_facts(
            self.db, space_id, fresh, scores, limit, current_only=False, chain=True
        )
        return [fh for fh in added if fh.fact.id not in present]


def _fuse_queries(
    per_query: Sequence[L1Candidates], weights: Sequence[float], cfg: L2Config
) -> L1Candidates:
    """多个查询的候选合成一份。

    max：每条取它在各查询里最好的（加权）名次分，再加一点各查询名次分之和打破平局——
    子查询各找一块信息（"现在几岁""哪年搬到美国"），每块的最佳命中都应该排到前面，
    而不是让"几个查询里都排中游"的轮胜出；
    rrf：名次分直接求和（共识优先），作为对照。
    """
    if len(per_query) == 1:
        return per_query[0]
    rrf_k = cfg.l1.rrf_k
    round_parts: dict[str, list[float]] = defaultdict(list)
    fact_parts: dict[str, list[float]] = defaultdict(list)
    units: dict[str, MemoryUnit] = {}
    facts: dict[str, Fact] = {}
    fact_rounds: dict[str, list[str]] = {}
    for cands, weight in zip(per_query, weights, strict=True):
        for rank, (rid, _) in enumerate(cands.ranked, start=1):
            round_parts[rid].append(weight / (rrf_k + rank))
        ordered = sorted(cands.fact_score, key=lambda f: (-cands.fact_score[f], f))
        for rank, fid in enumerate(ordered, start=1):
            fact_parts[fid].append(weight / (rrf_k + rank))
        units.update(cands.round_units)
        facts.update(cands.facts)
        fact_rounds.update(cands.fact_rounds)

    def combine(parts: list[float]) -> float:
        return max(parts) + 0.1 * sum(parts) if cfg.fusion == "max" else sum(parts)

    round_score = {rid: combine(parts) for rid, parts in round_parts.items()}
    first = per_query[0]
    return L1Candidates(
        query=first.query,
        ranked=sorted(round_score.items(), key=lambda item: (-item[1], item[0])),
        round_units=units,
        facts=facts,
        fact_score={fid: combine(parts) for fid, parts in fact_parts.items()},
        fact_rounds=fact_rounds,
        debug={"queries": len(per_query), "candidates": len(round_score)},
    )


def _add_rounds(rounds: Sequence[Hit], extra: L1Candidates, cfg: L2Config) -> list[Hit]:
    """补检索：装入还没有的轮，最多 followup_budget tokens。"""
    present = {h.unit.id for h in rounds}
    fresh = [(rid, score) for rid, score in extra.ranked if rid not in present]
    budget = replace(cfg.l1, budget_tokens=cfg.followup_budget)
    added = pack_rounds(fresh, extra.round_units, budget, route="l2_followup")
    start = len(rounds)
    return [h.model_copy(update={"rank": start + i}) for i, h in enumerate(added, start=1)]


def _add_seeds(seeds: dict[str, float], items: dict[str, float], mass: float) -> None:
    """把一组种子按分数比例分到总量 mass。"""
    total = sum(v for v in items.values() if v > 0)
    if total <= 0:
        return
    for node, value in items.items():
        if value > 0:
            seeds[node] = seeds.get(node, 0.0) + mass * value / total


def _folded(texts: Sequence[str]) -> set[str]:
    return {t.casefold() for t in texts}


def _since(started: float) -> float:
    return (time.perf_counter() - started) * 1000
