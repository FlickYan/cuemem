"""MemorySystem：把存储、向量化、写入、巩固、检索、回答组装在一起的门面。

上层（CLI、评测 runner、之后的 Agent 与 API 服务）只和它打交道。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import httpx2
from pydantic import BaseModel, Field

from cuemem.config import Settings
from cuemem.core.models import (
    Answer,
    Conversation,
    CoreMemory,
    Entity,
    Fact,
    Hit,
    MediaItem,
    SessionIn,
)
from cuemem.core.trace import TraceSink, start_trace
from cuemem.embed import CachedEmbedder, build_embedder, build_image_embedder
from cuemem.embed.base import TextEmbedder
from cuemem.embed.multimodal import ImageEmbedder
from cuemem.errors import ConsolidationError
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.read.agent import ToolReader
from cuemem.read.controller import (
    DEFAULT_MODEL,
    HYBRID_BAND,
    HYBRID_CONFIDENCE,
    LearnedController,
    LLMRouter,
    RouteDecision,
    Router,
    RuleController,
)
from cuemem.read.facts import FactAugmentedRetriever, ImageMode, MemoryView, Validity
from cuemem.read.l1 import L1Candidates, L1Config, L1Retriever
from cuemem.read.l2 import L2Config, L2Retriever
from cuemem.read.naive import NaiveRetriever
from cuemem.read.plan import QueryPlan, rule_plan
from cuemem.read.probe import Probe
from cuemem.read.reader import Reader
from cuemem.read.rerank import Reranker
from cuemem.store.media import MediaStore
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore
from cuemem.write.consolidate import (
    ConsolidationReport,
    ConsolidationWorker,
    Consolidator,
    scope,
)
from cuemem.write.online import IngestReport, OnlineWriter

AskMode = Literal["l0", "l1", "l2", "auto", "agent", "hatch", "facts", "naive"]
RecallMode = Literal["l0", "l1", "l2", "auto"]
Level = Literal["l0", "l1", "l2"]


@dataclass
class Recall:
    """一次检索（不回答）的结果：走了哪一档、交给回答模型的材料，以及过程记录。"""

    level: Level
    view: MemoryView
    decision: RouteDecision | None = None  # mode="auto" 时检索控制器的决定
    router: Router | None = None  # 实际用的控制器（没有模型文件时 learned 退回 rule）
    plan: QueryPlan | None = None  # L2 的检索计划（阅读提示用）
    details: dict[str, Any] = field(default_factory=dict)  # L2 的计划、充分性检查等


@dataclass
class ForgetReport:
    space_id: str
    turns_deleted: int
    sessions_affected: list[str]
    facts_before: int
    facts_after: int
    embed_cache_purged: int
    llm_cache_purged: int
    elapsed_ms: float
    images_deleted: int = 0  # 删除的原图文件数（同一张图被别的发言引用时保留）
    cascaded: int = 0  # 沿引用扩散多删的发言数（后来的回复引用过被删的记忆）
    traces_removed: int = 0  # 删除的 trace 条数（属性里有被删的原话）
    restated: int = 0  # 原始日志里复述了被删信息（没有引用、追不到）而一起删的发言数（N10）
    regrown: int = 0  # 重建后仍在复述被删信息、因而跟着删掉出处的事实数（N10）

    def merged(self, later: ForgetReport) -> ForgetReport:
        """同一次遗忘的后续几轮并进来：计数相加，删除前后的事实数取首尾。"""
        return replace(
            self,
            turns_deleted=self.turns_deleted + later.turns_deleted,
            sessions_affected=sorted({*self.sessions_affected, *later.sessions_affected}),
            facts_after=later.facts_after,
            embed_cache_purged=self.embed_cache_purged + later.embed_cache_purged,
            llm_cache_purged=self.llm_cache_purged + later.llm_cache_purged,
            elapsed_ms=self.elapsed_ms + later.elapsed_ms,
            images_deleted=self.images_deleted + later.images_deleted,
            cascaded=self.cascaded + later.cascaded,
            traces_removed=self.traces_removed + later.traces_removed,
            restated=self.restated + later.restated,
            regrown=self.regrown + later.regrown,
        )


REGROWN_SYSTEM = (
    "You check whether information the user asked to forget is still in their memory. "
    "You get one deleted fact and a numbered list of facts that remain. List the remaining "
    "facts that still reveal the deleted information, fully or in part: a restatement, the same "
    "detail in other words, or a vaguer version that still gives it away. A fact that only "
    "mentions the same person or topic, or says something different about it, does not count. "
    'Reply with JSON only: {"same": [numbers]}'
)


RESTATED_SYSTEM = (
    "You check whether information the user asked to forget is still in their conversation "
    "history. You get one deleted fact and numbered messages from the remaining conversation. "
    "List the messages that still reveal the deleted information, fully or in part: for example "
    "the assistant restating it in passing, or the same detail in other words. A message that "
    "only mentions the same person or topic, or says something different about it, does not "
    'count. Reply with JSON only: {"same": [numbers]}'
)


class RegrownOutput(BaseModel):
    same: list[int] = Field(default_factory=list)


def load_controller(settings: Settings) -> LearnedController | None:
    """按配置（retrieval.controller）加载检索控制器的模型文件；没有时返回 None（退回规则版）。"""
    path = Path(settings.retrieval.controller)
    if not path.is_absolute():
        path = DEFAULT_MODEL.parent / path
    return LearnedController.load(path) if path.exists() else None


def image_mode(cfg: L1Config) -> ImageMode:
    """检索用哪种图片描述，回答时就展示哪种（不用描述时只留 [Image] 占位）。"""
    if cfg.image_kind == "image":
        return "context"
    if cfg.image_kind == "image_iso":
        return "isolated"
    return "none"


def route_details(recall: Recall) -> dict[str, Any]:
    """控制器的决定，记在回答的 details["route"] 里（评测结果与 Agent 的轨迹面板都用）。"""
    decision = recall.decision
    assert decision is not None
    return {
        "router": recall.router,
        "level": decision.level,
        "see": decision.see,
        "reasons": decision.reasons,
        **({"scores": decision.scores} if decision.scores else {}),
    }


def build_reranker(
    settings: Settings,
    *,
    llm: LLMClient,
    ledger: Ledger | None,
    http_client: httpx2.Client | None = None,
) -> Reranker | None:
    """重排模型（L2 用）。没有对应供应商的 Key 时返回 None，L2 不做重排。"""
    provider = settings.providers.get(settings.rerank.provider)
    api_key = provider.api_key() if provider is not None else None
    if api_key is None:
        return None
    return Reranker(
        settings.rerank, api_key, http_client=http_client, cache=llm.cache, ledger=ledger
    )


class MemorySystem:
    def __init__(
        self,
        settings: Settings,
        *,
        db: MemoryDB,
        vectors: VectorStore,
        embedder: TextEmbedder,
        llm: LLMClient,
        ledger: Ledger,
        tracer: TraceSink | None = None,
        owns_ledger: bool = True,
        reranker: Reranker | None = None,
        media: MediaStore | None = None,
        image_embedder: ImageEmbedder | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.llm = llm
        self.ledger = ledger
        self.tracer = tracer
        self._owns_ledger = owns_ledger  # 评测时账本由多个线程共享，不能由单个系统关闭
        self.media = media
        self.image_embedder = image_embedder
        self.writer = OnlineWriter(
            db, vectors, embedder, media=media, image_embedder=image_embedder
        )
        self.consolidator = Consolidator(db, vectors, embedder, llm, settings.consolidation)
        self.retriever = NaiveRetriever(db, vectors, embedder)
        self.fact_retriever = FactAugmentedRetriever(db, vectors, embedder)
        self.l1 = L1Retriever(db, vectors, embedder, image_embedder=image_embedder)
        self.l2 = L2Retriever(db, vectors, embedder, llm, self.l1, reranker=reranker)
        self.probe = Probe(db, self.l1)
        self.controller = RuleController()
        # 检索控制器（N7 / N11）：模型文件由 scripts/n7_controller.py（v1）、
        # scripts/n11_controller_v2.py（v2，默认）训练导出；没有时退回规则版
        self.learned = load_controller(settings)
        self.router = LLMRouter(llm)
        self.reader = Reader(llm)
        self.agent = ToolReader(llm)
        self.worker: ConsolidationWorker | None = None
        # 进行中的会话（Agent 正在对话，N8）：暂不巩固，会话结束后再整体抽取一次
        self._open: set[tuple[str, str]] = set()
        self._open_lock = threading.Lock()

    @classmethod
    def from_settings(
        cls, settings: Settings, *, http_client: httpx2.Client | None = None
    ) -> MemorySystem:
        """按配置组装各组件。http_client 用于测试时注入模拟的网络层。"""
        paths = settings.paths
        ledger = Ledger(paths.ledger)
        embedder = build_embedder(settings, ledger=ledger, http_client=http_client)
        db = MemoryDB(paths.db)
        # 记录建库时的向量模型，防止换模型后新旧向量混在一起
        db.check_or_set_meta("embedder", f"{embedder.name}:{embedder.dim}")
        llm = LLMClient(
            settings,
            cache=ResponseCache(paths.llm_cache),
            ledger=ledger,
            http_client=http_client,
        )
        return cls(
            settings,
            db=db,
            vectors=VectorStore(paths.qdrant, dim=embedder.dim, url=settings.vector_store.url),
            embedder=embedder,
            llm=llm,
            ledger=ledger,
            tracer=TraceSink(paths.traces),
            reranker=build_reranker(settings, llm=llm, ledger=ledger, http_client=http_client),
            media=MediaStore(paths.media),
            image_embedder=build_image_embedder(settings, ledger=ledger, http_client=http_client),
        )

    @classmethod
    def ephemeral(
        cls,
        settings: Settings,
        *,
        embedder: TextEmbedder,
        llm: LLMClient,
        ledger: Ledger,
        reranker: Reranker | None = None,
        image_embedder: ImageEmbedder | None = None,
    ) -> MemorySystem:
        """评测用：数据库与向量库都放在内存里，用完即弃；向量缓存与 LLM 缓存由传入的组件共享。"""
        return cls(
            settings,
            db=MemoryDB(None),
            vectors=VectorStore(None, dim=embedder.dim),
            embedder=embedder,
            llm=llm,
            ledger=ledger,
            owns_ledger=False,
            reranker=reranker,
            media=MediaStore(settings.paths.media),
            image_embedder=image_embedder,
        )

    @classmethod
    def on_disk(
        cls,
        settings: Settings,
        directory: Path,
        *,
        embedder: TextEmbedder,
        llm: LLMClient,
        ledger: Ledger,
        reranker: Reranker | None = None,
        image_embedder: ImageEmbedder | None = None,
    ) -> MemorySystem:
        """评测用：每道题（每份对话历史）一个磁盘目录，写入与巩固的结果跨运行复用。

        换了向量模型会在打开时报 ConfigMismatchError，调用方应删掉目录重建。
        """
        db = MemoryDB(directory / "memory.sqlite")
        try:
            db.check_or_set_meta("embedder", f"{embedder.name}:{embedder.dim}")
        except Exception:
            db.close()
            raise
        return cls(
            settings,
            db=db,
            vectors=VectorStore(directory / "qdrant", dim=embedder.dim),
            embedder=embedder,
            llm=llm,
            ledger=ledger,
            owns_ledger=False,
            reranker=reranker,
            media=MediaStore(settings.paths.media),  # 原图在各空间之间共享（按内容哈希存放）
            image_embedder=image_embedder,
        )

    # ---------- 写入 ----------

    def ingest(self, conversation: Conversation, *, consolidate: bool = False) -> IngestReport:
        """在线写入（不调用 LLM），并为新会话登记巩固任务（进行中的会话除外）。

        consolidate=True 时立即同步巩固；否则如果后台线程已启动，就交给它异步完成。
        """
        space_id = conversation.space_id
        with start_trace("ingest", self.tracer, space=space_id) as trace:
            report = self.writer.ingest(conversation)
            queued = self.consolidator.enqueue(space_id, skip=self.open_sessions(space_id))
            trace.attrs.update(
                units_added=report.units_added, turns_added=report.turns_added, jobs=queued
            )
        if consolidate:
            self.consolidate(space_id)
        elif self.worker is not None and queued:
            self.worker.submit(space_id)
        return report

    def append(self, space_id: str, session: SessionIn, first_idx: int) -> IngestReport:
        """往会话里追加发言（Agent 每轮写回新的一问一答，序号从 first_idx 起），在线写入、
        不调用 LLM。会话进行中不登记巩固任务（见 open_session）。"""
        with start_trace("append", self.tracer, space=space_id, session=session.session_id):
            report = self.writer.append(space_id, session, first_idx)
            queued = self.consolidator.enqueue(
                space_id, [session.session_id], skip=self.open_sessions(space_id)
            )
        if queued and self.worker is not None:
            self.worker.submit(space_id)
        return report

    # ---------- 进行中的会话（Agent，N8） ----------

    def open_session(self, space_id: str, session_id: str) -> None:
        """标记会话进行中：每轮照常在线写入（可检索），但不登记巩固任务——
        否则半个会话会被抽取一遍，会话结束后整个会话又抽取一遍。"""
        with self._open_lock:
            self._open.add((space_id, session_id))

    def close_session(
        self, space_id: str, session_id: str, *, wait: bool = False
    ) -> ConsolidationReport | None:
        """会话结束：登记巩固任务；wait=True 时同步巩固并返回报告，否则交给后台线程。"""
        with self._open_lock:
            self._open.discard((space_id, session_id))
        self.consolidator.enqueue(space_id, [session_id])
        if wait:
            return self.consolidate(space_id)
        self.start_worker().submit(space_id)
        return None

    def open_sessions(self, space_id: str) -> set[str]:
        with self._open_lock:
            return {sid for space, sid in self._open if space == space_id}

    def consolidate(
        self, space_id: str, *, rebuild: bool = False, strict: bool = False
    ) -> ConsolidationReport:
        """执行待处理的巩固任务。rebuild=True（或抽取逻辑版本变化）时先清空派生记忆再全部重放。

        strict=True 时有会话抽取失败就抛出异常（评测用：宁可整题重试，也不要残缺的记忆）。
        """
        skip = self.open_sessions(space_id)
        with start_trace("consolidate", self.tracer, space=space_id) as trace:
            if rebuild or self.consolidator.stale(space_id):
                self.consolidator.rebuild(space_id, skip=skip)
            self.db.retry_failed_jobs(space_id)
            self.consolidator.enqueue(space_id, skip=skip)
            report = self.consolidator.run(space_id)
            trace.attrs.update(report.as_dict())
        if strict and report.failed:
            raise ConsolidationError(f"{len(report.failed)} 个会话抽取失败：{report.failed[:5]}")
        return report

    def start_worker(self) -> ConsolidationWorker:
        """启动后台巩固线程：此后 ingest 只登记任务，巩固在后台完成。"""
        if self.worker is None:
            self.worker = ConsolidationWorker(self.consolidator)
        return self.worker

    def forget(
        self,
        space_id: str,
        *,
        session_ids: Sequence[str] = (),
        turn_ids: Sequence[str] = (),
        follow_citations: bool = True,
        restated_turn_ids: Sequence[str] = (),
    ) -> ForgetReport:
        """遗忘（被遗忘权）：删除原始轮次，并清除由它们派生出的一切（D-021）。

        派生记忆是原始日志的函数，所以遗忘 = 从日志里删除 + 用剩下的日志重放派生记忆：
        事实、实体、摘要、核心记忆都重新生成；被删内容取代过的旧事实会自动恢复为当前有效。
        随后清理缓存：被删会话的抽取结果、该空间的对账与回答缓存、不再出现的文本的向量。

        follow_citations（N8）：助手后来的回复如果引用过要删的记忆，它就复述了被删的信息，
        那一轮也一起删，并继续往下追（溯源表记录了每次回复引用了什么）。

        restated_turn_ids（D-051）：按内容找到的转述，一起删，但不从它们出发沿引用扩散。
        它们只是顺带透露了被删的信息，所在那一轮里别的内容（用户在同一轮说的另一件事、
        由此抽出的事实、后来引用这些事实的回复）不是要删的。
        """
        started = time.perf_counter()
        with start_trace("forget", self.tracer, space=space_id) as trace:
            all_turns = self.db.get_turns(space_id)
            asked, sessions = set(turn_ids), set(session_ids)
            wanted_turns = {t.id for t in all_turns if t.id in asked or t.session_id in sessions}
            cascaded = 0
            if follow_citations and wanted_turns:
                expanded = self._follow_citations(space_id, wanted_turns)
                cascaded = len(expanded - wanted_turns)
                wanted_turns = expanded
            restated = set(restated_turn_ids)
            wanted_turns |= {t.id for t in all_turns if t.id in restated}
            targets = [t for t in all_turns if t.id in wanted_turns]
            affected = sorted({t.session_id for t in targets})
            if not targets:
                return ForgetReport(space_id, 0, [], 0, 0, 0, 0, 0.0)
            consolidated = bool(self.db.get_jobs(space_id))
            facts_before = len(self.db.get_facts(space_id))
            texts_before = {u.text for u in self.db.list_units(space_id)}

            # 1. 原始层：删除轮次与其中的图片（记录、图片向量；没有别处引用的原图文件），
            #    再按剩下的轮次重建受影响会话的"轮"
            old_rounds = self.db.list_units(space_id, kinds=["round"], session_ids=affected)
            self.vectors.delete(space_id, [u.id for u in old_rounds])
            self.db.delete_units(space_id, [u.id for u in old_rounds])
            doomed_media = self.db.get_media(space_id, turn_ids=[t.id for t in targets])
            deleted = self.db.delete_turns(space_id, [t.id for t in targets])
            self.vectors.delete_images(space_id, [m.id for m in doomed_media])
            images_deleted = 0
            for item in doomed_media:
                if self.media is not None and not self.db.media_sha_in_use(item.sha256):
                    images_deleted += int(self.media.delete(Path(item.path)))
            self.writer.reindex_sessions(space_id, affected)

            # 2. 派生层：清空并清理相关的 LLM 缓存，再从剩下的日志重放
            removed = self.db.clear_derived(space_id)
            self.vectors.delete(space_id, removed)
            llm_purged = 0
            if self.llm.cache is not None:
                parts = [f"session:{sid}" for sid in affected] + ["consolidate", "ask"]
                llm_purged = sum(self.llm.cache.delete_scope(scope(space_id, p)) for p in parts)
            if consolidated:
                self.consolidator.enqueue(space_id, skip=self.open_sessions(space_id))
                self.consolidator.run(space_id)

            # 3. trace：属性里有被删原话的条目（提问、回答原文）一并删除
            traces_removed = self._remove_traces(space_id, [t.text for t in targets])

            # 4. 向量缓存：删除已不在记忆里的文本（向量本身也可能泄露原文）
            gone = sorted(texts_before - {u.text for u in self.db.list_units(space_id)})
            embed_purged = (
                self.embedder.purge(gone) if isinstance(self.embedder, CachedEmbedder) else 0
            )
            self.db.vacuum()
            report = ForgetReport(
                space_id=space_id,
                turns_deleted=deleted,
                sessions_affected=affected,
                facts_before=facts_before,
                facts_after=len(self.db.get_facts(space_id)),
                embed_cache_purged=embed_purged,
                llm_cache_purged=llm_purged,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                images_deleted=images_deleted,
                cascaded=cascaded,
                traces_removed=traces_removed,
            )
            trace.attrs.update(turns_deleted=deleted, sessions=affected, cascaded=cascaded)
        return report

    def _remove_traces(self, space_id: str, texts: Sequence[str]) -> int:
        """删除这个空间里、属性中含有被删原话的 trace（太短的话不作依据，免得误删）。"""
        wanted = [t for t in texts if len(t.strip()) >= 12]
        if self.tracer is None or not wanted:
            return 0

        def doomed(record: dict[str, Any]) -> bool:
            attrs = record.get("attrs") or {}
            if attrs.get("space") != space_id:
                return False
            blob = " ".join(str(v) for v in attrs.values())
            return any(text in blob for text in wanted)

        return self.tracer.remove(doomed)

    def _follow_citations(self, space_id: str, turns: set[str]) -> set[str]:
        """沿引用扩散：要删的发言所在的轮、由它们抽取的事实、其中的图片，凡是被后来的回复
        引用过，那一轮也要删；新删的轮又可能被更后来的回复引用，直到没有新的为止。"""
        rounds = self.db.list_units(space_id, kinds=["round"])
        round_of = {t: u for u in rounds for t in u.source_turn_ids}
        facts = self.db.get_facts(space_id)
        media = self.db.get_media(space_id)
        doomed = set(turns)
        frontier = set(turns)
        while frontier:
            items = {round_of[t].id for t in frontier if t in round_of}
            items |= {f.id for f in facts if frontier & set(f.source_turn_ids)}
            items |= {m.id for m in media if m.source_turn_id in frontier}
            citing = set(self.db.citing_turns(space_id, sorted(items))) - doomed
            fresh: set[str] = set()
            for turn_id in citing:  # 整轮删除：用户的问话里也可能带着同样的信息
                unit = round_of.get(turn_id)
                fresh |= set(unit.source_turn_ids) if unit is not None else {turn_id}
            frontier = fresh - doomed
            doomed |= frontier
        return doomed

    def restating_turns(
        self,
        space_id: str,
        statements: Sequence[str],
        *,
        exclude: Collection[str] = (),
        limit: int = 8,
        floor: float = 0.2,
    ) -> list[str]:
        """原始日志里还有哪些发言复述了被删的信息（N10，D-051），返回这些发言的 ID。

        沿引用扩散只追得到引用过它的回复。助手在当时的对话里顺口转述的（"她 10 月 3 日生日，
        早点下单"），或者进行中的会话里问候时提了一句的，都没有引用，删了出处照样留在日志里。
        对每条要删的事实，按向量取最相近的几轮（包括进行中的会话），把其中的发言逐条交给
        便宜的模型，确认哪几条仍在透露它；只是提到同一个人或话题的不算。

        逐条而不是整轮判断：助手顺口转述时，同一轮里用户说的常常是别的事（"我跑鞋换成
        40 码了"），整轮删掉会连带删掉无关的记忆。
        """
        wanted = [s for s in dict.fromkeys(statements) if s.strip()]
        if not wanted:
            return []
        skip = set(exclude)
        found: list[str] = []
        vectors = self.embedder.embed(wanted, "query").dense
        for statement, vector in zip(wanted, vectors, strict=True):
            hits = self.vectors.search_dense(space_id, vector, limit * 2, kinds=["round"])
            ids = [h.unit_id for h in hits if h.score >= floor]
            rounds = [
                u for u in self.db.get_units(space_id, ids) if not skip & set(u.source_turn_ids)
            ]
            rounds = rounds[:limit]
            if not rounds:
                continue
            order = [t for u in rounds for t in u.source_turn_ids]
            by_id = {t.id: t for t in self.db.get_turns(space_id, ids=order)}
            turns = [by_id[t] for t in order if t in by_id]
            listing = "\n\n".join(
                f"[{n}] {t.role.capitalize()}: {t.text[:1500]}" for n, t in enumerate(turns, 1)
            )
            prompt = f'Deleted fact: "{statement}"\n\nConversation excerpts:\n{listing}'
            verdict = self.llm.structured(
                "extractor",
                [
                    {"role": "system", "content": RESTATED_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                RegrownOutput,
                cache_scope=scope(space_id, "consolidate"),
            ).value
            for n in verdict.same:
                if 1 <= n <= len(turns):
                    found.append(turns[n - 1].id)
        return list(dict.fromkeys(found))

    def regrown_facts(
        self, space_id: str, statements: Sequence[str], *, threshold: float = 0.55, limit: int = 8
    ) -> list[Fact]:
        """遗忘之后重建出来的事实里，哪些还在说被删的信息（N10）。

        沿引用扩散只追得到引用过它的回复；助手在当时的对话里顺口转述（"她 10 月 3 日生日，
        早点下单"）而没有引用的，删掉出处、重建之后，同样的信息会从这一轮重新抽出来。
        先按向量相似度挑候选（同一个人的其他信息也会进来，开发时实测 0.5–0.57，复述 0.6–0.9），
        再让便宜的模型逐条确认：完整复述、换了说法、只透露一部分都算，只是提到同一个人或话题不算。
        """
        facts = self.db.get_facts(space_id)
        wanted = [s for s in dict.fromkeys(statements) if s.strip()]
        if not facts or not wanted:
            return []
        vectors = self.embedder.embed([*wanted, *(f.statement for f in facts)], "document").dense
        similarity = vectors[: len(wanted)] @ vectors[len(wanted) :].T
        found: dict[str, Fact] = {}
        for row, statement in enumerate(wanted):
            order = [int(i) for i in similarity[row].argsort()[::-1][:limit]]
            candidates = [facts[i] for i in order if similarity[row, i] >= threshold]
            if not candidates:
                continue
            listing = "\n".join(f"{n}. {f.statement}" for n, f in enumerate(candidates, 1))
            verdict = self.llm.structured(
                "extractor",
                [
                    {"role": "system", "content": REGROWN_SYSTEM},
                    {
                        "role": "user",
                        "content": f'Deleted fact: "{statement}"\n\nRemaining facts:\n{listing}',
                    },
                ],
                RegrownOutput,
                cache_scope=scope(space_id, "consolidate"),
            ).value
            for n in verdict.same:
                if 1 <= n <= len(candidates):
                    found.setdefault(candidates[n - 1].id, candidates[n - 1])
        return list(found.values())

    def round_turns(self, space_id: str, turn_ids: Sequence[str]) -> list[str]:
        """把发言扩展成所在的整轮（用户的话 + 助手的回复）。遗忘一条事实或一张图时要删整轮：
        助手在同一轮的回复里往往复述了同样的信息（"Ana 十月三号生日快乐"），只删用户那句，
        之后照样检索得到。"""
        wanted = set(turn_ids)
        out = list(dict.fromkeys(turn_ids))
        for unit in self.db.list_units(space_id, kinds=["round"]):
            if wanted & set(unit.source_turn_ids):
                out.extend(t for t in unit.source_turn_ids if t not in out)
        return out

    # ---------- 读取 ----------

    def search(self, space_id: str, query: str, k: int | None = None) -> list[Hit]:
        return self.retriever.search(space_id, query, k or self.settings.retrieval.naive_k)

    def recall(
        self,
        space_id: str,
        query: str,
        *,
        k: int | None = None,
        fact_k: int | None = None,
        fact_keys: bool = True,
        current_only: bool = False,
        chain: bool = True,
    ) -> MemoryView:
        cfg = self.settings.retrieval
        return self.fact_retriever.search(
            space_id,
            query,
            k=k or cfg.naive_k,
            fact_k=fact_k or cfg.fact_k,
            pool=cfg.fact_pool,
            fact_keys=fact_keys,
            current_only=current_only,
            chain=chain,
        )

    def ask(
        self,
        space_id: str,
        question: str,
        *,
        question_time: datetime | None = None,
        k: int | None = None,
        mode: AskMode = "l1",
        validity: Validity = "full",
        show_facts: bool = True,
        profile: bool = True,
        fact_keys: bool = True,
        l1: L1Config | None = None,
        relative_time: bool = True,
        l2: L2Config | None = None,
        notes: bool | None = None,
        router: Router = "learned",
        lam: float | None = None,
    ) -> Answer:
        """检索并回答。

        - mode="l0"：只用常驻上下文（核心记忆，即用户画像），不检索（§6.1，N7 控制器的动作之一）；
        - mode="l1"（默认）：L1 快速召回——多粒度混合检索、时间感知、时间感知打包（N4）；
        - mode="l2"：L2 深度检索——查询规划、多查询召回、图扩散、充分性检查与补检索（N5）；
        - mode="auto"：检索控制器选档（N7）。router="learned"（默认）：探针特征 → 两段式控制器，
          在 L0 / L1 / L2 与"要不要再看原图"之间按 P̂(答对) − λ·成本选，lam 不填用模型里的默认值；
          "hybrid"：门控拿不准时再问 LLM；"llm"：零样本 LLM 路由；"rule"：规则版（B6r，N5）；
        - mode="agent"：模型自主检索（B3）——reader 带 memory_search 工具自己决定检索；
        - mode="hatch"：控制器先选档召回，再交给带 memory_search 工具的 reader，缺什么自己补查
          （N8 的 Agent 做法，B6a，D-042）；
        - mode="facts"：B2F，dense 检索 + 事实增强键 + 带有效期的事实 + 核心记忆（N3）；
        - mode="naive"：B2 基线，只在"轮"上做 dense 检索。
        notes：reader 先记笔记再回答（并按检索计划给阅读提示）；默认 L2 开、L1 关。
        还没巩固过的空间，l1 与 facts 模式也能用（只是没有事实可用）。
        """
        when = question_time or datetime.now()
        cache_scope = scope(space_id, "ask")
        with start_trace("ask", self.tracer, space=space_id, question=question, mode=mode) as trace:
            recall: Recall | None = None
            if mode in ("l0", "l1", "l2", "auto"):
                recall = self.retrieve(
                    space_id,
                    question,
                    question_time=when,
                    mode=mode,
                    router=router,
                    lam=lam,
                    k=k,
                    l1=l1,
                    l2=l2,
                    cache_scope=cache_scope,
                )
            if recall is not None and recall.level == "l0":
                answer = self.reader.answer_with_memory(
                    question, recall.view, question_time=when, cache_scope=cache_scope
                )
            elif recall is not None and recall.level == "l1":
                if notes:  # 笔记式阅读用规则版计划给阅读提示（不调用规划模型）
                    answer = self.reader.answer_deep(
                        question,
                        recall.view,
                        plan=rule_plan(question, when),
                        question_time=when,
                        cache_scope=cache_scope,
                    )
                else:
                    answer = self.reader.answer_with_memory(
                        question,
                        recall.view,
                        question_time=when,
                        show_facts=show_facts,
                        profile=profile,
                        relative_time=relative_time,
                        cache_scope=cache_scope,
                    )
            elif recall is not None:  # L2
                assert recall.plan is not None
                if notes is None or notes:
                    answer = self.reader.answer_deep(
                        question,
                        recall.view,
                        plan=recall.plan,
                        question_time=when,
                        cache_scope=cache_scope,
                    )
                else:
                    answer = self.reader.answer_with_memory(
                        question,
                        recall.view,
                        question_time=when,
                        relative_time=True,
                        cache_scope=cache_scope,
                    )
                answer.details = dict(recall.details)
            elif mode in ("agent", "hatch"):
                if mode == "hatch":  # 控制器先选档召回，模型可以再用 memory_search 补查（B6a）
                    recall = self.retrieve(
                        space_id,
                        question,
                        question_time=when,
                        mode="auto",
                        router=router,
                        lam=lam,
                        k=k,
                        l1=l1,
                        l2=l2,
                        cache_scope=cache_scope,
                    )
                answer = self.agent.answer(
                    question,
                    question_time=when,
                    search=lambda query, cfg: self.l1.search(space_id, query, when, cfg),
                    core=self.db.get_core_memory(space_id),
                    cache_scope=cache_scope,
                    recalled=recall.view if recall is not None else None,
                )
            elif mode == "naive":
                hits = self.search(space_id, question, k)
                answer = self.reader.answer(
                    question, hits, question_time=when, cache_scope=cache_scope
                )
            else:
                view = self.recall(
                    space_id,
                    question,
                    k=k,
                    fact_keys=fact_keys,
                    current_only=validity == "current",
                    chain=validity == "full",
                )
                answer = self.reader.answer_with_memory(
                    question,
                    view,
                    question_time=when,
                    validity=validity,
                    show_facts=show_facts,
                    profile=profile,
                    cache_scope=cache_scope,
                )
            if recall is not None and recall.decision is not None:
                answer.details["route"] = route_details(recall)
            if recall is not None and recall.level != "l0" and recall.view.attach:
                answer.details["attached"] = list(recall.view.attach)  # 附了原图的图片（再感知）
            trace.attrs.update(answer=answer.answer, citations=answer.citations)
        return answer

    def retrieve(
        self,
        space_id: str,
        question: str,
        *,
        question_time: datetime | None = None,
        mode: RecallMode = "auto",
        router: Router = "learned",
        lam: float | None = None,
        k: int | None = None,
        l1: L1Config | None = None,
        l2: L2Config | None = None,
        exclude: Collection[str] = (),
        cache_scope: str | None = None,
    ) -> Recall:
        """只检索、不回答：按 mode 取记忆（auto 由检索控制器选档），返回交给回答模型的材料。

        ask() 与 Agent（N8）共用这一步。exclude 是已经在上下文里的轮与图片的 ID
        （Agent 的工作记忆），不参与探针与检索。
        """
        when = question_time or datetime.now()
        cache_scope = cache_scope or scope(space_id, "ask")
        level: str = mode
        decision: RouteDecision | None = None
        used: Router | None = None
        candidates: L1Candidates | None = None
        if mode == "auto":
            decision, candidates, used = self._route(
                space_id, question, when, l1 or L1Config(), router, lam, cache_scope, exclude
            )
            level = decision.level.lower()
            if decision.see:  # 再感知：回答时附上最相关的 3 张原图（N6）
                l1 = replace(l1 or L1Config(), attach_images=3)
                base_l2 = l2 or L2Config()
                l2 = replace(base_l2, l1=replace(base_l2.l1, attach_images=3))
        if level == "l0":
            view = MemoryView(rounds=[], facts=[], core=self.db.get_core_memory(space_id))
            return Recall("l0", view, decision, used)
        if level == "l1":
            cfg = l1 or L1Config(k=k or self.settings.retrieval.naive_k)
            if candidates is not None:  # 探针已经打过分：直接打包，不重复检索
                result = self.l1.pack(space_id, candidates, cfg)
            else:
                result = self.l1.search(space_id, question, when, cfg, exclude=exclude)
            view = MemoryView(
                rounds=result.rounds,
                facts=result.facts,
                core=result.core,
                media=result.media,
                image_mode=image_mode(cfg),
                attach=result.attach,
            )
            return Recall("l1", view, decision, used)
        l2_cfg = l2 or L2Config()
        deep = self.l2.search(
            space_id, question, when, l2_cfg, cache_scope=cache_scope, exclude=exclude
        )
        view = MemoryView(
            rounds=deep.rounds,
            facts=deep.facts,
            core=deep.core,
            media=deep.media,
            image_mode=image_mode(l2_cfg.l1),
            attach=deep.attach,
        )
        return Recall("l2", view, decision, used, plan=deep.plan, details=deep.details())

    def _route(
        self,
        space_id: str,
        question: str,
        when: datetime,
        cfg: L1Config,
        router: Router,
        lam: float | None,
        cache_scope: str,
        exclude: Collection[str] = (),
    ) -> tuple[RouteDecision, L1Candidates | None, Router]:
        """检索控制器选档；学到的控制器先跑探针，选了 L1 时把探针的候选交回去直接打包。

        返回（决定，探针的候选，实际用的控制器）：没有模型文件时学到的控制器退回规则版。
        """
        if router == "llm":
            return self.router.decide(question, cache_scope=cache_scope), None, "llm"
        if router == "rule" or self.learned is None:
            return self.controller.decide(question, when), None, "rule"
        probe = self.probe.run(space_id, question, when, cfg, exclude=exclude)
        has_images = probe.features.get("m_media", 0.0) > 0
        p_none: float | None = None
        asked = False
        if router == "hybrid":
            gate = self.learned.gate(probe.features)
            if HYBRID_BAND[0] < gate < HYBRID_BAND[1]:  # 门控拿不准：问一次 LLM
                asked = True
                llm = self.router.decide(question, cache_scope=cache_scope)
                p_none = HYBRID_CONFIDENCE if llm.level == "L0" else 1 - HYBRID_CONFIDENCE
        decision = self.learned.decide(
            probe.features, has_images=has_images, lam=lam, p_none=p_none
        )
        scores = {**decision.scores, "probe_ms": round(probe.ms, 1)}
        if asked:
            scores["asked_llm"] = 1.0
        return replace(decision, scores=scores), probe.candidates, router

    def search_by_image(
        self, space_id: str, image: Path, k: int = 5
    ) -> list[tuple[MediaItem, float]]:
        """以图搜图（N6）："我之前给你看过这个吗？"——返回最相似的图片记录与余弦相似度。"""
        if self.image_embedder is None:
            return []
        vector = self.image_embedder.embed_images([image])[0]
        hits = self.vectors.search_images(space_id, vector, k)
        items = {m.id: m for m in self.db.get_media(space_id, ids=[h.unit_id for h in hits])}
        return [(items[h.unit_id], h.score) for h in hits if h.unit_id in items]

    # ---------- 检查（记忆检查器 CLI 与之后的界面用） ----------

    def facts(
        self,
        space_id: str,
        *,
        current_only: bool = False,
        valid_at: str | None = None,
        known_at: datetime | None = None,
    ) -> list[Fact]:
        return self.db.get_facts(
            space_id, current_only=current_only, valid_at=valid_at, known_at=known_at
        )

    def entities(self, space_id: str) -> list[Entity]:
        return self.db.get_entities(space_id)

    def core_memory(self, space_id: str) -> CoreMemory | None:
        return self.db.get_core_memory(space_id)

    def close(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.vectors.close()
        self.db.close()
        if self._owns_ledger:
            self.ledger.close()

    def __enter__(self) -> MemorySystem:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
