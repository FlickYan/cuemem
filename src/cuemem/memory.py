"""MemorySystem：把存储、向量化、写入、巩固、检索、回答组装在一起的门面。

上层（CLI、评测 runner、之后的 Agent 与 API 服务）只和它打交道。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Literal

import httpx2

from cuemem.config import Settings
from cuemem.core.models import Answer, Conversation, CoreMemory, Entity, Fact, Hit
from cuemem.core.trace import TraceSink, start_trace
from cuemem.embed import CachedEmbedder, build_embedder
from cuemem.embed.base import TextEmbedder
from cuemem.errors import ConsolidationError
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.read.facts import FactAugmentedRetriever, MemoryView, Validity
from cuemem.read.l1 import L1Config, L1Retriever
from cuemem.read.naive import NaiveRetriever
from cuemem.read.reader import Reader
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore
from cuemem.write.consolidate import (
    ConsolidationReport,
    ConsolidationWorker,
    Consolidator,
    scope,
)
from cuemem.write.online import IngestReport, OnlineWriter

AskMode = Literal["l1", "facts", "naive"]


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
    ) -> None:
        self.settings = settings
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.llm = llm
        self.ledger = ledger
        self.tracer = tracer
        self._owns_ledger = owns_ledger  # 评测时账本由多个线程共享，不能由单个系统关闭
        self.writer = OnlineWriter(db, vectors, embedder)
        self.consolidator = Consolidator(db, vectors, embedder, llm, settings.consolidation)
        self.retriever = NaiveRetriever(db, vectors, embedder)
        self.fact_retriever = FactAugmentedRetriever(db, vectors, embedder)
        self.l1 = L1Retriever(db, vectors, embedder)
        self.reader = Reader(llm)
        self.worker: ConsolidationWorker | None = None

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
        )

    @classmethod
    def ephemeral(
        cls, settings: Settings, *, embedder: TextEmbedder, llm: LLMClient, ledger: Ledger
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
        )

    # ---------- 写入 ----------

    def ingest(self, conversation: Conversation, *, consolidate: bool = False) -> IngestReport:
        """在线写入（不调用 LLM），并为新会话登记巩固任务。

        consolidate=True 时立即同步巩固；否则如果后台线程已启动，就交给它异步完成。
        """
        with start_trace("ingest", self.tracer, space=conversation.space_id) as trace:
            report = self.writer.ingest(conversation)
            queued = self.consolidator.enqueue(conversation.space_id)
            trace.attrs.update(
                units_added=report.units_added, turns_added=report.turns_added, jobs=queued
            )
        if consolidate:
            self.consolidate(conversation.space_id)
        elif self.worker is not None:
            self.worker.submit(conversation.space_id)
        return report

    def consolidate(
        self, space_id: str, *, rebuild: bool = False, strict: bool = False
    ) -> ConsolidationReport:
        """执行待处理的巩固任务。rebuild=True（或抽取逻辑版本变化）时先清空派生记忆再全部重放。

        strict=True 时有会话抽取失败就抛出异常（评测用：宁可整题重试，也不要残缺的记忆）。
        """
        with start_trace("consolidate", self.tracer, space=space_id) as trace:
            if rebuild or self.consolidator.stale(space_id):
                self.consolidator.rebuild(space_id)
            self.db.retry_failed_jobs(space_id)
            self.consolidator.enqueue(space_id)
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
    ) -> ForgetReport:
        """遗忘（被遗忘权）：删除原始轮次，并清除由它们派生出的一切（D-021）。

        派生记忆是原始日志的函数，所以遗忘 = 从日志里删除 + 用剩下的日志重放派生记忆：
        事实、实体、摘要、核心记忆都重新生成；被删内容取代过的旧事实会自动恢复为当前有效。
        随后清理缓存：被删会话的抽取结果、该空间的对账与回答缓存、不再出现的文本的向量。
        """
        started = time.perf_counter()
        with start_trace("forget", self.tracer, space=space_id) as trace:
            wanted_turns, wanted_sessions = set(turn_ids), set(session_ids)
            targets = [
                t
                for t in self.db.get_turns(space_id)
                if t.id in wanted_turns or t.session_id in wanted_sessions
            ]
            affected = sorted({t.session_id for t in targets})
            if not targets:
                return ForgetReport(space_id, 0, [], 0, 0, 0, 0, 0.0)
            consolidated = bool(self.db.get_jobs(space_id))
            facts_before = len(self.db.get_facts(space_id))
            texts_before = {u.text for u in self.db.list_units(space_id)}

            # 1. 原始层：删除轮次，按剩下的轮次重建受影响会话的"轮"
            old_rounds = self.db.list_units(space_id, kinds=["round"], session_ids=affected)
            self.vectors.delete(space_id, [u.id for u in old_rounds])
            self.db.delete_units(space_id, [u.id for u in old_rounds])
            deleted = self.db.delete_turns(space_id, [t.id for t in targets])
            self.writer.reindex_sessions(space_id, affected)

            # 2. 派生层：清空并清理相关的 LLM 缓存，再从剩下的日志重放
            removed = self.db.clear_derived(space_id)
            self.vectors.delete(space_id, removed)
            llm_purged = 0
            if self.llm.cache is not None:
                parts = [f"session:{sid}" for sid in affected] + ["consolidate", "ask"]
                llm_purged = sum(self.llm.cache.delete_scope(scope(space_id, p)) for p in parts)
            if consolidated:
                self.consolidator.enqueue(space_id)
                self.consolidator.run(space_id)

            # 3. 向量缓存：删除已不在记忆里的文本（向量本身也可能泄露原文）
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
            )
            trace.attrs.update(turns_deleted=deleted, sessions=affected)
        return report

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
    ) -> Answer:
        """检索并回答。

        - mode="l1"（默认）：L1 快速召回——多粒度混合检索、时间感知、时间感知打包（N4）；
        - mode="facts"：B2F，dense 检索 + 事实增强键 + 带有效期的事实 + 核心记忆（N3）；
        - mode="naive"：B2 基线，只在"轮"上做 dense 检索。
        还没巩固过的空间，l1 与 facts 模式也能用（只是没有事实可用）。
        """
        when = question_time or datetime.now()
        cache_scope = scope(space_id, "ask")
        with start_trace("ask", self.tracer, space=space_id, question=question, mode=mode) as trace:
            if mode == "l1":
                cfg = l1 or L1Config(k=k or self.settings.retrieval.naive_k)
                result = self.l1.search(space_id, question, when, cfg)
                answer = self.reader.answer_with_memory(
                    question,
                    MemoryView(rounds=result.rounds, facts=result.facts, core=result.core),
                    question_time=when,
                    show_facts=show_facts,
                    profile=profile,
                    relative_time=relative_time,
                    cache_scope=cache_scope,
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
            trace.attrs.update(answer=answer.answer, citations=answer.citations)
        return answer

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
