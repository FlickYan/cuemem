"""MemorySystem：把存储、向量化、写入、检索、回答组装在一起的门面。

上层（CLI、评测 runner、之后的 Agent 与 API 服务）只和它打交道。
"""

from __future__ import annotations

from datetime import datetime
from types import TracebackType

import httpx2

from cuemem.config import Settings
from cuemem.core.models import Answer, Conversation, Hit
from cuemem.core.trace import TraceSink, start_trace
from cuemem.embed import build_embedder
from cuemem.embed.base import TextEmbedder
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.read.naive import NaiveRetriever
from cuemem.read.reader import Reader
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore
from cuemem.write.online import IngestReport, OnlineWriter


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
    ) -> None:
        self.settings = settings
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.llm = llm
        self.ledger = ledger
        self.tracer = tracer
        self.writer = OnlineWriter(db, vectors, embedder)
        self.retriever = NaiveRetriever(db, vectors, embedder)
        self.reader = Reader(llm)

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

    def ingest(self, conversation: Conversation) -> IngestReport:
        with start_trace("ingest", self.tracer, space=conversation.space_id) as trace:
            report = self.writer.ingest(conversation)
            trace.attrs.update(units_added=report.units_added, turns_added=report.turns_added)
        return report

    def search(self, space_id: str, query: str, k: int | None = None) -> list[Hit]:
        return self.retriever.search(space_id, query, k or self.settings.retrieval.naive_k)

    def ask(
        self,
        space_id: str,
        question: str,
        *,
        question_time: datetime | None = None,
        k: int | None = None,
    ) -> Answer:
        when = question_time or datetime.now()
        with start_trace("ask", self.tracer, space=space_id, question=question) as trace:
            hits = self.search(space_id, question, k)
            answer = self.reader.answer(question, hits, question_time=when)
            trace.attrs.update(answer=answer.answer, citations=answer.citations)
        return answer

    def close(self) -> None:
        self.vectors.close()
        self.db.close()
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
