"""MemorySystem：把存储、向量化、写入、巩固、检索、回答组装在一起的门面。

上层（CLI、评测 runner、之后的 Agent 与 API 服务）只和它打交道。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
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
    Turn,
)
from cuemem.core.timeutil import naive_local
from cuemem.core.trace import TraceSink, current_trace, start_trace
from cuemem.core.types import AskMode, RecallMode
from cuemem.embed import CachedEmbedder, build_embedder, build_image_embedder
from cuemem.embed.base import TextEmbedder
from cuemem.embed.multimodal import DashScopeMultimodalEmbedder, ImageEmbedder
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
    MAX_ATTEMPTS,
    ConsolidationReport,
    ConsolidationWorker,
    Consolidator,
    scope,
)
from cuemem.write.extract import PIPELINE_VERSION
from cuemem.write.online import IngestReport, OnlineWriter

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
    # 学到的控制器的探针打过分的候选（mode="auto" 时）：L0 改走 L1 时直接打包（见 retrieve）
    candidates: L1Candidates | None = None


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


REGROW_ROUNDS = 3  # 遗忘后追查复述的轮数上限（每轮要重建一次派生记忆）


class RegrownOutput(BaseModel):
    same: list[int] = Field(default_factory=list)


def controller_path(name: str) -> Path:
    """检索控制器模型文件（retrieval.controller）的位置：相对路径相对 src/cuemem/read/。"""
    path = Path(name)
    return path if path.is_absolute() else DEFAULT_MODEL.parent / path


def load_controller(settings: Settings) -> LearnedController | None:
    """按配置（retrieval.controller）加载检索控制器的模型文件；没有时返回 None（退回规则版）。"""
    path = controller_path(settings.retrieval.controller)
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
    name = settings.rerank.provider
    provider = settings.providers.get(name) if name is not None else None
    api_key = provider.api_key() if provider is not None else None
    if api_key is None:
        return None
    return Reranker(
        settings.rerank, api_key, http_client=http_client, cache=llm.cache, ledger=ledger
    )


def _strings(value: Any) -> Iterator[str]:
    """JSON 结构里的全部字符串（trace 的 spans、events 里按 ID 记着检索到的记忆）。"""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _without_images(text: str) -> str:
    """发言原文去掉图片标记、合并空白：记忆代理拿用户消息的文字部分（evidence.message_text）
    做检索查询，写进日志的却是带 "<image>" 标记的原话（evidence.stored_text）。"""
    return " ".join(text.replace("<image>", " ").split())


class _ReadGate:
    """一个空间的读写门（见 MemorySystem.reading）：读取可以同时进行，遗忘独占。遗忘先等进行中的
    读取做完，等着的时候新来的读取排在它后面（不会一直等不到）。同一线程里读可以重入（ask 里的
    检索、Agent 一轮对话里的检索），遗忘里的读取直接通过，删除空间里的遗忘可以重入。"""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._readers = 0  # 持有读的线程数
        self._writer: int | None = None  # 持有写的线程
        self._writes = 0  # 写的重入深度
        self._waiting = 0  # 等着写的线程数
        self._local = threading.local()  # 本线程读的重入深度

    @contextmanager
    def shared(self) -> Iterator[None]:
        if self._writer == threading.get_ident():  # 遗忘自己要读：已经独占了
            yield
            return
        depth = getattr(self._local, "depth", 0)
        if depth == 0:
            with self._cond:
                while self._writer is not None or self._waiting:
                    self._cond.wait()
                self._readers += 1
        self._local.depth = depth + 1
        try:
            yield
        finally:
            self._local.depth = depth
            if depth == 0:
                with self._cond:
                    self._readers -= 1
                    self._cond.notify_all()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                self._writes += 1
            else:
                own = int(getattr(self._local, "depth", 0) > 0)  # 本线程自己持有的读不用等
                self._waiting += 1
                try:
                    while self._writer is not None or self._readers > own:
                        self._cond.wait()
                finally:
                    self._waiting -= 1
                self._writer, self._writes = me, 1
        try:
            yield
        finally:
            with self._cond:
                self._writes -= 1
                if self._writes == 0:
                    self._writer = None
                    self._cond.notify_all()


def _open_mark(space_id: str, session_id: str = "") -> str:
    """进行中的会话在数据库里的标记（元信息的键）；session_id 为空时是这个空间全部标记的
    共同前缀。空间 ID 带上长度：一个空间的前缀不会同时是另一个空间的前缀。"""
    return f"open-session:{len(space_id)}:{space_id}:{session_id}"


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
        owns_caches: bool = False,
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
        # LLM 缓存与向量缓存只在自己建的时候（from_settings）随系统关闭：评测时多个系统共用
        self._owns_caches = owns_caches
        self.media = media
        self.image_embedder = image_embedder
        self.writer = OnlineWriter(
            db, vectors, embedder, media=media, image_embedder=image_embedder
        )
        self.consolidator = Consolidator(
            db, vectors, embedder, llm, settings.consolidation, open_sessions=self.open_sessions
        )
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
        self._worker_lock = threading.Lock()
        # 进行中的会话（Agent 正在对话，N8）：暂不巩固，会话结束后再整体抽取一次
        self._open: set[tuple[str, str]] = set()
        self._open_lock = threading.Lock()
        # 同一会话的写入按顺序进行（SDK 的 add 在锁里分配发言序号）。按（空间，会话）分到固定的
        # 几十把锁上，不为每个会话建一把：不给 run_id 的 add 每次都是一个新会话
        self._session_locks = [threading.Lock() for _ in range(64)]
        self._gates: dict[str, _ReadGate] = {}  # 空间 → 读写门（见 reading）
        self._gates_guard = threading.Lock()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        http_client: httpx2.Client | None = None,
        on_peak_wait: Callable[[str, datetime], None] | None = None,
    ) -> MemorySystem:
        """按配置组装各组件。http_client 用于测试时注入模拟的网络层；on_peak_wait：wait 策略下
        遇到高峰、开始暂停时通知调用方（每段高峰每个供应商一次）。"""
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
            on_peak_wait=on_peak_wait,
        )
        return cls(
            settings,
            db=db,
            vectors=VectorStore(paths.qdrant, dim=embedder.dim, url=settings.vector_store.url),
            embedder=embedder,
            llm=llm,
            ledger=ledger,
            tracer=TraceSink(paths.traces),
            owns_caches=True,
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
        sessions = [(s, 0) for s in conversation.sessions]
        skip = self.open_sessions(space_id)  # 在事务之外取：open_session 持有它的锁时写数据库
        with start_trace("ingest", self.tracer, space=space_id) as trace:
            # 巩固任务与发言在同一个事务里提交：向量化出错时什么都不留（见 OnlineWriter.writing）
            with self.writer.writing(space_id, [sessions]) as report:
                queued = self.consolidator.enqueue(space_id, skip=skip)
            trace.attrs.update(
                units_added=report.units_added, turns_added=report.turns_added, jobs=queued
            )
        if consolidate:
            self.consolidate(space_id)
        elif self.worker is not None and queued:
            self.worker.submit(space_id)
        return report

    def append(
        self, space_id: str, session: SessionIn, first_idx: int, *, enqueue: bool = True
    ) -> IngestReport:
        """往会话里追加发言（Agent 每轮写回新的一问一答，序号从 first_idx 起），在线写入、
        不调用 LLM。会话进行中不登记巩固任务（见 open_session）；enqueue=False 时也不登记
        （SDK 的 infer=False：只记日志，下次 consolidate 时再抽取）。

        要么全部写进去，要么什么都不留：向量化或写向量出错时抛出异常、数据库里什么都没写，
        调用方可以原样重试。"""
        with self.appending(space_id, [(session, first_idx)], enqueue=enqueue) as report:
            return report

    @contextmanager
    def appending(
        self, space_id: str, pieces: Sequence[tuple[SessionIn, int]], *, enqueue: bool = True
    ) -> Iterator[IngestReport]:
        """同 append，pieces 是若干段（会话，第一句的序号），每段各向量化一次（导入文件时每段
        连续的发言一段，与以前逐段追加时的请求相同）。with 块里的写入（Agent 的溯源）与发言、
        巩固任务在同一个事务里提交，块里出错时一起撤销（块里只做数据库的写入：事务持有数据库
        的锁）。"""
        sessions = list(dict.fromkeys(s.session_id for s, _ in pieces))
        names = {"session": sessions[0]} if len(sessions) == 1 else {"sessions": sessions}
        skip = self.open_sessions(space_id)  # 在事务之外取：open_session 持有它的锁时写数据库
        queued = 0
        with (
            start_trace("append", self.tracer, space=space_id, **names),
            self.writer.writing(space_id, [[piece] for piece in pieces]) as report,
        ):
            if enqueue:
                queued = self.consolidator.enqueue(space_id, sessions, skip=skip)
            yield report
        if queued and self.worker is not None:
            self.worker.submit(space_id)

    # ---------- 进行中的会话（Agent，N8） ----------

    def open_session(self, space_id: str, session_id: str) -> None:
        """标记会话进行中：每轮照常在线写入（可检索），但不登记巩固任务——
        否则半个会话会被抽取一遍，会话结束后整个会话又抽取一遍。

        数据库里另外留一个标记：进程崩溃时没人再结束这个会话，下一个进程据此接手（adopt_open）。"""
        with self._open_lock:
            if (space_id, session_id) not in self._open:
                self._open.add((space_id, session_id))
                self.db.set_meta(_open_mark(space_id, session_id), datetime.now().isoformat())

    def close_session(
        self, space_id: str, session_id: str, *, wait: bool = False, submit: bool = True
    ) -> ConsolidationReport | None:
        """会话结束：登记巩固任务；wait=True 时同步巩固并返回报告，否则交给后台线程
        （submit=False 时只登记，下次巩固或后台线程启动时完成）。"""
        with self._open_lock:
            self._open.discard((space_id, session_id))
            self.db.delete_meta(_open_mark(space_id, session_id))
        self.consolidator.enqueue(space_id, [session_id])
        if wait:
            return self.consolidate(space_id)
        if submit:
            self.start_worker().submit(space_id)
        return None

    def open_sessions(self, space_id: str) -> set[str]:
        with self._open_lock:
            return {sid for space, sid in self._open if space == space_id}

    def session_lock(self, space_id: str, session_id: str) -> threading.Lock:
        """往这个会话里写入时持有的锁：从分配发言序号到写完为止。放在系统上而不是 SDK 的
        Memory 上：HTTP 服务、记忆代理与各个集成用 Memory.from_system 包住的是同一个系统。"""
        return self._session_locks[hash((space_id, session_id)) % len(self._session_locks)]

    def reading(self, space_id: str) -> AbstractContextManager[None]:
        """读这个空间的记忆、并把结果写下来（回答写进 LLM 缓存与 trace，Agent 的回复写进日志）时
        持有：可以同时有多个。遗忘等进行中的读取都做完才开始，期间新的读取等它做完——否则回答
        模型还在生成时遗忘清掉了缓存与 trace，回答随后又带着被删的内容写进去。同一线程里可以
        重入。"""
        return self._gate(space_id).shared()

    def _gate(self, space_id: str) -> _ReadGate:
        with self._gates_guard:
            return self._gates.setdefault(space_id, _ReadGate())

    def adopt_open(self, space_id: str, keep: Collection[str] = (), *, start: bool = True) -> int:
        """接手之前的进程留下的进行中的会话：进程崩溃（没来得及 close_session）时，它们没有
        登记巩固任务，也没人再结束它们。数据库里有标记、本进程里却不在进行中的会话（keep 除外：
        调用方马上要接着写），登记任务、交给后台线程，去掉标记。返回接手的会话数。
        start=False：没有后台线程时只登记、不为此启动它（SDK 的 background=False）。

        只认标记，不碰没有任务的其他会话：SDK 的 infer=False 写入的会话本来就不该抽取。
        记忆代理、SDK 与 HTTP 服务第一次碰到一个空间时各调用一次（同一个数据库同时只有一个
        进程在写时，本进程里不在进行中的就是崩溃留下的）。"""
        prefix = _open_mark(space_id)
        with self._open_lock:
            live = {sid for space, sid in self._open if space == space_id} | set(keep)
            marked = [key.removeprefix(prefix) for key in self.db.meta_keys(prefix)]
            leftover = [sid for sid in marked if sid not in live]
            if not leftover:
                return 0
            self.consolidator.enqueue(space_id, leftover)
            for sid in leftover:
                self.db.delete_meta(_open_mark(space_id, sid))
        if start or self.worker is not None:
            self.start_worker().submit(space_id)
        return len(leftover)

    def consolidate(
        self, space_id: str, *, rebuild: bool = False, strict: bool = False
    ) -> ConsolidationReport:
        """执行待处理的巩固任务。rebuild=True（或抽取逻辑版本变化）时先清空派生记忆再全部重放。

        strict=True 时有会话抽取失败就抛出异常（评测用：宁可整题重试，也不要残缺的记忆）。

        进行中的会话拿到锁之后再读（重建时在清空之后读，见 Consolidator.rebuild）：等锁可能要等
        后台线程做完一整轮巩固，这期间新开的会话不能被抽取半段，结束的会话也不能没有任务。
        """
        with (
            self.consolidator.exclusive(space_id),  # 清空重建不能和后台线程的巩固交错
            start_trace("consolidate", self.tracer, space=space_id) as trace,
        ):
            if rebuild or self.consolidator.stale(space_id):
                self.consolidator.rebuild(space_id)
            self.db.retry_failed_jobs(space_id)
            self.consolidator.enqueue(space_id, skip=self.open_sessions(space_id))
            report = self.consolidator.run(space_id)
            trace.attrs.update(report.as_dict())
        if strict and report.failed:
            raise ConsolidationError(f"{len(report.failed)} 个会话抽取失败：{report.failed[:5]}")
        return report

    def start_worker(self) -> ConsolidationWorker:
        """启动后台巩固线程：此后 ingest 只登记任务，巩固在后台完成。

        第一次启动时接着做上次没做完的：进程退出、崩溃或关闭超时时还待处理的任务，以及失败
        次数还没到上限的任务，所在的空间都提交一遍。"""
        with self._worker_lock:
            if self.worker is None:
                self.worker = ConsolidationWorker(self.consolidator, tracer=self.tracer)
                for space_id in self.db.spaces_with_work(PIPELINE_VERSION, MAX_ATTEMPTS):
                    self.worker.submit(space_id)
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
        同时清理缓存与 trace：被删会话的抽取结果、该空间的回答缓存、不再出现的文本的向量、
        被删发言的查询向量，以及含有被删原话或看过被删记忆的 trace（连同其提问的查询向量）。

        对账与实体消歧的缓存（请求里列着别的会话的事实与实体）不再整个删掉、重放时全部重新
        请求：重放之后只留下重放时原样用到的条目，其余删掉。重放的每个请求都是从剩下的日志
        算出来的，缓存键是请求的哈希，命中就说明条目的请求与它逐字相同，里面不会有被删的
        内容；列过被删的事实或实体、或者剩下的日志已经算不出来的条目，重放用不到，都删掉。
        这样只有输入真的变了的对账与消歧（被删会话之后、候选里有过它的内容的那些）要重新请求。
        不按会话给条目打标签：实体的描述可能来自好几个会话，库里没有记它的出处。

        follow_citations（N8）：助手后来的回复如果引用过要删的记忆，它就复述了被删的信息，
        那一轮也一起删，并继续往下追（溯源表记录了每次回复引用了什么）。

        restated_turn_ids（D-051）：按内容找到的转述，一起删，但不从它们出发沿引用扩散。
        它们只是顺带透露了被删的信息，所在那一轮里别的内容（用户在同一轮说的另一件事、
        由此抽出的事实、后来引用这些事实的回复）不是要删的。

        整个过程持有这个空间的巩固锁（Consolidator.exclusive）：后台线程正在巩固时先等它做完，
        否则它会把由被删轮次抽出的事实、向量与缓存写回来。进行中的回答、检索与 Agent 的对话
        也先等它们做完（reading），否则它们随后把看过的被删内容写进缓存、trace 与日志。

        重放失败（例如对账时余额不足）时照样清理 trace、向量缓存与 LLM 缓存（标记的条目只留下
        重放已经用到的），再把异常抛出去：原始日志已经删了，重试时找不到要删的发言。没做完的
        巩固任务留着，下次巩固时补齐。
        """
        started = time.perf_counter()
        with (
            self._gate(space_id).exclusive(),  # 先于巩固锁：读取不会去拿巩固锁
            self.consolidator.exclusive(space_id),
            start_trace("forget", self.tracer, space=space_id) as trace,
        ):
            targets, cascaded = self._forget_targets(
                space_id, session_ids, turn_ids, follow_citations, restated_turn_ids
            )
            affected = sorted({t.session_id for t in targets})
            if not targets:
                return ForgetReport(space_id, 0, [], 0, 0, 0, 0, 0.0)
            consolidated = bool(self.db.get_jobs(space_id))
            facts = self.db.get_facts(space_id)
            texts_before = {u.text for u in self.db.list_units(space_id)}
            doomed = {t.id for t in targets}

            # 0. LLM 缓存，在删日志之前：受影响会话的抽取与回答、遗忘时的复述检查（见
            #    restating_turns）直接删；对账与消歧的条目先标记，重放之后只留下重放时原样用到的
            #    （见上面的说明）。标记写在库里，进程中途退出时下次打开缓存就全部删掉
            cache = self.llm.cache
            held = scope(space_id, "consolidate")
            reused: set[str] = set()  # 重放时命中过的缓存键
            llm_purged = 0
            if cache is not None:
                parts = [f"session:{sid}" for sid in affected] + ["ask", "forget"]
                llm_purged = sum(cache.delete_scope(scope(space_id, p)) for p in parts)
                cache.hold(held)
            try:
                # 1. 原始层：删除轮次与其中的图片（记录、图片向量；没有别处引用的原图文件），
                #    再按剩下的轮次重建受影响会话的"轮"
                old_rounds = self.db.list_units(space_id, kinds=["round"], session_ids=affected)
                self.vectors.delete(space_id, [u.id for u in old_rounds])
                self.db.delete_units(space_id, [u.id for u in old_rounds])
                doomed_media = self.db.get_media(space_id, turn_ids=[t.id for t in targets])
                # 被删内容的记忆 ID：含有被删发言的轮、由它们抽出的事实、其中的图片（trace 按 ID
                # 记着回答模型看过哪些记忆）
                refs = {u.id for u in old_rounds if doomed & set(u.source_turn_ids)}
                refs |= {f.id for f in facts if doomed & set(f.source_turn_ids)}
                refs |= {m.id for m in doomed_media}
                deleted = self.db.delete_turns(space_id, [t.id for t in targets])
                self.vectors.delete_images(space_id, [m.id for m in doomed_media])
                images_deleted = 0
                for item in doomed_media:
                    if self.media is not None and not self.db.media_sha_in_use(item.sha256):
                        images_deleted += int(self.media.delete(Path(item.path)))
                self.writer.reindex_sessions(space_id, affected)

                # 2. 派生层：清空，再从剩下的日志重放
                removed = self.db.clear_derived(space_id)
                self.vectors.delete(space_id, removed)
                try:
                    if consolidated:
                        with cache.recording(reused) if cache is not None else nullcontext():
                            self.consolidator.enqueue(space_id, skip=self.open_sessions(space_id))
                            self.consolidator.run(space_id)
                finally:
                    # 3. trace：属性里有被删原话的条目（提问、回答原文），以及检索时给回答模型
                    #    看过被删记忆的条目（回答可能复述了它，trace 里却没有原话）。带图的发言
                    #    另外按去掉图片标记的文字找：记忆代理拿消息里的文字部分做检索查询
                    texts = [t.text for t in targets]
                    texts += [_without_images(t) for t in texts if "<image>" in t]
                    traces_removed, questions = self._remove_traces(space_id, texts, refs)

                    # 4. 向量缓存（向量本身也可能泄露原文）：已不在记忆里的文本；被删发言与上面
                    #    那些提问的查询向量（代理与 Agent 拿用户的原话做检索查询）。重放失败时派生
                    #    记忆只重建了一部分，多删的向量下次巩固时重新算
                    gone = sorted(texts_before - {u.text for u in self.db.list_units(space_id)})
                    embed_purged = 0
                    if isinstance(self.embedder, CachedEmbedder):
                        embed_purged = self.embedder.purge(gone)
                        embed_purged += self.embedder.purge([*texts, *questions], "query")
                    self.db.scrub()  # 两种缓存在删除时各自截断 WAL（见 delete_scope、delete_many）
            finally:  # 中途出错时重放用到的更少，删得更多
                if cache is not None:
                    llm_purged += cache.settle(held, reused)
            report = ForgetReport(
                space_id=space_id,
                turns_deleted=deleted,
                sessions_affected=affected,
                facts_before=len(facts),
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

    def _forget_targets(
        self,
        space_id: str,
        session_ids: Sequence[str],
        turn_ids: Sequence[str],
        follow_citations: bool,
        restated_turn_ids: Sequence[str],
    ) -> tuple[list[Turn], int]:
        """要删的发言（按日志顺序）与沿引用扩散多出来的条数。遗忘与预览共用。"""
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
        return [t for t in all_turns if t.id in wanted_turns], cascaded

    def _remove_traces(
        self, space_id: str, texts: Sequence[str], refs: Collection[str] = ()
    ) -> tuple[int, list[str]]:
        """删除这个空间里的 trace：属性中含有被删原话的（太短的话不作依据，免得误删），以及
        检索记录（spans、events）里有被删记忆 ID 的——回答模型看过它，回答可能复述了它。
        返回删除的条数与这些 trace 的提问（提问的查询向量也要清掉）。

        按 ID 宁多勿漏（D-046、D-051）：检索总会带上前几名候选，空间小的时候，遗忘一段会话
        可能删掉大部分回答的 trace。trace 只是排查用的 JSONL，产品里没有地方读它。"""
        wanted = [t for t in texts if len(t.strip()) >= 12]
        ids = set(refs)
        if self.tracer is None or not (wanted or ids):
            return 0, []
        questions: list[str] = []

        def doomed(record: dict[str, Any]) -> bool:
            attrs = record.get("attrs") or {}
            if attrs.get("space") != space_id:
                return False
            blob = " ".join(str(v) for v in attrs.values())
            hit = any(text in blob for text in wanted)
            if not hit and ids:
                hit = not ids.isdisjoint(_strings([record.get("spans"), record.get("events")]))
            if hit:
                questions.extend(str(attrs[k]) for k in ("question", "message") if attrs.get(k))
            return hit

        return self.tracer.remove(doomed), questions

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
                cache_scope=scope(space_id, "forget"),  # 遗忘时删掉（见 forget、forget_memories）
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
                cache_scope=scope(space_id, "forget"),
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

    def forget_memories(
        self,
        space_id: str,
        *,
        session_ids: Sequence[str] = (),
        turn_ids: Sequence[str] = (),
        fact_ids: Sequence[str] = (),
        media_ids: Sequence[str] = (),
    ) -> ForgetReport:
        """按记忆遗忘：会话、发言、事实或图片（服务层、SDK 与 MCP 共用，N12 从服务层移来）。

        原始日志是唯一的事实来源：删一条事实或一张图，删的是它出自的那一轮对话（用户的话与助手
        的回复），然后由剩下的日志重建所有派生记忆（D-021）。删事实时还要追没有引用、追不到的
        转述（D-051）：先在原始日志里找复述了它的发言（包括进行中的会话）一起删；重建之后如果
        同样的信息又从别处长了出来，再删它的出处，最多再追 REGROW_ROUNDS 轮。按内容找到的转述
        只删那条发言、不沿引用扩散：同一轮里用户说的别的事，以及后来引用这些事的回复，都不该连带删掉。
        """
        turns, restated, statements = self._forget_plan(space_id, turn_ids, fact_ids, media_ids)
        try:
            report = self.forget(
                space_id,
                session_ids=list(session_ids),
                turn_ids=list(dict.fromkeys(turns)),
                restated_turn_ids=restated,
            )
            report = replace(report, restated=len(restated))
            for _ in range(REGROW_ROUNDS if statements else 0):
                regrown = self.regrown_facts(space_id, statements)
                if not regrown:
                    break
                more = [t for f in regrown for t in f.source_turn_ids]  # 同样只删出处那条发言
                report = report.merged(
                    replace(self.forget(space_id, restated_turn_ids=more), regrown=len(regrown))
                )
        finally:
            # 找转述与兜底时为被删的陈述算过查询与文档两种向量，兜底那次还在 forget 清理之后：
            # 一并删掉（中途失败时也删）。兜底的请求里有被删的陈述，最后一次没有 forget 跟在后面
            # 替它清理，它的缓存条目（归属 forget）也在这里删
            purged = llm_purged = 0
            if statements and isinstance(self.embedder, CachedEmbedder):
                purged = self.embedder.purge(statements, "query") + self.embedder.purge(statements)
            if statements and self.llm.cache is not None:
                llm_purged = self.llm.cache.delete_scope(scope(space_id, "forget"))
        return replace(
            report,
            embed_cache_purged=report.embed_cache_purged + purged,
            llm_cache_purged=report.llm_cache_purged + llm_purged,
        )

    def _forget_plan(
        self,
        space_id: str,
        turn_ids: Sequence[str],
        fact_ids: Sequence[str],
        media_ids: Sequence[str],
    ) -> tuple[list[str], list[str], list[str]]:
        """按记忆遗忘的第一步：（要删的发言，按内容找到的转述，被删事实的陈述）。"""
        sources: list[str] = []
        statements: list[str] = []
        if fact_ids:
            for fact in self.db.get_facts(space_id, ids=list(fact_ids)):
                sources.extend(fact.source_turn_ids)
                statements.append(fact.statement)
        if media_ids:
            items = self.db.get_media(space_id, ids=list(media_ids))
            sources.extend(m.source_turn_id for m in items)
        # 事实与图片出自的发言扩展成整轮：助手的回复里常常复述了同样的信息
        turns = list(dict.fromkeys([*turn_ids, *self.round_turns(space_id, sources)]))
        restated = self.restating_turns(space_id, statements, exclude=turns)
        return turns, restated, statements

    def preview_forget(
        self,
        space_id: str,
        *,
        session_ids: Sequence[str] = (),
        turn_ids: Sequence[str] = (),
        fact_ids: Sequence[str] = (),
        media_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """遗忘的预演（N12，对标 Supermemory、Redis Agent Memory Server 的 dry_run）：
        会删哪些发言（含沿引用扩散与按内容找到的转述）、哪些事实会随之消失，但什么都不删。

        找转述要调用一次便宜的模型（与真正遗忘时相同，结果进缓存）。重建之后"又长出来"的
        事实（REGROW_ROUNDS）要真的重建才知道，预演里没有；事实一栏只是出处全在被删发言里的那些。
        """
        turns, restated, _ = self._forget_plan(space_id, turn_ids, fact_ids, media_ids)
        targets, cascaded = self._forget_targets(
            space_id, session_ids, turns, follow_citations=True, restated_turn_ids=restated
        )
        doomed = {t.id for t in targets}
        facts = [
            f
            for f in self.db.get_facts(space_id)
            if f.source_turn_ids and set(f.source_turn_ids) <= doomed
        ]
        return {
            "turns": [
                {"id": t.id, "session_id": t.session_id, "role": t.role, "text": t.text}
                for t in targets
            ],
            "sessions_affected": sorted({t.session_id for t in targets}),
            "cascaded": cascaded,
            "restated": len(restated),
            "facts": [{"id": f.id, "memory": f.statement} for f in facts],
            "images": len(self.db.get_media(space_id, turn_ids=sorted(doomed))) if doomed else 0,
        }

    def delete_space(self, space_id: str) -> ForgetReport:
        """删除整个记忆空间（SDK 的 delete_all，N12）：先按会话遗忘全部日志（连带清理缓存、
        trace 与没有别处引用的原图），再删掉空间本身的记录与向量集合。

        两步都在巩固锁里：并发写入登记的巩固不会落在遗忘与删除之间。读写门与遗忘一样先拿。"""
        with self._gate(space_id).exclusive(), self.consolidator.exclusive(space_id):
            report = self.forget(space_id, session_ids=sorted(self.db.get_sessions(space_id)))
            self.db.delete_space(space_id)
            self.vectors.delete_space(space_id)
            self.db.scrub()
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
        question_time 带时区时换算成本机时间（与存下来的时间一致，见 core/timeutil.py）。
        """
        when = naive_local(question_time) if question_time is not None else datetime.now()
        cache_scope = scope(space_id, "ask")
        with (
            self.reading(space_id),  # 遗忘等这次回答写完缓存与 trace
            start_trace("ask", self.tracer, space=space_id, question=question, mode=mode) as trace,
        ):
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
        candidates: L1Candidates | None = None,
    ) -> Recall:
        """只检索、不回答：按 mode 取记忆（auto 由检索控制器选档），返回交给回答模型的材料。

        ask() 与 Agent（N8）共用这一步。exclude 是已经在上下文里的轮与图片的 ID
        （Agent 的工作记忆），不参与探针与检索。

        探针打过分的候选不丢：控制器选 L1 时直接打包；选 L2 时原问题的召回用探针的结果（见
        L2Retriever.search）；返回的 Recall 带着它们，调用方把 L0 改走 L1 时（recall_listing 的
        at_least_l1）以 mode="l1" 与 candidates 传回来、直接打包，不再检索一遍。candidates 必须
        是同一个问题、同一个问题时间、同样的 exclude 与 L1 配置下的探针结果。

        不在 trace 里时（SDK 的 search、记忆代理、各框架集成与 MCP 的 recall）自己记一条
        retrieve trace：提问常常就是用户的原话，遗忘时按检索记录找到它、清掉它的查询向量（见
        _remove_traces）。ask、Agent 与评测在自己的 trace 里调用，不另记，trace 与用量统计不变。
        遗忘等进行中的检索写完 trace（reading）。
        """
        tracing: AbstractContextManager[object] = (
            nullcontext()
            if current_trace() is not None
            else start_trace("retrieve", self.tracer, space=space_id, question=question, mode=mode)
        )
        with self.reading(space_id), tracing:
            when = naive_local(question_time) if question_time is not None else datetime.now()
            cache_scope = cache_scope or scope(space_id, "ask")
            level: str = mode
            decision: RouteDecision | None = None
            used: Router | None = None
            if mode != "l1":
                candidates = None
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
                return Recall("l0", view, decision, used, candidates=candidates)
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
                return Recall("l1", view, decision, used, candidates=candidates)
            l2_cfg = l2 or L2Config()
            deep = self.l2.search(
                space_id,
                question,
                when,
                l2_cfg,
                cache_scope=cache_scope,
                exclude=exclude,
                probe=candidates,
            )
            view = MemoryView(
                rounds=deep.rounds,
                facts=deep.facts,
                core=deep.core,
                media=deep.media,
                image_mode=image_mode(l2_cfg.l1),
                attach=deep.attach,
            )
            return Recall(
                "l2",
                view,
                decision,
                used,
                plan=deep.plan,
                details=deep.details(),
                candidates=candidates,
            )

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
        if known_at is not None:  # 数据库里按字符串比较：带时区的先换算成本机时间
            known_at = naive_local(known_at)
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
        if self._owns_caches:  # 关掉最后一个连接时，SQLite 把 WAL 写回并删掉
            if self.llm.cache is not None:
                self.llm.cache.close()
            if isinstance(self.embedder, CachedEmbedder):
                self.embedder.cache.close()
            image = self.image_embedder
            if isinstance(image, DashScopeMultimodalEmbedder) and image.cache is not None:
                image.cache.close()  # 与文本向量同一个文件的另一个连接
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
