"""异步巩固：把原始日志整理成事实、实体、会话摘要与核心记忆（设计文档 §5.2）。

流水线（D-020）：

    待处理任务 ──并行──▶ A 抽取（无状态，每个会话一次 LLM 调用，可缓存）
                ──批量──▶ B 向量化（陈述、实体描述、摘要一次算完）
                ──串行──▶ C 按会话时间顺序：实体消歧 → 事实对账 → 先写向量、后写数据库
                ────────▶ D 重新生成核心记忆

- 抽取不依赖已有记忆，所以能并行；对账依赖之前所有会话的结果，所以必须按时间串行。
- 任务键 = 记忆空间 + 会话 + 会话内容指纹 + 流水线版本：同一任务只执行一次（幂等）；
  会话内容或抽取逻辑变了，任务键随之改变。
- 每个会话的结果在一个数据库事务里写入并把任务标记为完成；中途失败时任务保持待处理，
  重跑即可，已完成的会话不会重复处理。
- 派生数据都可以从原始日志重建：清空后全部任务重放一遍（LLM 与向量都命中缓存）。巩固过的会话
  又追加了发言时也这样重放，派生记忆始终是原始日志的函数（D-021，见 Consolidator.run）。
"""

from __future__ import annotations

import contextvars
import json
import logging
import queue
import threading
import time
from collections import Counter, OrderedDict, deque
from collections.abc import Callable, Collection, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from cuemem.config import ConsolidationConfig
from cuemem.core.ids import stable_id
from cuemem.core.models import Entity, Episode, Fact, MediaItem, MemoryUnit, Turn
from cuemem.core.timeutil import to_epoch
from cuemem.core.trace import TraceSink, record_event, span, start_trace
from cuemem.embed.base import FloatArray, SparseVector, TextEmbedder
from cuemem.embed.cache import CachedEmbedder
from cuemem.errors import ContentFilteredError, StructuredOutputError
from cuemem.llm.client import LLMClient
from cuemem.store.sqlite import MemoryDB, stored_entity, stored_fact
from cuemem.store.vector import VectorStore
from cuemem.write.caption import Captioner, CaptionMode, image_note, with_text
from cuemem.write.entities import (
    EntityResolver,
    EntityState,
    ResolutionReport,
    descriptor,
    is_user,
)
from cuemem.write.extract import (
    PIPELINE_VERSION,
    Extractor,
    SessionExtraction,
    participants,
    session_content_hash,
)
from cuemem.write.predicates import (
    legacy_value_key,
    normalize_predicate,
    normalize_value,
    registered_cardinality,
)
from cuemem.write.profile import build_core_memory
from cuemem.write.reconcile import FactState, Reconciler

Vector = tuple[FloatArray, SparseVector | None]

logger = logging.getLogger(__name__)

# 后台线程自动重试抽取失败的任务：第 n 次失败之后等 RETRY_BASE_S·4^(n-1) 秒（1、4 分钟），
# 失败 MAX_ATTEMPTS 次就不再自动重试（显式的 consolidate 仍会重试全部失败的任务）
MAX_ATTEMPTS = 3
RETRY_BASE_S = 60.0
# 跨巩固留着载入状态（_Mirror）的空间数，最近巩固过的优先；一个空间约是全部事实与实体，
# 再加当前事实的陈述向量（1 万条、1024 维时约 40 MB）
MIRRORED_SPACES = 4
DIGESTS_KEPT = 100_000  # 记着内容指纹的会话数（见 enqueue），超过时清空重算


@dataclass
class ConsolidationReport:
    space_id: str
    sessions: int = 0
    facts_extracted: int = 0
    added: int = 0
    superseded: int = 0
    merged: int = 0
    noop: int = 0
    entities_created: int = 0
    entities_linked: int = 0
    llm_calls: int = 0  # 抽取、对账、消歧三类调用的总数（含命中缓存的）
    rules: Counter[str] = field(default_factory=Counter)  # 对账结论来自哪条规则
    failed: list[str] = field(default_factory=list)  # 抽取失败的会话（下次重试）
    skipped: list[str] = field(default_factory=list)  # 被内容审核拒绝而跳过抽取的会话
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "facts_extracted": self.facts_extracted,
            "added": self.added,
            "superseded": self.superseded,
            "merged": self.merged,
            "noop": self.noop,
            "entities_created": self.entities_created,
            "entities_linked": self.entities_linked,
            "llm_calls": self.llm_calls,
            "rules": dict(self.rules),
            "failed": self.failed,
            "skipped": self.skipped,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


@dataclass
class _Mirror:
    """一个空间的事实、实体与建了检索单元的实体在库里的样子（与从库里读到的逐字段相同），以及
    向量化过的文本的向量（都来自向量缓存）。见 Consolidator._load_states。"""

    version: tuple[int, ...]  # 记下它时的 MemoryDB.derived_version
    embed_version: tuple[int, ...]  # 记下它时的 CachedEmbedder.version
    facts: dict[str, Fact]
    entities: dict[str, Entity]
    indexed: set[str]
    vectors: dict[str, Vector] = field(default_factory=dict)


class Consolidator:
    def __init__(
        self,
        db: MemoryDB,
        vectors: VectorStore,
        embedder: TextEmbedder,
        llm: LLMClient,
        config: ConsolidationConfig,
        *,
        extractor_model: str | None = None,
        open_sessions: Callable[[str], Collection[str]] | None = None,
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.config = config
        # 进行中的会话（MemorySystem.open_sessions）：重放时跳过，结束后再整体登记
        self.open_sessions: Callable[[str], Collection[str]] = open_sessions or (lambda _: ())
        self.extractor = Extractor(
            llm,
            model=extractor_model,
            assistant_chars=config.assistant_chars,
            max_chars=config.max_session_chars,
            fallback_model=config.fallback_model,
            instructions=config.instructions,
        )
        self.resolver = EntityResolver(
            llm,
            merge_threshold=config.entity_merge_threshold,
            gray_threshold=config.entity_gray_threshold,
            fallback_model=config.fallback_model,
        )
        self.reconciler = Reconciler(
            llm,
            duplicate_threshold=config.duplicate_threshold,
            conflict_threshold=config.conflict_threshold,
            multi_conflict_threshold=config.multi_conflict_threshold,
            fallback_model=config.fallback_model,
        )
        self.captioner = (
            Captioner(llm, max_side=config.caption_max_side, fallback_model=config.fallback_model)
            if config.caption_images
            else None
        )
        # 可重入：遗忘、重建持有这把锁时，内部再调用 run() 不会自己等自己（见 exclusive）
        self._space_locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()
        self._mirrors: OrderedDict[str, _Mirror] = OrderedDict()  # 见 _load_states
        self._mirrors_guard = threading.Lock()
        # （空间，会话）→（发言的版本，开始时间，内容指纹；没有发言时为 None），见 enqueue
        self._digests: dict[tuple[str, str], tuple[tuple[int, ...], datetime, str | None]] = {}
        self._digests_guard = threading.Lock()

    # ---------- 任务 ----------

    def enqueue(
        self,
        space_id: str,
        session_ids: Sequence[str] | None = None,
        *,
        skip: Collection[str] = (),
    ) -> int:
        """为会话登记巩固任务（幂等），返回新登记的任务数。

        skip：暂不巩固的会话（Agent 正在进行的会话，结束后再登记）。同一会话的内容变了
        （又追加了发言）时，新任务取代还没执行的旧任务（见 MemoryDB.enqueue_job）；这个会话
        已经巩固过时，执行前先从原始日志重放（见 run）。
        """
        sessions = self.db.get_sessions(space_id)
        todo: list[tuple[str, str]] = []
        for session_id in session_ids if session_ids is not None else sorted(sessions):
            if session_id not in sessions or session_id in skip:
                continue
            digest = self._digest(space_id, session_id, sessions[session_id])
            if digest is not None:
                todo.append((session_id, digest))
        added = 0
        with self.db.transaction():  # 一次提交（以前每个会话各提交一次）
            for session_id, digest in todo:
                job_id = stable_id(space_id, "consolidate", session_id, digest, PIPELINE_VERSION)
                added += self.db.enqueue_job(job_id, space_id, session_id, PIPELINE_VERSION, digest)
        return added

    def _digest(self, space_id: str, session_id: str, started_at: datetime) -> str | None:
        """会话的内容指纹（session_content_hash；没有发言时为 None）。发言没变过（MemoryDB 的
        turns_version）时用上次算的：显式的巩固与遗忘每次都为全部会话登记任务，以前每次都把每个
        会话的发言读出来重算一遍（800 个会话约 0.18 秒）。"""
        version = self.db.turns_version(space_id, session_id)
        key = (space_id, session_id)
        with self._digests_guard:
            known = self._digests.get(key)
        if known is not None and known[:2] == (version, started_at):
            return known[2]
        turns = self.db.get_turns(space_id, session_id)
        digest = session_content_hash(turns, started_at) if turns else None
        with self._digests_guard:
            if len(self._digests) >= DIGESTS_KEPT:
                self._digests.clear()
            self._digests[key] = (version, started_at, digest)
        return digest

    def fingerprint(self) -> str:
        """影响巩固结果的配置指纹：流水线版本、抽取模型、阈值等（并发数不算）。"""
        role = self.extractor.llm.settings.roles.get(self.extractor.role)
        model = self.extractor.model or (role.model if role else "")
        # 没有补充说明时不计入指纹：加这个字段之前巩固过的空间不会被判为过期
        exclude = {"workers"} | ({"instructions"} if self.config.instructions is None else set())
        config = json.dumps(self.config.model_dump(exclude=exclude), sort_keys=True)
        return stable_id(PIPELINE_VERSION, model, config)

    def stale(self, space_id: str) -> bool:
        """派生记忆是否过期：有用旧版本流水线处理过的会话，或巩固配置变了（需要重建）。"""
        jobs = self.db.get_jobs(space_id)
        if any(j["version"] != PIPELINE_VERSION for j in jobs):
            return True
        recorded = self.db.get_meta(f"consolidation:{space_id}")
        return bool(jobs) and recorded is not None and recorded != self.fingerprint()

    def rebuild(self, space_id: str, *, replay: bool = False) -> None:
        """清空派生记忆（数据库与向量），再为全部会话登记任务，进行中的会话（open_sessions）
        除外。进行中的会话在清空之后才读：这期间结束的会话（close_session 先把它移出进行中、
        再登记任务），任务要么在清空之后登记、不会被删，要么在这里登记。

        replay=True（会话又长了时的重放，见 _replay_regrown）：只为原来有任务的会话登记，
        SDK 的 infer=False 只记了日志的会话照旧等显式的 consolidate；内容没变的会话保留原来的
        结局（_keeps_outcome）。清空前的任务与清空在同一个事务里读：期间别的线程登记的任务
        不会被删掉而没人补上。"""
        with self.db.transaction():
            before = self.db.get_jobs(space_id) if replay else []
            removed = self.db.clear_derived(space_id)
        self.vectors.delete(space_id, removed)
        skip = self.open_sessions(space_id)
        if not replay:
            self.enqueue(space_id, skip=skip)
            return
        self.enqueue(space_id, sorted({j["session_id"] for j in before}), skip=skip)
        self.db.restore_jobs([j for j in before if _keeps_outcome(j)])

    def retry_due(
        self,
        space_id: str,
        *,
        max_attempts: int = MAX_ATTEMPTS,
        base_s: float = RETRY_BASE_S,
        now: datetime | None = None,
    ) -> int:
        """失败的任务里退避时间已到的改回待处理（后台线程用），返回改回的个数。"""
        due = [i for i, wait in self._retry_waits(space_id, max_attempts, base_s, now) if wait <= 0]
        return self.db.reset_jobs(due)

    def next_retry(
        self,
        space_id: str,
        *,
        max_attempts: int = MAX_ATTEMPTS,
        base_s: float = RETRY_BASE_S,
        now: datetime | None = None,
    ) -> float | None:
        """离最早一个失败的任务可以重试还有几秒（已经到了是 0）；没有要重试的任务时为 None。"""
        waits = [wait for _, wait in self._retry_waits(space_id, max_attempts, base_s, now)]
        return max(0.0, min(waits)) if waits else None

    def _retry_waits(
        self, space_id: str, max_attempts: int, base_s: float, now: datetime | None
    ) -> list[tuple[str, float]]:
        """（任务 ID，还要等几秒）：当前版本、失败次数还没到上限的任务。失败时间是 updated_at。"""
        now = now or datetime.now()
        out: list[tuple[str, float]] = []
        for job in self.db.get_jobs(space_id, status="failed"):
            if job["version"] != PIPELINE_VERSION or job["attempts"] >= max_attempts:
                continue
            backoff = timedelta(seconds=base_s * 4 ** max(job["attempts"] - 1, 0))
            ready = datetime.fromisoformat(job["updated_at"]) + backoff
            out.append((job["id"], (ready - now).total_seconds()))
        return out

    # ---------- 执行 ----------

    def run(self, space_id: str) -> ConsolidationReport:
        """执行该空间所有待处理的任务。同一空间同一时间只有一个巩固在跑。

        巩固过的会话又追加了发言时，先清空派生记忆、从原始日志重放（见 _replay_regrown）。"""
        with self._lock_for(space_id):
            self._replay_regrown(space_id)
            return self._run(space_id, scope(space_id, "consolidate"))

    def _replay_regrown(self, space_id: str) -> None:
        """会话巩固过之后又长了（SDK 每轮 add 同一个 run_id、线程结束后又接着聊、代理复用指定的
        会话）：新任务把整个会话重新抽取一遍，增量巩固却只增不减，之前那次抽取的事实还留着（换了
        说法的复述成了另一条事实）。派生记忆就不再是原始日志的函数，之后的遗忘、导入会改动无关的
        事实。所以先清空派生记忆、为原来有任务的会话重新登记（进行中的除外），不清 LLM 缓存：
        其余会话的抽取与时间在它之前的对账都命中缓存，只有长了的会话要重新抽取，时间在它之后的
        会话的对账可能要再问模型；本地的重放与空间大小成正比（D-045）。没有任务的会话（infer=False）
        不抽取；内容审核拒绝过的会话不再请求（拒绝不进缓存），失败的会话保留失败次数与退避时间。

        LLM 缓存关闭或只读（llm_cache_mode 不是 read_write）时不重放：关闭时要把所有会话重新抽取
        一遍，只读时重放的结果进不了缓存。照旧增量巩固，记一条警告。"""
        regrown = self.db.regrown_sessions(space_id, PIPELINE_VERSION)
        if not regrown:
            return
        llm = self.extractor.llm
        if llm.cache is None or llm.settings.llm_cache_mode != "read_write":
            logger.warning(
                "%s 的会话 %s 巩固过之后又追加了发言，LLM 缓存不可写，不从原始日志重放："
                "之前那次抽取的事实还留着",
                space_id,
                sorted(regrown),
            )
            return
        record_event("consolidate_regrown_replay", sessions=sorted(regrown))
        self.rebuild(space_id, replay=True)

    def _run(self, space_id: str, cache_scope: str) -> ConsolidationReport:
        started = time.perf_counter()
        report = ConsolidationReport(space_id)
        sessions = self.db.get_sessions(space_id)
        jobs = [
            j
            for j in self.db.get_jobs(space_id, status="pending")
            if j["session_id"] in sessions and j["version"] == PIPELINE_VERSION
        ]
        if not jobs:
            return report
        jobs.sort(key=lambda j: (sessions[j["session_id"]], j["session_id"]))
        turns_of = {j["session_id"]: self.db.get_turns(space_id, j["session_id"]) for j in jobs}

        # A0. 图片描述（N6）：先描述这些会话里的图片，抽取事实时把描述填回原话里图片的位置
        image_notes = self._caption_images(space_id, turns_of, report)

        # A. 并行抽取
        extractions: dict[str, SessionExtraction] = {}
        with (
            span("consolidate_extract", sessions=len(jobs)),
            ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as pool,
        ):
            # 每个任务带上当前上下文的副本：trace 是 ContextVar，线程池里默认看不到
            futures = {
                j["id"]: pool.submit(
                    contextvars.copy_context().run,
                    self.extractor.extract,
                    j["session_id"],
                    turns_of[j["session_id"]],
                    sessions[j["session_id"]],
                    cache_scope=scope(space_id, f"session:{j['session_id']}"),
                    image_notes=image_notes,
                )
                for j in jobs
            }
            session_of = {j["id"]: j["session_id"] for j in jobs}
            for job_id, future in futures.items():
                session_id = session_of[job_id]
                try:
                    extractions[job_id] = future.result()
                except ContentFilteredError as exc:
                    # 两家的内容审核都拒绝：跳过这个会话的抽取（原始对话仍可检索），不算失败，
                    # 因为重试也不会有不同结果
                    self.db.apply_consolidation(
                        facts=[],
                        entities=[],
                        episodes=[],
                        units=[],
                        job_id=job_id,
                        job_stats={"skipped": "content_filter", "error": str(exc)[:300]},
                    )
                    report.skipped.append(session_id)
                except Exception as exc:
                    self.db.fail_job(job_id, f"{type(exc).__name__}: {exc}")
                    report.failed.append(session_id)
        report.llm_calls += sum(e.attempts for e in extractions.values())

        # B. 批量向量化：陈述、实体描述、摘要
        texts: list[str] = []
        for ext in extractions.values():
            texts += [c.raw.statement for c in ext.facts]
            texts += [descriptor(m) for m in ext.entities]
            texts += [ext.summary] if ext.summary else []
        embedded = self._embed(texts)

        # C. 按会话时间顺序串行：消歧 → 对账 → 写入
        loaded = self.db.derived_version(space_id)
        mirror, entity_state, fact_state = self._load_states(space_id, loaded)
        if mirror is not None:
            mirror.vectors.update(embedded)
        for job in jobs:
            extraction = extractions.get(job["id"])
            if extraction is None:
                continue
            stats = self._apply(
                space_id,
                job["id"],
                extraction,
                turns_of[job["session_id"]],
                entity_state,
                fact_state,
                embedded,
                report,
                cache_scope,
                mirror,
            )
            record_event("consolidate_session", session=extraction.session_id, **stats)

        # D. 核心记忆：参与者（用户本人，或多人对话里的各位说话人）的重要事实
        names = self.db.speakers(space_id) or ["User"]
        people_ids = list(dict.fromkeys(entity_state.lookup(n) for n in names))
        people = [entity_state.entities[i] for i in people_ids if i is not None]
        core = build_core_memory(
            space_id,
            list(fact_state.facts.values()),
            people,
            max_chars=self.config.profile_max_chars,
            min_importance=self.config.profile_min_importance,
        )
        self.db.set_core_memory(core)
        self.db.set_meta(f"consolidation:{space_id}", self.fingerprint())
        self._keep_mirror(space_id, mirror, loaded)
        report.sessions = len(extractions)
        report.elapsed_ms = (time.perf_counter() - started) * 1000
        return report

    def _caption_images(
        self,
        space_id: str,
        turns_of: dict[str, list[Turn]],
        report: ConsolidationReport,
    ) -> dict[str, list[str]]:
        """为待处理会话里还没有描述的图片生成两种描述，写入图片记录与检索单元。

        返回 发言 ID → 各张图的上下文感知描述（按出现顺序），供抽取时填回原话。
        """
        all_turns = {t.id: t for turns in turns_of.values() for t in turns}
        media = self.db.get_media(space_id, turn_ids=list(all_turns)) if all_turns else []
        if not media:
            return {}
        if self.captioner is not None:
            todo = [m for m in media if not m.caption]
            if todo:
                self._run_captions(space_id, todo, turns_of, all_turns, report)
        by_turn: dict[str, list[MediaItem]] = {}
        for item in media:
            by_turn.setdefault(item.source_turn_id, []).append(item)
        notes: dict[str, list[str]] = {}
        for turn_id, items in by_turn.items():
            order = all_turns[turn_id].media_ids
            items.sort(key=lambda m: order.index(m.id) if m.id in order else len(order))
            notes[turn_id] = [image_note(m) for m in items if m.caption]
        return {k: v for k, v in notes.items() if v}

    def _run_captions(
        self,
        space_id: str,
        todo: list[MediaItem],
        turns_of: dict[str, list[Turn]],
        all_turns: dict[str, Turn],
        report: ConsolidationReport,
    ) -> None:
        assert self.captioner is not None
        captioner = self.captioner
        session_of = {t.id: t.session_id for t in all_turns.values()}
        modes: tuple[CaptionMode, ...] = ("context", "isolated")
        tasks = [(m, mode) for m in todo for mode in modes]

        def run(item: MediaItem, mode: CaptionMode) -> tuple[str, CaptionMode, Any]:
            session_id = session_of[item.source_turn_id]
            try:
                output = captioner.caption(
                    item,
                    turns_of[session_id],
                    mode=mode,
                    cache_scope=scope(space_id, f"session:{session_id}"),
                )
            except (ContentFilteredError, StructuredOutputError) as exc:
                # 两家都拒绝，或多次重试仍得不到合格的描述：这张图没有描述（原图仍可再感知），
                # 不让一张图拖垮整份历史的巩固
                record_event("caption_failed", media=item.id, mode=mode, error=type(exc).__name__)
                return item.id, mode, None
            return item.id, mode, output

        with (
            span("consolidate_caption", images=len(todo)),
            ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as pool,
        ):
            futures = [
                pool.submit(contextvars.copy_context().run, run, item, mode) for item, mode in tasks
            ]
            results = [f.result() for f in futures]
        by_id = {m.id: m for m in todo}
        for media_id, mode, output in results:
            report.llm_calls += 1
            if output is None:
                continue
            item = by_id[media_id]
            if mode == "context":
                item.caption = output.caption or "an image"
                item.ocr_text = output.text_in_image
                item.tags = output.tags
            else:
                item.caption_iso = with_text(output) or "an image"
        captioned = [m for m in todo if m.caption or m.caption_iso]
        self._index_images(space_id, captioned, all_turns)
        self.db.set_media_captions(captioned)

    def _index_images(
        self, space_id: str, items: Sequence[MediaItem], all_turns: dict[str, Turn]
    ) -> None:
        """图片描述作为检索单元（像事实一样指回原话所在的轮）：上下文感知与孤立各一种。"""
        units: list[MemoryUnit] = []
        for item in items:
            turn = all_turns[item.source_turn_id]
            label = turn.speaker or ("User" if turn.role == "user" else "Assistant")
            for kind, note in (
                ("image", image_note(item) if item.caption else ""),
                ("image_iso", image_note(item, isolated=True) if item.caption_iso else ""),
            ):
                if not note:
                    continue
                units.append(
                    MemoryUnit(
                        id=stable_id(space_id, kind, item.id),
                        space_id=space_id,
                        kind=kind,
                        session_id=turn.session_id,
                        text=f"{label} shared an image: {note}",
                        ts=item.ts,
                        source_turn_ids=[turn.id],
                        meta={"media_id": item.id},
                    )
                )
        if not units:
            return
        vectors = self._embed([u.text for u in units])
        dense = _stack([vectors[u.text][0] for u in units])
        sparse_rows = [vectors[u.text][1] for u in units]
        sparse = None if any(v is None for v in sparse_rows) else [v for v in sparse_rows if v]
        payloads = [
            {"kind": u.kind, "session_id": u.session_id, "ts": to_epoch(u.ts)} for u in units
        ]
        self.vectors.upsert(space_id, [u.id for u in units], dense, sparse, payloads)
        self.db.add_units(units)

    def _apply(
        self,
        space_id: str,
        job_id: str,
        ext: SessionExtraction,
        turns: Sequence[Any],
        entity_state: EntityState,
        fact_state: FactState,
        embedded: dict[str, Vector],
        report: ConsolidationReport,
        cache_scope: str,
        mirror: _Mirror | None,
    ) -> dict[str, Any]:
        ts = ext.started_at
        ents = ResolutionReport()
        self.resolver.ensure_people(entity_state, participants(turns), ts, ents)
        for candidate in ext.facts:  # 先登记用户的名字，抽取出的同名实体才能对应到用户本人
            raw = candidate.raw
            if normalize_predicate(raw.predicate) == "name" and is_user(raw.subject):
                self.resolver.name_user(entity_state, raw.object, ts, ents)
        dense_of = {text: vec[0] for text, vec in embedded.items()}
        self.resolver.resolve(
            entity_state, ext.entities, dense_of, ts, ents, cache_scope=cache_scope
        )

        new_facts: list[tuple[Fact, FloatArray]] = []
        said_on = {t.id: t.ts for t in turns}
        keys: dict[str, str] = {}  # 本次抽取里用过的事实 ID → 陈述的规范形式
        for candidate in ext.facts:
            raw = candidate.raw
            subject_id = self.resolver.subject_id(entity_state, raw.subject, ts, ents)
            if subject_id is None:  # 关于助手本身的事实不记
                continue
            predicate = normalize_predicate(raw.predicate)
            object_id = entity_state.lookup(raw.object) if raw.object else None
            # 事实 ID 照旧按原来的规范形式派生（已有的 ID 不变）；只有原来的形式撞上、陈述其实
            # 不同时（"...named 초코." 与 "...named 보리." 原来都成了 "...named"）才另派一个
            key = normalize_value(raw.statement)
            fact_id = stable_id(space_id, "fact", ext.session_id, legacy_value_key(raw.statement))
            if keys.setdefault(fact_id, key) != key:
                fact_id = stable_id(space_id, "fact", ext.session_id, key, "u")
            fact = Fact(
                id=fact_id,
                space_id=space_id,
                subject_id=subject_id,
                subject=entity_state.entities[subject_id].name,
                predicate=predicate,
                object=raw.object,
                statement=raw.statement.strip(),
                category=raw.category,
                cardinality=registered_cardinality(predicate) or raw.cardinality,
                negated=raw.negated,
                valid_from=raw.date,
                recorded_at=_said_at(candidate.source_turn_ids, said_on, ts),
                importance=raw.importance,
                entity_ids=list(dict.fromkeys([subject_id, *([object_id] if object_id else [])])),
                session_id=ext.session_id,
                source_turn_ids=candidate.source_turn_ids,
            )
            new_facts.append((fact, embedded[raw.statement][0]))
        report.facts_extracted += len(new_facts)
        said = {  # 用户说的原话：对账靠它分清"说错了"与"现实变了"
            t.id: f"{t.speaker}: {t.text}" if t.speaker else t.text
            for t in turns
            if t.role == "user"
        }
        rec = self.reconciler.reconcile(
            fact_state, new_facts, cache_scope=cache_scope, sources=said
        )

        # 先写向量（新事实、摘要、新实体；被取代的事实更新 payload），再在一个事务里写数据库
        units: list[MemoryUnit] = []
        index: list[tuple[MemoryUnit, Vector, dict[str, Any]]] = []
        added_ids = {op.fact_id for op in rec.operations if op.action in ("add", "supersede")}
        for fact in rec.changed.values():
            unit = _fact_unit(fact)
            units.append(unit)
            if fact.id in added_ids:
                index.append((unit, embedded[fact.statement], _fact_payload(fact)))
        retired = [f.id for f in rec.changed.values() if not f.is_current and f.id not in added_ids]
        episodes: list[Episode] = []
        if ext.summary:
            episode = Episode(
                id=stable_id(space_id, "episode", ext.session_id),
                space_id=space_id,
                session_id=ext.session_id,
                start_ts=ts,
                end_ts=max((t.ts for t in turns), default=ts),
                summary=ext.summary,
                turn_ids=ext.turn_ids,
            )
            episodes.append(episode)
            unit = MemoryUnit(
                id=episode.id,
                space_id=space_id,
                kind="summary",
                session_id=ext.session_id,
                text=ext.summary,
                ts=ts,
                source_turn_ids=ext.turn_ids,
            )
            units.append(unit)
            index.append((unit, embedded[ext.summary], {"kind": "summary"}))
        for entity in ents.changed.values():
            vec = embedded.get(entity.descriptor)
            if vec is None or entity.id not in ents.created_ids:  # 只索引新建且有描述向量的实体
                continue
            unit = _entity_unit(entity, ext.session_id)
            units.append(unit)
            index.append((unit, vec, {"kind": "entity"}))

        if index:
            dense = _stack([vec[0] for _, vec, _ in index])
            sparse = [vec[1] or SparseVector(indices=[], values=[]) for _, vec, _ in index]
            payloads = [
                {"session_id": unit.session_id, "ts": to_epoch(unit.ts), **extra}
                for unit, _, extra in index
            ]
            self.vectors.upsert(space_id, [u.id for u, _, _ in index], dense, sparse, payloads)
        self.vectors.set_payload(space_id, retired, {"current": False})
        stats = {
            "facts": len(new_facts),
            "added": rec.count("add"),
            "superseded": rec.count("supersede"),
            "merged": rec.count("merge"),
            "noop": rec.count("noop"),
            "entities_created": ents.created,
            "entities_linked": ents.linked_exact + ents.linked_vector + ents.linked_llm,
            "llm_calls": ext.attempts + rec.llm_calls + ents.llm_calls,
            "dropped_turn_refs": ext.dropped_turn_refs,
        }
        self.db.apply_consolidation(
            facts=list(rec.changed.values()),
            entities=list(ents.changed.values()),
            episodes=episodes,
            units=units,
            job_id=job_id,
            job_stats=stats,
        )
        if mirror is not None:  # 镜像跟着库走：记下写进去的样子
            mirror.facts.update((f.id, stored_fact(f)) for f in rec.changed.values())
            mirror.entities.update((e.id, stored_entity(e)) for e in ents.changed.values())
            mirror.indexed.update(u.id for u in units if u.kind == "entity")
        report.added += stats["added"]
        report.superseded += stats["superseded"]
        report.merged += stats["merged"]
        report.noop += stats["noop"]
        report.entities_created += ents.created
        report.entities_linked += stats["entities_linked"]
        report.llm_calls += rec.llm_calls + ents.llm_calls
        report.rules.update(f"{op.action}:{op.rule}" for op in rec.operations)
        return stats

    # ---------- 状态加载 ----------

    def _load_states(
        self, space_id: str, version: tuple[int, ...]
    ) -> tuple[_Mirror | None, EntityState, FactState]:
        """载入这个空间的实体与事实状态：库里的全部实体（按 first_seen、ID）与建了检索单元的
        实体的描述向量，库里的全部事实（按 recorded_at、ID）与当前事实的陈述向量。

        以前每次巩固都从库里读一遍全部事实与实体、逐条查向量缓存（9,600 条事实时约 0.75 秒，
        SDK 每次 add 都有一次巩固）。现在上一次巩固之后留下这个空间的镜像（_Mirror：库里的样子
        与查过的向量），派生记忆与向量缓存都没变过（version 是 MemoryDB.derived_version，见
        _keep_mirror）时只按库里的顺序取一遍 ID，状态、先后与向量都与从库里读的逐字段、逐位
        相同；还没查过的文本照旧交给向量器，请求也相同。向量器不带缓存时照旧每次从库里读。"""
        embed_version = self._embed_version()
        with self._mirrors_guard:
            mirror = self._mirrors.pop(space_id, None)  # 巩固出错时不放回去：下次从库里读
        entities: list[Entity] | None = None
        facts: list[Fact] = []
        if mirror is not None and (mirror.version, mirror.embed_version) == (
            version,
            embed_version,
        ):
            fact_ids, entity_ids = self.db.fact_ids(space_id), self.db.entity_ids(space_id)
            if mirror.facts.keys() == set(fact_ids) and mirror.entities.keys() == set(entity_ids):
                entities = [mirror.entities[i] for i in entity_ids]
                facts = [mirror.facts[i] for i in fact_ids]
        if mirror is None or entities is None:
            entities = self.db.get_entities(space_id)
            indexed = {u.id for u in self.db.list_units(space_id, kinds=["entity"])}
            facts = self.db.get_facts(space_id)
            mirror = None
            if embed_version is not None:
                mirror = _Mirror(
                    version,
                    embed_version,
                    {f.id: f for f in facts},
                    {e.id: e for e in entities},
                    indexed,
                )
        else:
            indexed = mirror.indexed
        entity_state = EntityState(space_id)
        vectors = self._vectors([e.descriptor for e in entities if e.id in indexed], mirror)
        for entity in entities:
            vec = vectors.get(entity.descriptor) if entity.id in indexed else None
            entity_state.add(entity, vec[0] if vec else None)
        fact_state = FactState()
        vectors = self._vectors([f.statement for f in facts if f.is_current], mirror)
        for fact in facts:
            vec = vectors.get(fact.statement) if fact.is_current else None
            fact_state.put(fact, vec[0] if vec else None)
        return mirror, entity_state, fact_state

    def _vectors(self, texts: Sequence[str], mirror: _Mirror | None) -> Mapping[str, Vector]:
        """同 _embed；有镜像时先用镜像里记下的向量（都是向量缓存里的，逐位相同），只把其余的交给
        向量器：记下的文本本来就都在缓存里，未命中、要发出的请求与全部交给向量器时相同。"""
        if mirror is None:
            return self._embed(texts)
        known = mirror.vectors
        known.update(self._embed([t for t in texts if t not in known]))
        return known

    def _keep_mirror(self, space_id: str, mirror: _Mirror | None, loaded: tuple[int, ...]) -> None:
        """巩固做完，留下镜像给下一次用。这期间这个空间的派生记忆只有本次巩固改过（改动派生记忆
        的操作都持有这个空间的锁，见 exclusive），所以镜像与库相同；有事务回滚过、别的连接写过
        这个库，或向量缓存删过条目、被别的连接改过时不留。"""
        if mirror is None:
            return
        version = self.db.derived_version(space_id)
        if version[1:] != loaded[1:] or self._embed_version() != mirror.embed_version:
            return
        mirror.version = version
        needed = {f.statement for f in mirror.facts.values() if f.is_current}
        needed |= {mirror.entities[i].descriptor for i in mirror.indexed if i in mirror.entities}
        mirror.vectors = {t: v for t, v in mirror.vectors.items() if t in needed}
        with self._mirrors_guard:
            self._mirrors[space_id] = mirror
            while len(self._mirrors) > MIRRORED_SPACES:
                self._mirrors.popitem(last=False)

    def _embed_version(self) -> tuple[int, ...] | None:
        return self.embedder.version() if isinstance(self.embedder, CachedEmbedder) else None

    def _embed(self, texts: Sequence[str]) -> dict[str, Vector]:
        unique = [t for t in dict.fromkeys(texts) if t]
        if not unique:
            return {}
        with span("embed", texts=len(unique)):
            batch = self.embedder.embed(unique, "document")
        return {
            text: (batch.dense[i], batch.sparse[i] if batch.sparse is not None else None)
            for i, text in enumerate(unique)
        }

    def exclusive(self, space_id: str) -> threading.RLock:
        """这个空间的派生记忆的独占锁（可重入），与 run() 是同一把。

        凡是要改动派生记忆的操作（遗忘、重建、删除空间）都在这把锁里做：后台线程正在巩固
        这个空间时，它在开头就读好了会话的轮次，事后会把由它们抽出的事实、向量与缓存写回去；
        在锁外删除原始日志、清空派生层，被删的内容就又回来了。持有这把锁时可以直接调用 run()。
        """
        return self._lock_for(space_id)

    def _lock_for(self, space_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._space_locks.setdefault(space_id, threading.RLock())


class ConsolidationWorker:
    """后台巩固：在线写入只负责登记任务，这个线程在空闲时把任务做完，不阻塞对话。

    同一时间只处理一个空间；同一空间的任务按会话时间顺序串行（对账需要）。抽取失败的会话
    （限流、余额不足、超时）按退避时间自动重试（见 Consolidator.retry_due），巩固整个抛出了
    异常（对账、向量化时出错）时整个空间按同样的退避再做（见 _retry_after_crash），不用等下一次
    显式的巩固。reports 与 errors 只留最近 100 条。
    """

    def __init__(
        self,
        consolidator: Consolidator,
        *,
        tracer: TraceSink | None = None,
        retry_base_s: float = RETRY_BASE_S,
    ) -> None:
        self.consolidator = consolidator
        self.tracer = tracer
        self.retry_base_s = retry_base_s
        self.stop_timeout_s = 30.0
        self.closing = False  # stop() 没等到线程结束：数据库随后关闭，之后的报错不再记录
        self.reports: deque[ConsolidationReport] = deque(maxlen=100)
        self.errors: deque[str] = deque(maxlen=100)
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._retry: dict[str, float] = {}  # 空间 → 可以重试的时刻（time.monotonic），只在线程里用
        self._crashes: dict[str, int] = {}  # 空间 → 巩固连续抛出异常的次数，只在线程里用
        self._thread = threading.Thread(target=self._loop, name="cuemem-consolidate", daemon=True)
        self._thread.start()

    def submit(self, space_id: str) -> None:
        self._queue.put(space_id)

    def wait(self) -> None:
        """等待已提交的空间全部处理完（其间到期的重试也等；还没到期的不等）。"""
        self._queue.join()

    def stop(self) -> None:
        """排在前面的空间做完再退出，最多等 stop_timeout_s 秒。等不到时不再等：调用方随后关闭
        数据库，线程之后的报错不再记录，没做完的任务留在数据库里，下次启动时接着做。"""
        self._queue.put(None)
        self._thread.join(timeout=self.stop_timeout_s)
        if self._thread.is_alive():
            self.closing = True
            logger.warning(
                "后台巩固 %.0f 秒内没有做完，不再等待：没做完的抽取下次启动时接着做",
                self.stop_timeout_s,
            )

    def _loop(self) -> None:
        while True:
            try:
                space_id = self._queue.get(timeout=self._until_due())
            except queue.Empty:
                self._resubmit_due()
                continue
            if space_id is None:
                self._queue.task_done()
                return
            try:
                self._consolidate(space_id)
                self._crashes.pop(space_id, None)
                self._resubmit_due()  # 到期的重试排在 task_done 之前：wait() 一并等它们
            except Exception as exc:  # 后台线程不能崩：记录下来，任务保持待处理，按退避再试
                if not self.closing:  # 关闭超时后数据库已经关了，这时的报错说明不了什么
                    self.errors.append(f"{space_id}: {type(exc).__name__}: {exc}")
                    self._retry_after_crash(space_id)
                    self._resubmit_due()
            finally:
                self._queue.task_done()

    def _consolidate(self, space_id: str) -> None:
        consolidator = self.consolidator
        base_s = self.retry_base_s
        with start_trace("consolidate", self.tracer, space=space_id, background=True) as trace:
            with consolidator.exclusive(space_id):
                consolidator.retry_due(space_id, base_s=base_s)
                report = consolidator.run(space_id)
                wait = consolidator.next_retry(space_id, base_s=base_s)
            trace.attrs.update(report.as_dict())
            if report.failed:  # 逐个会话的失败不会抛出来：记一条，便于在状态接口里看到
                failed = set(report.failed)
                jobs = consolidator.db.get_jobs(space_id, status="failed")
                error = next((j["error"] or "" for j in jobs if j["session_id"] in failed), "")
                record_event("consolidate_failed", sessions=report.failed, error=error[:200])
                self.errors.append(f"{space_id}: {len(failed)} 个会话抽取失败：{error[:200]}")
        self.reports.append(report)
        if wait is None:
            self._retry.pop(space_id, None)
        else:
            self._retry[space_id] = time.monotonic() + wait

    def _retry_after_crash(self, space_id: str) -> None:
        """巩固整个抛出了异常（对账、向量化、图片描述遇上余额不足、限流等；逐个会话的抽取失败
        不会抛出来，见 retry_due）：任务还是待处理，按同样的退避时间再做一遍，连续 MAX_ATTEMPTS
        次都抛出就不再自动重试，等这个空间的下一次提交或显式的巩固。"""
        crashes = self._crashes[space_id] = self._crashes.get(space_id, 0) + 1
        if crashes >= MAX_ATTEMPTS:
            return
        due = time.monotonic() + self.retry_base_s * 4 ** (crashes - 1)
        self._retry[space_id] = min(due, self._retry.get(space_id, due))

    def _until_due(self) -> float | None:
        """离最早一次重试还有几秒；没有要重试的空间时为 None（一直等新提交的空间）。"""
        if not self._retry:
            return None
        return max(0.0, min(self._retry.values()) - time.monotonic())

    def _resubmit_due(self) -> None:
        now = time.monotonic()
        for space_id, due in list(self._retry.items()):
            if due <= now:
                del self._retry[space_id]
                self._queue.put(space_id)


def scope(space_id: str, part: str) -> str:
    """LLM 缓存条目的归属标签：抽取按会话标记，对账与消歧按空间标记，回答按空间标记。
    遗忘时据此只清理受影响的条目，其余会话的抽取结果仍可复用（D-021）。"""
    return f"{space_id}|{part}"


def _keeps_outcome(job: Mapping[str, Any]) -> bool:
    """重放时内容没变（任务键相同）的会话保留的结局：待处理与失败的任务（失败次数、退避时间不
    清零，见 MAX_ATTEMPTS），以及被内容审核拒绝而跳过的任务（拒绝不进缓存，重放会再请求两家）。
    正常完成的任务要重放：清空之后它的事实没了，重新执行命中缓存。"""
    skipped = job["status"] == "done" and job["stats"].get("skipped") == "content_filter"
    return job["status"] in ("pending", "failed") or skipped


def _said_at(
    turn_ids: Sequence[str], said_on: Mapping[str, datetime], started: datetime
) -> datetime:
    """事实的记录时间（被说出的时间）：出处里最晚的一句发言不在会话开始的那天说的（复用的
    run_id、LangGraph 的线程跨了好几天），取那句发言的时间；否则取会话开始的时间。

    只按天区分，与抽取提示一致（换了日期的发言前才标出日期，见 render_session）：同一天里的
    会话（评测的会话、Agent 的一次对话）事实的记录时间与之前相同，派生记忆与之后的请求不变。"""
    latest = max((said_on[t] for t in turn_ids if t in said_on), default=started)
    return latest if latest.date() != started.date() else started


# ---------- 检索单元 ----------


def _fact_unit(fact: Fact) -> MemoryUnit:
    return MemoryUnit(
        id=fact.id,
        space_id=fact.space_id,
        kind="fact",
        session_id=fact.session_id,
        text=fact.statement,
        ts=fact.recorded_at,
        source_turn_ids=fact.source_turn_ids,
        meta={"subject_id": fact.subject_id, "predicate": fact.predicate},
    )


def _fact_payload(fact: Fact) -> dict[str, Any]:
    return {"kind": "fact", "current": fact.is_current, "subject_id": fact.subject_id}


def _entity_unit(entity: Entity, session_id: str) -> MemoryUnit:
    return MemoryUnit(
        id=entity.id,
        space_id=entity.space_id,
        kind="entity",
        session_id=session_id,
        text=entity.descriptor,
        ts=entity.first_seen,
        source_turn_ids=[],
        meta={"name": entity.name, "type": entity.type},
    )


def _stack(rows: Sequence[FloatArray]) -> FloatArray:
    return np.vstack(rows).astype(np.float32)
