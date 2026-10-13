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
- 派生数据都可以从原始日志重建：清空后全部任务重放一遍（LLM 与向量都命中缓存）。
"""

from __future__ import annotations

import contextvars
import json
import queue
import threading
import time
from collections import Counter
from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cuemem.config import ConsolidationConfig
from cuemem.core.ids import stable_id
from cuemem.core.models import Entity, Episode, Fact, MediaItem, MemoryUnit, Turn
from cuemem.core.timeutil import to_epoch
from cuemem.core.trace import record_event, span
from cuemem.embed.base import FloatArray, SparseVector, TextEmbedder
from cuemem.errors import ContentFilteredError, StructuredOutputError
from cuemem.llm.client import LLMClient
from cuemem.store.sqlite import MemoryDB
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
from cuemem.write.predicates import normalize_predicate, normalize_value, registered_cardinality
from cuemem.write.profile import build_core_memory
from cuemem.write.reconcile import FactState, Reconciler

Vector = tuple[FloatArray, SparseVector | None]


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
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.config = config
        self.extractor = Extractor(
            llm,
            model=extractor_model,
            assistant_chars=config.assistant_chars,
            max_chars=config.max_session_chars,
            fallback_model=config.fallback_model,
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
        self._space_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

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
        （又追加了发言）时，新任务取代还没执行的旧任务（见 MemoryDB.enqueue_job）。
        """
        sessions = self.db.get_sessions(space_id)
        added = 0
        for session_id in session_ids if session_ids is not None else sorted(sessions):
            turns = self.db.get_turns(space_id, session_id)
            if not turns or session_id not in sessions or session_id in skip:
                continue
            digest = session_content_hash(turns, sessions[session_id])
            job_id = stable_id(space_id, "consolidate", session_id, digest, PIPELINE_VERSION)
            added += self.db.enqueue_job(job_id, space_id, session_id, PIPELINE_VERSION, digest)
        return added

    def fingerprint(self) -> str:
        """影响巩固结果的配置指纹：流水线版本、抽取模型、阈值等（并发数不算）。"""
        role = self.extractor.llm.settings.roles.get(self.extractor.role)
        model = self.extractor.model or (role.model if role else "")
        config = json.dumps(self.config.model_dump(exclude={"workers"}), sort_keys=True)
        return stable_id(PIPELINE_VERSION, model, config)

    def stale(self, space_id: str) -> bool:
        """派生记忆是否过期：有用旧版本流水线处理过的会话，或巩固配置变了（需要重建）。"""
        jobs = self.db.get_jobs(space_id)
        if any(j["version"] != PIPELINE_VERSION for j in jobs):
            return True
        recorded = self.db.get_meta(f"consolidation:{space_id}")
        return bool(jobs) and recorded is not None and recorded != self.fingerprint()

    def rebuild(self, space_id: str, *, skip: Collection[str] = ()) -> None:
        """清空派生记忆（数据库与向量），再为全部会话（skip 里的除外）登记任务。"""
        removed = self.db.clear_derived(space_id)
        self.vectors.delete(space_id, removed)
        self.enqueue(space_id, skip=skip)

    # ---------- 执行 ----------

    def run(self, space_id: str) -> ConsolidationReport:
        """执行该空间所有待处理的任务。同一空间同一时间只有一个巩固在跑。"""
        with self._lock_for(space_id):
            return self._run(space_id, scope(space_id, "consolidate"))

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
            for job_id, future in futures.items():
                session_id = next(j["session_id"] for j in jobs if j["id"] == job_id)
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
        entity_state = self._load_entities(space_id)
        fact_state = self._load_facts(space_id)
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
        for candidate in ext.facts:
            raw = candidate.raw
            subject_id = self.resolver.subject_id(entity_state, raw.subject, ts, ents)
            if subject_id is None:  # 关于助手本身的事实不记
                continue
            predicate = normalize_predicate(raw.predicate)
            object_id = entity_state.lookup(raw.object) if raw.object else None
            fact = Fact(
                id=stable_id(space_id, "fact", ext.session_id, normalize_value(raw.statement)),
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
                recorded_at=ts,
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

    def _load_entities(self, space_id: str) -> EntityState:
        state = EntityState(space_id)
        entities = self.db.get_entities(space_id)
        indexed = {u.id for u in self.db.list_units(space_id, kinds=["entity"])}
        vectors = self._embed([e.descriptor for e in entities if e.id in indexed])
        for entity in entities:
            vec = vectors.get(entity.descriptor) if entity.id in indexed else None
            state.add(entity, vec[0] if vec else None)
        return state

    def _load_facts(self, space_id: str) -> FactState:
        state = FactState()
        facts = self.db.get_facts(space_id)
        vectors = self._embed([f.statement for f in facts if f.is_current])
        for fact in facts:
            state.facts[fact.id] = fact
            if fact.is_current and fact.statement in vectors:
                state.vectors[fact.id] = vectors[fact.statement][0]
        return state

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

    def _lock_for(self, space_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._space_locks.setdefault(space_id, threading.Lock())


class ConsolidationWorker:
    """后台巩固：在线写入只负责登记任务，这个线程在空闲时把任务做完，不阻塞对话。

    同一时间只处理一个空间；同一空间的任务按会话时间顺序串行（对账需要）。
    """

    def __init__(self, consolidator: Consolidator) -> None:
        self.consolidator = consolidator
        self.reports: list[ConsolidationReport] = []
        self.errors: list[str] = []
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="cuemem-consolidate", daemon=True)
        self._thread.start()

    def submit(self, space_id: str) -> None:
        self._queue.put(space_id)

    def wait(self) -> None:
        """等待已提交的空间全部处理完。"""
        self._queue.join()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=30)

    def _loop(self) -> None:
        while True:
            space_id = self._queue.get()
            try:
                if space_id is None:
                    return
                self.reports.append(self.consolidator.run(space_id))
            except Exception as exc:  # 后台线程不能崩：记录下来，任务保持待处理，下次再试
                self.errors.append(f"{space_id}: {type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()


def scope(space_id: str, part: str) -> str:
    """LLM 缓存条目的归属标签：抽取按会话标记，对账与消歧按空间标记，回答按空间标记。
    遗忘时据此只清理受影响的条目，其余会话的抽取结果仍可复用（D-021）。"""
    return f"{space_id}|{part}"


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
