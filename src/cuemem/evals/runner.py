"""评测运行器：写入 → 回答 → 判分，结果逐行追加到 JSONL。

- 断点续跑：已完成的（题目，方法）组合会被跳过；出错的组合记入 errors.jsonl，下次运行会重试；
- 并发：按对话历史分组，每组在一个线程里处理，同组题目共用一次写入（LoCoMo 一段对话约 200 题）；
- 用量：每个（题目，方法）在独立的 trace 里执行，从 trace 事件汇总 token、调用次数与名义成本；
  判分单独统计，不计入方法成本；
- 预算：按实际花费（不含本地缓存命中）累计，超过上限就不再开始新的组合。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2

from cuemem.config import PROJECT_ROOT, Settings
from cuemem.core.models import Hit
from cuemem.core.trace import Trace, start_trace
from cuemem.embed import build_embedder, build_image_embedder
from cuemem.embed.base import TextEmbedder
from cuemem.embed.multimodal import ImageEmbedder
from cuemem.errors import ConfigMismatchError
from cuemem.evals.datasets import (
    EvalSession,
    EvalSet,
    Sample,
    SessionIds,
    sample_evidence_turns,
    session_id_scheme,
    storage_name,
    to_conversation,
    to_session_ins,
)
from cuemem.evals.judge import Judge, token_f1
from cuemem.evals.methods import Method, MethodContext, build_methods
from cuemem.evals.metrics import retrieval_metrics
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.memory import MemorySystem, build_reranker
from cuemem.read.reader import Reader
from cuemem.read.rerank import Reranker
from cuemem.write.extract import PIPELINE_VERSION
from cuemem.write.online import build_rounds, build_turns


@dataclass
class RunConfig:
    name: str
    dataset: str
    split: str | None
    methods: list[str]
    k: int = 10
    workers: int = 4
    max_cost: float | None = None  # 美元，按实际花费计；None 表示不设上限
    limit: int | None = None  # 只跑前 N 道题（调试用）
    reuse_spaces: bool = True  # 记忆空间存在磁盘上跨运行复用（paths.eval_spaces）
    build_only: bool = False  # 只构建记忆空间（写入与巩固），不回答、不判分


@dataclass(frozen=True)
class ProbeQuery:
    """一次探针：在哪个空间、问什么、什么时候问；meta 原样写进 probe.jsonl。"""

    query_id: str
    dataset: str  # 记忆空间所在的数据集
    space_key: str
    question: str
    question_time: datetime
    meta: dict[str, Any] = field(default_factory=dict)
    # 空间的会话 ID 写法（见 datasets 的模块说明）；None 时按环境变量，与加载数据集时一致
    session_ids: SessionIds | None = None


@dataclass
class RunSummary:
    run_dir: Path
    planned: int
    completed: int
    errors: int
    spent: float  # 本次运行的实际花费（不含本地缓存命中）
    nominal_cost: float  # 本次运行的名义成本
    elapsed_s: float
    stopped_by_budget: bool


def usage_from_trace(trace: Trace) -> dict[str, Any]:
    """从 trace 事件汇总用量。cost 是名义成本；spent 只计真实发生的调用（本地缓存命中不花钱）。"""
    llm = [e for e in trace.events if e["kind"] == "llm"]
    emb = [e for e in trace.events if e["kind"] == "embedding"]
    rerank = [e for e in trace.events if e["kind"] == "rerank"]
    spans: dict[str, float] = {}
    for record in trace.spans:
        spans[record["name"]] = round(spans.get(record["name"], 0.0) + record["ms"], 2)
    emb_cost = sum(e.get("cost", 0.0) for e in emb)
    rerank_cost = sum(e["cost"] for e in rerank)
    rerank_spent = sum(e["cost"] for e in rerank if not e.get("from_cache"))
    return {
        "llm_calls": len(llm),
        "llm_cache_hits": sum(1 for e in llm if e["from_cache"]),
        # 按角色计调用次数：L2 的规划与充分性检查（planner）单独看得出来
        "llm_calls_by_role": dict(Counter(str(e.get("role")) for e in llm)),
        "input_tokens": sum(e["input_tokens"] for e in llm),
        "output_tokens": sum(e["output_tokens"] for e in llm),
        "cached_input_tokens": sum(e["cached_input_tokens"] for e in llm),
        "embedding_tokens": sum(e["tokens"] for e in emb),
        "rerank_calls": len(rerank),
        "rerank_tokens": sum(e["tokens"] for e in rerank),
        "cost": sum(e["cost"] for e in llm) + emb_cost + rerank_cost,
        # 实际花费：扣除供应商缓存折扣与错峰折扣（旧版事件没有 spent 字段时按名义成本计）
        "spent": sum(e.get("spent", e["cost"]) for e in llm if not e["from_cache"])
        + emb_cost
        + rerank_spent,
        "latency_ms": trace.total_ms,
        "spans": spans,
    }


class EvalRunner:
    def __init__(
        self,
        settings: Settings,
        config: RunConfig,
        *,
        http_client: httpx2.Client | None = None,
        on_start: Callable[[int], None] | None = None,
        on_advance: Callable[[], None] | None = None,
        on_peak_wait: Callable[[str, datetime], None] | None = None,
    ) -> None:
        self.settings = settings
        self.config = config
        self.run_dir = settings.paths.eval_runs / config.name
        self.methods = build_methods(config.methods, k=config.k)
        self.ledger = Ledger(settings.paths.ledger, run_id=f"eval:{config.name}")
        # 评测是批量任务：分时价格的高峰时段默认暂停（settings.peak_policy），等高峰结束再调用
        self.llm = LLMClient(
            settings,
            cache=ResponseCache(settings.paths.llm_cache),
            ledger=self.ledger,
            http_client=http_client,
            on_peak_wait=on_peak_wait,
        )
        self.reader = Reader(self.llm)
        self.judge = Judge(self.llm)
        self._http_client = http_client
        self._embedder: TextEmbedder | None = None
        self._reranker: Reranker | None = None
        self._reranker_checked = False
        self._image_embedder: ImageEmbedder | None = None
        self._image_embedder_checked = False
        self._on_start = on_start or (lambda total: None)
        self._on_advance = on_advance or (lambda: None)
        self._lock = threading.Lock()
        self._spent = 0.0
        self._nominal = 0.0
        self._completed = 0
        self._errors = 0
        self._stop = threading.Event()
        self._inner_workers = 1

    # ---------- 对外接口 ----------

    def run(self, evalset: EvalSet) -> RunSummary:
        started = time.perf_counter()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_info(evalset)
        done = self._load_done()
        samples = evalset.samples[: self.config.limit] if self.config.limit else evalset.samples
        groups: dict[str, list[tuple[Sample, list[Method]]]] = {}
        planned = 0
        for sample in samples:
            pending = [m for m in self.methods if (sample.sample_id, m.name) not in done]
            if self.config.build_only:  # 只构建：每份对话历史算一个任务
                pending = list(self.methods)
            if pending:
                groups.setdefault(sample.space_key, []).append((sample, pending))
                planned += len(pending)
        if self.config.build_only:
            planned = len(groups)
        self._on_start(planned)
        # 对话历史比线程少时（LoCoMo 一共 3 段对话），同一段历史里的题也并行回答
        self._inner_workers = max(1, self.config.workers // max(1, len(groups)))
        with ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as pool:
            futures = [
                pool.submit(self._run_group, evalset, key, items) for key, items in groups.items()
            ]
            for future in as_completed(futures):
                future.result()
        self.ledger.close()
        return RunSummary(
            run_dir=self.run_dir,
            planned=planned,
            completed=self._completed,
            errors=self._errors,
            spent=self._spent,
            nominal_cost=self._nominal,
            elapsed_s=time.perf_counter() - started,
            stopped_by_budget=self._stop.is_set(),
        )

    def probe(self, evalset: EvalSet) -> tuple[int, int]:
        """为每道题跑一次记忆探针（不调用 LLM），特征写入 probe.jsonl：N7 检索控制器的训练数据。"""
        samples = evalset.samples[: self.config.limit] if self.config.limit else evalset.samples
        queries = [
            ProbeQuery(
                s.sample_id,
                evalset.dataset,
                s.space_key,
                s.question,
                s.question_time,
                {
                    "dataset": s.dataset,
                    "space_key": s.space_key,
                    "question_type": s.question_type,
                    "is_abstention": s.is_abstention,
                },
                evalset.session_ids,
            )
            for s in samples
        ]
        return self.probe_queries(queries)

    def probe_queries(self, queries: Sequence[ProbeQuery]) -> tuple[int, int]:
        """在指定的记忆空间上跑探针（混合流的无关查询也走这里）。

        空间必须已经构建好（之前跑过需要索引的方法，或 --build-only）；探针只读不写。
        已经算过的查询跳过。返回（新算的条数，出错的条数）。
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "probe.jsonl"
        done: set[str] = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(line)["sample_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        groups: dict[tuple[str, str, SessionIds], list[ProbeQuery]] = {}
        for query in queries:
            if query.query_id not in done:
                scheme = query.session_ids or session_id_scheme(query.dataset)
                groups.setdefault((query.dataset, query.space_key, scheme), []).append(query)
        self._on_start(sum(len(items) for items in groups.values()))
        counts = Counter[str]()

        def run_group(
            dataset: str, space_key: str, scheme: SessionIds, items: list[ProbeQuery]
        ) -> None:
            space_id = f"{dataset}:{space_key}"
            system, _ = self._open_space(dataset, space_key, scheme)
            try:
                if system.db.space_stats(space_id)["rounds"] == 0:
                    raise RuntimeError(f"空间 {space_id} 还没有构建，先跑一次评测或 --build-only")
                for query in items:
                    result = system.probe.run(space_id, query.question, query.question_time)
                    self._append(
                        "probe.jsonl",
                        {
                            "sample_id": query.query_id,
                            **query.meta,
                            "features": {k: round(v, 5) for k, v in result.features.items()},
                            "probe_ms": round(result.ms, 1),
                        },
                    )
                    with self._lock:
                        counts["done"] += 1
                    self._on_advance()
            except Exception as exc:
                with self._lock:
                    counts["errors"] += len(items)
                self._append(
                    "errors.jsonl",
                    {
                        "space_key": space_key,
                        "stage": "probe",
                        "error": f"{type(exc).__name__}: {exc}",
                        "at": datetime.now().isoformat(timespec="seconds"),
                    },
                )
            finally:
                system.close()

        with ThreadPoolExecutor(max_workers=max(1, self.config.workers)) as pool:
            futures = [
                pool.submit(run_group, dataset, key, scheme, items)
                for (dataset, key, scheme), items in groups.items()
            ]
            for future in as_completed(futures):
                future.result()
        self.ledger.close()
        return counts["done"], counts["errors"]

    # ---------- 内部实现 ----------

    @property
    def embedder(self) -> TextEmbedder:
        """只有需要索引的方法才创建向量器，这样只跑 B0 / B1 时不需要向量接口。"""
        with self._lock:
            if self._embedder is None:
                self._embedder = build_embedder(
                    self.settings, ledger=self.ledger, http_client=self._http_client
                )
            return self._embedder

    @property
    def image_embedder(self) -> ImageEmbedder | None:
        """图片向量器（N6），所有题共用一个；没有 Key 时为 None。"""
        with self._lock:
            if self._image_embedder is None and not self._image_embedder_checked:
                self._image_embedder = build_image_embedder(
                    self.settings, ledger=self.ledger, http_client=self._http_client
                )
                self._image_embedder_checked = True
            return self._image_embedder

    @property
    def reranker(self) -> Reranker | None:
        """重排模型（L2 用），所有题共用一个（内部限制并发）。没有 Key 时为 None。"""
        with self._lock:
            if self._reranker is None and not self._reranker_checked:
                self._reranker = build_reranker(
                    self.settings, llm=self.llm, ledger=self.ledger, http_client=self._http_client
                )
                self._reranker_checked = True
            return self._reranker

    def _run_group(
        self, evalset: EvalSet, space_key: str, items: list[tuple[Sample, list[Method]]]
    ) -> None:
        if self._stop.is_set() or self._budget_exhausted():  # 预算用完后不再开始新的写入
            self._stop.set()
            return
        sessions = evalset.haystacks[space_key]
        space_id = f"{evalset.dataset}:{space_key}"
        system: MemorySystem | None = None
        turns = sum(len(s.turns) for s in sessions)
        build: dict[str, Any] = {}
        try:
            if any(m.needs_index for _, methods in items for m in methods):
                system, build = self._open_space(evalset.dataset, space_key, evalset.session_ids)
                with start_trace("ingest", space=space_id) as trace:
                    report = system.ingest(to_conversation(space_id, sessions))
                row = {
                    "space_key": space_key,
                    "stage": "ingest",
                    "units": report.units_added,
                    "turns": turns,
                    **usage_from_trace(trace),
                }
                self._record_stage(row, build, did_work=report.units_added > 0)
        except Exception as exc:  # 写入失败只影响需要索引的方法，其余方法照常评测
            items = self._drop_failed(items, exc, "ingest", lambda m: m.needs_index)
            if system is not None:
                system.close()
            system = None
        if system is not None and any(
            m.needs_consolidation for _, methods in items for m in methods
        ):
            try:
                with start_trace("consolidate", space=space_id) as trace:
                    consolidation = system.consolidate(space_id, strict=True)
                row = {
                    "space_key": space_key,
                    "stage": "consolidate",
                    "turns": turns,
                    **consolidation.as_dict(),
                    "facts_total": len(system.db.get_facts(space_id)),
                    "facts_current": len(system.db.get_facts(space_id, current_only=True)),
                    "entities": len(system.db.get_entities(space_id)),
                    **usage_from_trace(trace),
                }
                self._record_stage(row, build, did_work=consolidation.sessions > 0)
            except Exception as exc:  # 巩固失败只影响需要事实记忆的方法
                items = self._drop_failed(
                    items, exc, "consolidate", lambda m: m.needs_consolidation
                )
        if build.get("_path") and system is not None:
            path = Path(build.pop("_path"))
            path.write_text(json.dumps(build, ensure_ascii=False), encoding="utf-8")
        if self.config.build_only:
            if system is not None:
                system.close()
            with self._lock:
                self._completed += 1
            self._on_advance()
            return
        pairs = [(sample, method) for sample, methods in items for method in methods]

        def answer(pair: tuple[Sample, Method]) -> None:
            sample, method = pair
            if self._stop.is_set() or self._budget_exhausted():
                self._stop.set()
                return
            try:
                row = self._evaluate(
                    sample, method, sessions, system, space_id, evalset.session_ids
                )
            except Exception as exc:
                self._record_error(sample, method, exc, stage="answer")
                return
            self._append("results.jsonl", row)
            with self._lock:
                self._completed += 1
            self._on_advance()

        try:
            if self._inner_workers > 1 and len(pairs) > 1:
                with ThreadPoolExecutor(max_workers=self._inner_workers) as pool:
                    list(pool.map(answer, pairs))
            else:
                for pair in pairs:
                    if self._stop.is_set():
                        break
                    answer(pair)
        finally:
            if system is not None:
                system.close()

    def _evaluate(
        self,
        sample: Sample,
        method: Method,
        sessions: list[EvalSession],
        system: MemorySystem | None,
        space_id: str,
        session_ids: SessionIds = "native",
    ) -> dict[str, Any]:
        ctx = MethodContext(sample, sessions, self.reader, system, space_id, session_ids)
        with start_trace("answer", method=method.name, sample=sample.sample_id) as trace:
            output = method.answer(ctx)
        usage = usage_from_trace(trace)
        with start_trace("judge") as judge_trace:
            verdict = self.judge.judge(sample, output.response)
        judge_usage = usage_from_trace(judge_trace)
        self._add_cost(usage)
        self._add_cost(judge_usage)
        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "space_key": sample.space_key,
            "question_type": sample.question_type,
            "ability": sample.ability,
            "is_abstention": sample.is_abstention,
            "method": method.name,
            "question": sample.question,
            "answer": sample.answer,
            "response": output.response,
            "evidence_sufficient": output.evidence_sufficient,
            "correct": verdict.correct,
            "judge_raw": verdict.raw,
            "f1": token_f1(output.response, sample.answer) if sample.dataset == "locomo" else None,
            "retrieval": self._retrieval(sample, sessions, output.hits) if output.hits else None,
            "retrieved": [
                [h.unit.session_id, h.unit.meta.get("turn_indices", []), round(h.score, 4)]
                for h in output.hits
            ],
            "retrieved_facts": [
                [
                    fh.fact.statement,
                    "current" if fh.fact.is_current else fh.fact.supersede_reason,
                    fh.fact.session_id,
                    round(fh.score, 4),
                ]
                for fh in output.facts
            ],
            "details": output.details or None,
            "usage": usage,
            "judge_usage": judge_usage,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
        }

    @staticmethod
    def _retrieval(
        sample: Sample, sessions: list[EvalSession], hits: list[Hit]
    ) -> dict[str, float] | None:
        if sample.is_abstention:
            return None
        turns = sample_evidence_turns(sample, sessions)
        wanted = set(sample.evidence_session_ids)
        relevant_units = 0
        for session in to_session_ins(s for s in sessions if s.session_id in wanted):
            for unit in build_rounds(build_turns("eval", session)):
                indices = unit.meta["turn_indices"]
                if any((session.session_id, i) in turns for i in indices):
                    relevant_units += 1
        return retrieval_metrics(
            hits,
            evidence_sessions=sample.evidence_session_ids,
            evidence_turns=turns,
            relevant_units=relevant_units,
        )

    def _open_space(
        self, dataset: str, space_key: str, session_ids: SessionIds | None = None
    ) -> tuple[MemorySystem, dict[str, Any]]:
        """打开这份对话历史的记忆空间。复用时（reuse_spaces）存在磁盘上，下次运行直接用；
        build.json 记下首次构建时的写入与巩固统计，复用时报告里的写入成本照样有数。

        会话 ID 改写过（anon）与保留原始 ID（raw）的空间放在不同目录（space_dir），数据库里
        另记一份写法：对不上的空间作废重建，两种 ID 不会混在一个空间里。"""
        parts: dict[str, Any] = {
            "embedder": self.embedder,
            "llm": self.llm,
            "ledger": self.ledger,
            "reranker": self.reranker,
            "image_embedder": self.image_embedder,
        }
        if not self.config.reuse_spaces:
            return MemorySystem.ephemeral(self.settings, **parts), {}
        scheme = session_ids or session_id_scheme(dataset)
        directory = space_dir(self.settings, dataset, self.embedder, space_key, scheme)
        try:
            system = _on_disk(self.settings, directory, scheme, parts)
        except ConfigMismatchError:  # 换了向量模型或会话 ID 的写法：旧空间作废，重建
            shutil.rmtree(directory, ignore_errors=True)
            system = _on_disk(self.settings, directory, scheme, parts)
        path = directory / "build.json"
        build = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        build["_path"] = str(path)
        return system, build

    def _record_stage(self, row: dict[str, Any], build: dict[str, Any], *, did_work: bool) -> None:
        """记录写入或巩固的统计：这次真的做了就记实测值；复用已有空间时记首次构建的统计。"""
        stage = row["stage"]
        self._add_cost(row)  # 花费只算这次真实发生的
        if did_work or stage not in build:
            if "_path" in build:
                build[stage] = row
            self._append("ingest.jsonl", row)
            return
        self._append("ingest.jsonl", {**build[stage], "reused": True})

    def _check_resume(self, info: dict[str, Any], session_ids: SessionIds = "native") -> None:
        _check_resume_compatible(info, self.settings, self.config, session_ids)

    def _drop_failed(
        self,
        items: list[tuple[Sample, list[Method]]],
        exc: Exception,
        stage: str,
        affected: Callable[[Method], bool],
    ) -> list[tuple[Sample, list[Method]]]:
        """某个准备阶段失败：受影响的方法记为出错（下次重跑会重试），其余方法照常评测。"""
        for sample, methods in items:
            for method in methods:
                if affected(method):
                    self._record_error(sample, method, exc, stage=stage)
        return [(s, [m for m in ms if not affected(m)]) for s, ms in items]

    def _record_error(self, sample: Sample, method: Method, exc: Exception, *, stage: str) -> None:
        self._append(
            "errors.jsonl",
            {
                "sample_id": sample.sample_id,
                "method": method.name,
                "stage": stage,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
                "at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        with self._lock:
            self._errors += 1
        self._on_advance()

    def _add_cost(self, usage: dict[str, Any]) -> None:
        with self._lock:
            self._spent += usage["spent"]
            self._nominal += usage["cost"]

    def _budget_exhausted(self) -> bool:
        if self.config.max_cost is None:
            return False
        with self._lock:
            return self._spent >= self.config.max_cost

    def _append(self, filename: str, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock, (self.run_dir / filename).open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _load_done(self) -> set[tuple[str, str]]:
        path = self.run_dir / "results.jsonl"
        if not path.exists():
            return set()
        done = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # 进程被中断时最后一行可能不完整，跳过后会重新评测
                continue
            done.add((row["sample_id"], row["method"]))
        return done

    def _write_run_info(self, evalset: EvalSet) -> None:
        """记录本次运行的配置与环境，保证结果可追溯；续跑时保留首次的信息并追加续跑时间。"""
        path = self.run_dir / "run.json"
        info: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        now = datetime.now().isoformat(timespec="seconds")
        if not info:
            roles = {
                role: {
                    "model": rc.model,
                    "model_id": self.settings.models[rc.model].model_id,
                    "price": self.settings.models[rc.model].price.model_dump(),
                }
                for role, rc in self.settings.roles.items()
            }
            info = {
                "config": asdict(self.config),
                "samples": len(evalset.samples),
                "roles": roles,
                "embedding": {
                    "model": self.settings.embedding.model,
                    "dim": self.settings.embedding.dim,
                    "backend": self.settings.embedding.backend,
                },
                "retrieval": self.settings.retrieval.model_dump(),
                "consolidation": {
                    "pipeline_version": PIPELINE_VERSION,
                    **self.settings.consolidation.model_dump(),
                },
                "git": _git_version(),
                "created": now,
                "resumed": [],
            }
            if evalset.session_ids != "native":  # LoCoMo 的 run.json 不变
                info["session_ids"] = evalset.session_ids
        else:
            self._check_resume(info, evalset.session_ids)
            info["resumed"].append(now)
            info["config"]["methods"] = sorted(
                set(info["config"]["methods"]) | set(self.config.methods)
            )
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _check_resume_compatible(
    info: dict[str, Any],
    settings: Settings,
    config: RunConfig,
    session_ids: SessionIds = "native",
) -> None:
    """续跑同名运行时，关键配置必须一致；否则新旧结果会混在一个目录里（N4 差点发生过）。

    会话 ID 的写法也要一致：改写之前的运行没有记下写法，LongMemEval / MemLens 的都是原始 ID。"""
    previous = info.get("config", {})
    before_ids = info.get("session_ids") or session_id_scheme(str(previous.get("dataset")), True)
    checks = {
        "数据集": (previous.get("dataset"), config.dataset),
        "划分": (previous.get("split"), config.split),
        "会话 ID": (before_ids, session_ids),
        "向量模型": (info.get("embedding", {}).get("model"), settings.embedding.model),
        "回答模型": (
            info.get("roles", {}).get("reader", {}).get("model"),
            settings.roles["reader"].model if "reader" in settings.roles else None,
        ),
    }
    changed = [f"{name}：{old} → {new}" for name, (old, new) in checks.items() if old != new]
    if changed:
        raise ConfigMismatchError(
            f"运行 {config.name} 已存在，但关键配置变了（{'；'.join(changed)}）。"
            "续跑会把新旧结果混在一起：请换一个运行名（--name）。"
        )


def _safe_name(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", key)


def space_dir(
    settings: Settings,
    dataset: str,
    embedder: TextEmbedder,
    space_key: str,
    session_ids: SessionIds | None = None,
) -> Path:
    """评测记忆空间的目录：按向量模型分开存，换模型对比时两套空间可以共存。

    会话 ID 改写过的空间放在 <数据集>-anon 下，与之前（原始 ID）的空间分开（见 datasets 的
    storage_name）；session_ids 为 None 时按环境变量，与加载数据集时一致。LoCoMo 不变。"""
    model = _safe_name(f"{embedder.name}-{embedder.dim}")
    folder = storage_name(dataset, session_ids)
    return settings.paths.eval_spaces / folder / model / _safe_name(space_key)


SESSION_IDS_META = "session_ids"  # 评测空间的数据库里记下的会话 ID 写法


def _on_disk(
    settings: Settings, directory: Path, scheme: SessionIds, parts: dict[str, Any]
) -> MemorySystem:
    """打开磁盘上的评测空间；LongMemEval / MemLens 的空间核对会话 ID 的写法（不一致时报
    ConfigMismatchError，调用方删掉重建）。改写之前建的空间没有这条记录，都是原始 ID：
    它们在原来的目录里，只会被原始 ID 的运行打开，第一次打开时记为 raw。LoCoMo 不记。"""
    system = MemorySystem.on_disk(settings, directory, **parts)
    if scheme != "native":
        try:
            system.db.check_or_set_meta(SESSION_IDS_META, scheme)
        except ConfigMismatchError:
            system.close()
            raise
    return system


def _git_version() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}{'-dirty' if dirty else ''}"
