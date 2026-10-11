"""评测运行器：写入 → 回答 → 判分，结果逐行追加到 JSONL。

- 断点续跑：已完成的（题目，方法）组合会被跳过；出错的组合记入 errors.jsonl，下次运行会重试；
- 并发：按对话历史分组，每组在一个线程里处理，同组题目共用一次写入（LoCoMo 一段对话约 200 题）；
- 用量：每个（题目，方法）在独立的 trace 里执行，从 trace 事件汇总 token、调用次数与名义成本；
  判分单独统计，不计入方法成本；
- 预算：按实际花费（不含本地缓存命中）累计，超过上限就不再开始新的组合。
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2

from cuemem.config import PROJECT_ROOT, Settings
from cuemem.core.models import Hit
from cuemem.core.trace import Trace, start_trace
from cuemem.embed import build_embedder
from cuemem.embed.base import TextEmbedder
from cuemem.evals.datasets import (
    EvalSession,
    EvalSet,
    Sample,
    sample_evidence_turns,
    to_conversation,
    to_session_ins,
)
from cuemem.evals.judge import Judge, token_f1
from cuemem.evals.methods import Method, MethodContext, build_methods
from cuemem.evals.metrics import retrieval_metrics
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.memory import MemorySystem
from cuemem.read.reader import Reader
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
    spans: dict[str, float] = {}
    for record in trace.spans:
        spans[record["name"]] = round(spans.get(record["name"], 0.0) + record["ms"], 2)
    emb_cost = sum(e.get("cost", 0.0) for e in emb)
    return {
        "llm_calls": len(llm),
        "llm_cache_hits": sum(1 for e in llm if e["from_cache"]),
        "input_tokens": sum(e["input_tokens"] for e in llm),
        "output_tokens": sum(e["output_tokens"] for e in llm),
        "cached_input_tokens": sum(e["cached_input_tokens"] for e in llm),
        "embedding_tokens": sum(e["tokens"] for e in emb),
        "cost": sum(e["cost"] for e in llm) + emb_cost,
        "spent": sum(e["cost"] for e in llm if not e["from_cache"]) + emb_cost,
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
    ) -> None:
        self.settings = settings
        self.config = config
        self.run_dir = settings.paths.eval_runs / config.name
        self.methods = build_methods(config.methods, k=config.k)
        self.ledger = Ledger(settings.paths.ledger, run_id=f"eval:{config.name}")
        self.llm = LLMClient(
            settings,
            cache=ResponseCache(settings.paths.llm_cache),
            ledger=self.ledger,
            http_client=http_client,
        )
        self.reader = Reader(self.llm)
        self.judge = Judge(self.llm)
        self._http_client = http_client
        self._embedder: TextEmbedder | None = None
        self._on_start = on_start or (lambda total: None)
        self._on_advance = on_advance or (lambda: None)
        self._lock = threading.Lock()
        self._spent = 0.0
        self._nominal = 0.0
        self._completed = 0
        self._errors = 0
        self._stop = threading.Event()

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
            if pending:
                groups.setdefault(sample.space_key, []).append((sample, pending))
                planned += len(pending)
        self._on_start(planned)
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

    def _run_group(
        self, evalset: EvalSet, space_key: str, items: list[tuple[Sample, list[Method]]]
    ) -> None:
        if self._stop.is_set() or self._budget_exhausted():  # 预算用完后不再开始新的写入
            self._stop.set()
            return
        sessions = evalset.haystacks[space_key]
        space_id = f"{evalset.dataset}:{space_key}"
        system: MemorySystem | None = None
        try:
            if any(m.needs_index for _, methods in items for m in methods):
                system = MemorySystem.ephemeral(
                    self.settings, embedder=self.embedder, llm=self.llm, ledger=self.ledger
                )
                with start_trace("ingest", space=space_id) as trace:
                    report = system.ingest(to_conversation(space_id, sessions))
                usage = usage_from_trace(trace)
                self._add_cost(usage)
                self._append(
                    "ingest.jsonl",
                    {"space_key": space_key, "units": report.units_added, **usage},
                )
        except Exception as exc:  # 写入失败只影响需要索引的方法，其余方法照常评测
            for sample, methods in items:
                for method in methods:
                    if method.needs_index:
                        self._record_error(sample, method, exc, stage="ingest")
            items = [(s, [m for m in ms if not m.needs_index]) for s, ms in items]
            if system is not None:
                system.close()
            system = None
        try:
            for sample, methods in items:
                for method in methods:
                    if self._budget_exhausted():
                        self._stop.set()
                        return
                    try:
                        row = self._evaluate(sample, method, sessions, system, space_id)
                    except Exception as exc:
                        self._record_error(sample, method, exc, stage="answer")
                        continue
                    self._append("results.jsonl", row)
                    with self._lock:
                        self._completed += 1
                    self._on_advance()
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
    ) -> dict[str, Any]:
        ctx = MethodContext(sample, sessions, self.reader, system, space_id)
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
                "git": _git_version(),
                "created": now,
                "resumed": [],
            }
        else:
            info["resumed"].append(now)
            info["config"]["methods"] = sorted(
                set(info["config"]["methods"]) | set(self.config.methods)
            )
        path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
