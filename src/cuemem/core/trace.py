"""轻量 trace：记录一次请求内各阶段的耗时与关键事件，写成 JSONL 便于事后分析。

用法：
    with start_trace("ask", sink, space="demo"):
        with span("retrieve"):
            ...
        record_event("llm", model="deepseek-flash", input_tokens=812)

LLM 客户端与向量化模块会自动往当前 trace 里记事件；不在 trace 中时这些调用什么也不做。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any


class Trace:
    def __init__(self, name: str, **attrs: Any) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.attrs: dict[str, Any] = attrs
        self.started_at = datetime.now()
        self.spans: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.total_ms: float | None = None
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()

    def add_event(self, kind: str, **data: Any) -> None:
        with self._lock:
            self.events.append({"kind": kind, "t_ms": self._elapsed_ms(), **data})

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        record: dict[str, Any] = {"name": name, "start_ms": self._elapsed_ms(), **attrs}
        started = time.perf_counter()
        try:
            yield record
        finally:
            record["ms"] = round((time.perf_counter() - started) * 1000, 2)
            with self._lock:
                self.spans.append(record)

    def finish(self) -> None:
        self.total_ms = self._elapsed_ms()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "total_ms": self.total_ms,
            "attrs": self.attrs,
            "spans": self.spans,
            "events": self.events,
        }

    def _elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._t0) * 1000, 2)


class TraceSink:
    """把 trace 追加写入按日期分文件的 JSONL。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()

    def remove(self, predicate: Callable[[dict[str, Any]], bool]) -> int:
        """删除满足条件的 trace（遗忘时用：trace 的属性里有提问与回答的原文）。返回删除的条数。"""
        removed = 0
        with self._lock:
            for path in sorted(self.directory.glob("*.jsonl")):
                kept: list[str] = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        kept.append(line)
                        continue
                    if isinstance(record, dict) and predicate(record):
                        removed += 1
                    else:
                        kept.append(line)
                if removed:
                    partial = path.with_suffix(".jsonl.part")
                    partial.write_text("".join(f"{k}\n" for k in kept), encoding="utf-8")
                    partial.replace(path)
        return removed

    def write(self, trace: Trace) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{trace.started_at:%Y%m%d}.jsonl"
        line = json.dumps(trace.to_dict(), ensure_ascii=False, default=str)
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return path


_current: ContextVar[Trace | None] = ContextVar("cuemem_trace", default=None)


@contextmanager
def start_trace(name: str, sink: TraceSink | None = None, **attrs: Any) -> Iterator[Trace]:
    """开始一个 trace；如果已经在某个 trace 里（例如评测 runner 调用 MemorySystem.ask），
    就记为父 trace 的一个阶段，事件也都记到父 trace 上，便于外层统一统计用量。"""
    parent = _current.get()
    if parent is not None:
        with parent.span(name, **attrs):
            yield parent
        return
    trace = Trace(name, **attrs)
    token = _current.set(trace)
    try:
        yield trace
    finally:
        trace.finish()
        _current.reset(token)
        if sink is not None:
            sink.write(trace)


def current_trace() -> Trace | None:
    return _current.get()


def record_event(kind: str, **data: Any) -> None:
    """往当前 trace 记一个事件；不在 trace 中时什么也不做。"""
    trace = _current.get()
    if trace is not None:
        trace.add_event(kind, **data)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
    """在当前 trace 中为一个阶段计时；不在 trace 中时只返回一个普通字典。"""
    trace = _current.get()
    if trace is None:
        yield dict(attrs)
        return
    with trace.span(name, **attrs) as record:
        yield record
