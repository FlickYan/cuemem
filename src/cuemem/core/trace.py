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
import logging
import threading
import time
import uuid
from collections.abc import Callable, Generator, Iterator
from contextlib import closing, contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        self.failing = False  # 上一次写入失败了（见 start_trace：接连失败只记一次警告）
        self._lock = threading.Lock()

    def remove(
        self, predicate: Callable[[dict[str, Any]], bool], *, contains: str | None = None
    ) -> int:
        """删除满足条件的 trace（遗忘时用：trace 的属性里有提问与回答的原文）。返回删除的条数。

        contains：只解析含有这段文字的行，别的行原样留着、不交给 predicate（遗忘按空间 ID 的
        写法先筛一遍，见 MemorySystem._remove_traces；不解析别的用户的 trace，快十几倍）。
        predicate 对同一条 trace 可能调用两次（见下）。

        每个文件先不拿锁看一遍有没有要删的，没有就不碰它；有的才拿着写 trace 的锁重新读一遍、
        写回（期间追加的行不会丢）。以前整个过程拿着这把锁，每次请求结束写 trace 都要等全部用户
        的 trace 读写一遍；也不能只是一个文件一个文件地拿锁：删除的线程放开锁马上又拿到，写 trace
        的线程照样等到最后。不拿锁读的时候正在追加的最后一行可能只读到一半、当作不删：与这一行
        晚一点写、写在看完这个文件之后一样（遗忘期间这个空间没有进行中的读取，见
        MemorySystem.reading，它们的 trace 都在读取里写）。

        只重写有删除的文件（以前第一处删除之后的文件都重写一遍）。按 "\\n" 分行（逐行读文件）：
        splitlines 还在 U+2028、U+2029、U+0085 处断开，ensure_ascii=False 写出的 JSON 里这几个
        字符不转义，含有它们的 trace 断成几段解析不了的行、原样留下（遗忘删不掉提问里有它们的
        trace），重写时还把别的空间的这种行写成了几行。"""
        removed = 0
        for path in sorted(self.directory.glob("*.jsonl")):
            try:
                with closing(_judged(path, predicate, contains)) as lines:
                    if not any(doomed for _, doomed in lines):
                        continue
                with self._lock:
                    kept: list[str] = []
                    here = 0
                    for line, doomed in _judged(path, predicate, contains):
                        if doomed:
                            here += 1
                        else:
                            kept.append(line)
                    if here:
                        partial = path.with_suffix(".jsonl.part")
                        partial.write_text("".join(f"{k}\n" for k in kept), encoding="utf-8")
                        partial.replace(path)
                        removed += here
            except FileNotFoundError:  # 刚过期删掉了（见 prune）
                continue
        return removed

    def write(self, trace: Trace) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{trace.started_at:%Y%m%d}.jsonl"
        line = json.dumps(trace.to_dict(), ensure_ascii=False, default=str)
        with self._lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return path

    def prune(self, before: date) -> tuple[int, list[str]]:
        """删掉 before 之前那些天的 trace 文件（按文件名里的日期；包括重写到一半留下的
        .jsonl.part），返回删掉的文件数与其中 trace 的提问（属性里的 question、message：它们的
        查询向量要一起删，见 MemorySystem.prune_traces）。

        不拿锁读（以前的文件不会再追加，读的时候写 trace 不用等），拿着锁删：与写入、遗忘的重写
        错开。分行与 remove 相同。"""
        cutoff = f"{before:%Y%m%d}"
        questions: list[str] = []

        def collect(record: dict[str, Any]) -> bool:
            attrs = record.get("attrs")
            if isinstance(attrs, dict):
                questions.extend(str(attrs[k]) for k in ("question", "message") if attrs.get(k))
            return False

        files = 0
        for path in sorted(self.directory.glob("*.jsonl*")):
            day, _, suffix = path.name.partition(".")
            dated = len(day) == 8 and day.isdigit()
            if suffix not in ("jsonl", "jsonl.part") or not dated or day >= cutoff:
                continue
            try:
                for _ in _judged(path, collect, None):
                    pass
                with self._lock:
                    path.unlink()
            except FileNotFoundError:  # 别的线程刚删掉或换掉了它（遗忘重写时的 .part）
                continue
            files += 1
        return files, questions


def _judged(
    path: Path, predicate: Callable[[dict[str, Any]], bool], contains: str | None
) -> Generator[tuple[str, bool], None, None]:
    """逐行读一个 trace 文件，给出（行，要不要删）。只按 "\\n" 分行（见 TraceSink.remove）；
    不含 contains 的行与解析不了的行都不删，也不交给 predicate。"""
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if contains is not None and contains not in line:
                yield line, False
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                yield line, False
                continue
            yield line, isinstance(record, dict) and predicate(record)


_current: ContextVar[Trace | None] = ContextVar("cuemem_trace", default=None)


@contextmanager
def start_trace(name: str, sink: TraceSink | None = None, **attrs: Any) -> Iterator[Trace]:
    """开始一个 trace；如果已经在某个 trace 里（例如评测 runner 调用 MemorySystem.ask），
    就记为父 trace 的一个阶段，事件也都记到父 trace 上，便于外层统一统计用量。

    trace 只是排查用的记录，在块做完（写入已经提交、遗忘已经做完）之后才写：写不进去（磁盘满、
    目录不可写）时记一条警告，不抛出。以前它的 OSError 让已经提交的写入报错，调用方原样重试就
    写了两遍；块里出错时还盖住了原来的异常。接连失败只记一次，写成功之后再失败时再记。"""
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
            try:
                sink.write(trace)
            except Exception as exc:
                if not sink.failing:
                    logger.warning(
                        "trace 写不进 %s，这次（%s）与之后的 trace 不记，直到写入恢复：%s",
                        sink.directory,
                        name,
                        exc,
                    )
                sink.failing = True
            else:
                sink.failing = False


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
