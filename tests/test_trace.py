"""trace 的 JSONL（core/trace.py）：遗忘时删除、按天删除过期的文件。"""

from __future__ import annotations

import json
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from cuemem.core.trace import Trace, TraceSink


def _line(space: str, question: str, trace_id: str = "t") -> str:
    record = {
        "trace_id": trace_id,
        "name": "retrieve",
        "attrs": {"space": space, "question": question},
    }
    return json.dumps(record, ensure_ascii=False)


def _write_days(directory: Path, days: dict[str, list[str]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for day, lines in days.items():
        (directory / f"{day}.jsonl").write_text("".join(f"{x}\n" for x in lines), "utf-8")


def _victim(record: dict[str, Any]) -> bool:
    return bool(record["attrs"]["space"] == "victim")


class _CountingLock:
    """记下 TraceSink 的锁拿了几次。"""

    def __init__(self) -> None:
        self.inner = threading.Lock()
        self.taken = 0

    def __enter__(self) -> None:
        self.inner.acquire()
        self.taken += 1

    def __exit__(self, *_: object) -> None:
        self.inner.release()


def test_remove_rewrites_only_files_with_removals(tmp_path: Path) -> None:
    """只重写有删除的文件：以前删除的计数跨文件累加，第一处删除之后的文件都重写一遍。只在重写
    有删除的文件时拿写 trace 的锁：以前整个过程拿着它，别的请求写 trace 时要等全部文件读写完。"""
    directory = tmp_path / "traces"
    days = ["20260101", "20260102", "20260103"]
    _write_days(
        directory,
        {
            days[0]: [_line("victim", "secret"), _line("other", "a")],
            days[1]: [_line("other", "b")],
            days[2]: [_line("other", "c")],
        },
    )
    old = {d: (directory / f"{d}.jsonl").stat() for d in days}
    for d in days:  # 改时间之后再比：重写了的文件时间会变
        os.utime(directory / f"{d}.jsonl", ns=(1_000_000_000, 1_000_000_000))
    sink = TraceSink(directory)
    lock = _CountingLock()
    sink._lock = lock  # type: ignore[assignment]

    assert sink.remove(_victim) == 1
    after = {d: (directory / f"{d}.jsonl").stat() for d in days}
    assert after[days[0]].st_ino != old[days[0]].st_ino  # 有删除的文件换成了新写的
    for d in days[1:]:
        assert after[d].st_ino == old[d].st_ino and after[d].st_mtime_ns == 1_000_000_000
    assert lock.taken == 1
    assert (directory / f"{days[0]}.jsonl").read_text("utf-8") == _line("other", "a") + "\n"


@pytest.mark.parametrize("sep", ["\u2028", "\u2029", "\x85"])
def test_remove_keeps_other_lines_with_line_separators_intact(tmp_path: Path, sep: str) -> None:
    """按 "\\n" 分行：ensure_ascii=False 写出的 JSON 里 U+2028、U+2029、U+0085 不转义，以前
    splitlines 在这几个字符处断开，含有它们的 trace 断成几行解析不了的，遗忘删不掉提问里有它们
    的 trace，重写文件时还把别的空间的这种行拆成几行。"""
    directory = tmp_path / "traces"
    sink = TraceSink(directory)
    for space, question in [("other", f"first{sep}second"), ("victim", f"my PIN{sep}4321")]:
        trace = Trace("retrieve", space=space, question=question)
        trace.finish()
        sink.write(trace)

    assert sink.remove(_victim) == 1
    [path] = directory.glob("*.jsonl")
    lines = path.read_text("utf-8").split("\n")
    assert lines[-1] == "" and len(lines) == 2
    record = json.loads(lines[0])
    assert record["attrs"] == {"space": "other", "question": f"first{sep}second"}


def test_contains_prefilter_skips_parsing(tmp_path: Path) -> None:
    """contains：不含这段文字的行原样留着、不解析，也不交给 predicate（遗忘时按空间 ID 的写法
    先筛，见 MemorySystem._remove_traces）。"""
    directory = tmp_path / "traces"
    _write_days(
        directory,
        {
            "20260101": [_line("victim", "x"), _line("other", "y"), "not json"],
            "20260102": [_line("other", "z"), _line("victim", "w")],
        },
    )
    seen: list[str] = []

    def predicate(record: dict[str, Any]) -> bool:
        seen.append(record["attrs"]["space"])
        return _victim(record)

    needle = '"space": ' + json.dumps("victim", ensure_ascii=False)
    assert TraceSink(directory).remove(predicate, contains=needle) == 2
    assert seen and set(seen) == {"victim"}
    left = [p.read_text("utf-8") for p in sorted(directory.glob("*.jsonl"))]
    assert left == [f"{_line('other', 'y')}\nnot json\n", f"{_line('other', 'z')}\n"]


def test_prune_deletes_only_days_before_the_cutoff(tmp_path: Path) -> None:
    """按天删除过期的 trace 文件（traces_retention_days）：只删日期在截止日之前的（包括遗忘重写
    到一半留下的 .part），返回其中 trace 的提问与消息（它们的查询向量要一起删）。分行与 remove
    相同：提问里有 U+2028 也整条读出来。别的文件不碰。"""
    directory = tmp_path / "traces"
    chat = {"trace_id": "c", "name": "chat", "attrs": {"space": "u", "message": "hi"}}
    _write_days(
        directory,
        {
            "20260101": [_line("u", "old\u2028question"), json.dumps(chat), "not json"],
            "20260102": [_line("u", "kept")],
            "20260103": [_line("u", "new")],
        },
    )
    (directory / "20251231.jsonl.part").write_text(_line("u", "half") + "\n", "utf-8")
    (directory / "notes.jsonl").write_text(_line("u", "not a day") + "\n", "utf-8")
    (directory / "20251230.txt").write_text("other", "utf-8")
    sink = TraceSink(directory)

    files, questions = sink.prune(date(2026, 1, 2))
    assert files == 2
    assert sorted(questions) == ["half", "hi", "old\u2028question"]
    left = sorted(p.name for p in directory.iterdir())
    assert left == ["20251230.txt", "20260102.jsonl", "20260103.jsonl", "notes.jsonl"]
    assert sink.prune(date(2026, 1, 2)) == (0, [])
