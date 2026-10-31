"""trace 的 JSONL（core/trace.py）：遗忘时删除、按天删除过期的文件。"""

from __future__ import annotations

import errno
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


class _ShortWrite:
    """追加时只写进去一半（断在 "我" 这个字中间）就抛 ENOSPC 的文件，与磁盘满时的短写相同。"""

    def __init__(self, f: Any) -> None:
        self.f = f

    def __enter__(self) -> _ShortWrite:
        return self

    def __exit__(self, *_: object) -> None:
        self.f.close()

    def write(self, text: str) -> int:
        data = text.encode("utf-8")
        self.f.buffer.write(data[: data.index("我".encode()) + 1])
        self.f.flush()
        raise OSError(errno.ENOSPC, "No space left on device")


def test_a_failed_append_leaves_the_file_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """追加一行写到一半失败（磁盘满）：文件截回追加之前的样子，再把错误抛给 start_trace（它记一条
    警告）。以前半行留在文件里，下一次追加接在它后面：两条都解析不了，断在一个汉字中间时这一行
    还不是 UTF-8，之后遗忘与过期读到它就抛 UnicodeDecodeError。"""
    sink = TraceSink(tmp_path / "traces")
    path = sink.write(Trace("retrieve", space="alice", question="hello"))
    before = path.read_bytes()
    real_open = Path.open

    def short(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        f = real_open(self, mode, *args, **kwargs)
        return _ShortWrite(f) if "a" in mode else f

    monkeypatch.setattr(Path, "open", short)
    with pytest.raises(OSError, match="No space"):
        sink.write(Trace("retrieve", space="bob", question="我住在哪里"))
    monkeypatch.undo()
    assert path.read_bytes() == before

    sink.write(Trace("retrieve", space="carol", question="你好"))
    lines = path.read_bytes().decode("utf-8").splitlines()
    assert [json.loads(x)["attrs"]["space"] for x in lines] == ["alice", "carol"]


def test_lines_that_are_not_utf8_are_kept_byte_for_byte(tmp_path: Path) -> None:
    """不是 UTF-8 的行（这一版之前的写入断在一个汉字中间，下一次追加接在它后面）不解析、原样
    留着；以前整个文件按 UTF-8 读，remove 与 prune 都抛 UnicodeDecodeError（遗忘在提交之后
    报错，过期的 trace 删不掉）。合法的行写回去逐字节相同。"""
    directory = tmp_path / "traces"
    _write_days(directory, {"20260101": [_line("victim", "秘密"), _line("other", "留着")]})
    path = directory / "20260101.jsonl"
    glued = b'{"attrs": {"space": "victim", "question": "\xe6\x88' + _line("other", "x").encode()
    with path.open("ab") as f:
        f.write(glued + b"\n")
    sink = TraceSink(directory)

    assert sink.remove(_victim) == 1
    assert path.read_bytes() == (_line("other", "留着") + "\n").encode() + glued + b"\n"
    files, questions = sink.prune(date(2026, 1, 2))
    assert files == 1 and questions == ["留着"]
    assert not path.exists()


def test_remove_deletes_a_file_it_cannot_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """重写不了的文件（磁盘满：重写要先有这个文件那么大的空间）整个删掉，删除不要空间：同一天
    别的 trace 也没了，记一条警告，其中每条 trace 的提问放进 dropped（它们的查询向量要一起删）。
    以前 OSError 从这里抛出，遗忘已经提交了却跳过了清理向量缓存，还留下一个 .part。删也删不掉、
    读不了的文件记警告、名字放进 left，接着处理别的文件，不抛出。"""
    directory = tmp_path / "traces"
    _write_days(
        directory,
        {
            "20260101": [_line("victim", "secret"), _line("other", "a"), "not json"],
            "20260102": [_line("victim", "stuck"), _line("other", "b")],
            "20260103": [_line("other", "c")],
        },
    )
    (directory / "20260104.jsonl").mkdir()  # 读不了
    real_write, real_unlink = Path.write_text, Path.unlink

    def full(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.name.endswith(".jsonl.part"):
            real_write(self, "partial", encoding="utf-8")
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_write(self, *args, **kwargs)

    def stuck(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name == "20260102.jsonl":
            raise PermissionError(errno.EACCES, "Permission denied")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full)
    monkeypatch.setattr(Path, "unlink", stuck)
    dropped: list[str] = []
    left: list[str] = []
    with caplog.at_level("WARNING", "cuemem.core.trace"):
        assert TraceSink(directory).remove(_victim, dropped=dropped, left=left) == 1
    assert sorted(dropped) == ["a", "secret"]
    assert left == ["20260102.jsonl", "20260104.jsonl"]
    assert len(caplog.records) == 3
    assert sorted(p.name for p in directory.iterdir()) == [
        "20260102.jsonl",
        "20260103.jsonl",
        "20260104.jsonl",
    ]
    assert "stuck" in (directory / "20260102.jsonl").read_text("utf-8")
