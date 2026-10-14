"""Claude 的记忆工具（memory_20250818）接到 CueMem 上（N12）：/memories 目录的后端。

    import anthropic
    from cuemem import Memory
    from cuemem.integrations.claude import CueMemMemoryTool, with_recall

    memory = Memory()  # 默认配置用通义千问；只有 DeepSeek 的 Key 时 Memory(preset="deepseek")
    tool = CueMemMemoryTool(memory, user_id="alice")
    history, evidence = with_recall(
        [{"role": "user", "content": "I moved to Lisbon last month."}], memory, user_id="alice"
    )
    runner = anthropic.Anthropic().beta.messages.tool_runner(
        model="claude-opus-5-5", max_tokens=16000, tools=[tool], messages=history
    )
    runner.until_done()

按 anthropic 1.14 的 BetaAbstractMemoryTool 写（需要可选依赖 cuemem[anthropic]）；记忆工具是
客户端工具：Claude 发出 view / create / str_replace / insert / delete / rename，这里执行。
/memories 是个虚拟目录，里面有两类文件：

- 生成的只读文件：profile.md（核心记忆）、facts.md（当前事实，带事实 ID、日期与有效状态，注明
  出自哪个文件）、history/<事实 ID>.md（一条事实的版本链，目录里只列出有过变化的）。写入、改名、
  删除一律拒绝，并告诉 Claude 该怎么做；
- Claude 自己的文件：原文存在数据目录下的 claude_memory.sqlite（按记忆空间分开）。create、
  str_replace、insert 的每次改动另写一条笔记进 CueMem 的原始日志（一条笔记一个会话，与 MCP 的
  remember 相同，后台抽取成事实）。新文件写全文；之后的改动只写改了的行（连同所在的标题）：
  换掉的旧行注明"被取代"，删掉的旧行注明"不再适用"，都带上它最初写下的日期，对账据此把旧事实
  标成被取代（历史保留），没变的内容也不会被重复抽取。标题决定含义：一行从"# 过敏"挪到
  "# 以前的过敏"是改动（注明从哪里挪来），只在同一节里挪位置的不算。rename 只搬动文件与出处，
  不写笔记；delete 遗忘这个文件写进日志的全部笔记，由剩下的日志重建派生记忆，返回哪些事实不再
  记得、哪些旧版本恢复为当前。create 已有的文件时覆盖（Claude 的工具说明就是"创建或覆盖"）。

删掉一行不等于遗忘（不可逆的遗忘只在删除文件时发生）：删掉的那一行可能只是过时了，也可能是
整理；笔记写成"不再适用"，历史保留，工具结果告诉 Claude 要彻底遗忘就删掉文件再重建。

遗忘在两个方向上都成立（D-021）：文件的某条笔记在别处被遗忘（Memory.delete、MCP 的 forget、
转述追查）之后，这个文件从 /memories 里去掉——不知道被遗忘的信息在文件的哪几行，留着就泄露了；
它其余的笔记仍在 CueMem 里，记录保留（facts.md 注明出自"已去掉"的文件），删除这个路径或
clear_all_memory 照样能遗忘它们。整个记忆空间被删除（Memory.delete_all）之后，下次访问时这个
空间的文件全部清掉；直接用本模块的 delete_all 则当场清掉。文件原文不在 Memory.export 里（导出
的是原始日志，其中有每次改动的笔记）。

几个工具实例（例如每个请求一个）可以共用一个存储文件：同一进程里同一个记忆空间的读改写串行；
跨进程时保存前核对文件没有被别人改过，改过就撤销刚写的笔记并报错，让 Claude 重新查看。

路径只认 /memories 下的；拒绝 . 与 ..、隐藏名、反斜杠、控制字符、百分号编码、过深过长的路径。
单个文件、文件数、总字符数都有上限；查看超过 16000 字符时截断，提示用 view_range 往下翻。

with_recall / awith_recall（可选）：调用模型之前先过检索控制器（L0 只给画像），证据块与记忆
代理（api/proxy.py）同一个格式、同一个位置，放在最新一句用户消息的开头（后面跟着的
mid-conversation system 消息不算）。返回的列表就是之后的对话历史：下一轮在它后面追加，再调用
with_recall——只往后加，已经发出去的消息一个字节都不改（Claude Opus 5.5 等模型的 preserved
thinking 要求前缀不变，否则 400 或丢掉思考块，提示缓存也失效）。检索出错时照常返回原消息，
level 为 "error"，这一轮只是不带记忆。
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import re
import sqlite3
import threading
import unicodedata
import uuid
import warnings
import weakref
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple, cast

# 没装 anthropic 可选依赖时 mypy 也要能跑（CI 装了，本地 uv sync 默认不装）
from anthropic.lib.tools import (  # type: ignore[import-not-found, unused-ignore]
    BetaAbstractMemoryTool,
    BetaAsyncAbstractMemoryTool,
    ToolError,
)
from anthropic.types.beta import (  # type: ignore[import-not-found, unused-ignore]
    BetaCacheControlEphemeralParam,
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
    BetaMemoryTool20250818ViewCommand,
    BetaMessageParam,
)
from anyio import to_thread

from cuemem.api.proxy import MEMORY_NOTE, _inject
from cuemem.core.models import Fact, Turn
from cuemem.core.timeutil import day
from cuemem.core.trace import record_event
from cuemem.memory import RecallMode
from cuemem.read.controller import english_reason
from cuemem.read.facts import fact_status
from cuemem.read.reader import memory_blocks
from cuemem.sdk import AsyncMemory, Memory, space_for
from cuemem.store.sqlite import MemoryDB

__all__ = [
    "AsyncCueMemMemoryTool",
    "CueMemMemoryTool",
    "Evidence",
    "awith_recall",
    "delete_all",
    "with_recall",
]

logger = logging.getLogger(__name__)

ROOT = "/memories"
PROFILE = f"{ROOT}/profile.md"
FACTS = f"{ROOT}/facts.md"
HISTORY = f"{ROOT}/history"
STORE_NAME = "claude_memory.sqlite"  # 放在数据目录（paths.db 所在目录）下

MAX_FILE_CHARS = 20_000  # 单个文件
MAX_FILES = 200  # 每个记忆空间
MAX_TOTAL_CHARS = 400_000  # 每个记忆空间所有文件合计
MAX_VIEW_CHARS = 16_000  # 一次查看最多返回的字符数（Claude 的工具说明就是按这个数截断）
MAX_LINES = 999_999
LINE_WIDTH = 6
MAX_PATH_CHARS = 256
MAX_NAME_CHARS = 100
MAX_DEPTH = 8  # /memories 下最多几层
MAX_HISTORY_LISTED = 50  # history/ 目录里列出的版本链数（其余的按事实 ID 照样能看）
MAX_LISTED_FACTS = 20  # 删除文件的结果里最多列出的事实数

_PERCENT = re.compile(r"%[0-9A-Fa-f]{2}")

PROFILE_HEAD = (
    "# Profile\n"
    "The most important facts that are currently true about the user (core memory), kept up "
    "to date by the memory system. Read-only; already remembered, so don't copy it into your "
    "files.\n\n"
)
FACTS_HEAD = (
    "# Facts\n"
    "What the memory system currently knows, extracted from your memory files and past "
    "conversations and kept up to date automatically. Each line: [fact id] (when it was said; "
    "since when; status) statement. Read-only and already remembered, so don't copy it into "
    "your files. To make the memory system forget something you wrote, delete the file it came "
    "from.\n\n"
)
HISTORY_HEAD = (
    "# History\n"
    "Every recorded version of this fact, oldest first. Old versions are kept, not "
    'overwritten: "no longer true" means reality changed, "corrected" means the earlier '
    "version was wrong. Read-only.\n\n"
)
ROOT_NOTE = (
    "\n\nprofile.md, facts.md and history/ are kept up to date by the memory system from your "
    "memory files and past conversations, and are read-only. They and the memories recalled "
    "into the conversation are already remembered: don't copy them into your files. Every "
    "change to your own files is also saved to long-term memory, so write absolute dates "
    '("in 2026-09", not "last month"). A line you remove is recorded as no longer applying, and '
    "deleting a file makes the memory system forget what that file added."
)
REMOVED_NOTE = (
    "\n\nRemoving a line does not make the memory system forget it: it records that the line "
    "no longer applies (for example, that the user no longer lives there) and keeps its "
    "history. If the user asked you to forget something, delete this file and create it again "
    "without it: deleting a file makes the memory system forget what the file added."
)


class _Note(NamedTuple):
    """文件的一次改动写进 CueMem 原始日志的那条笔记。live=False：文件已经去掉，笔记还在。"""

    path: str
    session_id: str
    turn_id: str
    live: bool = True


class _State(NamedTuple):
    """一个记忆空间里 Claude 的文件与出处（每个命令开始时读一次）。"""

    files: dict[str, str]
    notes: list[_Note]  # 现有文件的笔记
    orphans: list[_Note]  # 已经去掉的文件留下的笔记
    turns: dict[str, Turn]  # 笔记在原始日志里的那一条


class _Conflict(Exception):
    """保存时发现文件已经被别的进程改过。"""


# ---------- 旁边的文件存储 ----------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    space_id TEXT NOT NULL,
    path TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (space_id, path)
);
CREATE TABLE IF NOT EXISTS notes (
    space_id TEXT NOT NULL,
    path TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    command TEXT NOT NULL,
    ts TEXT NOT NULL,
    live INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (space_id, turn_id)
);
CREATE INDEX IF NOT EXISTS notes_by_path ON notes (space_id, path);
CREATE TABLE IF NOT EXISTS spaces (
    space_id TEXT PRIMARY KEY,
    epoch TEXT NOT NULL
);
"""


class _FileStore:
    """Claude 写的文件原文，与每次改动写进原始日志的笔记（出处）。几个工具实例可以共用一个文件。

    里面是用户的个人信息：文件只给本人读写（-wal、-shm 随数据库文件的权限），删除时覆写
    （secure_delete），删完把 WAL 写回并截断，不在磁盘上留下被删的原文。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():  # 先以 0600 建好空文件，SQLite 建 -wal、-shm 时沿用这个权限
            with contextlib.suppress(FileExistsError):
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        for extra in (Path(f"{path}-wal"), Path(f"{path}-shm")):
            try:
                if extra.exists() and extra.stat().st_mode & 0o077:
                    os.chmod(extra, 0o600)
            except OSError:  # 别人的文件：改不了就算了
                pass
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA secure_delete=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(notes)")}
        if "live" not in columns:  # 早期版本的存储
            self._conn.execute("ALTER TABLE notes ADD COLUMN live INTEGER NOT NULL DEFAULT 1")
            self._conn.commit()

    def files(self, space: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, content FROM files WHERE space_id = ? ORDER BY path", (space,)
            ).fetchall()
        return {path: content for path, content in rows}

    def notes(self, space: str) -> list[_Note]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, session_id, turn_id, live FROM notes WHERE space_id = ? "
                "ORDER BY ts, turn_id",
                (space,),
            ).fetchall()
        return [_Note(path, session, turn, bool(live)) for path, session, turn, live in rows]

    def save(
        self,
        space: str,
        path: str,
        content: str,
        note: _Note | None,
        command: str,
        expected: str | None,
    ) -> None:
        """存原文与出处。expected 是读到的原文（None：读的时候还没有这个文件）；文件在此期间
        被别的进程改过时抛出 _Conflict，什么都不写。"""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._conn:
            if expected is None:
                cursor = self._conn.execute(
                    "INSERT INTO files (space_id, path, content, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT (space_id, path) DO NOTHING",
                    (space, path, content, now, now),
                )
            else:
                cursor = self._conn.execute(
                    "UPDATE files SET content = ?, updated_at = ? "
                    "WHERE space_id = ? AND path = ? AND content = ?",
                    (content, now, space, path, expected),
                )
            if cursor.rowcount != 1:
                raise _Conflict(path)
            if note is not None:
                self._conn.execute(
                    "INSERT INTO notes (space_id, path, session_id, turn_id, command, ts, live) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1)",
                    (space, path, note.session_id, note.turn_id, command, now),
                )

    def prune(self, space: str, paths: Sequence[str], dead_turns: Sequence[str]) -> None:
        """笔记在别处被遗忘了：去掉这些文件（它们其余的笔记留作记录），删掉死了的笔记。"""
        with self._lock, self._conn:
            for path in paths:
                self._conn.execute(
                    "DELETE FROM files WHERE space_id = ? AND path = ?", (space, path)
                )
                self._conn.execute(
                    "UPDATE notes SET live = 0 WHERE space_id = ? AND path = ? AND live = 1",
                    (space, path),
                )
            for turn in dead_turns:
                self._conn.execute(
                    "DELETE FROM notes WHERE space_id = ? AND turn_id = ?", (space, turn)
                )
        self._checkpoint()

    def remove(self, space: str, paths: Sequence[str]) -> None:
        """删除这些路径的文件与全部笔记（包括已经去掉的文件留下的）。"""
        with self._lock, self._conn:
            for path in paths:
                self._conn.execute(
                    "DELETE FROM files WHERE space_id = ? AND path = ?", (space, path)
                )
                self._conn.execute(
                    "DELETE FROM notes WHERE space_id = ? AND path = ?", (space, path)
                )
        self._checkpoint()

    def purge(self, space: str) -> int:
        """删掉这个记忆空间的一切，返回删掉的文件数。"""
        with self._lock, self._conn:
            removed = self._conn.execute("DELETE FROM files WHERE space_id = ?", (space,)).rowcount
            self._conn.execute("DELETE FROM notes WHERE space_id = ?", (space,))
            self._conn.execute("DELETE FROM spaces WHERE space_id = ?", (space,))
        self._checkpoint()
        return int(removed)

    def move(self, space: str, moves: Mapping[str, str]) -> None:
        """改名：文件与它的笔记一起搬（目标路径事先确认过都不存在）；已经去掉的文件的笔记不动。"""
        with self._lock, self._conn:
            for old, new in moves.items():
                self._conn.execute(
                    "UPDATE files SET path = ? WHERE space_id = ? AND path = ?", (new, space, old)
                )
                self._conn.execute(
                    "UPDATE notes SET path = ? WHERE space_id = ? AND path = ? AND live = 1",
                    (new, space, old),
                )
        self._checkpoint()

    def epoch(self, space: str) -> str | None:
        """第一次写入时 CueMem 里这个记忆空间的建立时间：之后对不上就是空间被删过。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT epoch FROM spaces WHERE space_id = ?", (space,)
            ).fetchone()
        return None if row is None else str(row[0])

    def set_epoch(self, space: str, epoch: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO spaces (space_id, epoch) VALUES (?, ?)", (space, epoch)
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _checkpoint(self) -> None:
        """把 WAL 写回数据库并截断：被删的原文不留在 WAL 里。"""
        with self._lock:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


class _SpaceLock:
    """同一个存储文件、同一个记忆空间的读改写串行（同一进程里的所有工具实例共用一把）。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()


_SPACE_LOCKS: weakref.WeakValueDictionary[tuple[str, str], _SpaceLock] = (
    weakref.WeakValueDictionary()
)
_SPACE_LOCKS_GUARD = threading.Lock()


def _space_lock(store: Path, space: str) -> _SpaceLock:
    key = (str(store.resolve()), space)
    with _SPACE_LOCKS_GUARD:
        lock = _SPACE_LOCKS.get(key)
        if lock is None:
            lock = _SPACE_LOCKS[key] = _SpaceLock()
        return lock


# ---------- 记忆工具 ----------


class CueMemMemoryTool(BetaAbstractMemoryTool):
    """Claude 记忆工具的后端：/memories 里是 CueMem 生成的只读视图与 Claude 自己的文件。

    memory：Memory 或 AsyncMemory；user_id、agent_id 定记忆空间（与 SDK 相同；不能含 "/"）。
    store：文件存储的 SQLite 路径（默认数据目录下的 claude_memory.sqlite）。
    wait=True：每次改动同步抽取成事实（测试、脚本用）；默认交给后台线程（Memory 默认开着）。
    线程安全；交给 client.beta.messages.tool_runner(tools=[...])，或直接调用各命令方法。
    """

    def __init__(
        self,
        memory: Memory | AsyncMemory,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        store: str | Path | None = None,
        wait: bool = False,
        max_file_chars: int = MAX_FILE_CHARS,
        max_files: int = MAX_FILES,
        max_total_chars: int = MAX_TOTAL_CHARS,
        cache_control: BetaCacheControlEphemeralParam | None = None,
    ) -> None:
        super().__init__(cache_control=cache_control)
        self.memory = _sync(memory)
        self.user_id = user_id
        self.agent_id = agent_id
        self.space = _space(user_id, agent_id)
        self.wait = wait
        self.max_file_chars = max_file_chars
        self.max_files = max_files
        self.max_total_chars = max_total_chars
        path = _store_path(self.memory, store)
        self.store = _FileStore(path)
        self._space_lock = _space_lock(path, self.space)

    # ---------- 命令 ----------

    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        path = _normalize(command.path)
        with self._space_lock.lock:
            state = self._own()
            files = state.files
            if path in files:
                return _show(path, files[path], command.view_range)
            if _is_history_file(path):
                return _show(path, self._history(path), command.view_range)
            generated = self._generated(state)
            if path in generated:
                return _show(path, generated[path], command.view_range)
            tree = {**generated, **files}
            if path not in (ROOT, HISTORY) and not _has_children(tree, path):
                raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
            listing = _listing(tree, path)
            return listing + ROOT_NOTE if path == ROOT else listing

    def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        path = _normalize(command.path)
        text = command.file_text
        if not isinstance(text, str):
            raise ToolError("Error: file_text must be a string.")
        with self._space_lock.lock:
            state = self._own()
            files = state.files
            self._writable(path, files)
            existed = path in files
            if not existed and len(files) >= self.max_files:
                raise ToolError(
                    f"Error: There are already {len(files)} memory files, the limit. Delete or "
                    "merge files you no longer need first."
                )
            self._check_size(path, text, files)
            if existed and files[path] == text:
                return f"File created successfully at: {path}"
            removed = False
            if existed:  # 覆盖：只写改了的行，没变的内容不再进日志（否则会被重复抽取）
                note, removed = _change_note(
                    path, files[path], text, "rewritten", self._written(state, path)
                )
            else:
                note = f"Memory note {path}:\n{text}" if text.strip() else None
            self._save(path, text, note, "create", files.get(path))
            if existed:
                return f"File overwritten successfully at: {path}" + (
                    REMOVED_NOTE if removed else ""
                )
            return f"File created successfully at: {path}"

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        path = _normalize(command.path)
        old = command.old_str
        new = getattr(command, "new_str", None) or ""  # new_str 可以省略：删掉 old_str
        if not isinstance(old, str) or not isinstance(new, str):
            raise ToolError("Error: old_str and new_str must be strings.")
        with self._space_lock.lock:
            state = self._own()
            files = state.files
            content = self._file(
                path, files, f"Error: The path {path} does not exist. Please provide a valid path."
            )
            if not old:
                raise ToolError("Error: old_str must not be empty.")
            count = content.count(old)
            if count == 0:
                raise ToolError(
                    f"No replacement was performed, old_str `{old}` did not appear verbatim in "
                    f"{path}."
                )
            if count > 1:
                lines = _occurrence_lines(content, old)
                raise ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str `{old}` in "
                    f"lines: {', '.join(map(str, lines))}. Please ensure it is unique"
                )
            pos = content.find(old)
            updated = content[:pos] + new + content[pos + len(old) :]
            self._check_size(path, updated, files)
            removed = False
            if updated != content:
                note, removed = _change_note(
                    path, content, updated, "edited", self._written(state, path)
                )
                self._save(path, updated, note, "str_replace", content)
            return (
                "The memory file has been edited. Here is the snippet showing the change (with "
                "line numbers):\n"
                + _snippet(updated, content[:pos].count("\n"))
                + (REMOVED_NOTE if removed else "")
            )

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        path = _normalize(command.path)
        line, text = command.insert_line, command.insert_text
        if not isinstance(text, str):
            raise ToolError("Error: insert_text must be a string.")
        with self._space_lock.lock:
            state = self._own()
            files = state.files
            content = self._file(path, files, f"Error: The path {path} does not exist")
            lines = _lines(content)
            if not isinstance(line, int) or isinstance(line, bool) or not 0 <= line <= len(lines):
                raise ToolError(
                    f"Error: Invalid `insert_line` parameter: {line}. It should be within the "
                    f"range of lines of the file: [0, {len(lines)}]"
                )
            added = text.rstrip("\n")
            updated = "\n".join([*lines[:line], added, *lines[line:]]) + "\n"
            self._check_size(path, updated, files)
            # 插入一个标题会改变它下面几行的含义：同样按改动前后的差别写笔记
            note, _ = _change_note(path, content, updated, "added", self._written(state, path))
            self._save(path, updated, note, "insert", content)
            return f"The file {path} has been edited."

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        path = _normalize(command.path)
        if path == ROOT:
            raise ToolError("Error: Cannot delete the /memories directory itself")
        with self._space_lock.lock:
            state = self._own()
            _not_generated(path)
            targets = [p for p in state.files if _inside(p, path)]
            # 已经去掉的文件留下的笔记：facts.md 注明出自这个路径，删除这个路径也遗忘它们
            removed = sorted(
                {n.path for n in state.orphans if _inside(n.path, path)} - set(targets)
            )
            if not targets and not removed:
                raise ToolError(f"Error: The path {path} does not exist")
            paths = targets + removed
            notes = [n for n in [*state.notes, *state.orphans] if n.path in paths]
            report = self._forget(notes, len(set(paths)))
            self.store.remove(self.space, paths)
            return f"Successfully deleted {path}\n{report}"

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        old, new = _normalize(command.old_path), _normalize(command.new_path)
        if old == ROOT:
            raise ToolError("Error: Cannot rename the /memories directory itself")
        with self._space_lock.lock:
            state = self._own()
            files = state.files
            _not_generated(old)
            _not_generated(new)
            if old in files:
                moves = {old: new}
            else:
                moves = {p: new + p[len(old) :] for p in files if p.startswith(old + "/")}
                if not moves:
                    raise ToolError(f"Error: The path {old} does not exist")
                if new.startswith(old + "/"):
                    raise ToolError(f"Error: Cannot move {old} into itself")
            tree = {**self._generated(state), **files}
            if new == ROOT or new in tree or _has_children(tree, new):
                raise ToolError(f"Error: The destination {new} already exists")
            for target in moves.values():
                _normalize(target)  # 搬过去之后也不能太深、太长
                _no_file_above(target, tree)
            self.store.move(self.space, moves)
            return f"Successfully renamed {old} to {new}"

    def clear_all_memory(self) -> str:
        """删除 Claude 写的全部文件，并遗忘它们（包括已经去掉的文件）写进日志的笔记。对话里
        直接写进 CueMem 的记忆不动；要删掉这个用户的全部记忆，用本模块的 delete_all。"""
        with self._space_lock.lock:
            state = self._own()
            notes = [*state.notes, *state.orphans]
            if not state.files and not notes:
                return "All memory cleared"
            report = self._forget(notes, len(set(state.files) | {n.path for n in notes}))
            self.store.purge(self.space)
            return f"All memory cleared\n{report}"

    # ---------- 其他 ----------

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> CueMemMemoryTool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # ---------- 内部 ----------

    def _own(self) -> _State:
        """Claude 的文件与笔记。记忆空间被删过（Memory.delete_all）就全部清掉；笔记在别处被遗忘
        了的文件先去掉（见模块说明）。"""
        db = self.memory.system.db
        recorded = self.store.epoch(self.space)
        if recorded is not None and recorded != _core_epoch(db, self.space):
            removed = self.store.purge(self.space)
            logger.info("Removed %d memory file(s) of a deleted memory space", removed)
            record_event("claude_memory_purged", space=self.space, files=removed)
        files = self.store.files(self.space)
        notes = self.store.notes(self.space)
        ids = [n.turn_id for n in notes]
        turns: dict[str, Turn] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            turns.update({t.id: t for t in db.get_turns(self.space, ids=chunk)})
        dead = [n for n in notes if n.turn_id not in turns]
        if dead:
            tainted = sorted({n.path for n in dead if n.live})
            self.store.prune(self.space, tainted, [n.turn_id for n in dead])
            if tainted:
                logger.info(
                    "Removed %d memory file(s) whose notes were forgotten elsewhere", len(tainted)
                )
                record_event("claude_memory_pruned", space=self.space, files=len(tainted))
            files = {p: c for p, c in files.items() if p not in tainted}
            notes = [
                n._replace(live=n.live and n.path not in tainted)
                for n in notes
                if n.turn_id in turns
            ]
        return _State(files, [n for n in notes if n.live], [n for n in notes if not n.live], turns)

    def _generated(self, state: _State) -> dict[str, str]:
        """生成的只读文件：profile.md、facts.md，以及 history/ 里有过变化的事实。"""
        facts = self.memory.system.facts(self.space)
        chains = _chains(facts)
        source = {n.turn_id: n.path for n in state.notes}
        source.update({n.turn_id: f"{n.path} (removed)" for n in state.orphans})
        out = {PROFILE: self._profile(), FACTS: _facts_file(facts, chains, source)}
        changed = {id(c): c for c in chains.values() if len(c) > 1}.values()
        latest = sorted(changed, key=lambda c: (c[-1].recorded_at, c[-1].id), reverse=True)
        for chain in latest[:MAX_HISTORY_LISTED]:
            out[f"{HISTORY}/{chain[-1].id}.md"] = _history_file(chain)
        return out

    def _profile(self) -> str:
        core = self.memory.system.core_memory(self.space)
        text = core.text if core is not None and core.text else ""
        return PROFILE_HEAD + (text or "Nothing is known about the user yet.") + "\n"

    def _history(self, path: str) -> str:
        """history/<事实 ID>.md：这条事实所在的整条版本链（只认这个空间的事实）。"""
        fact_id = path[len(HISTORY) + 1 : -len(".md")]
        chains = _chains(self.memory.system.facts(self.space))
        if fact_id not in chains:
            raise ToolError(f"The path {path} does not exist. Please provide a valid path.")
        return _history_file(chains[fact_id])

    def _written(self, state: _State, path: str) -> dict[str, str]:
        """这个文件里每一行最近一次写下（不是被引用）的日期：笔记里引用旧行时注明，抽取据此
        理解旧行里的相对时间，也知道它是哪天说的。"""
        written: dict[str, str] = {}
        for note in state.notes:
            turn = state.turns.get(note.turn_id)
            if note.path != path or turn is None:
                continue
            for line in turn.text.split("\n")[1:]:  # 第一行是 "Memory note ..."
                text = line.strip()
                if text and not text.startswith(("(", "#")):
                    written[text] = day(turn.ts)
        return written

    def _file(self, path: str, files: Mapping[str, str], missing: str) -> str:
        """要修改的文件：生成的只读文件报只读；目录与不存在的路径报不存在（按文档的约定）。"""
        _not_generated(path)
        if path not in files:
            raise ToolError(missing)
        return files[path]

    def _writable(self, path: str, files: Mapping[str, str]) -> None:
        if path == ROOT or _has_children(files, path):
            raise ToolError(f"Error: {path} is a directory")
        _not_generated(path)
        _no_file_above(path, files)

    def _check_size(self, path: str, content: str, files: Mapping[str, str]) -> None:
        if len(content) > self.max_file_chars:
            raise ToolError(
                f"Error: {path} would have {len(content)} characters; a memory file can have at "
                f"most {self.max_file_chars}. Keep only what matters or split it into files."
            )
        total = sum(len(c) for p, c in files.items() if p != path) + len(content)
        if total > self.max_total_chars:
            raise ToolError(
                f"Error: Memory files would have {total} characters in all, over the limit of "
                f"{self.max_total_chars}. Delete or shorten files you no longer need first."
            )

    def _save(
        self,
        path: str,
        content: str,
        note_text: str | None,
        command: str,
        expected: str | None,
    ) -> None:
        """先把这次改动写进 CueMem 的原始日志（写不进去就整个失败，文件不动），再存原文与出处。
        存不进去（包括文件刚被别的进程改过）时撤销刚写的笔记，文件与记忆保持一致。
        note_text 为 None（只是空白之类，没有新信息）时只存原文。

        会话 ID 带上微秒：同一秒里的几次改动（一轮里并行的工具调用）按 ID 排序就是先后顺序，
        后台一起巩固时对账才不会把新版本当成旧的。"""
        self._ensure_epoch()
        note = None
        if note_text is not None and note_text.strip():
            now = datetime.now()
            session = f"memfile-{now:%Y%m%d-%H%M%S-%f}-{uuid.uuid4().hex[:4]}"
            try:
                out = self.memory.add(
                    note_text.strip(),
                    user_id=self.user_id,
                    agent_id=self.agent_id,
                    run_id=session,
                    timestamp=now.replace(microsecond=0),
                    wait=self.wait,
                )
            except Exception as exc:
                logger.warning("Could not save a memory note: %s", exc, exc_info=True)
                raise ToolError(
                    f"Error: Could not save the change to long-term memory ({type(exc).__name__}"
                    f"); {path} was not changed. Try again later."
                ) from exc
            note = _Note(path, session, out["turn_ids"][0]) if out["turn_ids"] else None
        try:
            self.store.save(self.space, path, content, note, command, expected)
        except Exception as exc:
            if note is not None:
                self._rollback(note)
            if isinstance(exc, _Conflict):
                raise ToolError(
                    f"Error: {path} was changed by another conversation at the same time; "
                    "nothing was saved. View it again and redo the change."
                ) from exc
            logger.warning("Could not save memory file %s: %s", path, exc, exc_info=True)
            raise ToolError(
                f"Error: Could not save {path} ({type(exc).__name__}); it was not changed. Try "
                "again later."
            ) from exc

    def _rollback(self, note: _Note) -> None:
        """撤销刚写进日志、文件却没存上的笔记（否则记忆里有、哪个文件都不认）。"""
        try:
            self._settle()
            self.memory.system.forget_memories(self.space, session_ids=[note.session_id])
        except Exception as exc:
            logger.error("Could not undo memory note %s of %s: %s", note.turn_id, note.path, exc)
            record_event("claude_memory_rollback_failed", space=self.space, turn=note.turn_id)

    def _ensure_epoch(self) -> None:
        if self.store.epoch(self.space) is None:
            db = self.memory.system.db
            db.ensure_space(self.space)
            self.store.set_epoch(self.space, _core_epoch(db, self.space) or "")

    def _settle(self) -> None:
        """后台还在抽取这个空间的笔记的话，先在这里做完（只等这个空间，不等别的用户）。
        做不完（模型出错）也照样往下遗忘：被删的会话排着的抽取任务随之作废。"""
        if self.memory.system.worker is None:
            return
        try:
            self.memory.consolidate(user_id=self.user_id, agent_id=self.agent_id)
        except Exception as exc:
            logger.warning("Could not finish pending extraction before forgetting: %s", exc)

    def _forget(self, notes: Sequence[_Note], files: int) -> str:
        """遗忘这些笔记（删日志、重建派生记忆），并说明遗忘了什么。"""
        sessions = sorted({n.session_id for n in notes})
        if not sessions:
            subject = "It" if files == 1 else "They"
            return f"{subject} had added nothing to long-term memory."
        self._settle()
        system = self.memory.system
        before = system.facts(self.space)
        current = {f.id for f in before if f.is_current}
        try:
            report = system.forget_memories(self.space, session_ids=sessions)
        except Exception as exc:
            logger.warning("Could not forget memory notes: %s", exc, exc_info=True)
            raise ToolError(
                f"Error: Could not forget what {notes[0].path} added to long-term memory "
                f"({type(exc).__name__}); the file was kept. Try again later."
            ) from exc
        after = system.facts(self.space, current_only=True)
        now_ids = {f.id for f in after}
        gone = [f for f in before if f.id in current and f.id not in now_ids]
        known = {f.id for f in before}
        back = [f for f in after if f.id not in current and f.id in known]  # 旧版本恢复为当前
        own = report.turns_deleted - report.cascaded
        what = "this file" if files == 1 else f"these {files} files"
        repeated = (
            f" and {report.cascaded} later message(s) that repeated them" if report.cascaded else ""
        )
        lines = [
            f"Forgotten from long-term memory: the {own} saved change(s) of {what}{repeated}; "
            "everything derived only from them was removed."
        ]
        if gone:
            lines += ["No longer remembered:", *_statements(gone)]
        if back:
            lines += ["Current again (the newer version came from the deleted notes):"]
            lines += _statements(back)
        if not gone and not back:
            lines.append("No remembered fact depended only on it.")
        return "\n".join(lines)


class AsyncCueMemMemoryTool(BetaAsyncAbstractMemoryTool):
    """异步版本（交给 AsyncAnthropic 的 tool_runner）：同一个后端，命令在工作线程里执行。"""

    def __init__(
        self,
        memory: Memory | AsyncMemory,
        *,
        cache_control: BetaCacheControlEphemeralParam | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(cache_control=cache_control)
        self.sync = CueMemMemoryTool(memory, cache_control=cache_control, **kwargs)

    async def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        return await to_thread.run_sync(self.sync.view, command)

    async def create(self, command: BetaMemoryTool20250818CreateCommand) -> str:
        return await to_thread.run_sync(self.sync.create, command)

    async def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        return await to_thread.run_sync(self.sync.str_replace, command)

    async def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        return await to_thread.run_sync(self.sync.insert, command)

    async def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        return await to_thread.run_sync(self.sync.delete, command)

    async def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        return await to_thread.run_sync(self.sync.rename, command)

    async def clear_all_memory(self) -> str:
        return await to_thread.run_sync(self.sync.clear_all_memory)

    def close(self) -> None:
        self.sync.close()


def delete_all(
    memory: Memory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    store: str | Path | None = None,
) -> dict[str, Any]:
    """Memory.delete_all 加上 Claude 的文件：这个用户（或 Agent）的全部记忆，以及记忆工具
    存的文件原文，当场删除（被遗忘权）。只调用 Memory.delete_all 的话，文件在下次访问时才清掉。
    异步代码里放进线程执行：await asyncio.to_thread(delete_all, memory.sync, user_id=...)。"""
    space = _space(user_id, agent_id)
    out = memory.delete_all(user_id=user_id, agent_id=agent_id)
    path = _store_path(memory, store)
    if not path.exists():  # 从没用过记忆工具：不为了删除建一个空的存储
        out["memory_files_deleted"] = 0
        return out
    files = _FileStore(path)
    try:
        with _space_lock(files.path, space).lock:
            out["memory_files_deleted"] = files.purge(space)
    finally:
        files.close()
    return out


# ---------- 调用模型之前：检索控制器 ----------


@dataclass(frozen=True)
class Evidence:
    """一次自动回忆的结果。"""

    # "L0" / "L1" / "L2"；"none"：没有检索（没有记忆，或最新的不是用户的话）；
    # "error"：检索出错（reasons 里是错误），这一轮不带记忆
    level: str
    reasons: list[str]
    recalled: int  # 注入的记忆条数（画像、事实、对话片段）
    block: str  # 注入的证据块；空：什么都没加


def with_recall(
    messages: Sequence[Mapping[str, Any]],
    memory: Memory | AsyncMemory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    mode: RecallMode = "auto",
) -> tuple[list[BetaMessageParam], Evidence]:
    """对最新一句用户消息先跑检索控制器，把证据块放在它的开头（返回新的列表，原列表不动）。

    与记忆代理同一个格式：L0 只给画像（没有画像就什么都不加），L1 / L2 再加带有效期的事实与
    对话片段。末尾的 mid-conversation system 消息跳过，证据放进它前面那句用户消息；最新的是
    工具结果（工具循环的中途）或助手的话时不检索。

    返回的列表就是之后的历史：下一轮在它后面追加用户消息再调用（只往后加）。不要每轮从
    不带证据的消息重新拼——那样会改动已经发出去的消息，Claude Opus 5.5 这样的模型会拒绝
    （preserved thinking），提示缓存也会失效。检索出错时给出 RuntimeWarning，原样返回消息，
    level 为 "error"。这个函数会等网络请求：异步代码里用 awith_recall。
    """
    space = _space(user_id, agent_id)
    out = [dict(m) for m in messages]
    at = _newest_user(out)
    if at is None:
        return cast("list[BetaMessageParam]", out), Evidence("none", [], 0, "")
    try:
        evidence = _recall(out[at], _sync(memory), space, mode)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.debug("Recall failed", exc_info=True)
        record_event("claude_recall_failed", space=space, error=error)
        warnings.warn(
            f"CueMem could not recall memories; continuing without them: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return cast("list[BetaMessageParam]", out), Evidence("error", [error], 0, "")
    if evidence.block:
        out = [*_inject(out[: at + 1], evidence.block), *out[at + 1 :]]
    return cast("list[BetaMessageParam]", out), evidence


async def awith_recall(
    messages: Sequence[Mapping[str, Any]],
    memory: Memory | AsyncMemory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    mode: RecallMode = "auto",
) -> tuple[list[BetaMessageParam], Evidence]:
    """with_recall 的异步版本：检索（向量、路由与 L2 的模型调用）在工作线程里跑，不阻塞事件循环。"""
    call = functools.partial(
        with_recall, messages, memory, user_id=user_id, agent_id=agent_id, mode=mode
    )
    return await to_thread.run_sync(call)


def _newest_user(messages: Sequence[Mapping[str, Any]]) -> int | None:
    """最新一句用户消息的位置（末尾的 system 消息跳过）；最新的不是用户的话时返回 None。"""
    for at in range(len(messages) - 1, -1, -1):
        role = messages[at].get("role")
        if role == "system":
            continue
        return at if role == "user" else None
    return None


def _recall(message: Mapping[str, Any], memory: Memory, space: str, mode: RecallMode) -> Evidence:
    """用户消息 → 控制器选档、取回记忆、排成证据块（记忆代理的 _recall 去掉会话部分）。"""
    none = Evidence("none", [], 0, "")
    content = message.get("content")
    blocks = content if isinstance(content, list) else []
    if any(_block_get(b, "type") == "tool_result" for b in blocks):
        return none
    query = (
        content
        if isinstance(content, str)
        else " ".join(
            str(_block_get(b, "text") or "") for b in blocks if _block_get(b, "type") == "text"
        )
    )
    system = memory.system
    if not query.strip() or not system.db.get_sessions(space):
        return none
    now = datetime.now()
    recall = system.retrieve(space, query, question_time=now, mode=mode)
    view = recall.view
    if recall.level == "l0":  # 不需要回忆：只给画像
        view = replace(view, rounds=[], facts=[], attach=[])
    parts, refs, _ = memory_blocks(view, now, relative_time=True)
    if not view.rounds:  # 没取到对话片段时不留一个空的"(none)"块
        parts = parts[:-1]
    level = recall.level.upper()
    reasons = [english_reason(r) for r in recall.decision.reasons] if recall.decision else []
    if not parts:
        return Evidence(level, reasons, 0, "")
    has_profile = view.core is not None and bool(view.core.text)
    record_event("claude_recall", space=space, level=level, recalled=len(refs))
    return Evidence(
        level, reasons, len(refs) + int(has_profile), "\n\n".join([MEMORY_NOTE, *parts])
    )


# ---------- 路径 ----------


def _normalize(path: object) -> str:
    """校验并规范化路径：只认 /memories 下的；拒绝一切可能跳出目录或含义含糊的写法。"""
    if not isinstance(path, str) or not path:
        raise ToolError("Error: A path under /memories is required.")
    path = unicodedata.normalize("NFC", path)
    if path != ROOT and not path.startswith(ROOT + "/"):
        raise ToolError(f"Error: Path must start with /memories, got: {path}")
    if "\\" in path or any(ord(c) < 32 or ord(c) == 127 for c in path):
        raise ToolError(f"Error: Invalid character in path {path!r}")
    if _PERCENT.search(path):
        raise ToolError(f"Error: Percent-encoded paths are not allowed: {path}")
    if len(path) > MAX_PATH_CHARS:
        raise ToolError(f"Error: Path is longer than {MAX_PATH_CHARS} characters.")
    rest = path[len(ROOT) :].rstrip("/")  # 目录可以带结尾的 /
    names = rest.split("/")[1:] if rest else []
    for name in names:
        if not name:
            raise ToolError(f"Error: Empty path segment in {path}")
        if name in (".", "..") or name.startswith("."):
            raise ToolError(
                f"Error: {path} is not allowed: no '.', '..' or hidden names under /memories"
            )
        if name != name.strip():
            raise ToolError(f"Error: Path segments cannot start or end with spaces: {path}")
        if len(name) > MAX_NAME_CHARS:
            raise ToolError(f"Error: A file or directory name is longer than {MAX_NAME_CHARS}.")
    if len(names) > MAX_DEPTH:
        raise ToolError(f"Error: {path} is nested more than {MAX_DEPTH} levels deep.")
    return "/".join([ROOT, *names])


def _generated_path(path: str) -> bool:
    return path in (PROFILE, FACTS, HISTORY) or path.startswith(HISTORY + "/")


def _is_history_file(path: str) -> bool:
    return path.startswith(HISTORY + "/") and path.endswith(".md") and path.count("/") == 3


def _not_generated(path: str) -> None:
    if _generated_path(path):
        raise ToolError(
            f"Error: {path} is generated by the memory system and is read-only. Write your notes "
            "in another file under /memories; to make the memory system forget something you "
            "wrote, delete the file it is in."
        )


def _has_children(tree: Mapping[str, str], path: str) -> bool:
    return any(p.startswith(path + "/") for p in tree)


def _inside(path: str, top: str) -> bool:
    """path 就是 top，或者在目录 top 里面。"""
    return path == top or path.startswith(top + "/")


def _no_file_above(path: str, tree: Mapping[str, str]) -> None:
    """路径上方不能有同名的文件（一个路径不能既是文件又是目录）。"""
    parts = path.split("/")
    for i in range(3, len(parts)):
        above = "/".join(parts[:i])
        if above in tree or above in (PROFILE, FACTS):
            raise ToolError(f"Error: {above} is a file, not a directory")


# ---------- 显示 ----------


def _listing(tree: Mapping[str, str], path: str) -> str:
    """目录列表，与参考实现相同：两层以内，大小与路径之间用制表符，目录以 / 结尾。"""
    sizes: dict[tuple[str, ...], int] = {}
    dirs: set[tuple[str, ...]] = set()
    total = 0
    for name, content in tree.items():
        if not name.startswith(path + "/"):
            continue
        size = len(content.encode("utf-8"))
        total += size
        parts = tuple(name[len(path) + 1 :].split("/"))
        for depth in range(1, min(len(parts), 3)):  # 上面两层的目录
            dirs.add(parts[:depth])
            sizes[parts[:depth]] = sizes.get(parts[:depth], 0) + size
        if len(parts) <= 2:
            sizes[parts] = size
    if path == ROOT:  # history/ 总是列出来，没有变化过的事实时是空目录
        dirs.add(("history",))
        sizes.setdefault(("history",), 0)
    lines = [
        "Here're the files and directories up to 2 levels deep in "
        f"{path}, excluding hidden items and node_modules:",
        f"{_size(total)}\t{path}",
    ]
    for key in sorted(sizes):
        slash = "/" if key in dirs else ""
        lines.append(f"{_size(sizes[key])}\t{path}/{'/'.join(key)}{slash}")
    return "\n".join(lines)


def _show(path: str, content: str, view_range: object) -> str:
    """带行号的文件内容（行号 6 位右对齐、制表符分隔）；超过 MAX_VIEW_CHARS 时截断。"""
    lines = _lines(content) or [""]
    if len(lines) > MAX_LINES:
        raise ToolError(f"File {path} exceeds maximum line limit of 999,999 lines.")
    start, end = 1, len(lines)
    if view_range is not None:
        if (
            not isinstance(view_range, list)
            or len(view_range) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in view_range)
        ):
            raise ToolError("Error: view_range must be two line numbers: [start, end]")
        start, end = view_range
        if not 1 <= start <= len(lines):
            raise ToolError(
                f"Error: Invalid `view_range`: {view_range}. Its first element should be within "
                f"the range of lines of the file: [1, {len(lines)}]"
            )
        end = len(lines) if end == -1 else min(end, len(lines))
        if end < start:
            raise ToolError(
                f"Error: Invalid `view_range`: {view_range}. The second element should be -1 or "
                "not smaller than the first."
            )
    shown: list[str] = []
    used = 0
    last = start - 1
    for number in range(start, end + 1):
        line = f"{number:>{LINE_WIDTH}}\t{lines[number - 1]}"
        if shown and used + len(line) > MAX_VIEW_CHARS:
            break
        shown.append(line)
        used += len(line) + 1
        last = number
    text = f"Here's the content of {path} with line numbers:\n" + "\n".join(shown)
    if last < end:
        text += (
            f"\n[Truncated: showing lines {start}-{last} of {len(lines)}. View the rest with "
            f"view_range [{last + 1}, -1].]"
        )
    return text


def _snippet(content: str, line_index: int) -> str:
    """改动处前后各两行（与参考实现相同）。"""
    lines = content.split("\n")
    first, stop = max(0, line_index - 2), min(len(lines), line_index + 3)
    return "\n".join(f"{n + 1:>{LINE_WIDTH}}\t{lines[n]}" for n in range(first, stop))


def _occurrence_lines(content: str, needle: str) -> list[int]:
    found, start = [], 0
    while (pos := content.find(needle, start)) != -1:
        found.append(content[:pos].count("\n") + 1)
        start = pos + 1
    return found


def _lines(content: str) -> list[str]:
    """按行切开，结尾的换行不算一行空行。"""
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _size(n: int) -> str:
    """人能读的大小：5.5K、1.2M（与参考实现相同）。"""
    if n == 0:
        return "0B"
    unit = min((n.bit_length() - 1) // 10, 3)
    value = n / 1024**unit
    shown = f"{int(value)}" if value == int(value) else f"{value:.1f}"
    return shown + "BKMG"[unit]


# ---------- 笔记 ----------


class _Item(NamedTuple):
    """文件里的一行内容（标题本身不算），连同它上方的各级标题：标题决定这一行的含义。"""

    headings: tuple[str, ...]  # 从外到内
    line: str  # 原样（去掉行尾空白）
    text: str  # 去掉首尾空白，用来比较

    @property
    def key(self) -> tuple[tuple[str, ...], str]:
        return self.headings, self.text


def _items(content: str) -> list[_Item]:
    items: list[_Item] = []
    stack: list[tuple[int, str]] = []  # (级别, 标题)
    for line in _lines(content):
        text = line.strip()
        if not text:
            continue
        if line.startswith("#"):
            level = len(text) - len(text.lstrip("#"))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
            continue
        items.append(_Item(tuple(h for _, h in stack), line.rstrip(), text))
    return items


def _change_note(
    path: str, before: str, after: str, verb: str, written: Mapping[str, str]
) -> tuple[str | None, bool]:
    """改动写进日志的笔记，以及有没有删掉旧行（不是换掉）。

    只写改了的行（连同上方的标题）而不是整个文件：没变的内容一遍遍进日志，会被重复抽取（同一条
    事实取代自己），检索时也会挤掉别的对话片段。每一行连同它所在的标题比较：在同一节里挪位置不算
    改动；挪到别的标题下是改动（"# 待办" 下的一项挪到 "# 已完成"），注明从哪里挪来。换掉的旧行
    注明被取代，删掉的旧行注明不再适用，都带上它最初写下的日期（written：行 → 日期）。没有新
    信息（只改了空白或标题下没有内容）时返回 None。
    """
    old, new = _items(before), _items(after)
    hunks = [
        (range(i1, i2), range(j1, j2))
        for tag, i1, i2, j1, j2 in SequenceMatcher(
            None, [x.key for x in old], [x.key for x in new], autojunk=False
        ).get_opcodes()
        if tag != "equal"
    ]
    removed = [i for was, _ in hunks for i in was]
    added = [j for _, now in hunks for j in now]
    # 同一节里只是挪了位置：两边都不算
    same: dict[tuple[tuple[str, ...], str], list[int]] = defaultdict(list)
    for i in removed:
        same[old[i].key].append(i)
    gone: set[int] = set()
    kept: set[int] = set()
    for j in added:
        if same[new[j].key]:
            gone.add(same[new[j].key].pop(0))
            kept.add(j)
    # 挪到了别的标题下：记在新位置，注明原来在哪
    elsewhere: dict[str, list[int]] = defaultdict(list)
    for i in removed:
        if i not in gone:
            elsewhere[old[i].text].append(i)
    came_from: dict[int, int] = {}
    for j in added:
        if j not in kept and elsewhere[new[j].text]:
            came_from[j] = elsewhere[new[j].text].pop(0)
            gone.add(came_from[j])
    parts: list[str] = []
    shown: tuple[str, ...] = ()  # 已经写出的标题
    removed_any = False
    for was, now in hunks:
        printed = False
        for j in now:
            if j in kept:
                continue
            item = new[j]
            if item.headings != shown:
                common = 0
                while (
                    common < min(len(shown), len(item.headings))
                    and shown[common] == item.headings[common]
                ):
                    common += 1
                parts += item.headings[common:]
                shown = item.headings
            line = item.line
            if j in came_from:
                origin = old[came_from[j]].headings
                line += f' (moved here from "{origin[-1]}")' if origin else " (moved here)"
            parts.append(line)
            printed = True
        for i in was:
            if i in gone:
                continue
            removed_any = removed_any or not printed
            parts.append(_quote(old[i], written, shown, replaced=printed))
    if not parts:
        return None, removed_any
    return f"Memory note {path} ({verb}):\n" + "\n".join(parts), removed_any


def _quote(item: _Item, written: Mapping[str, str], shown: tuple[str, ...], replaced: bool) -> str:
    """换掉或删掉的一行旧内容：注明它原来在哪个标题下、哪天写的。"""
    where = f' under "{item.headings[-1]}"' if item.headings and item.headings != shown else ""
    when = f", written on {written[item.text]}" if item.text in written else ""
    if replaced:
        return f'(This replaces "{item.text}"{where}{when}.)'
    return f'(Removed "{item.text}"{where}{when}: it no longer applies.)'


# ---------- 生成的文件 ----------


def _chains(facts: Sequence[Fact]) -> dict[str, list[Fact]]:
    """事实 ID → 它所在的版本链，旧版本在前（同一条链的成员共用一个列表）。

    先后按取代关系排：被取代的在前，不只看记录时间——同一秒里写下又改掉的两版记录时间相同。
    """
    parent = {f.id: f.id for f in facts}
    successor = {f.id: f.superseded_by for f in facts if f.superseded_by in parent}

    def root(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for old, new in successor.items():
        parent[root(old)] = root(new or "")
    rank: dict[str, int] = {}  # 取代链上前面有几个版本（取最长的一条）
    for start in set(parent) - {s for s in successor.values() if s}:
        node: str | None = start
        for step in range(len(parent)):  # 步数有上限：数据坏了出现环也不会死循环
            if node is None:
                break
            rank[node] = max(rank.get(node, 0), step)
            node = successor.get(node)
    groups: dict[str, list[Fact]] = {}
    for fact in facts:
        groups.setdefault(root(fact.id), []).append(fact)
    for group in groups.values():
        group.sort(key=lambda f: (rank.get(f.id, 0), f.recorded_at, f.id))
    return {f.id: groups[root(f.id)] for f in facts}


def _fact_line(fact: Fact, refs: Mapping[str, str]) -> str:
    """一条事实，写法与 reader 的事实块相同（read/facts.py 的 format_facts），只是用真实 ID。"""
    notes = [f"said on {day(fact.recorded_at)}"]
    if fact.valid_from and fact.valid_from != day(fact.recorded_at):
        notes.append(("happened " if fact.category == "event" else "since ") + fact.valid_from)
    notes.append(fact_status(fact, dict(refs)))
    return f"[{fact.id}] ({'; '.join(notes)}) {fact.statement}"


def _facts_file(
    facts: Sequence[Fact], chains: Mapping[str, list[Fact]], source: Mapping[str, str]
) -> str:
    current = sorted(
        (f for f in facts if f.is_current), key=lambda f: (f.subject, f.recorded_at, f.id)
    )
    lines = []
    for fact in current:
        chain = chains[fact.id]
        extra = []
        files = sorted({source[t] for t in fact.source_turn_ids if t in source})
        if files:
            extra.append("from " + ", ".join(files))
        if len(chain) > 1:
            extra.append(f"earlier versions: {HISTORY}/{chain[-1].id}.md")
        suffix = f" ({'; '.join(extra)})" if extra else ""
        lines.append(_fact_line(fact, {}) + suffix)
    return FACTS_HEAD + ("\n".join(lines) if lines else "No facts yet.") + "\n"


def _history_file(chain: Sequence[Fact]) -> str:
    refs = {f.id: f.id for f in chain}
    lines = [f"{n}. {_fact_line(f, refs)}" for n, f in enumerate(chain, start=1)]
    return HISTORY_HEAD + "\n".join(lines) + "\n"


def _statements(facts: Sequence[Fact]) -> list[str]:
    shown = [f"- {f.statement}" for f in facts[:MAX_LISTED_FACTS]]
    if len(facts) > MAX_LISTED_FACTS:
        shown.append(f"- … and {len(facts) - MAX_LISTED_FACTS} more")
    return shown


# ---------- 其他 ----------


def _sync(memory: Memory | AsyncMemory) -> Memory:
    return memory.sync if isinstance(memory, AsyncMemory) else memory


def _space(user_id: str | None, agent_id: str | None) -> str:
    """记忆空间 ID（与 SDK 的 space_for 相同）。ID 里不许有 "/"，也不许 Agent 叫 "agent"：
    space_for 用 "/" 拼接，否则 user_id="a/b" 与 user_id="b"、agent_id="a" 会落进同一个空间，
    彼此看到对方的文件与记忆。"""
    for name, value in (("user_id", user_id), ("agent_id", agent_id)):
        if value is not None and "/" in value:
            raise ValueError(f"{name} must not contain '/' (got {value!r}).")
    if user_id and agent_id == "agent":
        raise ValueError('agent_id "agent" is reserved when a user_id is given.')
    return space_for(user_id, agent_id)


def _store_path(memory: Memory, store: str | Path | None) -> Path:
    return Path(store) if store is not None else memory.settings.paths.db.parent / STORE_NAME


def _core_epoch(db: MemoryDB, space: str) -> str | None:
    """CueMem 里这个记忆空间的建立时间；空间不存在（被删除了）时为 None。"""
    with db.transaction() as conn:
        row = conn.execute("SELECT created_at FROM spaces WHERE id = ?", (space,)).fetchone()
    return None if row is None else str(row[0])


def _block_get(block: Any, key: str) -> Any:
    """内容块可以是字典，也可以是 SDK 的对象（把上一轮的 response.content 原样放回时）。"""
    return block.get(key) if isinstance(block, Mapping) else getattr(block, key, None)
