"""Claude 的记忆工具（memory_20250818）接到 CueMem 上（N12）：/memories 目录的后端。

    import anthropic
    from cuemem import Memory
    from cuemem.integrations.claude import CueMemMemoryTool

    memory = Memory(preset="deepseek")
    tool = CueMemMemoryTool(memory, user_id="alice")
    runner = anthropic.Anthropic().beta.messages.tool_runner(
        model="claude-opus-5-5", max_tokens=16000, tools=[tool],
        messages=[{"role": "user", "content": "I moved to Lisbon last month."}],
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
  remember 相同，后台抽取成事实）。新文件写全文；之后的改动只写改了的行和它们原来的样子（只是挪了
  位置的不算），对账据此把旧事实标成被取代，没变的内容也不会被重复抽取。rename 只搬动文件与出处，
  不写笔记（没有新信息）；delete 遗忘这个文件写进日志的全部笔记，由剩下的日志重建派生记忆，返回
  哪些事实不再记得、哪些旧版本恢复为当前。create 已有的文件时覆盖（Claude 的工具说明就是
  "创建或覆盖"；报错的话它会先删再建，白白触发一次遗忘）。

遗忘在两个方向上都成立（D-021）：文件的某条笔记在别处被遗忘（Memory.delete、MCP 的 forget、
转述追查）之后，这个文件也从 /memories 里去掉——不知道被遗忘的信息在文件的哪几行，留着就泄露了；
文件里其余的信息仍在 CueMem 里（facts.md），Claude 需要时可以重写。

路径只认 /memories 下的；拒绝 . 与 ..、隐藏名、反斜杠、控制字符、百分号编码、过深过长的路径。
单个文件、文件数、总字符数都有上限；查看超过 16000 字符时截断，提示用 view_range 往下翻。

with_recall（可选）：调用模型之前先过检索控制器（L0 只给画像），证据块与记忆代理
（api/proxy.py）同一个格式、同一个位置，放在最后一句用户消息的开头。
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import unicodedata
import uuid
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple, cast

from anthropic.lib.tools import BetaAbstractMemoryTool, BetaAsyncAbstractMemoryTool, ToolError
from anthropic.types.beta import (
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
from cuemem.core.models import Fact
from cuemem.core.timeutil import day
from cuemem.core.trace import record_event
from cuemem.memory import RecallMode
from cuemem.read.controller import english_reason
from cuemem.read.facts import fact_status
from cuemem.read.reader import memory_blocks
from cuemem.sdk import AsyncMemory, Memory, space_for

__all__ = [
    "AsyncCueMemMemoryTool",
    "CueMemMemoryTool",
    "Evidence",
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
    "to date by the memory system. Read-only.\n\n"
)
FACTS_HEAD = (
    "# Facts\n"
    "What the memory system currently knows, extracted from your memory files and past "
    "conversations and kept up to date automatically. Each line: [fact id] (when it was said; "
    "since when; status) statement. Read-only: to make the memory system forget something you "
    "wrote, delete the file it came from.\n\n"
)
HISTORY_HEAD = (
    "# History\n"
    "Every recorded version of this fact, oldest first. Old versions are kept, not "
    'overwritten: "no longer true" means reality changed, "corrected" means the earlier '
    "version was wrong. Read-only.\n\n"
)
ROOT_NOTE = (
    "\n\nprofile.md, facts.md and history/ are kept up to date by the memory system from your "
    "memory files and past conversations, and are read-only. Every change to your own files is "
    "also saved to long-term memory; deleting a file makes the memory system forget what that "
    "file added."
)


class _Note(NamedTuple):
    """文件的一次改动写进 CueMem 原始日志的那条笔记。"""

    path: str
    session_id: str
    turn_id: str


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
    PRIMARY KEY (space_id, turn_id)
);
CREATE INDEX IF NOT EXISTS notes_by_path ON notes (space_id, path);
"""


class _FileStore:
    """Claude 写的文件原文，与每次改动写进原始日志的笔记（出处）。几个工具实例可以共用一个文件。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        if fresh:  # 文件里是用户的个人信息：只给本人读写
            os.chmod(path, 0o600)

    def files(self, space: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, content FROM files WHERE space_id = ? ORDER BY path", (space,)
            ).fetchall()
        return {path: content for path, content in rows}

    def notes(self, space: str, paths: Collection[str] | None = None) -> list[_Note]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, session_id, turn_id FROM notes WHERE space_id = ? "
                "ORDER BY ts, turn_id",
                (space,),
            ).fetchall()
        wanted = set(paths) if paths is not None else None
        return [_Note(*row) for row in rows if wanted is None or row[0] in wanted]

    def save(self, space: str, path: str, content: str, note: _Note | None, command: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO files (space_id, path, content, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT (space_id, path) "
                "DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
                (space, path, content, now, now),
            )
            if note is not None:
                self._conn.execute(
                    "INSERT INTO notes (space_id, path, session_id, turn_id, command, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (space, path, note.session_id, note.turn_id, command, now),
                )

    def remove(self, space: str, paths: Sequence[str]) -> None:
        with self._lock, self._conn:
            for path in paths:
                self._conn.execute(
                    "DELETE FROM files WHERE space_id = ? AND path = ?", (space, path)
                )
                self._conn.execute(
                    "DELETE FROM notes WHERE space_id = ? AND path = ?", (space, path)
                )

    def move(self, space: str, moves: Mapping[str, str]) -> None:
        """改名：文件与它的笔记一起搬（目标路径事先确认过都不存在）。"""
        with self._lock, self._conn:
            for old, new in moves.items():
                for table in ("files", "notes"):
                    self._conn.execute(
                        f"UPDATE {table} SET path = ? WHERE space_id = ? AND path = ?",
                        (new, space, old),
                    )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------- 记忆工具 ----------


class CueMemMemoryTool(BetaAbstractMemoryTool):
    """Claude 记忆工具的后端：/memories 里是 CueMem 生成的只读视图与 Claude 自己的文件。

    memory：Memory 或 AsyncMemory；user_id、agent_id 定记忆空间（与 SDK 相同）。
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
        self.space = space_for(user_id, agent_id)
        self.wait = wait
        self.max_file_chars = max_file_chars
        self.max_files = max_files
        self.max_total_chars = max_total_chars
        default = self.memory.settings.paths.db.parent / STORE_NAME
        self.store = _FileStore(Path(store) if store is not None else default)
        self._lock = threading.RLock()

    # ---------- 命令 ----------

    def view(self, command: BetaMemoryTool20250818ViewCommand) -> str:
        path = _normalize(command.path)
        with self._lock:
            files, notes = self._own()
            if path in files:
                return _show(path, files[path], command.view_range)
            if _is_history_file(path):
                return _show(path, self._history(path), command.view_range)
            generated = self._generated(notes)
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
        with self._lock:
            files, _ = self._own()
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
            if existed:  # 覆盖：只写改了的行，没变的内容不再进日志（否则会被重复抽取）
                note = _change_note(path, files[path], text, "rewritten")
            else:
                note = f"Memory note {path}:\n{text}" if text.strip() else None
            self._save(path, text, note, "create")
            if existed:
                return f"File overwritten successfully at: {path}"
            return f"File created successfully at: {path}"

    def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> str:
        path = _normalize(command.path)
        old = command.old_str
        new = getattr(command, "new_str", None) or ""  # new_str 可以省略：删掉 old_str
        if not isinstance(old, str) or not isinstance(new, str):
            raise ToolError("Error: old_str and new_str must be strings.")
        with self._lock:
            files, _ = self._own()
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
            if updated != content:
                note = _change_note(path, content, updated, "edited")
                self._save(path, updated, note, "str_replace")
            return (
                "The memory file has been edited. Here is the snippet showing the change (with "
                "line numbers):\n" + _snippet(updated, content[:pos].count("\n"))
            )

    def insert(self, command: BetaMemoryTool20250818InsertCommand) -> str:
        path = _normalize(command.path)
        line, text = command.insert_line, command.insert_text
        if not isinstance(text, str):
            raise ToolError("Error: insert_text must be a string.")
        with self._lock:
            files, _ = self._own()
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
            self._save(path, updated, _change_note(path, content, updated, "added"), "insert")
            return f"The file {path} has been edited."

    def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> str:
        path = _normalize(command.path)
        if path == ROOT:
            raise ToolError("Error: Cannot delete the /memories directory itself")
        with self._lock:
            files, _ = self._own()
            _not_generated(path)
            targets = [path] if path in files else [p for p in files if p.startswith(path + "/")]
            if not targets:
                raise ToolError(f"Error: The path {path} does not exist")
            report = self._forget(targets)
            self.store.remove(self.space, targets)
            return f"Successfully deleted {path}\n{report}"

    def rename(self, command: BetaMemoryTool20250818RenameCommand) -> str:
        old, new = _normalize(command.old_path), _normalize(command.new_path)
        if old == ROOT:
            raise ToolError("Error: Cannot rename the /memories directory itself")
        with self._lock:
            files, notes = self._own()
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
            tree = {**self._generated(notes), **files}
            if new == ROOT or new in tree or _has_children(tree, new):
                raise ToolError(f"Error: The destination {new} already exists")
            for target in moves.values():
                _normalize(target)  # 搬过去之后也不能太深、太长
                _no_file_above(target, tree)
            self.store.move(self.space, moves)
            return f"Successfully renamed {old} to {new}"

    def clear_all_memory(self) -> str:
        """删除 Claude 写的全部文件，并遗忘它们写进日志的笔记（生成的视图之外的对话记忆不动；
        要删掉这个用户的全部记忆，用 Memory.delete_all）。"""
        with self._lock:
            files, _ = self._own()
            if not files:
                return "All memory cleared"
            report = self._forget(sorted(files))
            self.store.remove(self.space, sorted(files))
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

    def _own(self) -> tuple[dict[str, str], list[_Note]]:
        """Claude 的文件与笔记。笔记在别处被遗忘了的文件先去掉（见模块说明）。"""
        files = self.store.files(self.space)
        notes = self.store.notes(self.space)
        if not notes:
            return files, notes
        ids = [n.turn_id for n in notes]
        alive: set[str] = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            alive |= {t.id for t in self.memory.system.db.get_turns(self.space, ids=chunk)}
        tainted = sorted({n.path for n in notes if n.turn_id not in alive})
        if not tainted:
            return files, notes
        self.store.remove(self.space, tainted)
        logger.info("Removed %d memory file(s) whose notes were forgotten elsewhere", len(tainted))
        record_event("claude_memory_pruned", space=self.space, files=len(tainted))
        kept = {p: c for p, c in files.items() if p not in tainted}
        return kept, [n for n in notes if n.path not in tainted]

    def _generated(self, notes: Sequence[_Note]) -> dict[str, str]:
        """生成的只读文件：profile.md、facts.md，以及 history/ 里有过变化的事实。"""
        facts = self.memory.system.facts(self.space)
        chains = _chains(facts)
        source = {n.turn_id: n.path for n in notes}
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

    def _save(self, path: str, content: str, note_text: str | None, command: str) -> None:
        """先把这次改动写进 CueMem 的原始日志（写不进去就整个失败，文件不动），再存原文与出处。
        note_text 为 None（只是空白之类，没有新信息）时只存原文。

        会话 ID 带上微秒：同一秒里的几次改动（一轮里并行的工具调用）按 ID 排序就是先后顺序，
        后台一起巩固时对账才不会把新版本当成旧的。"""
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
        self.store.save(self.space, path, content, note, command)

    def _forget(self, paths: Sequence[str]) -> str:
        """遗忘这些文件写进日志的全部笔记（删日志、重建派生记忆），并说明遗忘了什么。"""
        notes = self.store.notes(self.space, paths)
        sessions = sorted({n.session_id for n in notes})
        if not sessions:
            return "It had added nothing to long-term memory."
        self.memory.wait()  # 后台还在抽取这些笔记的话，等它做完再删
        system = self.memory.system
        before = system.facts(self.space)
        current = {f.id for f in before if f.is_current}
        try:
            report = system.forget_memories(self.space, session_ids=sessions)
        except Exception as exc:
            logger.warning("Could not forget memory notes: %s", exc, exc_info=True)
            raise ToolError(
                f"Error: Could not forget what {paths[0]} added to long-term memory "
                f"({type(exc).__name__}); the file was kept. Try again later."
            ) from exc
        after = system.facts(self.space, current_only=True)
        now_ids = {f.id for f in after}
        gone = [f for f in before if f.id in current and f.id not in now_ids]
        known = {f.id for f in before}
        back = [f for f in after if f.id not in current and f.id in known]  # 旧版本恢复为当前
        own = report.turns_deleted - report.cascaded
        what = "this file" if len(paths) == 1 else f"these {len(paths)} files"
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


# ---------- 调用模型之前：检索控制器 ----------


@dataclass(frozen=True)
class Evidence:
    """一次自动回忆的结果。"""

    level: str  # "L0" / "L1" / "L2"；"none"：没有检索（没有记忆，或最后一条不是用户的话）
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
    """对最后一句用户消息先跑检索控制器，把证据块放在它的开头（返回新的列表，原列表不动）。

    与记忆代理同一个格式：L0 只给画像（没有画像就什么都不加），L1 / L2 再加带有效期的事实与
    对话片段。最后一条是工具结果（工具循环的中途）时不检索。
    """
    out = [dict(m) for m in messages]
    evidence = _recall(out, _sync(memory), space_for(user_id, agent_id), mode)
    if evidence.block:
        out = _inject(out, evidence.block)
    return cast("list[BetaMessageParam]", out), evidence


def _recall(
    messages: list[dict[str, Any]], memory: Memory, space: str, mode: RecallMode
) -> Evidence:
    """最后一句用户消息 → 控制器选档、取回记忆、排成证据块（记忆代理的 _recall 去掉会话部分）。"""
    none = Evidence("none", [], 0, "")
    if not messages or messages[-1].get("role") != "user":
        return none
    content = messages[-1].get("content")
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


def _change_note(path: str, before: str, after: str, verb: str) -> str | None:
    """改动写进日志的笔记：改了的行（连同上方最近的标题），以及它们原来的样子。

    只写改动处而不是整个文件：没变的内容一遍遍进日志，会被重复抽取（同一条事实取代自己），
    检索时也会挤掉别的对话片段。只是挪了位置的行不算改动；没有新信息（只改了空白）时返回 None。
    """
    old, new = _lines(before), _lines(after)
    hunks = [
        (old[i1:i2], new[j1:j2], j1)
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
        if tag != "equal"
    ]
    removed = {x.strip() for was, _, _ in hunks for x in was}
    moved = removed & {x.strip() for _, now, _ in hunks for x in now}
    parts: list[str] = []
    shown = ""  # 已经写过的标题
    for was_lines, now_lines, at in hunks:
        was = [x for x in was_lines if x.strip() and x.strip() not in moved]
        now = [x for x in now_lines if x.strip() and x.strip() not in moved]
        quoted = "\n".join(was)
        if now:
            heading = _heading(new[:at])
            if heading and heading != shown and not now[0].startswith("#"):
                parts.append(heading)
                shown = heading
            parts += now
            if was:
                parts.append(f'(This replaces: "{quoted}")')
        elif was:
            parts.append(f'(Removed: "{quoted}")')
    return f"Memory note {path} ({verb}):\n" + "\n".join(parts) if parts else None


def _heading(lines: Sequence[str]) -> str:
    """上方最近的 Markdown 标题：给改动的那几行补上下文（"## 饮食" 下的 "- 不吃辣"）。"""
    return next((line for line in reversed(lines) if line.startswith("#")), "")


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


def _block_get(block: Any, key: str) -> Any:
    """内容块可以是字典，也可以是 SDK 的对象（把上一轮的 response.content 原样放回时）。"""
    return block.get(key) if isinstance(block, Mapping) else getattr(block, key, None)
