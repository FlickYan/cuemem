"""Claude 记忆工具（cuemem.integrations.claude）的文件存储：Claude 写的文件原文与每次改动的
笔记（出处），存在数据目录下的 claude_memory.sqlite（STORE_NAME）。

只用标准库的 sqlite3（不导入 anthropic 与 cuemem.sdk）：MemorySystem 删除空间与遗忘时也要清理
它（purge_space、prune_turns，第七轮体检），没装 cuemem[anthropic] 时同样要能删。"""

from __future__ import annotations

import contextlib
import os
import sqlite3
import threading
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

STORE_NAME = "claude_memory.sqlite"  # 放在数据目录（paths.db 所在目录）下


class _Note(NamedTuple):
    """文件的一次改动写进 CueMem 原始日志的那条笔记。live=False：文件已经去掉，笔记还在。"""

    path: str
    session_id: str
    turn_id: str
    live: bool = True


class _Conflict(Exception):
    """保存时发现文件已经被别的进程改过。"""


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


def purge_space(store: Path, space: str) -> int:
    """删掉存储里这个记忆空间的一切（文件、笔记与 spaces 里的那一行），返回删掉的文件数。
    MemorySystem.delete_space 调用（Memory.delete_all、reset、/api/v1 的 DELETE）。存储不存在
    时什么都不做，不为了删除建一个空的。

    不拿记忆工具的 _space_lock：删除空间在 MemorySystem 的独占里调用，工具却是先拿 _space_lock
    再调用记忆系统，反过来拿会互相等死。SQLite 的事务就够了：并发的工具保存要么核对原文不符
    （_Conflict，撤销它的笔记），要么存下一个笔记的发言已经不在的文件，下次 _own 把它去掉。"""
    if not store.exists():
        return 0
    files = _FileStore(store)
    try:
        return files.purge(space)
    finally:
        files.close()


def prune_turns(store: Path, space: str, turn_ids: Collection[str]) -> int:
    """这些发言被遗忘了（MemorySystem.forget 提交之后）：与记忆工具的 _own 对死了的笔记做的
    相同——笔记还在的文件去掉（不知道被遗忘的信息在文件的哪几行），死了的笔记删掉。返回去掉
    的文件数。存储不存在时什么都不做；不拿 _space_lock（见 purge_space）。"""
    if not turn_ids or not store.exists():
        return 0
    files = _FileStore(store)
    try:
        ids = sorted(set(turn_ids))
        tainted: set[str] = set()
        dead: list[str] = []
        with files._lock:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                marks = ", ".join("?" * len(chunk))
                rows = files._conn.execute(
                    "SELECT path, turn_id, live FROM notes"
                    f" WHERE space_id = ? AND turn_id IN ({marks})",
                    [space, *chunk],
                ).fetchall()
                tainted.update(path for path, _, live in rows if live)
                dead += [turn for _, turn, _ in rows]
        if not dead:
            return 0
        files.prune(space, sorted(tainted), dead)
        return len(tainted)
    finally:
        files.close()
