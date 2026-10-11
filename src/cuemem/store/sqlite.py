"""结构化存储（SQLite）：记忆空间、会话、原始轮次、记忆单元。

写入一律用 INSERT OR IGNORE 加确定性 ID，重复导入同一份数据不会产生重复记录。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from cuemem.core.models import MemoryUnit, Turn
from cuemem.errors import ConfigMismatchError

SCHEMA_VERSION = 2  # v2：turns 增加 speaker 列

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spaces (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    space_id TEXT NOT NULL,
    id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    PRIMARY KEY (space_id, id)
);
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    ts TEXT NOT NULL,
    media_ids TEXT NOT NULL DEFAULT '[]',
    speaker TEXT,
    UNIQUE (space_id, session_id, idx)
);
CREATE TABLE IF NOT EXISTS units (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    session_id TEXT NOT NULL,
    text TEXT NOT NULL,
    ts TEXT NOT NULL,
    source_turn_ids TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_turns_space ON turns (space_id, session_id, idx);
CREATE INDEX IF NOT EXISTS idx_units_space ON units (space_id, kind);
"""


class MemoryDB:
    def __init__(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(turns)")}
            if "speaker" not in columns:  # 从 v1 升级
                self._conn.execute("ALTER TABLE turns ADD COLUMN speaker TEXT")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    # ---------- 元信息 ----------

    def check_or_set_meta(self, key: str, value: str) -> None:
        """首次写入时记录配置指纹；之后若与当前配置不一致则报错（例如换了向量模型）。"""
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            if row is None:
                self._conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (key, value))
                self._conn.commit()
                return
        if row[0] != value:
            raise ConfigMismatchError(
                f"数据库 {self.path} 建立时的 {key} 是 {row[0]}，当前配置是 {value}。"
                "混用会导致向量不可比：请删除 data/ 目录后重新导入，"
                "或通过 CUEMEM_PATHS__* 指定新的存储位置。"
            )

    # ---------- 写入 ----------

    def ensure_space(self, space_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO spaces (id, created_at) VALUES (?, ?)",
                (space_id, datetime.now().isoformat()),
            )
            self._conn.commit()

    def add_session(self, space_id: str, session_id: str, started_at: datetime) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (space_id, id, started_at) VALUES (?, ?, ?)",
                (space_id, session_id, started_at.isoformat()),
            )
            self._conn.commit()

    def add_turns(self, turns: Sequence[Turn]) -> int:
        """写入原始轮次，返回实际新增的条数。"""
        rows = [
            (
                t.id,
                t.space_id,
                t.session_id,
                t.idx,
                t.role,
                t.text,
                t.ts.isoformat(),
                json.dumps(t.media_ids),
                t.speaker,
            )
            for t in turns
        ]
        with self._lock:
            cursor = self._conn.executemany(
                "INSERT OR IGNORE INTO turns"
                " (id, space_id, session_id, idx, role, text, ts, media_ids, speaker)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
            return max(cursor.rowcount, 0)

    def add_units(self, units: Sequence[MemoryUnit]) -> int:
        rows = [
            (
                u.id,
                u.space_id,
                u.kind,
                u.session_id,
                u.text,
                u.ts.isoformat(),
                json.dumps(u.source_turn_ids),
                json.dumps(u.meta, ensure_ascii=False),
            )
            for u in units
        ]
        with self._lock:
            cursor = self._conn.executemany(
                "INSERT OR IGNORE INTO units"
                " (id, space_id, kind, session_id, text, ts, source_turn_ids, meta)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
            return max(cursor.rowcount, 0)

    def delete_space(self, space_id: str) -> None:
        with self._lock:
            for table in ("units", "turns", "sessions"):
                self._conn.execute(f"DELETE FROM {table} WHERE space_id = ?", (space_id,))
            self._conn.execute("DELETE FROM spaces WHERE id = ?", (space_id,))
            self._conn.commit()

    # ---------- 读取 ----------

    def missing_unit_ids(self, space_id: str, ids: Sequence[str]) -> list[str]:
        """返回 ids 中尚未写入的那些，保持输入顺序。"""
        existing = {u.id for u in self.get_units(space_id, ids)}
        return [i for i in ids if i not in existing]

    def get_units(self, space_id: str, ids: Sequence[str]) -> list[MemoryUnit]:
        """按给定 ID 顺序返回记忆单元；不存在的 ID 会被跳过。"""
        rows: dict[str, MemoryUnit] = {}
        unique = list(dict.fromkeys(ids))
        for start in range(0, len(unique), 500):
            chunk = unique[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            with self._lock:
                fetched = self._conn.execute(
                    "SELECT id, space_id, kind, session_id, text, ts, source_turn_ids, meta"
                    f" FROM units WHERE space_id = ? AND id IN ({placeholders})",
                    [space_id, *chunk],
                ).fetchall()
            for row in fetched:
                unit = _row_to_unit(row)
                rows[unit.id] = unit
        return [rows[i] for i in ids if i in rows]

    def get_turns(self, space_id: str, session_id: str | None = None) -> list[Turn]:
        sql = "SELECT id, space_id, session_id, idx, role, text, ts, media_ids, speaker FROM turns"
        sql += " WHERE space_id = ?"
        args: list[Any] = [space_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            args.append(session_id)
        sql += " ORDER BY ts, session_id, idx"
        with self._lock:
            fetched = self._conn.execute(sql, args).fetchall()
        return [
            Turn(
                id=r[0],
                space_id=r[1],
                session_id=r[2],
                idx=r[3],
                role=r[4],
                text=r[5],
                ts=datetime.fromisoformat(r[6]),
                media_ids=json.loads(r[7]),
                speaker=r[8],
            )
            for r in fetched
        ]

    def list_spaces(self) -> list[dict[str, Any]]:
        sql = (
            "SELECT s.id, s.created_at,"
            " (SELECT COUNT(*) FROM sessions WHERE space_id = s.id),"
            " (SELECT COUNT(*) FROM turns WHERE space_id = s.id),"
            " (SELECT COUNT(*) FROM units WHERE space_id = s.id)"
            " FROM spaces s ORDER BY s.created_at"
        )
        with self._lock:
            fetched = self._conn.execute(sql).fetchall()
        return [
            {"space_id": r[0], "created_at": r[1], "sessions": r[2], "turns": r[3], "units": r[4]}
            for r in fetched
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_unit(row: Sequence[Any]) -> MemoryUnit:
    return MemoryUnit(
        id=row[0],
        space_id=row[1],
        kind=row[2],
        session_id=row[3],
        text=row[4],
        ts=datetime.fromisoformat(row[5]),
        source_turn_ids=json.loads(row[6]),
        meta=json.loads(row[7]),
    )
