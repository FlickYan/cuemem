"""结构化存储（SQLite）。

- 原始层：记忆空间、会话、原始轮次；
- 派生层（N3）：实体、事实、会话片段、核心记忆，以及巩固任务队列；
- 检索层：记忆单元（轮、事实、摘要、实体）。

原始层写入一律用 INSERT OR IGNORE 加确定性 ID，重复导入同一份数据不会产生重复记录。
派生层可以整体删除后从原始层重建（重放）。删除时开启 secure_delete，被删内容会被覆写，
不会残留在数据库文件的空闲页里（被遗忘权，见 D-021）。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from cuemem.core.models import CoreMemory, Entity, Episode, Fact, MemoryUnit, Turn
from cuemem.errors import ConfigMismatchError

SCHEMA_VERSION = 3  # v2：turns 增加 speaker 列；v3：实体、事实、片段、核心记忆、任务队列

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
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    aliases TEXT NOT NULL DEFAULT '[]',
    description TEXT NOT NULL DEFAULT '',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    mentions INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    statement TEXT NOT NULL,
    category TEXT NOT NULL,
    cardinality TEXT NOT NULL,
    negated INTEGER NOT NULL DEFAULT 0,
    valid_from TEXT,
    valid_to TEXT,
    recorded_at TEXT NOT NULL,
    invalidated_at TEXT,
    superseded_by TEXT,
    supersede_reason TEXT,
    importance INTEGER NOT NULL,
    entity_ids TEXT NOT NULL DEFAULT '[]',
    session_id TEXT NOT NULL,
    source_turn_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    summary TEXT NOT NULL,
    turn_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS core_memory (
    space_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    fact_ids TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS media (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    path TEXT NOT NULL,
    caption TEXT NOT NULL DEFAULT '',
    ocr_text TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    entity_ids TEXT NOT NULL DEFAULT '[]',
    source_turn_id TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    stats TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_space ON turns (space_id, session_id, idx);
CREATE INDEX IF NOT EXISTS idx_units_space ON units (space_id, kind, session_id);
CREATE INDEX IF NOT EXISTS idx_facts_space ON facts (space_id, subject_id, predicate);
CREATE INDEX IF NOT EXISTS idx_entities_space ON entities (space_id);
CREATE INDEX IF NOT EXISTS idx_jobs_space ON jobs (space_id, status);
"""

# 派生数据所在的表：forget 与重建时整体清空，再从原始轮次重放
DERIVED_TABLES = ("facts", "entities", "episodes", "core_memory", "jobs")
_FACT_COLUMNS = (
    "id, space_id, subject_id, subject, predicate, object, statement, category, cardinality,"
    " negated, valid_from, valid_to, recorded_at, invalidated_at, superseded_by,"
    " supersede_reason, importance, entity_ids, session_id, source_turn_ids"
)
_UNIT_COLUMNS = "id, space_id, kind, session_id, text, ts, source_turn_ids, meta"
_ENTITY_COLUMNS = "id, space_id, name, type, aliases, description, first_seen, last_seen, mentions"


class MemoryDB:
    def __init__(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA secure_delete=ON")  # 删除时覆写内容，不留在空闲页里
            self._conn.executescript(_SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(turns)")}
            if "speaker" not in columns:  # 从 v1 升级
                self._conn.execute("ALTER TABLE turns ADD COLUMN speaker TEXT")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """在一个事务里执行多条写入：要么全部生效，要么全部回滚。"""
        with self._lock:
            try:
                yield self._conn
            except BaseException:
                self._conn.rollback()
                raise
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

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    # ---------- 原始层 ----------

    def ensure_space(self, space_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO spaces (id, created_at) VALUES (?, ?)",
                (space_id, datetime.now().isoformat()),
            )

    def add_session(self, space_id: str, session_id: str, started_at: datetime) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (space_id, id, started_at) VALUES (?, ?, ?)",
                (space_id, session_id, started_at.isoformat()),
            )

    def get_sessions(self, space_id: str) -> dict[str, datetime]:
        """会话 ID → 开始时间。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, started_at FROM sessions WHERE space_id = ?", (space_id,)
            ).fetchall()
        return {r[0]: datetime.fromisoformat(r[1]) for r in rows}

    def speakers(self, space_id: str) -> list[str]:
        """多人对话里出现过的说话人名字（按首次出现排序）；普通的用户—助手对话返回空列表。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT speaker, MIN(ts) FROM turns WHERE space_id = ? AND speaker IS NOT NULL"
                " GROUP BY speaker ORDER BY MIN(ts), speaker",
                (space_id,),
            ).fetchall()
        return [r[0] for r in rows]

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
        with self.transaction() as conn:
            cursor = conn.executemany(
                "INSERT OR IGNORE INTO turns"
                " (id, space_id, session_id, idx, role, text, ts, media_ids, speaker)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return max(cursor.rowcount, 0)

    def get_turns(
        self,
        space_id: str,
        session_id: str | None = None,
        *,
        ids: Sequence[str] | None = None,
    ) -> list[Turn]:
        sql = "SELECT id, space_id, session_id, idx, role, text, ts, media_ids, speaker FROM turns"
        sql += " WHERE space_id = ?"
        args: list[Any] = [space_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            args.append(session_id)
        if ids is not None:
            if not ids:
                return []
            sql += f" AND id IN ({', '.join('?' for _ in ids)})"
            args.extend(ids)
        sql += " ORDER BY ts, session_id, idx"
        with self._lock:
            fetched = self._conn.execute(sql, args).fetchall()
        return [_row_to_turn(r) for r in fetched]

    def delete_turns(self, space_id: str, turn_ids: Sequence[str]) -> int:
        """删除原始轮次（遗忘）；会话里的轮次全部删除后，会话本身也删除。"""
        if not turn_ids:
            return 0
        with self.transaction() as conn:
            deleted = 0
            for start in range(0, len(turn_ids), 500):
                chunk = list(turn_ids[start : start + 500])
                marks = ", ".join("?" * len(chunk))
                cursor = conn.execute(
                    f"DELETE FROM turns WHERE space_id = ? AND id IN ({marks})",
                    [space_id, *chunk],
                )
                deleted += max(cursor.rowcount, 0)
            conn.execute(
                "DELETE FROM sessions WHERE space_id = ? AND id NOT IN"
                " (SELECT DISTINCT session_id FROM turns WHERE space_id = ?)",
                (space_id, space_id),
            )
        return deleted

    # ---------- 检索层：记忆单元 ----------

    def add_units(self, units: Sequence[MemoryUnit]) -> int:
        with self.transaction() as conn:
            return _insert_units(conn, units, replace=False)

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
                    f"SELECT {_UNIT_COLUMNS} FROM units"
                    f" WHERE space_id = ? AND id IN ({placeholders})",
                    [space_id, *chunk],
                ).fetchall()
            for row in fetched:
                unit = _row_to_unit(row)
                rows[unit.id] = unit
        return [rows[i] for i in ids if i in rows]

    def list_units(
        self,
        space_id: str,
        *,
        kinds: Sequence[str] | None = None,
        session_ids: Sequence[str] | None = None,
    ) -> list[MemoryUnit]:
        sql = f"SELECT {_UNIT_COLUMNS} FROM units WHERE space_id = ?"
        args: list[Any] = [space_id]
        if kinds is not None:
            sql += f" AND kind IN ({', '.join('?' for _ in kinds)})"
            args.extend(kinds)
        if session_ids is not None:
            if not session_ids:
                return []
            sql += f" AND session_id IN ({', '.join('?' for _ in session_ids)})"
            args.extend(session_ids)
        sql += " ORDER BY ts, id"
        with self._lock:
            fetched = self._conn.execute(sql, args).fetchall()
        return [_row_to_unit(r) for r in fetched]

    def delete_units(self, space_id: str, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        deleted = 0
        with self.transaction() as conn:
            for start in range(0, len(ids), 500):
                chunk = list(ids[start : start + 500])
                marks = ", ".join("?" * len(chunk))
                cursor = conn.execute(
                    f"DELETE FROM units WHERE space_id = ? AND id IN ({marks})",
                    [space_id, *chunk],
                )
                deleted += max(cursor.rowcount, 0)
        return deleted

    # ---------- 派生层：实体、事实、片段、核心记忆 ----------

    def apply_consolidation(
        self,
        *,
        facts: Sequence[Fact],
        entities: Sequence[Entity],
        episodes: Sequence[Episode],
        units: Sequence[MemoryUnit],
        job_id: str | None = None,
        job_stats: dict[str, Any] | None = None,
    ) -> None:
        """一个会话的巩固结果在一个事务里写入：事实（含被取代的旧事实）、实体、片段、
        检索单元，并把任务标记为完成。中途失败则整体回滚，任务保持待处理，重跑即可。"""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO facts ({_FACT_COLUMNS}) VALUES ({', '.join('?' * 20)})",
                [_fact_row(f) for f in facts],
            )
            conn.executemany(
                f"INSERT OR REPLACE INTO entities ({_ENTITY_COLUMNS})"
                f" VALUES ({', '.join('?' * 9)})",
                [_entity_row(e) for e in entities],
            )
            conn.executemany(
                "INSERT OR REPLACE INTO episodes"
                " (id, space_id, session_id, start_ts, end_ts, summary, turn_ids)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        e.id,
                        e.space_id,
                        e.session_id,
                        e.start_ts.isoformat(),
                        e.end_ts.isoformat(),
                        e.summary,
                        json.dumps(e.turn_ids),
                    )
                    for e in episodes
                ],
            )
            _insert_units(conn, units, replace=True)
            if job_id is not None:
                conn.execute(
                    "UPDATE jobs SET status = 'done', error = NULL, stats = ?, updated_at = ?"
                    " WHERE id = ?",
                    (json.dumps(job_stats or {}, ensure_ascii=False), now, job_id),
                )

    def get_facts(
        self,
        space_id: str,
        *,
        ids: Sequence[str] | None = None,
        current_only: bool = False,
        subject_id: str | None = None,
        superseded_by: Sequence[str] | None = None,
        valid_at: str | None = None,
        known_at: datetime | None = None,
    ) -> list[Fact]:
        """查询事实。两条时间轴可以分别查询：

        - valid_at（事件时间，ISO 日期）："那时现实中什么是成立的"，例如"2025 年 6 月我在哪工作"。
          不知道起始时间的事实，从它被说出的那天算起；被更正的事实从未成立，不会出现；
        - known_at（记录时间）："系统在那一刻相信什么"，用来回放当时的认知、审计与调试。
        """
        sql = f"SELECT {_FACT_COLUMNS} FROM facts WHERE space_id = ?"
        args: list[Any] = [space_id]
        if ids is not None:
            if not ids:
                return []
            sql += f" AND id IN ({', '.join('?' for _ in ids)})"
            args.extend(ids)
        if current_only:
            sql += " AND invalidated_at IS NULL"
        if subject_id is not None:
            sql += " AND subject_id = ?"
            args.append(subject_id)
        if superseded_by is not None:  # 被这些事实取代的旧版本
            if not superseded_by:
                return []
            sql += f" AND superseded_by IN ({', '.join('?' for _ in superseded_by)})"
            args.extend(superseded_by)
        if valid_at is not None:
            # 部分日期按字典序比较即可：'2023-05' <= '2023-05-07'
            sql += (
                " AND COALESCE(valid_from, substr(recorded_at, 1, 10)) <= ?"
                " AND (valid_to IS NULL OR ? < valid_to)"
            )
            args.extend([valid_at, valid_at])
            sql += " AND (supersede_reason IS NULL OR supersede_reason != 'corrected')"
        if known_at is not None:
            sql += " AND recorded_at <= ? AND (invalidated_at IS NULL OR ? < invalidated_at)"
            args.extend([known_at.isoformat(), known_at.isoformat()])
        sql += " ORDER BY recorded_at, id"
        with self._lock:
            fetched = self._conn.execute(sql, args).fetchall()
        facts = [_row_to_fact(r) for r in fetched]
        if ids is not None:
            order = {fid: i for i, fid in enumerate(ids)}
            facts.sort(key=lambda f: order.get(f.id, len(order)))
        return facts

    def get_entities(self, space_id: str, ids: Sequence[str] | None = None) -> list[Entity]:
        sql = f"SELECT {_ENTITY_COLUMNS} FROM entities WHERE space_id = ?"
        args: list[Any] = [space_id]
        if ids is not None:
            if not ids:
                return []
            sql += f" AND id IN ({', '.join('?' for _ in ids)})"
            args.extend(ids)
        sql += " ORDER BY first_seen, id"
        with self._lock:
            fetched = self._conn.execute(sql, args).fetchall()
        return [_row_to_entity(r) for r in fetched]

    def get_episodes(self, space_id: str) -> list[Episode]:
        with self._lock:
            fetched = self._conn.execute(
                "SELECT id, space_id, session_id, start_ts, end_ts, summary, turn_ids"
                " FROM episodes WHERE space_id = ? ORDER BY start_ts, id",
                (space_id,),
            ).fetchall()
        return [
            Episode(
                id=r[0],
                space_id=r[1],
                session_id=r[2],
                start_ts=datetime.fromisoformat(r[3]),
                end_ts=datetime.fromisoformat(r[4]),
                summary=r[5],
                turn_ids=json.loads(r[6]),
            )
            for r in fetched
        ]

    def set_core_memory(self, memory: CoreMemory) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO core_memory (space_id, text, fact_ids, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    memory.space_id,
                    memory.text,
                    json.dumps(memory.fact_ids),
                    memory.updated_at.isoformat(),
                ),
            )

    def get_core_memory(self, space_id: str) -> CoreMemory | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT space_id, text, fact_ids, updated_at FROM core_memory WHERE space_id = ?",
                (space_id,),
            ).fetchone()
        if row is None:
            return None
        return CoreMemory(
            space_id=row[0],
            text=row[1],
            fact_ids=json.loads(row[2]),
            updated_at=datetime.fromisoformat(row[3]),
        )

    def clear_derived(self, space_id: str) -> list[str]:
        """清空某个空间的全部派生数据（事实、实体、片段、核心记忆、任务，以及除"轮"以外的
        检索单元），返回被删除的检索单元 ID，供调用方同步删除向量。"""
        with self.transaction() as conn:
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM units WHERE space_id = ? AND kind != 'round'", (space_id,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM units WHERE space_id = ? AND kind != 'round'", (space_id,))
            for table in DERIVED_TABLES:
                conn.execute(f"DELETE FROM {table} WHERE space_id = ?", (space_id,))
        return ids

    # ---------- 巩固任务队列 ----------

    def enqueue_job(
        self, job_id: str, space_id: str, session_id: str, version: str, content_hash: str
    ) -> bool:
        """登记一个巩固任务；同一任务键已存在时什么也不做（幂等）。返回是否新登记。"""
        now = datetime.now().isoformat()
        with self.transaction() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO jobs"
                " (id, space_id, session_id, version, content_hash, status, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (job_id, space_id, session_id, version, content_hash, now, now),
            )
            return cursor.rowcount > 0

    def get_jobs(self, space_id: str, status: str | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, session_id, version, content_hash, status, attempts, error, stats,"
            " updated_at FROM jobs WHERE space_id = ?"
        )
        args: list[Any] = [space_id]
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        with self._lock:
            fetched = self._conn.execute(sql, args).fetchall()
        return [
            {
                "id": r[0],
                "session_id": r[1],
                "version": r[2],
                "content_hash": r[3],
                "status": r[4],
                "attempts": r[5],
                "error": r[6],
                "stats": json.loads(r[7]),
                "updated_at": r[8],
            }
            for r in fetched
        ]

    def fail_job(self, job_id: str, error: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'failed', attempts = attempts + 1, error = ?,"
                " updated_at = ? WHERE id = ?",
                (error[:2000], datetime.now().isoformat(), job_id),
            )

    def retry_failed_jobs(self, space_id: str) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = 'pending' WHERE space_id = ? AND status = 'failed'",
                (space_id,),
            )
            return max(cursor.rowcount, 0)

    # ---------- 空间级操作 ----------

    def delete_space(self, space_id: str) -> None:
        with self.transaction() as conn:
            for table in ("units", "turns", "sessions", "media", *DERIVED_TABLES):
                conn.execute(f"DELETE FROM {table} WHERE space_id = ?", (space_id,))
            conn.execute("DELETE FROM spaces WHERE id = ?", (space_id,))

    def list_spaces(self) -> list[dict[str, Any]]:
        sql = (
            "SELECT s.id, s.created_at,"
            " (SELECT COUNT(*) FROM sessions WHERE space_id = s.id),"
            " (SELECT COUNT(*) FROM turns WHERE space_id = s.id),"
            " (SELECT COUNT(*) FROM units WHERE space_id = s.id),"
            " (SELECT COUNT(*) FROM facts WHERE space_id = s.id AND invalidated_at IS NULL),"
            " (SELECT COUNT(*) FROM facts WHERE space_id = s.id),"
            " (SELECT COUNT(*) FROM entities WHERE space_id = s.id)"
            " FROM spaces s ORDER BY s.created_at"
        )
        with self._lock:
            fetched = self._conn.execute(sql).fetchall()
        return [
            {
                "space_id": r[0],
                "created_at": r[1],
                "sessions": r[2],
                "turns": r[3],
                "units": r[4],
                "current_facts": r[5],
                "facts": r[6],
                "entities": r[7],
            }
            for r in fetched
        ]

    def vacuum(self) -> None:
        """整理数据库文件（遗忘之后调用，进一步确保被删内容不留在文件里）。"""
        with self._lock:
            self._conn.commit()
            self._conn.execute("VACUUM")
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------- 行 ↔ 模型 ----------


def _insert_units(conn: sqlite3.Connection, units: Sequence[MemoryUnit], *, replace: bool) -> int:
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
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    cursor = conn.executemany(
        f"{verb} INTO units ({_UNIT_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    return max(cursor.rowcount, 0)


def _row_to_turn(r: Sequence[Any]) -> Turn:
    return Turn(
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


def _fact_row(f: Fact) -> tuple[Any, ...]:
    return (
        f.id,
        f.space_id,
        f.subject_id,
        f.subject,
        f.predicate,
        f.object,
        f.statement,
        f.category,
        f.cardinality,
        int(f.negated),
        f.valid_from,
        f.valid_to,
        f.recorded_at.isoformat(),
        f.invalidated_at.isoformat() if f.invalidated_at else None,
        f.superseded_by,
        f.supersede_reason,
        f.importance,
        json.dumps(f.entity_ids),
        f.session_id,
        json.dumps(f.source_turn_ids),
    )


def _row_to_fact(r: Sequence[Any]) -> Fact:
    return Fact(
        id=r[0],
        space_id=r[1],
        subject_id=r[2],
        subject=r[3],
        predicate=r[4],
        object=r[5],
        statement=r[6],
        category=r[7],
        cardinality=r[8],
        negated=bool(r[9]),
        valid_from=r[10],
        valid_to=r[11],
        recorded_at=datetime.fromisoformat(r[12]),
        invalidated_at=datetime.fromisoformat(r[13]) if r[13] else None,
        superseded_by=r[14],
        supersede_reason=r[15],
        importance=r[16],
        entity_ids=json.loads(r[17]),
        session_id=r[18],
        source_turn_ids=json.loads(r[19]),
    )


def _entity_row(e: Entity) -> tuple[Any, ...]:
    return (
        e.id,
        e.space_id,
        e.name,
        e.type,
        json.dumps(e.aliases, ensure_ascii=False),
        e.description,
        e.first_seen.isoformat(),
        e.last_seen.isoformat(),
        e.mentions,
    )


def _row_to_entity(r: Sequence[Any]) -> Entity:
    return Entity(
        id=r[0],
        space_id=r[1],
        name=r[2],
        type=r[3],
        aliases=json.loads(r[4]),
        description=r[5],
        first_seen=datetime.fromisoformat(r[6]),
        last_seen=datetime.fromisoformat(r[7]),
        mentions=r[8],
    )
