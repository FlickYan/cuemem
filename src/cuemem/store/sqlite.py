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
import logging
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from cuemem.core.models import (
    CoreMemory,
    Entity,
    Episode,
    Fact,
    MediaItem,
    MemoryUnit,
    Turn,
    TurnMeta,
)
from cuemem.core.timeutil import naive_local
from cuemem.errors import ConfigMismatchError

logger = logging.getLogger(__name__)

MemoryKind = Literal["fact", "excerpt", "image"]  # 按 ID 找到的记忆的种类（MemoryDB.memory_kind）

# v2：turns 增加 speaker 列；v3：实体、事实、片段、核心记忆、任务队列；
# v4：media 增加孤立描述与尺寸；v5：provenance（助手的回复引用了哪些记忆，遗忘时沿引用扩散）；
# v6：turn_meta（SDK 写入时附带的 metadata、过期时间与 update 的更正目标）。新表都用
# CREATE TABLE IF NOT EXISTS 建，打开已有的库时自动补上；v7：表不变，标出第二轮复审的 128 位
# 集合名（VectorStore.collection_name，D-069）。add_keys（第六轮体检，add 的 idempotency_key）
# 也是这样补上的，不升版本：旧代码不读它，只是 delete_space 时不删它的行。
#
# 旧代码读不懂新代码写下的东西时就要升版本：不只是 SQLite 的表，也包括 Qdrant 的集合名与
# payload。D-069 的 128 位集合名（第二轮复审）是在 v6 里改的、当时没升版本，所以那次复审
# 之前的 v6 提交读不到之后建的空间（检索为空）。打开比 SCHEMA_VERSION 新的库时拒绝
# （MemoryDB.__init__），user_version 也从不调低：降级走导出再导入
SCHEMA_VERSION = 7

# 新增的索引不用升版本：每次打开都执行 _SCHEMA，已有的库自动补上。facts 的
# (space_id, superseded_by) 与 (space_id, invalidated_at) 每次读取都用得到：沿版本链找被取代的
# 旧版本（choose_facts）、数当前有效的事实（space_stats）。jobs 的 (space_id, session_id)：
# 登记任务时找同一会话还没执行的旧任务（enqueue_jobs），清空派生层之后按会话重新登记时，
# 没有它每登记一个都要扫一遍已经登记的任务
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
    ts TEXT NOT NULL,
    caption_iso TEXT NOT NULL DEFAULT '',
    width INTEGER NOT NULL DEFAULT 0,
    height INTEGER NOT NULL DEFAULT 0
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
CREATE TABLE IF NOT EXISTS provenance (
    space_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    target TEXT NOT NULL,
    PRIMARY KEY (space_id, turn_id, target)
);
CREATE TABLE IF NOT EXISTS turn_meta (
    space_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    expires_at TEXT,
    edit_of TEXT,
    PRIMARY KEY (space_id, turn_id)
);
CREATE TABLE IF NOT EXISTS add_keys (
    space_id TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    request_hash TEXT,
    session_id TEXT,
    turn_ids TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (space_id, key_hash)
);
CREATE INDEX IF NOT EXISTS idx_provenance_target ON provenance (space_id, target);
CREATE INDEX IF NOT EXISTS idx_turns_space ON turns (space_id, session_id, idx);
CREATE INDEX IF NOT EXISTS idx_units_space ON units (space_id, kind, session_id);
CREATE INDEX IF NOT EXISTS idx_facts_space ON facts (space_id, subject_id, predicate);
CREATE INDEX IF NOT EXISTS idx_facts_superseded ON facts (space_id, superseded_by);
CREATE INDEX IF NOT EXISTS idx_facts_current ON facts (space_id, invalidated_at);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts (space_id, session_id);
CREATE INDEX IF NOT EXISTS idx_entities_space ON entities (space_id);
CREATE INDEX IF NOT EXISTS idx_jobs_space ON jobs (space_id, status);
CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs (space_id, session_id);
CREATE INDEX IF NOT EXISTS idx_media_space ON media (space_id, source_turn_id);
CREATE INDEX IF NOT EXISTS idx_add_keys_created ON add_keys (created_at);
"""

# 派生数据所在的表：forget 与重建时整体清空，再从原始轮次重放
DERIVED_TABLES = ("facts", "entities", "episodes", "core_memory", "jobs")
_FACT_COLUMNS = (
    "id, space_id, subject_id, subject, predicate, object, statement, category, cardinality,"
    " negated, valid_from, valid_to, recorded_at, invalidated_at, superseded_by,"
    " supersede_reason, importance, entity_ids, session_id, source_turn_ids"
)
# 按事件时间查事实（get_facts 的 valid_period）时，没有自己起点的更正版的起点。更正指旧值从未
# 成立（见 write/reconcile.py），新值从旧值的起点起就成立；以前从更正那天算起，查更正之前的
# 日期新旧两版都查不到。corrected_start(id, start)：start 是 id 往前那条更正链到头的一版（自己有
# 起点，或不再更正过别的）的起点，从链头往后传：种子是这样到头的被更正版，起点只传给没有自己
# 起点的后继版；起点取各条链头里最早的。UNION 按 (id, start) 去重，行数不超过事实数 × 不同的
# 起点数。以前存每一版与它所有祖先的对（深 d 的链约 d²/2 行），反复 update 同一条记忆时，深
# 1000 的链一次查询约 4 秒、期间占着全库的锁。参数：space_id 一个，放在整句最前
_CORRECTED_CHAIN = (
    "WITH RECURSIVE corrected_start(id, start) AS ("
    "SELECT f.id, COALESCE(p.valid_from, substr(p.recorded_at, 1, 10))"
    " FROM facts p JOIN facts f ON f.id = p.superseded_by"
    " WHERE p.space_id = ? AND p.supersede_reason = 'corrected' AND f.valid_from IS NULL"
    " AND (p.valid_from IS NOT NULL OR NOT EXISTS (SELECT 1 FROM facts q"
    " WHERE q.space_id = p.space_id AND q.superseded_by = p.id"
    " AND q.supersede_reason = 'corrected'))"
    " UNION SELECT f.id, c.start FROM corrected_start c JOIN facts a ON a.id = c.id"
    " JOIN facts f ON f.id = a.superseded_by"
    " WHERE a.supersede_reason = 'corrected' AND f.valid_from IS NULL)"
)
_CORRECTED_START = "(SELECT MIN(c.start) FROM corrected_start c WHERE c.id = facts.id)"
_UNIT_COLUMNS = "id, space_id, kind, session_id, text, ts, source_turn_ids, meta"
_ENTITY_COLUMNS = "id, space_id, name, type, aliases, description, first_seen, last_seen, mentions"
_MEDIA_COLUMNS = (
    "id, space_id, sha256, path, caption, ocr_text, tags, entity_ids, source_turn_id, ts,"
    " caption_iso, width, height"
)
IN_CHUNK = 900  # 一条 SQL 的 IN (...) 里最多放几个参数（SQLite 旧版本的上限是 999）
TURN_WRITES_KEPT = 100_000  # 记着写入次数的会话数（见 turns_version），超过时清空、换一代
_IDX = "next-idx:"
_LEGACY_IDX = "next_idx:"


def next_idx_key(space_id: str, session_id: str) -> str:
    """SDK 记下的会话里下一句发言的序号（元信息的键，见 sdk.Memory._next_idx）：遗忘删掉的末尾
    几句的序号不再复用。空间 ID 带上长度（与进行中会话的标记相同）：以前的键 legacy_idx_key 里，
    空间 'a:b' 的会话 'c' 与空间 'a' 的会话 'b:c' 是同一行，一个用户的写入会让另一个用户会话里的
    序号跳过一段（抽取时当成有发言被删过）。以前的键照旧读。"""
    return f"{_IDX}{len(space_id)}:{space_id}:{session_id}"


def legacy_idx_key(space_id: str, session_id: str) -> str:
    """以前的 next_idx 的键（有歧义，见 next_idx_key）：只读，不再写。"""
    return f"{_LEGACY_IDX}{space_id}:{session_id}"


ADD_KEY_TTL = timedelta(hours=24)  # add 的 idempotency_key 记多久（与 Stripe 相同）


class MemoryDB:
    def __init__(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._lock = threading.RLock()
        self._depth = 0  # 嵌套的 transaction() 层数（只在持有 _lock 时读写）
        # 本连接的写入计数（只在持有 _lock 时读写），见 derived_version 与 turns_version
        self._derived: dict[str, int] = {}  # 空间 → 事实、实体与实体检索单元改过几次
        self._turn_epochs: dict[str, int] = {}  # 空间 → 删过几次发言（遗忘、删除空间）
        self._turn_writes: dict[tuple[str, str], int] = {}  # （空间，会话）→ 写过几次发言
        self._turn_generation = 0  # _turn_writes 清空过几次（见 _count_turn_writes）
        self._rollbacks = 0  # 回滚过的事务与保存点
        with self._lock:
            # 先读版本，再切 WAL（切换就会写文件头）、建表、补列：更新的 CueMem 写的库原样不动。
            # 以前每次打开都把 user_version 设成 SCHEMA_VERSION，旧版本打开新库时悄悄调低，
            # 新的集合名与 turn_meta 都被忽略（检索为空）
            stored = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if stored > SCHEMA_VERSION:
                self._conn.close()  # 不留句柄：serve、proxy 这样长时间运行的进程里会泄漏
                raise ConfigMismatchError(
                    f"{path} 是更新的 CueMem 写的（数据格式 v{stored}，这个版本只支持到"
                    f" v{SCHEMA_VERSION}）：升级 CueMem（各种安装方式的做法见 README 的“升级”"
                    "一节），或者用新版本导出（cuemem export）再在这里导入"
                )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA secure_delete=ON")  # 删除时覆写内容，不留在空闲页里
            self._conn.executescript(_SCHEMA)
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(turns)")}
            if "speaker" not in columns:  # 从 v1 升级
                self._conn.execute("ALTER TABLE turns ADD COLUMN speaker TEXT")
            media_columns = {row[1] for row in self._conn.execute("PRAGMA table_info(media)")}
            for column, ddl in (
                ("caption_iso", "TEXT NOT NULL DEFAULT ''"),
                ("width", "INTEGER NOT NULL DEFAULT 0"),
                ("height", "INTEGER NOT NULL DEFAULT 0"),
            ):
                if column not in media_columns:  # 从 v3 升级
                    self._conn.execute(f"ALTER TABLE media ADD COLUMN {column} {ddl}")
            self._conn.execute(f"PRAGMA user_version = {max(stored, SCHEMA_VERSION)}")  # 从不调低
            self._conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """在一个事务里执行多条写入：要么全部生效，要么全部回滚。

        可以嵌套：只有最外层提交；里层是保存点，里层出错只撤销里层，异常照常往外抛。以前里层
        提交时连外层已经写的一起提交了，外层随后出错也撤不回来（在线写入把发言、"轮"与巩固
        任务放在一个事务里，见 OnlineWriter.writing）。"""
        with self._lock:
            if self._depth:
                name = f"nested_{self._depth}"
                self._conn.execute(f"SAVEPOINT {name}")
                self._depth += 1
                try:
                    yield self._conn
                except BaseException:
                    self._rollbacks += 1
                    if self._conn.in_transaction:  # 有的错误（磁盘满等）SQLite 已经整个回滚了
                        self._conn.execute(f"ROLLBACK TO {name}")
                        self._conn.execute(f"RELEASE {name}")
                    raise
                finally:
                    self._depth -= 1
                self._conn.execute(f"RELEASE {name}")
                return
            if not self._conn.in_transaction:  # 显式开始：里层的保存点不会自己成为事务、提前提交
                self._conn.execute("BEGIN")
            self._depth = 1
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._rollbacks += 1
                self._conn.rollback()
                raise
            finally:
                self._depth = 0

    def derived_version(self, space_id: str) -> tuple[int, ...]:
        """这个空间的事实、实体与实体检索单元的版本：两次读到的相同，期间就没有改过（巩固据此
        跨运行复用载入的状态，见 Consolidator._load_states）。

        本连接的写入由改动它们的方法计数（apply_consolidation、clear_derived、delete_space，
        以及写到或删掉实体检索单元的 add_units、delete_units；以后加的写入方法也要计数）；
        事务回滚过、别的连接（别的进程）提交过写入，也都算改过。"""
        with self._lock:
            return (self._derived.get(space_id, 0), self._rollbacks, self._data_version())

    def turns_version(self, space_id: str, session_id: str) -> tuple[int, ...]:
        """这个会话的发言的版本：两次读到的相同，期间这个会话的发言就没有变过（巩固据此不用
        每次都重算全部会话的内容指纹，见 Consolidator.enqueue）。发言只会追加（add_turns）或
        删除（delete_turns、delete_space），这几个方法计数（以后加的改动发言的方法也要计数）；
        回滚与别的连接的提交同样算变过。写入计数清空过（_count_turn_writes）也算变过。"""
        return self.turns_versions(space_id, [session_id])[session_id]

    def turns_versions(
        self, space_id: str, session_ids: Sequence[str]
    ) -> dict[str, tuple[int, ...]]:
        """同 turns_version，一次取一批会话：只拿一次锁、只查一次 PRAGMA data_version。巩固为
        全部会话登记任务时用它，以前每个会话各拿一次锁、各查一次。"""
        with self._lock:
            generation = self._turn_generation
            epoch = self._turn_epochs.get(space_id, 0)
            rollbacks, changed = self._rollbacks, self._data_version()
            return {
                sid: (
                    generation,
                    epoch,
                    self._turn_writes.get((space_id, sid), 0),
                    rollbacks,
                    changed,
                )
                for sid in session_ids
            }

    def turn_epoch(self, space_id: str) -> int:
        """这个空间经本连接删过几次发言（遗忘、删除空间）。写入失败、留着稍后重试的发言（记忆
        代理与 Agents 的会话）据此判断期间有没有遗忘过：遗忘之前说的话不该在之后写进来。"""
        with self._lock:
            return self._turn_epochs.get(space_id, 0)

    def _data_version(self) -> int:
        """别的连接每提交一次写入就变一次（本连接自己的提交不算）。"""
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def _bump(self, space_ids: Iterable[str]) -> None:
        for space_id in set(space_ids):
            self._derived[space_id] = self._derived.get(space_id, 0) + 1

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
                "混用会导致向量不可比：换一个新的数据目录（环境变量 CUEMEM_HOME，SDK 用 "
                "Memory(home=...)），或用 CUEMEM_PATHS__DB 与 CUEMEM_PATHS__QDRANT 指定新的存储"
                "位置；确定不要这里的记忆时，删掉这个数据库与向量库后重新导入。"
            )

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def delete_meta(self, key: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM meta WHERE key = ?", (key,))

    def meta_keys(self, prefix: str) -> list[str]:
        """以 prefix 开头的键。按区间查询而不用 LIKE：键里的空间 ID 可能有 % 与 _。"""
        with self._lock:
            return _keys_under(self._conn, prefix)

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
        """会话 ID → 开始时间：整个空间的会话（2 万个会话约 30 毫秒）。只问有没有会话、或者
        某几个会话时用下面三个方法（按主键查，与会话数无关）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, started_at FROM sessions WHERE space_id = ?", (space_id,)
            ).fetchall()
        return {r[0]: _dt(r[1]) for r in rows}

    def session_turn_counts(self, space_id: str) -> dict[str, int]:
        """会话 ID → 发言条数（记忆浏览器的会话列表）：只数，不解出每句发言。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, COUNT(*) FROM turns WHERE space_id = ? GROUP BY session_id",
                (space_id,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def has_sessions(self, space_id: str) -> bool:
        """这个空间有没有会话（检索之前判断空间是不是空的）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE space_id = ? LIMIT 1", (space_id,)
            ).fetchone()
        return row is not None

    def session_started(self, space_id: str, session_id: str) -> datetime | None:
        """这个会话的开始时间；没有这个会话时为 None。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT started_at FROM sessions WHERE space_id = ? AND id = ?",
                (space_id, session_id),
            ).fetchone()
        return None if row is None else _dt(row[0])

    def sessions_of(self, space_id: str, session_ids: Sequence[str]) -> dict[str, datetime]:
        """同 get_sessions，只取这几个会话（不在库里的不出现）。"""
        ids = list(dict.fromkeys(session_ids))
        found: dict[str, datetime] = {}
        with self._lock:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                marks = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT id, started_at FROM sessions WHERE space_id = ? AND id IN ({marks})",
                    [space_id, *chunk],
                ).fetchall()
                found.update((r[0], _dt(r[1])) for r in rows)
        return found

    def speakers(self, space_id: str, *, without_sessions: Sequence[str] = ()) -> list[str]:
        """多人对话里出现过的说话人名字（按首次出现排序）；普通的用户—助手对话返回空列表。
        without_sessions：不算这几个会话（遗忘的快速路径比较删掉它们之后会不会变）。"""
        sql = "SELECT speaker, MIN(ts) FROM turns WHERE space_id = ? AND speaker IS NOT NULL"
        args: list[Any] = [space_id]
        if without_sessions:
            sql += f" AND session_id NOT IN ({', '.join('?' * len(without_sessions))})"
            args += list(without_sessions)
        with self._lock:
            rows = self._conn.execute(sql + " GROUP BY speaker ORDER BY MIN(ts), speaker", args)
            fetched = rows.fetchall()
        return [r[0] for r in fetched]

    def add_turns(self, turns: Sequence[Turn], meta: Mapping[str, TurnMeta] | None = None) -> int:
        """写入原始轮次，返回实际新增的条数。

        meta：轮 ID → 附加信息（SDK 的 metadata 与过期时间），在同一个事务里写，只给这次真正
        写进去的发言写：已有的发言（重复导入、撞号）不会被换上别人的附加信息。"""
        if meta:
            return self._add_turns_with_meta(turns, meta)
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
            self._count_turn_writes(turns)
            return max(cursor.rowcount, 0)

    def _add_turns_with_meta(self, turns: Sequence[Turn], meta: Mapping[str, TurnMeta]) -> int:
        added = 0
        with self.transaction() as conn:
            for t in turns:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO turns"
                    " (id, space_id, session_id, idx, role, text, ts, media_ids, speaker)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    ),
                )
                if cursor.rowcount <= 0:
                    continue
                added += 1
                info = meta.get(t.id)
                if info is not None:
                    conn.execute(
                        "INSERT OR REPLACE INTO turn_meta"
                        " (space_id, turn_id, metadata, expires_at, edit_of)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (
                            t.space_id,
                            t.id,
                            json.dumps(info.metadata, ensure_ascii=False),
                            info.expires_at,
                            info.edit_of,
                        ),
                    )
            # 与不带附加信息的写入一样计数（turns_version）：以前不计，同一个 run_id 带着 metadata
            # 再写时巩固以为会话没变，新的发言不再抽取
            self._count_turn_writes(turns)
        return added

    def _count_turn_writes(self, turns: Sequence[Turn]) -> None:
        """给写到的会话的写入次数加一（turns_version，持有 _lock 时调用）。记着的会话到了
        TURN_WRITES_KEPT 个时先清空、换一代：不给 run_id 的 add 每次都是一个新会话，以前一直
        记着，常驻的服务每 100 万次写入约 250 MB。换了一代，每个会话的版本都算变过，巩固重算
        一遍同样的内容指纹（Consolidator._digest），登记的任务不变。"""
        if len(self._turn_writes) >= TURN_WRITES_KEPT:
            self._turn_writes.clear()
            self._turn_generation += 1
        for key in {(t.space_id, t.session_id) for t in turns}:
            self._turn_writes[key] = self._turn_writes.get(key, 0) + 1

    def get_turn_meta(
        self, space_id: str, turn_ids: Sequence[str] | None = None
    ) -> dict[str, TurnMeta]:
        """轮 ID → 附加信息；没有附加信息的发言不在结果里。turn_ids 为 None 时取整个空间的。"""
        sql = "SELECT turn_id, metadata, expires_at, edit_of FROM turn_meta WHERE space_id = ?"
        chunks: list[list[str] | None] = [None]
        if turn_ids is not None:
            unique = list(dict.fromkeys(turn_ids))
            chunks = [unique[i : i + 500] for i in range(0, len(unique), 500)]
        out: dict[str, TurnMeta] = {}
        for chunk in chunks:
            query, args = sql, [space_id]
            if chunk is not None:
                query += f" AND turn_id IN ({', '.join('?' * len(chunk))})"
                args += chunk
            with self._lock:
                fetched = self._conn.execute(query, args).fetchall()
            for r in fetched:
                out[r[0]] = TurnMeta(metadata=json.loads(r[1]), expires_at=r[2], edit_of=r[3])
        return out

    def expiring_turns(self, space_id: str) -> dict[str, str]:
        """带过期时间的发言：轮 ID → 存下的过期时间（见 expiry.expired_ids）。走 turn_meta 的主键，
        没有附加信息的空间什么都不读。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn_id, expires_at FROM turn_meta"
                " WHERE space_id = ? AND expires_at IS NOT NULL",
                (space_id,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def spaces_with_expiring(self) -> list[str]:
        """有带过期时间的发言的空间（定时删过期的记忆用，见 MemorySystem.purge_expired）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT space_id FROM turn_meta WHERE expires_at IS NOT NULL"
                " ORDER BY space_id"
            ).fetchall()
        return [r[0] for r in rows]

    def turn_sessions(self, space_id: str, turn_ids: Sequence[str]) -> set[str]:
        """这几句发言所在的会话（只读 session_id 一列，见 expiry.expired_ids）。"""
        sql = "SELECT session_id FROM turns WHERE space_id = ?"
        return {r[0] for r in self._in_chunks(sql, [space_id], "id", turn_ids)}

    def sources_in(self, space_id: str, session_ids: Sequence[str]) -> list[tuple[str, list[str]]]:
        """这几个会话里的检索单元与事实：（ID，出处的发言），只读这两列、不构造模型（过期的记忆
        每次检索都要算一遍，见 expiry.expired_ids）。"""
        rows: list[Sequence[Any]] = []
        for table in ("units", "facts"):
            sql = f"SELECT id, source_turn_ids FROM {table} WHERE space_id = ?"
            rows.extend(self._in_chunks(sql, [space_id], "session_id", session_ids))
        return [(r[0], json.loads(r[1])) for r in rows]

    def media_of_turns(self, space_id: str, turn_ids: Sequence[str]) -> list[str]:
        """这几句发言里的图片 ID（只读 id 一列，见 expiry.expired_ids）。"""
        sql = "SELECT id FROM media WHERE space_id = ?"
        return [r[0] for r in self._in_chunks(sql, [space_id], "source_turn_id", turn_ids)]

    def get_turns(
        self,
        space_id: str,
        session_id: str | None = None,
        *,
        ids: Sequence[str] | None = None,
        session_ids: Sequence[str] | None = None,
    ) -> list[Turn]:
        """按日志顺序（ts, session_id, idx）返回发言。ids、session_ids 只取其中的（与 session_id
        一起时取交集）；很长的列表分批查，再按同样的顺序合起来（见 _turn_rows）。"""
        sql = "SELECT id, space_id, session_id, idx, role, text, ts, media_ids, speaker FROM turns"
        sql += " WHERE space_id = ?"
        args: list[Any] = [space_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            args.append(session_id)
        if ids is not None and session_ids is not None:
            sessions = set(session_ids)
            rows = self._turn_rows(sql, args, "id", ids)
            return [_row_to_turn(r) for r in rows if r[2] in sessions]
        if ids is not None:
            return [_row_to_turn(r) for r in self._turn_rows(sql, args, "id", ids)]
        if session_ids is not None:
            return [_row_to_turn(r) for r in self._turn_rows(sql, args, "session_id", session_ids)]
        sql += " ORDER BY ts, session_id, idx"
        with self._lock:
            fetched = self._conn.execute(sql, args).fetchall()
        return [_row_to_turn(r) for r in fetched]

    def turns_of(
        self, space_id: str, *, ids: Sequence[str] = (), session_ids: Sequence[str] = ()
    ) -> list[Turn]:
        """ids 里的发言加上 session_ids 里的会话的全部发言（并集，去重），按日志顺序返回：遗忘
        只读要删的这几个会话，不再读整个空间（第五轮体检：5 万轮的空间删一个会话 4.2 秒）。"""
        sql = "SELECT id, space_id, session_id, idx, role, text, ts, media_ids, speaker FROM turns"
        sql += " WHERE space_id = ?"
        rows = {r[0]: r for r in self._turn_rows(sql, [space_id], "session_id", session_ids)}
        for r in self._turn_rows(sql, [space_id], "id", ids):
            rows.setdefault(r[0], r)
        return [_row_to_turn(r) for r in sorted(rows.values(), key=_turn_order)]

    def _in_chunks(
        self, sql: str, args: list[Any], column: str, values: Sequence[Any], order: str = ""
    ) -> list[Sequence[Any]]:
        """sql AND column IN values 分批执行（每批 IN_CHUNK 个，重复的值只查一次），行按批次
        接起来；给了 order 时每批 ORDER BY order，分了几批的由调用方再排一次。"""
        unique = list(dict.fromkeys(values))
        rows: list[Sequence[Any]] = []
        for start in range(0, len(unique), IN_CHUNK):
            chunk = unique[start : start + IN_CHUNK]
            query = f"{sql} AND {column} IN ({', '.join('?' for _ in chunk)})"
            query += f" ORDER BY {order}" if order else ""
            with self._lock:
                rows.extend(self._conn.execute(query, [*args, *chunk]).fetchall())
        return rows

    def _turn_rows(
        self, sql: str, args: list[Any], column: str, values: Sequence[str]
    ) -> list[Sequence[Any]]:
        """按 column IN values 分批（每批 IN_CHUNK 个）取发言的行，按 ORDER BY ts, session_id, idx
        的顺序合起来：按存的原文排序（ts 是 TEXT，以前写进去的带时区的行按原文比较），与一次查完
        的 ORDER BY 相同。"""
        unique = list(dict.fromkeys(values))
        rows: list[Sequence[Any]] = []
        for start in range(0, len(unique), IN_CHUNK):
            chunk = unique[start : start + IN_CHUNK]
            query = f"{sql} AND {column} IN ({', '.join('?' for _ in chunk)})"
            query += " ORDER BY ts, session_id, idx"
            with self._lock:
                rows.extend(self._conn.execute(query, [*args, *chunk]).fetchall())
        if len(unique) > IN_CHUNK:
            rows.sort(key=_turn_order)
        return rows

    def delete_turns(self, space_id: str, turn_ids: Sequence[str]) -> int:
        """删除原始轮次（遗忘）；会话里的轮次全部删除后，会话本身也删除，连同它的 next_idx
        （_drop_idle_idx）。还有发言的会话的 next_idx 留着：删掉的末尾几句的序号不再复用。"""
        if not turn_ids:
            return 0
        with self.transaction() as conn:
            self._turn_epochs[space_id] = self._turn_epochs.get(space_id, 0) + 1
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
            self._drop_idle_idx(conn, space_id)
            # 被删发言里的图片记录一并删除（原图文件由调用方按引用情况清理）
            conn.execute(
                "DELETE FROM media WHERE space_id = ? AND source_turn_id NOT IN"
                " (SELECT id FROM turns WHERE space_id = ?)",
                (space_id, space_id),
            )
            conn.execute(
                "DELETE FROM provenance WHERE space_id = ? AND turn_id NOT IN"
                " (SELECT id FROM turns WHERE space_id = ?)",
                (space_id, space_id),
            )
            # 附加信息（metadata 里也可能有个人信息）随发言一起删
            conn.execute(
                "DELETE FROM turn_meta WHERE space_id = ? AND turn_id NOT IN"
                " (SELECT id FROM turns WHERE space_id = ?)",
                (space_id, space_id),
            )
        return deleted

    # ---------- add 的 idempotency_key（第六轮体检） ----------

    def add_key_lookup(self, space_id: str, key_hash: str) -> dict[str, Any] | None:
        """这个空间里这个键（调用方给的键的 sha256）记下的那次写入：request_hash、session_id、
        turn_ids；遗忘过的（墓碑）三者都是 None。没有记过、或者记了超过 ADD_KEY_TTL 时 None。"""
        cutoff = (datetime.now() - ADD_KEY_TTL).isoformat()
        with self._lock:
            row = self._conn.execute(
                "SELECT request_hash, session_id, turn_ids FROM add_keys"
                " WHERE space_id = ? AND key_hash = ? AND created_at >= ?",
                (space_id, key_hash, cutoff),
            ).fetchone()
        if row is None:
            return None
        return {
            "request_hash": row[0],
            "session_id": row[1],
            "turn_ids": json.loads(row[2]) if row[2] is not None else None,
        }

    def add_key_insert(
        self,
        conn: sqlite3.Connection,
        space_id: str,
        key_hash: str,
        request_hash: str,
        session_id: str,
        turn_ids: Sequence[str],
    ) -> None:
        """记下一次带键的写入（在调用方写发言的事务里）。先删掉超过 ADD_KEY_TTL 的行（全部空间，
        按 idx_add_keys_created 只找过期的行：以前每次扫一遍整张表，50 万个键时约 33 毫秒，持着
        写锁，第七轮体检）。每次都删，不隔一段时间才删：同一个键过期之后再用时，过期的那一行得先
        删掉，否则下面的 INSERT 撞上主键、这次写入整个回滚。再用普通的 INSERT：同一个键已经由
        并发的另一次写入提交时 sqlite3.IntegrityError，调用方的事务整个回滚（连同这次的发言与
        向量），再按 add_key_lookup 返回先提交的那次。"""
        now = datetime.now()
        conn.execute(
            "DELETE FROM add_keys WHERE created_at < ?", ((now - ADD_KEY_TTL).isoformat(),)
        )
        conn.execute(
            "INSERT INTO add_keys"
            " (space_id, key_hash, request_hash, session_id, turn_ids, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                space_id,
                key_hash,
                request_hash,
                session_id,
                json.dumps(list(turn_ids)),
                now.isoformat(),
            ),
        )

    def add_key_forget(
        self, conn: sqlite3.Connection, space_id: str, turn_ids: Iterable[str]
    ) -> int:
        """遗忘这些发言时（在遗忘的事务里）：写下过其中任何一句的键只留下墓碑——request_hash、
        session_id 与 turn_ids 都置空，只剩调用方随机的键的 sha256。请求的摘要由被遗忘的内容算出，
        不能留着；晚到的重试按墓碑返回 forgotten，不再写一遍。返回改了几行。"""
        doomed = set(turn_ids)
        if not doomed:
            return 0
        rows = conn.execute(
            "SELECT key_hash, turn_ids FROM add_keys WHERE space_id = ? AND turn_ids IS NOT NULL",
            (space_id,),
        ).fetchall()
        hit = [key for key, ids in rows if doomed.intersection(json.loads(ids))]
        for key in hit:
            conn.execute(
                "UPDATE add_keys SET request_hash = NULL, session_id = NULL, turn_ids = NULL"
                " WHERE space_id = ? AND key_hash = ?",
                (space_id, key),
            )
        return len(hit)

    def _drop_idle_idx(self, conn: sqlite3.Connection, space_id: str) -> None:
        """删掉这个空间里已经没有发言的会话的 next_idx（next_idx_key，在调用方的事务里）：会话
        没有发言时 SDK 的序号本来就从 0 开始，留着的只是写着用户 ID 与写入时间的键（不给 run_id
        的 add 每次一个）。以前遗忘与删除空间都不删，delete_all 之后用户 ID 还在库里。

        以前的键有歧义：不能按前缀删（空间 'acme' 的前缀 'next_idx:acme:' 底下也有空间
        'acme:alice' 的键），每个键按每一种拆法核对，哪一种拆出的会话还在就留着。顺带删掉以前的
        遗忘留下的。新的键带着空间 ID 的长度，前缀底下的都是这个空间的。"""
        idle: list[tuple[str]] = []
        prefix = next_idx_key(space_id, "")
        for key in _keys_under(conn, prefix):
            if not _has_session(conn, space_id, key.removeprefix(prefix)):
                idle.append((key,))
        for key in _keys_under(conn, legacy_idx_key(space_id, "")):
            rest = key.removeprefix(_LEGACY_IDX)
            owners = [(rest[:i], rest[i + 1 :]) for i, c in enumerate(rest) if c == ":"]
            if not any(_has_session(conn, space, sid) for space, sid in owners):
                idle.append((key,))
        conn.executemany("DELETE FROM meta WHERE key = ?", idle)

    # ---------- 溯源：助手的回复引用了哪些记忆（N8） ----------

    def add_provenance(self, space_id: str, turn_id: str, targets: Sequence[str]) -> None:
        """记下一次回复（助手的发言）引用的记忆：事实、轮或图片的 ID。属于原始层，
        重建派生记忆时不清空；发言被删时一起删。"""
        if not targets:
            return
        with self.transaction() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO provenance (space_id, turn_id, target) VALUES (?, ?, ?)",
                [(space_id, turn_id, t) for t in dict.fromkeys(targets)],
            )

    def citing_turns(self, space_id: str, targets: Sequence[str]) -> list[str]:
        """引用过这些记忆的发言。targets 分批查（IN_CHUNK）：删掉整个用户时遗忘的轮、事实与图片
        可以有几万个，以前一条 SQL 超过 SQLite 的参数上限（3.32 起 32766）就报错、什么都没删。"""
        sql = "SELECT DISTINCT turn_id FROM provenance WHERE space_id = ?"
        rows = self._in_chunks(sql, [space_id], "target", targets)
        return list(dict.fromkeys(r[0] for r in rows))

    def provenance(self, space_id: str, turn_id: str) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT target FROM provenance WHERE space_id = ? AND turn_id = ?",
                (space_id, turn_id),
            ).fetchall()
        return [r[0] for r in rows]

    # ---------- 检索层：记忆单元 ----------

    def add_units(self, units: Sequence[MemoryUnit]) -> int:
        with self.transaction() as conn:
            self._bump(u.space_id for u in units if u.kind == "entity")
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
        if session_ids is not None:  # 很长的列表分批查，再按 ORDER BY ts, id 合起来
            unique = list(dict.fromkeys(session_ids))
            rows: list[Sequence[Any]] = []
            for start in range(0, len(unique), IN_CHUNK):
                chunk = unique[start : start + IN_CHUNK]
                query = f"{sql} AND session_id IN ({', '.join('?' for _ in chunk)})"
                with self._lock:
                    rows.extend(
                        self._conn.execute(query + " ORDER BY ts, id", [*args, *chunk]).fetchall()
                    )
            if len(unique) > IN_CHUNK:
                rows.sort(key=lambda r: (r[5], r[0]))
            return [_row_to_unit(r) for r in rows]
        sql += " ORDER BY ts, id"
        with self._lock:
            fetched = self._conn.execute(sql, args).fetchall()
        return [_row_to_unit(r) for r in fetched]

    def unit_texts(self, space_id: str, *, kinds_not: Sequence[str] = ()) -> list[str]:
        """这个空间的记忆单元的文本（不要 kinds_not 里的几种），只读 text 一列：遗忘时记下派生
        记忆的文本，不再把整个空间的记忆单元解出来（见 MemorySystem.forget）。"""
        sql = "SELECT text FROM units WHERE space_id = ?"
        if kinds_not:
            sql += f" AND kind NOT IN ({', '.join('?' for _ in kinds_not)})"
        with self._lock:
            rows = self._conn.execute(sql, [space_id, *kinds_not]).fetchall()
        return [r[0] for r in rows]

    def unit_counts(self, space_id: str) -> dict[str, int]:
        """这个空间每种记忆单元各有几个（doctor --vectors 与向量库里的点数对比）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, COUNT(*) FROM units WHERE space_id = ? GROUP BY kind ORDER BY kind",
                (space_id,),
            ).fetchall()
        return {str(kind): int(n) for kind, n in rows}

    def existing_texts(self, space_id: str, texts: Iterable[str]) -> set[str]:
        """texts 里还有记忆单元用着的那些（分批查）。"""
        unique = list(dict.fromkeys(texts))
        found: set[str] = set()
        for start in range(0, len(unique), IN_CHUNK):
            chunk = unique[start : start + IN_CHUNK]
            marks = ", ".join("?" for _ in chunk)
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT DISTINCT text FROM units WHERE space_id = ? AND text IN ({marks})",
                    [space_id, *chunk],
                ).fetchall()
            found.update(r[0] for r in rows)
        return found

    def some_unit_ids(self, space_id: str, limit: int = 3) -> list[str]:
        """这个空间的几个记忆单元 ID，"轮"优先（在线写入时一定写了向量）：向量库据此确认旧名字的
        集合是这个空间的（见 VectorStore.collection_name）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM units WHERE space_id = ? ORDER BY kind != 'round' LIMIT ?",
                (space_id, limit),
            ).fetchall()
        return [r[0] for r in rows]

    def has_units(self, space_id: str, kinds: Sequence[str]) -> bool:
        """空间里有没有这几种记忆单元（走 idx_units_space，不扫表）。"""
        marks = ", ".join("?" for _ in kinds)
        with self._lock:
            row = self._conn.execute(
                f"SELECT 1 FROM units WHERE space_id = ? AND kind IN ({marks}) LIMIT 1",
                [space_id, *kinds],
            ).fetchone()
        return row is not None

    def delete_units(self, space_id: str, ids: Sequence[str]) -> int:
        if not ids:
            return 0
        deleted = 0
        with self.transaction() as conn:
            for start in range(0, len(ids), 500):
                chunk = list(ids[start : start + 500])
                marks = ", ".join("?" * len(chunk))
                entity = conn.execute(  # 删到实体的检索单元时，巩固载入的状态就变了
                    "SELECT 1 FROM units WHERE space_id = ? AND kind = 'entity'"
                    f" AND id IN ({marks}) LIMIT 1",
                    [space_id, *chunk],
                ).fetchone()
                if entity is not None:
                    self._bump([space_id])
                cursor = conn.execute(
                    f"DELETE FROM units WHERE space_id = ? AND id IN ({marks})",
                    [space_id, *chunk],
                )
                deleted += max(cursor.rowcount, 0)
        return deleted

    # ---------- 图片（N6） ----------

    def add_media(self, items: Sequence[MediaItem]) -> int:
        """登记图片（在线写入时，还没有描述）。重复登记同一条什么也不做。"""
        rows = [
            (
                m.id,
                m.space_id,
                m.sha256,
                m.path,
                m.caption,
                m.ocr_text,
                json.dumps(m.tags, ensure_ascii=False),
                json.dumps(m.entity_ids),
                m.source_turn_id,
                m.ts.isoformat(),
                m.caption_iso,
                m.width,
                m.height,
            )
            for m in items
        ]
        with self.transaction() as conn:
            cursor = conn.executemany(
                f"INSERT OR IGNORE INTO media ({_MEDIA_COLUMNS}) VALUES ({', '.join('?' * 13)})",
                rows,
            )
            return max(cursor.rowcount, 0)

    def get_media(
        self,
        space_id: str,
        *,
        ids: Sequence[str] | None = None,
        turn_ids: Sequence[str] | None = None,
    ) -> list[MediaItem]:
        """图片，按 ts、id 排。ids 与 turn_ids 里长的那个分批查（IN_CHUNK）：遗忘整个用户时的
        turn_ids 可以有几万个，以前一条 SQL 超过 SQLite 的参数上限就报错、什么都没删。"""
        sql = f"SELECT {_MEDIA_COLUMNS} FROM media WHERE space_id = ?"
        args: list[Any] = [space_id]
        given = [(c, v) for c, v in (("id", ids), ("source_turn_id", turn_ids)) if v is not None]
        if any(not values for _, values in given):
            return []
        if not given:
            with self._lock:
                fetched = self._conn.execute(f"{sql} ORDER BY ts, id", args).fetchall()
            return [_row_to_media(r) for r in fetched]
        longest = max(range(len(given)), key=lambda i: len(given[i][1]))
        for i, (column, values) in enumerate(given):
            if i != longest:
                sql += f" AND {column} IN ({', '.join('?' for _ in values)})"
                args.extend(values)
        column, values = given[longest]
        fetched = self._in_chunks(sql, args, column, values, "ts, id")
        if len(set(values)) > IN_CHUNK:
            fetched.sort(key=lambda r: (r[9] or "", r[0]))  # 同 ORDER BY ts, id（NULL 在前）
        return [_row_to_media(r) for r in fetched]

    def set_media_captions(self, items: Sequence[MediaItem]) -> None:
        """写入巩固阶段生成的描述（上下文感知描述、孤立描述、图中文字、标签）。"""
        with self.transaction() as conn:
            conn.executemany(
                "UPDATE media SET caption = ?, caption_iso = ?, ocr_text = ?, tags = ?"
                " WHERE id = ? AND space_id = ?",
                [
                    (
                        m.caption,
                        m.caption_iso,
                        m.ocr_text,
                        json.dumps(m.tags, ensure_ascii=False),
                        m.id,
                        m.space_id,
                    )
                    for m in items
                ],
            )

    def space_stats(self, space_id: str) -> dict[str, int]:
        """一个空间的规模：会话数、轮数、当前有效的事实数、图片数（检索控制器的特征，N7）。"""
        sql = (
            "SELECT (SELECT COUNT(*) FROM sessions WHERE space_id = ?),"
            " (SELECT COUNT(*) FROM units WHERE space_id = ? AND kind = 'round'),"
            " (SELECT COUNT(*) FROM facts WHERE space_id = ? AND invalidated_at IS NULL),"
            " (SELECT COUNT(*) FROM media WHERE space_id = ?)"
        )
        with self._lock:
            row = self._conn.execute(sql, (space_id,) * 4).fetchone()
        return dict(zip(("sessions", "rounds", "facts", "media"), row, strict=True))

    def has_media(self, space_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM media WHERE space_id = ? LIMIT 1", (space_id,)
            ).fetchone()
        return row is not None

    def media_sha_in_use(self, sha256: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM media WHERE sha256 = ? LIMIT 1", (sha256,)
            ).fetchone()
        return row is not None

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
            self._bump(
                [
                    *(f.space_id for f in facts),
                    *(e.space_id for e in entities),
                    *(u.space_id for u in units if u.kind == "entity"),
                ]
            )
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
        session_ids: Sequence[str] | None = None,
        valid_period: tuple[str, str] | None = None,
        known_at: datetime | None = None,
    ) -> list[Fact]:
        """查询事实。两条时间轴可以分别查询：

        - valid_period（事件时间，(起, 止) 两个含在内的 YYYY-MM-DD，见 timeutil.valid_period）：
          "那时现实中什么是成立的"，例如"2025 年 6 月我在哪工作"。给的是一段时间（一年、一个月）
          时，这段时间里任何时候成立过的都算，先后几个值可能同时出现（终点是部分日期时，终点所在
          的那一年、那个月也算）。不知道起始时间的事实，从它被说出的那天算起；被更正的事实从未
          成立，不会出现，没有自己起点的更正版沿用它更正的那一版的起点；
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
        if session_ids is not None and not session_ids:  # 这几个会话里抽出的事实（见下）
            return []
        if valid_period is not None:
            # 与这段时间有重叠：起点不晚于段尾、终点晚于段首。存下的部分日期按字典序比较即可：
            # '2023-05' <= '2023-05-31'。查询的一端是部分日期时不行（'2023-05-07' <= '2023-05'
            # 不成立），所以查询总是换成完整的日期；只查一天时两端相同，与以前的结果一样
            lo, hi = valid_period
            # 没有自己起点的更正版，起点沿版本链取它更正的那一版的（见 _CORRECTED_START）
            sql = f"{_CORRECTED_CHAIN} {sql}"
            args.insert(0, space_id)
            sql += f" AND COALESCE(valid_from, {_CORRECTED_START}, substr(recorded_at, 1, 10)) <= ?"
            args.append(hi)
            if lo == hi:
                sql += " AND (valid_to IS NULL OR ? < valid_to)"
                args.append(lo)
            else:
                # 终点常是部分日期（取代时记成新值的起点，'2026'、'2026-06'）：'2026-01-01' <
                # '2026' 不成立，以前查 2026 年就没有换走的旧值。终点是这段时间开头所在的年、月时
                # 也算重叠。只能用 =：用 >= 的话旧值在以后每一年、每个月都会出现
                sql += (
                    " AND (valid_to IS NULL OR ? < valid_to"
                    " OR (length(valid_to) < 10 AND substr(?, 1, length(valid_to)) = valid_to))"
                )
                args.extend([lo, lo])
            sql += " AND (supersede_reason IS NULL OR supersede_reason != 'corrected')"
        if known_at is not None:
            sql += " AND recorded_at <= ? AND (invalidated_at IS NULL OR ? < invalidated_at)"
            args.extend([known_at.isoformat(), known_at.isoformat()])
        if session_ids is not None:  # 分批查（IN_CHUNK）：过期的发言可以出自几万个会话
            fetched = self._in_chunks(sql, args, "session_id", session_ids, "recorded_at, id")
            if len(set(session_ids)) > IN_CHUNK:
                fetched.sort(key=lambda r: (r[12], r[0]))  # 同 ORDER BY recorded_at, id
        else:
            with self._lock:
                fetched = self._conn.execute(f"{sql} ORDER BY recorded_at, id", args).fetchall()
        facts = [_row_to_fact(r) for r in fetched]
        if ids is not None:
            order = {fid: i for i, fid in enumerate(ids)}
            facts.sort(key=lambda f: order.get(f.id, len(order)))
        return facts

    def fact_ids(self, space_id: str) -> list[str]:
        """全部事实的 ID，顺序同 get_facts（不构造 Fact）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM facts WHERE space_id = ? ORDER BY recorded_at, id", (space_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def fact_links(self, space_id: str) -> dict[str, tuple[bool, str | None]]:
        """事实 ID →（是否当前有效，被谁取代），顺序同 get_facts（不构造 Fact）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, invalidated_at IS NULL, superseded_by FROM facts WHERE space_id = ?"
                " ORDER BY recorded_at, id",
                (space_id,),
            ).fetchall()
        return {r[0]: (bool(r[1]), r[2]) for r in rows}

    def memory_kind(self, memory_id: str) -> tuple[str, MemoryKind] | None:
        """SDK 按 ID 找的记忆：（所在的记忆空间，种类）——事实、对话片段（"轮"）或图片，就是
        search 结果里的三种 ID；都不是时 None。三张表的 id 都是主键，ID 由空间派生、全局唯一。
        以前只查事实：search 给出的对话片段与图片的 ID，get、delete 都当成不存在。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT space_id, 'fact', 0 FROM facts WHERE id = ?"
                " UNION ALL"
                " SELECT space_id, 'excerpt', 1 FROM units WHERE id = ? AND kind = 'round'"
                " UNION ALL"
                " SELECT space_id, 'image', 2 FROM media WHERE id = ?"
                " ORDER BY 3 LIMIT 1",
                (memory_id, memory_id, memory_id),
            ).fetchone()
        return (row[0], row[1]) if row else None

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

    def entity_ids(self, space_id: str) -> list[str]:
        """全部实体的 ID，顺序同 get_entities。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM entities WHERE space_id = ? ORDER BY first_seen, id", (space_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def entity_names(self, space_id: str) -> list[tuple[str, list[str]]]:
        """各实体的名字与别名（顺序同 get_entities）：只要名字时不用逐行构造 Entity。"""
        with self._lock:
            fetched = self._conn.execute(
                "SELECT name, aliases FROM entities WHERE space_id = ? ORDER BY first_seen, id",
                (space_id,),
            ).fetchall()
        return [(r[0], json.loads(r[1])) for r in fetched]

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
                start_ts=_dt(r[3]),
                end_ts=_dt(r[4]),
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
            updated_at=_dt(row[3]),
        )

    def derived_unit_ids(self, space_id: str) -> list[str]:
        """这个空间里除"轮"以外的检索单元（派生记忆）的 ID：清空之前先删它们的向量（见
        Consolidator.rebuild）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM units WHERE space_id = ? AND kind != 'round'", (space_id,)
            ).fetchall()
        return [r[0] for r in rows]

    def clear_derived(self, space_id: str) -> list[str]:
        """清空某个空间的全部派生数据（事实、实体、片段、核心记忆、任务，以及除"轮"以外的
        检索单元），返回被删除的检索单元 ID，供调用方同步删除向量。"""
        with self.transaction() as conn:
            self._bump([space_id])
            ids = [
                r[0]
                for r in conn.execute(
                    "SELECT id FROM units WHERE space_id = ? AND kind != 'round'", (space_id,)
                ).fetchall()
            ]
            conn.execute("DELETE FROM units WHERE space_id = ? AND kind != 'round'", (space_id,))
            for table in DERIVED_TABLES:
                conn.execute(f"DELETE FROM {table} WHERE space_id = ?", (space_id,))
            # 图片描述也是派生数据：清空后由巩固重新生成（命中缓存）；图片本身属于原始层
            conn.execute(
                "UPDATE media SET caption = '', caption_iso = '', ocr_text = '', tags = '[]'"
                " WHERE space_id = ?",
                (space_id,),
            )
        return ids

    def drop_inert_sessions(self, space_id: str, session_ids: Sequence[str]) -> list[str]:
        """遗忘的快速路径（第七轮体检，见 MemorySystem._inert_sessions）：整段删掉的会话什么都
        没留下（没有事实、没有新建或关联的实体），派生层里只有它们的片段与摘要单元、任务。删掉
        这些，连同这个空间里被取代的任务（重放时清空派生层，它们也不在了），别的已完成任务的
        失败次数清零（与重放相同），返回删掉的摘要单元 ID，供调用方删向量。在调用方的事务里
        调用（这里的事务是保存点）。"""
        ids = list(dict.fromkeys(session_ids))
        removed: list[str] = []
        with self.transaction() as conn:
            self._bump([space_id])
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                marks = ", ".join("?" * len(chunk))
                args = [space_id, *chunk]
                removed += [
                    r[0]
                    for r in conn.execute(
                        "SELECT id FROM units WHERE space_id = ? AND kind = 'summary'"
                        f" AND session_id IN ({marks})",
                        args,
                    )
                ]
                conn.execute(
                    "DELETE FROM units WHERE space_id = ? AND kind = 'summary'"
                    f" AND session_id IN ({marks})",
                    args,
                )
                for table in ("episodes", "jobs"):
                    conn.execute(
                        f"DELETE FROM {table} WHERE space_id = ? AND session_id IN ({marks})", args
                    )
            conn.execute(
                "DELETE FROM jobs WHERE space_id = ? AND status = 'superseded'", (space_id,)
            )
            # 别的会话已完成的任务：重放重新登记、重新执行（命中缓存），失败过几次的记录随之
            # 清零；内容审核跳过的保留原来的结局（write.consolidate._keeps_outcome）。与重放一致
            conn.execute(
                "UPDATE jobs SET attempts = 0, error = NULL WHERE space_id = ? AND status = 'done'"
                " AND (attempts != 0 OR error IS NOT NULL)"
                " AND COALESCE(json_extract(stats, '$.skipped'), '') != 'content_filter'",
                (space_id,),
            )
        return removed

    # ---------- 巩固任务队列 ----------

    def enqueue_job(
        self, job_id: str, space_id: str, session_id: str, version: str, content_hash: str
    ) -> bool:
        """登记一个巩固任务；同一任务键已存在时什么也不做（幂等）。返回是否新登记。

        新登记时，同一会话还没执行（或失败待重试）的旧任务标记为 superseded：会话又追加了
        发言，旧任务对应的是过时的内容，执行它只会把同一个会话多抽取一遍。
        """
        return self.enqueue_jobs(space_id, [(job_id, session_id, version, content_hash)]) > 0

    def enqueue_jobs(self, space_id: str, rows: Sequence[tuple[str, str, str, str]]) -> int:
        """按顺序登记一批任务（任务 ID，会话，流水线版本，内容指纹），每个与 enqueue_job 相同，
        在一个事务里、不为每个任务另开保存点。返回新登记的个数。

        任务多时（显式的巩固为全部会话登记）先读出这个空间已有的任务 ID，已有的不再插入：
        INSERT OR IGNORE 本来也什么都不做，结果相同，2 万个会话时省下约 30 毫秒。"""
        now = datetime.now().isoformat()
        added = 0
        with self.transaction() as conn:
            known: set[str] = set()
            if len(rows) > 100:
                known = {
                    r[0]
                    for r in conn.execute("SELECT id FROM jobs WHERE space_id = ?", (space_id,))
                }
            for job_id, session_id, version, content_hash in rows:
                if job_id in known:
                    continue
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO jobs (id, space_id, session_id, version, content_hash,"
                    " status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                    (job_id, space_id, session_id, version, content_hash, now, now),
                )
                if cursor.rowcount > 0:
                    added += 1
                    conn.execute(
                        "UPDATE jobs SET status = 'superseded', updated_at = ?"
                        " WHERE space_id = ? AND session_id = ? AND id != ?"
                        " AND status IN ('pending', 'failed')",
                        (now, space_id, session_id, job_id),
                    )
        return added

    def has_jobs(self, space_id: str) -> bool:
        """这个空间有没有巩固任务（任何状态）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM jobs WHERE space_id = ? LIMIT 1", (space_id,)
            ).fetchone()
        return row is not None

    def has_jobs_of_other_version(self, space_id: str, version: str) -> bool:
        """这个空间有没有按别的流水线版本登记的任务（派生记忆过期，见 Consolidator.stale）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM jobs WHERE space_id = ? AND version != ? LIMIT 1",
                (space_id, version),
            ).fetchone()
        return row is not None

    def job_counts(self, space_id: str) -> dict[str, int]:
        """这个空间每种状态的任务各有几个（走 idx_jobs_space，不解出每个任务）：状态查询只要数目，
        以前读出并 json.loads 每个任务，1 万个任务时 40 毫秒，界面每发一条消息查一次。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM jobs WHERE space_id = ? GROUP BY status", (space_id,)
            ).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def job_summary(self, version: str) -> list[dict[str, Any]]:
        """全部空间的任务概况（cuemem doctor，第八轮体检）：每个有任务的空间一项，按空间 ID 排序，
        有 space_id、counts（状态 → 个数）、other_version（按别的流水线版本登记的任务数）、
        error（失败任务里的一条错误：第一行、最多 200 个字符，错误可能引用模型输出里的用户原文）
        与 recorded（巩固指纹 consolidation:<空间>，没有时 None）。

        一次分组查询扫一遍任务表（命令行里用，不走索引也够快），失败的错误与指纹各一次查询。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT space_id, status, COUNT(*), SUM(version != ?) FROM jobs"
                " GROUP BY space_id, status",
                (version,),
            ).fetchall()
            errors = dict(
                self._conn.execute(
                    "SELECT space_id, MIN(error) FROM jobs WHERE status = 'failed'"
                    " GROUP BY space_id"
                ).fetchall()
            )
            prefix = "consolidation:"
            recorded = {
                str(key)[len(prefix) :]: value
                for key, value in self._conn.execute(
                    "SELECT key, value FROM meta WHERE key >= ? AND key < ?",
                    (prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1)),
                ).fetchall()
            }
        spaces: dict[str, dict[str, Any]] = {}
        for space_id, status, count, other in rows:
            item = spaces.setdefault(
                space_id,
                {"space_id": space_id, "counts": {}, "other_version": 0, "error": None},
            )
            item["counts"][status] = int(count)
            item["other_version"] += int(other or 0)
        for space_id, item in spaces.items():
            error = errors.get(space_id)
            if error:
                item["error"] = (str(error).splitlines() or [""])[0][:200]
            item["recorded"] = recorded.get(space_id)
        return [spaces[key] for key in sorted(spaces)]

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

    def reset_jobs(self, job_ids: Sequence[str]) -> int:
        """把这些失败的任务改回待处理（后台按退避时间重试）。期间被新任务取代的不动。"""
        if not job_ids:
            return 0
        marks = ",".join("?" * len(job_ids))
        with self.transaction() as conn:
            cursor = conn.execute(
                f"UPDATE jobs SET status = 'pending' WHERE id IN ({marks}) AND status = 'failed'",
                list(job_ids),
            )
            return max(cursor.rowcount, 0)

    def restore_jobs(self, jobs: Sequence[Mapping[str, Any]]) -> int:
        """把清空之前的任务状态写回同一任务键、刚登记的待处理任务（会话内容没变）：状态、
        失败次数、报错、统计与更新时间。返回写回的个数。"""
        rows = [
            (
                j["status"],
                j["attempts"],
                j["error"],
                json.dumps(j["stats"], ensure_ascii=False),
                j["updated_at"],
                j["id"],
            )
            for j in jobs
        ]
        restored = 0
        with self.transaction() as conn:
            for row in rows:
                cursor = conn.execute(
                    "UPDATE jobs SET status = ?, attempts = ?, error = ?, stats = ?,"
                    " updated_at = ? WHERE id = ? AND status = 'pending'",
                    row,
                )
                restored += max(cursor.rowcount, 0)
        return restored

    def regrown_sessions(self, space_id: str, version: str) -> set[str]:
        """巩固过之后又长了的会话：有待处理的任务，同一版本又有内容指纹不同的已完成任务。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT p.session_id FROM jobs p JOIN jobs d"
                " ON d.space_id = p.space_id AND d.session_id = p.session_id"
                " AND d.version = p.version"
                " WHERE p.space_id = ? AND p.version = ? AND p.status = 'pending'"
                " AND d.status = 'done' AND d.content_hash != p.content_hash",
                (space_id, version),
            ).fetchall()
        return {r[0] for r in rows}

    def spaces_with_work(self, version: str, max_attempts: int) -> list[str]:
        """有待处理的任务、或失败次数还没到上限的任务的空间（后台线程启动时接着做）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT space_id FROM jobs WHERE version = ?"
                " AND (status = 'pending' OR (status = 'failed' AND attempts < ?))"
                " ORDER BY space_id",
                (version, max_attempts),
            ).fetchall()
        return [r[0] for r in rows]

    # ---------- 空间级操作 ----------

    def delete_space(self, space_id: str) -> None:
        """删掉这个空间的全部记录，连同元信息里写着它的 ID 的：各会话的 next_idx 与巩固配置的
        指纹（consolidation:<空间>，见 Consolidator.stale；没有指纹的空间不算过期）。以前这两种
        都留着，delete_all 之后用户 ID（常常是邮箱）与每次 add 的时间还在库里。进行中会话的标记
        由 MemorySystem.delete_space 清理（本进程里还在进行中的要留着）。"""
        with self.transaction() as conn:
            self._bump([space_id])
            self._turn_epochs[space_id] = self._turn_epochs.get(space_id, 0) + 1
            raw = ("units", "turns", "sessions", "media", "provenance", "turn_meta", "add_keys")
            for table in (*raw, *DERIVED_TABLES):
                conn.execute(f"DELETE FROM {table} WHERE space_id = ?", (space_id,))
            conn.execute("DELETE FROM spaces WHERE id = ?", (space_id,))
            self._drop_idle_idx(conn, space_id)
            conn.execute("DELETE FROM meta WHERE key = ?", (f"consolidation:{space_id}",))
            # 写入计数也不再记着这个空间的会话：删除次数（_turn_epochs）在上面加过，版本不会重复
            for key in [k for k in self._turn_writes if k[0] == space_id]:
                del self._turn_writes[key]

    def space_ids(self) -> list[str]:
        """全部记忆空间的 ID（只读 spaces 表，不像 list_spaces 那样数会话与记忆）。"""
        with self._lock:
            return [row[0] for row in self._conn.execute("SELECT id FROM spaces")]

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

    def scrub(self) -> None:
        """遗忘之后调用：提交，把 WAL 写回数据库文件并截断，被删内容不留在 WAL 里。

        删除时已经覆写（secure_delete，溢出页也一样），写回之后文件里就没有被删的内容了，
        不用 VACUUM。VACUUM 要重写整个文件，库大时一次要几秒，期间这个库的读写全部停下；
        代价是文件不会因为遗忘变小。别的进程正在读这个库时可能截断不了：记一条警告。"""
        with self._lock:
            self._conn.commit()
            busy = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        if busy:
            logger.warning("记忆库的 WAL 没能截断（别的进程正在使用），被删的内容暂时还在 WAL 里")

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _keys_under(conn: sqlite3.Connection, prefix: str) -> list[str]:
    """元信息里以 prefix 开头的键（见 MemoryDB.meta_keys）。"""
    upper = prefix[:-1] + chr(ord(prefix[-1]) + 1)
    rows = conn.execute(
        "SELECT key FROM meta WHERE key >= ? AND key < ? ORDER BY key", (prefix, upper)
    ).fetchall()
    return [r[0] for r in rows]


def _has_session(conn: sqlite3.Connection, space_id: str, session_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sessions WHERE space_id = ? AND id = ?", (space_id, session_id)
    ).fetchone()
    return row is not None


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


def stored_fact(fact: Fact) -> Fact:
    """这条事实写进库之后再读出来的样子（与 get_facts 读到的逐字段相同）。"""
    return _row_to_fact(_fact_row(fact))


def stored_entity(entity: Entity) -> Entity:
    """这个实体写进库之后再读出来的样子（与 get_entities 读到的逐字段相同）。"""
    return _row_to_entity(_entity_row(entity))


def _dt(value: str) -> datetime:
    """读出的时间一律是本机时间、不带时区。写入时已经换算（见 core/models.py），这里照顾之前
    用带时区的时间写进去的行（'...+00:00'）：否则与新写入的行混在一起无法比较。"""
    return naive_local(datetime.fromisoformat(value))


def _turn_order(r: Sequence[Any]) -> tuple[Any, ...]:
    """发言的行按 ORDER BY ts, session_id, idx 的顺序（存的原文，与 SQLite 的比较相同）。"""
    return (r[6], r[2], r[3])


def _row_to_turn(r: Sequence[Any]) -> Turn:
    return Turn(
        id=r[0],
        space_id=r[1],
        session_id=r[2],
        idx=r[3],
        role=r[4],
        text=r[5],
        ts=_dt(r[6]),
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
        ts=_dt(row[5]),
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
        recorded_at=_dt(r[12]),
        invalidated_at=_dt(r[13]) if r[13] else None,
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
        first_seen=_dt(r[6]),
        last_seen=_dt(r[7]),
        mentions=r[8],
    )


def _row_to_media(r: Sequence[Any]) -> MediaItem:
    return MediaItem(
        id=r[0],
        space_id=r[1],
        sha256=r[2],
        path=r[3],
        caption=r[4],
        ocr_text=r[5],
        tags=json.loads(r[6]),
        entity_ids=json.loads(r[7]),
        source_turn_id=r[8],
        ts=_dt(r[9]),
        caption_iso=r[10],
        width=r[11],
        height=r[12],
    )
