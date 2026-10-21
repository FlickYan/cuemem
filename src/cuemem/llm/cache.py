"""本地 LLM 响应缓存（SQLite）。

供应商、模型、消息、参数完全相同的请求只调用一次 API：既省钱，也让评测结果可以复现。
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 正在记录命中的缓存键（见 ResponseCache.recording）。放在上下文里：巩固在线程池里抽取时复制
# 调用方的上下文，那些线程里的命中也记得到
_HITS: ContextVar[set[str] | None] = ContextVar("cuemem_llm_cache_hits", default=None)


def request_key(request: dict[str, Any]) -> str:
    """请求的规范化哈希：字段顺序不同但内容相同的请求得到同一个键。"""
    canonical = json.dumps(
        request, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        # 几个评测进程共用一份缓存：写锁多等一会儿（默认 5 秒），免得忙时报 "database is locked"
        self._conn = sqlite3.connect(
            str(path) if path else ":memory:", check_same_thread=False, timeout=60.0
        )
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA secure_delete=ON")  # 遗忘时删除的条目不留在空闲页里
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                " key TEXT PRIMARY KEY, model TEXT NOT NULL, response TEXT NOT NULL,"
                " created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, scope TEXT)"
            )
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(llm_cache)")}
            if "scope" not in columns:  # N3 之前建的缓存库
                self._conn.execute("ALTER TABLE llm_cache ADD COLUMN scope TEXT")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_llm_cache_scope ON llm_cache (scope)"
            )
            # 遗忘时标记、待核对的条目（见 hold、settle）
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache_held ("
                " key TEXT PRIMARY KEY, scope TEXT NOT NULL)"
            )
            self._conn.commit()
        self._drop_held()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
            hits = _HITS.get()
            if row is not None and hits is not None:
                hits.add(key)
        return json.loads(row[0]) if row else None

    @contextmanager
    def recording(self, hits: set[str] | None = None) -> Iterator[set[str]]:
        """把 with 块里命中过的缓存键记进 hits（包括复制了这个上下文的线程里的命中）。遗忘重放时
        用，见 settle。"""
        hits = set() if hits is None else hits
        token = _HITS.set(hits)
        try:
            yield hits
        finally:
            _HITS.reset(token)

    def hold(self, scope: str) -> int:
        """遗忘开始时标记这个归属下的全部条目，等重放之后核对（settle）：重放时原样用到的留下，
        其余删掉。标记写在库里：进程在核对之前退出时，下次打开缓存就把它们全部删掉。返回条数。"""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO llm_cache_held (key, scope)"
                " SELECT key, scope FROM llm_cache WHERE scope = ?",
                (scope,),
            )
            self._conn.commit()
            return max(cursor.rowcount, 0)

    def settle(self, scope: str, kept: Collection[str]) -> int:
        """核对 hold 标记的条目：键在 kept 里的（重放时命中过，见 recording）留下，其余删掉，
        删除时覆写、删了就截断 WAL。返回删掉的条数。"""
        with self._lock:
            held = self._conn.execute(
                "SELECT key FROM llm_cache_held WHERE scope = ?", (scope,)
            ).fetchall()
            doomed = [key for (key,) in held if key not in kept]
            deleted = 0
            for start in range(0, len(doomed), 500):
                chunk = doomed[start : start + 500]
                cursor = self._conn.execute(
                    f"DELETE FROM llm_cache WHERE key IN ({', '.join('?' * len(chunk))})", chunk
                )
                deleted += max(cursor.rowcount, 0)
            self._conn.execute("DELETE FROM llm_cache_held WHERE scope = ?", (scope,))
            self._conn.commit()  # 删条目与去掉标记一起提交：中途退出时标记还在
            if deleted:
                self._checkpoint()
            return deleted

    def _drop_held(self) -> None:
        """删掉还标记着的条目：上次遗忘重放到一半时进程退出了，没来得及核对。"""
        with self._lock:
            if self._conn.execute("SELECT 1 FROM llm_cache_held LIMIT 1").fetchone() is None:
                return
            cursor = self._conn.execute(
                "DELETE FROM llm_cache WHERE key IN (SELECT key FROM llm_cache_held)"
            )
            self._conn.execute("DELETE FROM llm_cache_held")
            self._conn.commit()
            self._checkpoint()
        logger.warning(
            "LLM 缓存里有上次遗忘没核对完的 %d 个条目（进程中途退出），已删除", cursor.rowcount
        )

    def put(
        self, key: str, model: str, response: dict[str, Any], *, scope: str | None = None
    ) -> None:
        """scope 标记这条缓存属于哪个记忆空间，遗忘时据此清理（见 delete_scope、hold）。"""
        payload = json.dumps(response, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, response, scope)"
                " VALUES (?, ?, ?, ?)",
                (key, model, payload, scope),
            )
            self._conn.commit()

    def delete_scope(self, scope: str) -> int:
        """删除某个记忆空间产生的全部缓存条目：回复里可能含有被遗忘的个人信息。

        删除时覆写（secure_delete），删了东西就把 WAL 写回并截断：否则被删的回复留在 WAL
        里，直到进程退出。"""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM llm_cache WHERE scope = ?", (scope,))
            self._conn.commit()
            deleted = max(cursor.rowcount, 0)
            if deleted:
                self._checkpoint()
            return deleted

    def responses(self, scope: str | None = None) -> list[str]:
        """返回缓存中的原始回复文本（测试与遗忘检查用）。"""
        sql = "SELECT response FROM llm_cache"
        args: tuple[Any, ...] = ()
        if scope is not None:
            sql += " WHERE scope = ?"
            args = (scope,)
        with self._lock:
            return [r[0] for r in self._conn.execute(sql, args).fetchall()]

    def __len__(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _checkpoint(self) -> None:
        """把 WAL 写回数据库并截断（调用方持有 self._lock）。别的进程（评测）正在读写这份缓存时
        可能做不完，被删的条目暂时还在 WAL 里：记一条警告。"""
        busy = self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        if busy:
            logger.warning("LLM 缓存的 WAL 没能截断（别的进程正在使用），被删的条目暂时还在 WAL 里")
