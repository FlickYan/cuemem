"""本地 LLM 响应缓存（SQLite）。

供应商、模型、消息、参数完全相同的请求只调用一次 API：既省钱，也让评测结果可以复现。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


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
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
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
            self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(
        self, key: str, model: str, response: dict[str, Any], *, scope: str | None = None
    ) -> None:
        """scope 标记这条缓存属于哪个记忆空间，遗忘时据此清理（见 delete_scope）。"""
        payload = json.dumps(response, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, model, response, scope)"
                " VALUES (?, ?, ?, ?)",
                (key, model, payload, scope),
            )
            self._conn.commit()

    def delete_scope(self, scope: str) -> int:
        """删除某个记忆空间产生的全部缓存条目：回复里可能含有被遗忘的个人信息。"""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM llm_cache WHERE scope = ?", (scope,))
            self._conn.commit()
            return max(cursor.rowcount, 0)

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
