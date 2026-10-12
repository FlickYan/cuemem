"""用量账本：记录每次模型调用的 token 与成本，便于按模型、角色、运行汇总（D-010）。

两种成本口径：
- cost（名义成本）：全部输入按"未命中缓存"的标价计算。同一个实验无论何时运行，成本都一样，便于比较。
- cost_cached：按供应商报告的缓存命中 token 打折后的估算，更接近真实账单（仍未计入错峰折扣）。
命中本地响应缓存的调用（from_cache=True）实际没有花钱，汇总时单独统计。
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from cuemem.config import Price
from cuemem.llm.types import Usage


class UsageRecord(BaseModel):
    ts: datetime
    run_id: str
    kind: Literal["chat", "embedding", "rerank"]
    role: str | None
    provider: str
    model: str
    input_tokens: int
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cost: float
    cost_cached: float
    currency: str
    from_cache: bool = False
    latency_ms: float = 0.0


def chat_cost(price: Price, usage: Usage) -> tuple[float, float]:
    """返回（名义成本，计入供应商缓存折扣后的成本），币种与标价一致。"""
    nominal = (usage.input_tokens * price.input + usage.output_tokens * price.output) / 1e6
    hit_price = price.cached_input if price.cached_input is not None else price.input
    cached = min(usage.cached_input_tokens, usage.input_tokens)
    discounted = (
        (usage.input_tokens - cached) * price.input
        + cached * hit_price
        + usage.output_tokens * price.output
    ) / 1e6
    return nominal, discounted


_COLUMNS = list(UsageRecord.model_fields)


class Ledger:
    def __init__(self, path: Path | None, *, run_id: str | None = None) -> None:
        # run_id 标识一次运行（一次 CLI 调用或一次评测），汇总时可以按它过滤
        self.run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS usage ("
                " ts TEXT NOT NULL, run_id TEXT NOT NULL, kind TEXT NOT NULL, role TEXT,"
                " provider TEXT NOT NULL, model TEXT NOT NULL,"
                " input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,"
                " cached_input_tokens INTEGER NOT NULL, cost REAL NOT NULL,"
                " cost_cached REAL NOT NULL, currency TEXT NOT NULL,"
                " from_cache INTEGER NOT NULL, latency_ms REAL NOT NULL)"
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_run ON usage (run_id)")
            self._conn.commit()

    def record(self, rec: UsageRecord) -> None:
        row = rec.model_dump()
        row["ts"] = rec.ts.isoformat()
        row["from_cache"] = int(rec.from_cache)
        columns = ", ".join(_COLUMNS)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._lock:
            self._conn.execute(
                f"INSERT INTO usage ({columns}) VALUES ({placeholders})",
                [row[c] for c in _COLUMNS],
            )
            self._conn.commit()

    def summary(
        self, *, run_id: str | None = None, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """按（类型、供应商、模型、角色、币种）汇总。spent 为估算的实际花费，不含本地缓存命中。"""
        conditions: list[str] = []
        args: list[str] = []
        if run_id is not None:
            conditions.append("run_id = ?")
            args.append(run_id)
        if since is not None:
            conditions.append("ts >= ?")
            args.append(since.isoformat())
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = (
            "SELECT kind, provider, model, role, currency, COUNT(*) AS calls,"
            " SUM(from_cache) AS cache_hits, SUM(input_tokens) AS input_tokens,"
            " SUM(output_tokens) AS output_tokens, SUM(cached_input_tokens) AS cached_input_tokens,"
            " SUM(cost) AS nominal_cost,"
            " SUM(CASE WHEN from_cache = 1 THEN 0 ELSE cost_cached END) AS spent"
            f" FROM usage {where}"
            " GROUP BY kind, provider, model, role, currency ORDER BY kind, provider, model, role"
        )
        with self._lock:
            cursor = self._conn.execute(sql, args)
            names = [d[0] for d in cursor.description]
            return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
