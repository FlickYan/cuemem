"""用量账本：记录每次模型调用的 token 与成本，便于按模型、角色、运行汇总（D-010）。

两种成本口径：
- cost（名义成本）：全部输入按"未命中缓存"的标价计算。同一个实验无论何时运行，成本都一样，便于比较。
- cost_cached：按供应商报告的缓存命中 token 打折后的估算，非高峰时段的调用再乘上错峰系数
  （LLMClient.account 记账前已经乘好），更接近真实账单。
命中本地响应缓存的调用（from_cache=True）实际没有花钱，汇总时单独统计。

另有一张 key_usage 表（第十轮体检 Q19）：记忆代理给写了 max_daily_tokens 的令牌按（令牌的名字，
服务器本地的日期）累计转发的 token 数，每日预算据此算。只在第一次记这样的用量时建
（add_key_usage）：评测与没有这种令牌的部署的账本文件里没有它，usage 表与它的写法不变。
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
from cuemem.store.files import spare_write


class UsageRecord(BaseModel):
    ts: datetime
    run_id: str
    kind: Literal["chat", "embedding", "rerank", "image"]  # image：生图（N8 演示素材）
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
    def __init__(
        self, path: Path | None, *, run_id: str | None = None, best_effort: bool = False
    ) -> None:
        """best_effort：磁盘满了、只读时 record 记一条警告、不抛出（见 store.files.spare_write）：
        已经付费的回复照样返回，只是少记一笔。只有 MemorySystem.from_settings 打开；评测与脚本
        用默认的 False。"""
        self._path = path
        self.best_effort = best_effort
        self.failing = False  # 上一次 record 因为磁盘写不进去（接连失败只记一次警告）
        # run_id 标识一次运行（一次 CLI 调用或一次评测），汇总时可以按它过滤
        self.run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        # 几个评测进程共用一个账本：写锁默认只等 5 秒，忙时会报 "database is locked"，这里等 60 秒
        self._conn = sqlite3.connect(
            str(path) if path else ":memory:", check_same_thread=False, timeout=60.0
        )
        self._lock = threading.Lock()
        with self._lock:
            try:
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
            except BaseException:  # 打开时出错（文件损坏、不是 SQLite 数据库）：不留句柄
                self._conn.close()
                raise

    def record(self, rec: UsageRecord) -> None:
        row = rec.model_dump()
        row["ts"] = rec.ts.isoformat()
        row["from_cache"] = int(rec.from_cache)
        columns = ", ".join(_COLUMNS)
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self._lock:
            try:
                self._conn.execute(
                    f"INSERT INTO usage ({columns}) VALUES ({placeholders})",
                    [row[c] for c in _COLUMNS],
                )
                self._conn.commit()
            except sqlite3.OperationalError as exc:
                spare_write(
                    self._conn,
                    exc,
                    best_effort=self.best_effort,
                    failing=self.failing,
                    what="用量账本",
                    path=self._path,
                )
                self.failing = True
            else:
                self.failing = False

    def add_key_usage(self, key: str, day: str, tokens: int, *, unmetered: bool = False) -> None:
        """记忆代理替令牌 key 在 day（服务器本地的日期，ISO 写法）转发的一次补全：tokens（输入加
        输出）加进这一天的合计，calls 加一；unmetered：上游没报用量的一次（tokens 是 0）。表在
        第一次调用时才建。磁盘写不进去时与 record 相同（best_effort 时记警告、不抛出）。"""
        with self._lock:
            try:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS key_usage ("
                    " key TEXT NOT NULL, day TEXT NOT NULL, tokens INTEGER NOT NULL,"
                    " calls INTEGER NOT NULL, unmetered INTEGER NOT NULL,"
                    " PRIMARY KEY (key, day))"
                )
                self._conn.execute(
                    "INSERT INTO key_usage (key, day, tokens, calls, unmetered)"
                    " VALUES (?, ?, ?, 1, ?)"
                    " ON CONFLICT (key, day) DO UPDATE SET tokens = tokens + excluded.tokens,"
                    " calls = calls + 1, unmetered = unmetered + excluded.unmetered",
                    (key, day, tokens, int(unmetered)),
                )
                self._conn.commit()
            except sqlite3.OperationalError as exc:
                spare_write(
                    self._conn,
                    exc,
                    best_effort=self.best_effort,
                    failing=self.failing,
                    what="用量账本",
                    path=self._path,
                )
                self.failing = True
            else:
                self.failing = False

    def key_tokens(self, key: str, day: str) -> int:
        """令牌 key 在 day 已经记下的 token 数；没有这张表、没有这一行或读不出来时 0（不建表）。"""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT tokens FROM key_usage WHERE key = ? AND day = ?", (key, day)
                ).fetchone()
            except sqlite3.OperationalError:
                return 0
        return int(row[0]) if row else 0

    def key_usage(self, day: str) -> list[dict[str, Any]]:
        """day 这一天各个令牌的用量（key、tokens、calls、unmetered），按名字排；没有这张表时空
        （不建表）。cuemem usage --by-key 用。"""
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT key, tokens, calls, unmetered FROM key_usage WHERE day = ?"
                    " ORDER BY key",
                    (day,),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        names = ("key", "tokens", "calls", "unmetered")
        return [dict(zip(names, row, strict=True)) for row in rows]

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
