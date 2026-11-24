"""用量账本里按令牌的用量（key_usage，第十轮体检 Q19）：只在记这种用量时才有这张表。"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from cuemem.llm.ledger import Ledger, UsageRecord


def tables(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {name for (name,) in conn.execute("SELECT name FROM sqlite_master")}
    finally:
        conn.close()


def columns(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(usage)")]
    finally:
        conn.close()


def test_recording_usage_never_creates_the_key_table(tmp_path: Path) -> None:
    """打开账本、记一行、汇总、读按令牌的用量都不建 key_usage：评测与没有每日预算的部署的账本
    文件不变（usage 表的列也不变）。"""
    path = tmp_path / "ledger.sqlite"
    ledger = Ledger(path)
    ledger.record(
        UsageRecord(
            ts=datetime(2026, 11, 24, 9, 0),
            run_id=ledger.run_id,
            kind="chat",
            role="proxy",
            provider="dashscope",
            model="qwen3.8-flash",
            input_tokens=120,
            output_tokens=30,
            cost=0.1,
            cost_cached=0.1,
            currency="CNY",
        )
    )
    assert ledger.summary()[0]["calls"] == 1
    assert ledger.key_tokens("metered", "2026-11-24") == 0
    assert ledger.key_usage("2026-11-24") == []
    ledger.close()
    assert "usage" in tables(path) and "key_usage" not in tables(path)
    assert columns(path) == list(UsageRecord.model_fields)


def test_key_usage_adds_up_per_key_and_day(tmp_path: Path) -> None:
    """add_key_usage 按（令牌，日期）累计 token 数与次数；另一个连接（另一个进程、重启之后）
    读到同样的数。"""
    path = tmp_path / "ledger.sqlite"
    ledger = Ledger(path)
    ledger.add_key_usage("metered", "2026-11-24", 150)
    ledger.add_key_usage("metered", "2026-11-24", 13)
    ledger.add_key_usage("metered", "2026-11-24", 0, unmetered=True)
    ledger.add_key_usage("metered", "2026-11-25", 7)
    ledger.add_key_usage("other", "2026-11-24", 5)
    assert ledger.key_tokens("metered", "2026-11-24") == 163
    assert ledger.key_tokens("metered", "2026-11-26") == 0
    other = Ledger(path)
    assert other.key_tokens("metered", "2026-11-24") == 163
    assert other.key_usage("2026-11-24") == [
        {"key": "metered", "tokens": 163, "calls": 3, "unmetered": 1},
        {"key": "other", "tokens": 5, "calls": 1, "unmetered": 0},
    ]
    other.close()
    ledger.close()
    assert columns(path) == list(UsageRecord.model_fields)  # usage 表不动
