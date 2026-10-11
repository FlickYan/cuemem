"""时间工具。基准数据里的时间大多不带时区，统一视为"对话时间"：无时区即按 UTC 处理。"""

from __future__ import annotations

from datetime import UTC, datetime


def to_epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()
