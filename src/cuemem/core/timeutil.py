"""时间工具。

不带时区的时间一律是本机的墙上时间，也就是"对话时间"：基准数据里的时间不带时区，
datetime.now() 与 datetime.fromtimestamp() 的缺省值也是本机时间。带时区的时间（ISO 字符串
里的 'Z'、'+02:00'）在写入与查询的入口换算成本机时间、去掉时区（naive_local）：两种时间
混在一起无法比较大小。to_epoch 把不带时区的时间按 UTC 换算，只是给向量库的载荷算一个数值，
没有按它过滤的地方。
事实的事件时间用"部分日期"字符串表示（YYYY / YYYY-MM / YYYY-MM-DD）：对话里常常只说
"五月""去年"，硬补成某一天会制造虚假的精度。同一格式的 ISO 字符串可以直接按字典序比较。
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

_PARTIAL_DATE = re.compile(r"^(\d{4})(?:[-/.](\d{1,2})(?:[-/.](\d{1,2}))?)?")


def naive_local(dt: datetime) -> datetime:
    """带时区的时间 → 本机的墙上时间（不带时区）；不带时区的原样返回。"""
    return dt if dt.tzinfo is None else dt.astimezone().replace(tzinfo=None)


def to_epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def day(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def normalize_partial_date(value: object) -> str | None:
    """把模型给出的日期规范成 YYYY / YYYY-MM / YYYY-MM-DD；无法识别或不合法时返回 None。

    也接受 "2023/05/07"、"2023-05-07T10:00" 这类写法（只取日期部分）。
    """
    if not isinstance(value, str):
        return None
    match = _PARTIAL_DATE.match(value.strip())
    if match is None:
        return None
    year, month, dom = match.groups()
    try:
        if dom is not None and month is not None:
            return datetime(int(year), int(month), int(dom)).strftime("%Y-%m-%d")
        if month is not None:
            return datetime(int(year), int(month), 1).strftime("%Y-%m")
        return f"{int(year):04d}"
    except ValueError:  # 2023-02-30 之类
        return None


def partial_date_start(value: str) -> datetime:
    """部分日期所指时间段的起点：'2023-05' → 2023-05-01 00:00。"""
    parts = [int(p) for p in value.split("-")]
    while len(parts) < 3:
        parts.append(1)
    return datetime(parts[0], parts[1], parts[2])
