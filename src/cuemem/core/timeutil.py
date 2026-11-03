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
from datetime import UTC, date, datetime, time, timedelta

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


def normalize_expiry(value: object) -> str | None:
    """过期时间（SDK 的 expiration_date、导入文件里的 expires_at）→ 存下的字符串：只给日期时是
    "YYYY-MM-DD"，到那天结束为止（与 Mem0 相同）；带时间的换算成本机时间。报错是英文：
    调用方是 SDK 与导入。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return naive_local(value).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        try:
            if len(text) == 10:
                return date.fromisoformat(text).isoformat()
            return naive_local(datetime.fromisoformat(text)).isoformat()
        except ValueError:
            pass
    raise ValueError(
        f"Invalid expiration date {value!r}: give an ISO date ('2026-12-31') or a datetime."
    )


def expiry_instant(value: str) -> datetime:
    """存下的过期时间（normalize_expiry 的结果）→ 过期的时刻：只有日期的，第二天 0 点过期。"""
    if len(value) == 10:
        return datetime.combine(date.fromisoformat(value), time()) + timedelta(days=1)
    return datetime.fromisoformat(value)
