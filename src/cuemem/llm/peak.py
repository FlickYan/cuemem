"""供应商的分时价格：判断当前是否处于高峰时段，以及高峰何时结束。

DeepSeek 工作日 09:00–12:00、14:00–18:00（北京时间）为高峰，价格是其他时段的两倍。
配置里的标价按高峰计（D-010，名义成本保证实验可比）；非高峰时段的实际花费 = 标价 × 折扣系数。
评测这类批量任务在高峰时段默认暂停，等高峰结束再继续调用（命中缓存的请求不受影响）。
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from cuemem.config import PeakHours


def _windows(cfg: PeakHours) -> list[tuple[time, time]]:
    out = []
    for window in cfg.windows:
        start, end = window.split("-")
        out.append((time.fromisoformat(start.strip()), time.fromisoformat(end.strip())))
    return out


def local_now(cfg: PeakHours, now: datetime | None = None) -> datetime:
    """换算到供应商定价所用的时区。"""
    tz = ZoneInfo(cfg.timezone)
    if now is None:
        return datetime.now(tz)
    if now.tzinfo is None:
        now = now.astimezone()  # 无时区的时间视为本机时间
    return now.astimezone(tz)


def peak_end(cfg: PeakHours, now: datetime | None = None) -> datetime | None:
    """如果 now 处于高峰时段，返回这一段高峰的结束时间；否则返回 None。"""
    local = local_now(cfg, now)
    if local.weekday() not in cfg.weekdays:
        return None
    for start, end in _windows(cfg):
        if start <= local.time() < end:
            return local.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)
    return None


def in_peak(cfg: PeakHours, now: datetime | None = None) -> bool:
    return peak_end(cfg, now) is not None


def next_peak_start(cfg: PeakHours, now: datetime | None = None) -> datetime:
    """下一段高峰的开始时间（now 处于高峰时返回当前这段的开始时间之后的下一段）。"""
    local = local_now(cfg, now)
    for days in range(8):
        day = (local + timedelta(days=days)).date()
        if day.weekday() not in cfg.weekdays:
            continue
        for start, _ in _windows(cfg):
            moment = datetime.combine(day, start, tzinfo=local.tzinfo)
            if moment > local:
                return moment
    raise ValueError("高峰时段配置为空")


def describe(cfg: PeakHours) -> str:
    names = "一二三四五六日"
    days = "、".join(f"周{names[d]}" for d in sorted(cfg.weekdays))
    if sorted(cfg.weekdays) == [0, 1, 2, 3, 4]:
        days = "工作日"
    return f"{days} {'、'.join(cfg.windows)}（{cfg.timezone}）"
