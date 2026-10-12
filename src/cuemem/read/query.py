"""问题分析（不调用 LLM）：时间表达与问题类型。

- **时间表达**：以**提问时间**为参照（不是系统当前时间），把"两周前""上周六""in May"
  "去年"换算成绝对时间区间。只做规则能确定的部分；解析不了就不给区间，宁缺毋滥。
- **问题类型**：问"第一次 / 最早"时按时间正序看、关掉新近性；问"现在 / 最近一次"时新的优先；
  问"以前 / 之前"时要带上被取代的旧事实；"一共几次 / 哪些"是聚合题，需要更全的证据。

这些信号用于 L1 的重排与打包（read/l1.py），也是 N7 检索控制器的特征。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

WEEKDAYS_EN = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
WEEKDAYS_ZH = "一二三四五六日"
MONTHS_EN = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "a couple of": 2,
    "a few": 3,
    "several": 3,
    "一": 1,
    "两": 2,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "几": 3,
}
# 每种单位"n 个单位之前"的容差：越粗的单位，说话人越不精确
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}
TOLERANCE_DAYS = {"day": 1, "week": 3, "month": 10, "year": 60}
FIXED_HOLIDAYS = {
    "valentine's day": (2, 14),
    "valentines day": (2, 14),
    "christmas eve": (12, 24),
    "christmas": (12, 25),
    "new year's eve": (12, 31),
    "new year's day": (1, 1),
    "halloween": (10, 31),
    "independence day": (7, 4),
    "情人节": (2, 14),
    "圣诞节": (12, 25),
    "平安夜": (12, 24),
}

_NUM = r"(\d+|a couple of|a few|several|an?|one|two|three|four|five|six|seven|eight|nine|ten)"
_ZH_NUM = r"(\d+|[一两二三四五六七八九十几])"
_UNIT_ZH = {"天": "day", "日": "day", "周": "week", "星期": "week", "个星期": "week",
            "个月": "month", "月": "month", "年": "year"}  # fmt: skip

FIRST_WORDS = re.compile(
    r"\b(first|earliest|initially|originally|at first|start(ed)? (using|doing))\b"
    r"|第一次|最早|最初|一开始",
    re.I,
)
LATEST_WORDS = re.compile(
    r"\b(current(ly)?|now|latest|most recent(ly)?|these days|nowadays|still)\b"
    r"|现在|目前|最近一次|如今|还在",
    re.I,
)
HISTORY_WORDS = re.compile(
    r"\b(previous(ly)?|before|used to|formerly|originally|old|prior)\b|之前|以前|原来|曾经|过去",
    re.I,
)
AGGREGATION_WORDS = re.compile(
    r"\b(how many|how much|total|in total|all (the )?|each|every|list|count|sum|combined"
    r"|altogether)\b"
    r"|多少|几次|几个|几种|总共|一共|所有|哪些|分别",
    re.I,
)
TEMPORAL_WORDS = re.compile(
    r"\b(when|what (date|day|time)|how long|how many (days|weeks|months|years)|ago|since"
    r"|before|after|order|first|last|earlier|later)\b|什么时候|哪天|几号|多久|多长时间|几天|之前|之后|先后",
    re.I,
)


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime
    expr: str  # 匹配到的原文

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end

    @property
    def days(self) -> float:
        return (self.end - self.start).total_seconds() / 86400


@dataclass(frozen=True)
class QueryInfo:
    text: str
    question_time: datetime
    time_range: TimeRange | None
    wants_first: bool
    wants_latest: bool
    wants_history: bool
    aggregation: bool
    temporal: bool


def analyze(question: str, question_time: datetime) -> QueryInfo:
    return QueryInfo(
        text=question,
        question_time=question_time,
        time_range=parse_time(question, question_time),
        wants_first=bool(FIRST_WORDS.search(question)),
        wants_latest=bool(LATEST_WORDS.search(question)),
        wants_history=bool(HISTORY_WORDS.search(question)),
        aggregation=bool(AGGREGATION_WORDS.search(question)),
        temporal=bool(TEMPORAL_WORDS.search(question)),
    )


def parse_time(text: str, now: datetime) -> TimeRange | None:
    """把时间表达换算成以 now（提问时间）为参照的区间 [start, end)；解析不了返回 None。"""
    lowered = text.lower()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for parser in _PARSERS:
        found = parser(lowered, now, today)
        if found is not None:
            return found
    return None


# ---------- 各类表达 ----------


def _day(start: datetime, expr: str, days: int = 1) -> TimeRange:
    return TimeRange(start, start + timedelta(days=days), expr)


def _relative_days(text: str, now: datetime, today: datetime) -> TimeRange | None:
    # 只收指向过去的表达："今天 / 今晚"在提问里通常指当下（"今晚看什么电影"），不是检索线索
    table = {
        "the day before yesterday": 2,
        "yesterday": 1,
        "last night": 1,
        "前天": 2,
        "昨天": 1,
        "昨晚": 1,
    }
    for expr, days in table.items():
        if expr in text:
            return _day(today - timedelta(days=days), expr)
    return None


def _ago(text: str, now: datetime, today: datetime) -> TimeRange | None:
    match = re.search(rf"\b{_NUM} (day|week|month|year)s? ago\b", text)
    if match:
        n, unit = NUMBERS.get(match.group(1)) or int(match.group(1)), match.group(2)
    else:
        zh = re.search(rf"{_ZH_NUM}(个星期|星期|个月|天|周|月|年)(以)?前", text)
        if zh is None:
            return None
        raw = zh.group(1)
        n = int(raw) if raw.isdigit() else NUMBERS[raw]
        unit = _UNIT_ZH[zh.group(2)]
        match = zh
    center = today - timedelta(days=n * UNIT_DAYS[unit])
    slack = timedelta(days=TOLERANCE_DAYS[unit])
    return TimeRange(center - slack, center + slack + timedelta(days=1), match.group(0))


def _past_window(text: str, now: datetime, today: datetime) -> TimeRange | None:
    match = re.search(rf"\b(?:past|last|previous) {_NUM} (day|week|month|year)s?\b", text)
    if match:
        n, unit = NUMBERS.get(match.group(1)) or int(match.group(1)), match.group(2)
    else:
        zh = re.search(rf"(过去|最近|近){_ZH_NUM}(个星期|星期|个月|天|周|月|年)", text)
        if zh is None:
            return None
        raw = zh.group(2)
        n = int(raw) if raw.isdigit() else NUMBERS[raw]
        unit = _UNIT_ZH[zh.group(3)]
        match = zh
    return TimeRange(
        today - timedelta(days=n * UNIT_DAYS[unit]), now + timedelta(seconds=1), match.group(0)
    )


def _last_weekday(text: str, now: datetime, today: datetime) -> TimeRange | None:
    match = re.search(
        r"\blast (monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b", text
    )
    zh = re.search(r"上(周|个?星期)([一二三四五六日天])", text)
    if match is None and zh is None:
        return None
    if match is not None:
        name = match.group(1)
        expr = match.group(0)
        target = 5 if name == "weekend" else WEEKDAYS_EN.index(name)
    else:
        assert zh is not None
        name = zh.group(2)
        expr = zh.group(0)
        target = 6 if name in "日天" else WEEKDAYS_ZH.index(name)
    back = (today.weekday() - target) % 7 or 7  # 最近一个已经过去的那天
    start = today - timedelta(days=back)
    return _day(start, expr, days=2 if match is not None and match.group(1) == "weekend" else 1)


def _calendar(text: str, now: datetime, today: datetime) -> TimeRange | None:
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    previous_month = (month_start - timedelta(days=1)).replace(day=1)
    table = [
        (r"\blast week\b|上周|上个星期|上星期", week_start - timedelta(days=7), week_start),
        (r"\bthis week\b|这周|本周|这个星期", week_start, now + timedelta(seconds=1)),
        (r"\blast month\b|上个月|上月", previous_month, month_start),
        (r"\bthis month\b|这个月|本月", month_start, now + timedelta(seconds=1)),
        (
            r"\blast year\b|去年",
            today.replace(year=today.year - 1, month=1, day=1),
            today.replace(month=1, day=1),
        ),
        (r"\bthis year\b|今年", today.replace(month=1, day=1), now + timedelta(seconds=1)),
    ]
    for pattern, start, end in table:
        match = re.search(pattern, text)
        if match:
            return TimeRange(start, end, match.group(0))
    return None


def _month_day(text: str, now: datetime, today: datetime) -> TimeRange | None:
    months = "|".join(MONTHS_EN)
    match = re.search(rf"\b({months}) (\d{{1,2}})(?:st|nd|rd|th)?(?:,? (\d{{4}}))?\b", text)
    zh = re.search(r"(\d{1,2})月(\d{1,2})[日号]", text)
    if match is not None:
        month, dom = MONTHS_EN.index(match.group(1)) + 1, int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        expr = match.group(0)
    elif zh is not None:
        month, dom, year, expr = int(zh.group(1)), int(zh.group(2)), None, zh.group(0)
    else:
        for name, (month, dom) in FIXED_HOLIDAYS.items():
            if name in text:
                return _resolve_day(month, dom, None, today, name)
        return None
    return _resolve_day(month, dom, year, today, expr)


def _resolve_day(
    month: int, dom: int, year: int | None, today: datetime, expr: str
) -> TimeRange | None:
    try:
        moment = today.replace(year=year or today.year, month=month, day=dom)
    except ValueError:
        return None
    if year is None and moment > today:  # 没说年份：取最近一个已经过去的
        moment = moment.replace(year=moment.year - 1)
    return _day(moment, expr)


def _month(text: str, now: datetime, today: datetime) -> TimeRange | None:
    months = "|".join(MONTHS_EN)
    # "may" 也是情态动词：要求前面有 in / during / since 之类的介词
    match = re.search(
        rf"\b(?:in|during|since|of|early|late|mid-?)\s*({months})(?:,? (\d{{4}}))?\b", text
    )
    zh = re.search(r"(?<![\d月])(\d{1,2})月份?(?![\d日号])", text)
    if match is not None:
        month = MONTHS_EN.index(match.group(1)) + 1
        year = int(match.group(2)) if match.group(2) else None
        expr = match.group(0)
    elif zh is not None and 1 <= int(zh.group(1)) <= 12:
        month, year, expr = int(zh.group(1)), None, zh.group(0)
    else:
        return None
    start = today.replace(year=year or today.year, month=month, day=1)
    if year is None and start > today:
        start = start.replace(year=start.year - 1)
    end = (start + timedelta(days=32)).replace(day=1)
    return TimeRange(start, end, expr)


def _year(text: str, now: datetime, today: datetime) -> TimeRange | None:
    match = re.search(r"\b(?:in|during|since) ((?:19|20)\d{2})\b|((?:19|20)\d{2})年", text)
    if match is None:
        return None
    year = int(match.group(1) or match.group(2))
    return TimeRange(datetime(year, 1, 1), datetime(year + 1, 1, 1), match.group(0))


# 越具体的表达越先匹配
_PARSERS = (
    _ago,
    _last_weekday,
    _month_day,
    _relative_days,
    _past_window,
    _calendar,
    _month,
    _year,
)
