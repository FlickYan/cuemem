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
    "元旦": (1, 1),
}

_NUM = r"(\d+|a couple of|a few|several|an?|one|two|three|four|five|six|seven|eight|nine|ten)"
# 中文数量：最多三位数字，或"十二 / 二十 / 二十三"这样的两位数，或单个数字。
# 前面的两个 (?<!…) 有两个用处：一长串数字只从第一位试一次（匹配时间保持线性）；
# 不从"一百二十""2023""1.5"的中间截一段出来。"2023年以前""3000年前"因此不再当成"N 年前"
_ZH_NUM = (
    r"(?<![\d.十百千万])(?<![百千万][一二两三四五六七八九])"
    r"(\d{1,3}|[二三四五六七八九]?十[一二三四五六七八九]?|[一两二三四五六七八九几])"
)
_ZH_MONTH = r"(1[0-2]|0?[1-9]|十[一二]?|[一二三四五六七八九])"
_ZH_DOM = r"(\d{1,2}|[二三]?十[一二三四五六七八九]?|[一二三四五六七八九])"
# 月、日、节日前面的年份：分组 1 是四位数字，分组 2 是"今年 / 去年 / 前年"
_ZH_YEAR = r"(?:((?:19|20)\d{2})年|(今年|去年|(?<![以之大从目提])前年))"
_ZH_DIGITS = "零一二三四五六七八九"
_HAN = re.compile(r"[\u4e00-\u9fff]")
# 只收线索词里用到的字，不做完整的繁简转换
_ZH_FOLD = str.maketrans(
    "週個兩幾過這聖誕節號禮後麼時長間總現還來經開種數較別從萬氣輕紀齡級０１２３４５６７８９\u3000",
    "周个两几过这圣诞节号礼后么时长间总现还来经开种数较别从万气轻纪龄级0123456789 ",
)
_ZH_GAP = re.compile(r"(?<=\d)\s+(?=[年月日号天周个星])|(?<=[年月])\s+(?=\d)")
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


# 规则版控制器（B6r，N5）的三条线索：先后、时长、比较；N7 起也是探针的查询线索特征
_NOT_ORDER = r"(?!\s+(?:week|month|year|weekend|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)"  # noqa: E501
ORDER_WORDS = re.compile(
    rf"\b(first|earliest|last{_NOT_ORDER}|latest|most recent(?:ly)?|before|after|order"
    r"|sequence)\b|第一次|最早|最后一次|最近一次|先后|之前|之后",
    re.I,
)
DURATION_WORDS = re.compile(
    r"\bhow (?:long|many (?:days|weeks|months|years|hours|minutes))\b"
    r"|多久|多长时间|几天|几周|几个月|几年",
    re.I,
)
COMPARE_WORDS = re.compile(
    r"\b(more|most|less|least|fewer|larger|largest|bigger|biggest|compared?|difference)\b"
    r"|更多|最多|最少|比较",
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


def cue_text(text: str) -> str:
    """找线索词、解析时间表达时用的文本；不含汉字的文本原样返回。

    含汉字时把线索词里用到的繁体字与全角数字换成简体与半角，"礼拜"换成"星期"，
    去掉"5 月 3 日""3 天前"里数字与单位之间的空格。只用于匹配，不改写要检索的问题。
    """
    if not _HAN.search(text):
        return text
    return _ZH_GAP.sub("", text.translate(_ZH_FOLD).replace("礼拜", "星期"))


def analyze(question: str, question_time: datetime) -> QueryInfo:
    cue = cue_text(question)
    return QueryInfo(
        text=question,
        question_time=question_time,
        time_range=parse_time(question, question_time),
        wants_first=bool(FIRST_WORDS.search(cue)),
        wants_latest=bool(LATEST_WORDS.search(cue)),
        wants_history=bool(HISTORY_WORDS.search(cue)),
        aggregation=bool(AGGREGATION_WORDS.search(cue)),
        temporal=bool(TEMPORAL_WORDS.search(cue)),
    )


def parse_time(text: str, now: datetime) -> TimeRange | None:
    """把时间表达换算成以 now（提问时间）为参照的区间 [start, end)；解析不了返回 None。"""
    lowered = cue_text(text.lower())
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for parser in _PARSERS:
        found = parser(lowered, now, today)
        if found is not None:
            return found
    return None


# ---------- 各类表达 ----------


def _zh_int(raw: str) -> int:
    if raw.isdigit():
        return int(raw)
    if "十" in raw:
        tens, _, ones = raw.partition("十")
        return (_ZH_DIGITS.index(tens) if tens else 1) * 10 + (
            _ZH_DIGITS.index(ones) if ones else 0
        )
    return NUMBERS[raw]


def _zh_year(digits: str | None, word: str | None, today: datetime) -> int | None:
    if digits:
        return int(digits)
    if word:
        return today.year - ("今年", "去年", "前年").index(word)
    return None


def _day(start: datetime, expr: str, days: int = 1) -> TimeRange:
    return TimeRange(start, start + timedelta(days=days), expr)


def _relative_days(text: str, now: datetime, today: datetime) -> TimeRange | None:
    # 只收指向过去的表达："今天 / 今晚"在提问里通常指当下（"今晚看什么电影"），不是检索线索
    table = {
        "the day before yesterday": 2,
        "yesterday": 1,
        "last night": 1,
        "昨天": 1,
        "昨晚": 1,
    }
    for expr, days in table.items():
        if expr in text:
            return _day(today - timedelta(days=days), expr)
    # "以前天天""目前天气""提前天数"里的"前天"不是日期
    zh = re.search(r"(?<![以之从目提])(大?)前天", text)
    if zh is not None:
        return _day(today - timedelta(days=3 if zh.group(1) else 2), zh.group(0))
    return None


def _ago(text: str, now: datetime, today: datetime) -> TimeRange | None:
    match = re.search(rf"\b{_NUM} (day|week|month|year)s? ago\b", text)
    if match:
        n, unit = NUMBERS.get(match.group(1)) or int(match.group(1)), match.group(2)
    else:
        zh = re.search(rf"{_ZH_NUM}(个星期|星期|个月|天|周|月|年)(以)?前", text)
        if zh is None:
            half = re.search(r"半(年|个月)(以)?前", text)
            if half is None:
                return None
            days, tolerance = (182, 60) if half.group(1) == "年" else (15, 7)
            center = today - timedelta(days=days)
            slack = timedelta(days=tolerance)
            return TimeRange(center - slack, center + slack + timedelta(days=1), half.group(0))
        n = _zh_int(zh.group(1))
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
        n = _zh_int(zh.group(2))
        unit = _UNIT_ZH[zh.group(3)]
        match = zh
    return TimeRange(
        today - timedelta(days=n * UNIT_DAYS[unit]), now + timedelta(seconds=1), match.group(0)
    )


def _last_weekday(text: str, now: datetime, today: datetime) -> TimeRange | None:
    match = re.search(
        r"\blast (monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend)\b", text
    )
    # "上周天气""上周日本之行"说的是上周，不是上周日：留给 _calendar
    zh = re.search(r"上(上?)个?(?:星期|周)([一二三四五六末]|日(?!本)|天(?!气))", text)
    if match is None and zh is None:
        return None
    if match is not None:
        name = match.group(1)
        expr = match.group(0)
        target = 5 if name == "weekend" else WEEKDAYS_EN.index(name)
    else:
        assert zh is not None
        # 中文的"上周三"按日历周算：上一个周一到周日里的周三，不是最近一个已经过去的周三
        name = zh.group(2)
        target = 5 if name == "末" else 6 if name in "日天" else WEEKDAYS_ZH.index(name)
        week_start = today - timedelta(days=today.weekday())
        start = week_start - timedelta(days=14 if zh.group(1) else 7) + timedelta(days=target)
        return _day(start, zh.group(0), days=2 if name == "末" else 1)
    back = (today.weekday() - target) % 7 or 7  # 最近一个已经过去的那天
    start = today - timedelta(days=back)
    return _day(start, expr, days=2 if match is not None and match.group(1) == "weekend" else 1)


def _calendar(text: str, now: datetime, today: datetime) -> TimeRange | None:
    week_start = today - timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    previous_month = (month_start - timedelta(days=1)).replace(day=1)
    two_months_ago = (previous_month - timedelta(days=1)).replace(day=1)
    table = [
        (r"上上(?:周|个?星期)", week_start - timedelta(days=14), week_start - timedelta(days=7)),
        (r"上上个?月", two_months_ago, previous_month),
        (
            r"(?<![以之大从目提])前年(?![轻纪龄代级])",
            today.replace(year=today.year - 2, month=1, day=1),
            today.replace(year=today.year - 1, month=1, day=1),
        ),
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
    zh = re.search(rf"(?:{_ZH_YEAR}的?)?(?<![\d十])(?:{_ZH_MONTH})月(?:{_ZH_DOM})[日号]", text)
    if match is not None:
        month, dom = MONTHS_EN.index(match.group(1)) + 1, int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        expr = match.group(0)
    elif zh is not None:
        month, dom, expr = _zh_int(zh.group(3)), _zh_int(zh.group(4)), zh.group(0)
        year = _zh_year(zh.group(1), zh.group(2), today)
    else:
        for name, (month, dom) in FIXED_HOLIDAYS.items():
            if name in text:
                dated = (
                    re.search(rf"{_ZH_YEAR}的?{re.escape(name)}", text)
                    if _HAN.search(name)
                    else None
                )
                if dated is not None:
                    year = _zh_year(dated.group(1), dated.group(2), today)
                    return _resolve_day(month, dom, year, today, dated.group(0))
                return _resolve_day(month, dom, None, today, name)
        return None
    return _resolve_day(month, dom, year, today, expr)


def _zh_dated_month(text: str, now: datetime, today: datetime) -> TimeRange | None:
    # "2024年3月""去年十月"：带年份的月份，要排在"去年"之前
    zh = re.search(rf"{_ZH_YEAR}的?(?:{_ZH_MONTH})月(?![\d日号])", text)
    if zh is None:
        return None
    year = _zh_year(zh.group(1), zh.group(2), today)
    assert year is not None
    start = today.replace(year=year, month=_zh_int(zh.group(3)), day=1)
    return TimeRange(start, (start + timedelta(days=32)).replace(day=1), zh.group(0))


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
    # 单独的"三月""十二月"；"三个月""每三月""五月天""十二月亮"之类不算
    han = re.search(
        r"(?<![\d个每这那几半多数第正腊一二三四五六七八九十])(十[一二]?|[一二三四五六七八九])月份?"
        r"(?![\d日号亮球光天一二三四五六七八九十])",
        text,
    )
    if match is not None:
        month = MONTHS_EN.index(match.group(1)) + 1
        year = int(match.group(2)) if match.group(2) else None
        expr = match.group(0)
    elif zh is not None and 1 <= int(zh.group(1)) <= 12:
        month, year, expr = int(zh.group(1)), None, zh.group(0)
    elif han is not None:
        month, year, expr = _zh_int(han.group(1)), None, han.group(0)
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
    _zh_dated_month,
    _relative_days,
    _past_window,
    _calendar,
    _month,
    _year,
)
