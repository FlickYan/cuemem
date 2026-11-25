"""L1 快速召回：时间表达解析、BM25 分词、融合方式与时间感知打包。"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword

from cuemem.config import Settings
from cuemem.core.models import Conversation
from cuemem.memory import MemorySystem
from cuemem.read.bm25 import BM25, tokenize
from cuemem.read.l1 import L1Config
from cuemem.read.plan import rule_plan
from cuemem.read.probe import query_features
from cuemem.read.query import _ZH_NUM, analyze, cue_text, parse_time
from cuemem.read.reader import relative_days

NOW = datetime(2023, 5, 30, 22, 0)  # 周二


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("What did I do two weeks ago?", "2023-05-13", "2023-05-20"),
        ("I got a gift last Saturday", "2023-05-27", "2023-05-28"),
        ("Where did I go last month?", "2023-04-01", "2023-05-01"),
        ("What did I buy in March?", "2023-03-01", "2023-04-01"),
        ("What happened on May 5th?", "2023-05-05", "2023-05-06"),
        ("the flight on Valentine's day", "2023-02-14", "2023-02-15"),
        ("How much did I read in the past 3 weeks?", "2023-05-09", "2023-05-30"),
        ("我上周六去了哪里？", "2023-05-27", "2023-05-28"),
        ("三天前我买了什么？", "2023-05-26", "2023-05-29"),
        ("我去年在哪工作？", "2022-01-01", "2023-01-01"),
        ("5月20日发生了什么？", "2023-05-20", "2023-05-21"),
    ],
)
def test_time_expressions(text: str, start: str, end: str) -> None:
    found = parse_time(text, NOW)
    assert found is not None
    assert (f"{found.start:%Y-%m-%d}", f"{found.end:%Y-%m-%d}") == (start, end)


@pytest.mark.parametrize(
    "text",
    [
        "What may I need for the trip?",  # may 是情态动词，不是五月
        "Can you recommend a movie to watch tonight?",  # 指当下，不是检索线索
        "How many fitness classes do I attend in a typical week?",
    ],
)
def test_no_false_time_ranges(text: str) -> None:
    assert parse_time(text, NOW) is None


@pytest.mark.parametrize("text", ["7" * 200_000, "７" * 200_000])
def test_ago_zh_is_linear_on_long_digit_runs(text: str) -> None:
    # 旧写法从一长串数字的每一位都重新试一遍，20 万位要几分钟，期间整个服务不响应
    started = time.perf_counter()
    assert parse_time(text, NOW) is None
    assert time.perf_counter() - started < 5


@pytest.mark.parametrize(
    "text",
    [
        "三天前我买了什么",
        "12天前",
        "3个月以前",
        "１２天前",
        "abc123天前",
        "2个星期前",
        "几年前",
        "1 2天前",
        "12 天前",  # 不匹配
        "天前",  # 不匹配
    ],
)
def test_ago_zh_matches_old_pattern(text: str) -> None:
    # 数字前面加了 (?<!…) 之后，这些输入匹配到的位置与分组必须与原来的写法一样
    old = re.search(r"(\d+|[一两二三四五六七八九十几])(个星期|星期|个月|天|周|月|年)(以)?前", text)
    new = re.search(rf"{_ZH_NUM}(个星期|星期|个月|天|周|月|年)(以)?前", text)
    assert (old is None) == (new is None)
    if old is not None and new is not None:
        assert (new.span(), new.groups()) == (old.span(), old.groups())


def test_ago_zh_four_digits_are_not_a_count() -> None:
    # 有意的改动："2023年前"原来读成"2023 年以前"（落在公元 5 年前后），现在读成 2023 年
    found = parse_time("2023年前", NOW)
    assert found is not None
    assert (found.start.year, found.end.year, found.expr) == (2023, 2024, "2023年")


def test_ago_zh_keeps_the_whole_digit_run() -> None:
    found = parse_time("12天前", NOW)  # 取整串 12，不是最后一位 2
    assert found is not None
    assert (f"{found.start:%Y-%m-%d}", f"{found.end:%Y-%m-%d}") == ("2023-05-17", "2023-05-20")


NOW_2026 = datetime(2026, 11, 24, 10, 0)  # 周二


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        # 年份在月、日、节日前面
        ("2023年5月12日我去了哪里", "2023-05-12", "2023-05-13"),
        ("2024年3月我买了什么", "2024-03-01", "2024-04-01"),
        ("2024年的5月", "2024-05-01", "2024-06-01"),
        ("去年十月", "2025-10-01", "2025-11-01"),
        ("去年3月", "2025-03-01", "2025-04-01"),
        ("前年5月", "2024-05-01", "2024-06-01"),
        ("去年元旦", "2025-01-01", "2025-01-02"),
        # 多个字的数字
        ("十二天前", "2026-11-11", "2026-11-14"),
        ("二十天前", "2026-11-03", "2026-11-06"),
        ("十一个月前", "2025-12-19", "2026-01-09"),
        ("最近十二天", "2026-11-12", "2026-11-24"),
        ("半年前", "2026-03-27", "2026-07-26"),
        ("半个月前", "2026-11-02", "2026-11-17"),
        # 相对的天、周、月、年
        ("大前天", "2026-11-21", "2026-11-22"),
        ("前天天气怎么样", "2026-11-22", "2026-11-23"),
        # 前天、昨天都说到时取前天，与改动前按表的顺序相同（第十轮复审）
        ("昨天和前天我都去了哪", "2026-11-22", "2026-11-23"),
        ("前天和昨天我都去了哪", "2026-11-22", "2026-11-23"),
        ("昨天我去了哪", "2026-11-23", "2026-11-24"),
        ("昨晚吃了什么", "2026-11-23", "2026-11-24"),
        ("上上周三", "2026-11-11", "2026-11-12"),
        ("上上周", "2026-11-09", "2026-11-16"),
        ("上上个月", "2026-09-01", "2026-10-01"),
        ("上周末", "2026-11-21", "2026-11-23"),
        ("上周日", "2026-11-22", "2026-11-23"),
        ("上周天气怎么样", "2026-11-16", "2026-11-23"),  # 上周，不是"上周天"
        ("上周日本之行", "2026-11-16", "2026-11-23"),
        ("前年", "2024-01-01", "2025-01-01"),
        # 繁体、全角、带空格
        ("上週三", "2026-11-18", "2026-11-19"),
        ("上礼拜三", "2026-11-18", "2026-11-19"),
        ("上禮拜三", "2026-11-18", "2026-11-19"),
        ("上週天氣怎麼樣", "2026-11-16", "2026-11-23"),
        ("兩週前", "2026-11-07", "2026-11-14"),
        ("三個星期前", "2026-10-31", "2026-11-07"),
        ("過去三個月", "2026-08-26", "2026-11-24"),
        ("這個月", "2026-11-01", "2026-11-24"),
        ("聖誕節", "2025-12-25", "2025-12-26"),
        ("5 月 3 日", "2026-05-03", "2026-05-04"),
        ("3 天前", "2026-11-20", "2026-11-23"),
        ("２０２４年", "2024-01-01", "2025-01-01"),
        # 汉字写的月、日
        ("三月五号", "2026-03-05", "2026-03-06"),
        ("十二月", "2025-12-01", "2026-01-01"),
        # 四位数字是年份，不是"N 年前"
        ("2023年以前我住在哪", "2023-01-01", "2024-01-01"),
        ("最近2024年的计划", "2024-01-01", "2025-01-01"),
        # 四位数字的天、周、个月照旧是数量（第十轮复审：不会是年份的单位）
        ("1000天前我在做什么", "2024-02-27", "2024-03-01"),
        ("1500个星期以前", "1998-02-21", "1998-02-28"),
        ("过去1000天", "2024-02-28", "2026-11-24"),
        ("最近1200周", "2003-11-25", "2026-11-24"),
    ],
)
def test_zh_time_expressions(text: str, start: str, end: str) -> None:
    found = parse_time(text, NOW_2026)
    assert found is not None
    assert (f"{found.start:%Y-%m-%d}", f"{found.end:%Y-%m-%d}") == (start, end)


@pytest.mark.parametrize(
    "text",
    [
        "我以前天天跑步",  # 不是"前天"
        "目前天气怎么样",
        "二十几天前",  # 不截成"几天前"
        "一百二十天前",  # 不截成"十天前"
        "1.5天前",
        "大前年",
        "五月天的演唱会",
        "三个月",
        "每三月",
        "3000年前的恐龙",  # 原来在这里抛 OverflowError
        "10000天前",  # 五位以上的数量不收
        "12345个月前",
        "1.2345天前",
        "1000月前",
        "最近2024周年庆",
    ],
)
def test_zh_no_false_time_ranges(text: str) -> None:
    assert parse_time(text, NOW_2026) is None


def test_zh_last_weekday_uses_the_calendar_week() -> None:
    friday = datetime(2026, 11, 27, 10, 0)
    found = parse_time("上周三", friday)  # 上一个日历周的周三，不是本周三（两天前）
    assert found is not None
    assert f"{found.start:%Y-%m-%d}" == "2026-11-18"
    english = parse_time("last wednesday", friday)  # 英文的规则不变：最近一个已经过去的周三
    assert english is not None
    assert f"{english.start:%Y-%m-%d}" == "2026-11-25"


def test_zh_holiday_takes_the_year_before_it() -> None:
    found = parse_time("去年圣诞节", datetime(2026, 12, 28, 10, 0))
    assert found is not None
    assert (f"{found.start:%Y-%m-%d}", found.expr) == ("2025-12-25", "去年圣诞节")


def test_zh_cues_in_traditional_characters() -> None:
    assert analyze("我現在住在哪裡", NOW_2026).wants_latest
    assert analyze("總共幾次", NOW_2026).aggregation
    assert analyze("什麼時候", NOW_2026).temporal
    info = analyze("之後我去了哪裡", NOW_2026)
    assert info.temporal and info.text == "之後我去了哪裡"  # 要检索的问题不改写
    assert query_features("花了多長時間", info)["q_duration"] == 1.0
    assert rule_plan("總共幾次", NOW_2026).answer_type == "count"


def test_cue_text_leaves_text_without_han_alone() -> None:
    for text in ("What did I do 3 days ago?", "１２ days ago", "in 2023 ", ""):
        assert cue_text(text) is text
    assert cue_text("上週三 ５ 月 ３ 日") == "上周三 5月3日"
    assert cue_text("我以前住在 Berlin 3 号楼") == "我以前住在 Berlin 3号楼"


def test_zh_number_is_linear_on_long_digit_runs() -> None:
    started = time.perf_counter()
    assert parse_time("1" * 20_000 + "天", NOW) is None
    assert parse_time("过去" + "1" * 20_000 + "天前", NOW) is None
    assert time.perf_counter() - started < 2


def test_english_questions_parse_as_before() -> None:
    """不含汉字的问题：时间区间与线索标志与改动前逐条相同。

    语料取自 tests/ 与 examples/ 里的英文问题与句子，另加二十句补上它们没有碰到的写法（数字的
    "N days ago"、the day before yesterday、last night、节日、past N days、带年份的日期等；
    第十轮复审）；期望值是改动前的 query.py（fixture 里 base 那个提交）算出来的，写成字面量，
    不随现在的代码变。
    """
    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "parse_time_english.json").read_text("utf-8")
    )
    moments = [datetime.fromisoformat(text) for text in fixture["now"]]
    flags = ("wants_first", "wants_latest", "wants_history", "aggregation", "temporal")
    assert len(fixture["questions"]) >= 150
    with_range = 0
    exprs: set[str] = set()
    for item in fixture["questions"]:
        text = item["text"]
        assert text.isascii()
        for moment, expected in zip(moments, item["expected"], strict=True):
            info = analyze(text, moment)
            assert info.time_range == parse_time(text, moment)
            found = info.time_range
            got = (
                None
                if found is None
                else [found.start.isoformat(sep=" "), found.end.isoformat(sep=" "), found.expr]
            )
            assert got == expected["range"], text
            assert [name for name in flags if getattr(info, name)] == expected["flags"], text
        with_range += item["expected"][0]["range"] is not None
        exprs.update(e["range"][2] for e in item["expected"] if e["range"] is not None)
    assert with_range >= 30
    assert len(exprs) >= 30  # 不同的时间表达：语料不能只是同几种写法的重复
    assert {"3 days ago", "the day before yesterday", "last night", "christmas"} <= exprs


def test_query_flags() -> None:
    info = analyze("How many times did I first go hiking before I moved?", NOW)
    assert info.aggregation and info.wants_first and info.wants_history and info.temporal
    assert analyze("我现在住在哪里？", NOW).wants_latest


def test_bm25_tokenizer_and_ranking() -> None:
    assert tokenize("User: I paid $8000 for 12.5 kg\nAssistant: ok") == [
        "paid",
        "8000",
        "12.5",
        "kg",
        "ok",
    ]
    assert "笔记" in tokenize("我买了笔记本")  # 中文二字组
    index = BM25(["User: my dog Max", "User: I bought layer feed for my chickens", "hello"])
    assert [i for i, _ in index.search("chickens feed", 3)] == [1]


def test_relative_days() -> None:
    assert relative_days(datetime(2023, 5, 30, 9), NOW) == "the same day as the question"
    assert relative_days(datetime(2023, 5, 27), NOW) == "3 days before the question"
    assert (
        relative_days(datetime(2023, 4, 30), NOW) == "30 days (about 4 weeks) before the question"
    )


@pytest.fixture
def toy(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[tuple[MemorySystem, ScriptedModel]]:
    model = ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    fake_api.chat_replies = [model]
    conv = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    system.ingest(conv, consolidate=True)
    yield system, model
    system.close()


def test_l1_finds_first_travel_mention(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    """N1 冒烟测试里朴素检索答错的那道题：关键词路与事实键把"巴厘岛旅行"那一轮带到前面。"""
    system, _ = toy
    question = "我第一次提到想去旅行是什么时候？想去哪里？"
    result = system.l1.search("demo", question, datetime(2025, 10, 10), L1Config(k=3))
    assert any("巴厘岛" in h.unit.text for h in result.rounds)
    assert any("巴厘岛" in fh.fact.statement for fh in result.facts)
    assert result.query.wants_first


def test_time_boost_prefers_rounds_in_range(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    system, _ = toy
    question = "我三个月前说了什么？"  # 提问 2025-09-28 → 三个月前约为 2025-06-20 那段对话
    when = datetime(2025, 9, 20)
    boosted = system.l1.search("demo", question, when, L1Config(k=1, time_weight=5.0))
    assert boosted.query.time_range is not None
    assert boosted.rounds[0].unit.session_id == "s3"


def test_l1_answer_has_relative_time(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    system, model = toy
    system.ask("demo", "我现在在哪工作？", question_time=datetime(2025, 10, 10))
    prompt = model.prompts["read"][-1]
    assert "days before the question" in prompt and "Notes from memory" in prompt
    system.ask(
        "demo", "我现在在哪工作？", question_time=datetime(2025, 10, 10), relative_time=False
    )
    assert "days before the question" not in model.prompts["read"][-1]


def test_fusion_modes_differ_only_in_how_lists_combine(
    toy: tuple[MemorySystem, ScriptedModel],
) -> None:
    system, _ = toy
    when = datetime(2025, 10, 10)
    for fusion in ("modality", "kind"):
        result = system.l1.search(
            "demo",
            "Max 喜欢去哪里散步？",
            when,
            L1Config(k=2, fusion=fusion),  # type: ignore[arg-type]
        )
        assert result.rounds and result.timings_ms["total"] > 0
        assert any("东海岸公园" in h.unit.text for h in result.rounds)


def test_token_budget_packing(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    from cuemem.read.l1 import estimate_tokens

    system, _ = toy
    when = datetime(2025, 10, 10)
    small = system.l1.search("demo", "我现在在哪工作？", when, L1Config(budget_tokens=40))
    large = system.l1.search("demo", "我现在在哪工作？", when, L1Config(budget_tokens=10_000))
    assert 1 <= len(small.rounds) < len(large.rounds) == 9  # 示例对话一共 9 轮
    assert sum(estimate_tokens(h.unit.text) for h in small.rounds) <= 40 or len(small.rounds) == 1
    assert estimate_tokens("我买了笔记本") == 7 and estimate_tokens("abcdefgh") == 3


def test_hidden_rounds_free_their_bm25_and_dense_slots(
    toy: tuple[MemorySystem, ScriptedModel],
) -> None:
    """hidden（过期的记忆，第七轮体检 P5）在各路检索里就排除：BM25 与 dense 每路仍取满 pool 条，
    结果与"取全部再去掉"的前 pool 条相同；exclude 照旧先取前 pool 条再去掉（N8 的做法不变）。"""
    system, _ = toy
    when = datetime(2025, 10, 10)
    question = "我现在在哪工作？"
    cfg = L1Config(pool=4, sparse="bm25")
    full = system.l1.candidates("demo", question, when, replace(cfg, pool=100))
    order = [uid for uid, _ in full.routes[("round", "dense")]]
    words = [uid for uid, _ in full.routes[("round", "sparse")]]
    assert len(order) == 9 and len(words) > 4  # 示例对话一共 9 轮
    hidden = frozenset(order[:3]) | frozenset(words[:2])
    found = system.l1.candidates("demo", question, when, cfg, hidden=hidden)
    assert [u for u, _ in found.routes[("round", "dense")]] == [
        u for u in order if u not in hidden
    ][:4]
    assert [u for u, _ in found.routes[("round", "sparse")]] == [
        u for u in words if u not in hidden
    ][:4]
    assert not {rid for rid, _ in found.ranked} & hidden
    excluded = system.l1.candidates("demo", question, when, cfg, exclude=hidden)
    assert len(excluded.routes[("round", "dense")]) == 4 - len(hidden & set(order[:4]))
