"""L1 快速召回：时间表达解析、BM25 分词、融合方式与时间感知打包。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword

from cuemem.config import Settings
from cuemem.core.models import Conversation
from cuemem.memory import MemorySystem
from cuemem.read.bm25 import BM25, tokenize
from cuemem.read.l1 import L1Config
from cuemem.read.query import analyze, parse_time
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
