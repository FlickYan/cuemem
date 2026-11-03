"""抽取：会话渲染、宽松解析（模型输出不规范时不应整条失败）、超长会话切窗、谓词与名字的规范化。"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.core.models import SessionIn, TurnIn
from cuemem.core.timeutil import normalize_partial_date
from cuemem.llm import LLMClient
from cuemem.write import build_turns
from cuemem.write.entities import normalize_name
from cuemem.write.extract import DATED_NOTE, ExtractionOutput, Extractor, render_session
from cuemem.write.predicates import (
    legacy_value_key,
    normalize_predicate,
    normalize_value,
    registered_cardinality,
)

WHEN = datetime(2023, 5, 20, 9, 0)


def session(*turns: tuple[str, str], speaker: str | None = None) -> list:  # type: ignore[type-arg]
    s = SessionIn(
        session_id="s1",
        started_at=WHEN,
        turns=[TurnIn(role=role, text=text, speaker=speaker) for role, text in turns],  # type: ignore[arg-type]
    )
    return build_turns("space", s)


def test_render_truncates_assistant_and_numbers_turns() -> None:
    turns = session(("user", "I adopted a dog."), ("assistant", "A" * 50))
    text = render_session(turns, WHEN, assistant_chars=10)
    assert "Conversation date: 2023-05-20 (Saturday)" in text
    assert "Participants: User, Assistant (AI)" in text
    assert "[t0] User: I adopted a dog." in text
    assert "[t1] Assistant: AAAAAAAAAA […]" in text
    assert "A" * 11 not in render_session(turns, WHEN, assistant_chars=10)
    assert "A" * 50 in render_session(turns, WHEN, assistant_chars=None)


def test_multi_speaker_sessions_use_names() -> None:
    turns = session(("user", "Hi Mel!"), speaker="Caroline")
    text = render_session(turns, WHEN, assistant_chars=10)
    assert "Participants: Caroline" in text and "[t0] Caroline: Hi Mel!" in text


def test_lenient_parsing() -> None:
    raw = {
        "summary": None,
        "facts": [
            {
                "statement": "  The user ran a 5K.  ",
                "importance": "7",
                "category": "hobby",
                "cardinality": "SINGLE",
                "date": "2023/5/7",
                "turns": ["t2", "[t3]", 4],
            },
            {"statement": "x", "importance": "high", "date": "last week", "turns": None},
        ],
        "entities": [{"name": "Max", "type": "dog", "aliases": "Maxie"}],
    }
    out = ExtractionOutput.model_validate(raw)
    first, second = out.facts
    assert first.statement == "The user ran a 5K."
    assert (first.importance, first.category, first.cardinality) == (5, "other", "single")
    assert first.date == "2023-05-07" and first.turns == [2, 3, 4]
    assert (second.importance, second.date, second.turns) == (3, None, [])
    assert out.entities[0].type == "other" and out.entities[0].aliases == ["Maxie"]
    assert out.summary == ""


def test_partial_dates() -> None:
    assert normalize_partial_date("2023-05") == "2023-05"
    assert normalize_partial_date("2023") == "2023"
    assert normalize_partial_date("2023-02-30") is None
    assert normalize_partial_date("2023-05-07T10:00:00") == "2023-05-07"
    assert normalize_partial_date(None) is None


def test_extractor_maps_turns_and_splits_long_sessions(
    settings: Settings, fake_api: FakeAPI
) -> None:
    reply = {
        "summary": "s",
        "facts": [{"statement": "f", "turns": [0, 99]}],
        "entities": [],
    }
    fake_api.chat_replies = [json.dumps(reply)]
    extractor = Extractor(LLMClient(settings, http_client=fake_api.client()), max_chars=30)
    turns = session(("user", "a" * 20), ("assistant", "b" * 20), ("user", "c" * 20))
    result = extractor.extract("s1", turns, WHEN)
    assert len(fake_api.chat_requests) == 3  # 每个窗口一次调用
    assert result.summary == "s s s"
    assert result.facts[0].source_turn_ids == [turns[0].id]
    assert result.dropped_turn_refs == 3  # 三个窗口的回复都引用了不存在的编号 99
    request = fake_api.chat_requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["enable_thinking"] is False  # 抽取用 qwen3.8-flash，关闭思考模式


def test_deleted_gaps_are_marked_only_where_messages_were_deleted(
    settings: Settings, fake_api: FakeAPI
) -> None:
    """遗忘留下的空缺标成 [deleted] 并加说明（剩下的代词不要套到别人身上）；没删过的会话、
    超长会话后面的窗口（编号本来就不从 0 起）都不标，渲染结果与之前相同、抽取缓存照常命中。"""
    turns = session(
        ("user", "My friend Lena runs."),
        ("assistant", "Nice."),
        ("user", "My sister Ana turns 30."),
        ("assistant", "Happy birthday to Ana!"),
        ("user", "I'll get those poles for her."),
        ("assistant", "Great."),
    )
    plain = render_session(turns, WHEN, assistant_chars=None)
    assert "[deleted]" not in plain and "deleted at the user's request" not in plain
    assert render_session(turns, WHEN, assistant_chars=None, deleted_from=0) == plain
    kept = [turns[i] for i in (0, 1, 4, 5)]
    text = render_session(kept, WHEN, assistant_chars=None, deleted_from=0)
    assert "deleted at the user's request" in text
    assert "[t1] Assistant: Nice.\n[deleted]\n[t4] User: I'll get those poles for her." in text

    reply = {"summary": "s", "facts": [], "entities": []}
    fake_api.chat_replies = [json.dumps(reply)]
    extractor = Extractor(LLMClient(settings, http_client=fake_api.client()), max_chars=30)
    extractor.extract("s1", turns, WHEN)  # 切成几个窗口，后面的窗口编号不从 0 起
    first = len(fake_api.chat_requests)
    assert first > 1
    extractor.extract("s1", [turns[i] for i in (0, 1, 2, 3, 5)], WHEN)  # 删了第 4 句
    prompts = [r["messages"][1]["content"] for r in fake_api.chat_requests]
    assert not any("[deleted]" in p for p in prompts[:first])
    marked = [p for p in prompts[first:] if "[deleted]" in p]
    assert len(marked) == 1 and "[deleted]\n[t5]" in marked[0]


def test_turns_said_on_another_day_get_a_date_line() -> None:
    """一段会话跨了几天（复用的 run_id、LangGraph 的线程）：换了日期的发言前标出日期，加一行
    说明。同一天里的会话（评测的会话、Agent 的一次对话）渲染结果与之前逐字相同。"""
    turns = session(
        ("user", "I adopted a dog."), ("assistant", "Nice."), ("user", "I moved yesterday.")
    )
    expected = (
        "Conversation date: 2023-05-20 (Saturday)\n"
        "Participants: User, Assistant (AI)\n"
        "Language: English\n"
        "\n"
        "[t0] User: I adopted a dog.\n"
        "[t1] Assistant: Nice.\n"
        "[t2] User: I moved yesterday."
    )
    assert render_session(turns, WHEN, assistant_chars=None) == expected
    same_day = [*turns[:2], turns[2].model_copy(update={"ts": WHEN + timedelta(hours=14)})]
    assert render_session(same_day, WHEN, assistant_chars=None) == expected

    later = [*turns[:2], turns[2].model_copy(update={"ts": datetime(2023, 5, 28, 18, 0)})]
    text = render_session(later, WHEN, assistant_chars=None)
    header, body = text.split("\n\n")
    assert header == expected.split("\n\n")[0] + "\n" + DATED_NOTE
    assert body == (
        "[t0] User: I adopted a dog.\n"
        "[t1] Assistant: Nice.\n"
        "[Date: 2023-05-28 (Sunday)]\n"
        "[t2] User: I moved yesterday."
    )


def test_predicate_normalization() -> None:
    assert normalize_predicate("Lives In") == "lives_in"
    assert normalize_predicate("employer") == "works_at"
    assert normalize_predicate("5K personal-best") == "5k_personal_best"
    assert registered_cardinality("job_title") == "single"
    assert registered_cardinality("visited") == "multi"
    assert registered_cardinality("laptop_budget") is None


def test_normalization_keeps_other_scripts() -> None:
    """西里尔、希腊、韩文、假名等文字不再被规范化成 ''：两个俄文名字是两个键，'서울' 与
    '부산' 不再相同，日文句子不再只剩汉字。"""
    assert normalize_name("Анна") == "анна" and normalize_name("Иван") == "иван"
    assert normalize_name("Μαρία") == "μαρία" and normalize_name("민준") == "민준"
    assert normalize_value("Москва") == "москва"
    assert normalize_value("서울") != normalize_value("부산")
    assert normalize_value("The user has a dog named 초코.") == "user has a dog named 초코"
    assert normalize_value("ユーザーは東京に住んでいる。") == "ユーザーは東京に住んでいる"
    assert normalize_predicate("живёт в") == "живёт_в"
    assert normalize_predicate("—") == "related_to"
    # 派生事实 ID 用的旧键不变
    assert legacy_value_key("The user has a dog named 초코.") == "user has a dog named"


def test_normalization_of_latin_and_chinese_is_unchanged() -> None:
    """拉丁字母（含带重音的）与中文的键与原来逐字相同：评测里已有的事实 ID、实体与对账不变。"""
    values = {
        "The user reads Brené Brown.": "user reads bren brown",
        "A Wüsthof chef's knife": "w sthof chef s knife",
        "Zoë": "zo",
        "x² café": "x caf",
        "用户住在北京。": "用户住在北京",
        "My dog Max (golden retriever)": "dog max golden retriever",
        "The user lives in New York City.": "user lives in new york city",
    }
    for text, key in values.items():
        assert normalize_value(text) == key == legacy_value_key(text)
    names = {
        "Brené Brown": "bren brown",
        "Wüsthof": "w sthof",
        "Zoë": "zo",
        "林晓": "林晓",
        "The User": "the user",
        "Dr. Who": "dr who",
    }
    for text, key in names.items():
        assert normalize_name(text) == key
    assert normalize_predicate("最喜欢的颜色") == "最喜欢的颜色"
