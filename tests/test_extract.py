"""抽取：会话渲染、宽松解析（模型输出不规范时不应整条失败）、超长会话切窗、谓词规范化。"""

from __future__ import annotations

import json
from datetime import datetime

from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.core.models import SessionIn, TurnIn
from cuemem.core.timeutil import normalize_partial_date
from cuemem.llm import LLMClient
from cuemem.write import build_turns
from cuemem.write.extract import ExtractionOutput, Extractor, render_session
from cuemem.write.predicates import normalize_predicate, registered_cardinality

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


def test_predicate_normalization() -> None:
    assert normalize_predicate("Lives In") == "lives_in"
    assert normalize_predicate("employer") == "works_at"
    assert normalize_predicate("5K personal-best") == "5k_personal_best"
    assert registered_cardinality("job_title") == "single"
    assert registered_cardinality("visited") == "multi"
    assert registered_cardinality("laptop_budget") is None
