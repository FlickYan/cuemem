from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from cuemem.evals.datasets import (
    RAW_SESSION_IDS_ENV,
    evidence_turns,
    load_locomo,
    load_longmemeval,
    sample_evidence_turns,
    to_conversation,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_longmemeval_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RAW_SESSION_IDS_ENV, raising=False)
    evalset = load_longmemeval(FIXTURES / "longmemeval_mini.json")
    by_id = {s.sample_id: s for s in evalset.samples}
    assert set(by_id) == {"q-ssu", "q-tr", "q-cat_abs"}

    ssu = by_id["q-ssu"]
    assert ssu.question_type == "single-session-user"
    assert ssu.ability == "information_extraction"
    assert not ssu.is_abstention
    assert ssu.question_time == datetime(2023, 6, 1, 10, 0)
    # 会话 ID 按时间顺序改写（原始 ID 会泄漏证据会话，见 test_eval_leak.py），原 ID 留作追溯
    assert evalset.session_ids == "anon"
    assert ssu.evidence_session_ids == ["session_1"]
    sessions = evalset.sessions_of(ssu)
    assert [s.session_id for s in sessions] == ["session_1", "session_2", "session_3"]
    assert [s.source_id for s in sessions] == ["ssu-a", "ssu-b", "ssu-c"]
    assert sessions[0].started_at == datetime(2023, 5, 20, 9, 0)
    assert evidence_turns(sessions) == {("session_1", 0)}

    assert by_id["q-tr"].answer == "12"  # 整数答案转成字符串
    assert by_id["q-tr"].ability == "temporal_reasoning"
    assert by_id["q-cat_abs"].is_abstention

    conversation = to_conversation("space", sessions)
    assert [t.role for t in conversation.sessions[0].turns] == ["user", "assistant"]


def test_longmemeval_loader_filters_ids() -> None:
    evalset = load_longmemeval(FIXTURES / "longmemeval_mini.json", ids={"q-tr"})
    assert [s.sample_id for s in evalset.samples] == ["q-tr"]
    assert set(evalset.haystacks) == {"q-tr"}


def test_locomo_loader() -> None:
    evalset = load_locomo(FIXTURES / "locomo_mini.json")
    by_type = {s.question_type: s for s in evalset.samples}
    assert set(by_type) == {"single-hop", "temporal", "adversarial", "multi-hop"}
    assert {s.space_key for s in evalset.samples} == {"conv-test"}  # 同一段对话的题共享历史

    sessions = evalset.haystacks["conv-test"]
    first = sessions[0].turns
    assert [t.speaker for t in first] == ["Alice", "Bob", "Alice"]
    assert {t.role for t in first} == {"user"}
    assert first[2].text.endswith("[shares a photo: a photo of a golden retriever puppy]")
    assert sessions[1].started_at == datetime(2023, 5, 20, 10, 0)

    single = by_type["single-hop"]
    assert single.question_time == datetime(2023, 5, 20, 10, 0)  # 取最后一个会话的时间
    assert sample_evidence_turns(single, sessions) == {("session_1", 2)}
    assert single.evidence_session_ids == ["session_1"]

    adversarial = by_type["adversarial"]
    assert adversarial.is_abstention
    assert "It was hard" in adversarial.answer

    multi = by_type["multi-hop"]
    assert multi.answer == "2023"
    # "D1:1; D2:1" 这种合并写法会被拆开
    assert sample_evidence_turns(multi, sessions) == {("session_1", 0), ("session_2", 0)}
