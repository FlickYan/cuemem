from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import pytest
from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.evals.datasets import Sample
from cuemem.evals.judge import Judge, judge_prompt, token_f1
from cuemem.llm import Ledger, LLMClient, ResponseCache

LONGMEMEVAL_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


def make_sample(question_type: str, *, abstention: bool = False) -> Sample:
    return Sample(
        dataset="longmemeval_s",
        sample_id="q_abs" if abstention else "q",
        space_key="q",
        question="What breed is my dog?",
        answer="golden retriever",
        question_type=question_type,
        ability="",
        is_abstention=abstention,
        question_time=datetime(2023, 6, 1),
    )


def load_official():  # type: ignore[no-untyped-def]
    path = Path(__file__).parent / "fixtures" / "longmemeval_official.py"
    spec = importlib.util.spec_from_file_location("longmemeval_official", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_anscheck_prompt


@pytest.mark.parametrize("abstention", [False, True])
@pytest.mark.parametrize("question_type", LONGMEMEVAL_TYPES)
def test_prompts_match_official_verbatim(question_type: str, abstention: bool) -> None:
    official = load_official()
    sample = make_sample(question_type, abstention=abstention)
    expected = official(
        question_type,
        sample.question,
        sample.answer,
        "It is a golden retriever.",
        abstention=abstention,
    )
    assert judge_prompt(sample, "It is a golden retriever.") == expected


def test_judge_call_follows_official_settings(settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = ["Yes.", "no"]
    llm = LLMClient(
        settings, cache=ResponseCache(None), ledger=Ledger(None), http_client=fake_api.client()
    )
    judge = Judge(llm)
    assert judge.judge(make_sample("multi-session"), "golden retriever").correct
    assert not judge.judge(make_sample("multi-session"), "poodle").correct

    request = fake_api.chat_requests[0]
    assert request["model"] == "qwen3.8-max"
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 10
    assert [m["role"] for m in request["messages"]] == ["user"]  # 只有一条 user 消息


def test_token_f1() -> None:
    assert token_f1("The golden retriever", "golden retriever") == 1.0
    assert token_f1("a poodle", "golden retriever") == 0.0
    assert token_f1("golden lab", "golden retriever") == pytest.approx(0.5)
