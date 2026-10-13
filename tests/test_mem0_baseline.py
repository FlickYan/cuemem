"""外部基线 M0（Mem0）的接线测试：不联网、不调用模型。

mem0ai 不是 CueMem 的依赖，平时这个文件整体跳过；要跑时临时装上：
    uv run --with mem0ai --with spacy --with fastembed pytest tests/test_mem0_baseline.py
"""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from cuemem.config import Settings
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.methods import MethodContext

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mem0") is None, reason="需要 mem0ai（uv run --with mem0ai ...）"
)


class FakeLLM:
    """只回答 Mem0 的抽取请求：把新消息里用户说的话原样记成一条记忆，并记下提示里的观察日期。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ledger = None
        self.observed: list[str] = []

    def chat(self, role: str, messages: list[dict[str, Any]], **_: Any) -> SimpleNamespace:
        prompt = str(messages[-1]["content"])
        observed = re.search(r"## Observation Date\n(\S+)", prompt)
        self.observed.append(observed.group(1) if observed else "")
        new = prompt.split("## New Messages\n")[1].split("\n\n## ")[0]  # "user: ……" 每行一条
        said = [
            line.removeprefix("user: ") for line in new.splitlines() if line.startswith("user: ")
        ]
        memories = [{"id": str(i), "text": text} for i, text in enumerate(said)]
        return SimpleNamespace(content=json.dumps({"memory": memories}))


class FakeReader:
    def __init__(self, llm: FakeLLM) -> None:
        self.llm = llm
        self.seen: list[list[Any]] = []

    def answer(self, question: str, hits: list[Any], **_: Any) -> SimpleNamespace:
        self.seen.append(hits)
        return SimpleNamespace(answer="ok", evidence_sufficient=bool(hits))


def _sessions() -> list[EvalSession]:
    def turn(role: str, text: str) -> EvalTurn:
        return EvalTurn(role=role, text=text)  # type: ignore[arg-type]

    return [
        EvalSession(
            "s1",
            datetime(2023, 5, 8, 13, 56),
            [
                turn("user", "I adopted a beagle named Biscuit yesterday."),
                turn("assistant", "Nice!"),
            ],
        ),
        EvalSession(
            "s2",
            datetime(2023, 6, 1, 9, 0),
            [turn("user", "Biscuit chewed my running shoes."), turn("assistant", "Oh no.")],
        ),
    ]


def test_mem0_baseline_builds_once_with_session_dates(offline_settings: Settings) -> None:
    from cuemem.evals.mem0_baseline import Mem0Baseline

    llm = FakeLLM(offline_settings)
    reader = FakeReader(llm)
    method = Mem0Baseline()
    sample = Sample(
        dataset="toy",
        sample_id="q1",
        space_key="conv",
        question="What is the name of my beagle Biscuit?",
        answer="Biscuit",
        question_type="single-session-user",
        ability="information_extraction",
        is_abstention=False,
        question_time=datetime(2023, 7, 1),
    )
    ctx = MethodContext(sample, _sessions(), reader, None, "conv")  # type: ignore[arg-type]

    first = method.answer(ctx)
    # 抽取提示里的"观察日期"是会话日期，不是运行当天：相对日期才会被换算到对的年份
    assert llm.observed == ["2023-05-08", "2023-06-01"]
    assert first.details["mem0_build"]["memories"] == 2
    texts = {h.unit.text: h.unit for h in first.hits}
    beagle = texts["I adopted a beagle named Biscuit yesterday."]
    assert beagle.ts == datetime(2023, 5, 8, 13, 56) and beagle.session_id == "s1"

    second = method.answer(ctx)  # 同一份对话历史：不再写入
    assert "mem0_build" not in second.details and len(llm.observed) == 2
    assert second.hits and reader.seen[-1] == second.hits


def test_mem0_baseline_gives_each_speaker_a_user_id(offline_settings: Settings) -> None:
    """两个人之间的对话（LoCoMo）：照 Mem0 论文的评测代码，每位说话人一个 user_id，
    自己的话是 user、对方的话是 assistant；检索两边都取，每条注明是谁的记忆。"""
    from cuemem.evals.mem0_baseline import Mem0Baseline

    def turn(speaker: str, text: str) -> EvalTurn:
        return EvalTurn(role="user", text=text, speaker=speaker)

    sessions = [
        EvalSession(
            "session_1",
            datetime(2023, 5, 8, 13, 56),
            [
                turn("Caroline", "I went to a support group yesterday."),
                turn("Melanie", "I painted a lake sunrise last week."),
            ],
        )
    ]
    llm = FakeLLM(offline_settings)
    sample = Sample(
        dataset="locomo",
        sample_id="q1",
        space_key="conv-1",
        # 两条记忆都要相关：Mem0 的检索默认丢掉分数低于 0.1 的
        question="Did Caroline go to a support group, and did Melanie paint a lake sunrise?",
        answer="yes",
        question_type="temporal",
        ability="temporal_reasoning",
        is_abstention=False,
        question_time=datetime(2023, 7, 1),
    )
    ctx = MethodContext(sample, sessions, FakeReader(llm), None, "locomo:conv-1")  # type: ignore[arg-type]
    out = Mem0Baseline().answer(ctx)
    assert out.details["mem0_build"]["perspectives"] == ["Caroline", "Melanie"]
    assert llm.observed == ["2023-05-08", "2023-05-08"]  # 每位说话人各写一遍
    # FakeLLM 只记 user 角色的话：各自的存储里只有自己说的，说明角色按视角互换了
    assert sorted(h.unit.text for h in out.hits) == [
        '[Caroline\'s memory; "User" = Caroline] Caroline: I went to a support group yesterday.',
        '[Melanie\'s memory; "User" = Melanie] Melanie: I painted a lake sunrise last week.',
    ]


class ListLLM(FakeLLM):
    """像通义的 JSON 模式偶尔那样，直接返回记忆列表，少了外层的 {"memory": …}。"""

    def chat(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> SimpleNamespace:
        reply = super().chat(role, messages, **kw)
        return SimpleNamespace(content=json.dumps(json.loads(reply.content)["memory"]))


def test_mem0_baseline_accepts_a_bare_list(offline_settings: Settings) -> None:
    from cuemem.evals.mem0_baseline import Mem0Baseline, _wrap_list

    assert json.loads(_wrap_list('```json\n[{"id": "0", "text": "x"}]\n```')) == {
        "memory": [{"id": "0", "text": "x"}]
    }
    assert _wrap_list('{"memory": []}') == '{"memory": []}'
    llm = ListLLM(offline_settings)
    sample = Sample(
        dataset="toy",
        sample_id="q1",
        space_key="conv",
        question="What is the name of my beagle Biscuit?",
        answer="Biscuit",
        question_type="single-session-user",
        ability="information_extraction",
        is_abstention=False,
        question_time=datetime(2023, 7, 1),
    )
    ctx = MethodContext(sample, _sessions(), FakeReader(llm), None, "conv")  # type: ignore[arg-type]
    out = Mem0Baseline().answer(ctx)
    assert out.details["mem0_build"]["memories"] == 2  # 两个会话的记忆都写进去了


class FlakyLLM(FakeLLM):
    """第二个会话的抽取总是失败（例如审核误拒，换了备用模型也不行）。"""

    def chat(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> SimpleNamespace:
        if "running shoes" in str(messages[-1]["content"]).split("## New Messages")[1]:
            raise RuntimeError("data_inspection_failed")
        return super().chat(role, messages, **kw)


def test_mem0_baseline_skips_a_failing_session(offline_settings: Settings) -> None:
    from cuemem.evals.mem0_baseline import Mem0Baseline

    llm = FlakyLLM(offline_settings)
    sample = Sample(
        dataset="toy",
        sample_id="q1",
        space_key="conv",
        question="What is my beagle Biscuit like?",
        answer="",
        question_type="single-session-user",
        ability="information_extraction",
        is_abstention=False,
        question_time=datetime(2023, 7, 1),
    )
    ctx = MethodContext(sample, _sessions(), FakeReader(llm), None, "conv")  # type: ignore[arg-type]
    out = Mem0Baseline().answer(ctx)
    build = out.details["mem0_build"]
    assert build["memories"] == 1 and len(build["failed_sessions"]) == 1
    assert "s2" in build["failed_sessions"][0]


def test_mem0_baseline_does_not_rebuild_after_a_failed_build(
    offline_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuemem.evals.mem0_baseline import Mem0Baseline

    method = Mem0Baseline()
    calls = []

    def broken(self: Any, key: str, sessions: Any) -> dict[str, Any]:
        calls.append(key)
        raise RuntimeError("vector store down")

    monkeypatch.setattr(Mem0Baseline, "_build", broken)
    llm = FakeLLM(offline_settings)
    sample = Sample(
        dataset="toy",
        sample_id="q1",
        space_key="conv",
        question="x",
        answer="",
        question_type="single-session-user",
        ability="information_extraction",
        is_abstention=False,
        question_time=datetime(2023, 7, 1),
    )
    ctx = MethodContext(sample, _sessions(), FakeReader(llm), None, "conv")  # type: ignore[arg-type]
    for _ in range(3):
        with pytest.raises(RuntimeError):
            method.answer(ctx)
    assert calls == ["conv"]  # 只尝试一次：同一份历史后面的题直接报错，不再花钱重建
