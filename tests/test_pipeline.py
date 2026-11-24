from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from fakes import TOY_CONVERSATION, FakeAPI

from cuemem.config import Settings
from cuemem.core.models import Conversation, SessionIn, TurnIn
from cuemem.memory import MemorySystem
from cuemem.write import build_rounds, build_turns


def test_rounds_group_assistant_replies() -> None:
    session = SessionIn(
        session_id="s",
        started_at=datetime(2025, 1, 1),
        turns=[
            TurnIn(role="assistant", text="欢迎"),
            TurnIn(role="user", text="问1"),
            TurnIn(role="assistant", text="答1a"),
            TurnIn(role="assistant", text="答1b"),
            TurnIn(role="user", text="问2"),
        ],
    )
    rounds = build_rounds(build_turns("space", session))
    assert [len(r.source_turn_ids) for r in rounds] == [1, 3, 1]
    assert rounds[1].text == "User: 问1\nAssistant: 答1a\nAssistant: 答1b"


def cite_golden_retriever(body: dict[str, Any]) -> str:
    """模拟 reader：在上下文里找到提到"金毛"的记忆并引用它。"""
    prompt = body["messages"][-1]["content"]
    match = re.search(r"\[(m\d+)\][^\[]*金毛", prompt)
    return json.dumps(
        {
            "answer": "金毛",
            "citations": [match.group(1)] if match else [],
            "evidence_sufficient": match is not None,
        }
    )


def test_ingest_and_ask_end_to_end(offline_settings: Settings, fake_api: FakeAPI) -> None:
    conversation = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    fake_api.chat_replies = [cite_golden_retriever]

    with MemorySystem.from_settings(offline_settings, http_client=fake_api.client()) as system:
        report = system.ingest(conversation)
        assert report.turns_added == sum(len(s.turns) for s in conversation.sessions)
        assert report.units_added == system.vectors.count("demo") == 9

        answer = system.ask(
            "demo", "Max 是什么品种的狗？", question_time=datetime(2025, 10, 10), k=5
        )
        assert answer.answer == "金毛"
        assert answer.evidence_sufficient
        [cited] = system.db.get_units("demo", [answer.refs[answer.citations[0]]])
        assert "金毛" in cited.text

        request = fake_api.chat_requests[-1]
        assert request["thinking"] == {"type": "disabled"}
        assert "Current date: 2025-10-10" in request["messages"][-1]["content"]

        again = system.ingest(conversation)  # 重复导入不产生新数据
        assert (again.turns_added, again.units_added) == (0, 0)

    [trace_file] = list(offline_settings.paths.traces.glob("*.jsonl"))
    traces = [json.loads(line) for line in trace_file.read_text("utf-8").splitlines()]
    ask_trace = next(t for t in traces if t["name"] == "ask")
    assert {s["name"] for s in ask_trace["spans"]} >= {"retrieve", "read"}
    assert any(e["kind"] == "llm" for e in ask_trace["events"])


def test_ingest_with_a_repeated_session_id_is_unchanged(
    offline_settings: Settings, fake_api: FakeAPI
) -> None:
    """LongMemEval 有 13 个样本同一个会话 ID 出现两次（内容相同、日期不同）。导入整段对话
    （评测建库的路）不做导入文件那样的"先出现的算"：那一"轮"照旧算两次、写两次向量，向量库里
    留的是后一次的时间，重建的评测空间与以前的逐项相同。数据库里的"轮"是先写的那个。"""

    def said(session_id: str, day: int, text: str) -> SessionIn:
        return SessionIn(
            session_id=session_id,
            started_at=datetime(2023, 5, day, 9),
            turns=[TurnIn(role="user", text=text)],
        )

    sessions = [
        said("dup", 25, "我养了一只猫"),
        said("other", 26, "我住在柏林"),
        said("dup", 28, "我养了一只猫"),
    ]
    sent: list[list[str]] = []
    with MemorySystem.from_settings(offline_settings, http_client=fake_api.client()) as system:
        embed = system.embedder.embed

        def spy(texts: Any, kind: Any) -> Any:
            sent.append(list(texts))
            return embed(texts, kind)

        system.embedder.embed = spy  # type: ignore[method-assign]
        report = system.ingest(Conversation(space_id="lme", sessions=sessions))
        assert sent == [["User: 我养了一只猫", "User: 我住在柏林", "User: 我养了一只猫"]]
        assert (report.turns_added, report.units_added) == (2, 3)
        assert system.vectors.count("lme") == 2
        [unit] = [u for u in system.db.list_units("lme", kinds=["round"]) if "猫" in u.text]
        assert unit.ts == datetime(2023, 5, 25, 9)
        records, _ = system.vectors._client.scroll(
            system.vectors.collection_name("lme"), limit=10, with_payload=True
        )
        [ts] = [r.payload["ts"] for r in records if r.payload and r.payload["unit_id"] == unit.id]
        assert ts == datetime(2023, 5, 28, 9).timestamp()
