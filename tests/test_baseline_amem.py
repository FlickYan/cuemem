"""外部基线 MA（A-MEM）：笔记格式、链接解析、检索结果到证据的转换、持久化。

不装 A-MEM 也能跑：A-MEM 只在 setup / build 里导入，这里测纯函数与替换进 A-MEM 的两个方法；
需要 A-MEM 的端到端写入与检索用 importorskip。"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cuemem.evals.baselines.amem import (
    MAX_TOKENS,
    NOTES,
    TOP_K,
    AMemBaseline,
    Note,
    _consolidate_memories,
    _Controller,
    _find_related_memories,
    clean_analysis,
    expand,
    link_targets,
    load_notes,
    note_content,
    note_document,
    note_text,
    parse_keywords,
    save_notes,
    strings,
    to_hits,
    when_text,
)
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.methods import build_methods

MAY8 = datetime(2023, 5, 8, 13, 56)
MAY25 = datetime(2023, 5, 25, 9, 5)


def _locomo() -> list[EvalSession]:
    later = EvalSession(
        "session_2",
        MAY25,
        [
            EvalTurn("user", "I ran a charity race last Saturday.", "Melanie"),
            EvalTurn("user", "  ", "Caroline"),  # 空话跳过
            EvalTurn("user", "Amazing! I joined a support group.", "Caroline"),
        ],
    )
    first = EvalSession(
        "session_1",
        MAY8,
        [
            EvalTurn("user", "I went to a support group yesterday.", "Caroline"),
            EvalTurn("user", "How was it?", "Melanie"),
        ],
    )
    return [later, first]  # 故意乱序：写入时按时间排


def _note(i: int, *, links: list[Any] | None = None, when: datetime = MAY8) -> Note:
    return Note(
        id=f"id-{i}",
        content=f"Speaker A says : line {i}",
        timestamp=when_text(when),
        session_id="session_1",
        session_time=when,
        keywords=[f"k{i}"],
        context=f"context {i}",
        tags=[f"t{i}"],
        links=links or [],
    )


def test_note_format_follows_the_paper() -> None:
    assert when_text(MAY8) == "1:56 pm on 8 May, 2023"  # LoCoMo 的 date_time 格式
    assert when_text(datetime(2023, 1, 2, 0, 5)) == "12:05 am on 2 January, 2023"
    assert when_text(MAY25) == "9:05 am on 25 May, 2023"
    turn = EvalTurn("user", " I went to a support group. ", "Caroline")
    assert note_content(turn) == "Speaker Caroline says : I went to a support group."
    assert note_content(EvalTurn("assistant", "Sure.")) == "Speaker assistant says : Sure."
    document = note_document("hi", "a chat", ["x", "y"], "t1, t2")
    assert document == "content:hi context:a chat keywords: x, y tags: t1, t2"
    assert note_document("hi", None, None, []) == "content:hi context:General keywords:  tags: "


def test_clean_analysis_keeps_usable_fields() -> None:
    raw = {"keywords": ["support group", " ", 3], "context": ["Talk", "about it"], "tags": "a, b"}
    assert clean_analysis(raw) == {
        "keywords": ["support group", "3"],
        "context": "Talk about it",
        "tags": ["a", "b"],
    }
    assert clean_analysis("not a dict") == {"keywords": [], "context": "General", "tags": []}
    assert strings(None) == [] and strings(7) == ["7"]


def test_link_targets_resolve_positions_and_ids() -> None:
    ids = {"id-0": 0, "id-1": 1, "id-2": 2, "id-3": 3}
    links = [2, "1", "memory index:3", "Memory Index: 2", "id-0", "id-9", 99, -1, True, None, {}]
    assert link_targets(links, ids, 4, own=1) == [2, 3, 0]  # 自己、重复、越界、对不上的都丢掉


def test_expand_follows_links_and_dedupes() -> None:
    notes = [_note(0), _note(1, links=["memory index:3", 0]), _note(2, links=[1]), _note(3)]
    ids = {n.id: i for i, n in enumerate(notes)}
    assert expand([1, 2, 3], notes, ids, k=3) == [(1, False), (3, True), (0, True), (2, False)]
    # 只取前 k 条，每条最多带 k 条链接
    assert expand([2, 1], notes, ids, k=1) == [(2, False), (1, True)]
    assert expand([], notes, ids, k=5) == []


def test_hits_carry_session_dates_and_paper_format() -> None:
    notes = [_note(0), _note(1, when=MAY25)]
    hits = to_hits("locomo_conv-1", notes, [(1, False), (0, True)], {1: 0.8})
    assert [h.unit.id for h in hits] == ["id-1", "id-0"]
    assert [h.route for h in hits] == ["amem", "amem_link"]
    assert hits[0].unit.ts == MAY25 and hits[0].unit.session_id == "session_1"
    assert hits[0].score == 0.8 and hits[1].score == 0.0 and hits[1].rank == 2
    assert hits[0].unit.text == note_text(notes[1])
    assert note_text(notes[1]).splitlines() == [
        "memory content: Speaker A says : line 1",
        "memory context: context 1",
        "memory keywords: k1",
        "memory tags: t1",
    ]


def test_parse_keywords() -> None:
    assert parse_keywords('{"keywords": "support group, Caro"}', "q") == "support group, Caro"
    assert parse_keywords('{"keywords": ["a", "b"]}', "q") == "a, b"
    assert parse_keywords("support cosmos group", "q") == "support cosmos group"  # 不是 JSON
    assert parse_keywords('{"keywords": ""}', "When?") == "When?"
    assert parse_keywords(None, "When?") == "When?"


def test_notes_round_trip(tmp_path: Path) -> None:
    notes = [_note(0, links=["memory index:1", 1]), _note(1, when=MAY25)]
    save_notes(tmp_path / NOTES, notes)
    assert load_notes(tmp_path / NOTES) == notes


def test_find_related_memories_labels_real_positions() -> None:
    """邻居标上在记忆库里的真实序号：A-MEM 的 process_memory 按这个序号改邻居、记链接。"""

    class Retriever:
        def __init__(self) -> None:
            self.upserted: list[tuple[str, str, dict[str, Any]]] = []
            self.consolidations = 0

        def search(self, query: str, k: int = 5) -> dict[str, Any]:
            return {"ids": [["n2", "gone", "n0"][:k]], "distances": [[0.1, 0.2, 0.3][:k]]}

        def upsert(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
            self.upserted += rows

    memories = OrderedDict(
        (
            f"n{i}",
            SimpleNamespace(
                id=f"n{i}",
                content=f"c{i}",
                timestamp="1:56 pm on 8 May, 2023",
                context=f"x{i}",
                keywords=[f"k{i}"],
                tags=[f"t{i}"],
            ),
        )
        for i in range(3)
    )
    system = SimpleNamespace(memories=memories, retriever=Retriever())
    text, indices = _find_related_memories(system, "query", k=5)
    assert indices == [2, 0]
    assert text.splitlines()[0] == (
        "memory index:2\ttalk start time:1:56 pm on 8 May, 2023\tmemory content: c2\t"
        "memory context: x2\tmemory keywords: ['k2']\tmemory tags: ['t2']"
    )
    assert _find_related_memories(SimpleNamespace(memories={}), "q") == ("", [])
    _consolidate_memories(system)
    assert [row[0] for row in system.retriever.upserted] == ["n0", "n1", "n2"]
    assert system.retriever.upserted[1][2]["context"] == "x1"
    assert system.retriever.consolidations == 1


def test_controller_sends_amem_request_and_counts_problems() -> None:
    """请求照 A-MEM 的 OpenAIController（系统提示、temperature），回复上限放宽；
    数失败、被截断与不是 JSON 的回复。"""
    replies: list[Any] = [('{"a": 1}', "stop"), ('{"a": "cut', "length"), RuntimeError("boom")]
    sent: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content, finish = reply
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    controller = _Controller(client, "deepseek-flash")
    schema = {"type": "json_object"}
    assert controller.get_completion("p", schema) == '{"a": 1}'
    assert sent[0]["messages"] == [
        {"role": "system", "content": "You must respond with a JSON object."},
        {"role": "user", "content": "p"},
    ]
    assert sent[0]["temperature"] == 0.7 and sent[0]["max_tokens"] == MAX_TOKENS == 2000
    assert sent[0]["response_format"] == schema and sent[0]["model"] == "deepseek-flash"
    controller.get_completion("p", schema, temperature=0.5)
    assert sent[1]["temperature"] == 0.5
    with pytest.raises(RuntimeError):
        controller.get_completion("p")
    assert "response_format" not in sent[2]
    assert (controller.calls, controller.failed) == (3, 1)
    assert (controller.truncated, controller.invalid_json) == (1, 1)
    assert controller.errors == ["RuntimeError: boom"]


def test_method_is_registered_without_the_package() -> None:
    (method,) = build_methods(["MA"])  # 构造时不导入 A-MEM
    assert isinstance(method, AMemBaseline)
    assert method.name == "MA" and method.system == "amem" and TOP_K == 10


# ---------- 装了 A-MEM 时：经网关端到端写入与检索 ----------


def _fake_model(body: dict[str, Any]) -> str:
    """假的抽取模型：按提示回 A-MEM 的笔记分析、演化决定，或检索用的关键词。"""
    prompt = body["messages"][-1]["content"]
    if "Generate a structured analysis" in prompt:
        content = prompt.rsplit("Content for analysis:", 1)[-1]
        keyword = "support group" if "support group" in content else "chat"
        return json.dumps({"keywords": [keyword], "context": "A talk.", "tags": ["life"]})
    if "memory evolution agent" in prompt:
        neighbors = re.findall(r"memory index:(\d+)", prompt)
        return json.dumps(
            {
                "should_evolve": True,
                "actions": ["strengthen", "update_neighbor"],
                "suggested_connections": [f"memory index:{neighbors[0]}"],
                "tags_to_update": ["linked"],
                "new_context_neighborhood": ["Updated context."] * len(neighbors),
                "new_tags_neighborhood": [["evolved"]] * len(neighbors),
            }
        )
    if "generate several keywords" in prompt:
        return json.dumps({"keywords": "support group"})
    return "{}"


def test_build_and_retrieve_through_the_gateway(
    offline_settings: Any, fake_api: Any, tmp_path: Path
) -> None:
    """装了 A-MEM 时：每句一条笔记，分析与演化经网关调用，写完存盘；重新打开后检索带出链接。"""
    pytest.importorskip("agentic_memory.memory_system")
    pytest.importorskip("chromadb")
    from cuemem.llm import Ledger, LLMClient, ResponseCache

    fake_api.chat_replies = [_fake_model]
    llm = LLMClient(
        offline_settings,
        cache=ResponseCache(None),
        ledger=Ledger(None),
        http_client=fake_api.client(),
    )
    baseline = AMemBaseline()
    baseline._setup(llm)
    key, sessions = "locomo_conv-1", _locomo()
    directory = baseline.root / key
    question = "When did Caroline go to a support group?"
    sample = Sample("locomo", "q1", "conv-1", question, "", "x", "x", False, MAY25)
    try:
        built = baseline._ensure_built(key, directory, sessions)
        assert built is not None and built["memories"] == 4
        assert built["failed_calls"] == 0 and built["invalid_json"] == 0
        assert built["truncated"] == 0 and built["max_tokens"] == MAX_TOKENS
        assert fake_api.chat_requests[0]["max_tokens"] == MAX_TOKENS  # 网关照传给上游
        assert built["amem_calls"] == 7 and built["chat_calls"] == 7  # 4 次分析 + 3 次演化
        assert built["evolved"] == 3 and built["links"] >= 3 and built["unresolved_links"] == 0
        notes = load_notes(directory / NOTES)
        assert [n.session_id for n in notes] == ["session_1"] * 2 + ["session_2"] * 2
        assert notes[0].content == "Speaker Caroline says : I went to a support group yesterday."
        assert notes[0].timestamp == "1:56 pm on 8 May, 2023"
        assert notes[0].tags == ["evolved"] and notes[0].context == "Updated context."
        assert notes[3].keywords == ["support group"] and notes[3].tags == ["linked"]

        baseline.close(key)  # 模拟重新运行：从磁盘打开
        hits, details = baseline.retrieve(key, directory, sample, sessions)
        assert details["amem_query"] == "support group"
        assert details["amem_retrieved"] == 4 and len(hits) == 4  # 只有 4 条，全取到
        assert details["amem_search"]["chat_calls"] == 1
        by_id = {h.unit.id: h for h in hits}
        assert by_id[notes[0].id].unit.ts == MAY8 and by_id[notes[3].id].unit.ts == MAY25
        assert by_id[notes[0].id].unit.text.startswith("memory content: Speaker Caroline says")
        usage = baseline.gateway.usage(baseline.bucket(key, "search"))
        assert usage["embedding_calls"] >= 1 and usage["chat_calls"] == 1
    finally:
        baseline.close(key)
        baseline.gateway.stop()
