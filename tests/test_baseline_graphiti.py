"""外部基线 MG（Graphiti）的测试：不联网、不调用模型。

纯函数（发言格式、时间、证据转换、episode 账本）直接测，不需要 graphiti-core；最后一个测试把
Graphiti 真正接到评测网关上（模型是脚本化的假接口），要装上它才跑：
    uv run --with "graphiti-core[kuzu]==0.30.2" pytest tests/test_baseline_graphiti.py
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.evals.baselines.graphiti import (
    MAX_MESSAGE_CHARS,
    EpisodeLedger,
    GraphitiBaseline,
    entity_hit,
    episode_body,
    fact_hit,
    fact_text,
    group_id,
    keyword_query,
    naive,
    reference_time,
)
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.methods import MethodContext, build_methods
from cuemem.llm import Ledger, LLMClient, ResponseCache
from cuemem.read.reader import Reader

MAY7 = datetime(2023, 5, 7, 13, 56)
MAY9 = datetime(2023, 5, 9, 18, 0)


def test_episode_body_uses_graphiti_message_format() -> None:
    named = EvalTurn(role="user", text="  I moved to Berlin.  ", speaker="Ana")
    assert episode_body(named) == "Ana: I moved to Berlin."
    assert episode_body(EvalTurn(role="assistant", text="Noted.")) == "assistant: Noted."
    assert episode_body(EvalTurn(role="user", text="   ")) is None
    long = episode_body(EvalTurn(role="user", text="x" * (MAX_MESSAGE_CHARS + 50)))
    assert long == "user: " + "x" * MAX_MESSAGE_CHARS  # Zep 的 LongMemEval 评测也截到 8000


def test_reference_time_keeps_the_day_and_orders_turns() -> None:
    first, third = reference_time(MAY7, 0), reference_time(MAY7, 2)
    assert first == MAY7.replace(tzinfo=UTC) and first < third
    assert (third - first).total_seconds() == 2 and third.date() == MAY7.date()
    aware = datetime(2023, 5, 7, 13, 56, tzinfo=UTC)
    assert reference_time(aware, 1).tzinfo is UTC
    assert naive(reference_time(MAY7, 0)) == MAY7


def test_group_id_and_keyword_query() -> None:
    assert group_id("locomo_conv-42") == "locomo_conv-42"
    assert group_id("longmemeval_s_abc.1") == "longmemeval_s_abc_1"
    # Kuzu 的全文索引不去问号："live?" 匹配不到 "lives"
    assert keyword_query("Where does Ana live?") == "Where does Ana live"
    assert keyword_query("When did Ana's dog, Max, arrive?") == "When did Ana s dog Max arrive"
    assert keyword_query("?!") == "?!"


def _ledger(tmp_path: Path) -> EpisodeLedger:
    ledger = EpisodeLedger(tmp_path / "episodes.json")
    ledger.add("e1", "session_1", 3, reference_time(MAY7, 3), ["ana", "berlin"])
    ledger.add("e2", "session_2", 0, reference_time(MAY9, 0), ["ana"])
    ledger.add("e3", "session_1", 5, reference_time(MAY7, 5), [])
    return ledger


def test_ledger_round_trip_and_sources(tmp_path: Path) -> None:
    _ledger(tmp_path).save()
    ledger = EpisodeLedger.load(tmp_path / "episodes.json")
    assert ledger.episodes["e1"] == ("session_1", 3, datetime(2023, 5, 7, 13, 56, 3))
    # 一条事实来自多条 episode：取最早那条的会话，以及该会话里提到它的所有发言
    session, turns, said = ledger.source(["e2", "e3", "e1", "unknown"])
    assert (session, turns, said) == ("session_1", [3, 5], datetime(2023, 5, 7, 13, 56, 3))
    assert ledger.source(["unknown"]) == ("", [], None)
    assert ledger.last_mention("ana") == ("session_2", MAY9)  # 最近一次提到
    assert ledger.last_mention("berlin")[0] == "session_1"
    assert ledger.last_mention("nobody") == ("", None)
    assert EpisodeLedger.load(tmp_path / "missing.json").episodes == {}


def test_facts_and_entities_become_dated_evidence(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    moved = SimpleNamespace(
        uuid="f1",
        fact="Ana moved to Berlin",
        valid_at=datetime(2023, 4, 30, tzinfo=UTC),  # 事件时间："上周"
        invalid_at=None,
        episodes=["e1"],
        reference_time=reference_time(MAY7, 3),
        created_at=datetime(2026, 10, 14, tzinfo=UTC),
    )
    hit = fact_hit("locomo_conv-1", moved, 0.5, 1, ledger)
    assert hit.unit.text == "Ana moved to Berlin (Date range: 2023-04-30 - present)"
    assert hit.unit.ts == datetime(2023, 4, 30) and hit.unit.ts.tzinfo is None
    assert hit.unit.session_id == "session_1" and hit.unit.meta["turn_indices"] == [3]
    assert hit.route == "graphiti_fact" and hit.rank == 1 and hit.score == 0.5

    undated = SimpleNamespace(**{**vars(moved), "valid_at": None, "uuid": "f2"})
    hit = fact_hit("locomo_conv-1", undated, 0.1, 2, ledger)
    assert hit.unit.ts == datetime(2023, 5, 7, 13, 56, 3)  # 没有事件时间：取说这句话的时间
    assert "(Date range: date unknown - present)" in hit.unit.text
    ended = fact_text("Ana lives in Paris", datetime(2022, 1, 1), datetime(2023, 4, 30, tzinfo=UTC))
    assert ended == "Ana lives in Paris (Date range: 2022-01-01 - 2023-04-30)"

    ana = SimpleNamespace(
        uuid="ana", name="Ana", summary="Ana lives in Berlin.", created_at=datetime(2026, 1, 1)
    )
    hit = entity_hit("locomo_conv-1", ana, 0.3, 1, ledger)
    assert hit.unit.text == "About Ana (entity summary): Ana lives in Berlin."
    assert hit.unit.kind == "entity" and hit.unit.ts == MAY9
    assert hit.unit.session_id == "session_2" and "turn_indices" not in hit.unit.meta
    stranger = SimpleNamespace(uuid="x", name="X", summary="", created_at=MAY7)
    assert entity_hit("k", stranger, 0.0, 2, ledger).unit.text.endswith("(no summary)")


def test_graph_built_with_another_embedder_is_refused(tmp_path: Path) -> None:
    # 换了向量模型（比如 deepseek 预设的本机 bge-small 换成通义）：图里的向量没法和查询比
    method = GraphitiBaseline()
    method.embedder = SimpleNamespace(name="qwen-embedding", dim=1024)  # type: ignore[assignment]
    (tmp_path / "build.json").write_text(json.dumps({"embedder": "BAAI/bge-small:384"}), "utf-8")
    question = SimpleNamespace(question="Where does Ana live?")
    with pytest.raises(RuntimeError, match="删掉这个目录重建"):
        method.retrieve("locomo_conv-1", tmp_path, question, [])  # type: ignore[arg-type]


def test_method_is_registered_without_the_package() -> None:
    # 外部包不是依赖：构造时不导入 graphiti_core，第一次回答（setup）时才导入
    (method,) = build_methods(["MG"])
    assert isinstance(method, GraphitiBaseline)
    assert (method.name, method.system) == ("MG", "graphiti")
    assert method.bucket("locomo_conv-42", "search") == "graphiti.search.locomo_conv-42"


# ---------- 接上真正的 Graphiti（需要 graphiti-core[kuzu]） ----------

READER_MARK = "personal assistant with long-term memory"


def _fake_model(body: dict[str, Any]) -> str:
    """脚本化的模型：认出 Graphiti 的各个提示（网关把 json_schema 写进了系统提示），
    从当前消息里"抽取"Ana 与 Berlin；回答提示返回 Berlin。"""
    messages = body["messages"]
    system = str(messages[0]["content"])
    if READER_MARK in system:
        return json.dumps({"answer": "Berlin", "evidence_sufficient": True, "citations": ["m1"]})
    assert body["response_format"] == {"type": "json_object"}
    found = re.search(r"conforming to this JSON Schema:\s*(\{.*\})\s*$", system, re.S)
    schema = json.loads(found.group(1)) if found else {}
    last = str(messages[-1]["content"])
    current = last.split("<CURRENT MESSAGE>")[-1].split("</CURRENT MESSAGE>")[0]
    names = [name for name in ("Ana", "Berlin") if name in current]
    if schema.get("title") == "ExtractedEntities":
        entities = [{"name": name, "entity_type_id": 0} for name in names]
        return json.dumps({"extracted_entities": entities})
    if schema.get("title") == "ExtractedEdges":
        edges = [
            {
                "source_entity_name": "Ana",
                "target_entity_name": "Berlin",
                "relation_type": "LIVES_IN",
                "fact": "Ana lives in Berlin",
                "valid_at": "2023-04-30T00:00:00Z",
                "invalid_at": None,
                "episode_indices": [0],
            }
        ]
        return json.dumps({"edges": edges if len(names) == 2 else []})
    return json.dumps(_empty(schema))


def _empty(schema: dict[str, Any]) -> dict[str, Any]:
    """其余提示（消歧、摘要……）：按 schema 给出"什么也没有"的回答。"""
    blank = {"array": [], "string": "", "integer": -1, "number": 0, "boolean": False}
    properties = schema.get("properties", {})
    return {
        name: blank.get(str(properties[name].get("type")))
        for name in schema.get("required", [])
        if name in properties
    }


def test_graphiti_end_to_end_through_the_gateway(
    offline_settings: Settings, fake_api: FakeAPI
) -> None:
    pytest.importorskip("graphiti_core")
    pytest.importorskip("kuzu")
    fake_api.chat_replies = [_fake_model]
    llm = LLMClient(
        offline_settings,
        cache=ResponseCache(None),
        ledger=Ledger(None),
        http_client=fake_api.client(),
    )
    sessions = [
        EvalSession(
            "session_1",
            MAY7,
            [
                EvalTurn(role="user", text="I moved to Berlin last week!", speaker="Ana"),
                EvalTurn(role="user", text="", speaker="Bo"),  # 空发言不写入，但占着下标
                EvalTurn(role="user", text="Congrats Ana, Berlin is great.", speaker="Bo"),
            ],
        ),
        EvalSession("session_2", MAY9, [EvalTurn(role="user", text="Hi again", speaker="Ana")]),
    ]
    sample = Sample(
        dataset="locomo",
        sample_id="conv-1-q0",
        space_key="conv-1",
        question="Where does Ana live?",
        answer="Berlin",
        question_type="single-hop",
        ability="single-hop",
        is_abstention=False,
        question_time=MAY9,
    )
    method = GraphitiBaseline()
    ctx = MethodContext(sample, sessions, Reader(llm), None, "locomo:conv-1")
    try:
        first = method.answer(ctx)
        build = first.details["graphiti_build"]
        assert build["episodes"] == 3 and build["failed_episodes"] == []
        assert build["entities"] == 2 and build["facts"] >= 1 and build["chat_calls"] > 0
        assert build["embedder"] == f"{method.embedder.name}:{method.embedder.dim}"
        assert first.response == "Berlin"
        fact = next(h for h in first.hits if h.route == "graphiti_fact")
        assert fact.unit.text == "Ana lives in Berlin (Date range: 2023-04-30 - present)"
        assert fact.unit.session_id == "session_1" and 0 in fact.unit.meta["turn_indices"]
        names = {h.unit.text.split(" (entity")[0] for h in first.hits if h.route != "graphiti_fact"}
        assert names == {"About Ana", "About Berlin"}
        assert first.details["search_usage"]["embedding_calls"] == 1

        method.close("locomo_conv-1")  # 关掉再打开：库与账本都从磁盘读
        second = method.answer(ctx)
        assert "graphiti_build" not in second.details
        assert [h.unit.text for h in second.hits] == [h.unit.text for h in first.hits]
        directory = method.root / "locomo_conv-1"
        assert json.loads((directory / "build.json").read_text("utf-8"))["episodes"] == 3
    finally:
        method.close("locomo_conv-1")
        method.gateway.stop()
