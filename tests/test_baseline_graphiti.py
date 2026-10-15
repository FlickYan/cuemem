"""外部基线 MG（Graphiti）的测试：不联网、不调用模型。

纯函数（发言格式、时间、证据转换、按轮计的检索指标、episode 账本、告警计数、重建判断）直接测，
不需要 graphiti-core；最后几个测试把 Graphiti 真正接到评测网关上（模型与重排都是脚本化的
假接口），要装上它才跑：
    uv run --with "graphiti-core[kuzu]==0.30.2" pytest tests/test_baseline_graphiti.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import REJECT, FakeAPI
from pydantic import BaseModel

from cuemem.config import Settings
from cuemem.core.trace import start_trace
from cuemem.evals.baselines.graphiti import (
    _WARNINGS,
    BUILD_PROTOCOL,
    MAX_MESSAGE_CHARS,
    EpisodeLedger,
    GraphitiBaseline,
    SchemaMismatch,
    WarningCounter,
    _resampling_client_type,
    check_episodes,
    check_reply,
    entity_hit,
    episode_body,
    fact_hit,
    fact_text,
    group_id,
    keyword_query,
    mark_rounds,
    naive,
    ranked,
    reference_time,
    round_starts,
    warning_key,
)
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import memory_hit
from cuemem.evals.methods import MethodContext, build_methods
from cuemem.evals.metrics import retrieval_metrics
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
    # 两段会话开始时间相同（LongMemEval 常见）：后一段接在前一段之后，不交错
    last_of_first = reference_time(MAY7, 5)
    second = reference_time(MAY7, 0, after=last_of_first)
    assert (second - last_of_first).total_seconds() == 1 and second.date() == MAY7.date()
    later = reference_time(MAY9, 0, after=last_of_first)  # 本来就更晚：不动
    assert later == MAY9.replace(tzinfo=UTC)


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
    assert ledger.last_mention("ana") == MAY9  # 最近一次提到
    assert ledger.last_mention("berlin") == datetime(2023, 5, 7, 13, 56, 3)
    assert ledger.last_mention("nobody") is None
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
    # 实体摘要跨会话累积：不标会话，不算进会话召回
    assert hit.unit.session_id == "" and "turn_indices" not in hit.unit.meta
    stranger = SimpleNamespace(uuid="x", name="X", summary="", created_at=MAY7)
    assert entity_hit("k", stranger, 0.0, 2, ledger).unit.text.endswith("(no summary)")


def _fact(session: str, turns: list[int], rank: int) -> Any:
    hit = memory_hit("k", f"fact:{rank}", "fact", MAY7, rank, session_id=session)
    hit.unit.meta["turn_indices"] = turns
    return hit


def test_metrics_count_each_round_once() -> None:
    # LoCoMo：两人都是 user，每条发言一轮。三条事实出自同一条证据发言，不能算三次
    locomo = EvalSession("session_7", MAY7, [EvalTurn(role="user", text="t")] * 4)
    hits = [_fact("session_7", [0], 1), _fact("session_7", [0], 2), _fact("session_7", [0, 2], 3)]
    mark_rounds(hits, [locomo])
    assert [h.unit.meta["turn_indices"] for h in hits] == [[0], [], [2]]
    metrics = retrieval_metrics(
        hits,
        evidence_sessions=["session_7"],
        evidence_turns={("session_7", 0), ("session_7", 2)},
        relevant_units=2,
    )
    assert metrics["turn_recall_all"] == 1.0 and metrics["turn_ndcg"] <= 1.0

    duplicates = [_fact("session_7", [0], rank) for rank in (1, 2, 3)]
    mark_rounds(duplicates, [locomo])
    metrics = retrieval_metrics(
        duplicates,
        evidence_sessions=["session_7"],
        evidence_turns={("session_7", 0), ("session_7", 2)},
        relevant_units=2,
    )
    assert metrics["turn_recall_all"] == 0.5 and metrics["turn_ndcg"] < 1.0  # 原来是 1.0、1.31

    # LongMemEval：一问一答为一轮，与 CueMem 的命中单位、relevant_units 相同
    chat = EvalSession(
        "s1",
        MAY7,
        [EvalTurn(role=role, text="t") for role in ("user", "assistant", "user", "assistant")],
    )
    assert round_starts(chat) == [0, 0, 2, 2]
    hits = [_fact("s1", [3], 1), _fact("s1", [2], 2), _fact("s1", [0], 3), _fact("other", [9], 4)]
    mark_rounds(hits, [chat])
    assert [h.unit.meta["turn_indices"] for h in hits] == [[2, 3], [], [0, 1], [9]]
    metrics = retrieval_metrics(
        hits, evidence_sessions=["s1"], evidence_turns={("s1", 2)}, relevant_units=1
    )
    assert metrics["turn_recall_all"] == 1.0 and metrics["turn_ndcg"] == 1.0


class _Edge(BaseModel):
    fact: str
    episode_indices: list[int] = []


class _Edges(BaseModel):
    """ExtractedEdges 的缩影：一个必填的列表字段，元素里有整数列表。"""

    edges: list[_Edge]


def test_replies_that_do_not_match_the_schema_are_resampled() -> None:
    """通义的 JSON 模式不保证 schema：裸列表、整数列表里的 null 当作解析失败（Graphiti 会重试）。"""
    check_reply({"edges": [{"fact": "Ana lives in Berlin", "episode_indices": [0]}]}, _Edges)
    check_reply({"edges": [], "note": "extra keys are ignored"}, _Edges)
    for bad in ([], [{"fact": "x"}], {"edges": [{"fact": "x", "episode_indices": [None]}]}, {}):
        with pytest.raises(SchemaMismatch) as caught:
            check_reply(bad, _Edges)
        assert isinstance(caught.value, json.JSONDecodeError)  # Graphiti 只对这类错误重试


def test_refused_episodes_have_their_own_allowance() -> None:
    """通义的内容审核会误拒平常的发言：跳过并计数，超过 max(2, 5%) 才作废；其他失败仍是 1%。"""
    check_episodes(629, ["s1#1: x"] * 6, ["s2#3: refused"] * 31)
    with pytest.raises(RuntimeError, match="写入失败"):
        check_episodes(629, ["s1#1: x"] * 7, [])
    with pytest.raises(RuntimeError, match="最多容许 31 条"):
        check_episodes(629, [], ["s2#3: refused"] * 32)
    check_episodes(3, [], ["s2#3: refused"] * 2)  # 很短的历史也容许两条
    with pytest.raises(RuntimeError, match="内容审核"):
        check_episodes(3, [], ["s2#3: refused"] * 3)


def test_cross_encoder_ranking_follows_graphiti_convention() -> None:
    assert ranked(["a", "b", "c"], [0.1, 0.9, 0.1]) == [("b", 0.9), ("a", 0.1), ("c", 0.1)]
    assert ranked([], []) == []


def test_graphiti_warnings_are_counted_per_build() -> None:
    handler = WarningCounter()
    unmatched = {
        "msg": "Source entity not found in nodes for edge relation: %s",
        "levelname": "WARNING",
    }
    decode = {"msg": "Error in generating LLM response: line 1 column 5", "levelname": "ERROR"}
    handler.emit(logging.makeLogRecord(unmatched))  # 不在写入中：不计、不报错
    counts: Counter[str] = Counter()
    token = _WARNINGS.set(counts)
    try:
        for relation in ("LIKES", "OWNS"):
            handler.emit(logging.makeLogRecord({**unmatched, "args": (relation,)}))
        handler.emit(logging.makeLogRecord(decode))
    finally:
        _WARNINGS.reset(token)
    assert counts == {
        "WARNING: Source entity not found in nodes for edge relation: %s": 2,
        "ERROR: Error in generating LLM response: line # column #": 1,
    }
    record = logging.makeLogRecord(
        {"msg": "Error executing Kuzu query: x\nMATCH ...", "levelname": "ERROR"}
    )
    assert warning_key(record) == "ERROR: Error executing Kuzu query: x"


def test_graph_from_another_setup_is_kept_or_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """换了写入协议（BUILD_PROTOCOL）：重建。换了抽取模型或向量模型：另一个协议的图，报错、不删
    （评测空间的目录不分协议）。早先的写入记录只有 graphiti_protocol、extractor 与 embedder
    （pds 的 locomo_conv-42 连 protocol 也没有），照样认得。"""
    from cuemem.errors import ConfigMismatchError

    method = GraphitiBaseline()
    method.gateway = SimpleNamespace(usage=lambda bucket: {"spent": 0.0})  # type: ignore[assignment]
    method.protocol = "qwen3.8-flash|qwen3.7-text-embedding:1024"  # 基类的指纹，_setup() 里设
    built: list[str] = []
    monkeypatch.setattr(method, "build", lambda key, directory, sessions: built.append(key) or {})
    directory = tmp_path / "locomo_conv-1"
    directory.mkdir()
    record = directory / "build.json"
    current = {"graphiti_protocol": BUILD_PROTOCOL, "extractor": "qwen3.8-flash"}
    current["embedder"] = "qwen3.7-text-embedding:1024"
    for legacy in (current, {**current, "protocol": method.protocol}):
        record.write_text(json.dumps(legacy), "utf-8")
        assert method._ensure_built("locomo_conv-1", directory, []) is None and built == []
    for other in (
        {"extractor": "deepseek-flash"},
        {"embedder": "x:384"},
        {"protocol": "deepseek-flash|BAAI/bge-small-en-v1.5:384"},
    ):
        record.write_text(json.dumps({**current, **other}), "utf-8")
        with pytest.raises(ConfigMismatchError, match="不复用，也不覆盖"):
            method._ensure_built("locomo_conv-1", directory, [])
        assert json.loads(record.read_text("utf-8")) == {**current, **other}
    assert built == []
    record.write_text(json.dumps({**current, "graphiti_protocol": 1}), "utf-8")
    stats = method._ensure_built("locomo_conv-1", directory, [])
    assert stats is not None and built == ["locomo_conv-1"]
    assert stats["build_tag"] == f"graphiti_protocol={BUILD_PROTOCOL}"
    assert stats["replaced"] == {"build_tag": "graphiti_protocol=1"}  # 换掉的是哪一版
    assert method._ensure_built("locomo_conv-1", directory, []) is None


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
    if "passport" in current:  # 内容审核误拒（主模型与备用模型都拒）
        return REJECT
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


@pytest.mark.parametrize("rerank", [True, False], ids=["cross_encoder", "rrf"])
def test_graphiti_end_to_end_through_the_gateway(
    offline_settings: Settings,
    fake_api: FakeAPI,
    monkeypatch: pytest.MonkeyPatch,
    rerank: bool,
) -> None:
    pytest.importorskip("graphiti_core")
    pytest.importorskip("kuzu")
    if not rerank:  # 没有重排模型（deepseek 预设）：事实检索退回 RRF
        monkeypatch.setattr("cuemem.memory.build_reranker", lambda *args, **kwargs: None)
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
        EvalSession(
            "session_3",
            datetime(2023, 5, 9, 20, 0),
            [EvalTurn(role="user", text="My passport expires soon.", speaker="Ana")],
        ),
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
        with start_trace("answer") as trace:
            first = method.answer(ctx)
        build = first.details["graphiti_build"]
        assert build["episodes"] == 3 and build["failed_episodes"] == []
        # 被内容审核拒绝的发言跳过并计数，不算失败（通义预设没有备用模型）
        (refused,) = build["refused_episodes"]
        assert refused.startswith("session_3#0: BadRequestError") and build["refusals"] >= 1
        assert build["entities"] == 2 and build["facts"] >= 1 and build["chat_calls"] > 0
        assert build["protocol"] == method.protocol
        assert build["protocol"].endswith(f"|{method.embedder.name}:{method.embedder.dim}")
        assert build["build_tag"] == f"graphiti_protocol={BUILD_PROTOCOL}"
        assert isinstance(build["warnings"], dict)
        assert first.response == "Berlin"
        fact = next(h for h in first.hits if h.route == "graphiti_fact")
        assert fact.unit.text == "Ana lives in Berlin (Date range: 2023-04-30 - present)"
        assert fact.unit.session_id == "session_1" and 0 in fact.unit.meta["turn_indices"]
        names = {h.unit.text.split(" (entity")[0] for h in first.hits if h.route != "graphiti_fact"}
        assert names == {"About Ana", "About Berlin"}
        assert all(h.unit.session_id == "" for h in first.hits if h.route == "graphiti_entity")
        # 查询向量由 CueMem 的向量器在这道题里算（query 模式），检索时不再经网关
        assert first.details["search_usage"]["embedding_calls"] == 0
        candidates = first.details["candidates"]
        assert set(candidates) == {"fact_bm25", "fact_cosine", "entity_bm25", "entity_cosine"}
        assert candidates["fact_bm25"] >= 1 and candidates["entity_bm25"] >= 1
        # 脚本化的模型有一次给的实体名对不上，Graphiti 丢掉那条事实并告警：计进写入记录
        unmatched = "WARNING: Target entity not found in nodes for edge relation: %s"
        assert build["warnings"][unmatched] == 1
        (logged,) = [k for k in build["warnings"] if k != unmatched]  # 被拒那次的错误日志
        assert logged.startswith("ERROR: Error in generating LLM response")
        kinds = {event["kind"] for event in trace.events}
        if rerank:
            assert first.details["edge_reranker"] == "cross_encoder:qwen3-rerank"
            # 重排对原问题打分（不是去掉标点的 BM25 查询），用量记进这道题的 trace
            assert fake_api.rerank_requests[-1]["input"]["query"] == "Where does Ana live?"
            assert "rerank" in kinds
        else:
            assert first.details["edge_reranker"] == "rrf" and not fake_api.rerank_requests
            assert "rerank" not in kinds

        method.close("locomo_conv-1")  # 关掉再打开：库与账本都从磁盘读
        second = method.answer(ctx)
        assert "graphiti_build" not in second.details
        assert [h.unit.text for h in second.hits] == [h.unit.text for h in first.hits]
        directory = method.root / "locomo_conv-1"
        assert json.loads((directory / "build.json").read_text("utf-8"))["episodes"] == 3
    finally:
        method.close("locomo_conv-1")
        method.gateway.stop()


def test_retries_resample_instead_of_replaying_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # Graphiti 对解析不了的回复原样重发；网关缓存会还回同一个坏回复。重试的请求要换个缓存键
    pytest.importorskip("graphiti_core")
    import tenacity
    from graphiti_core.llm_client.client import LLMClient as GraphitiLLM
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.prompts.models import Message

    retrying = GraphitiLLM._generate_response_with_retry.retry
    monkeypatch.setattr(retrying, "wait", tenacity.wait_none())  # 不等退避
    sent: list[int] = []
    replies = iter(["not json", '{"ok": true}'])

    async def create(**kwargs: Any) -> Any:
        sent.append(kwargs["max_tokens"])
        message = SimpleNamespace(content=next(replies))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client = _resampling_client_type()(LLMConfig(api_key="x", model="m"), client=fake)
    messages = [Message(role="system", content="Extract."), Message(role="user", content="Hi")]
    assert asyncio.run(client.generate_response(messages)) == {"ok": True}
    assert sent == [client.max_tokens, client.max_tokens - 1] and client.attempts == {}

    # 回复是 JSON 却不合 schema（通义回了裸列表）：同样换个缓存键重新采样，而不是整条 episode 失败
    from graphiti_core.prompts.extract_edges import ExtractedEdges

    sent.clear()
    replies = iter(["[]", '{"edges": []}'])
    messages = [Message(role="system", content="Extract."), Message(role="user", content="Hi")]
    reply = asyncio.run(client.generate_response(messages, response_model=ExtractedEdges))
    assert reply == {"edges": []}
    assert sent == [client.max_tokens, client.max_tokens - 1] and client.attempts == {}
