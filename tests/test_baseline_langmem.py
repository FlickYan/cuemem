"""外部基线 ML（LangMem）：消息格式、写入记录、证据转换与装回存储。全部离线。

纯函数直接测；需要 langmem 的端到端测试（经网关写入与检索）在没装 langmem 时跳过，用
uv run --with langmem==0.0.30 --with langchain-openai pytest tests/test_baseline_langmem.py 跑。
"""

from __future__ import annotations

import json
import random
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.evals.baselines import langmem as ml
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.methods import MethodContext, build_methods
from cuemem.llm import Ledger, LLMClient, ResponseCache
from cuemem.read.reader import Reader

DAY1 = datetime(2026, 8, 1, 10, 30)
DAY2 = datetime(2026, 8, 5, 18, 0)
IDENTITY = {"extractor": "deepseek-flash", "embedder": "hash", "dim": 2}


def _two_people() -> list[EvalSession]:
    return [
        EvalSession(
            "session_1",
            DAY1,
            [
                EvalTurn(role="user", text="I moved to Berlin last week.", speaker="Ana"),
                EvalTurn(role="user", text="  ", speaker="Ben"),
                EvalTurn(role="user", text="Welcome! How is it?", speaker="Ben"),
            ],
        ),
        EvalSession(
            "session_2",
            DAY2,
            [EvalTurn(role="user", text="I started a job at a bakery.", speaker="Ana")],
        ),
    ]


def test_session_messages_carry_the_date_and_the_speaker() -> None:
    messages = ml.session_messages(_two_people()[0])
    assert messages == [
        {"role": "user", "content": "2026-08-01 (Sat) 10:30 | Ana: I moved to Berlin last week."},
        {"role": "user", "content": "2026-08-01 (Sat) 10:30 | Ben: Welcome! How is it?"},
    ]
    chat = EvalSession(
        "s",
        DAY2,
        [EvalTurn(role="user", text="hi"), EvalTurn(role="assistant", text="Hello!")],
    )
    assert [m["role"] for m in ml.session_messages(chat)] == ["user", "assistant"]
    assert ml.session_messages(chat)[1]["content"] == "2026-08-05 (Wed) 18:00 | Hello!"


def test_record_writes_keeps_first_and_last_session() -> None:
    first, second = _two_people()
    written: dict[str, dict[str, str]] = {}
    puts = [{"namespace": ("memories",), "key": "k1", "value": {"kind": "Memory"}}]
    assert ml.record_writes(written, puts, first) == 0
    new = {"namespace": ("memories",), "key": "k2", "value": {"kind": "Memory"}}
    assert ml.record_writes(written, [*puts, new], second) == 1
    assert written["k1"] == {
        "first_session": "session_1",
        "first_time": DAY1.isoformat(),
        "session": "session_2",
        "time": DAY2.isoformat(),
    }
    assert written["k2"]["first_session"] == "session_2"


def test_hits_are_dated_by_the_session_that_wrote_them() -> None:
    value = {"kind": "Memory", "content": {"content": "Ana lives in Berlin."}}
    once = {
        "first_session": "session_1",
        "first_time": DAY1.isoformat(),
        "session": "session_1",
        "time": DAY1.isoformat(),
    }
    hit = ml.to_hit("conv", "k1", value, once, 3, 0.8, DAY2)
    assert hit.unit.text == "Ana lives in Berlin."
    assert hit.unit.ts == DAY1 and hit.unit.session_id == "session_1"
    assert hit.rank == 3 and hit.score == 0.8 and hit.route == "langmem"

    updated = {**once, "session": "session_2", "time": DAY2.isoformat()}
    hit = ml.to_hit("conv", "k1", value, updated, 1, None, DAY2)
    assert hit.unit.ts == DAY2 and hit.unit.session_id == "session_2"
    assert hit.unit.text.endswith("(first recorded 2026-08-01, last updated 2026-08-05)")

    other = ml.to_hit("conv", "k9", {"kind": "Note", "content": {"a": 1}}, None, 1, None, DAY2)
    assert other.unit.text == '{"a": 1}' and other.unit.ts == DAY2


def test_pack_skips_what_does_not_fit_but_keeps_one() -> None:
    sizes = [400, 4000, 400, 400]  # 约 100、1000、100、100 tokens
    hits = [
        ml.to_hit("c", f"k{i}", {"content": {"content": "x" * n}}, None, i, None, DAY1)
        for i, n in enumerate(sizes, start=1)
    ]
    kept, tokens = ml.pack(hits, 250)
    assert [h.unit.id for h in kept] == ["k1", "k3"] and tokens == 202
    kept, _ = ml.pack(hits[1:], 10)
    assert [h.unit.id for h in kept] == ["k2"]


def test_snapshot_is_json() -> None:
    item = SimpleNamespace(value={"kind": "Memory"}, created_at=DAY1, updated_at=DAY2)
    payload = ml.snapshot(
        {("memories",): {"k1": item}},
        {("memories",): {"k1": {"$": [0.5, 0.25]}}},
        {"k1": {"session": "session_1"}},
        IDENTITY,
    )
    row = json.loads(json.dumps(payload))["items"][0]
    assert row["namespace"] == ["memories"] and row["vectors"] == {"$": [0.5, 0.25]}
    assert row["written"] == {"session": "session_1"} and row["updated_at"] == DAY2.isoformat()


def test_store_written_by_other_models_is_refused(tmp_path: Path) -> None:
    path = tmp_path / ml.STORE_FILE
    path.write_text(json.dumps({"identity": IDENTITY, "items": []}), "utf-8")
    other = {**IDENTITY, "extractor": "qwen3.8-flash"}
    with pytest.raises(RuntimeError, match="换了模型"):
        ml.load_store(path, None, other)


def test_seeded_uuid_only_inside_a_build() -> None:
    shim = ml._SeededUuid()
    assert shim.uuid4() != shim.uuid4()  # 不在写入线程里：真随机
    assert shim.uuid5(shim.NAMESPACE_DNS, "x") == uuid.uuid5(uuid.NAMESPACE_DNS, "x")
    ml._ids.rng = random.Random("conv|0|s1")
    try:
        first = [shim.uuid4() for _ in range(3)]
        ml._ids.rng = random.Random("conv|0|s1")
        assert [shim.uuid4() for _ in range(3)] == first
        assert first[0].version == 4
    finally:
        ml._ids.rng = None


def test_method_builds_without_langmem() -> None:
    (method,) = build_methods(["ML"])
    assert isinstance(method, ml.LangMemBaseline)
    assert method.name == "ML" and method.system == "langmem"


# ---------- 端到端：经网关写入与检索（需要 langmem） ----------


def _fake_model(body: dict[str, Any]) -> str | dict[str, Any]:
    """LangMem 的抽取（带工具）：第一个会话新增一条记忆，第二个会话用 PatchDoc 改写它，
    第三个会话给一个改不了的补丁（路径不存在）；回答模型（不带工具）：返回固定答案。"""
    if not body.get("tools"):
        return '{"answer": "Berlin", "evidence_sufficient": true, "citations": ["m1"]}'
    transcript = " ".join(
        m["content"] for m in body["messages"] if m["role"] == "user" and m.get("content")
    )
    if "bakery" in transcript or "gym" in transcript:
        system = body["messages"][0]["content"]
        doc_id = re.search(r"<instance id=(\S+) ", system).group(1)  # type: ignore[union-attr]
        path = "/content" if "bakery" in transcript else "/no_such_field"
        patch = {
            "json_doc_id": doc_id,
            "planned_edits": "Add the news.",
            "patches": [
                {
                    "op": "replace",
                    "path": path,
                    "value": "Ana moved to Berlin and works at a bakery.",
                }
            ],
        }
        call = {"id": "c2", "name": "PatchDoc", "arguments": json.dumps(patch)}
        return {"tool_calls": [call]}
    memory = json.dumps({"content": "Ana moved to Berlin in late July 2026."})
    return {"tool_calls": [{"id": "c1", "name": "Memory", "arguments": memory}]}


def test_build_and_retrieve_through_the_gateway(
    offline_settings: Settings, fake_api: FakeAPI
) -> None:
    pytest.importorskip("langmem")
    pytest.importorskip("langchain_openai")
    llm = LLMClient(
        offline_settings,
        cache=ResponseCache(None),
        ledger=Ledger(None),
        http_client=fake_api.client(),
    )
    fake_api.chat_replies = [_fake_model]
    sample = Sample(
        dataset="locomo",
        sample_id="q1",
        space_key="conv-1",
        question="Where does Ana live?",
        answer="Berlin",
        question_type="single-hop",
        ability="single-hop",
        is_abstention=False,
        question_time=DAY2,
    )
    gym = [EvalTurn(role="user", text="I joined a gym.", speaker="Ben")]
    sessions = [*_two_people(), EvalSession("session_3", datetime(2026, 8, 9, 9, 0), gym)]
    ctx = MethodContext(sample, sessions, Reader(llm), None, "locomo:conv-1")
    method = ml.LangMemBaseline()
    try:
        out = method.answer(ctx)
        assert out.response == "Berlin"
        build = out.details["langmem_build"]
        assert build["memories"] == 1 and build["updates"] == 1
        assert build["failed_sessions"] == [] and build["chat_calls"] == 3
        assert build["embedded_texts"] >= 2
        # 第三个会话的补丁改不了：trustcall 只记日志、跳过，统计里要看得到
        assert build["skipped_edits"] == 1
        assert "Could not apply patch" in build["skipped_edit_examples"][0]
        extraction = [r for r in fake_api.chat_requests if r.get("tools")]
        # 还没有记忆时由模型决定调不调工具；有已有记忆时 trustcall 要求必须调用（"any"）
        assert [r.get("tool_choice") for r in extraction] == [None, "required", "required"]
        assert "2026-08-01 (Sat) 10:30 | Ana: I moved to Berlin" in json.dumps(extraction[0])

        (hit,) = out.hits
        assert hit.unit.text.startswith("Ana moved to Berlin and works at a bakery.")
        assert hit.unit.ts == DAY2 and hit.unit.session_id == "session_2"
        assert out.details["langmem_found"] == 1
        directory = method.root / "locomo_conv-1"
        stored = json.loads((directory / ml.STORE_FILE).read_text("utf-8"))
        assert stored["identity"] == {
            "extractor": "qwen3.8-flash",
            "embedder": method.embedder.name,
            "dim": offline_settings.embedding.dim,
        }
        assert build["extractor"] == "qwen3.8-flash"
        key = stored["items"][0]["key"]

        method.close("locomo_conv-1")  # 从磁盘装回：结果相同，不重建
        again = method.answer(ctx)
        assert "langmem_build" not in again.details
        assert [h.unit.text for h in again.hits] == [hit.unit.text]

        # 重建：记忆 ID 由（对话, 会话）定种子，与第一次相同
        method.close("locomo_conv-1")
        shutil.rmtree(directory)
        rebuilt = ml.LangMemBaseline()
        try:
            rebuilt.answer(ctx)
            stored = json.loads((directory / ml.STORE_FILE).read_text("utf-8"))
            assert stored["items"][0]["key"] == key
        finally:
            rebuilt.gateway.stop()
    finally:
        method.gateway.stop()
