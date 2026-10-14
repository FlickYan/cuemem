"""外部基线 MC（Cognee）：写入格式、检索结果到证据的转换、子进程协议。

不装 cognee 也能跑：Cognee 只在子进程里导入，这里测父进程一侧的纯函数与协议；
需要 cognee 的部分用 importorskip。"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from cuemem.evals.baselines.cognee import (
    BUDGET_TOKENS,
    CogneeBaseline,
    Worker,
    _mentions,
    dataset_name,
    interleave,
    manifest,
    session_items,
    split_prefix,
    to_hits,
    turn_pairs,
    worker_env,
)
from cuemem.evals.datasets import EvalSession, EvalTurn
from cuemem.evals.external import memory_hit
from cuemem.evals.methods import build_methods

MAY8 = datetime(2023, 5, 8, 13, 56)
MAY25 = datetime(2023, 5, 25, 13, 14)


def _locomo() -> list[EvalSession]:
    later = EvalSession(
        "session_2",
        MAY25,
        [
            EvalTurn("user", "I ran a charity race last Saturday.", "Melanie"),
            EvalTurn("user", "Amazing!", "Caroline"),
            EvalTurn("user", "  ", "Melanie"),  # 空话跳过
            EvalTurn("user", "We went camping too.", "Melanie"),
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


def test_turn_pairs_two_people_and_user_assistant() -> None:
    pairs = turn_pairs(_locomo()[0].turns)
    assert [[t.text for t in p] for p in pairs] == [
        ["I ran a charity race last Saturday.", "Amazing!"],
        ["We went camping too."],
    ]
    chat = [
        EvalTurn("assistant", "Welcome back."),
        EvalTurn("user", "Plan a trip"),
        EvalTurn("assistant", "Sure: day 1 ..."),
        EvalTurn("user", "Thanks"),
        EvalTurn("user", "One more thing"),
        EvalTurn("assistant", "Yes?"),
        EvalTurn("assistant", "Anything else?"),
    ]
    # BEAM 的配对：用户的一句开始新的一轮，紧跟的助手回复并入；多出来的助手回复单独一轮
    assert [[t.text for t in p] for p in turn_pairs(chat)] == [
        ["Welcome back."],
        ["Plan a trip", "Sure: day 1 ..."],
        ["Thanks"],
        ["One more thing", "Yes?"],
        ["Anything else?"],
    ]


def test_session_items_carry_position_date_and_speakers() -> None:
    items = session_items(3, _locomo()[0])
    assert items[0] == (
        "[Session 3, Turn 1, date 2023-05-25 (Thu) 13:14]\n"
        "Melanie: I ran a charity race last Saturday.\nCaroline: Amazing!"
    )
    assert items[1].startswith("[Session 3, Turn 2, date 2023-05-25 (Thu) 13:14]\nMelanie: We")
    assistant = EvalSession("s", MAY8, [EvalTurn("user", "hi"), EvalTurn("assistant", "hello")])
    assert session_items(1, assistant)[0].endswith("\nUser: hi\nAssistant: hello")


def test_long_rounds_are_split_into_parts() -> None:
    long = EvalSession(
        "s", MAY8, [EvalTurn("user", "Summarize."), EvalTurn("assistant", "word " * 3000)]
    )
    items = session_items(2, long, max_tokens=1000)
    assert len(items) >= 3
    assert items[0].startswith("[Session 2, Turn 1, Part 1/")
    assert all(len(item) < 1000 * 4 + 200 for item in items)
    for item in items:
        number, body = split_prefix(item)
        assert number == 2 and body and not body.startswith("[")


def test_manifest_orders_sessions_by_time() -> None:
    plan = manifest(_locomo())
    rows = plan["sessions"]
    assert [(r["number"], r["session_id"]) for r in rows] == [(1, "session_1"), (2, "session_2")]
    assert rows[0]["file"] == "sessions/s0001_session_1.json"
    assert rows[1]["items"][0].startswith("[Session 2, Turn 1, date 2023-05-25")
    assert dataset_name("locomo_conv-42") == "locomo_conv_42"
    assert split_prefix("no prefix") == (None, "no prefix")


def _rows() -> list[dict[str, Any]]:
    return [
        {"number": 1, "session_id": "session_1", "started_at": MAY8.isoformat()},
        {"number": 2, "session_id": "session_2", "started_at": MAY25.isoformat()},
    ]


def test_to_hits_dates_every_piece_of_evidence() -> None:
    reply = {
        "chunks": [
            {
                "id": "c2",
                "text": "[Session 2, Turn 1, date 2023-05-25 (Thu) 13:14]\nMelanie: race",
                "score": 0.2,
            },
            {"id": "c1", "text": "[Session 1, Turn 1, date x]\nCaroline: group", "score": None},
        ],
        "entities": [
            {"id": "e1", "block": "### Melanie (person)\n- Melanie ran a race", "sessions": [2, 1]},
            {"id": "e2", "block": "### Lake (place)", "sessions": []},
        ],
        "facts": [
            "Melanie -- likes -- camping",
            "This chunk is about: race made from [Session 1, Turn 1, date d] Caroline: grou.",
        ],
        "context_tokens": 99,
    }
    hits, details = to_hits("locomo:conv-1", reply, _rows())
    by_id = {h.unit.id: h for h in hits}
    assert by_id["chunk:c2"].unit.text == "Melanie: race"
    assert by_id["chunk:c2"].unit.ts == MAY25 and by_id["chunk:c2"].unit.session_id == "session_2"
    assert by_id["chunk:c1"].unit.ts == MAY8 and by_id["chunk:c2"].score == 0.2
    melanie = by_id["entity:e1"]
    assert melanie.unit.ts == MAY25 and melanie.unit.kind == "entity"  # 最后一次提到它的会话
    assert melanie.unit.text.endswith("(Mentioned in sessions on: 2023-05-08, 2023-05-25)")
    assert by_id["entity:e2"].unit.text.endswith("(Source session unknown)")
    assert by_id["fact:1"].unit.text.endswith("(knowledge-graph fact; date unknown)")
    assert by_id["fact:1"].unit.ts == MAY25  # 找不到来源：对话的最后一个会话
    dated = by_id["fact:2"]  # 文字里带着块的位置前缀：取那个会话
    assert dated.unit.ts == MAY8 and dated.unit.session_id == "session_1"
    assert not dated.unit.text.endswith("date unknown)")
    assert [h.route for h in hits][:3] == ["cognee_round", "cognee_entity", "cognee_fact"]
    assert details["cognee_chunks"] == 2 and details["cognee_entities"] == 2
    assert details["cognee_facts"] == 2 and details["cognee_context_tokens"] == 99
    assert details["evidence_tokens"] <= BUDGET_TOKENS
    assert to_hits("x", {"chunks": [], "entities": [], "facts": []}, _rows())[0] == []


def test_interleave_takes_lanes_in_turn_within_budget() -> None:
    def hit(name: str, words: int) -> Any:
        return memory_hit("s", name, "w " * words, MAY8, 1)

    chunks = [hit("c1", 100), hit("c2", 100), hit("c3", 100)]
    entities = [hit("e1", 10), hit("e2", 400)]
    kept, used = interleave([chunks, entities, []], 140)
    # c1（51）、e1（6）、c2（51）装下；e2 放不下跳过；c3 也放不下
    assert [h.unit.id for h in kept] == ["c1", "e1", "c2"] and used == 108
    huge = [hit("big", 5000)]
    assert [h.unit.id for h in interleave([huge], 10)[0]] == ["big"]  # 至少一条


def test_entity_mentions_come_from_chunks_in_the_graph() -> None:
    class Graph:
        async def get_neighborhood(self, ids: list[str], depth: int) -> Any:
            assert depth == 1 and ids == ["e1", "e2"]
            nodes = [
                ("c1", {"type": "DocumentChunk", "text": "[Session 4, Turn 2, date d]\nA: x"}),
                ("c2", {"type": "DocumentChunk", "text": "[Session 7, Turn 1, date d]\nB: y"}),
                ("t1", {"type": "EntityType", "name": "person"}),
            ]
            edges = [
                ("c1", "e1", "contains", {}),
                ("c2", "e1", "contains", {}),
                ("e2", "t1", "is_a", {}),
                ("c2", "e2", "contains", {}),
            ]
            return nodes, edges

    found = asyncio.run(_mentions(["e1", "e2"], Graph()))
    assert found == {"e1": {4, 7}, "e2": {7}}


def test_worker_env_points_cognee_at_the_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")  # 云上环境的 AWS 凭据会让 Cognee 改用 S3
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
    monkeypatch.setenv("CUEMEM_TEST_PLAIN", "kept")
    env = worker_env(tmp_path, "http://127.0.0.1:9/b/cognee.build.x/v1", "deepseek-flash", "m", 8)
    assert "AWS_ACCESS_KEY_ID" not in env and "DEEPSEEK_API_KEY" not in env
    assert env["CUEMEM_TEST_PLAIN"] == "kept"
    assert env["LLM_MODEL"] == "openai/deepseek-flash" and env["LLM_PROVIDER"] == "openai"
    assert (
        env["LLM_ENDPOINT"] == env["EMBEDDING_ENDPOINT"] == "http://127.0.0.1:9/b/cognee.build.x/v1"
    )
    assert env["EMBEDDING_PROVIDER"] == "openai_compatible" and env["EMBEDDING_DIMENSIONS"] == "8"
    assert env["TELEMETRY_DISABLED"] == "1" and env["ENABLE_BACKEND_ACCESS_CONTROL"] == "false"
    assert env["SYSTEM_ROOT_DIRECTORY"] == str(tmp_path / "cognee_system")
    assert env["DATA_ROOT_DIRECTORY"] == str(tmp_path / "cognee_data")


ECHO = """
import json, sys
print("noise on stderr", file=sys.stderr)
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request["query"] == "die":
        sys.exit(3)
    print(json.dumps({"ok": True, "echo": request["query"]}), flush=True)
"""


def test_worker_protocol_and_crash_reporting(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    worker = Worker([sys.executable, "-c", ECHO], {}, log)
    try:
        assert worker.receive(30) == {"ready": True}
        assert worker.ask({"query": "hello"}, timeout=30) == {"ok": True, "echo": "hello"}
        with pytest.raises(RuntimeError, match="noise on stderr"):
            worker.ask({"query": "die"}, timeout=30)
    finally:
        worker.stop()
    hang = Worker([sys.executable, "-c", "import time; time.sleep(60)"], {}, log)
    with pytest.raises(TimeoutError):
        hang.receive(0.5)
    assert hang.process.poll() is not None  # 超时后关掉了


def test_mc_is_constructed_without_cognee() -> None:
    (method,) = build_methods(["MC"])
    assert isinstance(method, CogneeBaseline)
    assert (method.name, method.system) == ("MC", "cognee")
    assert not method.needs_index and not method.needs_consolidation


def test_worker_imports_cognee_in_a_subprocess(tmp_path: Path) -> None:
    """装了 cognee 时：子进程能导入它、按给的目录配置好，并在空图上返回空结果。"""
    pytest.importorskip("cognee")
    env = worker_env(tmp_path, "http://127.0.0.1:9/v1", "x", "m", 8)
    (tmp_path / "manifest.json").write_text(json.dumps({"dataset": "d", "sessions": []}))
    argv = [sys.executable, "-m", "cuemem.evals.baselines.cognee", "serve", str(tmp_path)]
    worker = Worker(argv, env, tmp_path / "worker.log")
    try:
        assert worker.receive(300) == {"ready": True}
        reply = worker.ask({"query": "anything"}, timeout=120)
        assert reply["ok"], reply
        assert reply["chunks"] == [] and reply["entities"] == []
    finally:
        worker.stop()
    assert (tmp_path / "cognee_system" / "databases").is_dir()  # 数据库都建在给的目录下


def _graph_or_summary(body: dict[str, Any]) -> str:
    """假的抽取模型：图谱抽取（schema 里有 source_node_id）回一个小图，摘要回一句话。"""
    text = json.dumps(body)
    if "source_node_id" in text:
        graph = {
            "nodes": [
                {"id": "Caroline", "name": "Caroline", "type": "Person", "description": "A person"},
                {"id": "group", "name": "support group", "type": "Event", "description": "x"},
            ],
            "edges": [
                {
                    "source_node_id": "Caroline",
                    "target_node_id": "group",
                    "relationship_name": "attended",
                }
            ],
        }
        return json.dumps(graph)
    if "summary" in text:
        return json.dumps({"summary": "Caroline went to a support group.", "description": ""})
    return "test"


def test_build_and_retrieve_through_the_gateway(
    offline_settings: Any, fake_api: Any, tmp_path: Path
) -> None:
    """装了 cognee 时：写入子进程经网关抽取、向量化，检索子进程返回带日期的证据。"""
    pytest.importorskip("cognee")
    from cuemem.evals.datasets import Sample
    from cuemem.llm import Ledger, LLMClient, ResponseCache

    fake_api.chat_replies = [_graph_or_summary]
    llm = LLMClient(
        offline_settings,
        cache=ResponseCache(None),
        ledger=Ledger(None),
        http_client=fake_api.client(),
    )
    baseline = CogneeBaseline()
    baseline._setup(llm)
    key, sessions = "locomo_conv-1", _locomo()
    directory = baseline.root / key
    question = "Who went to a support group?"
    sample = Sample("locomo", "q1", "conv-1", question, "", "x", "x", False, MAY25)
    try:
        built = baseline._ensure_built(key, directory, sessions)
        assert built is not None and built["failed_sessions"] == []
        assert built["chunks_written"] == 3 and built["chat_calls"] >= 6  # 每块一次抽取一次摘要
        assert built["node_types"]["DocumentChunk"] == 3 and built["embedded_texts"] > 0
        hits, details = baseline.retrieve(key, directory, sample, sessions)
        assert details["cognee_chunks"] == 3 and details["cognee_entities"] >= 1
        rounds = {h.unit.session_id: h for h in hits if h.route == "cognee_round"}
        assert rounds["session_1"].unit.ts == MAY8 and rounds["session_2"].unit.ts == MAY25
        assert rounds["session_1"].unit.text.startswith("Caroline: I went to a support group")
        caroline = next(h for h in hits if h.unit.text.startswith("### caroline"))
        assert "(Mentioned in sessions on: 2023-05-08, 2023-05-25)" in caroline.unit.text
        assert baseline.gateway.usage(baseline.bucket(key, "search"))["embedding_calls"] >= 1
    finally:
        baseline.close(key)
        baseline.gateway.stop()
