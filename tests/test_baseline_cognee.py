"""外部基线 MC（Cognee）：写入格式、出处与日期、检索结果到证据的转换、子进程协议。

不装 cognee 也能跑：Cognee 只在子进程里导入，这里测父进程一侧的纯函数与协议；
需要 cognee 的部分用 importorskip（用 uv run --with cognee==1.6.4 pytest 运行）。"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cuemem.evals.baselines.cognee import (
    COGNEE_VERSION,
    MANIFEST,
    CogneeBaseline,
    GraphIndex,
    Worker,
    dataset_name,
    distill_session_id,
    manifest,
    session_items,
    split_prefix,
    to_hits,
    turn_pairs,
    two_people,
    worker_env,
)
from cuemem.evals.datasets import EvalSession, EvalTurn
from cuemem.evals.methods import build_methods

MAY8 = datetime(2023, 5, 8, 13, 56)
MAY25 = datetime(2023, 5, 25, 13, 14)
JUNE9 = datetime(2023, 6, 9, 10, 0)
JULY1 = datetime(2023, 7, 1, 9, 30)


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
    assert two_people(_locomo())
    pairs = turn_pairs(_locomo()[0].turns, True)
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
    assert not two_people([EvalSession("s", MAY8, chat)])
    # BEAM 的配对：用户的一句开始新的一轮，紧跟的助手回复并入；多出来的助手回复单独一轮
    assert [[t.text for t in p] for p in turn_pairs(chat, False)] == [
        ["Welcome back."],
        ["Plan a trip", "Sure: day 1 ..."],
        ["Thanks"],
        ["One more thing", "Yes?"],
        ["Anything else?"],
    ]


def test_session_items_follow_beam_format_turn_pair() -> None:
    items = session_items(3, _locomo()[0], True)
    assert items[0] == (
        "Session: 3\nTurn: 1\nTime anchor: 2023-05-25 (Thu) 13:14\n\n"
        "Melanie:\nI ran a charity race last Saturday.\n\nCaroline:\nAmazing!"
    )
    assert items[1] == (
        "Session: 3\nTurn: 2\nTime anchor: 2023-05-25 (Thu) 13:14\n\nMelanie:\nWe went camping too."
    )
    # 用户与助手：标签是 User/Assistant（BEAM 的 parse_beam_block 按它们找出每一轮）
    chat = EvalSession(
        "s", MAY8, [EvalTurn("user", "hi", "Ann"), EvalTurn("assistant", "hello", "Bot")]
    )
    item = session_items(1, chat, False)[0]
    assert item.endswith("\n\nUser:\nhi\n\nAssistant:\nhello")
    assert split_prefix(item) == (1, 1, "User:\nhi\n\nAssistant:\nhello")


def test_long_rounds_are_split_into_parts() -> None:
    long = EvalSession(
        "s", MAY8, [EvalTurn("user", "Summarize."), EvalTurn("assistant", "word " * 3000)]
    )
    items = session_items(2, long, False, max_tokens=1000)
    assert len(items) >= 3
    assert items[0].startswith("Session: 2\nTurn: 1\nPart: 1/")
    assert all(len(item) < 1000 * 4 + 200 for item in items)
    for item in items:
        number, turn, body = split_prefix(item)
        assert (number, turn) == (2, 1) and body and not body.startswith("Time anchor")


def test_manifest_orders_sessions_and_records_turn_indices() -> None:
    plan = manifest(_locomo())
    rows = plan["sessions"]
    assert plan["two_people"] is True
    assert [(r["number"], r["session_id"]) for r in rows] == [(1, "session_1"), (2, "session_2")]
    assert rows[0]["file"] == "sessions/s0001_session_1.json"
    assert rows[1]["items"][0].startswith("Session: 2\nTurn: 1\nTime anchor: 2023-05-25")
    # 第几轮 → 它的发言在会话里的下标（空话跳过，下标不变）
    assert rows[1]["rounds"] == [[0, 1], [3]] and rows[0]["rounds"] == [[0, 1]]
    assert dataset_name("locomo_conv-42") == "locomo_conv_42"


def test_split_prefix_reads_turns_and_distilled_lessons() -> None:
    assert split_prefix("no prefix") == (None, None, "no prefix")
    lesson = f"# Session learning (session {distill_session_id(7)})\n\nX.\n"
    assert split_prefix(lesson) == (7, None, "# Session learning\n\nX.\n")
    # 早先的写法（带着空间名）照样认得出编号
    old = "# Session learning (session s0007_longmemeval_s_q1)\n\nX.\n"
    assert split_prefix(old) == (7, None, "# Session learning\n\nX.\n")


def test_distill_session_id_carries_no_question_id() -> None:
    """经验文档的会话 ID 要交给 cognify 的模型：只能有会话编号，不能有题号
    （LongMemEval 拒答题的题号以 _abs 结尾）。"""
    assert distill_session_id(3) == "s0003_session"
    assert "abs" not in distill_session_id(3) and "longmemeval" not in distill_session_id(3)


def test_distilled_builds_with_old_lesson_ids_are_rebuilt(tmp_path: Path) -> None:
    """做了会话蒸馏、蒸馏 ID 还带着题号的旧写入不复用；没做蒸馏的（LoCoMo）照常复用。"""
    from cuemem.evals.baselines.cognee import LESSON_TAG

    calls: list[str] = []

    class Recording(CogneeBaseline):
        def build(self, key: str, directory: Path, sessions: Any) -> dict[str, Any]:
            calls.append(key)
            return {}

    baseline = Recording()
    baseline.protocol = "p"
    baseline.gateway = SimpleNamespace(usage=lambda bucket: {})
    for key, record, rebuilt in (
        ("locomo_conv-1", {"protocol": "p", "distill": False}, False),
        ("longmemeval_s-anon_q1", {"protocol": "p", "distill": True}, True),
        (
            "longmemeval_s-anon_q2",
            {"protocol": "p", "distill": True, "lesson_ids": LESSON_TAG, "session_ids": "anon"},
            False,
        ),
    ):
        directory = tmp_path / key
        directory.mkdir()
        (directory / "build.json").write_text(json.dumps(record))
        baseline._schemes[key] = "native" if key.startswith("locomo") else "anon"
        built = baseline._ensure_built(key, directory, [])
        assert (built is not None) == rebuilt, key
    assert calls == ["longmemeval_s-anon_q1"]


def _rows() -> list[dict[str, Any]]:
    return [
        {"number": 1, "session_id": "session_1", "started_at": MAY8.isoformat(), "rounds": [[0]]},
        {
            "number": 2,
            "session_id": "session_2",
            "started_at": MAY25.isoformat(),
            "rounds": [[0, 1], [3]],
        },
        {"number": 3, "session_id": "session_3", "started_at": JUNE9.isoformat(), "rounds": []},
        {"number": 4, "session_id": "session_4", "started_at": JULY1.isoformat(), "rounds": []},
    ]


def test_to_hits_dates_every_piece_of_evidence_by_its_source() -> None:
    reply = {
        "chunks": [
            {
                "id": "c2",
                "text": "Session: 2\nTurn: 2\nTime anchor: x\n\nMelanie:\nWe went camping too.",
                "score": 0.2,
            },
            {"id": "c1", "text": "Session: 1\nTurn: 1\nTime anchor: x\n\nCaroline:\nhi"},
            {"id": "c9", "text": "# Session learning (session s0003_d)\n\nLikes hiking."},
            {"id": "c0", "text": "stray"},
        ],
        "entities": [
            {
                "id": "e1",
                "header": "### melanie (person)\nRuns races.",
                "sessions": [2],  # 描述来自会话 2
                "edges": [
                    {"text": "melanie ran a charity race.", "sessions": [2]},
                    {"text": "melanie is a person.", "sessions": []},
                    {"text": "melanie went camping.", "sessions": [1, 2, 3, 4]},
                ],
            },
        ],
        "facts": [
            {"id": "f1", "text": "Melanie ran a race on 2023-05-20.", "sessions": [2]},
            {"id": "f2", "text": "Caroline attends a support group.", "sessions": [1, 3]},
            {"id": "f3", "text": "Melanie: A runner.", "sessions": []},
        ],
        "context_tokens": 99,
    }
    hits, details = to_hits("locomo_conv-1", reply, _rows())
    by_id = {h.unit.id: h for h in hits}
    camping = by_id["chunk:c2"]
    assert camping.unit.text == "Melanie:\nWe went camping too." and camping.score == 0.2
    assert camping.unit.ts == MAY25 and camping.unit.session_id == "session_2"
    assert camping.unit.meta["turn_indices"] == [3]  # 第 2 轮覆盖的发言
    assert by_id["chunk:c1"].unit.ts == MAY8 and by_id["chunk:c1"].unit.meta["turn_indices"] == [0]
    lesson = by_id["chunk:c9"]  # 会话蒸馏出的经验：取它所在的会话，没有发言下标
    assert lesson.unit.session_id == "session_3" and "turn_indices" not in lesson.unit.meta
    assert lesson.route == "cognee_lesson" and lesson.unit.kind == "round"
    assert lesson.unit.text.startswith("# Session learning\n\nLikes hiking.")
    stray = by_id["chunk:c0"]  # 认不出来源：第一个会话的时间，会话记为 unknown
    assert stray.unit.ts == MAY8 and stray.unit.session_id == "unknown"
    assert stray.unit.text == "stray (date unknown)"
    melanie = by_id["entity:e1"]
    assert melanie.unit.kind == "entity" and melanie.unit.ts == MAY25
    assert melanie.unit.text.splitlines() == [
        "### melanie (person)",
        "Runs races.",
        "- melanie ran a charity race. (from the conversation on 2023-05-25)",
        "- melanie is a person.",
        "- melanie went camping. (from the conversations on 2023-05-25, 2023-06-09, "
        "2023-07-01, and 1 earlier)",
    ]
    assert by_id["fact:f1"].unit.text == "Melanie ran a race on 2023-05-20."  # 一个出处：不另注
    assert by_id["fact:f1"].unit.ts == MAY25
    support = by_id["fact:f2"]  # 几个出处：取最近的会话，注明各次日期
    assert support.unit.ts == JUNE9 and support.unit.session_id == "session_3"
    assert support.unit.text.endswith("(from the conversations on 2023-05-08, 2023-06-09)")
    undated = by_id["fact:f3"]  # 找不到出处：不会被当成最新的记忆
    assert undated.unit.ts == MAY8 and undated.unit.session_id == "unknown"
    assert undated.unit.text.endswith("(date unknown)")
    assert {h.route for h in hits} == {
        "cognee_round",
        "cognee_lesson",
        "cognee_entity",
        "cognee_fact",
    }
    assert details["cognee_chunks"] == 4 and details["cognee_entities"] == 1
    assert details["cognee_facts"] == 3 and details["cognee_undated"] == 2
    assert details["cognee_context_tokens"] == 99 and details["evidence_tokens"] > 0
    assert to_hits("x", {"chunks": [], "entities": [], "facts": []}, _rows())[0] == []


def test_graph_index_finds_where_entities_and_facts_come_from() -> None:
    nodes = [
        ("c1", {"type": "DocumentChunk", "text": "Session: 4\nTurn: 2\nTime anchor: d\n\nA: x"}),
        ("c2", {"type": "DocumentChunk", "text": "Session: 7\nTurn: 1\nTime anchor: d\n\nB: y"}),
        ("e1", {"type": "Entity", "name": "nate"}),
        ("t1", {"type": "EntityType", "name": "person"}),
    ]
    edges = [
        ("c1", "e1", "contains", {"edge_text": "Document chunk mentions nate: Likes games."}),
        ("c2", "e1", "contains", {"edge_text": "Document chunk mentions nate: Has a dog."}),
        ("e1", "t1", "is_a", {"edge_text": "nate is a person.", "edge_object_id": "o1"}),
        ("e1", "e1", "won", {"edge_text": "Nate won a tournament.", "edge_object_id": "o2"}),
        ("s1", "c2", "made_from", {"edge_text": "Summary made from B: y", "edge_object_id": "o3"}),
    ]
    index = GraphIndex.build(nodes, edges, lambda rel, props: f"T:{props.get('edge_text') or rel}")
    assert index.chunk_session == {"c1": 4, "c2": 7}
    assert index.entity_sessions("e1", "Has a dog.") == [7]  # 描述来自哪个块
    assert index.entity_sessions("e1", "Something else") == [4, 7]  # 对不上：提到它的所有会话
    assert index.entity_sessions("nobody", "x") == []
    # 包含边的事实（"Nate: Likes games."）：起点就是块
    assert index.fact_edges("T:Document chunk mentions nate: Likes games.") == ([], {4})
    # 实体之间的关系：交给出处记录（edge_object_id）
    assert index.fact_edges("T:Nate won a tournament.") == (["o2"], set())
    # 摘要 → 块：终点是块
    assert index.fact_edges("T:Summary made from B: y") == (["o3"], {7})
    assert index.fact_edges("T:missing") == ([], set())


def test_worker_env_points_cognee_at_the_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "x")  # 云上环境的 AWS 凭据会让 Cognee 改用 S3
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret")
    monkeypatch.setenv("CUEMEM_TEST_PLAIN", "kept")
    url = "http://127.0.0.1:9/b/cognee.search.x/v1"
    env = worker_env(tmp_path / "space", url, "deepseek-flash", "m", 8)
    assert "AWS_ACCESS_KEY_ID" not in env and "DEEPSEEK_API_KEY" not in env
    assert env["CUEMEM_TEST_PLAIN"] == "kept"
    assert env["LLM_MODEL"] == "openai/deepseek-flash" and env["LLM_PROVIDER"] == "openai"
    assert env["LLM_ENDPOINT"] == env["EMBEDDING_ENDPOINT"] == url
    assert env["EMBEDDING_PROVIDER"] == "openai_compatible" and env["EMBEDDING_DIMENSIONS"] == "8"
    assert env["TELEMETRY_DISABLED"] == "1" and env["ENABLE_BACKEND_ACCESS_CONTROL"] == "false"
    assert env["SYSTEM_ROOT_DIRECTORY"] == str(tmp_path / "space" / "cognee_system")
    assert env["DATA_ROOT_DIRECTORY"] == str(tmp_path / "space" / "cognee_data")
    # Cognee 不读项目的 .env：指向空间目录里的空文件
    assert Path(env["COGNEE_ENV_FILE"]).read_text() == ""
    assert Path(env["COGNEE_ENV_FILE"]).parent == tmp_path / "space"
    assert env["LITELLM_MODE"] == "PRODUCTION"  # LiteLLM 导入时也会读 .env
    # 检索：不用会话缓存，问题按查询向量化
    assert env["CACHING"] == "false" and env["EMBEDDING_INPUT_TYPE"] == "query"
    build = worker_env(tmp_path / "space", url, "m", "m", 8, mode="build")
    assert build["CACHING"] == "false" and "EMBEDDING_INPUT_TYPE" not in build
    distill = worker_env(tmp_path / "space", url, "m", "m", 8, mode="build", distill=True)
    assert (distill["CACHING"], distill["CACHE_BACKEND"], distill["AUTO_FEEDBACK"]) == (
        "true",
        "fs",
        "true",
    )


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


def test_setup_pins_the_cognee_version(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    baseline = CogneeBaseline()
    baseline.protocol = "model|embedder:8"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.5.0")
    with pytest.raises(RuntimeError, match=f"cognee=={COGNEE_VERSION}"):
        baseline.setup()
    monkeypatch.setattr(importlib.metadata, "version", lambda name: COGNEE_VERSION)
    baseline.setup()
    # 协议里带上版本与写入格式：换了就重建
    assert baseline.protocol.startswith(f"model|embedder:8|cognee=={COGNEE_VERSION}|")


# ---------- 需要 cognee ----------


def test_cognee_does_not_read_the_project_dotenv(tmp_path: Path) -> None:
    """Cognee 导入时从当前目录往上找 .env 并覆盖环境变量；子进程的环境要挡住它。"""
    pytest.importorskip("cognee")
    project = tmp_path / "project"
    (project / "deep").mkdir(parents=True)
    (project / "pyproject.toml").write_text("")
    (project / ".env").write_text(
        "LLM_ENDPOINT=http://example.invalid/v1\nDASHSCOPE_API_KEY=sk-x\n"
    )
    env = worker_env(project / "space", "http://gateway/v1", "x", "m", 8)
    code = (
        "import os, cognee\n"
        "from cognee.infrastructure.llm.config import get_llm_config\n"
        "print(get_llm_config().llm_endpoint, os.environ.get('DASHSCOPE_API_KEY'))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=project / "deep",
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    assert out.stdout.split()[-2:] == ["http://gateway/v1", "None"]


def test_worker_imports_cognee_in_a_subprocess(tmp_path: Path) -> None:
    """装了 cognee 时：子进程能导入它、按给的目录配置好，并在空图上返回空结果。"""
    pytest.importorskip("cognee")
    env = worker_env(tmp_path, "http://127.0.0.1:9/v1", "x", "m", 8)
    (tmp_path / MANIFEST).write_text(json.dumps({"dataset": "d", "sessions": []}))
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


POISON = "POISONED-SESSION"


@pytest.fixture
def fake_endpoint() -> Iterator[str]:
    """一个最小的 OpenAI 兼容服务：抽取与摘要照 _graph_or_summary 回答，向量是 8 维的常数；
    请求里出现 POISON 时回 404（Cognee 不重试 404，测试不用等它的重试下限）。"""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # BaseHTTPRequestHandler 的接口
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/embeddings"):
                texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
                data = [
                    {"object": "embedding", "index": i, "embedding": [1.0, len(t) % 5] + [0.5] * 6}
                    for i, t in enumerate(texts)
                ]
                reply: dict[str, Any] = {
                    "object": "list",
                    "data": data,
                    "model": "m",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            elif POISON in json.dumps(body):
                self._send(404, {"error": {"message": "poisoned", "type": "not_found"}})
                return
            else:
                reply = {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": _graph_or_summary(body)},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            self._send(200, reply)

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()


def test_a_failing_session_does_not_take_later_sessions_down(
    tmp_path: Path, fake_endpoint: str
) -> None:
    """装了 cognee 时：中间一个会话一直失败，它被跳过、从数据集里删掉，后面的会话照常写进图里
    （不删的话，Cognee 会在之后每次 cognify 里重跑它、再把新会话一起回滚）。"""
    pytest.importorskip("cognee")
    sessions = [
        EvalSession(
            f"session_{n}",
            datetime(2023, 5, n, 10, 0),
            [
                EvalTurn("user", f"Day {n}: {POISON if n == 2 else 'fine'}.", "Caroline"),
                EvalTurn("user", "ok", "Melanie"),
            ],
        )
        for n in range(1, 5)
    ]
    baseline = CogneeBaseline()
    plan = manifest(sessions)
    plan = {"dataset": "conv", "distill": False, **plan}
    for row in plan["sessions"]:
        (tmp_path / row["file"]).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / row["file"]).write_text(json.dumps(row["items"]))
    (tmp_path / MANIFEST).write_text(json.dumps(plan))
    env = worker_env(tmp_path, fake_endpoint, "x", "m", 8, mode="build")
    worker = Worker(baseline._argv("build", tmp_path), env, tmp_path / "worker.log")
    try:
        reply = worker.receive(900)
    finally:
        worker.stop()
    assert reply["ok"], reply
    stats = reply["stats"]
    assert [f.split(":")[0] for f in stats["failed_sessions"]] == ["session_2"]
    assert stats["retried_sessions"] == ["session_2"]  # 删掉后重写过一次，仍然失败
    assert stats["dropped_documents"] == ["s0002_session_2", "s0002_session_2"]
    assert stats["sessions_without_chunks"] == []
    assert stats["node_types"]["DocumentChunk"] == 3  # 会话 1、3、4 各一块


def test_build_and_retrieve_through_the_gateway(
    offline_settings: Any, fake_api: Any, tmp_path: Path
) -> None:
    """装了 cognee 时：写入子进程经网关抽取、向量化，检索子进程返回带日期与出处的证据。"""
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
        assert built["protocol"].endswith(f"|cognee=={COGNEE_VERSION}|beam-turn-pair")
        assert built["chunks_written"] == 3 and built["chat_calls"] >= 6  # 每块一次抽取一次摘要
        assert built["node_types"]["DocumentChunk"] == 3 and built["embedded_texts"] > 0
        assert built["distill"] is False and built["sessions_without_chunks"] == []
        assert built["effective_chunk_size"] > 0
        hits, details = baseline.retrieve(key, directory, sample, sessions)
        assert details["cognee_chunks"] == 3 and details["cognee_entities"] >= 1
        rounds = {h.unit.session_id: h for h in hits if h.route == "cognee_round"}
        assert rounds["session_1"].unit.ts == MAY8 and rounds["session_2"].unit.ts == MAY25
        assert rounds["session_1"].unit.text.startswith("Caroline:\nI went to a support group")
        assert rounds["session_1"].unit.meta["turn_indices"] == [0, 1]
        caroline = next(h for h in hits if h.unit.text.startswith("### caroline"))
        assert caroline.unit.ts == MAY25  # 描述来自最后一次抽到它的块
        # 每条关系注明出处：三个块都抽出了同一条关系
        assert "(from the conversations on 2023-05-08, 2023-05-25)" in caroline.unit.text
        undated = [h.unit.text for h in hits if h.unit.session_id == "unknown"]
        assert details["cognee_undated"] == 0, undated
        assert baseline.gateway.usage(baseline.bucket(key, "search"))["embedding_calls"] >= 1
    finally:
        baseline.close(key)
        baseline.gateway.stop()
