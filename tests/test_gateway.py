"""评测网关与外部基线的公共部分（N12）：OpenAI 兼容的接口、按桶统计、写入缓存与预算。全部离线。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import ABORT_JSON, FakeAPI
from fastapi.testclient import TestClient
from openai import OpenAI

from cuemem.config import Settings
from cuemem.embed import build_embedder
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit, space_key
from cuemem.evals.gateway import Gateway
from cuemem.evals.methods import MethodContext, build_methods
from cuemem.llm import Ledger, LLMClient, ResponseCache
from cuemem.read.reader import Reader

HELLO = [{"role": "user", "content": "hi"}]


@pytest.fixture
def llm(offline_settings: Settings, fake_api: FakeAPI) -> LLMClient:
    return LLMClient(
        offline_settings,
        cache=ResponseCache(None),
        ledger=Ledger(None),
        http_client=fake_api.client(),
    )


@pytest.fixture
def gateway(offline_settings: Settings, llm: LLMClient) -> Gateway:
    return Gateway(llm, build_embedder(offline_settings, use_cache=False))


@pytest.fixture
def client(gateway: Gateway) -> Iterator[TestClient]:
    with TestClient(gateway.app()) as test_client:
        yield test_client


def test_chat_is_routed_to_the_extractor_model(
    client: TestClient, gateway: Gateway, fake_api: FakeAPI
) -> None:
    fake_api.chat_replies = ["Hello there"]
    response = client.post(
        "/b/graphiti.build.x/v1/chat/completions",
        json={"model": "gpt-4.1-mini", "messages": HELLO, "temperature": 0.3, "max_tokens": 50},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello there"
    assert body["model"] == "qwen3.8-flash"  # extractor 角色的模型，不管请求里写的是什么
    assert body["usage"]["prompt_tokens"] == 120
    sent = fake_api.chat_requests[-1]
    assert sent["model"] == "qwen3.8-flash" and sent["temperature"] == 0.3
    assert sent["max_tokens"] == 50
    usage = gateway.usage("graphiti.build.x")
    assert usage["chat_calls"] == 1 and usage["input_tokens"] == 120
    assert usage["cost"] > 0 and usage["models_requested"] == {"gpt-4.1-mini": 1}
    assert gateway.usage("other")["chat_calls"] == 0  # 各桶分开统计


def test_json_schema_is_downgraded_to_json_object(client: TestClient, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = ['{"name": "Ana"}']
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "system", "content": "Extract."}, *HELLO],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "Person", "schema": schema},
            },
        },
    )
    assert response.status_code == 200
    sent = fake_api.chat_requests[-1]
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["messages"][0]["content"].startswith(
        "Extract.\n\nOutput format: reply with one JSON"
    )
    assert json.dumps(schema) in sent["messages"][0]["content"]


def test_tool_calls_and_streaming(client: TestClient, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = [
        lambda body: {"tool_calls": [{"id": "c1", "name": "save", "arguments": '{"a": 1}'}]}
    ]
    tools = [{"type": "function", "function": {"name": "save", "parameters": {"type": "object"}}}]
    response = client.post(
        "/v1/chat/completions",
        json={"model": "x", "messages": HELLO, "tools": tools, "tool_choice": "required"},
    )
    call = response.json()["choices"][0]["message"]["tool_calls"][0]
    assert call == {
        "id": "c1",
        "type": "function",
        "function": {"name": "save", "arguments": '{"a": 1}'},
    }
    assert fake_api.chat_requests[-1]["tool_choice"] == "required"

    fake_api.chat_replies = ["streamed answer"]
    with client.stream(
        "POST", "/v1/chat/completions", json={"model": "x", "messages": HELLO, "stream": True}
    ) as stream:
        lines = [line for line in stream.iter_lines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    first = json.loads(lines[0][5:])
    assert first["choices"][0]["delta"]["content"] == "streamed answer"


def test_aborted_json_falls_back_and_errors_become_openai_errors(
    client: TestClient, gateway: Gateway, fake_api: FakeAPI
) -> None:
    fake_api.chat_replies = [
        lambda body: ABORT_JSON if body["model"] == "qwen3.8-flash" else '{"ok": true}'
    ]
    ok = client.post(
        "/b/b1/v1/chat/completions",
        json={"messages": HELLO, "response_format": {"type": "json_object"}},
    )
    assert ok.status_code == 200 and ok.json()["model"] == "deepseek-flash"
    assert gateway.usage("b1")["fallbacks"] == 1
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
    assert client.post("/v1/chat/completions", json={"messages": HELLO, "n": 2}).status_code == 400


def test_embeddings(client: TestClient, gateway: Gateway, offline_settings: Settings) -> None:
    response = client.post("/b/e/v1/embeddings", json={"model": "x", "input": ["a b", "c"]})
    data = response.json()["data"]
    assert [d["index"] for d in data] == [0, 1]
    assert len(data[0]["embedding"]) == offline_settings.embedding.dim
    single = client.post("/v1/embeddings", json={"input": "hello", "dimensions": 8}).json()
    assert len(single["data"][0]["embedding"]) == 8
    assert gateway.usage("e")["embedded_texts"] == 2
    models = client.get("/b/e/v1/models").json()
    assert {m["id"] for m in models["data"]} == {"qwen3.8-flash", gateway.embedder.name}


def test_openai_sdk_against_the_running_gateway(gateway: Gateway, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = ["from the SDK"]
    gateway.start()
    try:
        sdk = OpenAI(base_url=gateway.url("sdk"), api_key="unused", max_retries=0)
        reply = sdk.chat.completions.create(model="anything", messages=HELLO)  # type: ignore[arg-type]
        assert reply.choices[0].message.content == "from the SDK"
        vectors = sdk.embeddings.create(model="anything", input=["x", "y"])
        assert len(vectors.data) == 2
        assert gateway.usage("sdk")["chat_calls"] == 1
    finally:
        gateway.stop()


# ---------- 外部基线的公共部分 ----------


class _Toy(ExternalBaseline):
    """把每句话当作一条记忆存进 JSON 文件；写入时经网关调一次 LLM。"""

    name, label, system = "MT", "toy", "toy"

    def __init__(self) -> None:
        super().__init__()
        self.builds = 0

    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        self.builds += 1
        sdk = OpenAI(base_url=self.gateway.url(self.bucket(key)), api_key="unused", max_retries=0)
        sdk.chat.completions.create(model="toy", messages=HELLO)  # type: ignore[arg-type]
        lines = [
            {"text": t.text, "ts": s.started_at.isoformat()} for s in sessions for t in s.turns
        ]
        (directory / "memories.json").write_text(json.dumps(lines), "utf-8")
        return {"memories": len(lines)}

    def retrieve(
        self, key: str, directory: Path, sample: Sample, sessions: Sequence[EvalSession]
    ) -> tuple[list[Any], dict[str, Any]]:
        lines = json.loads((directory / "memories.json").read_text("utf-8"))
        hits = [
            memory_hit(key, f"m{i}", line["text"], datetime.fromisoformat(line["ts"]), i)
            for i, line in enumerate(lines, start=1)
        ]
        return hits, {"searched": True}


def test_external_baseline_builds_once_and_records_usage(
    offline_settings: Settings, llm: LLMClient, fake_api: FakeAPI
) -> None:
    fake_api.chat_replies = [
        lambda body: (
            '{"answer": "Berlin", "evidence_sufficient": true, "citations": ["m1"]}'
            if "Question:" in json.dumps(body)
            else "ok"
        )
    ]
    when = datetime(2026, 8, 1, 10)
    session = EvalSession(
        session_id="s1", started_at=when, turns=[EvalTurn(role="user", text="I live in Berlin")]
    )
    sample = Sample(
        dataset="locomo",
        sample_id="q1",
        space_key="conv-1",
        question="Where do I live?",
        answer="Berlin",
        question_type="single-hop",
        ability="single-hop",
        is_abstention=False,
        question_time=when,
    )
    toy = _Toy()
    ctx = MethodContext(sample, [session], Reader(llm), None, "locomo:conv-1")
    try:
        first = toy.answer(ctx)
        assert first.response == "Berlin" and first.details["memories"] == 1
        build = first.details["toy_build"]
        assert build["memories"] == 1 and build["sessions"] == 1 and build["chat_calls"] == 1
        stored = json.loads((toy.root / "locomo_conv-1" / "build.json").read_text("utf-8"))
        assert stored["chat_calls"] == 1
        second = toy.answer(ctx)  # 已写入：不再重建
        assert "toy_build" not in second.details and toy.builds == 1
    finally:
        toy.gateway.stop()
    assert space_key("locomo:conv-42") == "locomo_conv-42"


def test_external_methods_are_registered_lazily() -> None:
    with pytest.raises(ModuleNotFoundError):  # 适配器还没装上外部包时，构造时才报错
        build_methods(["MG"])
