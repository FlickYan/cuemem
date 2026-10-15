"""评测网关与外部基线的公共部分（N12）：OpenAI 兼容的接口、按桶统计、写入缓存与预算。全部离线。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import ABORT_JSON, NO_BALANCE, RATE_LIMIT, REJECT, FakeAPI
from fastapi.testclient import TestClient
from openai import OpenAI, RateLimitError

from cuemem.config import Settings
from cuemem.embed import build_embedder
from cuemem.embed.dashscope import DashScopeTextEmbedder
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit, space_key
from cuemem.evals.gateway import Gateway, _retry_after, embed_kind
from cuemem.evals.methods import MethodContext, build_methods
from cuemem.llm import Ledger, LLMClient, ResponseCache
from cuemem.llm.types import ChatResult
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


TOOLS = [{"type": "function", "function": {"name": "save", "parameters": {"type": "object"}}}]


def test_parallel_tool_calls_default_to_true_like_openai(
    client: TestClient, gateway: Gateway, fake_api: FakeAPI
) -> None:
    """外部系统按 OpenAI 的默认（true）写；通义不设为 true 时一次只回一个工具调用。"""
    two = [
        {"id": "c1", "name": "save", "arguments": '{"a": 1}'},
        {"id": "c2", "name": "save", "arguments": '{"a": 2}'},
    ]
    fake_api.chat_replies = [lambda body: {"tool_calls": two}]
    client.post("/b/t/v1/chat/completions", json={"messages": HELLO, "tools": TOOLS})
    assert fake_api.chat_requests[-1]["parallel_tool_calls"] is True
    client.post(
        "/b/t/v1/chat/completions",
        json={"messages": HELLO, "tools": TOOLS, "parallel_tool_calls": False},
    )
    assert fake_api.chat_requests[-1]["parallel_tool_calls"] is False  # 写了就照原样转发
    fake_api.chat_replies = ["no tool"]
    bye = [{"role": "user", "content": "bye"}]  # 换个内容：同样的请求会命中响应缓存
    client.post("/b/t/v1/chat/completions", json={"messages": bye, "tools": TOOLS})
    client.post("/b/t/v1/chat/completions", json={"messages": HELLO, "parallel_tool_calls": True})
    assert "parallel_tool_calls" not in fake_api.chat_requests[-1]  # 不带工具时不发
    usage = gateway.usage("t")
    assert usage["tool_requests"] == 3 and usage["tool_calls"] == 4
    assert usage["multi_tool_replies"] == 2 and usage["no_tool_replies"] == 1
    assert usage["chat_calls"] == 4


def test_truncated_replies_are_counted(
    client: TestClient, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    cut = ChatResult(content="half a", finish_reason="length", model_alias="m", model_id="m")
    monkeypatch.setattr(gateway.llm, "chat", lambda *args, **kwargs: cut)
    client.post("/b/t/v1/chat/completions", json={"messages": HELLO})
    assert gateway.usage("t")["truncated_replies"] == 1


def test_refusals_are_counted_apart_from_other_errors(
    client: TestClient, gateway: Gateway, fake_api: FakeAPI
) -> None:
    """主模型与备用模型都拒绝算一次拒绝；外部系统吞掉错误时（如 trustcall 改写成提示重试），
    调用方从桶里看得出是不是内容审核拒绝。"""
    fake_api.chat_replies = [lambda body: REJECT]
    assert client.post("/b/f/v1/chat/completions", json={"messages": HELLO}).status_code == 400
    fake_api.chat_replies = [lambda body: ABORT_JSON]
    assert client.post("/b/f/v1/chat/completions", json={"messages": HELLO}).status_code == 400
    usage = gateway.usage("f")
    assert usage["errors"] == 2 and usage["refusals"] == 1 and usage["chat_calls"] == 0


def test_search_buckets_embed_queries(
    client: TestClient, gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """检索阶段的桶按查询向量化（通义的 text_type=query），其余按文档。"""
    kinds: list[str] = []
    embed = gateway.embedder.embed

    def spy(texts: Sequence[str], kind: Any) -> Any:
        kinds.append(kind)
        return embed(texts, kind)

    monkeypatch.setattr(gateway.embedder, "embed", spy)
    for bucket in ("langmem.search.locomo_conv-1", "langmem.build.locomo_conv-1", "default"):
        client.post(f"/b/{bucket}/v1/embeddings", json={"input": "where does Ana live"})
    assert kinds == ["query", "document", "document"]
    assert embed_kind("graphiti.search.x") == "query" and embed_kind("search") == "document"


# ---------- 限流、上游错误与解析不了的 JSON ----------


def _max_retries(gateway: Gateway) -> int:
    settings = gateway.llm.settings
    provider = settings.models[settings.roles["extractor"].model].provider
    return settings.providers[provider].max_retries


def test_upstream_rate_limit_becomes_429_with_retry_after(
    client: TestClient, gateway: Gateway, fake_api: FakeAPI
) -> None:
    """上游限流：LLM 客户端按 SDK 的退避重试过（providers.*.max_retries 次）仍然 429 时，网关回 429
    并带上上游的 Retry-After；网关自己不再重试，也不换备用模型。按桶计入 rate_limited。"""
    fake_api.chat_replies = [RATE_LIMIT]
    fake_api.chat_rate_limit_headers = {"retry-after": "7", "retry-after-ms": "1"}
    response = client.post("/b/r/v1/chat/completions", json={"messages": HELLO})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "7"
    assert response.json()["error"]["type"] == "rate_limit_error"
    assert len(fake_api.chat_requests) == 1 + _max_retries(gateway)  # 只有 SDK 自己的重试
    assert {r["model"] for r in fake_api.chat_requests} == {"qwen3.8-flash"}
    usage = gateway.usage("r")
    assert usage["rate_limited"] == 1 and usage["errors"] == 1 and usage["refusals"] == 0
    assert usage["chat_calls"] == 0 and usage["fallbacks"] == 0

    # 上游只给了 retry-after-ms：换算成秒（向上取整）
    fake_api.chat_rate_limit_headers = {"retry-after-ms": "1"}
    again = client.post("/b/r/v1/chat/completions", json={"messages": HELLO})
    assert again.status_code == 429 and again.headers["retry-after"] == "1"
    assert gateway.usage("r")["rate_limited"] == 2


def test_rate_limit_reaches_openai_sdk_clients(gateway: Gateway, fake_api: FakeAPI) -> None:
    """外部系统的 OpenAI SDK 看到的是 RateLimitError（会照 Retry-After 退避），不是 500。"""
    fake_api.chat_replies = [RATE_LIMIT]
    fake_api.chat_rate_limit_headers = {"retry-after": "3", "retry-after-ms": "1"}
    gateway.start()
    try:
        sdk = OpenAI(base_url=gateway.url("sdk"), api_key="unused", max_retries=0)
        with pytest.raises(RateLimitError) as caught:
            sdk.chat.completions.create(model="anything", messages=HELLO)  # type: ignore[arg-type]
        assert caught.value.response.headers["retry-after"] == "3"
    finally:
        gateway.stop()
    assert gateway.usage("sdk")["rate_limited"] == 1


def test_retry_after_header_parsing() -> None:
    assert _retry_after({"retry-after": "12"}) == "12"
    date = "Wed, 21 Oct 2026 07:28:00 GMT"
    assert _retry_after({"retry-after": date}) == date  # HTTP 日期原样转发
    assert _retry_after({"retry-after-ms": "1500"}) == "2"
    assert _retry_after({"retry-after-ms": "soon"}) is None
    assert _retry_after({}) is None


def test_other_upstream_errors_become_502(
    client: TestClient, gateway: Gateway, fake_api: FakeAPI
) -> None:
    """余额不足（402）之类的上游错误回 502，带上上游的报错；不算限流。"""
    fake_api.chat_replies = [NO_BALANCE]
    response = client.post("/b/x/v1/chat/completions", json={"messages": HELLO})
    assert response.status_code == 502
    assert "Insufficient Balance" in response.json()["error"]["message"]
    usage = gateway.usage("x")
    assert usage["errors"] == 1 and usage["rate_limited"] == 0


def test_identical_retry_after_bad_json_gets_a_fresh_reply(
    client: TestClient, gateway: Gateway, fake_api: FakeAPI
) -> None:
    """外部系统（如 Cognee 的整次重试）拿到解析不了的 JSON 后把同样的请求再发一次：网关不从缓存里
    再给同一个坏回复，而是重新调用，新回复覆盖缓存；按桶计入 bad_json_replies 与 json_refreshes。"""
    fake_api.chat_replies = ['{"name": "An', '{"name": "Ana"}']
    body = {"messages": HELLO, "response_format": {"type": "json_object"}}

    def content() -> str:
        reply = client.post("/b/j/v1/chat/completions", json=body)
        return str(reply.json()["choices"][0]["message"]["content"])

    assert content() == '{"name": "An'
    assert content() == '{"name": "Ana"}'
    assert content() == '{"name": "Ana"}'  # 之后照常命中缓存
    assert len(fake_api.chat_requests) == 2
    usage = gateway.usage("j")
    assert usage["bad_json_replies"] == 1 and usage["json_refreshes"] == 1
    assert usage["chat_calls"] == 3 and usage["cache_hits"] == 1

    # json_schema 同样（网关改成 JSON 模式）；不要求 JSON 的回复不检查
    schema = {"type": "json_schema", "json_schema": {"name": "P", "schema": {"type": "object"}}}
    fake_api.chat_replies = ["not json"]
    client.post("/b/k/v1/chat/completions", json={"messages": HELLO, "response_format": schema})
    plain = {"messages": [{"role": "user", "content": "plain"}]}
    client.post("/b/k/v1/chat/completions", json=plain)
    client.post("/b/k/v1/chat/completions", json=plain)
    assert gateway.usage("k")["bad_json_replies"] == 1
    assert gateway.usage("k")["cache_hits"] == 1  # 文字回复照常命中缓存


def test_embedding_errors_become_429_or_502(
    settings: Settings, llm: LLMClient, fake_api: FakeAPI
) -> None:
    """向量器重试用尽仍被限流时回 429（带 Retry-After），其他错误回 502；网关不再重试。"""
    fake_api.embed_dim = settings.embedding.dim
    cfg = settings.embedding.model_copy(update={"max_attempts": 2})
    embedder = DashScopeTextEmbedder(
        cfg, "sk-test", http_client=fake_api.client(), sleep=lambda _: None
    )
    gateway = Gateway(llm, embedder)
    fake_api.embed_status = [429, 429, 400]
    fake_api.embed_retry_after = "3"
    with TestClient(gateway.app()) as test_client:
        limited = test_client.post("/b/e/v1/embeddings", json={"input": ["a"]})
        assert limited.status_code == 429 and limited.headers["retry-after"] == "3"
        assert test_client.post("/b/e/v1/embeddings", json={"input": ["a"]}).status_code == 502
        assert test_client.post("/b/e/v1/embeddings", json={"input": ["a"]}).status_code == 200
    assert len(fake_api.embed_requests) == 4  # 2（向量器自己的重试）+ 1 + 1
    usage = gateway.usage("e")
    assert usage["rate_limited"] == 1 and usage["embedding_errors"] == 2
    assert usage["embedding_calls"] == 3 and usage["embedded_texts"] == 1


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
        # 限流、坏 JSON 等计数与其他计数一样记进写入记录
        for counter in ("rate_limited", "bad_json_replies", "json_refreshes", "embedding_errors"):
            assert stored[counter] == 0 and build[counter] == 0
        second = toy.answer(ctx)  # 已写入：不再重建
        assert "toy_build" not in second.details and toy.builds == 1
        assert stored["protocol"] == f"qwen3.8-flash|{toy.embedder.name}:{toy.embedder.dim}"
        toy.protocol = "deepseek-flash|other:384"  # 换了协议：重建
        third = toy.answer(ctx)
        assert "toy_build" in third.details and toy.builds == 2
    finally:
        toy.gateway.stop()
    assert space_key("locomo:conv-42") == "locomo_conv-42"


def test_external_builds_keep_session_id_schemes_apart(llm: LLMClient, fake_api: FakeAPI) -> None:
    """外部系统的写入里存着会话 ID（证据抬头会显示它）：改写过的另存一处、记下写法；
    原始 ID 的写入沿用原来的位置；改写之前的写入（没记写法）不给改写后的运行用。"""
    fake_api.chat_replies = [
        lambda body: (
            '{"answer": "ok", "evidence_sufficient": true, "citations": []}'
            if "Question:" in json.dumps(body)
            else "ok"
        )
    ]
    when = datetime(2023, 5, 20, 9)
    turns = [EvalTurn(role="user", text="I hike in the Alps.")]
    sample = Sample(
        "longmemeval_s", "q1_abs", "q1_abs", "Where do I hike?", "", "x", "x", True, when
    )
    toy = _Toy()
    try:
        anon = MethodContext(
            sample,
            [EvalSession("session_1", when, turns, source_id="answer_1")],
            Reader(llm),
            None,
            "longmemeval_s:q1_abs",
            "anon",
        )
        raw = MethodContext(
            sample,
            [EvalSession("answer_1", when, turns, source_id="answer_1")],
            Reader(llm),
            None,
            "longmemeval_s:q1_abs",
            "raw",
        )
        # 改写之前的写入：位置与原始 ID 的相同，没有记写法
        legacy = toy.answer(MethodContext(sample, raw.sessions, Reader(llm), None, raw.space_id))
        assert toy.builds == 1 and "toy_build" in legacy.details
        record = toy.root / "longmemeval_s_q1_abs" / "build.json"
        assert "session_ids" not in json.loads(record.read_text("utf-8"))

        assert "toy_build" not in toy.answer(raw).details  # 原始 ID 的运行照常复用
        first = toy.answer(anon)  # 改写后的运行：另存一处，重新写入
        assert toy.builds == 2 and first.details["toy_build"]["session_ids"] == "anon"
        anon_dir = toy.root / "longmemeval_s-anon_q1_abs"
        assert json.loads((anon_dir / "build.json").read_text("utf-8"))["session_ids"] == "anon"
        assert "toy_build" not in toy.answer(anon).details and toy.builds == 2
        assert "toy_build" not in toy.answer(raw).details  # 原始 ID 的写入没被碰过

        # 放错了地方的写入（记的写法对不上）：重建
        (anon_dir / "build.json").write_text(json.dumps({"protocol": toy.protocol}), "utf-8")
        assert "toy_build" in toy.answer(anon).details and toy.builds == 3
    finally:
        toy.gateway.stop()


def test_external_methods_are_registered_lazily() -> None:
    # 适配器模块按需导入；外部包（这里是 graphiti-core）要到第一次回答时才导入，构造不需要它
    assert [m.name for m in build_methods(["MG"])] == ["MG"]
