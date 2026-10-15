"""评测网关与外部基线的公共部分（N12）：OpenAI 兼容的接口、按桶统计、写入缓存与预算。全部离线。"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import ABORT_JSON, NO_BALANCE, RATE_LIMIT, REJECT, FakeAPI
from fastapi.testclient import TestClient
from openai import OpenAI, RateLimitError

from cuemem.config import RoleConfig, Settings
from cuemem.core.trace import Trace, start_trace
from cuemem.embed import build_embedder
from cuemem.embed.dashscope import DashScopeTextEmbedder
from cuemem.errors import ConfigMismatchError
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import (
    ExternalBaseline,
    check_protocol,
    memory_hit,
    protocol_of,
    recorded_protocol,
    space_key,
)
from cuemem.evals.gateway import Gateway, _retry_after, embed_kind
from cuemem.evals.methods import MethodContext, build_methods
from cuemem.evals.report import run_protocol
from cuemem.evals.runner import usage_from_trace
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
        # 像 LangMem、Cognee 那样检索时经网关（search 桶）向量化问题，再像 A-MEM 那样改写一次查询
        url = self.gateway.url(self.bucket(key, "search"))
        sdk = OpenAI(base_url=url, api_key="unused", max_retries=0)
        sdk.embeddings.create(model="toy", input=[sample.question])
        rewrite = [{"role": "user", "content": f"Rewrite as a search query: {sample.question}"}]
        sdk.chat.completions.create(model="toy", messages=rewrite)  # type: ignore[arg-type]
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
        assert stored["build_tag"] == ""
    finally:
        toy.gateway.stop()
    assert space_key("locomo:conv-42") == "locomo_conv-42"


ANSWER = '{"answer": "Berlin", "evidence_sufficient": true, "citations": ["m1"]}'


def _answer_or_ok(body: dict[str, Any]) -> str:
    return ANSWER if "Question:" in json.dumps(body) else "ok"


def _toy_context(llm: LLMClient, question: str = "Where do I live?") -> MethodContext:
    when = datetime(2026, 8, 1, 10)
    session = EvalSession(
        session_id="s1", started_at=when, turns=[EvalTurn(role="user", text="I live in Berlin")]
    )
    sample = Sample(
        "locomo", question, "conv-1", question, "Berlin", "single-hop", "single-hop", False, when
    )
    return MethodContext(sample, [session], Reader(llm), None, "locomo:conv-1")


def test_external_builds_of_another_protocol_are_kept(llm: LLMClient, fake_api: FakeAPI) -> None:
    """评测空间的目录不分协议：另一个协议（抽取模型或向量模型）的写入报错，目录与文件原样不动；
    同一协议、写入标记（build_tag）变了才重建；没记协议的旧记录按当前协议复用。"""
    fake_api.chat_replies = [_answer_or_ok]
    ctx = _toy_context(llm)
    toy = _Toy()
    try:
        toy.answer(ctx)
        directory = toy.root / "locomo_conv-1"
        files = {p.name: p.read_bytes() for p in directory.iterdir()}
        current = toy.protocol
        pds = "deepseek-flash|BAAI/bge-small-en-v1.5:384"
        toy.protocol = pds
        with pytest.raises(ConfigMismatchError) as caught:
            toy.answer(ctx)
        message = str(caught.value)
        assert str(directory) in message and current in message and pds in message
        assert "CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen" in message
        assert {p.name: p.read_bytes() for p in directory.iterdir()} == files  # 一个字节都没动
        assert toy.builds == 1

        toy.protocol = current  # 记协议之前的旧记录：按当前协议复用
        record = directory / "build.json"
        legacy = json.loads(record.read_text("utf-8"))
        del legacy["protocol"], legacy["build_tag"]
        record.write_text(json.dumps(legacy), "utf-8")
        assert "toy_build" not in toy.answer(ctx).details and toy.builds == 1

        toy.build_tag = "toy@2"  # 写法变了（同一协议）：作废重建，记下换掉的是哪一版
        rebuilt = toy.answer(ctx).details["toy_build"]
        assert toy.builds == 2 and rebuilt["build_tag"] == "toy@2"
        assert rebuilt["replaced"] == {"build_tag": ""}
        assert "toy_build" not in toy.answer(ctx).details and toy.builds == 2
        # 早先写入标记接在 protocol 后面（A-MEM、Cognee 现存的写入）：照样认得
        record.write_text(json.dumps({"protocol": f"{current}|toy@2"}), "utf-8")
        assert "toy_build" not in toy.answer(ctx).details and toy.builds == 2
        record.write_text(json.dumps({"protocol": f"{pds}|toy@2"}), "utf-8")
        with pytest.raises(ConfigMismatchError):
            toy.answer(ctx)
        assert json.loads(record.read_text("utf-8")) == {"protocol": f"{pds}|toy@2"}

        # 半成品（有目录、没有 build.json）：照旧清掉重建
        record.unlink()
        (directory / "partial.tmp").write_text("x", "utf-8")
        assert "toy_build" in toy.answer(ctx).details and toy.builds == 3
        assert not (directory / "partial.tmp").exists()
    finally:
        toy.gateway.stop()


def test_search_usage_reaches_the_question(llm: LLMClient, fake_api: FakeAPI) -> None:
    """检索时经网关的请求（search 桶）在网关的服务线程里执行：用量事件转记进这道题的 trace，
    算进 results.jsonl 的 usage；details 的 search_usage 记下这道题经网关的部分。写入不算。"""
    fake_api.chat_replies = [_answer_or_ok]
    toy = _Toy()
    bucket = "toy.search.locomo_conv-1"
    try:
        with start_trace("answer") as trace:
            out = toy.answer(_toy_context(llm))
        assert toy.bucket("locomo_conv-1", "search") == bucket
        searched = out.details["search_usage"]
        assert searched["chat_calls"] == 1 and searched["embedding_calls"] == 1
        assert searched["cost"] == toy.gateway.usage(bucket)["cost"] > 0
        relayed = [e for e in trace.events if e.get("bucket") == bucket]
        assert [e["kind"] for e in relayed] == ["llm"]  # 哈希向量器不记事件
        answered = [e for e in trace.events if e["kind"] == "llm" and "bucket" not in e]
        assert len(answered) == 1  # 回答模型；写入那次 LLM 调用在别的线程，不在这里
        usage = usage_from_trace(trace)
        assert usage["llm_calls"] == 2 and usage["input_tokens"] == 240
        assert usage["cost"] == pytest.approx(searched["cost"] + answered[0]["cost"])
        assert toy.gateway.usage("toy.build.locomo_conv-1")["chat_calls"] == 1
    finally:
        toy.gateway.stop()


def test_search_usage_is_exact_under_parallel_questions(llm: LLMClient, fake_api: FakeAPI) -> None:
    """同一份对话历史的题并行回答（评测的 inner workers）：检索在空间锁里串行，每题只记到
    自己经网关的那一次调用，合起来等于桶里的总数。"""
    fake_api.chat_replies = [_answer_or_ok]
    toy = _Toy()
    questions = [f"Where do I live, take {i}?" for i in range(6)]

    def ask(question: str) -> tuple[dict[str, Any], dict[str, Any]]:
        with start_trace("answer") as trace:
            out = toy.answer(_toy_context(llm, question))
        return out.details["search_usage"], usage_from_trace(trace)

    bucket = "toy.search.locomo_conv-1"
    try:
        toy.answer(_toy_context(llm))  # 先写入
        before = toy.gateway.usage(bucket)
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(ask, questions))
        after = toy.gateway.usage(bucket)
        for searched, usage in results:
            assert searched["chat_calls"] == 1 and searched["embedding_calls"] == 1
            assert usage["llm_calls"] == 2  # 自己的检索一次、回答一次
        assert after["chat_calls"] - before["chat_calls"] == len(questions)
        spent = sum(s["cost"] for s, _ in results)
        assert spent == pytest.approx(after["cost"] - before["cost"]) and spent > 0
    finally:
        toy.gateway.stop()


def test_recorded_protocol_reads_every_existing_record_shape() -> None:
    """现存的写入记录（data/eval-spaces 与 data/eval-spaces-qwen）的几种写法都认得出协议。"""
    pds = "deepseek-flash|BAAI/bge-small-en-v1.5:384"
    assert recorded_protocol({"protocol": f"{pds}|amem@ceffb86|max_tokens=4096|v4"}) == pds
    assert recorded_protocol({"protocol": f"{pds}|cognee==1.6.4|beam-turn-pair"}) == pds
    assert recorded_protocol({"protocol": pds, "build_tag": "x"}) == pds
    # 记协议之前的 Graphiti（embedder 带维度）与 LangMem（维度另记在 dim）；extractor 是别名，
    # 与协议里的抽取模型同一写法
    graphiti = {"graphiti_protocol": 2, "extractor": "deepseek-flash"}
    assert recorded_protocol({**graphiti, "embedder": "BAAI/bge-small-en-v1.5:384"}) == pds
    langmem = {"extractor": "deepseek-flash", "embedder": "BAAI/bge-small-en-v1.5", "dim": 384}
    assert recorded_protocol(langmem) == pds
    thinking = {**langmem, "extractor": "deepseek-flash-thinking"}
    assert recorded_protocol(thinking) == "deepseek-flash-thinking|BAAI/bge-small-en-v1.5:384"
    assert recorded_protocol({"memories": 3, "perspectives": ["Ana"]}) is None  # Mem0 早先的


def test_protocol_tells_apart_settings_of_one_model(offline_settings: Settings) -> None:
    """协议里的抽取模型记配置里的别名：deepseek-flash 与开了思考的 deepseek-flash-thinking 是
    同一个 model_id，写出的记忆不同，算不同的协议，另一个的写入报错（按 model_id 核对时会被
    悄悄复用）。运行记录里的协议（report.run_protocol）同一写法。"""
    embedder = build_embedder(offline_settings, use_cache=False)

    def extractor(model: str) -> Settings:
        roles = {**offline_settings.roles, "extractor": RoleConfig(model=model)}
        return offline_settings.model_copy(update={"roles": roles})

    plain, thinking = extractor("deepseek-flash"), extractor("deepseek-flash-thinking")
    models = offline_settings.models
    assert models["deepseek-flash"].model_id == models["deepseek-flash-thinking"].model_id
    built, current = protocol_of(plain, embedder), protocol_of(thinking, embedder)
    assert built == f"deepseek-flash|{embedder.name}:{embedder.dim}"
    assert current == f"deepseek-flash-thinking|{embedder.name}:{embedder.dim}"
    directory = Path("data/eval-spaces/graphiti/locomo_conv-42")
    with pytest.raises(ConfigMismatchError, match="deepseek-flash-thinking"):
        check_protocol({"protocol": built}, current, directory, "graphiti")
    check_protocol({"protocol": current}, current, directory, "graphiti")
    info = {
        "roles": {"extractor": {"model": "deepseek-flash-thinking", "model_id": "deepseek-flash"}},
        "embedding": {"model": embedder.name, "dim": embedder.dim},
    }
    assert run_protocol(info) == current


def test_gateway_forward_relays_one_bucket(
    settings: Settings, llm: LLMClient, fake_api: FakeAPI
) -> None:
    """forward 期间这个桶的 LLM 与向量事件另外记进给定的 trace（标上桶名）；别的桶、forward
    之外的请求不记；桶里照常累计。"""
    fake_api.embed_dim = settings.embedding.dim
    embedder = DashScopeTextEmbedder(
        settings.embedding, "sk-test", http_client=fake_api.client(), sleep=lambda _: None
    )
    gateway = Gateway(llm, embedder)
    fake_api.chat_replies = ["ok"]
    target = Trace("answer")

    def chat(test_client: TestClient, bucket: str, text: str) -> None:
        body = {"messages": [{"role": "user", "content": text}]}
        assert test_client.post(f"/b/{bucket}/v1/chat/completions", json=body).status_code == 200

    with TestClient(gateway.app()) as test_client:
        with gateway.forward("x.search.k", target) as used:
            chat(test_client, "x.search.k", "one")
            chat(test_client, "other", "two")
            assert test_client.post("/b/x.search.k/v1/embeddings", json={"input": "q"}).is_success
        chat(test_client, "x.search.k", "three")
        with gateway.forward("x.search.k", None) as untraced:  # 不在 trace 里：只累计用量
            chat(test_client, "x.search.k", "four")
    assert [(e["kind"], e["bucket"]) for e in target.events] == [
        ("llm", "x.search.k"),
        ("embedding", "x.search.k"),
    ]
    usage = usage_from_trace(target)
    assert usage["llm_calls"] == 1 and usage["input_tokens"] == 120
    assert usage["embedding_tokens"] == gateway.usage("x.search.k")["embedding_tokens"] > 0
    assert gateway.usage("x.search.k")["chat_calls"] == 3
    assert (used["chat_calls"], used["embedding_calls"]) == (1, 1)
    assert used["cost"] == pytest.approx(usage["cost"], abs=1e-6)  # as_dict 保留 6 位小数
    assert used["models_requested"] == {"": 1}
    assert (untraced["chat_calls"], untraced["embedding_calls"]) == (1, 0)


def test_gateway_forward_leaves_late_requests_in_the_bucket(
    offline_settings: Settings, llm: LLMClient, fake_api: FakeAPI
) -> None:
    """请求按开始时的转记目标记账：一道题的检索放弃了（如 Cognee 的检索超时）、而它发出的请求在
    下一道题检索时才处理完，只进桶，不记到哪道题名下（以前记到了下一道题的 trace 与
    search_usage 里）。"""
    started, release = threading.Event(), threading.Event()

    def slow(body: dict[str, Any]) -> str:
        if "late" in json.dumps(body):
            started.set()
            assert release.wait(10)
        return "ok"

    fake_api.chat_replies = [slow]
    gateway = Gateway(llm, build_embedder(offline_settings, use_cache=False))
    first, second = Trace("q1"), Trace("q2")

    def chat(test_client: TestClient, text: str) -> None:
        body = {"messages": [{"role": "user", "content": text}]}
        assert test_client.post("/b/x.search.k/v1/chat/completions", json=body).is_success

    with TestClient(gateway.app()) as test_client, ThreadPoolExecutor(1) as pool:
        with gateway.forward("x.search.k", first) as used_first:
            late = pool.submit(chat, test_client, "late")
            assert started.wait(10)
        with gateway.forward("x.search.k", second) as used_second:
            release.set()
            late.result(10)
            chat(test_client, "on time")
    assert first.events == []
    assert [(e["kind"], e["bucket"]) for e in second.events] == [("llm", "x.search.k")]
    assert used_first["chat_calls"] == 0 and used_second["chat_calls"] == 1
    assert used_second["cost"] == pytest.approx(usage_from_trace(second)["cost"], abs=1e-6)
    assert gateway.usage("x.search.k")["chat_calls"] == 2  # 桶里两次都有


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
