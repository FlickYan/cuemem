"""OpenAI 兼容的记忆代理（N12，D-062）：想起 → 注入 → 转发 → 写回；流式与非流式。全部离线。"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import threading
import time
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import httpx2
import numpy as np
import pytest
from fakes import RATE_LIMIT, REJECT, TIMEOUT, FakeAPI
from fastapi.testclient import TestClient
from openai import OpenAI
from openai.types.responses import Response as OpenAIResponse
from PIL import Image, UnidentifiedImageError
from scripted import ScriptedModel
from test_sdk import where_i_live

from cuemem import Memory
from cuemem.api import proxy as proxy_module
from cuemem.api.proxy import MemoryProxy, _chat_to_response, _responses_to_chat, create_proxy_app
from cuemem.config import PathsConfig, Settings, load_settings
from cuemem.embed import CachedEmbedder
from cuemem.errors import ConcurrentWriteError, ContentFilteredError, EmbeddingAPIError
from cuemem.memory import MemorySystem
from cuemem.write.consolidate import Consolidator
from cuemem.write.online import worth_retrying

SYSTEM = {"role": "system", "content": "You are a helpful shopping assistant."}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Upstream:
    """上游模型：抽取等内部调用交给脚本化的模型，代理转发来的请求回答固定的话。"""

    def __init__(self) -> None:
        self.model = ScriptedModel(where_i_live)
        self.reply = "Happy to help, Alice!"
        self.proxied: list[dict[str, Any]] = []

    def __call__(self, body: dict[str, Any]) -> str | dict[str, Any]:
        if body["messages"][0].get("content") == SYSTEM["content"]:
            self.proxied.append(body)
            return self.reply
        return self.model(body)


@pytest.fixture
def upstream(fake_api: FakeAPI) -> Upstream:
    model = Upstream()
    fake_api.chat_replies = [model]
    return model


@pytest.fixture
def system(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[MemorySystem]:
    memory = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    yield memory
    memory.close()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def proxy(system: MemorySystem, fake_api: FakeAPI, clock: Clock) -> MemoryProxy:
    return MemoryProxy(system, http_client=fake_api.client(), clock=clock, mode="l1")


@pytest.fixture
def client(proxy: MemoryProxy) -> Iterator[TestClient]:
    with TestClient(create_proxy_app(proxy)) as test_client:
        yield test_client


def chat(
    client: TestClient, text: Any, history: list[dict[str, Any]] | None = None, **extra: Any
) -> Any:
    """发一句话（字符串或内容块）；history 是这段对话之前的消息（不给时是一段新对话的第一句）。"""
    headers = extra.pop("headers", {})
    messages = [SYSTEM, *(history or []), {"role": "user", "content": text}]
    body = {"model": "qwen3.8-flash", "messages": messages}
    return client.post("/v1/chat/completions", json={**body, **extra}, headers=headers)


def test_remembers_across_sessions(
    client: TestClient, proxy: MemoryProxy, system: MemorySystem, upstream: Upstream, clock: Clock
) -> None:
    first = chat(client, "I live in Berlin", user="alice")
    assert first.status_code == 200
    assert first.json()["choices"][0]["message"]["content"] == "Happy to help, Alice!"
    assert first.headers["x-cuemem-level"] == "none"  # 还没有记忆：不检索
    session = first.headers["x-cuemem-session"]
    # 原样转发：user 字段不带给上游，系统提示与消息不变
    sent = upstream.proxied[-1]
    assert "user" not in sent and sent["messages"][-1]["content"] == "I live in Berlin"
    assert sent["model"] == "qwen3.8-flash" and sent["enable_thinking"] is False
    # 写回：用户的话与回答进了原始日志；会话进行中，不抽取
    turns = system.db.get_turns("alice", session)
    assert [(t.role, t.text) for t in turns] == [
        ("user", "I live in Berlin"),
        ("assistant", "Happy to help, Alice!"),
    ]
    assert system.facts("alice") == []

    # 同一段对话的第二轮：前一轮已经在请求里，检索时排除，没有可注入的记忆
    history = [
        {"role": "user", "content": "I live in Berlin"},
        {"role": "assistant", "content": "Happy to help, Alice!"},
    ]
    second = chat(client, "Recommend a coat", history, user="alice")
    assert second.headers["x-cuemem-session"] == session
    assert "<memory>" not in upstream.proxied[-1]["messages"][-1]["content"]

    # 空闲超过 30 分钟：会话结束，后台抽取事实
    clock.now += 31 * 60
    assert proxy.close_idle() == 1
    assert system.worker is not None
    system.worker.wait()
    # 抽取替身把其余的用户发言也各记一条
    assert {f.statement for f in system.facts("alice")} == {
        "The user lives in Berlin.",
        "Recommend a coat",
    }

    # 新的会话：控制器取回事实，放在最后一句用户消息前面
    third = chat(client, "Where should I buy winter boots?", user="alice")
    assert third.headers["x-cuemem-session"] != session
    assert third.headers["x-cuemem-level"] == "L1"
    assert int(third.headers["x-cuemem-recalled"]) >= 1
    last = upstream.proxied[-1]["messages"]
    assert last[0] == SYSTEM  # 系统提示不动（前缀缓存照常命中）
    content = last[-1]["content"]
    assert content.startswith("<memory>\n") and "The user lives in Berlin." in content
    assert content.endswith("</memory>\n\nWhere should I buy winter boots?")
    # 写回的是用户的原话，不带注入的记忆
    stored = system.db.get_turns("alice", third.headers["x-cuemem-session"])
    assert stored[0].text == "Where should I buy winter boots?"


def test_expired_memories_are_not_injected(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    """第五轮体检 P7：过期的记忆（SDK 写入时给了 expiration_date）不进注入的 <memory> 块，画像里
    那一行也去掉。以前只有 SDK 的 search 与 profile 去掉它们，代理照样注入。"""
    memory = Memory.from_system(system)
    yesterday = date.today() - timedelta(days=1)
    memory.add("I live in Berlin", user_id="alice", expiration_date=yesterday, wait=True)
    memory.add("I like jazz", user_id="alice", wait=True)
    response = chat(client, "Where do I live? Do I like jazz?", user="alice")
    assert response.status_code == 200
    content = upstream.proxied[-1]["messages"][-1]["content"]
    assert content.startswith("<memory>\n") and "jazz" in content
    assert "Berlin" not in content


def test_a_new_conversation_starts_a_new_session(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    first = chat(client, "I live in Berlin", user="alice")
    # 30 分钟内开了一段新对话（请求里没有之前的消息）：上一段结束、交给后台抽取；
    # 上一段的原话已经不在请求里，可以被想起来（原始日志写入后立即可检索）
    second = chat(client, "Where should I buy winter boots?", user="alice")
    assert second.headers["x-cuemem-session"] != first.headers["x-cuemem-session"]
    assert "I live in Berlin" in upstream.proxied[-1]["messages"][-1]["content"]
    assert first.headers["x-cuemem-session"] not in system.open_sessions("alice")
    assert system.worker is not None
    system.worker.wait()
    assert "The user lives in Berlin." in {f.statement for f in system.facts("alice")}


def test_sessions_left_open_by_a_crash_are_adopted(
    n3_settings: Settings, fake_api: FakeAPI, upstream: Upstream
) -> None:
    """代理崩溃时还在进行中的会话（没来得及 close_all）：下一个进程第一次碰到这个用户时交给抽取。
    只认进行中的标记：infer=False 写入的会话照样不抽取；调用方指定、马上要接着写的会话不动。"""

    def body(text: str) -> dict[str, Any]:
        return {"model": "qwen3.8-flash", "messages": [SYSTEM, user(text)], "user": "alice"}

    def user(text: str) -> dict[str, str]:
        return {"role": "user", "content": text}

    crashed = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    proxy = MemoryProxy(crashed, http_client=fake_api.client(), mode="l1")
    left = proxy.complete(body("I live in Berlin"), {})[2]["X-CueMem-Session"]
    proxy.complete(body("I like jazz"), {"x-cuemem-session": "kept"})
    Memory.from_system(crashed).add("I like tea", user_id="alice", infer=False)
    assert crashed.open_sessions("alice") == {left, "kept"}
    crashed.close()  # 没有 close_all：进行中的两个会话没人交出去

    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        proxy = MemoryProxy(system, http_client=fake_api.client(), mode="l1")
        proxy.complete(body("Where do I live?"), {"x-cuemem-session": "kept"})
        assert system.worker is not None
        system.worker.wait()
        assert {j["session_id"]: j["status"] for j in system.db.get_jobs("alice")} == {left: "done"}
        assert {f.statement for f in system.facts("alice")} == {"The user lives in Berlin."}
        assert system.open_sessions("alice") == {"kept"}
        assert system.adopt_open("alice") == 0  # 接手过的会话不再有标记


def red_png_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.mark.parametrize("photo", [False, True], ids=["text", "photo"])
def test_forget_purges_the_query_vector_of_a_deleted_message(
    client: TestClient, system: MemorySystem, upstream: Upstream, photo: bool
) -> None:
    """代理拿用户的原话做检索查询（查询向量进了缓存），再把原话写进原始日志：遗忘这条发言时，
    它的查询向量也要从缓存里删掉（向量本身也可能泄露原文）。带照片的消息查询的是文字部分，
    写进日志的是带 "<image>" 标记的原话：按去掉标记的文字也要清。"""
    chat(client, "I live in Berlin", user="alice")
    secret = "My sister Ana's birthday is October 3, here is her photo"
    content: Any = secret
    if photo:
        image = {"type": "image_url", "image_url": {"url": red_png_url()}}
        content = [{"type": "text", "text": secret}, image]
    assert chat(client, content, user="alice").headers["x-cuemem-level"] != "none"
    embedder = system.embedder
    assert isinstance(embedder, CachedEmbedder) and embedder.cached([secret], "query")
    [turn] = [t for t in system.db.get_turns("alice") if t.text.startswith(secret)]
    assert turn.text == (f"{secret} <image>" if photo else secret)
    assert system.forget("alice", turn_ids=[turn.id]).turns_deleted == 1
    assert not embedder.cached([secret], "query")
    traces = system.settings.paths.traces.glob("*.jsonl")
    assert not any("Ana" in p.read_text("utf-8") for p in traces)  # 代理的检索 trace 也删了


@pytest.mark.parametrize("stream", [False, True])
def test_an_answer_built_on_memory_forgotten_meanwhile_is_not_written_back(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    clock: Clock,
    fake_api: FakeAPI,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
) -> None:
    """等上游回答的时候别的调用方删了注入的那条记忆（DELETE /api/v1/memories/{id}）：回答是照
    被删的记忆写的，以前照样写回日志，下次抽取时被删的事实又长了回来（D-068 记下的剩余限制；
    以前写回时才记遗忘次数）。现在想起之前记下，写回时变了就不写回答、记一个 proxy_write_dropped
    事件；用户的话照写（它不是从记忆来的）。流式与非流式一样。"""
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "cuemem.api.proxy.record_event", lambda kind, **data: events.append((kind, data))
    )
    chat(client, "I live in Berlin", user="alice")
    clock.now += 31 * 60
    proxy.close_idle()
    assert system.worker is not None
    system.worker.wait()
    [fact] = [f for f in system.facts("alice") if "Berlin" in f.statement]
    memory = Memory.from_system(system)
    injected: list[bool] = []

    def reply(body: dict[str, Any]) -> str | dict[str, Any]:
        if body["messages"][0].get("content") != SYSTEM["content"]:
            return upstream.model(body)
        injected.append("lives in Berlin" in json.dumps(body["messages"]))
        forget = threading.Thread(target=memory.delete, args=(fact.id,))  # 另一个调用方
        forget.start()
        forget.join()
        return "You live in Berlin, so a warm coat."

    fake_api.chat_replies = [reply]
    response = chat(client, "What coat do I need?", user="alice", stream=stream)
    assert response.status_code == 200 and "Berlin" in response.text  # 调用方照样拿到回答
    assert injected == [True]
    turns = system.db.get_turns("alice")
    assert [(t.role, t.text) for t in turns] == [("user", "What coat do I need?")]
    dropped = [data for kind, data in events if kind == "proxy_write_dropped"]
    assert dropped == [{"space": "alice", "reason": "forgotten", "part": "answer"}]

    fake_api.chat_replies = [upstream]  # 之后没有遗忘：回答照常写回
    history = [{"role": "user", "content": "What coat do I need?"}]
    chat(client, "And boots?", history, user="alice", stream=stream)
    texts = [t.text for t in system.db.get_turns("alice")]
    assert texts == ["What coat do I need?", "And boots?", "Happy to help, Alice!"]


def test_streaming_is_relayed_and_written_back(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    upstream.reply = "Berlin has great coffee"
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen3.8-flash",
            "stream": True,
            "messages": [SYSTEM, {"role": "user", "content": "I live in Berlin"}],
        },
        headers={"X-CueMem-User": "alice", "X-CueMem-Session": "s1"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-cuemem-session"] == "s1"
        lines = [line for line in response.iter_lines() if line.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    text = "".join(
        (json.loads(line[5:])["choices"] or [{"delta": {}}])[0]["delta"].get("content", "")
        for line in lines[:-1]
    )
    assert text == "Berlin has great coffee"
    turns = system.db.get_turns("alice", "s1")
    assert [t.text for t in turns] == ["I live in Berlin", "Berlin has great coffee"]


def test_non_ascii_user_ids_in_the_headers(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    """X-CueMem-User 里不是 ASCII 的 ID（第四轮体检）：原样的 UTF-8 字节（curl）与 RFC 8187 的
    utf-8''%E5%BC%A0%E4%B8%89 都写进 '张三'（以前 Starlette 按 Latin-1 解成 'å¼\\xa0ä¸\\x89'）；
    不是 UTF-8 的 400（OpenAI 格式，param user），不转发、不开会话。请求体的 user 字段照旧。"""
    for text, user in (("raw", "张三".encode()), ("encoded", b"utf-8''%E5%BC%A0%E4%B8%89")):
        response = chat(client, text, headers=[(b"x-cuemem-user", user)])
        assert response.status_code == 200, response.text
    assert chat(client, "body", user="张三").status_code == 200
    assert [t.text for t in system.db.get_turns("张三") if t.role == "user"] == [
        "raw",
        "encoded",
        "body",
    ]
    proxied = len(upstream.proxied)
    refused = chat(client, "bad", headers=[(b"x-cuemem-user", b"utf-8''%FF")], user="张三")
    assert refused.status_code == 400
    error = refused.json()["error"]
    assert error["type"] == "invalid_request_error" and error["param"] == "user"
    assert "X-CueMem-User" in error["message"] and "utf-8''" in error["message"]
    assert len(upstream.proxied) == proxied
    assert {s["space_id"] for s in system.db.list_spaces()} == {"张三"}


def _sse(text: str) -> bytes:
    chunk = {"choices": [{"index": 0, "delta": {"content": text}}]}
    return f"data: {json.dumps(chunk)}\n\n".encode()


class Answer(httpx2.SyncByteStream):
    """上游的流式回答：先说 texts，然后断掉（error）、一直说下去（endless）或以 last 收尾。"""

    def __init__(
        self,
        texts: list[str],
        *,
        error: Exception | None = None,
        endless: bool = False,
        last: bytes = b"data: [DONE]\n\n",
    ) -> None:
        self.texts, self.error, self.endless, self.last = texts, error, endless, last
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from map(_sse, self.texts)
        if self.error is not None:
            raise self.error
        while self.endless:
            yield _sse(" and")
        yield self.last

    def close(self) -> None:
        self.closed = True


def streaming(system: MemorySystem, answer: Answer) -> MemoryProxy:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=answer, headers={"content-type": "text/event-stream"})

    return MemoryProxy(system, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


ASK_PIN = {
    "model": "qwen3.8-flash",
    "stream": True,
    "user": "alice",
    "messages": [{"role": "user", "content": "What is my PIN?"}],
}


def test_a_broken_upstream_stream_is_not_remembered(system: MemorySystem) -> None:
    """上游说到一半断了（连接被重置）：调用方拿到错误，记忆里什么也不写（D-062，与非流式
    相同）。以前把半截的 "Your PIN is 12" 当成完整的回答写进了原始日志。"""
    reset = httpx2.ReadError("connection reset by peer")
    answer = Answer(["Your PIN is", " 12"], error=reset)
    _, chunks, _ = streaming(system, answer).stream(ASK_PIN, {})
    with pytest.raises(httpx2.ReadError):
        list(chunks)
    assert answer.closed and system.db.get_turns("alice") == []


def test_an_error_inside_the_stream_is_not_remembered(system: MemorySystem) -> None:
    """有的上游出错时在流里发 {"error": ...} 再正常关流：同样不写记忆。"""
    error = {"error": {"message": "overloaded", "type": "server_error"}}
    answer = Answer(["Your PIN is"], last=f"data: {json.dumps(error)}\n\n".encode())
    _, chunks, _ = streaming(system, answer).stream(ASK_PIN, {})
    assert b"overloaded" in b"".join(chunks)  # 原样转发给调用方
    assert answer.closed and system.db.get_turns("alice") == []


def test_a_stopped_stream_keeps_only_the_user_message(system: MemorySystem) -> None:
    """调用方中途不要了（用户停下）：关掉上游，只写回用户的话，半截的回答不算回答。"""
    answer = Answer(["Your PIN is"], endless=True)
    _, chunks, _ = streaming(system, answer).stream(ASK_PIN, {})
    for _ in range(4):  # 读过几块回答再停：以前会把读到的半截写成回答
        next(chunks)
    chunks.close()
    assert answer.closed
    assert [(t.role, t.text) for t in system.db.get_turns("alice")] == [("user", "What is my PIN?")]


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
def test_client_disconnect_closes_the_relay_at_once(
    system: MemorySystem, spec_version: str
) -> None:
    """客户端中途断开：路由立即关掉转发，上游的连接马上释放，用户的话按顺序写回。以前
    Starlette 不关同步迭代器，要等垃圾回收才收尾（写回排到下一轮之后，还可能在同一线程上
    等自己持有的会话锁）。2.4 起服务器在 send 时报 OSError，更早的版本由 receive 告知断开。"""
    answer = Answer(["Your PIN is"], endless=True)
    app_ = create_proxy_app(streaming(system, answer))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8787),
    }
    bodies = 0

    async def main() -> None:
        gone = anyio.Event()
        received = False

        async def receive() -> dict[str, Any]:
            nonlocal received
            if not received:
                received = True
                return {"type": "http.request", "body": json.dumps(ASK_PIN).encode()}
            await gone.wait()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            nonlocal bodies
            if message["type"] == "http.response.body":
                bodies += 1
                if bodies >= 3:
                    gone.set()
                    if spec_version == "2.4":
                        raise OSError("client went away")

        with contextlib.suppress(Exception):
            await app_(scope, receive, send)

    anyio.run(main)
    assert bodies >= 3
    assert answer.closed  # 不等垃圾回收
    assert [(t.role, t.text) for t in system.db.get_turns("alice")] == [("user", "What is my PIN?")]


def test_model_mapping_and_agent_scope(
    client: TestClient, system: MemorySystem, upstream: Upstream, fake_api: FakeAPI
) -> None:
    chat(client, "hi", model="deepseek-flash", headers={"X-CueMem-User": "bob"})
    sent = upstream.proxied[-1]
    assert sent["model"] == "deepseek-flash" and sent["thinking"] == {"type": "disabled"}
    chat(client, "hi", model="gpt-unknown", user="bob", headers={"X-CueMem-Agent": "shop"})
    assert upstream.proxied[-1]["model"] == "qwen3.8-flash"  # 不认识的模型：agent 角色的模型
    assert {s["space_id"] for s in system.db.list_spaces()} == {"bob", "shop/bob"}
    models = client.get("/v1/models").json()
    assert {m["id"] for m in models["data"]} >= {"deepseek-flash", "qwen3.8-flash"}


def test_without_a_user_requests_pass_through(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    response = chat(client, "What is the capital of France?")
    assert response.status_code == 200
    assert response.headers["x-cuemem-level"] == "none"
    assert "x-cuemem-session" not in response.headers
    assert system.db.list_spaces() == []


def test_upstream_errors_are_returned_and_not_remembered(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    upstream.reply = REJECT
    response = chat(client, "I live in Berlin", user="alice")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "data_inspection_failed"
    assert system.db.get_turns("alice") == []


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_rate_limits_and_timeouts_reach_the_caller(
    client: TestClient,
    system: MemorySystem,
    upstream: Upstream,
    fake_api: FakeAPI,
    stream: bool,
) -> None:
    """上游限流：429 连同上游的 Retry-After 转给调用方（OpenAI SDK 照它退避；以前只转发了
    响应体）。上游超时：OpenAI 格式的 504（以前是没有说明的 500）。都不写记忆。"""
    upstream.reply = RATE_LIMIT
    fake_api.chat_rate_limit_headers = {"retry-after": "7", "retry-after-ms": "7000"}
    limited = chat(client, "I live in Berlin", user="alice", stream=stream)
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "7" and limited.headers["retry-after-ms"] == "7000"

    upstream.reply = TIMEOUT
    late = chat(client, "I live in Berlin", user="alice", stream=stream)
    assert late.status_code == 504 and late.json()["error"]["type"] == "timeout"
    assert system.db.get_turns("alice") == []


class Outage:
    """向量接口写入时（文档向量）出错，status 为 None 时恢复。"""

    def __init__(self, system: MemorySystem, monkeypatch: pytest.MonkeyPatch) -> None:
        self.status: int | None = 503
        self._embed = system.embedder.embed
        monkeypatch.setattr(system.embedder, "embed", self)

    def __call__(self, texts: Any, kind: Any) -> Any:
        if self.status is not None and kind == "document":
            raise EmbeddingAPIError(f"HTTP {self.status}", status=self.status)
        return self._embed(texts, kind)


def test_write_backs_that_fail_are_retried(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """向量接口暂时不可用时写回失败：留着，下一次写回这个会话时先写它（照它当时的时间），会话
    结束之前再试一次。在线写入要么全写、要么不写（R1）：以前这一轮就永远丢了，之后的请求只写
    各自最后一句用户消息。"""
    outage = Outage(system, monkeypatch)
    before = datetime.now()
    first = chat(client, "I live in Berlin", user="alice")
    after = datetime.now()
    assert first.status_code == 200  # 写不进记忆不影响回答
    session = first.headers["x-cuemem-session"]
    assert system.db.get_turns("alice", session) == []

    outage.status = None
    history = [
        {"role": "user", "content": "I live in Berlin"},
        {"role": "assistant", "content": "Happy to help, Alice!"},
    ]
    chat(client, "I like jazz", history, user="alice")
    stored = system.db.get_turns("alice", session)
    assert [t.text for t in stored] == [
        "I live in Berlin",
        "Happy to help, Alice!",
        "I like jazz",
        "Happy to help, Alice!",
    ]
    assert before <= stored[0].ts <= after < stored[2].ts

    outage.status = 503  # 会话结束之前再写一次
    history += [
        {"role": "user", "content": "I like jazz"},
        {"role": "assistant", "content": "Happy to help, Alice!"},
    ]
    chat(client, "I like tea", history, user="alice")
    outage.status = None
    proxy.close_all()
    assert [t.text for t in system.db.get_turns("alice", session)][4:] == [
        "I like tea",
        "Happy to help, Alice!",
    ]
    system.consolidate("alice")
    assert {f.statement for f in system.facts("alice")} == {
        "The user lives in Berlin.",
        "I like jazz",
        "I like tea",
    }


def test_write_backs_that_cannot_succeed_are_dropped(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """不留着重试的：向量接口拒绝了请求本身（4xx：再写一次还是一样，留着它这个会话后面的写回
    都排在它后面）；写不进去的期间这个用户的记忆被删过（这是删之前说的话）；会话结束时还是写不
    进去的：留着的几次写回一起丢掉、警告一次（以前改成会话结束时照样留着、或只丢第一次接着写
    后面的，都没有测试失败）。"""
    outage = Outage(system, monkeypatch)
    outage.status = 400
    first = chat(client, "I live in Berlin", user="alice", headers={"X-CueMem-Session": "s1"})
    assert first.status_code == 200
    outage.status = None
    chat(client, "I like jazz", headers={"X-CueMem-Session": "s1"}, user="alice")
    assert [t.text for t in system.db.get_turns("alice", "s1")] == [
        "I like jazz",
        "Happy to help, Alice!",
    ]

    outage.status = 503
    chat(client, "My PIN is 4321", user="alice", headers={"X-CueMem-Session": "s1"})
    Memory.from_system(system).delete_all(user_id="alice")
    outage.status = None
    chat(client, "I like tea", user="alice", headers={"X-CueMem-Session": "s1"})
    texts = [t.text for t in system.db.get_turns("alice")]
    assert texts == ["I like tea", "Happy to help, Alice!"]

    outage.status = 503
    chat(client, "I like coffee", user="alice", headers={"X-CueMem-Session": "s1"})
    chat(client, "I like milk", user="alice", headers={"X-CueMem-Session": "s1"})
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="cuemem.api.proxy"):
        proxy.close_all()
    assert [t.text for t in system.db.get_turns("alice")] == texts
    dropped = [r.getMessage() for r in caplog.records if "could not store" in r.getMessage()]
    assert len(dropped) == 1 and "store 4 message(s)" in dropped[0], dropped


def test_write_backs_kept_during_an_outage_are_capped(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """第七轮体检 P14：故障期间一个会话留着重试的写回最多 MAX_UNSAVED 次：超出时丢掉最早的，
    记 proxy_write_dropped（reason=backlog）与一条警告。以前一直留着，长时间故障而调用方一直
    在说话时内存一直长（每次请求可以有 64 MiB 的图片），最后服务被杀、留着的全丢。"""
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "cuemem.api.proxy.record_event", lambda kind, **data: events.append((kind, data))
    )
    monkeypatch.setattr("cuemem.api.proxy.MAX_UNSAVED", 3)
    outage = Outage(system, monkeypatch)
    session = {"X-CueMem-Session": "s1"}
    with caplog.at_level(logging.WARNING, logger="cuemem.api.proxy"):
        for i in range(5):
            assert chat(client, f"note {i}", user="alice", headers=session).status_code == 200
    [run] = proxy._runs.values()
    assert [u.messages[0]["content"] for u in run.unsaved] == ["note 2", "note 3", "note 4"]
    assert run.unsaved_bytes == sum(u.size for u in run.unsaved) > 0
    backlog = [d for kind, d in events if kind == "proxy_write_dropped"]
    assert backlog == [{"space": "alice", "reason": "backlog"}] * 2
    warned = [r.getMessage() for r in caplog.records if "for a long time" in r.getMessage()]
    assert len(warned) == 2 and "store 2 message(s) of proxy session s1" in warned[0]

    outage.status = None
    chat(client, "note 5", user="alice", headers=session)
    assert [t.text for t in system.db.get_turns("alice", "s1") if t.role == "user"] == [
        "note 2",
        "note 3",
        "note 4",
        "note 5",
    ]
    assert run.unsaved == [] and run.unsaved_bytes == 0


def test_write_backs_kept_during_an_outage_are_capped_by_size(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """留着的写回也按大小限制（MAX_UNSAVED_BYTES，图片按 data URL 的长度算）：只按次数限制时
    100 次 64 MiB 的请求就是 6.4 GB。比上限还大的一次照样留着：单个请求已经由请求体上限限住。"""
    image = red_png_url()

    def content(i: int) -> list[dict[str, Any]]:
        return [
            {"type": "text", "text": f"photo {i}"},
            {"type": "image_url", "image_url": {"url": image}},
        ]

    one = len("photo 0") + len(image) + len("Happy to help, Alice!")
    monkeypatch.setattr("cuemem.api.proxy.MAX_UNSAVED_BYTES", 2 * one + 1)
    Outage(system, monkeypatch)
    session = {"X-CueMem-Session": "s1"}
    for i in range(4):
        assert chat(client, content(i), user="alice", headers=session).status_code == 200
    [run] = proxy._runs.values()
    assert [u.messages[0]["content"][0]["text"] for u in run.unsaved] == ["photo 2", "photo 3"]
    assert [u.size for u in run.unsaved] == [one, one] and run.unsaved_bytes == 2 * one

    monkeypatch.setattr("cuemem.api.proxy.MAX_UNSAVED_BYTES", 10)
    chat(client, content(4), user="alice", headers=session)
    assert [u.messages[0]["content"][0]["text"] for u in run.unsaved] == ["photo 4"]
    assert run.unsaved_bytes == one


def test_open_proxy_sessions_per_user_are_capped(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第七轮复审 P14：一个用户同时进行中的代理会话最多 max_open_runs 个
    （CUEMEM_MAX_OPEN_SESSIONS，默认 64）。再开新的时先停放最久没有请求、没有留着写回的那个
    （第八轮体检 P1：会话照旧进行中，见 test_parked_proxy_sessions_are_not_replayed）；都留着写回
    （故障期间）时 429。以前每个会话的写回有上限，会话数却不限：调用方每次换一个
    X-CueMem-Session，故障期间每个都能留 32 MiB。"""
    assert proxy.max_open_runs == 64
    proxy.max_open_runs = 2
    for i, sid in enumerate(("s1", "s2", "s3")):
        clock.now = float(i)
        assert chat(client, f"note {i}", user="alice", headers={"X-CueMem-Session": sid}).is_success
    assert sorted(sid for _, sid in proxy._runs) == ["s2", "s3"]  # s1 最久没有请求：停放了
    assert system.open_sessions("alice") == {"s1", "s2", "s3"}  # 照旧进行中
    assert chat(client, "hi", user="bob", headers={"X-CueMem-Session": "s1"}).is_success  # 各算各的

    outage = Outage(system, monkeypatch)
    for i, sid in enumerate(("s2", "s3")):
        clock.now = 10.0 + i
        chat(client, f"later {i}", user="alice", headers={"X-CueMem-Session": sid})
    refused = chat(client, "one more", user="alice", headers={"X-CueMem-Session": "s4"})
    assert refused.status_code == 429 and refused.headers["retry-after"] == "60"
    error = refused.json()["error"]
    assert error["code"] == "rate_limited" and "CUEMEM_MAX_OPEN_SESSIONS" in error["message"]
    assert sorted(sid for space, sid in proxy._runs if space == "alice") == ["s2", "s3"]
    assert upstream.proxied[-1]["messages"][-1]["content"] != "one more"  # 没有转发

    outage.status = None
    chat(client, "back", user="alice", headers={"X-CueMem-Session": "s2"})  # s2 留着的写进去了
    clock.now = 20.0
    assert chat(client, "one more", user="alice", headers={"X-CueMem-Session": "s4"}).is_success
    # s2 刚有过请求，却是唯一没有留着写回的：停放它；s3 留着的写回不丢
    assert sorted(sid for space, sid in proxy._runs if space == "alice") == ["s3", "s4"]
    assert "s2" in system.open_sessions("alice")
    assert [t.text for t in system.db.get_turns("alice", "s2") if t.role == "user"] == [
        "note 1",
        "later 0",
        "back",
    ]
    assert [u.messages[0]["content"] for u in proxy._runs[("alice", "s3")].unsaved] == ["later 1"]


def test_parked_proxy_sessions_are_not_replayed(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第八轮体检 P1：一个用户并行的会话多于上限时，为腾名额停放的会话照旧进行中，不交给抽取。
    以前直接结束它（交给抽取），调用方回来接着写时会话又打开、长了，下次腾名额时整个空间重放
    （rebuild(replay=True)）：3 个会话轮流、上限 2 时 4 轮重放 3 次。停放的会话闲置太久时照常
    交给抽取，每个会话只抽取一次。"""
    proxy.max_open_runs = 2
    replays: list[bool] = []
    rebuild = Consolidator.rebuild

    def spy(self: Consolidator, space_id: str, *, replay: bool = False) -> Any:
        replays.append(replay)
        return rebuild(self, space_id, replay=replay)

    monkeypatch.setattr(Consolidator, "rebuild", spy)
    sessions = ("s1", "s2", "s3")
    assert system.worker is not None
    for r in range(4):
        for i, sid in enumerate(sessions):
            clock.now = float(r * 3 + i)
            headers = {"X-CueMem-Session": sid}
            assert chat(client, f"note {r} {sid}", user="alice", headers=headers).is_success
        system.worker.wait()
    assert not any(replays)
    assert system.open_sessions("alice") == set(sessions)
    assert system.db.get_jobs("alice") == []  # 进行中：都还没交给抽取
    assert proxy.open_runs() == 2 and proxy.parked_runs() == 1

    clock.now += proxy.idle_s
    assert proxy.close_idle() == 3  # 停放的也闲置结束
    system.worker.wait()
    assert proxy.open_runs() == 0 and proxy.parked_runs() == 0
    assert system.open_sessions("alice") == set()
    jobs = system.db.get_jobs("alice")
    assert sorted(j["session_id"] for j in jobs) == sorted(sessions)
    assert all(j["status"] == "done" for j in jobs)
    assert "note 3 s1" in {f.statement for f in system.facts("alice")}
    assert not any(replays)


def test_parked_proxy_sessions_beyond_the_limit_are_ended(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """停放的会话最多 PARKED_KEPT 个（整个进程）：再多时最早停放的那个交给抽取；退出时
    （close_all）停放的也交出去。"""
    monkeypatch.setattr(proxy_module, "PARKED_KEPT", 1)
    proxy.max_open_runs = 1
    assert system.worker is not None
    for i, sid in enumerate(("s1", "s2")):
        clock.now = float(i)
        assert chat(client, f"note {i}", user="alice", headers={"X-CueMem-Session": sid}).is_success
    assert proxy.parked_runs() == 1 and system.db.get_jobs("alice") == []
    clock.now = 2.0
    assert chat(client, "note 2", user="alice", headers={"X-CueMem-Session": "s3"}).is_success
    system.worker.wait()
    assert proxy.parked_runs() == 1  # 停放了 s2，最早停放的 s1 交给抽取
    assert [j["session_id"] for j in system.db.get_jobs("alice")] == ["s1"]
    assert system.open_sessions("alice") == {"s2", "s3"}

    proxy.close_all()
    system.worker.wait()
    assert proxy.parked_runs() == 0 and system.open_sessions("alice") == set()
    assert sorted(j["session_id"] for j in system.db.get_jobs("alice")) == ["s1", "s2", "s3"]


def test_a_parked_implicit_session_keeps_its_id(
    client: TestClient, proxy: MemoryProxy, system: MemorySystem, upstream: Upstream, clock: Clock
) -> None:
    """没指定会话头的会话也会被停放：调用方接着这段对话时写进原来的会话；开始新的一段时先把它
    交给抽取。"""
    proxy.max_open_runs = 1
    first = chat(client, "I live in Berlin", user="alice")
    session = first.headers["x-cuemem-session"]
    clock.now = 1.0
    assert chat(client, "hi", user="alice", headers={"X-CueMem-Session": "s1"}).is_success
    assert proxy.parked_runs() == 1 and session in system.open_sessions("alice")
    history = [
        {"role": "user", "content": "I live in Berlin"},
        {"role": "assistant", "content": "Happy to help, Alice!"},
    ]
    clock.now = 2.0
    second = chat(client, "Recommend a coat", history, user="alice")
    assert second.headers["x-cuemem-session"] == session
    assert system.db.get_jobs("alice") == []
    clock.now = 3.0
    assert chat(client, "hi again", user="alice", headers={"X-CueMem-Session": "s1"}).is_success
    clock.now = 4.0
    third = chat(client, "A new topic", user="alice")  # 新的一段：停放的那段交给抽取
    assert third.headers["x-cuemem-session"] != session
    assert system.worker is not None
    system.worker.wait()
    assert [j["session_id"] for j in system.db.get_jobs("alice")] == [session]


# 在线写入出错之后：过一会儿原样再写一次有没有可能写进去（记忆代理与 Agents 的会话据此留着或丢掉）
WORTH_RETRYING: list[tuple[BaseException, bool]] = [
    (EmbeddingAPIError("no balance", status=402), True),
    (EmbeddingAPIError("timeout", status=408), True),
    (EmbeddingAPIError("throttled", status=429), True),
    (EmbeddingAPIError("unavailable", status=503), True),
    (EmbeddingAPIError("bad request", status=400), False),
    (EmbeddingAPIError("no response"), True),  # 连不上、超时
    (ConcurrentWriteError("collided"), False),  # 已经写进去一部分，再写就重复了
    (ContentFilteredError("blocked"), False),
    (UnidentifiedImageError("not an image"), False),
    (ValueError("too many pixels"), False),
    (TypeError("bad message"), False),
    (KeyError("k"), False),
    (OSError("qdrant down"), True),
]


@pytest.mark.parametrize(
    ("exc", "expected"),
    WORTH_RETRYING,
    ids=[f"{type(e).__name__}-{i}" for i, (e, _) in enumerate(WORTH_RETRYING)],
)
def test_which_write_errors_are_worth_retrying(exc: BaseException, expected: bool) -> None:
    """worth_retrying 逐行钉住：以前把向量接口的 402 / 408 改成不重试、把撞号
    （ConcurrentWriteError）或不合法的发言（ValueError）改成要重试，都没有测试失败。"""
    assert worth_retrying(exc) is expected


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_a_write_back_with_an_oversized_image_is_dropped_alone(
    client: TestClient,
    proxy: MemoryProxy,
    system: MemorySystem,
    upstream: Upstream,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """图片的像素数超过 Pillow 的 Image.MAX_IMAGE_PIXELS（几十 KB 的 PNG 可以有一亿多像素）：只丢
    这一次写回，这个会话之后的照常写。以前 Pillow 的 DecompressionBombError 不是 ValueError，算作
    值得重试：它一直排在最前面，之后的写回都写不进去，会话结束时全部丢掉。"""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)
    buffer = io.BytesIO()
    Image.new("1", (200, 200)).save(buffer, format="PNG")
    bomb = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    content = [
        {"type": "text", "text": "Look at my cat"},
        {"type": "image_url", "image_url": {"url": bomb}},
    ]
    headers = {"X-CueMem-Session": "s1"}
    assert chat(client, content, user="alice", headers=headers).status_code == 200
    assert chat(client, "My cat is called Momo", user="alice", headers=headers).status_code == 200
    texts = [t.text for t in system.db.get_turns("alice", "s1")]
    assert texts == ["My cat is called Momo", "Happy to help, Alice!"]
    proxy.close_all()
    assert [t.text for t in system.db.get_turns("alice", "s1")] == texts
    assert system.db.get_media("alice") == []


def test_tool_follow_ups_only_store_the_answer(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    body = {
        "model": "qwen3.8-flash",
        "user": "alice",
        "messages": [
            SYSTEM,
            {"role": "user", "content": "Find me boots"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "find", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "P01"},
        ],
    }
    response = client.post("/v1/chat/completions", json=body, headers={"X-CueMem-Session": "t1"})
    assert response.status_code == 200
    assert [(t.role, t.text) for t in system.db.get_turns("alice", "t1")] == [
        ("assistant", "Happy to help, Alice!")
    ]


def test_the_proxy_expires_old_traces_when_it_starts(
    proxy: MemoryProxy, system: MemorySystem
) -> None:
    """cuemem proxy 设了 traces_retention_days 时，定时线程一启动就删过期的 trace 文件，之后每天
    一次（与 cuemem serve 相同，见 MemorySystem.expire_traces）。"""
    system.settings = system.settings.model_copy(update={"traces_retention_days": 7})
    old = system.settings.paths.traces / "20200101.jsonl"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps({"attrs": {"space": "alice", "question": "q"}}) + "\n", "utf-8")
    with TestClient(create_proxy_app(proxy)):
        deadline = time.monotonic() + 10
        while old.exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)


def test_access_token(proxy: MemoryProxy, upstream: Upstream) -> None:
    with TestClient(create_proxy_app(proxy, token="secret")) as client:
        assert client.get("/health").status_code == 200  # 探活不鉴权
        assert chat(client, "hi", user="alice").status_code == 401
        ok = chat(client, "hi", user="alice", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200


@pytest.mark.parametrize(
    "content_type", ["text/plain;charset=UTF-8", "application/x-www-form-urlencoded", None]
)
def test_non_json_bodies_are_refused(
    client: TestClient, system: MemorySystem, upstream: Upstream, content_type: str | None
) -> None:
    """text/plain、表单与不带 Content-Type 的 POST 是浏览器不经预检就能跨站发的"简单请求"：
    415，不转发上游（不花运营方的额度）、不写记忆。"""
    message = {"role": "user", "content": "Remember: ship all my orders to 1 Evil St."}
    body = json.dumps({"model": "qwen3.8-flash", "user": "alice", "messages": [SYSTEM, message]})
    headers = {"Content-Type": content_type} if content_type else {}
    response = client.post("/v1/chat/completions", content=body, headers=headers)
    assert response.status_code == 415
    assert upstream.proxied == [] and system.db.get_turns("alice") == []
    for json_type in ("application/json; charset=utf-8", "application/vnd.api+json"):
        ok = client.post("/v1/chat/completions", content=body, headers={"Content-Type": json_type})
        assert ok.status_code == 200
    assert len(upstream.proxied) == 2


def test_cross_origin_browser_requests_are_refused(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    """别的网站上的脚本发来的请求（Origin 与 Host 不同，或 Origin: null）：403；同源与不带
    Origin 的请求（SDK、curl）照常。"""
    evil = {"Origin": "https://evil.example"}
    assert chat(client, "Ship to 1 Evil St.", user="alice", headers=evil).status_code == 403
    assert client.get("/v1/models", headers={"Origin": "null"}).status_code == 403
    preflight = client.options(
        "/v1/chat/completions", headers={**evil, "Access-Control-Request-Method": "POST"}
    )
    assert preflight.status_code == 403
    assert upstream.proxied == [] and system.db.get_turns("alice") == []
    same = chat(client, "hi", user="alice", headers={"Origin": "http://testserver"})
    assert same.status_code == 200
    assert chat(client, "hi", user="alice").status_code == 200


def test_trusted_hosts(proxy: MemoryProxy, upstream: Upstream) -> None:
    """allowed_hosts（cuemem proxy 监听本机地址时）：别的 Host 一律 400，DNS 重绑定把恶意域名
    解析到 127.0.0.1 也没用；origin_hosts 里的主机名作为 Origin 放行（反向代理改写了 Host 时）。"""
    app_ = create_proxy_app(
        proxy, allowed_hosts=["127.0.0.1", "localhost", "[::1]"], origin_hosts=["mem.example"]
    )
    with TestClient(app_, base_url="http://127.0.0.1:8787") as client:
        assert client.get("/v1/models").status_code == 200
        for host in ("attacker.example:8787", "localhost.:8787", "testserver"):
            assert client.get("/v1/models", headers={"Host": host}).status_code == 400
        assert client.get("/v1/models", headers={"Host": "[::1]:8787"}).status_code == 200
        rebound = chat(client, "hi", user="alice", headers={"Host": "attacker.example"})
        assert rebound.status_code == 400 and upstream.proxied == []
        proxied = {"Origin": "https://mem.example"}
        assert chat(client, "hi", user="alice", headers=proxied).status_code == 200
        other = {"Origin": "http://localhost:3000"}  # 本机别的端口上的网页也是别的来源
        assert chat(client, "hi", user="alice", headers=other).status_code == 403


def test_an_image_round_still_in_the_request_is_not_recalled(
    client: TestClient, system: MemorySystem, upstream: Upstream
) -> None:
    """发过照片的一轮原样还在请求里：不再被想起、重复注入（以前按纯文字比对，写进记忆的
    "… <image>" 永远对不上）。只算真正存下的图片：网络图片不存，也就没有 <image> 标记。"""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    content = [
        {"type": "text", "text": "Here is my red jacket"},
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
    ]
    shown = {"role": "user", "content": content}
    first = client.post(
        "/v1/chat/completions",
        json={"model": "qwen3.8-flash", "user": "alice", "messages": [SYSTEM, shown]},
    )
    session = first.headers["x-cuemem-session"]
    assert system.db.get_turns("alice", session)[0].text == "Here is my red jacket <image>"
    history = [shown, {"role": "assistant", "content": "Happy to help, Alice!"}]
    second = chat(client, "What colour was the jacket?", history, user="alice")
    assert second.headers["x-cuemem-session"] == session
    assert second.headers["x-cuemem-level"] == "L1"
    assert second.headers["x-cuemem-recalled"] == "0"
    assert "<memory>" not in upstream.proxied[-1]["messages"][-1]["content"]


def test_server_side_image_paths_are_not_stored(
    client: TestClient, system: MemorySystem, upstream: Upstream, tmp_path: Path
) -> None:
    secret = tmp_path / "server-only.png"
    Image.new("RGB", (8, 8), "red").save(secret)
    content = [
        {"type": "text", "text": "Look at these"},
        {"type": "image_url", "image_url": {"url": str(secret)}},
        {"type": "image_url", "image_url": {"url": f"file://{secret}"}},
        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
    ]
    body = {
        "model": "qwen3.8-flash",
        "user": "alice",
        "messages": [SYSTEM, {"role": "user", "content": content}],
    }
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    turns = system.db.get_turns("alice")
    assert turns[0].text == "Look at these" and not turns[0].media_ids
    assert system.db.get_media("alice") == []  # 代理所在机器上的文件不会被写进记忆


def test_recall_failure_does_not_fail_the_turn(
    client: TestClient, system: MemorySystem, upstream: Upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat(client, "I live in Berlin", user="alice")

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("vector store is down")

    monkeypatch.setattr(system, "retrieve", broken)
    response = chat(client, "Where do I live?", user="alice")
    assert response.status_code == 200
    assert response.headers["x-cuemem-level"] == "error"
    assert response.json()["choices"][0]["message"]["content"] == "Happy to help, Alice!"


# ---------- 模型、/v1/embeddings 与 /v1/responses（N12 之后） ----------


def sdk_for(client: TestClient) -> OpenAI:
    """官方 OpenAI SDK，经 TestClient 调用代理（不走网络）。"""
    return OpenAI(
        base_url="http://testserver/v1", api_key="unused", http_client=client, max_retries=0
    )


def test_the_model_actually_used_is_named_and_unknown_ones_warned(
    client: TestClient, upstream: Upstream, caplog: pytest.LogCaptureFixture
) -> None:
    """不认识的模型照旧改用 agent 角色的模型（不让已有的客户端出错），但不再悄无声息：响应头
    X-CueMem-Model 写明实际转发的模型，日志里每个模型名记一条警告。以前 model="gpt-4o" 被
    悄悄换成 qwen3.8-flash，响应里看不出来。"""
    hi = [SYSTEM, {"role": "user", "content": "hi"}]
    with caplog.at_level(logging.WARNING, logger="cuemem.api.proxy"):
        first = chat(client, "hi", model="gpt-4o", user="bob")
        passed = chat(client, "hi again", model="gpt-4o")  # 没有用户、原样转发的也写明
        known = chat(client, "hi", model="deepseek-flash", user="bob")
        streamed = {"model": "gpt-4o", "stream": True, "messages": hi}
        with client.stream("POST", "/v1/chat/completions", json=streamed) as response:
            assert response.headers["x-cuemem-model"] == "qwen3.8-flash"
            list(response.iter_lines())
    assert first.status_code == passed.status_code == known.status_code == 200
    assert first.headers["x-cuemem-model"] == passed.headers["x-cuemem-model"] == "qwen3.8-flash"
    assert upstream.proxied[0]["model"] == "qwen3.8-flash"
    assert known.headers["x-cuemem-model"] == "deepseek-flash"
    warned = [r.getMessage() for r in caplog.records if r.name == "cuemem.api.proxy"]
    assert len(warned) == 1 and "'gpt-4o'" in warned[0] and "qwen3.8-flash" in warned[0]


def test_unknown_models_can_be_rejected(
    system: MemorySystem, fake_api: FakeAPI, upstream: Upstream
) -> None:
    """unknown_model="reject"（参数、配置 proxy.unknown_model 或 --unknown-model）：不认识的模型
    400（OpenAI 的 model_not_found），不转发、不开会话、不检索、不写记忆；认识的模型与没给
    model 的请求照常。"""
    proxy = MemoryProxy(system, http_client=fake_api.client(), mode="l1", unknown_model="reject")
    with TestClient(create_proxy_app(proxy)) as client:
        refused = chat(client, "I live in Berlin", model="gpt-4o", user="alice")
        assert refused.status_code == 400
        error = refused.json()["error"]
        assert (error["code"], error["param"]) == ("model_not_found", "model")
        assert "'gpt-4o'" in error["message"] and "qwen3.8-flash" in error["message"]
        streamed = {"model": "gpt-4o", "stream": True, "user": "alice", "messages": [SYSTEM]}
        assert client.post("/v1/chat/completions", json=streamed).status_code == 400
        answered = client.post("/v1/responses", json={"model": "gpt-4o", "input": "hi"})
        assert answered.json()["error"]["code"] == "model_not_found"
        embedded = client.post("/v1/embeddings", json={"model": "text-embedding-3", "input": "x"})
        assert embedded.status_code == 400
        assert upstream.proxied == [] and system.db.list_spaces() == []
        assert system.open_sessions("alice") == set()
        assert chat(client, "hi", model="qwen3.8-flash", user="alice").status_code == 200
        unnamed = {"messages": [SYSTEM, {"role": "user", "content": "hi"}], "user": "alice"}
        assert client.post("/v1/chat/completions", json=unnamed).status_code == 200


@pytest.fixture
def openai_proxy(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_api: FakeAPI
) -> Iterator[MemoryProxy]:
    """--preset openai，只设了 OPENAI_API_KEY：目录里还有默认配置的 Qwen、DeepSeek 别名。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    loaded = load_settings(preset="openai")
    settings = loaded.model_copy(
        update={
            "paths": PathsConfig().resolve_against(tmp_path),
            "peak_policy": "allow",
            "embedding": loaded.embedding.model_copy(update={"backend": "hash", "dim": 256}),
        }
    )
    system = MemorySystem.from_settings(settings, http_client=fake_api.client())
    yield MemoryProxy(system, http_client=fake_api.client(), mode="l1")
    system.close()


def test_models_without_a_key_are_not_forwarded(
    openai_proxy: MemoryProxy, fake_api: FakeAPI, upstream: Upstream
) -> None:
    """目录里供应商的 Key 没设置的模型（--preset openai 下的 qwen3.8-flash）：503
    model_not_available，在开会话、检索与转发之前。以前不带鉴权头照样转发，用户的话连同想起的
    记忆发给了没配置 Key 的供应商。GET /v1/models 也只列转发得了的模型。"""
    system = openai_proxy.system
    with TestClient(create_proxy_app(openai_proxy)) as client:
        refused = chat(client, "I live in Berlin", user="alice")
        assert refused.status_code == 503
        assert refused.headers["x-should-retry"] == "false"
        error = refused.json()["error"]
        assert (error["code"], error["param"]) == ("model_not_available", "model")
        assert "'qwen3.8-flash'" in error["message"] and "DASHSCOPE_API_KEY" in error["message"]
        assert "gpt-5.4-mini" in error["message"] and "qwen3.8-max" not in error["message"]
        streamed = {"model": "deepseek-flash", "stream": True, "user": "bob", "messages": [SYSTEM]}
        assert client.post("/v1/chat/completions", json=streamed).status_code == 503
        answered = client.post(
            "/v1/responses", json={"model": "qwen3.8-flash", "input": "hi", "user": "alice"}
        )
        assert answered.status_code == 503
        assert answered.json()["error"]["code"] == "model_not_available"
        assert fake_api.chat_requests == [] and upstream.proxied == []
        assert system.db.list_spaces() == [] and system.open_sessions("alice") == set()
        listed = [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert "gpt-5.4-mini" in listed
        assert not {"qwen3.8-flash", "deepseek-flash", "qwen3-vl-plus"} & set(listed)
        ok = chat(client, "I live in Berlin", model="gpt-5.4-mini", user="alice")
        assert ok.status_code == 200 and fake_api.chat_requests[-1]["model"] == "gpt-5.4-mini"


def test_unknown_models_rejected_list_only_available_ones(
    openai_proxy: MemoryProxy, upstream: Upstream
) -> None:
    openai_proxy.unknown_model = "reject"
    with TestClient(create_proxy_app(openai_proxy)) as client:
        error = chat(client, "hi", model="gpt-4o", user="alice").json()["error"]
    assert error["code"] == "model_not_found"
    assert "gpt-5.4-mini" in error["message"] and "qwen3.8-flash" not in error["message"]


def test_requests_with_keys_keep_their_auth_header(proxy: MemoryProxy) -> None:
    """默认配置、Key 都设了：鉴权头与以前相同；不需要 Key 的本地供应商照旧用占位 Key。"""
    body = {"model": "qwen3.8-flash", "messages": [SYSTEM]}
    _, url, auth = proxy._upstream(body)
    assert auth == {"Authorization": "Bearer sk-test-dashscope"}
    assert url == "https://maas.qwencloudapi.com/compatible-mode/v1/chat/completions"
    _, _, auth = proxy._upstream({**body, "model": "deepseek-flash"})
    assert auth == {"Authorization": "Bearer sk-test-deepseek"}
    listed = {m["id"] for m in proxy.models()["data"]}
    assert set(proxy.system.settings.models) <= listed


def test_embeddings_pass_through_to_the_configured_embedder(
    client: TestClient, system: MemorySystem, caplog: pytest.LogCaptureFixture
) -> None:
    """/v1/embeddings（以前 404）：交给配置里的向量模型，OpenAI 的格式（float 或 base64，官方
    SDK 缺省请求 base64）。调用方的文本不是记忆：不进向量缓存（遗忘清不到）。"""
    embedder = system.embedder
    assert isinstance(embedder, CachedEmbedder)
    name = embedder.name
    texts = ["red running shoes", "a quiet cafe in Berlin"]
    expected = embedder.inner.embed(texts, "document").dense
    response = client.post("/v1/embeddings", json={"model": name, "input": texts})
    assert response.status_code == 200 and response.headers["x-cuemem-model"] == name
    body = response.json()
    assert body["object"] == "list" and body["model"] == name
    assert [d["index"] for d in body["data"]] == [0, 1]
    assert np.allclose([d["embedding"] for d in body["data"]], expected)
    assert embedder.cached(texts, "document") == set()
    encoded = client.post(
        "/v1/embeddings", json={"input": texts[0], "encoding_format": "base64"}
    ).json()
    vector = np.frombuffer(base64.b64decode(encoded["data"][0]["embedding"]), dtype="<f4")
    assert np.allclose(vector, expected[0])
    result = sdk_for(client).embeddings.create(model=name, input=texts)
    assert np.allclose([d.embedding for d in result.data], expected)
    assert name in {m["id"] for m in client.get("/v1/models").json()["data"]}
    with caplog.at_level(logging.WARNING, logger="cuemem.api.proxy"):  # 不认识的模型：照用
        other = client.post("/v1/embeddings", json={"model": "text-embedding-3", "input": "x"})
    assert other.status_code == 200 and other.headers["x-cuemem-model"] == name
    assert any("text-embedding-3" in r.getMessage() for r in caplog.records)
    for bad in (
        {"input": []},
        {"input": [1, 2, 3]},
        {"input": ["fine", ""]},
        {"input": "x", "dimensions": embedder.dim + 1},
        {"input": "x", "encoding_format": "int8"},
    ):
        refused = client.post("/v1/embeddings", json=bad)
        assert refused.status_code == 400 and refused.json()["error"]["param"], bad


def test_embedding_errors_are_explained_like_the_rest(
    client: TestClient, proxy: MemoryProxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/v1/embeddings 的向量接口出错时，状态码与 HTTP 接口、对话转发的相同（api/failures.py）：
    限流 429 与暂时不可用 503 带上游的 Retry-After，余额不足 402，超时 504，别的 502。以前除了
    限流都是 502（余额不足、超时也一样），503 也不带 Retry-After。"""
    timeout = EmbeddingAPIError("向量接口网络错误：timed out")
    timeout.__cause__ = httpx2.ReadTimeout("timed out")
    for error, status, retry in (
        (EmbeddingAPIError("HTTP 429", status=429, retry_after="3"), 429, "3"),
        (EmbeddingAPIError("HTTP 503", status=503, retry_after="9"), 503, "9"),
        (EmbeddingAPIError("HTTP 402", status=402), 402, None),
        (timeout, 504, None),
        (EmbeddingAPIError("HTTP 500", status=500), 502, None),
    ):

        def failing(texts: Any, kind: str, error: Exception = error) -> Any:
            raise error

        monkeypatch.setattr(proxy._embedder, "embed", failing)
        response = client.post("/v1/embeddings", json={"input": "x"})
        assert response.status_code == status, response.text
        assert response.headers.get("retry-after") == retry
        assert str(error) in response.json()["error"]["message"]


def test_responses_api_recalls_injects_and_writes_back(
    client: TestClient, proxy: MemoryProxy, system: MemorySystem, upstream: Upstream, clock: Clock
) -> None:
    """/v1/responses（以前 404）：非流式、纯文字输入的子集，换成 Chat Completions 走同样的想起、
    注入、转发与写回；官方 OpenAI SDK 的 responses.create 与 output_text 照常可用。"""
    first = sdk_for(client).responses.create(
        model="qwen3.8-flash",
        instructions=SYSTEM["content"],
        input="I live in Berlin",
        user="alice",
        max_output_tokens=200,
    )
    assert first.output_text == "Happy to help, Alice!" and first.status == "completed"
    assert first.usage is not None and first.usage.input_tokens == 120
    sent = upstream.proxied[-1]
    assert sent["messages"] == [SYSTEM, {"role": "user", "content": "I live in Berlin"}]
    assert sent["max_tokens"] == 200 and "user" not in sent
    assert [(t.role, t.text) for t in system.db.get_turns("alice")] == [
        ("user", "I live in Berlin"),
        ("assistant", "Happy to help, Alice!"),
    ]
    clock.now += 31 * 60  # 会话结束，后台抽取
    assert proxy.close_idle() == 1
    assert system.worker is not None
    system.worker.wait()

    asked = {"type": "input_text", "text": "Where should I buy winter boots?"}
    raw = client.post(
        "/v1/responses",
        json={
            "model": "qwen3.8-flash",
            "instructions": SYSTEM["content"],
            "input": [{"type": "message", "role": "user", "content": [asked]}],
            "user": "alice",
            "metadata": {"app": "shop"},
            "store": True,
            "tools": [],
            "stream": False,
        },
    )
    assert raw.status_code == 200, raw.text
    assert raw.headers["x-cuemem-level"] == "L1" and int(raw.headers["x-cuemem-recalled"]) >= 1
    assert raw.headers["x-cuemem-model"] == "qwen3.8-flash"
    content = upstream.proxied[-1]["messages"][-1]["content"]
    assert content.startswith("<memory>\n") and "The user lives in Berlin." in content
    response = OpenAIResponse.model_validate(raw.json())  # 严格按 OpenAI 的类型校验
    assert response.output_text == "Happy to help, Alice!"
    assert response.metadata == {"app": "shop"} and response.store is False
    session = raw.headers["x-cuemem-session"]
    assert system.db.get_turns("alice", session)[0].text == "Where should I buy winter boots?"


def test_responses_input_becomes_chat_messages() -> None:
    body = {
        "model": "m",
        "instructions": "Be brief.",
        "input": [
            {"role": "developer", "content": "Answer in English."},
            {"role": "user", "content": "I live in Berlin"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Noted.", "annotations": []}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": "Boots?"}]},
        ],
        "temperature": 0.2,
        "top_p": None,
    }
    assert _responses_to_chat(body) == {
        "model": "m",
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "system", "content": "Answer in English."},
            {"role": "user", "content": "I live in Berlin"},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "Boots?"},
        ],
        "temperature": 0.2,
    }


def test_a_truncated_answer_is_an_incomplete_response() -> None:
    data = {
        "model": "deepseek-flash",
        "created": 5,
        "choices": [{"message": {"content": "Half"}, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "total_tokens": 14,
            "prompt_cache_hit_tokens": 6,
        },
    }
    out = OpenAIResponse.model_validate(_chat_to_response({"max_output_tokens": 4}, data, "x"))
    assert out.status == "incomplete" and out.incomplete_details is not None
    assert out.incomplete_details.reason == "max_output_tokens"
    assert out.output_text == "Half" and out.model == "deepseek-flash"
    assert out.usage is not None and out.usage.input_tokens_details.cached_tokens == 6


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        ({"stream": True}, "stream"),
        ({"tools": [{"type": "function", "name": "f", "parameters": {}}]}, "tools"),
        ({"previous_response_id": "resp_1"}, "previous_response_id"),
        ({"text": {"format": {"type": "json_object"}}}, "text"),
        ({"input": [{"role": "user", "content": [{"type": "input_image"}]}]}, "input_image"),
        ({"input": [{"type": "function_call_output", "call_id": "c"}]}, "function_call_output"),
    ],
)
def test_responses_outside_the_subset_are_501(
    client: TestClient,
    system: MemorySystem,
    upstream: Upstream,
    extra: dict[str, Any],
    named: str,
) -> None:
    """子集以外的（流式、工具、存下来的回答、结构化输出、图片与工具调用的输入）：501，OpenAI 格式
    的出错写明哪一项，x-should-retry: false 让官方 SDK 不重试；不转发、不碰记忆。"""
    body = {"model": "qwen3.8-flash", "input": "I live in Berlin", "user": "alice", **extra}
    response = client.post("/v1/responses", json=body)
    assert response.status_code == 501 and response.headers["x-should-retry"] == "false"
    assert named in response.json()["error"]["message"]
    assert upstream.proxied == [] and system.db.list_spaces() == []


def test_responses_errors(client: TestClient, system: MemorySystem, upstream: Upstream) -> None:
    for bad in ({"input": []}, {"input": 5}, {"input": "x", "instructions": ["x"]}):
        refused = client.post("/v1/responses", json={"model": "qwen3.8-flash", **bad})
        assert refused.status_code == 400, bad
    upstream.reply = REJECT  # 上游的出错原样返回，不写记忆
    body = {"model": "qwen3.8-flash", "instructions": SYSTEM["content"], "input": "hi"}
    rejected = client.post("/v1/responses", json={**body, "user": "alice"})
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "data_inspection_failed"
    assert system.db.get_turns("alice") == []
