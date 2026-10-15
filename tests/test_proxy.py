"""OpenAI 兼容的记忆代理（N12，D-062）：想起 → 注入 → 转发 → 写回；流式与非流式。全部离线。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fakes import REJECT, FakeAPI
from fastapi.testclient import TestClient
from PIL import Image
from scripted import ScriptedModel
from test_sdk import where_i_live

from cuemem.api.proxy import MemoryProxy, create_proxy_app
from cuemem.config import Settings
from cuemem.memory import MemorySystem

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
    client: TestClient, text: str, history: list[dict[str, Any]] | None = None, **extra: Any
) -> Any:
    """发一句话；history 是这段对话之前的消息（不给时是一段新对话的第一句）。"""
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


def test_access_token(proxy: MemoryProxy, upstream: Upstream) -> None:
    with TestClient(create_proxy_app(proxy, token="secret")) as client:
        assert client.get("/health").status_code == 200  # 探活不鉴权
        assert chat(client, "hi", user="alice").status_code == 401
        ok = chat(client, "hi", user="alice", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200


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
