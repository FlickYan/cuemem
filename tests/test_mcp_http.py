"""MCP 的 Streamable HTTP（cuemem serve 的 /mcp，api/mcp.py 的 McpEndpoint）：令牌、作用域、Host
与 Origin 的检查与 /api 相同，工具与 stdio（cuemem mcp）相同。全部离线。

以前 MCP 只有 stdio：一个进程一个空间、独占数据目录，开着 cuemem serve 时 Claude Code 用不上
同一份记忆，也没有多租户的 MCP；/mcp 每个请求都是 404。"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fakes import FakeAPI
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripted import ScriptedModel, echo_extraction
from starlette.datastructures import Headers

from cuemem.agent import Agent
from cuemem.api.app import LOOPBACK_HOSTS, create_app
from cuemem.api.auth import Keys
from cuemem.api.mcp import McpEndpoint, build_server
from cuemem.api.service import ChatService
from cuemem.config import Settings
from cuemem.memory import MemorySystem

ADMIN_TOKEN = "admin-token-0123456789"
ALICE_TOKEN = "alice-token-0123456789"
SHOP_TOKEN = "shop-token-0123456789"
KEYS = f"""
keys:
  - name: alice-app
    token: {ALICE_TOKEN}
    scopes: [{{user: alice}}]
  - name: shop-backend
    token: {SHOP_TOKEN}
    scopes: [{{user: "*", agent: shop}}]
"""
URL = "http://127.0.0.1:7860"
JSON_RPC = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 0,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}


def headers(token: str | None = None, **extra: str) -> dict[str, str]:
    out = dict(JSON_RPC)
    if token is not None:
        out["Authorization"] = f"Bearer {token}"
    return out | extra


def call(name: str, **arguments: Any) -> dict[str, Any]:
    params = {"name": name, "arguments": arguments}
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}


def said(response: Any) -> str:
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    return "".join(c.get("text", "") for c in result["content"])


@pytest.fixture
def service(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[ChatService]:
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    svc = ChatService(system, Agent(system, router="rule"), default_space="shop-demo")
    yield svc
    svc.close()


@pytest.fixture
def app(service: ChatService, tmp_path: Path) -> FastAPI:
    # 与 cuemem serve 监听 127.0.0.1 时相同的 Host 白名单（cli._host_policy）
    return create_app(
        service,
        upload_dir=tmp_path / "uploads",
        ui=False,
        keys=Keys.parse(KEYS, ADMIN_TOKEN),
        allowed_hosts=list(LOOPBACK_HOSTS),
    )


def turns(service: ChatService) -> dict[str, list[str]]:
    return {
        row["space_id"]: [t.text for t in service.system.db.get_turns(row["space_id"])]
        for row in service.spaces()
    }


def test_tokens_and_scopes_are_checked_before_mcp(app: FastAPI, service: ChatService) -> None:
    """令牌不对 401；X-CueMem-User、X-CueMem-Agent 选的用户在作用域之外 403，什么都不写；不给
    时是令牌自己的用户（Principal.home），管理令牌与全是通配的令牌没有，400。"""
    with TestClient(app, base_url=URL) as http:
        for token in (None, "wrong"):
            refused = http.post("/mcp", json=INITIALIZE, headers=headers(token))
            assert refused.status_code == 401, refused.text
            assert refused.headers["www-authenticate"] == "Bearer"
            assert refused.json() == {"detail": "缺少或错误的访问令牌"}
        alice = headers(ALICE_TOKEN)
        assert http.post("/mcp", json=INITIALIZE, headers=alice).status_code == 200
        saved = http.post("/mcp", json=call("remember", text="I like tea"), headers=alice)
        assert said(saved).startswith("Saved as ")
        for scope in (
            {"X-CueMem-User": "bob"},
            {"X-CueMem-User": "alice", "X-CueMem-Agent": "agent"},  # = agent/alice
            {"X-CueMem-Agent": "alice"},
            {"X-CueMem-User": "alice/../bob"},
        ):
            response = http.post(
                "/mcp",
                json=call("remember", text="PIN 0000"),
                headers=headers(ALICE_TOKEN, **scope),
            )
            assert response.status_code == 403, scope
            assert "cannot access the memories of" in response.json()["detail"]
        for token in (SHOP_TOKEN, ADMIN_TOKEN):
            response = http.post("/mcp", json=call("profile"), headers=headers(token))
            assert response.status_code == 400 and "X-CueMem-User" in response.json()["detail"]
        shop = headers(SHOP_TOKEN, **{"X-CueMem-User": "carol", "X-CueMem-Agent": "shop"})
        said(http.post("/mcp", json=call("remember", text="I like coffee"), headers=shop))
        admin = headers(ADMIN_TOKEN, **{"X-CueMem-User": "dave"})
        said(http.post("/mcp", json=call("remember", text="I like juice"), headers=admin))
        assert turns(service) == {
            "alice": ["I like tea"],
            "shop/carol": ["I like coffee"],
            "dave": ["I like juice"],
        }
        assert "tea" in said(http.post("/mcp", json=call("recall", query="tea"), headers=alice))


def test_non_ascii_user_ids_in_the_headers(service: ChatService, tmp_path: Path) -> None:
    """X-CueMem-User 里不是 ASCII 的 ID（第四轮体检）：Starlette 按 Latin-1 解请求头，curl 发的
    UTF-8 '张三' 以前悄悄写进 'å¼\\xa0ä¸\\x89' 这个空间（只能访问张三的令牌是 403）；Node 的 fetch
    （Claude Code 的 --header）与 httpx 根本发不出 '张三'。现在原样的 UTF-8 字节与 RFC 8187 的
    utf-8''%E5%BC%A0%E4%B8%89 都是 '张三'；不是 UTF-8、带控制字符或写错的 utf-8'' 是 400。"""
    zhang = "zhang-token-0123456789"
    keys = Keys.parse(
        f"keys: [{{name: zhang, token: {zhang}, scopes: [{{user: 张三}}]}}]", ADMIN_TOKEN
    )
    app = create_app(
        service, upload_dir=tmp_path / "u", ui=False, keys=keys, allowed_hosts=list(LOOPBACK_HOSTS)
    )

    def raw(token: str, **ids: bytes) -> list[tuple[bytes, bytes]]:
        out = [(k.encode(), v.encode()) for k, v in headers(token).items()]
        return out + [(f"x-cuemem-{name}".encode(), value) for name, value in ids.items()]

    utf8 = "张三".encode()
    encoded = b"utf-8''%E5%BC%A0%E4%B8%89"
    with TestClient(app, base_url=URL) as http:
        for token, text, user in (
            (ADMIN_TOKEN, "raw UTF-8", utf8),
            (ADMIN_TOKEN, "percent-encoded", encoded),
            (zhang, "scoped raw", utf8),
            (zhang, "scoped encoded", encoded.upper().replace(b"UTF-8''", b"UTF-8'zh'")),
        ):
            said(http.post("/mcp", json=call("remember", text=text), headers=raw(token, user=user)))
        agent = raw(ADMIN_TOKEN, user=b"bob", agent=b"utf-8''caf%C3%A9")
        said(http.post("/mcp", json=call("remember", text="agent"), headers=agent))
        for bad in (b"utf-8''%FF", b"utf-8''%0Abob", b"utf-8''%E5%BC", b"utf-8''a b"):
            response = http.post("/mcp", json=call("profile"), headers=raw(ADMIN_TOKEN, user=bad))
            assert response.status_code == 400, bad
            detail = response.json()["detail"]
            assert "X-CueMem-User" in detail and "utf-8''" in detail
        response = http.post("/mcp", json=call("profile"), headers=raw(zhang, user=b"bob"))
        assert response.status_code == 403
    assert turns(service) == {
        "张三": ["raw UTF-8", "percent-encoded", "scoped raw", "scoped encoded"],
        "café/bob": ["agent"],
    }
    # Node 的 fetch 与 requests 把 'José' 按 Latin-1 发：不是 UTF-8 的字节照旧按 Latin-1 解，以前
    # 就对（TestClient 会把请求头重新编码成 UTF-8，带不了 Latin-1 的字节，所以直接调用）
    endpoint = McpEndpoint(service, keys)
    latin1 = Headers(
        raw=[(b"authorization", f"Bearer {ADMIN_TOKEN}".encode()), (b"x-cuemem-user", b"Jos\xe9")]
    )
    assert endpoint._space(latin1) == "José"
    assert endpoint._space(Headers(raw=[*latin1.raw[:1], (b"x-cuemem-user", b"alice")])) == "alice"


def test_the_browser_and_host_checks_cover_mcp(app: FastAPI) -> None:
    """别的网站发来的浏览器请求 403（SameOriginOnly，以前 /mcp 不在它的路径里）；Host 由整个应用
    的白名单检查（DNS 重绑定 400），不带端口的 Host 照常（MCP SDK 自己的检查是 421）。"""
    alice = headers(ALICE_TOKEN)
    with TestClient(app, base_url=URL) as http:
        evil = http.post(
            "/mcp", json=INITIALIZE, headers=alice | {"Origin": "https://evil.example"}
        )
        assert evil.status_code == 403
        same = http.post("/mcp", json=INITIALIZE, headers=alice | {"Origin": URL})
        assert same.status_code == 200
        rebound = alice | {"Host": "attacker.example:7860"}
        assert http.post("/mcp", json=INITIALIZE, headers=rebound).status_code == 400
    with TestClient(app, base_url="http://127.0.0.1") as http:
        assert http.post("/mcp", json=INITIALIZE, headers=alice).status_code == 200


def test_the_tools_are_the_same_as_over_stdio(app: FastAPI, service: ChatService) -> None:
    """工具的名字、说明与参数，以及返回的文字，与 cuemem mcp（stdio）逐字相同。"""
    stdio = build_server(service, "alice")
    listed = asyncio.run(stdio.list_tools())
    expected = [t.model_dump(mode="json", exclude_none=True, by_alias=True) for t in listed]
    alice = headers(ALICE_TOKEN)
    with TestClient(app, base_url=URL) as http:
        tools = http.post(
            "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=alice
        )
        assert tools.json()["result"]["tools"] == expected
        empty = said(http.post("/mcp", json=call("profile"), headers=alice))
        assert empty == "Nothing is known about the user yet."
        result = asyncio.run(stdio.call_tool("profile", {}))
        assert "".join(getattr(c, "text", "") for c in result.content) == empty


def test_each_request_uses_its_own_space(app: FastAPI, service: ChatService) -> None:
    """并发的请求各自写进自己的空间：空间经 ContextVar 交给在线程池里执行的工具（MCP SDK 或
    anyio 换了做法时这里失败）。"""

    async def run() -> list[int]:
        async with app.router.lifespan_context(app):
            transport = httpx2.ASGITransport(app=app)
            async with httpx2.AsyncClient(transport=transport, base_url=URL) as http:

                async def one(i: int) -> int:
                    response = await http.post(
                        "/mcp",
                        json=call("remember", text=f"note {i}"),
                        headers=headers(ADMIN_TOKEN, **{"X-CueMem-User": f"u{i}"}),
                    )
                    return response.status_code

                return await asyncio.gather(*(one(i) for i in range(12)))

    assert asyncio.run(run()) == [200] * 12
    assert turns(service) == {f"u{i}": [f"note {i}"] for i in range(12)}


def test_the_app_can_start_twice_and_without_mcp(service: ChatService, tmp_path: Path) -> None:
    """MCP 的会话管理器只能 run 一次：应用的 lifespan 每次进入都新建一个。mcp=False（cuemem serve
    --no-mcp）时没有 /mcp。"""
    app = create_app(service, upload_dir=tmp_path / "uploads", ui=False)
    user = headers(**{"X-CueMem-User": "alice"})
    for _ in range(2):
        with TestClient(app, base_url=URL) as http:
            assert http.post("/mcp", json=INITIALIZE, headers=user).status_code == 200
    without = create_app(service, upload_dir=tmp_path / "uploads", ui=False, mcp=False)
    with TestClient(without, base_url=URL) as http:
        assert http.post("/mcp", json=INITIALIZE, headers=user).status_code == 404


def test_mcp_works_beside_the_ui(service: ChatService, tmp_path: Path) -> None:
    """界面挂在根路径上（Gradio 的 lifespan 套在外面）：/mcp 照常。"""
    pytest.importorskip("gradio")
    app = create_app(service, upload_dir=tmp_path / "uploads")
    with TestClient(app, base_url=URL) as http:
        user = headers(**{"X-CueMem-User": "alice"})
        assert said(http.post("/mcp", json=call("remember", text="hi"), headers=user))
    assert turns(service) == {"alice": ["hi"]}
