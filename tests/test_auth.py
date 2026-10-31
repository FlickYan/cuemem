"""多租户的访问令牌（api/auth.py）：限定作用域的令牌只能访问自己的用户。/api、/api/v1、界面
（/gradio_api）与记忆代理逐个接口检查，并且试着越出作用域：路径里的 %2F 与 %252F、Agent 的空间
（agent/<x> 与 agent_id="agent"）、filters 里换用户、按 ID 找别人的记忆、接手别人的会话、导入到
别人的空间、代理的请求头。全部离线。"""

from __future__ import annotations

import base64
import functools
import hashlib
import io
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import starlette.requests
from fakes import FakeAPI
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from scripted import ScriptedModel, echo_extraction

from cuemem import MemoryClient
from cuemem.agent import Agent
from cuemem.api import auth
from cuemem.api.app import create_app
from cuemem.api.auth import ADMIN, Keys, Principal, Scope
from cuemem.api.proxy import MemoryProxy, create_proxy_app
from cuemem.api.service import ChatService
from cuemem.config import Settings
from cuemem.errors import AccessKeysError, MemoryAPIError
from cuemem.memory import MemorySystem
from cuemem.sdk import space_for, space_owner

ADMIN_TOKEN = "admin-token-0123456789"
ALICE_TOKEN = "alice-token-0123456789"
SHOP_TOKEN = "shop-token-0123456789"
SHOP_USERS_TOKEN = "shop-users-token-0123456789"


def _sha256(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


KEYS = f"""
keys:
  - name: alice-app
    sha256: {_sha256(ALICE_TOKEN)}
    scopes:
      - user: alice
  - name: shop-backend
    token: {SHOP_TOKEN}
    scopes:
      - {{user: "*", agent: shop}}
      - {{agent: shop}}
  - name: shop-users
    token: {SHOP_USERS_TOKEN}
    scopes:
      - {{user: "*", agent: shop}}
"""
# 每个空间各记一句"秘密"：alice 的令牌只能访问前两个，shop-backend 只能访问 shop 下面的与
# shop 自己的，shop-users 只能访问 shop 下面的用户（没有 {agent: shop} 那一项，碰不到 shop
# 自己的记忆）。agent/alice 是 Agent alice 自己的记忆（也是 space_for("alice", "agent")），
# alice/bob 是用户 bob 在名叫 alice 的 Agent 下的记忆
SPACES = ["alice", "shop/alice", "bob", "shop/bob", "agent/shop", "agent/alice", "alice/bob"]
REACH = {
    ALICE_TOKEN: {"alice", "shop/alice"},
    SHOP_TOKEN: {"shop/alice", "shop/bob", "agent/shop"},
    SHOP_USERS_TOKEN: {"shop/alice", "shop/bob"},
    ADMIN_TOKEN: set(SPACES),
}


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------- 作用域 ----------


@pytest.mark.parametrize(
    ("user_id", "agent_id"),
    [
        ("alice", None),
        ("alice", "shop"),
        (None, "shop"),
        ("a/b", None),
        ("%2F", "x/y"),
        ("100%", "%25"),
        ("agent", "shop"),
        (None, "agent"),
    ],
)
def test_space_owner_inverts_space_for(user_id: str | None, agent_id: str | None) -> None:
    assert space_owner(space_for(user_id, agent_id)) == (user_id, agent_id)


def test_space_owner_of_the_agent_collision_and_foreign_spaces() -> None:
    """agent_id="agent" 的用户空间与 Agent 自己的空间是同一个 ID：一律算 Agent 的。不是由
    space_for 得来的空间解不出主人（只有管理令牌能访问）。"""
    assert space_for("alice", "agent") == space_for(agent_id="alice") == "agent/alice"
    assert space_owner("agent/alice") == (None, "alice")
    for raw in ("", "/", "a/b/c", "alice/../bob", "a%2fb", "a%2", "agent/", "/alice", "x/"):
        assert space_owner(raw) is None, raw


def test_scopes_cover_only_the_named_users_and_agents() -> None:
    """{user: "*", agent: shop} 只管 shop 下面的用户，shop 自己的记忆要另有一项 {agent: shop}
    （以前唯一带它的令牌两项都有，把两条规则混成一条也没有测试失败）。"""
    keys = Keys.parse(KEYS)
    alice, shop, users = (keys.resolve(t) for t in (ALICE_TOKEN, SHOP_TOKEN, SHOP_USERS_TOKEN))
    assert alice is not None and shop is not None and users is not None
    for token, principal in ((ALICE_TOKEN, alice), (SHOP_TOKEN, shop), (SHOP_USERS_TOKEN, users)):
        assert {s for s in SPACES if principal.allows(s)} == REACH[token]
    assert not users.allows("agent/shop") and not Scope(("*",), ("shop",)).covers(None, "shop")
    for escape in ("alice%2Fbob", "alice/../bob", "x/y/alice", "", "Alice", "alice ", "agent/x"):
        assert not alice.allows(escape), escape
    assert alice.allows("bot%2Fx/alice") and not alice.allows(None)
    assert not shop.allows("shop")  # 名叫 shop 的用户，不是 Agent shop
    prefix = Principal("acme", scopes=(Scope(users=("acme:*",)),))
    assert prefix.allows("acme:bob") and not prefix.allows("acme") and not prefix.allows("acmex")
    assert ADMIN.allows("locomo/conv-26/x") and not alice.allows("locomo/conv-26/x")
    assert alice.home() == "alice" and shop.home() == "agent/shop"


def test_tokens_are_compared_as_digests_against_every_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """令牌只比 sha256 摘要，与每个令牌都比一遍（不提前返回，耗时与哪个令牌对上无关）。"""
    keys = Keys.parse(KEYS, ADMIN_TOKEN)
    calls: list[tuple[bytes, bytes]] = []

    def compare(a: bytes, b: bytes) -> bool:
        calls.append((a, b))
        return a == b

    monkeypatch.setattr(auth.secrets, "compare_digest", compare)
    for token, name in ((ADMIN_TOKEN, "admin"), (ALICE_TOKEN, "alice-app"), ("nope", None)):
        calls.clear()
        found = keys.resolve(token)
        assert (found.name if found else None) == name
        assert len(calls) == 4 and all(len(a) == len(b) == 32 for a, b in calls)
        assert all(token.encode() not in a + b for a, b in calls)


def _keys(*entries: str) -> str:
    return "keys:\n" + "".join(f"  - {{{entry}}}\n" for entry in entries)


ZEROS = "0" * 64


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("keys: [", "YAML"),
        ("{}", "keys"),
        (_keys("name: a, token: t, scopes: [{user: a}], admin: true"), "admin"),
        (_keys("name: a, scopes: [{user: a}]"), "sha256 与 token"),
        (_keys(f"name: a, token: t, sha256: '{ZEROS}', scopes: [{{user: a}}]"), "只写一个"),
        (_keys("name: a, sha256: abc, scopes: [{user: a}]"), "64"),
        (_keys("name: ' a', token: t, scopes: [{user: a}]"), "name"),
        (_keys("name: a, token: t, scopes: []"), "scopes"),
        (_keys("name: a, token: t, scopes: [{}]"), "user 与 agent"),
        (_keys("name: a, token: t, scopes: [{user: 'a*b'}]"), "a*b"),
        (_keys("name: a, token: t, scopes: [{user: []}]"), "空列表"),
        (_keys("name: a, token: t, scopes: [{user: a, team: x}]"), "team"),
        (
            _keys(
                "name: a, token: t, scopes: [{user: a}]", "name: a, token: u, scopes: [{user: b}]"
            ),
            "名字重复",
        ),
        (
            _keys(
                "name: a, token: t, scopes: [{user: a}]", "name: b, token: t, scopes: [{user: b}]"
            ),
            "两次",
        ),
    ],
)
def test_bad_key_files_are_refused(text: str, problem: str) -> None:
    with pytest.raises(AccessKeysError) as caught:
        Keys.parse(text, source="keys.yaml")
    assert problem in str(caught.value)


def test_key_errors_never_echo_a_token() -> None:
    """写错的值可能就是令牌：出错信息只说哪里错、错在哪，不带值。"""
    secret = "s3cr3t-value-0123456789"
    for text in (
        f"keys:\n  - {{name: a, token: [{secret}], scopes: [{{user: a}}]}}",
        f"keys:\n  - {{name: a, token: t, scopes: [{{user: a}}]}}\n  - {secret}",
        f"keys:\n  - {{name: a, token: {secret}, scopes: [{{user: a}}]\n",  # YAML 本身写错
    ):
        with pytest.raises(AccessKeysError) as caught:
            Keys.parse(text)
        assert secret not in str(caught.value)
    with pytest.raises(AccessKeysError, match="管理令牌"):
        Keys.parse(f"keys:\n  - {{name: a, token: {secret}, scopes: [{{user: a}}]}}", secret)


def test_keys_from_the_environment(tmp_path: Path) -> None:
    path = tmp_path / "keys.yaml"
    path.write_text(KEYS, encoding="utf-8")
    from_file = Keys.from_env({"CUEMEM_API_KEYS_FILE": str(path), "CUEMEM_API_TOKEN": ADMIN_TOKEN})
    assert (
        from_file.names == ["alice-app", "shop-backend", "shop-users"]
        and from_file.admin is not None
    )
    inline = Keys.from_env({"CUEMEM_API_KEYS": json.dumps({"keys": [
        {"name": "a", "token": "t", "scopes": [{"user": ["a", "b"]}]}
    ]})})  # fmt: skip
    found = inline.resolve("t")
    assert inline.admin is None and found is not None and found.allows("b")
    assert not Keys.from_env({}).enabled and not Keys.from_env({"CUEMEM_API_TOKEN": ""}).enabled
    with pytest.raises(AccessKeysError, match="只能设一个"):
        Keys.from_env({"CUEMEM_API_KEYS_FILE": str(path), "CUEMEM_API_KEYS": KEYS})
    with pytest.raises(AccessKeysError, match="读不出"):
        Keys.from_env({"CUEMEM_API_KEYS_FILE": str(tmp_path / "missing.yaml")})


def test_short_plaintext_tokens_are_recorded_by_name() -> None:
    """Keys.weak：短于 16 个字符的令牌（管理令牌记作 CUEMEM_API_TOKEN，限定作用域的记它的名字）。
    cuemem serve / proxy 监听非本机地址时据此拒绝启动；Keys 只存摘要，量不出长度，所以在读令牌时
    记下。sha256: 量不出长度，不记；不带令牌本身，也不参与相等比较。"""
    keys = Keys.parse(_keys("name: k, token: x, scopes: [{user: a}]"))
    assert keys.weak == ("k",)
    assert Keys.from_token("a").weak == ("CUEMEM_API_TOKEN",)
    assert Keys.from_token("a" * 16).weak == () and Keys.from_token(None).weak == ()
    assert Keys.parse(KEYS, ADMIN_TOKEN).weak == ()  # 摘要与 16 个字符以上的令牌
    mixed = Keys.parse(
        _keys(
            "name: alice-app, token: 换成你的令牌, scopes: [{user: alice}]",
            f"name: hashed, sha256: '{_sha256('x')}', scopes: [{{user: bob}}]",
            f"name: long, token: {'b' * 16}, scopes: [{{user: carol}}]",
        ),
        "your-token",
    )
    assert mixed.weak == ("CUEMEM_API_TOKEN", "alice-app")
    assert "your-token" not in repr(mixed) and "换成你的令牌" not in repr(mixed)
    assert mixed == Keys(mixed.admin, mixed.scoped)
    env = Keys.from_env({"CUEMEM_API_KEYS": _keys("name: k, token: x, scopes: [{user: a}]")})
    assert env.weak == ("k",)


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        (None, None),
        ("", None),
        ("alice", "alice"),
        ("a%2Fb", "a%2Fb"),  # 不以 utf-8' 开头的 ASCII 原样
        ("张三".encode().decode("latin-1"), "张三"),  # curl 原样发的 UTF-8，Starlette 按 Latin-1 解
        ("José".encode().decode("latin-1"), "José"),
        ("José", "José"),  # Node 的 fetch、requests 发的 Latin-1：不是 UTF-8，原样（以前就对）
        ("utf-8''%E5%BC%A0%E4%B8%89", "张三"),
        ("UTF-8'zh-CN'%e5%bc%a0%e4%b8%89", "张三"),
        ("utf-8''alice", "alice"),
        ("utf-8''", None),
    ],
)
def test_header_ids(wire: str | None, expected: str | None) -> None:
    """请求头 X-CueMem-User、X-CueMem-Agent 里的 ID（第四轮体检，见 auth.header_id）。"""
    assert auth.header_id(wire) == expected


@pytest.mark.parametrize(
    "wire", ["utf-8''%FF", "utf-8''%E5%BC", "utf-8''%0Abob", "utf-8''a b", "utf-8'", "utf-8''%zz"]
)
def test_bad_header_ids(wire: str) -> None:
    """utf-8' 开头却写错、解出来不是 UTF-8 或带控制字符：400，说明是哪个头、该怎么发。"""
    with pytest.raises(ValueError):
        auth.header_id(wire)
    with pytest.raises(ValueError, match=r"X-CueMem-Agent .*utf-8''%E5%BC%A0%E4%B8%89"):
        auth.identity_headers({"x-cuemem-user": "alice", "x-cuemem-agent": wire})


def test_ui_login_binds_a_scoped_key_to_its_name() -> None:
    """界面登录：限定作用域的令牌要用它的名字做用户名（登录后只知道用户名），管理令牌用户名任意。
    用户名对不上令牌的名字时，不会被当成管理令牌。"""
    keys = Keys.parse(KEYS, ADMIN_TOKEN)
    assert keys.login("alice-app", ALICE_TOKEN) and not keys.login("admin", ALICE_TOKEN)
    assert not keys.login("shop-backend", ALICE_TOKEN) and not keys.login("alice-app", "nope")
    assert keys.login("anyone", ADMIN_TOKEN)
    alice = keys.by_username("alice-app")
    assert alice is not None and not alice.admin and keys.by_username("anyone") == ADMIN
    assert keys.by_username(None) is None
    scoped_only = Keys.parse(KEYS)  # 没有管理令牌：别的用户名根本登录不进来，也不会成为管理员
    assert scoped_only.by_username("anyone") is None
    assert Keys().by_username(None) == ADMIN  # 没有配令牌：不登录，不限制


# ---------- HTTP ----------


@dataclass
class World:
    app: FastAPI
    http: TestClient
    service: ChatService
    system: MemorySystem
    proxy: MemoryProxy
    fake_api: FakeAPI
    facts: dict[str, str]  # 空间 → 那句秘密抽成的事实

    def turns(self) -> dict[str, int]:
        return {s: len(self.system.db.get_turns(s)) for s in SPACES}

    def forwarded(self) -> int:
        """代理转发给上游的请求数（后台抽取的请求不算）。"""
        return sum(
            "PIN" in str(r["messages"][-1].get("content")) for r in self.fake_api.chat_requests
        )


@pytest.fixture
def world(n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path) -> Iterator[World]:
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"), default_space="shop-demo")
    proxy = MemoryProxy(system, http_client=fake_api.client(), mode="l1")
    keys = Keys.parse(KEYS, ADMIN_TOKEN)
    app = create_app(service, upload_dir=tmp_path / "uploads", ui=False, keys=keys, proxy=proxy)
    facts = {}
    for space in SPACES:
        service.remember(space, f"Secret of {space}")
        service.consolidate(space)
        (fact,) = system.facts(space)
        facts[space] = fact.id
    assert system.worker is not None
    system.worker.wait()  # 后台的抽取做完，测试里数得清请求与发言
    try:
        with TestClient(app) as http:
            yield World(app, http, service, system, proxy, fake_api, facts)
    finally:
        proxy.close_all()
        service.close()


SPACE_ROUTES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("GET", "profile", None),
    ("GET", "facts", None),
    ("GET", "entities", None),
    ("GET", "graph", None),
    ("GET", "media", None),
    ("GET", "media/0123456789abcdef", None),
    ("GET", "sessions", None),
    ("GET", "status", None),
    ("GET", "export", None),
    ("POST", "recall", {"query": "secret", "mode": "l1"}),
    ("POST", "forget", {"session_ids": ["s1"], "dry_run": True}),
    ("POST", "consolidate", None),
]


@pytest.mark.parametrize("token", [ALICE_TOKEN, SHOP_TOKEN, SHOP_USERS_TOKEN, ADMIN_TOKEN])
def test_api_space_routes_stay_in_scope(world: World, token: str) -> None:
    """/api/spaces/{space}/… 的每个接口：作用域里的照常，之外的 403；"/" 写成 %2F 一样。"""
    headers = bearer(token)
    for space in SPACES:
        for path in (space, space.replace("/", "%2F")):
            for method, route, body in SPACE_ROUTES:
                response = world.http.request(
                    method, f"/api/spaces/{path}/{route}", json=body, headers=headers
                )
                if space in REACH[token]:
                    assert response.status_code in (200, 404), (path, route, response.text)
                else:
                    assert response.status_code == 403, (path, route)
                    assert response.json()["detail"].startswith("这个访问令牌不能访问记忆空间")
    listed = world.http.get("/api/spaces", headers=headers).json()
    assert {row["space_id"] for row in listed} == REACH[token]


def test_space_status_is_scoped_and_forgotten(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """/api/spaces/{space}/status 只给这个空间的后台出错：以前是全部空间最近的 3 条，alice 的令牌
    借自己的空间就看得到 bob 的空间 ID 与出错（结构化输出不合格时出错里引着模型的输出）。遗忘 bob
    的会话之后，他的出错也不再给出。"""
    model = ScriptedModel(echo_extraction)

    def reply(body: dict[str, Any]) -> Any:
        if "PIN" in str(body["messages"][-1]["content"]) or "schema" in str(body["messages"]):
            return '{"facts": [{"statement": "s", "negated": "My bank PIN is 4711"}]}'
        return model(body)

    fake_api.chat_replies = [reply]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"))
    keys = Keys.parse(KEYS, ADMIN_TOKEN)
    app = create_app(service, upload_dir=tmp_path / "uploads", ui=False, keys=keys)
    try:
        sid = service.remember("bob", "My bank PIN is 4711")["session_id"]
        assert system.worker is not None
        system.worker.wait()
        assert any(e.startswith("bob: ") for e in system.worker.errors)
        with TestClient(app) as http:
            mine = http.get("/api/spaces/alice/status", headers=bearer(ALICE_TOKEN))
            assert mine.status_code == 200 and mine.json()["errors"] == []
            assert "bob" not in mine.text and "PIN" not in mine.text
            his = http.get("/api/spaces/bob/status", headers=bearer(ADMIN_TOKEN)).json()
            assert his["failed"] == 1 and his["errors"]
            forget = {"session_ids": [sid]}
            assert http.post("/api/spaces/bob/forget", json=forget, headers=bearer(ADMIN_TOKEN))
            his = http.get("/api/spaces/bob/status", headers=bearer(ADMIN_TOKEN)).json()
            assert his == {"pending": 0, "failed": 0, "done": 0, "errors": []}
        assert not any(e.startswith("bob: ") for e in system.worker.errors)
    finally:
        service.close()


def test_api_escapes_through_the_path_are_refused(world: World) -> None:
    """alice 的令牌：编码过的 "/"、"%"、".." 解出来都不是 alice 的空间；真的遗忘也做不到。"""
    headers = bearer(ALICE_TOKEN)
    before = world.turns()
    for path in (
        "bob",
        "alice%2Fbob",  # Agent alice 下的用户 bob
        "alice%252Fbob",  # 用户 "alice/bob"
        "alice%2F..%2Fbob",
        "agent%2Falice",  # Agent alice 自己的记忆
        "agent/alice",
        "shop/bob",
        "x/y/alice",
        "%2Fbob",
    ):
        for route in ("facts", "export"):
            response = world.http.get(f"/api/spaces/{path}/{route}", headers=headers)
            assert response.status_code == 403, (path, route)
        forgot = world.http.post(
            f"/api/spaces/{path}/forget",
            json={"fact_ids": list(world.facts.values())},
            headers=headers,
        )
        assert forgot.status_code == 403, path
    # 在自己的空间里报别人的 ID：遗忘只在这个空间里找，别人的记忆不受影响
    theirs = [
        t for s in SPACES if s not in REACH[ALICE_TOKEN] for t in world.system.db.get_turns(s)
    ]
    forgot = world.http.post(
        "/api/spaces/alice/forget",
        json={
            "fact_ids": [f for s, f in world.facts.items() if s not in REACH[ALICE_TOKEN]],
            "turn_ids": [t.id for t in theirs],
            "session_ids": sorted({t.session_id for t in theirs}),
        },
        headers=headers,
    )
    assert forgot.status_code in (200, 404) and forgot.json().get("turns_deleted", 0) == 0
    assert world.turns() == before
    assert world.http.get("/api/spaces/%61lice/facts", headers=headers).status_code == 200  # alice


def test_every_api_route_needs_a_token(world: World) -> None:
    """/api 与 /v1 下的每个接口（探活除外）都要令牌：以后新加的接口忘了挂 guard 时这里失败。"""
    checked = 0
    for path, methods in world.app.openapi()["paths"].items():
        if not path.startswith(("/api/", "/v1/")) or path == "/api/health":
            continue
        url = re.sub(r"\{[^}]+\}", "x", path)
        for method in methods:
            response = world.http.request(method.upper(), url, json={})
            assert response.status_code == 401, (method, url)
            checked += 1
    assert checked > 30  # /api、/api/v1 与 /v1 的全部接口


def test_a_bad_token_is_refused_before_the_body_is_read(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """令牌不对时在读请求体之前就 401。以前 FastAPI 先把整个请求体读完、解析完（任意大的 JSON；
    表单里的文件先写进临时文件），才轮到 guard 检查令牌：不带令牌就能让服务器花掉内存与磁盘，
    写坏的 JSON 回 422 而不是 401。"""
    http = world.http
    for headers in ({}, bearer("wrong")):
        broken = {**headers, "content-type": "application/json"}
        for path in ("/api/v1/memories", "/api/sessions"):
            response = http.post(path, content=b"{not json", headers=broken)
            assert response.status_code == 401, (path, response.text)

    def unread(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the body was read before the token was checked")

    for name in ("body", "json", "form"):
        monkeypatch.setattr(starlette.requests.Request, name, unread)
    for headers in ({}, bearer("wrong")):
        for path, body in (
            ("/api/v1/memories", {"messages": "hi", "user_id": "alice"}),
            ("/api/v1/memories/search", {"query": "hi", "user_id": "alice"}),
            ("/api/v1/import", {"data": {"version": 2}, "user_id": "alice"}),
            ("/api/sessions", {"space": "alice"}),
            ("/api/spaces/alice/recall", {"query": "hi"}),
            ("/api/spaces/import", {"version": 2, "space_id": "alice"}),
            ("/v1/chat/completions", {"messages": []}),  # 记忆代理以前就是这样
        ):
            response = http.post(path, json=body, headers=headers)
            assert response.status_code == 401, (path, response.text)
            assert response.headers["www-authenticate"] == "Bearer"
            if path.startswith("/api/v1/"):
                assert response.json()["error"]["code"] == "unauthorized"
            elif path.startswith("/api/"):
                assert response.json() == {"detail": "缺少或错误的访问令牌"}
        upload = http.post(
            "/api/sessions/x/messages",
            data={"text": "hi"},
            files={"images": ("a.png", b"\0" * 4096, "image/png")},
            headers=headers,
        )
        assert upload.status_code == 401 and upload.headers["www-authenticate"] == "Bearer"
    # 请求体仍在接口的签名里：OpenAPI 不变（/docs 的 Authorize 也照旧）
    operation = world.app.openapi()["paths"]["/api/v1/memories"]["post"]
    assert "requestBody" in operation and operation["security"] == [{"HTTPBearer": []}]


def test_api_sessions_and_import_stay_in_scope(world: World) -> None:
    """开会话（含默认空间）、用别人的会话 ID 发消息与结束（当作不存在，404）、导入到别人的空间。"""
    alice, admin = bearer(ALICE_TOKEN), bearer(ADMIN_TOKEN)
    http = world.http
    assert http.post("/api/sessions", json={"space": "bob"}, headers=alice).status_code == 403
    assert http.post("/api/sessions", json={}, headers=alice).status_code == 403  # shop-demo
    own = http.post("/api/sessions", json={"space": "alice"}, headers=alice)
    assert own.status_code == 200 and own.json()["space"] == "alice"
    bobs = http.post("/api/sessions", json={"space": "bob"}, headers=admin).json()["session_id"]
    before, calls = world.turns(), len(world.fake_api.chat_requests)
    sent = http.post(f"/api/sessions/{bobs}/messages", data={"text": "hi"}, headers=alice)
    assert sent.status_code == 404 and sent.json() == {"detail": f"没有进行中的会话 {bobs}"}
    ended = http.post(f"/api/sessions/{bobs}/end", headers=alice)
    assert ended.status_code == 404 and bobs in world.service.sessions  # 还开着
    assert world.turns() == before and len(world.fake_api.chat_requests) == calls

    dump = http.get("/api/spaces/bob/export", headers=admin).json()
    for params in ({}, {"space": "bob"}, {"space": "agent/alice"}):
        refused = http.post("/api/spaces/import", json=dump, params=params, headers=alice)
        assert refused.status_code == 403, params
    assert world.turns() == before
    imported = http.post("/api/spaces/import", json=dump, params={"space": "alice"}, headers=alice)
    assert imported.status_code == 200 and imported.json()["space_id"] == "alice"
    assert len(world.system.db.get_turns("alice")) == before["alice"] + 1


def _forbidden(call: Callable[[], Any]) -> None:
    with pytest.raises(MemoryAPIError) as caught:
        call()
    assert caught.value.status == 403 and caught.value.code == "forbidden"


def test_v1_api_stays_in_scope(world: World) -> None:
    """/api/v1（MemoryClient）：点名作用域之外的用户（包括 filters 里的、Agent 的空间）403；
    按 ID 找别人的记忆与不存在的 ID 一样。reset 见 test_v1_reset_stays_in_scope。"""
    alice = MemoryClient("http://testserver", api_key=ALICE_TOKEN, http_client=world.http)
    admin = MemoryClient("http://testserver", api_key=ADMIN_TOKEN, http_client=world.http)
    before = world.turns()
    outside: list[dict[str, Any]] = [
        {"user_id": "bob"},
        {"user_id": "alice", "agent_id": "agent"},  # = agent/alice：Agent alice 自己的记忆
        {"agent_id": "alice"},
        {"user_id": "bob", "agent_id": "alice"},  # = alice/bob
        {"user_id": "alice/../bob"},
        {"user_id": "alice%2Fbob"},
    ]
    for scope in outside:
        calls: list[Callable[[], Any]] = [
            functools.partial(alice.add, "My PIN is 0000", infer=False, **scope),
            functools.partial(alice.search, "secret", mode="l1", **scope),
            functools.partial(alice.get_all, **scope),
            functools.partial(alice.delete_all, **scope),
            functools.partial(alice.profile, **scope),
            functools.partial(alice.answer, "What is the secret?", **scope),
            functools.partial(alice.export, **scope),
            functools.partial(alice.consolidate, **scope),
            functools.partial(alice.open_run, "r1", **scope),
            functools.partial(alice.end_run, "r1", **scope),
        ]
        for call in calls:
            _forbidden(call)
    _forbidden(lambda: alice.search("secret", user_id="alice", filters={"user_id": "bob"}))
    _forbidden(lambda: alice.get_all(filters={"user_id": "bob"}))
    _forbidden(lambda: alice.get_all(user_id="alice", filters={"agent_id": "agent"}))
    assert world.turns() == before  # 什么都没写、没删

    # 自己的照常
    mine = alice.get_all(filters={"user_id": "alice"})["results"]
    assert [f["memory"] for f in mine] == ["Secret of alice"]
    assert alice.get_all(user_id="alice", agent_id="shop")["results"]
    assert alice.add("I like tea", user_id="alice", infer=False)["status"] == "logged"
    assert {s["space_id"] for s in alice.spaces()} == REACH[ALICE_TOKEN]
    assert alice.get(world.facts["alice"]) is not None and alice.history(world.facts["alice"])

    # 按 ID：别人的记忆与不存在的一样
    bob = world.facts["bob"]
    assert alice.get(bob) is None and alice.history(bob) == []
    for call in (
        functools.partial(alice.update, bob, "My PIN is 0000"),
        functools.partial(alice.delete, bob),
        functools.partial(alice.delete, bob, dry_run=True),
    ):
        with pytest.raises(KeyError):
            call()
    theirs = world.http.get(f"/api/v1/memories/{bob}", headers=bearer(ALICE_TOKEN)).json()
    none = world.http.get("/api/v1/memories/0000000000000000", headers=bearer(ALICE_TOKEN)).json()
    assert theirs["error"] | {"message": ""} == none["error"] | {"message": ""}
    kept = admin.get(bob)
    assert kept is not None and kept["memory"] == "Secret of bob" and kept["status"] == "current"

    # 导入：文件里是 bob 的空间，或者点名 bob，都不行；导进自己的可以
    dump = admin.export(user_id="bob")
    _forbidden(lambda: alice.load(dump))
    _forbidden(lambda: alice.load(dump, user_id="bob"))
    assert alice.load(dump, user_id="alice")["space_id"] == "alice"
    assert len(world.system.db.get_turns("bob")) == before["bob"]
    assert {s["space_id"] for s in admin.spaces()} == set(SPACES)


def test_v1_reset_stays_in_scope(world: World) -> None:
    """限定作用域的令牌 reset：只删它作用域里的空间（每个都能用 delete_all 删掉），只结束这些
    空间里还开着的 run；别的空间、别的令牌开的 run 不碰。返回值与管理令牌的 reset 同样的键。
    以前限定作用域的令牌一律 403（Mem0 的测试收尾 client.reset() 换成 MemoryClient 就出错）。"""
    alice = MemoryClient("http://testserver", api_key=ALICE_TOKEN, http_client=world.http)
    shop = MemoryClient("http://testserver", api_key=SHOP_TOKEN, http_client=world.http)
    admin = MemoryClient("http://testserver", api_key=ADMIN_TOKEN, http_client=world.http)
    mine = alice.open_run("mine", user_id="alice")
    theirs = shop.open_run("theirs", user_id="bob", agent_id="shop")
    before = world.turns()

    out = alice.reset()
    assert out["message"] == "All memories in this key's scope deleted"
    assert sorted(out["spaces_deleted"]) == sorted(REACH[ALICE_TOKEN])
    assert out["turns_deleted"] == sum(before[s] for s in REACH[ALICE_TOKEN])
    assert out.keys() == {"message", "spaces_deleted", "turns_deleted", "images_deleted"}
    assert {s["space_id"] for s in admin.spaces()} == set(SPACES) - REACH[ALICE_TOKEN]
    assert {s: n for s, n in world.turns().items() if s not in REACH[ALICE_TOKEN]} == {
        s: n for s, n in before.items() if s not in REACH[ALICE_TOKEN]
    }
    assert world.system.open_sessions("alice") == set()  # alice 的 run 结束了
    assert world.system.open_sessions("shop/bob") == {theirs}  # shop 的还开着
    assert mine == "mine"

    everything = admin.reset()  # 管理令牌照旧删全部、结束全部
    assert everything["message"] == "All memories deleted"
    assert admin.spaces() == [] and world.system.open_sessions("shop/bob") == set()


def test_v1_excerpt_and_image_ids_stay_in_scope(world: World) -> None:
    """按 ID 找的对话片段与图片（search 给出的另外两种 ID）：别人的与不存在的一样，alice 的令牌
    get、media 是 None，delete（含预演）与 update 是 KeyError，bob 的发言一句不少；管理令牌照常
    取得到。片段的 ID 由空间、会话与序号派生，算得出来：Memory.delete 认了片段的 ID 之后，
    /api/v1 若仍只按事实查作用域，限定作用域的令牌就能删掉别人的对话。"""
    alice = MemoryClient("http://testserver", api_key=ALICE_TOKEN, http_client=world.http)
    admin = MemoryClient("http://testserver", api_key=ADMIN_TOKEN, http_client=world.http)
    photo = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(photo, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(photo.getvalue()).decode()
    content = [{"type": "text", "text": "My photo"}, {"type": "image_url", "image_url": url}]
    admin.add({"role": "user", "content": content}, user_id="bob", infer=False)
    excerpt = world.system.db.list_units("bob", kinds=["round"])[0].id
    (image,) = world.system.db.get_media("bob")
    before = world.turns()
    for memory_id in (excerpt, image.id):
        assert alice.get(memory_id) is None and alice.history(memory_id) == []
        assert alice.media(memory_id) is None
        for call in (
            functools.partial(alice.delete, memory_id),
            functools.partial(alice.delete, memory_id, dry_run=True),
            functools.partial(alice.update, memory_id, "My PIN is 0000"),
        ):
            with pytest.raises(KeyError):
                call()
        kept = admin.get(memory_id)
        assert kept is not None and kept["space_id"] == "bob"
    assert admin.media(image.id) == photo.getvalue()
    assert world.turns() == before


def test_v1_agent_backend_key(world: World) -> None:
    """shop-backend：每个用户在 shop 下的记忆与 shop 自己的记忆；用户本人的、名叫 shop 的用户的、
    别的 Agent 的都不行。"""
    shop = MemoryClient("http://testserver", api_key=SHOP_TOKEN, http_client=world.http)
    assert shop.add("hi", user_id="carol", agent_id="shop", infer=False)["status"] == "logged"
    assert shop.add("note", agent_id="shop", infer=False)["status"] == "logged"
    assert shop.get_all(user_id="bob", agent_id="shop")["results"]
    for scope in (
        {"user_id": "carol"},
        {"user_id": "shop"},
        {"user_id": "carol", "agent_id": "bot"},
        {"agent_id": "bot"},
        {"user_id": "bot", "agent_id": "agent"},  # = agent/bot：Agent bot 自己的记忆
    ):
        _forbidden(functools.partial(shop.get_all, **scope))
    # agent_id="agent" 的空间算作 Agent 的：space_for("shop", "agent") 就是 shop 自己的 agent/shop
    assert shop.get_all(user_id="shop", agent_id="agent")["results"]


def test_errors_keep_the_v1_shape(world: World) -> None:
    response = world.http.get(
        "/api/v1/memories", params={"user_id": "bob"}, headers=bearer(ALICE_TOKEN)
    )
    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "forbidden",
            "type": None,
            "message": "This API key cannot access the memory space 'bob'.",
        }
    }
    assert world.http.get("/api/v1/spaces").status_code == 401


# ---------- 记忆代理 ----------


def _chat(http: TestClient, token: str, *, path: str = "/v1/chat/completions", **extra: Any) -> Any:
    headers = {**bearer(token), **extra.pop("headers", {})}
    body = {"model": "qwen3.8-flash", "messages": [{"role": "user", "content": "What's my PIN?"}]}
    if path == "/v1/responses":
        body = {"model": "qwen3.8-flash", "input": "What's my PIN?"}
    return http.post(path, json={**body, **extra}, headers=headers)


def test_proxy_identity_stays_in_scope(world: World) -> None:
    """请求体的 user、X-CueMem-User、X-CueMem-Agent：作用域之外的 403（OpenAI 格式），在检索、
    开会话、转发之前；流式与 /v1/responses 一样。没有用户的请求原样转发，不碰记忆。"""
    http, before = world.http, world.turns()
    refused = [
        _chat(http, ALICE_TOKEN, user="bob"),
        _chat(http, ALICE_TOKEN, user="alice", headers={"X-CueMem-User": "bob"}),
        _chat(http, ALICE_TOKEN, user="alice", headers={"X-CueMem-Agent": "agent"}),
        _chat(http, ALICE_TOKEN, user="alice/../bob"),
        _chat(http, ALICE_TOKEN, user=["alice"]),
        _chat(http, ALICE_TOKEN, user="bob", stream=True),
        _chat(http, ALICE_TOKEN, path="/v1/responses", user="bob"),
        _chat(http, SHOP_TOKEN, user="bob"),  # 用户本人的记忆
        _chat(http, SHOP_TOKEN, user="bob", headers={"X-CueMem-Agent": "bot"}),
    ]
    for response in refused:
        assert response.status_code == 403, response.text
        error = response.json()["error"]
        assert error["code"] == "forbidden" and error["param"] == "user"
    assert world.turns() == before and world.forwarded() == 0  # 没转发
    assert world.proxy.open_runs() == 0  # 没开会话

    ok = _chat(http, ALICE_TOKEN, user="alice")
    assert ok.status_code == 200 and ok.headers["x-cuemem-session"]
    shop = _chat(http, SHOP_TOKEN, user="bob", headers={"X-CueMem-Agent": "shop"})
    assert shop.status_code == 200
    assert world.turns()["alice"] == before["alice"] + 2
    assert world.turns()["shop/bob"] == before["shop/bob"] + 2
    passthrough = _chat(http, ALICE_TOKEN)  # 没有用户：不碰记忆
    assert passthrough.status_code == 200 and passthrough.headers["x-cuemem-level"] == "none"
    assert _chat(http, ADMIN_TOKEN, user="bob").status_code == 200
    assert http.get("/v1/models", headers=bearer(ALICE_TOKEN)).status_code == 200
    embedded = http.post("/v1/embeddings", json={"input": "hi"}, headers=bearer(SHOP_TOKEN))
    assert embedded.status_code == 200
    assert _chat(http, "wrong-token", user="alice").status_code == 401


def test_proxy_default_user_is_checked_too(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """cuemem proxy --user demo：没给用户的请求用 demo 的记忆，作用域里没有 demo 的令牌 403。"""
    fake_api.chat_replies = ["Hi!"]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    proxy = MemoryProxy(system, default_user="demo", http_client=fake_api.client())
    app = create_proxy_app(proxy, keys=Keys.parse(KEYS, ADMIN_TOKEN))
    try:
        with TestClient(app) as http:
            assert _chat(http, ALICE_TOKEN).status_code == 403
            assert proxy.open_runs() == 0 and fake_api.chat_requests == []
            assert _chat(http, ALICE_TOKEN, user="alice").status_code == 200
            assert _chat(http, ADMIN_TOKEN).status_code == 200
            assert proxy.open_runs() == 2
    finally:
        proxy.close_all()
        system.close()


LIMITED_TOKEN = "limited-token-0123456789"
CAPPED_TOKEN = "capped-token-0123456789"
PLAIN_TOKEN = "plain-token-0123456789"
LIMITS = f"""
keys:
  - name: limited
    token: {LIMITED_TOKEN}
    scopes: [{{user: alice}}]
    models: [qwen3.8-flash]
  - name: capped
    token: {CAPPED_TOKEN}
    scopes: [{{user: alice}}]
    max_output_tokens: 1000
  - name: plain
    token: {PLAIN_TOKEN}
    scopes: [{{user: alice}}]
"""
# 贵的请求：最贵的模型、16 个回答、长输出，还打开目录里关掉的思考模式
EXPENSIVE = {"model": "qwen3.8-max", "n": 16, "max_tokens": 32768, "enable_thinking": True}


@pytest.fixture
def limited(
    n3_settings: Settings, fake_api: FakeAPI
) -> Iterator[tuple[TestClient, MemoryProxy, FakeAPI]]:
    fake_api.chat_replies = ["Hi!"]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    proxy = MemoryProxy(system, http_client=fake_api.client(), mode="l1")
    app = create_proxy_app(proxy, keys=Keys.parse(LIMITS, ADMIN_TOKEN))
    try:
        with TestClient(app) as http:
            yield http, proxy, fake_api
    finally:
        proxy.close_all()
        system.close()


def test_scoped_key_model_allowlist(limited: tuple[TestClient, MemoryProxy, FakeAPI]) -> None:
    """限定作用域的令牌可以写 models：只能用列出的模型（别名或模型 ID，按改用默认模型之后的算），
    之外的 403（OpenAI 格式，param model），在检索、开会话与转发之前；流式与 /v1/responses 一样。
    这种令牌的请求里，目录给的供应商参数（extra_body，例如关掉思考模式）盖过调用方给的。"""
    http, proxy, fake_api = limited
    for extra in ({"user": "alice"}, {}, {"user": "alice", "stream": True}):
        refused = _chat(http, LIMITED_TOKEN, **{**EXPENSIVE, **extra})
        assert refused.status_code == 403, refused.text
        error = refused.json()["error"]
        assert error["type"] == "permission_error" and error["param"] == "model"
        assert "qwen3.8-flash" in error["message"]
    responses = _chat(http, LIMITED_TOKEN, path="/v1/responses", model="qwen3.8-max", user="alice")
    assert responses.status_code == 403
    assert fake_api.chat_requests == [] and proxy.open_runs() == 0

    ok = _chat(http, LIMITED_TOKEN, user="alice", model="qwen3.8-flash", enable_thinking=True)
    assert ok.status_code == 200, ok.text
    assert fake_api.chat_requests[-1]["enable_thinking"] is False
    for model in (None, "gpt-4o"):  # 没给模型、不认识的模型：都是 agent 角色的 qwen3.8-flash
        body = {"messages": [{"role": "user", "content": "hi"}]}
        if model:
            body["model"] = model
        sent = http.post("/v1/chat/completions", json=body, headers=bearer(LIMITED_TOKEN))
        assert sent.status_code == 200, (model, sent.text)
    assert _chat(http, PLAIN_TOKEN, **EXPENSIVE).status_code == 200  # 没写 models 的照旧
    assert fake_api.chat_requests[-1]["enable_thinking"] is True
    assert _chat(http, ADMIN_TOKEN, **EXPENSIVE).status_code == 200


def test_scoped_key_output_cap(limited: tuple[TestClient, MemoryProxy, FakeAPI]) -> None:
    """max_output_tokens：转发的 max_tokens（max_completion_tokens）不超过它，没给时补上；n > 1
    的 400（param n），不转发。/v1/responses 的 max_output_tokens 同样。"""
    http, _, fake_api = limited
    assert _chat(http, CAPPED_TOKEN, max_tokens=32768).status_code == 200
    assert fake_api.chat_requests[-1]["max_tokens"] == 1000
    assert _chat(http, CAPPED_TOKEN, max_tokens=200).status_code == 200
    assert fake_api.chat_requests[-1]["max_tokens"] == 200
    assert _chat(http, CAPPED_TOKEN, user="alice").status_code == 200
    assert fake_api.chat_requests[-1]["max_tokens"] == 1000
    reasoning = _chat(http, CAPPED_TOKEN, max_completion_tokens=5000, max_tokens=4000)
    assert reasoning.status_code == 200
    sent = fake_api.chat_requests[-1]
    assert sent["max_completion_tokens"] == 1000 and sent["max_tokens"] == 1000
    responses = _chat(http, CAPPED_TOKEN, path="/v1/responses", max_output_tokens=9000)
    assert responses.status_code == 200 and fake_api.chat_requests[-1]["max_tokens"] == 1000
    seen = len(fake_api.chat_requests)
    for n in (2, 16):
        refused = _chat(http, CAPPED_TOKEN, n=n)
        assert refused.status_code == 400 and refused.json()["error"]["param"] == "n"
    assert _chat(http, CAPPED_TOKEN, n=1).status_code == 200
    assert len(fake_api.chat_requests) == seen + 1
    assert _chat(http, PLAIN_TOKEN, **EXPENSIVE).status_code == 200  # 没写的照旧
    assert fake_api.chat_requests[-1]["max_tokens"] == 32768
    assert fake_api.chat_requests[-1]["n"] == 16


def test_scoped_keys_without_limits_forward_the_same_bytes(
    limited: tuple[TestClient, MemoryProxy, FakeAPI],
) -> None:
    """没写这几个字段的限定作用域的令牌，转发给上游的请求体与管理令牌的逐字节相同。"""
    http, _, fake_api = limited
    body = {**EXPENSIVE, "temperature": 0.3, "tools": [{"type": "function", "function": {}}]}
    sent = []
    for token in (ADMIN_TOKEN, PLAIN_TOKEN):
        assert _chat(http, token, **body).status_code == 200
        sent.append(json.dumps(fake_api.chat_requests[-1], sort_keys=True))
    assert sent[0] == sent[1]


def test_key_limits_are_validated() -> None:
    """rate_per_minute、max_output_tokens 是正整数；models 不能是空列表，也不能有空的名字。"""
    for entry, problem in (
        ("rate_per_minute: 0", "rate_per_minute"),
        ("rate_per_minute: '5'", "rate_per_minute"),
        ("max_output_tokens: -1", "max_output_tokens"),
        ("models: []", "models"),
        ("models: ['']", "models"),
        ("models: qwen3.8-flash", "models"),
    ):
        with pytest.raises(AccessKeysError) as caught:
            Keys.parse(_keys(f"name: a, token: t, scopes: [{{user: a}}], {entry}"))
        assert problem in str(caught.value), entry
    keys = Keys.parse(
        _keys(
            "name: a, token: t, scopes: [{user: a}], rate_per_minute: 60, "
            "models: [qwen3.8-flash, deepseek-flash], max_output_tokens: 2000"
        )
    )
    principal = keys.resolve("t")
    assert principal is not None and principal.rate_per_minute == 60
    assert principal.models == ("qwen3.8-flash", "deepseek-flash")
    assert principal.max_output_tokens == 2000
    assert ADMIN.rate_per_minute is None and ADMIN.models is None


# ---------- 界面 ----------


def _ui_call(http: TestClient, name: str, data: list[Any]) -> str:
    """调用界面的一个事件（与浏览器里的界面走同一条路），返回 SSE 的全文。"""
    posted = http.post(f"/gradio_api/call/{name}", json={"data": data})
    assert posted.status_code == 200, posted.text
    return http.get(f"/gradio_api/call/{name}/{posted.json()['event_id']}").text


@pytest.mark.filterwarnings("ignore::pandas.errors.Pandas4Warning")  # Gradio 的队列里调用 pandas
def test_ui_stays_in_scope(n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path) -> None:
    """界面：限定作用域的令牌用自己的名字登录；读、遗忘、开会话、发消息都只能在作用域里，直接调
    /gradio_api 也一样。页面打开时默认空间换成作用域里的。"""
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"), default_space="shop-demo")
    for space in ("alice", "bob"):
        service.remember(space, f"Secret of {space}")
        service.consolidate(space)
    assert system.facts("bob")
    keys = Keys.parse(KEYS, ADMIN_TOKEN)
    app = create_app(service, upload_dir=tmp_path / "uploads", ui=True, keys=keys)
    try:
        with TestClient(app) as http:
            assert http.post("/gradio_api/call/load_memory", json={"data": []}).status_code == 401
            wrong = {"username": "admin", "password": ALICE_TOKEN}
            assert http.post("/login", data=wrong).status_code == 400
            ok = {"username": "alice-app", "password": ALICE_TOKEN}
            assert http.post("/login", data=ok).status_code == 200

            assert "Secret of alice" in _ui_call(http, "load_memory", ["alice", False, "fact"])
            for name, data in (
                ("load_memory", ["bob", False, "fact"]),
                ("load_memory", ["agent/alice", False, "fact"]),
                ("load_memory", ["", False, "fact"]),  # 默认空间 shop-demo
                ("choices", ["bob", "fact"]),
                ("forget", ["bob", "fact", None]),  # 在看要遗忘什么之前就拒绝
                ("start", ["bob", "", None]),
                ("send", [{"text": "hi", "files": []}, [], None, "bob", "", "l0", "rule", 30]),
            ):
                out = _ui_call(http, name, data)
                assert "event: error" in out and "Secret of bob" not in out, (name, data)
                assert "cannot open the memory space" in out, (name, out)
            assert system.facts("bob") and service.sessions == {}
            assert '"alice"' in _ui_call(http, "opened", [])  # 默认空间换成 alice

            admin = TestClient(app)
            login = admin.post("/login", data={"username": "me", "password": ADMIN_TOKEN})
            assert login.status_code == 200
            assert "Secret of bob" in _ui_call(admin, "load_memory", ["bob", False, "fact"])
            admin.close()
    finally:
        service.close()


@pytest.fixture
def ui_alice(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, MemorySystem]]:
    """挂上界面的应用，alice-app 已经登录；Gradio 的上传目录（缓存）在 tmp_path/gradio。"""
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path / "gradio"))
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"), default_space="alice")
    app = create_app(
        service, upload_dir=tmp_path / "uploads", ui=True, keys=Keys.parse(KEYS, ADMIN_TOKEN)
    )
    try:
        with TestClient(app) as http:
            login = {"username": "alice-app", "password": ALICE_TOKEN}
            assert http.post("/login", data=login).status_code == 200
            yield http, system
    finally:
        service.close()


def _png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _last_data(sse: str) -> Any:
    """SSE 里最后一个 data 事件的内容（事件的输出）。"""
    return json.loads([line[6:] for line in sse.splitlines() if line.startswith("data: ")][-1])


@pytest.mark.filterwarnings("ignore::pandas.errors.Pandas4Warning")
@pytest.mark.parametrize(
    ("planted", "space"), [("message", "alice"), ("history", "alice"), ("history", "bob")]
)
def test_ui_rejects_server_side_paths(
    ui_alice: tuple[TestClient, MemorySystem], tmp_path: Path, planted: str, space: str
) -> None:
    """界面是挂载的（gr.mount_gradio_app），从不 launch()，而 Gradio 只在 launch() 之后检查请求里
    的文件路径（FileData.path）。以前登录的用户（限定作用域的也行）点名服务器上的任何文件，Gradio
    就在事件函数之前把它复制进自己的缓存：消息里的图片进了自己的记忆、能导出；聊天记录这个输入
    没有文件类型的限制，.env 经 /gradio_api/file= 能下载（里面有管理令牌）。复制在事件函数之前，
    所以空间在作用域之外（bob）也一样。要经 HTTP 调用：直接调用事件函数不经过 Gradio 的预处理。"""
    http, system = ui_alice
    server = tmp_path / "server"
    server.mkdir()
    env = server / ".env"
    env.write_text(f"CUEMEM_API_TOKEN={ADMIN_TOKEN}\nDEEPSEEK_API_KEY=sk-real-secret\n")
    picture = server / "secret.png"
    picture.write_bytes(_png((200, 10, 10)))
    meta = {"_type": "gradio.FileData"}
    if planted == "message":
        message = {"text": "look", "files": [{"path": str(picture), "meta": meta}]}
        history: list[Any] = []
    else:
        file = {"type": "file", "file": {"path": str(env), "meta": meta}}
        message, history = {"text": "", "files": []}, [{"role": "user", "content": [file]}]
    out = _ui_call(http, "send", [message, history, None, space, "", "l0", "rule", 30])
    assert "event: error" in out, out
    assert "/gradio_api/file=" not in out and "sk-real-secret" not in out
    copies = [*(tmp_path / "gradio").rglob(".env"), *(tmp_path / "gradio").rglob("secret.png")]
    assert copies == []  # 没有复制进 Gradio 的缓存
    assert system.db.get_media("alice") == []


@pytest.mark.filterwarnings("ignore::pandas.errors.Pandas4Warning")
def test_ui_uploads_pass_the_path_checks(ui_alice: tuple[TestClient, MemorySystem]) -> None:
    """路径检查打开之后正常的用法照常：图片先传到 /gradio_api/upload，再带着它发消息；下一轮带着
    上一轮的聊天记录（里面是缓存里的图片）；记忆面板的图片在媒体库里。"""
    http, system = ui_alice
    uploaded = http.post(
        "/gradio_api/upload", files={"files": ("pic.png", _png((1, 2, 3)), "image/png")}
    )
    assert uploaded.status_code == 200, uploaded.text
    message = {
        "text": "see",
        "files": [{"path": uploaded.json()[0], "meta": {"_type": "gradio.FileData"}}],
    }
    first = _ui_call(http, "send", [message, [], None, "alice", "", "l0", "rule", 30])
    assert "event: error" not in first, first
    history, _, sid, *_ = _last_data(first)
    again = {"text": "again", "files": []}
    second = _ui_call(http, "send", [again, history, sid, "alice", "", "l0", "rule", 30])
    assert "event: error" not in second, second
    assert len(system.db.get_media("alice")) == 1
    panel = _ui_call(http, "load_memory", ["alice", False, "fact"])
    assert "event: error" not in panel, panel


def test_ui_ignores_a_session_of_another_space(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """页面状态里的会话属于作用域之外的空间时（会话 ID 不该能伪造，这里再查一遍）：结束、开新会话
    与发消息都当作没有这个会话，不结束它、也不往里写。直接调用界面的事件函数，像状态被换掉了。"""
    import gradio as gr

    from cuemem.api.ui import build_ui

    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"))
    keys = Keys.parse(KEYS, ADMIN_TOKEN)
    try:
        demo = build_ui(service, access=keys.by_username)
        events = {f.name: f.fn for f in demo.fns.values()}
        bob = service.start("bob").session_id  # 别人的会话
        alice = gr.Request(username="alice-app")
        assert events["end"](bob, alice) == (None, "No session to end.")
        new = events["start"]("alice", "", bob, alice)[0]
        assert new != bob and service.get(new).space_id == "alice"
        message = {"text": "My secret plan", "files": []}
        outputs = list(events["send"](message, [], bob, "alice", "", "l0", "rule", 30, alice))
        sid = outputs[-1][2]
        assert sid != bob and service.get(sid).space_id == "alice"
        assert not service.get(bob).closed and system.db.get_turns("bob") == []
        assert "My secret plan" in [t.text for t in system.db.get_turns("alice")]
    finally:
        service.close()


def test_ui_starts_a_new_session_after_an_idle_end(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """页面里的会话闲置太久、被清理线程结束了（ChatService.close_idle）：下一句话开一个新的会话，
    状态栏说明原因。以前页面一直拿着结束了的会话，之后每句话都显示 "⚠️ KeyError: '没有进行中的
    会话 …'"，直到点 New session。"""
    import gradio as gr

    from cuemem.api.ui import build_ui

    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"))
    service.idle_minutes = 30
    keys = Keys.parse(KEYS, ADMIN_TOKEN)
    try:
        demo = build_ui(service, access=keys.by_username)
        events = {f.name: f.fn for f in demo.fns.values()}
        alice = gr.Request(username="alice-app")
        old = events["start"]("alice", "", None, alice)[0]
        service.end(old)  # 关掉的标签页：清理线程把会话结束了（与 close_idle 相同）
        message = {"text": "I like jazz", "files": []}
        outputs = list(events["send"](message, [], old, "alice", "", "l0", "rule", 30, alice))
        history, _, sid, _, status = outputs[-1]
        assert sid != old and service.get(sid).space_id == "alice"
        assert history[-1]["role"] == "assistant" and "⚠️" not in str(history[-1]["content"])
        assert "ended after 30 idle minutes" in status
        assert [t.session_id for t in system.db.get_turns("alice")] == [sid, sid]
    finally:
        service.close()
