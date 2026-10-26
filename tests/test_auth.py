"""多租户的访问令牌（api/auth.py）：限定作用域的令牌只能访问自己的用户。/api、/api/v1、界面
（/gradio_api）与记忆代理逐个接口检查，并且试着越出作用域：路径里的 %2F 与 %252F、Agent 的空间
（agent/<x> 与 agent_id="agent"）、filters 里换用户、按 ID 找别人的记忆、接手别人的会话、导入到
别人的空间、代理的请求头。全部离线。"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI
from fastapi import FastAPI
from fastapi.testclient import TestClient
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
"""
# 每个空间各记一句"秘密"：alice 的令牌只能访问前两个，shop-backend 只能访问 shop 下面的与
# shop 自己的。agent/alice 是 Agent alice 自己的记忆（也是 space_for("alice", "agent")），
# alice/bob 是用户 bob 在名叫 alice 的 Agent 下的记忆
SPACES = ["alice", "shop/alice", "bob", "shop/bob", "agent/shop", "agent/alice", "alice/bob"]
REACH = {
    ALICE_TOKEN: {"alice", "shop/alice"},
    SHOP_TOKEN: {"shop/alice", "shop/bob", "agent/shop"},
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
    keys = Keys.parse(KEYS)
    alice, shop = (keys.resolve(t) for t in (ALICE_TOKEN, SHOP_TOKEN))
    assert alice is not None and shop is not None
    for token, principal in ((ALICE_TOKEN, alice), (SHOP_TOKEN, shop)):
        assert {s for s in SPACES if principal.allows(s)} == REACH[token]
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
        assert len(calls) == 3 and all(len(a) == len(b) == 32 for a, b in calls)
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
    assert from_file.names == ["alice-app", "shop-backend"] and from_file.admin is not None
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


@pytest.mark.parametrize("token", [ALICE_TOKEN, SHOP_TOKEN, ADMIN_TOKEN])
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
    按 ID 找别人的记忆与不存在的 ID 一样；reset 只有管理令牌能用。"""
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
    _forbidden(alice.reset)
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
    assert {s["space_id"] for s in admin.spaces()} == set(SPACES)  # reset 没有发生


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
