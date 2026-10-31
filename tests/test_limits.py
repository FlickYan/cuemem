"""请求体的上限（api/limits.py）：超过上限的请求在读请求体之前（分块传输时读到超过为止）就回
413，令牌对不对都一样；导入有自己的上限；界面的登录与事件接口也有。以前没有任何上限：FastAPI
与 Gradio 先把整个请求体读完、解析完，不带令牌的 9 MB JSON 就让内存涨两百多 MB。全部离线。"""

from __future__ import annotations

import importlib.util
import json
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import starlette.requests
from fakes import FakeAPI
from fastapi.testclient import TestClient
from scripted import ScriptedModel, echo_extraction

from cuemem.agent import Agent
from cuemem.api.app import create_app
from cuemem.api.auth import Keys
from cuemem.api.limits import MAX_BODY_BYTES, MAX_IMPORT_BYTES, OTHER_BODY_BYTES, body_limits
from cuemem.api.proxy import MemoryProxy, create_proxy_app
from cuemem.api.service import ChatService
from cuemem.config import Settings
from cuemem.errors import CueMemError
from cuemem.memory import MemorySystem
from cuemem.sdk import Memory

TOKEN = "admin-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
JSON = {"content-type": "application/json"}
needs_gradio = pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="界面是可选依赖 ui"
)


@dataclass
class Served:
    service: ChatService
    proxy: MemoryProxy
    tmp_path: Path

    def client(self, **kwargs: Any) -> TestClient:
        app = create_app(
            self.service,
            upload_dir=self.tmp_path / "uploads",
            proxy=self.proxy,
            **{"ui": False, "keys": Keys.from_token(TOKEN), **kwargs},
        )
        return TestClient(app)


@pytest.fixture
def served(n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path) -> Iterator[Served]:
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"), default_space="alice")
    proxy = MemoryProxy(system, http_client=fake_api.client(), mode="l1")
    try:
        yield Served(service, proxy, tmp_path)
    finally:
        proxy.close_all()
        service.close()


def _unread(monkeypatch: pytest.MonkeyPatch) -> None:
    """之后谁再读请求体就失败：上限要在那之前起作用。"""

    def unread(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an oversized body was read")

    for name in ("body", "json", "form"):
        monkeypatch.setattr(starlette.requests.Request, name, unread)


def _chunks(data: bytes, size: int = 512) -> Iterator[bytes]:
    """没有 Content-Length 的请求体（分块传输）。"""
    for start in range(0, len(data), size):
        yield data[start : start + size]


def _json(size: int) -> bytes:
    return json.dumps({"messages": "x" * size, "user_id": "alice", "query": "x"}).encode()


PATHS = ("/api/v1/memories", "/api/sessions", "/api/spaces/alice/recall", "/v1/chat/completions")


def _too_large(path: str, response: Any) -> None:
    assert response.status_code == 413, (path, response.text)
    if path.startswith("/api/v1/"):  # /api/v1 的出错格式
        error = response.json()["error"]
        assert error["code"] == "payload_too_large", error
        assert "1024 bytes" in error["message"] and "CUEMEM_MAX_BODY_BYTES" in error["message"]
    elif path.startswith("/api/"):
        assert response.json()["detail"] == (
            "请求体太大：这里的上限是 1024 字节（环境变量 CUEMEM_MAX_BODY_BYTES）"
        )
    else:
        assert "1024 bytes" in response.json()["detail"]


def test_oversized_bodies_get_413_before_they_are_read(
    served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    """max_body_bytes=1024：2 KiB 的请求体在 /api/v1、/api 与 /v1 都是 413。带 Content-Length 时
    令牌对不对都一样、一个字节都不读；分块传输时读到超过上限为止（不带令牌的先 401，不读）。"""
    http = served.client(max_body_bytes=1024)
    small = http.post("/api/v1/memories", content=_json(100), headers={**AUTH, **JSON})
    assert small.status_code == 200, small.text
    big = _json(2048)

    for path in PATHS:  # 分块传输：读到超过上限为止
        _too_large(path, http.post(path, content=_chunks(big), headers={**AUTH, **JSON}))
        anonymous = http.post(path, content=_chunks(big), headers=JSON)
        assert anonymous.status_code == 401, (path, anonymous.text)
    _unread(monkeypatch)
    for headers in (AUTH, {}, {"Authorization": "Bearer wrong"}):
        for path in PATHS:
            _too_large(path, http.post(path, content=big, headers={**headers, **JSON}))
        upload = http.post(
            "/api/sessions/x/messages",
            data={"text": "hi"},
            files={"images": ("a.png", b"\0" * 2048, "image/png")},
            headers=headers,
        )
        _too_large("/api/sessions/x/messages", upload)
    assert len(served.service.system.db.get_turns("alice")) == 1  # 只有那句小的写进去了


def test_imports_have_their_own_limit(served: Served) -> None:
    """导出的文件带着原图，导入（/api/spaces/import、/api/v1/import）用 max_import_bytes：比
    max_body_bytes 大的导入照常；超过 max_import_bytes 的 413。名叫 import 的空间不算导入接口。"""
    served.service.remember("alice", "My diary: " + "a long day. " * 200)
    loose = served.client(max_body_bytes=1024)
    dump = loose.get("/api/spaces/alice/export", headers=AUTH).json()
    assert len(json.dumps(dump)) > 2048
    copied = loose.post("/api/spaces/import", json=dump, params={"space": "copy"}, headers=AUTH)
    assert copied.status_code == 200, copied.text
    loaded = loose.post("/api/v1/import", json={"data": dump, "user_id": "carol"}, headers=AUTH)
    assert loaded.status_code == 200, loaded.text
    tight = served.client(max_body_bytes=1024, max_import_bytes=1024)
    refused = tight.post("/api/spaces/import", json=dump, params={"space": "c2"}, headers=AUTH)
    assert refused.status_code == 413 and "CUEMEM_MAX_IMPORT_BYTES" in refused.json()["detail"]
    refused = tight.post("/api/v1/import", json={"data": dump, "user_id": "c3"}, headers=AUTH)
    assert refused.status_code == 413 and refused.json()["error"]["code"] == "payload_too_large"
    named = loose.post("/api/spaces/import/recall", content=_json(2048), headers={**AUTH, **JSON})
    assert named.status_code == 413
    assert {"c2", "c3"}.isdisjoint(s["space_id"] for s in served.service.spaces())


ALICE_TOKEN, BOB_TOKEN = "alice-app-token-0123", "bob-app-token-0123"
SCOPED = Keys.parse(
    f"""
keys:
  - {{name: alice-app, token: {ALICE_TOKEN}, scopes: [{{user: alice}}]}}
  - {{name: bob-app, token: {BOB_TOKEN}, scopes: [{{user: bob}}]}}
""",
    TOKEN,
)
ALICE = {"Authorization": f"Bearer {ALICE_TOKEN}"}
BOB = {"Authorization": f"Bearer {BOB_TOKEN}"}


def test_scoped_keys_import_within_the_body_limit(served: Served) -> None:
    """导入的那一档（max_import_bytes）只给管理令牌：JSON 解析出来的对象是请求体的二十多倍，以前
    限定作用域的令牌导入 1 GiB 要约 25 GB 内存（作用域在解析之后才查）。限定作用域的令牌导入也用
    max_body_bytes，出错说明更大的导入要用管理令牌；管理令牌照旧。"""
    served.service.remember("alice", "My diary: " + "a long day. " * 200)
    http = served.client(keys=SCOPED, max_body_bytes=1024, max_import_bytes=1 << 20)
    dump = http.get("/api/spaces/alice/export", headers=AUTH).json()
    assert 2048 < len(json.dumps(dump)) < 1 << 20
    refused = http.post("/api/spaces/import", json=dump, params={"space": "alice"}, headers=ALICE)
    assert refused.status_code == 413
    assert refused.json()["detail"] == (
        "请求体太大：这里的上限是 1024 字节（环境变量 CUEMEM_MAX_BODY_BYTES）；更大的导入要用"
        "管理令牌"
    )
    refused = http.post("/api/v1/import", json={"data": dump, "user_id": "alice"}, headers=ALICE)
    error = refused.json()["error"]
    assert refused.status_code == 413 and error["code"] == "payload_too_large"
    assert "1024 bytes" in error["message"] and "admin token" in error["message"]
    small = served.client(keys=SCOPED, max_body_bytes=1 << 20, max_import_bytes=1 << 20)
    assert small.post("/api/v1/import", json={"data": dump}, headers=ALICE).status_code == 200
    copied = http.post("/api/v1/import", json={"data": dump, "user_id": "c1"}, headers=AUTH)
    assert copied.status_code == 200, copied.text


def test_a_scoped_key_has_a_budget_for_bodies_in_flight(
    served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一个限定作用域的令牌同时在读、在处理的请求体加起来不超过 max_body_bytes，超过时 429（带
    Retry-After）：解析出来的对象在接口处理完之前一直占着，以前一个令牌并发几个大请求就能撑爆共用
    的服务。别的令牌、管理令牌不受影响；请求结束就还回去，分块传输边读边算。"""
    entered, release = threading.Event(), threading.Event()
    add = Memory.add

    def slow(self: Memory, *args: Any, **kwargs: Any) -> Any:  # 接口还在处理第一个请求
        entered.set()
        assert release.wait(30)
        return add(self, *args, **kwargs)

    monkeypatch.setattr(Memory, "add", slow)
    body = json.dumps({"messages": "x" * 2400, "user_id": "alice", "infer": False}).encode()
    search = json.dumps({"query": "x" * 2400, "user_id": "alice"}).encode()
    with TestClient(
        create_app(
            served.service,
            upload_dir=served.tmp_path / "uploads",
            keys=SCOPED,
            ui=False,
            max_body_bytes=4096,
        )
    ) as http:
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(
                http.post, "/api/v1/memories", content=body, headers={**ALICE, **JSON}
            )
            assert entered.wait(30)
            for content in (search, _chunks(search)):  # 声明的长度，与分块传输
                busy = http.post(
                    "/api/v1/memories/search", content=content, headers={**ALICE, **JSON}
                )
                assert busy.status_code == 429, busy.text
                assert busy.headers["retry-after"] == "1"
                assert busy.json()["error"]["code"] == "rate_limited"
                assert "4096 bytes" in busy.json()["error"]["message"]
            short = json.dumps({"query": "where?", "user_id": "alice"}).encode()
            ok = http.post("/api/v1/memories/search", content=short, headers={**ALICE, **JSON})
            assert ok.status_code == 200, ok.text
            other = json.dumps({"query": "x" * 2400, "user_id": "bob"}).encode()
            for headers in (BOB, AUTH):
                fine = http.post(
                    "/api/v1/memories/search", content=other, headers={**headers, **JSON}
                )
                assert fine.status_code == 200, fine.text
            release.set()
            assert first.result(30).status_code == 200
        again = http.post("/api/v1/memories/search", content=search, headers={**ALICE, **JSON})
        assert again.status_code == 200, again.text


def test_scoped_key_rate_per_minute(served: Served, fake_api: FakeAPI) -> None:
    """限定作用域的令牌可以写 rate_per_minute：/api、/api/v1、/v1 与 /mcp 写的请求每分钟最多这么
    多个（令牌桶，容量就是这个数），超过时 429 带 Retry-After（等到下一个名额的秒数），不转发、
    不调用模型。没写的令牌与管理令牌不限；GET 不算。以前没有任何按令牌的次数或花费上限：一个只能
    访问自己用户的令牌，也能无限地经代理花运营者的额度。第四轮复审：PUT 与 DELETE 也算（更正当场
    抽取一次，删除重新抽取碰到的会话），以前只算 POST，名额用完之后照样 200、调用模型。"""
    from starlette.middleware import Middleware

    from cuemem.api.limits import BodyLimit

    keys = Keys.parse(
        f"""
keys:
  - {{name: alice-app, token: {ALICE_TOKEN}, scopes: [{{user: alice}}], rate_per_minute: 2}}
  - {{name: bob-app, token: {BOB_TOKEN}, scopes: [{{user: bob}}]}}
""",
        TOKEN,
    )
    now = [1000.0]
    app = create_app(
        served.service,
        upload_dir=served.tmp_path / "uploads",
        proxy=served.proxy,
        ui=False,
        keys=keys,
    )
    (limit,) = (m for m in app.user_middleware if isinstance(m, Middleware) and m.cls is BodyLimit)
    limit.kwargs["clock"] = lambda: now[0]  # 中间件在第一个请求时才建
    body = {"model": "qwen3.8-flash", "messages": [{"role": "user", "content": "rate-test"}]}

    def forwarded() -> int:  # 只数代理转发的请求（后台抽取的不算）
        return sum("rate-test" in str(r["messages"]) for r in fake_api.chat_requests)

    def chat(headers: dict[str, str]) -> Any:
        return http.post("/v1/chat/completions", json=body, headers=headers)

    with TestClient(app) as http:
        turns = [{"role": "user", "content": "I live in Lisbon."}]
        added = {"messages": turns, "user_id": "alice", "wait": True}
        assert http.post("/api/v1/memories", json=added, headers=AUTH).is_success
        listed = http.get("/api/v1/memories", params={"user_id": "alice"}, headers=AUTH)
        fact = listed.json()["results"][0]["id"]
        assert [chat(ALICE).status_code for _ in range(2)] == [200, 200]
        refused = chat(ALICE)
        assert refused.status_code == 429 and refused.headers["retry-after"] == "30"
        assert "2 requests per minute" in refused.json()["detail"]
        assert forwarded() == 2
        search = {"query": "x", "user_id": "alice"}
        v1 = http.post("/api/v1/memories/search", json=search, headers=ALICE)
        assert v1.status_code == 429 and v1.json()["error"]["code"] == "rate_limited"
        assert v1.headers["retry-after"] == "30"
        api = http.post("/api/sessions", json={"space": "alice"}, headers=ALICE)
        assert api.status_code == 429 and "每分钟最多 2 个" in api.json()["detail"]
        assert http.post("/mcp", json={}, headers=ALICE).status_code == 429
        calls = len(fake_api.chat_requests)
        correction = {"text": "I live in Porto."}
        for refused in (
            http.put(f"/api/v1/memories/{fact}", json=correction, headers=ALICE),
            http.delete(f"/api/v1/memories/{fact}", headers=ALICE),
            http.delete("/api/v1/memories", params={"user_id": "alice"}, headers=ALICE),
        ):
            assert refused.status_code == 429 and refused.json()["error"]["code"] == "rate_limited"
            assert refused.headers["retry-after"] == "30"
        assert len(fake_api.chat_requests) == calls  # 更正与删除都没有调用模型
        assert http.get("/api/v1/memories", params={"user_id": "alice"}, headers=ALICE).is_success
        for headers in (BOB, AUTH):  # 没写的令牌、管理令牌：不限
            assert [chat(headers).status_code for _ in range(5)] == [200] * 5
        now[0] += 29
        assert chat(ALICE).status_code == 429
        assert chat(ALICE).headers["retry-after"] == "1"
        now[0] += 1
        assert chat(ALICE).status_code == 200
        assert chat(ALICE).status_code == 429
        now[0] += 3600  # 攒不下超过 rate_per_minute 个
        assert [chat(ALICE).status_code for _ in range(3)] == [200, 200, 429]
    assert forwarded() == 2 + 10 + 1 + 2


def test_the_standalone_proxy_has_the_limit_too(
    served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cuemem proxy 的应用：/v1 一样 413，分块传输也是；默认上限 64 MiB。"""
    http = TestClient(create_proxy_app(served.proxy, max_body_bytes=1024))
    chunked = http.post("/v1/chat/completions", content=_chunks(_json(2048)), headers=JSON)
    _too_large("/v1/chat/completions", chunked)
    _unread(monkeypatch)
    for path in ("/v1/chat/completions", "/v1/embeddings", "/v1/responses"):
        _too_large(path, http.post(path, content=_json(2048), headers=JSON))
    default = TestClient(create_proxy_app(served.proxy))
    huge = {**JSON, "content-length": str(MAX_BODY_BYTES + 1)}  # 只看请求头，不真的发这么多
    assert default.post("/v1/chat/completions", content=b"{}", headers=huge).status_code == 413


@needs_gradio
@pytest.mark.filterwarnings("ignore::pandas.errors.Pandas4Warning")
def test_ui_routes_have_a_limit(served: Served, monkeypatch: pytest.MonkeyPatch) -> None:
    """界面的事件与登录接口也在解析之前查（固定 8 MiB：文件经 /gradio_api/upload 另传），上传
    接口用 max_body_bytes。以前不登录的 9 MB JSON 发到 /gradio_api/queue/join 照样被解析，/login
    能缓冲几百 MB 的表单。"""
    http = served.client(ui=True, max_body_bytes=1024)
    uploaded = http.post(
        "/gradio_api/upload", files={"files": ("a.png", b"\0" * 2048, "image/png")}
    )
    assert uploaded.status_code == 413, uploaded.text
    big = b"[" + b"{}," * (OTHER_BODY_BYTES // 3) + b"{}]"
    for path in ("/gradio_api/queue/join", "/gradio_api/run/predict"):
        chunked = http.post(path, content=_chunks(big, 1 << 16), headers=JSON)
        assert chunked.status_code == 413, (path, chunked.text)
    _unread(monkeypatch)
    for path in ("/gradio_api/queue/join", "/gradio_api/run/predict"):
        assert http.post(path, content=big, headers=JSON).status_code == 413, path
    form = b"username=x&password=" + b"a" * OTHER_BODY_BYTES
    login = http.post(
        "/login", content=form, headers={"content-type": "application/x-www-form-urlencoded"}
    )
    assert login.status_code == 413


def test_limits_from_the_environment() -> None:
    assert body_limits({}) == (MAX_BODY_BYTES, MAX_IMPORT_BYTES) == (64 << 20, 1 << 30)
    env = {"CUEMEM_MAX_BODY_BYTES": "2048", "CUEMEM_MAX_IMPORT_BYTES": " 4096 "}
    assert body_limits(env) == (2048, 4096)
    assert body_limits({"CUEMEM_MAX_BODY_BYTES": ""}) == (MAX_BODY_BYTES, MAX_IMPORT_BYTES)
    for name in ("CUEMEM_MAX_BODY_BYTES", "CUEMEM_MAX_IMPORT_BYTES"):
        for bad in ("0", "-1", "64MB", "1e6", "²"):
            with pytest.raises(CueMemError, match=name):
                body_limits({name: bad})
