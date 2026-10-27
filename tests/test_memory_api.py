"""Mem0 形状的 HTTP 记忆接口（/api/v1）与远程客户端 MemoryClient。同一段代码分别跑在本地的
Memory 与经 HTTP 的 MemoryClient 上，结果相同；方法签名逐个核对，不会走样。全部离线
（TestClient 代替网络）。"""

from __future__ import annotations

import asyncio
import base64
import importlib.resources
import inspect
import io
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx2
import pytest
from fakes import NO_BALANCE, PROJECT_ROOT, RATE_LIMIT, REJECT, TIMEOUT, FakeAPI
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import BaseModel, TypeAdapter
from scripted import ScriptedModel
from test_sdk import (
    AUG,
    SEP,
    moved,
    where_i_live,
    without_balance_for_extracting,
    without_balance_for_reconciling,
)

from cuemem import AsyncMemory, AsyncMemoryClient, Memory, MemoryClient
from cuemem.agent import Agent
from cuemem.api.app import create_app
from cuemem.api.auth import Keys, guard
from cuemem.api.memories import (
    PREFIX,
    V1ErrorBody,
    install_error_format,
    memories_router,
    openapi_v1,
    v1_part,
)
from cuemem.api.service import ChatService
from cuemem.config import Settings
from cuemem.errors import (
    ConcurrentWriteError,
    ContentFilteredError,
    EmbeddingAPIError,
    MemoryAPIError,
)
from cuemem.evidence import RAISED_REASON
from cuemem.memory import MemorySystem

Mem = Memory | MemoryClient


@pytest.fixture
def model() -> ScriptedModel:
    return ScriptedModel(
        where_i_live,
        reconcile=moved,
        reader=lambda prompt: {
            "answer": "You live in Munich.",
            "citations": ["f1"],
            "evidence_sufficient": True,
        },
    )


@pytest.fixture
def system(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> Iterator[MemorySystem]:
    fake_api.chat_replies = [model]
    opened = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    yield opened
    opened.close()


def v1_app(system: MemorySystem, keys: Keys | None = None) -> FastAPI:
    """只有记忆接口的应用：不启动后台线程，与 background=False 的本地 Memory 一样确定。"""
    app = FastAPI()
    deps = guard(keys or Keys())
    app.include_router(memories_router(Memory.from_system(system), dependencies=deps))
    install_error_format(app)
    return app


# /api/v1 的接口（路由的元数据与包住的 Memory 无关）：按它们找响应在 OpenAPI 里写的模型
V1_ROUTES = [r for r in memories_router(cast(Memory, None)).routes if isinstance(r, APIRoute)]
V1_ERRORS = (400, 401, 402, 403, 404, 409, 413, 422, 429, 502, 503, 504)


def _undocumented(value: Any, where: str = "") -> list[str]:
    """校验过的返回值里没写进模型的键（extra="allow" 的模型把它们放在 model_extra 里）。"""
    if isinstance(value, BaseModel):
        found = [f"{where}.{key}" for key in value.model_extra or {}]
        for name in type(value).model_fields:
            found += _undocumented(getattr(value, name), f"{where}.{name}")
        return found
    if isinstance(value, list):
        return [key for i, v in enumerate(value) for key in _undocumented(v, f"{where}[{i}]")]
    return []


def documented(seen: set[str]) -> Callable[[httpx2.Response], None]:
    """TestClient 的响应钩子：/api/v1 的 JSON 响应按 OpenAPI 里写的模型校验（200 按接口的模型，
    出错按 V1ErrorBody），SDK 给出的每个键都要写在模型里。走过的接口记进 seen。"""

    def check(response: httpx2.Response) -> None:
        method, path = response.request.method, response.request.url.path
        route = next(
            (r for r in V1_ROUTES if method in r.methods and r.path_regex.match(path)), None
        )
        if route is None:
            return
        seen.add(f"{method} {route.path}")
        if not response.headers.get("content-type", "").startswith("application/json"):
            return  # 图片的原文件
        response.read()
        model = route.responses[200]["model"] if response.status_code == 200 else V1ErrorBody
        parsed = TypeAdapter(model).validate_python(response.json())
        assert _undocumented(parsed) == [], (method, path, response.status_code)

    return check


@pytest.fixture(params=["local", "remote"])
def mem(request: pytest.FixtureRequest, system: MemorySystem) -> Iterator[Mem]:
    if request.param == "local":
        yield Memory.from_system(system)
        return
    with (
        TestClient(v1_app(system)) as http,
        MemoryClient("http://testserver", http_client=http) as client,
    ):
        http.event_hooks = {"request": [], "response": [documented(set())]}
        yield client


def current(mem: Mem, user_id: str) -> list[str]:
    return sorted(f["memory"] for f in mem.get_all(user_id=user_id)["results"])


# ---------- 签名 ----------


def _methods(cls: type) -> set[str]:
    return {n for n in dir(cls) if not n.startswith("_") and callable(getattr(cls, n))}


def test_the_client_has_the_same_methods_and_signatures_as_memory() -> None:
    """MemoryClient 的方法与参数（名字、种类、缺省值、类型注解）与 Memory 逐个相同：Memory 加了
    参数或方法而客户端没跟上时，这个测试失败。构造方式（from_config、from_system）只在本地有。"""
    # close_idle_runs 是服务端结束闲置的 run 用的（cuemem serve 的清理线程），远程调用没有意义
    local_only = {"from_config", "from_system", "close_idle_runs"}
    assert _methods(MemoryClient) == _methods(Memory) - local_only
    assert "status" in _methods(MemoryClient)  # 抽取的状态（第三轮体检）：本地与远程都有
    for name in [*sorted(_methods(Memory) - local_only), "__enter__", "__exit__"]:
        ours = inspect.signature(getattr(MemoryClient, name))
        theirs = inspect.signature(getattr(Memory, name))
        assert list(ours.parameters.values()) == list(theirs.parameters.values()), name
        if name != "__enter__":  # 返回自己
            assert ours.return_annotation == theirs.return_annotation, name
    ours = inspect.signature(AsyncMemoryClient.run)
    assert ours == inspect.signature(AsyncMemory.run)
    for name in ("__getattr__", "__aenter__", "__aexit__"):
        assert hasattr(AsyncMemoryClient, name)


WHEN = datetime(2026, 1, 2, 3, 4, 5)
# 每个参数一个不是缺省值的值（按参数名）；同名、意思不同的按（方法，参数）另给。Memory 加了参数
# 而这里没有时 KeyError：表要跟着改，新参数也要走一遍
ARGUMENTS: dict[str, Any] = {
    "messages": "hello",
    "user_id": "alice",
    "agent_id": "shop",
    "run_id": "r1",
    "metadata": {"k": "v"},
    "expiration_date": "2027-01-01",
    "timestamp": WHEN.isoformat(),
    "infer": False,
    "wait": True,
    "query": "q",
    "limit": 3,
    "mode": "l2",
    "at": WHEN,
    "router": "rule",
    "lam": 0.25,
    "filters": {"topic": "x"},
    "top_k": 4,
    "min_level": "l0",
    "show_expired": True,
    "question": "q?",
    "include_superseded": True,
    "valid_at": "2026-01-01",
    "known_at": WHEN,
    "memory_id": "m1",
    "media_id": "0123456789abcdef",
    "text": "new text",
    "data": "new text",  # update 的 data 与 text 是同一个参数
    "dry_run": True,
    "rebuild": True,
    "derived": False,
}
PER_METHOD: dict[tuple[str, str], Any] = {("load", "data"): {"version": 2, "space_id": "alice"}}
RETURNS: dict[str, Any] = {
    "history": [],
    "profile": "",
    "open_run": "r1",
    "spaces": [],
    "media": b"image",
    "wait": None,
}


class _Recorder:
    """站在 Memory 的位置，记下 /api/v1 交给它的参数（位置参数按 Memory 的签名换成名字）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        db = SimpleNamespace(memory_kind=lambda memory_id: None)  # 按 ID 找的记忆：没有主人
        self.system = SimpleNamespace(db=db)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        signature = inspect.signature(getattr(Memory, name))

        def call(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(None, *args, **kwargs).arguments
            self.calls.append((name, {k: v for k, v in bound.items() if k != "self"}))
            return RETURNS.get(name, {})

        return call


def _meaning(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """比较用：日期、时间换成 ISO 字符串（经 JSON 传过去）；update 的 text 与 data 合成一个
    （MemoryClient 合成后以 data 发送）。"""
    out = {k: v.isoformat() if isinstance(v, date) else v for k, v in kwargs.items()}
    if name == "update":
        data = out.pop("data", None)
        out["text"] = out["text"] if out.get("text") is not None else data
    return out


def _v1_recorder(recorder: _Recorder) -> FastAPI:
    app = FastAPI()
    app.include_router(memories_router(cast(Memory, recorder), dependencies=guard(Keys())))
    install_error_format(app)
    return app


@pytest.mark.parametrize("name", sorted(_methods(MemoryClient) - {"close", "run"}))
def test_every_argument_reaches_memory(name: str) -> None:
    """MemoryClient 的每个方法、每个参数（都给不是缺省值的值）经 /api/v1 原样交到服务端的
    Memory。以前本地与远程的测试只用缺省参数：客户端或服务端丢掉 search 的 run_id（远程就悄悄
    搜了整个用户的空间）也没有测试失败；lam、top_k、limit、export 的 derived、answer 的 mode 与
    router、consolidate 的 rebuild 同样。"""
    recorder = _Recorder()
    signature = inspect.signature(getattr(Memory, name))
    args = {p: PER_METHOD.get((name, p), ARGUMENTS[p]) for p in list(signature.parameters)[1:]}
    with (
        TestClient(_v1_recorder(recorder)) as http,
        MemoryClient("http://testserver", http_client=http) as client,
    ):
        getattr(client, name)(**args)
    called, got = recorder.calls[0]  # open_run 之后，关掉客户端时还会 end_run
    assert called == name
    assert _meaning(name, got) == _meaning(name, args)


@pytest.mark.parametrize(
    ("async_cls", "sync_cls"), [(AsyncMemory, Memory), (AsyncMemoryClient, MemoryClient)]
)
def test_the_async_classes_declare_every_method(async_cls: type, sync_cls: type) -> None:
    """AsyncMemory 与 AsyncMemoryClient 的每个方法都写明了：协程函数，参数与类型注解与同步的
    逐字相同，构造参数也相同。以前它们经 __getattr__ 转发，类上没有这些方法，类型检查器看到的
    都是 Any（await m.serach(...)、AsyncMemoryClient(api_kye=...) 都查不出来）。"""
    for name in sorted(_methods(Memory) - {"from_config", "from_system", "run"}):
        method = getattr(async_cls, name)  # 在类上找：以前只有实例的 __getattr__ 转发
        assert inspect.iscoroutinefunction(method), name
        ours, theirs = inspect.signature(method), inspect.signature(getattr(sync_cls, name))
        assert list(ours.parameters.values()) == list(theirs.parameters.values()), name
        assert ours.return_annotation == theirs.return_annotation, name
    ours, theirs = inspect.signature(async_cls.__init__), inspect.signature(sync_cls.__init__)
    assert list(ours.parameters.values()) == list(theirs.parameters.values())
    if async_cls is AsyncMemory:
        assert {"from_config", "from_system"} <= set(vars(AsyncMemory))


def test_async_memory_passes_other_attributes_through(system: MemorySystem) -> None:
    """不是方法的属性（.system、.settings）照旧取包着的 Memory 的；写错的名字 AttributeError。"""
    memory = AsyncMemory.from_system(system)
    assert memory.system is system and memory.settings is system.settings
    with pytest.raises(AttributeError):
        memory.serach  # type: ignore[attr-defined]  # noqa: B018
    assert asyncio.run(memory.spaces()) == []


def test_the_package_ships_its_type_hints() -> None:
    """py.typed（PEP 561）：没有它时用户的 mypy 跳过整个包（"missing library stubs or py.typed
    marker"），cuemem 的类型都成了 Any。CI 的 package 任务另查 wheel 里有它。"""
    assert importlib.resources.files("cuemem").joinpath("py.typed").is_file()


def test_the_client_does_not_load_the_memory_stack() -> None:
    """远程客户端只依赖 httpx2：不加载向量库、模型客户端与 SDK 本身。"""
    code = (
        "import json, sys, cuemem.client; "
        "print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('cuemem', 'qdrant_client', 'numpy', 'openai', 'fastapi'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == ["cuemem", "cuemem._async", "cuemem.client", "cuemem.errors"]


# ---------- 本地与远程，同样的结果 ----------


def test_add_events_history_and_time_travel(mem: Mem) -> None:
    first = mem.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    assert first["status"] == "consolidated"
    assert [(e["event"], e["memory"]) for e in first["results"]] == [
        ("ADD", "The user lives in Berlin.")
    ]
    berlin = first["results"][0]["id"]
    second = mem.add(
        [
            {"role": "user", "content": "I moved to Munich"},
            {"role": "assistant", "content": "Nice!"},
        ],
        user_id="alice",
        timestamp=SEP.isoformat(),
        wait=True,
    )
    (event,) = second["results"]
    assert (event["event"], event["reason"]) == ("UPDATE", "changed")
    assert event["previous_memory"] == ["The user lives in Berlin."]
    munich = event["id"]
    assert current(mem, "alice") == ["The user lives in Munich."]
    everything = mem.get_all(user_id="alice", include_superseded=True)["results"]
    assert {f["id"] for f in everything} == {berlin, munich}
    then = mem.get_all(user_id="alice", valid_at="2026-08-15")["results"]
    assert [f["memory"] for f in then] == ["The user lives in Berlin."]
    believed = mem.get_all(user_id="alice", known_at=AUG + timedelta(days=1))["results"]
    assert [f["memory"] for f in believed] == ["The user lives in Berlin."]
    old = mem.get(berlin)
    assert old is not None and old["superseded_by"] == munich and old["space_id"] == "alice"
    assert [(v["event"], v["id"]) for v in mem.history(munich)] == [
        ("ADD", berlin),
        ("UPDATE", munich),
    ]
    assert mem.get("no-such-id") is None and mem.history("no-such-id") == []
    assert "Munich" in mem.profile(user_id="alice")


def test_metadata_filters_and_expiration(mem: Mem) -> None:
    tea = mem.add("I like green tea", user_id="dan", metadata={"topic": "food", "stars": 5})
    assert tea["status"] == "queued"
    mem.add("I like chess", user_id="dan", metadata={"topic": "games", "stars": 3})
    mem.add("I like rain", user_id="dan", expiration_date=date.today() - timedelta(days=1))
    mem.add("I like snow", user_id="dan", expiration_date=date.today() + timedelta(days=9))
    mem.consolidate(user_id="dan")

    def facts(**where: Any) -> list[str]:
        found = mem.get_all(filters={"user_id": "dan", **where})["results"]
        return sorted(f["memory"] for f in found)

    every = {f["memory"]: f for f in mem.get_all(user_id="dan")["results"]}
    assert sorted(every) == ["I like chess", "I like green tea", "I like snow"]  # 雨已过期
    assert every["I like green tea"]["metadata"] == {"topic": "food", "stars": 5}
    assert every["I like snow"]["expiration_date"] == (date.today() + timedelta(9)).isoformat()
    assert facts(topic="food") == ["I like green tea"]
    assert facts(stars={"gte": 3, "lte": 4}) == ["I like chess"]
    assert facts(OR=[{"topic": "food"}, {"NOT": [{"topic": "*"}]}]) == [
        "I like green tea",
        "I like snow",
    ]
    found = mem.search("green tea", filters={"user_id": "dan", "topic": "food"}, mode="l1")
    assert found["results"] and {r["metadata"]["topic"] for r in found["results"]} == {"food"}
    with pytest.raises(ValueError, match="filter"):
        mem.get_all(filters={"user_id": "dan", "topic": {"regex": "f.*"}})
    with pytest.raises(ValueError, match="filter"):
        mem.search("tea", filters={"user_id": "dan", "OR": {"topic": "food"}})
    with pytest.raises(TypeError, match="metadata"):
        mem.add("x", user_id="dan", metadata=["not", "a", "dict"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="keys must be strings"):
        mem.add("x", user_id="dan", metadata={1: "x"})  # JSON 会把 1 悄悄变成 "1"
    with pytest.raises(ValueError, match="expiration date"):
        mem.add("x", user_id="dan", expiration_date="next week")
    with pytest.raises(ValueError, match="user_id"):
        mem.get_all()
    assert len(mem.get_all(user_id="dan")["results"]) == 3  # 报错时什么都没写

    # show_expired（Mem0 的写法）：过期的也列出；画像照旧不给。以前只有 export 与已知的 ID
    shown = mem.get_all(user_id="dan", show_expired=True)["results"]
    assert sorted(f["memory"] for f in shown) == [
        "I like chess",
        "I like green tea",
        "I like rain",
        "I like snow",
    ]
    rain = mem.search("Do I like rain?", user_id="dan", mode="l1", show_expired=True)
    assert any("rain" in r["memory"] for r in rain["results"])
    assert "rain" not in rain["profile"]
    hidden = mem.search("Do I like rain?", user_id="dan", mode="l1")["results"]
    assert not any("rain" in r["memory"] for r in hidden)


def test_mem0_call_shapes(mem: Mem) -> None:
    """Mem0 2.x 的写法：update(memory_id=, text=)（data 是它旧的名字）、只给 metadata 的 update
    报 ValueError（以前是 TypeError）；limit、top_k 要是正整数：以前 limit=0 给出全部结果，
    limit=-1 悄悄少给一条。本地与远程相同。"""
    berlin = mem.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)["results"][0]
    mem.add("I like chess", user_id="alice", wait=True)
    moved = mem.update(memory_id=berlin["id"], text="I moved to Munich")
    assert moved["message"] == "Memory updated"
    paris = mem.update(moved["results"][0]["id"], data="I live in Paris")
    assert paris["message"] == "Memory updated"
    current = paris["results"][0]["id"]
    for call in (
        lambda: mem.update(current, metadata={"k": 1}),
        lambda: mem.update(current, text="I live in Rome", data="I live in Oslo"),
    ):
        with pytest.raises(ValueError, match="text"):
            call()
    every = mem.get_all(user_id="alice")["results"]
    assert len(every) == 2 and len(mem.get_all(user_id="alice", limit=1)["results"]) == 1
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            mem.get_all(user_id="alice", limit=bad)
        with pytest.raises(ValueError, match="positive integer"):
            mem.search("Where do I live?", user_id="alice", limit=bad)
        with pytest.raises(ValueError, match="positive integer"):
            mem.search("Where do I live?", user_id="alice", top_k=bad)
    found = mem.search("Where do I live?", user_id="alice", mode="l1")["results"]
    assert len(found) > 1
    two = mem.search("Where do I live?", user_id="alice", mode="l1", top_k=len(found) - 1)
    assert len(two["results"]) == len(found) - 1


def test_client_host_and_base_url(system: MemorySystem) -> None:
    """MemoryClient(host=…) 是 Mem0 的写法，与 base_url 相同；没有 http(s):// 的地址（Mem0 的
    MemoryClient("m0-…") 把 API Key 放在第一个参数）在构造时报错，报错里不带这个值。以前要等
    第一次调用才报"Cannot reach"。"""
    with TestClient(v1_app(system)) as http:
        assert MemoryClient(host="http://testserver", http_client=http).spaces() == []
        with pytest.raises(ValueError, match="not both"):
            MemoryClient("http://testserver", host="http://testserver", http_client=http)
    for wrong in ("m0-secret-key", "localhost:7860"):
        with pytest.raises(ValueError, match="http://") as raised:
            MemoryClient(wrong)
        assert wrong not in str(raised.value)
    with pytest.raises(ValueError, match="http://"):
        AsyncMemoryClient("m0-secret-key")


def test_update_delete_reset_and_their_errors(mem: Mem) -> None:
    berlin = mem.add(
        "I live in Berlin", user_id="alice", timestamp=AUG, metadata={"src": "crm"}, wait=True
    )["results"][0]
    out = mem.update(berlin["id"], "I live in Paris", metadata={"checked": True})
    assert out["message"] == "Memory updated"
    (event,) = out["results"]
    assert (event["event"], event["reason"]) == ("UPDATE", "corrected")
    assert event["metadata"] == {"src": "crm", "checked": True}
    assert [v["memory"] for v in mem.history(berlin["id"])] == [
        "The user lives in Berlin.",
        "The user lives in Paris.",
    ]
    with pytest.raises(ValueError, match="already replaced"):
        mem.update(berlin["id"], "I live in Rome")
    with pytest.raises(ValueError, match="empty"):
        mem.update(event["id"], "  ")
    with pytest.raises(KeyError):
        mem.update("no-such-id", "I live in Rome")
    with pytest.raises(KeyError):
        mem.delete("no-such-id")

    preview = mem.delete(event["id"], dry_run=True)
    assert preview["message"].startswith("Dry run") and current(mem, "alice") == [
        "The user lives in Paris."
    ]
    assert mem.delete(event["id"])["turns_deleted"] == 1  # 遗忘这句更正：原来那条恢复
    assert current(mem, "alice") == ["The user lives in Berlin."]

    mem.add("I live in Rome", agent_id="bot", wait=True)
    out = mem.reset()
    assert sorted(out["spaces_deleted"]) == ["agent/bot", "alice"]
    assert mem.spaces() == []


def test_extraction_errors_after_the_write_are_not_errors(
    mem: Mem, system: MemorySystem, model: ScriptedModel, fake_api: FakeAPI
) -> None:
    """add(wait=True) 写进发言之后对账时余额不足：本地与远程一样不报错（/api/v1 是 200），status
    "queued"、error 写明，下次巩固时补上。以前 /api/v1 回 402，MemoryClient 抛出异常，调用方重试
    就把同一句话写了两遍。"""
    mem.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    broke = without_balance_for_reconciling(model, fake_api)
    out = mem.add("I moved to Munich", user_id="alice", timestamp=SEP, wait=True)
    assert out["status"] == "queued" and "Error code: 402" in out["error"]
    broke["on"] = False
    mem.consolidate(user_id="alice")
    assert current(mem, "alice") == ["The user lives in Munich."]
    texts = [t.text for t in system.db.get_turns("alice")]
    assert texts == ["I live in Berlin", "I moved to Munich"]


def test_status_reports_failed_extractions(
    mem: Mem, system: MemorySystem, model: ScriptedModel, fake_api: FakeAPI
) -> None:
    """status()（/api/v1/status）：本地与远程一样给出失败的会话、失败了几次、原因与后台线程什么时候
    重试，只看这个用户的。以前 SDK、MemoryClient 与 /api/v1 都没有办法知道后台抽取失败了。"""
    system.start_worker()
    without_balance_for_extracting(model, fake_api)
    sid = mem.add("I live in Berlin", user_id="alice")["session_id"]
    mem.wait()
    state = mem.status(user_id="alice")
    assert state["pending"] == 0 and state["background"] is True
    (failed,) = state["failed"]
    assert failed["session_id"] == sid and failed["attempts"] == 1
    assert "Error code: 402" in failed["error"] and "\n" not in failed["error"]
    assert datetime.fromisoformat(failed["retry_at"]) > datetime.now()
    assert len(state["errors"]) == 1 and "402" in state["errors"][0]
    assert mem.status(user_id="bob") == {
        "pending": 0,
        "failed": [],
        "background": True,
        "errors": [],
    }


def test_search_levels_and_answer(mem: Mem) -> None:
    mem.add("I live in Berlin", user_id="u", timestamp=AUG, wait=True)
    mem.add("I like chess", user_id="u", wait=True)
    found = mem.search("Where do I live?", user_id="u", mode="l1", at=SEP)
    assert found["level"] == "L1" and {r["type"] for r in found["results"]} == {"fact", "excerpt"}
    assert mem.search("Where do I live?", user_id="u", mode="l1", top_k=1)["results"][0]["score"]
    gate = mem.search("Hi there", user_id="u", min_level="l0")
    assert gate["level"] == "L0" and gate["results"] == []
    raised = mem.search("Hi there", user_id="u")
    assert raised["level"] == "L1" and raised["reasons"] == [RAISED_REASON]
    answered = mem.answer("Where do I live?", user_id="u", mode="l1", router="rule")
    assert answered["answer"] == "You live in Munich." and answered["citations"]


def test_runs_scopes_export_and_load(mem: Mem) -> None:
    with mem.run(user_id="carol", wait=True) as run_id:
        mem.add("I live in Oslo", user_id="carol", run_id=run_id)
        mem.add("I like skiing", user_id="carol", run_id=run_id)
        assert current(mem, "carol") == []  # run 进行中不抽取
    assert current(mem, "carol") == ["I like skiing", "The user lives in Oslo."]
    other = mem.open_run("r2", user_id="carol")
    mem.add("I like tea", user_id="carol", run_id=other)
    assert mem.end_run(other, user_id="carol", wait=True)["turns"] == 1
    assert [f["memory"] for f in mem.get_all(user_id="carol", run_id="r2")["results"]] == [
        "I like tea"
    ]
    assert mem.delete_all(user_id="carol", run_id="r2")["sessions_affected"] == ["r2"]
    assert "I like tea" not in current(mem, "carol")

    mem.add("I live in Paris", user_id="carol", agent_id="travel/bot", wait=True)
    assert {s["space_id"] for s in mem.spaces()} == {"carol", "travel%2Fbot/carol"}
    scoped = mem.get_all(user_id="carol", agent_id="travel/bot")["results"]
    assert [f["object"] for f in scoped] == ["Paris"]

    dump = mem.export(user_id="carol")
    assert dump["space_id"] == "carol" and dump["derived"]["facts"]
    report = mem.load(dump, user_id="dave", wait=True)
    assert report["space_id"] == "dave"
    assert current(mem, "dave") == current(mem, "carol")
    mem.wait()
    mem.delete_all(user_id="carol", agent_id="travel/bot")
    assert {s["space_id"] for s in mem.spaces()} == {"carol", "dave"}


def test_export_and_load_errors_are_the_same_remotely(system: MemorySystem) -> None:
    """导出不存在的空间（KeyError）、导入没有 space_id 的文件又没给 user_id（ValueError，以前本地
    是 KeyError('space_id')）：MemoryClient 抛出与本地相同的异常。以前远程的这两个是没有说明的
    500（MemoryAPIError）。"""
    nameless = {"format": "cuemem.space", "version": 2, "sessions": []}
    with (
        TestClient(v1_app(system), raise_server_exceptions=False) as http,
        MemoryClient("http://testserver", http_client=http) as client,
    ):
        for mem in (Memory.from_system(system), client):
            with pytest.raises(KeyError, match="No memory space"):
                mem.export(user_id="nobody")
            with pytest.raises(ValueError, match="no space_id"):
                mem.load(nameless)


def _export(sessions: Any) -> dict[str, Any]:
    return {"format": "cuemem.space", "version": 2, "space_id": "x", "sessions": sessions}


MALFORMED = {
    "no turns": _export([{"session_id": "s1", "started_at": "2026-08-01T10:00:00"}]),
    "started_at is a number": _export([{"session_id": "s1", "started_at": 5, "turns": []}]),
    "sessions is a string": _export("abc"),
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_exports_are_value_errors(mem: Mem, system: MemorySystem, name: str) -> None:
    """导入的文件缺键、类型不对：本地与远程都是 ValueError("Malformed export…")（/api/v1 的
    400），什么都不写。以前本地是 KeyError 或 TypeError，/api/v1 是没有说明的 500 或 400。"""
    with pytest.raises(ValueError, match="Malformed export"):
        mem.load(MALFORMED[name])
    assert system.db.list_spaces() == []


def test_concurrent_write_conflicts_are_409(
    system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """序号撞上已有的发言（ConcurrentWriteError）：/api/v1 回 409 conflict，MemoryClient 抛出同样的
    异常，什么都没写。以前没有测试守着这个状态码：改成 500 也没有测试失败。"""
    with TestClient(v1_app(system), raise_server_exceptions=False) as http:
        client = MemoryClient("http://testserver", http_client=http)
        client.add("I live in Berlin", user_id="alice", run_id="chat", infer=False)
        monkeypatch.setattr(Memory, "_next_idx", lambda self, space, session_id: 0)
        body = {"messages": "I like jazz", "user_id": "alice", "run_id": "chat", "infer": False}
        response = http.post("/api/v1/memories", json=body)
        assert response.status_code == 409
        error = response.json()["error"]
        assert (error["code"], error["type"]) == ("conflict", "ConcurrentWriteError")
        with pytest.raises(ConcurrentWriteError, match="chat"):
            client.add("I like jazz", user_id="alice", run_id="chat", infer=False)
    assert [t.text for t in system.db.get_turns("alice", "chat")] == ["I live in Berlin"]


def test_the_client_explains_a_wrong_base_url(system: MemorySystem) -> None:
    """base_url 照文档里 curl 的写法以 /api/v1 结尾时照样能用（以前每个调用都是 404 "Not Found"）；
    指向别的服务（记忆代理的端口、别的网站）时 404 的报错写明 base_url 要是 cuemem serve 的根
    地址；连不上时问一句服务开着没有。"""
    with TestClient(v1_app(system)) as http:
        for url in ("http://testserver/api/v1", "http://testserver/api/v1/"):
            assert MemoryClient(url, http_client=http).spaces() == []
    with TestClient(FastAPI()) as other:
        elsewhere = MemoryClient("http://testserver", http_client=other)
        for call in (lambda: elsewhere.spaces(), lambda: elsewhere.get("abc")):
            with pytest.raises(MemoryAPIError, match="root of `cuemem serve`") as raised:
                call()
            assert raised.value.status == 404 and raised.value.type is None

    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("Connection refused", request=request)

    with httpx2.Client(transport=httpx2.MockTransport(refuse)) as dead:
        with pytest.raises(MemoryAPIError, match="is `cuemem serve` running") as raised:
            MemoryClient("http://127.0.0.1:9", http_client=dead).spaces()
        assert raised.value.status is None


def test_images_reach_the_memory(mem: Mem, system: MemorySystem, tmp_path: Path) -> None:
    """add 的本地图片路径：本地 SDK 直接读；MemoryClient 先在客户端读成 data URL，服务器不读它
    自己磁盘上的文件。网络图片两边都报同样的错。"""
    photo = tmp_path / "shoe.png"
    Image.new("RGB", (16, 16), "blue").save(photo)
    content = [
        {"type": "text", "text": "My running shoes"},
        {"type": "image_url", "image_url": {"url": str(photo)}},
    ]
    out = mem.add({"role": "user", "content": content}, user_id="alice", infer=False)
    assert out["status"] == "logged"
    (turn,) = system.db.get_turns("alice")
    assert turn.text == "My running shoes <image>" and len(turn.media_ids) == 1
    assert len(system.db.get_media("alice")) == 1 and photo.exists()
    web = [{"type": "image_url", "image_url": "https://x.test/a.png"}]
    with pytest.raises(ValueError, match="download"):
        mem.add({"role": "user", "content": web}, user_id="alice")
    gone = [{"type": "image_url", "image_url": "/no/such.png"}]
    with pytest.raises(ValueError, match="not found"):
        mem.add({"role": "user", "content": gone}, user_id="alice")


def test_every_id_search_returns_works_by_id(
    mem: Mem, system: MemorySystem, tmp_path: Path
) -> None:
    """search 给出的三种 ID（事实、对话片段、图片）get 都取得到、delete 都删得掉（删那一轮对话，
    与 MCP 的 forget 相同），图片的原文件用 media 取（path 是服务器上的路径）。以前片段与图片的
    ID get 是 None、delete 是 KeyError：infer=False 写入的会话 search 只给出片段，Mem0 的
    for r in results: delete(r["id"]) 第一条就出错。本地与远程相同。"""
    mem.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    mem.add("I like chess a lot", user_id="alice", infer=False)
    photo = tmp_path / "shoe.png"
    Image.new("RGB", (16, 16), "blue").save(photo)
    content = [
        {"type": "text", "text": "My running shoes"},
        {"type": "image_url", "image_url": {"url": str(photo)}},
    ]
    mem.add({"role": "user", "content": content}, user_id="alice", infer=False)
    found = mem.search("Where do I live? chess running shoes", user_id="alice", mode="l1")
    assert {r["type"] for r in found["results"]} == {"fact", "excerpt"}
    (image,) = found["images"]
    for r in [*found["results"], image]:
        item = mem.get(r["id"])
        assert item is not None and item["type"] == r["type"] and item["space_id"] == "alice"
        assert item["memory"] == r["memory"] and "score" not in item
        if r["type"] != "fact":
            assert mem.history(r["id"]) == []  # 只有事实有版本
            with pytest.raises(ValueError, match="Only facts can be updated"):
                mem.update(r["id"], "I like go")
    assert mem.media(image["id"]) == photo.read_bytes()
    assert mem.media(found["results"][0]["id"]) is None and mem.media("no-such-id") is None

    chess = next(r for r in found["results"] if "chess" in r["memory"])
    preview = mem.delete(chess["id"], dry_run=True)
    assert [t["text"] for t in preview["turns"]] == ["I like chess a lot"]
    assert len(system.db.get_turns("alice")) == 3  # 预演什么都没删
    assert mem.delete(chess["id"])["turns_deleted"] == 1
    assert mem.delete(image["id"])["turns_deleted"] == 1
    assert mem.get(image["id"]) is None and mem.media(image["id"]) is None
    assert [t.text for t in system.db.get_turns("alice")] == ["I live in Berlin"]
    with pytest.raises(KeyError):
        mem.delete(chess["id"])  # 已经删了：与不存在的一样

    # infer=False 写入的会话：search 只给出片段，Mem0 的写法逐条删
    mem.add("I like tea", user_id="bob", infer=False)
    mem.add("I like jazz", user_id="bob", infer=False)
    listed = mem.search("tea jazz", user_id="bob", mode="l1")["results"]
    assert len(listed) == 2 and {r["type"] for r in listed} == {"excerpt"}
    for r in listed:
        mem.delete(r["id"])
    assert system.db.get_turns("bob") == []


def test_closing_the_client_ends_its_runs(system: MemorySystem) -> None:
    """与 Memory.close 相同：这个客户端开始、还没结束的 run 交给抽取，服务端不再当它进行中。"""
    with TestClient(v1_app(system)) as http:
        client = MemoryClient("http://testserver", http_client=http)
        run_id = client.open_run(user_id="eve")
        client.add("I live in Lisbon", user_id="eve", run_id=run_id)
        assert system.open_sessions("eve") == {run_id}
        client.close()
        assert system.open_sessions("eve") == set()
        assert [j["session_id"] for j in system.db.get_jobs("eve")] == [run_id]


@pytest.mark.parametrize("where", ["local", "remote"])
def test_idle_remote_run_is_ended_and_reopened(system: MemorySystem, where: str) -> None:
    """open_run 之后闲置的 run 由服务端的 close_idle_runs 交给抽取（cuemem serve 每分钟调用）：以前
    调用方崩溃、没有 end_run 时，这段对话要等服务重启。调用方之后接着往这个 run 里 add 时重新打开
    （不登记新的任务），end_run 时整段只登记一次；end_run 之后再写就是一个普通的会话。"""
    server = Memory.from_system(system)
    app = FastAPI()
    app.include_router(memories_router(server, dependencies=guard(Keys())))
    install_error_format(app)

    def pending() -> list[str]:
        return [j["session_id"] for j in system.db.get_jobs("eve", status="pending")]

    with TestClient(app) as http:
        mem: Mem = (
            server if where == "local" else MemoryClient("http://testserver", http_client=http)
        )
        rid = mem.open_run(user_id="eve")
        mem.add("I live in Lisbon", user_id="eve", run_id=rid)
        later = time.monotonic() + 29 * 60
        assert server.close_idle_runs(30 * 60, now=later) == 0 and pending() == []
        later = time.monotonic() + 31 * 60
        assert server.close_idle_runs(30 * 60, now=later) == 1
        assert system.open_sessions("eve") == set() and pending() == [rid]

        mem.add("I like tea", user_id="eve", run_id=rid)  # 接着写：重新打开
        assert system.open_sessions("eve") == {rid} and pending() == [rid]
        with pytest.raises(ValueError, match="still open"):
            mem.add("I like jazz", user_id="eve", run_id=rid, wait=True)
        assert mem.end_run(rid, user_id="eve")["turns"] == 2
        assert system.open_sessions("eve") == set() and pending() == [rid]
        assert len(system.db.get_jobs("eve", status="superseded")) == 1
        mem.add("I like rain", user_id="eve", run_id=rid)  # end_run 之后不再重新打开
        assert system.open_sessions("eve") == set()


def test_async_client(system: MemorySystem) -> None:
    async def main(client: AsyncMemoryClient) -> list[str]:
        async with client.run(user_id="eve", wait=True) as run_id:
            await client.add("I live in Lisbon", user_id="eve", run_id=run_id)
        found = await client.search("Where do I live?", user_id="eve", mode="l1")
        return [r["memory"] for r in found["results"] if r["type"] == "fact"]

    with TestClient(v1_app(system)) as http:
        client = AsyncMemoryClient("http://testserver", http_client=http)
        assert asyncio.run(main(client)) == ["The user lives in Lisbon."]


# ---------- HTTP ----------


def test_cuemem_serve_mounts_the_v1_api_behind_the_token(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel, tmp_path: Path
) -> None:
    """cuemem serve（create_app）挂上 /api/v1：与其他接口同一个令牌、同一个记忆系统；出错是英文的
    {"error": {"code", "type", "message"}}，别的接口的出错格式不变。curl 的写法见 doc_notes。"""
    fake_api.chat_replies = [model]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"), default_space="shop")
    app = create_app(service, upload_dir=tmp_path / "uploads", ui=False, token="t0k")
    try:
        with TestClient(app) as http:
            denied = http.post("/api/v1/memories", json={"messages": "hi", "user_id": "alice"})
            assert denied.status_code == 401
            assert denied.json()["error"]["code"] == "unauthorized"
            assert http.get("/api/spaces").json() == {"detail": "缺少或错误的访问令牌"}
            with pytest.raises(MemoryAPIError) as caught:
                MemoryClient("http://testserver", api_key="wrong", http_client=http).spaces()
            assert caught.value.status == 401 and caught.value.code == "unauthorized"

            auth = {"Authorization": "Bearer t0k"}
            body = {"messages": "I live in Berlin", "user_id": "alice", "agent_id": "bot"}
            added = http.post("/api/v1/memories", json={**body, "wait": True}, headers=auth)
            assert added.status_code == 200, added.text
            (event,) = added.json()["results"]
            assert event["event"] == "ADD" and event["memory"] == "The user lives in Berlin."
            # 同一个记忆系统：/api 的浏览接口看得到，空间里的 "/" 不用转义
            facts = http.get("/api/spaces/bot/alice/facts", headers=auth).json()
            assert [f["id"] for f in facts] == [event["id"]]
            for where, found in (('{"topic": "*"}', []), ('{"NOT": [{"topic": "*"}]}', [event])):
                listed = http.get(
                    "/api/v1/memories",
                    params={"user_id": "alice", "agent_id": "bot", "filters": where},
                    headers=auth,
                )
                assert [f["id"] for f in listed.json()["results"]] == [f["id"] for f in found]

            missing = http.get("/api/v1/memories/nope", headers=auth)
            assert missing.status_code == 404
            assert missing.json() == {
                "error": {
                    "code": "not_found",
                    "type": "KeyError",
                    "message": "No memory with id nope",
                }
            }
            bad = http.post(
                "/api/v1/memories/search", json={"query": "x", "mode": "l9"}, headers=auth
            )
            assert bad.status_code == 422 and bad.json()["error"]["type"] == "ValueError"
            nobody = http.post("/api/v1/memories", json={"messages": "hi"}, headers=auth)
            assert nobody.status_code == 400 and "user_id" in nobody.json()["error"]["message"]
            filters = http.get("/api/v1/memories?user_id=alice&filters=[1]", headers=auth)
            assert filters.status_code == 400

            client = MemoryClient("http://testserver", api_key="t0k", http_client=http)
            raw = http.get(f"/api/v1/memories/{event['id']}", headers=auth).json()
            assert client.get(event["id"]) == raw
            assert [s["space_id"] for s in client.spaces()] == ["bot/alice"]
    finally:
        service.close()


def test_the_openapi_documents_every_v1_response() -> None:
    """/openapi.json 里 /api/v1 的每个接口：200 指向写明形状的模型（图片的原文件是 image/*），每个
    出错的状态码都是 V1ErrorBody（422 也是）。以前 200 只写着"任意对象"，422 写的是 FastAPI 的
    HTTPValidationError（实际不会出现），别的出错一个都没写。"""
    spec = openapi_v1()
    operations = [(p, m, op) for p, ops in spec["paths"].items() for m, op in ops.items()]
    assert len(operations) == sum(len(r.methods) for r in V1_ROUTES) == 19
    for path, method, op in operations:
        for status in V1_ERRORS:
            schema = op["responses"][str(status)]["content"]["application/json"]["schema"]
            assert schema == {"$ref": "#/components/schemas/V1ErrorBody"}, (path, method, status)
        ok = op["responses"]["200"]["content"]
        if path == f"{PREFIX}/media/{{media_id}}":
            assert list(ok) == ["image/*"]
        else:
            assert "#/components/schemas/" in json.dumps(ok["application/json"]["schema"]), path
    assert "HTTPValidationError" not in json.dumps(spec)


def test_the_openapi_snapshot_is_current(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel, tmp_path: Path
) -> None:
    """docs/openapi-v1.json（按它生成别的语言的客户端）与接口一致：改了接口或模型之后要运行
    scripts/openapi_v1.py。cuemem serve（create_app）的 /openapi.json 里 /api/v1 的部分与它相同。"""
    snapshot = json.loads((PROJECT_ROOT / "docs" / "openapi-v1.json").read_text(encoding="utf-8"))
    assert snapshot == openapi_v1(), "过时了：运行 uv run python scripts/openapi_v1.py"
    fake_api.chat_replies = [model]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"))
    try:
        app = create_app(service, upload_dir=tmp_path / "uploads", ui=False, token="t0k")
        assert {**v1_part(app.openapi()), "info": None} == {**snapshot, "info": None}
    finally:
        service.close()


def test_v1_responses_keep_the_json_serializer(
    system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """文档里的模型不改变返回值：接口照旧按 dict 序列化（NaN 写成 null，不是 500）。用
    response_model=None 写文档的话 FastAPI 改走 json.dumps(allow_nan=False)，分数是 NaN 时 500。"""
    for route in V1_ROUTES:
        if route.path != f"{PREFIX}/media/{{media_id}}":  # 图片的原文件不是 JSON
            assert route.response_field is not None, route.path
    nan = {"id": "x", "score": float("nan"), "memory": "m"}
    monkeypatch.setattr(Memory, "get", lambda self, memory_id: nan)
    with TestClient(v1_app(system)) as http:
        response = http.get("/api/v1/memories/x")
    assert response.status_code == 200 and response.json() == {**nan, "score": None}


def test_every_v1_response_matches_its_documented_model(
    system: MemorySystem, tmp_path: Path
) -> None:
    """每个 /api/v1 接口经 MemoryClient 走一遍：返回的 JSON 照 OpenAPI 里写的模型校验，SDK 给出的
    每个键都写在模型里，出错是 V1ErrorBody。（本地与远程的测试经 mem 的远程那一半也这样校验。）"""
    seen: set[str] = set()
    photo = tmp_path / "shoe.png"
    Image.new("RGB", (16, 16), "blue").save(photo)
    image = [
        {"type": "text", "text": "My running shoes"},
        {"type": "image_url", "image_url": {"url": str(photo)}},
    ]
    with (
        TestClient(v1_app(system)) as http,
        MemoryClient("http://testserver", http_client=http) as mem,
    ):
        http.event_hooks = {"request": [], "response": [documented(seen)]}
        added = mem.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
        berlin = added["results"][0]
        mem.add("I like chess a lot", user_id="alice", metadata={"k": 1}, infer=False)
        mem.add({"role": "user", "content": image}, user_id="alice", infer=False)
        found = mem.search("Where do I live? chess running shoes", user_id="alice", mode="l1")
        assert {r["type"] for r in found["results"]} == {"fact", "excerpt"} and found["images"]
        for item in [*found["results"], *found["images"]]:
            assert mem.get(item["id"]) is not None
        assert mem.media(found["images"][0]["id"]) == photo.read_bytes()
        assert mem.get_all(user_id="alice", show_expired=True)["results"]
        events = mem.update(berlin["id"], "I live in Paris")["results"]
        paris = next(e["id"] for e in events if e["event"] == "UPDATE")
        assert len(mem.history(paris)) == 2
        assert mem.delete(paris, dry_run=True)["turns"]
        assert mem.delete(paris)["turns_deleted"] == 1
        assert mem.answer("Where do I live?", user_id="alice", mode="l1", router="rule")
        assert "Berlin" in mem.profile(user_id="alice")
        with mem.run(user_id="bob", wait=True) as run_id:
            mem.add("I like tea", user_id="bob", run_id=run_id)
        later = mem.open_run(user_id="bob")
        mem.add("I like jazz", user_id="bob", run_id=later)
        assert mem.end_run(later, user_id="bob")["consolidation"] == "queued"
        assert mem.end_run(mem.open_run(user_id="bob"), user_id="bob")["consolidation"] is None
        mem.consolidate(user_id="bob")
        mem.wait()
        dump = mem.export(user_id="alice")
        assert mem.load(dump, user_id="carol", wait=True)["consolidation"]
        assert mem.spaces() and mem.delete_all(user_id="carol")["turns_deleted"]
        for call in (lambda: mem.update("no-such-id", "x"), lambda: mem.export(user_id="nobody")):
            with pytest.raises(KeyError):
                call()
        bad = http.post("/api/v1/memories/search", json={"query": "x", "mode": "l9"})
        assert bad.status_code == 422
        assert mem.reset()["spaces_deleted"]
    assert seen == {f"{m} {r.path}" for r in V1_ROUTES for m in r.methods}


def test_the_server_does_not_read_its_own_files(system: MemorySystem, tmp_path: Path) -> None:
    """经 HTTP 写入的图片只收 data URL：本地路径与 file:// 指向服务器上的文件，远程调用方不能借此
    把服务器上的文件写进自己的记忆（与记忆代理相同）。"""
    secret = tmp_path / "server-only.png"
    Image.new("RGB", (8, 8), "red").save(secret)
    with TestClient(v1_app(system)) as http:
        for url in (str(secret), f"file://{secret}", ""):
            block = {"type": "image_url", "image_url": {"url": url}}
            message = {"role": "user", "content": [{"type": "text", "text": "look"}, block]}
            response = http.post("/api/v1/memories", json={"messages": message, "user_id": "a"})
            assert response.status_code == 400, url
            assert "data URL" in response.json()["error"]["message"]
    assert system.db.list_spaces() == [] and system.db.get_media("a") == []


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_oversized_images_are_refused(
    system: MemorySystem, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """像素数超过 Pillow 的 Image.MAX_IMAGE_PIXELS 的图片（解压炸弹：几十 KB 的 PNG 解码要几百
    MB）：400 ValueError，什么都不写、不存原图，也不交给图片向量器解码。以前超过两倍的抛出
    DecompressionBombError（500），一到两倍之间的照样存下，每次用到都解码一遍。"""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)
    embedder = system.image_embedder
    assert embedder is not None
    embedded: list[Any] = []
    monkeypatch.setattr(embedder, "embed_images", embedded.append)
    with TestClient(v1_app(system), raise_server_exceptions=False) as http:
        for side in (200, 120):  # 两倍以上；一到两倍之间
            photo = io.BytesIO()
            Image.new("RGB", (side, side), "white").save(photo, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(photo.getvalue()).decode()
            block = {"type": "image_url", "image_url": {"url": url}}
            message = {"role": "user", "content": [{"type": "text", "text": "look"}, block]}
            body = {"messages": message, "user_id": "a", "infer": False}
            response = http.post("/api/v1/memories", json=body)
            assert response.status_code == 400, (side, response.text)
            error = response.json()["error"]
            assert error["type"] == "ValueError" and "too large" in error["message"]
    assert system.db.list_spaces() == [] and system.db.get_media("a") == []
    assert embedded == [] and not list(system.settings.paths.media.rglob("*.png"))


def test_v1_explains_provider_errors(
    system: MemorySystem, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模型供应商出错时 /api/v1 与 /api 一样给出状态码与说明（api/failures.py）：限流 429 带上游的
    Retry-After、余额不足 402、超时 504、内容审核拒绝 422；向量接口限流也是 429。MemoryClient 抛出
    MemoryAPIError（status、code），内容审核与向量接口的出错照本地的类型抛出。以前 openai 的异常
    是没有说明的 500，内容审核与向量接口的限流是 502。"""
    system.llm._sleep = lambda _: None  # LLM 客户端的重试不真等
    Memory.from_system(system).add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    fake_api.chat_rate_limit_headers = {"retry-after": "7"}
    question = {"question": "Where do I live?", "user_id": "alice", "mode": "l1"}
    with TestClient(v1_app(system), raise_server_exceptions=False) as http:
        client = MemoryClient("http://testserver", http_client=http)
        for marker, status, code, said in (
            (RATE_LIMIT, 429, "rate_limited", "rate-limiting"),
            (NO_BALANCE, 402, "no_balance", "balance"),
            (TIMEOUT, 504, "timeout", "did not respond"),
            (REJECT, 422, "content_filtered", "content filter"),
        ):
            fake_api.chat_replies = [marker]
            response = http.post("/api/v1/answer", json=question)
            assert response.status_code == status, response.text
            error = response.json()["error"]
            assert error["code"] == code and said in error["message"], error
            assert response.headers.get("retry-after") == ("7" if status == 429 else None)
            expected = ContentFilteredError if status == 422 else MemoryAPIError
            with pytest.raises(expected) as raised:
                client.answer(**question)
            if isinstance(raised.value, MemoryAPIError):
                assert (raised.value.status, raised.value.code) == (status, code)
                # 与本地的 openai 异常一样带着 Retry-After 与异常类型：以前没有 retry_after
                assert raised.value.retry_after == ("7" if status == 429 else None)
                assert raised.value.type == error["type"] and error["type"]

        def throttled(texts: Any, kind: str) -> Any:
            raise EmbeddingAPIError(
                "向量接口返回 HTTP 429：Throttling", status=429, retry_after="3"
            )

        monkeypatch.setattr(system.embedder, "embed", throttled)
        search = {"query": "where?", "user_id": "alice", "mode": "l1"}
        response = http.post("/api/v1/memories/search", json=search)
        assert response.status_code == 429 and response.headers["retry-after"] == "3"
        assert response.json()["error"]["type"] == "EmbeddingAPIError"
        with pytest.raises(EmbeddingAPIError) as remote:
            client.search("where?", user_id="alice", mode="l1")
        with pytest.raises(EmbeddingAPIError) as local:
            Memory.from_system(system).search("where?", user_id="alice", mode="l1")
        # 与本地一样带着上游的状态码与 Retry-After：以前远程的两个都是 None，按它们退避的代码
        # 换成 MemoryClient 就不退避了
        assert (remote.value.status, remote.value.retry_after) == (429, "3")
        assert (local.value.status, local.value.retry_after) == (429, "3")
