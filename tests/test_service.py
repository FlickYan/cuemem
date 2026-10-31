"""服务层、HTTP 接口、商品工具、MCP、界面渲染与剧本回放（N8）。全部离线。"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import re
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fakes import NO_BALANCE, RATE_LIMIT, REJECT, TIMEOUT, FakeAPI
from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError
from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image
from scripted import ScriptedModel, echo_extraction

from cuemem.agent import Agent, ChatSession
from cuemem.agent.shop import SHOP_PERSONA, Catalog, Product, shop_tools
from cuemem.api.app import _adopt_after, _sweep, create_app
from cuemem.api.auth import Keys
from cuemem.api.failures import provider_failure
from cuemem.api.mcp import build_server
from cuemem.api.proxy import MemoryProxy, create_proxy_app
from cuemem.api.replay import DemoScript, check, replay
from cuemem.api.service import ChatService
from cuemem.api.ui import _assistant, build_ui, render_graph, render_trace
from cuemem.config import Settings
from cuemem.errors import ContentFilteredError, EmbeddingAPIError
from cuemem.memory import MemorySystem, _open_mark
from cuemem.sdk import Memory
from cuemem.store.media import image_data_url

DAY = datetime(2026, 8, 28, 19, 30)


def _picture(path: Path, color: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 48), color).save(path)
    return path


@pytest.fixture
def catalog_root(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    _picture(root / "images" / "P01.jpg", "white")
    _picture(root / "images" / "P02.jpg", "brown")
    _picture(root / "images" / "P03.jpg", "teal")
    return root


@pytest.fixture
def model() -> ScriptedModel:
    return ScriptedModel(echo_extraction)


@pytest.fixture
def service(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel, catalog_root: Path
) -> Iterator[ChatService]:
    fake_api.chat_replies = [model]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    products = [
        Product(id="P01", name="Aero Knit Runner", category="running shoes", price=110,
                materials=["knit (vegan)"], sizes=["EU 38", "EU 42"], image="images/P01.jpg"),
        Product(id="P02", name="Tempo Leather Trainer", category="running shoes", price=95,
                materials=["leather"], image="images/P02.jpg"),
        Product(id="P03", name="Peak 20L Daypack", category="hiking gear", price=55,
                image="images/P03.jpg"),
    ]  # fmt: skip
    catalog = Catalog(
        products, catalog_root, embedder=system.embedder, image_embedder=system.image_embedder
    )
    agent = Agent(system, persona=SHOP_PERSONA, tools=shop_tools(catalog), router="rule")
    svc = ChatService(system, agent, catalog=catalog, default_space="shop")
    yield svc
    svc.close()


def _call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"tool_calls": [{"id": "c1", "name": name, "arguments": json.dumps(arguments)}]}


def test_shop_scenario_needs_its_catalog(
    n3_settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """导购演示的商品目录只在源码仓库里（不进 wheel）：找不到时报 CueMemError、指向
    --scenario plain 与 catalog_path，而且不先打开记忆系统。"""
    from cuemem.errors import CueMemError

    missing = tmp_path / "site-packages" / "demo" / "shop" / "catalog.json"
    opened: list[Settings] = []
    monkeypatch.setattr(MemorySystem, "from_settings", lambda s, **kw: opened.append(s))
    with pytest.raises(CueMemError, match="--scenario plain") as caught:
        ChatService.create(n3_settings, scenario="shop", catalog_path=missing)
    assert str(missing) in str(caught.value) and "catalog_path" in str(caught.value)
    monkeypatch.setattr("cuemem.api.service.SHOP_CATALOG", missing)
    with pytest.raises(CueMemError, match="--scenario plain"):
        ChatService.create(n3_settings)  # 默认就是导购演示
    assert opened == []


# ---------- 商品工具 ----------


def test_catalog_search_filters(service: ChatService) -> None:
    assert service.catalog is not None
    found = service.catalog.search("running shoes", max_price=100)
    assert found[0][0].id == "P02" and "P01" not in [p.id for p, _ in found]  # 超出预算的不给
    hiking = service.catalog.search("gift", category="hiking")
    assert [p.id for p, _ in hiking] == ["P03"]
    assert "sizes EU 38–EU 42" in service.catalog.products[0].line()


def test_find_similar_from_upload_and_memory(
    service: ChatService, model: ScriptedModel, catalog_root: Path
) -> None:
    photo = catalog_root / "images" / "P01.jpg"  # 哈希向量：同一张图相似度为 1
    model.agent = lambda body: (
        _call("find_similar_products", image="u1")
        if not any(m["role"] == "tool" for m in body["messages"])
        else "The Aero Knit Runner looks just like it."
    )
    session = service.start("shop", at=DAY)
    reply = service.chat(session.session_id, "Do you have these?", images=[photo], mode="l0")
    record = reply.trace.tools[0]
    assert record.name == "find_similar_products"
    assert record.data["products"][0]["id"] == "P01"
    assert reply.products[0]["mentioned"] and reply.products[0]["image"].endswith("P01.jpg")
    service.end(session.session_id, wait=True)

    # 下一个会话：照片从记忆里召回（i1），按它找同款
    model.agent = lambda body: (
        _call("find_similar_products", image="i1", max_price=200)
        if not any(m["role"] == "tool" for m in body["messages"])
        else "Here you go."
    )
    later = service.start("shop", at=datetime(2026, 10, 10, 10, 0))
    reply = service.chat(later.session_id, "Find shoes like the pair I showed you", mode="l1")
    assert any(item.kind == "image" for item in reply.trace.recalled)
    assert reply.trace.tools[0].data["products"][0]["id"] == "P01"
    unknown = service.agent.tools["find_similar_products"]
    missing = service.agent._run_tool(unknown.name, '{"image": "i9"}', _ctx(service, later))
    assert "No picture called 'i9'" in missing[1].text


def _ctx(service: ChatService, session: Any) -> Any:
    from cuemem.agent.tools import ToolContext
    from cuemem.read.agent import SearchSession
    from cuemem.read.l1 import L1Config

    memory = SearchSession(question_time=DAY, search=lambda q, c: None, config=L1Config())  # type: ignore[arg-type,return-value]
    return ToolContext(service.system, session.space_id, DAY, memory)


# ---------- 服务层 ----------


def test_session_lifecycle_and_memory_views(service: ChatService, model: ScriptedModel) -> None:
    session = service.start("shop", at=DAY)
    service.chat(session.session_id, "I'm vegan and I wear EU 39.", mode="l0")
    assert service.session_list("shop")[0].open
    out = service.end(session.session_id, wait=True)
    assert out["turns"] == 2 and out["consolidation"]["sessions"] == 1
    facts = service.facts("shop")
    assert any("vegan" in f["statement"] for f in facts) and facts[0]["status"] == "current"
    assert service.profile("shop")
    assert service.status("shop")["pending"] == 0
    with pytest.raises(KeyError):
        service.get(session.session_id)
    graph = service.graph("shop")
    assert {"nodes", "edges"} <= set(graph)
    assert "svg" in render_graph(graph) or "No entities" in render_graph(graph)


def test_idle_chat_sessions_are_ended(service: ChatService, tmp_path: Path) -> None:
    """闲置超过 idle_s 的会话由 close_idle 结束、交给后台抽取（HTTP 应用的清理线程每分钟调用）；
    发消息重新计时。以前只有 end 接口与界面的按钮结束会话：关掉的标签页、崩溃的客户端留下的会话
    一直开着，服务运行期间一直不抽取。结束之后再往这个会话发消息是 404，说明里写着闲置结束。"""
    now = [1000.0]
    chat = ChatService(service.system, service.agent, clock=lambda: now[0])
    sid = chat.start("shop", at=DAY).session_id
    chat.chat(sid, "I'm vegan and I wear EU 39.", mode="l0")
    now[0] += 25 * 60
    chat.chat(sid, "I like trail running.", mode="l0")  # 重新计时
    now[0] += 29 * 60
    assert chat.close_idle(30 * 60) == 0 and chat.is_open(sid)
    now[0] += 2 * 60
    assert chat.close_idle(30 * 60) == 1
    assert service.system.worker is not None
    service.system.worker.wait()
    assert [j["status"] for j in service.system.db.get_jobs("shop")] == ["done"]
    assert any("vegan" in f["statement"] for f in chat.facts("shop"))
    assert chat.sessions == {} and service.system.open_sessions("shop") == set()
    assert service.system.db.meta_keys("open-session:") == []
    chat.idle_minutes = 30
    client = TestClient(create_app(chat, upload_dir=tmp_path / "uploads", ui=False))
    reply = client.post(f"/api/sessions/{sid}/messages", data={"text": "hi"})
    assert reply.status_code == 404 and "闲置 30 分钟" in reply.json()["detail"]

    busy = chat.start("shop", at=DAY)  # 正在回答的会话：等这一轮做完再结束
    busy.lock.acquire()
    now[0] += 31 * 60
    sweeping = threading.Thread(target=chat.close_idle, args=(30 * 60,))
    sweeping.start()
    sweeping.join(0.3)
    assert sweeping.is_alive() and not busy.closed
    busy.lock.release()
    sweeping.join(5)
    assert busy.closed and not chat.is_open(busy.session_id)


def test_the_app_sweeps_idle_sessions_while_it_runs(service: ChatService, tmp_path: Path) -> None:
    """create_app 的清理线程只在应用运行期间（lifespan）跑，idle_minutes=None 时不起；一项出错不影响
    别的，线程也不退出。"""

    def sweepers() -> int:
        return sum(t.name == "cuemem-api-sweep" for t in threading.enumerate())

    before = sweepers()
    with TestClient(create_app(service, upload_dir=tmp_path / "up", ui=False, idle_minutes=30)):
        assert sweepers() == before + 1 and service.idle_minutes == 30
    assert sweepers() == before
    service.idle_minutes = None
    with TestClient(create_app(service, upload_dir=tmp_path / "up", ui=False, idle_minutes=None)):
        assert sweepers() == before and service.idle_minutes is None

    calls: list[str] = []
    stop = threading.Event()

    def broken() -> None:
        calls.append("broken")
        raise RuntimeError("boom")

    def fine() -> None:
        calls.append("fine")
        if calls.count("fine") == 2:
            stop.set()

    _sweep(stop, 0.01, broken, fine)
    assert calls == ["broken", "fine", "broken", "fine"]


def test_serve_expires_old_traces_when_it_starts(service: ChatService, tmp_path: Path) -> None:
    """设了 traces_retention_days 时，create_app 的定时线程一启动就删过期的 trace 文件（之后每天
    一次，见 MemorySystem.expire_traces），不结束空闲的会话（idle_minutes=None）时也删。以前 trace
    一直留着。"""
    system = service.system
    system.settings = system.settings.model_copy(update={"traces_retention_days": 7})
    old = system.settings.paths.traces / "20200101.jsonl"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(json.dumps({"attrs": {"space": "shop", "question": "q"}}) + "\n", "utf-8")
    app = create_app(service, upload_dir=tmp_path / "uploads", ui=False, idle_minutes=None)
    with TestClient(app):
        deadline = time.monotonic() + 10
        while old.exists():
            assert time.monotonic() < deadline
            time.sleep(0.02)


def test_graph_links_entities_named_in_statements(
    service: ChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """实体图里，陈述中出现了名字或别名（按词边界）的实体连一条边。先按子串筛再跑正则：
    名字多过 re 的缓存时，以前每条事实都把每个名字的正则重新编译一遍。"""
    from cuemem.core.models import Entity, Fact

    def entity(eid: str, name: str, *aliases: str) -> Entity:
        return Entity(
            id=eid, space_id="g", name=name, type="other", aliases=list(aliases),
            first_seen=DAY, last_seen=DAY,
        )  # fmt: skip

    def fact(fid: str, subject: str, statement: str, obj: str) -> Fact:
        return Fact(
            id=fid, space_id="g", subject_id=subject, subject=subject, predicate="p",
            object=obj, statement=statement, recorded_at=DAY, session_id="s",
            source_turn_ids=[],
        )  # fmt: skip

    entities = [
        entity("u", "User"),
        entity("max", "Max", "Maxie"),
        entity("lis", "Lisbon"),
        *(entity(f"p{i}", f"Person {i:03d}", f"nick{i}") for i in range(600)),
    ]
    facts = [
        fact("f1", "u", "The user walks Max in Lisbon.", "a dog"),
        fact("f2", "u", "The user likes maximalism.", "art"),  # "max" 只是子串
        fact("f3", "max", "Maxie sleeps a lot.", "sleep"),
        fact("f4", "u", "The user met nick12 and Person 007.", "friends"),
    ]
    service.system.db.apply_consolidation(facts=facts, entities=entities, episodes=[], units=[])
    searches: list[str] = []
    search = re.search

    def counting(pattern: str, text: str, *args: Any) -> Any:
        searches.append(pattern)
        return search(pattern, text, *args)

    monkeypatch.setattr(re, "search", counting)
    graph = service.graph("g")
    edges = {(e["source"], e["target"]) for e in graph["edges"]}
    assert edges == {("u", "max"), ("u", "lis"), ("u", "p12"), ("u", "p7")}
    assert len(searches) <= 12  # 以前 4 条事实 × 1,200 多个名字


def test_forget_fact_removes_whole_round(service: ChatService, model: ScriptedModel) -> None:
    model.agent = lambda body: "Happy early birthday to Ana on October 3!"
    session = service.start("shop", at=DAY)
    service.chat(session.session_id, "My sister Ana turns 30 on October 3.", mode="l0")
    service.end(session.session_id, wait=True)
    fact = next(f for f in service.facts("shop") if "Ana" in f["statement"])
    live = service.start("shop", at=datetime(2026, 9, 1))
    report = service.forget("shop", fact_ids=[fact["id"]])
    assert report.turns_deleted == 2  # 用户的话与助手复述的回复一起删
    assert not service.system.db.get_turns("shop", session.session_id)
    assert not any("Ana" in f["statement"] for f in service.facts("shop"))
    assert live.window == []
    # 已经同步过、记下了遗忘的次数：下一轮不再同步一遍（不再重取画像）
    assert live.forget_epoch == service.system.forget_epoch("shop") == 1


def test_forget_prunes_open_sessions_even_when_the_rebuild_fails(
    service: ChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """重建派生记忆失败（例如余额不足）时，原始日志已经删了：进行中的会话照样把被删的轮从
    工作记忆里去掉，出错写在报告的 rebuild_error 里（不再抛给调用方：删除已经做了）。"""
    first = service.start("shop", at=DAY)
    service.chat(first.session_id, "I'm vegan, so no leather please.", mode="l0")
    service.end(first.session_id, wait=True)  # 有巩固过的会话：遗忘时要重建
    live = service.start("shop", at=datetime(2026, 9, 1))
    service.chat(live.session_id, "My sister Ana turns 30 on October 3.", mode="l0")
    assert len(live.window) == 1
    said = service.system.db.get_turns("shop", live.session_id)[0]

    def broken(space_id: str) -> Any:
        raise RuntimeError("Insufficient Balance")

    monkeypatch.setattr(service.system.consolidator, "run", broken)
    report = service.forget("shop", turn_ids=[said.id])
    assert (
        report.turns_deleted == 1 and report.rebuild_error == "RuntimeError: Insufficient Balance"
    )
    assert live.window == []


SECRET = "My bank PIN 4321 is secret."
EVE_TOKEN, ADMIN_TOKEN = "eve-token-0123456789", "admin-token-0123456789"
AS_EVE = {"Authorization": f"Bearer {EVE_TOKEN}"}
AS_ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


def _secret_then_live(service: ChatService, model: ScriptedModel) -> tuple[str, ChatSession]:
    """eve 在一段结束了的会话里说过 SECRET（画像里有由它抽出的事实），之后开着一个会话、聊过
    一轮：（结束了的会话，进行中的会话）。"""
    model.agent = lambda body: "Noted."
    old = service.start("eve", at=DAY)
    service.chat(old.session_id, SECRET, mode="l0")
    service.end(old.session_id, wait=True)
    live = service.start("eve", at=datetime(2026, 9, 1))
    service.chat(live.session_id, "I like green tea.", mode="l0")
    assert any("4321" in statement for _, _, statement in live.profile)
    return old.session_id, live


def _sent_since(fake_api: FakeAPI, n: int) -> str:
    return json.dumps(fake_api.chat_requests[n:], ensure_ascii=False)


@pytest.mark.parametrize(
    "how", ["delete", "delete_all", "delete_all_run", "admin_reset", "scoped_reset"]
)
def test_v1_deletions_reach_open_chat_sessions(
    service: ChatService, model: ScriptedModel, fake_api: FakeAPI, tmp_path: Path, how: str
) -> None:
    """/api/v1 的删除（一条记忆、delete_all、delete_all(run_id)、管理令牌与限定作用域的令牌的
    reset）不经 ChatService.forget：以前进行中的会话（界面、/api）的画像快照还留着被删的事实，
    下一轮照样发给模型，回复复述了还写回日志（D-021、D-046）。现在会话在下一轮的读取里发现这个
    空间遗忘过，先同步再回答。"""
    keys = Keys.parse(
        f"keys:\n  - name: eve-app\n    token: {EVE_TOKEN}\n    scopes:\n      - user: eve\n",
        ADMIN_TOKEN,
    )
    app = create_app(
        service, upload_dir=tmp_path / "up", ui=False, mcp=False, keys=keys, idle_minutes=None
    )
    with TestClient(app) as http:
        old, live = _secret_then_live(service, model)
        listed = http.get("/api/v1/memories", params={"user_id": "eve"}, headers=AS_EVE).json()
        fact = next(f["id"] for f in listed["results"] if "4321" in f["memory"])
        if how == "delete":
            done = http.delete(f"/api/v1/memories/{fact}", headers=AS_EVE)
        elif how in ("delete_all", "delete_all_run"):
            params = {"user_id": "eve"} | ({"run_id": old} if how == "delete_all_run" else {})
            done = http.delete("/api/v1/memories", params=params, headers=AS_EVE)
        else:
            done = http.post("/api/v1/reset", headers=AS_ADMIN if how == "admin_reset" else AS_EVE)
        assert done.status_code == 200, done.text
        assert not any("4321" in t.text for t in service.system.db.get_turns("eve"))
        n = len(fake_api.chat_requests)
        reply = http.post(
            f"/api/sessions/{live.session_id}/messages",
            data={"text": "Hi again", "mode": "l0"},
            headers=AS_ADMIN,
        )
        assert reply.status_code == 200, reply.text
        assert "4321" not in _sent_since(fake_api, n)
        assert live.profile == []


def test_a_turn_waiting_behind_a_forget_does_not_send_what_it_deleted(
    service: ChatService,
    model: ScriptedModel,
    fake_api: FakeAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """这一轮已经拿着会话锁、在等读取（排在一次遗忘后面）：以前遗忘做完之后它照样带着旧的画像
    发给模型——ChatService.forget 的同步要等会话锁，也就是等这一轮做完。现在这一轮拿到读取时
    自己先同步。"""
    _, live = _secret_then_live(service, model)
    fact = next(f["id"] for f in service.facts("eve") if "4321" in f["statement"])
    inside, release = threading.Event(), threading.Event()
    delete_files = service.system._delete_image_files

    def held(items: Any) -> Any:  # 遗忘提交之后、还独占着这个空间时停下
        inside.set()
        release.wait(10)
        return delete_files(items)

    monkeypatch.setattr(service.system, "_delete_image_files", held)
    forgetting = threading.Thread(target=service.forget, args=("eve",), kwargs={"fact_ids": [fact]})
    forgetting.start()
    assert inside.wait(10)
    n, asked = len(fake_api.chat_requests), model.calls["agent"]
    chatting = threading.Thread(
        target=service.chat, args=(live.session_id, "Hi again"), kwargs={"mode": "l0"}
    )
    chatting.start()
    deadline = time.monotonic() + 10
    while not live.lock.locked():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    time.sleep(0.1)  # 走到读取那里等着
    release.set()
    forgetting.join(10)
    chatting.join(10)
    assert not forgetting.is_alive() and not chatting.is_alive()
    assert model.calls["agent"] == asked + 1
    assert "4321" not in _sent_since(fake_api, n)
    assert live.profile == []


def test_a_v1_delete_all_does_not_deadlock_with_a_waiting_turn(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """delete_all（删除空间）里的遗忘在删除空间的独占里重入。遗忘做完时就去同步进行中的会话的话
    （要拿会话锁），会与拿着会话锁、等读取的这一轮互相等死：删除等会话锁，这一轮等删除让出这个
    空间。会话在自己这一轮的读取里同步，删除的线程不碰会话锁。

    自己建服务：卡死时不能再结束会话、关闭系统（拿不到锁），线程留给进程退出时收拾。"""
    fake_api.chat_replies = [model]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"))
    _, live = _secret_then_live(service, model)
    inside, release = threading.Event(), threading.Event()
    delete_files = system._delete_image_files

    def held(items: Any) -> Any:  # 删除空间里的遗忘提交之后停下
        inside.set()
        release.wait(10)
        return delete_files(items)

    system._delete_image_files = held  # type: ignore[method-assign]
    v1 = Memory.from_system(system)
    deleting = threading.Thread(target=v1.delete_all, kwargs={"user_id": "eve"}, daemon=True)
    deleting.start()
    assert inside.wait(10)
    n = len(fake_api.chat_requests)
    chatting = threading.Thread(
        target=service.chat, args=(live.session_id, "Hi again"), kwargs={"mode": "l0"}, daemon=True
    )
    chatting.start()
    deadline = time.monotonic() + 10
    while not live.lock.locked():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    time.sleep(0.1)
    release.set()
    deleting.join(10)
    chatting.join(10)
    if deleting.is_alive() or chatting.is_alive():
        pytest.fail("delete_all 与等着读取的这一轮互相等死")
    try:
        assert "4321" not in _sent_since(fake_api, n)
        assert live.profile == [] and system.db.get_turns("eve", live.session_id) != []
    finally:
        service.close()


def test_sessions_left_open_by_a_crash_are_adopted(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """HTTP 与界面的会话在进程崩溃时还在进行中（没有结束、没有 service.close()）：下一个进程
    第一次在这个空间开会话时交给抽取；接着写的那个会话（指定了 session_id）不接手。"""
    fake_api.chat_replies = [model]

    def open_service() -> ChatService:
        system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
        return ChatService(system, Agent(system, router="rule"), default_space="shop")

    crashed = open_service()
    left = crashed.start("shop", at=DAY).session_id
    crashed.chat(left, "I live in Berlin", mode="l0")
    kept = crashed.start("shop", at=DAY, session_id="kept").session_id
    crashed.chat(kept, "I like jazz", mode="l0")
    crashed.system.close()  # 进程被杀：两个会话都没人结束

    service = open_service()
    try:
        service.start("shop", session_id="kept")
        assert service.system.worker is not None
        service.system.worker.wait()
        jobs = {j["session_id"]: j["status"] for j in service.system.db.get_jobs("shop")}
        assert jobs == {left: "done"}
        assert "I live in Berlin" in {f.statement for f in service.system.facts("shop")}
        assert service.system.open_sessions("shop") == {"kept"}
    finally:
        service.close()


def _leave_open(n3_settings: Settings, fake_api: FakeAPI) -> tuple[str, str]:
    """一个进程留下两个进行中的会话就退出了：eve 的 /api/v1 run（cuemem serve 停止时不结束它）与
    zoe 的界面会话（进程崩溃，没有 service.close()）。返回（run_id，界面的会话）。"""
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    server = Memory.from_system(system)
    run_id = server.open_run(user_id="eve")
    server.add("I live in Lisbon", user_id="eve", run_id=run_id)
    chat = ChatService(system, Agent(system, router="rule"))
    left = chat.start("zoe", at=DAY).session_id
    chat.chat(left, "I like jazz", mode="l0")
    system.close()
    return run_id, left


def test_left_open_sessions_are_handed_over_after_one_idle_period(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """之前的进程留下的进行中的会话，启动一个闲置期之后由清理线程的这一项交给抽取，即使再也没人碰
    这些空间（以前只在有人再写这个空间时接手：/api/v1 的 search、MCP 与界面的浏览都不接手）。
    一个闲置期之内不动（调用方可能马上接着写），之后本进程里进行中的会话不动；只做一次。"""
    fake_api.chat_replies = [model]
    run_id, left = _leave_open(n3_settings, fake_api)
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    try:
        now = [1000.0]
        adopt = _adopt_after(system, 60, clock=lambda: now[0])
        now[0] += 59
        adopt()
        assert len(system.db.meta_keys("open-session:")) == 2 and system.db.get_jobs("eve") == []
        system.open_session("zoe", "live")
        now[0] += 1
        adopt()
        assert system.worker is not None
        system.worker.wait()
        jobs = [*system.db.get_jobs("eve"), *system.db.get_jobs("zoe")]
        assert {j["session_id"]: j["status"] for j in jobs} == {run_id: "done", left: "done"}
        assert "I live in Lisbon" in {f.statement for f in system.facts("eve")}
        assert system.db.meta_keys("open-session:") == [_open_mark("zoe", "live")]
        assert system.open_sessions("zoe") == {"live"}
        system.db.set_meta(_open_mark("zoe", "later"), DAY.isoformat())
        now[0] += 3600
        adopt()  # 只做一次
        assert _open_mark("zoe", "later") in system.db.meta_keys("open-session:")
    finally:
        system.close()


@pytest.mark.parametrize("app", ["serve", "proxy"])
def test_serve_and_proxy_hand_over_left_open_sessions(
    n3_settings: Settings,
    fake_api: FakeAPI,
    model: ScriptedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    app: str,
) -> None:
    """cuemem serve（create_app）与 cuemem proxy（create_proxy_app）的清理线程装着这一项：之前的
    进程留下的会话在启动一个闲置期（idle_minutes）之后交给抽取。以前重启之后这些标记一直留着。"""
    fake_api.chat_replies = [model]
    run_id, left = _leave_open(n3_settings, fake_api)
    monkeypatch.setattr("cuemem.api.app.SWEEP_INTERVAL_S", 0.01)
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    if app == "serve":
        service = ChatService(system, Agent(system, router="rule"))
        built = create_app(service, upload_dir=tmp_path / "up", ui=False, idle_minutes=0.001)
        close: Callable[[], None] = service.close
    else:
        proxy = MemoryProxy(system, http_client=fake_api.client(), idle_minutes=0.001)
        built = create_proxy_app(proxy)
        close = system.close
    try:
        with TestClient(built):
            deadline = time.monotonic() + 5
            while system.db.meta_keys("open-session:"):
                assert time.monotonic() < deadline
                time.sleep(0.01)
        assert system.worker is not None
        system.worker.wait()
        jobs = [*system.db.get_jobs("eve"), *system.db.get_jobs("zoe")]
        assert {j["session_id"]: j["status"] for j in jobs} == {run_id: "done", left: "done"}
    finally:
        close()


def test_forget_keeps_other_sessions_reply_cache(
    service: ChatService, model: ScriptedModel
) -> None:
    """回答缓存按会话归属（N11）：遗忘一条事实只清受影响会话的缓存，别的会话回放时仍然命中。"""
    from cuemem.write.consolidate import scope

    model.agent = lambda body: "Noted!"
    first = service.start("shop", at=DAY)
    service.chat(first.session_id, "I'm vegan, so no leather please.", mode="l0")
    service.end(first.session_id, wait=True)
    second = service.start("shop", at=datetime(2026, 9, 12))
    service.chat(second.session_id, "My sister Ana turns 30 on October 3.", mode="l0")
    service.end(second.session_id, wait=True)
    cache = service.system.llm.cache
    assert cache is not None
    before = len(cache.responses(scope("shop", f"session:{first.session_id}")))
    fact = next(f for f in service.facts("shop") if "Ana" in f["statement"])
    service.forget("shop", fact_ids=[fact["id"]])
    assert len(cache.responses(scope("shop", f"session:{first.session_id}"))) == before > 0
    assert not cache.responses(scope("shop", f"session:{second.session_id}"))


BIRTHDAY_CHAT = ("My sister Ana turns 30 on October 3.", "I'll get her the poles.", "Ana hikes.")


def _birthday_chat(
    service: ChatService,
    model: ScriptedModel,
    replies: list[str],
    said: Sequence[str] = BIRTHDAY_CHAT,
) -> str:
    """一段已巩固的会话：用户说了 Ana 的生日，助手后来没有引用、顺口转述了一次。

    返回那条事实的 ID。
    """
    answers = iter(replies)
    model.agent = lambda body: next(answers)

    def extraction(prompt: str) -> dict[str, Any]:
        out = echo_extraction(prompt)  # 用户的每句话记成一条事实
        said = any("October 3" in f["statement"] for f in out["facts"])
        for n, text in re.findall(r"\[t(\d+)\] Assistant: (.*)", prompt):
            # 用户的原话还在时不重复记；原话删掉后，助手的转述成了唯一的出处
            if "October 3" in text and not said:
                out["facts"].append(
                    {**out["facts"][0], "statement": text, "turns": [int(n)], "subject": "Ana",
                     "predicate": "birthday", "object": "October 3"}
                )  # fmt: skip
        return out

    model.extractions = extraction
    session = service.start("shop", at=DAY)
    for text in said:
        service.chat(session.session_id, text, mode="l0")
    service.end(session.session_id, wait=True)
    return next(f["id"] for f in service.facts("shop") if f["statement"].startswith("My sister"))


def _flag(prompt: str, header: str) -> dict[str, Any]:
    """假的复述检查：编号列表里提到 October 3 的都算复述（只回答给定的那一种检查）。"""
    if header not in prompt:
        return {"same": []}
    listing = prompt.split(header)[1]
    parts = re.split(r"^(?:\[(\d+)\]|(\d+)\.) ", listing, flags=re.M)[1:]
    items = [
        (int(a or b), text)
        for a, b, text in zip(parts[::3], parts[1::3], parts[2::3], strict=False)
    ]
    return {"same": [n for n, text in items if "October 3" in text]}


def test_forget_removes_uncited_restatements_in_the_log(
    service: ChatService, model: ScriptedModel
) -> None:
    """没有引用、追不到的转述（D-051）：已巩固会话里助手顺口说的，进行中的会话里问候时提的，
    都在原始日志里找出来一起删；只删那条回复，同一轮里用户的话、只是提到同一个人的都保留。"""
    replies = ["Happy early birthday!", "Great pick. Ana turns 30 on October 3.", "Nice!",
               "Welcome back! Ana's big day was October 3."]  # fmt: skip
    fact_id = _birthday_chat(service, model, replies)
    model.regrown = lambda prompt: _flag(prompt, "Conversation excerpts:")
    live = service.start("shop", at=datetime(2026, 10, 10))
    service.chat(live.session_id, "Hi again", mode="l0")  # 问候里提到了生日，没有引用
    report = service.forget("shop", fact_ids=[fact_id])
    statements = [f["statement"] for f in service.facts("shop")]
    assert report.restated == 2 and report.regrown == 0  # 两条回复：已巩固的转述与进行中的问候
    assert report.turns_deleted == 4
    assert not any("October 3" in s for s in statements)
    assert "Ana hikes." in statements  # 只是提到同一个人，不算复述
    assert "I'll get her the poles." in statements  # 转述那一轮里用户的话保留
    assert live.window == []  # 问候的回复删了，这一轮从工作记忆里整轮去掉
    assert [t.text for t in service.system.db.get_turns("shop", live.session_id)] == ["Hi again"]


def test_forget_catches_facts_that_grow_back(service: ChatService, model: ScriptedModel) -> None:
    """日志里没找出来的转述，重建之后会重新抽成事实：再删它的出处（兜底）。"""
    fact_id = _birthday_chat(
        service, model, ["Happy early birthday!", "Great pick. Ana turns 30 on October 3.", "Nice!"]
    )
    model.regrown = lambda prompt: _flag(prompt, "Remaining facts:")
    report = service.forget("shop", fact_ids=[fact_id])
    statements = [f["statement"] for f in service.facts("shop")]
    assert report.restated == 0 and report.regrown == 1 and report.turns_deleted == 3
    assert not any("October 3" in s for s in statements) and "Ana hikes." in statements
    assert "I'll get her the poles." in statements  # 只删长出它的那条回复


def test_forget_restatement_does_not_take_its_round_along(
    service: ChatService, model: ScriptedModel
) -> None:
    """转述只删那条回复，也不从它出发沿引用扩散（重录演示时发现）：助手在"换成 40 码"那一轮
    顺口提了生日；整轮删掉会连带删掉用户的 40 码，再沿引用删掉后来所有引用 40 码的回答。"""
    replies = ["Happy early birthday!", "Noted, EU 40. And Ana turns 30 on October 3.", "Nice!"]
    said = (
        "My sister Ana turns 30 on October 3.",
        "I switched to EU 40 for running.",
        "Ana hikes.",
    )
    fact_id = _birthday_chat(service, model, replies, said)
    size = next(f["id"] for f in service.facts("shop") if "EU 40" in f["statement"])
    model.agent = lambda body: "You wear EU 40 now."
    live = service.start("shop", at=datetime(2026, 10, 10))
    service.chat(live.session_id, "What size do I wear now?", mode="l0")
    answer = service.system.db.get_turns("shop", live.session_id)[-1]
    service.system.db.add_provenance("shop", answer.id, [size])  # 这条回答引用了 40 码
    model.regrown = lambda prompt: _flag(prompt, "Conversation excerpts:")
    report = service.forget("shop", fact_ids=[fact_id])
    statements = [f["statement"] for f in service.facts("shop")]
    assert report.restated == 1 and report.cascaded == 0 and report.turns_deleted == 3
    assert "I switched to EU 40 for running." in statements
    assert len(service.system.db.get_turns("shop", live.session_id)) == 2
    assert not any("October 3" in s for s in statements)


# ---------- HTTP 接口 ----------


def _holding(root: Path, data: bytes, *, skip: Path) -> list[Path]:
    """root 下（skip 之外）内容与 data 相同的文件。"""
    digest = hashlib.sha256(data).hexdigest()
    return [
        p
        for p in root.rglob("*")
        if p.is_file()
        and skip not in p.parents
        and hashlib.sha256(p.read_bytes()).hexdigest() == digest
    ]


def test_uploads_left_by_earlier_versions_are_removed(service: ChatService, tmp_path: Path) -> None:
    """以前的版本把经 HTTP 发的照片一直留在上传目录里，遗忘之后磁盘上还有一份：建应用时清掉。
    只删上传会写的图片，别的文件不动。"""
    uploads = tmp_path / "uploads"
    old = _picture(uploads / "0f1e2d3c4b5a69788796a5b4c3d2e1f0.jpg", "red")
    _picture(uploads / "older.PNG", "blue")
    notes = uploads / "notes.txt"
    notes.write_text("keep me", encoding="utf-8")
    create_app(service, upload_dir=uploads, ui=False)
    assert not old.exists() and sorted(p.name for p in uploads.iterdir()) == ["notes.txt"]
    create_app(service, upload_dir=tmp_path / "missing", ui=False)  # 目录还不存在也行


def test_http_api(service: ChatService, tmp_path: Path, catalog_root: Path) -> None:
    uploads = tmp_path / "uploads"
    client = TestClient(create_app(service, upload_dir=uploads, ui=False))
    assert client.get("/api/health").json()["ok"]
    started = client.post("/api/sessions", json={"space": "shop", "at": "2026-08-28T19:30"})
    sid = started.json()["session_id"]
    photo = (catalog_root / "images" / "P03.jpg").read_bytes()
    reply = client.post(
        f"/api/sessions/{sid}/messages",
        data={"text": "Look at my new backpack", "mode": "l0"},
        files=[("images", ("pack.jpg", photo, "image/jpeg"))],
    )
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["trace"]["level"] == "L0" and body["text"] == "OK"
    # 上传的文件只用这一轮：图片存进了记忆，会话改读那份
    assert list(uploads.iterdir()) == []
    assert len(_holding(tmp_path, photo, skip=catalog_root)) == 1
    media = client.get("/api/spaces/shop/media").json()
    assert len(media) == 1
    assert client.get(f"/api/spaces/shop/media/{media[0]['id']}").status_code == 200
    assert client.post(f"/api/sessions/{sid}/end", params={"wait": True}).status_code == 200
    assert client.post(f"/api/sessions/{sid}/messages", data={"text": "hi"}).status_code == 404
    facts = client.get("/api/spaces/shop/facts").json()
    assert facts and client.get("/api/spaces/shop/sessions").json()[0]["turns"] == 2
    recalled = client.post("/api/spaces/shop/recall", json={"query": "backpack", "mode": "l1"})
    assert recalled.json()["level"] == "L1" and recalled.json()["excerpts"]
    bad = client.post("/api/spaces/shop/recall", json={"query": "x", "mode": "l9"})
    assert bad.status_code == 422
    forgot = client.post("/api/spaces/shop/forget", json={"session_ids": [sid]}).json()
    assert forgot["turns_deleted"] == 2 and forgot["images_deleted"] == 1
    # 遗忘之后数据目录里没有这张照片的任何副本（以前上传目录里留着原样的一份）
    assert _holding(tmp_path, photo, skip=catalog_root) == []
    rejected = client.post(
        f"/api/sessions/{client.post('/api/sessions', json={}).json()['session_id']}/messages",
        data={"text": "hi"},
        files=[
            ("images", ("ok.jpg", photo, "image/jpeg")),
            ("images", ("x.txt", b"not an image", "text/plain")),
        ],
    )
    assert rejected.status_code == 415
    assert list(uploads.iterdir()) == []  # 第二个文件被拒，先存下的第一个也删掉


def test_http_api_explains_provider_errors(
    service: ChatService,
    model: ScriptedModel,
    fake_api: FakeAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """模型供应商出错时 HTTP 接口给出状态码与说明，不是没有说明的 500：限流 429（带上游的
    Retry-After，重试过之后）、余额不足 402、超时 504、内容审核拒绝 422；向量接口限流也是 429。"""
    service.system.llm._sleep = lambda _: None  # LLM 客户端的重试不真等
    app = create_app(service, upload_dir=tmp_path / "uploads", ui=False)
    client = TestClient(app, raise_server_exceptions=False)
    sid = client.post("/api/sessions", json={"space": "alice"}).json()["session_id"]
    fake_api.chat_rate_limit_headers = {"retry-after": "7"}
    for reply, status, said in (
        (RATE_LIMIT, 429, "限流"),
        (NO_BALANCE, 402, "余额不足"),
        (TIMEOUT, 504, "超时"),
        (REJECT, 422, "内容审核"),
    ):
        model.agent = lambda body, reply=reply: reply
        response = client.post(f"/api/sessions/{sid}/messages", data={"text": "hi", "mode": "l0"})
        assert response.status_code == status, response.text
        assert said in response.json()["detail"]
        assert response.headers.get("retry-after") == ("7" if status == 429 else None)

    def throttled(texts: Sequence[str], kind: str) -> Any:
        raise EmbeddingAPIError("向量接口返回 HTTP 429：Throttling", status=429, retry_after="3")

    monkeypatch.setattr(service.system.embedder, "embed", throttled)
    response = client.post("/api/spaces/alice/recall", json={"query": "shoes", "mode": "l1"})
    assert response.status_code == 429 and response.headers["retry-after"] == "3"


UPSTREAM = httpx2.Request("POST", "https://upstream.invalid/v1/x")


def _upstream(code: int, **headers: str) -> APIStatusError:
    response = httpx2.Response(code, headers=headers, request=UPSTREAM)
    return APIStatusError("upstream", response=response, body=None)


def _embedding_timeout() -> EmbeddingAPIError:
    error = EmbeddingAPIError("timed out")
    error.__cause__ = httpx2.ReadTimeout("timed out", request=UPSTREAM)
    return error


# 供应商的错误 → （回给调用方的状态码，种类，Retry-After）
FAILURES: list[tuple[BaseException, tuple[int, str, str | None]]] = [
    (_upstream(429, **{"retry-after": "7"}), (429, "rate_limited", "7")),
    (_upstream(402), (402, "no_balance", None)),
    (_upstream(503, **{"retry-after": "9"}), (503, "unavailable", "9")),
    (_upstream(408), (504, "timeout", None)),
    (_upstream(504), (504, "timeout", None)),
    (_upstream(500), (502, "upstream_error", None)),
    (_upstream(401), (502, "upstream_error", None)),
    (APITimeoutError(request=UPSTREAM), (504, "timeout", None)),
    (APIConnectionError(request=UPSTREAM), (502, "upstream_error", None)),
    (ContentFilteredError("blocked"), (422, "content_filtered", None)),
    (EmbeddingAPIError("throttled", status=429, retry_after="3"), (429, "rate_limited", "3")),
    (EmbeddingAPIError("bad json"), (502, "upstream_error", None)),
    (_embedding_timeout(), (504, "timeout", None)),
]


@pytest.mark.parametrize(
    ("exc", "expected"),
    FAILURES,
    ids=[f"{type(e).__name__}-{i}" for i, (e, _) in enumerate(FAILURES)],
)
def test_provider_failures_map_to_documented_statuses(
    exc: BaseException, expected: tuple[int, str, str | None]
) -> None:
    """HTTP 接口与 MCP 共用的状态码表（D-069）逐行钉住：以前把上游的 504 改成不算超时（回 502）、
    把连不上改成回 500，都没有测试失败。"""
    failure = provider_failure(exc)
    assert failure is not None
    assert (failure.status, failure.kind, failure.retry_after) == expected
    assert provider_failure(ValueError("ours")) is None  # 不是供应商的错误：照常处理


def test_http_photo_stays_in_the_conversation(
    service: ChatService, fake_api: FakeAPI, tmp_path: Path, catalog_root: Path
) -> None:
    """上传的文件删掉之后，同一会话的下一轮照常（200），工作记忆里的那张图逐字不变。"""
    client = TestClient(create_app(service, upload_dir=tmp_path / "uploads", ui=False))
    sid = client.post("/api/sessions", json={"space": "shop", "at": "2026-08-28T19:30"}).json()[
        "session_id"
    ]
    picture = catalog_root / "images" / "P03.jpg"
    first = client.post(
        f"/api/sessions/{sid}/messages",
        data={"text": "Look at my new backpack", "mode": "l0"},
        files=[("images", ("pack.jpg", picture.read_bytes(), "image/jpeg"))],
    )
    assert first.status_code == 200
    seen = len(fake_api.chat_requests)
    second = client.post(f"/api/sessions/{sid}/messages", data={"text": "Colour?", "mode": "l0"})
    assert second.status_code == 200, second.text
    (request,) = fake_api.chat_requests[seen:]
    urls = [
        b["image_url"]["url"]
        for m in request["messages"]
        if isinstance(m["content"], list)
        for b in m["content"]
        if b.get("type") == "image_url"
    ]
    assert urls == [image_data_url(picture)]


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_http_chat_refuses_oversized_images_before_calling_the_model(
    service: ChatService, fake_api: FakeAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """对话里上传的图片像素数超过 Image.MAX_IMAGE_PIXELS（解压炸弹）：409，在召回、解码与调用
    模型之前，什么都不写、不存原图。以前一到两倍之间的先解码、发给模型，写进记忆时才 409（回答
    丢了、钱照付）；超过两倍的是 Pillow 的 DecompressionBombError，500。"""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)
    app = create_app(service, upload_dir=tmp_path / "uploads", ui=False)
    client = TestClient(app, raise_server_exceptions=False)
    sid = client.post("/api/sessions", json={"space": "shop"}).json()["session_id"]
    for side in (200, 120):  # 两倍以上；一到两倍之间
        photo = io.BytesIO()
        Image.new("RGB", (side, side), "white").save(photo, format="PNG")
        seen = len(fake_api.chat_requests)
        response = client.post(
            f"/api/sessions/{sid}/messages",
            data={"text": "Look at this", "mode": "l0"},
            files=[("images", ("big.png", photo.getvalue(), "image/png"))],
        )
        assert response.status_code == 409, (side, response.text)
        assert "Image too large" in response.json()["detail"]
        assert len(fake_api.chat_requests) == seen  # 没有调用模型
    db = service.system.db
    assert db.get_turns("shop") == [] and db.get_media("shop") == []
    assert not list(service.system.settings.paths.media.rglob("*.png"))
    again = client.post(f"/api/sessions/{sid}/messages", data={"text": "Hi", "mode": "l0"})
    assert again.status_code == 200, again.text  # 会话照常


def test_http_api_token(service: ChatService, tmp_path: Path) -> None:
    """设置访问令牌后：健康检查不鉴权，其余接口要带正确的 Bearer 令牌。"""
    client = TestClient(
        create_app(service, upload_dir=tmp_path / "uploads", ui=False, token="fake-token")
    )
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/spaces").status_code == 401
    wrong = client.get("/api/spaces", headers={"Authorization": "Bearer fake-wrong"})
    assert wrong.status_code == 401 and wrong.headers["www-authenticate"] == "Bearer"
    ok = client.get("/api/spaces", headers={"Authorization": "Bearer fake-token"})
    assert ok.status_code == 200
    forget = client.post("/api/spaces/shop/forget", json={"session_ids": ["s1"]})
    assert forget.status_code == 401  # 删除记忆同样要令牌


def test_http_api_agent_scoped_space(service: ChatService, tmp_path: Path) -> None:
    """带 Agent 的空间 ID 里有 "/"（SDK 的 agent_id、代理的 X-CueMem-Agent：bot/alice、
    agent/shop）：以前每个空间接口都 404，浏览、检索、导出、遗忘都做不到；"/" 写成 %2F 也一样。"""
    from cuemem.core.models import SessionIn, TurnIn
    from cuemem.sdk import space_for

    client = TestClient(create_app(service, upload_dir=tmp_path / "uploads", ui=False))
    spaces = [space_for("alice", "bot"), space_for(agent_id="shop")]
    assert spaces == ["bot/alice", "agent/shop"]
    for space in spaces:
        turns = [TurnIn(role="user", text="I live in Berlin"), TurnIn(role="assistant", text="Ok")]
        service.system.append(space, SessionIn(session_id="s1", started_at=DAY, turns=turns), 0)
        service.system.consolidate(space)
    for space in spaces:
        for path in (space, space.replace("/", "%2F")):
            base = f"/api/spaces/{path}"
            assert client.get(f"{base}/sessions").json()[0]["turns"] == 2
            assert client.get(f"{base}/facts").json()
            assert client.get(f"{base}/status").status_code == 200
            assert client.get(f"{base}/export").json()["space_id"] == space
            recalled = client.post(f"{base}/recall", json={"query": "Berlin", "mode": "l1"})
            assert recalled.status_code == 200 and recalled.json()["excerpts"]
            preview = client.post(f"{base}/forget", json={"session_ids": ["s1"], "dry_run": True})
            assert preview.status_code == 200 and len(preview.json()["turns"]) == 2
        forgot = client.post(f"/api/spaces/{space}/forget", json={"session_ids": ["s1"]})
        assert forgot.status_code == 200 and forgot.json()["turns_deleted"] == 2
        assert service.system.db.get_turns(space) == []


def test_http_api_refuses_cross_origin_requests(service: ChatService, tmp_path: Path) -> None:
    """别的网站发来的浏览器请求一律 403。预检请求（OPTIONS）匹配不到接口，以前会落到根路径上的
    Gradio，它对非本机的 Host（Docker、局域网）放行任何来源，之后跨站的 JSON POST 就能遗忘、导入。
    同源（/docs 的 Try it out）、不带 Origin 的请求与界面照常。"""
    client = TestClient(create_app(service, upload_dir=tmp_path / "uploads", ui=True))
    lan, evil = "192.168.1.10:7860", "https://evil.example"
    for path in ("/api/spaces/shop/forget", "/api/spaces/import", "/api/sessions"):
        preflight = client.options(
            path,
            headers={
                "Host": lan,
                "Origin": evil,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 403, path
        assert "access-control-allow-origin" not in preflight.headers
    started = client.post("/api/sessions", json={}, headers={"Host": lan, "Origin": evil})
    assert started.status_code == 403 and service.sessions == {}
    assert client.get("/api/spaces", headers={"Host": lan, "Origin": "null"}).status_code == 403
    same = client.post("/api/sessions", json={}, headers={"Host": lan, "Origin": f"http://{lan}"})
    assert same.status_code == 200
    assert client.get("/api/spaces").status_code == 200
    assert client.get("/", headers={"Host": lan, "Origin": evil}).status_code == 200  # 界面


@pytest.mark.parametrize("bind", ["::1", "127.0.0.2", "127.0.0.1"])
@pytest.mark.filterwarnings("ignore::pandas.errors.Pandas4Warning")  # Gradio 的队列里调用 pandas
def test_ui_api_refuses_cross_origin_requests(
    service: ChatService, tmp_path: Path, bind: str
) -> None:
    """界面自己的接口（/gradio_api）同样只接受同源的浏览器请求。Gradio 只在 Host 正好是
    localhost、127.0.0.1、0.0.0.0 时限制来源：监听 ::1、127.0.0.2（不要令牌的本机地址）时它放行
    任何来源，别的网页就能读出、遗忘任何空间的记忆；上传不经预检，监听 127.0.0.1 时也照样写文件。"""
    from cuemem.cli import _host_policy

    service.remember("alice", "My PIN is 4321")
    service.consolidate("alice")
    allowed, origins = _host_policy(bind)
    app_ = create_app(
        service,
        upload_dir=tmp_path / "uploads",
        ui=True,
        allowed_hosts=allowed,
        origin_hosts=origins,
    )
    host = f"[{bind}]:7860" if ":" in bind else f"{bind}:7860"
    evil = {"Host": host, "Origin": "https://evil.example"}
    call = "/gradio_api/call/load_memory"
    with TestClient(app_) as client:
        preflight = client.options(
            call,
            headers={
                **evil,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 403
        assert "access-control-allow-origin" not in preflight.headers
        posted = client.post(call, json={"data": ["alice", False, "fact"]}, headers=evil)
        assert posted.status_code == 403
        assert "access-control-allow-origin" not in posted.headers
        upload = client.post(
            "/gradio_api/upload", files={"files": ("x.txt", b"A", "text/plain")}, headers=evil
        )
        assert upload.status_code == 403

        same = {"Host": host, "Origin": f"http://{host}"}  # 界面本身
        posted = client.post(call, json={"data": ["alice", False, "fact"]}, headers=same)
        assert posted.status_code == 200
        event = posted.json()["event_id"]
        assert "4321" in client.get(f"{call}/{event}", headers=same).text
        assert client.get("/", headers={"Host": host}).status_code == 200


def test_http_api_trusted_hosts(service: ChatService, tmp_path: Path) -> None:
    """allowed_hosts（cuemem serve 监听本机地址时）：接口与界面都只接受这些 Host，
    防 DNS 重绑定。"""
    app_ = create_app(
        service, upload_dir=tmp_path / "uploads", ui=True, allowed_hosts=["127.0.0.1"]
    )
    client = TestClient(app_, base_url="http://127.0.0.1:7860")
    assert client.get("/api/spaces").status_code == 200
    assert client.get("/").status_code == 200
    for path in ("/api/spaces", "/api/spaces/shop/export", "/"):
        assert client.get(path, headers={"Host": "attacker.example:7860"}).status_code == 400


def test_v1_host_and_origin_refusals_have_the_v1_error_shape(
    service: ChatService, tmp_path: Path
) -> None:
    """OpenAPI 写着 /api/v1 的每个出错都是 V1ErrorBody，按它生成的客户端据此解析：Host 不对（400）
    与别的网站发来的浏览器请求（403）也是。以前是纯文本的 "Invalid host header" 与中文的
    {"detail": …}。/api 与界面照旧。"""
    from cuemem.api.memories import V1ErrorBody

    app_ = create_app(
        service, upload_dir=tmp_path / "uploads", ui=False, allowed_hosts=["127.0.0.1"]
    )
    client = TestClient(app_, base_url="http://127.0.0.1:7860")
    attacker, evil = {"Host": "attacker.example:7860"}, {"Origin": "https://evil.example"}
    refused = client.get("/api/v1/spaces", headers=attacker)
    assert refused.status_code == 400
    error = V1ErrorBody.model_validate(refused.json()).error
    assert error.code == "invalid_request" and "CUEMEM_ALLOWED_HOSTS" in error.message
    body = {"messages": "I like tea", "user_id": "alice", "infer": False}
    refused = client.post("/api/v1/memories", json=body, headers=evil)
    assert refused.status_code == 403
    error = V1ErrorBody.model_validate(refused.json()).error
    assert error.code == "forbidden" and "Cross-site" in error.message
    assert service.system.db.get_turns("alice") == []
    assert client.get("/api/v1/spaces").status_code == 200
    assert client.get("/api/spaces", headers=attacker).text == "Invalid host header"
    started = client.post("/api/sessions", json={}, headers=evil)
    assert started.json() == {"detail": "不接受别的网站发来的浏览器请求"}


def test_http_import_rejects_bad_attachments(service: ChatService, tmp_path: Path) -> None:
    """导入的附件不是图片、或 sha256 是一条路径：409（不是 500），什么都不写。"""
    client = TestClient(create_app(service, upload_dir=tmp_path / "uploads", ui=False))
    image = {"sha256": str(tmp_path / "planted"), "mime": "text/x-python", "data": "cHJpbnQoMSk="}
    turn = {"idx": 0, "role": "user", "text": "<image>", "ts": None, "images": [image]}
    session = {"session_id": "s1", "started_at": "2026-08-28T19:30:00", "turns": [turn]}
    data = {"format": "cuemem.space", "version": 1, "space_id": "x", "sessions": [session]}
    response = client.post("/api/spaces/import", json=data)
    assert response.status_code == 409 and "not a valid image" in response.json()["detail"]
    assert list(tmp_path.glob("planted*")) == []
    assert service.system.db.list_spaces() == []


@pytest.mark.parametrize(
    "sessions",
    [
        [{"session_id": "s1", "started_at": "2026-08-28T19:30:00"}],  # 没有 turns
        [{"session_id": "s1", "started_at": 5, "turns": []}],
        "abc",
    ],
)
def test_http_import_rejects_malformed_files(
    service: ChatService, tmp_path: Path, sessions: Any
) -> None:
    """导入的文件缺键、类型不对：与别的坏文件一样 409（Malformed export），什么都不写。以前缺
    turns 是 404 {"detail": "turns"}，类型不对是没有说明的 500。"""
    client = TestClient(
        create_app(service, upload_dir=tmp_path / "uploads", ui=False),
        raise_server_exceptions=False,
    )
    data = {"format": "cuemem.space", "version": 2, "space_id": "x", "sessions": sessions}
    response = client.post("/api/spaces/import", json=data)
    assert response.status_code == 409 and "Malformed export" in response.json()["detail"]
    assert service.system.db.list_spaces() == []


# ---------- MCP ----------


def test_mcp_tools(service: ChatService) -> None:
    server = build_server(service, "notes")
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert names == {"remember", "recall", "forget", "profile"}

    def call(name: str, **arguments: Any) -> str:
        result = asyncio.run(server.call_tool(name, arguments))
        return "".join(getattr(c, "text", "") for c in result.content)

    assert call("remember", text="I prefer aisle seats on flights.").startswith("Saved as ")
    if service.system.worker is not None:  # 等后台巩固做完
        service.system.worker.wait()
    found = call("recall", query="aisle seats", deep=False)
    assert "aisle" in found and "Memory controller:" in found
    excerpt = next(line for line in found.splitlines() if line.startswith("["))
    memory_id = excerpt[1 : excerpt.index("]")]
    assert "Deleted" in call("forget", ids=[memory_id])
    assert "aisle" not in call("recall", query="aisle seats")


def test_mcp_tools_explain_provider_errors(
    service: ChatService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """供应商出错时工具回一句模型看得懂的话（以前只有 "Error executing tool recall"）。"""
    server = build_server(service, "notes")

    def throttled(texts: Sequence[str], kind: str) -> Any:
        raise EmbeddingAPIError("向量接口返回 HTTP 429：Throttling", status=429, retry_after="3")

    monkeypatch.setattr(service.system.embedder, "embed", throttled)
    for name, arguments in (("remember", {"text": "I like tea."}), ("recall", {"query": "tea"})):
        with pytest.raises(ToolError, match=r"rate-limiting requests \(HTTP 429\).*about 3 s"):
            asyncio.run(server.call_tool(name, arguments))


def test_a_forget_whose_rebuild_fails_still_reports_the_deletion(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """遗忘删掉原始日志之后、重建派生记忆时对账余额不足：MCP 的 forget 与 HTTP 的遗忘照实说删了、
    重建下次巩固时补上。以前 MCP 回 "run out of balance (HTTP 402). Retrying will not help"、HTTP
    回 402，调用方的模型（以及用户）以为这条隐私相关的删除没有发生，其实已经删了。"""
    from test_sdk import moved, where_i_live, without_balance_for_reconciling

    model = ScriptedModel(where_i_live, reconcile=moved)
    fake_api.chat_replies = [model]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        service = ChatService(system, Agent(system, router="rule"))
        notes = [
            service.remember("notes", text, at=datetime(2026, 8, day + 1))["session_id"]
            for day, text in enumerate(
                ("I live in Berlin", "I moved to Zurich", "I moved to Munich", "I moved to Rome")
            )
        ]
        system.consolidate("notes")
        ids = {f.statement: f.id for f in system.facts("notes")}
        broke = without_balance_for_reconciling(model, fake_api)

        result = asyncio.run(
            build_server(service, "notes").call_tool(
                "forget", {"ids": [ids["The user lives in Zurich."]]}
            )
        )
        said = "".join(getattr(c, "text", "") for c in result.content)
        assert said.startswith("Deleted 1 conversation turn(s). Rebuilding the memory derived")
        assert "Error code: 402" in said and "next consolidation" in said
        with TestClient(create_app(service, upload_dir=tmp_path / "up", ui=False)) as http:
            body = {"session_ids": [notes[2]]}  # 派生记忆只重建了一部分：按会话遗忘慕尼黑
            response = http.post("/api/spaces/notes/forget", json=body)
        assert response.status_code == 200 and response.json()["turns_deleted"] == 1
        assert response.json()["warning"].startswith("已经删除；用剩下的日志重建派生记忆时出错")
        assert [t.text for t in system.db.get_turns("notes")] == [
            "I live in Berlin",
            "I moved to Rome",
        ]

        broke["on"] = False
        system.consolidate("notes")
        current = [f.statement for f in system.facts("notes") if f.is_current]
        assert current == ["The user lives in Rome."]


# ---------- 界面与剧本 ----------


def test_ui_builds_and_renders(service: ChatService, model: ScriptedModel) -> None:
    # 界面的临时文件（上传、对话与图库里的副本）一小时后清掉：Gradio 没法在遗忘时删它们
    assert build_ui(service).delete_cache == (3600, 3600)
    model.agent = lambda body: "You wear EU 39 [m1]."
    session = service.start("shop", at=DAY)
    service.chat(session.session_id, "I wear EU 39.", mode="l0")
    reply = service.chat(session.session_id, "What size do I wear?", mode="l0")
    html = render_trace(reply, service)
    for section in ("Controller", "Context budget", "Recalled memories", "Usage"):
        assert section in html
    assert _assistant(reply)[0] == "You wear EU 39."  # 没有召回：编号是编造的，去掉


def test_citations_render_as_superscripts(service: ChatService, model: ScriptedModel) -> None:
    model.agent = lambda body: "Your budget is $180 [f1, m2]."
    session = service.start("shop", at=DAY)
    reply = service.chat(session.session_id, "budget?", mode="l0")
    rendered = _assistant(reply)[0]
    assert rendered == "Your budget is $180."  # 没有召回，编号不存在：去掉
    reply.citations = ["f1", "m2"]
    assert _assistant(reply)[0] == 'Your budget is $180<sup class="cite">f1, m2</sup>.'
    reply.text = "From $120 [f1][m2] to $180."
    assert _assistant(reply)[0] == 'From $120<sup class="cite">f1, m2</sup> to $180.'


def test_replay_checks_expectations(
    service: ChatService, model: ScriptedModel, tmp_path: Path
) -> None:
    answers = iter(["Hi Maya! Here are some shoes.", "Sure, noted.", "You wear EU 39, Maya."])
    model.agent = lambda body: next(answers)
    script = DemoScript.model_validate(
        {
            "space": "shop",
            "sessions": [
                {
                    "at": "2026-08-28T19:30",
                    "messages": [
                        {"text": "Hi, I'm Maya.", "expect": [["Maya"]]},
                        {"text": "I wear EU 39.", "expect_not": ["leather"]},
                    ],
                },
                {
                    "at": "2026-10-10T10:00",
                    "messages": [
                        {"text": "What size do I wear?", "expect": [["40"]], "expect_level": "L1"}
                    ],
                },
            ],
        }
    )
    results = replay(service, script, root=tmp_path)
    assert [r.passed for r in results] == [True, True, False]
    assert results[-1].failures == ["缺少 40"] and results[-1].level == "L1"
    assert check("Size 40", script.sessions[1].messages[0]) == []


def test_trace_renders_cited_profile_lines(service: ChatService, model: ScriptedModel) -> None:
    session = service.start("shop", at=DAY)
    service.chat(session.session_id, "I'm vegan.", mode="l0")
    service.end(session.session_id, wait=True)
    model.agent = lambda body: "Noted, you're vegan [p1]."
    later = service.start("shop", at=datetime(2026, 9, 1))
    reply = service.chat(later.session_id, "Remind me what you know.", mode="l0")
    assert [i.kind for i in reply.trace.recalled] == ["profile"]
    assert "profile" in render_trace(reply, service)
