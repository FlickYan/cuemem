"""服务层、HTTP 接口、商品工具、MCP、界面渲染与剧本回放（N8）。全部离线。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI
from fastapi.testclient import TestClient
from PIL import Image
from scripted import ScriptedModel, echo_extraction

from cuemem.agent import Agent
from cuemem.agent.shop import SHOP_PERSONA, Catalog, Product, shop_tools
from cuemem.api.app import create_app
from cuemem.api.mcp import build_server
from cuemem.api.replay import DemoScript, check, replay
from cuemem.api.service import ChatService
from cuemem.api.ui import _assistant, build_ui, render_graph, render_trace
from cuemem.config import Settings
from cuemem.memory import MemorySystem

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


# ---------- HTTP 接口 ----------


def test_http_api(service: ChatService, tmp_path: Path, catalog_root: Path) -> None:
    client = TestClient(create_app(service, upload_dir=tmp_path / "uploads", ui=False))
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
    rejected = client.post(
        f"/api/sessions/{client.post('/api/sessions', json={}).json()['session_id']}/messages",
        data={"text": "hi"},
        files=[("images", ("x.txt", b"not an image", "text/plain"))],
    )
    assert rejected.status_code == 415


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


# ---------- 界面与剧本 ----------


def test_ui_builds_and_renders(service: ChatService, model: ScriptedModel) -> None:
    assert build_ui(service) is not None
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
