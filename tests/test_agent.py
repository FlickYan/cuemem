"""Agent Runtime（N8）：上下文分区与稳定前缀、工作记忆的淘汰与检索排除、工具循环、引用、
在线写回与会话结束后的巩固。全部离线：哈希向量 + 脚本化模型。"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from PIL import Image
from scripted import TOY_EXTRACTIONS, ScriptedModel, echo_extraction, reconcile_by_keyword

from cuemem.agent import Agent, ContextBudget, Tool, ToolResult
from cuemem.agent.context import Exchange, citations, evict, strip_citations
from cuemem.config import Settings
from cuemem.core.models import Conversation
from cuemem.embed import CachedEmbedder
from cuemem.embed.base import EmbedKind
from cuemem.errors import EmbeddingAPIError
from cuemem.memory import MemorySystem
from cuemem.store.media import image_data_url

DAY = datetime(2025, 10, 10, 19, 0)


def _extract(prompt: str) -> dict[str, Any]:
    """玩具对话按日期给预先写好的抽取结果；Agent 新聊的会话把用户的每句话记成一条事实。"""
    for date, out in TOY_EXTRACTIONS.items():
        if f"Conversation date: {date}" in prompt:
            return out
    return echo_extraction(prompt)


@pytest.fixture
def model() -> ScriptedModel:
    return ScriptedModel(_extract, reconcile=reconcile_by_keyword("搬到", "上海"))


@pytest.fixture
def system(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> Iterator[MemorySystem]:
    fake_api.chat_replies = [model]
    memory = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    conv = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    memory.ingest(conv, consolidate=True)
    yield memory
    memory.close()


def _user_texts(body: dict[str, Any]) -> str:
    content = body["messages"][-1]["content"]
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if b.get("type") == "text")
    return str(content)


def test_stable_prefix_across_turns(system: MemorySystem, model: ScriptedModel) -> None:
    answers = iter(["好的，记住了。", "Max 是金毛 [f2]。", "不客气！"])
    model.agent = lambda body: next(answers)
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    for text in ["你好", "我的狗是什么品种？", "谢谢"]:
        agent.chat(session, text)
    first, second, third = model.agent_requests
    # 每一轮的请求以上一轮请求（去掉最后一条用户消息）为前缀，只在末尾追加
    assert second["messages"][: len(first["messages"]) - 1] == first["messages"][:-1]
    assert third["messages"][: len(second["messages"]) - 1] == second["messages"][:-1]
    # 召回的记忆与日期只在最后一条消息里；历史里是用户的原话与去掉引用编号的回答
    assert "Current date: 2025-10-10" in _user_texts(second)
    assert second["messages"][1] == {"role": "user", "content": "你好"}
    assert third["messages"][4] == {"role": "assistant", "content": "Max 是金毛。"}
    # 画像在系统提示里（会话开始时的快照），每条带编号，可以引用
    assert "About the user (profile" in first["messages"][0]["content"]
    assert "] 用户名叫林晓。" in first["messages"][0]["content"]


def test_recall_cites_and_traces(system: MemorySystem, model: ScriptedModel) -> None:
    model.agent = lambda body: "Max 是一只金毛 [f2]，你上次也提到过 [m1, x9]。"
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    reply = agent.chat(session, "Max 是什么品种？")
    assert reply.citations == ["f2", "m1"]  # 编造的编号去掉
    assert reply.plain == "Max 是一只金毛，你上次也提到过。"
    trace = reply.trace
    assert trace.level == "L1" and trace.router == "rule"
    kinds = {item.kind for item in trace.recalled}
    assert {"fact", "excerpt"} <= kinds
    assert any(item.cited and item.ref == "f2" for item in trace.recalled)
    assert set(trace.partitions) == {"system", "tools", "core", "working", "recalled", "message"}
    assert trace.partitions["recalled"] > 0 and trace.usage["llm_calls"] == 1
    assert trace.timings["total"] >= trace.timings["llm"]


def test_working_memory_is_written_and_excluded(system: MemorySystem, model: ScriptedModel) -> None:
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    agent.chat(session, "我在找一台轻薄笔记本，预算一万。")
    turns = system.db.get_turns("demo", session.session_id)
    assert [t.role for t in turns] == ["user", "assistant"]  # 在线写入了原始日志
    reply = agent.chat(session, "轻薄笔记本的预算是多少？")
    # 第一轮的原话就在工作记忆里，检索时排除：召回的片段都来自更早的会话
    sessions = {item.session_id for item in reply.trace.recalled if item.kind == "excerpt"}
    assert session.session_id not in sessions
    # 会话进行中不登记巩固任务
    assert not [j for j in system.db.get_jobs("demo") if j["session_id"] == session.session_id]


def test_a_failed_write_back_leaves_nothing_and_the_turn_can_be_redone(
    system: MemorySystem, model: ScriptedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """写回这一轮时向量接口出错：这一问一答与它的溯源都不留，会话的序号不动；同一句话重来一次
    只写一遍，"轮"与溯源都齐。以前发言先提交了：没有"轮"、也没有溯源（遗忘时追不到这次回复），
    重来时同样的序号又被悄悄跳过（INSERT OR IGNORE）。"""
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    embed = system.embedder.embed

    def documents_down(texts: Sequence[str], kind: EmbedKind) -> Any:
        if kind == "document":  # 检索时的查询照常向量化
            raise EmbeddingAPIError("向量接口返回 HTTP 503", status=503)
        return embed(texts, kind)

    model.agent = lambda body: "Max 是金毛 [f2]。"
    monkeypatch.setattr(system.embedder, "embed", documents_down)
    with pytest.raises(EmbeddingAPIError):
        agent.chat(session, "Max 是什么品种？")
    assert system.db.get_turns("demo", session.session_id) == [] and session.next_idx == 0

    monkeypatch.setattr(system.embedder, "embed", embed)
    agent.chat(session, "Max 是什么品种？")  # 回答命中 LLM 缓存，与上一次相同
    turns = system.db.get_turns("demo", session.session_id)
    assert [(t.idx, t.text) for t in turns] == [(0, "Max 是什么品种？"), (1, "Max 是金毛。")]
    rounds = system.db.list_units("demo", kinds=["round"], session_ids=[session.session_id])
    assert [u.text for u in rounds] == ["User: Max 是什么品种？\nAssistant: Max 是金毛。"]
    assert len(system.db.provenance("demo", turns[1].id)) == 1 and session.next_idx == 2


def test_session_end_consolidates(system: MemorySystem, model: ScriptedModel) -> None:
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    agent.chat(session, "我下个月要去新加坡出差。")
    system.consolidate("demo")  # 别处触发的巩固也不碰进行中的会话
    assert not [j for j in system.db.get_jobs("demo") if j["session_id"] == session.session_id]
    report = agent.end(session, wait=True)
    assert report is not None and report.sessions == 1
    facts = [f for f in system.facts("demo") if f.session_id == session.session_id]
    assert any("新加坡" in f.statement for f in facts)
    with pytest.raises(ValueError, match="已经结束"):
        agent.chat(session, "还在吗？")


def test_aware_and_naive_times_in_one_session(system: MemorySystem, model: ScriptedModel) -> None:
    """HTTP 接口传来的 at 可能带时区（'Z'）：换算成本机时间。以前带时区开始的会话再用不带时区的
    时间聊（或者反过来），比较"时间不倒流"时就报 TypeError。"""
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY.astimezone(UTC))
    assert session.started_at == DAY and session.started_at.tzinfo is None
    agent.chat(session, "你好", at=DAY + timedelta(minutes=1))
    later = (DAY + timedelta(minutes=2)).astimezone(UTC)
    reply = agent.chat(session, "我下个月要去新加坡出差。", at=later)
    assert reply.at == DAY + timedelta(minutes=2)
    turns = system.db.get_turns("demo", session.session_id)
    assert all(t.ts.tzinfo is None for t in turns) and len(turns) == 4
    report = agent.end(session, wait=True)
    assert report is not None and report.sessions == 1


def test_memory_search_escape_hatch(system: MemorySystem, model: ScriptedModel) -> None:
    steps: list[dict[str, Any]] = []

    def reply(body: dict[str, Any]) -> str | dict[str, Any]:
        steps.append(body)
        if len(steps) == 1:
            args = json.dumps({"query": "狗粮口味"}, ensure_ascii=False)
            return {"tool_calls": [{"id": "c1", "name": "memory_search", "arguments": args}]}
        tool = next(m for m in body["messages"] if m["role"] == "tool")
        found = re.search(r"\[([fm]\d+)\]", tool["content"])
        assert found is not None
        return f"Max 爱吃鸡肉味的 [{found.group(1)}]。"

    model.agent = reply
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    out = agent.chat(session, "你好", mode="l0")  # 控制器没有召回：模型自己查
    assert out.trace.level == "L0"
    assert [t.name for t in out.trace.tools] == ["memory_search"]
    assert out.citations and out.trace.recalled
    assert all(item.via == "memory_search" for item in out.trace.recalled)
    assert steps[0]["tool_choice"] == "auto"


def test_forget_purges_the_query_vectors_of_the_agents_own_searches(
    system: MemorySystem, model: ScriptedModel
) -> None:
    """模型用 memory_search 自己查的话（看过记忆之后写的，可能复述了被删的内容）也向量化成查询
    向量。以前 chat 的 trace 里没记它们（tool 阶段不记参数），遗忘找不到这些查询向量；现在记成
    agent 事件（与 ToolReader 的同一个样子），遗忘时一起删。"""
    query = "新加坡 电商公司 AI 产品经理"
    steps: list[dict[str, Any]] = []

    def reply(body: dict[str, Any]) -> str | dict[str, Any]:
        steps.append(body)
        if len(steps) == 1:
            args = json.dumps({"query": query}, ensure_ascii=False)
            return {"tool_calls": [{"id": "c1", "name": "memory_search", "arguments": args}]}
        return "你在新加坡做 AI 产品经理。"

    model.agent = reply
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    agent.chat(session, "我现在做什么工作？", mode="l0")
    assert isinstance(system.embedder, CachedEmbedder)
    assert system.embedder.cached([query], "query") == {query}

    report = system.forget("demo", session_ids=["s4"])
    assert report.traces_removed >= 1
    assert system.embedder.cached([query], "query") == set()


def test_tool_loop_stops_and_reports_errors(system: MemorySystem, model: ScriptedModel) -> None:
    def broken(args: dict[str, Any], ctx: Any) -> ToolResult:
        raise RuntimeError("catalog offline")

    tool = Tool("search_products", "Search the catalog.", {"type": "object"}, broken)
    model.agent = lambda body: {
        "tool_calls": [{"id": "c1", "name": "search_products", "arguments": "{}"}]
    }
    agent = Agent(system, tools=[tool], router="rule", max_steps=3)
    session = agent.start("demo", at=DAY)
    agent.chat(session, "有什么跑鞋？", mode="l0")
    requests = model.agent_requests
    assert len(requests) == 3 and requests[-1]["tool_choice"] == "none"
    tool_messages = [m for m in requests[-1]["messages"] if m["role"] == "tool"]
    assert tool_messages[0]["content"] == "Tool error: RuntimeError: catalog offline"


def test_images_in_chat(system: MemorySystem, model: ScriptedModel, tmp_path: Path) -> None:
    picture = tmp_path / "shoes.png"
    Image.new("RGB", (64, 48), "white").save(picture)
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    agent.chat(session, "我朋友穿的这双鞋好看吗？", images=[picture], mode="l0")
    agent.chat(session, "那双鞋是什么颜色？", mode="l0")
    first, second = model.agent_requests
    assert any(b.get("type") == "image_url" for b in first["messages"][-1]["content"])
    # 下一轮里图片随工作记忆原样保留（编号 u1）
    history = second["messages"][1]["content"]
    assert any("[Picture u1]" in b.get("text", "") for b in history)
    media = system.db.get_media("demo")
    assert len(media) == 1 and Path(media[0].path).exists()
    assert media[0].id in agent._window_ids(session)  # 检索时排除工作记忆里的图


def test_pictures_are_read_from_memory_once_written(
    system: MemorySystem, model: ScriptedModel, tmp_path: Path
) -> None:
    """用户发的图写进记忆之后，会话（工作记忆、找同款用的 u1）改读媒体库里的那份：调用方的
    文件（HTTP 的上传、界面的临时文件）用完就能删，遗忘删掉原图后不留副本；之后的请求不变。"""
    picture = tmp_path / "shoes.png"
    Image.new("RGB", (64, 48), "white").save(picture)
    sent = image_data_url(picture)
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    agent.chat(session, "我朋友穿的这双鞋好看吗？", images=[picture], mode="l0")
    (stored,) = system.db.get_media("demo")
    assert Path(stored.path).is_relative_to(system.settings.paths.media)
    assert session.uploads == {"u1": stored.path}
    assert session.window[-1].images == [("u1", stored.path)]
    picture.unlink()  # 调用方删掉自己的文件：下一轮照常，请求里的图逐字不变
    agent.chat(session, "那双鞋是什么颜色？", mode="l0")
    history = model.agent_requests[-1]["messages"][1]["content"]
    assert [b["image_url"]["url"] for b in history if b.get("type") == "image_url"] == [sent]


def test_eviction_in_chunks() -> None:
    budget = ContextBudget(working=100, evict_to=0.5)
    window = [Exchange("x" * 120, "y" * 40, DAY, user_idx=2 * i) for i in range(4)]
    assert evict(window, budget) == 3 and len(window) == 1  # 一次淘汰到预算的一半以内
    small = [Exchange("hi", "hello", DAY)]
    assert evict(small, budget) == 0


def test_eviction_makes_old_turns_retrievable(system: MemorySystem, model: ScriptedModel) -> None:
    agent = Agent(system, router="rule", budget=ContextBudget(working=25))
    session = agent.start("demo", at=DAY)
    agent.chat(session, "我最喜欢的颜色是墨绿色，衣服鞋子都挑这个颜色。")
    agent.chat(session, "另外，我穿 42 码的鞋。")
    assert len(session.window) == 1  # 第一轮被淘汰出工作记忆
    reply = agent.chat(session, "我最喜欢什么颜色？")
    texts = [item.text for item in reply.trace.recalled if item.kind == "excerpt"]
    assert any("墨绿色" in t for t in texts)  # 淘汰的内容已经写入长期记忆，能检索到


def test_citation_helpers() -> None:
    refs = {"f1": "a", "m2": "b", "i1": "c"}
    assert citations("预算是一万 [f1]，见 [m2, i1] 和 [f9]。", refs) == ["f1", "m2", "i1"]
    assert strip_citations("预算是一万 [f1]，见 [m2, i1]。") == "预算是一万，见。"
    assert strip_citations("Size 42 [f1].") == "Size 42."
    assert strip_citations("See [Picture u1] and P03.") == "See [Picture u1] and P03."


def test_simulated_clock_moves_forward(system: MemorySystem) -> None:
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    assert session.simulated and session.now() >= DAY
    reply = agent.chat(session, "现在几点？", mode="l0")
    assert reply.at >= DAY and reply.at.date() == DAY.date()


def test_learned_controller_in_agent(system: MemorySystem, model: ScriptedModel) -> None:
    agent = Agent(system)  # 默认：学到的控制器（探针 + 两段式）
    session = agent.start("demo", at=DAY)
    agent.chat(session, "Max 是什么品种？")
    reply = agent.chat(session, "Max 是什么品种？")  # 同一句再问：上一轮的原话不算探针命中
    trace = reply.trace
    assert trace.router == "learned" and "p_none" in trace.scores
    assert trace.level in ("L0", "L1", "L2")


def test_newer_job_supersedes_pending(system: MemorySystem) -> None:
    db = system.db
    assert db.enqueue_job("j1", "demo", "s9", "v", "hash-a")
    assert db.enqueue_job("j2", "demo", "s9", "v", "hash-b")  # 会话又追加了发言
    assert not db.enqueue_job("j2", "demo", "s9", "v", "hash-b")  # 幂等
    status = {j["id"]: j["status"] for j in db.get_jobs("demo") if j["session_id"] == "s9"}
    assert status == {"j1": "superseded", "j2": "pending"}


def test_forget_inside_open_session(system: MemorySystem, model: ScriptedModel) -> None:
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    agent.chat(session, "我妹妹 Ana 的生日是 10 月 3 日。", mode="l0")
    agent.chat(session, "我穿 40 码的鞋。", mode="l0")
    turns = system.db.get_turns("demo", session.session_id)
    doomed = [t.id for t in turns if t.idx in (0, 1)]
    system.forget("demo", turn_ids=doomed)
    assert agent.prune(session) == 1 and len(session.window) == 1
    agent.chat(session, "我穿多大的鞋？", mode="l0")
    texts = [t.text for t in system.db.get_turns("demo", session.session_id)]
    assert not any("Ana" in t for t in texts)  # 删掉的不会被写回来
    assert [t.idx for t in system.db.get_turns("demo", session.session_id)] == [2, 3, 4, 5]
    history = model.agent_requests[-1]["messages"]
    assert not any("Ana" in str(m["content"]) for m in history)


def test_the_next_turn_drops_what_a_forget_deleted(
    system: MemorySystem, model: ScriptedModel
) -> None:
    """遗忘之后没人调用 agent.prune（SDK、/api/v1 的删除与 reset 都不调用）：下一轮在读取里发现
    这个空间遗忘过（MemorySystem.forget_epoch），先去掉被删的轮、重取画像再回答。以前被删的话
    还在工作记忆里，照样发给模型。"""
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    agent.chat(session, "我妹妹 Ana 的生日是 10 月 3 日。", mode="l0")
    agent.chat(session, "我穿 40 码的鞋。", mode="l0")
    turns = system.db.get_turns("demo", session.session_id)
    system.forget("demo", turn_ids=[t.id for t in turns if t.idx in (0, 1)])
    agent.chat(session, "我穿多大的鞋？", mode="l0")
    history = model.agent_requests[-1]["messages"]
    assert not any("Ana" in str(m["content"]) for m in history)
    assert [e.user_idx for e in session.window] == [2, 4]
    assert session.forget_epoch == system.forget_epoch("demo") > 0


def test_turns_without_a_forget_do_not_resync(
    system: MemorySystem, model: ScriptedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有遗忘时每轮只比一次遗忘的次数，不同步、不重取画像：请求与以前相同，稳定前缀不变。"""
    agent = Agent(system, router="rule")

    def never(session: Any) -> int:
        raise AssertionError("没有遗忘，不该同步会话")

    monkeypatch.setattr(agent, "_prune_locked", never)
    session = agent.start("demo", at=DAY)
    for text in ["你好", "我的狗是什么品种？"]:
        agent.chat(session, text)
    first, second = model.agent_requests
    assert second["messages"][: len(first["messages"]) - 1] == first["messages"][:-1]
    assert session.forget_epoch == system.forget_epoch("demo") == 0


def test_forget_follows_citations(system: MemorySystem, model: ScriptedModel) -> None:
    agent = Agent(system, router="rule")
    first = agent.start("demo", at=DAY)
    agent.chat(first, "我妹妹 Ana 的生日是 10 月 3 日。", mode="l0")
    agent.end(first, wait=True)
    fact = next(f for f in system.facts("demo") if "Ana" in f.statement)

    def greet(body: dict[str, Any]) -> str:
        system_prompt = body["messages"][0]["content"]
        found = re.search(r"\[(p\d+)\] 我妹妹 Ana", system_prompt)
        assert found is not None  # 画像里的这条事实带着编号
        return f"别忘了 Ana 10 月 3 日的生日 [{found.group(1)}]。"

    model.agent = greet
    second = agent.start("demo", at=datetime(2025, 10, 1, 9, 0))
    reply = agent.chat(second, "早上好", mode="l0")
    assert reply.citations and reply.trace.recalled[0].kind == "profile"
    later = system.db.get_turns("demo", second.session_id)
    assert system.db.provenance("demo", later[1].id) == [fact.id]

    report = system.forget("demo", turn_ids=fact.source_turn_ids)
    assert report.cascaded == 2  # 复述了生日的那一轮（问候与回复）也删掉
    assert not system.db.get_turns("demo", second.session_id)
    assert agent.prune(second) == 1 and second.window == []
    kept = system.forget("demo", turn_ids=[], follow_citations=True)
    assert kept.turns_deleted == 0


def test_naive_layout_moves_recall_into_system_prompt(
    system: MemorySystem, model: ScriptedModel
) -> None:
    agent = Agent(system, router="rule", layout="naive")
    session = agent.start("demo", at=DAY)
    agent.chat(session, "你好")
    agent.chat(session, "Max 是什么品种？")
    first, second = model.agent_requests
    assert "Recalled memories" in second["messages"][0]["content"]
    assert second["messages"][-1] == {"role": "user", "content": "Max 是什么品种？"}
    assert first["messages"][0] != second["messages"][0]  # 前缀每轮都变


def test_recommended_products_are_the_bold_ones() -> None:
    from cuemem.agent.runtime import ToolRecord, _products

    cards = [{"id": "P01", "name": "Aero Knit Runner"}, {"id": "P04", "name": "Tempo Leather"}]
    record = ToolRecord(
        name="search_products", arguments={}, result="", ms=1, data={"products": cards}
    )
    answer = "**Aero Knit Runner — $110** is vegan. I skipped the Tempo Leather (leather)."
    flags = {p["id"]: p["mentioned"] for p in _products([record], answer)}
    assert flags == {"P01": True, "P04": False}
    plain = {p["id"]: p["mentioned"] for p in _products([record], "Try the Aero Knit Runner.")}
    assert plain == {"P01": True, "P04": False}  # 没有加粗：提到就算


def test_forget_removes_traces_with_the_text(system: MemorySystem, model: ScriptedModel) -> None:
    agent = Agent(system, router="rule")
    session = agent.start("demo", at=DAY)
    secret = "我的银行卡密码提示是外婆家的门牌号。"
    agent.chat(session, secret, mode="l0")
    agent.chat(session, "今天天气不错。", mode="l0")
    assert system.tracer is not None
    traces = "".join(p.read_text("utf-8") for p in system.tracer.directory.glob("*.jsonl"))
    assert secret in traces
    turns = [t.id for t in system.db.get_turns("demo", session.session_id) if t.idx in (0, 1)]
    report = system.forget("demo", turn_ids=turns)
    assert report.traces_removed == 1
    traces = "".join(p.read_text("utf-8") for p in system.tracer.directory.glob("*.jsonl"))
    assert secret not in traces and "今天天气不错。" in traces
