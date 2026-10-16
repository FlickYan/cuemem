"""LangGraph / LangChain 集成（N12）：记忆工具、create_agent 中间件、create_react_agent 钩子。

全部离线：LangChain 的假对话模型当 Agent 的模型，CueMem 内部的抽取等调用交给脚本化的模型。
"""

from __future__ import annotations

import asyncio
import base64
import io
import tempfile
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain.agents")

from fakes import FakeAPI
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image
from pydantic import BaseModel, Field
from scripted import ScriptedModel, growing_extraction
from test_sdk import moved, where_i_live

import cuemem.integrations.langgraph as lg
from cuemem import AsyncMemory, Memory
from cuemem.api.mcp import build_server
from cuemem.config import Settings
from cuemem.evidence import MEMORY_NOTE
from cuemem.integrations.langgraph import (
    CueMemMiddleware,
    ThreadMemory,
    aend_thread,
    end_thread,
    memory_tools,
    recall_hook,
    save_hook,
    session_for,
)


class Recorder(FakeMessagesListChatModel):
    """假的对话模型：按顺序给出预设的回复，记下每次收到的消息；bind_tools 原样返回自己。"""

    seen: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Recorder:
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kw)


def ask(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def thread(name: str, **configurable: Any) -> dict[str, Any]:
    return {"configurable": {"thread_id": name, **configurable}}


def turns(memory: Memory, space: str, thread_id: str) -> list[tuple[str, str]]:
    return [(t.role, t.text) for t in memory.system.db.get_turns(space, session_for(thread_id))]


def facts(memory: Memory, user_id: str = "alice") -> list[str]:
    return [f["memory"] for f in memory.get_all(user_id=user_id)["results"]]


def tool_call(name: str, call_id: str = "c1", **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def counting_retrieve(memory: Memory, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """记下每次检索的问题（检查一个回合只召回一次）。"""
    calls: list[str] = []
    retrieve = memory.system.retrieve

    def counting(space: str, question: str, **kwargs: Any) -> Any:
        calls.append(question)
        return retrieve(space, question, **kwargs)

    monkeypatch.setattr(memory.system, "retrieve", counting)
    return calls


@tool
def weather(city: str) -> str:
    """Current weather in a city."""
    return f"Sunny in {city}."


@pytest.fixture
def memory(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[Memory]:
    fake_api.chat_replies = [ScriptedModel(where_i_live)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        yield m


@pytest.fixture
def lives_in_berlin(memory: Memory) -> Memory:
    memory.add("I live in Berlin", user_id="alice", wait=True)
    return memory


# ---------- 工具 ----------


def test_tools_match_the_mcp_server(memory: Memory) -> None:
    server = build_server(None, "x")  # type: ignore[arg-type]
    listed = asyncio.run(server.list_tools())
    mcp = {t.name: " ".join((t.description or "").split()) for t in listed}
    tools = memory_tools(memory, user_id="alice")
    assert [t.name for t in tools] == ["recall", "remember", "forget"]
    for t in tools:  # 同名工具、同样的说明（profile 不单独做工具：recall 的结果里带着画像）
        assert t.description == mcp[t.name]
    # 只有模型要填的参数出现在工具的参数里，config 由 LangChain 注入
    assert set(tools[0].tool_call_schema.model_json_schema()["properties"]) == {"query", "deep"}


def test_tools_take_the_user_from_the_config(memory: Memory) -> None:
    recall, remember, forget = memory_tools(memory)  # 不固定用户：每次调用从 config 里取
    alice = {"configurable": {"user_id": "alice"}}
    saved = remember.invoke({"text": "I live in Berlin"}, config=alice)
    assert saved.startswith("Saved as ")
    memory.consolidate(user_id="alice")
    (fact,) = memory.get_all(user_id="alice")["results"]
    assert fact["memory"] == "The user lives in Berlin."

    found = recall.invoke({"query": "Where do I live?"}, config=alice)
    assert found.startswith("Memory controller: L")
    assert f"[{fact['id']}]" in found and "The user lives in Berlin." in found
    # 别的用户什么都看不到；没有用户时给模型一条错误结果，不会写进某个默认空间
    bob = recall.invoke({"query": "Where do I live?"}, config={"configurable": {"user_id": "bob"}})
    assert bob.startswith("Memory controller: L0 (no memories yet)")  # 空的空间不检索
    assert "No memories were retrieved" in bob
    assert "needs a user_id" in recall.invoke({"query": "Where do I live?"})

    gone = forget.invoke({"ids": [fact["id"]]}, config=alice)
    assert gone == "Deleted 1 conversation turn(s); facts 1 -> 0."
    assert memory.get_all(user_id="alice")["results"] == []


def test_tools_accept_a_template_and_agent_scope(lives_in_berlin: Memory) -> None:
    memory = lives_in_berlin
    recall, remember, _ = memory_tools(memory, user_id="{langgraph_user_id}", agent_id="coach")
    config = {"configurable": {"langgraph_user_id": "alice"}}
    remember.invoke({"text": "I run every Sunday"}, config=config)
    assert {s["space_id"] for s in memory.spaces()} == {"alice", "coach/alice"}
    # 这个 Agent 的空间与用户自己的空间隔离
    assert "Berlin" not in recall.invoke({"query": "Where do I live?"}, config=config)


# ---------- create_agent + 中间件 ----------


def test_middleware_recalls_what_another_thread_said(memory: Memory) -> None:
    model = Recorder(responses=[AIMessage("Noted!"), AIMessage("You live in Berlin.")])
    middleware = CueMemMiddleware(memory, mode="l1")
    agent = create_agent(model, middleware=[middleware], checkpointer=InMemorySaver())

    first = thread("t1", user_id="alice")
    out = agent.invoke(ask("I live in Berlin"), first)
    assert out["cuemem"]["level"] == "none"  # 还没有记忆：不检索
    # 写回：这一问一答进了线程 t1 的会话；会话进行中，不抽取
    assert turns(memory, "alice", "t1") == [("user", "I live in Berlin"), ("assistant", "Noted!")]
    assert session_for("t1") in memory.system.open_sessions("alice")
    assert memory.get_all(user_id="alice")["results"] == []

    ended = middleware.end_thread(first, wait=True)
    assert ended["session_id"] == session_for("t1") and ended["turns"] == 2
    assert ended["consolidation"]["sessions"] == 1
    assert [f["memory"] for f in memory.get_all(user_id="alice")["results"]] == [
        "The user lives in Berlin."
    ]

    out = agent.invoke(ask("Where do I live?"), thread("t2", user_id="alice"))
    assert out["cuemem"]["level"] == "L1" and out["cuemem"]["recalled"] >= 2
    sent = model.seen[-1][-1]
    assert isinstance(sent, HumanMessage) and isinstance(sent.content, str)
    assert sent.content.startswith(f"<memory>\n{MEMORY_NOTE}\n\n")
    assert "The user lives in Berlin." in sent.content and "I live in Berlin" in sent.content
    assert sent.content.endswith("</memory>\n\nWhere do I live?")
    # 证据块只给模型看：线程里的消息不变，写回的也是原话；状态里只有召回结果，没有证据块
    assert out["messages"][0].content == "Where do I live?"
    assert set(out["cuemem"]) == {"message_id", "level", "reasons", "recalled"}
    assert "cuemem_turn" not in out
    assert turns(memory, "alice", "t2") == [
        ("user", "Where do I live?"),
        ("assistant", "You live in Berlin."),
    ]


def test_rounds_already_in_the_thread_are_not_recalled(memory: Memory) -> None:
    model = Recorder(responses=[AIMessage("Noted!"), AIMessage("A warm one.")])
    agent = create_agent(
        model, middleware=[CueMemMiddleware(memory, mode="l1")], checkpointer=InMemorySaver()
    )
    config = thread("t1", user_id="alice")
    agent.invoke(ask("I live in Berlin"), config)
    out = agent.invoke(ask("Recommend a coat"), config)
    # 唯一的一轮就在上下文里：检索时排除，没有可注入的记忆
    assert out["cuemem"]["level"] == "L1" and out["cuemem"]["recalled"] == 0
    assert model.seen[-1][-1].content == "Recommend a coat"
    assert [m.content for m in model.seen[-1]] == [
        "I live in Berlin",
        "Noted!",
        "Recommend a coat",
    ]
    assert len(turns(memory, "alice", "t1")) == 4


def test_tool_calls_reuse_the_evidence_and_only_the_answer_is_saved(
    lives_in_berlin: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = lives_in_berlin
    calls = counting_retrieve(memory, monkeypatch)
    model = Recorder(
        responses=[
            AIMessage("", tool_calls=[tool_call("weather", city="Berlin")]),
            AIMessage("Sunny, no umbrella needed."),
        ]
    )
    agent = create_agent(
        model, [weather], middleware=[CueMemMiddleware(memory, user_id="alice", mode="l1")]
    )
    out = agent.invoke(ask("Do I need an umbrella today?"), thread("t1"))
    assert isinstance(out["messages"][2], ToolMessage)
    assert calls == ["Do I need an umbrella today?"]  # 一个回合只召回一次
    first, second = (seen[0] for seen in model.seen)
    assert first.content == second.content and "lives in Berlin" in str(first.content)
    # 调用工具的中间步骤与工具结果不写回
    assert turns(memory, "alice", "t1") == [
        ("user", "Do I need an umbrella today?"),
        ("assistant", "Sunny, no umbrella needed."),
    ]


def test_middleware_can_register_the_memory_tools(memory: Memory) -> None:
    text = "I am vegetarian"
    call = {"name": "remember", "args": {"text": text}, "id": "c1", "type": "tool_call"}
    model = Recorder(responses=[AIMessage("", tool_calls=[call]), AIMessage("Got it.")])
    agent = create_agent(model, middleware=[CueMemMiddleware(memory, tools=True)])
    out = agent.invoke(ask("Remember that I am vegetarian"), thread("t1", user_id="alice"))
    assert out["messages"][2].content.startswith("Saved as ")
    notes = [s for s in memory.system.db.get_sessions("alice") if s != session_for("t1")]
    assert len(notes) == 1 and notes[0].startswith("note-")  # 与 MCP 的 remember 相同
    assert [t.text for t in memory.system.db.get_turns("alice", notes[0])] == ["I am vegetarian"]


@dataclass
class Context:
    user_id: str


def test_user_from_the_graph_context(memory: Memory) -> None:
    model = Recorder(responses=[AIMessage("Hi Bob.")])
    agent = create_agent(model, middleware=[CueMemMiddleware(memory)], context_schema=Context)
    agent.invoke(ask("Hello"), thread("t1"), context=Context(user_id="bob"))
    assert turns(memory, "bob", "t1") == [("user", "Hello"), ("assistant", "Hi Bob.")]


def test_without_a_user_memory_is_skipped(memory: Memory) -> None:
    model = Recorder(responses=[AIMessage("Hello!")])
    agent = create_agent(model, middleware=[CueMemMiddleware(memory)])
    with pytest.warns(UserWarning, match="needs a user_id"):
        out = agent.invoke(ask("Hi"), thread("t1"))
    assert out["messages"][-1].content == "Hello!" and out["cuemem"] is None
    assert model.seen[-1][-1].content == "Hi"
    assert memory.spaces() == []


def test_without_a_thread_each_exchange_is_its_own_session(memory: Memory) -> None:
    model = Recorder(responses=[AIMessage("Noted!")])
    agent = create_agent(model, middleware=[CueMemMiddleware(memory, user_id="alice")])
    agent.invoke(ask("I live in Berlin"))
    (session,) = memory.system.db.get_sessions("alice")
    assert session.startswith("add-")  # 不是进行中的会话：抽取照常登记
    memory.consolidate(user_id="alice")
    assert memory.get_all(user_id="alice")["results"][0]["memory"] == "The user lives in Berlin."


def test_async_agent(lives_in_berlin: Memory) -> None:
    memory = lives_in_berlin
    model = Recorder(responses=[AIMessage("Berlin.")])
    agent = create_agent(
        model,
        middleware=[CueMemMiddleware(memory, user_id="alice", mode="l1")],
        checkpointer=InMemorySaver(),
    )
    out = asyncio.run(agent.ainvoke(ask("Where do I live?"), thread("t1")))
    assert out["cuemem"]["level"] == "L1"
    assert "lives in Berlin" in str(model.seen[-1][-1].content)
    assert turns(memory, "alice", "t1") == [("user", "Where do I live?"), ("assistant", "Berlin.")]


def test_controller_decides_by_default(lives_in_berlin: Memory) -> None:
    model = Recorder(responses=[AIMessage("Berlin.")])
    agent = create_agent(model, middleware=[CueMemMiddleware(lives_in_berlin, user_id="alice")])
    out = agent.invoke(ask("Where do I live?"), thread("t1"))
    assert out["cuemem"]["level"] in {"L0", "L1", "L2"} and out["cuemem"]["reasons"]


# ---------- create_react_agent 的钩子 ----------


def test_react_agent_hooks(lives_in_berlin: Memory) -> None:
    from langgraph.prebuilt import create_react_agent

    memory = lives_in_berlin
    model = Recorder(responses=[AIMessage("You live in Berlin.")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # create_react_agent 在 LangGraph 1.0 起弃用
        agent = create_react_agent(
            model,
            [],
            pre_model_hook=recall_hook(memory, user_id="alice", mode="l1"),
            post_model_hook=save_hook(memory, user_id="alice"),
            checkpointer=InMemorySaver(),
        )
    config = thread("t1")
    out = agent.invoke(ask("Where do I live?"), config)
    sent = model.seen[-1][-1]
    assert str(sent.content).startswith("<memory>\n") and "lives in Berlin" in str(sent.content)
    assert out["messages"][0].content == "Where do I live?"  # llm_input_messages 不改线程的消息
    assert turns(memory, "alice", "t1") == [
        ("user", "Where do I live?"),
        ("assistant", "You live in Berlin."),
    ]
    assert end_thread(memory, config, user_id="alice", wait=True)["turns"] == 2
    assert session_for("t1") not in memory.system.open_sessions("alice")


def test_images_are_kept_and_web_images_skipped(
    memory: Memory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    png = base64.b64encode(buffer.getvalue()).decode()
    content = [
        {"type": "text", "text": "My new running shoes"},
        {"type": "image", "base64": png, "mime_type": "image/png"},
        {"type": "image", "url": "https://x.test/a.png"},  # 网络图片不下载（与记忆代理相同）
    ]
    model = Recorder(responses=[AIMessage("Nice!"), AIMessage("Red ones.")])
    agent = create_agent(
        model, middleware=[CueMemMiddleware(memory, mode="l1")], checkpointer=InMemorySaver()
    )
    config = thread("t1", user_id="alice")
    agent.invoke({"messages": [HumanMessage(content=content)]}, config)
    assert turns(memory, "alice", "t1") == [
        ("user", "My new running shoes <image>"),
        ("assistant", "Nice!"),
    ]
    (stored,) = memory.system.db.get_media("alice")
    assert Path(stored.path).exists() and not list(scratch.iterdir())
    # 带图的那一轮原样还在线程里：下一回合检索时照样排除
    out = agent.invoke(ask("Which shoes did I show you?"), config)
    assert out["cuemem"]["recalled"] == 0


def test_session_ids() -> None:
    assert session_for("3f2a-11") == "lg-3f2a-11"
    odd = session_for("user 1/thread 2")
    assert odd.startswith("lg-user_1_thread_2-") and odd != session_for("user_1/thread 2")
    assert len(session_for("x" * 200)) < 70


# ---------- 回合怎么结束都写回 ----------


class Answer(BaseModel):
    city: str


def test_structured_output_turn_is_saved(memory: Memory) -> None:
    # 模型不支持原生结构化输出（例如 DeepSeek）时 create_agent 用 ToolStrategy：回合停在工具结果上
    model = Recorder(responses=[AIMessage("", tool_calls=[tool_call("Answer", city="Berlin")])])
    agent = create_agent(
        model,
        middleware=[CueMemMiddleware(memory, user_id="alice")],
        response_format=ToolStrategy(Answer),
    )
    out = agent.invoke(ask("Which city do I live in?"), thread("t1"))
    assert out["structured_response"] == Answer(city="Berlin")
    assert isinstance(out["messages"][-1], ToolMessage)
    assert turns(memory, "alice", "t1") == [
        ("user", "Which city do I live in?"),
        ("assistant", '{"city":"Berlin"}'),
    ]


def test_return_direct_tool_turn_is_saved(memory: Memory) -> None:
    @tool(return_direct=True)
    def order_status(order: str) -> str:
        """Shipping status of an order."""
        return f"Order {order} ships tomorrow."

    model = Recorder(responses=[AIMessage("", tool_calls=[tool_call("order_status", order="42")])])
    agent = create_agent(
        model, [order_status], middleware=[CueMemMiddleware(memory, user_id="alice")]
    )
    agent.invoke(ask("Where is order 42?"), thread("t1"))
    assert turns(memory, "alice", "t1") == [
        ("user", "Where is order 42?"),
        ("assistant", "Order 42 ships tomorrow."),
    ]


def test_summarization_does_not_replace_the_user_turn(
    lives_in_berlin: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = lives_in_berlin
    calls = counting_retrieve(memory, monkeypatch)
    steps = [
        AIMessage("", tool_calls=[tool_call("weather", f"c{i}", city=city)])
        for i, city in enumerate(["Berlin", "Paris", "Rome"])
    ]
    model = Recorder(responses=[*steps, AIMessage("Sunny everywhere.")])
    summarizer = FakeMessagesListChatModel(responses=[AIMessage(f"summary {i}") for i in range(5)])
    agent = create_agent(
        model,
        [weather],
        middleware=[
            SummarizationMiddleware(summarizer, trigger=("messages", 4), keep=("messages", 2)),
            CueMemMiddleware(memory, user_id="alice", mode="l1"),
        ],
    )
    question = "Weather in Berlin, Paris and Rome?"
    out = agent.invoke(ask(question), thread("t1"))
    assert out["messages"][0].additional_kwargs.get("lc_source") == "summarization"
    assert all(m.content != question for m in out["messages"])  # 用户的话被总结掉了
    # 总结不是用户说的：不再召回一次，写回的是真正的问题
    assert calls == [question]
    assert turns(memory, "alice", "t1") == [("user", question), ("assistant", "Sunny everywhere.")]
    # 证据块仍然跟着这一回合：放在对话总结的开头
    sent = next(m for m in model.seen[-1] if m.additional_kwargs.get("lc_source"))
    assert str(sent.content).startswith("<memory>\n") and "lives in Berlin" in str(sent.content)


# ---------- 出错时不中断 ----------


def down(*args: Any, **kwargs: Any) -> Any:
    raise ConnectionError("embedding service is down")


def test_memory_tool_errors_go_back_to_the_model(
    lives_in_berlin: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = lives_in_berlin
    model = Recorder(
        responses=[
            AIMessage("", tool_calls=[tool_call("recall", query="Where do I live?")]),
            AIMessage("I can't recall that."),
        ]
    )
    agent = create_agent(model, middleware=[CueMemMiddleware(memory, tools=True)])
    with pytest.warns(UserWarning, match="needs a user_id"):
        out = agent.invoke(ask("Where do I live?"), thread("t1"))
    error = out["messages"][2]
    assert isinstance(error, ToolMessage) and error.status == "error"
    assert "needs a user_id" in str(error.content)
    assert out["messages"][-1].content == "I can't recall that."

    monkeypatch.setattr(memory.system, "retrieve", down)
    recall, _, _ = memory_tools(memory, user_id="alice")
    found = recall.invoke(tool_call("recall", query="Where do I live?"))
    assert found.status == "error" and "embedding service is down" in str(found.content)


def test_recall_and_write_failures_do_not_break_the_turn(
    lives_in_berlin: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = lives_in_berlin
    retrieve = memory.system.retrieve
    monkeypatch.setattr(memory.system, "retrieve", down)
    model = Recorder(responses=[AIMessage("No idea."), AIMessage("Still no idea.")])
    agent = create_agent(model, middleware=[CueMemMiddleware(memory, user_id="alice")])
    with pytest.warns(RuntimeWarning, match="could not recall"):
        out = agent.invoke(ask("Where do I live?"), thread("t1"))
    assert out["cuemem"]["level"] == "error" and out["messages"][-1].content == "No idea."
    assert model.seen[-1][-1].content == "Where do I live?"
    assert turns(memory, "alice", "t1") == [("user", "Where do I live?"), ("assistant", "No idea.")]

    monkeypatch.setattr(memory.system, "retrieve", retrieve)
    monkeypatch.setattr(memory, "add", down)
    with pytest.warns(RuntimeWarning, match="could not save"):
        out = agent.invoke(ask("Where do I live?"), thread("t2"))
    assert out["messages"][-1].content == "Still no idea."


# ---------- 身份、图片与遗忘 ----------


@dataclass
class ServerUser:
    """LangGraph Server 放进 configurable["langgraph_auth_user"] 的登录用户（BaseUser）。"""

    identity: str
    is_authenticated: bool = True


def test_the_authenticated_server_user_wins(memory: Memory) -> None:
    model = Recorder(responses=[AIMessage("Hi.")])
    agent = create_agent(model, middleware=[CueMemMiddleware(memory)])
    alice = ServerUser("alice")
    agent.invoke(ask("Hello"), thread("t1", langgraph_auth_user=alice))
    assert turns(memory, "alice", "t1") == [("user", "Hello"), ("assistant", "Hi.")]
    # 客户端在 configurable 里冒充别人：不碰记忆
    with pytest.warns(UserWarning, match="does not match the authenticated"):
        out = agent.invoke(ask("Hello"), thread("t2", user_id="bob", langgraph_auth_user=alice))
    assert out["cuemem"] is None and memory.system.db.get_sessions("bob") == {}
    recall, _, _ = memory_tools(memory)
    mismatch = {"configurable": {"user_id": "bob", "langgraph_auth_user": alice}}
    assert "does not match" in recall.invoke({"query": "Hello"}, config=mismatch)
    # 客户端传来的字典不是登录用户
    agent.invoke(ask("Hi"), thread("t3", user_id="carol", langgraph_auth_user={"identity": "x"}))
    assert turns(memory, "carol", "t3") == [("user", "Hi"), ("assistant", "Hi.")]
    # 写成模板表示开发者信任客户端给的值
    trusting = create_agent(model, middleware=[CueMemMiddleware(memory, user_id="{user_id}")])
    trusting.invoke(ask("Hey"), thread("t4", user_id="dave", langgraph_auth_user=alice))
    assert turns(memory, "dave", "t4") == [("user", "Hey"), ("assistant", "Hi.")]


def test_local_images_are_never_read(memory: Memory, tmp_path: Path) -> None:
    secret = tmp_path / "server_private.png"
    Image.new("RGB", (8, 8), "blue").save(secret)
    content = [
        {"type": "text", "text": "Look at this"},
        {"type": "image_url", "image_url": {"url": f"file://{secret}"}},
        {"type": "image_url", "image_url": {"url": str(secret)}},
        {"type": "image", "url": f"file://{secret}"},
        {"type": "image", "file_id": "file-123"},  # 只有引用、没有内容：不写，也不报错
    ]
    model = Recorder(responses=[AIMessage("Nice.")])
    agent = create_agent(model, middleware=[CueMemMiddleware(memory, user_id="mallory")])
    agent.invoke({"messages": [HumanMessage(content=content)]}, thread("t1"))
    assert turns(memory, "mallory", "t1") == [("user", "Look at this"), ("assistant", "Nice.")]
    assert memory.system.db.get_media("mallory") == []


@pytest.mark.parametrize("run_async", [False, True])
def test_forget_drops_the_evidence_of_the_running_turn(
    lives_in_berlin: Memory, run_async: bool
) -> None:
    memory = lives_in_berlin
    (fact,) = memory.get_all(user_id="alice")["results"]
    model = Recorder(
        responses=[
            AIMessage("", tool_calls=[tool_call("forget", ids=[fact["id"]])]),
            AIMessage("Done, I forgot it."),
        ]
    )
    cuemem = CueMemMiddleware(memory, user_id="alice", mode="l1", tools=True)
    agent = create_agent(model, middleware=[cuemem], checkpointer=InMemorySaver())
    config = thread("t1")
    request = ask("Please forget where I live")
    if run_async:
        out = asyncio.run(agent.ainvoke(request, config))
    else:
        out = agent.invoke(request, config)
    assert "lives in Berlin" in str(model.seen[0][-1].content)
    assert out["messages"][2].content == "Deleted 1 conversation turn(s); facts 1 -> 0."
    # 遗忘之后的模型调用重新召回：看不到刚删掉的记忆，状态里的召回结果也跟着更新
    assert "Berlin" not in " ".join(str(m.content) for m in model.seen[1])
    assert out["cuemem"]["recalled"] == 0
    # 证据块不进状态，也就不进检查点
    history = list(agent.get_state_history(config))
    assert history and not any("Berlin" in str(snapshot.values) for snapshot in history)


# ---------- 会话的生命周期 ----------


def test_idle_threads_are_handed_to_extraction(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [1000.0]
    monkeypatch.setattr(lg, "_clock", lambda: now[0])
    model = Recorder(responses=[AIMessage("Noted!"), AIMessage("Hi.")])
    cuemem = CueMemMiddleware(memory, idle_minutes=30)
    agent = create_agent(model, middleware=[cuemem], checkpointer=InMemorySaver())
    agent.invoke(ask("I live in Berlin"), thread("t1", user_id="alice"))
    assert session_for("t1") in memory.system.open_sessions("alice")
    now[0] += 31 * 60  # 半个多小时后另一个线程来了：t1 空闲太久，交给后台抽取
    agent.invoke(ask("Hello"), thread("t2", user_id="alice"))
    assert memory.system.open_sessions("alice") == {session_for("t2")}
    memory.wait()
    assert facts(memory) == ["The user lives in Berlin."]
    # 应用退出前：进行中的线程都交出去
    assert cuemem.close() == 1 and memory.system.open_sessions("alice") == set()


def test_threads_left_open_by_an_earlier_process(n3_settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = [ScriptedModel(where_i_live)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as before:
        middleware = [CueMemMiddleware(before, user_id="alice")]
        agent = create_agent(Recorder(responses=[AIMessage("Noted!")]), middleware=middleware)
        agent.invoke(ask("I live in Berlin"), thread("t1"))
    # 进程退出时没有 end_thread 也没有 close：下一个进程第一次碰到这个用户时交给抽取
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as after:
        middleware = [CueMemMiddleware(after, user_id="alice")]
        agent = create_agent(Recorder(responses=[AIMessage("Hi.")]), middleware=middleware)
        agent.invoke(ask("Hello"), thread("t2"))
        after.wait()
        assert facts(after) == ["The user lives in Berlin."]
        assert after.system.open_sessions("alice") == {session_for("t2")}


def test_a_thread_can_go_on_after_it_ended(memory: Memory, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = [ScriptedModel(where_i_live, reconcile=moved)]
    model = Recorder(responses=[AIMessage("Noted!"), AIMessage("Congrats on the move!")])
    cuemem = CueMemMiddleware(memory, user_id="alice", mode="l1")
    agent = create_agent(model, middleware=[cuemem], checkpointer=InMemorySaver())
    config = thread("t1")
    agent.invoke(ask("I live in Berlin"), config)
    assert cuemem.end_thread(config, wait=True)["turns"] == 2
    agent.invoke(ask("I moved to Munich"), config)
    assert session_for("t1") in memory.system.open_sessions("alice")
    ended = cuemem.end_thread(config, wait=True)  # 整段重新抽取：新的取代旧的，不重复
    assert ended["turns"] == 4 and ended["consolidation"]["sessions"] == 1
    assert facts(memory) == ["The user lives in Munich."]
    everything = memory.get_all(user_id="alice", include_superseded=True)["results"]
    assert sorted(f["memory"] for f in everything) == [
        "The user lives in Berlin.",
        "The user lives in Munich.",
    ]


def test_a_resumed_thread_ends_like_a_replay(memory: Memory, fake_api: FakeAPI) -> None:
    """线程结束之后又接着聊：再次结束时整段重新抽取，之前那次抽取的说法不再留着，事实与从原始
    日志重建的一样（D-045）。"""
    fake_api.chat_replies = [ScriptedModel(growing_extraction)]
    model = Recorder(responses=[AIMessage("Nice!"), AIMessage("Cool.")])
    cuemem = CueMemMiddleware(memory, user_id="alice", mode="l1")
    agent = create_agent(model, middleware=[cuemem], checkpointer=InMemorySaver())
    config = thread("t1")
    agent.invoke(ask("I like hiking"), config)
    cuemem.end_thread(config, wait=True)
    assert facts(memory) == ["The user likes hiking."]
    agent.invoke(ask("Mostly in the Alps"), config)
    cuemem.end_thread(config, wait=True)
    grown = sorted(facts(memory))
    assert grown == ["The user enjoys hiking in the Alps.", "The user said: Mostly in the Alps"]
    memory.consolidate(user_id="alice", rebuild=True)
    assert sorted(facts(memory)) == grown


def test_async_memory_and_async_ending(lives_in_berlin: Memory) -> None:
    wrapped = AsyncMemory.__new__(AsyncMemory)  # 与 Memory 共用同一个记忆系统
    wrapped.sync = lives_in_berlin
    model = Recorder(responses=[AIMessage("Berlin.")])
    cuemem = CueMemMiddleware(wrapped, user_id="alice", mode="l1", tools=True)
    agent = create_agent(model, middleware=[cuemem], checkpointer=InMemorySaver())
    config = thread("t1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 没有 "coroutine was never awaited"
        asyncio.run(agent.ainvoke(ask("Where do I live?"), config))
        found = asyncio.run(cuemem.tools[0].ainvoke({"query": "Where do I live?"}))
    assert "lives in Berlin" in found
    assert turns(lives_in_berlin, "alice", "t1") == [
        ("user", "Where do I live?"),
        ("assistant", "Berlin."),
    ]
    assert asyncio.run(cuemem.aend_thread(config, wait=True))["turns"] == 2
    assert asyncio.run(aend_thread(wrapped, thread("t9"), user_id="alice"))["turns"] == 0


def test_react_agent_hooks_async(lives_in_berlin: Memory) -> None:
    from langgraph.prebuilt import create_react_agent

    memory = lives_in_berlin
    model = Recorder(responses=[AIMessage("You live in Berlin.")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = create_react_agent(
            model,
            [],
            pre_model_hook=recall_hook(memory, user_id="alice", mode="l1"),
            post_model_hook=save_hook(memory, user_id="alice"),
        )
    asyncio.run(agent.ainvoke(ask("Where do I live?"), thread("t1")))
    assert "lives in Berlin" in str(model.seen[-1][-1].content)
    assert turns(memory, "alice", "t1") == [
        ("user", "Where do I live?"),
        ("assistant", "You live in Berlin."),
    ]


def test_messages_without_ids_are_kept_per_thread(memory: Memory) -> None:
    # 自己搭的图没用 add_messages 时消息没有 ID：不按内容去重，两个线程各写各的
    threads = ThreadMemory(memory, user_id="alice")
    exchange = [HumanMessage("yes"), AIMessage("OK.")]
    assert threads.save(exchange, thread("a")) == 2
    assert threads.save(exchange, thread("b")) == 2
    assert turns(memory, "alice", "b") == [("user", "yes"), ("assistant", "OK.")]


def test_concurrent_saves_keep_the_turn_order(memory: Memory) -> None:
    threads = ThreadMemory(memory, user_id="alice")

    def one(i: int) -> int:
        exchange = [HumanMessage(f"q{i}", id=f"h{i}"), AIMessage(f"a{i}", id=f"a{i}")]
        return threads.save(exchange, thread("t1"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(one, range(16))) == 32
    stored = memory.system.db.get_turns("alice", session_for("t1"))
    assert len({t.idx for t in stored}) == 32
    assert threads.save([HumanMessage("q0", id="h0"), AIMessage("a0", id="a0")], thread("t1")) == 0
