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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain.agents")

from fakes import FakeAPI
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image
from pydantic import Field
from scripted import ScriptedModel
from test_sdk import where_i_live

from cuemem import Memory
from cuemem.api.mcp import build_server
from cuemem.api.proxy import MEMORY_NOTE
from cuemem.config import Settings
from cuemem.integrations.langgraph import (
    CueMemMiddleware,
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
    # 别的用户什么都看不到；没有用户时报错，不会写进某个默认空间
    bob = recall.invoke({"query": "Where do I live?"}, config={"configurable": {"user_id": "bob"}})
    assert "No memories were retrieved" in bob
    with pytest.raises(ValueError, match="user_id"):
        recall.invoke({"query": "Where do I live?"})

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
    # 证据块只给模型看：线程里的消息不变，写回的也是原话
    assert out["messages"][0].content == "Where do I live?"
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
    assert out["cuemem"]["level"] == "L1" and out["cuemem"]["block"] == ""
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
    calls: list[str] = []
    retrieve = memory.system.retrieve

    def counting(space: str, question: str, **kwargs: Any) -> Any:
        calls.append(question)
        return retrieve(space, question, **kwargs)

    monkeypatch.setattr(memory.system, "retrieve", counting)

    @tool
    def weather(city: str) -> str:
        """Current weather in a city."""
        return f"Sunny in {city}."

    call = {"name": "weather", "args": {"city": "Berlin"}, "id": "c1", "type": "tool_call"}
    model = Recorder(
        responses=[AIMessage("", tool_calls=[call]), AIMessage("Sunny, no umbrella needed.")]
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
    assert out["cuemem"]["block"] == ""


def test_session_ids() -> None:
    assert session_for("3f2a-11") == "lg-3f2a-11"
    odd = session_for("user 1/thread 2")
    assert odd.startswith("lg-user_1_thread_2-") and odd != session_for("user_1/thread 2")
    assert len(session_for("x" * 200)) < 70
