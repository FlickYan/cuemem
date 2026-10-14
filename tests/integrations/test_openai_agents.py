"""OpenAI Agents SDK 集成（N12）：记忆工具、调用模型前的自动回忆、写进长期记忆的会话。

全部离线：Agent 用 SDK 自带的 agents.testing.ScriptedModel，CueMem 用假的模型接口与哈希向量。
"""

from __future__ import annotations

import asyncio
import base64
import io
import itertools
from collections.abc import Iterator
from typing import Any, cast

import pytest

pytest.importorskip("agents")

from agents import (
    Agent,
    RunConfig,
    Runner,
    Session,
    SessionSettings,
    SQLiteSession,
    TResponseInputItem,
)
from agents.testing import ModelCall, assistant_message, function_call
from agents.testing import ScriptedModel as AgentScript
from fakes import FakeAPI
from PIL import Image
from scripted import ScriptedModel, echo_extraction

from cuemem import AsyncMemory, Memory
from cuemem.api.mcp import build_server
from cuemem.api.proxy import MEMORY_NOTE
from cuemem.config import Settings
from cuemem.integrations.openai_agents import (
    CueMemSession,
    memory_input_filter,
    memory_tools,
)

_ids = itertools.count()


def reply(text: str) -> list[Any]:
    """一步模型输出：一条助手消息（每条消息的 ID 不同，SDK 会按 ID 去重）。"""
    return [assistant_message(text, item_id=f"msg-{next(_ids)}")]


def call(name: str, arguments: dict[str, Any]) -> list[Any]:
    n = next(_ids)
    return [function_call(name, arguments, call_id=f"call-{n}")]


def last_user_text(model_call: ModelCall) -> str:
    """一次模型调用里最新一句用户消息的文字。"""
    for item in reversed(model_call.input):
        if isinstance(item, dict) and item.get("role") == "user":
            content = item["content"]
            if isinstance(content, str):
                return content
            return "".join(part.get("text", "") for part in content)
    raise AssertionError("no user message")


def tool_outputs(model_call: ModelCall) -> list[str]:
    return [
        str(item["output"])
        for item in model_call.input
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]


def run(
    agent: Agent[Any],
    text: str,
    *,
    session: Session | None = None,
    config: RunConfig | None = None,
) -> Any:
    config = config or RunConfig()
    config.tracing_disabled = True  # 不往 OpenAI 发 trace
    return asyncio.run(Runner.run(agent, text, session=session, run_config=config))


@pytest.fixture
def memory(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[Memory]:
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        yield m


def turns(memory: Memory, session_id: str, space: str = "alice") -> list[tuple[str, str]]:
    return [(t.role, t.text) for t in memory.system.db.get_turns(space, session_id)]


def facts(memory: Memory) -> list[str]:
    return [f["memory"] for f in memory.get_all(user_id="alice")["results"]]


# ---------- 工具 ----------


def test_tools_share_names_and_descriptions_with_mcp(memory: Memory) -> None:
    server = build_server(cast(Any, None), "alice")
    mcp = {
        t.name: " ".join((t.description or "").split()) for t in asyncio.run(server.list_tools())
    }
    tools = memory_tools(memory, user_id="alice")
    assert [t.name for t in tools] == ["recall", "remember", "forget"]
    for tool in tools:
        assert tool.description == mcp[tool.name]
    assert list(tools[0].params_json_schema["properties"]) == ["query", "deep"]
    with pytest.raises(ValueError, match="user_id"):
        memory_tools(memory)


def test_tools_recall_remember_and_forget_in_a_run(memory: Memory) -> None:
    memory.add("I live in Berlin", user_id="alice", wait=True)
    (berlin,) = memory.get_all(user_id="alice")["results"]
    script = AgentScript(
        [
            call("recall", {"query": "Where do I live?", "deep": False}),
            call("remember", {"text": "I prefer window seats"}),
            call("forget", {"ids": [berlin["id"]]}),
            reply("Done."),
        ]
    )
    agent = Agent(name="a", model=script, tools=memory_tools(memory, user_id="alice"))
    result = run(agent, "hi")
    assert result.final_output == "Done."

    recalled = tool_outputs(script.calls[1])[0]
    assert recalled.startswith("Memory controller: L")
    assert f"[{berlin['id']}]" in recalled and "I live in Berlin" in recalled
    saved = tool_outputs(script.calls[2])[1]
    assert saved.startswith("Saved as ")
    notes = [t for t in memory.system.db.get_turns("alice") if t.text == "I prefer window seats"]
    assert len(notes) == 1 and notes[0].session_id.startswith("note-")
    deleted = tool_outputs(script.calls[3])[2]
    assert deleted.startswith("Deleted 1 conversation turn(s)")
    assert "I live in Berlin" not in facts(memory)
    assert not [t for t in memory.system.db.get_turns("alice") if "Berlin" in t.text]


# ---------- 自动回忆 ----------


def test_input_filter_injects_evidence_without_touching_history(memory: Memory) -> None:
    memory.add("I live in Berlin", user_id="alice", wait=True)
    script = AgentScript([reply("You live in Berlin.")])
    recall = memory_input_filter(memory, user_id="alice", mode="l1")
    session = SQLiteSession("plain")
    run(
        Agent(name="a", model=script),
        "Where do I live?",
        session=session,
        config=RunConfig(call_model_input_filter=recall),
    )
    sent = last_user_text(script.calls[0])
    assert sent.startswith(f"<memory>\n{MEMORY_NOTE}\n\n")
    assert sent.endswith("</memory>\n\nWhere do I live?")
    assert "I live in Berlin" in sent and "Retrieved conversation excerpts:" in sent
    assert recall.last is not None and recall.last.level == "L1" and recall.last.recalled >= 2
    # 存下的历史是原话：证据块只在这一次发给模型的副本里
    stored = asyncio.run(session.get_items())
    assert stored[0] == {"role": "user", "content": "Where do I live?"}


def test_l0_injects_only_the_profile_or_nothing(memory: Memory) -> None:
    memory.add("I live in Berlin", user_id="alice", wait=True)
    assert memory.profile(user_id="alice")
    script = AgentScript([reply("Hi!"), reply("Hi!"), reply("Hi!")])
    agent = Agent(name="a", model=script)

    profile_only = memory_input_filter(memory, user_id="alice", mode="l0")
    run(agent, "Hello there", config=RunConfig(call_model_input_filter=profile_only))
    sent = last_user_text(script.calls[0])
    assert "About the user (core memory, current facts):" in sent
    assert "Retrieved conversation excerpts" not in sent and "Facts" not in sent
    assert profile_only.last is not None and profile_only.last.level == "L0"

    # 只记了日志、还没有画像的用户：L0 什么都不加
    memory.add("I live in Rome", user_id="bob", infer=False)
    nothing = memory_input_filter(memory, user_id="bob", mode="l0")
    run(agent, "Hello there", config=RunConfig(call_model_input_filter=nothing))
    assert last_user_text(script.calls[1]) == "Hello there"

    # 没有任何记忆的用户：不检索
    empty = memory_input_filter(memory, user_id="carol")
    run(agent, "Hello there", config=RunConfig(call_model_input_filter=empty))
    assert last_user_text(script.calls[2]) == "Hello there"
    assert empty.last is not None and empty.last.level == "none"


def test_auto_mode_runs_the_controller(memory: Memory) -> None:
    memory.add("I live in Berlin", user_id="alice", wait=True)
    script = AgentScript([reply("ok")])
    recall = memory_input_filter(memory, user_id="alice")
    run(
        Agent(name="a", model=script),
        "Where do I live?",
        config=RunConfig(call_model_input_filter=recall),
    )
    assert recall.last is not None
    assert recall.last.level in {"L0", "L1", "L2"} and recall.last.reasons


def test_follow_up_calls_reuse_the_evidence_until_something_is_forgotten(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory.add("I live in Berlin", user_id="alice", wait=True)
    (berlin,) = memory.get_all(user_id="alice")["results"]
    retrieve = memory.system.retrieve
    calls: list[str] = []

    def counted(space: str, query: str, **kwargs: Any) -> Any:
        calls.append(query)
        return retrieve(space, query, **kwargs)

    monkeypatch.setattr(memory.system, "retrieve", counted)
    script = AgentScript(
        [
            call("remember", {"text": "I like jazz"}),
            call("forget", {"ids": [berlin["id"]]}),
            reply("Done."),
        ]
    )
    agent = Agent(name="a", model=script, tools=memory_tools(memory, user_id="alice"))
    recall = memory_input_filter(memory, user_id="alice", mode="l1")
    run(agent, "Where do I live?", config=RunConfig(call_model_input_filter=recall))
    first, second, third = (last_user_text(c) for c in script.calls)
    # 工具调用之后的再次调用：同一份证据，不重复检索
    assert "I live in Berlin" in first and second == first
    # 遗忘之后重新检索：被删的不再出现
    assert "Berlin" not in third
    assert calls == ["Where do I live?", "Where do I live?"]


def test_recall_errors_do_not_block_the_run(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory.add("I live in Berlin", user_id="alice", wait=True)

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(memory.system, "retrieve", broken)
    script = AgentScript([reply("ok")])
    recall = memory_input_filter(memory, user_id="alice", mode="l1")
    result = run(
        Agent(name="a", model=script),
        "Where do I live?",
        config=RunConfig(call_model_input_filter=recall),
    )
    assert result.final_output == "ok"
    assert last_user_text(script.calls[0]) == "Where do I live?"


# ---------- 会话 ----------


def test_session_mirrors_messages_and_a_later_session_recalls_them(memory: Memory) -> None:
    script = AgentScript([reply("Noted!"), reply("You live in Berlin.")])
    agent = Agent(name="a", model=script)
    recall = memory_input_filter(memory, user_id="alice", mode="l1")
    config = RunConfig(call_model_input_filter=recall)

    monday = CueMemSession("monday", memory, user_id="alice")
    assert isinstance(monday, Session)
    run(agent, "I live in Berlin", session=monday, config=config)
    assert turns(memory, "monday") == [("user", "I live in Berlin"), ("assistant", "Noted!")]
    # 进行中：立即可检索，但还没有抽取事实
    assert "monday" in memory.system.open_sessions("alice")
    assert facts(memory) == []
    assert memory.search("Berlin", user_id="alice", mode="l1")["results"]

    report = asyncio.run(monday.end(wait=True))
    assert report is not None and report["turns"] == 2
    assert report["consolidation"]["sessions"] == 1
    assert "monday" not in memory.system.open_sessions("alice")
    assert facts(memory) == ["I live in Berlin"]
    assert len(asyncio.run(monday.get_items())) == 2  # 短期历史原样保留

    tuesday = CueMemSession("tuesday", memory, user_id="alice")
    run(agent, "Where do I live?", session=tuesday, config=config)
    sent = last_user_text(script.calls[1])
    assert "I live in Berlin" in sent
    # 这句问话已经写进了进行中的会话，但它就在输入里：不当作记忆再给一遍
    assert recall.last is not None and "Where do I live" not in recall.last.block
    assert asyncio.run(CueMemSession("empty", memory, user_id="alice").end()) is None


def test_rounds_already_in_context_are_not_recalled(memory: Memory) -> None:
    script = AgentScript([reply("Noted!"), reply("Berlin.")])
    agent = Agent(name="a", model=script)
    recall = memory_input_filter(memory, user_id="alice", mode="l1")
    config = RunConfig(call_model_input_filter=recall)
    session = CueMemSession("monday", memory, user_id="alice")
    run(agent, "I live in Berlin", session=session, config=config)
    run(agent, "Where do I live?", session=session, config=config)
    # 唯一的记忆就是这段对话本身，已经原样在历史里：什么都不加
    assert last_user_text(script.calls[1]) == "Where do I live?"
    assert recall.last is not None and recall.last.level == "L1" and recall.last.block == ""


def test_pop_item_forgets_the_message(memory: Memory) -> None:
    script = AgentScript([reply("Noted!")])
    session = CueMemSession("monday", memory, user_id="alice")
    run(Agent(name="a", model=script), "I live in Berlin", session=session)

    popped = asyncio.run(session.pop_item())
    assert popped is not None and popped.get("role") == "assistant"
    assert turns(memory, "monday") == [("user", "I live in Berlin")]
    # 另一个对象（例如进程重启之后）没有写入记录：按内容找到同一句
    again = CueMemSession("monday", memory, user_id="alice", inner=session.inner)
    popped = asyncio.run(again.pop_item())
    assert popped == {"role": "user", "content": "I live in Berlin"}
    assert turns(memory, "monday") == []
    assert asyncio.run(again.pop_item()) is None


def test_clear_session_forgets_the_whole_conversation(memory: Memory) -> None:
    script = AgentScript([reply("Noted!"), reply("Hi!")])
    agent = Agent(name="a", model=script)
    session = CueMemSession("monday", memory, user_id="alice")
    run(agent, "I live in Berlin", session=session)
    asyncio.run(session.end(wait=True))
    assert facts(memory) == ["I live in Berlin"]

    asyncio.run(session.clear_session())
    assert asyncio.run(session.get_items()) == []
    assert turns(memory, "monday") == []
    assert facts(memory) == []  # 派生记忆按剩下的日志重建
    # 清空之后接着用：同一个会话从头开始
    run(agent, "Hello", session=session)
    assert [t.idx for t in memory.system.db.get_turns("alice", "monday")] == [0, 1]


def test_forgetting_prunes_whole_rounds_from_the_short_term_history(memory: Memory) -> None:
    script = AgentScript([reply("Noted!"), reply("Cool."), reply("Sure.")])
    agent = Agent(name="a", model=script)
    # 内层会话默认只读最近两条：去掉被遗忘的轮时不能丢了更早的历史
    inner = SQLiteSession("monday", session_settings=SessionSettings(limit=2))
    session = CueMemSession("monday", memory, user_id="alice", inner=inner)
    for text in ("I live in Berlin", "I like jazz", "I have a cat"):
        run(agent, text, session=session)

    (jazz,) = [t for t in memory.system.db.get_turns("alice", "monday") if t.text == "I like jazz"]
    memory.system.forget_memories("alice", turn_ids=[jazz.id])  # 例如 forget 工具或 Memory.delete
    assert len(asyncio.run(session.get_items())) == 2
    everything = asyncio.run(inner.get_items(100))
    assert [(i.get("role"), _text(i)) for i in everything] == [
        ("user", "I live in Berlin"),
        ("assistant", "Noted!"),
        ("user", "I have a cat"),
        ("assistant", "Sure."),
    ]


def test_only_text_and_inline_images_are_stored(memory: Memory) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    items = cast(
        list[TResponseInputItem],
        [
            {"role": "system", "content": "Be nice."},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "My new shoes"},
                    {"type": "input_image", "image_url": data_url, "detail": "auto"},
                    {"type": "input_image", "image_url": "https://x.test/a.png", "detail": "auto"},
                ],
            },
            {"type": "function_call", "call_id": "c1", "name": "recall", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "Memory controller: L1"},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"role": "assistant", "content": [{"type": "refusal", "refusal": "No."}]},
        ],
    )
    session = CueMemSession("monday", memory, user_id="alice")
    asyncio.run(session.add_items(items))
    assert turns(memory, "monday") == [("user", "My new shoes <image>")]
    assert len(memory.system.db.get_media("alice")) == 1
    assert len(asyncio.run(session.get_items())) == len(items)


def test_async_with_ends_the_conversation(memory: Memory) -> None:
    script = AgentScript([reply("Noted!")])
    agent = Agent(name="a", model=script)
    config = RunConfig(tracing_disabled=True)

    async def conversation() -> None:
        async with CueMemSession("monday", memory, user_id="alice") as session:
            await Runner.run(agent, "I live in Berlin", session=session, run_config=config)
            assert "monday" in memory.system.open_sessions("alice")
        assert "monday" not in memory.system.open_sessions("alice")

    asyncio.run(conversation())
    memory.wait()  # 退出时交给后台抽取
    assert facts(memory) == ["I live in Berlin"]


def test_storage_errors_do_not_break_the_conversation(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("disk full")

    monkeypatch.setattr(memory, "add", broken)
    session = CueMemSession("monday", memory, user_id="alice")
    result = run(Agent(name="a", model=AgentScript([reply("Noted!")])), "Hi", session=session)
    assert result.final_output == "Noted!"
    assert len(asyncio.run(session.get_items())) == 2  # 短期历史照常保存


def test_async_memory_is_accepted(memory: Memory) -> None:
    wrapped = AsyncMemory.__new__(AsyncMemory)
    wrapped.sync = memory
    assert CueMemSession("s", wrapped, user_id="alice").memory is memory
    assert memory_input_filter(wrapped, user_id="alice").memory is memory


def _text(item: Any) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)
