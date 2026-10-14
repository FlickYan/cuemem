"""LangGraph / LangChain 集成（N12）：记忆工具、控制器先行的自动召回、按线程写回对话。

    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from cuemem import Memory
    from cuemem.integrations.langgraph import CueMemMiddleware, end_thread

    memory = Memory(preset="deepseek")
    agent = create_agent(llm, tools, middleware=[CueMemMiddleware(memory)],
                         checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1", "user_id": "alice"}}
    agent.invoke({"messages": [{"role": "user", "content": "I moved to Munich"}]}, config)
    end_thread(memory, config)        # 这段对话结束：整段交给抽取

三块，可以分开用：

- memory_tools：recall / remember / forget 三个工具，名字与说明和 MCP Server 一致（api/mcp.py）；
- 自动召回：每个用户回合先过检索控制器（L0 不检索、只给画像；L1 / L2），取回的记忆排成与记忆代理
  相同的证据块，放在最后一句用户消息的前面，只给这次模型调用看，不写进线程的消息；
  同一回合里工具调用之后的模型调用沿用这个证据块，不重复检索，前缀缓存照常命中；
- 写回：最终回答之后把这一问一答在线写入（不调用 LLM，立即可检索）。一个 LangGraph 线程
  （thread_id）对应一个 CueMem 会话，进行中不抽取；end_thread 时整段抽取
  （与记忆代理、Agent 相同，D-045）。

接法：推荐 LangChain 1.x 的 create_agent + CueMemMiddleware（before_model 选档召回、
wrap_model_call 注入、after_model 写回）。langgraph.prebuilt.create_react_agent（1.0 起弃用）
用 recall_hook / save_hook 作 pre_model_hook / post_model_hook；自己搭的 StateGraph 也可以
把这两个钩子当节点用。

身份：user_id / agent_id 构造时固定；不填时运行时从 config["configurable"] 取，再看 context
（与 LangMem 的命名空间模板相同的做法）；写成 "{langgraph_user_id}" 这样的模板时从指定的键取。
都没有时不碰记忆（与记忆代理相同），并给出警告。

不提供 BaseStore 适配：CueMem 不是键值存储。写进去的是对话，事实由抽取得出、ID 由系统给，
更新是取代旧版本而不是覆盖，put(key, value) 之后 get(key) 拿不回原样的值。
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import threading
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Annotated, Any, NotRequired, cast

# 框架是可选依赖（uv sync --extra langgraph）；没装时 mypy 把它们当作 Any，不报找不到模块
from langchain.agents.middleware import (  # type: ignore[import-not-found, unused-ignore]
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain.agents.middleware.types import OmitFromSchema  # type: ignore[import-not-found, unused-ignore]
from langchain_core.messages import (  # type: ignore[import-not-found, unused-ignore]
    AIMessage,
    BaseMessage,
    HumanMessage,
    convert_to_openai_messages,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda  # type: ignore[import-not-found, unused-ignore]
from langchain_core.tools import BaseTool, StructuredTool  # type: ignore[import-not-found, unused-ignore]
from langgraph.config import get_config  # type: ignore[import-not-found, unused-ignore]
from langgraph.runtime import Runtime, get_runtime  # type: ignore[import-not-found, unused-ignore]

from cuemem.api.mcp import format_recall
from cuemem.api.proxy import MEMORY_NOTE
from cuemem.core.trace import record_event
from cuemem.memory import RecallMode
from cuemem.read.controller import english_reason
from cuemem.read.reader import memory_blocks
from cuemem.sdk import Memory, space_for

__all__ = [
    "CueMemMiddleware",
    "CueMemState",
    "Evidence",
    "ThreadMemory",
    "end_thread",
    "memory_tools",
    "recall_hook",
    "save_hook",
    "session_for",
]

# 工具说明与 MCP Server（api/mcp.py）的工具一字不差
RECALL_DESCRIPTION = (
    "Recall memories relevant to the query. The memory controller decides whether a fast "
    "search is enough or a deeper one is needed; deep=true forces the deep search. Returns the "
    "user's profile, facts with validity, and conversation excerpts, each with an id and a date."
)
REMEMBER_DESCRIPTION = (
    "Save something worth remembering about the user (a preference, a fact, a plan). It can be "
    "recalled right away and is turned into structured facts in the background."
)
FORGET_DESCRIPTION = (
    "Forget memories by the ids that recall returned (fact ids or excerpt ids). Deletes the "
    "conversation they came from and rebuilds everything derived from it."
)

_SAFE = re.compile(r"[^A-Za-z0-9_.:-]")
_TEMPLATE = re.compile(r"^\{(\w+)\}$")
_KEEP = 4096  # 进程内记住的已写入消息与已召回回合数
_OUTPUT_ONLY = OmitFromSchema(input=True, output=False)


@dataclass
class Evidence:
    """一个用户回合的召回结果：控制器选的档与理由、注入的证据块。"""

    message_id: str  # 证据块属于哪一句用户消息
    level: str  # L0 / L1 / L2；"none"：没有检索（还没有记忆，或这句话没有文字）
    reasons: list[str] = field(default_factory=list)
    recalled: int = 0  # 注入的记忆条数（画像、事实、对话片段）
    block: str = ""  # 证据块（不含 <memory> 标签）；空表示什么都不加


class CueMemState(AgentState[Any]):
    """中间件加到 Agent 状态里的一项：这一回合召回了什么（输出里看得到，输入里不用给）。"""

    # 没装可选依赖时 mypy 看不出 AgentState 是 TypedDict，NotRequired 会报错
    cuemem: NotRequired[Annotated[dict[str, Any] | None, _OUTPUT_ONLY]]  # type: ignore[valid-type, unused-ignore]


def session_for(thread_id: str) -> str:
    """LangGraph 线程 → CueMem 会话 ID：同一个线程总是同一个会话。"""
    safe = _SAFE.sub("_", thread_id)
    if safe != thread_id or len(safe) > 64:  # 改过字符或太长：加上哈希，避免不同线程撞在一起
        safe = f"{safe[:48]}-{hashlib.sha1(thread_id.encode()).hexdigest()[:8]}"
    return f"lg-{safe}"


class ThreadMemory:
    """LangGraph 线程与 CueMem 会话的对应：召回、写回、结束线程。中间件、钩子与工具共用；
    自己搭图时也可以直接调用。线程安全：同一会话的写入按顺序进行。"""

    def __init__(
        self,
        memory: Memory,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: RecallMode = "auto",
    ) -> None:
        self.memory = memory
        self.user_id = user_id
        self.agent_id = agent_id
        self.mode: RecallMode = mode
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._saved: OrderedDict[str, None] = OrderedDict()  # 写过的消息（空间 + 消息 ID）
        self._recalled: OrderedDict[str, Evidence] = OrderedDict()  # 钩子用：每回合只召回一次

    # ---------- 身份 ----------

    def identity(
        self, config: RunnableConfig | None, context: Any = None
    ) -> tuple[str | None, str | None]:
        """（user_id, agent_id）：构造时固定的，或者从 config["configurable"] / context 里取。"""
        return (
            _resolve(self.user_id, "user_id", config, context),
            _resolve(self.agent_id, "agent_id", config, context),
        )

    def space(self, config: RunnableConfig | None, context: Any = None) -> str:
        """记忆空间；没有身份时报错（工具与 end_thread 用）。"""
        user, agent = self.identity(config, context)
        if not (user or agent):
            raise ValueError(_NO_IDENTITY)
        return space_for(user, agent)

    # ---------- 召回 ----------

    def recall(
        self, messages: Sequence[BaseMessage], config: RunnableConfig | None, context: Any = None
    ) -> Evidence | None:
        """对最后一句用户消息跑检索控制器，取回的记忆排成与记忆代理相同的证据块。

        当前线程里已经写进记忆、原样还在消息里的轮，检索时排除（名额留给上下文里没有的）。
        没有用户消息、或没有身份时返回 None（不碰记忆）。"""
        index = _last_human(messages)
        if index is None:
            return None
        user, agent = self.identity(config, context)
        if not (user or agent):
            warnings.warn(f"{_NO_IDENTITY} Memory is skipped for this call.", stacklevel=2)
            return None
        space = space_for(user, agent)
        message = messages[index]
        evidence = Evidence(_message_id(message), "none")
        query = str(message.text)
        system = self.memory.system
        if not query.strip() or not system.db.get_sessions(space):
            return evidence
        thread = _thread_id(config)
        earlier = {_stored_text(m).strip() for m in messages[:index] if _is_turn(m)}
        exclude = _in_context(self.memory, space, session_for(thread), earlier) if thread else []
        now = datetime.now()
        found = system.retrieve(space, query, question_time=now, mode=self.mode, exclude=exclude)
        evidence.level = found.level.upper()
        if found.decision is not None:
            evidence.reasons = [english_reason(r) for r in found.decision.reasons]
        view = found.view
        if found.level == "l0":  # 不需要回忆：只给画像（没有画像就什么都不加）
            view = replace(view, rounds=[], facts=[], attach=[])
        blocks, refs, _ = memory_blocks(view, now, relative_time=True)
        if not view.rounds:  # 没取到对话片段时不留一个空的"(none)"块
            blocks = blocks[:-1]
        has_profile = view.core is not None and bool(view.core.text)
        evidence.recalled = len(refs) + int(has_profile)
        if blocks:
            evidence.block = "\n\n".join([MEMORY_NOTE, *blocks])
        record_event(
            "langgraph_recall", space=space, level=evidence.level, recalled=evidence.recalled
        )
        return evidence

    def recall_once(
        self, messages: Sequence[BaseMessage], config: RunnableConfig | None, context: Any = None
    ) -> Evidence | None:
        """同一回合（同一句用户消息）只召回一次，工具调用之后的模型调用沿用（进程内缓存）。"""
        index = _last_human(messages)
        user, agent = self.identity(config, context)
        if index is None or not (user or agent):
            return self.recall(messages, config, context)
        key = f"{space_for(user, agent)}\0{_message_id(messages[index])}"
        with self._guard:
            cached = self._recalled.get(key)
        if cached is not None:
            return cached
        evidence = self.recall(messages, config, context)
        if evidence is not None:
            with self._guard:
                _remember(self._recalled, key, evidence)
        return evidence

    # ---------- 写回 ----------

    def save(
        self, messages: Sequence[BaseMessage], config: RunnableConfig | None, context: Any = None
    ) -> int:
        """最终回答之后：把最后一句用户消息与回答在线写入（不调用 LLM，立即可检索）。

        有 thread_id 时写进这个线程的会话（标记为进行中，不抽取）；没有时这一问一答自成一个
        会话，交给后台抽取。写过的消息不再写。返回写入的发言数。写入失败只给警告：
        回答已经生成了，不因为记忆写不进去而丢掉（与记忆代理相同）。"""
        index = _last_human(messages)
        if index is None or not _is_final(messages[-1]):
            return 0  # 还在调用工具：等最终回答
        user, agent = self.identity(config, context)
        if not (user or agent):
            return 0
        space = space_for(user, agent)
        thread = _thread_id(config)
        sid = session_for(thread) if thread else None
        try:
            with self._lock(f"{space}\0{sid}"):
                picked = [
                    m
                    for m in messages[index:]
                    if _storable(m) and f"{space}\0{_message_id(m)}" not in self._saved
                ]
                if not picked:
                    return 0
                if sid is not None:
                    self.memory.system.open_session(space, sid)  # 进行中：照常写入，但不抽取
                self.memory.add(
                    [_openai(m) for m in picked], user_id=user, agent_id=agent, run_id=sid
                )
                with self._guard:
                    for m in picked:
                        _remember(self._saved, f"{space}\0{_message_id(m)}", None)
        except Exception as exc:
            record_event("langgraph_write_failed", error=f"{type(exc).__name__}: {exc}")
            warnings.warn(f"CueMem could not save the turn: {exc}", RuntimeWarning, stacklevel=2)
            return 0
        return len(picked)

    def end(
        self, config: RunnableConfig | None, context: Any = None, *, wait: bool = False
    ) -> dict[str, Any]:
        """结束线程：它的会话交给抽取（事实、对账、画像）；wait=True 时同步完成并返回报告。
        之后同一线程再有对话，还写进这个会话，下次结束时整段重新抽取。"""
        thread = _thread_id(config)
        if thread is None:
            raise ValueError("No thread_id in config['configurable']: there is no thread to end.")
        space = self.space(config, context)
        sid = session_for(thread)
        turns = len(self.memory.system.db.get_turns(space, sid))
        out: dict[str, Any] = {"session_id": sid, "turns": turns, "consolidation": None}
        if turns:
            report = self.memory.system.close_session(space, sid, wait=wait)
            out["consolidation"] = report.as_dict() if report is not None else "queued"
        return out

    def _lock(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())


def end_thread(
    memory: Memory,
    config: RunnableConfig,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    context: Any = None,
    wait: bool = False,
) -> dict[str, Any]:
    """结束一个 LangGraph 线程（config 与调用 Agent 时的相同）：整段对话交给抽取。"""
    return ThreadMemory(memory, user_id=user_id, agent_id=agent_id).end(config, context, wait=wait)


# ---------- 工具 ----------


def memory_tools(
    memory: Memory, *, user_id: str | None = None, agent_id: str | None = None
) -> list[BaseTool]:
    """recall / remember / forget 三个 LangChain 工具，与 MCP Server 的同名工具一致。

    user_id / agent_id 不填时，每次调用从 config["configurable"]（或图的 context）里取，
    同一组工具可以服务多个用户。recall 的结果里带着画像，所以没有单独的 profile 工具。"""
    threads = ThreadMemory(memory, user_id=user_id, agent_id=agent_id)

    def who(config: RunnableConfig) -> tuple[str | None, str | None]:
        user, agent = threads.identity(config, _context())
        if not (user or agent):  # 没有身份时报错，不会写进某个默认空间
            raise ValueError(_NO_IDENTITY)
        return user, agent

    def recall(query: str, config: RunnableConfig, deep: bool = False) -> str:
        user, agent = who(config)
        return format_recall(_recall(memory, user, agent, query, deep))

    def remember(text: str, config: RunnableConfig) -> str:
        user, agent = who(config)
        # 一句话一个会话，与 MCP 的 remember 相同：在线写入、后台抽取
        when = datetime.now().replace(microsecond=0)
        note = f"note-{when:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        out = memory.add(text, user_id=user, agent_id=agent, run_id=note, timestamp=when)
        return f"Saved as {out['turn_ids'][0]}." if out["turn_ids"] else "Nothing to save."

    def forget(ids: list[str], config: RunnableConfig) -> str:
        space = space_for(*who(config))
        system = memory.system
        fact_ids = [f.id for f in system.db.get_facts(space, ids=list(ids))]
        rounds = system.db.get_units(space, list(ids))
        turns = [t for unit in rounds if unit.kind == "round" for t in unit.source_turn_ids]
        report = system.forget_memories(space, fact_ids=fact_ids, turn_ids=turns)
        return (
            f"Deleted {report.turns_deleted} conversation turn(s); "
            f"facts {report.facts_before} -> {report.facts_after}."
        )

    return [
        StructuredTool.from_function(recall, name="recall", description=RECALL_DESCRIPTION),
        StructuredTool.from_function(remember, name="remember", description=REMEMBER_DESCRIPTION),
        StructuredTool.from_function(forget, name="forget", description=FORGET_DESCRIPTION),
    ]


def _recall(
    memory: Memory, user: str | None, agent: str | None, query: str, deep: bool
) -> dict[str, Any]:
    """MCP recall 的语义：调用方已经决定要查记忆，控制器选了 L0 时改走 L1，
    控制器只决定要不要查得更深。返回 format_recall 要的字典。"""
    found = memory.search(query, user_id=user, agent_id=agent, mode="l2" if deep else "auto")
    reasons = found["reasons"]
    if found["level"] == "L0" and memory.system.db.get_sessions(space_for(user, agent)):
        found = memory.search(query, user_id=user, agent_id=agent, mode="l1")
        reasons = ["raised from L0: the caller asked to recall"]
    results = found["results"]
    excerpts = [
        {"id": r["id"], "ts": r["created_at"], "text": r["memory"]}
        for r in results
        if r["type"] == "excerpt"
    ]
    return {
        "level": found["level"],
        "reasons": reasons,
        "profile": found["profile"],
        "facts": [
            {
                "id": r["id"],
                "statement": r["memory"],
                "said_on": r["created_at"],
                "status": r["status"],
            }
            for r in results
            if r["type"] == "fact"
        ],
        "excerpts": sorted(excerpts, key=lambda e: e["ts"]),
    }


# ---------- create_agent 中间件（推荐） ----------


class CueMemMiddleware(AgentMiddleware[CueMemState, Any, Any]):
    """LangChain create_agent 的中间件：控制器先行的自动召回 + 按线程写回。

    before_model：新的用户回合先过检索控制器，结果记在状态的 cuemem 项里（档位、理由、证据块）；
    wrap_model_call：把证据块放在那句用户消息前面，只给这次模型调用（线程的消息不变）；
    after_model：最终回答之后写回这一问一答。tools=True 时顺带注册 recall / remember / forget。
    mode：auto 由控制器选档；l1 / l2 每回合都检索；l0 只给画像。"""

    state_schema = CueMemState

    def __init__(
        self,
        memory: Memory,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: RecallMode = "auto",
        save: bool = True,
        tools: bool = False,
    ) -> None:
        super().__init__()
        self.threads = ThreadMemory(memory, user_id=user_id, agent_id=agent_id, mode=mode)
        self.save_turns = save
        self.tools = memory_tools(memory, user_id=user_id, agent_id=agent_id) if tools else []

    def before_model(self, state: CueMemState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self._recall(state, _config(), runtime.context)

    async def abefore_model(
        self, state: CueMemState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._recall, state, _config(), runtime.context)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        return handler(_with_evidence(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        return await handler(_with_evidence(request))

    def after_model(self, state: CueMemState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        if self.save_turns:
            self.threads.save(state["messages"], _config(), runtime.context)
        return None

    async def aafter_model(
        self, state: CueMemState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        if self.save_turns:
            await asyncio.to_thread(
                self.threads.save, state["messages"], _config(), runtime.context
            )
        return None

    def end_thread(
        self, config: RunnableConfig, *, context: Any = None, wait: bool = False
    ) -> dict[str, Any]:
        """结束一个线程：它的会话交给抽取（见 ThreadMemory.end）。"""
        return self.threads.end(config, context, wait=wait)

    def _recall(
        self, state: CueMemState, config: RunnableConfig, context: Any
    ) -> dict[str, Any] | None:
        messages = state["messages"]
        index = _last_human(messages)
        if index is None:
            return None
        current = state.get("cuemem")
        if current and current.get("message_id") == _message_id(messages[index]):
            return None  # 同一回合：工具调用之后的模型调用沿用这一回合的证据块
        evidence = self.threads.recall(messages, config, context)
        return {"cuemem": asdict(evidence) if evidence is not None else None}


def _with_evidence(request: ModelRequest[Any]) -> ModelRequest[Any]:
    evidence = cast("Mapping[str, Any]", request.state).get("cuemem")
    if not evidence or not evidence.get("block"):
        return request
    messages = inject(request.messages, evidence["message_id"], evidence["block"])
    return request.override(messages=messages)


# ---------- create_react_agent 的钩子（也可以当自己搭的图里的节点） ----------


def recall_hook(
    memory: Memory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    mode: RecallMode = "auto",
) -> RunnableLambda[Mapping[str, Any], dict[str, Any]]:
    """pre_model_hook：返回 llm_input_messages（注入了证据块的消息），线程的 messages 不变。"""
    threads = ThreadMemory(memory, user_id=user_id, agent_id=agent_id, mode=mode)

    def run(state: Mapping[str, Any], config: RunnableConfig, context: Any) -> dict[str, Any]:
        messages = list(state["messages"])
        evidence = threads.recall_once(messages, config, context)
        if evidence is not None and evidence.block:
            messages = inject(messages, evidence.message_id, evidence.block)
        return {"llm_input_messages": messages}

    def sync(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return run(state, config, _context())

    async def other(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
        return await asyncio.to_thread(run, state, config, _context())

    return RunnableLambda(sync, afunc=other, name="cuemem_recall")


def save_hook(
    memory: Memory, *, user_id: str | None = None, agent_id: str | None = None
) -> RunnableLambda[Mapping[str, Any], dict[str, Any]]:
    """post_model_hook：最终回答之后写回这一问一答（见 ThreadMemory.save），不改状态。"""
    threads = ThreadMemory(memory, user_id=user_id, agent_id=agent_id)

    def sync(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
        threads.save(state["messages"], config, _context())
        return {}

    async def other(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
        await asyncio.to_thread(threads.save, state["messages"], config, _context())
        return {}

    return RunnableLambda(sync, afunc=other, name="cuemem_save")


def inject[M: BaseMessage](messages: Sequence[M], message_id: str, block: str) -> list[M]:
    """把证据块放在那句用户消息的开头（与记忆代理的格式相同），返回新的列表，原消息不动。"""
    out = list(messages)
    note = f"<memory>\n{block}\n</memory>\n\n"
    for i in range(len(out) - 1, -1, -1):
        message = out[i]
        if isinstance(message, HumanMessage) and _message_id(message) == message_id:
            content = message.content
            if isinstance(content, str):
                new: str | list[str | dict[Any, Any]] = note + content
            else:
                new = [{"type": "text", "text": note}, *content]
            out[i] = message.model_copy(update={"content": new})
            break
    return out


# ---------- 内部 ----------

_NO_IDENTITY = (
    "CueMem needs a user_id or agent_id: pass it to the middleware, hooks or tools, or set "
    "config['configurable']['user_id'] (or user_id in the graph context)."
)


def _resolve(
    fixed: str | None, key: str, config: RunnableConfig | None, context: Any
) -> str | None:
    """固定值原样用；None 时从 config["configurable"][key] 或 context 取；"{name}" 从 name 取。"""
    if fixed is not None:
        template = _TEMPLATE.match(fixed)
        if template is None:
            return fixed
        key = template.group(1)
    value = ((config or {}).get("configurable") or {}).get(key)
    if value is None and context is not None:
        value = context.get(key) if isinstance(context, Mapping) else getattr(context, key, None)
    return str(value) if value is not None and str(value) else None


def _config() -> RunnableConfig:
    """当前图运行的 config；不在图里运行时是空的。"""
    try:
        return get_config()
    except RuntimeError:
        return {}


def _context() -> Any:
    """当前图运行的 context（Runtime.context）；不在图里运行时为 None。"""
    try:
        runtime = get_runtime()
    except RuntimeError:
        return None
    return getattr(runtime, "context", None)


def _thread_id(config: RunnableConfig | None) -> str | None:
    value = ((config or {}).get("configurable") or {}).get("thread_id")
    return str(value) if value is not None else None


def _last_human(messages: Sequence[BaseMessage]) -> int | None:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i
    return None


def _message_id(message: BaseMessage) -> str:
    """消息 ID（add_messages 会给每条消息分配）；没有时用内容的哈希。"""
    if message.id:
        return message.id
    return "sha1:" + hashlib.sha1(f"{message.type}\0{message.content}".encode()).hexdigest()


def _is_turn(message: BaseMessage) -> bool:
    return isinstance(message, (HumanMessage, AIMessage))


def _is_final(message: BaseMessage) -> bool:
    """最终回答：AI 消息、不再调用工具。"""
    return isinstance(message, AIMessage) and not message.tool_calls


def _storable(message: BaseMessage) -> bool:
    """写回记忆的消息：用户的话，以及有文字的最终回答（调用工具的中间步骤不写）。"""
    if isinstance(message, HumanMessage):
        return True
    return _is_final(message) and bool(str(message.text).strip())


def _openai(message: BaseMessage) -> dict[str, Any]:
    """LangChain 消息 → 写进 CueMem 的 OpenAI 格式消息。网络图片不下载（与记忆代理相同）。"""
    if isinstance(message, AIMessage):
        return {"role": "assistant", "content": str(message.text)}
    converted = convert_to_openai_messages(message)
    content = converted.get("content")
    if isinstance(content, list):
        content = [b for b in content if not _web_image(b)]
    out: dict[str, Any] = {"role": "user", "content": content}
    if converted.get("name"):
        out["name"] = converted["name"]
    return out


def _stored_text(message: BaseMessage) -> str:
    """写进记忆后的发言文本（与 SDK 的 parse_messages 相同：图片在文本里记作 <image>）。"""
    content = _openai(message)["content"]
    if not isinstance(content, list):
        return str(content or "")
    parts = [
        str(b.get("text", "")) if b.get("type") == "text" else "<image>"
        for b in content
        if isinstance(b, Mapping) and b.get("type") in ("text", "image_url")
    ]
    return " ".join(p for p in parts if p)


def _web_image(block: Any) -> bool:
    if not isinstance(block, Mapping) or block.get("type") != "image_url":
        return False
    value = block.get("image_url")
    url = str(value.get("url", "") if isinstance(value, Mapping) else value or "")
    return url.startswith(("http://", "https://"))


def _in_context(memory: Memory, space: str, session_id: str, present: set[str]) -> list[str]:
    """当前线程的会话里，原样还在消息里的轮（检索时排除）。按内容比对：线程的消息可能被
    裁剪或总结过，不能假定写进去的都还在（与记忆代理相同）。"""
    if not present:
        return []
    db = memory.system.db
    rounds = db.list_units(space, kinds=["round"], session_ids=[session_id])
    turns = {t.id: t for t in db.get_turns(space, session_id)}
    return [
        u.id
        for u in rounds
        if all(t in turns and turns[t].text.strip() in present for t in u.source_turn_ids)
    ]


def _remember[V](cache: OrderedDict[str, V], key: str, value: V) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _KEEP:
        cache.popitem(last=False)
