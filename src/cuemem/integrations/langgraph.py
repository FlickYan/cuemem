"""LangGraph / LangChain 集成（N12）：记忆工具、控制器先行的自动召回、按线程写回对话。

    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from cuemem import Memory
    from cuemem.integrations.langgraph import CueMemMiddleware

    memory = Memory(preset="deepseek")
    cuemem = CueMemMiddleware(memory)
    agent = create_agent(llm, tools, middleware=[cuemem], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t1", "user_id": "alice"}}
    agent.invoke({"messages": [{"role": "user", "content": "I moved to Munich"}]}, config)
    cuemem.end_thread(config)     # 这段对话结束：整段交给抽取（不调用时空闲 30 分钟后自动交给抽取）
    cuemem.close()                # 应用退出前：进行中的线程都交给抽取

三块，可以分开用：

- memory_tools：recall / remember / forget 三个工具，名字与说明和 MCP Server 一致（api/mcp.py）；
  出错时把错误作为工具结果交给模型，不中断 Agent；
- 自动召回：每个用户回合先过检索控制器（L0 不检索、只给画像；L1 / L2），取回的记忆排成与记忆代理
  相同的证据块，放在这一回合的用户消息前面，只给模型调用看，不写进线程的消息；
  同一回合里工具调用之后的模型调用沿用这个证据块，不重复检索，前缀缓存照常命中；
- 写回：回合结束时把用户的话与回答在线写入（不调用 LLM，立即可检索）。回答是最终的 AI 消息；
  结构化输出（response_format）时是结构化的结果，return_direct 工具结束时是工具的结果。

接法：推荐 LangChain 1.x 的 create_agent + CueMemMiddleware（before_agent 选档召回、
wrap_model_call 注入、after_agent 写回：每次调用各一次，不管从哪条路结束）。
langgraph.prebuilt.create_react_agent（1.0 起弃用）用 recall_hook / save_hook 作
pre_model_hook / post_model_hook；自己搭的 StateGraph 也可以把这两个钩子当节点用。

会话：一个 LangGraph 线程（thread_id）对应一个 CueMem 会话，进行中不抽取；end_thread、空闲超过
idle_minutes、或 close() 时整段交给抽取（与记忆代理、Agent 相同，D-045）。进程没来得及交出去的
（崩溃、没调用 close），下一个进程第一次碰到这个用户时交出去。

身份：user_id / agent_id 构造时固定；不填时运行时从 config["configurable"] 取，再看 context
（与 LangMem 的命名空间模板相同的做法）；写成 "{langgraph_user_id}" 这样的模板时从指定的键取。
LangGraph Server 上 configurable 与 context 都由客户端给：有登录用户（runtime.server_info.user）
时不填 user_id 就用它的 identity，客户端给的 user_id 与它不同时不碰记忆（写成模板表示信任客户端）。
都没有时不碰记忆（与记忆代理相同），并给出警告。

失败：召回或写回失败只给 RuntimeWarning，这一回合照常回答（只是不带记忆 / 没记下）。

遗忘：证据块不进图的状态与检查点，只缓存在进程里，forget 工具之后作废、重新召回。线程自己的
消息在 LangGraph 的检查点里，不归 CueMem 管；create_react_agent 的 llm_input_messages 是状态，
钩子注入的证据块会进它的检查点。

图片：只存 base64（data URL）图片；网络图片、本地路径与 file:// 都不存——消息内容来自终端用户，
不能让它把服务器上的文件读进记忆。

不提供 BaseStore 适配：CueMem 不是键值存储。写进去的是对话，事实由抽取得出、ID 由系统给，
更新是取代旧版本而不是覆盖，put(key, value) 之后 get(key) 拿不回原样的值。
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import re
import threading
import time
import uuid
import warnings
import weakref
from collections import Counter, OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig, RunnableLambda  # type: ignore[import-not-found, unused-ignore]
from langchain_core.tools import BaseTool, StructuredTool, ToolException  # type: ignore[import-not-found, unused-ignore]
from langgraph.config import get_config  # type: ignore[import-not-found, unused-ignore]
from langgraph.runtime import Runtime, get_runtime  # type: ignore[import-not-found, unused-ignore]
from langgraph.types import Command  # type: ignore[import-not-found, unused-ignore]
from pydantic import BaseModel

from cuemem.core.trace import record_event
from cuemem.evidence import format_listing, recall_evidence, recall_listing, resolve_ids
from cuemem.memory import MemorySystem, RecallMode
from cuemem.sdk import AsyncMemory, Memory, space_for

__all__ = [
    "CueMemMiddleware",
    "CueMemState",
    "Evidence",
    "ThreadMemory",
    "aend_thread",
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
_KEEP = 4096  # 进程内记住的已写入消息、已召回回合与接手过的空间数
_STRIPES = 64  # 写入锁的个数（按会话散列，个数固定，不随线程数增长）
_OUTPUT_ONLY = OmitFromSchema(input=True, output=False)
_PRIVATE = OmitFromSchema(input=True, output=True)
_clock = time.monotonic  # 空闲计时（测试里换成假时钟）


@dataclass
class Evidence:
    """一个用户回合的召回结果：控制器选的档与理由、注入的证据块。"""

    message_id: str  # 证据块属于哪一句用户消息
    level: str  # L0 / L1 / L2；"none"：没有检索（还没有记忆，或没有文字）；"error"：召回失败
    reasons: list[str] = field(default_factory=list)
    recalled: int = 0  # 注入的记忆条数（画像、事实、对话片段）
    block: str = ""  # 证据块（不含 <memory> 标签）；空表示什么都不加

    def summary(self) -> dict[str, Any]:
        """记进 Agent 状态的部分：档位、理由与条数。证据块本身不进状态，也就不进检查点。"""
        return {
            "message_id": self.message_id,
            "level": self.level,
            "reasons": list(self.reasons),
            "recalled": self.recalled,
        }


class CueMemState(AgentState[Any]):
    """中间件加到 Agent 状态里的两项：cuemem 是这一回合的召回结果（Evidence.summary，输出里
    看得到，输入里不用给）；cuemem_turn 是这一回合的用户消息（私有：回合结束时写回它，
    中途被 SummarizationMiddleware 总结掉了也还在）。"""

    # 没装可选依赖时 mypy 看不出 AgentState 是 TypedDict，NotRequired 会报错
    cuemem: NotRequired[Annotated[dict[str, Any] | None, _OUTPUT_ONLY]]  # type: ignore[valid-type, unused-ignore]
    cuemem_turn: NotRequired[Annotated[BaseMessage | None, _PRIVATE]]  # type: ignore[valid-type, unused-ignore]


def session_for(thread_id: str) -> str:
    """LangGraph 线程 → CueMem 会话 ID：同一个线程总是同一个会话。"""
    safe = _SAFE.sub("_", thread_id)
    if safe != thread_id or len(safe) > 64:  # 改过字符或太长：加上哈希，避免不同线程撞在一起
        safe = f"{safe[:48]}-{hashlib.sha1(thread_id.encode()).hexdigest()[:8]}"
    return f"lg-{safe}"


class _Shared:
    """同一个记忆系统在这个进程里的线程状态。ThreadMemory、中间件、钩子与工具共用：
    按请求新建 Agent（每次一个新的中间件）也不会丢掉进行中的线程、缓存与锁。"""

    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.stripes = [threading.Lock() for _ in range(_STRIPES)]
        self.active: dict[tuple[str, str], float] = {}  # 进行中的线程会话 → 最近一次活动
        self.adopted: OrderedDict[str, None] = OrderedDict()  # 接手过的空间
        self.saved: OrderedDict[str, None] = OrderedDict()  # 写过的消息（空间、会话、消息 ID）
        self.evidence: OrderedDict[str, tuple[int, Evidence]] = OrderedDict()  # 每回合的证据块
        self.forgets: Counter[str] = Counter()  # 每个空间遗忘过几次：之前缓存的证据块作废

    def lock(self, key: str) -> threading.Lock:
        """同一会话的写入按顺序进行（发言序号只增不减）。"""
        return self.stripes[hash(key) % _STRIPES]


_SHARED: weakref.WeakKeyDictionary[MemorySystem, _Shared] = weakref.WeakKeyDictionary()
_SHARED_GUARD = threading.Lock()


def _shared(system: MemorySystem) -> _Shared:
    with _SHARED_GUARD:
        shared = _SHARED.get(system)
        if shared is None:
            shared = _SHARED[system] = _Shared()
        return shared


class ThreadMemory:
    """LangGraph 线程与 CueMem 会话的对应：召回、写回、结束线程。中间件、钩子与工具共用；
    自己搭图时也可以直接调用。线程安全；memory 可以是 Memory 或 AsyncMemory。

    idle_minutes：线程空闲这么久，它的会话交给后台抽取（之后再有对话还写进这个会话，下次交出时
    整段重新抽取）；None 表示只在 end / close 时交出。"""

    def __init__(
        self,
        memory: Memory | AsyncMemory,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: RecallMode = "auto",
        idle_minutes: float | None = 30.0,
    ) -> None:
        self.memory = _sync(memory)
        self.user_id = user_id
        self.agent_id = agent_id
        self.mode: RecallMode = mode
        self.idle_minutes = idle_minutes
        self._shared = _shared(self.memory.system)

    # ---------- 身份 ----------

    def identity(
        self, config: RunnableConfig | None, context: Any = None
    ) -> tuple[str | None, str | None]:
        """（user_id, agent_id）：构造时固定的、LangGraph Server 的登录用户，或者从
        config["configurable"] / context 里取。没有身份、或与登录用户不符时报 ValueError。"""
        user = _resolve(self.user_id, "user_id", config, context, auth=_server_user(config))
        agent = _resolve(self.agent_id, "agent_id", config, context)
        if not (user or agent):
            raise ValueError(_NO_IDENTITY)
        return user, agent

    def space(self, config: RunnableConfig | None, context: Any = None) -> str:
        """记忆空间；没有身份时报 ValueError。"""
        return space_for(*self.identity(config, context))

    # ---------- 召回 ----------

    def recall(
        self,
        messages: Sequence[BaseMessage],
        config: RunnableConfig | None,
        context: Any = None,
        *,
        turn: BaseMessage | None = None,
    ) -> Evidence | None:
        """对这一回合的用户消息（默认是最后一句用户消息，对话总结不算）跑检索控制器，
        取回的记忆排成与记忆代理相同的证据块。当前线程里已经写进记忆、原样还在消息里的轮，
        检索时排除（名额留给上下文里没有的）。

        没有用户消息或没有身份时返回 None（不碰记忆；没有身份时给出警告）。召回失败时给出
        RuntimeWarning，返回 level 为 "error" 的空证据：这一回合照常回答，只是不带记忆。"""
        found = _turn(messages, turn)
        if found is None:
            return None
        message, earlier = found
        try:
            user, agent = self.identity(config, context)
        except ValueError as exc:
            warnings.warn(f"{exc} Memory is skipped for this call.", stacklevel=2)
            return None
        space = space_for(user, agent)
        thread = _thread_id(config)
        evidence = Evidence(_key_of(message), "none")
        try:
            self._adopt(space, thread)
            self.close_idle()
            self._retrieve(evidence, space, message, earlier, thread)
        except Exception as exc:
            record_event("langgraph_recall_failed", space=space, error=_err(exc))
            warnings.warn(
                f"CueMem could not recall memories; answering without them: {_err(exc)}",
                RuntimeWarning,
                stacklevel=2,
            )
            return Evidence(evidence.message_id, "error")
        return evidence

    def recall_once(
        self,
        messages: Sequence[BaseMessage],
        config: RunnableConfig | None,
        context: Any = None,
        *,
        turn: BaseMessage | None = None,
    ) -> Evidence | None:
        """同一回合只召回一次：同一句用户消息的证据块缓存在进程里，工具调用之后的模型调用沿用；
        forget 工具遗忘之后作废、重新召回。没有 ID 的消息（图里没用 add_messages）不缓存。"""
        found = _turn(messages, turn)
        if found is None:
            return None
        slot = self._slot(found[0], config, context)
        if slot is None:
            return self.recall(messages, config, context, turn=found[0])
        key, space = slot
        shared = self._shared
        with shared.guard:
            generation = shared.forgets[space]
            item = shared.evidence.get(key)
        if item is not None and item[0] == generation:
            return item[1]
        evidence = self.recall(messages, config, context, turn=found[0])
        if evidence is not None:
            with shared.guard:
                _remember(shared.evidence, key, (generation, evidence))
        return evidence

    def cached(
        self, message: BaseMessage, config: RunnableConfig | None, context: Any = None
    ) -> Evidence | None:
        """进程里缓存的这句用户消息的证据块；没有（或遗忘之后作废了）时为 None，不检索。"""
        slot = self._slot(message, config, context)
        if slot is None:
            return None
        key, space = slot
        with self._shared.guard:
            item = self._shared.evidence.get(key)
            current = self._shared.forgets[space]
        return item[1] if item is not None and item[0] == current else None

    # ---------- 写回 ----------

    def save(
        self, messages: Sequence[BaseMessage], config: RunnableConfig | None, context: Any = None
    ) -> int:
        """最终回答之后（最后一条是不再调用工具的 AI 消息）：写回最后一句用户消息与这个回答
        （见 save_turn）。还在调用工具时什么都不做。返回写入的发言数。"""
        if not messages or not _is_final(messages[-1]):
            return 0  # 还在调用工具：等最终回答
        found = _turn(messages, None)
        return self.save_turn(found[0] if found else None, messages[-1], config, context)

    def save_turn(
        self,
        user: BaseMessage | None,
        reply: BaseMessage | None,
        config: RunnableConfig | None,
        context: Any = None,
    ) -> int:
        """把一回合（用户消息与回答，可以只有其一）在线写入（不调用 LLM，立即可检索）。

        有 thread_id 时写进这个线程的会话（进行中，不抽取，见 end）；没有时这一回合自成一个
        会话，交给后台抽取。带 ID 的消息只写一次。返回写入的发言数。没有身份时不写（召回时
        已经警告过）；写入失败只给警告：回答已经生成了，不因为记忆写不进去而丢掉（与记忆代理相同）。"""
        picked = [m for m in (user, reply) if m is not None and _storable(m)]
        if not picked:
            return 0
        try:
            user_id, agent_id = self.identity(config, context)
        except ValueError:
            return 0
        space = space_for(user_id, agent_id)
        thread = _thread_id(config)
        sid = session_for(thread) if thread else None
        shared = self._shared
        keys = [f"{space}\0{sid or ''}\0{m.id}" if m.id else None for m in picked]
        try:
            self._adopt(space, thread)
            self.close_idle()
            with shared.lock(f"{space}\0{sid}"):
                with shared.guard:
                    fresh = [
                        (k, m) for k, m in zip(keys, picked, strict=True) if k not in shared.saved
                    ]
                if not fresh:
                    return 0
                if sid is not None:
                    self.memory.system.open_session(space, sid)  # 进行中：照常写入，但不抽取
                    with shared.guard:
                        shared.active[(space, sid)] = _clock()
                self.memory.add(
                    [_openai(m) for _, m in fresh], user_id=user_id, agent_id=agent_id, run_id=sid
                )
                with shared.guard:
                    for k, _ in fresh:
                        if k is not None:
                            _remember(shared.saved, k, None)
        except Exception as exc:
            record_event("langgraph_write_failed", space=space, error=_err(exc))
            warnings.warn(
                f"CueMem could not save the turn: {_err(exc)}", RuntimeWarning, stacklevel=2
            )
            return 0
        return len(fresh)

    # ---------- 会话 ----------

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
        with self._shared.guard:
            self._shared.active.pop((space, sid), None)
        turns = len(self.memory.system.db.get_turns(space, sid))
        out: dict[str, Any] = {"session_id": sid, "turns": turns, "consolidation": None}
        if turns:
            report = self.memory.system.close_session(space, sid, wait=wait)
            out["consolidation"] = report.as_dict() if report is not None else "queued"
        return out

    def close_idle(self) -> int:
        """空闲超过 idle_minutes 的线程会话交给后台抽取（与记忆代理相同）；返回交出的会话数。
        每次召回与写回时顺带检查；应用也可以定时调用。"""
        if self.idle_minutes is None:
            return 0
        limit = self.idle_minutes * 60
        now = _clock()
        shared = self._shared
        with shared.guard:
            stale = [k for k, seen in shared.active.items() if now - seen >= limit]
            for k in stale:
                del shared.active[k]
        self._close(stale)
        return len(stale)

    def close(self) -> int:
        """应用退出前调用（在 Memory.close 之前）：这个记忆系统进行中的线程会话都交给后台抽取
        （与记忆代理的 close_all 相同）。后台没做完的任务留在数据库里，下次巩固时完成。"""
        with self._shared.guard:
            sessions = list(self._shared.active)
            self._shared.active.clear()
        self._close(sessions)
        return len(sessions)

    # ---------- 内部 ----------

    def _retrieve(
        self,
        evidence: Evidence,
        space: str,
        message: BaseMessage,
        earlier: Sequence[BaseMessage],
        thread: str | None,
    ) -> None:
        query = str(message.text)
        system = self.memory.system
        if not query.strip() or not system.db.get_sessions(space):
            return
        present = {_stored_text(m).strip() for m in earlier if _is_turn(m)}
        exclude = _in_context(system, space, session_for(thread), present) if thread else []
        found = recall_evidence(system, space, query, mode=self.mode, exclude=exclude)
        evidence.level, evidence.reasons = found.level, found.reasons
        evidence.recalled, evidence.block = found.recalled, found.block
        record_event(
            "langgraph_recall", space=space, level=evidence.level, recalled=evidence.recalled
        )

    def _slot(
        self, message: BaseMessage, config: RunnableConfig | None, context: Any
    ) -> tuple[str, str] | None:
        """证据块的缓存键与空间；消息没有 ID 或没有身份时为 None（不缓存）。"""
        if not message.id:
            return None
        try:
            space = self.space(config, context)
        except ValueError:
            return None
        return f"{space}\0{_thread_id(config) or ''}\0{message.id}", space

    def _forgot(self, space: str) -> None:
        """这个空间刚遗忘过：进程里缓存的证据块作废，之后的模型调用重新召回。"""
        with self._shared.guard:
            self._shared.forgets[space] += 1

    def _adopt(self, space: str, thread: str | None) -> None:
        """这个进程第一次碰到这个空间：之前的进程（或之前的 Memory 对象）留下的线程会话——
        进行中就退出了，没人再交出去——交给后台抽取。当前线程的会话除外：它还要接着写。"""
        shared = self._shared
        with shared.guard:
            if space in shared.adopted:
                return
            _remember(shared.adopted, space, None)
        system = self.memory.system
        current = session_for(thread) if thread else None
        busy = system.open_sessions(space)
        leftover = [
            sid
            for sid in system.db.get_sessions(space)
            if sid.startswith("lg-") and sid != current and sid not in busy
        ]
        if not leftover:
            return
        added = system.consolidator.enqueue(space, leftover)
        if added or system.db.get_jobs(space, status="pending"):
            system.start_worker().submit(space)

    def _close(self, sessions: Iterable[tuple[str, str]]) -> None:
        for space, sid in sessions:
            try:
                self.memory.system.close_session(space, sid)
            except Exception as exc:
                record_event("langgraph_close_failed", space=space, error=_err(exc))
                warnings.warn(
                    f"CueMem could not hand thread {sid} to extraction: {_err(exc)}",
                    RuntimeWarning,
                    stacklevel=3,
                )


def end_thread(
    memory: Memory | AsyncMemory,
    config: RunnableConfig,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    context: Any = None,
    wait: bool = False,
) -> dict[str, Any]:
    """结束一个 LangGraph 线程（config 与调用 Agent 时的相同）：整段对话交给抽取。"""
    return ThreadMemory(memory, user_id=user_id, agent_id=agent_id).end(config, context, wait=wait)


async def aend_thread(
    memory: Memory | AsyncMemory,
    config: RunnableConfig,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    context: Any = None,
    wait: bool = False,
) -> dict[str, Any]:
    """end_thread 的异步版本：在线程里执行（wait=True 时整段抽取不占用事件循环）。"""
    threads = ThreadMemory(memory, user_id=user_id, agent_id=agent_id)
    return await asyncio.to_thread(threads.end, config, context, wait=wait)


# ---------- 工具 ----------


def memory_tools(
    memory: Memory | AsyncMemory, *, user_id: str | None = None, agent_id: str | None = None
) -> list[BaseTool]:
    """recall / remember / forget 三个 LangChain 工具，与 MCP Server 的同名工具一致。

    user_id / agent_id 不填时，每次调用从登录用户、config["configurable"]（或图的 context）里取，
    同一组工具可以服务多个用户。recall 的结果里带着画像，所以没有单独的 profile 工具。
    出错（包括没有身份）时模型收到一条错误的工具结果（status="error"），Agent 不中断。"""
    threads = ThreadMemory(memory, user_id=user_id, agent_id=agent_id)
    sync = threads.memory

    def who(config: RunnableConfig) -> tuple[str | None, str | None]:
        return threads.identity(config, _context())

    def recall(query: str, config: RunnableConfig, deep: bool = False) -> str:
        def run() -> str:
            return format_listing(_recall(sync.system, space_for(*who(config)), query, deep))

        return _tool("recall", run)

    def remember(text: str, config: RunnableConfig) -> str:
        def run() -> str:
            user, agent = who(config)
            # 一句话一个会话，与 MCP 的 remember 相同：在线写入、后台抽取
            when = datetime.now().replace(microsecond=0)
            note = f"note-{when:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
            out = sync.add(text, user_id=user, agent_id=agent, run_id=note, timestamp=when)
            return f"Saved as {out['turn_ids'][0]}." if out["turn_ids"] else "Nothing to save."

        return _tool("remember", run)

    def forget(ids: list[str], config: RunnableConfig) -> str:
        def run() -> str:
            space = space_for(*who(config))
            system = sync.system
            fact_ids, turns = resolve_ids(system, space, ids)
            try:
                report = system.forget_memories(space, fact_ids=fact_ids, turn_ids=turns)
            finally:  # 这一回合缓存的证据块可能含有刚删掉的内容：作废，下次模型调用重新召回
                threads._forgot(space)
            return (
                f"Deleted {report.turns_deleted} conversation turn(s); "
                f"facts {report.facts_before} -> {report.facts_after}."
            )

        return _tool("forget", run)

    return [
        StructuredTool.from_function(
            func,
            name=name,
            description=description,
            handle_tool_error=True,
        )
        for func, name, description in (
            (recall, "recall", RECALL_DESCRIPTION),
            (remember, "remember", REMEMBER_DESCRIPTION),
            (forget, "forget", FORGET_DESCRIPTION),
        )
    ]


def _tool(name: str, run: Callable[[], str]) -> str:
    """工具出错时抛 ToolException：LangChain 把它变成给模型的错误结果，不中断整个 Agent
    （与 MCP Server 把错误作为工具结果返回相同）。"""
    try:
        return run()
    except Exception as exc:
        record_event("langgraph_tool_failed", tool=name, error=_err(exc))
        raise ToolException(f"Memory {name} failed: {exc}") from exc


def _recall(system: MemorySystem, space: str, query: str, deep: bool) -> dict[str, Any]:
    """与 MCP 的 recall 相同的清单（控制器选了 L0 时改走 L1，见 recall_listing）。
    空间里还没有记忆时与 SDK 的 search 相同：不检索（不为问题算向量），只给画像。"""
    if not system.db.get_sessions(space):
        core = system.core_memory(space)
        profile = core.text if core is not None else ""
        return {
            "level": "L0",
            "reasons": ["no memories yet"],
            "profile": profile,
            "facts": [],
            "excerpts": [],
        }
    return recall_listing(system, space, query, mode="l2" if deep else "auto", at_least_l1=True)


# ---------- create_agent 中间件（推荐） ----------


class CueMemMiddleware(AgentMiddleware[CueMemState, Any, Any]):
    """LangChain create_agent 的中间件：控制器先行的自动召回 + 按线程写回。

    before_agent：这一回合的用户消息先过检索控制器，结果记在状态的 cuemem 项里（档位、理由、
    条数），证据块缓存在进程里；wrap_model_call：把证据块放在那句用户消息前面，只给模型调用
    （线程的消息不变）；after_agent：回合结束时写回用户的话与回答（最终回答、结构化输出或
    return_direct 工具的结果）。tools=True 时顺带注册 recall / remember / forget。
    mode：auto 由控制器选档；l1 / l2 每回合都检索；l0 只给画像。idle_minutes 见 ThreadMemory。"""

    state_schema = CueMemState

    def __init__(
        self,
        memory: Memory | AsyncMemory,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: RecallMode = "auto",
        save: bool = True,
        tools: bool = False,
        idle_minutes: float | None = 30.0,
    ) -> None:
        super().__init__()
        self.threads = ThreadMemory(
            memory, user_id=user_id, agent_id=agent_id, mode=mode, idle_minutes=idle_minutes
        )
        self.save_turns = save
        self.tools = memory_tools(memory, user_id=user_id, agent_id=agent_id) if tools else []

    def before_agent(self, state: CueMemState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self._start(state, _config(), runtime.context)

    async def abefore_agent(
        self, state: CueMemState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._start, state, _config(), runtime.context)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        turn = _turn_of(request)
        if turn is None:
            return handler(request)
        config, context = _config(), request.runtime.context
        evidence = self.threads.recall_once(request.messages, config, context, turn=turn)
        return _respond(request, evidence, handler(_with_evidence(request, evidence)))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        turn = _turn_of(request)
        if turn is None:
            return await handler(request)
        config, context = _config(), request.runtime.context
        evidence = self.threads.cached(turn, config, context)
        if evidence is None:  # 遗忘之后作废了，或换了进程：重新召回（在线程里，不占用事件循环）
            evidence = await asyncio.to_thread(
                self.threads.recall_once, request.messages, config, context, turn=turn
            )
        return _respond(request, evidence, await handler(_with_evidence(request, evidence)))

    def after_agent(self, state: CueMemState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        if self.save_turns:
            self.threads.save_turn(*_exchange(state), _config(), runtime.context)
        return None

    async def aafter_agent(
        self, state: CueMemState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        if self.save_turns:
            user, reply = _exchange(state)
            await asyncio.to_thread(self.threads.save_turn, user, reply, _config(), runtime.context)
        return None

    def end_thread(
        self, config: RunnableConfig, *, context: Any = None, wait: bool = False
    ) -> dict[str, Any]:
        """结束一个线程：它的会话交给抽取（见 ThreadMemory.end）。"""
        return self.threads.end(config, context, wait=wait)

    async def aend_thread(
        self, config: RunnableConfig, *, context: Any = None, wait: bool = False
    ) -> dict[str, Any]:
        """end_thread 的异步版本：在线程里执行。"""
        return await asyncio.to_thread(self.threads.end, config, context, wait=wait)

    def close(self) -> int:
        """应用退出前：进行中的线程会话都交给抽取（见 ThreadMemory.close）。"""
        return self.threads.close()

    async def aclose(self) -> int:
        return await asyncio.to_thread(self.threads.close)

    def _start(self, state: CueMemState, config: RunnableConfig, context: Any) -> dict[str, Any]:
        messages = state["messages"]
        found = _turn(messages, None)
        if found is None:
            return {"cuemem": None, "cuemem_turn": None}
        evidence = self.threads.recall_once(messages, config, context, turn=found[0])
        summary = evidence.summary() if evidence is not None else None
        return {"cuemem": summary, "cuemem_turn": found[0]}


def _state(request: ModelRequest[Any]) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", request.state)


def _turn_of(request: ModelRequest[Any]) -> BaseMessage | None:
    """这一回合的用户消息；这一回合没有召回（没有用户消息、没有身份）时为 None。"""
    state = _state(request)
    return state.get("cuemem_turn") if state.get("cuemem") is not None else None


def _with_evidence(request: ModelRequest[Any], evidence: Evidence | None) -> ModelRequest[Any]:
    if evidence is None or not evidence.block:
        return request
    return request.override(messages=inject(request.messages, evidence.message_id, evidence.block))


def _respond(
    request: ModelRequest[Any], evidence: Evidence | None, response: ModelResponse[Any]
) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
    """遗忘之后重新召回过时，把状态里的召回结果一并更新。"""
    summary = evidence.summary() if evidence is not None else None
    if summary == _state(request).get("cuemem"):
        return response
    return ExtendedModelResponse(
        model_response=response, command=Command(update={"cuemem": summary})
    )


def _exchange(state: Mapping[str, Any]) -> tuple[BaseMessage | None, BaseMessage | None]:
    """回合结束时要写回的用户消息与回答。回答：最终的 AI 消息；结构化输出时是结构化的结果
    （每次模型调用都会清掉旧的，所以有值就是这一回合的）；return_direct 工具结束时是工具的结果。"""
    messages: Sequence[BaseMessage] = state.get("messages") or []
    user = state.get("cuemem_turn")
    if user is None:
        found = _turn(messages, None)
        user = found[0] if found else None
    last = messages[-1] if messages else None
    structured = state.get("structured_response")
    if last is None or isinstance(last, HumanMessage):  # 没有调用模型就结束了（例如被拦下）
        return user, None
    if structured is not None:
        return user, AIMessage(content=_render(structured), id=last.id)
    if isinstance(last, AIMessage):
        return user, last if _is_final(last) else None
    if isinstance(last, ToolMessage):
        return user, AIMessage(content=str(last.text), id=last.id)
    return user, None


# ---------- create_react_agent 的钩子（也可以当自己搭的图里的节点） ----------


def recall_hook(
    memory: Memory | AsyncMemory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    mode: RecallMode = "auto",
    idle_minutes: float | None = 30.0,
) -> RunnableLambda[Mapping[str, Any], dict[str, Any]]:
    """pre_model_hook：返回 llm_input_messages（注入了证据块的消息），线程的 messages 不变。"""
    threads = ThreadMemory(
        memory, user_id=user_id, agent_id=agent_id, mode=mode, idle_minutes=idle_minutes
    )

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
    memory: Memory | AsyncMemory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    idle_minutes: float | None = 30.0,
) -> RunnableLambda[Mapping[str, Any], dict[str, Any]]:
    """post_model_hook：最终回答之后写回这一问一答（见 ThreadMemory.save），不改状态。
    return_direct 工具结束的回合之后没有模型调用，这个钩子看不到（create_agent 的中间件看得到）。"""
    threads = ThreadMemory(memory, user_id=user_id, agent_id=agent_id, idle_minutes=idle_minutes)

    def sync(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
        threads.save(state["messages"], config, _context())
        return {}

    async def other(state: Mapping[str, Any], config: RunnableConfig) -> dict[str, Any]:
        await asyncio.to_thread(threads.save, state["messages"], config, _context())
        return {}

    return RunnableLambda(sync, afunc=other, name="cuemem_save")


def inject[M: BaseMessage](messages: Sequence[M], message_id: str, block: str) -> list[M]:
    """把证据块放在那句用户消息的开头（与记忆代理的格式相同），返回新的列表，原消息不动。
    那句已经不在消息里（被 SummarizationMiddleware 总结掉了）时放在对话总结的开头：
    总结代替了之前的对话，证据块仍然跟着这一回合。"""
    out = list(messages)
    target = _find(out, lambda m: not _is_summary(m) and _key_of(m) == message_id)
    if target is None:
        target = _find(out, _is_summary)
    if target is None:
        return out
    message = out[target]
    note = f"<memory>\n{block}\n</memory>\n\n"
    content = message.content
    if isinstance(content, str):
        new: str | list[str | dict[Any, Any]] = note + content
    else:
        new = [{"type": "text", "text": note}, *content]
    out[target] = message.model_copy(update={"content": new})
    return out


# ---------- 内部 ----------

_NO_IDENTITY = (
    "CueMem needs a user_id or agent_id: pass it to the middleware, hooks or tools, or set "
    "config['configurable']['user_id'] (or user_id in the graph context)."
)
_NOT_SIGNED_IN_USER = (
    "The user_id in config['configurable'] or the graph context does not match the "
    "authenticated LangGraph Server user, so CueMem does not use it. Leave it out to use the "
    "authenticated user, or pass user_id='{user_id}' to trust the client-supplied value."
)


def _sync(memory: Memory | AsyncMemory) -> Memory:
    """AsyncMemory 换成它包着的 Memory：异步的钩子已经在线程里执行，这里只用同步方法。"""
    return memory.sync if isinstance(memory, AsyncMemory) else memory


def _resolve(
    fixed: str | None,
    key: str,
    config: RunnableConfig | None,
    context: Any,
    *,
    auth: str | None = None,
) -> str | None:
    """固定值原样用；"{name}" 从 configurable / context 的 name 取（开发者指定，照用）；
    None 时有登录用户就用它（客户端给的值与它不同时报错），否则从 key 取。"""
    if fixed is not None:
        template = _TEMPLATE.match(fixed)
        return fixed if template is None else _lookup(template.group(1), config, context)
    value = _lookup(key, config, context)
    if auth is not None and value is not None and value != auth:
        raise ValueError(_NOT_SIGNED_IN_USER)
    return auth or value


def _lookup(key: str, config: RunnableConfig | None, context: Any) -> str | None:
    value = ((config or {}).get("configurable") or {}).get(key)
    if value is None and context is not None:
        value = context.get(key) if isinstance(context, Mapping) else getattr(context, key, None)
    return str(value) if value is not None and str(value) else None


def _server_user(config: RunnableConfig | None) -> str | None:
    """LangGraph Server 的登录用户（runtime.server_info.user 也从这里来）。与 LangGraph 一样
    只认带 identity 属性的对象：客户端在 configurable 里传来的字典不算。"""
    user = ((config or {}).get("configurable") or {}).get("langgraph_auth_user")
    try:
        if getattr(user, "is_authenticated", True) is False:
            return None
        identity = getattr(user, "identity", None)
    except Exception:
        return None
    return str(identity) if identity else None


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


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_summary(message: BaseMessage) -> bool:
    """SummarizationMiddleware 换上的对话总结：形式上是用户消息，其实不是用户说的话。"""
    return (
        isinstance(message, HumanMessage)
        and message.additional_kwargs.get("lc_source") == "summarization"
    )


def _is_user(message: BaseMessage) -> bool:
    return isinstance(message, HumanMessage) and not _is_summary(message)


def _find(messages: Sequence[BaseMessage], match: Callable[[BaseMessage], bool]) -> int | None:
    """从后往前找第一条符合的消息。"""
    for i in range(len(messages) - 1, -1, -1):
        if match(messages[i]):
            return i
    return None


def _turn(
    messages: Sequence[BaseMessage], turn: BaseMessage | None
) -> tuple[BaseMessage, Sequence[BaseMessage]] | None:
    """这一回合的用户消息与它之前的消息。turn 不在消息里（被总结掉了）时，其余消息都算之前的。"""
    if turn is None:
        index = _find(messages, _is_user)
        return None if index is None else (messages[index], messages[:index])
    index = _find(messages, lambda m: m is turn or (bool(turn.id) and m.id == turn.id))
    return turn, messages if index is None else messages[:index]


def _key_of(message: BaseMessage) -> str:
    """消息 ID（add_messages 会给每条消息分配）；没有时用内容的哈希。"""
    if message.id:
        return message.id
    return "sha1:" + hashlib.sha1(f"{message.type}\0{message.content}".encode()).hexdigest()


def _is_turn(message: BaseMessage) -> bool:
    return _is_user(message) or isinstance(message, AIMessage)


def _is_final(message: BaseMessage) -> bool:
    """最终回答：AI 消息、不再调用工具。"""
    return isinstance(message, AIMessage) and not message.tool_calls


def _storable(message: BaseMessage) -> bool:
    """写回记忆的消息：用户的话，以及有文字的最终回答（调用工具的中间步骤与对话总结不写）。"""
    if isinstance(message, HumanMessage):
        return not _is_summary(message)
    return _is_final(message) and bool(str(message.text).strip())


def _openai(message: BaseMessage) -> dict[str, Any]:
    """LangChain 消息 → 写进 CueMem 的 OpenAI 格式消息。图片只留 base64（data URL）。"""
    if isinstance(message, AIMessage):
        return {"role": "assistant", "content": str(message.text)}
    content = message.content
    out: dict[str, Any] = {"role": "user", "content": content}
    if not isinstance(content, str):
        out["content"] = [part for block in content if (part := _part(block)) is not None]
    if message.name:
        out["name"] = message.name
    return out


def _part(block: str | dict[Any, Any]) -> dict[str, Any] | None:
    """一个内容块 → OpenAI 格式：文字照留；图片只留 data URL——网络图片不下载（与记忆代理
    相同），本地路径与 file:// 不读（消息来自终端用户）；其他块不写。"""
    if isinstance(block, str):
        return {"type": "text", "text": block}
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "text": str(block.get("text", ""))}
    if kind == "image_url":  # OpenAI 格式
        value = block.get("image_url")
        url = str(value.get("url", "") if isinstance(value, Mapping) else value or "")
    elif kind == "image":  # LangChain 的标准格式（base64 / url），以及旧的 source_type 写法
        data = block.get("base64") or (block.get("data") if block.get("source_type") else None)
        mime = block.get("mime_type") or "image/png"
        url = f"data:{mime};base64,{data}" if data else str(block.get("url") or "")
    else:
        return None
    return {"type": "image_url", "image_url": {"url": url}} if url.startswith("data:") else None


def _stored_text(message: BaseMessage) -> str:
    """写进记忆后的发言文本（与 SDK 的 parse_messages 相同：图片在文本里记作 <image>）。"""
    content = _openai(message)["content"]
    if not isinstance(content, list):
        return str(content or "")
    parts = [str(b.get("text", "")) if b["type"] == "text" else "<image>" for b in content]
    return " ".join(p for p in parts if p)


def _render(value: Any) -> str:
    """结构化输出 → 写进记忆的回答（JSON）。"""
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _in_context(system: MemorySystem, space: str, session_id: str, present: set[str]) -> list[str]:
    """当前线程的会话里，原样还在消息里的轮（检索时排除）。按内容比对：线程的消息可能被
    裁剪或总结过，不能假定写进去的都还在（与记忆代理相同）。"""
    if not present:
        return []
    db = system.db
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
