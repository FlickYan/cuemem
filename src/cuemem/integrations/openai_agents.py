"""OpenAI Agents SDK 集成（N12）：记忆工具、调用模型前先过检索控制器、对话写进长期记忆。

    from agents import Agent, RunConfig, Runner
    from cuemem import Memory
    from cuemem.integrations.openai_agents import CueMemSession, memory_input_filter, memory_tools

    memory = Memory(preset="deepseek")
    agent = Agent(name="Assistant", model=..., tools=memory_tools(memory, user_id="alice"))
    config = RunConfig(
        call_model_input_filter=memory_input_filter(memory, user_id="alice"),
        trace_include_sensitive_data=False,  # 注入的记忆不上传到 trace 后台
    )
    async with CueMemSession("chat-1", memory, user_id="alice") as session:
        await Runner.run(agent, "I moved to Lisbon", session=session, run_config=config)

三样东西，可以分开用（需要 openai-agents，可选依赖 cuemem[openai-agents]）：

- memory_tools：recall / remember / forget 三个工具，名字、说明与语义与 MCP（api/mcp.py）相同；
- memory_input_filter：交给 RunConfig.call_model_input_filter，SDK 每次调用模型之前把输入给它。
  对最新一句用户消息先跑检索控制器：L0 不检索、只给画像（没有画像就什么都不加），L1 / L2 取回
  事实与对话片段，排成与记忆代理相同的证据块（cuemem.evidence）。只改这次发给模型的副本。每一轮的
  第一次调用都重新检索；同一轮里工具调用之后的再次调用沿用这一轮的证据（遗忘之后重新检索）。
  进行中的会话里、原样已在输入中的轮不参与检索；
- CueMemSession：Session 协议的实现。内层会话（默认内存里的 SQLiteSession）原样保存短期历史；
  用户与助手的文字消息同时写进 CueMem（在线写入、立即可检索；进行中不抽取事实）。

证据块的位置（placement）：
- "message"（默认）：与记忆代理相同，放在这句用户消息的开头。系统提示与之前的消息不动，供应商的
  前缀缓存照常命中。用于历史由客户端保存的用法（Session、result.to_input_list()）；
- "instructions"：接在这次调用的 instructions 后面，每次请求单独给。服务端保存对话状态时
  （conversation_id、previous_response_id、auto_previous_response_id）用这个：改过的用户消息会被
  SDK 当成另一条，原消息每次调用都重发，带着证据块存到服务端（遗忘管不到）。用 "message" 时过滤器
  察觉到这种情况（这一轮的工具结果前面没有工具调用）会给出警告、不再注入。

会话（D-045）：一个 SDK 会话在 CueMem 里是一段或几段会话（session_for 给出第一段的 ID；不同的
session_id 不会撞在一起）。写入时标记为进行中、不抽取；end()、async with 结束、空闲超过
idle_minutes（默认 30 分钟）或 close_sessions() 时交给后台抽取，进程正常退出时只登记、由下一个
进程完成。之后同一个 SDK 会话再有对话，写进新的一段：对话日期是新的，交出过的不重新抽取。
不用 async with、也不调用 end() 时，空闲之后照样交出；进程崩溃没来得及交出的，下一个进程第一次
碰到这个用户时交出。
按请求新建 CueMemSession 的服务（一段对话跨多个请求）不要每个请求都 end()：那样一问一答各成一段。

一问一答为一轮：用户消息先留着，等助手的文字回复来了一起写入；同一轮里工具调用之后的回复并进
这一轮。模型没有回应过的用户消息（输入护栏拦下的、调用出错的）不写进长期记忆。

遗忘（D-021）：pop_item 撤回的消息、clear_session 清掉的整段对话（各段都算），在长期记忆里也一并
遗忘：删原始日志、按剩下的日志重建派生记忆。先在 CueMem 里遗忘，成功了才动内层会话，失败时抛出、
内层不变。只删这些发言：别的对话里转述过的不追（这里不记录回复引用了哪些记忆）。反过来，用
forget 工具或 Memory.delete 删掉的发言，下次读历史时也从内层会话里去掉所在的整轮（与 Agent 的
prune 相同）。

身份：工具与过滤器的 user_id / agent_id 可以是字符串，也可以是函数，按这次运行的上下文
（Runner.run 的 context）返回 ID。多个用户共用一个模块级的 Agent 与 RunConfig 时要用函数，
否则所有人读写的是同一个记忆空间。CueMemSession 每段对话一个，身份在构造时给。

trace：SDK 默认把模型的输入（包括注入的记忆）上传到 OpenAI 的 trace 后台；不需要时
RunConfig(trace_include_sensitive_data=False) 或 tracing_disabled=True。
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import hashlib
import logging
import re
import threading
import time
import weakref
from collections import Counter, OrderedDict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import TracebackType
from typing import Any, Literal, cast

# 框架是可选依赖（uv sync --extra openai-agents）；没装时 mypy 把它们当作 Any，不报找不到模块
from agents import (  # type: ignore[import-not-found, unused-ignore]
    RunContextWrapper,
    Session,
    SessionSettings,
    SQLiteSession,
    Tool,
    TResponseInputItem,
    function_tool,
)
from agents.run_config import (  # type: ignore[import-not-found, unused-ignore]
    CallModelData,
    ModelInputData,
)

from cuemem.core.ids import sortable_session_id
from cuemem.core.trace import record_event
from cuemem.evidence import (
    format_listing,
    in_context_rounds,
    recall_evidence,
    recall_listing,
    resolve_ids,
    stored_text,
)
from cuemem.memory import ForgetReport, MemorySystem, RecallMode
from cuemem.read.controller import Router
from cuemem.sdk import AsyncMemory, Memory, space_for

__all__ = [
    "CueMemSession",
    "Evidence",
    "MemoryInputFilter",
    "close_idle_sessions",
    "close_sessions",
    "memory_input_filter",
    "memory_tools",
    "session_for",
]

logger = logging.getLogger(__name__)

# 与 MCP 工具（api/mcp.py）的说明逐字相同
RECALL_DESCRIPTION = (
    "Recall memories relevant to the query. The memory controller decides whether a fast "
    "search is enough or a deeper one is needed; deep=true forces the deep search. Returns the "
    "user's profile, facts with validity, and conversation excerpts, each with an id and a date."
)
REMEMBER_DESCRIPTION = (
    "Save something worth remembering about the user (a preference, a fact, a plan). "
    "It can be recalled right away and is turned into structured facts in the background."
)
FORGET_DESCRIPTION = (
    "Forget memories by the ids that recall returned (fact ids or excerpt ids). Deletes "
    "the conversation they came from and rebuilds everything derived from it."
)

Placement = Literal["message", "instructions"]
# user_id / agent_id：固定的 ID，或者由这次运行的上下文（Runner.run 的 context）给出 ID 的函数
Identity = str | Callable[[Any], str | None] | None

_TEXT_PARTS = ("input_text", "output_text", "text")
_SAFE = re.compile(r"[^A-Za-z0-9_.:-]")
_PREFIX = "oa-"  # 这个集成写的 CueMem 会话：进程崩溃后据此找到没交出去的
_EVERYTHING = 2**62  # 读全部历史时显式给的上限：limit=None 会被内层的 session_settings.limit 截断
_STRIPES = 64
_CACHE_SIZE = 64
_CACHE_TTL = 600.0  # 秒：同一轮里工具调用之后沿用证据的时限
_TAIL = 64  # 接手别的对象开了头的一轮时，往回看内层会话的条数

_clock = time.monotonic


# ---------- 进程里共用的状态 ----------


class _Shared:
    """同一个记忆系统在这个进程里的会话状态：所有 CueMemSession、过滤器与工具共用
    （按请求新建对象也不会丢掉进行中的会话）。"""

    def __init__(self) -> None:
        self.guard = threading.Lock()
        self.stripes = [threading.Lock() for _ in range(_STRIPES)]
        # 进行中的段 →（最近一次写入，空闲多少秒后交出；None：只在 end / close_sessions 时交出）
        self.active: dict[tuple[str, str], tuple[float, float | None]] = {}
        self.adopted: OrderedDict[str, None] = OrderedDict()  # 接手过的空间
        self.forgets: Counter[str] = Counter()  # 每个空间经这里遗忘过几次：缓存的证据作废

    def lock(self, space: str, session: str) -> threading.Lock:
        """同一段对话（各段共用）的写入按顺序进行：发言序号只增不减。"""
        return self.stripes[hash(f"{space}\0{session.split('~', 1)[0]}") % _STRIPES]

    def generation(self, space: str) -> int:
        with self.guard:
            return self.forgets[space]

    def forgot(self, space: str) -> None:
        with self.guard:
            self.forgets[space] += 1


_SHARED: weakref.WeakKeyDictionary[MemorySystem, _Shared] = weakref.WeakKeyDictionary()
_SHARED_GUARD = threading.Lock()


def _shared(system: MemorySystem) -> _Shared:
    with _SHARED_GUARD:
        shared = _SHARED.get(system)
        if shared is None:
            shared = _SHARED[system] = _Shared()
        return shared


def close_idle_sessions(memory: Memory | AsyncMemory) -> int:
    """空闲超过 idle_minutes 的会话交给后台抽取，返回交出的段数。写入与召回时顺带检查；
    长时间没有请求的服务也可以定时调用。"""
    return _sweep(_sync(memory).system)


def close_sessions(memory: Memory | AsyncMemory) -> int:
    """进行中的会话都交给后台抽取（应用退出前、Memory.close 之前调用；与记忆代理的 close_all
    相同）。后台没做完的任务留在数据库里，下次巩固时完成。返回交出的段数。"""
    system = _sync(memory).system
    shared = _shared(system)
    with shared.guard:
        sessions = list(shared.active)
        shared.active.clear()
    for space, sid in sessions:
        _hand_over(system, space, sid)
    return len(sessions)


def _sweep(system: MemorySystem) -> int:
    shared = _shared(system)
    with shared.guard:
        now = _clock()
        stale = [
            key
            for key, (seen, limit) in shared.active.items()
            if limit is not None and now - seen >= limit
        ]
    closed = 0
    for space, sid in stale:
        lock = shared.lock(space, sid)
        if not lock.acquire(blocking=False):  # 正在写：下次再看
            continue
        try:
            with shared.guard:
                seen, limit = shared.active.get((space, sid), (0.0, None))
                if limit is None or _clock() - seen < limit:  # 刚写过，或者已经交出
                    continue
                del shared.active[(space, sid)]
            _hand_over(system, space, sid)
            closed += 1
        finally:
            lock.release()
    return closed


def _hand_over(system: MemorySystem, space: str, sid: str) -> None:
    """一段会话交给后台抽取；出错只记日志（数据都在，下次巩固或接手时再交）。"""
    try:
        system.close_session(space, sid)
    except Exception as exc:
        logger.warning("CueMem could not hand session %s to extraction (%s).", sid, _err(exc))
        record_event("agents_close_failed", space=space, error=_err(exc))


def _adopt(system: MemorySystem, space: str) -> None:
    """这个进程第一次碰到这个空间：之前的进程（崩溃、没调用 close_sessions）留下的进行中的段，
    没人再交出去，现在交给后台抽取。已经抽取过的段内容没变，不会重新抽取。"""
    shared = _shared(system)
    with shared.guard:
        if space in shared.adopted:
            return
        shared.adopted[space] = None
        while len(shared.adopted) > 4096:
            shared.adopted.popitem(last=False)
    busy = system.open_sessions(space)
    leftover = [
        sid for sid in system.db.get_sessions(space) if sid.startswith(_PREFIX) and sid not in busy
    ]
    if not leftover:
        return
    added = system.consolidator.enqueue(space, leftover)
    if added or system.db.get_jobs(space, status="pending"):
        system.start_worker().submit(space)


@atexit.register
def _at_exit() -> None:
    """进程正常退出：进行中的段登记抽取（不启动后台线程，下一个进程碰到这个用户时完成）。"""
    for system, shared in list(_SHARED.items()):
        with shared.guard:
            sessions = list(shared.active)
            shared.active.clear()
        for space, sid in sessions:
            with contextlib.suppress(Exception):  # 记忆系统已经关闭：下一个进程接手
                system.consolidator.enqueue(space, [sid])


# ---------- 工具 ----------


@dataclass(frozen=True)
class _Who:
    user_id: Identity
    agent_id: Identity

    @classmethod
    def of(cls, user_id: Identity, agent_id: Identity) -> _Who:
        if not callable(user_id) and not callable(agent_id):
            space_for(user_id, agent_id)  # 两个都没有时报错
        return cls(user_id, agent_id)

    def ids(self, context: Any) -> tuple[str | None, str | None]:
        user = self.user_id(context) if callable(self.user_id) else self.user_id
        agent = self.agent_id(context) if callable(self.agent_id) else self.agent_id
        return user, agent

    def space(self, context: Any) -> str:
        return space_for(*self.ids(context))


def memory_tools(
    memory: Memory | AsyncMemory,
    *,
    user_id: Identity = None,
    agent_id: Identity = None,
) -> list[Tool]:
    """recall / remember / forget 三个工具（与 MCP 同名同义），作用于 user_id、agent_id 的记忆空间
    （给函数时按每次运行的上下文取 ID）。

    recall 至少做一次快速召回（模型已经决定要查），控制器只决定要不要查得更深；结果原样交给模型
    去推理，这里不调用回答模型。画像不单独做工具：memory_input_filter 每一轮都会带上。
    与 CueMemSession 一起用时对话已经在写入，remember 只用于用户明确要求记住的事。
    返回的都是 FunctionTool（类型标成 Tool，好直接交给 Agent(tools=...)）。
    """
    sync = _sync(memory)
    who = _Who.of(user_id, agent_id)

    @function_tool(name_override="recall", description_override=RECALL_DESCRIPTION)
    def recall(ctx: RunContextWrapper[Any], query: str, deep: bool = False) -> str:
        # 与 MCP 的 recall 相同（at_least_l1）：控制器选了 L0 时改走 L1
        space = who.space(ctx.context)
        mode: RecallMode = "l2" if deep else "auto"
        return format_listing(
            recall_listing(sync.system, space, query, mode=mode, at_least_l1=True)
        )

    @function_tool(name_override="remember", description_override=REMEMBER_DESCRIPTION)
    def remember(ctx: RunContextWrapper[Any], text: str) -> str:
        user, agent = who.ids(ctx.context)
        when = datetime.now()  # 不截到秒：同一秒里的几条笔记也分得出先后
        note = sortable_session_id("note", when)
        out = sync.add(text, user_id=user, agent_id=agent, run_id=note, timestamp=when)
        if not out["turn_ids"]:
            return "Nothing was saved: the text is empty."
        return f"Saved as {out['turn_ids'][0]}."

    @function_tool(name_override="forget", description_override=FORGET_DESCRIPTION)
    def forget(ctx: RunContextWrapper[Any], ids: list[str]) -> str:
        report = _forget_ids(sync.system, who.space(ctx.context), ids)
        return (
            f"Deleted {report.turns_deleted} conversation turn(s); "
            f"facts {report.facts_before} -> {report.facts_after}."
        )

    return [recall, remember, forget]


def _forget_ids(system: MemorySystem, space: str, ids: Sequence[str]) -> ForgetReport:
    """按 recall 给出的 ID 遗忘：事实 ID，或对话片段（轮）的 ID。缓存的证据随之作废。"""
    fact_ids, turns = resolve_ids(system, space, ids)
    report = system.forget_memories(space, fact_ids=fact_ids, turn_ids=turns)
    _shared(system).forgot(space)
    return report


# ---------- 调用模型之前：自动回忆 ----------


@dataclass(frozen=True)
class Evidence:
    """一次自动回忆的结果。不直接用 cuemem.evidence.Evidence：这里的会进缓存、在并发的运行之间
    共用，保持不可变；level 多一个 "none"；也不带 Recall（缓存不留住整份检索结果）。"""

    query: str
    level: str  # "L0" / "L1" / "L2"；"none"：这个空间还没有记忆
    reasons: list[str]
    recalled: int  # 注入的记忆条数（画像、事实、对话片段）
    block: str  # 注入的证据块；空：什么都没加


class MemoryInputFilter:
    """RunConfig.call_model_input_filter：调用模型之前，按检索控制器的决定把记忆放进输入。

    last 是最近一次的结果（日志与测试用；几个 run 并发共用一个过滤器时只是其中之一）。
    """

    def __init__(
        self,
        memory: Memory | AsyncMemory,
        *,
        user_id: Identity = None,
        agent_id: Identity = None,
        mode: RecallMode = "auto",
        router: Router = "learned",
        lam: float | None = None,
        placement: Placement = "message",
    ) -> None:
        if placement not in ("message", "instructions"):
            raise ValueError("placement must be 'message' or 'instructions'.")
        self.memory = _sync(memory)
        self.who = _Who.of(user_id, agent_id)
        self.mode: RecallMode = mode
        self.router: Router = router
        self.lam = lam
        self.placement: Placement = placement
        self.last: Evidence | None = None
        self._cache: OrderedDict[tuple[Any, ...], tuple[float, Evidence]] = OrderedDict()
        # 每个空间（与运行上下文）最近一次的证据：服务端保存对话时，工具调用之后的输入里没有用户消息
        self._recent: OrderedDict[tuple[str, int | None], tuple[float, Evidence]] = OrderedDict()
        self._lock = threading.Lock()
        self._warned = False

    async def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        model_data = data.model_data
        items = model_data.input
        found = _latest_user(items)
        if found is None:
            if self.placement == "instructions" and _has_tool_results(items):
                return self._follow(data)
            return model_data
        index, message = found
        query = _text(message.get("content")).strip()
        if not query:
            return model_data
        follow_up = index < len(items) - 1  # 这一轮里工具调用之后的再次调用
        if follow_up and self.placement == "message" and _server_managed(items, index):
            self._warn_server_managed()
            return model_data
        try:
            space = self.who.space(data.context)
        except ValueError as exc:
            logger.warning("CueMem: %s Calling the model without memories.", exc)
            return model_data
        system = self.memory.system
        key = (space, _shared(system).generation(space), index, query, _history(items, index))
        evidence = self._get(self._cache, key) if follow_up else None
        if evidence is None:  # 每一轮的第一次调用都重新检索
            # 按写进记忆后的样子比对（_mirrorable 之后：只有 data URL 的图片记作 <image>）
            pending = [p for p in map(_mirrorable, items) if p is not None]
            present = {stored_text(p.message).strip() for p in pending}
            try:
                evidence = await asyncio.to_thread(self._recall, space, query, present)
            except Exception as exc:  # 记忆出错不挡住对话：这一次不带记忆
                logger.warning(
                    "CueMem recall failed (%s); calling the model without memories.", _err(exc)
                )
                record_event("agents_recall_failed", space=space, error=_err(exc))
                return model_data
            self._put(self._cache, key, evidence)
        self._put(self._recent, _run_key(space, data.context), evidence)
        self.last = evidence
        return self._apply(model_data, evidence, found)

    def _follow(self, data: CallModelData[Any]) -> ModelInputData:
        """服务端保存对话状态时工具调用之后的调用：输入里只有工具结果，instructions 不会从上一次
        响应带过来，再给一次这一轮的证据。"""
        try:
            space = self.who.space(data.context)
        except ValueError:
            return data.model_data
        evidence = self._get(self._recent, _run_key(space, data.context))
        return self._apply(data.model_data, evidence, None) if evidence else data.model_data

    def _apply(
        self,
        model_data: ModelInputData,
        evidence: Evidence,
        found: tuple[int, Mapping[str, Any]] | None,
    ) -> ModelInputData:
        if not evidence.block:
            return model_data
        note = f"<memory>\n{evidence.block}\n</memory>"
        items = model_data.input
        if self.placement == "instructions" or found is None:
            base = model_data.instructions
            return ModelInputData(input=items, instructions=f"{base}\n\n{note}" if base else note)
        index, message = found
        injected = list(items)
        injected[index] = cast(TResponseInputItem, _with_note(message, note))
        return ModelInputData(input=injected, instructions=model_data.instructions)

    def _recall(self, space: str, query: str, present: Collection[str]) -> Evidence:
        """与记忆代理相同：控制器选档、排出证据块；L0 只给画像。"""
        system = self.memory.system
        _adopt(system, space)
        _sweep(system)
        if not system.db.get_sessions(space):
            return Evidence(query, "none", ["no memories yet"], 0, "")
        found = recall_evidence(
            system,
            space,
            query,
            mode=self.mode,
            router=self.router,
            lam=self.lam,
            exclude=in_context_rounds(system, space, sorted(system.open_sessions(space)), present),
        )
        record_event("agents_recall", space=space, level=found.level, recalled=found.recalled)
        return Evidence(query, found.level, found.reasons, found.recalled, found.block)

    def _get(self, cache: OrderedDict[Any, tuple[float, Evidence]], key: Any) -> Evidence | None:
        with self._lock:
            entry = cache.get(key)
        if entry is None or _clock() - entry[0] > _CACHE_TTL:
            return None
        return entry[1]

    def _put(
        self, cache: OrderedDict[Any, tuple[float, Evidence]], key: Any, value: Evidence
    ) -> None:
        with self._lock:
            cache[key] = (_clock(), value)
            cache.move_to_end(key)
            while len(cache) > _CACHE_SIZE:
                cache.popitem(last=False)

    def _warn_server_managed(self) -> None:
        if self._warned:
            return
        self._warned = True
        logger.warning(
            "CueMem: the conversation state is kept on the server (conversation_id or "
            "previous_response_id), so memories placed in the user message would be stored "
            "there and the message re-sent. Skipping injection; use "
            "memory_input_filter(..., placement='instructions') with server-managed state."
        )


def memory_input_filter(
    memory: Memory | AsyncMemory,
    *,
    user_id: Identity = None,
    agent_id: Identity = None,
    mode: RecallMode = "auto",
    router: Router = "learned",
    lam: float | None = None,
    placement: Placement = "message",
) -> MemoryInputFilter:
    """RunConfig(call_model_input_filter=memory_input_filter(memory, user_id=...))。

    mode="auto"（默认）由检索控制器选 L0 / L1 / L2；"l1" / "l2" 强制每次检索，"l0" 只给画像。
    router、lam 同 Memory.search。placement 见模块说明：服务端保存对话状态时用 "instructions"。
    """
    return MemoryInputFilter(
        memory,
        user_id=user_id,
        agent_id=agent_id,
        mode=mode,
        router=router,
        lam=lam,
        placement=placement,
    )


def _with_note(message: Mapping[str, Any], note: str) -> dict[str, Any]:
    """把证据块放在这句用户消息的开头（新的条目，不改原来的）。不用 inject_evidence：这里是
    Responses 格式（内容块是 input_text），这句用户消息也不一定是最后一个条目（后面可能还有
    工具调用）。"""
    content = message.get("content")
    if isinstance(content, list):
        return {**message, "content": [{"type": "input_text", "text": f"{note}\n\n"}, *content]}
    return {**message, "content": f"{note}\n\n{content or ''}"}


def _server_managed(items: Sequence[Any], index: int) -> bool:
    """服务端保存对话状态的迹象：这一轮的工具结果前面没有对应的工具调用（调用在服务端）。"""
    after = [item for item in items[index + 1 :] if isinstance(item, Mapping)]
    calls = {item.get("call_id") for item in after if item.get("type") == "function_call"}
    return any(
        item.get("type") == "function_call_output" and item.get("call_id") not in calls
        for item in after
    )


def _has_tool_results(items: Sequence[Any]) -> bool:
    return any(
        isinstance(item, Mapping) and str(item.get("type", "")).endswith("_output")
        for item in items
    )


def _run_key(space: str, context: Any) -> tuple[str, int | None]:
    """最近一次证据的键：空间，加上这次运行的上下文对象（有的话；按请求新建的上下文区分并发的运行）。"""
    return space, id(context) if context is not None else None


def _history(items: Sequence[Any], index: int) -> int:
    """这句用户消息之前的历史（证据缓存的键：不同的对话里同一位置的同一句话不算同一轮）。"""
    return hash(tuple(_key(item) for item in items[:index]))


# ---------- 会话：短期历史 + 长期记忆 ----------


def session_for(session_id: str) -> str:
    """SDK 的 session_id → CueMem 里第一段会话的 ID（之后各段是 "<ID>~2"、"<ID>~3"……）。
    改过字符或太长时加上哈希，不同的 session_id 不会撞在一起。"""
    safe = _SAFE.sub("_", session_id)
    if safe != session_id or len(safe) > 64:
        safe = f"{safe[:48]}-{hashlib.sha1(session_id.encode()).hexdigest()[:8]}"
    return f"{_PREFIX}{safe}"


@dataclass(frozen=True)
class _Pending:
    key: tuple[str, str]  # （角色，文字）：与内层会话的条目比对
    message: dict[str, Any]  # 写进记忆的 OpenAI 格式消息


@dataclass(frozen=True)
class _Mirrored:
    turn_id: str
    key: tuple[str, str]


class CueMemSession:
    """Agents SDK 的 Session：内层会话保存原样的短期历史，用户与助手的文字消息同时写进 CueMem。

    写入、分段、交出与遗忘见模块说明。cuemem_session_id 是现在写入的那一段，
    cuemem_session_ids() 是这段对话在 CueMem 里的各段。工具调用、推理等条目只进内层会话；
    图片只存 data URL 的（网络图片不下载，与记忆代理相同）。

    内层会话由调用方管理，end() 不关闭它；默认的 SQLiteSession 在内存里，跨请求、跨进程保存短期
    历史要传 inner=SQLiteSession(session_id, "conversations.db") 之类。pop_item 与 clear_session
    要重建派生记忆（与 Memory.delete 相同的代价），比内层的同名操作慢。
    """

    session_settings: SessionSettings | None = None

    def __init__(
        self,
        session_id: str,
        memory: Memory | AsyncMemory,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        inner: Session | None = None,
        idle_minutes: float | None = 30.0,
    ) -> None:
        self.session_id = session_id
        self.memory = _sync(memory)
        self.user_id = user_id
        self.agent_id = agent_id
        self.space = space_for(user_id, agent_id)
        self.inner: Session = inner if inner is not None else SQLiteSession(session_id)
        self.session_settings = getattr(self.inner, "session_settings", None)
        self.idle_minutes = idle_minutes
        self._base = session_for(session_id)
        self._segment: str | None = None
        self._held: list[_Pending] = []  # 等回复的用户消息
        self._answered = False  # 模型回应过它们（工具调用等），只是还没有文字回复
        self._in_turn = False  # 最近写入的是这一轮助手的回复：之后的回复并进这一轮
        self._seen_user = False
        self._mirrored: list[_Mirrored] = []
        self._lock = threading.Lock()
        self._shared = _shared(self.memory.system)

    @property
    def cuemem_session_id(self) -> str:
        """现在写入的 CueMem 会话（这个对象还没写过时是第一段的 ID）。"""
        return self._segment or self._base

    def cuemem_session_ids(self) -> list[str]:
        """这段对话在 CueMem 里的各段会话，从早到晚。"""
        return _segments(self.memory.system, self.space, self._base)

    # ---------- Session 协议 ----------

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        try:
            await self._prune()
        except Exception as exc:  # 同步不了遗忘也照常读历史
            logger.warning("CueMem could not prune session %s (%s).", self.session_id, _err(exc))
            record_event("agents_prune_failed", space=self.space, error=_err(exc))
        return await self.inner.get_items(limit)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        await self.inner.add_items(items)
        try:
            lead = await self._lead(items)
            await asyncio.to_thread(self._take, items, lead)
        except Exception as exc:  # 写不进长期记忆不影响对话（与记忆代理的写回相同）
            logger.warning(
                "CueMem could not store messages of session %s (%s).", self.session_id, _err(exc)
            )
            record_event("agents_mirror_failed", space=self.space, error=_err(exc))

    async def pop_item(self) -> TResponseInputItem | None:
        last = await self.inner.get_items(1)
        key = _key(last[-1]) if last else None
        if key is not None:  # 先在长期记忆里遗忘：失败时抛出，内层不动
            try:
                await asyncio.to_thread(self._forget_item, key)
            except Exception as exc:
                record_event("agents_forget_failed", space=self.space, error=_err(exc))
                raise
        return await self.inner.pop_item()

    async def clear_session(self) -> None:
        try:
            await asyncio.to_thread(self._forget_all)
        except Exception as exc:
            record_event("agents_forget_failed", space=self.space, error=_err(exc))
            raise
        await self.inner.clear_session()

    # ---------- 会话边界 ----------

    async def end(self, *, wait: bool = False) -> dict[str, Any] | None:
        """对话结束：进行中的这一段交给抽取；没有进行中的段、或段里没有消息时返回 None。

        还在等回复的用户消息先处理（模型回应过的写入，没回应过的丢掉）。wait=True 时返回之前
        抽取已经完成（后台线程正好在巩固同一个空间时由它完成，报告里的计数就是 0）。
        async with 退出时自动调用（后台抽取）。之后再有对话写进新的一段。"""
        return await asyncio.to_thread(self._end, wait)

    async def __aenter__(self) -> CueMemSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.end()

    # ---------- 写入（在线程里执行：记忆系统的调用是同步的） ----------

    async def _lead(self, items: Sequence[Any]) -> list[_Pending] | None:
        """这个对象没见过这一轮的用户消息、却先收到助手的回复（例如 RunState 在另一个请求里恢复
        运行）：从内层会话往回找这一轮的用户消息。None：不用找；空列表：前面是同一轮助手的话。"""
        if self._seen_user or self._held or self._in_turn:
            return None
        first = next((p for p in map(_mirrorable, items) if p is not None), None)
        if first is None or first.key[0] != "assistant":
            return None
        tail = await self.inner.get_items(len(items) + _TAIL)
        users: list[_Pending] = []
        for item in reversed(tail[: max(len(tail) - len(items), 0)]):
            pending = _mirrorable(item)
            if pending is None:
                continue
            if pending.key[0] == "assistant":
                return users[::-1]
            users.append(pending)
        return users[::-1] or None  # 往前没有助手的话：不当作同一轮的接续

    def _take(self, items: Sequence[Any], lead: list[_Pending] | None) -> None:
        """一批会话条目：用户消息留着等回复，回复来了与它们一起写入（同一轮）。"""
        system = self.memory.system
        _adopt(system, self.space)
        _sweep(system)
        with self._lock:
            carried = bool(self._held)  # 上一批留下的
            if lead:
                self._held, self._answered = list(lead), True
            elif lead is not None:
                self._in_turn = True
            batch: list[_Pending] = []
            merge = False
            for item in items:
                pending = _mirrorable(item)
                if pending is None:
                    if self._held and _is_output(item):
                        self._answered = True
                    continue
                if pending.key[0] == "user":
                    self._seen_user = True
                    if carried:
                        batch.extend(self._settle())
                        carried = False
                    self._held.append(pending)
                    self._in_turn = False
                    continue
                if not batch and not self._held and self._in_turn:
                    merge = True  # 同一轮里工具调用之后的回复
                batch.extend(self._held)
                batch.append(pending)
                self._held, self._answered, self._in_turn = [], False, True
            if batch:
                self._write(batch, merge=merge)

    def _settle(self) -> list[_Pending]:
        """之前留下的用户消息：模型回应过（只是没有文字回复，例如只调用了工具）就写入；
        没回应过（输入护栏拦下、调用出错）就丢掉，不进长期记忆。"""
        held, answered = self._held, self._answered
        self._held, self._answered = [], False
        if answered or not held:
            return held
        logger.info(
            "CueMem did not store %d user message(s) of session %s: the model never answered.",
            len(held),
            self.session_id,
        )
        record_event("agents_unanswered_dropped", space=self.space, messages=len(held))
        return []

    def _write(self, batch: list[_Pending], *, merge: bool = False) -> None:
        system = self.memory.system
        with self._shared.lock(self.space, self._base):
            segment = self._open_segment()
            out = self.memory.add(
                [p.message for p in batch],
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=segment,
            )
            ids: list[str] = out["turn_ids"]
            if merge and ids:
                _merge_round(system, self.space, segment, ids[0])
            self._mirrored.extend(
                _Mirrored(turn_id, p.key) for p, turn_id in zip(batch, ids, strict=True)
            )
            self._touch(segment)

    def _open_segment(self) -> str:
        """现在写入的段：这个对象的段还在进行就接着写；同一对话的另一个对象（例如按请求新建的）
        开着的段也接着写；否则开新的一段（交出过的段不再写入）。"""
        system = self.memory.system
        busy = system.open_sessions(self.space)
        segment = self._segment
        if segment is None or segment not in busy:
            existing = _segments(system, self.space, self._base)
            if existing and existing[-1] in busy:
                segment = existing[-1]
            else:
                number = _number(existing[-1]) + 1 if existing else 1
                segment = self._base if number == 1 else f"{self._base}~{number}"
            # 进行中：照常写入，但不登记抽取（否则半个会话抽一遍，结束后整个会话又抽一遍）
            system.open_session(self.space, segment)
            self._segment = segment
        self._touch(segment)
        return segment

    def _touch(self, segment: str) -> None:
        limit = self.idle_minutes * 60 if self.idle_minutes is not None else None
        with self._shared.guard:
            self._shared.active[(self.space, segment)] = (_clock(), limit)

    # ---------- 遗忘与结束 ----------

    def _forget_item(self, key: tuple[str, str]) -> None:
        """弹出的消息在长期记忆里遗忘。还在等回复的直接丢掉；本对象写入的按记录找；
        没有记录（例如进程重启过）时按内容找这段对话里最近的同一句。"""
        system = self.memory.system
        with self._lock:
            for k in range(len(self._held) - 1, -1, -1):
                if self._held[k].key == key:
                    del self._held[k]
                    self._answered = self._answered and bool(self._held)
                    return
            index = next(
                (k for k in range(len(self._mirrored) - 1, -1, -1) if self._mirrored[k].key == key),
                None,
            )
            with self._shared.lock(self.space, self._base):
                if index is None:
                    turn_id = self._find(key)
                else:
                    recorded = self._mirrored[index].turn_id
                    gone = not system.db.get_turns(self.space, ids=[recorded])
                    turn_id = None if gone else recorded  # 已经被别处遗忘了：不再找同样的话
                if turn_id is not None:
                    system.forget_memories(self.space, turn_ids=[turn_id])
                    self._shared.forgot(self.space)
            if index is not None:
                del self._mirrored[index]
            self._in_turn = False

    def _find(self, key: tuple[str, str]) -> str | None:
        system = self.memory.system
        for segment in reversed(_segments(system, self.space, self._base)):
            turns = sorted(system.db.get_turns(self.space, segment), key=lambda t: t.idx)
            for turn in reversed(turns):
                if (turn.role, turn.text.strip()) == key:
                    return turn.id
        return None

    def _forget_all(self) -> None:
        """整段对话（各段）在长期记忆里遗忘；之后再写从头开始。"""
        system = self.memory.system
        with self._lock, self._shared.lock(self.space, self._base):
            segments = _segments(system, self.space, self._base)
            if segments:
                system.forget_memories(self.space, session_ids=segments)
                self._shared.forgot(self.space)
            self._held, self._answered, self._in_turn = [], False, False
            self._mirrored.clear()

    def _end(self, wait: bool) -> dict[str, Any] | None:
        system = self.memory.system
        with self._lock:
            batch = self._settle()
            if batch:
                self._write(batch)
            self._in_turn = False
            with self._shared.lock(self.space, self._base):
                busy = system.open_sessions(self.space)
                mine = [*_segments(system, self.space, self._base), self._segment]
                ending = list(dict.fromkeys(s for s in mine if s is not None and s in busy))
                self._segment = None
                with self._shared.guard:
                    for segment in ending:
                        self._shared.active.pop((self.space, segment), None)
            out: dict[str, Any] | None = None
            for segment in ending:
                turns = len(system.db.get_turns(self.space, segment))
                report = system.close_session(self.space, segment, wait=wait)
                if turns:
                    out = {
                        "session_id": segment,
                        "turns": turns,
                        "consolidation": report.as_dict() if report is not None else "queued",
                    }
            return out

    # ---------- 遗忘之后同步短期历史 ----------

    def _forgotten(self) -> list[_Mirrored]:
        """本对象写入、已在长期记忆里被遗忘的消息。"""
        system = self.memory.system
        with self._lock:
            mirrored = list(self._mirrored)
        existing = _existing(system, self.space, [m.turn_id for m in mirrored])
        return [m for m in mirrored if m.turn_id not in existing]

    async def _prune(self) -> None:
        """被删的消息所在的整轮（从用户的一句到下一句之前）从内层去掉。只动尾部：从第一处要删的
        地方起弹出，再把其中保留的放回去；中途出错时也把弹出的、该保留的放回去。"""
        if not self._mirrored:
            return
        gone = await asyncio.to_thread(self._forgotten)
        if not gone:
            return
        items = await self.inner.get_items(_EVERYTHING)
        keep = _keep(items, {m.key for m in gone})
        first = next((k for k, kept in enumerate(keep) if not kept), len(items))
        popped = 0
        try:
            while popped < len(items) - first:
                await self.inner.pop_item()
                popped += 1
        finally:
            restore = [items[k] for k in range(len(items) - popped, len(items)) if keep[k]]
            if restore:
                await self.inner.add_items(restore)
        done = {id(m) for m in gone}
        with self._lock:
            self._mirrored = [m for m in self._mirrored if id(m) not in done]


def _segments(system: MemorySystem, space: str, base: str) -> list[str]:
    """一段 SDK 对话在 CueMem 里的各段会话，从早到晚。"""
    found = [
        sid
        for sid in system.db.get_sessions(space)
        if sid == base or (sid.startswith(f"{base}~") and sid[len(base) + 1 :].isdigit())
    ]
    return sorted(found, key=_number)


def _number(segment: str) -> int:
    _, mark, number = segment.rpartition("~")
    return int(number) if mark and number.isdigit() else 1


def _merge_round(system: MemorySystem, space: str, segment: str, first_id: str) -> None:
    """同一轮里工具调用之后的回复并进这一轮：上一轮与新写入的"轮"换成按现存的发言重建的
    （writer.replace_rounds：先向量化，出错时原来的"轮"都还在）。"""
    turns = sorted(system.db.get_turns(space, segment), key=lambda t: t.idx)
    position = next((k for k, t in enumerate(turns) if t.id == first_id), 0)
    if position == 0:
        return
    touched = {turns[position - 1].id} | {t.id for t in turns[position:]}
    stale = [
        u.id
        for u in system.db.list_units(space, kinds=["round"], session_ids=[segment])
        if touched & set(u.source_turn_ids)
    ]
    system.writer.replace_rounds(space, segment, stale)


def _existing(system: MemorySystem, space: str, turn_ids: Sequence[str]) -> set[str]:
    found: set[str] = set()
    for start in range(0, len(turn_ids), 500):
        chunk = list(turn_ids[start : start + 500])
        found |= {t.id for t in system.db.get_turns(space, ids=chunk)}
    return found


# ---------- 条目 ----------


def _sync(memory: Memory | AsyncMemory) -> Memory:
    return memory.sync if isinstance(memory, AsyncMemory) else memory


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _message(item: Any) -> Mapping[str, Any] | None:
    """用户或助手的消息条目；工具调用、推理、系统消息等返回 None。"""
    if not isinstance(item, Mapping) or item.get("role") not in ("user", "assistant"):
        return None
    return item if item.get("type", "message") == "message" else None


def _is_output(item: Any) -> bool:
    """模型产出的条目（工具调用与结果、推理、没有文字的助手消息等）：说明模型回应过这一轮。"""
    if not isinstance(item, Mapping):
        return False
    if item.get("type", "message") != "message":
        return True
    return item.get("role") == "assistant"


def _latest_user(items: Sequence[Any]) -> tuple[int, Mapping[str, Any]] | None:
    for index in range(len(items) - 1, -1, -1):
        message = _message(items[index])
        if message is not None and message["role"] == "user":
            return index, message
    return None


def _text(content: Any) -> str:
    """内容（字符串或内容块）里的文字；图片、文件等不算。不用 message_text：Responses 格式的文字块
    是 input_text / output_text，不只是 text。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part["text"])
            for part in content
            if isinstance(part, Mapping) and part.get("type") in _TEXT_PARTS and part.get("text")
        )
    return ""


def _key(item: Any) -> tuple[str, str] | None:
    message = _message(item)
    if message is None:
        return None
    text = _text(message.get("content")).strip()
    return (str(message["role"]), text) if text else None


def _mirrorable(item: Any) -> _Pending | None:
    """会话条目 → 要写进记忆的消息（OpenAI 格式）。没有文字的消息不写。"""
    key = _key(item)
    message = _message(item)
    if key is None or message is None:
        return None
    content = message.get("content")
    if isinstance(content, str):
        return _Pending(key, {"role": key[0], "content": content})
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if not isinstance(part, Mapping):
            continue
        if part.get("type") in _TEXT_PARTS and part.get("text"):
            blocks.append({"type": "text", "text": str(part["text"])})
        elif part.get("type") == "input_image" and str(part.get("image_url", "")).startswith(
            "data:"
        ):
            blocks.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
    return _Pending(key, {"role": key[0], "content": blocks})


def _keep(items: Sequence[Any], gone: Collection[tuple[str, str]]) -> list[bool]:
    """每个条目要不要留下：含有被遗忘消息的整轮都去掉（用户的话或助手的回复被删，都整轮去掉，
    与 Agent.prune 相同）。"""
    rounds: list[list[int]] = []
    for index, item in enumerate(items):
        key = _key(item)
        if not rounds or (key is not None and key[0] == "user"):
            rounds.append([index])
        else:
            rounds[-1].append(index)
    keep = [True] * len(items)
    for group in rounds:
        if any(_key(items[k]) in gone for k in group):
            for k in group:
                keep[k] = False
    return keep
