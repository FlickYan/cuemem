"""OpenAI Agents SDK 集成（N12）：记忆工具、调用模型前先过检索控制器、对话写进长期记忆。

    from agents import Agent, RunConfig, Runner
    from cuemem import Memory
    from cuemem.integrations.openai_agents import CueMemSession, memory_input_filter, memory_tools

    memory = Memory(preset="deepseek")
    agent = Agent(name="Assistant", model=..., tools=memory_tools(memory, user_id="alice"))
    config = RunConfig(call_model_input_filter=memory_input_filter(memory, user_id="alice"))
    async with CueMemSession("chat-1", memory, user_id="alice") as session:
        await Runner.run(agent, "I moved to Lisbon", session=session, run_config=config)

三样东西，可以分开用（需要 openai-agents，可选依赖 cuemem[openai-agents]）：

- memory_tools：recall / remember / forget 三个工具，名字、说明与语义与 MCP（api/mcp.py）相同；
- memory_input_filter：交给 RunConfig.call_model_input_filter，SDK 每次调用模型之前把输入给它。
  对最新一句用户消息先跑检索控制器：L0 不检索、只给画像（没有画像就什么都不加），L1 / L2 取回
  事实与对话片段。证据块与记忆代理（api/proxy.py）同一个格式、同一个位置：放在这句用户消息的开头。
  只改这次发给模型的副本，会话里存的历史不动；同一轮里工具调用之后的再次调用沿用这一轮的证据
  （遗忘之后重新检索），不重复跑控制器。进行中的会话里、原样已在输入中的轮不参与检索；
- CueMemSession：Session 协议的实现。内层会话（默认内存里的 SQLiteSession）原样保存短期历史；
  用户与助手的文字消息同时写进 CueMem 里同名的会话（在线写入、立即可检索；进行中不抽取事实），
  end() 时交给巩固（与记忆代理、Agent 的会话边界相同，D-045）。

遗忘（D-021）：pop_item 撤回的消息、clear_session 清掉的整段对话，在长期记忆里也一并遗忘
（删原始日志、重建派生记忆，有转述的话连带删掉）；反过来，用 forget 工具或 Memory.delete 删掉的
发言，下次读历史时也从内层会话里去掉所在的整轮（与 Agent 的 prune 相同）。
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from types import TracebackType
from typing import Any, cast

from agents import (
    Session,
    SessionSettings,
    SQLiteSession,
    Tool,
    TResponseInputItem,
    function_tool,
)
from agents.run_config import CallModelData, ModelInputData

from cuemem.api.mcp import format_recall
from cuemem.api.proxy import MEMORY_NOTE
from cuemem.api.service import _status
from cuemem.core.trace import record_event
from cuemem.memory import ForgetReport, MemorySystem, RecallMode
from cuemem.read.controller import Router, english_reason
from cuemem.read.reader import memory_blocks
from cuemem.sdk import AsyncMemory, Memory, space_for

__all__ = [
    "CueMemSession",
    "Evidence",
    "MemoryInputFilter",
    "memory_input_filter",
    "memory_tools",
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

_TEXT_PARTS = ("input_text", "output_text", "text")
_SESSION_ID = re.compile(r"[^A-Za-z0-9_.:-]")
_EVERYTHING = 2**62  # 读全部历史时显式给的上限：limit=None 会被内层的 session_settings.limit 截断

# 每个记忆系统的遗忘代数：经这里遗忘之后，自动回忆缓存的证据作废
_generations: weakref.WeakKeyDictionary[MemorySystem, int] = weakref.WeakKeyDictionary()


# ---------- 工具 ----------


def memory_tools(
    memory: Memory | AsyncMemory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
) -> list[Tool]:
    """recall / remember / forget 三个工具（与 MCP 同名同义），作用于 user_id、agent_id 的记忆空间。

    recall 至少做一次快速召回（模型已经决定要查），控制器只决定要不要查得更深；结果原样交给模型
    去推理，这里不调用回答模型。画像不单独做工具：memory_input_filter 每一轮都会带上。
    返回的都是 FunctionTool（类型标成 Tool，好直接交给 Agent(tools=...)）。
    """
    sync = _sync(memory)
    space = space_for(user_id, agent_id)

    @function_tool(name_override="recall", description_override=RECALL_DESCRIPTION)
    def recall(query: str, deep: bool = False) -> str:
        return format_recall(_recall(sync.system, space, query, deep=deep))

    @function_tool(name_override="remember", description_override=REMEMBER_DESCRIPTION)
    def remember(text: str) -> str:
        when = datetime.now().replace(microsecond=0)
        note = f"note-{when:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        out = sync.add(text, user_id=user_id, agent_id=agent_id, run_id=note, timestamp=when)
        if not out["turn_ids"]:
            return "Nothing was saved: the text is empty."
        return f"Saved as {out['turn_ids'][0]}."

    @function_tool(name_override="forget", description_override=FORGET_DESCRIPTION)
    def forget(ids: list[str]) -> str:
        report = _forget_ids(sync.system, space, ids)
        return (
            f"Deleted {report.turns_deleted} conversation turn(s); "
            f"facts {report.facts_before} -> {report.facts_after}."
        )

    return [recall, remember, forget]


def _recall(system: MemorySystem, space: str, query: str, *, deep: bool) -> dict[str, Any]:
    """MCP recall 的检索部分（ChatService.recall，at_least_l1）：控制器选了 L0 时改走 L1。"""
    recall = system.retrieve(space, query, mode="l2" if deep else "auto")
    raised = recall.level == "l0"
    if raised:
        recall = system.retrieve(space, query, mode="l1")
    view, decision = recall.view, recall.decision
    reasons = (
        ["raised from L0: the caller asked to recall"]
        if raised
        else [english_reason(r) for r in decision.reasons]
        if decision
        else []
    )
    return {
        "level": recall.level.upper(),
        "reasons": reasons,
        "profile": view.core.text if view.core is not None else "",
        "facts": [
            {
                "id": fh.fact.id,
                "statement": fh.fact.statement,
                "said_on": fh.fact.recorded_at.isoformat(),
                "status": _status(fh.fact),
            }
            for fh in view.facts
        ],
        "excerpts": [
            {"id": hit.unit.id, "ts": hit.unit.ts.isoformat(), "text": hit.unit.text}
            for hit in sorted(view.rounds, key=lambda h: h.unit.ts)
        ],
    }


def _forget_ids(system: MemorySystem, space: str, ids: Sequence[str]) -> ForgetReport:
    """按 recall 给出的 ID 遗忘（ChatService.forget_ids）：事实 ID，或对话片段（轮）的 ID。"""
    fact_ids = [f.id for f in system.db.get_facts(space, ids=list(ids))]
    rounds = system.db.get_units(space, list(ids))
    turns = [t for unit in rounds if unit.kind == "round" for t in unit.source_turn_ids]
    report = system.forget_memories(space, fact_ids=fact_ids, turn_ids=turns)
    _bump(system)
    return report


# ---------- 调用模型之前：自动回忆 ----------


@dataclass(frozen=True)
class Evidence:
    """一次自动回忆的结果。"""

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
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: RecallMode = "auto",
        router: Router = "learned",
        lam: float | None = None,
    ) -> None:
        self.memory = _sync(memory)
        self.space = space_for(user_id, agent_id)
        self.mode: RecallMode = mode
        self.router: Router = router
        self.lam = lam
        self.last: Evidence | None = None
        self._cache: OrderedDict[tuple[int, int, str], Evidence] = OrderedDict()
        self._lock = threading.Lock()

    async def __call__(self, data: CallModelData[Any]) -> ModelInputData:
        items = data.model_data.input
        found = _latest_user(items)
        if found is None:
            return data.model_data
        index, message = found
        query = _text(message.get("content")).strip()
        if not query:
            return data.model_data
        # 同一轮里（工具调用之后再调用模型）这句话的位置不变：沿用这一轮的证据
        key = (_generations.get(self.memory.system, 0), index, query)
        with self._lock:
            evidence = self._cache.get(key)
        if evidence is None:
            present = {_text(m.get("content")).strip() for m in _messages(items)}
            try:
                evidence = await asyncio.to_thread(self._recall, query, present)
            except Exception as exc:  # 记忆出错不挡住对话：这一次不带记忆
                logger.warning(
                    "CueMem recall failed (%s: %s); calling the model without memories.",
                    type(exc).__name__,
                    exc,
                )
                return data.model_data
            with self._lock:
                self._cache[key] = evidence
                while len(self._cache) > 64:
                    self._cache.popitem(last=False)
        self.last = evidence
        if not evidence.block:
            return data.model_data
        injected = list(items)
        injected[index] = cast(TResponseInputItem, _with_block(message, evidence.block))
        return ModelInputData(input=injected, instructions=data.model_data.instructions)

    def _recall(self, query: str, present: Collection[str]) -> Evidence:
        """与记忆代理的 _recall 相同：控制器选档、排出证据块；L0 只给画像。"""
        system = self.memory.system
        if not system.db.get_sessions(self.space):
            return Evidence(query, "none", ["no memories yet"], 0, "")
        now = datetime.now()
        recall = system.retrieve(
            self.space,
            query,
            question_time=now,
            mode=self.mode,
            router=self.router,
            lam=self.lam,
            exclude=_in_context(system, self.space, present),
        )
        reasons = [english_reason(r) for r in recall.decision.reasons] if recall.decision else []
        view = recall.view
        if recall.level == "l0":  # 不需要回忆：只给画像（没有画像就什么都不加）
            view = replace(view, rounds=[], facts=[], attach=[])
        blocks, refs, _ = memory_blocks(view, now, relative_time=True)
        if not view.rounds:  # 没取到对话片段时不留一个空的"(none)"块
            blocks = blocks[:-1]
        has_profile = view.core is not None and bool(view.core.text)
        recalled = len(refs) + int(has_profile)
        block = "\n\n".join([MEMORY_NOTE, *blocks]) if blocks else ""
        level = recall.level.upper()
        record_event("agents_recall", space=self.space, level=level, recalled=recalled)
        return Evidence(query, level, reasons, recalled, block)


def memory_input_filter(
    memory: Memory | AsyncMemory,
    *,
    user_id: str | None = None,
    agent_id: str | None = None,
    mode: RecallMode = "auto",
    router: Router = "learned",
    lam: float | None = None,
) -> MemoryInputFilter:
    """RunConfig(call_model_input_filter=memory_input_filter(memory, user_id=...))。

    mode="auto"（默认）由检索控制器选 L0 / L1 / L2；"l1" / "l2" 强制每次检索，"l0" 只给画像。
    router、lam 同 Memory.search。
    """
    return MemoryInputFilter(
        memory, user_id=user_id, agent_id=agent_id, mode=mode, router=router, lam=lam
    )


def _in_context(system: MemorySystem, space: str, present: Collection[str]) -> list[str]:
    """进行中的会话里、原样已在输入中的轮（检索时排除，名额留给输入里没有的）。按内容比对，
    与记忆代理相同：调用方可能删减了历史，不能假定写进去的都还在。"""
    sessions = sorted(system.open_sessions(space))
    if not sessions or not present:
        return []
    turns = {t.id: t for sid in sessions for t in system.db.get_turns(space, sid)}
    rounds = system.db.list_units(space, kinds=["round"], session_ids=sessions)
    return [
        u.id
        for u in rounds
        if all(t in turns and turns[t].text.strip() in present for t in u.source_turn_ids)
    ]


def _with_block(message: Mapping[str, Any], block: str) -> dict[str, Any]:
    """把证据块放在这句用户消息的开头（新的条目，不改原来的）。"""
    note = f"<memory>\n{block}\n</memory>\n\n"
    content = message.get("content")
    if isinstance(content, list):
        return {**message, "content": [{"type": "input_text", "text": note}, *content]}
    return {**message, "content": note + (content or "")}


# ---------- 会话：短期历史 + 长期记忆 ----------


@dataclass(frozen=True)
class _Mirrored:
    turn_id: str
    key: tuple[str, str]  # （角色，文字）


class CueMemSession:
    """Agents SDK 的 Session：内层会话保存原样的短期历史，用户与助手的文字消息同时写进 CueMem。

    一个 Agents SDK 会话对应 CueMem 里同名的一个会话（cuemem_session_id）。第一次写入时标记为
    进行中：每轮在线写入（不调用 LLM、立即可检索），但不抽取事实；end()（或 async with 结束时）
    交给巩固。会话结束后还可以接着用，同一个 CueMem 会话重新进行，下次 end() 时整体重新抽取。
    工具调用、推理等条目只进内层会话；图片只存 data URL 的（网络图片不下载，与记忆代理相同）。

    - pop_item：内层弹出最后一条；它是写进记忆的消息时，在长期记忆里遗忘这一句；
    - clear_session：清空内层；CueMem 里的这个会话整体遗忘；
    - get_items：先把长期记忆里已被遗忘的消息所在的整轮从内层去掉（只认本对象写入的消息），再读。

    遗忘要重建派生记忆（与 Memory.delete 相同的代价），所以 pop_item 与 clear_session 比内层的
    同名操作慢。内层会话由调用方管理，end() 不关闭它。
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
    ) -> None:
        self.session_id = session_id
        self.memory = _sync(memory)
        self.user_id = user_id
        self.agent_id = agent_id
        self.space = space_for(user_id, agent_id)
        self.inner: Session = inner if inner is not None else SQLiteSession(session_id)
        self.session_settings = getattr(self.inner, "session_settings", None)
        self.cuemem_session_id = _SESSION_ID.sub("_", session_id)[:80] or "agents-session"
        self._mirrored: list[_Mirrored] = []
        self._dirty = False  # 上次 end() 之后写过消息
        self._lock = threading.Lock()

    # ---------- Session 协议 ----------

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        await self._prune()
        return await self.inner.get_items(limit)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        await self.inner.add_items(items)
        pending = [m for m in (_mirrorable(item) for item in items) if m is not None]
        if not pending:
            return
        try:
            await asyncio.to_thread(self._mirror, pending)
        except Exception as exc:  # 写不进长期记忆不影响对话（与记忆代理的写回相同）
            logger.warning(
                "CueMem could not store messages of session %s (%s: %s).",
                self.session_id,
                type(exc).__name__,
                exc,
            )
            record_event("agents_mirror_failed", error=f"{type(exc).__name__}: {exc}")

    async def pop_item(self) -> TResponseInputItem | None:
        item = await self.inner.pop_item()
        key = _key(item) if item is not None else None
        if key is not None:
            await asyncio.to_thread(self._forget_popped, key)
        return item

    async def clear_session(self) -> None:
        await self.inner.clear_session()
        await asyncio.to_thread(self._forget_session)

    # ---------- 会话边界 ----------

    async def end(self, *, wait: bool = False) -> dict[str, Any] | None:
        """对话结束：CueMem 里的这个会话交给巩固；会话里还没有消息时什么都不做，返回 None。

        wait=True 时返回之前抽取已经完成（后台线程正好在巩固同一个空间时由它完成，报告里的
        计数就是 0）。async with 退出时，这之后写过消息就自动调用一次（后台抽取）。"""
        return await asyncio.to_thread(self._end, wait)

    async def __aenter__(self) -> CueMemSession:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._dirty:
            await self.end()

    # ---------- 内部（在线程里执行：记忆系统的调用是同步的） ----------

    def _mirror(self, pending: list[tuple[tuple[str, str], dict[str, Any]]]) -> None:
        system = self.memory.system
        with self._lock:
            # 进行中：照常写入，但不登记抽取（否则半个会话抽一遍，结束后整个会话又抽一遍）
            system.open_session(self.space, self.cuemem_session_id)
            out = self.memory.add(
                [message for _, message in pending],
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.cuemem_session_id,
            )
            self._mirrored.extend(
                _Mirrored(turn_id, key)
                for (key, _), turn_id in zip(pending, out["turn_ids"], strict=True)
            )
            self._dirty = True

    def _forget_popped(self, key: tuple[str, str]) -> None:
        """弹出的消息在长期记忆里遗忘：本对象写入的按记录找；否则（例如进程重启过）按内容找
        这个会话里最近的同一句。"""
        system = self.memory.system
        with self._lock:
            turns = system.db.get_turns(self.space, self.cuemem_session_id)
            existing = {t.id for t in turns}
            turn_id = None
            for k in range(len(self._mirrored) - 1, -1, -1):
                if self._mirrored[k].key == key:
                    turn_id = self._mirrored.pop(k).turn_id
                    break
            if turn_id is None:
                turn_id = next(
                    (t.id for t in reversed(turns) if (t.role, t.text.strip()) == key), None
                )
            if turn_id is None or turn_id not in existing:
                return
            system.forget_memories(self.space, turn_ids=[turn_id])
            _bump(system)

    def _forget_session(self) -> None:
        system = self.memory.system
        with self._lock:
            self._mirrored.clear()
            if system.db.get_turns(self.space, self.cuemem_session_id):
                system.forget_memories(self.space, session_ids=[self.cuemem_session_id])
                _bump(system)

    def _end(self, wait: bool) -> dict[str, Any] | None:
        system = self.memory.system
        with self._lock:
            self._dirty = False
            turns = system.db.get_turns(self.space, self.cuemem_session_id)
            if not turns:
                return None
            report = system.close_session(self.space, self.cuemem_session_id, wait=wait)
        return {
            "session_id": self.cuemem_session_id,
            "turns": len(turns),
            "consolidation": report.as_dict() if report is not None else "queued",
        }

    def _forgotten(self) -> set[tuple[str, str]]:
        """本对象写入、已在长期记忆里被遗忘的消息。"""
        system = self.memory.system
        with self._lock:
            if not self._mirrored:
                return set()
            existing = {t.id for t in system.db.get_turns(self.space, self.cuemem_session_id)}
            gone = {m.key for m in self._mirrored if m.turn_id not in existing}
            self._mirrored = [m for m in self._mirrored if m.turn_id in existing]
            return gone

    async def _prune(self) -> None:
        """遗忘之后同步短期历史：被删的消息所在的整轮（从用户的一句到下一句之前）从内层去掉。
        只动尾部：从第一处要删的地方起弹出，再把其中保留的放回去。"""
        if not self._mirrored:
            return
        gone = await asyncio.to_thread(self._forgotten)
        if not gone:
            return
        items = await self.inner.get_items(_EVERYTHING)
        kept = _drop_rounds(items, gone)
        first = next(
            (k for k, (a, b) in enumerate(zip(items, kept, strict=False)) if a is not b),
            len(kept),
        )
        for _ in range(len(items) - first):
            await self.inner.pop_item()
        if kept[first:]:
            await self.inner.add_items(kept[first:])


# ---------- 条目 ----------


def _sync(memory: Memory | AsyncMemory) -> Memory:
    return memory.sync if isinstance(memory, AsyncMemory) else memory


def _bump(system: MemorySystem) -> None:
    _generations[system] = _generations.get(system, 0) + 1


def _message(item: Any) -> Mapping[str, Any] | None:
    """用户或助手的消息条目；工具调用、推理、系统消息等返回 None。"""
    if not isinstance(item, Mapping) or item.get("role") not in ("user", "assistant"):
        return None
    return item if item.get("type", "message") == "message" else None


def _messages(items: Sequence[Any]) -> list[Mapping[str, Any]]:
    return [m for m in (_message(item) for item in items) if m is not None]


def _latest_user(items: Sequence[Any]) -> tuple[int, Mapping[str, Any]] | None:
    for index in range(len(items) - 1, -1, -1):
        message = _message(items[index])
        if message is not None and message["role"] == "user":
            return index, message
    return None


def _text(content: Any) -> str:
    """内容（字符串或内容块）里的文字；图片、文件等不算。"""
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


def _mirrorable(item: Any) -> tuple[tuple[str, str], dict[str, Any]] | None:
    """会话条目 → （比对用的键，写进记忆的 OpenAI 格式消息）。没有文字的消息不写。"""
    key = _key(item)
    message = _message(item)
    if key is None or message is None:
        return None
    content = message.get("content")
    if isinstance(content, str):
        return key, {"role": key[0], "content": content}
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
    return key, {"role": key[0], "content": blocks}


def _drop_rounds(items: list[TResponseInputItem], gone: set[tuple[str, str]]) -> list[Any]:
    """去掉含有被遗忘消息的整轮：用户的话或助手的回复被删，都整轮去掉（与 Agent.prune 相同）。"""
    rounds: list[list[TResponseInputItem]] = []
    for item in items:
        key = _key(item)
        if not rounds or (key is not None and key[0] == "user"):
            rounds.append([item])
        else:
            rounds[-1].append(item)
    return [item for r in rounds if not any(_key(i) in gone for i in r) for item in r]
