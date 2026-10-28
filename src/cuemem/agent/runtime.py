"""Agent Runtime（N8）：自研的对话循环，把记忆系统接进一个真正的多轮 Agent。

每条用户消息：

1. **想起**：检索控制器（N7）用探针决定 L0 / L1 / L2 与要不要再看原图，取回记忆；
   当前会话最近几轮已经原样在上下文里（工作记忆），检索时排除，名额留给上下文里没有的；
2. **装配**：按分区预算组装请求，稳定前缀在前、易变的召回结果在最后（agent/context.py）；
3. **回答**：模型可以调用工具——memory_search 是逃生通道（控制器没取到的自己再查），
   领域工具（例如商品检索）由场景提供；回答里用 [f2]、[m3] 引用记忆；
4. **写回**：用户的话（含图片）与回答在线写入原始日志并建立索引（不调用 LLM）；
   会话进行中不巩固，结束时整个会话交给后台抽取事实、更新画像（agent 关掉会话即可）。

每一轮返回一份轨迹：控制器的决定与理由、各分区 token、召回与引用的记忆、工具调用、
用量（含供应商前缀缓存命中的输入 token）与各阶段耗时——Demo 的"记忆轨迹"面板直接展示它。
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from cuemem.agent.context import (
    ContextBudget,
    Exchange,
    Layout,
    build_context,
    citations,
    evict,
    strip_citations,
)
from cuemem.agent.tools import Tool, ToolContext, ToolResult, memory_tools
from cuemem.core.ids import sortable_session_id, stable_id
from cuemem.core.models import SessionIn, TurnIn
from cuemem.core.timeutil import naive_local
from cuemem.core.trace import Trace, span, start_trace
from cuemem.llm.client import Message
from cuemem.memory import MemorySystem, Recall, RecallMode
from cuemem.read.agent import SearchSession
from cuemem.read.controller import Router
from cuemem.read.facts import fact_status
from cuemem.read.l1 import L1Config
from cuemem.store.media import check_image
from cuemem.write.consolidate import ConsolidationReport, scope
from cuemem.write.online import media_id

AGENT_RULES = """How your memory works:
- "About the user" below is the user's profile from earlier conversations: current facts, with ids [p1], [p2], …
- With each user message, the memory system may attach recalled memories: facts [f1] with when they were said and whether they are still current, conversation excerpts [m1] with their dates, and images [i1] described in words (sometimes with the original picture attached). The headers say how long before "the question" each memory is; the question is the user's current message.
- Use the profile and recalled memories when they help and ignore the rest. Whenever you use one, cite its id in square brackets right after the sentence, like [p2], [f2] or [m3].
- If a memory was superseded, rely on the newer one, and mention the change when it matters.
- If you need something from past conversations that was not recalled, call memory_search with a short query. Do not search for what is already in this conversation.
- Never invent things about the user. If you don't know, say so or ask.
- Answer the user's latest message directly. Reply in the user's language; be concise and friendly."""  # noqa: E501


@dataclass(frozen=True)
class Persona:
    """Agent 的角色与领域说明（系统提示里不变的部分）。"""

    name: str = "Cue"
    role: str = "a personal assistant with long-term memory"
    instructions: str = ""

    def prompt(self) -> str:
        text = f"You are {self.name}, {self.role}.\n\n{AGENT_RULES}"
        if self.instructions:
            text += f"\n\n{self.instructions}"
        return text


class ToolRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    result: str  # 给模型看的结果（截断）
    ms: float
    data: dict[str, Any] = Field(default_factory=dict)


class RecalledItem(BaseModel):
    """本轮给模型看过的一条记忆。"""

    ref: str  # p1 / f1 / m1 / i1
    kind: Literal["profile", "fact", "excerpt", "image"]
    id: str  # 事实 ID、轮的 ID 或图片 ID
    text: str
    when: datetime | None = None
    status: str = ""  # 事实的有效状态：current / no longer true … / corrected …
    session_id: str | None = None
    path: str | None = None  # 图片的本地路径
    via: Literal["controller", "memory_search"] = "controller"
    attached: bool = False  # 原图附给了模型（再感知）
    cited: bool = False


class TurnTrace(BaseModel):
    trace_id: str
    level: str  # L0 / L1 / L2
    see: bool = False
    router: str | None = None
    reasons: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    partitions: dict[str, int] = Field(default_factory=dict)
    budget: dict[str, int] = Field(default_factory=dict)
    recalled: list[RecalledItem] = Field(default_factory=list)
    tools: list[ToolRecord] = Field(default_factory=list)
    usage: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    deep: dict[str, Any] = Field(default_factory=dict)  # L2 的检索计划与充分性检查
    window: int = 0  # 工作记忆里的轮数（本轮之后）
    evicted: int = 0  # 本轮之后淘汰出工作记忆的轮数
    prefix_messages: int = 0  # 稳定前缀的消息条数


class Reply(BaseModel):
    session_id: str
    at: datetime
    text: str  # 回答原文（含引用编号）
    plain: str  # 去掉引用编号（写入记忆、放进工作记忆的版本）
    citations: list[str] = Field(default_factory=list)
    products: list[dict[str, Any]] = Field(default_factory=list)  # 领域工具返回的商品等
    trace: TurnTrace


@dataclass
class ChatSession:
    """一次对话（一个会话）。时间可以模拟：给定开始时间后，会话内的时间按真实流逝往前走。"""

    space_id: str
    session_id: str
    started_at: datetime
    # 会话开始时的画像快照（稳定前缀的一部分）：（编号 p1，事实 ID，陈述）
    profile: list[tuple[str, str, str]] = field(default_factory=list)
    simulated: bool = False
    next_idx: int = 0  # 下一句发言的序号（只增不减：遗忘删掉的序号不再复用）
    window: list[Exchange] = field(default_factory=list)
    uploads: dict[str, str] = field(default_factory=dict)  # u1 → 本地路径（存进记忆后是那份）
    upload_turns: dict[str, int] = field(default_factory=dict)  # u1 → 所在发言的序号
    closed: bool = False
    _clock: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def now(self) -> datetime:
        """当前时间：真实会话就是现在；模拟的会话从给定的开始时间起，按真实流逝往前走。"""
        if not self.simulated:  # 不截到秒：与 SDK、各个集成的写入在同一秒里也分得出先后
            return datetime.now()
        return self.started_at + timedelta(seconds=round(time.monotonic() - self._clock))

    @property
    def last_at(self) -> datetime | None:
        return self.window[-1].at if self.window else None


class Agent:
    def __init__(
        self,
        system: MemorySystem,
        *,
        persona: Persona | None = None,
        tools: Sequence[Tool] = (),
        budget: ContextBudget | None = None,
        role: str = "agent",
        router: Router = "learned",
        lam: float | None = None,
        max_steps: int = 4,
        layout: Layout = "stable",
    ) -> None:
        self.layout: Layout = layout  # 上下文布局；"naive" 只用于对照实验（scripts/n8_context.py）
        self.system = system
        self.persona = persona or Persona()
        self.tools = {t.name: t for t in [*memory_tools(), *tools]}
        self.budget = budget or ContextBudget()
        self.role = role
        self.router: Router = router
        self.lam = lam
        self.max_steps = max_steps

    # ---------- 会话 ----------

    def start(
        self, space_id: str, *, at: datetime | None = None, session_id: str | None = None
    ) -> ChatSession:
        """开始一个会话：标记为进行中（暂不巩固），取一次核心记忆快照。
        at（模拟的开始时间）带时区时换算成本机时间。"""
        started = naive_local(at).replace(microsecond=0) if at is not None else datetime.now()
        sid = session_id or sortable_session_id("chat", started)
        self.system.open_session(space_id, sid)
        return ChatSession(
            space_id=space_id,
            session_id=sid,
            started_at=started,
            profile=self._profile(space_id),
            simulated=at is not None,
        )

    def _profile(self, space_id: str) -> list[tuple[str, str, str]]:
        """画像（核心记忆）里的事实，按画像里的顺序编号 p1、p2……，回答时可以引用。"""
        core = self.system.core_memory(space_id)
        if core is None or not core.fact_ids:
            return []
        facts = {f.id: f for f in self.system.db.get_facts(space_id, ids=core.fact_ids)}
        kept = [fid for fid in core.fact_ids if fid in facts]
        return [(f"p{i}", fid, facts[fid].statement) for i, fid in enumerate(kept, start=1)]

    def end(self, session: ChatSession, *, wait: bool = False) -> ConsolidationReport | None:
        """结束会话：整个会话交给巩固（抽取事实、对账、更新画像）。wait=True 时同步完成。"""
        with session.lock:
            if session.closed:
                return None
            session.closed = True
        if session.next_idx == 0:
            self.system.close_session(session.space_id, session.session_id)
            return None
        return self.system.close_session(session.space_id, session.session_id, wait=wait)

    def prune(self, session: ChatSession) -> int:
        """遗忘之后同步会话状态：工作记忆里被删的轮去掉，图片编号作废，核心记忆快照重取
        （画像是按剩下的日志重建的）。返回去掉的轮数。

        用户的话或助手的回复被删，都整轮去掉：转述只删回复（D-051），留下一句没有回复的问话
        在上下文里没有意义。"""
        with session.lock:
            space, sid = session.space_id, session.session_id
            existing = {t.id for t in self.system.db.get_turns(space, sid)}
            kept = [
                e
                for e in session.window
                if stable_id(space, sid, e.user_idx) in existing
                and stable_id(space, sid, e.user_idx + 1) in existing
            ]
            removed = len(session.window) - len(kept)
            session.window = kept
            for label, idx in list(session.upload_turns.items()):
                if stable_id(space, sid, idx) not in existing:
                    session.uploads.pop(label, None)
                    session.upload_turns.pop(label, None)
            session.profile = self._profile(space)
            return removed

    # ---------- 一轮对话 ----------

    def chat(
        self,
        session: ChatSession,
        text: str,
        *,
        images: Sequence[str | Path] = (),
        at: datetime | None = None,
        mode: RecallMode = "auto",
        router: Router | None = None,
        lam: float | None = None,
    ) -> Reply:
        with session.lock:
            if session.closed:
                raise ValueError(f"会话 {session.session_id} 已经结束")
            # 像素太多的图片（解压炸弹）在召回、解码与调用模型之前就拒绝（ValueError，/api 是
            # 409）。以前先解码、发给模型，写进记忆时才被拒绝：回答丢了、钱照付；超过两倍上限的是
            # Pillow 的 DecompressionBombError（500）
            for path in images:
                check_image(path)
            return self._chat(session, text, [str(p) for p in images], at, mode, router, lam)

    def _chat(
        self,
        session: ChatSession,
        text: str,
        images: list[str],
        at: datetime | None,
        mode: RecallMode,
        router: Router | None,
        lam: float | None,
    ) -> Reply:
        space = session.space_id
        when = naive_local(at) if at is not None else session.now()
        if session.last_at is not None and when < session.last_at:
            when = session.last_at  # 时间不倒流
        # 回答缓存按会话归属（N11）：遗忘只清受影响会话的缓存。整个空间共用一个归属时，删一条事实
        # 就会清掉所有会话的回答缓存，剧本回放的其他会话也得重新调用模型，检查时好时坏。
        # 缓存只存请求的哈希与回答；复述了被删信息的回答会被遗忘找出来（D-051），所在会话随之清理
        cache_scope = scope(space, f"session:{session.session_id}")
        started = time.perf_counter()
        first = len(session.uploads) + 1
        labels = [f"u{first + i}" for i in range(len(images))]
        uploads = list(zip(labels, images, strict=True))
        timings: dict[str, float] = {}
        with (
            self.system.reading(space),  # 遗忘等这一轮做完：回复写进日志、溯源记下之后才清理
            start_trace(
                "chat", self.system.tracer, space=space, session=session.session_id, message=text
            ) as trace,
        ):
            exclude = self._window_ids(session)
            l1 = L1Config(budget_tokens=self.budget.recalled)
            with span("recall"):
                recall = self.system.retrieve(
                    space,
                    text,
                    question_time=when,
                    mode=mode,
                    router=router or self.router,
                    lam=lam if lam is not None else self.lam,
                    l1=l1,
                    exclude=exclude,
                    cache_scope=cache_scope,
                )
            timings["recall"] = _ms(started)
            schemas = [t.schema() for t in self.tools.values()]
            ctx = build_context(
                persona=self.persona.prompt(),
                profile=session.profile,
                window=session.window,
                text=text,
                when=when,
                view=recall.view,
                uploads=uploads,
                budget=self.budget,
                tools=schemas,
                layout=self.layout,
            )
            memory = SearchSession(
                question_time=when,
                search=lambda query, cfg: self.system.l1.search(
                    space, query, when, cfg, exclude=exclude
                ),
                config=replace(L1Config(), budget_tokens=2000),
                rounds=list(recall.view.rounds),
                facts=list(ctx.facts),
                refs=dict(ctx.refs),
            )
            tool_ctx = ToolContext(
                system=self.system,
                space_id=space,
                when=when,
                memory=memory,
                uploads={**session.uploads, **dict(uploads)},
            )
            llm_started = time.perf_counter()
            answer, records = self._loop(ctx.messages, schemas, tool_ctx, cache_scope)
            timings["llm"] = _ms(llm_started)
            cited = citations(answer, memory.refs)
            plain = strip_citations(answer)

            write_started = time.perf_counter()
            user_idx = session.next_idx
            exchange = SessionIn(
                session_id=session.session_id,
                started_at=session.started_at,
                turns=[
                    TurnIn(role="user", text=text, ts=when, images=images),
                    TurnIn(role="assistant", text=plain, ts=when),
                ],
            )
            with self.system.appending(space, [(exchange, user_idx)]):
                # 溯源：这次回复引用了哪些记忆（遗忘时沿引用扩散，见 MemorySystem.forget）。与这一轮
                # 在同一个事务里提交：写入出错时都不留，这一轮重来时不会写两遍
                self.system.db.add_provenance(
                    space,
                    stable_id(space, session.session_id, user_idx + 1),
                    [memory.refs[ref] for ref in cited],
                )
            session.next_idx += 2
            # 之后各轮（工作记忆里的图、找同款的 u1）改读记忆里存的那份，字节相同、请求不变：
            # 调用方的文件（HTTP 的上传、界面的临时文件）用完就能删，遗忘删掉原图后不留副本
            uploads = self._stored_uploads(space, session.session_id, user_idx, uploads)
            timings["write"] = _ms(write_started)
            for label, path in uploads:
                session.uploads[label] = path
                session.upload_turns[label] = user_idx
            session.window.append(Exchange(text, plain, when, uploads, user_idx))
            evicted = evict(session.window, self.budget)
            timings["total"] = _ms(started)
            trace.attrs.update(answer=plain, citations=cited, level=_level(recall))
        return Reply(
            session_id=session.session_id,
            at=when,
            text=answer,
            plain=plain,
            citations=cited,
            products=_products(records, plain),
            trace=TurnTrace(
                trace_id=trace.id,
                level=_level(recall),
                see=bool(recall.decision and recall.decision.see),
                router=recall.router,
                reasons=list(recall.decision.reasons) if recall.decision else [],
                scores=dict(recall.decision.scores) if recall.decision else {},
                partitions=ctx.partitions,
                budget=self.budget.as_dict(),
                deep=_deep(recall),
                recalled=self._recalled(space, recall, memory, ctx.refs, cited, session.profile),
                tools=records,
                usage=_usage(trace),
                timings={k: round(v, 1) for k, v in timings.items()},
                window=len(session.window),
                evicted=evicted,
                prefix_messages=ctx.prefix,
            ),
        )

    def _stored_uploads(
        self, space: str, session_id: str, user_idx: int, uploads: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """这一轮的图片 → 媒体库里存的路径（没有媒体库、没存下的保留原路径）。"""
        if not uploads:
            return uploads
        turn_id = stable_id(space, session_id, user_idx)
        ids = [media_id(space, turn_id, i) for i in range(len(uploads))]
        stored = {m.id: m.path for m in self.system.db.get_media(space, ids=ids)}
        return [
            (label, stored.get(mid, path)) for (label, path), mid in zip(uploads, ids, strict=True)
        ]

    def _loop(
        self,
        messages: list[Message],
        schemas: list[dict[str, Any]],
        ctx: ToolContext,
        cache_scope: str,
    ) -> tuple[str, list[ToolRecord]]:
        """工具调用循环：模型调用工具就执行并把结果接在后面，直到它给出文字回答。
        最后一步不再允许调用工具。"""
        records: list[ToolRecord] = []
        for step in range(1, self.max_steps + 1):
            final = step == self.max_steps
            with span("agent_step", step=step):
                result = self.system.llm.chat(
                    self.role,
                    messages,
                    tools=schemas,
                    tool_choice="none" if final else "auto",
                    cache_scope=cache_scope,
                )
            if not result.tool_calls or final:
                return (result.content or "").strip(), records
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in result.tool_calls
                    ],
                }
            )
            for call in result.tool_calls:
                started = time.perf_counter()
                args, outcome = self._run_tool(call.name, call.arguments, ctx)
                records.append(
                    ToolRecord(
                        name=call.name,
                        arguments=args,
                        result=outcome.text[:2000],
                        ms=round(_ms(started), 1),
                        data=outcome.data,
                    )
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.text})
        return "", records  # 走不到这里：最后一步总会返回

    def _run_tool(
        self, name: str, arguments: str, ctx: ToolContext
    ) -> tuple[dict[str, Any], ToolResult]:
        tool = self.tools.get(name)
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {}, ToolResult("Invalid arguments: expected a JSON object.")
        if not isinstance(args, dict):
            return {}, ToolResult("Invalid arguments: expected a JSON object.")
        if tool is None:
            return args, ToolResult(f"Unknown tool: {name}")
        with span("tool", tool=name):
            try:
                return args, tool.run(args, ctx)
            except Exception as exc:  # 工具出错不能让整轮对话失败：把错误告诉模型
                return args, ToolResult(f"Tool error: {type(exc).__name__}: {exc}")

    # ---------- 辅助 ----------

    def _window_ids(self, session: ChatSession) -> set[str]:
        """工作记忆里的轮与图片的 ID：检索时排除（它们已经原样在上下文里）。"""
        ids: set[str] = set()
        for exchange in session.window:
            first = exchange.user_idx
            ids.add(stable_id(session.space_id, "round", session.session_id, first))
            turn_id = stable_id(session.space_id, session.session_id, first)
            for i in range(len(exchange.images)):
                ids.add(media_id(session.space_id, turn_id, i))
        return ids

    def _recalled(
        self,
        space_id: str,
        recall: Recall,
        memory: SearchSession,
        shown: dict[str, str],
        cited: list[str],
        session_profile: list[tuple[str, str, str]],
    ) -> list[RecalledItem]:
        """本轮给模型看过的记忆（控制器召回的，加上模型自己用 memory_search 查到的）。"""
        rounds = {h.unit.id: h for h in memory.rounds}
        facts = {fh.fact.id: fh for fh in memory.facts}
        fact_refs = {fid: ref for ref, fid in memory.refs.items() if ref.startswith("f")}
        attached = set(recall.view.attach)
        media_ids = [v for k, v in memory.refs.items() if k.startswith("i")]
        media = {m.id: m for m in self.system.db.get_media(space_id, ids=media_ids)}
        items: list[RecalledItem] = []
        profile = {fid: statement for _, fid, statement in session_profile}
        for ref, target in memory.refs.items():
            via: Literal["controller", "memory_search"] = (
                "controller" if ref in shown else "memory_search"
            )
            if ref.startswith("p"):
                if ref in cited:  # 画像每轮都在，只列出这一轮引用了的
                    items.append(
                        RecalledItem(
                            ref=ref,
                            kind="profile",
                            id=target,
                            text=profile.get(target, ""),
                            status="current",
                            cited=True,
                        )
                    )
            elif ref.startswith("f") and target in facts:
                fact = facts[target].fact
                items.append(
                    RecalledItem(
                        ref=ref,
                        kind="fact",
                        id=fact.id,
                        text=fact.statement,
                        when=fact.recorded_at,
                        status=fact_status(fact, fact_refs),
                        session_id=fact.session_id,
                        via=via,
                        cited=ref in cited,
                    )
                )
            elif ref.startswith("m") and target in rounds:
                unit = rounds[target].unit
                items.append(
                    RecalledItem(
                        ref=ref,
                        kind="excerpt",
                        id=unit.id,
                        text=unit.text,
                        when=unit.ts,
                        session_id=unit.session_id,
                        via=via,
                        cited=ref in cited,
                    )
                )
            elif ref.startswith("i") and target in media:
                item = media[target]
                items.append(
                    RecalledItem(
                        ref=ref,
                        kind="image",
                        id=item.id,
                        text=item.caption or "(not described yet)",
                        when=item.ts,
                        path=item.path,
                        via=via,
                        attached=item.id in attached,
                        cited=ref in cited,
                    )
                )
        return items


def _deep(recall: Recall) -> dict[str, Any]:
    """L2 的过程摘要：规划出的子查询与每次充分性检查（补检索了什么）。"""
    if recall.level != "l2":
        return {}
    plan = recall.details.get("plan", {})
    return {
        "answer_type": plan.get("answer_type"),
        "sub_queries": plan.get("sub_queries", []),
        "time_range": plan.get("time_range"),
        "checks": [
            {k: c.get(k) for k in ("sufficient", "missing", "queries", "added_rounds")}
            for c in recall.details.get("checks", [])
        ],
    }


def _level(recall: Recall) -> str:
    return recall.level.upper()


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _usage(trace: Trace) -> dict[str, float]:
    """本轮的模型用量：调用次数、输入（其中命中供应商前缀缓存的）、输出、名义成本与实际花费。"""
    usage = {
        "llm_calls": 0.0,
        "cache_hits": 0.0,  # 命中本地响应缓存（没有真正调用 API）
        "input_tokens": 0.0,
        "cached_input_tokens": 0.0,
        "output_tokens": 0.0,
        "cost": 0.0,
        "spent": 0.0,
        "embedding_calls": 0.0,
    }
    for event in trace.events:
        if event["kind"] == "llm":
            usage["llm_calls"] += 1
            usage["cache_hits"] += float(bool(event.get("from_cache")))
            usage["input_tokens"] += event.get("input_tokens", 0)
            usage["cached_input_tokens"] += event.get("cached_input_tokens", 0)
            usage["output_tokens"] += event.get("output_tokens", 0)
            usage["cost"] += event.get("cost", 0.0)
            usage["spent"] += event.get("spent", 0.0)
        elif event["kind"] == "embedding":
            usage["embedding_calls"] += 1
            usage["cost"] += event.get("cost", 0.0)
            usage["spent"] += event.get("cost", 0.0)
    return {k: round(v, 6) for k, v in usage.items()}


def _products(records: Sequence[ToolRecord], answer: str) -> list[dict[str, Any]]:
    """领域工具返回的商品（去重、保持顺序），标出回答推荐了的：名字加粗出现才算推荐
    （"跳过了皮鞋 Tempo Leather Trainer"不算）；一个都没加粗时退回到"提到就算"。"""
    seen: dict[str, dict[str, Any]] = {}
    for record in records:
        for product in record.data.get("products", []):
            seen.setdefault(str(product.get("id")), dict(product))
    bold = " ".join(re.findall(r"\*\*(.+?)\*\*", answer)).lower()
    names = {pid: str(p.get("name", "")).lower() for pid, p in seen.items()}
    if not any(name and name in bold for name in names.values()):
        bold = answer.lower()
    for pid, product in seen.items():
        product["mentioned"] = bool(names[pid]) and names[pid] in bold
    return list(seen.values())
