"""服务层（N8）：API 与界面共用的一个对象——记忆系统、Agent、进行中的会话，以及记忆浏览器
要看的各种视图（画像、事实时间线、实体图、图片、会话），和遗忘。

线程安全：FastAPI 在线程池里处理请求；记忆系统的各部件自带锁，同一会话的消息由会话锁串行。
"""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx2

from cuemem.agent import Agent, ChatSession, Persona, Reply
from cuemem.agent.shop import SHOP_PERSONA, Catalog, shop_tools
from cuemem.config import PROJECT_ROOT, Settings
from cuemem.core.ids import stable_id
from cuemem.core.models import Fact, SessionIn, TurnIn
from cuemem.memory import ForgetReport, MemorySystem, RecallMode
from cuemem.read.controller import Router

Scenario = Literal["shop", "plain"]
# 控制器给的理由用中文标签（CLI 是中文的）；接口、界面与 MCP 给英文
REGROW_ROUNDS = 3  # 遗忘后追查复述的轮数上限（每轮要重建一次派生记忆）

REASON_EN = {
    "单点事实": "single fact",
    "多会话 / 多跳": "multi-session",
    "时间推理": "temporal",
    "知识更新": "knowledge update",
    "偏好与建议": "preference / advice",
    "开放推断": "open inference",
    "拒答": "probably not in memory",
    "与记忆无关": "no memory needed",
}


def english_reason(text: str) -> str:
    for zh, en in REASON_EN.items():
        if text.startswith(zh):
            return en + text[len(zh) :]
    return text.replace("llm:", "LLM said ")


SHOP_CATALOG = PROJECT_ROOT / "demo" / "shop" / "catalog.json"


@dataclass
class SessionInfo:
    space_id: str
    session_id: str
    started_at: datetime
    open: bool
    turns: int


class ChatService:
    def __init__(
        self,
        system: MemorySystem,
        agent: Agent,
        *,
        catalog: Catalog | None = None,
        default_space: str = "default",
    ) -> None:
        self.system = system
        self.agent = agent
        self.catalog = catalog
        self.default_space = default_space
        self.sessions: dict[str, ChatSession] = {}
        self._lock = threading.Lock()
        system.start_worker()  # 会话结束后的巩固在后台完成

    @classmethod
    def create(
        cls,
        settings: Settings,
        *,
        scenario: Scenario = "shop",
        catalog_path: Path | None = None,
        default_space: str = "default",
        router: Router = "learned",
        lam: float | None = None,
        http_client: httpx2.Client | None = None,
    ) -> ChatService:
        system = MemorySystem.from_settings(settings, http_client=http_client)
        catalog = None
        persona: Persona | None = None
        tools = []
        if scenario == "shop":
            catalog = Catalog.load(
                catalog_path or SHOP_CATALOG,
                embedder=system.embedder,
                image_embedder=system.image_embedder,
            )
            persona = SHOP_PERSONA
            tools = shop_tools(catalog)
        agent = Agent(system, persona=persona, tools=tools, router=router, lam=lam)
        return cls(system, agent, catalog=catalog, default_space=default_space)

    # ---------- 会话 ----------

    def start(
        self,
        space_id: str | None = None,
        *,
        at: datetime | None = None,
        session_id: str | None = None,
    ) -> ChatSession:
        session = self.agent.start(space_id or self.default_space, at=at, session_id=session_id)
        with self._lock:
            self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ChatSession:
        with self._lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"没有进行中的会话 {session_id}")
        return session

    def chat(
        self,
        session_id: str,
        text: str,
        *,
        images: Sequence[str | Path] = (),
        at: datetime | None = None,
        mode: RecallMode = "auto",
        router: Router | None = None,
        lam: float | None = None,
    ) -> Reply:
        return self.agent.chat(
            self.get(session_id), text, images=images, at=at, mode=mode, router=router, lam=lam
        )

    def end(self, session_id: str, *, wait: bool = False) -> dict[str, Any]:
        session = self.get(session_id)
        report = self.agent.end(session, wait=wait)
        with self._lock:
            self.sessions.pop(session_id, None)
        out: dict[str, Any] = {"session_id": session_id, "turns": session.next_idx}
        if report is not None:
            out["consolidation"] = report.as_dict()
        elif session.next_idx:
            out["consolidation"] = "queued"
        return out

    def open_sessions(self, space_id: str) -> list[ChatSession]:
        with self._lock:
            return [s for s in self.sessions.values() if s.space_id == space_id]

    # ---------- 记忆即服务：只检索、不回答 ----------

    def recall(
        self,
        space_id: str,
        query: str,
        *,
        mode: RecallMode = "auto",
        router: Router | None = None,
        lam: float | None = None,
        at: datetime | None = None,
        at_least_l1: bool = False,
    ) -> dict[str, Any]:
        """检索控制器选档、取回记忆，原样交给调用方（另一个 Agent）去推理，不调用回答模型。

        at_least_l1：调用方已经决定要查记忆（例如 MCP 的 recall），控制器选了 L0 时改走 L1，
        控制器只决定要不要查得更深。
        """
        recall = self.system.retrieve(
            space_id,
            query,
            question_time=at,
            mode=mode,
            router=router or self.agent.router,
            lam=lam,
        )
        raised = at_least_l1 and recall.level == "l0"
        if raised:
            recall = self.system.retrieve(space_id, query, question_time=at, mode="l1")
        view = recall.view
        decision = recall.decision
        return {
            "level": recall.level.upper(),
            "see": bool(decision and decision.see),
            "reasons": (
                ["raised from L0: the caller asked to recall"]
                if raised
                else [english_reason(r) for r in decision.reasons]
                if decision
                else []
            ),
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
                {
                    "id": hit.unit.id,
                    "session_id": hit.unit.session_id,
                    "ts": hit.unit.ts.isoformat(),
                    "text": hit.unit.text,
                    "turn_ids": hit.unit.source_turn_ids,
                }
                for hit in sorted(view.rounds, key=lambda h: h.unit.ts)
            ],
            "images": [
                {"id": m.id, "caption": m.caption, "path": m.path, "attached": m.id in view.attach}
                for m in view.media.values()
            ],
        }

    def remember(self, space_id: str, text: str, *, at: datetime | None = None) -> dict[str, str]:
        """记一条笔记（MCP 的 remember）：写成只有一句话的会话，在线写入、后台巩固。"""
        when = (at or datetime.now()).replace(microsecond=0)
        sid = f"note-{when:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
        self.system.append(
            space_id,
            SessionIn(
                session_id=sid,
                started_at=when,
                turns=[TurnIn(role="user", text=text, ts=when)],
            ),
            first_idx=0,
        )
        return {"session_id": sid, "turn_id": stable_id(space_id, sid, 0)}

    def forget_ids(self, space_id: str, ids: Sequence[str]) -> ForgetReport:
        """按 recall 给出的 ID 遗忘：事实 ID，或对话片段（轮）的 ID。"""
        fact_ids = [f.id for f in self.system.db.get_facts(space_id, ids=list(ids))]
        rounds = self.system.db.get_units(space_id, list(ids))
        turns = [t for unit in rounds if unit.kind == "round" for t in unit.source_turn_ids]
        return self.forget(space_id, fact_ids=fact_ids, turn_ids=turns)

    # ---------- 记忆浏览器 ----------

    def spaces(self) -> list[dict[str, Any]]:
        return self.system.db.list_spaces()

    def profile(self, space_id: str) -> str:
        core = self.system.core_memory(space_id)
        return core.text if core is not None else ""

    def facts(self, space_id: str, *, current_only: bool = False) -> list[dict[str, Any]]:
        facts = self.system.facts(space_id)
        by_id = {f.id: f for f in facts}
        rows = []
        for fact in sorted(facts, key=lambda f: (f.recorded_at, f.id)):
            if current_only and not fact.is_current:
                continue
            rows.append(
                {
                    "id": fact.id,
                    "statement": fact.statement,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "category": fact.category,
                    "said_on": fact.recorded_at.isoformat(),
                    "valid_from": fact.valid_from,
                    "valid_to": fact.valid_to,
                    "current": fact.is_current,
                    "status": _status(fact),
                    "superseded_by": fact.superseded_by,
                    "replaced_by": (
                        by_id[fact.superseded_by].statement if fact.superseded_by in by_id else None
                    ),
                    "session_id": fact.session_id,
                    "importance": fact.importance,
                }
            )
        return rows

    def entities(self, space_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "aliases": e.aliases,
                "description": e.description,
                "mentions": e.mentions,
            }
            for e in self.system.entities(space_id)
        ]

    def graph(self, space_id: str) -> dict[str, Any]:
        """实体图：实体为点；当前有效的事实把主语和它提到的其他实体连起来（宾语，或陈述里
        出现了名字或别名的实体），边上写谓词。"""
        entities = {e.id: e for e in self.system.entities(space_id)}
        names = {
            eid: [n.lower() for n in [e.name, *e.aliases] if len(n) >= 3]
            for eid, e in entities.items()
        }
        edges = []
        degree: dict[str, int] = {}
        seen: set[tuple[str, str]] = set()
        for fact in self.system.facts(space_id, current_only=True):
            if fact.subject_id not in entities:
                continue
            degree[fact.subject_id] = degree.get(fact.subject_id, 0) + 1
            text = f"{fact.statement} {fact.object}".lower()
            linked = set(fact.entity_ids) | {
                eid
                for eid, aliases in names.items()
                if any(re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text) for a in aliases)
            }
            for other in sorted(linked - {fact.subject_id}):
                pair = (min(fact.subject_id, other), max(fact.subject_id, other))
                if other in entities and pair not in seen:  # 每对实体只画一条边
                    seen.add(pair)
                    edges.append(
                        {"source": fact.subject_id, "target": other, "label": fact.predicate}
                    )
                    degree[other] = degree.get(other, 0) + 1
        nodes = [
            {"id": e.id, "name": _display(e.name, e.aliases), "type": e.type,
             "facts": degree.get(e.id, 0)}
            for e in entities.values()
            if degree.get(e.id, 0) > 0
        ]  # fmt: skip
        return {"nodes": nodes, "edges": edges}

    def media(self, space_id: str) -> list[dict[str, Any]]:
        return [
            {
                "id": m.id,
                "path": m.path,
                "caption": m.caption,
                "ts": m.ts.isoformat(),
                "turn_id": m.source_turn_id,
            }
            for m in sorted(self.system.db.get_media(space_id), key=lambda m: m.ts)
        ]

    def session_list(self, space_id: str) -> list[SessionInfo]:
        started = self.system.db.get_sessions(space_id)
        counts: dict[str, int] = {}
        for turn in self.system.db.get_turns(space_id):
            counts[turn.session_id] = counts.get(turn.session_id, 0) + 1
        live = {s.session_id for s in self.open_sessions(space_id)}
        return [
            SessionInfo(space_id, sid, at, sid in live, counts.get(sid, 0))
            for sid, at in sorted(started.items(), key=lambda item: item[1])
        ]

    def status(self, space_id: str) -> dict[str, Any]:
        jobs = self.system.db.get_jobs(space_id)
        worker = self.system.worker
        return {
            "pending": sum(1 for j in jobs if j["status"] == "pending"),
            "failed": sum(1 for j in jobs if j["status"] == "failed"),
            "done": sum(1 for j in jobs if j["status"] == "done"),
            "errors": list(worker.errors[-3:]) if worker is not None else [],
        }

    # ---------- 遗忘与巩固 ----------

    def forget(
        self,
        space_id: str,
        *,
        session_ids: Sequence[str] = (),
        turn_ids: Sequence[str] = (),
        fact_ids: Sequence[str] = (),
        media_ids: Sequence[str] = (),
    ) -> ForgetReport:
        """遗忘。原始日志是唯一的事实来源：删一条事实或一张图，删的是它出自的那一轮对话
        （用户的话与助手的回复），然后由剩下的日志重建所有派生记忆（D-021）。
        删事实时还要追没有引用、追不到的转述（D-051）：先在原始日志里找复述了它的轮
        （包括进行中的会话）一起删；重建之后如果同样的信息又从别处长了出来，再删它的出处，
        最多再追 REGROW_ROUNDS 轮。进行中的会话同步去掉被删的内容。"""
        sources: list[str] = []
        statements: list[str] = []
        if fact_ids:
            for fact in self.system.db.get_facts(space_id, ids=list(fact_ids)):
                sources.extend(fact.source_turn_ids)
                statements.append(fact.statement)
        if media_ids:
            sources.extend(
                m.source_turn_id for m in self.system.db.get_media(space_id, ids=list(media_ids))
            )
        # 事实与图片出自的发言扩展成整轮：助手的回复里常常复述了同样的信息
        turns = [*turn_ids, *self.system.round_turns(space_id, sources)]
        restated = self.system.round_turns(
            space_id, self.system.restating_turns(space_id, statements, exclude=turns)
        )
        report = self.system.forget(
            space_id,
            session_ids=list(session_ids),
            turn_ids=list(dict.fromkeys([*turns, *restated])),
        )
        report = replace(report, restated=len(restated))
        for _ in range(REGROW_ROUNDS if statements else 0):
            regrown = self.system.regrown_facts(space_id, statements)
            if not regrown:
                break
            more = self.system.round_turns(
                space_id, [t for f in regrown for t in f.source_turn_ids]
            )
            report = report.merged(
                replace(self.system.forget(space_id, turn_ids=more), regrown=len(regrown))
            )
        for session in self.open_sessions(space_id):
            self.agent.prune(session)
        return report

    def consolidate(self, space_id: str) -> dict[str, Any]:
        return self.system.consolidate(space_id).as_dict()

    def close(self) -> None:
        for session in list(self.sessions.values()):
            if not session.closed:
                self.agent.end(session)  # 登记巩固任务；进程退出后下次巩固时完成
        self.system.close()


def _status(fact: Fact) -> str:
    """界面上的有效状态：当前有效 / 何时因现实变化而失效 / 何时被更正。"""
    if fact.is_current:
        return "current"
    when = f"{fact.invalidated_at:%Y-%m-%d}" if fact.invalidated_at else "?"
    if fact.supersede_reason == "corrected":
        return f"corrected on {when}"
    return f"changed on {when}" + (f" (until {fact.valid_to})" if fact.valid_to else "")


def _display(name: str, aliases: list[str]) -> str:
    """用户本人在实体图里显示真名（抽取时登记为别名）。"""
    if name.lower() in ("user", "the user", "用户") and aliases:
        return f"{aliases[0]} (user)"
    return name
