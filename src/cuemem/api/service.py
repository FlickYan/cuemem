"""服务层（N8）：API 与界面共用的一个对象——记忆系统、Agent、进行中的会话，以及记忆浏览器
要看的各种视图（画像、事实时间线、实体图、图片、会话），和遗忘。

线程安全：FastAPI 在线程池里处理请求；记忆系统的各部件自带锁，同一会话的消息由会话锁串行。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2

from cuemem.agent import Agent, ChatSession, Persona, Reply
from cuemem.agent.shop import SHOP_PERSONA, Catalog, shop_tools
from cuemem.config import PROJECT_ROOT, Settings
from cuemem.core.ids import sortable_session_id, stable_id
from cuemem.core.models import SessionIn, TurnIn
from cuemem.core.types import RecallMode, Router, Scenario
from cuemem.errors import CueMemError
from cuemem.evidence import fact_state, recall_listing, resolve_ids
from cuemem.memory import ForgetReport, MemorySystem

SHOP_CATALOG = PROJECT_ROOT / "demo" / "shop" / "catalog.json"  # 只在源码仓库里有，不进 wheel

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    space_id: str
    session_id: str
    started_at: datetime
    open: bool
    turns: int


class ChatService:
    """clock：会话最后一次活动的时间用的时钟（close_idle 据此判断闲置，测试里可以换）。
    idle_minutes：会话闲置多久由 HTTP 应用的清理线程结束（create_app 设置；None 是不结束），
    只用于说明。"""

    def __init__(
        self,
        system: MemorySystem,
        agent: Agent,
        *,
        catalog: Catalog | None = None,
        default_space: str = "default",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.system = system
        self.agent = agent
        self.catalog = catalog
        self.default_space = default_space
        self.sessions: dict[str, ChatSession] = {}
        self.idle_minutes: float | None = None
        self._clock = clock
        self._seen: dict[str, float] = {}  # 会话 → 最后一次活动（开始、发消息）的时刻
        self._adopted: set[str] = set()  # 接手过崩溃前留下的会话的空间（见 start）
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
        catalog_path = catalog_path or SHOP_CATALOG
        if scenario == "shop" and not catalog_path.is_file():  # 在打开记忆系统之前检查
            raise CueMemError(
                f"找不到导购演示的商品目录 {catalog_path}：它只在源码仓库里（demo/shop/），"
                "pip 安装的包里没有。改用通用助手（cuemem serve --scenario plain，"
                '代码里 scenario="plain"），或用 catalog_path 指定商品目录'
            )
        system = MemorySystem.from_settings(settings, http_client=http_client)
        catalog = None
        persona: Persona | None = None
        tools = []
        if scenario == "shop":
            catalog = Catalog.load(
                catalog_path, embedder=system.embedder, image_embedder=system.image_embedder
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
        space = space_id or self.default_space
        with self._lock:
            first = space not in self._adopted
            self._adopted.add(space)
        if first:  # 之前的进程崩溃时还在进行中的会话交给抽取；接着写的那个会话除外
            self.system.adopt_open(space, keep=[session_id] if session_id else [])
        session = self.agent.start(space, at=at, session_id=session_id)
        with self._lock:
            self.sessions[session.session_id] = session
            self._seen[session.session_id] = self._clock()
        return session

    def get(self, session_id: str) -> ChatSession:
        with self._lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(self._missing(session_id))
        return session

    def is_open(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self.sessions

    def _missing(self, session_id: str) -> str:
        idle = f"（闲置 {self.idle_minutes:g} 分钟的会话自动结束）" if self.idle_minutes else ""
        return f"没有进行中的会话 {session_id}：不存在或已经结束{idle}，请开始一个新的会话"

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
        # 取会话与记下这次活动在同一把锁里：清理线程（close_idle）在这把锁里判断闲置、取走会话，
        # 不会结束一个正要发消息的会话
        with self._lock:
            session = self.sessions.get(session_id)
            if session is not None:
                self._seen[session_id] = self._clock()
        if session is None:
            raise KeyError(self._missing(session_id))
        return self.agent.chat(
            session, text, images=images, at=at, mode=mode, router=router, lam=lam
        )

    def end(self, session_id: str, *, wait: bool = False) -> dict[str, Any]:
        session = self.get(session_id)
        report = self.agent.end(session, wait=wait)
        with self._lock:
            self.sessions.pop(session_id, None)
            self._seen.pop(session_id, None)
        out: dict[str, Any] = {"session_id": session_id, "turns": session.next_idx}
        if report is not None:
            out["consolidation"] = report.as_dict()
        elif session.next_idx:
            out["consolidation"] = "queued"
        return out

    def close_idle(self, idle_s: float) -> int:
        """闲置（开始或上一条消息之后）超过 idle_s 秒的会话结束、交给后台抽取，与 end 相同；返回
        结束的个数。HTTP 应用的清理线程每分钟调用一次（create_app 的 idle_minutes）。以前只有
        /api/sessions/{id}/end 与界面的按钮结束会话：关掉的标签页、崩溃的客户端留下的会话一直
        开着，服务运行期间一直不抽取。之后再往这个会话发消息是 404（与不存在的会话一样，不告诉
        作用域之外的调用方这个 ID 存在过），界面开一个新的会话。

        判断闲置与取走会话在发消息记下活动的同一把锁里（见 chat）：正要发消息的会话不会被结束。
        正在回答的会话（持有会话锁）等这一轮做完再结束，这一轮也一起抽取。"""
        now = self._clock()
        with self._lock:
            stale = [sid for sid, seen in self._seen.items() if now - seen >= idle_s]
            sessions = [self.sessions.pop(sid) for sid in stale if sid in self.sessions]
            for sid in stale:
                del self._seen[sid]
        for session in sessions:
            try:
                self.agent.end(session)
            except Exception:  # 一个会话结束不了不影响别的
                logger.warning("闲置的会话 %s 没能结束", session.session_id, exc_info=True)
        return len(sessions)

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
        """检索控制器选档、取回记忆，原样交给调用方（另一个 Agent）去推理（见 evidence.py）。"""
        return recall_listing(
            self.system,
            space_id,
            query,
            mode=mode,
            router=router or self.agent.router,
            lam=lam,
            at=at,
            at_least_l1=at_least_l1,
        )

    def remember(self, space_id: str, text: str, *, at: datetime | None = None) -> dict[str, str]:
        """记一条笔记（MCP 的 remember）：写成只有一句话的会话，在线写入、后台巩固。"""
        # 现在的时间不截到秒：与 SDK、各个集成写入的会话在同一秒里也分得出先后
        when = at.replace(microsecond=0) if at is not None else datetime.now()
        sid = sortable_session_id("note", when)
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
        fact_ids, turns = resolve_ids(self.system, space_id, ids)
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
                    "status": fact_state(fact),
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
            # 先按子串筛：每个名字一个正则，名字多过 re 的缓存后每条事实都要全部重新编译
            linked = set(fact.entity_ids) | {
                eid
                for eid, aliases in names.items()
                if any(
                    a in text and re.search(rf"(?<!\w){re.escape(a)}(?!\w)", text) for a in aliases
                )
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
        """巩固任务的状态与后台线程最近的出错，都只是这个空间的：以前 errors 是全部空间的最近
        3 条，限定作用域的令牌借自己的空间就看得到别的用户的空间 ID 与出错里引着的模型输出。"""
        jobs = self.system.db.get_jobs(space_id)
        worker = self.system.worker
        return {
            "pending": sum(1 for j in jobs if j["status"] == "pending"),
            "failed": sum(1 for j in jobs if j["status"] == "failed"),
            "done": sum(1 for j in jobs if j["status"] == "done"),
            "errors": worker.errors_for(space_id)[-3:] if worker is not None else [],
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
        """遗忘（逻辑见 MemorySystem.forget_memories）；进行中的会话同步去掉被删的内容。
        删除做完之后重建派生记忆出错时不抛出，写在报告的 rebuild_error 里（HTTP 接口与 MCP 据此
        说明删除已经做了）。"""
        try:
            return self.system.forget_memories(
                space_id,
                session_ids=session_ids,
                turn_ids=turn_ids,
                fact_ids=fact_ids,
                media_ids=media_ids,
                report_rebuild_errors=True,
            )
        finally:  # 重建失败时原始日志也已经删了：进行中的会话照样去掉被删的轮
            for session in self.open_sessions(space_id):
                self.agent.prune(session)

    def consolidate(self, space_id: str) -> dict[str, Any]:
        return self.system.consolidate(space_id).as_dict()

    def close(self) -> None:
        for session in list(self.sessions.values()):
            if not session.closed:
                self.agent.end(session)  # 登记巩固任务；进程退出后下次巩固时完成
        self.system.close()


def _display(name: str, aliases: list[str]) -> str:
    """用户本人在实体图里显示真名（抽取时登记为别名）。"""
    if name.lower() in ("user", "the user", "用户") and aliases:
        return f"{aliases[0]} (user)"
    return name
