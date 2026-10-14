"""Python SDK（N12，D-061）：照 Mem0 的接口形状包一层，几行代码就能用上 CueMem。

    from cuemem import Memory

    m = Memory(preset="openai")                       # 或 "ollama"（本地），不填用默认配置
    m.add([{"role": "user", "content": "I moved to Berlin last month"}], user_id="alice")
    m.search("Where do I live?", user_id="alice")     # 检索控制器决定要不要查、查多深
    m.answer("Where do I live?", user_id="alice")     # 检索并回答（带引用）
    m.history(fact_id)                                # 一条事实的版本链：何时被取代、为什么

与 Mem0 对应的方法名与参数（add / search / get_all / get / history / delete / delete_all）保持
一致，返回值也是 {"results": [...]}、每条带 "memory" 文本；不同的地方是 CueMem 自己的设计：

- 记忆空间：user_id 与 agent_id 合起来定一个空间（各空间完全隔离）；run_id 是空间里的一个会话；
- 写入分两步：add 先在线写入原始日志（不调用 LLM，立即可检索），抽取事实在后台完成；
  wait=True 时同步抽取，并按 Mem0 的格式返回 ADD / UPDATE 事件；
- 事实不就地改写、也不删除旧版本：被取代的事实留在版本链里（history），get_all 默认只给当前的；
- search 先过检索控制器：与记忆无关的问题返回 L0（只有画像，不检索），mode="l1" 可以强制检索；
- delete 是"被遗忘权"：删掉事实出自的对话，再用剩下的日志重建，连带清理转述、缓存与原图。

返回值、参数名与报错信息用英文（与 HTTP 接口、MCP、界面一致）；SDK 只组装已有的组件，
检索、巩固、遗忘的逻辑都在 MemorySystem 里。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import functools
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import httpx2

from cuemem.config import Settings, load_settings
from cuemem.core.ids import stable_id
from cuemem.core.models import Answer, Fact, FactHit, Hit, MediaItem, SessionIn, TurnIn
from cuemem.memory import ForgetReport, MemorySystem, Recall
from cuemem.read.controller import Router, english_reason
from cuemem.read.facts import fact_status

SearchMode = Literal["auto", "l0", "l1", "l2"]
AnswerMode = Literal["auto", "l0", "l1", "l2", "agent", "hatch"]
Message = Mapping[str, Any]
MessagesIn = str | Message | Sequence[Message]

_IMAGE_SUFFIX = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def space_for(user_id: str | None = None, agent_id: str | None = None) -> str:
    """user_id、agent_id → 记忆空间 ID。两者都给时是这个 Agent 对这个用户的记忆。"""
    if user_id and agent_id:
        return f"{agent_id}/{user_id}"
    if user_id:
        return user_id
    if agent_id:
        return f"agent/{agent_id}"
    raise ValueError("Give user_id, agent_id, or both: memories are kept per user and per agent.")


class Memory:
    """CueMem 的 Python SDK。线程安全；用完调用 close()，或用 with 语句。

    config：嵌套的字典，与默认配置（及预设）深度合并，例如
    {"roles": {"reader": {"model": "qwen3.8-flash"}}}；home：数据目录（默认 CUEMEM_HOME，
    源码仓库里是仓库根目录，否则 ~/.cuemem）；background：后台线程抽取事实（默认开）。
    """

    def __init__(
        self,
        config: Mapping[str, Any] | Settings | None = None,
        *,
        preset: str | None = None,
        home: str | Path | None = None,
        background: bool = True,
        http_client: httpx2.Client | None = None,
    ) -> None:
        if isinstance(config, Settings):
            settings = config
        else:
            settings = load_settings(preset=preset, home=home, **dict(config or {}))
        self.settings = settings
        self.system = MemorySystem.from_settings(settings, http_client=http_client)
        if background:
            self.system.start_worker()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> Memory:
        """与 Mem0 同名的构造方式：config 是嵌套的配置字典（字段见 configs/default.yaml）。"""
        return cls(config, **kwargs)

    @classmethod
    def from_system(cls, system: MemorySystem) -> Memory:
        """包住一个已经打开的记忆系统（同一个进程里共用，例如 HTTP 服务与记忆代理）。
        本地模式的向量库同一时间只能被一个对象打开，不能再 Memory() 一个新的。"""
        memory = cls.__new__(cls)
        memory.settings = system.settings
        memory.system = system
        return memory

    # ---------- 写入 ----------

    def add(
        self,
        messages: MessagesIn,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        timestamp: datetime | str | float | None = None,
        infer: bool = True,
        wait: bool = False,
    ) -> dict[str, Any]:
        """写入对话。messages 可以是一句话、一条消息或 OpenAI 格式的消息列表（内容块里的
        image_url 可以是 data URL 或本地路径，图片会存进记忆）；system / tool 消息不写入。

        run_id：写进同一个会话（一次对话），不填时这次 add 自成一个会话。
        infer=False：只写原始日志、不抽取事实（之后调用 consolidate）。
        wait=True：同步抽取，返回新增与取代的事实（Mem0 的 ADD / UPDATE 事件）。
        """
        space = space_for(user_id, agent_id)
        when = _as_datetime(timestamp) or datetime.now().replace(microsecond=0)
        turns, temporary = parse_messages(messages, when)
        try:
            if not turns:
                return {"results": [], "session_id": None, "turn_ids": [], "status": "empty"}
            sid = run_id or f"add-{when:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
            first = self._next_idx(space, sid)
            started = self.system.db.get_sessions(space).get(sid, when)
            before = {f.id: f for f in self.system.db.get_facts(space)} if wait else {}
            self.system.append(
                space,
                SessionIn(session_id=sid, started_at=started, turns=turns),
                first,
                enqueue=infer,
            )
            self.system.db.set_meta(_idx_key(space, sid), str(first + len(turns)))
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
        turn_ids = [stable_id(space, sid, first + i) for i in range(len(turns))]
        out: dict[str, Any] = {"results": [], "session_id": sid, "turn_ids": turn_ids}
        if not infer:
            out["status"] = "logged"
        elif wait:
            report = self.system.consolidate(space)
            out["results"] = _events(before, self.system.db.get_facts(space), user_id, agent_id)
            out["status"] = "consolidated"
            out["consolidation"] = report.as_dict()
        else:
            out["status"] = "queued"
        return out

    def consolidate(
        self, *, user_id: str | None = None, agent_id: str | None = None, rebuild: bool = False
    ) -> dict[str, Any]:
        """同步执行待处理的抽取（包括 infer=False 写入的会话）。rebuild=True 时全部重新抽取。"""
        return self.system.consolidate(space_for(user_id, agent_id), rebuild=rebuild).as_dict()

    def wait(self) -> None:
        """等后台线程把已登记的抽取做完（测试、脚本退出前用）。"""
        if self.system.worker is not None:
            self.system.worker.wait()

    # ---------- 读取 ----------

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
        mode: SearchMode = "auto",
        at: datetime | None = None,
        router: Router = "learned",
        lam: float | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """检索（不回答）。mode="auto" 由检索控制器选 L0 / L1 / L2，结果里的 level 与 reasons
        说明它为什么这么选；L0 表示这句话不需要回忆，只返回画像。limit 截断结果条数。

        filters={"user_id": …, "agent_id": …} 与 top_k 是 Mem0 v3 的写法，与 user_id、limit 等价。
        """
        user_id, agent_id = _scope(user_id, agent_id, filters)
        limit = limit or top_k
        space = space_for(user_id, agent_id)
        if not self.system.db.get_sessions(space):
            return _empty(self.system, space)
        recall = self.system.retrieve(
            space, query, question_time=at, mode=mode, router=router, lam=lam
        )
        return _recall_out(recall, user_id, agent_id, limit)

    def answer(
        self,
        question: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: AnswerMode = "auto",
        at: datetime | None = None,
        router: Router = "learned",
        lam: float | None = None,
    ) -> dict[str, Any]:
        """检索并回答，回答里的引用指向具体的事实与对话片段。"""
        space = space_for(user_id, agent_id)
        answer = self.system.ask(
            space, question, question_time=at, mode=mode, router=router, lam=lam
        )
        return _answer_out(answer)

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        include_superseded: bool = False,
        valid_at: str | None = None,
        known_at: datetime | None = None,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """全部事实：默认只给当前有效的。两条时间轴可以分别回放：valid_at="2025-06" 是那时
        现实中成立的事实，known_at 是系统在那一刻相信的事实。"""
        user_id, agent_id = _scope(user_id, agent_id, filters)
        limit = limit or top_k
        space = space_for(user_id, agent_id)
        current = not include_superseded and valid_at is None and known_at is None
        facts = self.system.facts(space, current_only=current, valid_at=valid_at, known_at=known_at)
        items = [_fact_item(f, None, user_id, agent_id) for f in facts]
        return {"results": items[:limit] if limit else items}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """按 ID 取一条事实（不用给 user_id：ID 里已经带着空间）。"""
        found = self._find(memory_id)
        return _fact_item(found[1], None, None, None, space=found[0]) if found else None

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        """一条事实的版本链，从最早到最新：每个版本何时记下、何时因为什么被取代。"""
        found = self._find(memory_id)
        if found is None:
            return []
        space, fact = found
        chain = [fact]
        older = self.system.db.get_facts(space, superseded_by=[fact.id])
        while older:  # 往前追：被它取代的旧版本（一条事实可以同时取代几条）
            chain = [*older, *chain]
            older = self.system.db.get_facts(space, superseded_by=[f.id for f in older])
        newer = fact
        while newer.superseded_by:  # 往后追：取代它的新版本
            nxt = self.system.db.get_facts(space, ids=[newer.superseded_by])
            if not nxt:
                break
            newer = nxt[0]
            chain.append(newer)
        events = []
        for version in sorted(chain, key=lambda f: (f.recorded_at, f.id)):
            item = _fact_item(version, None, None, None, space=space)
            item["event"] = (
                "ADD" if not any(f.superseded_by == version.id for f in chain) else "UPDATE"
            )
            events.append(item)
        return events

    def profile(self, *, user_id: str | None = None, agent_id: str | None = None) -> str:
        """用户画像（核心记忆）：重要且当前有效的事实，适合常驻在系统提示里。"""
        core = self.system.core_memory(space_for(user_id, agent_id))
        return core.text if core is not None else ""

    # ---------- 遗忘 ----------

    def delete(self, memory_id: str) -> dict[str, Any]:
        """遗忘一条事实（被遗忘权）：删掉它出自的那一轮对话与没有引用的转述，再用剩下的日志
        重建派生记忆。被它取代过的旧事实会恢复为当前有效。"""
        found = self._find(memory_id)
        if found is None:
            raise KeyError(f"No memory with id {memory_id}")
        report = self.system.forget_memories(found[0], fact_ids=[memory_id])
        return {"message": "Memory deleted", **_forget_out(report)}

    def delete_all(
        self, *, user_id: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
        """删除这个用户（或 Agent）的全部记忆：原始日志、派生记忆、向量、缓存与原图。"""
        report = self.system.delete_space(space_for(user_id, agent_id))
        return {"message": "All memories deleted", **_forget_out(report)}

    # ---------- 其他 ----------

    def spaces(self) -> list[dict[str, Any]]:
        return self.system.db.list_spaces()

    def close(self) -> None:
        self.system.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _find(self, memory_id: str) -> tuple[str, Fact] | None:
        space = self.system.db.fact_space(memory_id)
        facts = self.system.db.get_facts(space, ids=[memory_id]) if space is not None else []
        return (space, facts[0]) if space is not None and facts else None

    def _next_idx(self, space: str, session_id: str) -> int:
        """会话里下一句发言的序号：只增不减，遗忘删掉的序号不再复用（与 Agent 相同）；
        新会话（包括删掉后重建的空间）从 0 开始，否则抽取时会被当成有发言被删过。"""
        turns = self.system.db.get_turns(space, session_id)
        if not turns:
            return 0
        recorded = self.system.db.get_meta(_idx_key(space, session_id))
        return max(int(recorded or 0), max(t.idx for t in turns) + 1)


class AsyncMemory:
    """Memory 的异步版本（与 Mem0 的 AsyncMemory 对应）：同样的方法，在线程池里执行。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.sync = Memory(*args, **kwargs)

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> AsyncMemory:
        return cls(config, **kwargs)

    def __getattr__(self, name: str) -> Any:
        method = getattr(self.sync, name)
        if not callable(method) or name.startswith("_"):
            return method
        return _threaded(method)

    async def __aenter__(self) -> AsyncMemory:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.to_thread(self.sync.close)


def _threaded[**P, R](method: Callable[P, R]) -> Callable[P, Any]:
    @functools.wraps(method)
    async def run(*args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.to_thread(method, *args, **kwargs)

    return run


# ---------- 输入 ----------


def parse_messages(messages: MessagesIn, when: datetime) -> tuple[list[TurnIn], list[Path]]:
    """OpenAI 格式的消息 → 发言。返回发言与需要事后删除的临时图片文件。"""
    if isinstance(messages, str):
        items: list[Message] = [{"role": "user", "content": messages}]
    elif isinstance(messages, Mapping):
        items = [messages]
    else:
        items = list(messages)
    turns: list[TurnIn] = []
    temporary: list[Path] = []
    for message in items:
        role = message.get("role", "user")
        if role not in ("user", "assistant"):
            continue
        text, images, made = _content(message.get("content"))
        temporary.extend(made)
        if not text.strip() and not images:
            continue
        turns.append(
            TurnIn(role=role, text=text, ts=when, speaker=message.get("name"), images=images)
        )
    return turns, temporary


def _content(content: Any) -> tuple[str, list[str], list[Path]]:
    """内容（字符串或内容块）→ 文本、图片路径、临时文件。图片在文本里用 "<image>" 标出位置。"""
    if content is None:
        return "", [], []
    if isinstance(content, str):
        return content, [], []
    parts: list[str] = []
    images: list[str] = []
    made: list[Path] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            parts.append(block.get("text", ""))
        elif kind == "image_url":
            url = (
                block["image_url"]["url"]
                if isinstance(block["image_url"], dict)
                else block["image_url"]
            )
            path, temporary = _image_file(url)
            images.append(str(path))
            if temporary:
                made.append(path)
            parts.append("<image>")
    return " ".join(p for p in parts if p), images, made


def _image_file(url: str) -> tuple[Path, bool]:
    """image_url → 本地文件（data URL 先写成临时文件）。网络图片要先下载。"""
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        mime = header[5:].split(";")[0]
        try:
            data = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 data in an image data URL.") from exc
        with tempfile.NamedTemporaryFile(
            suffix=_IMAGE_SUFFIX.get(mime, ".img"), prefix="cuemem-", delete=False
        ) as handle:
            handle.write(data)
        return Path(handle.name), True
    if url.startswith(("http://", "https://")):
        raise ValueError(
            "Image URLs are not fetched: download the image and pass a local path or a data URL."
        )
    path = Path(url.removeprefix("file://")).expanduser()
    if not path.is_file():
        raise ValueError(f"Image not found: {url}")
    return path, False


def _scope(
    user_id: str | None, agent_id: str | None, filters: Mapping[str, Any] | None
) -> tuple[str | None, str | None]:
    """Mem0 v3 把 user_id、agent_id 放在 filters 里；两种写法都认。"""
    if not filters:
        return user_id, agent_id
    unknown = set(filters) - {"user_id", "agent_id", "run_id"}
    if unknown:
        raise ValueError(
            f"Unsupported filters: {sorted(unknown)}. Memories are scoped by user_id and agent_id."
        )
    return filters.get("user_id", user_id), filters.get("agent_id", agent_id)


def _as_datetime(value: datetime | str | float | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return datetime.fromtimestamp(value)


def _idx_key(space: str, session_id: str) -> str:
    return f"next_idx:{space}:{session_id}"


# ---------- 输出 ----------


def _fact_item(
    fact: Fact,
    score: float | None,
    user_id: str | None,
    agent_id: str | None,
    *,
    space: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": fact.id,
        "memory": fact.statement,
        "type": "fact",
        "subject": fact.subject,
        "predicate": fact.predicate,
        "object": fact.object,
        "category": fact.category,
        "importance": fact.importance,
        "status": fact_status(fact, {}),
        "valid_from": fact.valid_from,
        "valid_to": fact.valid_to,
        "created_at": fact.recorded_at.isoformat(),
        "invalidated_at": fact.invalidated_at.isoformat() if fact.invalidated_at else None,
        "superseded_by": fact.superseded_by,
        "session_id": fact.session_id,
        "source_turn_ids": fact.source_turn_ids,
    }
    if score is not None:
        item["score"] = round(score, 4)
    if space is not None:
        item["space_id"] = space
    if user_id is not None:
        item["user_id"] = user_id
    if agent_id is not None:
        item["agent_id"] = agent_id
    return item


def _excerpt_item(hit: Hit) -> dict[str, Any]:
    return {
        "id": hit.unit.id,
        "memory": hit.unit.text,
        "type": "excerpt",
        "score": round(hit.score, 4),
        "created_at": hit.unit.ts.isoformat(),
        "session_id": hit.unit.session_id,
        "source_turn_ids": hit.unit.source_turn_ids,
    }


def _image_item(item: MediaItem, attached: bool) -> dict[str, Any]:
    return {
        "id": item.id,
        "memory": item.caption,
        "type": "image",
        "path": item.path,
        "created_at": item.ts.isoformat(),
        "attached": attached,
    }


def _recall_out(
    recall: Recall, user_id: str | None, agent_id: str | None, limit: int | None
) -> dict[str, Any]:
    view = recall.view
    facts: list[FactHit] = view.facts
    items = [_fact_item(fh.fact, fh.score, user_id, agent_id) for fh in facts]
    items += [_excerpt_item(h) for h in sorted(view.rounds, key=lambda h: -h.score)]
    decision = recall.decision
    return {
        "results": items[:limit] if limit else items,
        "images": [_image_item(m, m.id in view.attach) for m in view.media.values()],
        "profile": view.core.text if view.core is not None else "",
        "level": recall.level.upper(),
        "see": bool(decision and decision.see),
        "reasons": [english_reason(r) for r in decision.reasons] if decision else [],
    }


def _empty(system: MemorySystem, space: str) -> dict[str, Any]:
    core = system.core_memory(space)
    return {
        "results": [],
        "images": [],
        "profile": core.text if core is not None else "",
        "level": "L0",
        "see": False,
        "reasons": ["no memories yet"],
    }


def _answer_out(answer: Answer) -> dict[str, Any]:
    facts = {fh.fact.id: fh.fact.statement for fh in answer.facts}
    units = {h.unit.id: h.unit.text for h in answer.hits}
    route = answer.details.get("route") or {}
    return {
        "answer": answer.answer,
        "evidence_sufficient": answer.evidence_sufficient,
        "citations": [
            {
                "ref": ref,
                "id": answer.refs[ref],
                "memory": facts.get(answer.refs[ref]) or units.get(answer.refs[ref], ""),
            }
            for ref in answer.citations
            if ref in answer.refs
        ],
        "level": str(route.get("level", "")).upper() or None,
        "reasons": [english_reason(r) for r in route.get("reasons", [])],
    }


def _events(
    before: Mapping[str, Fact], after: Sequence[Fact], user_id: str | None, agent_id: str | None
) -> list[dict[str, Any]]:
    """抽取前后的事实对比 → Mem0 风格的事件：新事实 ADD；取代了旧事实的新事实 UPDATE，
    附上被取代的旧版本（旧版本仍留在版本链里，不会被删除）。"""
    replaced: dict[str, list[Fact]] = {}
    for fact in after:
        old = before.get(fact.id)
        if old is not None and old.is_current and fact.superseded_by:
            replaced.setdefault(fact.superseded_by, []).append(fact)
    events = []
    for fact in after:
        if fact.id in before:
            continue
        item = _fact_item(fact, None, user_id, agent_id)
        previous = replaced.get(fact.id, [])
        item["event"] = "UPDATE" if previous else "ADD"
        if previous:
            item["previous_memory"] = [p.statement for p in previous]
            item["reason"] = previous[0].supersede_reason
        events.append(item)
    return events


def _forget_out(report: ForgetReport) -> dict[str, Any]:
    return {
        "turns_deleted": report.turns_deleted,
        "sessions_affected": report.sessions_affected,
        "facts_before": report.facts_before,
        "facts_after": report.facts_after,
        "images_deleted": report.images_deleted,
        "restated": report.restated,
    }
