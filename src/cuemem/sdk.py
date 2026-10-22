"""Python SDK（N12，D-061）：照 Mem0 的接口形状包一层，几行代码就能用上 CueMem。

    from cuemem import Memory

    m = Memory(preset="openai")                       # 或 "ollama"（本地），不填用默认配置
    m.add([{"role": "user", "content": "I moved to Berlin last month"}], user_id="alice")
    m.search("Where do I live?", user_id="alice")     # 检索控制器决定要不要查、查多深
    m.answer("Where do I live?", user_id="alice")     # 检索并回答（带引用）
    m.history(fact_id)                                # 一条事实的版本链：何时被取代、为什么

与 Mem0 对应的方法名与参数（add / search / get_all / get / history / update / delete /
delete_all / reset）保持一致，返回值也是 {"results": [...]}、每条带 "memory" 文本；不同的地方是
CueMem 自己的设计：

- 记忆空间：user_id 与 agent_id 合起来定一个空间（各空间完全隔离）；run_id 是空间里的一个会话，
  search / get_all / delete_all 给了 run_id 时只看、只删这个会话的记忆；
- add 的 metadata 随发言存在原始层（不进抽取），事实与对话片段带上出处各句的 metadata，
  filters 里可以按它筛（Mem0 的 eq / ne / in / gte / lte… 与 AND / OR / NOT）；expiration_date
  过期之后 search、get_all 与画像里不再出现（get 与 history 照样取得到，数据要 delete 才删）；
- 写入分两步：add 先在线写入原始日志（不调用 LLM，立即可检索），抽取事实在后台完成；
  wait=True 时同步抽取，并按 Mem0 的格式返回 ADD / UPDATE 事件；一段对话分多次写入时用
  run() 包起来，结束时整段只抽取一次；
- 事实不就地改写、也不删除旧版本：被取代的事实留在版本链里（history），get_all 默认只给当前的；
  update 是记进原始日志的一句更正，由它抽出的新事实取代原来那条；
- search 先过检索控制器，由它决定查多深；与 MCP、各框架集成的 recall 相同，不低于 L1（控制器
  选了 L0 时改走 L1，reasons 写明），min_level="l0" 时照控制器的决定、与记忆无关的问题只给画像；
  answer 照旧由控制器决定要不要回忆；
- delete 是"被遗忘权"：删掉事实出自的对话，再用剩下的日志重建，连带清理转述、缓存与原图。

返回值、参数名与 SDK 自己的报错信息用英文（与 HTTP 的 /api/v1 记忆接口、MCP、界面一致；
较早的 /api 接口与核心模块更深处的报错是中文，见 docs/en/api.md）；SDK 只组装已有的组件，
检索、巩固、遗忘的逻辑都在 MemorySystem 里。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import functools
import json
import logging
import tempfile
import threading
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

import httpx2

from cuemem.config import Settings, load_settings
from cuemem.core.ids import sortable_session_id, stable_id
from cuemem.core.models import (
    Answer,
    CoreMemory,
    Fact,
    FactHit,
    Hit,
    MediaItem,
    SessionIn,
    TurnIn,
    TurnMeta,
)
from cuemem.core.timeutil import expiry_instant, naive_local, normalize_expiry
from cuemem.core.types import PeakPolicy, Router
from cuemem.errors import ConcurrentWriteError

# 记忆系统（向量库、模型客户端）在 Memory() 时才导入：import cuemem.sdk 只要零点几秒
if TYPE_CHECKING:
    from cuemem.memory import ForgetReport, MemorySystem, Recall

logger = logging.getLogger(__name__)

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
    """user_id、agent_id → 记忆空间 ID。两者都给时是这个 Agent 对这个用户的记忆。

    "/" 是分隔符，所以 ID 里的 "/" 与 "%" 先转义（URL 的百分号编码）：否则 user_id="agent/x"
    与 agent_id="x" 会落进同一个空间。不含这两个字符的 ID 原样保留（"alice" 还是 "alice"）。"""
    if user_id and agent_id:
        return f"{_part(agent_id)}/{_part(user_id)}"
    if user_id:
        return _part(user_id)
    if agent_id:
        return f"agent/{_part(agent_id)}"
    raise ValueError("Give user_id, agent_id, or both: memories are kept per user and per agent.")


def space_owner(space: str) -> tuple[str | None, str | None] | None:
    """space_for 的逆：这个空间是哪个（user_id，agent_id）的（访问令牌的作用域据此判断）。

    "agent/x" 是 Agent x 自己的记忆（None，"x"）：用户 x 在名叫 agent 的 Agent 下的记忆
    （space_for("x", "agent")）也落在这里，但不能因此算作用户 x 的，否则只能访问用户 x 的令牌
    借 agent_id="agent" 就读到了 Agent x 的记忆。不是由 space_for 得来的空间（评测、手工起的名字
    里有两个以上的 "/"，或者 "%" 的写法与 space_for 不同）返回 None。解出来的 ID 再过一遍
    space_for，逐字相同才算。"""
    left, slash, right = space.partition("/")
    owner: tuple[str | None, str | None]
    if not slash:
        owner = (_unpart(space), None)
    elif left == "agent":
        owner = (None, _unpart(right))
    else:
        owner = (_unpart(right), _unpart(left))
    try:
        return owner if space_for(*owner) == space else None
    except ValueError:  # 空的 ID
        return None


def _part(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _unpart(value: str) -> str:
    return value.replace("%2F", "/").replace("%25", "%")


def _log_peak_wait(provider: str, until: datetime) -> None:
    """peak_policy="wait" 时开始暂停：记一条警告，不让调用方以为卡死了。"""
    logger.warning(
        "%s peak hours: pausing model calls until %s (peak_policy='wait'; "
        "pass peak_policy='warn' to call now at peak price).",
        provider,
        until.isoformat(timespec="minutes"),
    )


class Memory:
    """CueMem 的 Python SDK。线程安全；用完调用 close()，或用 with 语句。

    config：嵌套的字典，与默认配置（及预设）深度合并，例如
    {"roles": {"reader": {"model": "qwen3.8-flash"}}}；home：数据目录（默认 CUEMEM_HOME，
    源码仓库里是仓库根目录，否则 ~/.cuemem）；instructions：给抽取的补充说明（记什么、不记什么）；
    background：后台线程抽取事实（默认开）。

    peak_policy：DeepSeek 分时价格的高峰时段里需要真实调用时怎么办（默认配置的回答模型就是
    DeepSeek）。SDK 是交互式的，缺省 "warn"：照常按高峰价调用，trace 里记一条 peak_call，与
    cuemem ask / serve / mcp / proxy 一致（D-024）。批量写入想等高峰结束、拿错峰折扣时传 "wait"
    （开始等待时记一条警告日志），"allow" 不提醒。这个参数与 config 里的 peak_policy 优先于环境
    变量 CUEMEM_PEAK_POLICY 与 YAML 配置（与 CLI 相同）；config 是 Settings 对象时保留它自己的
    值，除非显式传了这个参数。
    """

    def __init__(
        self,
        config: Mapping[str, Any] | Settings | None = None,
        *,
        preset: str | None = None,
        home: str | Path | None = None,
        instructions: str | None = None,
        background: bool = True,
        peak_policy: PeakPolicy | None = None,
        http_client: httpx2.Client | None = None,
    ) -> None:
        if isinstance(config, Settings):
            settings = config
            if peak_policy is not None:
                settings = settings.model_copy(update={"peak_policy": peak_policy})
        else:
            overrides = dict(config or {})
            if peak_policy is not None:
                overrides["peak_policy"] = peak_policy
            # 交互式调用不继承配置里批量任务用的 wait：高峰最长 4 小时，调用方看起来像卡死（D-024）
            overrides.setdefault("peak_policy", "warn")
            settings = load_settings(preset=preset, home=home, **overrides)
        if instructions is not None:  # 抽取的补充说明（Mem0 叫 custom_instructions）
            consolidation = settings.consolidation.model_copy(update={"instructions": instructions})
            settings = settings.model_copy(update={"consolidation": consolidation})
        self.settings = settings
        from cuemem.memory import MemorySystem

        self.system = MemorySystem.from_settings(
            settings, http_client=http_client, on_peak_wait=_log_peak_wait
        )
        self._runs: set[tuple[str, str]] = set()  # open_run 开始、还没结束的（空间，run_id）
        self._adopted: set[str] = set()  # 接手过崩溃前留下的 run 的空间（见 _adopt）
        self._runs_lock = threading.Lock()
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
        memory._runs = set()
        memory._adopted = set()
        memory._runs_lock = threading.Lock()
        return memory

    def _adopt(self, space: str, keep: Sequence[str] = ()) -> None:
        """这个对象第一次碰到这个空间：之前的进程崩溃时还开着的 run（open_run 了、没来得及
        end_run 或 close()），以及 Agent 与记忆代理的会话，没有抽取任务、也没人再结束它们，
        交给抽取（MemorySystem.adopt_open；没有后台线程时只登记，下次 consolidate 时完成）。
        keep：调用方马上要接着写的 run（open_run 重新打开它）。"""
        with self._runs_lock:
            if space in self._adopted:
                return
            self._adopted.add(space)
        self.system.adopt_open(space, keep=keep, start=False)

    # ---------- 写入 ----------

    def add(
        self,
        messages: MessagesIn,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        expiration_date: date | datetime | str | None = None,
        timestamp: datetime | str | float | None = None,
        infer: bool = True,
        wait: bool = False,
    ) -> dict[str, Any]:
        """写入对话。messages 可以是一句话、一条消息或 OpenAI 格式的消息列表（内容块里的
        image_url 可以是 data URL 或本地路径，图片会存进记忆）；system / tool 消息不写入。

        run_id：写进同一个会话（一次对话），不填时这次 add 自成一个会话。每轮 add 一次同一个
        run_id 时，每次都把整段对话重新抽取一遍（巩固过的会话又长了还要重放派生记忆）：一段
        对话分多次写入时用 run()（或 open_run / end_run）包起来，结束时只抽取一次。
        metadata：JSON 对象，记在这次写入的每句发言上（不进抽取，不改变抽取的请求）；由它们
        抽出的事实、对话片段在结果里带上 metadata，search / get_all 的 filters 可以按它筛。
        expiration_date：过期时间（Mem0 的写法）。只给日期时到那天结束为止；过期之后 search、
        get_all 与画像里不再出现，get / history 照样取得到。同一件事后来又说过一次、没给过期
        时间，它就不过期。
        infer=False：只写原始日志、不抽取事实（之后调用 consolidate）。
        wait=True：同步抽取，返回新增与取代的事实（Mem0 的 ADD / UPDATE 事件）；run 进行中时
        不能用（这段对话结束时才抽取），会报 ValueError。发言写进去之后抽取出错（模型供应商
        余额不足、限流等）时不抛出：status 是 "queued"，error 写明原因，下次巩固时补上——抛出的话
        调用方重试就写了两遍。写进去之前出错（向量接口不可用等）时照常抛出，什么都没写，可以原样
        重试。
        """
        space = space_for(user_id, agent_id)
        if infer and wait and run_id is not None and run_id in self.system.open_sessions(space):
            raise ValueError(
                f"Run {run_id!r} is still open: its memories are extracted when it ends. "
                "Add without wait=True, then call end_run(..., wait=True)."
            )
        info = _turn_meta(metadata, expiration_date)
        # 缺省时间不截到秒：同一秒里的几次 add 也分得出先后（巩固按开始时间排序对账）
        when = _as_datetime(timestamp) or datetime.now()
        turns, temporary = parse_messages(messages, when)
        try:
            if not turns:
                return {"results": [], "session_id": None, "turn_ids": [], "status": "empty"}
            self._adopt(space)  # 没有重新 open_run 的 run_id 就是一个普通的会话，一起接手
            sid = run_id or sortable_session_id("add", when)
            # 只取 ID 与是否当前有效：以前前后各读一遍全部事实（9,600 条时每次约 0.29 秒）
            before = self.system.db.fact_links(space) if wait else {}
            # 发言序号在会话锁里分配、写完再放：同一个 run_id 的并发 add 不会撞号
            with self.system.session_lock(space, sid):
                first = self._next_idx(space, sid)
                started = self.system.db.get_sessions(space).get(sid, when)
                written = self.system.append(
                    space,
                    SessionIn(session_id=sid, started_at=started, turns=turns),
                    first,
                    enqueue=infer,
                    meta=[info] * len(turns) if info is not None else None,
                )
                self.system.db.set_meta(_idx_key(space, sid), str(first + len(turns)))
            if written.turns_added != len(turns):  # 撞号的发言被 INSERT OR IGNORE 悄悄跳过了
                raise ConcurrentWriteError(
                    f"Only {written.turns_added} of {len(turns)} messages were written to session "
                    f"{sid!r}: their turn numbers were already taken by another write."
                )
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)
        turn_ids = [stable_id(space, sid, first + i) for i in range(len(turns))]
        out: dict[str, Any] = {"results": [], "session_id": sid, "turn_ids": turn_ids}
        if not infer:
            out["status"] = "logged"
        elif wait:
            try:
                report = self.system.consolidate(space)
            except Exception as exc:  # 发言已经写进去了：抛出的话调用方重试，就写了两遍（R1）
                out["status"] = "queued"
                out["error"] = _not_extracted(exc)
            else:
                out["results"] = _events(self.system, space, before, user_id, agent_id)
                out["status"] = "consolidated"
                out["consolidation"] = report.as_dict()
        else:
            out["status"] = "queued"
        return out

    def open_run(
        self, run_id: str | None = None, *, user_id: str | None = None, agent_id: str | None = None
    ) -> str:
        """开始一段对话（run）：之后 add(run_id=...) 照常在线写入、立即可检索，但不抽取事实，
        end_run 时整段只抽取一次。返回 run_id（不给时生成一个）。

        进程崩溃时还开着的 run 由下一个进程第一次碰到这个用户时交给抽取（见 _adopt）；用同一个
        run_id 再 open_run 就是接着这段对话写，它不被接手。"""
        space = space_for(user_id, agent_id)
        rid = run_id or sortable_session_id("run", datetime.now())
        self._adopt(space, keep=[rid])
        self.system.open_session(space, rid)
        with self._runs_lock:
            self._runs.add((space, rid))
        return rid

    def end_run(
        self,
        run_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """结束一段对话：整段交给抽取。wait=True 时同步抽取并返回报告；否则交给后台线程
        （没开后台线程时只登记，下次 consolidate 时完成），consolidation 是 "queued"。"""
        space = space_for(user_id, agent_id)
        with self._runs_lock:
            self._runs.discard((space, run_id))
        turns = len(self.system.db.get_turns(space, run_id))
        if not turns:  # 什么都没写：只结束，不抽取
            self.system.close_session(space, run_id, submit=False)
            return {"session_id": run_id, "turns": 0, "consolidation": None}
        background = self.system.worker is not None
        report = self.system.close_session(space, run_id, wait=wait, submit=background)
        return {
            "session_id": run_id,
            "turns": turns,
            "consolidation": report.as_dict() if report is not None else "queued",
        }

    @contextmanager
    def run(
        self,
        run_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> Iterator[str]:
        """with m.run(user_id="alice") as run_id: ...：进入时 open_run，退出时 end_run（出错时
        也结束）。wait=True 时退出前同步抽取。"""
        rid = self.open_run(run_id, user_id=user_id, agent_id=agent_id)
        try:
            yield rid
        finally:
            self.end_run(rid, user_id=user_id, agent_id=agent_id, wait=wait)

    def update(
        self, memory_id: str, data: str, *, metadata: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """改一条记忆（Mem0 的 update）。事实不就地改写（派生记忆是原始日志的函数，D-021、
        D-061）：像用户那样说一句更正——"Correction: <data>" 记进一段新的会话（edit-…），原始层
        记下它要取代的事实（edit_of）——再同步巩固：由这句抽出的新事实以"说错了"（corrected）
        取代原来那条，不问 LLM；history() 里看得到前后两个版本，遗忘这句更正时原来那条恢复。

        新事实带上原来那条的 metadata（metadata 参数里的键覆盖它）与过期时间。返回 Mem0 风格的
        事件（UPDATE 附上被取代的旧版本）；从 data 里没有抽出事实时，原来那条不变，message 写明，
        这句更正也从日志里遗忘（turn_ids 是空的）。
        更正记下之后抽取出错时不抛出（与 add 的 wait=True 相同）：message 与 error 写明，下次
        巩固时补上。
        只能改当前有效的版本（ValueError）；ID 不存在时 KeyError。"""
        found = self._find(memory_id)
        if found is None:
            raise KeyError(f"No memory with id {memory_id}")
        space, fact = found
        if not fact.is_current:
            raise ValueError(
                f"Memory {memory_id} was already replaced by {fact.superseded_by}: "
                "update the current version (see history())."
            )
        if not data.strip():
            raise ValueError("data is empty: to remove a memory, call delete().")
        sources = _Sources.load(self.system, space, fact.source_turn_ids)
        inherited = {**sources.metadata(fact.source_turn_ids), **dict(metadata or {})}
        info = _turn_meta(inherited, sources.expires(fact.source_turn_ids)) or TurnMeta()
        info = info.model_copy(update={"edit_of": memory_id})
        self._adopt(space)
        # 巩固按会话开始的时间排序：更正排在原来那条之后（调用方写入时可能给了将来的时间）
        when = max(datetime.now(), fact.recorded_at + timedelta(microseconds=1))
        sid = sortable_session_id("edit", when)
        turn = TurnIn(role="user", text=f"Correction: {data.strip()}", ts=when)
        before = self.system.db.fact_links(space)
        with self.system.session_lock(space, sid):
            self.system.append(
                space, SessionIn(session_id=sid, started_at=when, turns=[turn]), 0, meta=[info]
            )
            self.system.db.set_meta(_idx_key(space, sid), "1")
        failed = "Correction logged, but extraction failed: it is retried on consolidate()"
        turn_ids = [stable_id(space, sid, 0)]
        try:
            report = self.system.consolidate(space)
        except Exception as exc:  # 更正已经记下了：抛出的话调用方重试，就记了两句更正
            return {
                "message": failed,
                "id": memory_id,
                "results": [],
                "session_id": sid,
                "turn_ids": turn_ids,
                "error": _not_extracted(exc),
            }
        (after,) = self.system.db.get_facts(space, ids=[memory_id]) or [fact]
        extra: dict[str, Any] = {}
        if not after.is_current:
            message = "Memory updated"
        elif sid in report.failed:
            message = failed
        else:
            # 没有抽出取代它的事实：这句更正也不留在日志里。否则它作为对话片段与原来那条一起被
            # 检索出来、随导出带走，换了抽取模型重建时还可能取代原来那条——与"没有改"不符
            message = "No memory could be extracted from data: the memory was not changed"
            dropped = self.system.forget_memories(
                space, session_ids=[sid], report_rebuild_errors=True
            )
            turn_ids = []
            if dropped.rebuild_error:
                extra["error"] = _forget_out(dropped)["warning"]
        events = _events(self.system, space, before, None, None)
        for event in events:
            event["space_id"] = space
        return {
            "message": message,
            "id": memory_id,
            "results": events,
            "session_id": sid,
            "turn_ids": turn_ids,
            "consolidation": report.as_dict(),
            **extra,
        }

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
        run_id: str | None = None,
        limit: int | None = None,
        mode: SearchMode = "auto",
        at: datetime | None = None,
        router: Router = "learned",
        lam: float | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        min_level: Literal["l0", "l1"] = "l1",
    ) -> dict[str, Any]:
        """检索（不回答）。mode="auto" 由检索控制器选档，结果里的 level 与 reasons 说明它为什么
        这么选；limit 截断结果条数。调用 search 就是已经决定要查记忆：与 MCP、各框架集成的
        recall 相同，控制器选了 L0（这句话不需要回忆）时改走 L1，reasons 写明，控制器只决定要不要
        查得更深。min_level="l0" 时照控制器的决定，L0 只返回画像；mode="l0" 直接只要画像。

        filters={"user_id": …, "agent_id": …, "run_id": …} 与 top_k 是 Mem0 v3 的写法，与
        user_id、limit 等价。run_id：只要这个会话里的事实、对话片段与图片。filters 里的其他键是
        metadata 的条件（见 _check_where）。检索照常在整个空间里做（控制器与 L1 / L2 不变），
        再按会话、条件与过期时间筛，所以结果可能少于 limit；画像是整个空间的（去掉过期的事实）。
        """
        user_id, agent_id, run_id, where = _scope(user_id, agent_id, run_id, filters)
        limit = limit or top_k
        space = space_for(user_id, agent_id)
        if not self.system.db.get_sessions(space):
            return _empty(self.system, space)
        from cuemem.evidence import RAISED_REASON

        # 两次检索用同一个问题时间：改走 L1 时打包的是探针在这个时间打过分的候选
        when = naive_local(at) if at is not None else datetime.now()
        recall = self.system.retrieve(
            space, query, question_time=when, mode=mode, router=router, lam=lam
        )
        raised = mode == "auto" and min_level == "l1" and recall.level == "l0"
        if raised:  # 与 evidence.recall_listing 的 at_least_l1 相同：不把探针的检索再做一遍
            recall = self.system.retrieve(
                space, query, question_time=when, mode="l1", candidates=recall.candidates
            )
        select = _Select(run_id, where, _scope_record(user_id, agent_id), datetime.now())
        out = _recall_out(self.system, space, recall, user_id, agent_id, limit, select)
        if raised:
            out["reasons"] = [RAISED_REASON]
        return out

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
        run_id: str | None = None,
        include_superseded: bool = False,
        valid_at: str | None = None,
        known_at: datetime | None = None,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """全部事实：默认只给当前有效的。两条时间轴可以分别回放：valid_at="2025-06" 是那时
        现实中成立的事实，known_at 是系统在那一刻相信的事实。run_id：只要这个会话里记下的事实；
        filters 里的其他键是 metadata 的条件（同 search）。过期的事实不给。"""
        user_id, agent_id, run_id, where = _scope(user_id, agent_id, run_id, filters)
        limit = limit or top_k
        space = space_for(user_id, agent_id)
        current = not include_superseded and valid_at is None and known_at is None
        facts = self.system.facts(space, current_only=current, valid_at=valid_at, known_at=known_at)
        sources = _Sources.load(self.system, space, [t for f in facts for t in f.source_turn_ids])
        select = _Select(run_id, where, _scope_record(user_id, agent_id), datetime.now())
        items = [
            _fact_item(f, None, user_id, agent_id, sources)
            for f in facts
            if select.keep(f.session_id, f.source_turn_ids, sources)
        ]
        return {"results": items[:limit] if limit else items}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """按 ID 取一条事实（不用给 user_id：ID 里已经带着空间）。过期的也取得到。"""
        found = self._find(memory_id)
        if found is None:
            return None
        space, fact = found
        sources = _Sources.load(self.system, space, fact.source_turn_ids)
        return _fact_item(fact, None, None, None, sources, space=space)

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
        sources = _Sources.load(self.system, space, [t for f in chain for t in f.source_turn_ids])
        events = []
        for version in sorted(chain, key=lambda f: (f.recorded_at, f.id)):
            item = _fact_item(version, None, None, None, sources, space=space)
            item["event"] = (
                "ADD" if not any(f.superseded_by == version.id for f in chain) else "UPDATE"
            )
            events.append(item)
        return events

    def profile(self, *, user_id: str | None = None, agent_id: str | None = None) -> str:
        """用户画像（核心记忆）：重要且当前有效的事实，适合常驻在系统提示里。过期的事实不给。"""
        space = space_for(user_id, agent_id)
        return _profile(self.system, space, self.system.core_memory(space), datetime.now())

    # ---------- 遗忘 ----------

    def delete(self, memory_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        """遗忘一条事实（被遗忘权）：删掉它出自的那一轮对话与没有引用的转述，再用剩下的日志
        重建派生记忆。被它取代过的旧事实会恢复为当前有效。

        dry_run=True 只预演：返回会删哪些发言、哪些事实会随之消失，什么都不删。删除做完之后
        重建出错（模型供应商余额不足等）时不抛出：已经删了，warning 写明重建下次巩固时补上。"""
        found = self._find(memory_id)
        if found is None:
            raise KeyError(f"No memory with id {memory_id}")
        if dry_run:
            return {
                "message": "Dry run: nothing was deleted",
                **self.system.preview_forget(found[0], fact_ids=[memory_id]),
            }
        report = self.system.forget_memories(
            found[0], fact_ids=[memory_id], report_rebuild_errors=True
        )
        return {"message": "Memory deleted", **_forget_out(report)}

    def delete_all(
        self, *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        """删除这个用户（或 Agent）的全部记忆：原始日志、派生记忆、向量、缓存与原图。
        run_id：只删这个会话（与按会话遗忘相同，再用剩下的日志重建派生记忆），空间留着。"""
        space = space_for(user_id, agent_id)
        if run_id is not None:
            report = self.system.forget_memories(
                space, session_ids=[run_id], report_rebuild_errors=True
            )
            return {"message": "Memories of this run deleted", **_forget_out(report)}
        report = self.system.delete_space(space)
        return {"message": "All memories deleted", **_forget_out(report)}

    def reset(self) -> dict[str, Any]:
        """删除这个数据目录里的全部记忆（Mem0 的 reset）：每个空间都按 delete_all 删除（原始
        日志、派生记忆、向量、缓存与原图），包括 HTTP 服务、记忆代理与演示写下的空间；还开着的
        run 一并结束。不可恢复。"""
        with self._runs_lock:
            runs = sorted(self._runs)
            self._runs.clear()
            self._adopted.clear()
        for space, rid in runs:
            self.system.close_session(space, rid, submit=False)
        spaces = [s["space_id"] for s in self.system.db.list_spaces()]
        reports = [self.system.delete_space(space) for space in spaces]
        return {
            "message": "All memories deleted",
            "spaces_deleted": spaces,
            "turns_deleted": sum(r.turns_deleted for r in reports),
            "images_deleted": sum(r.images_deleted for r in reports),
        }

    # ---------- 导出与导入 ----------

    def export(
        self, *, user_id: str | None = None, agent_id: str | None = None, derived: bool = True
    ) -> dict[str, Any]:
        """导出这个用户（或 Agent）的记忆：原始日志与图片原文件，可选附上事实与画像的快照。
        格式见 cuemem/portable.py；json.dump 之后可以在别的机器上 load。"""
        from cuemem.portable import export_space

        return export_space(self.system, space_for(user_id, agent_id), derived=derived)

    def load(
        self,
        data: Mapping[str, Any],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """导入 export 的结果：写回原始日志，派生记忆重新抽取（wait=True 时同步完成）。
        不给 user_id、agent_id 时导入到文件里原来的空间。"""
        from cuemem.portable import import_space

        space = space_for(user_id, agent_id) if user_id or agent_id else None
        report = import_space(self.system, data, space_id=space, consolidate=wait)
        if not wait and self.system.worker is not None:
            self.system.worker.submit(report["space_id"])
        return report

    # ---------- 其他 ----------

    def spaces(self) -> list[dict[str, Any]]:
        return self.system.db.list_spaces()

    def close(self) -> None:
        """关闭。还没结束的 run 先结束：只登记抽取任务（存在数据库里），下次巩固或后台线程
        启动时完成。"""
        with self._runs_lock:
            runs = sorted(self._runs)
            self._runs.clear()
        for space, rid in runs:
            self.system.close_session(space, rid, submit=False)
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

    @asynccontextmanager
    async def run(
        self,
        run_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> AsyncIterator[str]:
        """async with m.run(user_id="alice") as run_id: ...（见 Memory.run）。"""
        sync = self.sync
        rid = await asyncio.to_thread(sync.open_run, run_id, user_id=user_id, agent_id=agent_id)
        try:
            yield rid
        finally:
            await asyncio.to_thread(
                sync.end_run, rid, user_id=user_id, agent_id=agent_id, wait=wait
            )

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


def _turn_meta(
    metadata: Mapping[str, Any] | None, expiration_date: date | datetime | str | None
) -> TurnMeta | None:
    """add 的 metadata 与 expiration_date → 每句发言的附加信息；都没给时是 None（不写附加信息）。
    metadata 要能写成 JSON（导出、HTTP 接口都要用），按 JSON 往返一遍再存。"""
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a dict.")
    if any(not isinstance(key, str) for key in (metadata or {})):
        raise ValueError("metadata keys must be strings.")
    try:
        meta = json.loads(json.dumps(dict(metadata or {}), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata must be JSON-serializable: {exc}") from exc
    expires = normalize_expiry(expiration_date)
    return TurnMeta(metadata=meta, expires_at=expires) if meta or expires else None


_SCOPE_KEYS = ("user_id", "agent_id", "run_id")
_OPERATORS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "nin", "contains", "icontains")
_LOGIC = ("AND", "OR", "NOT")


def _scope(
    user_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    filters: Mapping[str, Any] | None,
) -> tuple[str | None, str | None, str | None, dict[str, Any] | None]:
    """Mem0 v3 把 user_id、agent_id、run_id 放在 filters 里；两种写法都认。返回（user_id，
    agent_id，run_id，metadata 的条件）。filters 里别的键都是 metadata 的条件（见 _check_where）；
    run_id 写成条件（{"run_id": {"in": [...]}}）时也按条件筛，user_id、agent_id 只能是一个 ID：
    它们决定在哪个记忆空间里找。"""
    if not filters:
        return user_id, agent_id, run_id, None
    where = {k: v for k, v in filters.items() if k not in _SCOPE_KEYS}
    for key in ("user_id", "agent_id"):
        if filters.get(key) is not None and not isinstance(filters[key], str):
            raise ValueError(
                f"filters[{key!r}] must be a single id: memories are kept per user and per agent."
            )
    scoped_run = filters.get("run_id")
    if scoped_run is not None and not isinstance(scoped_run, str):
        where["run_id"] = scoped_run
        scoped_run = None
    _check_where(where)
    return (
        filters.get("user_id") or user_id,
        filters.get("agent_id") or agent_id,
        scoped_run or run_id,
        where or None,
    )


def _check_where(where: Mapping[str, Any]) -> None:
    """检查 metadata 的条件（Mem0 的 filters 写法），写错时报 ValueError，而不是悄悄什么都不给：

    - {"key": value}：等于；{"key": "*"}：有这个键；几个键并列时都要满足；
    - {"key": {"eq" | "ne" | "gt" | "gte" | "lt" | "lte": v, "in" | "nin": [...],
      "contains" | "icontains": v}}：几个运算并列时都要满足；没有这个键时只有 ne、nin 成立；
    - {"AND": [条件, ...]}、{"OR": [...]}、{"NOT": [...]}（NOT：一个都不满足）。

    条件里的 user_id、agent_id、run_id 指记忆所在的空间与会话，其余的键指 metadata。"""
    for key, cond in where.items():
        if key in _LOGIC:
            if isinstance(cond, str | bytes | Mapping) or not isinstance(cond, Sequence):
                raise ValueError(f"filters[{key!r}] must be a list of conditions.")
            for item in cond:
                if not isinstance(item, Mapping):
                    raise ValueError(f"filters[{key!r}] must be a list of conditions.")
                _check_where(item)
        elif isinstance(cond, Mapping):
            unknown = sorted(set(cond) - set(_OPERATORS))
            if unknown or not cond:
                raise ValueError(
                    f"Unsupported filter operators for {key!r}: {unknown}. "
                    f"Use {', '.join(_OPERATORS)}."
                )
            for op in ("in", "nin"):
                if op in cond and (
                    isinstance(cond[op], str | bytes | Mapping)
                    or not isinstance(cond[op], Sequence)
                ):
                    raise ValueError(f"filters[{key!r}][{op!r}] must be a list.")


def _matches(where: Mapping[str, Any], record: Mapping[str, Any]) -> bool:
    return all(_match(key, cond, record) for key, cond in where.items())


def _match(key: str, cond: Any, record: Mapping[str, Any]) -> bool:
    if key == "AND":
        return all(_matches(c, record) for c in cond)
    if key == "OR":
        return any(_matches(c, record) for c in cond)
    if key == "NOT":
        return not any(_matches(c, record) for c in cond)
    present, value = key in record, record.get(key)
    if not isinstance(cond, Mapping):
        return present and (cond == "*" or value == cond)
    return all(_compare(op, present, value, arg) for op, arg in cond.items())


def _compare(op: str, present: bool, value: Any, arg: Any) -> bool:
    if op == "ne":
        return not present or value != arg
    if op == "nin":
        return not present or value not in arg
    if not present:
        return False
    if op == "eq":
        return bool(value == arg)
    if op == "in":
        return value in arg
    if op in ("contains", "icontains"):
        fold = (lambda s: s.casefold()) if op == "icontains" else (lambda s: s)
        if isinstance(value, str) and isinstance(arg, str):
            return fold(arg) in fold(value)
        if isinstance(value, list):  # 列表里有这一项
            return any(
                fold(v) == fold(arg) if isinstance(v, str) and isinstance(arg, str) else v == arg
                for v in value
            )
        return False
    try:
        if op == "gt":
            return bool(value > arg)
        if op == "gte":
            return bool(value >= arg)
        if op == "lt":
            return bool(value < arg)
        return bool(value <= arg)
    except TypeError:  # 类型不能比较（数字与字符串）：不算满足
        return False


def _scope_record(user_id: str | None, agent_id: str | None) -> dict[str, str]:
    """条件里也可以写 user_id、agent_id（Mem0 的 AND 里常这样写）：每条记忆都带着它们。"""
    return {k: v for k, v in (("user_id", user_id), ("agent_id", agent_id)) if v is not None}


@dataclass
class _Sources:
    """一批记忆出处各句的附加信息（turn_meta）。事实与对话片段的 metadata 是出处各句的并集
    （同一个键取后面那句的）；过期时间取出处里最晚的，有一句没给过期时间就不过期：同一件事
    后来又说过一次、没说什么时候过期，它就一直有效。"""

    meta: Mapping[str, TurnMeta]

    @classmethod
    def load(cls, system: MemorySystem, space: str, turn_ids: Iterable[str]) -> _Sources:
        ids = list(dict.fromkeys(turn_ids))
        return cls(system.db.get_turn_meta(space, ids) if ids else {})

    def metadata(self, turn_ids: Sequence[str]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for turn_id in turn_ids:
            info = self.meta.get(turn_id)
            if info is not None:
                merged.update(info.metadata)
        return merged

    def expires(self, turn_ids: Sequence[str]) -> str | None:
        stamps: list[str] = []
        for turn_id in turn_ids:
            info = self.meta.get(turn_id)
            if info is None or info.expires_at is None:
                return None
            stamps.append(info.expires_at)
        return max(stamps, key=expiry_instant) if stamps else None

    def expired(self, turn_ids: Sequence[str], now: datetime) -> bool:
        stamp = self.expires(turn_ids)
        return stamp is not None and expiry_instant(stamp) <= now


@dataclass
class _Select:
    """search / get_all 的事后筛选：会话（run_id）、metadata 的条件与过期时间。"""

    run_id: str | None
    where: Mapping[str, Any] | None
    scope: Mapping[str, str]
    now: datetime

    def keep(self, session_id: str | None, turn_ids: Sequence[str], sources: _Sources) -> bool:
        if self.run_id is not None and session_id != self.run_id:
            return False
        if sources.expired(turn_ids, self.now):
            return False
        if not self.where:
            return True
        record = {**sources.metadata(turn_ids), **self.scope}
        if session_id is not None:
            record["run_id"] = session_id
        return _matches(self.where, record)


def _profile(system: MemorySystem, space: str, core: CoreMemory | None, now: datetime) -> str:
    """画像去掉已经过期的事实。画像是事实表的物化视图（write/profile.py）：每条事实一行
    "- 陈述"，多人时前面有一行人名；某人名下的事实都过期了，人名那一行也去掉。"""
    if core is None:
        return ""
    facts = system.db.get_facts(space, ids=core.fact_ids)
    sources = _Sources.load(system, space, [t for f in facts for t in f.source_turn_ids])
    gone = {f"- {f.statement}" for f in facts if sources.expired(f.source_turn_ids, now)}
    if not gone:
        return core.text
    lines = [line for line in core.text.split("\n") if line not in gone]
    return "\n".join(
        line
        for i, line in enumerate(lines)
        if line.startswith("- ") or (i + 1 < len(lines) and lines[i + 1].startswith("- "))
    )


def _as_datetime(value: datetime | str | float | None) -> datetime | None:
    """timestamp 参数 → 本机时间（不带时区）。带时区的（'...Z'）先换算：生成的会话 ID 里的
    时间与存下来的开始时间一致。数字是 Unix 时间戳。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return naive_local(value)
    if isinstance(value, str):
        return naive_local(datetime.fromisoformat(value))
    return datetime.fromtimestamp(value)


def _idx_key(space: str, session_id: str) -> str:
    return f"next_idx:{space}:{session_id}"


# ---------- 输出 ----------


def _fact_item(
    fact: Fact,
    score: float | None,
    user_id: str | None,
    agent_id: str | None,
    sources: _Sources,
    *,
    space: str | None = None,
) -> dict[str, Any]:
    from cuemem.read.facts import fact_status

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
        "metadata": sources.metadata(fact.source_turn_ids),
        "expiration_date": sources.expires(fact.source_turn_ids),
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


def _excerpt_item(hit: Hit, sources: _Sources) -> dict[str, Any]:
    return {
        "id": hit.unit.id,
        "memory": hit.unit.text,
        "type": "excerpt",
        "score": round(hit.score, 4),
        "created_at": hit.unit.ts.isoformat(),
        "session_id": hit.unit.session_id,
        "source_turn_ids": hit.unit.source_turn_ids,
        "metadata": sources.metadata(hit.unit.source_turn_ids),
        "expiration_date": sources.expires(hit.unit.source_turn_ids),
    }


def _image_item(item: MediaItem, attached: bool, sources: _Sources) -> dict[str, Any]:
    return {
        "id": item.id,
        "memory": item.caption,
        "type": "image",
        "path": item.path,
        "created_at": item.ts.isoformat(),
        "attached": attached,
        "metadata": sources.metadata([item.source_turn_id]),
        "expiration_date": sources.expires([item.source_turn_id]),
    }


def _recall_out(
    system: MemorySystem,
    space: str,
    recall: Recall,
    user_id: str | None,
    agent_id: str | None,
    limit: int | None,
    select: _Select,
) -> dict[str, Any]:
    """检索结果 → SDK 的返回值。检索本身不分会话、不看 metadata 与过期时间，在这里事后筛。"""
    from cuemem.read.controller import english_reason

    view = recall.view
    media = list(view.media.values())
    sources = _Sources.load(
        system,
        space,
        [
            *(t for fh in view.facts for t in fh.fact.source_turn_ids),
            *(t for h in view.rounds for t in h.unit.source_turn_ids),
            *(m.source_turn_id for m in media),
        ],
    )
    facts: list[FactHit] = [
        fh for fh in view.facts if select.keep(fh.fact.session_id, fh.fact.source_turn_ids, sources)
    ]
    rounds = [
        h for h in view.rounds if select.keep(h.unit.session_id, h.unit.source_turn_ids, sources)
    ]
    # 图片来自装入的轮（l1.collect_media）：会话按它所在的那一轮算
    session_of = {
        m: h.unit.session_id for h in view.rounds for m in h.unit.meta.get("media_ids", [])
    }
    media = [m for m in media if select.keep(session_of.get(m.id), [m.source_turn_id], sources)]
    items = [_fact_item(fh.fact, fh.score, user_id, agent_id, sources) for fh in facts]
    items += [_excerpt_item(h, sources) for h in sorted(rounds, key=lambda h: -h.score)]
    decision = recall.decision
    return {
        "results": items[:limit] if limit else items,
        "images": [_image_item(m, m.id in view.attach, sources) for m in media],
        "profile": _profile(system, space, view.core, select.now),
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
    from cuemem.read.controller import english_reason

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


def _not_extracted(exc: Exception) -> str:
    """发言已经写进原始日志（巩固任务也登记了），同步抽取却出错（例如对账时余额不足、向量接口
    暂时不可用）时回给调用方的说明。不抛出：抛出的话调用方以为没写进去，重试就写了两遍。没做完的
    任务留着，下次巩固（consolidate()，或后台线程下一次处理这个空间）时补上。"""
    logger.warning("Extraction failed after the messages were stored: %s", exc, exc_info=True)
    return (
        f"Extraction failed ({type(exc).__name__}: {str(exc)[:300]}). The messages are stored; "
        "their extraction is retried on the next consolidation. Do not add them again."
    )


def _events(
    system: MemorySystem,
    space: str,
    before: Mapping[str, tuple[bool, str | None]],
    user_id: str | None,
    agent_id: str | None,
) -> list[dict[str, Any]]:
    """抽取前后的事实对比 → Mem0 风格的事件：新事实 ADD；取代了旧事实的新事实 UPDATE，
    附上被取代的旧版本（旧版本仍留在版本链里，不会被删除）。before 是抽取前的
    MemoryDB.fact_links：只读出新的事实与这次被取代的事实（顺序同 get_facts）。"""
    db = system.db
    wanted = [
        fid
        for fid, (_, superseded_by) in db.fact_links(space).items()
        if fid not in before or (before[fid][0] and superseded_by)
    ]
    after = [
        f for i in range(0, len(wanted), 500) for f in db.get_facts(space, ids=wanted[i : i + 500])
    ]
    fresh = [t for f in after if f.id not in before for t in f.source_turn_ids]
    sources = _Sources.load(system, space, fresh)
    replaced: dict[str, list[Fact]] = {}
    for fact in after:
        old = before.get(fact.id)
        if old is not None and old[0] and fact.superseded_by:
            replaced.setdefault(fact.superseded_by, []).append(fact)
    events = []
    for fact in after:
        if fact.id in before:
            continue
        item = _fact_item(fact, None, user_id, agent_id, sources)
        previous = replaced.get(fact.id, [])
        item["event"] = "UPDATE" if previous else "ADD"
        if previous:
            item["previous_memory"] = [p.statement for p in previous]
            item["reason"] = previous[0].supersede_reason
        events.append(item)
    return events


def _forget_out(report: ForgetReport) -> dict[str, Any]:
    out: dict[str, Any] = {
        "turns_deleted": report.turns_deleted,
        "sessions_affected": report.sessions_affected,
        "facts_before": report.facts_before,
        "facts_after": report.facts_after,
        "images_deleted": report.images_deleted,
        "restated": report.restated,
    }
    if report.rebuild_error:  # 删除已经做了（见 MemorySystem.forget 的 report_rebuild_errors）
        out["warning"] = (
            f"The memories were deleted, but rebuilding what is derived from the rest failed "
            f"({report.rebuild_error}); it finishes on the next consolidation, and some facts "
            "may be missing until then."
        )
    return out
