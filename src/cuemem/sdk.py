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
  过期之后 search、get_all 与画像里不再出现（show_expired=True 时 search、get_all 照样列出，
  get 与 history 照样取得到，数据要 delete 才删）；
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
import hashlib
import json
import logging
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Annotated, Any, Literal

import httpx2
from pydantic import Field, TypeAdapter, ValidationError

from cuemem._async import AsyncMethods
from cuemem.config import Settings, load_settings, presets, unknown_keys
from cuemem.core.ids import sortable_session_id, stable_id
from cuemem.core.models import (
    Answer,
    CoreMemory,
    Fact,
    FactHit,
    MediaItem,
    MemoryUnit,
    SessionIn,
    TurnIn,
    TurnMeta,
)
from cuemem.core.timeutil import naive_local, normalize_expiry
from cuemem.core.types import PeakPolicy, Router
from cuemem.errors import ConcurrentWriteError, IdempotencyConflictError, SessionLimitError
from cuemem.expiry import Sources as _Sources
from cuemem.expiry import visible_core
from cuemem.store.sqlite import legacy_idx_key, next_idx_key

# 记忆系统（向量库、模型客户端）在 Memory() 时才导入：import cuemem.sdk 只要零点几秒
if TYPE_CHECKING:
    from cuemem.memory import ForgetReport, MemorySystem, Recall
    from cuemem.store.sqlite import MemoryKind

logger = logging.getLogger(__name__)

IDLE_CLOSED_KEPT = 10_000  # 记着多少个闲置结束的 run（见 close_idle_runs），之后再 add 时重新打开
SearchMode = Literal["auto", "l0", "l1", "l2"]
AnswerMode = Literal["auto", "l0", "l1", "l2", "agent", "hatch"]
MinLevel = Literal["l0", "l1"]  # search 的 min_level（签名里照写 Literal：客户端不导入这里）
# 控制器的 λ（按需付费的权重）：不小于 0 的有限数。/api/v1 的请求模型用同一个类型
Lam = Annotated[float | None, Field(ge=0, allow_inf_nan=False)]
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
    # 别的类型（user_id=42）以前在 _part 里 AttributeError；/api/v1 的请求校验报 ValueError
    for name, value in (("user_id", user_id), ("agent_id", agent_id)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
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
    {"roles": {"reader": {"model": "qwen3.8-flash"}}}；有不认识的键（写错的、Mem0 格式的配置）时
    报 ValueError，不悄悄用默认的模型供应商（见 _check_config）；home：数据目录（默认 CUEMEM_HOME，
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
            _check_config(overrides)
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
        self._init_runs()
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
        memory._init_runs()
        return memory

    def _init_runs(self) -> None:
        self._runs: set[tuple[str, str]] = set()  # open_run 开始、还没结束的（空间，run_id）
        self._run_seen: dict[tuple[str, str], float] = {}  # 这些 run 最后一次活动的时刻
        # close_idle_runs 结束的 run（按结束的先后，最多 IDLE_CLOSED_KEPT 个）：再 add 时重新打开
        self._idle_closed: dict[tuple[str, str], None] = {}
        self._adopted: set[str] = set()  # 接手过崩溃前留下的 run 的空间（见 _adopt）
        self._opening: dict[str, int] = {}  # 空间 → 正在 open_run（还没登记）的 run 数
        self._runs_lock = threading.Lock()
        # 一个空间同时开着的 run 的上限（open_run），None 是不限。cuemem serve 给 /api/v1 设成
        # CUEMEM_MAX_OPEN_SESSIONS（默认 64）；本地的 SDK 不限
        self.max_open_runs: int | None = None

    def _adopt(self, space: str, keep: Sequence[str] = ()) -> None:
        """这个对象第一次碰到这个空间：之前的进程崩溃时还开着的 run（open_run 了、没来得及
        end_run 或 close()），以及 Agent 与记忆代理的会话，没有抽取任务、也没人再结束它们，
        交给抽取（MemorySystem.adopt_open；没有后台线程时只登记，下次 consolidate 时完成）。
        keep：调用方马上要接着写的 run（open_run 重新打开它；add 写的 run 已经由 _resume 重新算作
        进行中，不用列出）。"""
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
        idempotency_key: str | None = None,
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
        调用方重试就写了两遍。这个会话的抽取失败（不抛出，记在任务上）时同样（以前 status 照样是
        "consolidated"、没有 error）。写进去之前出错（向量接口不可用等）时照常抛出，什么都没写，
        可以原样重试；写进去之后 trace 写不进去时只记一条警告（见 core/trace.py 的 start_trace）。
        后台抽取的状态与出错见 status()。
        idempotency_key：调用方给的键（1 到 255 个可打印的 ASCII 字符，例如 uuid4），重试时原样
        再给（第六轮体检，默认不用）。同一个用户下 24 小时内这个键已经写过同样的请求时不再写，
        返回 status "duplicate"、那次的 session_id 与 turn_ids、replayed True（results 是空的；
        wait=True 时先等这个空间的抽取做完，这个会话的抽取失败时加上 error，那次的会话是还开着的
        run 时不等）；那次写下的发言已经遗忘时 status 是 "forgotten"；同一个键用于不同的
        请求（消息或别的参数不同，wait 不算）时抛 IdempotencyConflictError，什么都不写。并发的
        同一个键只有一个写进去，别的回滚、按 duplicate 返回。
        """
        space = space_for(user_id, agent_id)
        keyed: tuple[str, str] | None = None
        if idempotency_key is not None:  # 不给键时与以前完全相同，不多一次查询
            keyed = (
                _key_hash(idempotency_key),
                _request_hash(
                    messages, user_id, agent_id, run_id, metadata, expiration_date, timestamp, infer
                ),
            )
            replayed = self._replayed(space, *keyed)
            if replayed is not None:
                return self._settle_replay(space, replayed, wait=wait and infer)
        if run_id is not None:
            self._resume(space, run_id)
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
                # 按主键只查这个会话：以前读出整个空间的会话（2 万个会话时约 20 毫秒）
                started = self.system.db.session_started(space, sid)
                if started is None:
                    started = when
                session = SessionIn(session_id=sid, started_at=started, turns=turns)
                extra = None
                if info is not None:
                    extra = {(sid, first + i): info for i in range(len(turns))}
                # 下一句的序号与发言在同一个事务里提交：以前另起一个事务写，出错时发言已经写进去，
                # 调用方原样重试就写了两遍
                try:
                    with self.system.appending(
                        space, [(session, first)], enqueue=infer, meta=extra
                    ) as written:
                        db = self.system.db
                        db.set_meta(next_idx_key(space, sid), str(first + len(turns)))
                        if keyed is not None:  # 与发言在同一个事务里：一起提交，一起回滚
                            with db.transaction() as conn:
                                db.add_key_insert(
                                    conn,
                                    space,
                                    *keyed,
                                    sid,
                                    [stable_id(space, sid, first + i) for i in range(len(turns))],
                                )
                        # 撞号的发言被 INSERT OR IGNORE 悄悄跳过了：在事务里抛出，发言、键与暂存的
                        # 向量一起撤销。以前提交之后才查：键已经记下，原样重试返回 duplicate 与
                        # 别人的发言，这次的发言丢了（第七轮体检）
                        if written.turns_added != len(turns):
                            raise ConcurrentWriteError(
                                f"Only {written.turns_added} of {len(turns)} messages could be "
                                f"written to session {sid!r}: their turn numbers were already "
                                "taken by another write. Nothing was written; retry the add."
                            )
                except sqlite3.IntegrityError:
                    # 并发的同一个键先提交了：这次的发言与暂存的向量已经随事务回滚
                    replayed = self._replayed(space, *keyed) if keyed is not None else None
                    if replayed is None:
                        raise
                    return self._settle_replay(space, replayed, wait=wait and infer)
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
                if sid in report.failed:  # 这个会话的抽取失败了（不抛出）：与上面一样说明
                    out["status"] = "queued"
                    out["error"] = _not_extracted(_job_error(self.system, space, sid))
        else:
            out["status"] = "queued"
        return out

    def _replayed(self, space: str, key_hash: str, request_hash: str) -> dict[str, Any] | None:
        """这个键已经写过时 add 的返回值（见 add 的 idempotency_key）；没写过时 None。"""
        found = self.system.db.add_key_lookup(space, key_hash)
        if found is None:
            return None
        if found["request_hash"] is None:  # 墓碑：那次写下的发言已经遗忘
            return {
                "results": [],
                "session_id": None,
                "turn_ids": [],
                "status": "forgotten",
                "replayed": True,
            }
        if found["request_hash"] != request_hash:
            raise IdempotencyConflictError(
                "This idempotency_key was already used for a different add request for this "
                "user; repeat the original request unchanged, or use a new key."
            )
        return {
            "results": [],
            "session_id": found["session_id"],
            "turn_ids": found["turn_ids"],
            "status": "duplicate",
            "replayed": True,
        }

    def _settle_replay(self, space: str, out: dict[str, Any], *, wait: bool) -> dict[str, Any]:
        """wait=True 的重试按 duplicate 返回之前，先等这个空间的抽取做完（第七轮体检）：
        MemoryClient 超时之后让调用方用同一个键重试，最常见的是 add(wait=True) 发言已经写进去、
        抽取还没做完；以前重试立即返回，紧接着的 search 找不到这次的事实。consolidate 持有这个
        空间的巩固锁：原来那次请求还在抽取时等它做完，之后没有剩下的事。那次的会话是还开着的
        run 时不抽取（结束时才抽取），照旧返回。这个会话的抽取失败时加上 error，status 仍是
        duplicate、results 仍是空的。"""
        if not wait or out["status"] != "duplicate":
            return out
        sid = out["session_id"]
        if sid in self.system.open_sessions(space):
            return out
        try:
            report = self.system.consolidate(space)
        except Exception as exc:  # 与 add(wait=True) 相同：发言写进去了，不抛出
            out["error"] = _not_extracted(exc)
        else:
            if sid in report.failed:
                out["error"] = _not_extracted(_job_error(self.system, space, sid))
        return out

    def open_run(
        self, run_id: str | None = None, *, user_id: str | None = None, agent_id: str | None = None
    ) -> str:
        """开始一段对话（run）：之后 add(run_id=...) 照常在线写入、立即可检索，但不抽取事实，
        end_run 时整段只抽取一次。返回 run_id（不给时生成一个）。

        进程崩溃时还开着的 run 由下一个进程第一次碰到这个用户时交给抽取（见 _adopt）；用同一个
        run_id 再 open_run 或 add 就是接着这段对话写，它不被接手（见 _resume）。cuemem serve 停止时
        /api/v1 的 run 留着，重启之后接着写；没人接着写的，启动一个闲置期之后交给抽取，在那之前
        别的请求第一次碰到这个用户时也不接手（MemorySystem.defer_adoption）。/api/v1 上闲置（默认
        30 分钟没有 add）的 run 先交给抽取，之后再 add 时重新打开（close_idle_runs）；本进程接手过
        的 run 同样（MemorySystem.resume_left_open）。"""
        space = space_for(user_id, agent_id)
        rid = run_id or sortable_session_id("run", datetime.now())
        with self._runs_lock:
            self._make_room(space, rid)
            self._opening[space] = self._opening.get(space, 0) + 1
        try:
            self._adopt(space, keep=[rid])
            self.system.open_session(space, rid)
            with self._runs_lock:
                self._runs.add((space, rid))
                self._run_seen[(space, rid)] = time.monotonic()
                self._idle_closed.pop((space, rid), None)
        finally:
            with self._runs_lock:
                self._opening[space] -= 1
                if not self._opening[space]:
                    del self._opening[space]
        return rid

    def _make_room(self, space: str, run_id: str) -> None:
        """open_run 的上限（调用方持有 _runs_lock）：max_open_runs 给了、这个空间开着的（加上正在
        开的）run 已经这么多、run_id 又不是其中之一时 SessionLimitError（/api/v1 回 429）。不替
        调用方结束别的 run：那会让它们提前抽取。以前 /api/v1 不限，一个令牌可以一直 open_run，
        每个都占着内存与一个进行中的标记，直到闲置 30 分钟才结束。"""
        limit = self.max_open_runs
        if limit is None or (space, run_id) in self._runs:
            return
        count = sum(1 for s, _ in self._runs if s == space) + self._opening.get(space, 0)
        if count >= limit:
            raise SessionLimitError(
                f"User space {space} already has {count} open runs (the limit is {limit}, set by "
                "CUEMEM_MAX_OPEN_SESSIONS): end one with end_run, or retry after idle runs "
                "have been ended."
            )

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
            self._run_seen.pop((space, run_id), None)
            self._idle_closed.pop((space, run_id), None)
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

    def close_idle_runs(self, idle_s: float, *, now: float | None = None) -> int:
        """open_run 开始、之后 idle_s 秒没有 add 的 run 交给抽取（与 end_run(wait=False) 相同），
        返回结束的个数。同一个 run_id 之后再 add 时重新打开（_resume）：调用方没有 end_run，还在
        接着这段对话写，结束时整段只抽取一次。cuemem serve 每分钟为 /api/v1 调用一次：远程的调用方
        崩溃、没有 end_run 时，以前这段对话要等服务重启才交给抽取。本地的 SDK 不调用就不会结束。
        now：time.monotonic() 的时刻（测试用）。"""
        clock = time.monotonic() if now is None else now
        with self._runs_lock:
            stale = [k for k in self._runs if clock - self._run_seen.get(k, clock) >= idle_s]
        closed = 0
        for space, rid in stale:
            # 在会话锁里：与 add 写这个 run（_resume、写入）一个一个来，正要写的不会被结束
            with self.system.session_lock(space, rid):
                with self._runs_lock:
                    seen = self._run_seen.get((space, rid), clock)
                    if (space, rid) not in self._runs or clock - seen < idle_s:
                        continue
                    self._runs.discard((space, rid))
                    self._run_seen.pop((space, rid), None)
                    self._idle_closed[(space, rid)] = None
                    while len(self._idle_closed) > IDLE_CLOSED_KEPT:  # 最早结束的不再记着
                        del self._idle_closed[next(iter(self._idle_closed))]
                turns = self.system.db.get_turns(space, rid)
                submit = bool(turns) and self.system.worker is not None
                self.system.close_session(space, rid, submit=submit)
            closed += 1
        return closed

    def _resume(self, space: str, run_id: str) -> None:
        """add 写进 run 之前：还开着的记下这次活动（close_idle_runs 据此判断闲置）；闲置太久、被
        close_idle_runs 结束了的重新打开（open_session），结束时整段只抽取一次。不重新打开的话，
        之后每次 add 都把整段重新抽取、从原始日志重放一遍。

        之前的进程留下、还开着的 run（cuemem serve 停止时不结束 /api/v1 的 run，崩溃时也来不及
        结束）同样接着写（MemorySystem.resume_left_open），与用同一个 run_id 再 open_run 相同。
        以前 add 里的 _adopt 把它交给抽取：跨过一次部署的 run 每写一句都要再抽取、重放一遍。本进程
        已经接手（交给抽取）了的也重新打开，与闲置结束的一样。没记着的（之前的进程闲置结束的，或者
        接手的记录超过了 HANDED_OVER_KEPT 个）照旧当作普通的会话，每次 add 都抽取一遍。"""
        key = (space, run_id)
        with self._runs_lock:
            if key in self._runs:
                self._run_seen[key] = time.monotonic()
                return
            idle = key in self._idle_closed
        # 锁的顺序与 close_idle_runs 相同：先会话锁，再 _runs_lock（resume_left_open 拿 _open_lock）
        with self.system.session_lock(space, run_id):
            if not idle:
                if self.system.resume_left_open(space, run_id):
                    with self._runs_lock:
                        self._runs.add(key)
                        self._run_seen[key] = time.monotonic()
                return
            with self._runs_lock:
                if key not in self._idle_closed:  # 期间别的 add 已经重新打开了
                    if key in self._runs:
                        self._run_seen[key] = time.monotonic()
                    return
                del self._idle_closed[key]
            self.system.open_session(space, run_id)
            with self._runs_lock:
                self._runs.add(key)
                self._run_seen[key] = time.monotonic()

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        data: str | None = None,
    ) -> dict[str, Any]:
        """改一条记忆（Mem0 的 update）。事实不就地改写（派生记忆是原始日志的函数，D-021、
        D-061）：像用户那样说一句更正——"Correction: <text>" 记进一段新的会话（edit-…），原始层
        记下它要取代的事实（edit_of）——再同步巩固：由这句抽出的新事实以"说错了"（corrected）
        取代原来那条，不问 LLM；history() 里看得到前后两个版本，遗忘这句更正时原来那条恢复。

        text 与 data 是同一个参数（Mem0 2.x 叫 text，data 是它旧的名字）：给一个就行，两个都给且
        不同时 ValueError。只给 metadata 时 ValueError：metadata 记在原始层的发言上，要随一句
        更正写进去。新事实带上原来那条的 metadata（metadata 参数里的键覆盖它）与过期时间。返回
        Mem0 风格的事件（UPDATE 附上被取代的旧版本）；从 text 里没有抽出事实时，原来那条不变，
        message 写明，这句更正也从日志里遗忘（turn_ids 是空的）。
        更正记下之后抽取出错时不抛出（与 add 的 wait=True 相同）：message 与 error 写明，下次
        巩固时补上。
        只能改当前有效的版本（ValueError）；对话片段与图片的 ID 不能改（ValueError，要改就
        delete）；ID 不存在时 KeyError。"""
        data = _correction(text, data)
        found = self._find(memory_id)
        if found is None:
            located = self._locate(memory_id)
            if located is not None:
                raise ValueError(
                    f"Only facts can be updated: {memory_id} is an {located[1]}. "
                    "Delete it instead (delete() forgets the conversation turns it came from)."
                )
            raise KeyError(f"No memory with id {memory_id}")
        space, fact = found
        if not fact.is_current:
            raise ValueError(
                f"Memory {memory_id} was already replaced by {fact.superseded_by}: "
                "update the current version (see history())."
            )
        if not data.strip():
            raise ValueError("text is empty: to remove a memory, call delete().")
        sources = _Sources.load(self.system.db, space, fact.source_turn_ids)
        inherited = {**sources.metadata(fact.source_turn_ids), **dict(metadata or {})}
        info = _turn_meta(inherited, sources.expires(fact.source_turn_ids)) or TurnMeta()
        info = info.model_copy(update={"edit_of": memory_id})
        self._adopt(space)
        # 巩固按会话开始的时间排序：更正排在原来那条之后（调用方写入时可能给了将来的时间）
        when = max(datetime.now(), fact.recorded_at + timedelta(microseconds=1))
        sid = sortable_session_id("edit", when)
        turn = TurnIn(role="user", text=f"Correction: {data.strip()}", ts=when)
        before = self.system.db.fact_links(space)
        session = SessionIn(session_id=sid, started_at=when, turns=[turn])
        with (
            self.system.session_lock(space, sid),
            self.system.appending(space, [(session, 0)], meta={(sid, 0): info}),
        ):
            self.system.db.set_meta(next_idx_key(space, sid), "1")  # 与更正在同一个事务里（见 add）
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
            extra["error"] = _not_extracted(_job_error(self.system, space, sid))
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

    def wait(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """等后台线程把已登记的抽取做完（测试、脚本退出前用）。抽取失败时不抛出（与 Mem0
        相同），后台线程记一条警告日志；失败的会话与原因见 status()。

        给了 user_id 或 agent_id 时只等这个用户（或 Agent）的抽取，不给时等全部。timeout 秒内
        没做完时抛 TimeoutError（抽取在后台照常继续）；不给时一直等。timeout 是负数时
        ValueError（与 /api/v1 相同，没有后台线程时也报）。"""
        if timeout is not None and timeout < 0:
            raise ValueError("timeout: must be >= 0")
        worker = self.system.worker
        if worker is None:
            return
        spaces = None
        if user_id is not None or agent_id is not None:
            space = space_for(user_id, agent_id)
            spaces = space.__eq__
        if not worker.wait(spaces, timeout):
            raise TimeoutError(f"Extraction did not finish within {timeout}s")

    def status(self, *, user_id: str | None = None, agent_id: str | None = None) -> dict[str, Any]:
        """这个用户（或 Agent）的记忆的抽取状态：pending 是等着抽取的会话数；failed 是抽取失败的
        会话，每个带 session_id、attempts（失败了几次）、error（出错的第一行）与 retry_at（后台
        线程什么时候自动重试，ISO 时间；不再自动重试——失败次数到了上限、没有后台线程——时为
        None，要显式地 consolidate()）；background 是有没有后台线程；errors 是后台线程在这个空间
        最近的几条出错。以前抽取失败（没配好密钥、余额不足）时 add() 照常返回、wait() 正常结束，
        get_all() 一直是空的，SDK 里看不到原因。"""
        from cuemem.write.consolidate import MAX_ATTEMPTS, RETRY_BASE_S
        from cuemem.write.extract import PIPELINE_VERSION

        space = space_for(user_id, agent_id)
        # 只数待处理的、只读失败的（以前读出并解开每个任务，1 万个任务时 40 毫秒）。按会话排：
        # 与以前读全部任务时的顺序相同（那条查询走 idx_jobs_session），同一会话的按写入的先后
        db = self.system.db
        jobs = sorted(db.get_jobs(space, status="failed"), key=lambda j: j["session_id"])
        worker = self.system.worker
        base_s = worker.retry_base_s if worker is not None else RETRY_BASE_S
        failed = []
        for job in jobs:
            retry_at = None
            due = job["attempts"] < MAX_ATTEMPTS and job["version"] == PIPELINE_VERSION
            if due and worker is not None:  # 与后台线程的退避相同（Consolidator.retry_due）
                wait = timedelta(seconds=base_s * 4 ** max(job["attempts"] - 1, 0))
                retry_at = (datetime.fromisoformat(job["updated_at"]) + wait).isoformat()
            lines = (job["error"] or "").splitlines()
            failed.append(
                {
                    "session_id": job["session_id"],
                    "attempts": job["attempts"],
                    "error": lines[0][:300] if lines else "",
                    "retry_at": retry_at,
                }
            )
        return {
            "pending": db.job_counts(space).get("pending", 0),
            "failed": failed,
            "background": worker is not None,
            # 只给这个空间的：worker.errors 里有别的用户的空间 ID 与出错
            "errors": worker.errors_for(space)[-3:] if worker is not None else [],
        }

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
        at: datetime | date | str | None = None,
        router: Router = "learned",
        lam: float | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        min_level: Literal["l0", "l1"] = "l1",
        show_expired: bool = False,
    ) -> dict[str, Any]:
        """检索（不回答）。mode="auto" 由检索控制器选档，结果里的 level 与 reasons 说明它为什么
        这么选；limit 截断结果条数（正整数，否则 ValueError）。调用 search 就是已经决定要查记忆：
        与 MCP、各框架集成的 recall 相同，控制器选了 L0（这句话不需要回忆）时改走 L1，reasons
        写明，控制器只决定要不要查得更深。min_level="l0" 时照控制器的决定，L0 只返回画像；
        mode="l0" 直接只要画像。

        filters={"user_id": …, "agent_id": …, "run_id": …} 与 top_k 是 Mem0 v3 的写法，与
        user_id、limit 等价。run_id：只要这个会话里的事实、对话片段与图片。filters 里的其他键是
        metadata 的条件（见 _check_where）；Mem0 保留的 created_at、updated_at 与 categories 在
        metadata 没有这个键时对记忆自己的字段（见 _Select）。检索照常在整个空间里做（控制器与
        L1 / L2 不变），再按会话、条件与过期时间筛，所以结果可能少于 limit；show_expired=True 时
        过期的也给（Mem0 的写法）。画像是整个空间的（总是去掉过期的事实）。

        score 只用来排序（L1 是排名融合的分数，第一名约 0.016），不是相似度：没有 Mem0 的
        threshold（按 Mem0 默认的 0.1 截断会什么都不剩）；Mem0 的 rerank 对应 mode="l2"。
        """
        # 先于一切检查：空间还是空的时，写错的 mode 也要报错，而不是返回空结果
        _valid("mode", SearchMode, mode)
        _valid("router", Router, router)
        _valid("min_level", MinLevel, min_level)
        lam = _valid("lam", Lam, lam)
        _count("limit", limit)
        _count("top_k", top_k)
        user_id, agent_id, run_id, where = _scope(user_id, agent_id, run_id, filters)
        limit = limit if limit is not None else top_k
        space = space_for(user_id, agent_id)
        moment = _moment("at", at)  # 空间还是空的时，写错的 at 也报错
        if not self.system.db.has_sessions(space):
            return _empty(self.system, space)
        from cuemem.evidence import RAISED_REASON

        # 两次检索用同一个问题时间：改走 L1 时打包的是探针在这个时间打过分的候选
        when = moment if moment is not None else datetime.now()
        # 从检索到读出各句的 metadata 与画像一直持有读：改走 L1 时打包的是探针取到的记忆，两次检索
        # 之间插进来的遗忘会让结果里留着已经删掉的事实与对话片段（见 MemorySystem.retrieve）
        with self.system.reading(space):
            recall = self.system.retrieve(
                space, query, question_time=when, mode=mode, router=router, lam=lam
            )
            raised = mode == "auto" and min_level == "l1" and recall.level == "l0"
            if raised:  # 与 evidence.recall_listing 的 at_least_l1 相同：不把探针的检索再做一遍
                recall = self.system.retrieve(
                    space,
                    query,
                    question_time=when,
                    mode="l1",
                    candidates=recall.candidates,
                    expired=recall.expired,
                )
            select = _Select(
                run_id, where, _scope_record(user_id, agent_id), datetime.now(), show_expired
            )
            out = _recall_out(self.system, space, recall, user_id, agent_id, limit, select)
        if raised:
            out["reasons"] = [RAISED_REASON]
        return out

    def context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        at: datetime | date | str | None = None,
        mode: SearchMode = "auto",
        router: Router = "learned",
        lam: float | None = None,
        min_level: Literal["l0", "l1"] = "l1",
        wrap: bool = True,
    ) -> dict[str, Any]:
        """放进提示词的记忆（第六轮体检 P17）：记忆代理注入的同一段证据块（evidence.py 的
        evidence_from），不调用模型。带日期与有效状态的事实、按时间排列的对话片段、画像，记忆里的
        <memory、</memory 已经转义，开头的 MEMORY_NOTE 声明这些是资料、不是指令。自己拿 search
        的结果拼提示词时这些都没有。

        返回 {context, level, reasons, recalled, see, refs}。wrap=True 时 context 包在
        "<memory>\\n…\\n</memory>" 里（记忆代理放在最后一句用户消息开头的那段，去掉末尾的空行），
        wrap=False 时只有块本身；没有要给的内容时是空字符串。recalled 是条数（画像算一条），
        refs 是块里的编号 → 记忆 ID（Memory.get 认得）。

        与 search 相同，min_level="l1"（缺省）时控制器选了 L0 也改走 L1，reasons 写明；
        min_level="l0" 时照控制器的决定，与记忆代理逐字相同。过期的记忆不给，画像里也去掉。
        mode、router、min_level、lam 写错时 ValueError（与 search 同一句话）。"""
        _valid("mode", SearchMode, mode)
        _valid("router", Router, router)
        _valid("min_level", MinLevel, min_level)
        lam = _valid("lam", Lam, lam)
        space = space_for(user_id, agent_id)
        moment = _moment("at", at)
        if not self.system.db.has_sessions(space):
            return {
                "context": "",
                "level": "L0",
                "reasons": ["no memories yet"],
                "recalled": 0,
                "see": False,
                "refs": {},
            }
        from cuemem.evidence import context_evidence

        evidence = context_evidence(
            self.system,
            space,
            query,
            at=moment,
            mode=mode,
            router=router,
            lam=lam,
            min_level=min_level,
        )
        block = evidence.block
        return {
            "context": f"<memory>\n{block}\n</memory>" if wrap and block else block,
            "level": evidence.level,
            "reasons": evidence.reasons,
            "recalled": evidence.recalled,
            "see": evidence.see,
            "refs": evidence.refs,
        }

    def answer(
        self,
        question: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: AnswerMode = "auto",
        at: datetime | date | str | None = None,
        router: Router = "learned",
        lam: float | None = None,
    ) -> dict[str, Any]:
        """检索并回答，回答里的引用指向具体的事实与对话片段。过期的记忆（expiration_date 过了）
        不用，画像里也去掉，同 search。mode、router、lam 写错时 ValueError（与 /api/v1 相同）：
        MemorySystem.ask 的 "naive"、"facts" 是评测用的基线，不经过 SDK。"""
        _valid("mode", AnswerMode, mode)
        _valid("router", Router, router)
        lam = _valid("lam", Lam, lam)
        space = space_for(user_id, agent_id)
        answer = self.system.ask(
            space,
            question,
            question_time=_moment("at", at),
            mode=mode,
            router=router,
            lam=lam,
            hide_expired=True,
        )
        return _answer_out(answer)

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_superseded: bool = False,
        valid_at: str | date | None = None,
        known_at: datetime | date | str | None = None,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        show_expired: bool = False,
    ) -> dict[str, Any]:
        """全部事实：默认只给当前有效的。两条时间轴可以分别回放：valid_at 是那时现实中成立的
        事实，写成 "2025"、"2025-06"、"2025-06-07" 或 date；一年、一个月指这段时间里任何时候
        成立过的，先后几个值可能都在（别的写法 ValueError）。known_at 是系统在那一刻相信的事实
        （datetime、date、ISO 文本或 Unix 时间戳，同 search 的 at，见 _moment）。
        run_id：只要这个会话里记下的事实；filters 里的其他键是 metadata 的条件（同 search）。过期
        的事实默认不给，show_expired=True 时也给（以前除了 export 与已知的 ID，没有办法列出过期的
        记忆）。limit 同 search。"""
        _count("limit", limit)
        _count("top_k", top_k)
        user_id, agent_id, run_id, where = _scope(user_id, agent_id, run_id, filters)
        limit = limit if limit is not None else top_k
        space = space_for(user_id, agent_id)
        moment = _moment("known_at", known_at)
        current = not include_superseded and valid_at is None and moment is None
        facts = self.system.facts(space, current_only=current, valid_at=valid_at, known_at=moment)
        sources = _Sources.load(
            self.system.db, space, [t for f in facts for t in f.source_turn_ids]
        )
        select = _Select(
            run_id, where, _scope_record(user_id, agent_id), datetime.now(), show_expired
        )
        items = [
            _fact_item(f, None, user_id, agent_id, sources)
            for f in facts
            if select.keep(
                f.session_id, f.source_turn_ids, sources, _mem0_fields(f.recorded_at, f.category)
            )
        ]
        return {"results": items[:limit] if limit is not None else items}

    def get(self, memory_id: str) -> dict[str, Any] | None:
        """按 ID 取一条记忆（不用给 user_id：ID 里已经带着空间）：事实，或 search 给出的对话片段
        （type "excerpt"）与图片（type "image"，原文件用 media() 取）。过期的也取得到。以前只认
        事实的 ID，search 给出的另外两种都当成不存在。"""
        located = self._locate(memory_id)
        if located is None:
            return None
        space, kind = located
        if kind == "excerpt":
            units = self.system.db.get_units(space, [memory_id])
            if not units:  # 刚好被遗忘删掉了
                return None
            sources = _Sources.load(self.system.db, space, units[0].source_turn_ids)
            return {**_unit_item(units[0], None, sources), "space_id": space}
        if kind == "image":
            image = self._image(memory_id)
            if image is None:
                return None
            sources = _Sources.load(self.system.db, space, [image.source_turn_id])
            return {**_image_item(image, False, sources), "space_id": space}
        found = self._find(memory_id)
        if found is None:
            return None
        space, fact = found
        sources = _Sources.load(self.system.db, space, fact.source_turn_ids)
        return _fact_item(fact, None, None, None, sources, space=space)

    def media(self, media_id: str) -> bytes | None:
        """一张图片的原文件（search 结果 images 里的 id）：图片项里的 path 是服务器上的路径，
        远程调用方（MemoryClient）用这个方法取图片。没有这张图片、或原图已经不在时是 None。"""
        image = self._image(media_id)
        if image is None:
            return None
        path = Path(image.path)
        return path.read_bytes() if path.is_file() else None

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        """一条事实的版本链，从最早到最新：每个版本何时记下、何时因为什么被取代。只有事实有
        版本：对话片段与图片的 ID 与不存在的一样，是空列表。"""
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
        sources = _Sources.load(
            self.system.db, space, [t for f in chain for t in f.source_turn_ids]
        )
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
        重建派生记忆。被它取代过的旧事实会恢复为当前有效。memory_id 也可以是 search 给出的
        对话片段或图片的 ID：删掉那一轮对话（图片所在的那一轮），与 MCP、各框架集成的 forget
        相同。以前这两种 ID 报 KeyError，Mem0 的 for r in results: delete(r["id"]) 中途出错。
        与 Mem0 不同的是删的是整轮对话：同一轮的对话片段、由它抽出的别的事实随之消失，之后再
        delete 它们的 ID 是 KeyError（与不存在的一样），逐条删时要接住。每次 delete 都用剩下的
        日志重放一遍，重放还可能给留下的事实换新的 ID：拿着之前 get_all 的 ID 逐条删时，没删的
        事实也可能 KeyError。一次删好几条用 delete_many：先认出全部 ID，每个空间只重放一次。

        dry_run=True 只预演：返回会删哪些发言、哪些事实会随之消失，什么都不删。删除做完之后
        重建出错（模型供应商余额不足等）时不抛出：已经删了，warning 写明重建下次巩固时补上。

        一句发言都没删、这条事实却还在（它出处的发言已经不在日志里，例如之前的版本遗忘到一半
        出错留下的）时 message 是 "Nothing deleted"，warning 让调用方 consolidate(rebuild=True)
        （检索仍然缺对话片段时再 cuemem reindex 补齐向量）：以前照样回 "Memory deleted"。"""
        located = self._locate(memory_id)
        if located is not None and located[1] != "fact":
            return self._delete_source(located[0], located[1], memory_id, dry_run=dry_run)
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
        if not report.turns_deleted and self.system.db.get_facts(found[0], ids=[memory_id]):
            return {"message": "Nothing deleted", **_forget_out(report), "warning": _STUCK}
        return {"message": "Memory deleted", **_forget_out(report)}

    def delete_many(self, memory_ids: Sequence[str], *, dry_run: bool = False) -> dict[str, Any]:
        """一次遗忘好几条记忆（事实、对话片段或图片的 ID，与 delete 认的相同）：先认出全部 ID，
        每个空间调用一次遗忘、重放一次。逐条 delete 时 k 条是 k 次重放，每次都独占这个空间；
        同一轮里的两个 ID、重放之后换了 ID 的事实也不会中途 KeyError。

        最多 DELETE_MANY_MAX（100）个 ID（去重之后），多了 ValueError：删事实时找转述要对每条
        陈述调用一次便宜的模型。memory_ids 是单个字符串时 TypeError。不认识的 ID 不报错，放进
        返回值的 missing（经 /api/v1 时作用域之外的也在这里，与不存在的一样）；deleted 是认出来、
        删掉（预演时是会删）的 ID。

        dry_run=True 只预演，返回 delete 的预演的各项（几个空间合起来）。删除的返回值与 delete
        的相同，计数是几个空间相加；重建出错时 warning 只写一次。好几个空间时逐个删，中途出错
        时前面的空间已经删了。"""
        return self._delete_many(memory_ids, dry_run=dry_run)

    def batch_delete(self, memories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Mem0 的 MemoryClient.batch_delete 的写法：memories=[{"memory_id": ...}, ...]，与
        delete_many 相同（不预演）。"""
        if isinstance(memories, str | Mapping):
            raise TypeError('memories must be a list of {"memory_id": ...} objects.')
        ids: list[str] = []
        for item in memories:
            if not isinstance(item, Mapping) or "memory_id" not in item:
                raise ValueError('Each item of memories must be an object with "memory_id".')
            ids.append(item["memory_id"])
        return self.delete_many(ids)

    def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """删除这个用户（或 Agent）的全部记忆：原始日志、派生记忆、向量、缓存与原图，以及这个
        空间的 trace 与库里写着它的 ID 的元信息（见 MemorySystem.delete_space）。
        run_id：只删这个会话（与按会话遗忘相同，再用剩下的日志重建派生记忆），空间留着。日志里
        没有这个会话的发言（run_id 写错了）时什么都不做，message 是 "No such run: nothing
        deleted"：以前照样回 "Memories of this run deleted"，与删掉了分不出来。

        dry_run=True 只预演这个会话的删除（与 delete 的 dry_run、cuemem forget --session
        --dry-run 相同）：返回会删哪些发言、哪些事实会随之消失，什么都不删。只能与 run_id 一起
        用（ValueError）：不给 run_id 时删的是整个用户，没有预演。"""
        space = space_for(user_id, agent_id)
        if dry_run and run_id is None:
            raise ValueError(
                "dry_run needs run_id; delete_all without run_id deletes the whole user"
            )
        if run_id is not None:
            if dry_run:
                return {
                    "message": "Dry run: nothing was deleted",
                    **self.system.preview_forget(space, session_ids=[run_id]),
                }
            if not self.system.db.get_turns(space, run_id):  # 只读这一个会话
                from cuemem.memory import ForgetReport

                empty = ForgetReport(space, 0, [], 0, 0, 0, 0, 0.0)
                return {"message": "No such run: nothing deleted", **_forget_out(empty)}
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
        return self._reset()

    def _reset(self, keep: Callable[[str], bool] | None = None) -> dict[str, Any]:
        """reset 的实现。keep：只删它认的空间（/api/v1/reset 收到限定作用域的令牌时是
        Principal.allows），也只结束、只忘掉这些空间里的 run 与接手记录，别的空间不碰；None 时
        是全部。不做成 reset 的公开参数：MemoryClient 与 Memory 的签名逐字相同。"""

        def chosen(space: str) -> bool:
            return keep is None or keep(space)

        with self._runs_lock:
            runs = sorted(r for r in self._runs if chosen(r[0]))
            self._runs.difference_update(runs)
            self._run_seen = {k: t for k, t in self._run_seen.items() if not chosen(k[0])}
            self._idle_closed = {k: v for k, v in self._idle_closed.items() if not chosen(k[0])}
            self._adopted.difference_update([s for s in self._adopted if chosen(s)])
        for space, rid in runs:
            self.system.close_session(space, rid, submit=False)
        spaces = [s["space_id"] for s in self.system.db.list_spaces() if chosen(s["space_id"])]
        # 这些空间的 trace 最后一起删：每个空间删一遍要把 trace 文件读一遍。中途出错（例如向量库
        # 连不上）时也删掉已经开始删的空间的 trace：删完了的不在空间列表里，重试时没人再删它们
        reports: list[ForgetReport] = []
        started: list[str] = []
        try:
            for space in spaces:
                started.append(space)
                reports.append(self.system.delete_space(space, purge_traces=False))
        finally:
            self.system.purge_space_traces(started)
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
            self._run_seen.clear()
            self._idle_closed.clear()
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

    def _locate(self, memory_id: str) -> tuple[str, MemoryKind] | None:
        """（空间，种类）：search 给出的三种 ID——事实、对话片段、图片（MemoryDB.memory_kind）。"""
        return self.system.db.memory_kind(memory_id)

    def _find(self, memory_id: str) -> tuple[str, Fact] | None:
        located = self._locate(memory_id)
        if located is None or located[1] != "fact":
            return None
        facts = self.system.db.get_facts(located[0], ids=[memory_id])
        return (located[0], facts[0]) if facts else None

    def _image(self, media_id: str) -> MediaItem | None:
        located = self._locate(media_id)
        if located is None or located[1] != "image":
            return None
        items = self.system.db.get_media(located[0], ids=[media_id])
        return items[0] if items else None

    def _delete_source(
        self, space: str, kind: MemoryKind, memory_id: str, *, dry_run: bool
    ) -> dict[str, Any]:
        """delete 一个对话片段或一张图片：按发言（片段那一轮的发言）或按图片遗忘，与 MCP、
        各框架集成的 forget（evidence.resolve_ids）相同。"""
        from cuemem.evidence import resolve_ids

        turns = resolve_ids(self.system, space, [memory_id])[1] if kind == "excerpt" else []
        media = [memory_id] if kind == "image" else []
        if dry_run:
            return {
                "message": "Dry run: nothing was deleted",
                **self.system.preview_forget(space, turn_ids=turns, media_ids=media),
            }
        report = self.system.forget_memories(
            space, turn_ids=turns, media_ids=media, report_rebuild_errors=True
        )
        return {"message": "Memory deleted", **_forget_out(report)}

    def _delete_many(
        self,
        memory_ids: Sequence[str],
        *,
        dry_run: bool,
        keep: Callable[[str], bool] | None = None,
    ) -> dict[str, Any]:
        """delete_many 的实现。keep：只删它认的空间里的记忆（/api/v1 收到限定作用域的令牌时是
        Principal.allows），别的空间的 ID 与不存在的一样放进 missing；None 时不限。不做成公开
        参数：MemoryClient 与 Memory 的签名逐字相同（与 _reset 一样）。"""
        from cuemem.evidence import resolve_ids

        ids = _memory_ids(memory_ids)
        # 空间 → 种类 → ID：先全部认出来，再删（删了之后别的 ID 可能就找不到了）
        plan: dict[str, dict[MemoryKind, list[str]]] = {}
        for memory_id in ids:
            located = self._locate(memory_id)
            if located is not None and (keep is None or keep(located[0])):
                plan.setdefault(located[0], {}).setdefault(located[1], []).append(memory_id)
        for space, kinds in plan.items():  # 与 delete 相同：库里没有的事实算不存在
            if kinds.get("fact"):
                found = {f.id for f in self.system.db.get_facts(space, ids=kinds["fact"])}
                kinds["fact"] = [i for i in kinds["fact"] if i in found]
        chosen = {i for kinds in plan.values() for group in kinds.values() for i in group}
        deleted = [i for i in ids if i in chosen]
        missing = [i for i in ids if i not in chosen]
        targets = []
        for space in sorted(plan):
            kinds = plan[space]
            excerpts = kinds.get("excerpt", [])
            turns = resolve_ids(self.system, space, excerpts)[1] if excerpts else []
            if turns or kinds.get("fact") or kinds.get("image"):
                targets.append((space, kinds.get("fact", []), turns, kinds.get("image", [])))

        if dry_run:
            previews = [
                self.system.preview_forget(space, fact_ids=facts, turn_ids=turns, media_ids=media)
                for space, facts, turns, media in targets
            ]
            return {
                "message": "Dry run: nothing was deleted",
                "deleted": deleted,
                "missing": missing,
                "turns": [t for p in previews for t in p["turns"]],
                "sessions_affected": sorted({s for p in previews for s in p["sessions_affected"]}),
                "cascaded": sum(p["cascaded"] for p in previews),
                "restated": sum(p["restated"] for p in previews),
                "facts": [f for p in previews for f in p["facts"]],
                "images": sum(p["images"] for p in previews),
            }

        reports: list[ForgetReport] = []
        stuck: set[str] = set()
        for space, facts, turns, media in targets:
            report = self.system.forget_memories(
                space, fact_ids=facts, turn_ids=turns, media_ids=media, report_rebuild_errors=True
            )
            reports.append(report)
            if not report.turns_deleted and facts:  # 同 delete 的 "Nothing deleted"
                stuck.update(f.id for f in self.system.db.get_facts(space, ids=facts))
        turns_deleted = sum(r.turns_deleted for r in reports)
        out: dict[str, Any] = {
            "message": "Memories deleted" if turns_deleted else "Nothing deleted",
            "deleted": [i for i in deleted if i not in stuck],
            "missing": missing,
            "turns_deleted": turns_deleted,
            "sessions_affected": sorted({s for r in reports for s in r.sessions_affected}),
            "facts_before": sum(r.facts_before for r in reports),
            "facts_after": sum(r.facts_after for r in reports),
            "images_deleted": sum(r.images_deleted for r in reports),
            "restated": sum(r.restated for r in reports),
        }
        # 重建出错的说明只写一次（与 delete 相同的措辞），出处不在日志里的事实再加一句
        failed = next((r for r in reports if r.rebuild_error), None)
        warnings = [_forget_out(failed)["warning"]] if failed is not None else []
        if stuck:
            warnings.insert(0, _STUCK)
        if warnings:
            out["warning"] = " ".join(warnings)
        return out

    def _next_idx(self, space: str, session_id: str) -> int:
        """会话里下一句发言的序号：只增不减，遗忘删掉的序号不再复用（与 Agent 相同）；
        新会话（包括删掉后重建的空间）从 0 开始，否则抽取时会被当成有发言被删过。记在
        next_idx_key 下；以前的版本记在 legacy_idx_key 下的照旧读。"""
        turns = self.system.db.get_turns(space, session_id)
        if not turns:
            return 0
        db = self.system.db
        recorded = db.get_meta(next_idx_key(space, session_id))
        if recorded is None:
            recorded = db.get_meta(legacy_idx_key(space, session_id))
        return max(int(recorded or 0), max(t.idx for t in turns) + 1)


class AsyncMemory(AsyncMethods):
    """Memory 的异步版本（与 Mem0 的 AsyncMemory 对应）：同样的方法与参数，在线程池里执行
    （cuemem/_async.py）。self.sync 是包着的 Memory；别的属性（.system、.settings）取它的。
    取消一次调用停不下它，被取消的写入照样可能生效，见 AsyncMethods。"""

    sync: Memory

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
        self.sync = Memory(
            config,
            preset=preset,
            home=home,
            instructions=instructions,
            background=background,
            peak_policy=peak_policy,
            http_client=http_client,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any], **kwargs: Any) -> AsyncMemory:
        return cls(config, **kwargs)

    @classmethod
    def from_system(cls, system: MemorySystem) -> AsyncMemory:
        """包住一个已经打开的记忆系统（见 Memory.from_system）。"""
        memory = cls.__new__(cls)
        memory.sync = Memory.from_system(system)
        return memory

    async def close_idle_runs(self, idle_s: float, *, now: float | None = None) -> int:
        """见 Memory.close_idle_runs（只有本地的有：MemoryClient 没有这个方法）。"""
        return await asyncio.to_thread(self.sync.close_idle_runs, idle_s, now=now)


# ---------- 输入 ----------

# Mem0 配置字典的键（Mem0 2.2 的 MemoryConfig）：出现时报错里附上换成 CueMem 写法的提示
_MEM0_KEYS = {
    "llm",
    "embedder",
    "vector_store.provider",
    "vector_store.config",
    "custom_instructions",
    "history_db_path",
    "reranker",
    "graph_store",
    "version",
}
# 是 Memory() 与 load_settings 的参数、不是配置字段的键：以前写进 config 时被悄悄丢掉（preset、
# home 报"multiple values"，config_path、env_file 被当成 load_settings 自己的参数）
_KEYWORD_KEYS = ("preset", "home", "config_path", "env_file")


def _check_config(config: Mapping[str, Any]) -> None:
    """Memory(config) 的配置字典里有配置模型不认识的键时报 ValueError（config.unknown_keys）。

    以前不认识的键被悄悄丢掉：写错的键（consolidation.instruction）不起作用，Mem0 格式的配置
    （llm: openai……）照样用默认的模型供应商，用户的话发给了调用方没有配置过的供应商。只查
    Memory() 收到的字典：Settings 对象、CUEMEM_* 环境变量与 YAML（CLI、评测都经过）不变。"""
    unknown = unknown_keys(config)
    if not unknown:
        return
    listed = ", ".join(f"{key} (did you mean {near}?)" if near else key for key, near in unknown)
    message = (
        f"Unknown config keys: {listed}. The config dict is merged into CueMem's own "
        "configuration (see configs/default.yaml); unknown keys are refused, not ignored."
    )
    keys = {key for key, _ in unknown}
    keywords = [key for key in _KEYWORD_KEYS if key in keys]
    if keywords:
        message += " Pass " + ", ".join(f"{k}=" for k in keywords) + " as keyword arguments."
    if keys & _MEM0_KEYS:
        choices = ", ".join(f'"{name}"' for name in presets())
        message += (
            " This looks like a Mem0 config, which CueMem does not read: choose the models "
            f'with a preset, Memory(preset="openai") (presets: {choices}), or with "models" '
            'and "roles" as in configs/default.yaml; pass custom_instructions as '
            "Memory(instructions=...); use a Qdrant server with "
            '{"vector_store": {"url": "http://localhost:6333"}}.'
        )
    raise ValueError(message)


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


IDEMPOTENCY_KEY_MAX = 255  # idempotency_key 最多这么多个字符（Idempotency-Key 头同）


def _key_hash(key: str) -> str:
    """idempotency_key 的 sha256（库里只存它）。键要是 1 到 255 个可打印的 ASCII 字符：它也要能
    放进 HTTP 头（MemoryClient 的 Idempotency-Key），否则 ValueError。"""
    if (
        not isinstance(key, str)
        or not 0 < len(key) <= IDEMPOTENCY_KEY_MAX
        or not all(" " <= c <= "~" for c in key)
    ):
        raise ValueError(
            f"idempotency_key must be 1 to {IDEMPOTENCY_KEY_MAX} printable ASCII characters."
        )
    return hashlib.sha256(key.encode()).hexdigest()


def _request_hash(
    messages: Any,
    user_id: str | None,
    agent_id: str | None,
    run_id: str | None,
    metadata: Mapping[str, Any] | None,
    expiration_date: date | datetime | str | None,
    timestamp: datetime | str | float | None,
    infer: bool,
) -> str:
    """一次 add 的请求的摘要（wait 不算）：同一个 idempotency_key 再来时据此判断是不是同一个请求。
    日期与时间按 ISO 写法算：本地调用的 datetime 与 MemoryClient 发来的 ISO 文本相同。"""

    def plain(value: Any) -> Any:
        if isinstance(value, date):  # datetime 也是 date
            return value.isoformat()
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"cannot hash {type(value).__name__}")

    request = [messages, user_id, agent_id, run_id, metadata, expiration_date, timestamp, infer]
    text = json.dumps(request, sort_keys=True, ensure_ascii=False, default=plain)
    return hashlib.sha256(text.encode()).hexdigest()


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


def _correction(text: str | None, data: str | None) -> str:
    """update 的 text 与 data（Mem0 2.x 的新旧两个名字）→ 更正的内容。MemoryClient.update 先在
    客户端做第一项检查（报错相同），都没给时由服务端的这里报错。"""
    if text is not None and data is not None and text != data:
        raise ValueError("Give text or data (the same argument), not two different values.")
    chosen = text if text is not None else data
    if chosen is None:
        raise ValueError(
            "update needs text: CueMem keeps metadata on the conversation turns a memory came "
            "from, so pass text= (the corrected memory) together with metadata=."
        )
    return chosen


@functools.cache
def _adapter(kind: Any) -> TypeAdapter[Any]:
    return TypeAdapter(kind)


def _valid(name: str, kind: Any, value: Any) -> Any:
    """search / answer 的 mode、router、min_level、lam：与 /api/v1 的请求模型（SearchRequest、
    AnswerRequest）同一个类型、同一句话（HTTP 的是 "Invalid request: body.mode: …"），本地与远程
    抛同样的 ValueError。以前本地不查：mode="l9" 悄悄走 L2，router="bogus" 走学习版控制器，
    min_level="l2" 只给画像。返回校验后的值（lam 的整数换成浮点数，同 HTTP）。"""
    try:
        return _adapter(kind).validate_python(value)
    except ValidationError as exc:
        raise ValueError(f"{name}: {exc.errors()[0]['msg']}") from None


def _count(name: str, value: int | None) -> None:
    """search / get_all 的 limit、top_k：None 或正整数。以前 limit=0 给出全部结果，limit=-1 悄悄
    少给一条（items[:-1]）。MemoryClient 有一份相同的检查（client._count）。"""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer (got {value!r}).")


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
      值是列表（如 categories）时 eq、ne、in、nin 逐项对（见 _compare）；
    - {"AND": [条件, ...]}、{"OR": [...]}、{"NOT": [...]}（NOT：一个都不满足）。

    条件里的 user_id、agent_id、run_id 指记忆所在的空间与会话，created_at、updated_at 与
    categories 也对记忆自己的字段（metadata 里没有这个键时，见 _Select），其余的键指 metadata。"""
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
        return present and (cond == "*" or _compare("eq", present, value, cond))
    return all(_compare(op, present, value, arg) for op, arg in cond.items())


def _compare(op: str, present: bool, value: Any, arg: Any) -> bool:
    if isinstance(value, list) and op in ("eq", "ne", "in", "nin"):
        # 值是列表（categories 等）时逐项对：in 有一项在 arg 里，nin 一项都不在，eq / ne 对列表里
        # 的项；参数本身是列表时 eq / ne 比整个列表，in / nin 也照旧认整个列表
        if op in ("eq", "ne"):
            hit = value == arg if isinstance(arg, list) else arg in value
        else:
            hit = value in arg or any(v in arg for v in value)
        return not hit if op in ("ne", "nin") else bool(hit)
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
class _Select:
    """search / get_all 的事后筛选：会话（run_id）、metadata 的条件与过期时间（show_expired 时
    不看过期时间；画像另算，总是去掉过期的）。

    条件里 Mem0 保留的几个名字（created_at、updated_at、categories）也对记忆自己的字段（fields，
    见 _mem0_fields）：以前只对 metadata，照 Mem0 写的 {"created_at": {"gte": …}}、
    {"categories": {"contains": …}} 悄悄什么都不给。metadata 里有同名的键时以 metadata 为准
    （以前能筛出来的照样筛出来），user_id、agent_id、run_id 总是指空间与会话。"""

    run_id: str | None
    where: Mapping[str, Any] | None
    scope: Mapping[str, str]
    now: datetime
    show_expired: bool = False

    def keep(
        self,
        session_id: str | None,
        turn_ids: Sequence[str],
        sources: _Sources,
        fields: Mapping[str, Any] | None = None,
    ) -> bool:
        if self.run_id is not None and session_id != self.run_id:
            return False
        if not self.show_expired and sources.expired(turn_ids, self.now):
            return False
        if not self.where:
            return True
        record = {**(fields or {}), **sources.metadata(turn_ids), **self.scope}
        if session_id is not None:
            record["run_id"] = session_id
        return _matches(self.where, record)


def _mem0_fields(created_at: datetime, category: str | None = None) -> dict[str, Any]:
    """filters 里 Mem0 保留的名字对应的记忆自己的字段（见 _Select）：created_at、updated_at 是
    结果里 created_at 的同一个 ISO 字符串（按字符串比较，{"gte": "2026-01-01"} 就能用），事实
    另有 categories（它的 category，一个元素的列表）。"""
    stamp = created_at.isoformat()
    out: dict[str, Any] = {"created_at": stamp, "updated_at": stamp}
    if category is not None:
        out["categories"] = [category]
    return out


def _profile(system: MemorySystem, space: str, core: CoreMemory | None, now: datetime) -> str:
    """画像去掉已经过期的事实（规则见 expiry.visible_core）。"""
    shown = visible_core(system.db, space, core, now)
    return shown.text if shown is not None else ""


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


def _moment(name: str, value: datetime | date | str | float | None) -> datetime | None:
    """search、context、answer 的 at 与 get_all 的 known_at → 本机时间（不带时区，见
    naive_local）。与 /api/v1（MemoryClient）收的写法相同（第七轮体检）：datetime；date 是那天
    零点；ISO 文本（带时区的先换算）；数字是 Unix 时间戳。以前只认 datetime，别的写法在空间里
    有记忆时 AttributeError。别的写法 ValueError，开头是参数名。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return naive_local(value)
    if isinstance(value, date):  # datetime 也是 date：放在后面
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return naive_local(datetime.fromisoformat(value))
        except ValueError:
            raise ValueError(f"{name}: invalid ISO date or datetime {value!r}") from None
    if isinstance(value, int | float) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value)
        except (OverflowError, OSError, ValueError):
            raise ValueError(f"{name}: invalid Unix timestamp {value!r}") from None
    raise ValueError(f"{name}: expected a datetime, date, ISO string or Unix timestamp")


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


def _unit_item(unit: MemoryUnit, score: float | None, sources: _Sources) -> dict[str, Any]:
    """对话片段（"轮"）→ 结果里的一项：search 的带 score，get 的不带。"""
    item: dict[str, Any] = {"id": unit.id, "memory": unit.text, "type": "excerpt"}
    if score is not None:
        item["score"] = round(score, 4)
    return {
        **item,
        "created_at": unit.ts.isoformat(),
        "session_id": unit.session_id,
        "source_turn_ids": unit.source_turn_ids,
        "metadata": sources.metadata(unit.source_turn_ids),
        "expiration_date": sources.expires(unit.source_turn_ids),
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
        system.db,
        space,
        [
            *(t for fh in view.facts for t in fh.fact.source_turn_ids),
            *(t for h in view.rounds for t in h.unit.source_turn_ids),
            *(m.source_turn_id for m in media),
        ],
    )
    facts: list[FactHit] = [
        fh
        for fh in view.facts
        if select.keep(
            fh.fact.session_id,
            fh.fact.source_turn_ids,
            sources,
            _mem0_fields(fh.fact.recorded_at, fh.fact.category),
        )
    ]
    rounds = [
        h
        for h in view.rounds
        if select.keep(h.unit.session_id, h.unit.source_turn_ids, sources, _mem0_fields(h.unit.ts))
    ]
    # 图片来自装入的轮（l1.collect_media）：会话按它所在的那一轮算
    session_of = {
        m: h.unit.session_id for h in view.rounds for m in h.unit.meta.get("media_ids", [])
    }
    media = [
        m
        for m in media
        if select.keep(session_of.get(m.id), [m.source_turn_id], sources, _mem0_fields(m.ts))
    ]
    items = [_fact_item(fh.fact, fh.score, user_id, agent_id, sources) for fh in facts]
    items += [_unit_item(h.unit, h.score, sources) for h in sorted(rounds, key=lambda h: -h.score)]
    decision = recall.decision
    return {
        "results": items[:limit] if limit is not None else items,
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


def _not_extracted(error: Exception | str) -> str:
    """发言已经写进原始日志（巩固任务也登记了），同步抽取却出错（例如对账时余额不足、向量接口
    暂时不可用）时回给调用方的说明。不抛出：抛出的话调用方以为没写进去，重试就写了两遍。没做完的
    任务留着，下次巩固（consolidate()，或后台线程下一次处理这个空间）时补上。

    error：巩固抛出的异常，或者抽取失败的任务记下的出错（"<类型>: <说明>"，见 _job_error：逐个
    会话的抽取失败不抛出）。日志只记出错的第一行：结构化输出不合格时，之后几行引着模型的输出。"""
    if isinstance(error, Exception):
        name, text = type(error).__name__, str(error)
    else:
        name, _, text = (error or "Error: unknown").partition(": ")
    lines = text.splitlines()
    first = lines[0][:300] if lines else ""
    logger.warning("Extraction failed after the messages were stored: %s: %s", name, first)
    return (
        f"Extraction failed ({name}: {text[:300]}). The messages are stored; "
        "their extraction is retried on the next consolidation. Do not add them again."
    )


def _job_error(system: MemorySystem, space: str, session_id: str) -> str:
    """这个会话抽取失败的任务记下的出错（"<类型>: <说明>"）。"""
    jobs = system.db.get_jobs(space, status="failed")
    return next((j["error"] or "" for j in jobs if j["session_id"] == session_id), "")


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
    sources = _Sources.load(system.db, space, fresh)
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


DELETE_MANY_MAX = 100  # delete_many 一次最多这么多个 ID：删事实时每条陈述调用一次模型找转述
# delete 一句发言都没删、事实却还在时的说明（delete_many 同）
_STUCK = (
    "Nothing was deleted: none of the conversation turns this memory came from "
    "are in the log any more (an earlier version can leave this behind when a "
    "forget failed partway). Run consolidate(rebuild=True) to rebuild the "
    "memories from the log; if search still misses conversation excerpts "
    "afterwards, `cuemem reindex` restores the vectors the log still has."
)


def _memory_ids(memory_ids: Sequence[str]) -> list[str]:
    """delete_many 的 ID：字符串的列表（单个字符串 TypeError），按顺序去重，最多
    DELETE_MANY_MAX 个（ValueError）。"""
    if isinstance(memory_ids, str | bytes) or not isinstance(memory_ids, Sequence):
        raise TypeError("memory_ids must be a list of memory ids, not a single id.")
    if not all(isinstance(i, str) for i in memory_ids):
        raise TypeError("memory_ids must be a list of strings.")
    ids = list(dict.fromkeys(memory_ids))
    if len(ids) > DELETE_MANY_MAX:
        raise ValueError(
            f"At most {DELETE_MANY_MAX} memory ids per call (got {len(ids)}); split them into "
            "several calls."
        )
    return ids


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
