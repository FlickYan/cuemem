"""过期的记忆（Mem0 的 expiration_date）：哪些事实、对话片段与图片已经过期，画像里去掉哪几行。

过期时间存在发言上（turn_meta.expires_at，SDK 写入时给的）。一条记忆的出处各句都过期了它才
过期：同一件事后来又说过一次、没说什么时候过期，它就一直有效。以前只有 SDK 的 search、get_all
与 profile 按它筛（sdk._Sources、_profile）；第五轮体检（P7）把规则挪到这里，MemorySystem 的
retrieve / ask（hide_expired=True）也用它：answer()、记忆代理、MCP 与各框架集成都不再给出过期的
记忆。评测与命令行不打开 hide_expired：它们的空间从不写 turn_meta，请求本来也不会变。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cuemem.core.models import CoreMemory, TurnMeta
from cuemem.core.timeutil import expiry_instant

if TYPE_CHECKING:
    from cuemem.store.sqlite import MemoryDB


@dataclass
class Sources:
    """一批记忆出处各句的附加信息（turn_meta）。事实与对话片段的 metadata 是出处各句的并集
    （同一个键取后面那句的）；过期时间取出处里最晚的，有一句没给过期时间就不过期：同一件事
    后来又说过一次、没说什么时候过期，它就一直有效。"""

    meta: Mapping[str, TurnMeta]

    @classmethod
    def load(cls, db: MemoryDB, space: str, turn_ids: Iterable[str]) -> Sources:
        ids = list(dict.fromkeys(turn_ids))
        return cls(db.get_turn_meta(space, ids) if ids else {})

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


def visible_core(
    db: MemoryDB, space: str, core: CoreMemory | None, now: datetime
) -> CoreMemory | None:
    """画像去掉已经过期的事实。画像是事实表的物化视图（write/profile.py）：每条事实一行
    "- 陈述"，多人时前面有一行人名；某人名下的事实都过期了，人名那一行也去掉。没有过期的
    事实时原样返回同一个对象。"""
    if core is None:
        return None
    facts = db.get_facts(space, ids=core.fact_ids)
    sources = Sources.load(db, space, [t for f in facts for t in f.source_turn_ids])
    expired = [f for f in facts if sources.expired(f.source_turn_ids, now)]
    if not expired:
        return core
    gone = {f"- {f.statement}" for f in expired}
    lines = [line for line in core.text.split("\n") if line not in gone]
    text = "\n".join(
        line
        for i, line in enumerate(lines)
        if line.startswith("- ") or (i + 1 < len(lines) and lines[i + 1].startswith("- "))
    )
    dropped = {f.id for f in expired}
    kept = [fid for fid in core.fact_ids if fid not in dropped]
    return core.model_copy(update={"text": text, "fact_ids": kept})


def expired_ids(db: MemoryDB, space: str, now: datetime) -> frozenset[str]:
    """这个空间里已经过期的记忆：对话片段等检索单元、事实与图片的 ID（检索的 exclude 用）。

    没有带过期时间的发言时（绝大多数空间，评测的空间都是）只查一次 turn_meta、返回空集合。
    否则只看过期发言所在的会话：检索单元与事实的出处都是同一会话里的发言（对账合并来源时沿用
    旧事实的会话），图片按它出自的发言。规则同 Sources.expired：出处各句都过期了才算。"""
    stamps = db.expiring_turns(space)
    gone = {turn for turn, stamp in stamps.items() if expiry_instant(stamp) <= now}
    if not gone:
        return frozenset()
    turns = sorted(gone)
    sessions: set[str] = set()
    for start in range(0, len(turns), 500):
        sessions.update(t.session_id for t in db.get_turns(space, ids=turns[start : start + 500]))

    def dead(turn_ids: Sequence[str]) -> bool:
        return bool(turn_ids) and all(t in gone for t in turn_ids)

    within = sorted(sessions)
    ids = {u.id for u in db.list_units(space, session_ids=within) if dead(u.source_turn_ids)}
    ids.update(f.id for f in db.get_facts(space, session_ids=within) if dead(f.source_turn_ids))
    for start in range(0, len(turns), 500):
        ids.update(m.id for m in db.get_media(space, turn_ids=turns[start : start + 500]))
    return frozenset(ids)
