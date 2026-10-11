"""在线写入（同步、不调用 LLM）：追加原始轮次 → 切分为"轮" → 向量化 → 写入索引。

巩固（抽取事实、实体、摘要）不在这里做，而是登记成任务，由 write/consolidate.py 异步完成。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cuemem.core.ids import stable_id
from cuemem.core.models import Conversation, MemoryUnit, SessionIn, Turn
from cuemem.core.timeutil import to_epoch
from cuemem.core.trace import span
from cuemem.embed.base import TextEmbedder
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore


@dataclass
class IngestReport:
    space_id: str
    sessions: int
    turns_added: int
    units_added: int
    embed_tokens: int
    elapsed_ms: float


def build_turns(space_id: str, session: SessionIn) -> list[Turn]:
    return [
        Turn(
            id=stable_id(space_id, session.session_id, idx),
            space_id=space_id,
            session_id=session.session_id,
            idx=idx,
            role=turn.role,
            text=turn.text,
            ts=turn.ts or session.started_at,
            speaker=turn.speaker,
        )
        for idx, turn in enumerate(session.turns)
    ]


def build_rounds(turns: Sequence[Turn]) -> list[MemoryUnit]:
    """一问一答为一"轮"：每次用户发言开启新的一轮，其后的助手回复并入该轮。

    多人对话里每个人都是 user 角色，因此每条发言各成一轮；文本前缀用说话人名字。
    """
    groups: list[list[Turn]] = []
    for turn in turns:
        if turn.role == "user" or not groups:
            groups.append([turn])
        else:
            groups[-1].append(turn)
    units = []
    for group in groups:
        first = group[0]
        text = "\n".join(f"{_label(t)}: {t.text}" for t in group)
        units.append(
            MemoryUnit(
                id=stable_id(first.space_id, "round", first.session_id, first.idx),
                space_id=first.space_id,
                kind="round",
                session_id=first.session_id,
                text=text,
                ts=first.ts,
                source_turn_ids=[t.id for t in group],
                meta={"turn_indices": [t.idx for t in group]},  # 评测时据此判断是否命中证据
            )
        )
    return units


def _label(turn: Turn) -> str:
    if turn.speaker:
        return turn.speaker
    return "User" if turn.role == "user" else "Assistant"


class OnlineWriter:
    def __init__(self, db: MemoryDB, vectors: VectorStore, embedder: TextEmbedder) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder

    def ingest(self, conversation: Conversation) -> IngestReport:
        started = time.perf_counter()
        space_id = conversation.space_id
        self.db.ensure_space(space_id)
        turns_added = 0
        units: list[MemoryUnit] = []
        for session in conversation.sessions:
            self.db.add_session(space_id, session.session_id, session.started_at)
            turns = build_turns(space_id, session)
            turns_added += self.db.add_turns(turns)
            units.extend(build_rounds(turns))

        new_units, tokens = self._index(space_id, units)
        return IngestReport(
            space_id=space_id,
            sessions=len(conversation.sessions),
            turns_added=turns_added,
            units_added=len(new_units),
            embed_tokens=tokens,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def reindex_sessions(self, space_id: str, session_ids: Sequence[str]) -> int:
        """按数据库里现存的轮次重建这些会话的"轮"（遗忘部分轮次之后使用）。

        调用方需先删除这些会话原有的"轮"；返回新写入的单元数。
        """
        units: list[MemoryUnit] = []
        for session_id in session_ids:
            units.extend(build_rounds(self.db.get_turns(space_id, session_id)))
        new_units, _ = self._index(space_id, units)
        return len(new_units)

    def _index(self, space_id: str, units: Sequence[MemoryUnit]) -> tuple[list[MemoryUnit], int]:
        missing = set(self.db.missing_unit_ids(space_id, [u.id for u in units]))
        new_units = [u for u in units if u.id in missing]
        if not new_units:
            return [], 0
        with span("embed", texts=len(new_units)):
            batch = self.embedder.embed([u.text for u in new_units], "document")
        self.vectors.upsert(
            space_id,
            [u.id for u in new_units],
            batch.dense,
            batch.sparse,
            [_payload(u) for u in new_units],
        )
        # 先写向量、后写数据库：数据库里有这条记录，就说明向量一定已经写入；
        # 中途失败时重跑会补齐（向量化命中缓存，不会重复计费）。
        self.db.add_units(new_units)
        return new_units, batch.tokens


def _payload(unit: MemoryUnit) -> dict[str, Any]:
    return {"kind": unit.kind, "session_id": unit.session_id, "ts": to_epoch(unit.ts)}
