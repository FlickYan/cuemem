"""在线写入（同步、不调用 LLM）：追加原始轮次 → 切分为"轮" → 向量化 → 写入索引。

带图片的发言（N6）：原图按内容哈希存进媒体库，登记图片记录，并算出图片向量（以文搜图用）。
巩固（抽取事实、实体、摘要、图片描述）不在这里做，而是登记成任务，由 write/consolidate.py 异步完成。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuemem.core.ids import stable_id
from cuemem.core.models import Conversation, MediaItem, MemoryUnit, SessionIn, Turn
from cuemem.core.timeutil import to_epoch
from cuemem.core.trace import span
from cuemem.embed.base import TextEmbedder
from cuemem.embed.multimodal import ImageEmbedder
from cuemem.store.media import MediaStore
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
    images_added: int = 0


def media_id(space_id: str, turn_id: str, index: int) -> str:
    """一次发言里第 index 张图的记录 ID（Agent 据此在检索时排除工作记忆里的图片）。"""
    return stable_id(space_id, "media", turn_id, index)


def build_turns(space_id: str, session: SessionIn, first_idx: int = 0) -> list[Turn]:
    """发言的序号从 first_idx 起（导入整段对话时从 0 起；Agent 追加新的一问一答时接着往下排）。"""
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
        for idx, turn in enumerate(session.turns, start=first_idx)
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
        # turn_indices：评测时据此判断是否命中证据
        meta: dict[str, Any] = {"turn_indices": [t.idx for t in group]}
        media_ids = [m for t in group for m in t.media_ids]
        if media_ids:
            meta["media_ids"] = media_ids  # 这一轮里的图片，按出现顺序（回答时把描述填回原位）
        units.append(
            MemoryUnit(
                id=stable_id(first.space_id, "round", first.session_id, first.idx),
                space_id=first.space_id,
                kind="round",
                session_id=first.session_id,
                text=text,
                ts=first.ts,
                source_turn_ids=[t.id for t in group],
                meta=meta,
            )
        )
    return units


def _label(turn: Turn) -> str:
    if turn.speaker:
        return turn.speaker
    return "User" if turn.role == "user" else "Assistant"


class OnlineWriter:
    def __init__(
        self,
        db: MemoryDB,
        vectors: VectorStore,
        embedder: TextEmbedder,
        *,
        media: MediaStore | None = None,
        image_embedder: ImageEmbedder | None = None,
    ) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.media = media  # 为空时忽略图片（文本里的 "<image>" 标记照常保留）
        self.image_embedder = image_embedder

    def ingest(self, conversation: Conversation) -> IngestReport:
        return self._write(conversation.space_id, [(s, 0) for s in conversation.sessions])

    def append(self, space_id: str, session: SessionIn, first_idx: int) -> IngestReport:
        """往一个会话里追加发言（Agent 每轮写入新的一问一答），序号从 first_idx 起。

        与重新导入整个会话不同，已有的发言不再处理；遗忘删掉的发言也不会被写回来。
        """
        return self._write(space_id, [(session, first_idx)])

    def _write(self, space_id: str, sessions: Sequence[tuple[SessionIn, int]]) -> IngestReport:
        started = time.perf_counter()
        self.db.ensure_space(space_id)
        turns_added = 0
        units: list[MemoryUnit] = []
        media: list[MediaItem] = []
        for session, first_idx in sessions:
            self.db.add_session(space_id, session.session_id, session.started_at)
            turns = build_turns(space_id, session, first_idx)
            if self.media is not None:
                media.extend(self._attach_images(session, turns))
            turns_added += self.db.add_turns(turns)
            units.extend(build_rounds(turns))

        images_added = self._index_images(space_id, media)
        new_units, tokens = self._index(space_id, units)
        return IngestReport(
            space_id=space_id,
            sessions=len(sessions),
            turns_added=turns_added,
            units_added=len(new_units),
            embed_tokens=tokens,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            images_added=images_added,
        )

    def _attach_images(self, session: SessionIn, turns: Sequence[Turn]) -> list[MediaItem]:
        """把发言里的图片存进媒体库，生成图片记录，并把图片 ID 记在发言上。"""
        assert self.media is not None
        items: list[MediaItem] = []
        for turn, turn_in in zip(turns, session.turns, strict=True):
            for i, path in enumerate(turn_in.images):
                stored = self.media.put(Path(path))
                items.append(
                    MediaItem(
                        id=media_id(turn.space_id, turn.id, i),
                        space_id=turn.space_id,
                        sha256=stored.sha256,
                        path=str(stored.path),
                        source_turn_id=turn.id,
                        ts=turn.ts,
                        width=stored.width,
                        height=stored.height,
                    )
                )
            turn.media_ids = [m.id for m in items if m.source_turn_id == turn.id]
        return items

    def _index_images(self, space_id: str, items: Sequence[MediaItem]) -> int:
        """新图片：先写图片向量，再登记图片记录（与文本单元一样，数据库里有就说明向量已写入）。"""
        if not items:
            return 0
        existing = {m.id for m in self.db.get_media(space_id, ids=[m.id for m in items])}
        new = [m for m in items if m.id not in existing]
        if not new:
            return 0
        if self.image_embedder is not None:
            with span("embed_images", images=len(new)):
                vectors = self.image_embedder.embed_images([Path(m.path) for m in new])
            self.vectors.upsert_images(
                space_id,
                [m.id for m in new],
                vectors,
                [{"turn_id": m.source_turn_id, "ts": to_epoch(m.ts)} for m in new],
            )
        return self.db.add_media(new)

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
