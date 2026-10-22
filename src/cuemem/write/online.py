"""在线写入（同步、不调用 LLM）：原始轮次 → 切分为"轮" → 向量化、写入索引 → 一起提交。

带图片的发言（N6）：原图按内容哈希存进媒体库，登记图片记录，并算出图片向量（以文搜图用）。
巩固（抽取事实、实体、摘要、图片描述）不在这里做，而是登记成任务，由 write/consolidate.py 异步完成。

一次写入要么全部写进去、要么什么都不留（见 OnlineWriter.writing）：向量化或写向量出错时调用方
可以原样重试，发言不会写两遍（重试有没有用见 worth_retrying）。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import UnidentifiedImageError

from cuemem.core.ids import stable_id
from cuemem.core.models import MediaItem, MemoryUnit, SessionIn, Turn, TurnMeta
from cuemem.core.timeutil import to_epoch
from cuemem.core.trace import span
from cuemem.embed.base import EmbeddingBatch, TextEmbedder
from cuemem.embed.multimodal import ImageEmbedder
from cuemem.errors import ConcurrentWriteError, ContentFilteredError, EmbeddingAPIError
from cuemem.store.media import MediaStore, StoredImage
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore

logger = logging.getLogger(__name__)


def worth_retrying(exc: BaseException) -> bool:
    """在线写入出错之后，过一会儿把同样的发言原样再写一次有没有可能写进去：记忆代理与 Agents 的
    会话据此决定留着重试还是丢掉。向量接口暂时不可用、限流、超时、连不上、余额不足，向量库或
    数据库出错时有。没有的：向量接口拒绝了请求本身（别的 4xx）、内容审核、发言本身不合法（图片
    读不出来等），留着它们，这个会话后面的写入都得排在它们后面；撞号（ConcurrentWriteError）时
    已经写进去了一部分，再写就重复了。"""
    if isinstance(
        exc,
        ConcurrentWriteError
        | ContentFilteredError
        | UnidentifiedImageError
        | ValueError
        | TypeError
        | LookupError,
    ):
        return False
    if isinstance(exc, EmbeddingAPIError) and exc.status is not None:
        return exc.status in (402, 408, 429) or exc.status >= 500
    return isinstance(exc, Exception)


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


@dataclass
class _Staged:
    """一次写入里已经写进向量库、还没写进数据库的部分（见 OnlineWriter.writing）。"""

    space_id: str
    # 会话、发言与要随发言写的附加信息（轮 ID → TurnMeta）
    pieces: list[tuple[SessionIn, list[Turn], dict[str, TurnMeta]]] = field(default_factory=list)
    media: list[MediaItem] = field(default_factory=list)  # 新的图片记录
    units: list[MemoryUnit] = field(default_factory=list)  # 新的"轮"
    stored: list[StoredImage] = field(default_factory=list)  # 存进媒体库的原图（每次 put）
    tokens: int = 0


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

    @contextmanager
    def writing(
        self,
        space_id: str,
        batches: Sequence[Sequence[tuple[SessionIn, int]]],
        meta: Mapping[tuple[str, int], TurnMeta | None] | None = None,
    ) -> Iterator[IngestReport]:
        """写入这些会话的发言，要么全部写进去，要么什么都不留。每项是会话与第一句的序号（导入
        整段对话时从 0 起；Agent 追加新的一问一答时接着往下排）。with 块里调用方的写入（登记巩固
        任务、溯源）与发言在同一个事务里提交，块里出错时一起撤销。

        先存原图、向量化、写向量（点的 ID 由发言的位置决定，重写是幂等的），再在一个事务里写会话、
        发言、图片记录与"轮"。以前先提交发言、再向量化：向量接口或向量库出错时日志里已经有了这些
        发言，却没有"轮"、也没有巩固任务，调用方重试又把同样的话写一遍。现在出错时数据库里什么
        都没写，已经写进向量库的点尽量删掉，存进媒体库、别处没有用到的原图也删掉（见 _discard）。

        每批向量化一次，与以前逐次写入时的请求相同：导入整段对话是一批，导入文件时每段连续的
        发言一批。已有的发言不再处理；遗忘删掉的发言也不会被写回来。

        meta：（会话，发言序号）→ 附加信息（SDK 的 metadata、过期时间与更正目标，导入的文件），
        与发言在同一个事务里写，只写给真正写进去的发言；不影响"轮"与向量化。
        """
        started = time.perf_counter()
        staged = _Staged(space_id)
        try:
            for pieces in batches:
                self._stage(staged, pieces, meta)
            with self.db.transaction():
                report = self._commit(staged, started)
                yield report
        except BaseException:
            self._discard(staged)
            raise
        if self.media is not None:  # 图片记录提交了：之后遗忘时按记录删原图
            for image in staged.stored:
                self.media.release(image)

    def _stage(
        self,
        staged: _Staged,
        pieces: Sequence[tuple[SessionIn, int]],
        meta: Mapping[tuple[str, int], TurnMeta | None] | None = None,
    ) -> None:
        """一批会话：建发言与"轮"、存原图，把这批里新的图片与"轮"向量化、写进向量库（不写数据库）。
        前面几批已经要写的不算新的，与以前逐批提交时相同（导入的文件里序号重复时）。"""
        space_id = staged.space_id
        units: list[MemoryUnit] = []
        media: list[MediaItem] = []
        for session, first_idx in pieces:
            turns = build_turns(space_id, session, first_idx)
            if self.media is not None:
                media.extend(self._attach_images(staged, session, turns))
            extra: dict[str, TurnMeta] = {}
            if meta:
                for t in turns:
                    info = meta.get((t.session_id, t.idx))
                    if info is not None:
                        extra[t.id] = info
            staged.pieces.append((session, turns, extra))
            units.extend(build_rounds(turns))

        if media:  # 新图片：先写图片向量，再登记图片记录（数据库里有就说明向量已写入）
            known = {m.id for m in self.db.get_media(space_id, ids=[m.id for m in media])}
            known |= {m.id for m in staged.media}
            new_media = [m for m in media if m.id not in known]
            if new_media and self.image_embedder is not None:
                with span("embed_images", images=len(new_media)):
                    vectors = self.image_embedder.embed_images([Path(m.path) for m in new_media])
                staged.media.extend(new_media)
                self.vectors.upsert_images(
                    space_id,
                    [m.id for m in new_media],
                    vectors,
                    [{"turn_id": m.source_turn_id, "ts": to_epoch(m.ts)} for m in new_media],
                )
            else:
                staged.media.extend(new_media)

        missing = set(self.db.missing_unit_ids(space_id, [u.id for u in units]))
        missing -= {u.id for u in staged.units}
        new_units = [u for u in units if u.id in missing]
        if new_units:
            embedded = self._embed(new_units)
            staged.units.extend(new_units)
            staged.tokens += embedded.tokens
            self._upsert(space_id, new_units, embedded)

    def _commit(self, staged: _Staged, started: float) -> IngestReport:
        """在事务里写会话、发言、图片记录与"轮"（向量都已写入）。"""
        space_id = staged.space_id
        self.db.ensure_space(space_id)
        turns_added = 0
        for session, turns, extra in staged.pieces:
            self.db.add_session(space_id, session.session_id, session.started_at)
            turns_added += self.db.add_turns(turns, extra)
        images_added = self.db.add_media(staged.media) if staged.media else 0
        if staged.units:
            self.db.add_units(staged.units)
        return IngestReport(
            space_id=space_id,
            sessions=len(staged.pieces),
            turns_added=turns_added,
            units_added=len(staged.units),
            embed_tokens=staged.tokens,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            images_added=images_added,
        )

    def _discard(self, staged: _Staged) -> None:
        """写入失败：删掉这次写进向量库、数据库里却没有记录的点。先查数据库："数据库里有记录，
        向量就一定在"这条不变式要保住（同一会话的写入由调用方串行，例如 SDK 的会话锁）。删不掉
        （向量库连不上）时记一条警告：取不到记录的点不会被检索返回，重试时同样的 ID 覆盖它们。

        这次存进媒体库的原图，没有图片记录引用、也没有别的写入正要用时删掉（与遗忘相同的规则，
        MediaStore.delete_unused）：原图按内容哈希存放，别的记录、别的空间可能也在用。以前一律
        不删：没有任何记录指向它，遗忘与删除空间都找不到它，例如向量接口不可用时发来的证件照片
        一直留在磁盘上。"""
        space_id = staged.space_id
        try:
            if staged.units:
                ids = [u.id for u in staged.units]
                self.vectors.delete(space_id, self.db.missing_unit_ids(space_id, ids))
            if staged.media and self.image_embedder is not None:
                kept = {m.id for m in self.db.get_media(space_id, ids=[m.id for m in staged.media])}
                self.vectors.delete_images(
                    space_id, [m.id for m in staged.media if m.id not in kept]
                )
        except Exception:
            logger.warning(
                "在线写入失败后没能删掉这次写进向量库的点（重试时会覆盖）", exc_info=True
            )
        if self.media is not None:
            for image in staged.stored:
                try:
                    self.media.release(image, in_use=self.db.media_sha_in_use)
                except Exception:
                    logger.warning("在线写入失败后没能删掉存进媒体库的原图", exc_info=True)

    def _attach_images(
        self, staged: _Staged, session: SessionIn, turns: Sequence[Turn]
    ) -> list[MediaItem]:
        """把发言里的图片存进媒体库（记在 staged.stored 上，失败时 _discard 据此删掉），生成图片
        记录，并把图片 ID 记在发言上。"""
        assert self.media is not None
        items: list[MediaItem] = []
        for turn, turn_in in zip(turns, session.turns, strict=True):
            for i, path in enumerate(turn_in.images):
                stored = self.media.put(Path(path))
                staged.stored.append(stored)
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

    def reindex_sessions(self, space_id: str, session_ids: Sequence[str]) -> int:
        """按数据库里现存的轮次重建这些会话的"轮"（遗忘部分轮次之后使用）。

        调用方需先删除这些会话原有的"轮"；返回新写入的单元数。
        """
        units: list[MemoryUnit] = []
        for session_id in session_ids:
            units.extend(build_rounds(self.db.get_turns(space_id, session_id)))
        missing = set(self.db.missing_unit_ids(space_id, [u.id for u in units]))
        new_units = [u for u in units if u.id in missing]
        if new_units:
            self._upsert(space_id, new_units, self._embed(new_units))
            # 先写向量、后写数据库：数据库里有这条记录，就说明向量一定已经写入；
            # 中途失败时重跑会补齐（向量化命中缓存，不会重复计费）。
            self.db.add_units(new_units)
        return len(new_units)

    def replace_rounds(self, space_id: str, session_id: str, stale: Sequence[str]) -> int:
        """把会话里的这些"轮"换成按现存发言重建的（例如同一轮里工具调用之后的回复并进这一轮），
        返回新写入的单元数。先向量化、写向量，再在一个事务里删掉旧的、写进新的：向量化或写向量
        出错时数据库里什么都不变。以前先删旧的再重建，出错时这些发言一个"轮"都没有了。
        不再用的旧向量最后删，删不掉时只记警告（取不到记录的点不会被检索返回）。"""
        rounds = build_rounds(self.db.get_turns(space_id, session_id))
        gone = set(stale)
        missing = set(self.db.missing_unit_ids(space_id, [u.id for u in rounds])) | gone
        fresh = [u for u in rounds if u.id in missing]
        if fresh:
            self._upsert(space_id, fresh, self._embed(fresh))
        with self.db.transaction():
            self.db.delete_units(space_id, sorted(gone))
            if fresh:
                self.db.add_units(fresh)
        try:
            self.vectors.delete(space_id, sorted(gone - {u.id for u in fresh}))
        except Exception:
            logger.warning('换掉的"轮"的旧向量没能删掉', exc_info=True)
        return len(fresh)

    def _embed(self, units: Sequence[MemoryUnit]) -> EmbeddingBatch:
        with span("embed", texts=len(units)):
            return self.embedder.embed([u.text for u in units], "document")

    def _upsert(self, space_id: str, units: Sequence[MemoryUnit], batch: EmbeddingBatch) -> None:
        self.vectors.upsert(
            space_id,
            [u.id for u in units],
            batch.dense,
            batch.sparse,
            [_payload(u) for u in units],
        )


def _payload(unit: MemoryUnit) -> dict[str, Any]:
    return {"kind": unit.kind, "session_id": unit.session_id, "ts": to_epoch(unit.ts)}
