"""向量索引（Qdrant 本地模式）：每个记忆空间一个集合，同时存 dense 与 sparse 向量。

本地模式不需要启动服务，适合笔记本；接口不变即可切换到 Qdrant 服务端（D-005）。本地模式每个集合
一直开着一个 SQLite 文件（有图片的空间另有一个图片集合），启动时全部打开、全部向量读进内存：用户
多的服务改用服务端（打不开时的说明见 _open_failed）。
"""

from __future__ import annotations

import errno
import functools
import gc
import json
import os
import re
import sqlite3
import threading
import traceback
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from cuemem.config import cli_command
from cuemem.core.ids import point_uuid, stable_id
from cuemem.embed.base import FloatArray, SparseVector
from cuemem.errors import StorageLockedError, VectorStoreOpenError

DENSE = "dense"
SPARSE = "sparse"
IMAGE_SUFFIX = "_img"  # 每个空间的图片向量放在单独的集合里（维度不同）
# 检索时用来过滤的 payload 字段（_build_filter）
PAYLOAD_INDEXES = {"kind": models.PayloadSchemaType.KEYWORD, "ts": models.PayloadSchemaType.FLOAT}


@dataclass(frozen=True)
class VectorHit:
    unit_id: str
    score: float


class VectorStore:
    def __init__(
        self, path: Path | None, dim: int, *, url: str | None = None, api_key: str | None = None
    ) -> None:
        """url 非空时连接 Qdrant 服务端；否则用本地模式（path 为空则只放在内存里）。

        api_key：服务端的 API Key（放在 api-key 请求头里，本地模式不用）。url 要写明 http:// 或
        https://：不写时 qdrant-client 在有 Key 时改用 https。经 http:// 发 Key 时 qdrant-client
        每个客户端警告一次 "Api key is used with an insecure connection."（本机与 docker compose
        的内部网络上无妨）。

        owned_units：空间 → 数据库里这个空间的几个记忆单元 ID（MemorySystem 设为
        MemoryDB.some_unit_ids），用来确认旧名字的集合是不是这个空间的（见 collection_name）；
        没设时（单独用的向量库，没有数据库可以核对）旧名字的集合照旧用。"""
        self.dim = dim
        self.remote = url is not None
        self.owned_units: Callable[[str], Sequence[str]] | None = None
        self._names: dict[str, str] = {}  # 空间 → 集合名（解析一次）
        self._path = path if url is None else None  # 本地模式的目录（出错时的说明用）
        if url is not None:
            self._client = QdrantClient(url=url, api_key=api_key)
        elif path is None:
            self._client = QdrantClient(location=":memory:")
        else:
            path.mkdir(parents=True, exist_ok=True)
            try:
                self._client = QdrantClient(path=str(path))
            except RuntimeError as exc:
                if "already accessed" not in str(exc):
                    raise
                raise StorageLockedError(
                    f"向量库 {path} 正被另一个进程使用"
                    "（Qdrant 本地模式同一时间只允许一个进程打开）。"
                    "先停掉那个进程（比如正在运行的 cuemem serve），或者改用 Qdrant 服务端："
                    "设置 CUEMEM_VECTOR_STORE__URL，见 docs/design.md 的部署一节"
                ) from exc
            except (sqlite3.OperationalError, OSError) as exc:
                # 以前原样抛出 sqlite3.OperationalError（不是 CueMemError）：cuemem serve、proxy
                # 与 migrate-vectors 都打印调用栈退出，看不出是打开的文件太多
                raise _open_failed(path, exc) from exc
            except (ValueError, KeyError, TypeError) as exc:
                # meta.json 读不出（多半是空的：qdrant-client 1.19 每次建、删集合都先清空它再写，
                # 正好在这时被杀掉）。以前原样抛出 JSONDecodeError，cuemem export 也用不了
                raise _meta_unreadable(path, exc) from exc
        # 统一加锁：本地模式不是线程安全的；服务端模式下同步客户端被多个线程同时使用也会出错
        # （N11 压测里出现 "Bad file descriptor"）
        self._lock = threading.RLock()

    def collection_name(self, space_id: str) -> str:
        """这个空间的集合名：前 40 个字符（不是字母数字的换成 "_"）加上空间 ID 的 128 位摘要。

        以前的名字（legacy_name）只带 32 位摘要：前 40 个字符相同、8 位十六进制又撞上的两个空间
        落进同一个集合，检索不按空间过滤、delete_space 删掉整个集合。限定作用域的令牌可以随便起
        agent_id，离线算约 2^32 次哈希就能撞上别人（例如 "shop/<用户的 UUID>"）的集合：删掉它，
        或者往它的检索结果里塞东西。

        升级之前建的旧名字集合（包括评测的构建）照旧用，但只给能证明是自己的空间：数据库里这个
        空间的记忆单元在这个集合里（只读，不往库里写东西）。别的空间——包括为了撞上它而起的——
        用新名字。新名字的集合已经有了时总用它。"""
        with self._lock:
            name = self._names.get(space_id)
            if name is None:
                name = self._names[space_id] = self._resolve(space_id)
            return name

    @staticmethod
    def legacy_name(space_id: str) -> str:
        """第二轮复审之前的集合名（32 位摘要），见 collection_name。"""
        return f"space_{_safe(space_id)}_{stable_id(space_id, length=8)}"

    def _resolve(self, space_id: str) -> str:
        name = f"space_{_safe(space_id)}_{stable_id(space_id, length=32)}"
        legacy = self.legacy_name(space_id)
        if self._client.collection_exists(name) or not self._client.collection_exists(legacy):
            return name
        if self.owned_units is None:
            return legacy
        ids = [point_uuid(u) for u in self.owned_units(space_id)]
        mine = ids and self._client.retrieve(legacy, ids=ids, with_payload=False)
        return legacy if mine else name

    def ensure_space(self, space_id: str) -> None:
        name = self.collection_name(space_id)
        with self._lock:
            if not self._client.collection_exists(name):
                with self._creating():
                    self._client.create_collection(
                        name,
                        vectors_config={
                            DENSE: models.VectorParams(
                                size=self.dim, distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={SPARSE: models.SparseVectorParams()},
                    )
                if self.remote:  # 服务端按 payload 建索引，过滤检索不必扫全集合（本地模式不支持）
                    for field, schema in PAYLOAD_INDEXES.items():
                        self._client.create_payload_index(name, field, field_schema=schema)

    def upsert(
        self,
        space_id: str,
        unit_ids: Sequence[str],
        dense: FloatArray,
        sparse: Sequence[SparseVector] | None,
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        points = []
        for row, unit_id in enumerate(unit_ids):
            vector: dict[str, Any] = {DENSE: dense[row].tolist()}
            if sparse is not None and len(sparse[row]) > 0:
                vector[SPARSE] = models.SparseVector(
                    indices=sparse[row].indices, values=sparse[row].values
                )
            points.append(
                models.PointStruct(
                    id=point_uuid(unit_id),
                    vector=vector,
                    payload={"unit_id": unit_id, **payloads[row]},
                )
            )
        with self._lock:
            self.ensure_space(space_id)
            name = self.collection_name(space_id)
            with self._batched(name):
                self._client.upsert(name, points=points)

    def search_dense(
        self,
        space_id: str,
        vector: FloatArray,
        k: int,
        *,
        kinds: Sequence[str] | None = None,
        ts_range: tuple[float | None, float | None] | None = None,
        hidden_ids: Collection[str] = (),
    ) -> list[VectorHit]:
        """hidden_ids：不参与检索的单元（过期的记忆，见 MemorySystem.retrieve 的 hide_expired）：
        在检索里就排除，前 k 条都是别的单元。空的时过滤条件与以前完全相同。"""
        query_filter = _build_filter(kinds, ts_range, hidden_ids)
        return self._query(space_id, vector.tolist(), DENSE, k, query_filter)

    def search_sparse(
        self,
        space_id: str,
        vector: SparseVector,
        k: int,
        *,
        kinds: Sequence[str] | None = None,
        ts_range: tuple[float | None, float | None] | None = None,
        hidden_ids: Collection[str] = (),
    ) -> list[VectorHit]:
        """hidden_ids 同 search_dense。"""
        if len(vector) == 0:
            return []
        query = models.SparseVector(indices=vector.indices, values=vector.values)
        query_filter = _build_filter(kinds, ts_range, hidden_ids)
        return self._query(space_id, query, SPARSE, k, query_filter)

    def indexed(self, space_id: str) -> bool:
        """服务端是否已建完索引（大批写入后后台建 HNSW，期间检索变慢）；本地模式总是 True。"""
        name = self.collection_name(space_id)
        with self._lock:
            if not self.remote or not self._client.collection_exists(name):
                return True
            info = self._client.get_collection(name)
        return info.status == models.CollectionStatus.GREEN

    def count(self, space_id: str, *, kinds: Sequence[str] | None = None) -> int:
        name = self.collection_name(space_id)
        with self._lock:
            if not self._client.collection_exists(name):
                return 0
            self._fast_kind_mask(name)
            return int(self._client.count(name, count_filter=_build_filter(kinds, None)).count)

    def delete(self, space_id: str, unit_ids: Sequence[str]) -> None:
        """按记忆单元 ID 删除向量点（遗忘、重建派生记忆时使用）。"""
        name = self.collection_name(space_id)
        if not unit_ids:
            return
        with self._lock:
            if not self._client.collection_exists(name):
                return
            with self._batched(name):
                self._client.delete(
                    name,
                    points_selector=models.PointIdsList(points=[point_uuid(u) for u in unit_ids]),
                )

    def set_payload(self, space_id: str, unit_ids: Sequence[str], payload: dict[str, Any]) -> None:
        """更新若干点的 payload，例如事实被取代后把 current 改为 False，供过滤检索使用。"""
        name = self.collection_name(space_id)
        if not unit_ids:
            return
        with self._lock:
            if not self._client.collection_exists(name):
                return
            with self._batched(name):
                self._client.set_payload(
                    name, payload=payload, points=[point_uuid(u) for u in unit_ids]
                )

    # ---------- 图片向量（N6）：每个空间另建一个集合，维度与文本向量不同 ----------

    def upsert_images(
        self,
        space_id: str,
        media_ids: Sequence[str],
        vectors: FloatArray,
        payloads: Sequence[dict[str, Any]],
    ) -> None:
        if not media_ids:
            return
        name = self._image_collection(space_id)
        points = [
            models.PointStruct(
                id=point_uuid(media_id),
                vector={DENSE: vectors[row].tolist()},
                payload={"media_id": media_id, **payloads[row]},
            )
            for row, media_id in enumerate(media_ids)
        ]
        with self._lock:
            if not self._client.collection_exists(name):
                with self._creating():
                    self._client.create_collection(
                        name,
                        vectors_config={
                            DENSE: models.VectorParams(
                                size=int(vectors.shape[1]), distance=models.Distance.COSINE
                            )
                        },
                    )
            with self._batched(name):
                self._client.upsert(name, points=points)

    def search_images(
        self, space_id: str, vector: FloatArray, k: int, *, hidden_ids: Collection[str] = ()
    ) -> list[VectorHit]:
        """以文搜图 / 以图搜图：返回（图片 ID，余弦相似度）。hidden_ids：不参与检索的图片
        （过期的，同 search_dense）；空的时请求与以前相同（不带过滤条件）。"""
        name = self._image_collection(space_id)
        query_filter = _build_filter(None, None, hidden_ids)
        with self._lock:
            if not self._client.collection_exists(name):
                return []
            self._fast_kind_mask(name)
            response = self._client.query_points(
                name,
                query=vector.tolist(),
                using=DENSE,
                limit=k,
                query_filter=query_filter,
                with_payload=["media_id"],
            )
        return [
            VectorHit(unit_id=str(point.payload["media_id"]), score=float(point.score))
            for point in response.points
            if point.payload
        ]

    def delete_images(self, space_id: str, media_ids: Sequence[str]) -> None:
        name = self._image_collection(space_id)
        if not media_ids:
            return
        with self._lock:
            if not self._client.collection_exists(name):
                return
            with self._batched(name):
                self._client.delete(
                    name,
                    points_selector=models.PointIdsList(points=[point_uuid(m) for m in media_ids]),
                )

    def _image_collection(self, space_id: str) -> str:
        return f"{self.collection_name(space_id)}{IMAGE_SUFFIX}"

    @contextmanager
    def _batched(self, name: str) -> Iterator[None]:
        """一次写入（upsert、delete、set_payload）只提交一次本地模式的 SQLite（调用方持有
        self._lock）。qdrant-client 的本地模式每写一个点提交一次（CollectionPersistence.persist
        与 delete），默认的回滚日志加 synchronous=FULL，每个点一次 fsync：遗忘一次删掉 1 万个
        派生向量，全程拿着 self._lock，别的用户的检索最多等了 13.6 秒（第五轮体检）。这里把
        这个集合的连接换成 commit 什么都不做的替身，做完再提交一次；点、顺序与内容都不变。

        出错时也提交：内存里的集合已经改了，写下已经写的，磁盘与内存才一致（与以前每个点一提交
        相同）。进程中途被杀时这一次写入在磁盘上整个没有（以前留下前面一部分点）：遗忘先删向量、
        后删记录，这等于什么都没删，正是遗忘当作安全的状态。

        用的是 qdrant-client 的内部属性（QdrantLocal.collections、LocalCollection.storage、
        CollectionPersistence.storage）：对不上时（服务端模式、只放在内存里、以后的 qdrant-client
        改了）照旧每个点提交一次。"""
        local = getattr(self._client, "_client", None) if not self.remote and self._path else None
        collection = getattr(local, "collections", {}).get(name) if local is not None else None
        persistence: Any = getattr(collection, "storage", None)
        real = getattr(persistence, "storage", None)
        if not isinstance(real, sqlite3.Connection):
            yield
            return
        persistence.storage = _NoCommit(real)
        try:
            yield
        finally:
            persistence.storage = real
            real.commit()

    def _fast_kind_mask(self, name: str) -> None:
        """给本地模式的这个集合换上按 kind 过滤的快速掩码（调用方持有 self._lock，见 _fast_mask）。

        只改这一个集合对象（不改 LocalCollection 类：同一进程里别的 QdrantClient(path=...)，
        例如 M0 基线的 mem0，照旧）；每次检索前看一眼，delete_space 之后重建的集合与启动时打开的
        集合都会换上。用的是 qdrant-client 的内部属性：对不上时（服务端模式、以后的 qdrant-client
        改了）什么都不做，照旧用上游的逐点过滤。"""
        if self.remote:
            return
        local = getattr(self._client, "_client", None)
        collection = getattr(local, "collections", {}).get(name)
        if collection is None or getattr(collection, "_cuemem_fast_mask", False):
            return
        upstream = getattr(collection, "_payload_and_non_deleted_mask", None)
        if not (
            callable(upstream)
            and isinstance(getattr(collection, "payload", None), list)
            and isinstance(getattr(collection, "deleted", None), np.ndarray)
            and isinstance(getattr(collection, "deleted_per_vector", None), dict)
        ):
            return
        collection._payload_and_non_deleted_mask = functools.partial(
            _fast_mask, collection, upstream
        )
        collection._cuemem_fast_mask = True

    @contextmanager
    def _creating(self) -> Iterator[None]:
        """建集合（调用方持有 self._lock）。本地模式下建不了时（多半是打开的文件用光了：每个集合
        一个文件）抛出写明原因的 VectorStoreOpenError，写入、后台线程的出错与 HTTP 接口都看得到；
        以前是 sqlite3.OperationalError: unable to open database file。"""
        try:
            yield
        except (sqlite3.OperationalError, OSError) as exc:
            if self._path is None:
                raise
            collections = len(self._client.get_collections().collections)  # 在内存里，不用文件
            raise _open_failed(self._path, exc, collections) from exc

    def dense_vectors(self, space_id: str, unit_ids: Sequence[str]) -> dict[str, FloatArray]:
        """按记忆单元 ID 取回已存的 dense 向量（L2 的实体图用来给语义相近的事实连边）。"""
        name = self.collection_name(space_id)
        if not unit_ids:
            return {}
        with self._lock:
            if not self._client.collection_exists(name):
                return {}
            points = self._client.retrieve(
                name,
                ids=[point_uuid(u) for u in unit_ids],
                with_payload=["unit_id"],
                with_vectors=[DENSE],
            )
        out: dict[str, FloatArray] = {}
        for point in points:
            vector = point.vector.get(DENSE) if isinstance(point.vector, dict) else None
            if point.payload and vector is not None:
                out[str(point.payload["unit_id"])] = np.asarray(vector, dtype=np.float32)
        return out

    def unit_ids(self, space_id: str, *, kinds: Sequence[str] | None = None) -> list[str]:
        """列出空间里的全部记忆单元 ID（测试、一致性检查与 reindex 用）。"""
        name = self.collection_name(space_id)
        out: list[str] = []
        with self._lock:
            if not self._client.collection_exists(name):
                return out
            self._fast_kind_mask(name)
            offset = None
            while True:
                points, offset = self._client.scroll(
                    name,
                    scroll_filter=_build_filter(kinds, None),
                    limit=256,
                    offset=offset,
                    with_payload=["unit_id"],
                    with_vectors=False,
                )
                out.extend(str(p.payload["unit_id"]) for p in points if p.payload)
                if offset is None:
                    return out

    def media_ids(self, space_id: str) -> list[str]:
        """列出空间的图片集合里的全部图片 ID（一致性检查与 reindex 用）。"""
        name = self._image_collection(space_id)
        out: list[str] = []
        with self._lock:
            if not self._client.collection_exists(name):
                return out
            offset = None
            while True:
                points, offset = self._client.scroll(
                    name, limit=256, offset=offset, with_payload=["media_id"], with_vectors=False
                )
                out.extend(str(p.payload["media_id"]) for p in points if p.payload)
                if offset is None:
                    return out

    def copy_into(self, target: VectorStore, *, batch: int = 256) -> dict[str, int]:
        """把所有集合（文字与图片，向量与 payload 原样）复制到 target，返回各集合的点数。

        用于从本地模式迁到服务端：数据库里已有的记忆单元不会被重新写入向量库（写入是幂等的），
        所以切换前要先把向量搬过去。target 里已有的同名点会被覆盖。
        """
        counts: dict[str, int] = {}
        with target._lock:
            target._client.get_collections()  # 先确认连得上目标，免得复制到一半才失败
        with self._lock:
            names = [c.name for c in self._client.get_collections().collections]
        for name in names:
            with self._lock:
                params = self._client.get_collection(name).config.params
            with target._lock:
                if not target._client.collection_exists(name):
                    target._client.create_collection(
                        name,
                        vectors_config=params.vectors,
                        sparse_vectors_config=params.sparse_vectors,
                    )
                    if target.remote and not name.endswith(IMAGE_SUFFIX):
                        for field, schema in PAYLOAD_INDEXES.items():
                            target._client.create_payload_index(name, field, field_schema=schema)
            offset = None
            counts[name] = 0
            while True:
                with self._lock:
                    records, offset = self._client.scroll(
                        name, limit=batch, offset=offset, with_payload=True, with_vectors=True
                    )
                points = [
                    models.PointStruct(id=r.id, vector=r.vector or {}, payload=r.payload)
                    for r in records
                ]
                if points:
                    with target._lock, target._batched(name):
                        target._client.upsert(name, points=points)
                counts[name] += len(points)
                if offset is None:
                    break
        return counts

    def delete_space(self, space_id: str) -> None:
        """删掉这个空间的两个集合（文字与图片）。要在数据库删掉这个空间的记忆单元之前调用：旧名字
        的集合靠它们确认是这个空间的（见 collection_name）。"""
        with self._lock:
            for name in (self.collection_name(space_id), self._image_collection(space_id)):
                if self._client.collection_exists(name):
                    self._client.delete_collection(name)
            self._names.pop(space_id, None)

    def close(self) -> None:
        with self._lock:
            self._client.close()

    def _query(
        self,
        space_id: str,
        query: list[float] | models.SparseVector,
        using: str,
        k: int,
        query_filter: models.Filter | None,
    ) -> list[VectorHit]:
        name = self.collection_name(space_id)
        with self._lock:
            if not self._client.collection_exists(name):
                return []
            self._fast_kind_mask(name)
            response = self._client.query_points(
                name,
                query=query,
                using=using,
                limit=k,
                query_filter=query_filter,
                with_payload=["unit_id"],
            )
        return [
            VectorHit(unit_id=str(point.payload["unit_id"]), score=float(point.score))
            for point in response.points
            if point.payload
        ]


class _NoCommit:
    """本地模式集合的 SQLite 连接的替身：commit 什么都不做，其余原样交给连接（见 _batched）。"""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def commit(self) -> None:
        pass

    # 较新的 qdrant-client（1.20 起）在 persist 与 delete 里写 `with self.storage:`，用连接自己的
    # 上下文管理器提交或回滚；特殊方法不经过 __getattr__，要在这里写出来。进入与退出都不提交：
    # 和 commit 一样留给 _batched 最后提交一次（出错时也提交已经写下的，见 _batched）。
    def __enter__(self) -> _NoCommit:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def _kind_only(query_filter: Any) -> frozenset[str] | None:
    """过滤条件只有"kind 是其中之一"（_build_filter 只给 kinds 时的样子）时返回这几个 kind，
    别的过滤条件（带 ts 范围、should、must_not 等）返回 None。"""
    if not isinstance(query_filter, models.Filter):
        return None
    if query_filter.should or query_filter.must_not or query_filter.min_should:
        return None
    return _kinds_in(query_filter.must)


def _kind_and_hidden(query_filter: Any) -> tuple[frozenset[str] | None, list[str]] | None:
    """_build_filter 给了 hidden_ids 时的样子：must 只有"kind 是其中之一"或者没有，must_not
    只有一个 HasIdCondition。返回（这几个 kind，没有时 None；要排除的点 ID），别的样子返回
    None。"""
    if not isinstance(query_filter, models.Filter):
        return None
    if query_filter.should or query_filter.min_should:
        return None
    must_not = query_filter.must_not
    if not isinstance(must_not, list) or len(must_not) != 1:
        return None
    condition = must_not[0]
    if not isinstance(condition, models.HasIdCondition):
        return None
    if not all(isinstance(point, str) for point in condition.has_id):
        return None
    hidden = [str(point) for point in condition.has_id]
    if query_filter.must is None or query_filter.must == []:
        return None, hidden
    kinds = _kinds_in(query_filter.must)
    return (kinds, hidden) if kinds is not None else None


def _kinds_in(must: Any) -> frozenset[str] | None:
    """must 只有一条"kind 是其中之一"时返回这几个 kind，否则 None。"""
    conditions = must if isinstance(must, list) else [must]
    if len(conditions) != 1:
        return None
    condition = conditions[0]
    if not (
        isinstance(condition, models.FieldCondition)
        and condition.key == "kind"
        and isinstance(condition.match, models.MatchAny)
        and all(isinstance(kind, str) for kind in condition.match.any)
        and condition.range is None
        and condition.geo_bounding_box is None
        and condition.geo_radius is None
        and condition.geo_polygon is None
        and condition.values_count is None
        and condition.is_empty is None
        and condition.is_null is None
    ):
        return None
    return frozenset(str(kind) for kind in condition.match.any)


def _fast_mask(
    collection: Any,
    upstream: Callable[..., Any],
    payload_filter: models.Filter | None,
    vector_name: str | None = None,
) -> Any:
    """本地模式 LocalCollection._payload_and_non_deleted_mask 的替身，结果与上游逐位相同。

    上游对每个点在 Python 里跑一遍 check_filter（每个点约 10 微秒），L1 每种 kind 检索一次，
    全程拿着 VectorStore 的全局锁：2 万个点的空间每次检索单是过滤就要 0.2 秒以上（第五轮体检）。
    只按 kind 过滤、而且每个点的 kind 都是字符串时，这里直接比较字符串（与上游的 MatchAny 对
    字符串的判断相同）；别的过滤条件、或者有点的 kind 不是字符串（上游对列表逐个元素匹配）时
    照旧交给上游。

    再排除一批点（hidden_ids，过期的记忆：must_not 一个 HasIdCondition）时，上游每个点都把整个
    ID 列表重新组成元组、查一遍缓存，几千个过期的点就是几秒；这里按点 ID 直接找到下标、置为
    不要。存下的点 ID 是规范写法的 UUID 字符串（point_uuid），与上游的 normalize_point_id 相同。"""
    from qdrant_client.local.payload_filters import validate_filter

    hidden: list[str] = []
    wanted = _kind_only(payload_filter)
    if wanted is None:
        parsed = _kind_and_hidden(payload_filter)
        if parsed is None or not isinstance(getattr(collection, "ids", None), dict):
            return upstream(payload_filter, vector_name)
        wanted, hidden = parsed
    validate_filter(payload_filter)
    if wanted is None:
        mask = np.ones(len(collection.payload), dtype=bool)
    else:
        kinds = [payload.get("kind") for payload in collection.payload]
        if not all(isinstance(kind, str) for kind in kinds):
            return upstream(payload_filter, vector_name)
        mask = np.fromiter((kind in wanted for kind in kinds), dtype=bool, count=len(kinds))
    for point in hidden:
        index = collection.ids.get(point)
        if index is not None:
            mask[index] = False
    mask = mask & ~collection.deleted
    if vector_name is not None:
        mask = mask & ~collection.deleted_per_vector[vector_name]
    return mask


def _safe(space_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", space_id)[:40]


def _open_failed(
    path: Path, exc: Exception, collections: int | None = None
) -> VectorStoreOpenError:
    """本地模式的向量库打不开或建不了集合时的出错，写明原因与办法。collections：集合数（没给时
    读 meta.json，不打开向量库）。

    先放掉打开到一半的集合：它们各占着一个文件，引用它们的只有这个异常的调用栈，不放掉的话进程
    （例如接住了出错的 SDK 应用）的文件一直用光着，这里也读不了 meta.json。集合数超过打开文件数
    软上限的一半（cuemem doctor 标黄的条件），或者出错本身就是文件用光了，算作文件太多。"""
    traceback.clear_frames(exc.__traceback__)
    gc.collect()
    if collections is None:
        with suppress(OSError, ValueError, KeyError, TypeError):
            collections = len(json.loads((path / "meta.json").read_text("utf-8"))["collections"])
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        unlimited = resource.RLIM_INFINITY
    except ImportError:  # Windows
        soft = hard = unlimited = -1
    exhausted = isinstance(exc, OSError) and exc.errno in (errno.EMFILE, errno.ENFILE)
    crowded = collections is not None and soft != unlimited and collections > soft / 2
    if not (exhausted or crowded):
        return VectorStoreOpenError(
            f"本地模式的向量库 {path} 打不开（{exc}）：检查这个目录的权限与磁盘空间"
        )
    count = f"这里有 {collections} 个集合，" if collections is not None else ""
    limit = "不限" if soft == unlimited else str(soft)
    cap = "不限" if hard == unlimited else str(hard)
    migrate = cli_command("migrate-vectors --to URL")
    return VectorStoreOpenError(
        f"本地模式的向量库 {path} 打不开（{exc}）：本地模式每个集合（每个记忆空间一个，有图片的"
        f"空间两个）一直开着一个文件，{count}打开文件数的上限是 {limit}（硬上限 {cap}）。先调高"
        f" ulimit -n（不够时连硬上限一起调）让它打得开，再用 {migrate} 把向量复制到 Qdrant"
        " 服务端，并设置 CUEMEM_VECTOR_STORE__URL（见 docs/design.md 的部署一节）"
    )


def _meta_unreadable(path: Path, exc: Exception) -> VectorStoreOpenError:
    """本地模式的 meta.json 读不出时的出错：向量还在，写明怎么修、不要删 meta.json。先放掉打开到
    一半的集合（同 _open_failed）。"""
    traceback.clear_frames(exc.__traceback__)
    gc.collect()
    return VectorStoreOpenError(
        f"本地模式的向量库 {path / 'meta.json'} 读不出（{type(exc).__name__}: {exc}），向量还在"
        f" {path / 'collection'} 下面：运行 {cli_command('doctor --repair-vectors')} 按这些集合"
        "重写 meta.json。不要手动删掉 meta.json：Qdrant 会写一个空的，这些集合就再也找不到了"
        f"（{cli_command('export')} 不用向量库，照样能导出）"
    )


def _valid_collection(entry: Any) -> bool:
    """meta.json 里一个集合的配置 Qdrant 打开时认不认（QdrantLocal 载入时同样构造
    CreateCollection）。"""
    try:
        models.CreateCollection(**entry)
    except (ValueError, TypeError):  # pydantic 的 ValidationError 是 ValueError
        return False
    return True


def repair_local_meta(path: Path, *, text_dim: int, image_dim: int) -> dict[str, Any] | None:
    """按 collection/ 下的集合重写本地模式的 meta.json（cuemem doctor --repair-vectors）。

    只在 meta.json 读不出、不在了、或者漏了 collection/ 下的集合时动手，否则返回 None。直接删掉
    meta.json 不行：Qdrant 会写一个空的，旧集合没人认，下次写入在同名目录上重新建集合。另一个
    进程开着这个向量库时（.lock 被锁着）拒绝：StorageLockedError。

    meta.json 读得出、Qdrant 认的部分原样保留（某个集合的配置不合法时当作漏了它）；其余集合的
    配置照 VectorStore 建集合时的写：文字集合 dense（余弦）加 sparse，图片集合（名字以
    IMAGE_SUFFIX 结尾）只有 dense。dense 的维度从存着的一个点读（只读打开；不同向量模型的构建
    可以并存，不能用现在的配置），空集合用 text_dim 或 image_dim。原来的 meta.json 改名为
    meta.json.broken-<时间>，新文件先写临时文件、fsync 再换上，写完重新打开核对一遍。返回补上
    的文字、图片与空集合数、点数与备份的位置。"""
    import time

    import portalocker
    from qdrant_client._pydantic_compat import to_dict

    meta = path / "meta.json"
    root = path / "collection"
    names = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    try:
        kept = dict(json.loads(meta.read_text("utf-8"))["collections"])
    except FileNotFoundError:
        kept = {}
    except (ValueError, KeyError, TypeError):
        kept = None  # 读不出：整个重写
    # 读得出、某个集合的配置却不合法（Qdrant 打开时 ValidationError，VectorStore 报 meta.json
    # 读不出）：当作漏了，照存着的点重写（第五轮复审：以前原样保留、返回 None，说好的修复什么
    # 都不做，向量库一直打不开）
    invalid = [name for name, entry in (kept or {}).items() if not _valid_collection(entry)]
    if kept is not None:
        kept = {name: entry for name, entry in kept.items() if name not in invalid}
    if kept is not None and not invalid and all(name in kept for name in names):
        return None
    lock_path = path / ".lock"
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as lock:  # 与 qdrant-client 打开时加的是同一把锁
        flags = portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
        try:
            portalocker.lock(lock, flags)
        except portalocker.exceptions.LockException as exc:
            raise StorageLockedError(
                f"向量库 {path} 正被另一个进程使用：先停掉那个进程（比如正在运行的 cuemem serve）"
                "再修"
            ) from exc
        try:
            collections = dict(kept or {})
            report = {"text": 0, "image": 0, "empty": 0, "points": 0}
            for name in names:
                if name in collections:
                    continue
                image = name.endswith(IMAGE_SUFFIX)
                size, points = _stored_dim(root / name / "storage.sqlite")
                if size is None:
                    size = image_dim if image else text_dim
                    report["empty"] += 1
                report["image" if image else "text"] += 1
                report["points"] += points
                dense = models.VectorParams(size=size, distance=models.Distance.COSINE)
                config = models.CreateCollection(
                    vectors={DENSE: dense},
                    sparse_vectors=None if image else {SPARSE: models.SparseVectorParams()},
                    metadata=None,
                )
                collections[name] = to_dict(config)
            # 与 qdrant-client 1.19 的 QdrantLocal._save 写的一样
            text = json.dumps({"collections": collections, "aliases": {}})
            backup = None
            if meta.exists():
                backup = meta.with_name(f"meta.json.broken-{time.strftime('%Y%m%d-%H%M%S')}")
                meta.rename(backup)
            tmp = meta.with_name("meta.json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, meta)
            with suppress(OSError):  # 目录项也落盘（Windows 上打不开目录，跳过）
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        finally:
            portalocker.unlock(lock)
    client = QdrantClient(path=str(path))  # 核对：打得开、集合都在
    try:
        found = {c.name for c in client.get_collections().collections}
    finally:
        client.close()
    if found != set(collections):
        raise VectorStoreOpenError(
            f"重写 {meta} 之后集合对不上：{sorted(set(collections) ^ found)}"
        )
    return {**report, "backup": str(backup) if backup else None}


def _stored_dim(storage: Path) -> tuple[int | None, int]:
    """集合文件里一个点的 dense 维度（没有点时 None）与点数。只读打开；只读打不开时（被杀掉时留下
    了回滚日志，只读模式回滚不了）照常打开，Qdrant 下次打开时也是这样回滚的。"""
    if not storage.is_file():
        return None, 0
    try:
        conn = sqlite3.connect(f"{storage.resolve().as_uri()}?mode=ro", uri=True)
        try:
            row, count = _first_point(conn)
        finally:
            conn.close()
    except sqlite3.OperationalError:
        conn = sqlite3.connect(storage)
        try:
            row, count = _first_point(conn)
        finally:
            conn.close()
    if row is None:
        return None, 0
    import pickle

    point = pickle.loads(row)  # Qdrant 本地模式自己的格式（LocalCollection 也是这样读的）
    vector = point.vector.get(DENSE) if isinstance(point.vector, dict) else None
    if not isinstance(vector, list) or not vector:
        raise VectorStoreOpenError(f"{storage} 里的点没有 {DENSE} 向量，不是 CueMem 建的集合")
    return len(vector), count


def _first_point(conn: sqlite3.Connection) -> tuple[bytes | None, int]:
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'points'").fetchone():
        return None, 0
    count = int(conn.execute("SELECT COUNT(*) FROM points").fetchone()[0])
    row = conn.execute("SELECT point FROM points LIMIT 1").fetchone()
    return (bytes(row[0]) if row else None), count


def _build_filter(
    kinds: Sequence[str] | None,
    ts_range: tuple[float | None, float | None] | None,
    hidden_ids: Collection[str] = (),
) -> models.Filter | None:
    """hidden_ids 不为空时再加一条 must_not（这些单元的点 ID）；空的时与以前完全相同。"""
    conditions: list[models.Condition] = []
    if kinds:
        conditions.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))))
    if ts_range is not None:
        low, high = ts_range
        conditions.append(models.FieldCondition(key="ts", range=models.Range(gte=low, lte=high)))
    if not hidden_ids:
        return models.Filter(must=conditions) if conditions else None
    hidden: list[models.ExtendedPointId] = list(_hidden_points(frozenset(hidden_ids)))
    return models.Filter(must=conditions or None, must_not=[models.HasIdCondition(has_id=hidden)])


@functools.lru_cache(maxsize=8)
def _hidden_points(unit_ids: frozenset[str]) -> tuple[str, ...]:
    """单元 ID → 点 ID（排好序）。同一次检索的几路用同一批过期的记忆，只算一次。"""
    return tuple(sorted(point_uuid(u) for u in unit_ids))
