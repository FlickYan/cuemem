"""向量索引（Qdrant 本地模式）：每个记忆空间一个集合，同时存 dense 与 sparse 向量。

本地模式不需要启动服务，适合笔记本；接口不变即可切换到 Qdrant 服务端（D-005）。
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from cuemem.core.ids import point_uuid, stable_id
from cuemem.embed.base import FloatArray, SparseVector
from cuemem.errors import StorageLockedError

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
    def __init__(self, path: Path | None, dim: int, *, url: str | None = None) -> None:
        """url 非空时连接 Qdrant 服务端；否则用本地模式（path 为空则只放在内存里）。

        owned_units：空间 → 数据库里这个空间的几个记忆单元 ID（MemorySystem 设为
        MemoryDB.some_unit_ids），用来确认旧名字的集合是不是这个空间的（见 collection_name）；
        没设时（单独用的向量库，没有数据库可以核对）旧名字的集合照旧用。"""
        self.dim = dim
        self.remote = url is not None
        self.owned_units: Callable[[str], Sequence[str]] | None = None
        self._names: dict[str, str] = {}  # 空间 → 集合名（解析一次）
        if url is not None:
            self._client = QdrantClient(url=url)
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
                self._client.create_collection(
                    name,
                    vectors_config={
                        DENSE: models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
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
            self._client.upsert(self.collection_name(space_id), points=points)

    def search_dense(
        self,
        space_id: str,
        vector: FloatArray,
        k: int,
        *,
        kinds: Sequence[str] | None = None,
        ts_range: tuple[float | None, float | None] | None = None,
    ) -> list[VectorHit]:
        return self._query(space_id, vector.tolist(), DENSE, k, _build_filter(kinds, ts_range))

    def search_sparse(
        self,
        space_id: str,
        vector: SparseVector,
        k: int,
        *,
        kinds: Sequence[str] | None = None,
        ts_range: tuple[float | None, float | None] | None = None,
    ) -> list[VectorHit]:
        if len(vector) == 0:
            return []
        query = models.SparseVector(indices=vector.indices, values=vector.values)
        return self._query(space_id, query, SPARSE, k, _build_filter(kinds, ts_range))

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
            return int(self._client.count(name, count_filter=_build_filter(kinds, None)).count)

    def delete(self, space_id: str, unit_ids: Sequence[str]) -> None:
        """按记忆单元 ID 删除向量点（遗忘、重建派生记忆时使用）。"""
        name = self.collection_name(space_id)
        if not unit_ids:
            return
        with self._lock:
            if not self._client.collection_exists(name):
                return
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
                self._client.create_collection(
                    name,
                    vectors_config={
                        DENSE: models.VectorParams(
                            size=int(vectors.shape[1]), distance=models.Distance.COSINE
                        )
                    },
                )
            self._client.upsert(name, points=points)

    def search_images(self, space_id: str, vector: FloatArray, k: int) -> list[VectorHit]:
        """以文搜图 / 以图搜图：返回（图片 ID，余弦相似度）。"""
        name = self._image_collection(space_id)
        with self._lock:
            if not self._client.collection_exists(name):
                return []
            response = self._client.query_points(
                name, query=vector.tolist(), using=DENSE, limit=k, with_payload=["media_id"]
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
            self._client.delete(
                name,
                points_selector=models.PointIdsList(points=[point_uuid(m) for m in media_ids]),
            )

    def _image_collection(self, space_id: str) -> str:
        return f"{self.collection_name(space_id)}{IMAGE_SUFFIX}"

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
        """列出空间里的全部记忆单元 ID（测试与一致性检查用）。"""
        name = self.collection_name(space_id)
        out: list[str] = []
        with self._lock:
            if not self._client.collection_exists(name):
                return out
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
                    with target._lock:
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


def _safe(space_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", space_id)[:40]


def _build_filter(
    kinds: Sequence[str] | None, ts_range: tuple[float | None, float | None] | None
) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if kinds:
        conditions.append(models.FieldCondition(key="kind", match=models.MatchAny(any=list(kinds))))
    if ts_range is not None:
        low, high = ts_range
        conditions.append(models.FieldCondition(key="ts", range=models.Range(gte=low, lte=high)))
    return models.Filter(must=conditions) if conditions else None
