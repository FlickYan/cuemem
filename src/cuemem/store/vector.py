"""向量索引（Qdrant 本地模式）：每个记忆空间一个集合，同时存 dense 与 sparse 向量。

本地模式不需要启动服务，适合笔记本；接口不变即可切换到 Qdrant 服务端（D-005）。
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from cuemem.core.ids import point_uuid, stable_id
from cuemem.embed.base import FloatArray, SparseVector

DENSE = "dense"
SPARSE = "sparse"


@dataclass(frozen=True)
class VectorHit:
    unit_id: str
    score: float


class VectorStore:
    def __init__(self, path: Path | None, dim: int, *, url: str | None = None) -> None:
        """url 非空时连接 Qdrant 服务端；否则用本地模式（path 为空则只放在内存里）。"""
        self.dim = dim
        if url is not None:
            self._client = QdrantClient(url=url)
        elif path is None:
            self._client = QdrantClient(location=":memory:")
        else:
            path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(path))
        self._lock = threading.RLock()  # 本地模式不是线程安全的，统一加锁

    @staticmethod
    def collection_name(space_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", space_id)[:40]
        return f"space_{safe}_{stable_id(space_id, length=8)}"

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

    def delete_space(self, space_id: str) -> None:
        name = self.collection_name(space_id)
        with self._lock:
            if self._client.collection_exists(name):
                self._client.delete_collection(name)

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
