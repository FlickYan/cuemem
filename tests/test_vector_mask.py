"""本地模式向量库按 kind 过滤的快速掩码（store/vector.py 的 _fast_mask）：结果与 qdrant-client
的逐点过滤逐位相同，只换掉这个 VectorStore 自己的集合。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.local import local_collection
from qdrant_client.local.payload_filters import calculate_payload_mask

from cuemem.embed import SparseVector
from cuemem.store import VectorStore
from cuemem.store import vector as vector_module

KINDS = ["round", "fact", "summary", "entity", "image", "image_iso"]


def _store(tmp_path: Path, *, n: int = 60) -> VectorStore:
    """每种 kind 都有、删掉了几个点、有的点没有稀疏向量的本地模式向量库。"""
    store = VectorStore(tmp_path / "qdrant", dim=4)
    rng = np.random.default_rng(0)
    ids = [f"u{i}" for i in range(n)]
    dense = rng.random((n, 4)).astype(np.float32)
    sparse = [
        SparseVector(indices=[i % 7, 7 + i % 5], values=[1.0, 0.5])
        if i % 4
        else SparseVector(indices=[], values=[])
        for i in range(n)
    ]
    payloads = [{"kind": KINDS[i % len(KINDS)], "ts": float(i)} for i in range(n)]
    store.upsert("s", ids, dense, sparse, payloads)
    store.delete("s", ids[::7])
    return store


def _collection(store: VectorStore) -> Any:
    return store._client._client.collections[store.collection_name("s")]


def _kind_filter(kinds: list[str]) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="kind", match=models.MatchAny(any=kinds))])


def _upstream(coll: Any, payload_filter: Any, vector_name: str | None) -> Any:
    mask = calculate_payload_mask(
        coll.payload, payload_filter, coll.ids_inv, coll.deleted_per_vector
    )
    mask = mask & ~coll.deleted
    if vector_name is not None:
        mask = mask & ~coll.deleted_per_vector[vector_name]
    return mask


@pytest.fixture
def counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """数 qdrant-client 的逐点过滤被调用了几次。"""
    calls: list[int] = []
    upstream = local_collection.calculate_payload_mask

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return upstream(*args, **kwargs)

    monkeypatch.setattr(local_collection, "calculate_payload_mask", counting)
    return calls


def test_kind_filter_skips_the_per_point_python_filter(tmp_path: Path, counted: list[int]) -> None:
    """只按 kind 过滤的检索不再对每个点跑一遍 check_filter（第五轮体检：2 万个点每次检索 0.2 秒
    以上，L1 每种 kind 一次，全程拿着向量库的全局锁），结果不变。"""
    store = _store(tmp_path)
    query = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float32)
    hits = store.search_dense("s", query, 50, kinds=["fact"])
    sparse_hits = store.search_sparse(
        "s", SparseVector(indices=[1, 8], values=[1.0, 1.0]), 50, kinds=["round", "fact"]
    )
    assert store.count("s", kinds=["summary"]) == 8
    assert counted == []
    assert hits and {int(h.unit_id[1:]) % len(KINDS) for h in hits} == {1}
    assert all(int(h.unit_id[1:]) % 7 for h in hits)  # 删掉的点不在
    assert sparse_hits and {int(h.unit_id[1:]) % len(KINDS) for h in sparse_hits} <= {0, 1}
    assert all(int(h.unit_id[1:]) % 4 for h in sparse_hits)  # 没有稀疏向量的点不在

    # 与上游的逐点过滤比：同一个集合、同一个查询，结果（顺序与分数）一模一样
    coll = _collection(store)
    fast = coll._payload_and_non_deleted_mask
    coll._payload_and_non_deleted_mask = fast.args[1]  # functools.partial(_fast_mask, coll, 上游)
    upstream_hits = store.search_dense("s", query, 50, kinds=["fact"])
    coll._payload_and_non_deleted_mask = fast
    assert [(h.unit_id, h.score) for h in hits] == [(h.unit_id, h.score) for h in upstream_hits]
    store.close()


@pytest.mark.parametrize("vector_name", [None, "dense", "sparse"])
def test_fast_mask_equals_upstream(tmp_path: Path, vector_name: str | None) -> None:
    store = _store(tmp_path)
    store.search_dense("s", np.ones(4, dtype=np.float32), 1)  # 换上快速掩码
    coll = _collection(store)
    upstream = coll._payload_and_non_deleted_mask.args[1]
    for kinds in [[k] for k in KINDS] + [["round", "fact"], ["nope"], []]:
        payload_filter = _kind_filter(kinds)
        expected = _upstream(coll, payload_filter, vector_name)
        got = vector_module._fast_mask(coll, upstream, payload_filter, vector_name)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert np.array_equal(got, expected), kinds
    store.close()


def test_other_filters_and_list_kinds_go_to_upstream(tmp_path: Path, counted: list[int]) -> None:
    """带 ts 范围的过滤、或者有点的 kind 不是字符串（上游对列表逐个元素匹配）时照旧交给上游。"""
    store = _store(tmp_path)
    query = np.array([1.0, 0.5, 0.2, 0.1], dtype=np.float32)
    in_range = store.search_dense("s", query, 50, kinds=["fact"], ts_range=(10.0, 40.0))
    assert len(counted) == 1
    assert in_range and all(10 <= int(h.unit_id[1:]) <= 40 for h in in_range)

    store.upsert(
        "s",
        ["listed"],
        query.reshape(1, 4),
        None,
        [{"kind": ["fact"], "ts": 1.0}],  # CueMem 自己不写这种 payload
    )
    coll = _collection(store)
    upstream = coll._payload_and_non_deleted_mask.args[1]
    payload_filter = _kind_filter(["fact"])
    before = len(counted)
    got = vector_module._fast_mask(coll, upstream, payload_filter, "dense")
    assert len(counted) == before + 1
    assert np.array_equal(got, _upstream(coll, payload_filter, "dense"))
    assert "listed" in [h.unit_id for h in store.search_dense("s", query, 50, kinds=["fact"])]
    store.close()


def test_only_this_stores_collections_are_patched(tmp_path: Path, counted: list[int]) -> None:
    """不改 LocalCollection 类：同一进程里别的 QdrantClient(path=...)（例如 M0 基线的 mem0）照旧
    用上游的逐点过滤。delete_space 之后重建的集合也换上快速掩码。"""
    store = _store(tmp_path)
    store.search_dense("s", np.ones(4, dtype=np.float32), 1, kinds=["fact"])
    assert "_payload_and_non_deleted_mask" in vars(_collection(store))
    assert not hasattr(local_collection.LocalCollection, "_cuemem_fast_mask")

    other = QdrantClient(path=str(tmp_path / "other"))
    other.create_collection(
        "x", vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.DOT)}
    )
    other.upsert(
        "x",
        points=[models.PointStruct(id=1, vector={"dense": [1, 0, 0, 0]}, payload={"kind": "fact"})],
    )
    found = other.query_points(
        "x", query=[1, 0, 0, 0], using="dense", query_filter=_kind_filter(["fact"])
    )
    assert [p.id for p in found.points] == [1]
    assert counted == [1]
    assert "_payload_and_non_deleted_mask" not in vars(other._client.collections["x"])
    other.close()

    store.delete_space("s")
    store.upsert("s", ["a"], np.ones((1, 4), dtype=np.float32), None, [{"kind": "fact", "ts": 0.0}])
    assert [h.unit_id for h in store.search_dense("s", np.ones(4), 5, kinds=["fact"])] == ["a"]
    assert counted == [1]
    store.close()
