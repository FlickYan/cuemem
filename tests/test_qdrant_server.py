"""Qdrant 服务端模式（vector_store.url，docker-compose 推荐的部署）：只在服务端才走的分支——建集合
时的 payload 索引、indexed()、迁到服务端时给目标建索引——以及服务端上的过滤检索、删除空间与旧名字
集合的归属核对。别的测试都用本地模式，这些分支以前从没跑过。

设了 CUEMEM_TEST_QDRANT_URL 才跑（CI 的 qdrant-server 任务起一个要 Key 的 qdrant/qdrant:v1.19.1）：
    docker run -d -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.19.1
    CUEMEM_TEST_QDRANT_URL=http://localhost:6333 uv run pytest tests/test_qdrant_server.py
服务端设了 QDRANT__SERVICE__API_KEY 时把同一个 Key 设成 CUEMEM_TEST_QDRANT_API_KEY：每个连接都带上
它，另有一个测试确认不带 Key 的连接被拒绝（服务端真的在查 Key）。

不整份重跑别的测试文件：它们的空间名相同会互相撞上，还有只对本地模式成立的断言（存储锁、同分时
的先后）。每个测试用自己的空间名，结束时删掉名字里带它的集合。
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from cuemem.embed import SparseVector
from cuemem.embed.base import l2_normalize
from cuemem.store.vector import IMAGE_SUFFIX, PAYLOAD_INDEXES, VectorStore

URL = os.environ.get("CUEMEM_TEST_QDRANT_URL")
KEY = os.environ.get("CUEMEM_TEST_QDRANT_API_KEY") or None
pytestmark = [
    pytest.mark.skipif(not URL, reason="需要 Qdrant 服务端：设 CUEMEM_TEST_QDRANT_URL"),
    # 经 http:// 发 Key 时 qdrant-client 每个客户端警告一次（CI 与本机上无妨）
    pytest.mark.filterwarnings("ignore:Api key is used with an insecure connection"),
]

DENSE = l2_normalize(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32))
SPARSE = [
    SparseVector(indices=[1, 2], values=[1.0, 0.5]),
    SparseVector(indices=[3], values=[1.0]),
    SparseVector(indices=[], values=[]),
]
PAYLOADS = [
    {"kind": "round", "ts": 100.0},
    {"kind": "fact", "ts": 200.0},
    {"kind": "round", "ts": 300.0},
]


@pytest.fixture
def token() -> str:
    """这个测试的空间名里带的记号（集合名里也带着它，结束时按它删）。"""
    return f"ci{uuid.uuid4().hex[:12]}"


@pytest.fixture
def server(token: str) -> Iterator[VectorStore]:
    store = VectorStore(None, dim=4, url=URL, api_key=KEY)
    try:
        yield store
    finally:
        for collection in store._client.get_collections().collections:
            if token in collection.name:
                store._client.delete_collection(collection.name)
        store.close()


def _indexes(store: VectorStore, name: str) -> dict[str, object]:
    info = store._client.get_collection(name)
    return {key: index.data_type for key, index in info.payload_schema.items()}


def _wait_indexed(store: VectorStore, space: str) -> None:
    deadline = time.monotonic() + 60
    while not store.indexed(space):
        assert time.monotonic() < deadline, "Qdrant 一分钟内没有建完索引"
        time.sleep(0.2)


def test_server_collections_are_indexed_and_filtered(server: VectorStore, token: str) -> None:
    """新集合按 kind、ts 建 payload 索引（过滤检索不必扫全集合）；按种类、时间过滤，删点、改 payload
    之后的计数与本地模式相同。"""
    space = f"{token}/alice"
    server.upsert(space, ["a", "b", "c"], DENSE, SPARSE, PAYLOADS)
    _wait_indexed(server, space)
    assert _indexes(server, server.collection_name(space)) == dict(PAYLOAD_INDEXES)
    assert [h.unit_id for h in server.search_dense(space, DENSE[0], 2)] == ["a", "c"]
    assert [h.unit_id for h in server.search_dense(space, DENSE[0], 3, kinds=["fact"])] == ["b"]
    later = server.search_dense(space, DENSE[0], 3, ts_range=(150.0, None))
    assert [h.unit_id for h in later] == ["c", "b"]
    assert [h.unit_id for h in server.search_sparse(space, SPARSE[1], 3)] == ["b"]
    assert server.count(space) == 3 and server.count(space, kinds=["round"]) == 2
    server.set_payload(space, ["b"], {"kind": "round"})
    server.delete(space, ["a"])
    assert server.count(space) == 2 and server.count(space, kinds=["round"]) == 2
    assert sorted(server.unit_ids(space)) == ["b", "c"]
    assert set(server.dense_vectors(space, ["b"])) == {"b"}


def test_deleting_a_space_drops_both_collections(server: VectorStore, token: str) -> None:
    space = f"{token}/bob"
    server.upsert(space, ["a", "b", "c"], DENSE, SPARSE, PAYLOADS)
    server.upsert_images(space, ["m1"], l2_normalize(np.ones((1, 2), dtype=np.float32)), [{}])
    name = server.collection_name(space)
    assert server._client.collection_exists(name + IMAGE_SUFFIX)
    server.delete_space(space)
    assert not server._client.collection_exists(name)
    assert not server._client.collection_exists(name + IMAGE_SUFFIX)
    assert server.count(space) == 0 and server.search_dense(space, DENSE[0], 3) == []


def test_old_collection_names_need_proof_of_ownership(server: VectorStore, token: str) -> None:
    """升级之前的旧名字集合（32 位摘要）只给能证明是自己的空间：数据库里它的记忆单元在里面。"""
    space = f"{token}/carol"
    legacy = VectorStore.legacy_name(space)
    server._names[space] = legacy  # 升级之前建的
    server.upsert(space, ["u1"], DENSE[:1], SPARSE[:1], PAYLOADS[:1])
    assert server._client.collection_exists(legacy)

    owner = VectorStore(None, dim=4, url=URL, api_key=KEY)  # 升级之后的新进程
    owner.owned_units = lambda s: ["u1"] if s == space else []
    stranger = VectorStore(None, dim=4, url=URL, api_key=KEY)
    stranger.owned_units = lambda s: ["someone-else"]
    try:
        assert owner.collection_name(space) == legacy and owner.count(space) == 1
        assert stranger.collection_name(space) != legacy and stranger.count(space) == 0
    finally:
        owner.close()
        stranger.close()


def test_local_vectors_migrate_into_the_server(
    server: VectorStore, token: str, tmp_path: Path
) -> None:
    """cuemem migrate-vectors 的做法（copy_into）：文字与图片集合原样搬到服务端，服务端的文字集合
    建好 payload 索引，检索结果与本地相同。"""
    space = f"{token}/dave"
    source = VectorStore(tmp_path / "local", dim=4)
    try:
        source.upsert(space, ["a", "b", "c"], DENSE, SPARSE, PAYLOADS)
        source.upsert_images(space, ["m1"], l2_normalize(np.ones((1, 2), dtype=np.float32)), [{}])
        counts = source.copy_into(server)
        name = server.collection_name(space)
        assert counts == {name: 3, name + IMAGE_SUFFIX: 1}
        _wait_indexed(server, space)
        assert _indexes(server, name) == dict(PAYLOAD_INDEXES)
        assert _indexes(server, name + IMAGE_SUFFIX) == {}
        before = [h.unit_id for h in source.search_dense(space, DENSE[1], 3)]
        assert [h.unit_id for h in server.search_dense(space, DENSE[1], 3)] == before
        assert [h.unit_id for h in server.search_images(space, np.ones(2) / np.sqrt(2), 1)] == [
            "m1"
        ]
    finally:
        source.close()


@pytest.mark.skipif(not KEY, reason="服务端没设 API Key：设 CUEMEM_TEST_QDRANT_API_KEY")
def test_a_keyed_server_refuses_a_store_without_the_key(server: VectorStore, token: str) -> None:
    """服务端真的在查 Key（CI 给 Qdrant 设了 QDRANT__SERVICE__API_KEY）：带 Key 的连接照常，不带的
    401。以前 VectorStore 发不了 Key，连不上这样的服务端（Qdrant Cloud 也一样）。"""
    from qdrant_client.http.exceptions import UnexpectedResponse

    space = f"{token}/erin"
    server.upsert(space, ["a"], DENSE[:1], SPARSE[:1], PAYLOADS[:1])
    assert server.count(space) == 1
    keyless = VectorStore(None, dim=4, url=URL)
    try:
        with pytest.raises(UnexpectedResponse) as refused:
            keyless.count(space)
        assert refused.value.status_code == 401
    finally:
        keyless.close()
