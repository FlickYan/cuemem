from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeAPI

from cuemem.config import MultimodalEmbeddingConfig, Settings
from cuemem.core.ids import content_hash
from cuemem.embed import (
    CachedEmbedder,
    DashScopeTextEmbedder,
    EmbeddingBatch,
    EmbeddingCache,
    EmbedKind,
    HashEmbedder,
    SparseVector,
)
from cuemem.embed.multimodal import DashScopeMultimodalEmbedder
from cuemem.errors import EmbeddingAPIError
from cuemem.llm import Ledger


def make_embedder(
    settings: Settings,
    fake_api: FakeAPI,
    *,
    sleep: Callable[[float], None] = lambda _: None,
    **overrides: object,
) -> DashScopeTextEmbedder:
    fake_api.embed_dim = settings.embedding.dim
    cfg = settings.embedding.model_copy(update=overrides)
    return DashScopeTextEmbedder(
        cfg, "sk-test", http_client=fake_api.client(), ledger=Ledger(None), sleep=sleep
    )


def test_dashscope_batches_and_parses(settings: Settings, fake_api: FakeAPI) -> None:
    # 稀疏向量的解析仍要测：text-embedding-v4 + sparse 是可选配置
    embedder = make_embedder(settings, fake_api, model="text-embedding-v4", sparse=True)
    batch = embedder.embed([f"text {i}" for i in range(23)], "document")

    assert len(fake_api.embed_requests) == 3  # batch_size = 10
    first = fake_api.embed_requests[0]
    assert first["model"] == "text-embedding-v4"
    assert first["parameters"] == {
        "dimension": 1024,
        "text_type": "document",
        "output_type": "dense&sparse",
    }
    assert batch.dense.shape == (23, 1024)
    assert np.allclose(np.linalg.norm(batch.dense, axis=1), 1.0, atol=1e-5)
    assert batch.sparse is not None
    # 重复下标取最大权重，并按下标排序
    assert batch.sparse[0] == SparseVector(indices=[3, 7], values=[0.4, 0.9])
    assert batch.tokens == 5 * 23


def test_instruct_only_applies_to_queries(settings: Settings, fake_api: FakeAPI) -> None:
    embedder = make_embedder(settings, fake_api, query_instruct="Find relevant memories")
    embedder.embed(["q"], "query")
    embedder.embed(["d"], "document")
    query_req, doc_req = fake_api.embed_requests
    assert query_req["parameters"]["instruct"] == "Find relevant memories"
    assert "instruct" not in doc_req["parameters"]


def test_retries_on_throttling(settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.embed_status = [429, 200]
    batch = make_embedder(settings, fake_api).embed(["hello"], "document")
    assert len(fake_api.embed_requests) == 2
    assert batch.dense.shape[0] == 1


def test_client_error_is_not_retried(settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.embed_status = [400]
    with pytest.raises(EmbeddingAPIError, match="400"):
        make_embedder(settings, fake_api).embed(["x"], "document")
    assert len(fake_api.embed_requests) == 1


class CountingEmbedder(HashEmbedder):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
        self.calls.append(list(texts))
        return super().embed(texts, kind)


def test_cached_embedder_only_embeds_misses(tmp_path: Path) -> None:
    inner = CountingEmbedder(64)
    cached = CachedEmbedder(inner, EmbeddingCache(tmp_path / "embed.sqlite"))

    first = cached.embed(["苹果", "香蕉", "苹果"], "document")
    assert inner.calls == [["苹果", "香蕉"]]  # 同一批里的重复文本只算一次
    second = cached.embed(["香蕉", "橙子"], "document")
    assert inner.calls[-1] == ["橙子"]
    assert np.allclose(first.dense[1], second.dense[0])
    assert first.sparse is not None and second.sparse is not None
    assert first.sparse[1] == second.sparse[0]

    cached.embed(["苹果"], "query")  # query 与 document 分开缓存
    assert inner.calls[-1] == ["苹果"]

    reopened = CachedEmbedder(inner, EmbeddingCache(tmp_path / "embed.sqlite"))
    reopened.embed(["橙子"], "document")  # 持久化：新实例直接命中
    assert len(inner.calls) == 3


class NamedEmbedder:
    """只用来算缓存键：名字、维度与稀疏开关取默认配置的向量模型。"""

    name, dim, supports_sparse = "qwen3.7-text-embedding", 1024, False

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
        raise AssertionError("算缓存键不需要向量化")


def test_embedding_cache_keys_are_frozen(tmp_path: Path) -> None:
    """文本与图片向量的缓存键钉死（与基线提交 67226a8 上相同）：键变了，data/cache 里已有的
    向量全部失效，重放发表的数字时要重新向量化。"""
    cached = CachedEmbedder(
        NamedEmbedder(),
        EmbeddingCache(tmp_path / "embed.sqlite"),
        instruct="Find memories that answer the question",
    )
    text = "我领养了一只金毛，叫 Max。"
    query_key = "d59f46a006fe056ace094ecb5400655cde3237cf456b97f8dd0bbb913d000685"
    document_key = "a7b11e3bc09380393d665e3b8bfec7489e4dd4fb60dd16de99ee8d05a9bafc56"
    assert cached._key(text, "query") == query_key  # 查询带指令
    assert cached._key(text, "document") == document_key

    images = DashScopeMultimodalEmbedder(MultimodalEmbeddingConfig(), "sk-test")
    text_key = "45dc995a9495b881a515e65d263dfbc7bb8daaf18552c4b3f11cc0646019cb81"
    image_key = "e0287fb67548371dae3846ce385b95fe64978b4c9dfc7828dfe61fb7fafe3d34"
    assert images._key("text", content_hash("a red sign that says CUE 42")) == text_key
    image = hashlib.sha256(b"\x89PNG not really an image").hexdigest()
    assert images._key("image", image) == image_key


def test_hash_embedder_reflects_lexical_overlap() -> None:
    batch = HashEmbedder(256).embed(
        ["我领养了一只金毛，叫 Max", "Max 是一只金毛", "笔记本预算 6000 元"], "document"
    )
    similarity = batch.dense @ batch.dense.T
    assert similarity[0, 1] > similarity[0, 2]


def test_retry_after_header_is_honored(settings: Settings, fake_api: FakeAPI) -> None:
    waits: list[float] = []
    fake_api.embed_status = [429, 200]
    fake_api.embed_retry_after = "3"
    make_embedder(settings, fake_api, sleep=waits.append).embed(["hello"], "document")
    assert waits == [3.0]


def test_gives_up_after_max_attempts(settings: Settings, fake_api: FakeAPI) -> None:
    waits: list[float] = []
    fake_api.embed_status = [503] * 10
    embedder = make_embedder(settings, fake_api, sleep=waits.append, max_attempts=3)
    with pytest.raises(EmbeddingAPIError, match="503"):
        embedder.embed(["hello"], "document")
    assert len(fake_api.embed_requests) == 3
    assert len(waits) == 2
    assert waits[0] < waits[1]  # 指数退避


class FlakyEmbedder(HashEmbedder):
    """第 fail_on 次调用时抛出限流错误，用来检验分块缓存。"""

    def __init__(self, dim: int, fail_on: int) -> None:
        super().__init__(dim)
        self.calls = 0
        self.fail_on = fail_on

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
        self.calls += 1
        if self.calls == self.fail_on:
            raise EmbeddingAPIError("HTTP 429")
        return super().embed(texts, kind)


def test_cached_embedder_keeps_partial_progress(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "embed.sqlite")
    flaky = CachedEmbedder(FlakyEmbedder(32, fail_on=2), cache, chunk_size=2)
    texts = ["一", "二", "三", "四"]
    with pytest.raises(EmbeddingAPIError):
        flaky.embed(texts, "document")  # 第一块成功、第二块失败
    retry_inner = CountingEmbedder(32)
    batch = CachedEmbedder(retry_inner, cache, chunk_size=2).embed(texts, "document")
    assert retry_inner.calls == [["三", "四"]]  # 只补算失败的那一块
    assert batch.dense.shape == (4, 32)


def test_default_embedding_is_dense_only(settings: Settings, fake_api: FakeAPI) -> None:
    """默认 qwen3.7-text-embedding 只取 dense：稀疏路由本地 BM25 承担（D-026）。"""
    batch = make_embedder(settings, fake_api).embed(["hello"], "document")
    request = fake_api.embed_requests[0]
    assert request["model"] == "qwen3.7-text-embedding"
    assert request["parameters"]["output_type"] == "dense"
    assert batch.sparse is None
