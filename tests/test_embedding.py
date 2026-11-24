from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx2
import numpy as np
import pytest
from fakes import FakeAPI

from cuemem.config import MultimodalEmbeddingConfig, Settings
from cuemem.core.ids import content_hash
from cuemem.core.trace import start_trace
from cuemem.embed import (
    CachedEmbedder,
    DashScopeTextEmbedder,
    EmbeddingBatch,
    EmbeddingCache,
    EmbedKind,
    HashEmbedder,
    SparseVector,
    build_embedder,
)
from cuemem.embed.multimodal import DashScopeMultimodalEmbedder
from cuemem.errors import EmbeddingAPIError
from cuemem.llm import Ledger
from cuemem.memory import MemorySystem


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


class SlowAPI:
    """FakeAPI 外面包一层：每个向量请求花 delay 秒，记下同时在途的请求数的峰值。"""

    def __init__(self, fake_api: FakeAPI, delay: float = 0.02) -> None:
        self.fake_api = fake_api
        self.delay = delay
        self.flying = 0
        self.peak = 0
        self.lock = threading.Lock()

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        with self.lock:
            self.flying += 1
            self.peak = max(self.peak, self.flying)
        try:
            time.sleep(self.delay)
            with self.lock:  # FakeAPI 的预设状态码序列不是线程安全的
                return self.fake_api._handle(request)
        finally:
            with self.lock:
                self.flying -= 1


def slow_embedder(
    settings: Settings, api: SlowAPI, ledger: Ledger, **overrides: object
) -> DashScopeTextEmbedder:
    api.fake_api.embed_dim = settings.embedding.dim
    parallel = bool(overrides.pop("parallel", False))
    return DashScopeTextEmbedder(
        settings.embedding.model_copy(update=overrides),
        "sk-test",
        http_client=api.client(),
        ledger=ledger,
        sleep=lambda _: None,
        parallel=parallel,
    )


def ledger_tokens(ledger: Ledger) -> list[int]:
    rows = ledger._conn.execute("SELECT input_tokens FROM usage ORDER BY rowid").fetchall()
    return [int(row[0]) for row in rows]


def test_parallel_batches_overlap_up_to_max_concurrency(settings: Settings) -> None:
    """parallel 的向量器（MemorySystem.from_settings 建的）：一次调用的几个请求同时发，在途的不
    超过 max_concurrency；请求体、向量、tokens 与账本里的行（按输入的顺序）都与逐个发时相同。
    默认（评测）还是一个接一个，请求的顺序就是输入的顺序。"""
    texts = [f"text number {i} " + "x" * (i % 7) for i in range(65)]  # 7 个请求，最后一个 5 段

    one_by_one, ledger_a = SlowAPI(FakeAPI()), Ledger(None)
    expected = slow_embedder(settings, one_by_one, ledger_a, max_concurrency=3).embed(
        texts, "document"
    )
    assert one_by_one.peak == 1
    in_order = one_by_one.fake_api.embed_requests
    assert [t for r in in_order for t in r["input"]["texts"]] == texts

    together, ledger_b = SlowAPI(FakeAPI()), Ledger(None)
    embedder = slow_embedder(settings, together, ledger_b, max_concurrency=3, parallel=True)
    got = embedder.embed(texts, "document")
    assert 2 <= together.peak <= 3
    assert np.array_equal(got.dense, expected.dense) and got.tokens == expected.tokens

    def canonical(requests: list[dict[str, object]]) -> list[str]:
        return sorted(json.dumps(r, sort_keys=True) for r in requests)

    assert canonical(together.fake_api.embed_requests) == canonical(in_order)
    assert ledger_tokens(ledger_b) == ledger_tokens(ledger_a) == [50] * 6 + [25]

    # 只有一个请求的调用（检索时的问题、一次 add）不建线程池
    together.peak = 0
    embedder.embed(texts[:10], "query")
    assert together.peak == 1

    # max_concurrency 为 1 时 parallel 不起作用
    single = SlowAPI(FakeAPI())
    slow_embedder(settings, single, Ledger(None), max_concurrency=1, parallel=True).embed(
        texts, "document"
    )
    assert single.peak == 1


def test_parallel_batches_stop_at_the_first_failure(settings: Settings) -> None:
    """同时发的请求里有一个失败：排着队的不再发，已经回来的照样记账（已经付费），抛出那个错误。"""
    api, ledger = SlowAPI(FakeAPI()), Ledger(None)
    api.fake_api.embed_status = [400]  # 第一个到的请求失败
    embedder = slow_embedder(settings, api, ledger, max_concurrency=2, parallel=True)
    with pytest.raises(EmbeddingAPIError, match="400"):
        embedder.embed([f"text {i}" for i in range(200)], "document")  # 20 个请求
    sent = len(api.fake_api.embed_requests)
    assert 2 <= sent < 20
    assert len(ledger_tokens(ledger)) == sent - 1


def test_a_failed_parallel_batch_does_not_shift_the_accounting(settings: Settings) -> None:
    """同时发的三个请求（10、10、5 段）里第一个失败、另外两个回来了：embedding 事件的 texts 是
    回来的那两批的段数。以前成功的结果从第一批起对着数，第三批的 5 段记成了 10。"""
    fake = FakeAPI()
    fake.embed_dim = settings.embedding.dim
    lock = threading.Lock()

    def handle(request: httpx2.Request) -> httpx2.Response:
        if "text 0" in json.loads(request.content)["input"]["texts"]:
            time.sleep(0.2)  # 等另外两个回来
            return httpx2.Response(400, json={"message": "bad request"})
        with lock:
            return fake._handle(request)

    embedder = DashScopeTextEmbedder(
        settings.embedding.model_copy(update={"max_concurrency": 3}),
        "sk-test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
        ledger=Ledger(None),
        sleep=lambda _: None,
        parallel=True,
    )
    with start_trace("embed") as trace, pytest.raises(EmbeddingAPIError, match="400"):
        embedder.embed([f"text {i}" for i in range(25)], "document")
    assert [e["texts"] for e in trace.events if e["kind"] == "embedding"] == [10, 5]


def test_parallel_batches_keep_trace_events(settings: Settings) -> None:
    """线程里的重试照样记进调用方的 trace（任务带着调用方的上下文），记账的事件按输入的顺序。"""
    api = SlowAPI(FakeAPI())
    api.fake_api.embed_status = [429, 200]
    embedder = slow_embedder(settings, api, Ledger(None), max_concurrency=2, parallel=True)
    with start_trace("embed") as trace:
        embedder.embed([f"text {i}" for i in range(25)], "document")
    kinds = [e["kind"] for e in trace.events]
    assert kinds.count("embedding_retry") == 1
    assert [e["texts"] for e in trace.events if e["kind"] == "embedding"] == [10, 10, 5]


def test_only_from_settings_builds_a_parallel_embedder(
    settings: Settings, fake_api: FakeAPI
) -> None:
    def parallel(embedder: object) -> bool:
        assert isinstance(embedder, CachedEmbedder)
        assert isinstance(embedder.inner, DashScopeTextEmbedder)
        return embedder.inner._parallel

    embedder = build_embedder(settings, http_client=fake_api.client())  # 评测与脚本
    assert isinstance(embedder, CachedEmbedder)
    try:
        assert not parallel(embedder)
    finally:
        embedder.cache.close()
    system = MemorySystem.from_settings(settings, http_client=fake_api.client())
    try:
        assert parallel(system.embedder)
    finally:
        system.close()
