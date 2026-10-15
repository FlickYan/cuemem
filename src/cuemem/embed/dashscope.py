"""通义千问文本向量（DashScope 原生接口）：一次请求同时返回 dense 与 sparse 向量。

不用 OpenAI 兼容接口的原因：兼容接口只返回 dense 向量，也不支持 text_type / instruct 参数。
"""

from __future__ import annotations

import contextlib
import random
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import httpx2
import numpy as np

from cuemem.config import EmbeddingConfig
from cuemem.core.trace import record_event
from cuemem.embed.base import EmbeddingBatch, EmbedKind, FloatArray, SparseVector, l2_normalize
from cuemem.errors import EmbeddingAPIError
from cuemem.llm.ledger import Ledger, UsageRecord

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class DashScopeTextEmbedder:
    def __init__(
        self,
        cfg: EmbeddingConfig,
        api_key: str,
        *,
        http_client: httpx2.Client | None = None,
        ledger: Ledger | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.name = cfg.model
        self.dim = cfg.dim
        self.supports_sparse = cfg.sparse
        self._api_key = api_key
        self._url = cfg.native_url
        self._http = http_client or httpx2.Client(timeout=cfg.timeout_s)
        self._ledger = ledger
        self._max_attempts = max(1, cfg.max_attempts)
        self._sleep = sleep  # 测试时替换为空函数，避免真的等待
        # 评测时多个线程共用同一个向量器：限制在途请求数，避免触发按分钟计的配额限流
        self._slots = threading.BoundedSemaphore(max(1, cfg.max_concurrency))

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
        dense_parts: list[FloatArray] = []
        sparse: list[SparseVector] = []
        tokens = 0
        for start in range(0, len(texts), self.cfg.batch_size):
            # 接口拒绝空字符串，用空格占位；过长文本截断
            batch = [
                t[: self.cfg.max_chars] or " " for t in texts[start : start + self.cfg.batch_size]
            ]
            data, latency_ms = self._post(self._body(batch, kind))
            vectors, sparse_vectors, used = self._parse(data, expected=len(batch))
            dense_parts.append(vectors)
            sparse.extend(sparse_vectors)
            tokens += used
            self._account(used, latency_ms, len(batch))
        dense = (
            l2_normalize(np.vstack(dense_parts))
            if dense_parts
            else np.zeros((0, self.dim), dtype=np.float32)
        )
        return EmbeddingBatch(
            dense=dense, sparse=sparse if self.cfg.sparse else None, tokens=tokens
        )

    def _body(self, batch: list[str], kind: EmbedKind) -> dict[str, Any]:
        params: dict[str, Any] = {
            "dimension": self.cfg.dim,
            "text_type": kind,
            "output_type": "dense&sparse" if self.cfg.sparse else "dense",
        }
        if kind == "query" and self.cfg.query_instruct:
            params["instruct"] = self.cfg.query_instruct
        return {"model": self.cfg.model, "input": {"texts": batch}, "parameters": params}

    def _parse(
        self, data: dict[str, Any], *, expected: int
    ) -> tuple[FloatArray, list[SparseVector], int]:
        return _parse(data, expected=expected)

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(1, self._max_attempts + 1):
            started = time.perf_counter()
            try:
                with self._slots:
                    response = self._http.post(self._url, json=body, headers=headers)
            except httpx2.TransportError as exc:
                if attempt == self._max_attempts:
                    raise EmbeddingAPIError(f"向量接口网络错误：{exc}") from exc
                self._wait(attempt, status=None, retry_after=None)
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code == 200:
                return response.json(), latency_ms
            if response.status_code in _RETRYABLE_STATUS and attempt < self._max_attempts:
                self._wait(attempt, response.status_code, response.headers.get("retry-after"))
                continue
            raise EmbeddingAPIError(
                f"向量接口返回 HTTP {response.status_code}：{response.text[:300]}",
                status=response.status_code,
                retry_after=response.headers.get("retry-after"),
            )
        raise EmbeddingAPIError("向量接口重试次数用尽")

    def _wait(self, attempt: int, status: int | None, retry_after: str | None) -> None:
        """退避等待：服务端给了 Retry-After 就照做，否则指数退避（1、2、4……秒，最多 30 秒）。"""
        seconds = _backoff(attempt)
        if retry_after is not None:
            with contextlib.suppress(ValueError):
                seconds = min(60.0, max(0.0, float(retry_after)))
        record_event("embedding_retry", status=status, attempt=attempt, wait_s=round(seconds, 2))
        self._sleep(seconds)

    def _account(self, tokens: int, latency_ms: float, n_texts: int) -> None:
        cost = tokens * self.cfg.price_per_m / 1e6
        if self._ledger is not None:
            self._ledger.record(
                UsageRecord(
                    ts=datetime.now(),
                    run_id=self._ledger.run_id,
                    kind="embedding",
                    role=None,
                    provider=self.cfg.provider,
                    model=self.cfg.model,
                    input_tokens=tokens,
                    cost=cost,
                    cost_cached=cost,
                    currency=self.cfg.currency,
                    latency_ms=latency_ms,
                )
            )
        record_event(
            "embedding",
            model=self.cfg.model,
            texts=n_texts,
            tokens=tokens,
            cost=round(cost, 8),
            latency_ms=round(latency_ms, 1),
        )


def _parse(data: dict[str, Any], *, expected: int) -> tuple[FloatArray, list[SparseVector], int]:
    output = data.get("output")
    if not isinstance(output, dict) or "embeddings" not in output:
        raise EmbeddingAPIError(f"向量接口返回格式异常：{data.get('code')} {data.get('message')}")
    items = sorted(output["embeddings"], key=lambda item: item.get("text_index", 0))
    if len(items) != expected:
        raise EmbeddingAPIError(f"向量数量不符：期望 {expected}，实际 {len(items)}")
    dense = np.asarray([item["embedding"] for item in items], dtype=np.float32)
    sparse = [
        SparseVector.from_pairs(
            (int(entry["index"]), float(entry["value"]))
            for entry in item.get("sparse_embedding") or []
        )
        for item in items
    ]
    tokens = int((data.get("usage") or {}).get("total_tokens", 0))
    return dense, sparse, tokens


def _backoff(attempt: int) -> float:
    return float(min(30, 2 ** (attempt - 1))) + random.uniform(0, 0.5)
