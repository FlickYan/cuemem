"""图片向量（N6）：图片与文字映射到同一个向量空间，用于以文搜图、以图搜图。

用 QwenCloud 的 tongyi-embedding-vision-plus（原生接口，1152 维）。它对图片和文字分别给出向量，
不是把图片连同对话一起编码的"融合向量"（qwen3-vl-embedding 能做，但 QwenCloud 上没有开放）。
图片按内容哈希缓存，同一张图只向量化一次。离线测试用 HashMultimodalEmbedder。
"""

from __future__ import annotations

import contextlib
import hashlib
import random
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import httpx2
import numpy as np

from cuemem.config import MultimodalEmbeddingConfig
from cuemem.core.ids import content_hash
from cuemem.core.trace import record_event
from cuemem.embed.base import FloatArray, l2_normalize
from cuemem.embed.cache import EmbeddingCache
from cuemem.errors import EmbeddingAPIError
from cuemem.llm.ledger import Ledger, UsageRecord
from cuemem.store.media import image_data_url

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ImageEmbedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_images(self, paths: Sequence[Path]) -> FloatArray: ...

    def embed_texts(self, texts: Sequence[str]) -> FloatArray: ...


class DashScopeMultimodalEmbedder:
    def __init__(
        self,
        cfg: MultimodalEmbeddingConfig,
        api_key: str,
        *,
        http_client: httpx2.Client | None = None,
        ledger: Ledger | None = None,
        cache: EmbeddingCache | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.cfg = cfg
        self.name = cfg.model
        self.dim = cfg.dim
        self._api_key = api_key
        self._http = http_client or httpx2.Client(timeout=cfg.timeout_s)
        self._ledger = ledger
        self._cache = cache
        self._sleep = sleep
        self._slots = threading.BoundedSemaphore(max(1, cfg.max_concurrency))

    def embed_images(self, paths: Sequence[Path]) -> FloatArray:
        keys = [self._key("image", _file_hash(p)) for p in paths]
        return self._embed(keys, [{"image": str(p)} for p in paths])

    def embed_texts(self, texts: Sequence[str]) -> FloatArray:
        keys = [self._key("text", content_hash(t)) for t in texts]
        return self._embed(keys, [{"text": t or " "} for t in texts])

    def _embed(self, keys: list[str], contents: list[dict[str, str]]) -> FloatArray:
        found: dict[str, FloatArray] = {}
        if self._cache is not None:
            found = {k: v[0] for k, v in self._cache.get_many(keys).items()}
        todo: list[int] = []  # 需要真正请求的下标（同一批里的重复内容只请求一次）
        seen: set[str] = set()
        for i, key in enumerate(keys):
            if key not in found and key not in seen:
                seen.add(key)
                todo.append(i)
        for start in range(0, len(todo), self.cfg.batch_size):
            batch = todo[start : start + self.cfg.batch_size]
            body = {
                "model": self.cfg.model,
                "input": {"contents": [self._content(contents[i]) for i in batch]},
            }
            data, latency_ms = self._post(body)
            vectors, tokens = _parse(data, expected=len(batch))
            items = []
            for row, i in enumerate(batch):
                found[keys[i]] = vectors[row]
                items.append((keys[i], vectors[row], None))
            if self._cache is not None:
                self._cache.put_many(items)
            self._account(tokens, latency_ms, len(batch))
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.vstack([found[k] for k in keys]))

    def _content(self, content: dict[str, str]) -> dict[str, str]:
        if "image" in content:
            return {"image": image_data_url(content["image"], max_side=self.cfg.max_side)}
        return content

    def _key(self, kind: str, digest: str) -> str:
        return content_hash("\x1f".join(["mm", self.cfg.model, str(self.cfg.dim), kind, digest]))

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(1, self.cfg.max_attempts + 1):
            started = time.perf_counter()
            try:
                with self._slots:
                    response = self._http.post(self.cfg.url, json=body, headers=headers)
            except httpx2.TransportError as exc:
                if attempt == self.cfg.max_attempts:
                    raise EmbeddingAPIError(f"图片向量接口网络错误：{exc}") from exc
                self._wait(attempt, None)
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code == 200:
                return response.json(), latency_ms
            if response.status_code in _RETRYABLE_STATUS and attempt < self.cfg.max_attempts:
                self._wait(attempt, response.headers.get("retry-after"))
                continue
            raise EmbeddingAPIError(
                f"图片向量接口返回 HTTP {response.status_code}：{response.text[:300]}"
            )
        raise EmbeddingAPIError("图片向量接口重试次数用尽")

    def _wait(self, attempt: int, retry_after: str | None) -> None:
        seconds = float(min(30, 2 ** (attempt - 1))) + random.uniform(0, 0.5)
        if retry_after is not None:
            with contextlib.suppress(ValueError):
                seconds = min(60.0, max(0.0, float(retry_after)))
        self._sleep(seconds)

    def _account(self, tokens: int, latency_ms: float, n: int) -> None:
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
            texts=n,
            tokens=tokens,
            cost=round(cost, 8),
            latency_ms=round(latency_ms, 1),
        )


class HashMultimodalEmbedder:
    """离线测试用：图片按文件内容、文字按文本内容生成确定的伪随机向量（没有语义）。"""

    def __init__(self, dim: int = 64) -> None:
        self.name = "hash-mm"
        self.dim = dim

    def embed_images(self, paths: Sequence[Path]) -> FloatArray:
        return self._vectors([_file_hash(p) for p in paths])

    def embed_texts(self, texts: Sequence[str]) -> FloatArray:
        return self._vectors([content_hash(t) for t in texts])

    def _vectors(self, digests: Sequence[str]) -> FloatArray:
        rows = [np.random.default_rng(int(d[:16], 16)).standard_normal(self.dim) for d in digests]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.vstack(rows))


def _file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse(data: dict[str, Any], *, expected: int) -> tuple[FloatArray, int]:
    output = data.get("output")
    if not isinstance(output, dict) or "embeddings" not in output:
        raise EmbeddingAPIError(
            f"图片向量接口返回格式异常：{data.get('code')} {data.get('message')}"
        )
    items = sorted(output["embeddings"], key=lambda item: item.get("index", 0))
    if len(items) != expected:
        raise EmbeddingAPIError(f"图片向量数量不符：期望 {expected}，实际 {len(items)}")
    vectors = np.asarray([item["embedding"] for item in items], dtype=np.float32)
    usage = data.get("usage") or {}
    tokens = int(usage.get("input_tokens") or usage.get("total_tokens") or 0)
    return vectors, tokens
