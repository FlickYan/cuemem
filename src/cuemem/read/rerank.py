"""重排（cross-encoder）：把问题和每段候选文本一起送进重排模型打相关度分。

向量检索把问题和文本各自编码再比较，重排模型把两者放在一起读，判断更准，但每次都要调用接口
（QwenCloud 的 qwen3-rerank，按输入 token 计费），延迟也在几百毫秒。所以只用在 L2：
对多路召回融合后的前几十条候选重新排序。

和 LLM 调用一样：相同请求命中本地响应缓存（同一张缓存表，按记忆空间标记归属，遗忘时一并清理），
用量记进账本与 trace。
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import httpx2

from cuemem.config import RerankConfig
from cuemem.core.trace import record_event
from cuemem.errors import CueMemError
from cuemem.llm.cache import ResponseCache, request_key
from cuemem.llm.ledger import Ledger, UsageRecord

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RerankAPIError(CueMemError):
    """重排接口返回错误或格式异常。"""


class Reranker:
    def __init__(
        self,
        cfg: RerankConfig,
        api_key: str,
        *,
        http_client: httpx2.Client | None = None,
        cache: ResponseCache | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.cfg = cfg
        self._api_key = api_key
        self._http = http_client or httpx2.Client(timeout=cfg.timeout_s)
        self._cache = cache
        self._ledger = ledger
        self._slots = threading.BoundedSemaphore(max(1, cfg.max_concurrency))

    def scores(
        self, query: str, documents: Sequence[str], *, cache_scope: str | None = None
    ) -> list[float]:
        """返回与 documents 一一对应的相关度分数（0–1，越大越相关）。"""
        if not documents:
            return []
        docs = [d[: self.cfg.max_chars] or " " for d in documents]
        key = request_key({"kind": "rerank", "model": self.cfg.model, "query": query, "docs": docs})
        cached = self._cache.get(key) if self._cache is not None else None
        if cached is not None:
            self._account(int(cached.get("tokens", 0)), 0.0, len(docs), from_cache=True)
            return [float(x) for x in cached["scores"]]
        body = {
            "model": self.cfg.model,
            "input": {"query": query, "documents": docs},
            "parameters": {"return_documents": False, "top_n": len(docs)},
        }
        data, latency_ms = self._post(body)
        scores, tokens = _parse(data, expected=len(docs))
        if self._cache is not None:
            self._cache.put(
                key, self.cfg.model, {"scores": scores, "tokens": tokens}, scope=cache_scope
            )
        self._account(tokens, latency_ms, len(docs), from_cache=False)
        return scores

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        for attempt in range(1, self.cfg.max_attempts + 1):
            started = time.perf_counter()
            try:
                with self._slots:
                    response = self._http.post(self.cfg.url, json=body, headers=headers)
            except httpx2.TransportError as exc:
                if attempt == self.cfg.max_attempts:
                    raise RerankAPIError(f"重排接口网络错误：{exc}") from exc
                time.sleep(min(30, 2 ** (attempt - 1)) + random.uniform(0, 0.5))
                continue
            latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code == 200:
                return response.json(), latency_ms
            if response.status_code in _RETRYABLE_STATUS and attempt < self.cfg.max_attempts:
                time.sleep(min(30, 2 ** (attempt - 1)) + random.uniform(0, 0.5))
                continue
            raise RerankAPIError(f"重排接口返回 HTTP {response.status_code}：{response.text[:300]}")
        raise RerankAPIError("重排接口重试次数用尽")

    def _account(self, tokens: int, latency_ms: float, n_docs: int, *, from_cache: bool) -> None:
        cost = tokens * self.cfg.price_per_m / 1e6
        if self._ledger is not None:
            self._ledger.record(
                UsageRecord(
                    ts=datetime.now(),
                    run_id=self._ledger.run_id,
                    kind="rerank",
                    role=None,
                    provider=self.cfg.provider,
                    model=self.cfg.model,
                    input_tokens=tokens,
                    cost=cost,
                    cost_cached=cost,
                    currency=self.cfg.currency,
                    from_cache=from_cache,
                    latency_ms=latency_ms,
                )
            )
        record_event(
            "rerank",
            model=self.cfg.model,
            docs=n_docs,
            tokens=tokens,
            cost=round(cost, 8),
            from_cache=from_cache,
            latency_ms=round(latency_ms, 1),
        )


def _parse(data: dict[str, Any], *, expected: int) -> tuple[list[float], int]:
    output = data.get("output")
    if not isinstance(output, dict) or "results" not in output:
        raise RerankAPIError(f"重排接口返回格式异常：{data.get('code')} {data.get('message')}")
    scores = [0.0] * expected
    for item in output["results"]:
        index = int(item["index"])
        if 0 <= index < expected:
            scores[index] = float(item["relevance_score"])
    tokens = int((data.get("usage") or {}).get("total_tokens", 0))
    return scores, tokens
