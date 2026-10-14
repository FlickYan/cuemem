"""OpenAI 兼容的文本向量接口（POST {base_url}/embeddings，N12）。

OpenAI、Ollama、vLLM、LM Studio、硅基流动等都提供这个接口，换供应商只改配置。它只返回 dense
向量；关键词检索本来就由本地 BM25 承担（D-026），不受影响。

重试、退避、并发上限与记账沿用通义向量器的实现，只换请求体与响应的解析。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx2
import numpy as np

from cuemem.config import EmbeddingConfig
from cuemem.embed.base import EmbedKind, FloatArray, SparseVector
from cuemem.embed.dashscope import DashScopeTextEmbedder
from cuemem.errors import EmbeddingAPIError
from cuemem.llm.ledger import Ledger


class OpenAICompatibleEmbedder(DashScopeTextEmbedder):
    def __init__(
        self,
        cfg: EmbeddingConfig,
        api_key: str,
        base_url: str,
        *,
        http_client: httpx2.Client | None = None,
        ledger: Ledger | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        cfg = cfg.model_copy(update={"sparse": False})  # 兼容接口只有 dense 向量
        super().__init__(cfg, api_key, http_client=http_client, ledger=ledger, sleep=sleep)
        self._url = base_url.rstrip("/") + "/embeddings"

    def _body(self, batch: list[str], kind: EmbedKind) -> dict[str, Any]:
        # 兼容接口不区分查询与文档；配置了查询指令（Qwen3-Embedding 之类的指令模型）时拼在查询前面
        if kind == "query" and self.cfg.query_instruct:
            batch = [f"Instruct: {self.cfg.query_instruct}\nQuery: {text}" for text in batch]
        body: dict[str, Any] = {"model": self.cfg.model, "input": batch}
        if self.cfg.send_dimensions:
            body["dimensions"] = self.cfg.dim
        return body

    def _parse(
        self, data: dict[str, Any], *, expected: int
    ) -> tuple[FloatArray, list[SparseVector], int]:
        items = data.get("data")
        if not isinstance(items, list):
            raise EmbeddingAPIError(f"向量接口返回格式异常：{str(data)[:300]}")
        items = sorted(items, key=lambda item: item.get("index", 0))
        if len(items) != expected:
            raise EmbeddingAPIError(f"向量数量不符：期望 {expected}，实际 {len(items)}")
        dense = np.asarray([item["embedding"] for item in items], dtype=np.float32)
        if dense.shape[1] != self.dim:
            raise EmbeddingAPIError(
                f"向量维度不符：配置 {self.dim}，模型 {self.cfg.model} 返回 {dense.shape[1]}；"
                "请改 embedding.dim（或打开 embedding.send_dimensions）"
            )
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        return dense, [], tokens
