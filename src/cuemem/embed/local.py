"""本机向量（fastembed，ONNX 推理，N12）：不要 Key、不要 Ollama，pip 装上就能用。

    uv sync --extra local          # 或 pip install "cuemem[local]"
    CUEMEM_PRESET=deepseek         # 只有 DeepSeek 的 Key 时：向量在本机算

模型第一次用到时从 Hugging Face 下载（bge-small-en-v1.5 约 130 MB），缓存在数据目录下。
CPU 上的速度与文本长度关系很大：4 核的开发机上，100 个字符约 20 条 / 秒，1000 个字符约 2 条 / 秒。
短对话（LoCoMo）够用；LongMemEval 这种几千万字符的历史要用 API 向量或 GPU。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from cuemem.config import EmbeddingConfig
from cuemem.core.trace import record_event
from cuemem.embed.base import EmbeddingBatch, EmbedKind, l2_normalize
from cuemem.errors import EmbeddingAPIError
from cuemem.llm.ledger import Ledger, UsageRecord


class FastEmbedTextEmbedder:
    def __init__(
        self,
        cfg: EmbeddingConfig,
        *,
        cache_dir: Path | None = None,
        ledger: Ledger | None = None,
        threads: int | None = None,
    ) -> None:
        self.cfg = cfg
        self.name = cfg.model
        self.dim = cfg.dim
        self.supports_sparse = False
        self._cache_dir = cache_dir
        self._threads = threads
        self._ledger = ledger
        self._model: Any = None
        self._lock = threading.Lock()  # ONNX 会话自己用满所有核：同一时间只跑一批

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(dense=np.zeros((0, self.dim), dtype=np.float32))
        prefix = self.cfg.query_instruct if kind == "query" and self.cfg.query_instruct else ""
        batch = [prefix + (t[: self.cfg.max_chars] or " ") for t in texts]
        started = time.perf_counter()
        with self._lock:
            model = self._load()
            rows = list(model.embed(batch, batch_size=self.cfg.batch_size))
        dense = l2_normalize(np.asarray(rows, dtype=np.float32))
        if dense.shape[1] != self.dim:
            raise EmbeddingAPIError(
                f"向量维度不符：配置 {self.dim}，模型 {self.cfg.model} 返回 {dense.shape[1]}；"
                "请改 embedding.dim"
            )
        latency_ms = (time.perf_counter() - started) * 1000
        self._account(len(batch), latency_ms)
        return EmbeddingBatch(dense=dense, sparse=None, tokens=0)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding  # type: ignore[import-not-found, unused-ignore]
            except ImportError as exc:  # pragma: no cover - 取决于是否装了可选依赖
                raise EmbeddingAPIError(
                    "本机向量需要可选依赖 fastembed：uv sync --extra local（或 pip install "
                    '"cuemem[local]"）'
                ) from exc
            self._model = TextEmbedding(
                self.cfg.model,
                cache_dir=str(self._cache_dir) if self._cache_dir is not None else None,
                threads=self._threads,
            )
        return self._model

    def _account(self, n_texts: int, latency_ms: float) -> None:
        if self._ledger is not None:  # 不花钱，但记下调用次数与耗时，便于和 API 向量比较
            self._ledger.record(
                UsageRecord(
                    ts=datetime.now(),
                    run_id=self._ledger.run_id,
                    kind="embedding",
                    role=None,
                    provider="local",
                    model=self.cfg.model,
                    input_tokens=0,
                    cost=0.0,
                    cost_cached=0.0,
                    currency=self.cfg.currency,
                    latency_ms=latency_ms,
                )
            )
        record_event(
            "embedding", model=self.cfg.model, texts=n_texts, tokens=0, cost=0.0,
            latency_ms=round(latency_ms, 1),
        )  # fmt: skip
