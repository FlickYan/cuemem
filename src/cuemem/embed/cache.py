"""向量缓存：按"模型 + 维度 + 类型 + 指令 + 文本"的哈希缓存结果，同一段文本只向量化一次。

评测时大量会话在不同样本间重复出现，这层缓存能省下大部分向量化费用。
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cuemem.core.ids import content_hash
from cuemem.embed.base import EmbeddingBatch, EmbedKind, FloatArray, SparseVector, TextEmbedder

CachedVector = tuple[FloatArray, SparseVector | None]


class EmbeddingCache:
    def __init__(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embed_cache ("
                " key TEXT PRIMARY KEY, dense BLOB NOT NULL, sparse_idx BLOB, sparse_val BLOB)"
            )
            self._conn.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, CachedVector]:
        found: dict[str, CachedVector] = {}
        unique = list(dict.fromkeys(keys))
        for start in range(0, len(unique), 500):
            chunk = unique[start : start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            with self._lock:
                rows = self._conn.execute(
                    "SELECT key, dense, sparse_idx, sparse_val FROM embed_cache"
                    f" WHERE key IN ({placeholders})",
                    chunk,
                ).fetchall()
            for key, dense, sparse_idx, sparse_val in rows:
                sparse = None
                if sparse_idx is not None:
                    sparse = SparseVector(
                        indices=np.frombuffer(sparse_idx, dtype=np.uint32).tolist(),
                        values=np.frombuffer(sparse_val, dtype=np.float32).tolist(),
                    )
                found[key] = (np.frombuffer(dense, dtype=np.float32).copy(), sparse)
        return found

    def put_many(self, items: Sequence[tuple[str, FloatArray, SparseVector | None]]) -> None:
        rows = [
            (
                key,
                np.asarray(dense, dtype=np.float32).tobytes(),
                None if sparse is None else np.asarray(sparse.indices, dtype=np.uint32).tobytes(),
                None if sparse is None else np.asarray(sparse.values, dtype=np.float32).tobytes(),
            )
            for key, dense, sparse in items
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embed_cache (key, dense, sparse_idx, sparse_val)"
                " VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class CachedEmbedder:
    """包装任意 TextEmbedder：先查缓存，只把未命中的文本交给内部模型。"""

    def __init__(
        self,
        inner: TextEmbedder,
        cache: EmbeddingCache,
        *,
        instruct: str | None = None,
        chunk_size: int = 50,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.instruct = instruct
        self.chunk_size = chunk_size

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def dim(self) -> int:
        return self.inner.dim

    @property
    def supports_sparse(self) -> bool:
        return self.inner.supports_sparse

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
        keys = [self._key(text, kind) for text in texts]
        found = self.cache.get_many(keys)
        missing: dict[str, str] = {}  # 键 → 文本，同一批里的重复文本只算一次
        for key, text in zip(keys, texts, strict=True):
            if key not in found:
                missing.setdefault(key, text)
        tokens = 0
        pending = list(missing.items())
        # 分块向量化，每块完成就写入缓存：中途失败时，已经付费的结果不会丢
        for start in range(0, len(pending), self.chunk_size):
            chunk = pending[start : start + self.chunk_size]
            batch = self.inner.embed([text for _, text in chunk], kind)
            new_items: list[tuple[str, FloatArray, SparseVector | None]] = []
            for row, (key, _) in enumerate(chunk):
                sparse = batch.sparse[row] if batch.sparse is not None else None
                found[key] = (batch.dense[row], sparse)
                new_items.append((key, batch.dense[row], sparse))
            self.cache.put_many(new_items)
            tokens += batch.tokens
        if not keys:
            return EmbeddingBatch(dense=np.zeros((0, self.dim), dtype=np.float32), tokens=0)
        dense = np.vstack([found[key][0] for key in keys]).astype(np.float32)
        sparse_out = (
            [found[key][1] or SparseVector(indices=[], values=[]) for key in keys]
            if self.supports_sparse
            else None
        )
        return EmbeddingBatch(dense=dense, sparse=sparse_out, tokens=tokens)

    def _key(self, text: str, kind: EmbedKind) -> str:
        instruct = (self.instruct or "") if kind == "query" else ""
        parts = [self.inner.name, str(self.inner.dim), str(self.inner.supports_sparse), kind]
        return content_hash("\x1f".join([*parts, instruct, text]))
