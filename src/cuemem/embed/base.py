"""向量化接口。

dense 向量用于语义检索；sparse 向量（词项 → 权重）用于关键词检索。两者在 N4 中融合。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

EmbedKind = Literal["query", "document"]
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, float]]) -> SparseVector:
        """合并重复下标（取最大权重）并按下标排序：向量库要求下标唯一。"""
        merged: dict[int, float] = {}
        for index, value in pairs:
            if index not in merged or value > merged[index]:
                merged[index] = value
        ordered = sorted(merged)
        return cls(indices=ordered, values=[merged[i] for i in ordered])

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class EmbeddingBatch:
    dense: FloatArray  # 形状 (n, dim)，已做 L2 归一化
    sparse: list[SparseVector] | None = None
    tokens: int = 0  # 本次实际计费的 token 数（缓存命中的部分不计）


class TextEmbedder(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def supports_sparse(self) -> bool: ...

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch: ...


def l2_normalize(matrix: NDArray[np.floating]) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)
