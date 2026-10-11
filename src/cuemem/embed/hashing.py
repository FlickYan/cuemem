"""离线哈希向量器：不需要网络和模型，用于单元测试和没有 Key 时的演示。

原理：把文本切成词（英文单词、中文单字及相邻二字组），用特征哈希映射到固定维度。
共享的词越多，向量越相似。它没有真正的语义能力，只保证"字面相近 → 向量相近"。
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from cuemem.embed.base import EmbeddingBatch, EmbedKind, SparseVector, l2_normalize

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_SPARSE_SPACE = 2**31 - 1


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    bigrams = [a + b for a, b in itertools.pairwise(tokens) if _is_cjk(a) and _is_cjk(b)]
    return tokens + bigrams


def _is_cjk(token: str) -> bool:
    return len(token) == 1 and "\u4e00" <= token <= "\u9fff"


def _hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


class HashEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim
        self.supports_sparse = True

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
        dense = np.zeros((len(texts), self.dim), dtype=np.float32)
        sparse: list[SparseVector] = []
        for row, text in enumerate(texts):
            pairs: list[tuple[int, float]] = []
            for token, tf in Counter(tokenize(text)).items():
                h = _hash(token)
                weight = 1.0 + math.log(tf)
                sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
                dense[row, h % self.dim] += sign * weight
                pairs.append((h % _SPARSE_SPACE, weight))
            sparse.append(SparseVector.from_pairs(pairs))
        return EmbeddingBatch(dense=l2_normalize(dense), sparse=sparse, tokens=0)
