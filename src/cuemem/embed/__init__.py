"""向量化：接口、实现（通义 DashScope / 离线哈希）与缓存。"""

from __future__ import annotations

import httpx2

from cuemem.config import Settings
from cuemem.embed.base import EmbeddingBatch, EmbedKind, SparseVector, TextEmbedder
from cuemem.embed.cache import CachedEmbedder, EmbeddingCache
from cuemem.embed.dashscope import DashScopeTextEmbedder
from cuemem.embed.hashing import HashEmbedder
from cuemem.errors import MissingAPIKeyError
from cuemem.llm.ledger import Ledger

__all__ = [
    "CachedEmbedder",
    "DashScopeTextEmbedder",
    "EmbedKind",
    "EmbeddingBatch",
    "EmbeddingCache",
    "HashEmbedder",
    "SparseVector",
    "TextEmbedder",
    "build_embedder",
]


def build_embedder(
    settings: Settings,
    *,
    ledger: Ledger | None = None,
    http_client: httpx2.Client | None = None,
    use_cache: bool = True,
) -> TextEmbedder:
    """按配置构造向量器；默认外面包一层内容哈希缓存。"""
    cfg = settings.embedding
    inner: TextEmbedder
    if cfg.backend == "hash":
        inner = HashEmbedder(cfg.dim)
    else:
        provider = settings.providers[cfg.provider]
        api_key = provider.api_key()
        if api_key is None:
            raise MissingAPIKeyError(
                f"向量化需要环境变量 {provider.api_key_env}，请运行 uv run cuemem configure；"
                "离线试用可设置 CUEMEM_EMBEDDING__BACKEND=hash"
            )
        inner = DashScopeTextEmbedder(cfg, api_key, http_client=http_client, ledger=ledger)
    if not use_cache:
        return inner
    return CachedEmbedder(
        inner, EmbeddingCache(settings.paths.embed_cache), instruct=cfg.query_instruct
    )
