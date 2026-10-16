"""向量化：接口、实现（通义 DashScope / OpenAI 兼容接口 / 离线哈希）与缓存。"""

from __future__ import annotations

import httpx2

from cuemem.config import Settings, missing_key_remedy
from cuemem.embed.base import EmbeddingBatch, EmbedKind, SparseVector, TextEmbedder
from cuemem.embed.cache import CachedEmbedder, EmbeddingCache
from cuemem.embed.dashscope import DashScopeTextEmbedder
from cuemem.embed.hashing import HashEmbedder
from cuemem.embed.local import FastEmbedTextEmbedder
from cuemem.embed.multimodal import (
    DashScopeMultimodalEmbedder,
    HashMultimodalEmbedder,
    ImageEmbedder,
)
from cuemem.embed.openai_compat import OpenAICompatibleEmbedder
from cuemem.errors import MissingAPIKeyError
from cuemem.llm.ledger import Ledger

__all__ = [
    "CachedEmbedder",
    "DashScopeTextEmbedder",
    "EmbedKind",
    "EmbeddingBatch",
    "EmbeddingCache",
    "FastEmbedTextEmbedder",
    "HashEmbedder",
    "ImageEmbedder",
    "OpenAICompatibleEmbedder",
    "SparseVector",
    "TextEmbedder",
    "build_embedder",
    "build_image_embedder",
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
    elif cfg.backend == "fastembed":
        inner = FastEmbedTextEmbedder(cfg, cache_dir=settings.paths.models, ledger=ledger)
    else:
        provider = settings.providers[cfg.provider]
        api_key = provider.api_key()
        if api_key is None:
            raise MissingAPIKeyError(
                f"向量化需要环境变量 {provider.api_key_env}，{missing_key_remedy(cfg.provider)}；"
                "离线试用可设置 CUEMEM_EMBEDDING__BACKEND=hash"
            )
        if cfg.backend == "openai":
            inner = OpenAICompatibleEmbedder(
                cfg, api_key, provider.base_url, http_client=http_client, ledger=ledger
            )
        else:
            inner = DashScopeTextEmbedder(cfg, api_key, http_client=http_client, ledger=ledger)
    if not use_cache:
        return inner
    return CachedEmbedder(
        inner, EmbeddingCache(settings.paths.embed_cache), instruct=cfg.query_instruct
    )


def build_image_embedder(
    settings: Settings,
    *,
    ledger: Ledger | None = None,
    http_client: httpx2.Client | None = None,
) -> ImageEmbedder | None:
    """图片向量器（N6）。文本向量用离线哈希时也用哈希；没有 Key 时返回 None（不建图片向量）。"""
    cfg = settings.multimodal_embedding
    if cfg.backend == "none":
        return None
    if cfg.backend == "hash" or settings.embedding.backend == "hash":
        return HashMultimodalEmbedder()
    provider = settings.providers.get(cfg.provider)
    api_key = provider.api_key() if provider is not None else None
    if api_key is None:
        return None
    return DashScopeMultimodalEmbedder(
        cfg,
        api_key,
        http_client=http_client,
        ledger=ledger,
        cache=EmbeddingCache(settings.paths.embed_cache),
    )
