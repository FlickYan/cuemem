"""向量化：接口、实现（通义 DashScope / OpenAI 兼容接口 / 离线哈希）与缓存。

按需导入（与 cuemem/__init__.py 相同）：以前这里把每个后端都导入一遍，经图片向量 → 媒体库 →
store 包连带加载了 Qdrant 客户端，只用接口（embed.base）也要约 1 秒。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from cuemem.config import Settings, missing_key_remedy
from cuemem.errors import MissingAPIKeyError

if TYPE_CHECKING:
    import httpx2

    from cuemem.embed.base import EmbeddingBatch, EmbedKind, SparseVector, TextEmbedder
    from cuemem.embed.cache import CachedEmbedder, EmbeddingCache
    from cuemem.embed.dashscope import DashScopeTextEmbedder
    from cuemem.embed.hashing import HashEmbedder
    from cuemem.embed.local import FastEmbedTextEmbedder
    from cuemem.embed.multimodal import ImageEmbedder
    from cuemem.embed.openai_compat import OpenAICompatibleEmbedder
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
_SOURCES = {
    "CachedEmbedder": "cuemem.embed.cache",
    "DashScopeMultimodalEmbedder": "cuemem.embed.multimodal",
    "DashScopeTextEmbedder": "cuemem.embed.dashscope",
    "EmbedKind": "cuemem.embed.base",
    "EmbeddingBatch": "cuemem.embed.base",
    "EmbeddingCache": "cuemem.embed.cache",
    "FastEmbedTextEmbedder": "cuemem.embed.local",
    "HashEmbedder": "cuemem.embed.hashing",
    "HashMultimodalEmbedder": "cuemem.embed.multimodal",
    "ImageEmbedder": "cuemem.embed.multimodal",
    "OpenAICompatibleEmbedder": "cuemem.embed.openai_compat",
    "SparseVector": "cuemem.embed.base",
    "TextEmbedder": "cuemem.embed.base",
}


def __getattr__(name: str) -> Any:
    if name in _SOURCES:
        return getattr(import_module(_SOURCES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_embedder(
    settings: Settings,
    *,
    ledger: Ledger | None = None,
    http_client: httpx2.Client | None = None,
    use_cache: bool = True,
) -> TextEmbedder:
    """按配置构造向量器；默认外面包一层内容哈希缓存。"""
    from cuemem.embed.cache import CachedEmbedder, EmbeddingCache

    cfg = settings.embedding
    inner: TextEmbedder
    if cfg.backend == "hash":
        from cuemem.embed.hashing import HashEmbedder

        inner = HashEmbedder(cfg.dim)
    elif cfg.backend == "fastembed":
        from cuemem.embed.local import FastEmbedTextEmbedder

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
            from cuemem.embed.openai_compat import OpenAICompatibleEmbedder

            inner = OpenAICompatibleEmbedder(
                cfg, api_key, provider.base_url, http_client=http_client, ledger=ledger
            )
        else:
            from cuemem.embed.dashscope import DashScopeTextEmbedder

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
    cache: EmbeddingCache | None = None,
) -> ImageEmbedder | None:
    """图片向量器（N6）。文本向量用离线哈希时也用哈希；没有 Key 时返回 None（不建图片向量）。

    cache：向量缓存，缺省另开一个连接（同一个文件）。MemorySystem 传文本向量器的那个：另开的连接
    每写一次缓存（以文搜图时没见过的问题、写入带图的发言），文本那个连接读到的 data_version 就
    变一次，巩固就当向量缓存被别人改过，丢掉所有空间的镜像（见 Consolidator._load_states）。"""
    from cuemem.embed.cache import EmbeddingCache
    from cuemem.embed.multimodal import DashScopeMultimodalEmbedder, HashMultimodalEmbedder

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
        cache=cache if cache is not None else EmbeddingCache(settings.paths.embed_cache),
    )
