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
    "image_vectors_enabled",
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
    cache_best_effort: bool = False,
    parallel: bool = False,
) -> TextEmbedder:
    """按配置构造向量器；默认外面包一层内容哈希缓存。cache_best_effort：缓存写不进去（磁盘满了、
    只读）时只记警告（见 EmbeddingCache），MemorySystem.from_settings 打开，评测与脚本不打开。

    parallel：一次调用要发几个请求时同时发（最多 embedding.max_concurrency 个在途，见
    DashScopeTextEmbedder._send_all；本地的 hash、fastembed 后端没有请求，不受影响）。同样只有
    MemorySystem.from_settings 打开：评测与脚本的请求一个接一个，顺序照旧。"""
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
                "离线试用可在单独的数据目录（CUEMEM_HOME）里设置 CUEMEM_EMBEDDING__BACKEND=hash"
                "（数据目录会记住向量模型）"
            )
        if cfg.backend == "openai":
            from cuemem.embed.openai_compat import OpenAICompatibleEmbedder

            inner = OpenAICompatibleEmbedder(
                cfg,
                api_key,
                provider.base_url,
                http_client=http_client,
                ledger=ledger,
                parallel=parallel,
            )
        else:
            from cuemem.embed.dashscope import DashScopeTextEmbedder

            inner = DashScopeTextEmbedder(
                cfg, api_key, http_client=http_client, ledger=ledger, parallel=parallel
            )
    if not use_cache:
        return inner
    from cuemem.store.files import opening

    with opening(settings.paths.embed_cache):  # 打不开时 DatabaseOpenError（一行说明）
        cache = EmbeddingCache(settings.paths.embed_cache, best_effort=cache_best_effort)
    return CachedEmbedder(inner, cache, instruct=cfg.query_instruct)


def image_vectors_enabled(settings: Settings) -> bool:
    """建不建图片向量：与 build_image_embedder 返回不返回 None 的判断相同（改一处时两处一起改），
    只是不打开向量缓存、不建向量器。doctor --vectors 据此决定比不比图片向量。"""
    cfg = settings.multimodal_embedding
    if cfg.backend == "none":
        return False
    if cfg.backend == "hash" or settings.embedding.backend == "hash":
        return True
    provider = settings.providers.get(cfg.provider)
    return provider is not None and provider.api_key() is not None


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
