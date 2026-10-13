"""配置加载。

优先级（高 → 低）：显式参数 > 环境变量（前缀 CUEMEM_，嵌套用 __）> YAML 配置文件。
API Key 不写进配置文件，而是从环境变量或项目根目录的 .env 中读取（用 cuemem configure 填写）。
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# QwenCloud（阿里模型平台的国际品牌）的原生向量接口：只有原生接口能同时返回 dense 与 sparse 向量
QWENCLOUD_EMBEDDING_URL = (
    "https://maas.qwencloudapi.com/api/v1/services/embeddings/text-embedding/text-embedding"
)

QWENCLOUD_RERANK_URL = (
    "https://maas.qwencloudapi.com/api/v1/services/rerank/text-rerank/text-rerank"
)
QWENCLOUD_MULTIMODAL_EMBEDDING_URL = (
    "https://maas.qwencloudapi.com/api/v1/services/embeddings/multimodal-embedding/"
    "multimodal-embedding"
)

# YAML 文件路径通过 ContextVar 传给配置源，避免修改全局环境变量
_config_path: ContextVar[Path] = ContextVar("cuemem_config_path", default=DEFAULT_CONFIG)

Currency = Literal["CNY", "USD"]


class PeakHours(BaseModel):
    """分时价格：这些时段是高峰，价格是其他时段的 1 / off_peak_factor 倍。

    配置里的模型标价按高峰计（D-010）；非高峰时段的实际花费 = 标价 × off_peak_factor。
    """

    timezone: str = "Asia/Shanghai"
    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])  # 周一 = 0
    windows: list[str] = Field(default_factory=lambda: ["09:00-12:00", "14:00-18:00"])
    off_peak_factor: float = 0.5


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str
    max_concurrency: int = 8
    timeout_s: float = 120.0
    max_retries: int = 3
    peak_hours: PeakHours | None = None  # 没有分时价格的供应商为空

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None


class Price(BaseModel):
    """每百万 token 的名义标价：高峰时段、未命中缓存。"""

    input: float
    output: float
    cached_input: float | None = None  # 供应商侧缓存命中价；None 表示与 input 相同


class ModelSpec(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider: str
    model_id: str
    price: Price
    currency: Currency = "USD"
    context_window: int | None = None
    vision: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)  # 供应商特有参数，如关闭思考模式


class RoleConfig(BaseModel):
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None


class EmbeddingConfig(BaseModel):
    backend: Literal["dashscope", "hash"] = "dashscope"
    provider: str = "qwen"
    model: str = "text-embedding-v4"
    dim: int = 1024
    sparse: bool = True
    batch_size: int = 10
    max_chars: int = 6000
    query_instruct: str | None = None
    native_url: str = QWENCLOUD_EMBEDDING_URL
    price_per_m: float = 0.07
    currency: Currency = "USD"
    timeout_s: float = 60.0
    max_concurrency: int = 2  # 同时在途的向量请求数；QwenCloud 按分钟限额，并发过高会被 429 限流
    max_attempts: int = 6  # 429 / 5xx / 网络错误的最大尝试次数（指数退避，单次最多等 30 秒）


class MultimodalEmbeddingConfig(BaseModel):
    """图片向量（N6）：图片与文字映射到同一个向量空间，可以以文搜图、以图搜图。"""

    backend: Literal["dashscope", "hash"] = "dashscope"
    provider: str = "qwen"
    model: str = "tongyi-embedding-vision-plus"
    dim: int = 1152
    url: str = QWENCLOUD_MULTIMODAL_EMBEDDING_URL
    price_per_m: float = 0.108  # 美元 / 百万 tokens（国际站标价，图片与文字同价）
    currency: Currency = "USD"
    batch_size: int = 8
    max_side: int = 768  # 送去向量化的图片最长边（像素）
    timeout_s: float = 60.0
    max_concurrency: int = 2
    max_attempts: int = 6


class RerankConfig(BaseModel):
    """重排模型（N5，L2 用）：QwenCloud 的 qwen3-rerank，原生接口。"""

    provider: str = "qwen"
    model: str = "qwen3-rerank"
    url: str = QWENCLOUD_RERANK_URL
    price_per_m: float = 0.10  # 美元 / 百万 tokens（国际站标价）
    currency: Currency = "USD"
    max_chars: int = 4000  # 每段文本最多送多少字符
    timeout_s: float = 60.0
    max_concurrency: int = 4
    max_attempts: int = 4


class PathsConfig(BaseModel):
    db: Path = Path("data/cuemem.sqlite")
    qdrant: Path = Path("data/qdrant")
    llm_cache: Path = Path("data/cache/llm.sqlite")
    embed_cache: Path = Path("data/cache/embed.sqlite")
    ledger: Path = Path("data/ledger.sqlite")
    traces: Path = Path("runs/traces")
    datasets: Path = Path("data/datasets")  # 下载的评测数据集
    eval_runs: Path = Path("runs/eval")  # 每次评测运行一个子目录
    eval_spaces: Path = Path("data/eval-spaces")  # 评测用的记忆空间（写入与巩固的结果），跨运行复用
    media: Path = Path("data/media")  # 原图，按内容哈希存放（N6）

    def resolve_against(self, root: Path) -> PathsConfig:
        """把相对路径解析为以 root 为基准的绝对路径。"""
        resolved = {name: p if p.is_absolute() else root / p for name, p in self}
        return PathsConfig.model_validate(resolved)


class VectorStoreConfig(BaseModel):
    url: str | None = None  # 为空用本地模式（paths.qdrant）；填 http://localhost:6333 改用服务端


class RetrievalConfig(BaseModel):
    naive_k: int = 8
    fact_k: int = 10  # 事实增强检索：放进上下文的事实条数（不含沿版本链补进来的）
    fact_pool: int = 50  # 事实增强检索：一次向量检索取回的候选数（轮与事实合计）
    # 检索控制器的模型文件：相对路径相对 src/cuemem/read/。默认是 v2（N11）；
    # 复现 N7 / N9 的数字时用 controller_model_v1.json
    controller: str = "controller_model.json"


class ConsolidationConfig(BaseModel):
    """异步巩固（N3）的参数。相似度阈值针对 text-embedding-v4 的余弦相似度。"""

    assistant_chars: int | None = 600  # 抽取时每条助手回复最多保留的字符数；None 表示不截断
    max_session_chars: int = 24_000  # 单次抽取的会话文本上限，超过时按轮次切成多个窗口
    workers: int = 8  # 同一空间内并行抽取的会话数
    fallback_model: str | None = "deepseek-flash"  # 内容审核误拒时改用的模型（另一家供应商）
    duplicate_threshold: float = 0.93  # 陈述相似度超过它视为同一条事实（MERGE）
    conflict_threshold: float = 0.75  # 单值事实与已有事实相似度超过它，交给 LLM 判断是否冲突
    multi_conflict_threshold: float = 0.88  # 多值事实要更相似才交给 LLM
    entity_merge_threshold: float = 0.90  # 实体描述相似度超过它直接合并
    entity_gray_threshold: float = 0.75  # 介于两个阈值之间的实体交给 LLM 确认
    profile_max_chars: int = 1600  # 核心记忆的长度上限（约 400 tokens）
    profile_min_importance: int = 4  # 进入核心记忆的最低重要度（1–5）
    caption_images: bool = True  # 巩固时为图片生成描述（上下文感知与孤立两种，N6）
    caption_max_side: int = 768  # 送去描述的图片最长边（像素）


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CUEMEM_", env_nested_delimiter="__", extra="ignore"
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm_cache_mode: Literal["read_write", "read_only", "off"] = "read_write"
    # 高峰时段遇到需要真实调用的请求：wait 暂停到高峰结束（批量任务默认）；
    # warn 照常调用但记录提醒；allow 照常调用
    peak_policy: Literal["wait", "warn", "allow"] = "wait"
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    multimodal_embedding: MultimodalEmbeddingConfig = Field(
        default_factory=MultimodalEmbeddingConfig
    )
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=_config_path.get())
        return (init_settings, env_settings, yaml_source)

    @model_validator(mode="after")
    def _check_references(self) -> Settings:
        for name, spec in self.models.items():
            if spec.provider not in self.providers:
                raise ValueError(f"模型 {name} 引用了未定义的供应商 {spec.provider}")
        for role, rc in self.roles.items():
            if rc.model not in self.models:
                raise ValueError(f"角色 {role} 引用了未定义的模型 {rc.model}")
        if self.embedding.backend == "dashscope" and self.embedding.provider not in self.providers:
            raise ValueError(f"向量配置引用了未定义的供应商 {self.embedding.provider}")
        if self.rerank.provider not in self.providers:
            raise ValueError(f"重排配置引用了未定义的供应商 {self.rerank.provider}")
        return self


def env_file_path(root: Path = PROJECT_ROOT) -> Path:
    """本项目读写的 .env 位置：环境变量 CUEMEM_ENV_FILE 优先，否则为项目根目录下的 .env。"""
    return Path(os.environ.get("CUEMEM_ENV_FILE") or root / ".env")


def load_settings(
    config_path: Path | str | None = None,
    *,
    env_file: Path | None = None,
    **overrides: Any,
) -> Settings:
    """加载配置；路径字段相对于项目根目录（配置文件所在目录的上一级）解析。

    已存在的环境变量优先于 .env 中的值。
    """
    path = Path(config_path or os.environ.get("CUEMEM_CONFIG") or DEFAULT_CONFIG).resolve()
    root = path.parent.parent
    load_dotenv(env_file or env_file_path(root), override=False)
    token = _config_path.set(path)
    try:
        settings = Settings(**overrides)
    finally:
        _config_path.reset(token)
    return settings.model_copy(update={"paths": settings.paths.resolve_against(root)})
