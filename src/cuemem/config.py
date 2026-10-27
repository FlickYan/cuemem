"""配置加载。

优先级（高 → 低）：显式参数 > 环境变量（前缀 CUEMEM_，嵌套用 __）> 预设 > YAML 配置文件。
API Key 不写进配置文件，而是从环境变量或数据目录的 .env 中读取（用 cuemem configure 填写）。

默认配置随包发布（cuemem/configs/default.yaml），pip 安装后也能用（N12，D-060）。
数据目录：CUEMEM_HOME 优先；在源码仓库里运行时是仓库根目录（与之前一样），否则是 ~/.cuemem。
预设（presets/*.yaml）叠加在默认配置上，换掉模型供应商：openai、ollama（本地，不要 Key）。
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

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]  # 源码仓库的根目录（src 布局）；pip 安装后没有意义
DEFAULT_CONFIG = PACKAGE_DIR / "configs" / "default.yaml"
PRESETS_DIR = PACKAGE_DIR / "configs" / "presets"

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

# YAML 文件路径（默认配置在前、预设在后，深度合并）通过 ContextVar 传给配置源，避免修改全局环境变量
_config_paths: ContextVar[tuple[Path, ...]] = ContextVar(
    "cuemem_config_paths", default=(DEFAULT_CONFIG,)
)

Currency = Literal["CNY", "USD"]


class PeakHours(BaseModel):
    """分时价格：这些时段是高峰，价格是其他时段的 1 / off_peak_factor 倍。

    配置里的模型标价按高峰计（D-010）；非高峰时段的实际花费 = 标价 × off_peak_factor。
    """

    timezone: str = "Asia/Shanghai"
    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])  # 周一 = 0
    windows: list[str] = Field(default_factory=lambda: ["09:00-12:00", "14:00-18:00"])
    off_peak_factor: float = 0.5


# 不需要 Key 的供应商（本地模型）用的占位 Key：OpenAI SDK 要求 Key 非空，本地服务会忽略它
LOCAL_API_KEY = "local"


class ProviderConfig(BaseModel):
    base_url: str
    api_key_env: str | None = None  # 为空表示不需要 Key（Ollama、vLLM 等本地服务，N12）
    max_concurrency: int = 8
    timeout_s: float = 120.0
    # 限流、5xx、连不上时重试几次（LLMClient 照 OpenAI SDK 的规则退避）；超时只在调用方没有
    # 备用模型时重试，有备用模型时超时一次就换模型
    max_retries: int = 3
    peak_hours: PeakHours | None = None  # 没有分时价格的供应商为空

    def api_key(self) -> str | None:
        if self.api_key_env is None:
            return LOCAL_API_KEY
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
    # 推理模型（如 OpenAI 的 GPT-5 系列）只接受默认温度：请求里不带 temperature（N12）
    fixed_temperature: bool = False
    # 输出长度上限的参数名：OpenAI 的新模型只认 max_completion_tokens
    max_tokens_param: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"


class RoleConfig(BaseModel):
    model: str
    temperature: float = 0.0
    max_tokens: int | None = None


class EmbeddingConfig(BaseModel):
    # dashscope：通义原生接口（dense + sparse）；openai：OpenAI 兼容的 /embeddings 接口
    # （OpenAI、Ollama、vLLM、硅基流动……，地址取供应商的 base_url，N12）；
    # fastembed：本机 ONNX 推理，不要 Key（可选依赖 cuemem[local]，N12）；hash：离线哈希向量
    backend: Literal["dashscope", "openai", "fastembed", "hash"] = "dashscope"
    provider: str = "qwen"
    model: str = "text-embedding-v4"
    dim: int = 1024
    sparse: bool = True
    batch_size: int = 10
    max_chars: int = 6000
    query_instruct: str | None = None
    # openai 后端：请求里带 dimensions（只有支持截断维度的模型接受，如 text-embedding-3）
    send_dimensions: bool = False
    native_url: str = QWENCLOUD_EMBEDDING_URL
    price_per_m: float = 0.07
    currency: Currency = "USD"
    timeout_s: float = 60.0
    max_concurrency: int = 2  # 同时在途的向量请求数；QwenCloud 按分钟限额，并发过高会被 429 限流
    max_attempts: int = 6  # 429 / 5xx / 网络错误的最大尝试次数（指数退避，单次最多等 30 秒）


class MultimodalEmbeddingConfig(BaseModel):
    """图片向量（N6）：图片与文字映射到同一个向量空间，可以以文搜图、以图搜图。

    none：不建图片向量（没有图文向量模型的供应商，N12）；图片仍然保存、生成描述、按描述检索。
    """

    backend: Literal["dashscope", "hash", "none"] = "dashscope"
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
    """重排模型（N5，L2 用）：QwenCloud 的 qwen3-rerank，原生接口。provider 为空时 L2 不重排。"""

    provider: str | None = "qwen"
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
    models: Path = Path("data/models")  # 本机模型（fastembed 下载的向量模型，N12）

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
    # 检索控制器的模型文件：相对路径相对 src/cuemem/read/。默认是 v3（N11 之后，D-058）；
    # 复现 N7 / N9 的数字时用 controller_model_v1.json，复现 N11 时用 controller_model_v2.json
    controller: str = "controller_model.json"


UnknownModel = Literal["remap", "reject"]


class ProxyConfig(BaseModel):
    """记忆代理（cuemem proxy 与 cuemem serve 的 /v1，D-062）。"""

    # 请求的 model 不是模型目录里的别名或模型 ID 时：remap 改用 agent 角色的模型（响应头
    # X-CueMem-Model 写明实际用的模型，日志里记一条警告）；reject 返回 400（model_not_found）
    unknown_model: UnknownModel = "remap"


class ConsolidationConfig(BaseModel):
    """异步巩固（N3）的参数。相似度阈值针对 text-embedding-v4 的余弦相似度。"""

    assistant_chars: int | None = 600  # 抽取时每条助手回复最多保留的字符数；None 表示不截断
    max_session_chars: int = 24_000  # 单次抽取的会话文本上限，超过时按轮次切成多个窗口
    workers: int = 8  # 同一空间内并行抽取的会话数
    # 后台巩固同时处理的空间数（同一空间仍只有一个在巩固）；与 workers 一样不计入巩固配置的指纹
    background_workers: int = Field(default=4, ge=1)
    fallback_model: str | None = "deepseek-flash"  # 内容审核误拒时改用的模型（另一家供应商）
    duplicate_threshold: float = 0.93  # 陈述相似度超过它视为同一条事实（MERGE）
    conflict_threshold: float = 0.75  # 单值事实与已有事实相似度超过它，交给 LLM 判断是否冲突
    multi_conflict_threshold: float = 0.88  # 多值事实要更相似才交给 LLM
    entity_merge_threshold: float = 0.90  # 实体描述相似度超过它直接合并
    entity_gray_threshold: float = 0.75  # 介于两个阈值之间的实体交给 LLM 确认
    profile_max_chars: int = 1600  # 核心记忆的长度上限（约 400 tokens）
    profile_min_importance: int = 4  # 进入核心记忆的最低重要度（1–5）
    # 应用给抽取的补充说明（N12，对标 Mem0 的 custom_instructions、Graphiti 的
    # custom_extraction_instructions）：例如"导购场景：记下尺码、预算、品牌偏好，不记寒暄"。
    # 改了会触发重建派生记忆（巩固配置的指纹变了）
    instructions: str | None = None
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
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    # trace 保留几天（cuemem serve 与 cuemem proxy 启动时与之后每天删一遍更早的 trace 文件，连同
    # 其中提问的查询向量，见 MemorySystem.prune_traces）；None 一直留着
    traces_retention_days: int | None = Field(default=None, ge=1)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(
            settings_cls, yaml_file=list(_config_paths.get()), deep_merge=True
        )
        return (init_settings, env_settings, yaml_source)

    @model_validator(mode="after")
    def _check_references(self) -> Settings:
        for name, spec in self.models.items():
            if spec.provider not in self.providers:
                raise ValueError(f"模型 {name} 引用了未定义的供应商 {spec.provider}")
        for role, rc in self.roles.items():
            if rc.model not in self.models:
                raise ValueError(f"角色 {role} 引用了未定义的模型 {rc.model}")
        local = self.embedding.backend in ("hash", "fastembed")
        if not local and self.embedding.provider not in self.providers:
            raise ValueError(f"向量配置引用了未定义的供应商 {self.embedding.provider}")
        if self.rerank.provider is not None and self.rerank.provider not in self.providers:
            raise ValueError(f"重排配置引用了未定义的供应商 {self.rerank.provider}")
        return self

    def used_providers(self) -> list[str]:
        """实际用到的供应商：角色的模型、向量、图片向量、重排引用到的（按配置里的顺序）。

        预设叠加在默认配置上，默认的供应商仍在配置里但用不到；configure 只问用得到的 Key。
        """
        used = {self.models[rc.model].provider for rc in self.roles.values()}
        if self.embedding.backend not in ("hash", "fastembed"):
            used.add(self.embedding.provider)
        if self.multimodal_embedding.backend == "dashscope":
            used.add(self.multimodal_embedding.provider)
        if self.rerank.provider is not None:
            used.add(self.rerank.provider)
        return [name for name in self.providers if name in used]


def source_checkout() -> bool:
    """是否在源码仓库里运行（uv sync 的可编辑安装也算）：数据、评测与演示文件都在仓库里。"""
    return (PROJECT_ROOT / "pyproject.toml").is_file() and (PACKAGE_DIR.parent.name == "src")


def home_dir() -> Path:
    """数据目录：配置里的相对路径以它为基准，.env 也放在这里。

    CUEMEM_HOME 优先；在源码仓库里运行时是仓库根目录（data/、runs/ 与之前一样）；
    pip 安装后是 ~/.cuemem，不往 site-packages 里写东西。
    """
    if env := os.environ.get("CUEMEM_HOME"):
        return Path(env).expanduser().resolve()
    return PROJECT_ROOT if source_checkout() else Path.home() / ".cuemem"


def env_file_path(root: Path | None = None) -> Path:
    """本项目读写的 .env 位置：环境变量 CUEMEM_ENV_FILE 优先，否则为数据目录下的 .env。"""
    return Path(os.environ.get("CUEMEM_ENV_FILE") or (root or home_dir()) / ".env")


def cli_command(args: str) -> str:
    """提示里让用户运行的命令：源码仓库里是 uv run cuemem …，pip 安装后直接是 cuemem …。"""
    return f"{'uv run ' if source_checkout() else ''}cuemem {args}"


def extra_remedy(extra: str) -> str:
    """缺可选依赖时的安装命令：源码仓库里用 uv sync --extra，pip 安装后用 pip install。"""
    return f"uv sync --extra {extra}" if source_checkout() else f'pip install "cuemem[{extra}]"'


def missing_key_remedy(provider: str) -> str:
    """缺 Key 时的解决办法。只设了 OPENAI_API_KEY、缺的却是别家的 Key 时，多半是没换预设。"""
    remedy = f"请运行 {cli_command('configure')}"
    if provider != "openai" and os.environ.get("OPENAI_API_KEY"):
        remedy += (
            f"；只有 OpenAI 的 Key 时用 openai 预设：{cli_command('--preset openai …')}、"
            'Memory(preset="openai")，或设 CUEMEM_PRESET=openai'
        )
    return remedy


def presets() -> list[str]:
    """可用的预设名（presets/*.yaml）。"""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def preset_path(name: str) -> Path:
    """预设名或 YAML 文件路径 → 文件路径。"""
    path = PRESETS_DIR / f"{name}.yaml"
    if path.is_file():
        return path
    if Path(name).expanduser().is_file():
        return Path(name).expanduser().resolve()
    raise ValueError(f"没有预设 {name}；可用的有：{', '.join(presets())}")


def load_settings(
    config_path: Path | str | None = None,
    *,
    preset: str | None = None,
    home: Path | str | None = None,
    env_file: Path | None = None,
    **overrides: Any,
) -> Settings:
    """加载配置；路径字段相对于数据目录解析（home 不填时为 home_dir()）。

    preset：叠加在配置文件上的预设（名字或 YAML 路径），不填时看环境变量 CUEMEM_PRESET。
    overrides：嵌套的字典与 YAML 深度合并（例如 embedding={"model": ...} 只改这一个字段）。
    已存在的环境变量优先于 .env 中的值。
    """
    path = Path(config_path or os.environ.get("CUEMEM_CONFIG") or DEFAULT_CONFIG).resolve()
    root = Path(home).expanduser().resolve() if home is not None else home_dir()
    load_dotenv(env_file or env_file_path(root), override=False)
    files = [path]
    if chosen := preset or os.environ.get("CUEMEM_PRESET"):
        files.append(preset_path(chosen))
    token = _config_paths.set(tuple(files))
    try:
        settings = Settings(**overrides)
    finally:
        _config_paths.reset(token)
    return settings.model_copy(update={"paths": settings.paths.resolve_against(root)})
