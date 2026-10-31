"""换模型供应商（N12，D-060）：随包发布的默认配置、数据目录、预设、OpenAI 兼容的向量接口、
推理模型的参数差异、本地模型不要 Key、回答模型看不了图时不附图。全部离线。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeAPI
from PIL import Image

from cuemem.config import (
    DEFAULT_CONFIG,
    LOCAL_API_KEY,
    PACKAGE_DIR,
    PathsConfig,
    Settings,
    home_dir,
    load_settings,
    presets,
)
from cuemem.embed import (
    FastEmbedTextEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
    build_image_embedder,
)
from cuemem.errors import EmbeddingAPIError
from cuemem.llm import Ledger, LLMClient, ResponseCache
from cuemem.memory import build_reranker
from cuemem.read.reader import Reader

HELLO = [{"role": "user", "content": "hi"}]
READER_REPLY = '{"answer": "Blue.", "evidence_sufficient": true, "citations": []}'


def test_default_config_ships_inside_the_package() -> None:
    assert DEFAULT_CONFIG.is_relative_to(PACKAGE_DIR)
    assert DEFAULT_CONFIG.is_file()
    assert {"openai", "ollama", "deepseek", "qwen"} <= set(presets())


def test_home_dir_can_be_overridden(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    assert home_dir() == (tmp_path / "home").resolve()
    loaded = load_settings()
    assert loaded.paths.db == (tmp_path / "home" / "data" / "cuemem.sqlite").resolve()


def test_openai_preset_overlays_the_defaults(clean_env: None) -> None:
    loaded = load_settings(preset="openai")
    assert {rc.model for rc in loaded.roles.values()} == {"gpt-5.4-mini"}
    assert loaded.embedding.backend == "openai"
    assert loaded.embedding.dim == 1536
    assert loaded.embedding.sparse is False  # 其余字段仍来自默认配置
    assert loaded.rerank.provider is None
    assert loaded.multimodal_embedding.backend == "none"
    assert "deepseek" in loaded.providers  # 默认的供应商还在，只是用不到
    assert loaded.used_providers() == ["openai"]


def test_preset_from_environment(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUEMEM_PRESET", "ollama")
    loaded = load_settings()
    assert loaded.roles["reader"].model == "llama3.1-8b"
    assert loaded.used_providers() == ["ollama"]
    assert loaded.providers["ollama"].api_key() == LOCAL_API_KEY  # 本地模型不要 Key
    assert loaded.consolidation.caption_images is False


def test_unknown_preset_is_rejected(clean_env: None) -> None:
    with pytest.raises(ValueError, match="openai"):
        load_settings(preset="no-such-preset")


def test_missing_key_errors_name_the_right_remedy(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺 Key 的报错（命令行提示照旧是中文）：pip 安装后要运行的是 cuemem configure（源码仓库里
    才加 uv run）；只设了 OPENAI_API_KEY、缺的却是别家的 Key 时，提示换成 openai 预设。"""
    from cuemem import config
    from cuemem.errors import MissingAPIKeyError

    settings = load_settings(home=tmp_path, peak_policy="allow")
    monkeypatch.setattr(config, "source_checkout", lambda: False)
    with pytest.raises(MissingAPIKeyError) as caught:
        build_embedder(settings)
    assert "DASHSCOPE_API_KEY，请运行 cuemem configure" in str(caught.value)
    assert "openai" not in str(caught.value)
    # 离线试用的提示说清楚要另用数据目录：数据目录会记住向量模型，命令失败了也一样
    assert "单独的数据目录（CUEMEM_HOME）里设置 CUEMEM_EMBEDDING__BACKEND=hash" in str(caught.value)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    with pytest.raises(MissingAPIKeyError, match=r'Memory\(preset="openai"\)'):
        build_embedder(settings)
    with pytest.raises(MissingAPIKeyError, match="--preset openai"):
        LLMClient(settings).chat("reader", HELLO)
    monkeypatch.setattr(config, "source_checkout", lambda: True)
    with pytest.raises(MissingAPIKeyError, match="请运行 uv run cuemem configure"):
        build_embedder(settings)


def test_install_hints_point_at_github_not_pypi(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CueMem 还没有发布到 PyPI：pip 安装后缺可选依赖时的提示从 GitHub 装（"cuemem[x] @ git+…"），
    不按名字到 PyPI 上找，那里的 cuemem 不是这个项目（谁都能注册）。源码仓库里照旧是 uv sync。
    本机向量缺 fastembed 时的报错用同一个提示（以前写死了 pip install "cuemem[local]"）。"""
    import sys

    from cuemem import config

    monkeypatch.setattr(config, "source_checkout", lambda: False)
    for extra in ("ui", "eval", "local", "integrations"):
        hint = config.extra_remedy(extra)
        assert hint == f'pip install "cuemem[{extra}] @ git+https://github.com/FlickYan/cuemem"'
    monkeypatch.setitem(sys.modules, "fastembed", None)  # 当作没装
    embedder = build_embedder(load_settings(preset="deepseek", home=tmp_path), use_cache=False)
    with pytest.raises(EmbeddingAPIError) as caught:
        embedder.embed(["x"], "document")
    assert config.extra_remedy("local") in str(caught.value)
    monkeypatch.setattr(config, "INSTALL_URL", None)  # 发布到 PyPI 之后
    assert config.extra_remedy("ui") == 'pip install "cuemem[ui]"'
    monkeypatch.setattr(config, "source_checkout", lambda: True)
    assert config.extra_remedy("local") == "uv sync --extra local"
    with pytest.raises(EmbeddingAPIError, match=r"fastembed：uv sync --extra local$"):
        embedder.embed(["x"], "document")


def test_install_hints_in_the_docker_image_rebuild_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """仓库的 Docker 镜像里（依赖在 /opt/venv，没有 pip，镜像里也没有 git）缺可选依赖时，提示带上
    它重新构建镜像（默认的 ui、local 照样带上）。以前提示从 GitHub pip install，在容器里走不通：
    系统的 pip 装进别的环境，也没有 git（第四轮复审）。"""
    import importlib.util
    import sys

    from cuemem import config

    monkeypatch.setattr(config, "source_checkout", lambda: False)
    monkeypatch.setattr(sys, "prefix", "/opt/venv")
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "pip" else real(name, *a)
    )
    assert config.docker_image()
    rebuild = 'docker build --build-arg EXTRAS="--extra ui --extra local'
    assert config.extra_remedy("ui").startswith(rebuild + '" -t cuemem .')
    assert config.extra_remedy("local").startswith(rebuild + '" -t cuemem .')
    assert config.extra_remedy("eval").startswith(rebuild + ' --extra eval" -t cuemem .')
    monkeypatch.setattr(sys, "prefix", "/home/me/.venv")  # 别处的虚拟环境照旧从 GitHub 装
    assert not config.docker_image()
    assert config.extra_remedy("ui").startswith('pip install "cuemem[ui] @ ')


@pytest.fixture
def openai_settings(clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai")
    loaded = load_settings(preset="openai")
    return loaded.model_copy(
        update={"paths": PathsConfig().resolve_against(tmp_path), "peak_policy": "allow"}
    )


def test_reasoning_models_get_their_own_parameters(
    openai_settings: Settings, fake_api: FakeAPI
) -> None:
    llm = LLMClient(openai_settings, cache=ResponseCache(None), http_client=fake_api.client())
    llm.chat("agent", HELLO)
    request = fake_api.chat_requests[-1]
    assert request["model"] == "gpt-5.4-mini"
    assert "temperature" not in request  # 推理模型只接受默认温度
    assert request["max_completion_tokens"] == 4000
    assert "max_tokens" not in request
    assert request["reasoning_effort"] == "low"


def test_classic_models_keep_their_request_unchanged(settings: Settings, fake_api: FakeAPI) -> None:
    # 默认配置的请求逐字不变，已有的响应缓存照常命中
    llm = LLMClient(settings, cache=ResponseCache(None), http_client=fake_api.client())
    llm.chat("agent", HELLO)
    request = fake_api.chat_requests[-1]
    assert request["temperature"] == 0.0
    assert request["max_tokens"] == 1200


def test_openai_compatible_embedder(openai_settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.embed_dim = 1536
    embedder = build_embedder(openai_settings, http_client=fake_api.client(), use_cache=False)
    assert isinstance(embedder, OpenAICompatibleEmbedder)
    batch = embedder.embed([f"text {i}" for i in range(70)], "query")
    assert len(fake_api.embed_requests) == 2  # batch_size = 64
    first = fake_api.embed_requests[0]
    assert first == {"model": "text-embedding-3-small", "input": first["input"]}
    assert batch.dense.shape == (70, 1536)
    assert batch.sparse is None
    assert batch.tokens == 5 * 70
    # 响应顺序被打乱过：按 index 排回原位（文本长度不同，向量的第一维不同）
    assert batch.dense[0, 0] != batch.dense[10, 0]


def test_openai_embedder_options(openai_settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.embed_dim = 1536
    cfg = openai_settings.embedding.model_copy(
        update={"send_dimensions": True, "query_instruct": "Find the memory"}
    )
    embedder = OpenAICompatibleEmbedder(
        cfg, "sk-test", "https://example.test/v1/", http_client=fake_api.client()
    )
    embedder.embed(["where do I live"], "query")
    embedder.embed(["I live in Berlin"], "document")
    query, document = fake_api.embed_requests
    assert query["dimensions"] == 1536
    assert query["input"] == ["Instruct: Find the memory\nQuery: where do I live"]
    assert document["input"] == ["I live in Berlin"]


def test_openai_embedder_checks_the_dimension(openai_settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.embed_dim = 8
    embedder = build_embedder(openai_settings, http_client=fake_api.client(), use_cache=False)
    with pytest.raises(EmbeddingAPIError, match=r"embedding\.dim"):
        embedder.embed(["hello"], "document")


def test_optional_components_are_skipped(openai_settings: Settings) -> None:
    llm = LLMClient(openai_settings)
    assert build_reranker(openai_settings, llm=llm, ledger=Ledger(None)) is None
    assert build_image_embedder(openai_settings) is None


def test_reader_drops_pictures_for_text_only_models(
    clean_env: None, tmp_path: Path, fake_api: FakeAPI
) -> None:
    loaded = load_settings(preset="ollama")
    loaded = loaded.model_copy(update={"paths": PathsConfig().resolve_against(tmp_path)})
    fake_api.chat_replies = [READER_REPLY]
    llm = LLMClient(loaded, cache=ResponseCache(None), http_client=fake_api.client())
    assert not llm.vision("reader")
    picture = tmp_path / "p.png"
    Image.new("RGB", (8, 8), "blue").save(picture)
    answer = Reader(llm)._read(
        "What color?", "(none)", datetime(2026, 1, 1), pictures=[("Picture 1", str(picture))]
    )
    assert answer.answer == "Blue."
    content = fake_api.chat_requests[-1]["messages"][-1]["content"]
    assert isinstance(content, str)  # 没有图片内容块


class _FakeFastEmbed:
    """fastembed.TextEmbedding 的替身：按文本长度生成向量，记下收到的文本。"""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.seen: list[str] = []

    def embed(self, texts: list[str], batch_size: int = 32) -> Iterator[list[float]]:
        for text in texts:
            self.seen.append(text)
            yield [float(len(text) % 5 + 1)] + [0.5] * (self.dim - 1)


def test_deepseek_preset_runs_embeddings_locally(clean_env: None, tmp_path: Path) -> None:
    loaded = load_settings(preset="deepseek", home=tmp_path)
    assert {rc.model for rc in loaded.roles.values()} == {"deepseek-flash"}
    assert loaded.embedding.backend == "fastembed" and loaded.embedding.dim == 384
    assert loaded.used_providers() == ["deepseek"]  # 向量不需要任何供应商的 Key
    assert build_image_embedder(loaded) is None and loaded.rerank.provider is None
    embedder = build_embedder(loaded, use_cache=False)
    assert isinstance(embedder, FastEmbedTextEmbedder)
    fake = _FakeFastEmbed(384)
    embedder._model = fake  # 不下载模型
    batch = embedder.embed(["I live in Berlin", ""], "query")
    assert batch.dense.shape == (2, 384) and batch.sparse is None
    assert fake.seen == ["I live in Berlin", " "]  # 空文本用空格占位；不加查询指令
    assert abs(float(np.linalg.norm(batch.dense[0])) - 1.0) < 1e-5
    with pytest.raises(EmbeddingAPIError, match=r"embedding\.dim"):
        embedder._model = _FakeFastEmbed(8)
        embedder.embed(["x"], "document")
