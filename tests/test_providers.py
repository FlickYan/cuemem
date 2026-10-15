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
