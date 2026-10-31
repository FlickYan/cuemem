from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from cuemem.config import (
    DEFAULT_CONFIG,
    PRESETS_DIR,
    PROJECT_ROOT,
    QWENCLOUD_EMBEDDING_URL,
    Settings,
    load_settings,
    unknown_keys,
)


def test_default_config(settings: Settings) -> None:
    assert settings.roles["reader"].model == "deepseek-flash"
    reader = settings.models["deepseek-flash"]
    assert reader.provider == "deepseek"
    assert reader.extra_body == {"thinking": {"type": "disabled"}}
    # judge 与主 reader 来自不同家族（D-008）
    judge = settings.models[settings.roles["judge"].model]
    assert judge.provider != reader.provider
    # 成本统一按美元计（D-013）
    assert {spec.currency for spec in settings.models.values()} == {"USD"}
    assert settings.embedding.currency == "USD"


def test_qwen_goes_through_qwencloud(settings: Settings) -> None:
    qwen = settings.providers["qwen"]
    assert qwen.base_url == "https://maas.qwencloudapi.com/compatible-mode/v1"
    assert qwen.api_key_env == "DASHSCOPE_API_KEY"
    assert settings.embedding.native_url == QWENCLOUD_EMBEDDING_URL


def test_paths_resolve_against_project_root(clean_env: None) -> None:
    loaded = load_settings()
    assert loaded.paths.db == PROJECT_ROOT / "data" / "cuemem.sqlite"
    assert loaded.paths.traces == PROJECT_ROOT / "runs" / "traces"


def test_env_overrides_are_deep_merged(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUEMEM_ROLES__READER__MODEL", "qwen3.8-flash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "512")
    loaded = load_settings()
    assert loaded.roles["reader"].model == "qwen3.8-flash"
    assert loaded.roles["reader"].temperature == 0.0  # 其余字段仍来自 YAML
    assert loaded.roles["judge"].model == "qwen3.8-max"  # 其他角色不受影响
    assert loaded.embedding.dim == 512
    assert loaded.embedding.model == "qwen3.7-text-embedding"


def test_trace_retention_is_off_unless_set(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """trace 默认一直留着；CUEMEM_TRACES_RETENTION_DAYS 设保留天数（至少 1 天）。"""
    assert load_settings().traces_retention_days is None
    monkeypatch.setenv("CUEMEM_TRACES_RETENTION_DAYS", "30")
    assert load_settings().traces_retention_days == 30
    monkeypatch.setenv("CUEMEM_TRACES_RETENTION_DAYS", "0")
    with pytest.raises(ValidationError, match="traces_retention_days"):
        load_settings()


def test_vector_store_api_key_comes_from_the_environment(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qdrant 服务端的 API Key：CUEMEM_VECTOR_STORE__API_KEY（或 .env）读进 vector_store.api_key，
    空字符串当没设，repr 里看不到。以前没有这个字段，Key 被悄悄丢掉，连不上要 Key 的服务端。"""
    assert load_settings().vector_store.api_key is None
    monkeypatch.setenv("CUEMEM_VECTOR_STORE__URL", "http://qdrant:6333")
    monkeypatch.setenv("CUEMEM_VECTOR_STORE__API_KEY", "k-secret-value")
    loaded = load_settings()
    assert loaded.vector_store.api_key is not None
    assert loaded.vector_store.api_key.get_secret_value() == "k-secret-value"
    assert loaded.vector_store.secret() == "k-secret-value"
    assert loaded.vector_store.url == "http://qdrant:6333"
    assert "k-secret-value" not in repr(loaded) and "k-secret-value" not in str(loaded)
    monkeypatch.setenv("CUEMEM_VECTOR_STORE__API_KEY", "")
    assert load_settings().vector_store.api_key is None
    assert load_settings().vector_store.secret() is None


def test_unknown_model_reference_is_rejected(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUEMEM_ROLES__READER__MODEL", "no-such-model")
    with pytest.raises(ValidationError, match="no-such-model"):
        load_settings()


def test_shipped_configs_have_no_unknown_keys() -> None:
    """默认配置与每个预设都只用配置模型认识的键：Memory(config) 对不认识的键报错，YAML 本身
    照旧宽松（CLI 与评测都读它），这个测试防止它们悄悄写错。"""
    for path in [DEFAULT_CONFIG, *sorted(PRESETS_DIR.glob("*.yaml"))]:
        assert unknown_keys(yaml.safe_load(path.read_text(encoding="utf-8")) or {}) == [], path


def test_unknown_keys_walks_the_config_models() -> None:
    """子模型、X | None、dict[str, 子模型] 的每个值都查，extra_body 里是供应商的参数，不查；
    附上最像的已知键。"""
    found = unknown_keys(
        {
            "retreival": {},
            "consolidation": {"instruction": "x", "workers": 2},
            "retrieval": {"controler": "v1.json"},
            "proxy": {"unknown_models": "reject"},
            "roles": {"reader": {"modle": "x", "temperature": 0.1}},
            "models": {"m": {"extra_body": {"anything": 1}, "pricee": 1}},
            "providers": {"p": {"base_url": "u", "peak_hours": {"windowz": []}}},
            "llm": {"provider": "openai"},
        }
    )
    assert found == [
        ("retreival", "retrieval"),
        ("consolidation.instruction", "consolidation.instructions"),
        ("retrieval.controler", "retrieval.controller"),
        ("proxy.unknown_models", "proxy.unknown_model"),
        ("roles.reader.modle", "roles.reader.model"),
        ("models.m.pricee", "models.m.price"),
        ("providers.p.peak_hours.windowz", "providers.p.peak_hours.windows"),
        ("llm", None),
    ]
    assert unknown_keys({"embedding": {"backend": "hash", "dim": 64}, "peak_policy": "warn"}) == []
