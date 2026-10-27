from __future__ import annotations

import pytest
from pydantic import ValidationError

from cuemem.config import PROJECT_ROOT, QWENCLOUD_EMBEDDING_URL, Settings, load_settings


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


def test_unknown_model_reference_is_rejected(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUEMEM_ROLES__READER__MODEL", "no-such-model")
    with pytest.raises(ValidationError, match="no-such-model"):
        load_settings()
