"""测试公共设施：离线配置与模拟接口。所有测试都不访问网络，也不读取项目里的 .env。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fakes import FakeAPI

from cuemem.config import PathsConfig, Settings, load_settings

# 测试中 load_dotenv 可能写入这些变量；先登记到 monkeypatch，测试结束时自动恢复原状
GUARDED_ENV = (
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "CUEMEM_PROVIDERS__QWEN__BASE_URL",
    "CUEMEM_EMBEDDING__NATIVE_URL",
)


@pytest.fixture
def clean_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉相关环境变量，并让 .env 指向临时目录里的文件（不碰项目真实的 .env）。"""
    for name in list(os.environ):
        if name.startswith("CUEMEM_"):
            monkeypatch.delenv(name)
    for name in GUARDED_ENV:
        monkeypatch.setenv(name, "placeholder")  # 记录原始状态，结束时还原
        monkeypatch.delenv(name)
    monkeypatch.setenv("CUEMEM_ENV_FILE", str(tmp_path / "test.env"))


@pytest.fixture
def settings(clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """默认配置 + 假的 API Key + 所有数据写到临时目录。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    loaded = load_settings()
    return loaded.model_copy(update={"paths": PathsConfig().resolve_against(tmp_path)})


@pytest.fixture
def offline_settings(settings: Settings) -> Settings:
    """改用离线哈希向量器：有真实的字面相似度，适合测检索链路。"""
    embedding = settings.embedding.model_copy(update={"backend": "hash", "dim": 256})
    return settings.model_copy(update={"embedding": embedding})


@pytest.fixture
def fake_api() -> FakeAPI:
    return FakeAPI()
