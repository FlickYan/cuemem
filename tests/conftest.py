"""测试公共设施：离线配置与模拟接口。所有测试都不访问网络（netguard 守着），也不读取项目里的
.env。"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import netguard
import pytest
from fakes import FakeAPI

from cuemem.config import PathsConfig, Settings, load_settings

# 测试中 load_dotenv 可能写入这些变量；先登记到 monkeypatch，测试结束时自动恢复原状
GUARDED_ENV = (
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
    "CUEMEM_PROVIDERS__QWEN__BASE_URL",
    "CUEMEM_EMBEDDING__NATIVE_URL",
)
# 离线的测试会话里删掉：代理（见 netguard）与模型供应商的 Key（D-069 记下过一个测试草稿真的
# 调用了上游；没有真 Key 时，漏网的请求也只会被拒绝）
OFFLINE_UNSET = (
    *netguard.PROXY_ENV,
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
)
# 第三方库导入或使用时自己联网的开关：Gradio 的使用统计（以前界面的测试每次向 api.gradio.app 与
# huggingface.co 发请求）；LiteLLM 导入时下载模型价格表（在主进程里导入 cognee 的测试，适配层给
# 子进程设了同样的变量）
OFFLINE_SET = {"GRADIO_ANALYTICS_ENABLED": "False", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}
LIVE = pytest.StashKey[bool]()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """按 -m 筛完之后：选中了联网测试（uv run pytest -m live）时不装网络守卫。"""
    config.stash[LIVE] = any(item.get_closest_marker("live") for item in items)


@pytest.fixture(scope="session", autouse=True)
def offline(request: pytest.FixtureRequest) -> Iterator[None]:
    """整个测试会话不访问网络：删掉代理与供应商 Key 的环境变量，关掉第三方库自己联网的开关，
    装上 netguard。"""
    if request.config.stash.get(LIVE, False):
        yield
        return
    with pytest.MonkeyPatch.context() as patch:
        netguard.install(patch)  # 先装：记下删掉之前的代理
        for name in OFFLINE_UNSET:
            patch.delenv(name, raising=False)
        for name, value in OFFLINE_SET.items():
            patch.setenv(name, value)
        yield


@pytest.fixture(autouse=True)
def no_network(offline: None) -> Iterator[None]:
    """测试里有访问网络的尝试（哪怕异常被后台线程吞掉了）就让它失败；上一个测试结束之后才发生的
    （后台线程）记在下一个测试上，说明里写着当时在跑的测试。"""
    earlier = netguard.unreported()
    if earlier:
        pytest.fail(f"上一个测试结束之后有访问网络的尝试：{earlier}")
    yield
    attempts = netguard.unreported()
    if attempts:
        pytest.fail(f"测试试图访问网络：{attempts}")


@pytest.fixture(autouse=True)
def restore_umask() -> Iterator[None]:
    """命令行一启动就收紧 umask（cli._private_umask）：CliRunner 在测试进程里运行它，测试结束时
    还原，不影响后面的测试。"""
    mask = os.umask(0o022)
    os.umask(mask)
    yield
    os.umask(mask)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """最后一个测试结束之后（后台线程）才有的尝试：整个会话算失败。"""
    late = netguard.unreported()
    if late:
        print(f"\n测试结束之后有访问网络的尝试：{late}")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


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
    # 测试不能依赖真实时钟：分时价格的高峰时段里也不暂停（暂停逻辑在 test_peak.py 里用假时钟测）
    return loaded.model_copy(
        update={"paths": PathsConfig().resolve_against(tmp_path), "peak_policy": "allow"}
    )


@pytest.fixture
def offline_settings(settings: Settings) -> Settings:
    """改用离线哈希向量器：有真实的字面相似度，适合测检索链路。"""
    embedding = settings.embedding.model_copy(update={"backend": "hash", "dim": 256})
    return settings.model_copy(update={"embedding": embedding})


@pytest.fixture
def fake_api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture
def n3_settings(offline_settings: Settings) -> Settings:
    """巩固相关测试用的配置：哈希向量的相似度没有语义，把相似度阈值调到几乎只认精确匹配，
    让对账与消歧的走向完全由规则和脚本化的模型决定。"""
    consolidation = offline_settings.consolidation.model_copy(
        update={
            "duplicate_threshold": 0.9999,
            "conflict_threshold": 0.9999,
            "multi_conflict_threshold": 0.9999,
            "entity_merge_threshold": 0.9999,
            "entity_gray_threshold": 0.9998,
            "workers": 2,
        }
    )
    return offline_settings.model_copy(update={"consolidation": consolidation})
