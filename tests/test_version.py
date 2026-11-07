"""版本号（第八轮体检 P21）：每合入一轮体检升一次次版本号（第八轮 0.8.0），只在 pyproject.toml
里改；cuemem.__version__ 是同一个字面量（不在 import cuemem 时读包的元数据，启动时间不变），这里
核对两处与 uv.lock、OpenAPI 快照、API 文档一致。以前 0.1.0 几百个提交没变过，pip install -U
从 GitHub 装时什么都不做，构建报告也看不出是哪一版。"""

from __future__ import annotations

import json
import re
import tomllib

from fakes import PROJECT_ROOT

import cuemem


def _project_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as f:
        return str(tomllib.load(f)["project"]["version"])


def test_package_version_matches_pyproject() -> None:
    assert cuemem.__version__ == _project_version()


def test_version_is_round_based_and_everywhere_the_same() -> None:
    version = _project_version()
    assert re.fullmatch(r"0\.\d+\.\d+", version), version
    assert int(version.split(".")[1]) >= 8  # 第八轮起按体检轮次编号，不再退回 0.1.0
    with (PROJECT_ROOT / "uv.lock").open("rb") as f:
        lock = tomllib.load(f)
    [entry] = [p for p in lock["package"] if p["name"] == "cuemem"]
    assert entry["version"] == version  # 不一致时 uv sync --locked（CI 与 Docker）失败
    spec = json.loads((PROJECT_ROOT / "docs" / "openapi-v1.json").read_text("utf-8"))
    assert spec["info"]["version"] == version
    assert f"Signatures are those of CueMem {version} " in (
        PROJECT_ROOT / "docs" / "en" / "api.md"
    ).read_text("utf-8")
    assert f"签名以 CueMem {version} 为准" in (PROJECT_ROOT / "docs" / "api.md").read_text("utf-8")
