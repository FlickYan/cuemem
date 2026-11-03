"""CI 的供应链（第四轮体检）：工作流里的第三方 Action 按提交号固定、令牌只读、每个任务有超时；
CI、Docker 镜像与 build-system 用同一个 uv 版本，它在 uv_build 的范围里（构建 wheel 用 uv 自带的
uv_build，不从 PyPI 现取一份没有哈希校验的）；Dependabot 与依赖审计的配置。只读仓库里的文件，
离线。"""

from __future__ import annotations

import json
import re
import shlex
import tomllib
from pathlib import Path
from typing import Any

import yaml
from fakes import PROJECT_ROOT
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

WORKFLOWS = sorted((PROJECT_ROOT / ".github" / "workflows").glob("*.yml"))
# 提交号后面跟着版本的注释：升级时看得出是哪个版本，Dependabot 也照着改
PINNED = re.compile(r"^\s*(?:-\s*)?uses:\s*[^@\s]+@[0-9a-f]{40} # v\d+\.\d+\.\d+\s*$")


def _workflow(path: Path) -> dict[Any, Any]:
    # 不用 "on" 这个键：PyYAML 按 YAML 1.1 把它读成 True
    loaded = yaml.safe_load(path.read_text("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _steps(path: Path) -> list[dict[str, Any]]:
    return [step for job in _workflow(path)["jobs"].values() for step in job.get("steps", [])]


def _docker_uv() -> str:
    """Dockerfile 里 uv 的版本（ghcr.io/astral-sh/uv:<版本>@sha256:<镜像摘要>）。"""
    found = re.findall(
        r"COPY --from=ghcr\.io/astral-sh/uv:([0-9.]+)@sha256:[0-9a-f]{64} ",
        (PROJECT_ROOT / "Dockerfile").read_text("utf-8"),
    )
    assert len(found) == 1, "Dockerfile 的 uv 要按版本与镜像摘要固定"
    return str(found[0])


def test_actions_are_pinned_by_commit() -> None:
    """每个 uses: 都是完整的提交号（标签可以被改写、挪到别的提交上），后面注释着版本。以前是
    actions/checkout@v4 与 astral-sh/setup-uv@v6：后者停在 v6.8.0，装 uv 时不校验哈希。"""
    assert WORKFLOWS
    for path in WORKFLOWS:
        steps = [step for step in _steps(path) if "uses" in step]
        assert steps, path.name
        for step in steps:
            assert re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", step["uses"]), (path.name, step)
        lines = [line for line in path.read_text("utf-8").splitlines() if "uses:" in line]
        assert len(lines) == len(steps) and all(PINNED.match(line) for line in lines), lines
        for step in steps:
            if step["uses"].startswith("actions/checkout@"):
                # 不把令牌留在 .git/config 里：后面的步骤（第三方的包、测试）读不到
                assert step.get("with", {}).get("persist-credentials") is False, path.name


def test_ci_docker_and_build_backend_agree_on_uv() -> None:
    """CI 装的 uv 与 Dockerfile 的相同，并且在 build-system 声明的 uv_build 范围里：uv 用自带的
    uv_build 构建 wheel。以前 CI 不指定版本、装的是最新的 0.13.6（setup-uv v6 的哈希表里没有它，
    没有校验），Docker 与开发机是 0.11.32，都不在 >=0.12.13,<0.13.0 里，构建时从 PyPI 现取一份。"""
    version = _docker_uv()
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text("utf-8"))
    [backend] = [Requirement(r) for r in pyproject["build-system"]["requires"]]
    assert canonicalize_name(backend.name) == "uv-build", backend
    assert backend.specifier.contains(version), (backend, version)
    for path in WORKFLOWS:
        setups = [s for s in _steps(path) if s.get("uses", "").startswith("astral-sh/setup-uv@")]
        assert setups, path.name
        for step in setups:
            assert str(step.get("with", {}).get("version")) == version, (path.name, step)


def test_workflows_have_read_only_tokens_timeouts_and_cancel_stale_pr_runs() -> None:
    """令牌只读写在工作流里（以前只靠仓库设置）；每个任务有超时（默认 360 分钟）；同一个 PR 的新
    推送取消还在跑的旧运行，push 各跑各的（组名用 run_id 兜底，不按分支排队）。"""
    for path in WORKFLOWS:
        workflow = _workflow(path)
        assert workflow.get("permissions") == {"contents": "read"}, path.name
        for name, job in workflow["jobs"].items():
            minutes = job.get("timeout-minutes")
            assert isinstance(minutes, int) and 0 < minutes <= 120, (path.name, name)
        concurrency = workflow.get("concurrency") or {}
        assert concurrency.get("cancel-in-progress") is True, path.name
        assert "github.event.pull_request.number || github.run_id" in concurrency["group"]


def test_dependabot_watches_actions_the_lock_and_the_base_image() -> None:
    """Dependabot 每周查工作流的 Action、uv.lock 与 Dockerfile 的基础镜像（以前一个都没有：锁文件
    里 Pillow 12.3.0 的 41 条安全公告没有任何提醒）。镜像的 Python 停在 .python-version 的小版本；
    不管 docker-compose.yml：那里的 qdrant/qdrant 与 ci.yml 服务端任务的版本要一起升级。"""
    config = _workflow(PROJECT_ROOT / ".github" / "dependabot.yml")
    updates = {entry["package-ecosystem"]: entry for entry in config["updates"]}
    assert set(updates) == {"github-actions", "uv", "docker"}
    for entry in updates.values():
        assert entry["directory"] == "/" and entry["schedule"]["interval"] == "weekly", entry
    # 新发布的版本等几天再用（冷却期不管安全更新）；大版本不并进小版本与补丁的 PR
    assert updates["uv"]["cooldown"]["default-days"] >= 3
    assert [group["update-types"] for group in updates["uv"]["groups"].values()] == [
        ["minor", "patch"]
    ]
    [python] = [rule for rule in updates["docker"]["ignore"] if rule["dependency-name"] == "python"]
    assert set(python["update-types"]) == {
        "version-update:semver-major",
        "version-update:semver-minor",
    }
    pinned = (PROJECT_ROOT / ".python-version").read_text("utf-8").strip()
    bases = re.findall(r"^FROM (\S+)", (PROJECT_ROOT / "Dockerfile").read_text("utf-8"), re.M)
    assert bases and all(base == f"python:{pinned}-slim" for base in bases), bases


def test_dependency_audit_reads_hashed_exports_and_is_not_required() -> None:
    """依赖审计（pip-audit）：--disable-pip 只认全部带哈希的需求文件，所以导出不能加 --no-hashes，
    也不能带上项目本身（"-e ." 没有哈希，--no-emit-project）：缺了哪一样 pip-audit 都直接退出 1，
    什么也没查。一步查镜像装的依赖（与 Dockerfile 的 EXTRAS 相同），一步查全部依赖。它不在规则集
    的必需检查里（新的公告不需要代码改动就会出现），规则集要求的四项都是 ci.yml 的任务。"""
    path = PROJECT_ROOT / ".github" / "workflows" / "audit.yml"
    commands = [
        shlex.split(line)
        for step in _steps(path)
        for line in step.get("run", "").replace("\\\n", " ").splitlines()
        if line.strip()
    ]
    exports = [c for c in commands if c[:2] == ["uv", "export"]]
    audits = [c for c in commands if c[0] == "uvx"]
    assert len(exports) == len(audits) == 2
    for export, audit in zip(exports, audits, strict=True):
        assert "--locked" in export and "--no-emit-project" in export, export
        assert "--no-hashes" not in export and "--format" in export, export
        output = export[export.index("-o") + 1]
        assert re.fullmatch(r"pip-audit==\d+\.\d+\.\d+", audit[1]), audit
        assert audit[2:] == ["-r", output, "--disable-pip"], audit
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text("utf-8")
    [extras] = re.findall(r'^ARG EXTRAS="([^"]*)"', dockerfile, re.M)
    assert " ".join(exports[0]).count(f"--no-dev {extras} ") == 1, exports[0]
    assert "--all-extras" in exports[1] and "--all-groups" in exports[1], exports[1]

    ruleset = json.loads(
        (PROJECT_ROOT / ".github" / "rulesets" / "protect-main.json").read_text("utf-8")
    )
    [checks] = [r for r in ruleset["rules"] if r["type"] == "required_status_checks"]
    required = {c["context"] for c in checks["parameters"]["required_status_checks"]}
    ci_jobs = set(_workflow(PROJECT_ROOT / ".github" / "workflows" / "ci.yml")["jobs"])
    assert required == {"test", "package", "docker", "qdrant-server"} <= ci_jobs
    assert "audit" not in required and "audit" not in ci_jobs
