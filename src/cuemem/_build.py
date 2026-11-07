"""这份 CueMem 是从哪个提交构建的（cuemem --version 与 doctor 的第一行，第八轮体检 P19）。

版本号不随每次提交变：pip install -U 同一个版本号时不重装，docker compose up -d 照旧用旧镜像，
以前没有办法看出跑着的是哪一版。不由 import cuemem 导入（可能要起一个 git 子进程），只在命令行
里用；也不放进不鉴权的 /api/health（公开部署时会告诉任何人缺哪些修复）。
"""

from __future__ import annotations

import json
import os
import subprocess
from functools import cache
from pathlib import Path

BUILD_ENV = "CUEMEM_BUILD"  # Docker 镜像构建时写进来（Dockerfile 的 ARG CUEMEM_BUILD）
SHORT = 12  # 提交号取前 12 位，与 git rev-parse --short=12 相同


@cache
def build_id() -> str | None:
    """构建标识，依次看：环境变量 CUEMEM_BUILD（不为空、不是 unknown）；从 git 安装时 pip 记下的
    direct_url.json 里的提交号（工作区改过时加 +dirty）；源码仓库里的 git rev-parse（2 秒超时）。
    都没有时 None。从不抛出异常，结果缓存。"""
    env = os.environ.get(BUILD_ENV, "").strip()
    if env and env != "unknown":
        return env
    try:
        return _from_direct_url() or _from_checkout()
    except Exception:  # 只是展示用：读不出就说不知道，不该让命令失败
        return None


def _from_direct_url() -> str | None:
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        text = distribution("cuemem").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not text:
        return None
    vcs = json.loads(text).get("vcs_info")
    if not isinstance(vcs, dict) or not isinstance(vcs.get("commit_id"), str):
        return None
    commit = vcs["commit_id"][:SHORT]
    return commit + "+dirty" if vcs.get("dirty") else commit


def _from_checkout() -> str | None:
    import cuemem

    root = Path(cuemem.__file__).resolve().parents[2]  # src/cuemem/__init__.py → 仓库根目录
    if not (root / ".git").exists():
        return None
    done = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"--short={SHORT}", "HEAD"],
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
    )
    commit = done.stdout.strip()
    return commit if done.returncode == 0 and commit else None


def version_line() -> str:
    """cuemem --version 与 doctor 第一行：cuemem <版本> (build <标识>|unknown)。"""
    from cuemem import __version__

    return f"cuemem {__version__} (build {build_id() or 'unknown'})"
