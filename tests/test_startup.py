"""启动与可选依赖：命令行、SDK 与轻的模块不连带加载用不到的重依赖；界面与评测是可选依赖。

以前 cuemem --help 要 3.5 秒、import cuemem.sdk 要 2.4–2.9 秒：几个包的 __init__ 把每个实现都
导入一遍（只用向量接口 embed.base，也经图片向量 → 媒体库 → store 包加载了 Qdrant 客户端），命令行
在模块顶层导入评测、报告（matplotlib）与整个记忆系统。导入放在子进程里做（这个进程早就加载过
这些模块），看导入之后 sys.modules 里有什么；不按耗时断言，免得在忙的机器上不稳。
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

HEAVY = (
    "qdrant_client",
    "openai",
    "matplotlib",
    "gradio",
    "sklearn",
    "scipy",
    "fastembed",
    "onnxruntime",
    "cuemem.memory",
    "cuemem.evals.report",
    "cuemem.evals.runner",
)


def heavy_after(code: str) -> list[str]:
    """在新的解释器里执行 code，返回之后已经加载的重模块。"""
    probe = f"{code}\nimport json, sys\nprint(json.dumps(sorted(sys.modules)))"
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    )
    modules = set(json.loads(out.stdout.splitlines()[-1]))
    return [name for name in HEAVY if name in modules]


def cli_call(*args: str) -> str:
    return (
        "from cuemem.cli import app\n"
        "try:\n"
        f"    app({list(args)!r}, prog_name='cuemem')\n"
        "except SystemExit:\n"
        "    pass"
    )


@pytest.mark.parametrize(
    "code",
    [
        "import cuemem.cli",
        cli_call("--help"),
        cli_call("eval", "--help"),
        cli_call("mcp", "--help"),
        cli_call("serve", "--help"),
        "import cuemem.sdk",
        "from cuemem import Memory",
    ],
)
def test_cli_and_sdk_import_without_heavy_dependencies(code: str) -> None:
    assert heavy_after(code) == []


@pytest.mark.parametrize(
    ("module", "allowed"),
    [
        ("cuemem.embed.base", []),
        ("cuemem.embed", []),
        ("cuemem.store.sqlite", []),
        ("cuemem.store.media", []),
        ("cuemem.llm.ledger", []),
        ("cuemem.llm.cache", []),
        ("cuemem.read.query", []),
        ("cuemem.write.extract", ["openai"]),  # 抽取要调用模型客户端
        # 打开记忆系统才要向量库与模型客户端（装了 local 时 Qdrant 客户端自己导入 fastembed）
        ("cuemem.memory", ["qdrant_client", "openai", "fastembed", "onnxruntime", "cuemem.memory"]),
    ],
)
def test_light_modules_do_not_pull_in_the_vector_store(module: str, allowed: list[str]) -> None:
    assert [m for m in heavy_after(f"import {module}") if m not in allowed] == []


def test_report_tables_work_without_matplotlib() -> None:
    """只读结果、汇总表格（判分抽检、命令行的总表）不要 matplotlib：它在可选依赖 eval 里。"""
    code = (
        "import sys\n"
        "sys.modules['matplotlib'] = None  # 当作没装：再导入就报 ModuleNotFoundError\n"
        "from cuemem.evals.report import load_rows, method_stats\n"
        "assert method_stats([]) == [] and sys.modules['matplotlib'] is None"
    )
    heavy_after(code)  # 子进程出错时 check=True 抛出
