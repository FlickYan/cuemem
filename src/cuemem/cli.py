"""命令行入口。

cuemem [--preset openai|ollama|deepseek|qwen] …   换模型供应商（也可设 CUEMEM_PRESET，N12）
cuemem configure                      交互式填写 API Key（写入 .env）
cuemem doctor [--live]                检查配置、API Key 与连通性（--repair-vectors 修本地向量库）
cuemem ingest FILE [--consolidate]    导入对话（可选：立即巩固）
cuemem consolidate --space S          巩固：抽取事实与实体、对账、摘要、核心记忆
cuemem ask "问题" --space S [--at T]  检索记忆并回答
cuemem inspect facts|entities|sources|profile --space S   记忆检查器
cuemem forget --space S --session X   遗忘：删除会话及由它派生的一切（--fact、--dry-run 预演）
cuemem export --space S [-o FILE]     导出记忆空间：原始日志、图片与事实快照（N12）
cuemem import FILE [--space S]        导入：写回原始日志，再重新巩固（N12）
cuemem spaces                         列出记忆空间
cuemem usage [--today] [--run ID]     汇总调用量与成本
cuemem data download [NAME...]       下载评测数据集
cuemem eval split / run / report     评测：划分开发集、运行、生成报告
cuemem eval audit RUN [--score]      判分抽检：导出待核对条目 / 统计一致率
cuemem chat --space S [--at T]        在终端里与 Agent 对话（N8）
cuemem serve [--port 7860]            启动 HTTP 接口与演示界面（FastAPI + Gradio）
cuemem demo SCRIPT [--until N]        按演示剧本回放对话并核对回答
cuemem mcp --space S                  以 MCP Server（stdio）提供 remember / recall / forget
cuemem proxy [--port 8787]            OpenAI 兼容的记忆代理：改 base_url 即接入（N12）
cuemem migrate-vectors --to URL       把本地向量库复制到 Qdrant 服务端（N11）
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import logging
import os
import shlex
import shutil
import signal
import sqlite3
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, cast, get_args

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from cuemem.config import (
    Settings,
    UnknownModel,
    cli_command,
    env_file_path,
    extra_remedy,
    load_settings,
)
from cuemem.core.timeutil import naive_local, valid_period
from cuemem.core.types import AskMode, RecallMode, Router, Scenario
from cuemem.envfile import atomic_write_text, mask, read_env, update_env
from cuemem.errors import (
    ConfigMismatchError,
    CueMemError,
    DatabaseOpenError,
    DatasetChecksumError,
)

# 记忆系统、评测、Agent 与服务在用到它们的命令里才导入：cuemem --help、cuemem mcp 这些命令
# 不加载 matplotlib、评测与用不到的模块（以前每个命令启动都要 3 秒多）
if TYPE_CHECKING:
    from cuemem.agent import Reply
    from cuemem.api.auth import AnyKeys, KeyRing, Keys
    from cuemem.api.service import ChatService
    from cuemem.core.models import Answer, Fact
    from cuemem.evals.datasets import EvalSet
    from cuemem.evals.runner import RunSummary
    from cuemem.memory import ForgetReport, MemorySystem
    from cuemem.store.sqlite import MemoryDB
    from cuemem.write.consolidate import ConsolidationReport

app = typer.Typer(
    help="CueMem：线索驱动的多模态 Agent 长期记忆系统",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
LOW_DISK = 200 * 2**20  # doctor 在数据目录剩余空间低于它时标黄


@app.callback()
def main(
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            help="换模型供应商的预设：openai、ollama（本地，不要 Key）、deepseek、qwen，"
            "或预设 YAML 的路径；也可以设环境变量 CUEMEM_PRESET",
        ),
    ] = None,
) -> None:
    """CueMem：线索驱动的多模态 Agent 长期记忆系统"""
    _raise_open_file_limit()
    _private_umask()
    if preset:  # 各命令都通过 load_settings 读配置，它看这个环境变量（N12）
        os.environ["CUEMEM_PRESET"] = preset


def _raise_open_file_limit() -> tuple[int, int] | None:
    """把这个进程打开文件数的软上限调到硬上限（最多 65536），返回调整之后的（软，硬）上限；
    没有 resource 模块（Windows）时返回 None。从不调低。

    Qdrant 本地模式每个集合（每个记忆空间一个，有图片的空间两个）一直开着一个文件、启动时全部
    打开：软上限常见的 1024（Linux、systemd）与 256（macOS）只够几百个用户，超过之后 cuemem serve
    建不了新的空间，重启时打不开向量库，cuemem migrate-vectors 也打不开、搬不走。macOS 的硬上限
    是不限，超过 kern.maxfilesperproc 时 setrlimit 报 ValueError，依次退到 10240、4096。

    只在命令行里调：库（MemorySystem、SDK 的 Memory）不改进程级别的设置，那是用户自己的应用；
    它们打不开时抛出写明原因的 VectorStoreOpenError。"""
    try:
        import resource
    except ImportError:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return soft, hard
    top = 65536 if hard == resource.RLIM_INFINITY else min(hard, 65536)
    for target in (top, 10240, 4096):
        if target <= soft or (hard != resource.RLIM_INFINITY and target > hard):
            continue
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except (ValueError, OSError):
            continue
        break
    return resource.getrlimit(resource.RLIMIT_NOFILE)


def _private_umask() -> int:
    """把这个进程的 umask 收紧到至少 077，返回收紧之后的值；从不放宽（已经更严的照旧）。

    命令行写的文件里有对话原文、提问与图片：数据库、缓存、trace、export -o 的导出、评测结果、
    上传的图片、下载的模型，以及经 .part 换名的重写。以前按常见的 umask 022 建成 0644，同一台
    机器上别的账户都能读（第四轮体检）。与 _raise_open_file_limit 一样只在命令行里调：库与 SDK
    不改进程级别的设置，它们新建的数据目录是 0700（store/files.py）。"""
    mask = os.umask(0o077) | 0o077
    os.umask(mask)
    return mask


DATETIME_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]
CURRENCY_SYMBOL = {"CNY": "¥", "USD": "$"}
PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "qwen": "QwenCloud（通义千问）",
    "openai": "OpenAI",
    "ollama": "Ollama（本地）",
}
KEY_HINTS = {"qwen": "以 sk-ws- 开头"}  # 帮助区分同一平台上的其他 Key
SECRET_WORDS = ("key", "token", "secret", "password")  # 配置写错时不回显这些字段收到的值


def _settings(*, stderr: bool = False, **overrides: Any) -> Settings:
    """读配置（load_settings）；配置写错时只打印一行原因、以 1 退出，不打印调用栈。

    写错的预设名（ValueError）与不合法的值（pydantic 的 ValidationError，也是 ValueError；例如
    docker run --env-file 把行尾的 # 注释算进了值里）以前在多数命令里打出几十行调用栈：退出码同样
    是 1，容器同样反复重启，原因却埋在调用栈里。SDK 不变，照旧抛出这些异常。stderr：mcp 的 stdout
    是 MCP 的通道。"""
    try:
        return load_settings(**overrides)
    except (CueMemError, ValueError) as exc:
        out = Console(stderr=True) if stderr else console
        out.print(f"[red]错误：[/red]配置有误：{escape(_config_problem(exc))}")
        raise typer.Exit(1) from exc


def _config_problem(exc: Exception) -> str:
    """配置错误的一行说明：pydantic 的 ValidationError 每个字段一句（字段、原因、收到的值，过长时
    截断），不要它多行的格式与文档链接；别的错误原样。

    配置里有密钥（vector_store.api_key，Qdrant 服务端的 Key）：名字像密钥的字段不回显收到的值，
    收到的是一整段配置（字典，例如整份配置的引用检查）时也不回显，里面可能带着密钥；与
    api/auth.py 写错令牌时一样。这一行会进 docker compose logs。"""
    if not isinstance(exc, ValidationError):
        return str(exc)
    parts = []
    for error in exc.errors(include_url=False):
        where = ".".join(str(part) for part in error["loc"]) or "配置"
        secret = any(word in str(part).lower() for part in error["loc"] for word in SECRET_WORDS)
        if secret or isinstance(error.get("input"), dict):
            parts.append(f"{where}：{error['msg']}")
            continue
        got = repr(error.get("input"))
        got = got if len(got) <= 60 else got[:57] + "…"
        parts.append(f"{where}：{error['msg']}（收到 {got}）")
    return "；".join(parts)


@contextmanager
def _open_system() -> Iterator[MemorySystem]:
    """组装 MemorySystem；把 CueMem 的已知错误转成友好的提示。

    交互式命令只发少量请求：分时价格的高峰时段照常调用（记录提醒），不像评测那样暂停。
    """
    from cuemem.memory import MemorySystem

    system: MemorySystem | None = None
    try:
        settings = _settings(peak_policy="warn")
        _peak_note(settings)
        try:
            system = MemorySystem.from_settings(settings)
        except DatabaseOpenError as exc:  # 已经说明了哪个文件：不再包一层 _unopenable
            console.print(f"[red]错误：[/red]{escape(str(exc))}")
            raise typer.Exit(1) from exc
        except sqlite3.OperationalError as exc:  # 只管打开时：用的时候出的错照常抛出
            console.print(_unopenable(settings, exc))
            raise typer.Exit(1) from exc
        yield system
    except CueMemError as exc:
        console.print(f"[red]错误：[/red]{exc}")
        raise typer.Exit(1) from exc
    finally:
        if system is not None:
            system.close()


def _unopenable(settings: Settings, exc: sqlite3.OperationalError) -> str:
    """打开数据库、账本或缓存时 SQLite 出错（目录只读、磁盘满了）：一行说明，不打印调用栈。"""
    where = escape(str(settings.paths.db.parent))
    return (
        f"[red]错误：[/red]{where} 里的数据库打不开（{escape(str(exc))}）：检查目录权限与磁盘空间"
    )


@contextmanager
def _open_db_only() -> Iterator[MemoryDB]:
    """只打开数据库（cuemem export、spaces、inspect）：不用向量库、模型与缓存，向量库打不开、
    或者 cuemem serve 正开着它的时候也能用（不拿 MemorySystem.from_settings 的数据库锁）。"""
    from cuemem.store.files import private_dirs, storage_dirs
    from cuemem.store.sqlite import MemoryDB

    settings = _settings()
    private_dirs(*storage_dirs(settings))
    try:
        db = MemoryDB(settings.paths.db)
    except sqlite3.OperationalError as exc:
        console.print(_unopenable(settings, exc))
        raise typer.Exit(1) from exc
    except CueMemError as exc:
        console.print(f"[red]错误：[/red]{exc}")
        raise typer.Exit(1) from exc
    try:
        yield db
    finally:
        db.close()


@app.command()
def configure() -> None:
    """交互式填写各供应商的 API Key，写入 .env（Key 输入时不显示）。"""
    settings = _settings()
    path = env_file_path()
    current = read_env(path)
    console.print(
        f"配置将写入 [bold]{path}[/bold]（已在 .gitignore 中，不会被提交）。\n"
        "Key 输入时屏幕上不显示任何字符；直接回车表示保留当前值。\n"
    )
    # 只问用得到的供应商（预设叠加在默认配置上，默认的供应商仍在配置里），不要 Key 的跳过
    needed = {
        name: provider
        for name, provider in settings.providers.items()
        if name in settings.used_providers() and provider.api_key_env is not None
    }
    updates: dict[str, str | None] = {}
    for name, provider in needed.items():
        assert provider.api_key_env is not None
        label = PROVIDER_LABELS.get(name, name)
        hint = f"（{KEY_HINTS[name]}）" if name in KEY_HINTS else ""
        updates[provider.api_key_env] = _prompt_secret(
            f"{label} API Key{hint}", current.get(provider.api_key_env)
        )
    update_env(path, updates)

    summary = Table(title=f"已写入 {path.name}（仅本人可读写）")
    for column in ("供应商", "环境变量", "Key", "接口"):
        summary.add_column(column)
    for name, provider in needed.items():
        assert provider.api_key_env is not None
        key = updates[provider.api_key_env]
        summary.add_row(
            PROVIDER_LABELS.get(name, name),
            provider.api_key_env,
            mask(key) if key else "[red]未填写[/red]",
            provider.base_url,
        )
    console.print(summary)
    console.print(f"下一步：[bold]{cli_command('doctor --live')}[/bold]")


@app.command()
def doctor(
    live: Annotated[
        bool, typer.Option("--live", help="实际调用每个模型与向量接口做连通性测试（费用极低）")
    ] = False,
    repair_vectors: Annotated[
        bool,
        typer.Option(
            "--repair-vectors",
            help="本地向量库的 meta.json 读不出时，按 collection/ 下的集合重写它（先停掉 serve）",
        ),
    ] = False,
    vectors: Annotated[
        bool,
        typer.Option(
            "--vectors",
            help="逐个空间比较数据库与向量库的点数（要打开向量库：本地模式下先停掉 serve）",
        ),
    ] = False,
) -> None:
    """检查配置、API Key、数据目录与密钥文件的权限、剩余空间，以及（可选）向量库与数据库是否
    一致、各模型的连通性。"""
    settings = _settings()
    if repair_vectors:
        _repair_vectors(settings)
    _print_config(settings)
    _print_local_store(settings)
    _print_disk(settings)
    _print_media_journal(settings)
    if vectors:
        _print_vector_gaps(settings)
    for line in _loose_modes(settings):
        console.print(line)
    if live:
        _live_check(settings)
    else:
        console.print("提示：加 [bold]--live[/bold] 会实际调用各模型做连通性测试。")


@app.command()
def ingest(
    path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, help="对话 JSON，格式见 examples/"),
    ],
    space: Annotated[str | None, typer.Option(help="覆盖文件里的 space_id")] = None,
    consolidate: Annotated[
        bool, typer.Option("--consolidate", help="导入后立即巩固（抽取事实、实体与摘要）")
    ] = False,
) -> None:
    """导入一段多会话对话（可重复执行，不会产生重复数据）。"""
    from cuemem.core.models import Conversation

    conversation = Conversation.model_validate_json(path.read_text(encoding="utf-8"))
    if space:
        conversation = conversation.model_copy(update={"space_id": space})
    with _open_system() as system:
        report = system.ingest(conversation)
        console.print(
            f"记忆空间 [bold]{report.space_id}[/bold]：{report.sessions} 个会话，"
            f"新增 {report.turns_added} 条轮次、{report.units_added} 个记忆单元，"
            f"耗时 {report.elapsed_ms:.0f} ms"
        )
        if consolidate:
            _print_consolidation(system.consolidate(report.space_id))
        else:
            console.print("[dim]提示：运行 cuemem consolidate 抽取事实与实体（需要 LLM）。[/dim]")
        _print_usage_rows(system.ledger.summary(run_id=system.ledger.run_id))


@app.command("export")
def export_cmd(
    space: Annotated[str, typer.Option(help="记忆空间 ID")],
    out: Annotated[
        Path | None, typer.Option("--out", "-o", help="输出文件；不填则输出到标准输出")
    ] = None,
    no_derived: Annotated[
        bool, typer.Option("--no-derived", help="不附派生记忆（事实与画像）的快照")
    ] = False,
) -> None:
    """导出记忆空间（N12）：原始日志与图片原文件，可选附上事实与画像的快照。"""
    from cuemem.portable import export_space

    with _open_db_only() as db:  # 不用向量库：向量库坏了（见 doctor --repair-vectors）也能导出
        try:
            data = export_space(db, space, derived=not no_derived)
        except KeyError as exc:
            console.print(f"[red]没有记忆空间 {space}[/red]")
            raise typer.Exit(1) from exc
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if out is None:
        print(text)
        return
    target = out.resolve()
    atomic = not target.exists() or target.is_file()  # /dev/stdout、FIFO 之类直接写
    try:
        if atomic:  # 写到一半出错（磁盘满）时旧的备份不动；以前先截断，新旧两份都没了
            atomic_write_text(out, text, fsync=True)
        else:
            out.write_text(text, encoding="utf-8")
    except OSError as exc:
        kept = f"；{escape(str(out))} 未改动" if atomic else ""
        console.print(f"[red]导出失败：{escape(str(exc))}{kept}[/red]")
        raise typer.Exit(1) from exc
    turns = sum(len(s["turns"]) for s in data["sessions"])
    console.print(f"已导出 {len(data['sessions'])} 个会话、{turns} 条发言到 [bold]{out}[/bold]")


@app.command("import")
def import_cmd(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="cuemem export 的文件")],
    space: Annotated[str | None, typer.Option(help="导入到这个空间（默认用文件里的）")] = None,
    consolidate: Annotated[
        bool, typer.Option("--consolidate", help="导入后立即巩固（重新抽取事实，需要 LLM）")
    ] = False,
) -> None:
    """导入 cuemem export 的文件（N12）：写回原始日志，派生记忆由巩固重新生成；可重复执行。"""
    from cuemem.portable import import_space

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:  # 以前是调用栈
        console.print(f"[red]不是有效的 CueMem 导出文件（被截断了？）：{escape(str(exc))}[/red]")
        raise typer.Exit(1) from exc
    with _open_system() as system:
        try:
            report = import_space(system, data, space_id=space, consolidate=consolidate)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc
        console.print(
            f"记忆空间 [bold]{report['space_id']}[/bold]：{report['sessions']} 个会话，"
            f"新增 {report['turns_added']} 条发言、{report['images_added']} 张图片"
        )
        if "consolidation" not in report:
            console.print(
                "[dim]提示：运行 cuemem consolidate 重新抽取事实与实体（需要 LLM）。[/dim]"
            )


@app.command("consolidate")
def consolidate_cmd(
    space: Annotated[str, typer.Option(help="记忆空间 ID")],
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="清空派生记忆后从原始日志全部重放（命中缓存）")
    ] = False,
) -> None:
    """巩固：把原始对话整理成带有效期的事实、实体、会话摘要与核心记忆（幂等，可重复执行）。"""
    with _open_system() as system:
        _print_consolidation(system.consolidate(space, rebuild=rebuild))
        _print_usage_rows(system.ledger.summary(run_id=system.ledger.run_id))


@app.command()
def reindex(
    space: Annotated[str | None, typer.Option(help="记忆空间 ID")] = None,
    all_spaces: Annotated[bool, typer.Option("--all", help="数据库里的全部空间")] = False,
    full: Annotated[
        bool, typer.Option("--full", help="全部重新写一遍，不只补缺的（命中向量缓存）")
    ] = False,
) -> None:
    """按数据库补齐向量库：补上缺的向量、删掉多余的点（不调用 LLM）。向量库丢了、坏了，或者与
    数据库不是同一时刻的备份时用；先用 cuemem doctor --vectors 查哪些空间对不上。顺带删掉进程
    被杀掉时没有提交的写入、没做完的遗忘留在媒体库里的原图（cuemem doctor 提示有没有）。"""
    if (space is None) == (not all_spaces):
        console.print("[red]指定 --space 或 --all 其中一个[/red]")
        raise typer.Exit(2)
    with _open_system() as system:
        targets = [s["space_id"] for s in system.db.list_spaces()]
        if space is not None:
            if space not in targets:
                console.print(f"[red]没有记忆空间 {escape(space)}[/red]")
                raise typer.Exit(1)
            targets = [space]
        for target in targets:
            report = system.reindex(target, only_missing=not full)
            added = {k: v for k, v in report.items() if k in _REINDEX_KINDS and v}
            filled = "、".join(f"{k} {v}" for k, v in added.items())
            line = f"补上 {filled}" if filled else "没有缺的"
            line += f"；删掉多余的点 {report['orphans'] + report.get('image_orphans', 0)} 个"
            if report.get("missing_files"):
                line += f"；{report['missing_files']} 张图片的原图不在了，补不上"
            if report.get("facts_missing"):
                line += f"；{report['facts_missing']} 个事实单元在事实表里找不到，补不上"
            console.print(f"记忆空间 [bold]{escape(target)}[/bold]：{line}")
        # 被杀掉的写入与遗忘留在媒体库里的原图（第七轮体检）：只删标记点名、没有记录引用的
        recovered, _ = system.recover_media()
        if recovered:
            console.print(f"媒体库：删掉 {recovered} 张没有提交的写入或遗忘留下的原图")
        _print_usage_rows(system.ledger.summary(run_id=system.ledger.run_id))


# reindex 的报告里按种类计数的几项（其余是删掉的点与补不上的原图）
_REINDEX_KINDS = ("round", "fact", "summary", "entity", "image", "image_iso", "images")


@app.command()
def forget(
    space: Annotated[
        str | None, typer.Option(help="记忆空间 ID（--expired 时不给是所有空间）")
    ] = None,
    session: Annotated[list[str] | None, typer.Option(help="要遗忘的会话 ID（可多次指定）")] = None,
    turn: Annotated[list[str] | None, typer.Option(help="要遗忘的轮次 ID（可多次指定）")] = None,
    fact: Annotated[
        list[str] | None,
        typer.Option(help="要遗忘的事实 ID（可多次指定）：删它出自的对话与没有引用的转述"),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="只预演：列出会删哪些发言与事实，什么都不删")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="不再确认，直接删除")] = False,
    expired: Annotated[
        bool,
        typer.Option(
            "--expired",
            help="删掉已经过期的发言（写入时给了 expiration_date 的），连同派生的记忆",
        ),
    ] = False,
    older_than_days: Annotated[
        int, typer.Option("--older-than-days", min=0, help="--expired 时只删过期这么多天以上的")
    ] = 0,
) -> None:
    """遗忘：删除原始轮次，并清除由它们派生的事实、摘要、向量与缓存（不可恢复）。

    --expired：删过期的记忆（cuemem serve / proxy 设了 CUEMEM_EXPIRED_RETENTION_DAYS 时每天自己
    删，只用 SDK 时用这个命令）。只删过期的发言本身，后来引用了它、没给过期时间的回复留着。"""
    if expired:
        if session or turn or fact or dry_run:
            console.print("[red]--expired 不能与 --session、--turn、--fact、--dry-run 一起用[/red]")
            raise typer.Exit(1)
        _forget_expired(space, older_than_days, yes)
        return
    if space is None:
        console.print("[red]请用 --space 指定记忆空间[/red]")
        raise typer.Exit(1)
    if not session and not turn and not fact:
        console.print("[red]请用 --session、--turn 或 --fact 指定要遗忘的内容[/red]")
        raise typer.Exit(1)
    if dry_run:
        with _open_system() as system:
            plan = system.preview_forget(
                space, session_ids=session or [], turn_ids=turn or [], fact_ids=fact or []
            )
        _print_forget_plan(plan)
        return
    targets = ", ".join([*(session or []), *(turn or []), *(fact or [])])
    if not yes and not typer.confirm(f"将永久删除 {targets} 及其派生的全部记忆，确定吗？"):
        raise typer.Exit(1)
    with _open_system() as system:
        report = system.forget_memories(
            space, session_ids=session or [], turn_ids=turn or [], fact_ids=fact or []
        )
        if report.turns_deleted == 0:
            console.print("没有找到要删除的内容。")
            return
        _print_forget_report(report)


def _forget_expired(space: str | None, days: int, yes: bool) -> None:
    where = f"空间 {space}" if space is not None else "所有空间"
    if not yes and not typer.confirm(
        f"将永久删除{where}里过期 {days} 天以上的发言及其派生的全部记忆，确定吗？"
    ):
        raise typer.Exit(1)
    with _open_system() as system:
        reports = system.purge_expired(space, older_than=timedelta(days=days))
    if not reports:
        console.print("没有找到要删除的内容。")
        return
    for report in reports.values():
        console.print(f"空间 {report.space_id}：", end="")
        _print_forget_report(report)


def _print_forget_report(report: ForgetReport) -> None:
    console.print(
        f"已删除 {report.turns_deleted} 条轮次（会话 {', '.join(report.sessions_affected)}）；"
        f"事实 {report.facts_before} → {report.facts_after} 条（已按剩余日志重建）；"
        f"清理向量缓存 {report.embed_cache_purged} 条、LLM 缓存 {report.llm_cache_purged} 条；"
        f"耗时 {report.elapsed_ms:.0f} ms"
    )
    if report.rebuild_error:
        console.print(
            f"[yellow]删除已经做了，重建派生记忆时出错（{report.rebuild_error}），"
            "下次巩固时补齐[/yellow]"
        )


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="要问的问题")],
    space: Annotated[str, typer.Option(help="记忆空间 ID")],
    k: Annotated[int | None, typer.Option(help="检索条数，默认取配置")] = None,
    at: Annotated[
        datetime | None,
        typer.Option(formats=DATETIME_FORMATS, help="提问时间，如 2025-10-10；默认为当前时间"),
    ] = None,
    show_context: Annotated[bool, typer.Option("--show-context", help="显示检索到的记忆")] = False,
    mode: Annotated[
        str,
        typer.Option(
            help="l1：L1 快速召回（默认）；l2：深度检索；auto：检索控制器选档（L0 / L1 / L2）；"
            "l0：只用核心记忆；agent：模型自主检索；facts：事实增强检索（B2F）；naive：朴素检索（B2）"
        ),
    ] = "l1",
    router: Annotated[
        str,
        typer.Option(
            help="mode=auto 时用哪种控制器：learned（默认，探针 + 两段式）、"
            "hybrid（拿不准时问 LLM）、rule（规则版 B6r）、llm（零样本 LLM 路由）"
        ),
    ] = "learned",
    lam: Annotated[
        float | None, typer.Option(help="成本旋钮 λ：越大越省钱；默认取控制器模型里的值")
    ] = None,
    show_expired: Annotated[
        bool,
        typer.Option("--show-expired", help="也用已经过期的记忆（写入时给了 expiration_date 的）"),
    ] = False,
) -> None:
    """检索记忆并回答问题。已经过期的记忆默认不用，同 SDK 的 answer（见 expiry.py）。"""
    if mode not in get_args(AskMode):
        console.print(f"[red]--mode 只能是 {'、'.join(get_args(AskMode))}[/red]")
        raise typer.Exit(1)
    if router not in get_args(Router):
        console.print(f"[red]--router 只能是 {'、'.join(get_args(Router))}[/red]")
        raise typer.Exit(1)
    with _open_system() as system:
        started = time.perf_counter()
        answer = system.ask(
            space,
            question,
            question_time=at,
            k=k,
            mode=cast(AskMode, mode),
            router=cast(Router, router),
            lam=lam,
            hide_expired=not show_expired,  # 没有过期的记忆时请求与不打开时逐字相同
        )
        elapsed = (time.perf_counter() - started) * 1000
        if show_context:
            _print_hits(answer)
            _print_process(answer)
        console.print(Panel(answer.answer, title="回答", expand=False))
        cited = ", ".join(answer.citations) or "无"
        sufficient = "是" if answer.evidence_sufficient else "否"
        console.print(f"引用：{cited} · 证据充分：{sufficient} · 总耗时 {elapsed:.0f} ms")
        _print_usage_rows(system.ledger.summary(run_id=system.ledger.run_id))


@app.command("search-image")
def search_image(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="要查找的图片")],
    space: Annotated[str, typer.Option(help="记忆空间 ID")],
    k: Annotated[int, typer.Option(help="返回几张")] = 5,
) -> None:
    """以图搜图：在记忆里找出最相似的图片（"我之前给你看过这个吗？"）。"""
    with _open_system() as system:
        found = system.search_by_image(space, image, k)
        if not found:
            console.print("没有找到图片（这个空间里没有图片，或没有配置图片向量）。")
            return
        table = Table(title=f"最相似的 {len(found)} 张图片")
        for column in ("相似度", "时间", "描述", "原图"):
            table.add_column(column, overflow="fold")
        for item, score in found:
            caption = item.caption or "（还没有描述，运行 cuemem consolidate 生成）"
            table.add_row(f"{score:.3f}", f"{item.ts:%Y-%m-%d}", caption[:120], item.path)
        console.print(table)


@app.command()
def spaces() -> None:
    """列出所有记忆空间。"""
    with _open_db_only() as db:
        rows = db.list_spaces()
    if not rows:
        console.print("还没有任何记忆空间。先运行 cuemem ingest。")
        return
    table = Table(title="记忆空间")
    for column in ("空间", "会话", "轮次", "记忆单元", "事实（当前 / 全部）", "实体", "创建时间"):
        table.add_column(column)
    for r in rows:
        table.add_row(
            r["space_id"],
            str(r["sessions"]),
            str(r["turns"]),
            str(r["units"]),
            f"{r['current_facts']} / {r['facts']}",
            str(r["entities"]),
            r["created_at"][:19],
        )
    console.print(table)


@app.command()
def usage(
    today: Annotated[bool, typer.Option("--today", help="只统计今天")] = False,
    run: Annotated[str | None, typer.Option(help="只统计某次运行（run_id）")] = None,
) -> None:
    """汇总模型调用量与成本。"""
    from cuemem.llm.ledger import Ledger

    settings = _settings()
    ledger = Ledger(settings.paths.ledger)
    since = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) if today else None
    rows = ledger.summary(run_id=run, since=since)
    ledger.close()
    if not rows:
        console.print("没有调用记录。")
        return
    _print_usage_rows(rows, detailed=True)


# ---------- Agent、服务与演示（N8） ----------

SCENARIO_HELP = "shop：电商导购演示（商品目录与按图找同款）；plain：通用助手"


def _service(
    space: str,
    scenario: str,
    router: str,
    lam: float | None,
    *,
    catalog_path: Path | None = None,
    check: Callable[[], None] | None = None,
) -> tuple[Settings, ChatService]:
    """读配置、建服务。check 在读完配置（数据目录的 .env 已经加载）、建服务之前调用：服务一建好
    后台线程就接着做上次没做完的抽取，要拒绝启动的话得在这之前（serve 的访问令牌检查）。"""
    if scenario not in get_args(Scenario):
        console.print(f"[red]--scenario 只能是 {'、'.join(get_args(Scenario))}[/red]")
        raise typer.Exit(1)
    if router not in get_args(Router):
        console.print(f"[red]--router 只能是 {'、'.join(get_args(Router))}[/red]")
        raise typer.Exit(1)
    from cuemem.api.service import ChatService

    # 缺 Key、换了向量模型、没有商品目录等：提示一行原因，不打印调用栈。退出码与调用栈时一样是 1，
    # docker compose 的 restart: unless-stopped 照样反复重启，这一行在 docker compose logs 里
    try:
        settings = _settings(peak_policy="warn")
        if check is not None:
            check()
        _peak_note(settings)
        service = ChatService.create(
            settings,
            scenario=cast(Scenario, scenario),
            catalog_path=catalog_path,
            default_space=space,
            router=cast(Router, router),
            lam=lam,
        )
    except CueMemError as exc:
        console.print(f"[red]错误：[/red]{escape(str(exc))}")
        raise typer.Exit(1) from exc
    return settings, service


def _default_scenario() -> str:
    """serve 不指定 --scenario 时：源码仓库里有导购演示的商品目录，用 shop；pip 安装后没有，
    用通用助手。"""
    from cuemem.api.service import SHOP_CATALOG

    if SHOP_CATALOG.is_file():
        return "shop"
    console.print("[dim]导购演示的商品目录只在源码仓库里：这里用通用助手（--scenario plain）[/dim]")
    return "plain"


INSECURE_HELP = (
    "监听非本机地址（或 CUEMEM_ALLOWED_HOSTS 列了别的主机名、经反向代理对外）又没有设置 "
    "CUEMEM_API_TOKEN 时也照常启动（不鉴权：能连到这台机器的人都能读写、删除记忆，用你的上游 "
    "Key）。只在可信的网络里用"
)
WEAK_TOKEN_HELP = (
    "监听非本机地址（或 CUEMEM_ALLOWED_HOSTS 列了别的主机名）时也接受短于 16 个字符的访问令牌"
)
GRADIO_TEMP_ENV = "GRADIO_TEMP_DIR"
PLAIN_HTTP_NOTE = (
    "[yellow]注意：[/yellow]服务是明文 HTTP：访问令牌（也是界面的密码）与记忆都不加密地经过网络。"
    "可信的网络之外请在前面放 TLS 反向代理（README 的 Public deployment / 公网部署）"
)


def _is_loopback(host: str) -> bool:
    """监听地址是不是只有本机能连（127.0.0.0/8、::1、localhost）。"""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _allowed_hosts_env() -> list[str]:
    """CUEMEM_ALLOWED_HOSTS 列出的主机名（逗号分隔，去空白、小写）。_host_policy 与
    _exposed_names 共用这一个解析。"""
    from cuemem.api.app import HOSTS_ENV

    return [h.strip().lower() for h in os.environ.get(HOSTS_ENV, "").split(",") if h.strip()]


def _loopback_name(name: str) -> bool:
    """CUEMEM_ALLOWED_HOSTS 里的一项是不是只指本机：回环 IP（可带 [] 与端口）、localhost、
    *.localhost（RFC 6761）。"""
    if name.startswith("["):
        name = name[1:].split("]", 1)[0]
    elif name.count(":") == 1:
        name = name.split(":", 1)[0]
    return _is_loopback(name) or name.endswith(".localhost")


def _exposed_names() -> list[str]:
    """CUEMEM_ALLOWED_HOSTS 里不只指本机的项（含 "*"）。监听本机地址时列了这些就是经反向代理
    对外（README 的 Public deployment / 公网部署），要和监听非本机地址一样设令牌（第五轮体检）。"""
    return [h for h in _allowed_hosts_env() if h == "*" or not _loopback_name(h)]


def _access_keys(allow_weak: bool = True) -> KeyRing:
    """访问令牌：CUEMEM_API_TOKEN 与限定作用域的令牌（CUEMEM_API_KEYS_FILE 或 CUEMEM_API_KEYS，
    见 api/auth.py）。在读完配置（数据目录的 .env 已加载）之后、打开记忆系统之前调用；写错时
    AccessKeysError（CueMemError），由调用方打印原因。

    设了 CUEMEM_API_KEYS_FILE 时运行期间改了文件会重新读入（KeyRing，第六轮体检）；allow_weak 为
    False（_weak_refused）时新的文件里有短令牌就不换，与启动时的检查一致。"""
    from cuemem.api.auth import KeyRing

    return KeyRing.from_env(allow_weak=allow_weak)


def _weak_refused(host: str, insecure: bool, allow_weak: bool) -> bool:
    """_require_token 会不会因为短令牌拒绝启动（运行期间重新读入令牌文件时也按它查）。"""
    if insecure or allow_weak:
        return False
    return not _is_loopback(host) or bool(_exposed_names())


def _keys_note(keys: AnyKeys) -> str:
    from cuemem.api.auth import TOKEN_ENV

    parts = [f"管理令牌（{TOKEN_ENV}）"] if keys.admin is not None else []
    if keys.names:
        parts.append(f"{len(keys.names)} 个限定作用域的令牌（{'、'.join(keys.names)}）")
    return "、".join(parts)


def _require_token(host: str, keys: AnyKeys, insecure: bool, allow_weak: bool = False) -> None:
    """监听非本机地址时必须设访问令牌：没有令牌时拒绝启动，除非显式加 --insecure-no-auth。
    容器里总是监听 0.0.0.0（看不出宿主机只映射到了本机），所以镜像也要设令牌。只有限定作用域的
    令牌、没有管理令牌也算设了。

    令牌也要够长（第四轮体检）：直接给出的令牌短于 16 个字符（Keys.weak；以前 README 的例子就是
    your-token）时同样拒绝启动，除非加 --insecure-weak-token（allow_weak）。信息里只有令牌的名字。
    sha256: 给出的量不出长度，不查。加了 --insecure-no-auth（insecure）时两项都不查：设了令牌时
    照样鉴权，短的令牌也放过。

    监听本机地址但 CUEMEM_ALLOWED_HOSTS 列了别的主机名（或 "*"）时同样要查（第五轮体检）：
    这是 README 公网部署的做法，TLS 反向代理转发进来的请求都会被接受。"""
    from cuemem.api.app import HOSTS_ENV
    from cuemem.api.auth import KEYS_FILE_ENV, MIN_TOKEN_CHARS, TOKEN_ENV

    if insecure:
        return
    exposed = _exposed_names() if _is_loopback(host) else []
    if _is_loopback(host) and not exposed:
        return
    where = f"监听 {escape(host)}"
    if exposed:
        where += f"，但 {HOSTS_ENV} 列出了 {escape('、'.join(exposed))}（经反向代理对外），"
    else:
        where += " 但"
    if not keys.enabled:
        console.print(
            f"[red]{where}没有设置 {TOKEN_ENV}，拒绝启动：能连到这台机器的人都能"
            f"读写、删除记忆，用你的上游 Key。[/red]\n在 .env 里设置 {TOKEN_ENV}（例如用 openssl "
            f"rand -hex 32 生成），或用 {KEYS_FILE_ENV} 给出限定作用域的令牌；只在可信的网络里"
            "使用时也可以加 --insecure-no-auth。"
        )
        raise typer.Exit(2)
    if keys.weak and _weak_refused(host, insecure, allow_weak):
        console.print(
            f"[red]{where}这些访问令牌短于 {MIN_TOKEN_CHARS} 个字符，拒绝启动："
            f"{escape('、'.join(keys.weak))}。[/red]\n换成随机生成的长令牌（例如 openssl rand "
            f"-hex 32）；sha256: 给出的令牌不查长度。确实要用短令牌时加 --insecure-weak-token。"
        )
        raise typer.Exit(2)


UNKNOWN_MODEL_HELP = (
    "记忆代理收到模型目录里没有的 model 时：remap 改用 agent 角色的模型（响应头 X-CueMem-Model "
    "写明，日志里记一条警告）；reject 返回 400。默认取配置 proxy.unknown_model（remap）"
)
# serve 与 proxy 收到 SIGTERM 后，等进行中的请求（例如流式的 /v1 响应）最多这么多秒，之后才接着
# 收尾（记忆代理重试缓冲着的写回、结束会话、等后台巩固最多 ConsolidationWorker.stop_timeout_s）。
# 以前不限：一个不结束的连接拖住收尾，docker-compose.yml 的 stop_grace_period 一过就被 SIGKILL
GRACEFUL_SHUTDOWN_S = 20
IDLE_HELP = (
    "会话闲置多少分钟后结束并交给抽取：界面与 /api 的会话、/api/v1 的 run、/v1 记忆代理的会话。"
    "之后再往 /api 的会话发消息是 404，界面开一个新的会话，/api/v1 的 run 再 add 时重新打开"
)


def _unknown_model(value: str | None) -> UnknownModel | None:
    """--unknown-model 的检查：写错时提示、退出码 1（在打开记忆系统之前）。"""
    if value is None:
        return None
    if value in get_args(UnknownModel):
        return cast(UnknownModel, value)
    console.print(f"[red]--unknown-model 只能是 {'、'.join(get_args(UnknownModel))}[/red]")
    raise typer.Exit(1)


class HostPolicy(NamedTuple):
    """create_app / create_proxy_app 的 allowed_hosts、origin_hosts 与 allow_ip_hosts。"""

    allowed: list[str] | None  # Host 白名单；None 不查
    origins: list[str]  # 除同源外放行的 Origin 主机名
    ip_literals: bool  # 另外接受任何 IP 地址作为 Host


def _host_policy(host: str, *, authenticated: bool = True) -> HostPolicy:
    """防 DNS 重绑定（恶意域名解析到这台机器后，它的网页就算同源）的 Host 检查：

    - 监听本机地址：只接受 Host 为本机的请求；
    - 监听别的地址又设了访问令牌（authenticated）：不查 Host（Docker、局域网用什么地址访问都行，
      别的网页拿不到令牌）；
    - 监听别的地址又不鉴权（--insecure-no-auth，第四轮体检）：接受任何 IP 地址、localhost 与
      监听的主机名，别的主机名（局域网的 nas.local、compose 里的服务名）要列在
      CUEMEM_ALLOWED_HOSTS 里。以前也不查：没有令牌，任何网页都能经重绑定读写记忆、花上游的额度。

    CUEMEM_ALLOWED_HOSTS 列出的主机名两边都放行，含 "*" 时不查 Host、不限 Origin。"""
    from cuemem.api.app import LOOPBACK_HOSTS

    extra = _allowed_hosts_env()
    if "*" in extra:
        return HostPolicy(None, extra, False)
    if _is_loopback(host):
        bind = f"[{host}]" if ":" in host else host.lower()  # 例如 127.0.0.2
        return HostPolicy(list(dict.fromkeys([*LOOPBACK_HOSTS, bind, *extra])), extra, False)
    if authenticated:
        return HostPolicy(None, extra, False)
    try:
        ipaddress.ip_address(host)
        named = []  # 0.0.0.0、::、局域网地址：IP 地址都接受
    except ValueError:
        named = [host.lower()] if host else []
    return HostPolicy(list(dict.fromkeys(["localhost", *named, *extra])), extra, True)


def _no_auth_warning(host: str, can: str) -> None:
    """--insecure-no-auth 对外时的启动提示：监听非本机地址，或监听本机地址但
    CUEMEM_ALLOWED_HOSTS 列了别的主机名（经反向代理对外，第五轮复审：以前这时什么都不提示）。"""
    from cuemem.api.app import HOSTS_ENV
    from cuemem.api.auth import TOKEN_ENV

    exposed = _exposed_names() if _is_loopback(host) else []
    via = f"，{HOSTS_ENV} 列出了 {escape('、'.join(exposed))}（经反向代理对外）" if exposed else ""
    console.print(
        f"[yellow]注意：[/yellow]监听 {escape(host)}{via}，但没有设置 {TOKEN_ENV}"
        f"（--insecure-no-auth），能连到这台机器的人都能{can}"
    )


def _no_auth_hosts_note(hosts: HostPolicy) -> None:
    """--insecure-no-auth 监听非本机地址时的启动提示：只接受哪些 Host（别的回 400
    "Invalid host header"，局域网里用主机名访问的人不知道为什么）。"""
    from cuemem.api.app import HOSTS_ENV

    if not hosts.ip_literals:  # CUEMEM_ALLOWED_HOSTS 含 "*"：不查 Host
        return
    console.print(
        f"只接受 IP 地址、{escape('、'.join(hosts.allowed or []))} 作为 Host（防 DNS 重绑定）；"
        f"用别的主机名访问时把它加进 {HOSTS_ENV}（逗号分隔）"
    )


@contextmanager
def _gradio_temp_dir(ui: bool) -> Iterator[None]:
    """界面（Gradio）放上传与图库副本的目录（第四轮体检）：运营者没设 GRADIO_TEMP_DIR 时换成这次
    启动自己的目录（mkdtemp：0700、名字随机，别的账户不能先建好），停下时删掉。

    Gradio 默认用 /tmp/gradio：整台机器的 Gradio 应用共用、目录是 0755，界面的图库把媒体库里的
    原图复制进去（copy2，权限照旧），别的本地用户能读；那个目录被别的账户先建好时，上传要么失败、
    要么交给了它。不放在数据目录下：进程崩溃时照片的副本不留在 /data 卷上。Gradio 在 create_app
    里才第一次导入，建 Blocks 与 App 时读这个变量。"""
    previous = os.environ.get(GRADIO_TEMP_ENV)
    if not ui or previous:
        yield
        return
    directory = tempfile.mkdtemp(prefix="cuemem-gradio-")
    os.environ[GRADIO_TEMP_ENV] = directory
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(GRADIO_TEMP_ENV, None)
        else:
            os.environ[GRADIO_TEMP_ENV] = previous
        shutil.rmtree(directory, ignore_errors=True)


LOG_LEVEL_ENV = "CUEMEM_LOG_LEVEL"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _configure_logging() -> None:
    """serve、proxy 与 mcp 的日志（第八轮体检 P13）：一条一行，带时间、级别与 logger 名字，写到
    stderr（stdio 的 MCP 用 stdout 当协议通道）。级别取 CUEMEM_LOG_LEVEL（DEBUG、INFO、WARNING、
    ERROR、CRITICAL，默认 WARNING），写错时说明并退出（退出码 2）。只读进程的环境变量：这时还没有
    读入 .env。

    要在建 MCPServer 之前调用：它的构造函数调用 logging.basicConfig 装上 RichHandler，root 已经
    有处理器时那次调用什么也不做。以前装了 MCP（serve 默认）时日志是 Rich 的框，非终端里按 80 列
    折行（grep 找不到整句），一次出错几百行；--no-mcp 时 root 没有处理器，警告走 Python 的
    lastResort，没有时间、级别与 logger 名字。不用 force=True：不拆掉别人（pytest 的 caplog、
    嵌入 CueMem 的应用）已经装上的处理器；SDK 与库的导入路径不碰日志设置。"""
    raw = os.environ.get(LOG_LEVEL_ENV, "").strip()
    level = logging.WARNING
    if raw:
        named = logging.getLevelNamesMapping().get(raw.upper())
        if named is None or named == logging.NOTSET:
            Console(stderr=True).print(
                f"[red]{LOG_LEVEL_ENV} 只能是 DEBUG、INFO、WARNING、ERROR、CRITICAL[/red]"
                f"（现在是 {escape(raw)}）"
            )
            raise typer.Exit(2)
        level = named
    logging.basicConfig(level=level, format=LOG_FORMAT, stream=sys.stderr)


def _sigterm_to_exit(signum: int, frame: object) -> None:
    """SIGTERM 变成 SystemExit(143)：经过 serve 与 proxy 的 finally 收尾后再退出。"""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)  # 收尾时再来一次 SIGTERM：照旧直接结束
    raise SystemExit(128 + signum)


@contextmanager
def _sigterm_runs_cleanup() -> Iterator[None]:
    """uvicorn.run 期间 SIGTERM 也走收尾（第八轮体检 P2）。

    uvicorn 收到 SIGTERM 先等进行中的请求，再恢复原来的处理函数并把信号重发给自己。原来是默认
    处理（SIG_DFL）时进程当场结束：记忆代理缓冲着的写回、结束会话、等后台巩固、WAL 检查点与
    界面的临时目录（原图的副本）都被跳过。容器里 cuemem 是 1 号进程时内核忽略这次重发，所以
    以前只有 Docker 不加 init 时才收尾；systemd、kill、--init 都不收尾。这里装上的处理函数被
    uvicorn 当作"原来的"，重发时抛 SystemExit，finally 照常执行，退出码仍是 143。"""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _sigterm_to_exit)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="监听地址")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="端口")] = 7860,
    space: Annotated[str, typer.Option(help="界面默认的记忆空间")] = "shop-demo",
    scenario: Annotated[
        str | None,
        typer.Option(help=SCENARIO_HELP + "；默认在源码仓库里是 shop，pip 安装后是 plain"),
    ] = None,
    router: Annotated[str, typer.Option(help="默认的检索控制器")] = "learned",
    lam: Annotated[float | None, typer.Option(help="成本旋钮 λ；默认取控制器模型里的值")] = None,
    no_ui: Annotated[bool, typer.Option("--no-ui", help="只启动 HTTP 接口")] = False,
    no_mcp: Annotated[
        bool, typer.Option("--no-mcp", help="不在 /mcp 提供 MCP（Streamable HTTP）")
    ] = False,
    insecure_no_auth: Annotated[
        bool, typer.Option("--insecure-no-auth", help=INSECURE_HELP)
    ] = False,
    insecure_weak_token: Annotated[
        bool, typer.Option("--insecure-weak-token", help=WEAK_TOKEN_HELP)
    ] = False,
    unknown_model: Annotated[str | None, typer.Option(help=UNKNOWN_MODEL_HELP)] = None,
    idle_minutes: Annotated[float, typer.Option(help=IDLE_HELP)] = 30.0,
    access_log: Annotated[
        bool, typer.Option("--access-log/--no-access-log", help="每个请求记一行访问日志")
    ] = False,
) -> None:
    """启动 HTTP 接口（/api，文档在 /docs）、记忆代理（/v1）、MCP（/mcp）与演示界面（Gradio，
    根路径）。

    界面要用可选依赖 ui（Gradio）；没装时只启动 HTTP 接口与记忆代理，并说明怎么装。/mcp 与
    cuemem mcp 的工具相同，共用这个进程的记忆系统（cuemem mcp 不能与 serve 同时打开数据目录），
    请求头 X-CueMem-User 选用户（不是 ASCII 的 ID 写成 utf-8''%E5%BC%A0%E4%B8%89 这样的 UTF-8
    百分号编码），令牌同 /api。"""
    _configure_logging()  # 在建 MCPServer 之前（见 _configure_logging）
    import uvicorn

    from cuemem.api.app import create_app
    from cuemem.api.auth import KeyRing, Keys
    from cuemem.api.limits import MAX_BODY_BYTES, MAX_IMPORT_BYTES, body_limits, concurrency_limit
    from cuemem.api.proxy import MemoryProxy
    from cuemem.api.service import open_session_limit

    if not idle_minutes > 0:  # 在打开记忆系统之前检查
        console.print("[red]--idle-minutes 要大于 0[/red]")
        raise typer.Exit(1)
    ui = not no_ui
    if ui and importlib.util.find_spec("gradio") is None:
        console.print(
            "[yellow]注意：[/yellow]演示界面要用可选依赖 ui（Gradio），没有安装："
            f"{escape(extra_remedy('ui'))}。这次只启动 HTTP 接口与记忆代理"
        )
        ui = False
    unknown = _unknown_model(unknown_model)
    keys = KeyRing(Keys())
    limits = MAX_BODY_BYTES, MAX_IMPORT_BYTES

    def check() -> None:  # 加载配置时已读入 .env；建服务之前检查，被拒绝时不接着做上次的抽取
        nonlocal keys, limits
        keys = _access_keys(not _weak_refused(host, insecure_no_auth, insecure_weak_token))
        limits = body_limits()  # CUEMEM_MAX_BODY_BYTES、CUEMEM_MAX_IMPORT_BYTES；写错时退出码 1
        concurrency_limit()  # CUEMEM_MAX_CONCURRENT_PER_KEY（建应用时再读一次）
        open_session_limit()  # CUEMEM_MAX_OPEN_SESSIONS（建服务时再读一次）
        _require_token(host, keys, insecure_no_auth, insecure_weak_token)

    settings, service = _service(space, scenario or _default_scenario(), router, lam, check=check)
    hosts = _host_policy(host, authenticated=keys.enabled)
    # 同一个记忆系统：经代理写入的对话在界面里也能看到
    proxy = MemoryProxy(service.system, unknown_model=unknown, idle_minutes=idle_minutes)
    try:
        with _gradio_temp_dir(ui):
            app_ = create_app(
                service,
                upload_dir=settings.paths.media.parent / "uploads",
                ui=ui,
                keys=keys,
                proxy=proxy,
                allowed_hosts=hosts.allowed,
                origin_hosts=hosts.origins,
                allow_ip_hosts=hosts.ip_literals,
                idle_minutes=idle_minutes,
                max_body_bytes=limits[0],
                max_import_bytes=limits[1],
                mcp=not no_mcp,
            )
            where = f"界面：[bold]http://{host}:{port}/[/bold] · " if ui else ""
            console.print(
                f"{where}接口文档：http://{host}:{port}/docs"
                f" · OpenAI 兼容的记忆代理：http://{host}:{port}/v1"
            )
            if not no_mcp:
                console.print(
                    f"MCP（Streamable HTTP）：http://{host}:{port}/mcp"
                    "（请求头 X-CueMem-User 选用户，不是 ASCII 的写成 utf-8''<百分号编码>；"
                    "令牌同 /api；--no-mcp 关掉）"
                )
            if keys.enabled:
                console.print(
                    f"已启用访问令牌：{_keys_note(keys)}。接口要带 Bearer 令牌，界面要先登录"
                )
                # 容器里总是 0.0.0.0：Docker 也提示；本机地址经反向代理对外时也提示
                if not _is_loopback(host) or _exposed_names():
                    console.print(PLAIN_HTTP_NOTE)
            elif not _is_loopback(host) or _exposed_names():  # 加了 --insecure-no-auth
                _no_auth_warning(host, "读写、删除记忆")
                _no_auth_hosts_note(hosts)
            for line in _loose_modes():
                console.print(line)
            with _sigterm_runs_cleanup():
                uvicorn.run(
                    app_,
                    host=host,
                    port=port,
                    # 访问日志是 INFO：要它时 uvicorn 的启动信息也一起出来
                    log_level="info" if access_log else "warning",
                    access_log=access_log,
                    timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
                )
    finally:
        proxy.close_all()
        service.close()


@app.command()
def proxy(
    host: Annotated[str, typer.Option(help="监听地址")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="端口")] = 8787,
    user: Annotated[
        str | None,
        typer.Option(
            help="请求里没有 user 字段、也没有 X-CueMem-User 头时用的用户；不填则不碰记忆"
        ),
    ] = None,
    mode: Annotated[str, typer.Option(help="检索档位：auto（控制器选）、l0、l1、l2")] = "auto",
    idle_minutes: Annotated[float, typer.Option(help="会话空闲多少分钟后结束并抽取事实")] = 30.0,
    insecure_no_auth: Annotated[
        bool, typer.Option("--insecure-no-auth", help=INSECURE_HELP)
    ] = False,
    insecure_weak_token: Annotated[
        bool, typer.Option("--insecure-weak-token", help=WEAK_TOKEN_HELP)
    ] = False,
    unknown_model: Annotated[str | None, typer.Option(help=UNKNOWN_MODEL_HELP)] = None,
    access_log: Annotated[
        bool, typer.Option("--access-log/--no-access-log", help="每个请求记一行访问日志")
    ] = True,
) -> None:
    """OpenAI 兼容的记忆代理（N12）：把应用的 base_url 换成 http://HOST:PORT/v1 就有了长期记忆。"""
    _configure_logging()
    import uvicorn

    from cuemem.api.limits import body_limits, concurrency_limit, open_session_limit
    from cuemem.api.proxy import MemoryProxy, create_proxy_app
    from cuemem.memory import MemorySystem

    if mode not in get_args(RecallMode):
        console.print(f"[red]--mode 只能是 {'、'.join(get_args(RecallMode))}[/red]")
        raise typer.Exit(1)
    unknown = _unknown_model(unknown_model)
    settings = _settings(peak_policy="warn")
    try:
        # load_settings 已读入 .env
        keys = _access_keys(not _weak_refused(host, insecure_no_auth, insecure_weak_token))
        max_body_bytes, _ = body_limits()  # CUEMEM_MAX_BODY_BYTES（代理没有导入接口）
        concurrency_limit()  # CUEMEM_MAX_CONCURRENT_PER_KEY（建应用时再读一次）
        open_session_limit()  # CUEMEM_MAX_OPEN_SESSIONS（建代理时再读一次）
    except CueMemError as exc:
        console.print(f"[red]错误：[/red]{escape(str(exc))}")
        raise typer.Exit(1) from exc
    _require_token(host, keys, insecure_no_auth, insecure_weak_token)
    hosts = _host_policy(host, authenticated=keys.enabled)
    try:
        system = MemorySystem.from_settings(settings)
    except CueMemError as exc:  # 缺 Key、换了向量模型等：提示原因，不打印调用栈
        console.print(f"[red]错误：[/red]{escape(str(exc))}")
        raise typer.Exit(1) from exc
    memory_proxy = MemoryProxy(
        system,
        default_user=user,
        mode=cast(RecallMode, mode),
        idle_minutes=idle_minutes,
        unknown_model=unknown,
    )
    try:
        console.print(
            f"记忆代理：[bold]http://{host}:{port}/v1[/bold]（OpenAI 兼容；"
            f"上游默认模型 {settings.roles['agent'].model}）"
        )
        console.print(
            "用户：请求体的 user 字段，或请求头 X-CueMem-User"
            "（不是 ASCII 的写成 utf-8''<百分号编码>）"
            + (f"；默认 {user}" if user else "；没有时原样转发、不碰记忆")
        )
        if keys.enabled:
            console.print(f"已启用访问令牌：{_keys_note(keys)}。OpenAI SDK 的 api_key 填令牌")
            if not _is_loopback(host) or _exposed_names():
                console.print(PLAIN_HTTP_NOTE)
        elif not _is_loopback(host) or _exposed_names():  # 加了 --insecure-no-auth
            _no_auth_warning(host, "用你的上游 Key、读写记忆")
            _no_auth_hosts_note(hosts)
        for line in _loose_modes():
            console.print(line)
        app_ = create_proxy_app(
            memory_proxy,
            keys=keys,
            allowed_hosts=hosts.allowed,
            origin_hosts=hosts.origins,
            allow_ip_hosts=hosts.ip_literals,
            max_body_bytes=max_body_bytes,
        )
        with _sigterm_runs_cleanup():
            uvicorn.run(
                app_,
                host=host,
                port=port,
                access_log=access_log,
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
            )
    finally:
        memory_proxy.close_all()
        system.close()


keys_app = typer.Typer(
    help="限定作用域的访问令牌（CUEMEM_API_KEYS_FILE 的文件）：生成、列出、吊销。运行中的 serve 与 "
    "proxy 一秒内读到改动，不用重启",
    no_args_is_help=True,
)
app.add_typer(keys_app, name="keys")
KEYS_FILE_HELP = "令牌文件；默认取环境变量 CUEMEM_API_KEYS_FILE（数据目录的 .env 里设的也算）"
# keys add 新建文件时写在开头
KEYS_FILE_HEADER = (
    "# CueMem 限定作用域的访问令牌（格式见 cuemem/api/auth.py）。\n"
    "# 只存令牌的 sha256，不存令牌本身；用 cuemem keys add / revoke 增删，\n"
    "# 运行中的 cuemem serve / proxy 一秒内读到改动\n"
)


def _keys_path(file: Path | None, *, edit: bool) -> Path | None:
    """keys 命令的令牌文件：--file，否则 CUEMEM_API_KEYS_FILE（先读入数据目录的 .env，与 serve
    一样，已经设了的环境变量优先）。都没有时：edit（add、revoke）报错退出；list 返回 None（看
    CUEMEM_API_KEYS 里的）。令牌写在 CUEMEM_API_KEYS 里时 add、revoke 拒绝：改了文件服务也不读。"""
    from dotenv import load_dotenv

    from cuemem.api.auth import KEYS_ENV, KEYS_FILE_ENV

    load_dotenv(env_file_path(), override=False)
    err = Console(stderr=True)
    if edit and os.environ.get(KEYS_ENV):
        err.print(
            f"[red]错误：[/red]令牌写在环境变量 {KEYS_ENV} 里，cuemem keys 只改文件：把它们移进一个"
            f"文件，用 {KEYS_FILE_ENV} 指向它（两个只能设一个）"
        )
        raise typer.Exit(1)
    if file is None and (env := os.environ.get(KEYS_FILE_ENV)):
        file = Path(env)
    if file is None and edit:
        err.print(
            f"[red]错误：[/red]没有设置 {KEYS_FILE_ENV}：用 --file 指定令牌文件，并在 .env 里设置 "
            f"{KEYS_FILE_ENV}，cuemem serve / proxy 才会读它"
        )
        raise typer.Exit(1)
    return file.expanduser() if file is not None else None


def _read_keys(path: Path) -> tuple[str, Keys]:
    """令牌文件的文本与解析结果；读不出、写错时打印原因退出（出错信息里不带令牌本身）。"""
    from cuemem.api.auth import Keys

    try:
        text = path.read_text(encoding="utf-8")
        return text, Keys.parse(text, source=str(path))
    except OSError as exc:
        Console(stderr=True).print(
            f"[red]错误：[/red]读不出 {escape(str(path))}：{escape(exc.strerror or str(exc))}"
        )
        raise typer.Exit(1) from exc
    except CueMemError as exc:
        Console(stderr=True).print(f"[red]错误：[/red]{escape(str(exc))}")
        raise typer.Exit(1) from exc


def _write_keys(path: Path, text: str) -> None:
    """换名替换（envfile.atomic_write_text：运行中的服务读到的不是旧的就是新的），再改成 0600。
    所在的目录要已经有（keys add 事先查过）：不替人建目录，路径写错时不悄悄建出一个没人读的文件。"""
    os.chmod(atomic_write_text(path, text), 0o600)


@keys_app.command("add")
def keys_add(
    name: Annotated[str, typer.Argument(help="令牌的名字（也是界面登录的用户名）")],
    user: Annotated[
        list[str] | None,
        typer.Option("--user", help='能访问的用户 ID；可重复；"*" 是任何用户，"acme:*" 是前缀'),
    ] = None,
    agent: Annotated[
        list[str] | None,
        typer.Option(
            "--agent",
            help="与 --user 一起：只限这些 Agent 下面的记忆；只写 --agent：这些 Agent 自己的记忆",
        ),
    ] = None,
    rate_per_minute: Annotated[
        int | None, typer.Option(min=1, help="每分钟最多这么多个写的请求（429）")
    ] = None,
    models: Annotated[
        list[str] | None,
        typer.Option("--models", help="记忆代理只能用这些模型（别名或模型 ID）；可重复"),
    ] = None,
    max_output_tokens: Annotated[
        int | None, typer.Option(min=1, help="记忆代理转发的 max_tokens 上限")
    ] = None,
    max_concurrent: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="同时在处理的请求最多这么多个（429）；不给时用 CUEMEM_MAX_CONCURRENT_PER_KEY",
        ),
    ] = None,
    max_spaces: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="最多这么多个记忆空间：要建新空间的写入 429，已有的照常；不给时不限",
        ),
    ] = None,
    file: Annotated[Path | None, typer.Option("--file", help=KEYS_FILE_HELP)] = None,
) -> None:
    """生成一个限定作用域的令牌，把它的 sha256 加进令牌文件（注释与写法留着，文件改成 0600）。

    令牌只打印这一次，只打到标准输出（TOKEN=$(cuemem keys add …) 这样接住），文件里不存它；
    别的提示都在标准错误。文件原来就有时，运行中的 cuemem serve / proxy 一秒内开始认它；文件不在
    时新建（所在的目录要已经有，否则报错退出，不打印令牌），服务要用指向它的
    CUEMEM_API_KEYS_FILE 启动（或重启）才读它。"""
    import hashlib
    import secrets

    from cuemem.api.auth import Keys, append_key, key_entry

    err = Console(stderr=True)
    path = _keys_path(file, edit=True)
    assert path is not None
    if not user and not agent:
        err.print("[red]错误：[/red]至少给一个 --user 或 --agent（令牌能访问谁的记忆）")
        raise typer.Exit(1)
    shown = escape(str(path.resolve()))  # 相对路径按各自进程的当前目录解析：提示里给绝对路径
    exists = path.exists()
    if not exists and not path.resolve().parent.is_dir():
        err.print(
            f"[red]错误：[/red]{shown} 不在，它所在的目录 {escape(str(path.resolve().parent))} "
            "也不在：路径是不是写错了（或者用了容器里的路径）？要新建令牌文件，先建好目录"
        )
        raise typer.Exit(1)
    text, old = _read_keys(path) if exists else ("", Keys())
    if name in old.names:
        err.print(f"[red]错误：[/red]{escape(str(path))} 里已经有名叫 {escape(name)} 的令牌")
        raise typer.Exit(1)
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()

    def entry(indent: str) -> str:
        return key_entry(
            name,
            digest,
            users=user or (),
            agents=agent or (),
            rate_per_minute=rate_per_minute,
            models=models,
            max_output_tokens=max_output_tokens,
            max_concurrent=max_concurrent,
            max_spaces=max_spaces,
            indent=indent,
        )

    new = append_key(text, entry) if exists else f"{KEYS_FILE_HEADER}keys:\n{entry('  ')}"
    try:  # 核对：原来的令牌一个不少，新的这个认得出、名字对
        parsed = Keys.parse(new, source=str(path)) if new is not None else None
    except CueMemError as exc:
        if not exists:  # 新建的文件只有这一项：是参数写错了（例如名字前后有空格）
            err.print(f"[red]错误：[/red]{escape(str(exc))}")
            raise typer.Exit(1) from exc
        parsed = None
    found = parsed.resolve(token) if parsed is not None else None
    if (
        new is None
        or parsed is None
        or parsed.scoped[:-1] != old.scoped
        or found is None
        or found.name != name
    ):
        err.print(
            f"{escape(str(path))} 的 keys: 不是一项一行开头的列表（例如写成了 keys: [...]），没有"
            f"改它。把下面这几行加进它的 keys: 列表：\n\n{escape(entry('  '))}\n"
            "贴进去之前这个令牌不能用（退出码 1）"
        )
        print(token)
        raise typer.Exit(1)
    else:
        _write_keys(path, new)
        if exists:
            err.print(
                f"已把 {escape(name)} 加进 {shown}（只存 sha256）。令牌只显示这一次，"
                "在标准输出；运行中的 cuemem serve / proxy 一秒内开始认它"
            )
        else:
            err.print(
                f"已新建 {shown}（0600），把 {escape(name)} 加了进去（只存 sha256）。令牌只显示"
                "这一次，在标准输出。cuemem serve / proxy 只有用 "
                f"CUEMEM_API_KEYS_FILE={shown} 启动时才读它（在 Docker 里是容器里的路径）："
                "带上这个设置启动或重启它们"
            )
        if user and not agent and max_spaces is None:
            err.print(
                "[yellow]提醒[/yellow]：只给了 --user、没给 --agent，这个令牌可以用任意 agent_id，"
                "每个新的 agent_id 是一个新的记忆空间（各有一个向量集合，本地模式下一直开着一个"
                "文件）。交给不太可信的调用方时加上 --agent 或 --max-spaces"
            )
    print(token)


def _scope_text(scope: Any) -> str:
    users = f"user={','.join(scope.users)}" if scope.users is not None else ""
    agents = f"agent={','.join(scope.agents)}" if scope.agents is not None else ""
    return " ".join(x for x in (users, agents) if x)


@keys_app.command("list")
def keys_list(
    file: Annotated[Path | None, typer.Option("--file", help=KEYS_FILE_HELP)] = None,
) -> None:
    """列出限定作用域的令牌：名字、作用域与用量限制（不显示令牌与摘要）。"""
    from cuemem.api.auth import KEYS_ENV, Keys

    path = _keys_path(file, edit=False)
    if path is not None:
        _, keys = _read_keys(path)
        where = str(path)
    elif inline := os.environ.get(KEYS_ENV):
        try:
            keys = Keys.parse(inline, source=KEYS_ENV)
        except CueMemError as exc:
            Console(stderr=True).print(f"[red]错误：[/red]{escape(str(exc))}")
            raise typer.Exit(1) from exc
        where = KEYS_ENV
    else:
        console.print("没有设置限定作用域的令牌（CUEMEM_API_KEYS_FILE 或 CUEMEM_API_KEYS）")
        return
    table = Table(title=f"{escape(where)}：{len(keys.names)} 个令牌")
    columns = ("名字", "作用域", "每分钟写", "同时处理", "空间上限", "模型", "max_output_tokens")
    for column in columns:
        table.add_column(column)
    for _, p in keys.scoped:
        table.add_row(
            escape(p.name),
            escape("；".join(_scope_text(s) for s in p.scopes)),
            str(p.rate_per_minute or "—"),
            str(p.max_concurrent or "—"),
            str(p.max_spaces or "—"),
            escape(", ".join(p.models)) if p.models is not None else "—",
            str(p.max_output_tokens or "—"),
        )
    console.print(table)


@keys_app.command("revoke")
def keys_revoke(
    name: Annotated[str, typer.Argument(help="要吊销的令牌的名字")],
    file: Annotated[Path | None, typer.Option("--file", help=KEYS_FILE_HELP)] = None,
) -> None:
    """从令牌文件里删掉这个令牌的那几行（别的行与注释不动）。运行中的 cuemem serve / proxy
    一秒内不再认它，不用重启。删不干净（与别的内容写在同一行这样的写法）时不改文件，说明要手工删
    哪几行。"""
    from cuemem.api.auth import TOKEN_ENV, Keys, key_lines, remove_key

    err = Console(stderr=True)
    path = _keys_path(file, edit=True)
    assert path is not None
    text, old = _read_keys(path)
    if name not in old.names:
        others = "、".join(old.names) or "没有"
        err.print(
            f"[red]错误：[/red]{escape(str(path))} 里没有名叫 {escape(name)} 的令牌"
            f"（有：{escape(others)}）"
        )
        raise typer.Exit(1)
    new = remove_key(text, name)
    try:
        parsed = Keys.parse(new, source=str(path)) if new is not None else None
    except CueMemError:
        parsed = None
    if parsed is None or parsed.scoped != tuple(k for k in old.scoped if k[1].name != name):
        where = key_lines(text, name)
        lines = f"第 {where[0] + 1}–{where[1] + 1} 行" if where is not None else "它的那几行"
        err.print(
            f"[red]没能确定地删掉 {escape(name)}，没有改文件。[/red]请手工删掉 "
            f"{escape(str(path))} 的{lines}（名叫 {escape(name)} 的那一项）"
        )
        raise typer.Exit(1)
    assert new is not None
    _write_keys(path, new)
    err.print(
        f"已从 {escape(str(path.resolve()))} 删掉 {escape(name)}；"
        "运行中的 cuemem serve / proxy 一秒内不再认它"
    )
    if not parsed.scoped and not os.environ.get(TOKEN_ENV):
        err.print(
            f"[yellow]注意：[/yellow]这是最后一个令牌，又没有设置 {TOKEN_ENV}：运行中的服务不会因此"
            "关掉鉴权，仍认原来的令牌（包括这一个），要停掉服务（或加一个令牌）"
        )


@app.command()
def chat(
    space: Annotated[str, typer.Option(help="记忆空间 ID")],
    scenario: Annotated[str, typer.Option(help=SCENARIO_HELP)] = "plain",
    at: Annotated[
        datetime | None,
        typer.Option(formats=DATETIME_FORMATS, help="模拟的会话开始时间；默认为现在"),
    ] = None,
    router: Annotated[str, typer.Option(help="检索控制器")] = "learned",
    lam: Annotated[float | None, typer.Option(help="成本旋钮 λ")] = None,
) -> None:
    """在终端里与 Agent 对话。"/image 路径 文字" 附图；/end 结束会话并巩固；/quit 退出。"""
    _, service = _service(space, scenario, router, lam)
    session = service.start(space, at=at)
    console.print(
        f"会话 [bold]{session.session_id}[/bold]（{session.started_at:%Y-%m-%d %H:%M}）。"
        "/image 路径 文字：附图；/end：结束并巩固；/quit：退出（会话交给后台巩固）。"
    )
    try:
        while True:
            try:
                line = console.input("[bold green]你[/bold green] › ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line == "/quit":
                break
            if line == "/end":
                out = service.end(session.session_id, wait=True)
                console.print(f"[dim]会话结束：{out.get('consolidation')}[/dim]")
                session = service.start(space)
                console.print(f"[dim]新会话 {session.session_id}[/dim]")
                continue
            images: list[str] = []
            if line.startswith("/image "):
                _, path, *rest = line.split(" ", 2)
                images, line = [path], (rest[0] if rest else "")
            reply = service.chat(session.session_id, line, images=images)
            console.print(Panel(escape(reply.text), title="Cue", expand=False))
            console.print(f"[dim]{_trace_line(reply)}[/dim]")
    finally:
        service.close()


@app.command()
def demo(
    script: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="演示剧本 JSON")],
    space: Annotated[str | None, typer.Option(help="记忆空间；默认取剧本里的")] = None,
    until: Annotated[
        int | None, typer.Option(help="只回放前 N 段会话（现场演示前建好记忆）")
    ] = None,
    fresh: Annotated[
        bool, typer.Option("--fresh", help="先清空这个记忆空间（LLM 缓存保留，回放结果可复现）")
    ] = False,
    router: Annotated[str, typer.Option(help="检索控制器")] = "learned",
    lam: Annotated[float | None, typer.Option(help="成本旋钮 λ")] = None,
    report: Annotated[Path | None, typer.Option(help="回放报告（JSON）存放路径")] = None,
) -> None:
    """按演示剧本回放对话：每段会话结束后同步巩固；核对回答里该有、不该有的内容。"""
    from cuemem.api.replay import DemoScript, TurnResult, replay, save_report

    plan = DemoScript.load(script)
    space_id = space or plan.space
    # 剧本旁边有商品目录就用它（demo/shop/ 里就是这样）：不在源码仓库里也能回放
    catalog = script.parent / "catalog.json"
    settings, service = _service(
        space_id, "shop", router, lam, catalog_path=catalog if catalog.is_file() else None
    )
    try:
        if fresh:
            # 在巩固锁里清空：后台线程一启动就接着做上次中途退出时没做完的任务，可能正在巩固
            # 这个空间，锁外清空的话它随后又把事实与向量写回来。不用 system.delete_space：
            # 它连 LLM 缓存一起清，回放就不可复现了
            system = service.system
            with system.consolidator.exclusive(space_id):
                system.vectors.delete_space(space_id)  # 先于数据库，见 VectorStore.delete_space
                system.db.delete_space(space_id)
            console.print(f"[dim]已清空记忆空间 {space_id}[/dim]")

        def show(result: TurnResult) -> None:
            mark = "[green]✓[/green]" if result.passed else "[red]✗[/red]"
            head = f"S{result.session}·{result.index} {mark}"
            if result.forgot is not None:
                console.print(f"{head} 遗忘 {escape(str(result.forgot))}")
                return
            assert result.reply is not None
            level = result.level + (
                f"（期望 {result.expected_level}）"
                if result.expected_level and result.expected_level != result.level
                else ""
            )
            console.print(f"{head} [bold]{level}[/bold] {escape(result.text)}")
            console.print(f"    [cyan]{escape(result.reply.text)}[/cyan]")
            console.print(f"    [dim]{_trace_line(result.reply)}[/dim]")
            for failure in result.failures:
                console.print(f"    [red]{failure}[/red]")

        results = replay(
            service,
            plan,
            root=script.parent,
            space=space_id,
            until=until,
            router=cast(Router, router),
            lam=lam,
            on_turn=show,
        )
        target = report or settings.paths.eval_runs.parent / "demo" / (
            f"{space_id}-{datetime.now():%Y%m%d-%H%M%S}.json"
        )
        save_report(results, target, script=str(script), space=space_id, router=router)
        passed = sum(r.passed for r in results)
        console.print(f"通过 {passed} / {len(results)}；报告：{target}")
        _print_usage_rows(service.system.ledger.summary(run_id=service.system.ledger.run_id))
    finally:
        service.close()


@app.command("migrate-vectors")
def migrate_vectors(
    to: Annotated[str, typer.Option("--to", help="Qdrant 服务端地址，如 http://localhost:6333")],
) -> None:
    """把本地模式的向量库原样复制到 Qdrant 服务端（改用服务端之前运行一次）。

    数据库里已有的记忆不会被重新写入向量库，所以直接切换会得到一个空的向量库。服务端要 API Key
    时先在环境变量或 .env 里设 CUEMEM_VECTOR_STORE__API_KEY（不做成参数：会留在 shell 历史与进程
    列表里）。
    """
    from cuemem.store.vector import VectorStore

    settings = _settings()
    if settings.vector_store.url:
        console.print(
            "[red]当前已经在用服务端[/red]（CUEMEM_VECTOR_STORE__URL）：这个命令只从本地模式迁出"
        )
        raise typer.Exit(1)
    try:
        source = VectorStore(settings.paths.qdrant, dim=settings.embedding.dim)
    except CueMemError as exc:
        console.print(f"[red]错误：[/red]{exc}")
        raise typer.Exit(1) from exc
    api_key = settings.vector_store.secret()
    target = VectorStore(None, dim=settings.embedding.dim, url=to, api_key=api_key)
    try:
        counts = source.copy_into(target)
    except Exception as exc:  # 命令行的边界：连不上服务端、版本不兼容……都给一句能看懂的话
        console.print(f"[red]复制失败[/red]（{to}）：{exc}")
        raise typer.Exit(1) from exc
    finally:
        source.close()
        target.close()
    if not counts:
        console.print("本地向量库里没有集合，不需要迁移")
        return
    table = Table(title=f"已复制到 {to}")
    table.add_column("集合")
    table.add_column("点数", justify="right")
    for name, count in counts.items():
        table.add_row(name, str(count))
    console.print(table)
    console.print(
        f"下一步：在 .env 里加上 [bold]CUEMEM_VECTOR_STORE__URL={to}[/bold]"
        + (
            "，CUEMEM_VECTOR_STORE__API_KEY 也写进 .env"
            if api_key
            else "（这个服务端没要 API Key；给 Qdrant 设了 QDRANT__SERVICE__API_KEY 之后，"
            "同一个 Key 写成 CUEMEM_VECTOR_STORE__API_KEY）"
        )
    )


@app.command("mcp")
def mcp_cmd(
    space: Annotated[str, typer.Option(help="记忆空间 ID")] = "default",
    router: Annotated[str, typer.Option(help="recall 用的检索控制器")] = "learned",
) -> None:
    """以 MCP Server（stdio）提供记忆工具，让其他 Agent（如 Claude Code）直接使用这套记忆。

    独占数据目录：开着 cuemem serve 时改用它的 /mcp（同样的工具，HTTP）。"""
    _configure_logging()  # 在导入 MCP、建 MCPServer 之前；日志写到 stderr
    from cuemem.api.mcp import build_server
    from cuemem.api.service import ChatService

    settings = _settings(stderr=True, peak_policy="warn")
    try:  # 缺 Key、向量库打不开等：说明原因，不打印调用栈；写到 stderr（stdout 是 MCP 的通道）
        service = ChatService.create(
            settings, scenario="plain", default_space=space, router=cast(Router, router)
        )
    except CueMemError as exc:
        Console(stderr=True).print(f"[red]错误：[/red]{escape(str(exc))}")
        raise typer.Exit(1) from exc
    try:
        build_server(service, space).run("stdio")
    finally:
        service.close()


def _trace_line(reply: Reply) -> str:
    """一轮对话的轨迹摘要：档位、召回与引用、工具、用量与耗时。"""
    t = reply.trace
    usage = t.usage
    parts = [
        f"{t.level}{'+see' if t.see else ''}（{t.router or '指定'}）",
        f"召回 {len(t.recalled)} 条",
        f"引用 {', '.join(reply.citations) or '无'}",
        f"工具 {', '.join(r.name for r in t.tools) or '无'}",
        f"输入 {int(usage.get('input_tokens', 0)):,}"
        f"（前缀缓存 {int(usage.get('cached_input_tokens', 0)):,}）",
        f"输出 {int(usage.get('output_tokens', 0)):,}",
        f"${usage.get('spent', 0.0):.4f}",
        f"{t.timings.get('total', 0) / 1000:.1f} s",
    ]
    return " · ".join(parts)


# ---------- 记忆检查器 ----------

inspect_app = typer.Typer(help="记忆检查器：事实时间线、实体、来源、核心记忆", no_args_is_help=True)
app.add_typer(inspect_app, name="inspect")


@inspect_app.command("facts")
def inspect_facts(
    space: Annotated[str, typer.Option(help="记忆空间 ID")],
    subject: Annotated[str | None, typer.Option(help="只看某个主语（实体名）")] = None,
    current: Annotated[bool, typer.Option("--current", help="只看当前有效的事实")] = False,
    valid_at: Annotated[
        str | None,
        typer.Option(
            "--valid-at",
            help="事件时间：那时现实中成立的事实，写成 2025、2025-06 或 2025-06-07；"
            "一年、一个月指这段时间里任何时候成立过的",
        ),
    ] = None,
    known_at: Annotated[
        datetime | None,
        typer.Option("--known-at", formats=DATETIME_FORMATS, help="记录时间：系统那时相信的事实"),
    ] = None,
    hide_expired: Annotated[
        bool, typer.Option("--hide-expired", help="不列出已经过期的事实（expiration_date）")
    ] = False,
) -> None:
    """事实时间线：按主语分组、按时间排列，标出被取代的版本与已经过期的事实（检查用，默认
    都列出；--hide-expired 时不列出过期的，同检索与记忆浏览器）。"""
    if valid_at is not None:
        try:
            valid_period(valid_at)
        except ValueError:
            raise typer.BadParameter(
                f"{valid_at!r} 不是 ISO 日期：写成 YYYY、YYYY-MM 或 YYYY-MM-DD",
                param_hint="--valid-at",
            ) from None
    from cuemem.expiry import expired_ids

    with _open_db_only() as db:  # 只读数据库：cuemem serve 开着它的时候也能用（见 _open_db_only）
        facts = db.get_facts(
            space,
            current_only=current,
            valid_period=valid_period(valid_at) if valid_at is not None else None,
            known_at=naive_local(known_at) if known_at is not None else None,
        )
        if subject:
            facts = [f for f in facts if f.subject.casefold() == subject.casefold()]
        gone = expired_ids(db, space, datetime.now())
        if hide_expired:
            facts = [f for f in facts if f.id not in gone]
        if not facts:
            console.print("没有符合条件的事实。还没巩固的话，先运行 cuemem consolidate。")
            return
        ids = {f.id: i for i, f in enumerate(facts, start=1)}
        table = Table(title=f"事实时间线（{len(facts)} 条）")
        for column in ("#", "主语", "谓词", "陈述", "事件时间", "说出时间", "状态", "ID"):
            table.add_column(column, overflow="fold")
        for fact in sorted(facts, key=lambda f: (f.subject, f.predicate, f.recorded_at)):
            table.add_row(
                str(ids[fact.id]),
                fact.subject,
                fact.predicate,
                fact.statement,
                _span_text(fact.valid_from, fact.valid_to),
                f"{fact.recorded_at:%Y-%m-%d}",
                _fact_status(fact, ids, expired=fact.id in gone),
                fact.id,
            )
        console.print(table)


@inspect_app.command("entities")
def inspect_entities(space: Annotated[str, typer.Option(help="记忆空间 ID")]) -> None:
    """实体列表：名字、类型、别名、提及次数。"""
    with _open_db_only() as db:
        entities = db.get_entities(space)
        if not entities:
            console.print("没有实体。还没巩固的话，先运行 cuemem consolidate。")
            return
        table = Table(title=f"实体（{len(entities)} 个）")
        for column in ("名字", "类型", "别名", "描述", "提及", "首次 / 最近"):
            table.add_column(column, overflow="fold")
        for e in sorted(entities, key=lambda e: (-e.mentions, e.name)):
            table.add_row(
                e.name,
                e.type,
                ", ".join(e.aliases) or "-",
                e.description or "-",
                str(e.mentions),
                f"{e.first_seen:%Y-%m-%d} / {e.last_seen:%Y-%m-%d}",
            )
        console.print(table)


@inspect_app.command("sources")
def inspect_sources(
    fact_id: Annotated[str, typer.Argument(help="事实 ID（见 inspect facts 的最后一列）")],
    space: Annotated[str, typer.Option(help="记忆空间 ID")],
) -> None:
    """溯源：一条事实来自哪些原始轮次，以及它的版本链。"""
    with _open_db_only() as db:
        found = db.get_facts(space, ids=[fact_id])
        if not found:
            console.print(f"[red]找不到事实 {fact_id}[/red]")
            raise typer.Exit(1)
        fact = found[0]
        console.print(Panel(fact.statement, title=f"事实 {fact.id}", expand=False))
        console.print(
            f"主语 {fact.subject} · 谓词 {fact.predicate} · 宾语 {fact.object} · "
            f"重要度 {fact.importance} · 事件时间 {_span_text(fact.valid_from, fact.valid_to)} · "
            f"记录于 {fact.recorded_at:%Y-%m-%d %H:%M}"
        )
        chain = db.get_facts(space, superseded_by=[fact.id])
        if fact.superseded_by:
            chain += db.get_facts(space, ids=[fact.superseded_by])
        for other in chain:
            relation = "取代了" if other.superseded_by == fact.id else "被它取代"
            console.print(f"[dim]{relation}：{other.statement}（{other.id}）[/dim]")
        turns = db.get_turns(space, ids=fact.source_turn_ids)
        for t in turns:
            who = t.speaker or ("用户" if t.role == "user" else "助手")
            console.print(f"[bold]{t.ts:%Y-%m-%d} · {t.session_id} · {who}[/bold]：{t.text}")


@inspect_app.command("profile")
def inspect_profile(space: Annotated[str, typer.Option(help="记忆空间 ID")]) -> None:
    """核心记忆（用户画像）：重要且当前有效的事实，每轮常驻上下文。"""
    with _open_db_only() as db:
        core = db.get_core_memory(space)
        if core is None or not core.text:
            console.print("还没有核心记忆。先运行 cuemem consolidate。")
            return
        title = f"核心记忆（{len(core.fact_ids)} 条事实，{len(core.text)} 字符）"
        console.print(Panel(core.text, title=title, expand=False))


# ---------- 评测 ----------

data_app = typer.Typer(help="评测数据集", no_args_is_help=True)
eval_app = typer.Typer(help="评测：划分、运行、报告", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")


@data_app.command("download")
def data_download(
    names: Annotated[
        list[str] | None, typer.Argument(help="数据集名，默认全部（可选的只在点名时下载）")
    ] = None,
) -> None:
    """下载评测数据集到 data/datasets（已存在则跳过）。下载地址固定在上游的提交上，下载后核对
    sha256，对不上就不保存、报错。"""
    from cuemem.evals.datasets import DATASETS
    from cuemem.evals.datasets import download as download_dataset

    settings = _settings()
    for name in names or [n for n, spec in DATASETS.items() if not spec.optional]:
        if name not in DATASETS:
            console.print(f"[red]未知数据集 {name}[/red]，可选：{', '.join(DATASETS)}")
            raise typer.Exit(1)
        console.print(f"{name}：{DATASETS[name].description}")
        try:
            path = download_dataset(name, settings.paths.datasets)
        except DatasetChecksumError as exc:
            console.print(f"[red]错误：[/red]{escape(str(exc))}")
            raise typer.Exit(1) from exc
        console.print(f"  → {path}（{path.stat().st_size / 1e6:.1f} MB）")


@eval_app.command("split")
def eval_split(
    dataset: Annotated[str, typer.Option(help="数据集")] = "longmemeval_s",
    name: Annotated[str, typer.Option(help="划分名")] = "dev100",
    size: Annotated[int, typer.Option(help="抽取的题数")] = 100,
    seed: Annotated[int, typer.Option(help="随机种子")] = 0,
    parent: Annotated[
        str | None, typer.Option(help="从已有划分中再抽（如从 dev100 抽 pilot10）")
    ] = None,
    rest: Annotated[
        str | None, typer.Option(help="把剩余的题另存为这个划分名（如 test400）")
    ] = None,
) -> None:
    """按（题型，是否拒答）分层抽样，生成划分文件（提交进仓库，保证可复现）。"""
    from cuemem.evals.splits import save_split, stratified_ids

    settings = _settings()
    evalset = _load_evalset(settings, dataset, parent)
    chosen = stratified_ids(evalset.samples, size, seed=seed)
    note = f"从 {parent} 分层抽样" if parent else "按（题型，是否拒答）分层抽样"
    path = save_split(dataset, name, chosen, seed=seed, note=note)
    console.print(f"已保存 {path}（{len(chosen)} 题）")
    if rest:
        remaining = sorted({s.sample_id for s in evalset.samples} - set(chosen))
        rest_path = save_split(dataset, rest, remaining, seed=seed, note=f"{name} 之外的其余题目")
        console.print(f"已保存 {rest_path}（{len(remaining)} 题）")
    picked = set(chosen)
    counts: dict[str, list[int]] = {}
    for s in evalset.samples:
        total, selected = counts.setdefault(s.question_type, [0, 0])
        counts[s.question_type] = [total + 1, selected + (s.sample_id in picked)]
    table = Table(title="各题型题数（全部 / 抽中）")
    for column in ("题型", "全部", "抽中"):
        table.add_column(column)
    for qtype, (total, selected) in sorted(counts.items()):
        table.add_row(qtype, str(total), str(selected))
    console.print(table)


# eval run --fail-on-error 的退出码（与配置错误的 1、参数错误的 2 区分开）：有组合出错时是 3，
# 达到花费上限提前停止时是 4（复现脚本据此区分发表时就有的出错与没跑完）
ERRORS_EXIT = 3
BUDGET_EXIT = 4


@eval_app.command("run")
def eval_run(
    dataset: Annotated[str, typer.Option(help="数据集")] = "longmemeval_s",
    split: Annotated[str, typer.Option(help="划分名；all 表示全部题目")] = "dev100",
    methods: Annotated[str, typer.Option(help="逗号分隔：B0,B1,B2,ORACLE")] = "B0,B1,B2,ORACLE",
    k: Annotated[int, typer.Option(help="B2 的检索条数")] = 10,
    workers: Annotated[int, typer.Option(help="并发线程数")] = 4,
    max_cost: Annotated[
        float, typer.Option(help="实际花费上限（美元），超过后不再开始新题")
    ] = 10.0,
    limit: Annotated[int | None, typer.Option(help="只跑前 N 题（调试用）")] = None,
    types: Annotated[
        str | None, typer.Option(help="只跑这些题型（逗号分隔）；LoCoMo 默认排除对抗题")
    ] = None,
    name: Annotated[str | None, typer.Option(help="运行名；同名运行会断点续跑")] = None,
    peak: Annotated[
        str | None,
        typer.Option(help="高峰时段策略：wait 暂停到高峰结束（默认）/ warn / allow"),
    ] = None,
    build_only: Annotated[
        bool,
        typer.Option(
            "--build-only",
            help="只构建记忆空间（写入与巩固），不回答、不判分；"
            "构建只用通义的模型，可以在 DeepSeek 高峰时段做",
        ),
    ] = False,
    rebuild_spaces: Annotated[
        bool,
        typer.Option(
            "--rebuild-spaces",
            help="磁盘上的记忆空间是按另一套巩固配置（或旧版流水线）建的时，按这次的配置重建；"
            "默认报错、不动它",
        ),
    ] = False,
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error",
            help=f"照常生成报告，但有组合出错时以退出码 {ERRORS_EXIT}、达到花费上限提前停止时以 "
            f"{BUDGET_EXIT} 结束（复现脚本据此不复制不完整的报告）；默认这两种情况退出码都是 0",
        ),
    ] = False,
) -> None:
    """运行评测：写入 → 回答 → 判分；结果写入 runs/eval/<运行名>/，最后生成报告。"""
    if peak is not None and peak not in ("wait", "warn", "allow"):
        console.print("[red]--peak 只能是 wait、warn 或 allow[/red]")
        raise typer.Exit(1)
    if not build_only:  # 最后的报告要画图：先说明要装什么，不白跑一遍
        _require_plotting()
    from cuemem.evals.datasets import protocol_samples
    from cuemem.evals.report import build_report
    from cuemem.evals.runner import EvalRunner, RunConfig

    settings = _settings()
    if peak is not None:
        settings = settings.model_copy(update={"peak_policy": peak})
    for line in _peak_status(settings):
        console.print(line)
    split_name = None if split == "all" else split
    evalset = _load_evalset(settings, dataset, split_name)
    # 给了 --types 就只评这些题型；没给时 LoCoMo 默认不评对抗题（与文献一致）
    wanted = {t.strip() for t in types.split(",")} if types else None
    evalset.samples = protocol_samples(evalset, wanted)
    run_name = name or f"{dataset}-{split}-{datetime.now():%Y%m%d-%H%M}"
    config = RunConfig(
        name=run_name,
        dataset=dataset,
        split=split_name,
        methods=[m.strip() for m in methods.split(",") if m.strip()],
        k=k,
        workers=workers,
        max_cost=max_cost,
        limit=limit,
        build_only=build_only,
        rebuild_spaces=rebuild_spaces,
    )
    count = min(limit, len(evalset.samples)) if limit else len(evalset.samples)
    console.print(f"运行 [bold]{run_name}[/bold]：{count} 题 × {config.methods}")
    with _progress() as progress:
        task = progress.add_task("评测", total=None)

        def on_peak_wait(provider: str, until: datetime) -> None:
            label = PROVIDER_LABELS.get(provider, provider)
            progress.console.print(
                f"[yellow]{label} 进入高峰时段：需要真实调用的请求暂停到 {until:%H:%M}，"
                "命中缓存的继续执行（--peak allow 可关闭）[/yellow]"
            )

        runner = EvalRunner(
            settings,
            config,
            on_start=lambda total: progress.update(task, total=total),
            on_advance=lambda: progress.advance(task),
            on_peak_wait=on_peak_wait,
        )
        try:
            summary = runner.run(evalset)
        except ConfigMismatchError as exc:
            progress.stop()
            console.print(f"[red]错误：[/red]{exc}")
            raise typer.Exit(1) from exc
    if build_only:
        console.print(
            f"构建完成：{summary.completed} / {summary.planned} 份对话历史，"
            f"出错 {summary.errors} 份；耗时 {summary.elapsed_s:.0f} 秒；"
            f"本次实际花费 ${summary.spent:.4f}"
        )
        if summary.errors:
            _print_errors(summary.run_dir)
        _exit_if_partial(summary, fail_on_error)
        return
    report_path = build_report(summary.run_dir, eval_spaces=settings.paths.eval_spaces)
    note = "（已达到花费上限，提前停止）" if summary.stopped_by_budget else ""
    console.print(
        f"完成 {summary.completed} / {summary.planned} 个组合，出错 {summary.errors} 个{note}；"
        f"耗时 {summary.elapsed_s:.0f} 秒；本次实际花费 ${summary.spent:.4f}"
        f"（名义成本 ${summary.nominal_cost:.4f}）"
    )
    if summary.errors:
        _print_errors(summary.run_dir)
    _print_method_stats(summary.run_dir)
    console.print(f"报告：{report_path}")
    _exit_if_partial(summary, fail_on_error)


def _exit_if_partial(summary: RunSummary, fail_on_error: bool) -> None:
    """--fail-on-error：这次运行不完整时以非零退出码结束（报告照常生成）。"""
    if not fail_on_error or not (summary.errors or summary.stopped_by_budget):
        return
    code = BUDGET_EXIT if summary.stopped_by_budget else ERRORS_EXIT
    reason = "达到花费上限提前停止" if summary.stopped_by_budget else "有组合出错"
    console.print(f"[red]这次运行不完整（{reason}），按 --fail-on-error 以退出码 {code} 结束[/red]")
    raise typer.Exit(code)


@eval_app.command("probe")
def eval_probe(
    dataset: Annotated[str, typer.Option(help="数据集")] = "longmemeval_s",
    split: Annotated[str, typer.Option(help="划分名；all 表示全部题目")] = "dev100",
    name: Annotated[str, typer.Option(help="运行名：特征写入 runs/eval/<运行名>/probe.jsonl")] = "",
    limit: Annotated[int | None, typer.Option(help="只跑前 N 题")] = None,
    workers: Annotated[int, typer.Option(help="并发线程数")] = 8,
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error",
            help=f"有题出错（空间没建好等）时以退出码 {ERRORS_EXIT} 结束（复现脚本据此停下："
            "控制器的分析会悄悄略过没有探针特征的题）；默认退出码是 0",
        ),
    ] = False,
) -> None:
    """跑记忆探针（一次检索，不调用 LLM）：每道题的检索控制器特征（N7）。空间要先构建好。"""
    if not name:
        console.print("[red]请用 --name 指定运行名（与评测运行同名，特征与结果放在一起）[/red]")
        raise typer.Exit(1)
    from cuemem.evals.datasets import protocol_samples
    from cuemem.evals.runner import EvalRunner, RunConfig

    settings = _settings()
    split_name = None if split == "all" else split
    evalset = _load_evalset(settings, dataset, split_name)
    evalset.samples = protocol_samples(evalset)  # 与评测一致：LoCoMo 不含对抗题
    config = RunConfig(
        name=name, dataset=dataset, split=split_name, methods=[], workers=workers, limit=limit
    )
    with _progress() as progress:
        task = progress.add_task("探针", total=None)
        runner = EvalRunner(
            settings,
            config,
            on_start=lambda total: progress.update(task, total=total),
            on_advance=lambda: progress.advance(task),
        )
        done, errors = runner.probe(evalset)
    console.print(f"完成 {done} 题，出错 {errors} 题：{runner.run_dir / 'probe.jsonl'}")
    if fail_on_error and errors:
        console.print(
            f"[red]有题出错（报错见 {runner.run_dir / 'errors.jsonl'}），"
            f"按 --fail-on-error 以退出码 {ERRORS_EXIT} 结束[/red]"
        )
        raise typer.Exit(ERRORS_EXIT)


@eval_app.command("report")
def eval_report(
    run: Annotated[str, typer.Argument(help="运行名或运行目录")],
) -> None:
    """重新生成某次运行的报告。"""
    _require_plotting()
    from cuemem.evals.report import build_report

    settings = _settings()
    run_dir = Path(run) if Path(run).exists() else settings.paths.eval_runs / run
    if not (run_dir / "run.json").exists():
        console.print(f"[red]找不到运行目录 {run_dir}[/red]")
        raise typer.Exit(1)
    path = build_report(run_dir, eval_spaces=settings.paths.eval_spaces)
    _print_method_stats(run_dir)
    console.print(f"报告：{path}")


@eval_app.command("audit")
def eval_audit(
    run: Annotated[str, typer.Argument(help="运行名或运行目录")],
    n: Annotated[int, typer.Option(help="抽检条数（判对、判错各一半）")] = 50,
    seed: Annotated[int, typer.Option(help="随机种子")] = 0,
    score: Annotated[bool, typer.Option("--score", help="统计已勾选的抽检结果")] = False,
) -> None:
    """判分抽检：导出待核对的 Markdown，或统计人工核对后的一致率。"""
    from cuemem.evals.audit import export_audit, score_audit
    from cuemem.evals.report import load_rows

    settings = _settings()
    run_dir = Path(run) if Path(run).exists() else settings.paths.eval_runs / run
    path = run_dir / "judge-audit.md"
    if score:
        if not path.exists():
            console.print(f"[red]找不到 {path}[/red]，请先不带 --score 运行一次导出")
            raise typer.Exit(1)
        agree, disagree, pending = score_audit(path)
        checked = agree + disagree
        rate = f"{agree / checked:.0%}" if checked else "—"
        console.print(f"同意 {agree}，不同意 {disagree}，未勾选 {pending}；一致率 {rate}")
        return
    if path.exists():
        console.print(f"[yellow]{path} 已存在[/yellow]，为避免覆盖你的勾选，请先改名或删除")
        raise typer.Exit(1)
    count = export_audit(load_rows(run_dir, "results.jsonl"), path, n=n, seed=seed)
    console.print(f"已导出 {count} 条到 {path}")


def _require_plotting() -> None:
    """评测报告要画图：没装可选依赖 eval（matplotlib）时说明要装什么，退出码 1。"""
    if importlib.util.find_spec("matplotlib") is None:
        console.print(
            "[red]错误：[/red]评测报告要画图，需要可选依赖 eval（matplotlib、scikit-learn），"
            f"没有安装：{escape(extra_remedy('eval'))}"
        )
        raise typer.Exit(1)


@contextmanager
def _progress() -> Iterator[Any]:
    """评测与探针的进度条。"""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    with Progress(*columns, TimeElapsedColumn(), console=console) as progress:
        yield progress


def _load_evalset(settings: Settings, dataset: str, split: str | None) -> EvalSet:
    from cuemem.evals.datasets import (
        DATASETS,
        RAW_LABELS_VALUE,
        RAW_SESSION_IDS_ENV,
        dataset_path,
    )
    from cuemem.evals.datasets import load as load_dataset
    from cuemem.evals.splits import load_split

    if dataset not in DATASETS:
        console.print(f"[red]未知数据集 {dataset}[/red]，可选：{', '.join(DATASETS)}")
        raise typer.Exit(1)
    path = dataset_path(settings.paths.datasets, dataset)
    if not path.exists():
        console.print(
            f"[red]缺少数据文件 {path}[/red]，请先运行：{cli_command('data download ' + dataset)}"
        )
        raise typer.Exit(1)
    ids = load_split(dataset, split) if split else None
    evalset = load_dataset(dataset, path, ids=ids)
    if evalset.session_ids == "raw":
        console.print(
            f"[yellow]{RAW_SESSION_IDS_ENV}=1：保留数据集的原始会话 ID（LongMemEval 的证据会话以 "
            "answer_ 开头，拒答题的还带 _abs，会泄漏给回答模型），只用于复现这个泄漏的消融[/yellow]"
        )
    elif evalset.raw_labels:
        console.print(
            f"[yellow]{RAW_SESSION_IDS_ENV}={RAW_LABELS_VALUE}：会话 ID 照常改写，但证据抬头显示"
            "原始 ID（泄漏的干净消融，只用于测量泄漏的影响）[/yellow]"
        )
    return evalset


def _print_errors(run_dir: Path, limit: int = 3) -> None:
    """出错的组合按报错归并后列出（errors.jsonl 里每道题各记一次）：配置问题（如外部基线的写入
    目录里是另一个协议的写入）一眼看得到报错与解决办法。"""
    from cuemem.evals.report import error_groups, pending_errors

    groups = error_groups(pending_errors(run_dir))
    for method, error, n in groups[:limit]:
        console.print(f"[red]{escape(method)} × {n}[/red]：{escape(error)}", highlight=False)
    if len(groups) > limit:
        console.print(f"另有 {len(groups) - limit} 种报错，详见 {run_dir / 'errors.jsonl'}")


def _print_method_stats(run_dir: Path) -> None:
    from cuemem.evals.report import load_rows, method_stats

    stats = method_stats(load_rows(run_dir, "results.jsonl"))
    table = Table(title="总体结果")
    for column in ("方法", "题数", "准确率", "95% 置信区间", "平均输入 tokens", "名义成本 / 题"):
        table.add_column(column)
    for s in stats:
        table.add_row(
            s.method,
            str(s.n),
            f"{s.accuracy:.1%}",
            f"{s.ci[0]:.1%} – {s.ci[1]:.1%}",
            f"{s.input_tokens:,.0f}",
            f"${s.cost:.5f}",
        )
    console.print(table)


# ---------- 交互辅助 ----------


def _prompt_secret(label: str, existing: str | None) -> str | None:
    hint = f"（当前 {mask(existing)}，回车保留）" if existing else ""
    value = typer.prompt(f"{label}{hint}", default="", show_default=False, hide_input=True)
    return value.strip() or existing


# ---------- 输出辅助 ----------


def _peak_note(settings: Settings) -> None:
    """当前处于某个供应商的高峰时段时提醒一句（交互式命令照常调用）。"""
    from cuemem.llm.peak import peak_end

    for name, provider in settings.providers.items():
        cfg = provider.peak_hours
        if cfg is not None and (end := peak_end(cfg)) is not None:
            console.print(
                f"[dim]提示：现在是 {PROVIDER_LABELS.get(name, name)} 的高峰时段"
                f"（价格为平时的 {1 / cfg.off_peak_factor:g} 倍，到 {end:%H:%M} 结束）[/dim]"
            )


def _peak_status(settings: Settings) -> list[str]:
    from cuemem.llm.peak import describe, next_peak_start, peak_end

    lines = []
    for name, provider in settings.providers.items():
        cfg = provider.peak_hours
        if cfg is None:
            continue
        end = peak_end(cfg)
        state = (
            f"[yellow]高峰中，到 {end:%H:%M} 结束[/yellow]"
            if end
            else f"[green]非高峰[/green]，下一段高峰 {next_peak_start(cfg):%m-%d %H:%M}"
        )
        lines.append(
            f"{PROVIDER_LABELS.get(name, name)} 高峰时段：{describe(cfg)}，"
            f"价格为平时的 {1 / cfg.off_peak_factor:g} 倍；当前{state}"
        )
    return lines


def _print_config(settings: Settings) -> None:
    providers = Table(title="供应商")
    for column in ("名称", "Base URL", "Key 环境变量", "状态"):
        providers.add_column(column)
    used = settings.used_providers()
    for name, p in settings.providers.items():
        if p.api_key_env is None:
            status = "[green]不需要 Key[/green]"
        else:
            status = "[green]已设置[/green]" if p.api_key() else "[red]未设置 Key[/red]"
        if name not in used:
            status += "（未使用）"
        providers.add_row(name, p.base_url, p.api_key_env or "—", status)
    console.print(providers)

    roles = Table(title="角色 → 模型")
    for column in ("角色", "模型别名", "供应商", "模型 ID", "标价（输入 / 输出，每百万）"):
        roles.add_column(column)
    for role, rc in settings.roles.items():
        spec = settings.models[rc.model]
        symbol = CURRENCY_SYMBOL.get(spec.currency, spec.currency)
        price = f"{symbol}{spec.price.input:g} / {symbol}{spec.price.output:g}"
        roles.add_row(role, rc.model, spec.provider, spec.model_id, price)
    console.print(roles)

    emb = settings.embedding
    detail = (
        f"离线哈希向量，{emb.dim} 维（仅用于测试与演示）"
        if emb.backend == "hash"
        else f"{emb.model}（{emb.provider}，{emb.backend} 接口），{emb.dim} 维，"
        f"sparse={'开' if emb.sparse and emb.backend == 'dashscope' else '关'}"
    )
    console.print(f"向量：{detail}")
    vector_store = f"本地模式 {settings.paths.qdrant}"
    if settings.vector_store.url:
        has_key = settings.vector_store.api_key is not None
        vector_store = f"{settings.vector_store.url}（API Key {'已设' if has_key else '未设'}）"
    console.print(f"数据库：{settings.paths.db}")
    console.print(f"向量库：{vector_store}")
    for line in _peak_status(settings):
        console.print(line)
    console.print(f"批量任务（评测）在高峰时段的策略：{settings.peak_policy}")


def _print_local_store(settings: Settings) -> None:
    """本地模式的向量库有几个集合，与打开文件数的上限（main 已经调到硬上限）比一比：每个集合
    一直开着一个文件、启动时全部打开。只读 meta.json、不打开向量库，cuemem serve 正在用它时也能
    查。集合超过软上限的一半时标黄，建议改用 Qdrant 服务端；多到再建一个就占到留给别的文件的
    余量（store.vector.reserved_files，VectorStore._room_for 拒绝建新空间的同一条线）时标红。"""
    meta = settings.paths.qdrant / "meta.json"
    if settings.vector_store.url:
        return
    repair = (
        f"运行 {cli_command('doctor --repair-vectors')} 按 collection/ 下的集合重写它（不要手动"
        f"删掉 meta.json）；{cli_command('export')} 不用向量库，照样能导出"
    )
    if not meta.is_file():
        root = settings.paths.qdrant / "collection"
        if root.is_dir() and any(p.is_dir() for p in root.iterdir()):
            console.print(f"[yellow]{escape(str(meta))} 不在了，集合还在[/yellow]：{repair}")
        return
    try:
        collections = len(json.loads(meta.read_text("utf-8"))["collections"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        console.print(
            f"[yellow]读不出 {escape(str(meta))}[/yellow]：{escape(str(exc))}。向量还在：{repair}"
        )
        return
    line = f"本地向量库：{collections} 个集合（每个一直开着一个文件，启动时全部打开）"
    limits = _raise_open_file_limit()
    if limits is None:
        console.print(line)
        return
    import resource

    soft = limits[0]
    shown = ["不限" if v == resource.RLIM_INFINITY else str(v) for v in limits]
    line += f"，打开文件数上限 {shown[0]}（硬上限 {shown[1]}）"
    if soft == resource.RLIM_INFINITY or collections <= soft / 2:
        console.print(line)
        return
    from cuemem.store.vector import reserved_files

    if collections + 1 > soft - reserved_files(soft):  # 与 VectorStore._room_for 同一条线
        console.print(
            f"[red]{line}：已经不能再建新的记忆空间（写入新空间的请求 503 capacity，已有的"
            f"照常可用；打开文件数要给数据库、图片与连接留 {reserved_files(soft)} 个）。删掉"
            "不用的空间、调高 ulimit -n，或者改用 Qdrant 服务端："
            f"{cli_command('migrate-vectors --to URL')}，再设置 CUEMEM_VECTOR_STORE__URL[/red]"
        )
        return
    console.print(
        f"[yellow]{line}：集合再多就打不开、建不了新的空间。用户多时改用 Qdrant 服务端："
        f"{cli_command('migrate-vectors --to URL')}，再设置 CUEMEM_VECTOR_STORE__URL[/yellow]"
    )


def _repair_vectors(settings: Settings) -> None:
    """doctor --repair-vectors：见 store/vector.py 的 repair_local_meta。"""
    from cuemem.store.vector import repair_local_meta

    if settings.vector_store.url:
        console.print("用的是 Qdrant 服务端（CUEMEM_VECTOR_STORE__URL），没有本地的 meta.json 要修")
        return
    meta = settings.paths.qdrant / "meta.json"
    try:
        report = repair_local_meta(
            settings.paths.qdrant,
            text_dim=settings.embedding.dim,
            image_dim=settings.multimodal_embedding.dim,
        )
    except CueMemError as exc:
        console.print(f"[red]错误：[/red]{exc}")
        raise typer.Exit(1) from exc
    if report is None:
        console.print(f"{escape(str(meta))} 读得出、集合都在，不用修")
        return
    backup = f"，原来的文件改名为 {escape(report['backup'])}" if report["backup"] else ""
    console.print(
        f"[green]已重写 {escape(str(meta))}[/green]：补上 {report['text']} 个文字集合、"
        f"{report['image']} 个图片集合（其中 {report['empty']} 个是空的），共 {report['points']}"
        f" 个点{backup}"
    )


def _print_vector_gaps(settings: Settings) -> None:
    """doctor --vectors：每个空间按种类比较数据库里的记忆单元与向量库里的点（图片另比），对不上
    的标黄、写明 cuemem reindex 的命令。向量库丢了、坏了，或者与数据库不是同一时刻的备份时，
    检索悄悄少了结果（get_all 照样列得出），以前 doctor 看不出来。

    要打开向量库：本地模式同一时间只允许一个进程打开，cuemem serve 正开着它时跳过（doctor 的
    其余检查只读 meta.json，照样能做）。"""
    from cuemem.embed import image_vectors_enabled
    from cuemem.errors import StorageLockedError
    from cuemem.memory import vector_gaps
    from cuemem.store.vector import VectorStore

    if not settings.paths.db.is_file():
        console.print("向量库检查：还没有数据库，没有要比的")
        return
    with _open_db_only() as db:
        try:
            store = VectorStore(
                settings.paths.qdrant,
                dim=settings.embedding.dim,
                url=settings.vector_store.url,
                api_key=settings.vector_store.secret(),
            )
        except StorageLockedError:
            console.print(
                "[yellow]向量库检查跳过：本地向量库正被另一个进程使用（多半是 cuemem serve）"
                "[/yellow]：先停掉它再查，或者改用 Qdrant 服务端"
            )
            return
        except CueMemError as exc:
            console.print(f"[yellow]向量库检查跳过：[/yellow]{exc}")
            return
        store.owned_units = db.some_unit_ids  # 与 MemorySystem 相同（见 collection_name）
        images = image_vectors_enabled(settings)
        spaces = [s["space_id"] for s in db.list_spaces()]
        try:
            gaps = {space: vector_gaps(db, store, space, images=images) for space in spaces}
        except Exception as exc:  # 服务端连不上等：一行说明，doctor 的其余检查照常
            console.print(f"[yellow]向量库检查没做完：[/yellow]{escape(str(exc))}")
            return
        finally:
            store.close()
    bad = {space: found for space, found in gaps.items() if found}
    if not bad:
        console.print(f"向量库：{len(spaces)} 个空间的点数与数据库一致")
        return
    for space, found in bad.items():
        detail = "、".join(f"{kind} {want} → {got}" for kind, (want, got) in found.items())
        command = escape(cli_command(f"reindex --space {shlex.quote(space)}"))
        console.print(
            f"[yellow]空间 {escape(space)} 的向量库与数据库对不上[/yellow]"
            f"（数据库 → 向量库：{detail}）：运行 {command}"
        )
    console.print(f"全部补齐：{cli_command('reindex --all')}（不调用 LLM，命中向量缓存时不发请求）")


def _print_disk(settings: Settings) -> None:
    """数据目录写不写得进去、还剩多少空间：只读、满了时打开数据库与写入都会失败。"""
    data = settings.paths.db.parent
    probe = data
    while not probe.exists() and probe != probe.parent:  # 还没建：看会建在哪里
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    line = f"数据目录 {escape(str(data))}：剩余空间 {free / 2**20:,.0f} MB"
    if not os.access(probe, os.W_OK):
        console.print(f"[yellow]{line}，但没有写权限[/yellow]：写入、巩固与导入都会失败")
    elif free < LOW_DISK:
        console.print(f"[yellow]{line}[/yellow]：空间不够时写入、巩固与导入会失败")
    else:
        console.print(line)


def _print_media_journal(settings: Settings) -> None:
    """进程被杀掉时没有提交的写入、没做完的遗忘留在媒体库里的原图（第七轮体检，见
    store/media.py）：有的话提示 cuemem reindex 删掉。只读标记，不打开数据库。"""
    from cuemem.store.media import MediaStore

    media = MediaStore(settings.paths.media, owner=str(settings.paths.db.absolute()))
    try:
        stale = media.pending()
    except OSError:
        return
    if stale:
        console.print(
            f"[yellow]媒体库里有 {stale} 个被中断的写入或遗忘[/yellow]：可能留下了没有记录的原图，"
            "运行 [bold]cuemem reindex --all[/bold] 删掉（cuemem serve 与 proxy 启动时也会删）"
        )


def _loose_modes(settings: Settings | None = None) -> list[str]:
    """同一台机器上别的用户能读的数据目录（给了 settings 时）与密钥文件（.env、
    CUEMEM_API_KEYS_FILE）：每个一行黄字与收紧的命令（第四轮体检）。只提醒、不改：已经存在的
    目录与文件可能有意给备份用户留了读权限。数据目录只看最上一级（里面的目录由它管着：上一级是
    0700 时别人进不去）。Windows 没有这些权限位，不查。"""
    if os.name != "posix":
        return []
    from cuemem.api.auth import KEYS_FILE_ENV

    lines: list[str] = []
    if settings is not None:
        from cuemem.store.files import storage_dirs

        dirs = storage_dirs(settings)
        for d in dirs:
            if any(d != top and d.is_relative_to(top) for top in dirs):
                continue  # 上一级不存在时它也不存在
            if mode := _shared_mode(d, directory=True):
                lines.append(
                    f"[yellow]注意：[/yellow]数据目录 {escape(str(d))} 的权限是 {mode:o}，别的用户"
                    f"也能读里面的对话、缓存与 trace：chmod 700 {escape(str(d))}"
                )
    secrets = [env_file_path()]
    if keys_file := os.environ.get(KEYS_FILE_ENV):
        secrets.append(Path(keys_file).expanduser())
    for path in dict.fromkeys(secrets):
        if mode := _shared_mode(path, directory=False):
            lines.append(
                f"[yellow]注意：[/yellow]{escape(str(path))} 的权限是 {mode:o}，别的用户也能读"
                f"里面的 Key 与令牌：chmod 600 {escape(str(path))}"
            )
    return lines


def _shared_mode(path: Path, *, directory: bool) -> int:
    """path 是目录（directory）或普通文件、组或其他用户有任何权限时返回权限位；不是、不存在或
    读不出时返回 0。"""
    try:
        st = path.stat()
    except OSError:
        return 0
    kind = stat.S_ISDIR(st.st_mode) if directory else stat.S_ISREG(st.st_mode)
    mode = stat.S_IMODE(st.st_mode)
    return mode if kind and mode & 0o077 else 0


def _live_check(settings: Settings) -> None:
    from cuemem.embed import build_embedder
    from cuemem.llm.client import LLMClient
    from cuemem.llm.ledger import Ledger

    ledger = Ledger(settings.paths.ledger)
    llm = LLMClient(settings, ledger=ledger)  # 不挂缓存：要测的是真实连通性
    table = Table(title="连通性测试（不走缓存）")
    for column in ("模型", "结果", "延迟", "输入 / 输出 tokens", "回复或错误"):
        table.add_column(column)
    for alias in sorted({rc.model for rc in settings.roles.values()}):
        spec = settings.models[alias]
        if settings.providers[spec.provider].api_key() is None:
            table.add_row(alias, "[yellow]跳过[/yellow]", "-", "-", "缺少 API Key")
            continue
        try:
            result = llm.chat(
                "doctor",
                [{"role": "user", "content": "Reply with the single word: OK"}],
                model=alias,
                max_tokens=16,
                use_cache=False,
            )
            table.add_row(
                alias,
                "[green]成功[/green]",
                f"{result.latency_ms:.0f} ms",
                f"{result.usage.input_tokens} / {result.usage.output_tokens}",
                (result.content or "").strip()[:30],
            )
        except Exception as exc:
            table.add_row(alias, "[red]失败[/red]", "-", "-", f"{type(exc).__name__}: {exc}"[:120])
    console.print(table)

    try:
        embedder = build_embedder(settings, ledger=ledger, use_cache=False)
        started = time.perf_counter()
        batch = embedder.embed(["你好，世界", "hello world"], "document")
        latency = (time.perf_counter() - started) * 1000
        nnz = [len(s) for s in batch.sparse] if batch.sparse is not None else []
        console.print(
            f"向量接口：[green]成功[/green] · {latency:.0f} ms · dense 维度 {batch.dense.shape[1]}"
            f" · sparse 非零项 {nnz} · {batch.tokens} tokens"
        )
    except Exception as exc:
        console.print(f"向量接口：[red]失败[/red] · {type(exc).__name__}: {exc}")

    for provider in settings.used_providers():
        if settings.providers[provider].api_key() is None:
            continue
        try:
            available = set(llm.list_models(provider))
        except Exception as exc:
            console.print(f"{provider} 模型列表：无法获取（{type(exc).__name__}）")
            continue
        configured = sorted(
            {s.model_id for s in settings.models.values() if s.provider == provider}
        )
        missing = [m for m in configured if m not in available]
        note = f"未在列表中：{missing}" if missing else "配置中的模型都在列表中"
        console.print(f"{provider} 模型列表：共 {len(available)} 个；{note}")
    ledger.close()


def _print_forget_plan(plan: dict[str, Any]) -> None:
    if not plan["turns"]:
        console.print("没有找到要删除的内容。")
        return
    table = Table(title=f"预演：将删除 {len(plan['turns'])} 条发言（什么都还没删）")
    for column in ("会话", "角色", "发言"):
        table.add_column(column)
    for turn in plan["turns"]:
        table.add_row(turn["session_id"], turn["role"], escape(turn["text"][:120]))
    console.print(table)
    console.print(
        f"沿引用扩散 {plan['cascaded']} 条；按内容找到的转述 {plan['restated']} 条；"
        f"原图 {plan['images']} 张；会话 {', '.join(plan['sessions_affected'])}"
    )
    for item in plan["facts"]:
        console.print(f"  事实将随之消失：{escape(item['memory'])}")
    console.print("[dim]重建后还会再查一遍有没有从别处长出来的同样信息，预演里没有这一步。[/dim]")


def _print_consolidation(report: ConsolidationReport) -> None:
    if report.sessions == 0 and not report.failed:
        console.print("没有待巩固的会话（任务都已完成）。")
        return
    console.print(
        f"巩固 {report.sessions} 个会话：抽取事实 {report.facts_extracted} 条 → "
        f"新增 {report.added} · 取代 {report.superseded} · 合并 {report.merged} · "
        f"重放 {report.noop}；新建实体 {report.entities_created} 个；"
        f"LLM 调用 {report.llm_calls} 次；耗时 {report.elapsed_ms / 1000:.1f} 秒"
    )
    if report.failed:
        console.print(
            f"[yellow]{len(report.failed)} 个会话抽取失败，下次运行会重试：[/yellow]"
            f"{', '.join(report.failed[:5])}"
        )


def _span_text(start: str | None, end: str | None) -> str:
    if not start and not end:
        return "-"
    return f"{start or '?'} → {end or '至今'}"


def _fact_status(fact: Fact, ids: dict[str, int], *, expired: bool = False) -> str:
    """expired：出处各句都过期了（expiration_date，见 expiry.py），检索与记忆浏览器不给出它。"""
    if fact.is_current:
        return "[red]已过期[/red]" if expired else "[green]当前[/green]"
    successor = f"#{ids[fact.superseded_by]}" if fact.superseded_by in ids else "新版本"
    gone = " · [red]已过期[/red]" if expired else ""
    if fact.supersede_reason == "corrected":
        return f"[yellow]已更正[/yellow]（见 {successor}）{gone}"
    return f"[dim]已变化[/dim]（见 {successor}）{gone}"


def _print_hits(answer: Answer) -> None:
    ref_of = {unit_id: ref for ref, unit_id in answer.refs.items()}
    table = Table(title=f"检索到的记忆（{len(answer.hits)} 条，按相关度排序）")
    for column in ("排名", "编号", "时间", "分数", "内容"):
        table.add_column(column)
    for hit in answer.hits:
        snippet = hit.unit.text.replace("\n", " ")
        table.add_row(
            str(hit.rank),
            ref_of.get(hit.unit.id, "-"),
            f"{hit.unit.ts:%Y-%m-%d}",
            f"{hit.score:.3f}",
            snippet[:80] + ("…" if len(snippet) > 80 else ""),
        )
    console.print(table)
    if answer.facts:
        facts = Table(title=f"检索到的事实（{len(answer.facts)} 条，含沿版本链补齐的）")
        for column in ("编号", "说出时间", "状态", "分数", "陈述"):
            facts.add_column(column, overflow="fold")
        for fh in answer.facts:
            fact = fh.fact
            status = (
                "当前"
                if fact.is_current
                else ("已更正" if fact.supersede_reason == "corrected" else "已变化")
            )
            facts.add_row(
                ref_of.get(fact.id, "-"),
                f"{fact.recorded_at:%Y-%m-%d}",
                status,
                "版本链" if fh.via_chain else f"{fh.score:.3f}",
                fact.statement,
            )
        console.print(facts)


def _print_process(answer: Answer) -> None:
    """L2 的检索计划与充分性检查、控制器的选择、B3 的检索记录、阅读笔记。"""
    details = answer.details
    route = details.get("route")
    if route:
        reasons = "、".join(route["reasons"]) or "无"
        see = "，附原图再看" if route.get("see") else ""
        if route.get("router") == "rule":
            console.print(f"[bold]控制器[/bold]：走 {route['level']}（触发的线索：{reasons}）")
        elif route.get("router") == "llm":
            console.print(f"[bold]控制器[/bold]（llm）：走 {route['level']}{see}（{reasons}）")
        else:
            scores = route.get("scores", {})
            extra = "，问了 LLM" if scores.get("asked_llm") else ""
            console.print(
                f"[bold]控制器[/bold]（{route.get('router')}）：走 {route['level']}{see}"
                f"（推理类型：{reasons}；与记忆无关的概率 {scores.get('p_none', 0):.0%}{extra}）"
            )
    plan = details.get("plan")
    if plan:
        console.print(
            f"[bold]检索计划[/bold]（{'LLM' if plan['source'] == 'llm' else '规则'}）："
            f"答案类型 {plan['answer_type']}，需要穷尽 {'是' if plan['exhaustive'] else '否'}，"
            f"时间区间 {plan['time_range'] or '无'}"
        )
        for query in plan["sub_queries"]:
            console.print(f"  · 子查询：{query}")
    for check in details.get("checks", []):
        verdict = "够了" if check["sufficient"] else f"不够（{check['missing']}）"
        added = f"，补检索 {check['added_rounds']} 轮" if check.get("added_rounds") else ""
        console.print(f"[bold]充分性检查 {check['iteration']}[/bold]：{verdict}{added}")
        for query in check["queries"]:
            console.print(f"  · 补充查询：{query}")
    for query in details.get("searches", []):
        console.print(f"[bold]模型检索[/bold]：{query}")
    for note in answer.notes:
        console.print(f"[dim]笔记：{note}[/dim]")


def _print_usage_rows(rows: list[dict[str, Any]], *, detailed: bool = False) -> None:
    if not rows:
        return
    if not detailed:
        for r in rows:
            symbol = CURRENCY_SYMBOL.get(r["currency"], r["currency"])
            who = f"（{r['role']}）" if r["role"] else ""
            cached = f"，其中本地缓存命中 {r['cache_hits']} 次" if r["cache_hits"] else ""
            console.print(
                f"[dim]{r['model']}{who}：调用 {r['calls']} 次{cached}，"
                f"输入 {r['input_tokens']} / 输出 {r['output_tokens']} tokens，"
                f"名义成本 {symbol}{r['nominal_cost']:.4f}[/dim]"
            )
        return
    table = Table(title="调用量与成本")
    columns = ("类型", "模型", "角色", "调用", "缓存命中", "输入", "输出", "名义成本", "估算花费")
    for column in columns:
        table.add_column(column)
    for r in rows:
        symbol = CURRENCY_SYMBOL.get(r["currency"], r["currency"])
        table.add_row(
            r["kind"],
            r["model"],
            r["role"] or "-",
            str(r["calls"]),
            str(r["cache_hits"]),
            str(r["input_tokens"]),
            str(r["output_tokens"]),
            f"{symbol}{r['nominal_cost']:.4f}",
            f"{symbol}{r['spent']:.4f}",
        )
    console.print(table)
    console.print(
        "[dim]名义成本按高峰、未命中缓存的标价计算，便于实验间比较；"
        "估算花费扣除了本地缓存命中与供应商缓存折扣，未计入错峰折扣与免费额度。[/dim]"
    )
