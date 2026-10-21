"""命令行入口。

cuemem [--preset openai|ollama] …     换模型供应商（也可设 CUEMEM_PRESET，N12）
cuemem configure                      交互式填写 API Key（写入 .env）
cuemem doctor [--live]                检查配置、API Key 与连通性
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
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast, get_args

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from cuemem.config import Settings, cli_command, env_file_path, extra_remedy, load_settings
from cuemem.core.types import AskMode, RecallMode, Router, Scenario
from cuemem.envfile import mask, read_env, update_env
from cuemem.errors import ConfigMismatchError, CueMemError

# 记忆系统、评测、Agent 与服务在用到它们的命令里才导入：cuemem --help、cuemem mcp 这些命令
# 不加载 matplotlib、评测与用不到的模块（以前每个命令启动都要 3 秒多）
if TYPE_CHECKING:
    from cuemem.agent import Reply
    from cuemem.api.service import ChatService
    from cuemem.core.models import Answer, Fact
    from cuemem.evals.datasets import EvalSet
    from cuemem.evals.runner import RunSummary
    from cuemem.memory import MemorySystem
    from cuemem.write.consolidate import ConsolidationReport

app = typer.Typer(
    help="CueMem：线索驱动的多模态 Agent 长期记忆系统",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def main(
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            help="换模型供应商的预设：openai、ollama（本地，不要 Key），或预设 YAML 的路径；"
            "也可以设环境变量 CUEMEM_PRESET",
        ),
    ] = None,
) -> None:
    """CueMem：线索驱动的多模态 Agent 长期记忆系统"""
    if preset:  # 各命令都通过 load_settings 读配置，它看这个环境变量（N12）
        os.environ["CUEMEM_PRESET"] = preset


DATETIME_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]
CURRENCY_SYMBOL = {"CNY": "¥", "USD": "$"}
PROVIDER_LABELS = {
    "deepseek": "DeepSeek",
    "qwen": "QwenCloud（通义千问）",
    "openai": "OpenAI",
    "ollama": "Ollama（本地）",
}
KEY_HINTS = {"qwen": "以 sk-ws- 开头"}  # 帮助区分同一平台上的其他 Key


@contextmanager
def _open_system() -> Iterator[MemorySystem]:
    """组装 MemorySystem；把 CueMem 的已知错误转成友好的提示。

    交互式命令只发少量请求：分时价格的高峰时段照常调用（记录提醒），不像评测那样暂停。
    """
    from cuemem.memory import MemorySystem

    system: MemorySystem | None = None
    try:
        settings = load_settings(peak_policy="warn")
        _peak_note(settings)
        system = MemorySystem.from_settings(settings)
        yield system
    except CueMemError as exc:
        console.print(f"[red]错误：[/red]{exc}")
        raise typer.Exit(1) from exc
    finally:
        if system is not None:
            system.close()


@app.command()
def configure() -> None:
    """交互式填写各供应商的 API Key，写入 .env（Key 输入时不显示）。"""
    settings = load_settings()
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
) -> None:
    """检查配置、API Key，以及（可选）各模型的连通性。"""
    settings = load_settings()
    _print_config(settings)
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

    with _open_system() as system:
        try:
            data = export_space(system, space, derived=not no_derived)
        except KeyError as exc:
            console.print(f"[red]没有记忆空间 {space}[/red]")
            raise typer.Exit(1) from exc
    text = json.dumps(data, ensure_ascii=False, indent=1)
    if out is None:
        print(text)
        return
    out.write_text(text, encoding="utf-8")
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

    data = json.loads(path.read_text(encoding="utf-8"))
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
def forget(
    space: Annotated[str, typer.Option(help="记忆空间 ID")],
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
) -> None:
    """遗忘：删除原始轮次，并清除由它们派生的事实、摘要、向量与缓存（不可恢复）。"""
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
        console.print(
            f"已删除 {report.turns_deleted} 条轮次（会话 {', '.join(report.sessions_affected)}）；"
            f"事实 {report.facts_before} → {report.facts_after} 条（已按剩余日志重建）；"
            f"清理向量缓存 {report.embed_cache_purged} 条、LLM 缓存 {report.llm_cache_purged} 条；"
            f"耗时 {report.elapsed_ms:.0f} ms"
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
) -> None:
    """检索记忆并回答问题。"""
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
    from cuemem.store.sqlite import MemoryDB

    settings = load_settings()
    db = MemoryDB(settings.paths.db)
    rows = db.list_spaces()
    db.close()
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

    settings = load_settings()
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

    try:  # 缺 Key、换了向量模型、没有商品目录等：提示原因，不打印调用栈（Docker 里会反复重启）
        settings = load_settings(peak_policy="warn")
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
    "监听非本机地址又没有设置 CUEMEM_API_TOKEN 时也照常启动（不鉴权：能连到这台机器的人都能"
    "读写、删除记忆，用你的上游 Key）。只在可信的网络里用"
)


def _is_loopback(host: str) -> bool:
    """监听地址是不是只有本机能连（127.0.0.0/8、::1、localhost）。"""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host.lower() == "localhost"


def _require_token(host: str, token: str | None, insecure: bool) -> None:
    """监听非本机地址时必须设访问令牌：没有令牌时拒绝启动，除非显式加 --insecure-no-auth。
    容器里总是监听 0.0.0.0（看不出宿主机只映射到了本机），所以镜像也要设令牌。"""
    from cuemem.api.app import TOKEN_ENV

    if token or insecure or _is_loopback(host):
        return
    console.print(
        f"[red]监听 {escape(host)} 但没有设置 {TOKEN_ENV}，拒绝启动：能连到这台机器的人都能"
        f"读写、删除记忆，用你的上游 Key。[/red]\n在 .env 里设置 {TOKEN_ENV}（例如用 openssl rand"
        " -hex 32 生成）；只在可信的网络里使用时也可以加 --insecure-no-auth。"
    )
    raise typer.Exit(2)


def _host_policy(host: str) -> tuple[list[str] | None, list[str]]:
    """（Host 白名单，除同源外放行的 Origin 主机名）。监听本机地址时只接受 Host 为本机的请求
    （防 DNS 重绑定：恶意域名解析到 127.0.0.1 后，它的网页就算同源）；监听别的地址时不查 Host
    （Docker、局域网用什么地址访问都行，靠访问令牌）。CUEMEM_ALLOWED_HOSTS 列出的主机名两边都放行，
    含 "*" 时不查 Host、不限 Origin。"""
    from cuemem.api.app import HOSTS_ENV, LOOPBACK_HOSTS

    extra = [h.strip().lower() for h in os.environ.get(HOSTS_ENV, "").split(",") if h.strip()]
    if "*" in extra or not _is_loopback(host):
        return None, extra
    bind = f"[{host}]" if ":" in host else host.lower()  # 例如 127.0.0.2
    return list(dict.fromkeys([*LOOPBACK_HOSTS, bind, *extra])), extra


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
    insecure_no_auth: Annotated[
        bool, typer.Option("--insecure-no-auth", help=INSECURE_HELP)
    ] = False,
) -> None:
    """启动 HTTP 接口（/api，文档在 /docs）与演示界面（Gradio，根路径）。

    界面要用可选依赖 ui（Gradio）；没装时只启动 HTTP 接口与记忆代理，并说明怎么装。"""
    import uvicorn

    from cuemem.api.app import TOKEN_ENV, create_app
    from cuemem.api.proxy import MemoryProxy

    ui = not no_ui
    if ui and importlib.util.find_spec("gradio") is None:
        console.print(
            "[yellow]注意：[/yellow]演示界面要用可选依赖 ui（Gradio），没有安装："
            f"{escape(extra_remedy('ui'))}。这次只启动 HTTP 接口与记忆代理"
        )
        ui = False

    def check() -> None:  # 加载配置时已读入 .env；建服务之前检查，被拒绝时不接着做上次的抽取
        _require_token(host, os.environ.get(TOKEN_ENV) or None, insecure_no_auth)

    settings, service = _service(space, scenario or _default_scenario(), router, lam, check=check)
    token = os.environ.get(TOKEN_ENV) or None
    allowed_hosts, origin_hosts = _host_policy(host)
    proxy = MemoryProxy(service.system)  # 同一个记忆系统：经代理写入的对话在界面里也能看到
    try:
        app_ = create_app(
            service,
            upload_dir=settings.paths.media.parent / "uploads",
            ui=ui,
            token=token,
            proxy=proxy,
            allowed_hosts=allowed_hosts,
            origin_hosts=origin_hosts,
        )
        where = f"界面：[bold]http://{host}:{port}/[/bold] · " if ui else ""
        console.print(
            f"{where}接口文档：http://{host}:{port}/docs"
            f" · OpenAI 兼容的记忆代理：http://{host}:{port}/v1"
        )
        if token:
            console.print(f"已启用访问令牌（{TOKEN_ENV}）：接口要带 Bearer 令牌，界面要先登录")
        elif not _is_loopback(host):  # 加了 --insecure-no-auth
            console.print(
                f"[yellow]注意：[/yellow]监听 {escape(host)} 但没有设置 {TOKEN_ENV}"
                "（--insecure-no-auth），能连到这台机器的人都能读写、删除记忆"
            )
        uvicorn.run(app_, host=host, port=port, log_level="warning")
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
) -> None:
    """OpenAI 兼容的记忆代理（N12）：把应用的 base_url 换成 http://HOST:PORT/v1 就有了长期记忆。"""
    import uvicorn

    from cuemem.api.app import TOKEN_ENV
    from cuemem.api.proxy import MemoryProxy, create_proxy_app
    from cuemem.memory import MemorySystem

    if mode not in get_args(RecallMode):
        console.print(f"[red]--mode 只能是 {'、'.join(get_args(RecallMode))}[/red]")
        raise typer.Exit(1)
    settings = load_settings(peak_policy="warn")
    token = os.environ.get(TOKEN_ENV) or None  # load_settings 已读入 .env
    _require_token(host, token, insecure_no_auth)
    allowed_hosts, origin_hosts = _host_policy(host)
    try:
        system = MemorySystem.from_settings(settings)
    except CueMemError as exc:  # 缺 Key、换了向量模型等：提示原因，不打印调用栈
        console.print(f"[red]错误：[/red]{escape(str(exc))}")
        raise typer.Exit(1) from exc
    memory_proxy = MemoryProxy(
        system, default_user=user, mode=cast(RecallMode, mode), idle_minutes=idle_minutes
    )
    try:
        console.print(
            f"记忆代理：[bold]http://{host}:{port}/v1[/bold]（OpenAI 兼容；"
            f"上游默认模型 {settings.roles['agent'].model}）"
        )
        console.print(
            "用户：请求体的 user 字段，或请求头 X-CueMem-User"
            + (f"；默认 {user}" if user else "；没有时原样转发、不碰记忆")
        )
        if token:
            console.print(f"已启用访问令牌（{TOKEN_ENV}）：OpenAI SDK 的 api_key 填这个令牌")
        elif not _is_loopback(host):  # 加了 --insecure-no-auth
            console.print(
                f"[yellow]注意：[/yellow]监听 {escape(host)} 但没有设置 {TOKEN_ENV}"
                "（--insecure-no-auth），能连到这台机器的人都能用你的上游 Key、读写记忆"
            )
        app_ = create_proxy_app(
            memory_proxy, token=token, allowed_hosts=allowed_hosts, origin_hosts=origin_hosts
        )
        uvicorn.run(app_, host=host, port=port)
    finally:
        memory_proxy.close_all()
        system.close()


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
                system.db.delete_space(space_id)
                system.vectors.delete_space(space_id)
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

    数据库里已有的记忆不会被重新写入向量库，所以直接切换会得到一个空的向量库。
    """
    from cuemem.store.vector import VectorStore

    settings = load_settings()
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
    target = VectorStore(None, dim=settings.embedding.dim, url=to)
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
    console.print(f"下一步：在 .env 里加上 [bold]CUEMEM_VECTOR_STORE__URL={to}[/bold]")


@app.command("mcp")
def mcp_cmd(
    space: Annotated[str, typer.Option(help="记忆空间 ID")] = "default",
    router: Annotated[str, typer.Option(help="recall 用的检索控制器")] = "learned",
) -> None:
    """以 MCP Server（stdio）提供记忆工具，让其他 Agent（如 Claude Code）直接使用这套记忆。"""
    from cuemem.api.mcp import build_server
    from cuemem.api.service import ChatService

    settings = load_settings(peak_policy="warn")
    service = ChatService.create(
        settings, scenario="plain", default_space=space, router=cast(Router, router)
    )
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
        str | None, typer.Option("--valid-at", help="事件时间：那时现实中成立的事实，如 2025-06")
    ] = None,
    known_at: Annotated[
        datetime | None,
        typer.Option("--known-at", formats=DATETIME_FORMATS, help="记录时间：系统那时相信的事实"),
    ] = None,
) -> None:
    """事实时间线：按主语分组、按时间排列，标出被取代的版本。"""
    with _open_system() as system:
        facts = system.facts(space, current_only=current, valid_at=valid_at, known_at=known_at)
        if subject:
            facts = [f for f in facts if f.subject.casefold() == subject.casefold()]
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
                _fact_status(fact, ids),
                fact.id,
            )
        console.print(table)


@inspect_app.command("entities")
def inspect_entities(space: Annotated[str, typer.Option(help="记忆空间 ID")]) -> None:
    """实体列表：名字、类型、别名、提及次数。"""
    with _open_system() as system:
        entities = system.entities(space)
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
    with _open_system() as system:
        found = system.db.get_facts(space, ids=[fact_id])
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
        chain = system.db.get_facts(space, superseded_by=[fact.id])
        if fact.superseded_by:
            chain += system.db.get_facts(space, ids=[fact.superseded_by])
        for other in chain:
            relation = "取代了" if other.superseded_by == fact.id else "被它取代"
            console.print(f"[dim]{relation}：{other.statement}（{other.id}）[/dim]")
        turns = system.db.get_turns(space, ids=fact.source_turn_ids)
        for t in turns:
            who = t.speaker or ("用户" if t.role == "user" else "助手")
            console.print(f"[bold]{t.ts:%Y-%m-%d} · {t.session_id} · {who}[/bold]：{t.text}")


@inspect_app.command("profile")
def inspect_profile(space: Annotated[str, typer.Option(help="记忆空间 ID")]) -> None:
    """核心记忆（用户画像）：重要且当前有效的事实，每轮常驻上下文。"""
    with _open_system() as system:
        core = system.core_memory(space)
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
    """下载评测数据集到 data/datasets（已存在则跳过）。"""
    from cuemem.evals.datasets import DATASETS
    from cuemem.evals.datasets import download as download_dataset

    settings = load_settings()
    for name in names or [n for n, spec in DATASETS.items() if not spec.optional]:
        if name not in DATASETS:
            console.print(f"[red]未知数据集 {name}[/red]，可选：{', '.join(DATASETS)}")
            raise typer.Exit(1)
        console.print(f"{name}：{DATASETS[name].description}")
        path = download_dataset(name, settings.paths.datasets)
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

    settings = load_settings()
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
    from cuemem.evals.report import build_report
    from cuemem.evals.runner import EvalRunner, RunConfig

    settings = load_settings()
    if peak is not None:
        settings = settings.model_copy(update={"peak_policy": peak})
    for line in _peak_status(settings):
        console.print(line)
    split_name = None if split == "all" else split
    evalset = _load_evalset(settings, dataset, split_name)
    if types:
        wanted = {t.strip() for t in types.split(",")}
        evalset.samples = [s for s in evalset.samples if s.question_type in wanted]
    elif dataset == "locomo":  # 与文献一致：默认不评对抗题
        evalset.samples = [s for s in evalset.samples if s.question_type != "adversarial"]
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
    from cuemem.evals.runner import EvalRunner, RunConfig

    settings = load_settings()
    split_name = None if split == "all" else split
    evalset = _load_evalset(settings, dataset, split_name)
    if dataset == "locomo":  # 与评测一致：不含对抗题
        evalset.samples = [s for s in evalset.samples if s.question_type != "adversarial"]
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

    settings = load_settings()
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

    settings = load_settings()
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
    vector_store = settings.vector_store.url or f"本地模式 {settings.paths.qdrant}"
    console.print(f"数据库：{settings.paths.db}")
    console.print(f"向量库：{vector_store}")
    for line in _peak_status(settings):
        console.print(line)
    console.print(f"批量任务（评测）在高峰时段的策略：{settings.peak_policy}")


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


def _fact_status(fact: Fact, ids: dict[str, int]) -> str:
    if fact.is_current:
        return "[green]当前[/green]"
    successor = f"#{ids[fact.superseded_by]}" if fact.superseded_by in ids else "新版本"
    if fact.supersede_reason == "corrected":
        return f"[yellow]已更正[/yellow]（见 {successor}）"
    return f"[dim]已变化[/dim]（见 {successor}）"


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
