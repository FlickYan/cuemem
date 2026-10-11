"""命令行入口。

cuemem configure                      交互式填写 API Key（写入 .env）
cuemem doctor [--live]                检查配置、API Key 与连通性
cuemem ingest FILE [--space S]        导入对话
cuemem ask "问题" --space S [--at T]  检索记忆并回答
cuemem spaces                         列出记忆空间
cuemem usage [--today] [--run ID]     汇总调用量与成本
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cuemem.config import Settings, env_file_path, load_settings
from cuemem.core.models import Answer, Conversation
from cuemem.embed import build_embedder
from cuemem.envfile import mask, read_env, update_env
from cuemem.errors import CueMemError
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.memory import MemorySystem
from cuemem.store.sqlite import MemoryDB

app = typer.Typer(
    help="CueMem：线索驱动的多模态 Agent 长期记忆系统",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DATETIME_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]
CURRENCY_SYMBOL = {"CNY": "¥", "USD": "$"}
PROVIDER_LABELS = {"deepseek": "DeepSeek", "qwen": "QwenCloud（通义千问）"}
KEY_HINTS = {"qwen": "以 sk-ws- 开头"}  # 帮助区分同一平台上的其他 Key


@contextmanager
def _open_system() -> Iterator[MemorySystem]:
    """组装 MemorySystem；把 CueMem 的已知错误转成友好的提示。"""
    system: MemorySystem | None = None
    try:
        system = MemorySystem.from_settings(load_settings())
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
    updates: dict[str, str | None] = {}
    for name, provider in settings.providers.items():
        label = PROVIDER_LABELS.get(name, name)
        hint = f"（{KEY_HINTS[name]}）" if name in KEY_HINTS else ""
        updates[provider.api_key_env] = _prompt_secret(
            f"{label} API Key{hint}", current.get(provider.api_key_env)
        )
    update_env(path, updates)

    summary = Table(title=f"已写入 {path.name}（仅本人可读写）")
    for column in ("供应商", "环境变量", "Key", "接口"):
        summary.add_column(column)
    for name, provider in settings.providers.items():
        key = updates[provider.api_key_env]
        summary.add_row(
            PROVIDER_LABELS.get(name, name),
            provider.api_key_env,
            mask(key) if key else "[red]未填写[/red]",
            provider.base_url,
        )
    console.print(summary)
    console.print("下一步：[bold]uv run cuemem doctor --live[/bold]")


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
) -> None:
    """导入一段多会话对话（可重复执行，不会产生重复数据）。"""
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
        _print_usage_rows(system.ledger.summary(run_id=system.ledger.run_id))


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
) -> None:
    """检索记忆并回答问题（N1：朴素 dense 检索，即 B2 基线）。"""
    with _open_system() as system:
        started = time.perf_counter()
        answer = system.ask(space, question, question_time=at, k=k)
        elapsed = (time.perf_counter() - started) * 1000
        if show_context:
            _print_hits(answer)
        console.print(Panel(answer.answer, title="回答", expand=False))
        cited = ", ".join(answer.citations) or "无"
        sufficient = "是" if answer.evidence_sufficient else "否"
        console.print(f"引用：{cited} · 证据充分：{sufficient} · 总耗时 {elapsed:.0f} ms")
        _print_usage_rows(system.ledger.summary(run_id=system.ledger.run_id))


@app.command()
def spaces() -> None:
    """列出所有记忆空间。"""
    settings = load_settings()
    db = MemoryDB(settings.paths.db)
    rows = db.list_spaces()
    db.close()
    if not rows:
        console.print("还没有任何记忆空间。先运行 cuemem ingest。")
        return
    table = Table(title="记忆空间")
    for column in ("空间", "会话", "轮次", "记忆单元", "创建时间"):
        table.add_column(column)
    for r in rows:
        table.add_row(
            r["space_id"],
            str(r["sessions"]),
            str(r["turns"]),
            str(r["units"]),
            r["created_at"][:19],
        )
    console.print(table)


@app.command()
def usage(
    today: Annotated[bool, typer.Option("--today", help="只统计今天")] = False,
    run: Annotated[str | None, typer.Option(help="只统计某次运行（run_id）")] = None,
) -> None:
    """汇总模型调用量与成本。"""
    settings = load_settings()
    ledger = Ledger(settings.paths.ledger)
    since = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) if today else None
    rows = ledger.summary(run_id=run, since=since)
    ledger.close()
    if not rows:
        console.print("没有调用记录。")
        return
    _print_usage_rows(rows, detailed=True)


# ---------- 交互辅助 ----------


def _prompt_secret(label: str, existing: str | None) -> str | None:
    hint = f"（当前 {mask(existing)}，回车保留）" if existing else ""
    value = typer.prompt(f"{label}{hint}", default="", show_default=False, hide_input=True)
    return value.strip() or existing


# ---------- 输出辅助 ----------


def _print_config(settings: Settings) -> None:
    providers = Table(title="供应商")
    for column in ("名称", "Base URL", "Key 环境变量", "状态"):
        providers.add_column(column)
    for name, p in settings.providers.items():
        status = "[green]已设置[/green]" if p.api_key() else "[red]未设置 Key[/red]"
        providers.add_row(name, p.base_url, p.api_key_env, status)
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
        f"{emb.model}（{emb.provider}），{emb.dim} 维，sparse={'开' if emb.sparse else '关'}"
        if emb.backend == "dashscope"
        else f"离线哈希向量，{emb.dim} 维（仅用于测试与演示）"
    )
    console.print(f"向量：{detail}")
    vector_store = settings.vector_store.url or f"本地模式 {settings.paths.qdrant}"
    console.print(f"数据库：{settings.paths.db}")
    console.print(f"向量库：{vector_store}")


def _live_check(settings: Settings) -> None:
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

    for provider in settings.providers:
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
