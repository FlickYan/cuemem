"""命令行入口。

cuemem configure                      交互式填写 API Key（写入 .env）
cuemem doctor [--live]                检查配置、API Key 与连通性
cuemem ingest FILE [--consolidate]    导入对话（可选：立即巩固）
cuemem consolidate --space S          巩固：抽取事实与实体、对账、摘要、核心记忆
cuemem ask "问题" --space S [--at T]  检索记忆并回答
cuemem inspect facts|entities|sources|profile --space S   记忆检查器
cuemem forget --space S --session X   遗忘：删除会话及由它派生的一切
cuemem spaces                         列出记忆空间
cuemem usage [--today] [--run ID]     汇总调用量与成本
cuemem data download [NAME...]       下载评测数据集
cuemem eval split / run / report     评测：划分开发集、运行、生成报告
cuemem eval audit RUN [--score]      判分抽检：导出待核对条目 / 统计一致率
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
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from cuemem.config import Settings, env_file_path, load_settings
from cuemem.core.models import Answer, Conversation, Fact
from cuemem.embed import build_embedder
from cuemem.envfile import mask, read_env, update_env
from cuemem.errors import CueMemError
from cuemem.evals.audit import export_audit, score_audit
from cuemem.evals.datasets import DATASETS, EvalSet, dataset_path
from cuemem.evals.datasets import download as download_dataset
from cuemem.evals.datasets import load as load_dataset
from cuemem.evals.report import build_report, load_rows, method_stats
from cuemem.evals.runner import EvalRunner, RunConfig
from cuemem.evals.splits import load_split, save_split, stratified_ids
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.llm.peak import describe, next_peak_start, peak_end
from cuemem.memory import MemorySystem
from cuemem.store.sqlite import MemoryDB
from cuemem.write.consolidate import ConsolidationReport

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
    """组装 MemorySystem；把 CueMem 的已知错误转成友好的提示。

    交互式命令只发少量请求：分时价格的高峰时段照常调用（记录提醒），不像评测那样暂停。
    """
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
    consolidate: Annotated[
        bool, typer.Option("--consolidate", help="导入后立即巩固（抽取事实、实体与摘要）")
    ] = False,
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
        if consolidate:
            _print_consolidation(system.consolidate(report.space_id))
        else:
            console.print("[dim]提示：运行 cuemem consolidate 抽取事实与实体（需要 LLM）。[/dim]")
        _print_usage_rows(system.ledger.summary(run_id=system.ledger.run_id))


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
    yes: Annotated[bool, typer.Option("--yes", "-y", help="不再确认，直接删除")] = False,
) -> None:
    """遗忘：删除原始轮次，并清除由它们派生的事实、摘要、向量与缓存（不可恢复）。"""
    if not session and not turn:
        console.print("[red]请用 --session 或 --turn 指定要遗忘的内容[/red]")
        raise typer.Exit(1)
    targets = ", ".join([*(session or []), *(turn or [])])
    if not yes and not typer.confirm(f"将永久删除 {targets} 及其派生的全部记忆，确定吗？"):
        raise typer.Exit(1)
    with _open_system() as system:
        report = system.forget(space, session_ids=session or [], turn_ids=turn or [])
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
        str, typer.Option(help="facts：事实增强检索（B2F，默认）；naive：朴素检索（B2）")
    ] = "facts",
) -> None:
    """检索记忆并回答问题。"""
    if mode not in ("facts", "naive"):
        console.print("[red]--mode 只能是 facts 或 naive[/red]")
        raise typer.Exit(1)
    with _open_system() as system:
        started = time.perf_counter()
        answer = system.ask(
            space, question, question_time=at, k=k, mode="naive" if mode == "naive" else "facts"
        )
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
    settings = load_settings()
    ledger = Ledger(settings.paths.ledger)
    since = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) if today else None
    rows = ledger.summary(run_id=run, since=since)
    ledger.close()
    if not rows:
        console.print("没有调用记录。")
        return
    _print_usage_rows(rows, detailed=True)


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
    names: Annotated[list[str] | None, typer.Argument(help="数据集名，默认全部")] = None,
) -> None:
    """下载评测数据集到 data/datasets（已存在则跳过）。"""
    settings = load_settings()
    for name in names or list(DATASETS):
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
) -> None:
    """运行评测：写入 → 回答 → 判分；结果写入 runs/eval/<运行名>/，最后生成报告。"""
    if peak is not None and peak not in ("wait", "warn", "allow"):
        console.print("[red]--peak 只能是 wait、warn 或 allow[/red]")
        raise typer.Exit(1)
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
    )
    count = min(limit, len(evalset.samples)) if limit else len(evalset.samples)
    console.print(f"运行 [bold]{run_name}[/bold]：{count} 题 × {config.methods}")
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
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
        summary = runner.run(evalset)
    report_path = build_report(summary.run_dir)
    note = "（已达到花费上限，提前停止）" if summary.stopped_by_budget else ""
    console.print(
        f"完成 {summary.completed} / {summary.planned} 个组合，出错 {summary.errors} 个{note}；"
        f"耗时 {summary.elapsed_s:.0f} 秒；本次实际花费 ${summary.spent:.4f}"
        f"（名义成本 ${summary.nominal_cost:.4f}）"
    )
    _print_method_stats(summary.run_dir)
    console.print(f"报告：{report_path}")


@eval_app.command("report")
def eval_report(
    run: Annotated[str, typer.Argument(help="运行名或运行目录")],
) -> None:
    """重新生成某次运行的报告。"""
    settings = load_settings()
    run_dir = Path(run) if Path(run).exists() else settings.paths.eval_runs / run
    if not (run_dir / "run.json").exists():
        console.print(f"[red]找不到运行目录 {run_dir}[/red]")
        raise typer.Exit(1)
    path = build_report(run_dir)
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


def _load_evalset(settings: Settings, dataset: str, split: str | None) -> EvalSet:
    if dataset not in DATASETS:
        console.print(f"[red]未知数据集 {dataset}[/red]，可选：{', '.join(DATASETS)}")
        raise typer.Exit(1)
    path = dataset_path(settings.paths.datasets, dataset)
    if not path.exists():
        console.print(
            f"[red]缺少数据文件 {path}[/red]，请先运行：uv run cuemem data download {dataset}"
        )
        raise typer.Exit(1)
    ids = load_split(dataset, split) if split else None
    return load_dataset(dataset, path, ids=ids)


def _print_method_stats(run_dir: Path) -> None:
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
    for name, provider in settings.providers.items():
        cfg = provider.peak_hours
        if cfg is not None and (end := peak_end(cfg)) is not None:
            console.print(
                f"[dim]提示：现在是 {PROVIDER_LABELS.get(name, name)} 的高峰时段"
                f"（价格为平时的 {1 / cfg.off_peak_factor:g} 倍，到 {end:%H:%M} 结束）[/dim]"
            )


def _peak_status(settings: Settings) -> list[str]:
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
    for line in _peak_status(settings):
        console.print(line)
    console.print(f"批量任务（评测）在高峰时段的策略：{settings.peak_policy}")


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
