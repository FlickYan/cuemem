"""评测报告：把 results.jsonl 汇总成 Markdown 表格，并画"准确率–成本"散点图。

图按可视化规范绘制：系列颜色按方法固定分配（不随排名变化），散点图最多 3 个系列色
（ORACLE 是上限参照，用中性色空心点）；浅色、深色各渲染一张，Markdown 里用 <picture>
按读者的主题自动切换。图中文字用英文，避免 CI 环境缺少中文字体。
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.transforms import Bbox

from cuemem.evals.metrics import bootstrap_ci, mean, percentile

METHOD_LABELS = {
    "B0": ("无记忆", "B0 No memory"),
    "B1": ("全上下文", "B1 Full context"),
    "B2": ("朴素检索", "B2 Naive RAG"),
    "ORACLE": ("只给证据会话（上限参照）", "Oracle: evidence only"),
}
METHOD_ORDER = ["B0", "B1", "B2", "ORACLE"]

TYPE_LABELS = {
    "single-session-user": "单会话 · 用户说过",
    "single-session-assistant": "单会话 · 助手说过",
    "single-session-preference": "单会话 · 偏好",
    "multi-session": "多会话推理",
    "temporal-reasoning": "时间推理",
    "knowledge-update": "知识更新",
    "single-hop": "单跳",
    "multi-hop": "多跳",
    "open-domain": "开放域",
    "temporal": "时间",
    "adversarial": "对抗（应拒答）",
}

RETRIEVAL_LABELS = {
    "session_recall_any": ("会话召回（任一）", "前 k 条里至少命中一个证据会话的比例"),
    "session_recall_all": ("会话召回（全部）", "证据会话被前 k 条覆盖的平均比例"),
    "turn_recall_any": ("发言召回（任一）", "前 k 条里至少有一条包含证据发言的比例"),
    "turn_ndcg": ("发言 NDCG", "包含证据发言的轮排得越靠前越高"),
}


# 可视化规范的参考配色（已用 validate_palette.js 在 --pairs all 下验证前 3 色，深浅两套）
@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str  # 主墨色：标题
    ink2: str  # 次墨色：轴标题、标签
    muted: str  # 刻度与说明文字
    grid: str
    axis: str
    series: dict[str, str]  # 方法 → 系列颜色（按方法固定，不随排名变化）


THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series={"B0": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a"},
    ),
    "dark": Theme(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series={"B0": "#3987e5", "B1": "#d95926", "B2": "#199e70"},
    ),
}


@dataclass
class MethodStats:
    method: str
    n: int
    accuracy: float
    ci: tuple[float, float]
    by_type: dict[str, tuple[int, float]]  # 题型 → (题数, 准确率)
    abstention: tuple[int, float] | None
    input_tokens: float
    output_tokens: float
    llm_calls: float
    cost: float  # 每题平均名义成本（只算回答，不含判分与写入）
    total_cost: float
    latency_p50: float
    latency_p95: float
    retrieval: dict[str, float]
    judge_cost: float
    f1: float | None


def load_rows(run_dir: Path, filename: str) -> list[dict[str, Any]]:
    path = run_dir / filename
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # 进程被中断时最后一行可能不完整
            continue
    return rows


def method_stats(rows: Sequence[dict[str, Any]]) -> list[MethodStats]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    stats = []
    for method in sorted(by_method, key=_method_rank):
        items = by_method[method]
        correct = [1.0 if r["correct"] else 0.0 for r in items]
        types: dict[str, list[float]] = defaultdict(list)
        for r, c in zip(items, correct, strict=True):
            if not r["is_abstention"]:
                types[r["question_type"]].append(c)
        abstained = [c for r, c in zip(items, correct, strict=True) if r["is_abstention"]]
        retrieval: dict[str, list[float]] = defaultdict(list)
        for r in items:
            for key, value in (r.get("retrieval") or {}).items():
                retrieval[key].append(value)
        latencies = [r["usage"]["latency_ms"] for r in items if r["usage"]["latency_ms"]]
        f1s = [r["f1"] for r in items if r.get("f1") is not None]
        stats.append(
            MethodStats(
                method=method,
                n=len(items),
                accuracy=mean(correct),
                ci=bootstrap_ci(correct),
                by_type={t: (len(v), mean(v)) for t, v in types.items()},
                abstention=(len(abstained), mean(abstained)) if abstained else None,
                input_tokens=mean([r["usage"]["input_tokens"] for r in items]),
                output_tokens=mean([r["usage"]["output_tokens"] for r in items]),
                llm_calls=mean([r["usage"]["llm_calls"] for r in items]),
                cost=mean([r["usage"]["cost"] for r in items]),
                total_cost=sum(r["usage"]["cost"] for r in items),
                latency_p50=percentile(latencies, 50),
                latency_p95=percentile(latencies, 95),
                retrieval={k: mean(v) for k, v in retrieval.items()},
                judge_cost=sum(r["judge_usage"]["cost"] for r in items),
                f1=mean(f1s) if f1s else None,
            )
        )
    return stats


def build_report(run_dir: Path) -> Path:
    """生成 report.md 与两张散点图，返回 report.md 的路径。"""
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = load_rows(run_dir, "results.jsonl")
    ingests = load_rows(run_dir, "ingest.jsonl")
    done = {(r["sample_id"], r["method"]) for r in rows}
    # 出错后重跑成功的组合不再算作错误
    errors = [
        e for e in load_rows(run_dir, "errors.jsonl") if (e["sample_id"], e["method"]) not in done
    ]
    stats = method_stats(rows)
    has_plot = bool(stats)
    if has_plot:
        for mode in THEMES:
            plot_pareto(stats, run_dir / f"pareto-{mode}.png", mode=mode, info=info)
    markdown = render_markdown(info, rows, stats, ingests, errors, has_plot=has_plot)
    path = run_dir / "report.md"
    path.write_text(markdown, encoding="utf-8")
    return path


# ---------- Markdown ----------


def error_attribution(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """把检索类方法答错的题分成三类（拒答题除外）：
    - retrieval_miss：前 k 条里没有任何证据发言；
    - noisy_or_incomplete：命中了证据但答错，而 ORACLE 答对——检索结果干扰多或证据不全；
    - reading：连 ORACLE 也答错——问题主要在阅读与推理。
    没有跑 ORACLE 时，后两类合并为 hit_but_wrong。
    """
    oracle = {r["sample_id"]: r["correct"] for r in rows if r["method"] == "ORACLE"}
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        retrieval = r.get("retrieval")
        if r["correct"] or r["is_abstention"] or not retrieval:
            continue
        hit = retrieval.get("turn_recall_any", retrieval.get("session_recall_any", 0.0)) > 0
        if not hit:
            label = "retrieval_miss"
        elif r["sample_id"] not in oracle:
            label = "hit_but_wrong"
        else:
            label = "reading" if not oracle[r["sample_id"]] else "noisy_or_incomplete"
        out[r["method"]][label] += 1
    return {method: dict(counts) for method, counts in out.items()}


ATTRIBUTION_LABELS = {
    "retrieval_miss": "检索未命中：前 k 条里没有任何证据发言",
    "noisy_or_incomplete": "命中证据但答错，ORACLE 答对：干扰多或证据不全",
    "reading": "ORACLE 也答错：阅读与推理问题",
    "hit_but_wrong": "命中证据但答错（未跑 ORACLE，无法细分）",
}


def render_markdown(
    info: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    stats: Sequence[MethodStats],
    ingests: Sequence[dict[str, Any]],
    errors: Sequence[dict[str, Any]],
    *,
    has_plot: bool,
) -> str:
    cfg = info["config"]
    roles = info["roles"]
    lines = [
        f"# 评测报告：{cfg['name']}",
        "",
        f"- 数据集：`{cfg['dataset']}`"
        + (f"（划分 `{cfg['split']}`）" if cfg.get("split") else "")
        + f" · 题数 {info['samples']}"
        + (f"（只跑前 {cfg['limit']} 题）" if cfg.get("limit") else ""),
        f"- 回答模型：`{roles['reader']['model_id']}` · 判分模型：`{roles['judge']['model_id']}`"
        f" · 向量：`{info['embedding']['model']}`（{info['embedding']['dim']} 维）",
        f"- B2 检索条数 k = {cfg['k']} · 代码版本 `{info['git']}` · 首次运行 {info['created']}",
        "",
        "## 总体结果",
        "",
        "| 方法 | 题数 | 准确率 | 95% 置信区间 | 平均输入 tokens | 平均输出 tokens"
        " | 平均 LLM 调用 | 名义成本 / 题 | 延迟 p50 / p95 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in stats:
        lines.append(
            f"| {s.method} {_label_zh(s.method)} | {s.n} | **{s.accuracy:.1%}** "
            f"| {s.ci[0]:.1%} – {s.ci[1]:.1%} | {s.input_tokens:,.0f} | {s.output_tokens:,.0f} "
            f"| {s.llm_calls:.2f} | ${s.cost:.5f} | {_seconds(s.latency_p50)} / "
            f"{_seconds(s.latency_p95)} |"
        )
    if has_plot:
        lines += [
            "",
            "<picture>",
            '  <source media="(prefers-color-scheme: dark)" srcset="pareto-dark.png">',
            '  <img alt="各方法的准确率与每题名义成本（对数坐标）" src="pareto-light.png"'
            ' width="720">',
            "</picture>",
        ]
    if any(s.f1 is not None for s in stats):
        lines += ["", "token F1（LoCoMo 原论文口径）："]
        lines += [f"- {s.method}：{s.f1:.3f}" for s in stats if s.f1 is not None]

    types = sorted({t for s in stats for t in s.by_type}, key=_type_rank)
    if types:
        header = " | ".join(f"{s.method}" for s in stats)
        lines += ["", "## 分题型准确率", "", f"| 题型 | 题数 | {header} |"]
        lines.append("|---|---|" + "---|" * len(stats))
        for t in types:
            n = max(s.by_type.get(t, (0, 0.0))[0] for s in stats)
            cells = " | ".join(f"{s.by_type[t][1]:.0%}" if t in s.by_type else "—" for s in stats)
            lines.append(f"| {TYPE_LABELS.get(t, t)} | {n} | {cells} |")
        if any(s.abstention for s in stats):
            n = max((s.abstention or (0, 0.0))[0] for s in stats)
            cells = " | ".join(f"{s.abstention[1]:.0%}" if s.abstention else "—" for s in stats)
            lines.append(f'| 拒答题（应回答"不知道"） | {n} | {cells} |')

    with_retrieval = [s for s in stats if s.retrieval]
    if with_retrieval:
        lines += ["", f"## 检索质量（前 k = {cfg['k']} 条，不含拒答题）", ""]
        lines.append("| 指标 | " + " | ".join(s.method for s in with_retrieval) + " | 含义 |")
        lines.append("|---|" + "---|" * len(with_retrieval) + "---|")
        for key, (name, meaning) in RETRIEVAL_LABELS.items():
            if any(key in s.retrieval for s in with_retrieval):
                cells = " | ".join(
                    f"{s.retrieval[key]:.1%}" if key in s.retrieval else "—" for s in with_retrieval
                )
                lines.append(f"| {name} | {cells} | {meaning} |")

    attribution = error_attribution(rows)
    if attribution:
        lines += ["", "## 错误归因（检索类方法答错的题，不含拒答题）", ""]
        lines += ["| 原因 | " + " | ".join(attribution) + " |", "|---|" + "---|" * len(attribution)]
        for key, label in ATTRIBUTION_LABELS.items():
            if any(key in counts for counts in attribution.values()):
                cells = " | ".join(str(counts.get(key, 0)) for counts in attribution.values())
                lines.append(f"| {label} | {cells} |")

    ingest_cost = sum(r["cost"] for r in ingests)
    histories = len({r["space_key"] for r in ingests})  # 续跑可能重复写入同一份历史
    ingest_tokens = sum(r["embedding_tokens"] for r in ingests)
    answer_cost = sum(s.total_cost for s in stats)
    judge_cost = sum(s.judge_cost for s in stats)
    lines += [
        "",
        "## 成本汇总（名义成本，美元）",
        "",
        "| 项目 | 名义成本 | 说明 |",
        "|---|---|---|",
    ]
    for s in stats:
        lines.append(f"| 回答 · {s.method} | ${s.total_cost:.4f} | {s.n} 题 |")
    lines += [
        f"| 写入（向量化） | ${ingest_cost:.4f} | {histories} 份对话历史，"
        f"{ingest_tokens:,} tokens；只有 B2 需要，一次写入可服务多次提问 |",
        f"| 判分 | ${judge_cost:.4f} | 评测本身的开销，不计入方法成本 |",
        f"| **合计** | **${answer_cost + ingest_cost + judge_cost:.4f}** | |",
        "",
        "## 说明",
        "",
        "- 准确率由 LLM 判分得出，使用 LongMemEval 官方模板（逐字照搬）；"
        "判分模型与回答模型来自不同家族（D-008）。",
        "- 名义成本按高峰时段、未命中缓存的标价计算，便于不同时间的运行相互比较（D-010）；"
        "实际花费因缓存、错峰折扣会更低。",
        '- "名义成本 / 题"只算回答本身；写入成本是一次性的，单独列出。',
        "- 置信区间为 bootstrap 95%（2000 次重采样，固定种子）。",
    ]
    if errors:
        lines.append(
            f"- 有 {len(errors)} 个组合出错未计入结果，详见 `errors.jsonl`；重新运行会自动重试。"
        )
    return "\n".join(lines) + "\n"


# ---------- 图 ----------


def plot_pareto(
    stats: Sequence[MethodStats], path: Path, *, mode: str, info: dict[str, Any]
) -> None:
    """准确率–每题名义成本散点图（x 为对数坐标），带 95% 置信区间竖线。"""
    theme = THEMES[mode]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)

    points = [s for s in stats if s.cost > 0]
    for s in points:
        x, y = s.cost, s.accuracy * 100
        color = theme.series.get(s.method)
        hollow = color is None  # ORACLE 等参照点：中性色空心点
        line_color = theme.ink2 if color is None else color
        ax.vlines(
            x, s.ci[0] * 100, s.ci[1] * 100, color=line_color, linewidth=0.9, alpha=0.6, zorder=2
        )
        ax.plot(
            x,
            y,
            marker="o",
            markersize=8,
            markerfacecolor=theme.surface if hollow else line_color,
            markeredgecolor=line_color
            if hollow
            else theme.surface,  # 实心点外圈是 2px 背景色描边环
            markeredgewidth=1.6 if hollow else 1.0,
            linestyle="none",
            zorder=3,
            label=_label_en(s.method),
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    if points:
        xs = [s.cost for s in points]
        ax.set_xlim(min(xs) / 3, max(xs) * 12)  # 右侧多留空间，让最贵那个点的标签能放在右边
    ax.set_ylim(-4, 104)  # 上下留白，0% 与 100% 的点不会压在坐标轴上
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color=theme.grid, linewidth=0.45)
    ax.grid(axis="x", which="major", color=theme.grid, linewidth=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(0.45)
    ax.tick_params(colors=theme.muted, labelsize=8.5, which="both")
    ax.set_xlabel("Nominal cost per question (USD, log scale)", color=theme.ink2, fontsize=9)
    ax.set_ylabel("Accuracy (judged)", color=theme.ink2, fontsize=9)
    cfg = info["config"]
    subtitle = f"{cfg['dataset']}" + (f" · {cfg['split']}" if cfg.get("split") else "")
    subtitle += f" · {max(s.n for s in stats)} questions · bars: 95% CI"
    ax.set_title("Accuracy vs. cost per question", loc="left", color=theme.ink, fontsize=11, pad=20)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=theme.muted, fontsize=8.5)
    legend = ax.legend(
        loc="lower right", frameon=False, fontsize=8.5, labelcolor=theme.ink2, handletextpad=0.4
    )
    fig.tight_layout()
    _place_labels(fig, ax, points, legend, theme)
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)


def _place_labels(
    fig: Figure, ax: Axes, points: Sequence[MethodStats], legend: Legend, theme: Theme
) -> None:
    """直接标签的防碰撞摆放：依次尝试右上、右下、左上、左下，避开点、区间竖线、图例与已放的标签。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    to_px = ax.transData.transform
    occupied: list[Bbox] = [legend.get_window_extent(renderer)]
    for s in points:
        cx, cy = to_px((s.cost, s.accuracy * 100))
        occupied.append(Bbox.from_extents(cx - 9, cy - 9, cx + 9, cy + 9))
        _, low = to_px((s.cost, s.ci[0] * 100))
        _, high = to_px((s.cost, s.ci[1] * 100))
        occupied.append(Bbox.from_extents(cx - 6, low, cx + 6, high))  # 竖线两侧留出间距
    area = ax.get_window_extent(renderer)
    candidates = [
        (8, 5, "left", "bottom"),
        (8, -5, "left", "top"),
        (-8, 5, "right", "bottom"),
        (-8, -5, "right", "top"),
    ]
    for s in sorted(points, key=lambda p: p.cost):
        anchor = (s.cost, s.accuracy * 100)
        own = Bbox.from_extents(*(to_px(anchor) - 9), *(to_px(anchor) + 9))
        others = [box for box in occupied if not _same(box, own)]
        text = None
        for dx, dy, ha, va in candidates:
            if text is not None:
                text.remove()
            text = ax.annotate(
                f"{_label_en(s.method)}  {s.accuracy:.0%}",
                anchor,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                color=theme.ink2,
                fontsize=8.5,
                zorder=4,
            )
            box = text.get_window_extent(renderer).padded(3)
            inside = (
                area.x0 <= box.x0 and box.x1 <= area.x1 and area.y0 <= box.y0 and box.y1 <= area.y1
            )
            if inside and not any(box.overlaps(o) for o in others):
                break
        if text is not None:
            occupied.append(text.get_window_extent(renderer))


def _same(a: Bbox, b: Bbox) -> bool:
    return bool(abs(a.x0 - b.x0) < 0.5 and abs(a.y0 - b.y0) < 0.5)


# ---------- 辅助 ----------


def _method_rank(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def _type_rank(question_type: str) -> int:
    order = list(TYPE_LABELS)
    return order.index(question_type) if question_type in order else len(order)


def _label_zh(method: str) -> str:
    return METHOD_LABELS.get(method, (method, method))[0]


def _label_en(method: str) -> str:
    return METHOD_LABELS.get(method, (method, method))[1]


def _seconds(ms: float) -> str:
    return "—" if ms != ms else f"{ms / 1000:.1f}s"  # NaN 检查
