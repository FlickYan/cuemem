"""评测报告：把 results.jsonl 汇总成 Markdown 表格，并画"准确率–成本"散点图。

图按可视化规范绘制：系列颜色按方法固定分配（不随排名变化），散点图最多 3 个系列色
（蓝 / 青 / 黄，已用 validate_palette.js 在 --pairs all 下验证深浅两套）。N4 及之前的运行
分给 B2、B2F、B4；有 L2 的运行（N5 起）只画当前的三条主线：L1（黄，与 N4 的 B4 同色）、
L2（蓝）、规则版控制器（青），B2、B2F 只进表格。B0（下限）、B1（全上下文）、ORACLE（上限）
与 B3（模型自主检索）是参照点，用中性色空心标记、形状区分；消融变体只出现在表格里。
浅色、深色各渲染一张，Markdown 里用 <picture> 按读者的主题自动切换。图中文字用英文，
避免 CI 环境缺少中文字体。
"""

from __future__ import annotations

import json
import re
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
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from matplotlib.transforms import Bbox

from cuemem.evals.metrics import bootstrap_ci, mean, percentile

METHOD_LABELS = {
    "B0": ("无记忆", "B0 No memory"),
    "B1": ("全上下文", "B1 Full context"),
    "B2": ("朴素检索", "B2 Naive RAG"),
    "B2F": ("朴素检索 + 事实记忆", "B2F Naive RAG + facts"),
    "B2F-keys": ("消融：只用事实增强键", "B2F keys only"),
    "B2F-current": ("消融：只给当前事实（覆盖式更新）", "B2F current facts only"),
    "B2F-plain": ("消融：事实不标有效期", "B2F no validity labels"),
    "B2F-noprofile": ("消融：不给核心记忆", "B2F no core memory"),
    "B4": ("L1 快速召回", "B4 L1 fast recall"),
    "B4-k10": ("消融：固定取 10 轮", "B4 top-10 rounds"),
    "B4-k5": ("消融：固定取 5 轮", "B4 top-5 rounds"),
    "B4-b2k": ("消融：预算 2k tokens", "B4 2k-token budget"),
    "B4-notime": ("消融：不做时间加分", "B4 no time boost"),
    "B4-norel": ("消融：不标相对时间", "B4 no relative time"),
    "B4-nobm25": ("消融：去掉 BM25（N5 起的 L1 默认）", "L1 (no BM25)"),
    "B4-b6k": ("对照：L1 预算 6k tokens（与 L2 相同）", "L1 6k budget"),
    "B4-note": ("对照：L1 + 先记笔记再回答", "L1 + notes"),
    "B5": ("L2 深度检索", "B5 L2 deep retrieval"),
    "B5-noplan": ("消融：不做查询规划", "B5 no planning"),
    "B5-noiter": ("消融：不做充分性检查与补检索", "B5 no follow-up"),
    "B5-nonote": ("消融：不记笔记", "B5 no notes"),
    "B5-ppr": ("变体：加图扩散（PPR）", "B5 + PPR"),
    "B6r": ("规则版控制器（L1 / L2）", "B6r rule router"),
    "B3": ("模型自主检索（ReAct）", "B3 model-driven search"),
    "ORACLE": ("只给证据会话（上限参照）", "Oracle: evidence only"),
}
METHOD_ORDER = list(METHOD_LABELS)
REFERENCE_MARKERS = {"B0": "s", "B1": "D", "ORACLE": "o", "B3": "^"}  # 参照点：中性色空心

# 每类运行画哪些方法、谁用哪个系列色（槽位：蓝、青、黄）；消融变体只进表格，不进图
SERIES_SLOTS = ("blue", "aqua", "yellow")
PROFILES = {
    "n4": {"B2F": "blue", "B2": "aqua", "B4": "yellow"},
    "n5": {"B5": "blue", "B6r": "aqua", "B4-nobm25": "yellow"},
}

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
    "session_recall_any": ("会话召回（任一）", "检索结果里至少命中一个证据会话的比例"),
    "session_recall_all": ("会话召回（全部）", "证据会话被检索结果覆盖的平均比例"),
    "turn_recall_any": ("发言召回（任一）", "检索结果里至少有一条包含证据发言的比例"),
    "turn_recall_all": ("发言召回（全部）", "包含证据发言的轮被检索结果覆盖的平均比例"),
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
    series: dict[str, str]  # 系列槽位 → 颜色


THEMES = {
    "light": Theme(
        surface="#fcfcfb",
        ink="#0b0b0b",
        ink2="#52514e",
        muted="#898781",
        grid="#e1e0d9",
        axis="#c3c2b7",
        series={"blue": "#2a78d6", "aqua": "#1baf7a", "yellow": "#eda100"},
    ),
    "dark": Theme(
        surface="#1a1a19",
        ink="#ffffff",
        ink2="#c3c2b7",
        muted="#898781",
        grid="#2c2c2a",
        axis="#383835",
        series={"blue": "#3987e5", "aqua": "#199e70", "yellow": "#c98500"},
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
    rounds: float | None  # 平均交给回答模型的轮数（与检索指标同口径：不含拒答题）
    uses_facts: bool
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
        # 按题目 ID 排序：结果按完成顺序写入，排序后 bootstrap 的重采样才与运行顺序无关
        items = sorted(by_method[method], key=lambda r: r["sample_id"])
        correct = [1.0 if r["correct"] else 0.0 for r in items]
        types: dict[str, list[float]] = defaultdict(list)
        for r, c in zip(items, correct, strict=True):
            if not r["is_abstention"]:
                types[r["question_type"]].append(c)
        abstained = [c for r, c in zip(items, correct, strict=True) if r["is_abstention"]]
        retrieval: dict[str, list[float]] = defaultdict(list)
        rounds = []
        for r in items:
            for key, value in (r.get("retrieval") or {}).items():
                retrieval[key].append(value)
            if r.get("retrieval"):
                rounds.append(len(r.get("retrieved") or []))
        # 延迟只统计真的调用了模型的题：命中本地缓存的回答没有意义上的延迟
        latencies = [
            r["usage"]["latency_ms"]
            for r in items
            if r["usage"]["latency_ms"] and r["usage"]["llm_cache_hits"] < r["usage"]["llm_calls"]
        ]
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
                rounds=mean(rounds) if rounds else None,
                uses_facts=any(r.get("retrieved_facts") for r in items),
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
    - retrieval_miss：交给回答模型的轮里没有任何证据发言；
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


# 粗略识别"不知道"式回答：有答案的题答错时，区分"答成了别的"与"说不知道"（误拒答）
DONT_KNOW = re.compile(
    r"\b(i don't know|i do not know|i don't have (any )?(information|record|details)"
    r"|i can't (tell|determine|give|say)|i cannot (tell|determine|give|say)"
    r"|no (information|record|mention) (of|about)|(isn't|is not|aren't|are not) mentioned"
    r"|not enough information)\b|不知道|没有(相关)?(的)?(信息|记录)",
    re.IGNORECASE,
)


def false_abstentions(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """每种方法在有答案的题上答错、且回答像"不知道"的题数（误拒答，按措辞粗略识别）。"""
    out: dict[str, int] = defaultdict(int)
    for r in rows:
        if not r["correct"] and not r["is_abstention"] and DONT_KNOW.search(r["response"] or ""):
            out[r["method"]] += 1
    return dict(out)


ATTRIBUTION_LABELS = {
    "retrieval_miss": "检索未命中：交给回答模型的轮里没有任何证据发言",
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
        lines += [
            "",
            "## 检索质量（不含拒答题）",
            "",
            "检索结果指交给回答模型的全部轮：固定条数的方法取前 k 轮，按 token 预算装入的方法"
            "（B4、B5 系列）轮数随每轮的长短变化，B5 的补检索还会再加几轮。"
            "轮数不同的方法，召回不能直接比较。",
            "",
        ]
        lines.append("| 指标 | " + " | ".join(s.method for s in with_retrieval) + " | 含义 |")
        lines.append("|---|" + "---|" * len(with_retrieval) + "---|")
        if all(s.rounds is not None for s in with_retrieval):
            cells = " | ".join(f"{s.rounds:.1f}" for s in with_retrieval)
            lines.append(f"| 平均轮数 | {cells} | 每题交给回答模型的轮数 |")
        for key, (name, meaning) in RETRIEVAL_LABELS.items():
            if any(key in s.retrieval for s in with_retrieval):
                cells = " | ".join(
                    f"{s.retrieval[key]:.1%}" if key in s.retrieval else "—" for s in with_retrieval
                )
                lines.append(f"| {name} | {cells} | {meaning} |")

    found = error_attribution(rows)
    attribution = {s.method: found[s.method] for s in stats if s.method in found}  # 按方法顺序
    if attribution:
        lines += ["", "## 错误归因（检索类方法答错的题，不含拒答题）", ""]
        lines += ["| 原因 | " + " | ".join(attribution) + " |", "|---|" + "---|" * len(attribution)]
        for key, label in ATTRIBUTION_LABELS.items():
            if any(key in counts for counts in attribution.values()):
                cells = " | ".join(str(counts.get(key, 0)) for counts in attribution.values())
                lines.append(f"| {label} | {cells} |")
        refusals = false_abstentions(rows)
        cells = " | ".join(str(refusals.get(method, 0)) for method in attribution)
        lines.append(f'| 其中回答"不知道"（误拒答，按措辞粗略识别） | {cells} |')

    lines += _l2_section(rows, stats)

    online = _latest_per_space([r for r in ingests if r.get("stage", "ingest") == "ingest"])
    consolidated = _latest_per_space([r for r in ingests if r.get("stage") == "consolidate"])
    if consolidated:
        lines += ["", *_consolidation_section(consolidated)]
    ingest_cost = sum(r["cost"] for r in online)
    ingest_tokens = sum(r["embedding_tokens"] for r in online)
    consolidate_cost = sum(r["cost"] for r in consolidated)
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
    lines.append(
        f"| 写入（向量化） | ${ingest_cost:.4f} | {len(online)} 份对话历史，"
        f"{ingest_tokens:,} tokens；检索类方法需要，一次写入可服务多次提问 |"
    )
    if consolidated:
        families = "、".join(dict.fromkeys(s.method.split("-")[0] for s in stats if s.uses_facts))
        lines.append(
            f"| 巩固（抽取、对账、摘要） | ${consolidate_cost:.4f} | "
            f"{len(consolidated)} 份对话历史；只有 {families or '用到事实的方法'} 系列需要，"
            "同样是一次性成本 |"
        )
    total = answer_cost + ingest_cost + consolidate_cost + judge_cost
    lines += [
        f"| 判分 | ${judge_cost:.4f} | 评测本身的开销，不计入方法成本 |",
        f"| **合计** | **${total:.4f}** | |",
        "",
        "## 说明",
        "",
        "- 准确率由 LLM 判分得出，使用 LongMemEval 官方模板（逐字照搬）；"
        "判分模型与回答模型来自不同家族（D-008）。",
        "- 名义成本按高峰时段、未命中缓存的标价计算，便于不同时间的运行相互比较（D-010）；"
        "实际花费因缓存、错峰折扣会更低。",
        '- "名义成本 / 题"只算回答本身；写入与巩固成本是一次性的，单独列出。',
        "- 置信区间为 bootstrap 95%（2000 次重采样，固定种子）。",
        "- 延迟只统计实际调用了模型的题；全部命中本地缓存的方法显示为 —。",
    ]
    if errors:
        lines.append(
            f"- 有 {len(errors)} 个组合出错未计入结果，详见 `errors.jsonl`；重新运行会自动重试。"
        )
    return "\n".join(lines) + "\n"


def _l2_section(rows: Sequence[dict[str, Any]], stats: Sequence[MethodStats]) -> list[str]:
    """L2 的检索过程（N5）：查询规划、充分性检查与补检索；规则版控制器走 L2 的比例。"""
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    routed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        details = row.get("details") or {}
        if details.get("plan"):
            by_method[row["method"]].append(row)
        if details.get("route"):
            routed[row["method"]].append(row)
    if not by_method and not routed:
        return []
    lines = ["", "## L2 检索过程", ""]
    methods = [s.method for s in stats if s.method in by_method]
    if methods:
        lines += ["| 项目 | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
        for name, cells in _l2_rows([by_method[m] for m in methods]):
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for method in (s.method for s in stats if s.method in routed):
        items = routed[method]
        to_l2 = [r for r in items if r["details"]["route"]["level"] == "L2"]
        reasons: dict[str, int] = defaultdict(int)
        for r in to_l2:
            for reason in r["details"]["route"]["reasons"]:
                reasons[reason] += 1
        listed = "、".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        lines += [
            "",
            f"{method}：{len(to_l2)} / {len(items)} 题走 L2（{len(to_l2) / len(items):.0%}）"
            + (f"；触发的线索：{listed}" if listed else "")
            + "。",
        ]
    return lines


def _l2_rows(groups: Sequence[list[dict[str, Any]]]) -> list[tuple[str, list[str]]]:
    def share(test: Any) -> list[str]:
        return [f"{sum(1 for r in g if test(r['details'])) / len(g):.0%}" for g in groups]

    def average(value: Any, fmt: str = ".2f") -> list[str]:
        return [format(mean([value(r) for r in g]), fmt) for g in groups]

    return [
        ("走 L2 的题数", [str(len(g)) for g in groups]),
        ("查询规划由 LLM 完成", share(lambda d: d["plan"]["source"] == "llm")),
        ("平均子查询数", average(lambda r: len(r["details"]["plan"]["sub_queries"]))),
        ("判为需要穷尽（计数 / 求和 / 列举）", share(lambda d: d["plan"]["exhaustive"])),
        ("带时间区间", share(lambda d: d["plan"]["time_range"])),
        ("平均充分性检查次数", average(lambda r: len(r["details"]["checks"]))),
        ("至少补检索一次", share(lambda d: any(c.get("added_rounds") for c in d["checks"]))),
        (
            "平均证据 tokens（轮）",
            average(lambda r: r["details"].get("evidence_tokens", 0), ",.0f"),
        ),
        (
            "每题规划与检查的调用次数",
            average(lambda r: r["usage"].get("llm_calls_by_role", {}).get("planner", 0)),
        ),
    ]


def _latest_per_space(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """续跑可能对同一份历史重复写入：每份历史只取最后一次。"""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[row["space_key"]] = row
    return list(latest.values())


def _consolidation_section(rows: Sequence[dict[str, Any]]) -> list[str]:
    """写入路径（N3）的统计：抽取了多少事实、对账动作的分布、写入成本（每千条发言）。"""
    sessions = sum(r["sessions"] for r in rows)
    turns = sum(r["turns"] for r in rows)
    facts = sum(r["facts_extracted"] for r in rows)
    actions = {k: sum(r[k] for r in rows) for k in ("added", "superseded", "merged", "noop")}
    total_actions = max(1, sum(actions.values()))
    input_tokens = sum(r["input_tokens"] for r in rows)
    output_tokens = sum(r["output_tokens"] for r in rows)
    cost = sum(r["cost"] for r in rows)
    calls = sum(r["llm_calls"] for r in rows)
    rules: dict[str, int] = defaultdict(int)
    for r in rows:
        for key, value in r.get("rules", {}).items():
            rules[key] += value
    llm_decided = sum(v for k, v in rules.items() if k.endswith(":llm"))
    per_k = 1000 / max(1, turns)
    current = sum(r["facts_current"] for r in rows)
    total = sum(r["facts_total"] for r in rows)
    elapsed = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    share = " / ".join(f"{actions[k] / total_actions:.1%}" for k in actions)
    return [
        "## 写入路径：巩固（N3）",
        "",
        "| 项目 | 数值 | 说明 |",
        "|---|---|---|",
        f"| 对话历史 / 会话 / 发言 | {len(rows)} / {sessions:,} / {turns:,} | |",
        f"| 抽取出的事实 | {facts:,}（每会话 {facts / max(1, sessions):.1f} 条） | |",
        f"| 对账：新增 / 取代 / 合并 / 重放 | {share} | 取代 {actions['superseded']:,} 次 |",
        f"| 由 LLM 裁决的对账 | {llm_decided:,}（占 {llm_decided / total_actions:.1%}） | "
        "其余由规则直接判定 |",
        f"| 事实：当前有效 / 全部 | {current:,} / {total:,} | 被取代的旧事实保留为历史 |",
        f"| 实体 | {sum(r['entities'] for r in rows):,} | |",
        f"| LLM 调用（抽取 + 对账 + 消歧） | {calls:,} | 含命中缓存的调用 |",
        f"| 每千条发言的 token（输入 / 输出） | {input_tokens * per_k:,.0f} / "
        f"{output_tokens * per_k:,.0f} | 抽取时助手回复只保留开头 |",
        f"| 每千条发言的名义成本 | ${cost * per_k:.4f} | 含向量化 |",
        f"| 每份历史的巩固耗时 p50 / p95 | {_seconds(percentile(elapsed, 50))} / "
        f"{_seconds(percentile(elapsed, 95))} | 受供应商并发上限影响 |",
    ]


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

    profile = chart_profile(stats)
    plotted = set(profile) | set(REFERENCE_MARKERS)
    points = [s for s in stats if s.cost > 0 and s.method in plotted]
    for s in points:
        x, y = s.cost, s.accuracy * 100
        slot = profile.get(s.method)
        color = theme.series[slot] if slot else None
        hollow = color is None  # B0、ORACLE 等参照点：中性色空心点
        line_color = theme.ink2 if color is None else color
        ax.vlines(
            x, s.ci[0] * 100, s.ci[1] * 100, color=line_color, linewidth=0.9, alpha=0.6, zorder=2
        )
        ax.plot(
            x,
            y,
            marker=REFERENCE_MARKERS.get(s.method, "o"),
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
    if points:
        xs = [s.cost for s in points]
        ax.set_xlim(min(xs) / 3, max(xs) * 12)  # 右侧多留空间，让最贵那个点的标签能放在右边
    low, high = ax.get_xlim()
    # 跨度不到两个数量级时按 1-2-5 取主刻度，否则只取 10 的整数次幂；次刻度不标数字，
    # 避免出现与美元格式不一致的科学计数法标签
    subs = (1.0, 2.0, 5.0) if high / low < 100 else (1.0,)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
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
    """直接标签的防碰撞摆放：先试紧贴点的四个方位（右上、右下、左上、左下），都放不下时
    逐步拉远并画一条细引线连回自己的点；避开点、区间竖线、图例与已放的标签。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
    to_px = ax.transData.transform
    # 每个障碍物记下属于哪个方法：摆某个点的标签时，它自己的点和区间竖线不算障碍
    occupied: list[tuple[str, Bbox]] = [("", legend.get_window_extent(renderer))]
    for s in points:
        cx, cy = to_px((s.cost, s.accuracy * 100))
        occupied.append((s.method, Bbox.from_extents(cx - 9, cy - 9, cx + 9, cy + 9)))
        _, low = to_px((s.cost, s.ci[0] * 100))
        _, high = to_px((s.cost, s.ci[1] * 100))
        occupied.append((s.method, Bbox.from_extents(cx - 6, low, cx + 6, high)))
    area = ax.get_window_extent(renderer)
    candidates: list[tuple[float, float, str, str]] = []
    offsets = [(8.0, 5.0), (8.0, -5.0), (-8.0, 5.0), (-8.0, -5.0)] + [
        (sx * 14.0, sy * d) for d in (22.0, 36.0, 50.0) for sx in (1, -1) for sy in (1, -1)
    ]
    for ox, oy in offsets:
        candidates.append((ox, oy, "left" if ox > 0 else "right", "bottom" if oy > 0 else "top"))
    centers = {s.method: to_px((s.cost, s.accuracy * 100)) for s in points}
    for s in sorted(points, key=lambda p: p.cost):
        anchor = (s.cost, s.accuracy * 100)
        others = [box for owner, box in occupied if owner != s.method]
        neighbours = [c for m, c in centers.items() if m != s.method]
        best: tuple[float, tuple[float, float, str, str]] | None = None
        chosen = None
        for candidate in candidates:
            text = _draw_label(ax, s, anchor, candidate, theme)
            box = text.get_window_extent(renderer).padded(3)
            inside = (
                area.x0 <= box.x0 and box.x1 <= area.x1 and area.y0 <= box.y0 and box.y1 <= area.y1
            )
            # 紧贴摆放时，标签必须离自己的点最近，否则读者会把标签对到另一个点上；有引线时不需要
            mine = _distance(box, centers[s.method])
            leader = abs(candidate[1]) > 5
            ambiguous = not leader and any(_distance(box, c) < mine + 6 for c in neighbours)
            # 代价：越界最重，其次是贴得离别的点更近（会对错号），再次是与其他元素的重叠面积
            clashes = (
                sum(_overlap(box, o) for o in others)
                + (0 if inside else 1e6)
                + (1e5 if ambiguous else 0)
            )
            text.remove()
            if clashes == 0:
                chosen = candidate
                break
            if best is None or clashes < best[0]:
                best = (clashes, candidate)
        if chosen is None and best is not None:  # 都放不下：选重叠面积最小的位置
            chosen = best[1]
        if chosen is not None:
            text = _draw_label(ax, s, anchor, chosen, theme)
            occupied.append((s.method, text.get_window_extent(renderer)))


def _draw_label(
    ax: Axes,
    s: MethodStats,
    anchor: tuple[float, float],
    candidate: tuple[float, float, str, str],
    theme: Theme,
) -> Any:
    dx, dy, ha, va = candidate
    leader = abs(dy) > 5  # 离点较远时画引线，读者能对上是哪个点
    return ax.annotate(
        f"{_label_en(s.method)}  {s.accuracy:.0%}",
        anchor,
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        color=theme.ink2,
        fontsize=8.5,
        zorder=4,
        arrowprops=(
            {"arrowstyle": "-", "color": theme.muted, "lw": 0.6, "shrinkB": 5} if leader else None
        ),
    )


def _overlap(a: Bbox, b: Bbox) -> float:
    width = min(a.x1, b.x1) - max(a.x0, b.x0)
    height = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(0.0, width) * max(0.0, height)


def _distance(box: Bbox, point: Any) -> float:
    """点到矩形的距离（像素），点在矩形内时为 0。"""
    dx = max(box.x0 - point[0], 0.0, point[0] - box.x1)
    dy = max(box.y0 - point[1], 0.0, point[1] - box.y1)
    return float((dx * dx + dy * dy) ** 0.5)


# ---------- 辅助 ----------


def chart_profile(stats: Sequence[MethodStats]) -> dict[str, str]:
    """有 L2（B5）的运行画 N5 的三条主线，否则沿用 N4 的配色。"""
    return PROFILES["n5" if any(s.method == "B5" for s in stats) else "n4"]


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
