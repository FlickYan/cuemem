"""N9：测试集上的最终评测，汇总成 docs/results/n9-final/results.json 与图。

只读评测运行的结果（runs/eval/<运行名>/），不调用 DeepSeek；混合流上零样本 LLM 路由（B6z、B6h）
的路由结果用通义，缓存在 runs/eval/n9-final/。缺哪个运行就跳过那一部分。

1. 主表：四个测试集上各方法的准确率（95% 置信区间）、每题名义成本、输入 tokens、分题型；
2. 冻结的控制器：N7 导出的模型直接在测试集上选档（与 B6 的真实运行逐题核对），λ 扫描的曲线，
   与固定档位、规则、模型自主检索、"控制器先选、模型兜底"的配对比较；
3. 混合流（测试集版，scripts/n9_mixed.py）：20% 记忆题时的成本、记忆题准确率与误召回；
4. L2 对 L1 的分题型配对检验；LoCoMo 上的图扩散；写入路径与有效期（H4）；
5. 换回答模型（H3）：test100r / test200r 上各回答模型 × 各方法；
6. 错误分析：主方法答错的题分类。

最后由 scripts/n9_tables.py 把结果渲染成 docs/results/n9-final/tables.md。

用法：
    uv run python scripts/n9_report.py
"""

from __future__ import annotations

import json
import sys
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullFormatter

from cuemem.config import PROJECT_ROOT
from cuemem.evals import routing as R
from cuemem.evals.report import (
    THEMES,
    error_attribution,
    false_abstentions,
    load_rows,
    method_stats,
)
from cuemem.evals.splits import load_split
from cuemem.read.controller import HYBRID_BAND, HYBRID_CONFIDENCE, V1_MODEL, LearnedController

sys.path.insert(0, str(Path(__file__).parent))
import n7_controller as N7
import n9_errors as E
import n9_tables

warnings.filterwarnings("ignore", category=UserWarning)

RUNS_DIR = PROJECT_ROOT / "runs" / "eval"
OUT = PROJECT_ROOT / "docs" / "results" / "n9-final"
CACHE = RUNS_DIR / "n9-final"
TEST_RUNS = {
    "longmemeval_s": "test400-n9",
    "locomo": "locomo-test7-n9",
    "memlens_32k": "memlens32-test-n9",
    "memlens_128k": "memlens128-test-n9",
}
DATASET_LABELS = {
    "longmemeval_s": "LongMemEval test400",
    "locomo": "LoCoMo 后 7 段对话（1,155 题）",
    "memlens_32k": "MemLens 32K（官方 195 题）",
    "memlens_128k": "MemLens 128K（官方 195 题）",
}
DATASET_EN = {
    "longmemeval_s": "LongMemEval test400",
    "locomo": "LoCoMo conversations 4-10",
    "memlens_32k": "MemLens 32K (195)",
    "memlens_128k": "MemLens 128K (195)",
}
MIXED_RUN = "mixed-n9"
MIX_RATIO = 0.2
# 换回答模型（H3）：主回答模型的结果取全量运行里子集的那部分
READER_SPLITS = {"longmemeval_s": "test100r", "locomo": "test200r"}
READERS = {
    "deepseek-flash": {"longmemeval_s": "test400-n9", "locomo": "locomo-test7-n9"},
    "qwen3.8-flash": {"longmemeval_s": "test100r-qflash", "locomo": "locomo-test200r-qflash"},
    "qwen3.6-35b-a3b": {"longmemeval_s": "test100r-qsmall", "locomo": "locomo-test200r-qsmall"},
    "deepseek-v4-pro": {"longmemeval_s": "test100r-dspro", "locomo": "locomo-test200r-dspro"},
}
READER_METHODS = ("L0", "B4-nobm25", "B5", "B6r", "B6", "B3", "B6a")


def available(name: str) -> bool:
    return (RUNS_DIR / name / "results.jsonl").exists()


def pct(x: float) -> float:
    return round(100 * x, 1)


# ---------- 1. 主表 ----------


def main_table(run: str, keep: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
    rows = load_rows(RUNS_DIR / run, "results.jsonl")
    if keep is not None:
        rows = [r for r in rows if keep(r)]
    stats = method_stats(rows)
    table: dict[str, Any] = {}
    for s in stats:
        entry = asdict(s)
        entry["ci"] = list(s.ci)
        table[s.method] = entry
    return table


# ---------- 2. 冻结的控制器 ----------


def frozen(
    items: Sequence[R.Item], ctrl: LearnedController, lam: float, name: str
) -> R.PolicyResult:
    """N7 导出的控制器直接选档（与运行时 ask(mode="auto") 相同的计算）。"""
    choices = []
    for it in items:
        d = ctrl.decide(it.features, has_images=it.has_images, lam=lam)
        choices.append(R.take(it, it.variant(d.level, d.see)))
    return R.PolicyResult(name, choices, lam)


def hybrid(
    items: Sequence[R.Item],
    ctrl: LearnedController,
    routes: dict[str, dict[str, Any]],
    lam: float,
) -> tuple[R.PolicyResult, list[bool]]:
    """B6h：门控拿不准（P(none) 在区间内）时问一次 LLM，用它的判断代替门控。"""
    choices, asked = [], []
    for it in items:
        gate = ctrl.gate(it.features)
        p_none = None
        ask = HYBRID_BAND[0] < gate < HYBRID_BAND[1]
        if ask:
            llm_l0 = routes[it.sample_id]["level"] == "L0"
            p_none = HYBRID_CONFIDENCE if llm_l0 else 1 - HYBRID_CONFIDENCE
        d = ctrl.decide(it.features, has_images=it.has_images, lam=lam, p_none=p_none)
        choice = R.take(it, it.variant(d.level, d.see))
        if ask:
            choice.cost += routes[it.sample_id]["cost"]
        choices.append(choice)
        asked.append(ask)
    return R.PolicyResult("B6h", choices, lam), asked


def policy_row(p: R.PolicyResult) -> dict[str, float]:
    return {
        "accuracy": pct(p.accuracy),
        "cost_per_1k": round(1000 * p.cost, 3),
        "l0": pct(p.share("L0")),
        "l2": pct(p.share("L2")),
        "see": pct(float(np.mean([c.action.endswith(R.SEE) for c in p.choices]))),
        "n": len(p.choices),
    }


def versus_l2(items: Sequence[R.Item], policy: R.PolicyResult) -> dict[str, int]:
    """与"总是 L2"逐题比：赢、输的题按策略选的档分开数（输在 L0 = 没检索，输在 L1 = 没升级）。"""
    out: Counter[str] = Counter()
    for c, it in zip(policy.choices, items, strict=True):
        deep = it.actions[it.variant("L2")].correct
        if deep and not c.correct:
            out[f"lost_at_{R.level_of(c.action)}"] += 1
        elif c.correct and not deep:
            out[f"won_at_{R.level_of(c.action)}"] += 1
    return dict(sorted(out.items()))


def controller_section(items: Sequence[R.Item], ctrl: LearnedController) -> dict[str, Any]:
    lam = ctrl.lam
    b6 = frozen(items, ctrl, lam, "B6")
    # MemLens 上的固定档位按 N6 的默认做法带再感知；不带原图的版本只作参照
    policies: dict[str, R.PolicyResult] = {
        "always-L0": R.fixed(items, "L0"),
        "always-L1": R.fixed(items, "L1"),
        "always-L2": R.fixed(items, "L2"),
        "B6r": R.rule(items),
        "B6 (frozen)": b6,
        "type-oracle": R.type_oracle(items),
        "oracle": R.oracle(items),
    }
    if any(it.has_images for it in items):
        policies["always-L1 (no photos)"] = R.fixed(items, "L1", see=False)
        policies["always-L2 (no photos)"] = R.fixed(items, "L2", see=False)
    for ref in ("B3", "B6a", "B6h", "B6z", "B4-note", "B1", "B0", "ORACLE"):
        got = R.reference(items, ref)
        if got is not None:
            policies[ref] = got
    # 冻结的选档与 B6 真实运行逐题核对（真实运行记下了实际选的档）
    live = [it.references["B6"].route for it in items if "B6" in it.references]
    replay = [
        R.level_of(c.action)
        for c, it in zip(b6.choices, items, strict=True)
        if "B6" in it.references
    ]
    agree = sum(a == b for a, b in zip(live, replay, strict=True))
    curve = [frozen(items, ctrl, x, f"B6@{x:g}") for x in R.LAMBDAS]
    tests: dict[str, Any] = {}
    for a in ("B6 (frozen)", "B6a", "B3"):
        if a not in policies:
            continue
        tests[a] = {}
        for b in ("always-L1", "always-L2", "B6r", "B3", "B6a", "B6 (frozen)"):
            if b == a or b not in policies:
                continue
            w, l_, p = R.paired(policies[a], policies[b])
            tests[a][b] = {"wins": w, "losses": l_, "p": round(p, 4)}
    return {
        "lambda": lam,
        "n": len(items),
        "live_replay_agreement": [agree, len(live)],
        "policies": {k: policy_row(v) for k, v in policies.items()},
        "curve": [policy_row(p) | {"lam": p.lam} for p in curve],
        "tests": tests,
        "routes": dict(Counter(c.action for c in b6.choices)),
        "b6_vs_l2": versus_l2(items, b6),
    }


def probe_latency(run: str) -> dict[str, float]:
    """探针（控制器的输入）的耗时：一次不调用 LLM 的 L1 打分。"""
    ms = [float(r["probe_ms"]) for r in load_rows(RUNS_DIR / run, "probe.jsonl")]
    return {f"p{q}": round(float(np.percentile(ms, q)), 1) for q in (50, 95)}


# ---------- 3. 混合流 ----------


def mixed_section(memory: Sequence[R.Item], ctrl: LearnedController) -> dict[str, Any] | None:
    if not (RUNS_DIR / MIXED_RUN / "probe.jsonl").exists():
        return None
    mixed = R.load_mixed_items(RUNS_DIR / MIXED_RUN, memory)
    items = list(memory) + mixed
    N7.CACHE = CACHE  # 零样本 LLM 路由的结果缓存到 N9 自己的目录
    routes = N7.llm_routes(items)
    b6 = frozen(items, ctrl, ctrl.lam, "B6")
    b6h, asked = hybrid(items, ctrl, routes, ctrl.lam)
    policies = {
        "always-L1": R.fixed(items, "L1"),
        "always-L2": R.fixed(items, "L2"),
        "B6r": R.rule(items),
        "B6z": N7.llm_policy(items, routes),
        "B6h": b6h,
        "B6": b6,
    }
    out: dict[str, Any] = {
        "memory": sum(it.is_memory for it in items),
        "no_memory": dict(Counter(it.dataset for it in mixed)),
        "ratio": MIX_RATIO,
        "hybrid_asked": pct(float(np.mean(asked))),
        "gate_auc": gate_auc(items, ctrl),
    }
    for name, policy in policies.items():
        m = N7.mixed_metrics(policy, items, MIX_RATIO)
        out[name] = {
            "memory_accuracy": pct(m["memory_accuracy"]),
            "cost_per_1k": round(1000 * m["cost"], 3),
            "false_recall": pct(m["false_recall"]),
            "memory_l0": pct(m["memory_l0"]),
        }
    curve = []
    for lam in R.LAMBDAS:
        m = N7.mixed_metrics(frozen(items, ctrl, lam, "B6"), items, MIX_RATIO)
        curve.append(
            {
                "lam": lam,
                "memory_accuracy": pct(m["memory_accuracy"]),
                "cost_per_1k": round(1000 * m["cost"], 3),
                "false_recall": pct(m["false_recall"]),
            }
        )
    out["curve"] = curve
    by_home = {}
    for home in sorted({it.home for it in items}):
        mask = [it.home == home for it in items]
        sub = [it for it, m in zip(items, mask, strict=True) if m]
        m = N7.mixed_metrics(b6.subset(mask), sub, MIX_RATIO)
        l1 = N7.mixed_metrics(policies["always-L1"].subset(mask), sub, MIX_RATIO)
        by_home[home] = {
            "B6_cost_per_1k": round(1000 * m["cost"], 3),
            "L1_cost_per_1k": round(1000 * l1["cost"], 3),
            "B6_memory_accuracy": pct(m["memory_accuracy"]),
            "L1_memory_accuracy": pct(l1["memory_accuracy"]),
            "false_recall": pct(m["false_recall"]),
            "memory_l0": pct(m["memory_l0"]),
        }
    out["by_dataset"] = by_home
    return out


def gate_auc(items: Sequence[R.Item], ctrl: LearnedController) -> dict[str, float]:
    """冻结的门控区分"记忆题"与"无关查询"的 AUC（H1）。

    分别按全部、无关查询的来源、记忆空间所在的数据集计算。
    """
    scores = [ctrl.gate(it.features) for it in items]
    labels = [not it.is_memory for it in items]

    def on(keep: Callable[[R.Item], bool]) -> float:
        idx = [i for i, it in enumerate(items) if keep(it)]
        return round(N7.auc([labels[i] for i in idx], [scores[i] for i in idx]), 4)

    out = {"all": on(lambda it: True)}
    for source in ("topic", "dolly"):
        out[source] = on(lambda it, s=source: it.is_memory or it.dataset == f"mixed:{s}")
    for home in sorted({it.home for it in items}):
        out[home] = on(lambda it, h=home: it.home == h)
    return out


# ---------- 4. 配对比较 ----------


def paired_by_type(
    rows: Sequence[dict[str, Any]],
    a: str,
    b: str,
    *,
    keep: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """两种方法逐题配对：总体与分题型的胜负与符号检验。"""
    by: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        if keep is None or keep(r):
            by[r["sample_id"]][r["method"]] = r
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    out: dict[str, Any] = {}
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for x, y in pairs:
        key = "abstention" if x["is_abstention"] else x["question_type"]
        groups[key].append((x, y))
        groups["all"].append((x, y))
    for key, items in groups.items():
        wins = sum(1 for x, y in items if x["correct"] and not y["correct"])
        losses = sum(1 for x, y in items if y["correct"] and not x["correct"])
        out[key] = {
            "n": len(items),
            a: pct(float(np.mean([x["correct"] for x, _ in items]))),
            b: pct(float(np.mean([y["correct"] for _, y in items]))),
            "wins": wins,
            "losses": losses,
            "p": round(R.sign_test(wins, losses), 4),
        }
    return out


def photo_section(run: str, methods: Sequence[str] = ("B4-see", "B5-see", "B6")) -> dict[str, Any]:
    """MemLens：回答时附上的原图里有没有证据图，以及有、没有两种情况下的准确率。

    证据图 = 证据会话里关键图片（needle_images/）的记录 ID。每道题平均 4.6 张，所以"附上了任意一张"
    是宽松的口径；"排第一的附图是证据图"更能说明找图找得准。
    """
    dataset, evalset = E.load_evalset(run)
    samples = {s.sample_id: s for s in evalset.samples}
    rows = load_rows(RUNS_DIR / run, "results.jsonl")
    out: dict[str, Any] = {}
    for method in methods:
        n = top1 = any_hit = 0
        correct: dict[bool, list[bool]] = {True: [], False: []}
        for r in rows:
            if r["method"] != method:
                continue
            needles = E.needle_images(dataset, evalset, samples[r["sample_id"]])
            if not needles:  # 不需要看图的题
                continue
            details = E._field(r, "details") or {}
            attached = list(details.get("attached") or []) if isinstance(details, dict) else []
            hit = bool(set(attached) & needles)
            n += 1
            any_hit += hit
            top1 += bool(attached) and attached[0] in needles
            correct[hit].append(bool(r["correct"]))
        if n:
            out[method] = {
                "image_questions": n,
                "needle_attached": pct(any_hit / n),
                "needle_first": pct(top1 / n),
                "accuracy_attached": pct(float(np.mean(correct[True]))) if correct[True] else None,
                "accuracy_not_attached": (
                    pct(float(np.mean(correct[False]))) if correct[False] else None
                ),
                "not_attached": len(correct[False]),
            }
    return out


# ---------- 5. 换回答模型 ----------


def reader_items(dataset: str, run: str, ids: set[str]) -> list[R.Item]:
    """某个回答模型在子集上的全信息结果；探针特征与回答模型无关，取主运行的。"""
    run_dir = RUNS_DIR / run
    probe = run_dir / "probe.jsonl"
    main_probe = RUNS_DIR / TEST_RUNS[dataset] / "probe.jsonl"
    if not probe.exists() and main_probe.exists():
        probe.write_text(main_probe.read_text(encoding="utf-8"), encoding="utf-8")
    if not probe.exists():
        return []
    return [it for it in R.load_items(run_dir) if it.sample_id in ids]


def readers_section(ctrl: LearnedController) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset, split in READER_SPLITS.items():
        ids = set(load_split(dataset, split))
        common: set[str] | None = None
        loaded: dict[str, list[R.Item]] = {}
        for reader, runs in READERS.items():
            run = runs.get(dataset)
            if run is None or not available(run):
                continue
            items = reader_items(dataset, run, ids)
            if not items:
                continue
            loaded[reader] = items
            got = {it.sample_id for it in items if all(m in it.references for m in ("B3", "B6a"))}
            common = got if common is None else common & got
        if not loaded or not common:
            continue
        table: dict[str, Any] = {}
        for reader, items in loaded.items():
            items = [it for it in items if it.sample_id in common]
            policies = {
                "L0": R.fixed(items, "L0"),
                "B4-nobm25": R.fixed(items, "L1"),
                "B5": R.fixed(items, "L2"),
                "B6r": R.rule(items),
                "B6": frozen(items, ctrl, ctrl.lam, "B6"),
            }
            for ref in ("B3", "B6a"):
                got_ref = R.reference(items, ref)
                if got_ref is not None:
                    policies[ref] = got_ref
            entry = {k: policy_row(v) for k, v in policies.items()}
            tests = {}
            for a, b in (("B6a", "B3"), ("B6a", "B5"), ("B6a", "B6"), ("B3", "B5"), ("B6", "B5")):
                if a in policies and b in policies:
                    w, l_, p = R.paired(policies[a], policies[b])
                    tests[f"{a} vs {b}"] = {"wins": w, "losses": l_, "p": round(p, 4)}
            entry["tests"] = tests
            table[reader] = entry
        out[dataset] = {"n": len(common), "readers": table}
    return out


# ---------- 6. 错误分析 ----------


def error_section(run: str, methods: Sequence[str]) -> dict[str, Any]:
    rows = [r for r in load_rows(RUNS_DIR / run, "results.jsonl") if r["method"] in methods]
    attribution = error_attribution(rows)
    refusals = false_abstentions(rows)
    out: dict[str, Any] = {}
    for method in methods:
        mine = [r for r in rows if r["method"] == method]
        if not mine:
            continue
        wrong = [r for r in mine if not r["correct"]]
        by_type = Counter("abstention" if r["is_abstention"] else r["question_type"] for r in wrong)
        out[method] = {
            "n": len(mine),
            "wrong": len(wrong),
            "by_type": dict(by_type),
            "attribution": attribution.get(method, {}),
            "false_abstention": refusals.get(method, 0),
            "answered_unanswerable": sum(1 for r in wrong if r["is_abstention"]),
        }
    return out


# ---------- 图 ----------

POINTS = {  # 方法 → (标记, 英文标签)
    "B0": ("o", "B0 no memory"),
    "L0": ("o", "L0 profile only"),
    "B1": ("H", "B1 full context"),
    "B2": ("v", "B2 naive RAG"),
    "always-L1": ("s", "L1 fast recall (+ photos on MemLens)"),
    "B4-note": ("p", "L1 + notes"),
    "always-L2": ("D", "L2 deep recall (+ photos on MemLens)"),
    "B6r": ("^", "B6r rules"),
    "B3": ("P", "B3 agentic (ReAct)"),
    "B6a": ("X", "B6a controller + escape hatch"),
    "B6z": ("<", "B6z zero-shot LLM router"),
}


def _style(ax: Any, theme: Any) -> None:
    ax.set_facecolor(theme.surface)
    ax.grid(color=theme.grid, linewidth=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.axis)
        ax.spines[side].set_linewidth(0.45)
    ax.tick_params(colors=theme.muted, labelsize=8, which="both")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))


def plot_pareto(sections: dict[str, dict[str, Any]], path_for: Callable[[str], Path]) -> None:
    """每个测试集一个面板：各方法一个点（中性色、形状区分），学到的控制器随 λ 一条曲线。"""
    datasets = [d for d in TEST_RUNS if d in sections]
    if not datasets:
        return
    for mode, theme in THEMES.items():
        cols = min(2, len(datasets))
        rows_n = (len(datasets) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(5.4 * cols, 3.9 * rows_n), squeeze=False)
        fig.patch.set_facecolor(theme.surface)
        handles: dict[str, Any] = {}
        for ax, dataset in zip(axes.flat, datasets, strict=False):
            sec = sections[dataset]
            _style(ax, theme)
            curve = sec["curve"]
            xs = [c["cost_per_1k"] for c in curve]
            ys = [c["accuracy"] for c in curve]
            (line,) = ax.plot(
                xs, ys, color=theme.series["blue"], linewidth=2, marker="o", markersize=3.5,
                zorder=3,
            )  # fmt: skip
            handles["B6 learned controller (λ sweep)"] = line
            b6 = sec["policies"]["B6 (frozen)"]
            star = ax.scatter(
                [b6["cost_per_1k"]], [b6["accuracy"]], s=90, marker="*",
                color=theme.series["blue"], edgecolor=theme.surface, linewidth=1.5, zorder=5,
            )  # fmt: skip
            handles["B6 at default λ = 30"] = star
            for key, (marker, label) in POINTS.items():
                p = sec["policies"].get(key)
                if p is None or (key == "B1" and p["cost_per_1k"] > 20):
                    continue
                color = theme.series["yellow"] if key == "B6a" else theme.ink2
                h = ax.scatter(
                    [p["cost_per_1k"]], [p["accuracy"]], s=42, marker=marker, color=color,
                    edgecolor=theme.surface, linewidth=1.2, zorder=4,
                )  # fmt: skip
                handles.setdefault(label, h)
            # set_xscale 会换掉格式器：之后再设美元刻度，并关掉挤在一起的次刻度标签
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20, 50]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(DATASET_EN[dataset], color=theme.ink, fontsize=10, loc="left")
            ax.set_xlabel("nominal cost per 1k questions (log)", color=theme.ink2, fontsize=8)
            ax.set_ylabel("accuracy", color=theme.ink2, fontsize=8)
        for ax in list(axes.flat)[len(datasets) :]:
            ax.set_visible(False)
        legend = fig.legend(
            list(handles.values()), list(handles.keys()), loc="lower center", ncol=3,
            frameon=False, fontsize=8, labelcolor=theme.ink2,
        )  # fmt: skip
        legend.set_zorder(10)
        fig.tight_layout(rect=(0, 0.11 if rows_n > 1 else 0.17, 1, 1))
        fig.savefig(path_for(mode), dpi=160, facecolor=theme.surface)
        plt.close(fig)


def plot_readers(section: dict[str, Any], path_for: Callable[[str], Path]) -> None:
    """换回答模型：每个数据集一个面板，横轴是方法，每个回答模型一条线。"""
    datasets = [d for d in READER_SPLITS if d in section]
    if not datasets:
        return
    methods = ("L0", "B4-nobm25", "B5", "B6r", "B6", "B3", "B6a")
    labels = ("L0", "L1", "L2", "B6r", "B6", "B3", "B6a")
    order = ("qwen3.6-35b-a3b", "qwen3.8-flash", "deepseek-flash", "deepseek-v4-pro")
    for mode, theme in THEMES.items():
        fig, axes = plt.subplots(
            1, len(datasets), figsize=(5.4 * len(datasets), 3.6), squeeze=False
        )
        fig.patch.set_facecolor(theme.surface)
        palette = [theme.series["yellow"], theme.series["aqua"], theme.series["blue"], theme.ink2]
        handles = {}
        for ax, dataset in zip(axes.flat, datasets, strict=True):
            readers = section[dataset]["readers"]
            ax.set_facecolor(theme.surface)
            ax.grid(color=theme.grid, linewidth=0.45, axis="y")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(theme.axis)
                ax.spines[side].set_linewidth(0.45)
            ax.tick_params(colors=theme.muted, labelsize=8)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
            for color, reader in zip(palette, order, strict=True):
                if reader not in readers:
                    continue
                ys = [readers[reader].get(m, {}).get("accuracy", np.nan) for m in methods]
                (h,) = ax.plot(
                    range(len(methods)), ys, marker="o", markersize=4, linewidth=2, color=color
                )
                handles[reader] = h
            ax.set_xticks(range(len(methods)), labels)
            ax.set_title(
                f"{DATASET_EN[dataset].split(' ')[0]} ({section[dataset]['n']} questions)",
                color=theme.ink, fontsize=10, loc="left",
            )  # fmt: skip
            ax.set_ylabel("accuracy", color=theme.ink2, fontsize=8)
        fig.legend(
            list(handles.values()), list(handles.keys()), loc="lower center", ncol=4,
            frameon=False, fontsize=8, labelcolor=theme.ink2,
        )  # fmt: skip
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(path_for(mode), dpi=160, facecolor=theme.surface)
        plt.close(fig)


# ---------- 主流程 ----------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    ctrl = LearnedController.load(V1_MODEL)  # N9 冻结评测的是 v1
    results: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "lambda": ctrl.lam,
        "main": {},
        "controller": {},
        "comparisons": {},
        "errors": {},
    }
    memory_items: list[R.Item] = []
    for dataset, run in TEST_RUNS.items():
        if not available(run):
            continue
        results["main"][dataset] = main_table(run)
        if (RUNS_DIR / run / "probe.jsonl").exists():
            items = R.load_items(RUNS_DIR / run)
            if items:
                results["controller"][dataset] = controller_section(items, ctrl)
                results["controller"][dataset]["probe_ms"] = probe_latency(run)
                if (
                    dataset != "memlens_128k"
                ):  # 混合流与 N7 一致：每道题只算一次（128K 与 32K 同题）
                    memory_items.extend(items)
        rows = load_rows(RUNS_DIR / run, "results.jsonl")
        comps: dict[str, Any] = {}
        for a, b in (
            ("B5", "B4-nobm25"),
            ("B5-see", "B4-see"),
            ("B5-ppr", "B5"),
            ("B4-note", "B4-nobm25"),
            ("B2F", "B2"),
            ("B2F", "B2F-current"),
            ("B2F", "B2F-plain"),
            ("B5", "B5-nonote"),
            ("B5", "B5-noiter"),
            ("B6a", "B3"),
            ("B6a", "B5"),
            ("B4-nobm25", "B1"),
            ("B5", "B1"),
            ("B4-see", "B4-mm"),
        ):
            got = paired_by_type(rows, a, b)
            if got:
                comps[f"{a} vs {b}"] = got
        results["comparisons"][dataset] = comps
        results["errors"][dataset] = error_section(run, ("B6", "B6a", "B5", "B4-nobm25", "B3"))
        if dataset.startswith("memlens"):
            results.setdefault("photos", {})[dataset] = photo_section(run)
    if memory_items:
        mixed = mixed_section(memory_items, ctrl)
        if mixed is not None:
            results["mixed"] = mixed
    results["readers"] = readers_section(ctrl)
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8"
    )
    if results["controller"]:
        plot_pareto(results["controller"], lambda mode: OUT / f"pareto-{mode}.png")
    if results["readers"]:
        plot_readers(results["readers"], lambda mode: OUT / f"readers-{mode}.png")
    print(f"结果：{OUT / 'results.json'}，表格：{n9_tables.write(results)}")
    for dataset, sec in results["controller"].items():
        b6 = sec["policies"]["B6 (frozen)"]
        agree = sec["live_replay_agreement"]
        print(
            f"{dataset}: B6 {b6['accuracy']}% ${b6['cost_per_1k']}/千题 · 真实运行一致 "
            f"{agree[0]}/{agree[1]}"
        )


if __name__ == "__main__":
    main()
