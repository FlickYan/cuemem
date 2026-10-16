"""N7：检索控制器的离线评测与训练，结果写进 docs/results/n7-controller/。

步骤（都不调用 DeepSeek；零样本 LLM 路由只用通义）：

1. 读三个开发集的全信息标注（L0 / L1 / L2，MemLens 另有再感知）与探针特征，读混合流的无关查询
   （scripts/n7_mixed.py 构建）；
2. 零样本 LLM 路由（B6z）：qwen3.8-flash 只看问题文本选一档，结果缓存在 runs/eval/n7-controller/；
3. 交叉拟合（按对话历史分 5 折、重复 5 次，每道题的预测都来自没见过它的模型）：两段式控制器
   （特征 q / p / qp）、直接拟合每档的答对概率、词袋；训练数据包括混合流的无关查询；
4. 按 λ 扫出帕累托曲线，与固定档位、规则（B6r）、模型自主检索（B3）、反应式升级、按不确定度
   逐级升级、理想路由比较；默认 λ 按事先定好的规则选；
5. 跨数据集：在两个数据集上训练、第三个上测试；
6. 混合流：记忆题与无关查询按 1:1 与 1:4 混合；
7. 用全部开发数据训练最终的控制器，导出 src/cuemem/read/controller_model_v1.json
   （N9 冻结评测的那一版；N11 的 v2 由 scripts/n11_controller_v2.py 导出为
   controller_model_v2.json）。

v1 是冻结评测用的那一版，复现脚本只核对、不覆盖它（--no-export --check）：重新训练的结果与
仓库里的文件（导出日期除外）对不上时以非零退出码结束。有意更新控制器时不加这两个参数，并更新
tests/test_frozen_config.py 里钉住的哈希。

用法：
    uv run python scripts/n7_controller.py                      # 评测 + 导出 v1
    uv run python scripts/n7_controller.py --no-export --check  # 只评测，核对 v1 没有变
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from cuemem.config import PROJECT_ROOT, load_settings
from cuemem.core.trace import start_trace
from cuemem.evals import routing as R
from cuemem.evals.report import THEMES
from cuemem.evals.runner import usage_from_trace
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.read.controller import HYBRID_BAND, HYBRID_CONFIDENCE, V1_MODEL, LLMRouter

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RUNS = ("dev100-n7", "locomo-3conv-n7", "memlens32-dev100-n7")
MIXED_RUN = "mixed-n7"
OUT = PROJECT_ROOT / "docs" / "results" / "n7-controller"
CACHE = PROJECT_ROOT / "runs" / "eval" / "n7-controller"
DATASET_LABELS = {
    "longmemeval_s": "LongMemEval dev100",
    "locomo": "LoCoMo 前 3 段对话",
    "memlens_32k": "MemLens 32K dev100",
}
DATASET_EN = {"longmemeval_s": "LongMemEval", "locomo": "LoCoMo", "memlens_32k": "MemLens 32K"}
MIX_RATIOS = (0.5, 0.2)  # 记忆题在流量里的占比
TOLERANCE = 0.01  # "质量优先"的 λ：开发集合并后准确率不低于 always-L2 减 1 个点的最大 λ
BAND = HYBRID_BAND  # 混合方案：门控的 P(none) 落在这个区间里才问 LLM（事先定的，不在评测数据上调）


# ---------- 零样本 LLM 路由 ----------


def llm_routes(items: Sequence[R.Item]) -> dict[str, dict[str, Any]]:
    """每条查询的 LLM 路由结果（level、see、cost），缓存在 runs/eval/n7-controller/。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "router_llm.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["sample_id"]] = row
    todo = [it for it in items if it.sample_id not in done]
    if todo:
        settings = load_settings()
        ledger = Ledger(settings.paths.ledger, run_id="n7-router")
        llm = LLMClient(settings, cache=ResponseCache(settings.paths.llm_cache), ledger=ledger)
        router = LLMRouter(llm)

        def route(item: R.Item) -> dict[str, Any]:
            with start_trace("route") as trace:
                decision = router.decide(item.question)
            usage = usage_from_trace(trace)
            return {
                "sample_id": item.sample_id,
                "level": decision.level,
                "see": decision.see,
                "reasons": decision.reasons,
                "cost": usage["cost"],
            }

        with ThreadPoolExecutor(max_workers=8) as pool, path.open("a", encoding="utf-8") as f:
            for row in pool.map(route, todo):
                done[row["sample_id"]] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        ledger.close()
    return done


def llm_policy(items: Sequence[R.Item], routes: dict[str, dict[str, Any]]) -> R.PolicyResult:
    choices = []
    for it in items:
        r = routes[it.sample_id]
        choice = R.take(it, it.variant(r["level"], see=bool(r["see"])))
        choice.cost += r["cost"]  # 路由本身的一次调用
        choices.append(choice)
    return R.PolicyResult("B6z", choices)


# ---------- 评测 ----------


def subset(policy: R.PolicyResult, items: Sequence[R.Item], keep: Any) -> R.PolicyResult:
    return policy.subset([bool(keep(it)) for it in items])


def choose_lambda(curve: Sequence[R.PolicyResult], target: float) -> R.PolicyResult:
    """事先定好的规则：准确率不低于 target 的最大 λ（最省钱）；都达不到时取 λ = 0。"""
    ok = [p for p in curve if p.accuracy >= target - 1e-12]
    return max(ok, key=lambda p: p.lam or 0.0) if ok else min(curve, key=lambda p: p.lam or 0.0)


def mixed_metrics(
    policy: R.PolicyResult, items: Sequence[R.Item], ratio: float
) -> dict[str, float]:
    mem = subset(policy, items, lambda it: it.is_memory)
    none = subset(policy, items, lambda it: not it.is_memory)
    false_recall = float(np.mean([R.level_of(c.action) != "L0" for c in none.choices]))
    missed = float(np.mean([R.level_of(c.action) == "L0" and not c.correct for c in mem.choices]))
    return {
        "memory_accuracy": mem.accuracy,
        "cost": ratio * mem.cost + (1 - ratio) * none.cost,
        "memory_cost": mem.cost,
        "none_cost": none.cost,
        "false_recall": false_recall,
        "missed_recall": missed,
        "memory_l0": mem.share("L0"),
    }


def auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def none_auc(items: Sequence[R.Item], preds: R.Predictions, source: str | None = None) -> float:
    idx = [
        i
        for i, it in enumerate(items)
        if it.is_memory or source is None or it.dataset == f"mixed:{source}"
    ]
    return auc(
        [not items[i].is_memory for i in idx],
        [preds.type_proba[i].get("none", 0.0) for i in idx],
    )


def type_accuracy(items: Sequence[R.Item], preds: R.Predictions) -> float:
    """记忆题的推理类型分对了多少：只在记忆题的类型里取最大（不算"与记忆无关"）。"""
    right = []
    for it, tp in zip(items, preds.type_proba, strict=True):
        if it.is_memory and tp:
            memory = {t: p for t, p in tp.items() if t != "none"}
            right.append(max(memory, key=lambda t: memory[t]) == it.rtype)
    return float(np.mean(right))


def leave_one_dataset_out(items: Sequence[R.Item], fit: Any) -> R.Predictions:
    """跨数据集：在另外两个数据集（含借用它们空间的无关查询）上训练，在这一个上测试。"""
    homes = sorted({it.home for it in items})
    splits = []
    for home in homes:
        test = [i for i, it in enumerate(items) if it.home == home]
        train = [i for i, it in enumerate(items) if it.home != home]
        splits.append((train, test))
    return R.crossfit(items, fit, splits=splits)


def row(policy: R.PolicyResult) -> dict[str, float]:
    return {
        "accuracy": policy.accuracy,
        "cost": policy.cost,
        "l0": policy.share("L0"),
        "l2": policy.share("L2"),
        "see": float(np.mean([c.action.endswith(R.SEE) for c in policy.choices])),
        "escalation": policy.escalation,
        "n": len(policy.choices),
    }


# ---------- 图 ----------


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


POINT_STYLE = {  # 参照点：中性色，形状区分
    "always-L1": ("s", "always-L1"),
    "always-L2": ("D", "always-L2"),
    "B6r": ("^", "B6r rules"),
    "B3": ("P", "B3 agentic (ReAct)"),
    "B6z": ("X", "B6z zero-shot LLM router"),
    "reactive": ("v", "reactive (L1, escalate if unsure)"),
    "cascade": ("<", "uncertainty cascade (L0, L1, L2)"),
    "oracle": ("*", "outcome oracle"),
}


def plot_curves(
    panels: list[tuple[str, dict[str, list[R.PolicyResult]], dict[str, R.PolicyResult]]],
    path: Path,
    *,
    mode: str,
    title: str,
    subtitle: str,
    ylabel: str,
) -> None:
    theme = THEMES[mode]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    width = 4.0 * len(panels) + 0.6 if len(panels) > 1 else 7.6
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.6), dpi=160, squeeze=False)
    fig.patch.set_facecolor(theme.surface)
    colors = [theme.series["blue"], theme.series["aqua"], theme.series["yellow"]]
    handles: dict[str, Any] = {}
    for ax, (name, curves, points) in zip(axes[0], panels, strict=True):
        _style(ax, theme)
        for color, (label, curve) in zip(colors, curves.items(), strict=False):
            ordered = sorted(curve, key=lambda p: -(p.lam or 0.0))  # λ 从大到小：成本从低到高
            xs = [p.cost * 1000 for p in ordered]
            ys = [p.accuracy * 100 for p in ordered]
            (line,) = ax.plot(
                xs, ys, color=color, linewidth=1.8, marker="o", markersize=3.5,
                markeredgecolor=theme.surface, markeredgewidth=0.6, zorder=3, label=label,
            )  # fmt: skip
            handles.setdefault(label, line)
        for key, policy in points.items():
            marker, label = POINT_STYLE[key]
            hollow = key == "oracle"
            (pt,) = ax.plot(
                policy.cost * 1000, policy.accuracy * 100, marker=marker, markersize=8,
                linestyle="none", markerfacecolor=theme.surface if hollow else theme.ink2,
                markeredgecolor=theme.ink2 if hollow else theme.surface, markeredgewidth=1.2,
                zorder=4, label=label,
            )  # fmt: skip
            handles.setdefault(label, pt)
        # 纵轴只看有意义的一段：参照点下方留 8 个点，曲线在 λ 很大时一路降到 L0 的部分截掉
        ref = [p.accuracy * 100 for p in points.values()]
        low = max(0.0, min(ref) - 8)
        high = min(100.0, max([*ref, *(p.accuracy * 100 for c in curves.values() for p in c)]) + 3)
        ax.set_ylim(low, high)
        visible = [p.cost * 1000 for c in curves.values() for p in c if p.accuracy * 100 >= low] + [
            p.cost * 1000 for p in points.values()
        ]
        ax.set_xlim(min(visible) * 0.92, max(visible) * 1.05)
        ax.set_title(name, loc="left", color=theme.ink, fontsize=10)
        ax.set_xlabel("Nominal cost per 1,000 questions (USD)", color=theme.ink2, fontsize=8.5)
    axes[0][0].set_ylabel(ylabel, color=theme.ink2, fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", color=theme.ink, fontsize=11.5, y=0.985)
    fig.text(0.01, 0.92, subtitle, color=theme.muted, fontsize=8.5)
    ncol = 5 if len(panels) > 1 else 3
    fig.legend(
        handles.values(), handles.keys(), loc="lower center", ncol=ncol,
        frameon=False, fontsize=8, labelcolor=theme.ink2, bbox_to_anchor=(0.5, -0.01),
    )  # fmt: skip
    rows = (len(handles) + ncol - 1) // ncol
    fig.tight_layout(rect=(0, 0.03 + 0.05 * rows, 1, 0.93))
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)


# ---------- 主流程 ----------


def fmt_cost(value: float) -> str:
    return f"${value * 1000:.2f}"


def check_export(path: Path, fresh: dict[str, Any]) -> None:
    """--check：重新训练的控制器与仓库里冻结的那份（导出日期除外）对不上时，以非零退出码结束。"""
    drift = R.controller_drift(fresh, json.loads(path.read_text(encoding="utf-8")))
    if drift:
        shown = "、".join(drift[:10]) + (f" 等 {len(drift)} 处" if len(drift) > 10 else "")
        sys.exit(
            f"重新训练的控制器与 {path.name} 对不上：{shown}。冻结评测用的是仓库里的那份；"
            "有意更新时不加 --no-export / --check 重新导出，"
            "并更新 tests/test_frozen_config.py 里的哈希"
        )
    print(f"核对通过：重新训练的控制器与 {path.name} 一致（导出日期除外）")


def add_export_flags(parser: argparse.ArgumentParser, filename: str) -> None:
    """三个训练脚本（N7 的 v1、N11 的 v2 与 v3）共用的导出开关。"""
    parser.add_argument("--no-export", action="store_true", help=f"只评测，不覆盖 {filename}")
    parser.add_argument(
        "--check",
        action="store_true",
        help=f"重新训练的结果与仓库里的 {filename} 比对，对不上时以非零退出码结束（复现脚本用）",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    add_export_flags(parser, V1_MODEL.name)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    runs = PROJECT_ROOT / "runs" / "eval"
    memory = [it for name in RUNS for it in R.load_items(runs / name)]
    mixed = R.load_mixed_items(runs / MIXED_RUN, memory)
    items = memory + mixed
    datasets = sorted({it.dataset for it in memory}, key=list(DATASET_LABELS).index)
    print(f"记忆题 {len(memory)}，无关查询 {len(mixed)}")

    routes = llm_routes(items)
    names = {fs: R.feature_names(items, fs) for fs in R.FEATURE_SETS}
    fits = {
        "qp": lambda tr: R.fit_two_stage(tr, names["qp"]),
        "q": lambda tr: R.fit_two_stage(tr, names["q"]),
        "p": lambda tr: R.fit_two_stage(tr, names["p"]),
        "text": R.fit_text,
        "direct": lambda tr: R.fit_direct(tr, names["qp"]),
    }
    preds = {k: R.crossfit(items, f, repeats=args.repeats) for k, f in fits.items()}
    llm_none = [routes[it.sample_id]["level"] == "L0" for it in items]
    hybrid_preds, asked = R.hybrid(preds["qp"], llm_none, band=BAND, confidence=HYBRID_CONFIDENCE)
    router_cost = [
        routes[it.sample_id]["cost"] if a else 0.0 for it, a in zip(items, asked, strict=True)
    ]
    curves = {k: R.sweep(items, p, name=f"B6-{k}") for k, p in preds.items()}
    curves["hybrid"] = R.sweep(items, hybrid_preds, name="B6h", extra=router_cost)
    curves["qp+reactive"] = R.sweep(items, preds["qp"], name="B6+reactive", escalate=True)

    is_mem = [it.is_memory for it in items]
    fixed = {
        "always-L0": R.fixed(items, "L0"),
        "always-L1": R.fixed(items, "L1"),
        "always-L2": R.fixed(items, "L2"),
        "B6r": R.rule(items),
        "B6z": llm_policy(items, routes),
        "reactive": R.reactive(items),
        "cascade": R.cascade(items),
        "oracle": R.oracle(items),
        "type-oracle": R.type_oracle(items),
    }
    # 默认 λ（事先定的产品规则）：记忆题（三个开发集合并）上准确率不低于 always-L1 的最大 λ，
    # 即"不比总是检索差的前提下最省钱"
    mem_curve = [p.subset(is_mem) for p in curves["qp"]]
    l1_accuracy = fixed["always-L1"].subset(is_mem).accuracy
    chosen = choose_lambda(mem_curve, l1_accuracy)
    lam = float(chosen.lam or 0.0)
    l2_target = fixed["always-L2"].subset(is_mem).accuracy - TOLERANCE
    quality = choose_lambda(mem_curve, l2_target)
    lam_quality = float(quality.lam or 0.0)
    print(f"默认 λ = {lam:g}（记忆题 {chosen.accuracy:.3f}，{fmt_cost(chosen.cost)} / 千题）")
    print(f"质量优先 λ = {lam_quality:g}（记忆题 {quality.accuracy:.3f}）")
    b6 = R.learned(items, preds["qp"], lam, name="B6")
    # 交叉拟合的选档（每道题都来自没见过它的模型），H3 换回答模型时直接复用（scripts/n7_h3.py）
    (CACHE / "b6_crossfit.json").write_text(
        json.dumps({it.sample_id: c.action for it, c in zip(items, b6.choices, strict=True)}),
        encoding="utf-8",
    )
    b6h = R.learned(items, hybrid_preds, lam, name="B6h", extra=router_cost)
    b6_quality = R.learned(items, preds["qp"], lam_quality, name="B6-quality")
    b6_react = R.learned(items, preds["qp"], lam, name="B6+reactive", escalate=True)
    policies = {**fixed, "B6": b6, "B6h": b6h, "B6-quality": b6_quality, "B6+reactive": b6_react}

    results: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "lambda": lam,
        "lambda_quality": lam_quality,
        "lambdas": list(R.LAMBDAS),
        "band": list(BAND),
        "llm_calls_hybrid": {
            "all": float(np.mean(asked)),
            "memory": float(np.mean([a for a, m in zip(asked, is_mem, strict=True) if m])),
            "none": float(np.mean([a for a, m in zip(asked, is_mem, strict=True) if not m])),
        },
        "memory": {},
        "mixed": {},
        "features": {},
        "cross_dataset": {},
    }

    # --- 记忆基准：每个数据集与合并 ---
    scopes = [("all", lambda it: it.is_memory)] + [
        (ds, lambda it, ds=ds: it.dataset == ds) for ds in datasets
    ]
    for scope, keep in scopes:
        mask = [bool(keep(it)) for it in items]
        scope_items = [it for it, m in zip(items, mask, strict=True) if m]
        table: dict[str, Any] = {k: row(pol.subset(mask)) for k, pol in policies.items()}
        for ref in ("B3", "B4-note"):
            got = R.reference(scope_items, ref)
            if got is not None:
                table[ref] = row(got)
        table["curves"] = {
            k: [row(p.subset(mask)) | {"lam": p.lam} for p in c] for k, c in curves.items()
        }
        tests = {}
        for name in ("B6", "B6h"):
            sub = policies[name].subset(mask)
            tests[name] = {}
            for other in ("always-L1", "always-L2", "B6r", "B6z"):
                w, l_, pv = R.paired(sub, policies[other].subset(mask))
                tests[name][other] = {"wins": w, "losses": l_, "p": pv}
            b3 = R.reference(scope_items, "B3")
            if b3 is not None:
                w, l_, pv = R.paired(sub, b3)
                tests[name]["B3"] = {"wins": w, "losses": l_, "p": pv}
        table["tests"] = tests
        table["agreement"] = R.routing_agreement(b6.subset(mask), scope_items)
        results["memory"][scope] = table

    # --- 混合流 ---
    for ratio in MIX_RATIOS:
        key = f"{round(ratio * 100)}%"
        entry: dict[str, Any] = {k: mixed_metrics(pol, items, ratio) for k, pol in policies.items()}
        for fs in ("q", "p", "text", "direct"):
            entry[f"B6-{fs}"] = mixed_metrics(
                R.learned(items, preds[fs], lam, name=f"B6-{fs}"), items, ratio
            )
        entry["curves"] = {
            k: [mixed_metrics(p, items, ratio) | {"lam": p.lam} for p in curves[k]]
            for k in ("qp", "q", "p", "text", "hybrid")
        }
        results["mixed"][key] = entry

    # --- 特征（H1）：L0 决策的 AUC、记忆题推理类型的准确率、曲线下面积 ---
    lo = fixed["always-L1"].subset(is_mem).cost
    hi = fixed["always-L2"].subset(is_mem).cost
    for fs in ("q", "p", "qp", "text", "direct"):
        pr = preds[fs]
        entry = {"curve_area": R.area_under_curve([p.subset(is_mem) for p in curves[fs]], lo, hi)}
        if pr.p_none:
            entry["none_auc"] = none_auc(items, pr)
            entry["none_auc_topic"] = none_auc(items, pr, "topic")
            entry["none_auc_dolly"] = none_auc(items, pr, "dolly")
            entry["type_accuracy"] = type_accuracy(items, pr)
        results["features"][fs] = entry
    results["features"]["llm"] = {
        "none_as_l0": float(np.mean([x for x, m in zip(llm_none, is_mem, strict=True) if not m])),
        "memory_as_l0": float(np.mean([x for x, m in zip(llm_none, is_mem, strict=True) if m])),
    }
    results["features"]["hybrid"] = {
        "none_as_l0": float(
            np.mean([b6h_l0 for b6h_l0, m in zip(_l0(b6h), is_mem, strict=True) if not m])
        ),
        "memory_as_l0": float(np.mean([x for x, m in zip(_l0(b6h), is_mem, strict=True) if m])),
    }
    results["features"]["qp_at_lambda"] = {
        "none_as_l0": float(np.mean([x for x, m in zip(_l0(b6), is_mem, strict=True) if not m])),
        "memory_as_l0": float(np.mean([x for x, m in zip(_l0(b6), is_mem, strict=True) if m])),
    }

    # --- 跨数据集 ---
    for fs in ("qp", "q", "text"):
        cross = leave_one_dataset_out(items, fits[fs])
        cross_b6 = R.learned(items, cross, lam, name=f"cross-{fs}")
        in_domain = R.learned(items, preds[fs], lam)
        entry = {}
        for ds in datasets:
            mask = [it.dataset == ds for it in items]
            home = [it.home == ds for it in items]
            entry[ds] = {
                "in_domain": row(in_domain.subset(mask)),
                "cross": row(cross_b6.subset(mask)),
                "cross_none_auc": auc(
                    [not items[i].is_memory for i in range(len(items)) if home[i]],
                    [cross.p_none[i] for i in range(len(items)) if home[i]],
                )
                if cross.p_none
                else None,
                "in_domain_none_auc": auc(
                    [not items[i].is_memory for i in range(len(items)) if home[i]],
                    [preds[fs].p_none[i] for i in range(len(items)) if home[i]],
                )
                if preds[fs].p_none
                else None,
            }
        results["cross_dataset"][fs] = entry

    # --- 跨数据集 + 少量目标场景的数据：能不能把迁移失败救回来 ---
    results["adaptation"] = adaptation(items, datasets, names["qp"], lam)

    # --- 最终模型：全部开发数据训练，导出 ---
    final = R.fit_two_stage(items, names["qp"])
    coef_rank: dict[str, Any] = {}
    memory_types = final.types[:-1] if final.gate_coef is not None else final.types
    for i, t in enumerate(memory_types):
        order = np.argsort(-final.coef[i])
        coef_rank[t] = [(final.features[j], round(float(final.coef[i][j]), 3)) for j in order[:5]]
    if final.gate_coef is not None:
        order = np.argsort(-final.gate_coef)
        coef_rank["none"] = [
            (final.features[j], round(float(final.gate_coef[j]), 3)) for j in order[:5]
        ]
        coef_rank["memory"] = [
            (final.features[j], round(float(final.gate_coef[j]), 3)) for j in order[::-1][:5]
        ]
    results["coefficients"] = coef_rank
    results["accuracy_table"] = final.accuracy.level
    results["see_gain"] = final.accuracy.see_gain
    results["costs"] = {**final.costs.level, "see": final.costs.see}
    meta = {
        "trained_on": {
            "memory": {ds: sum(it.dataset == ds for it in memory) for ds in datasets},
            "no_memory": dict(Counter(it.dataset for it in mixed)),
        },
        "lambda_rule": "开发集记忆题合并后，准确率不低于 always-L1 的最大 λ",
        "hybrid_band": list(BAND),
        "created": datetime.now().strftime("%Y-%m-%d"),
    }
    model = final.to_json(lam=lam, meta=meta)
    if args.check:  # 在写任何结果之前：对不上时 docs/results 也不动
        check_export(V1_MODEL, model)
    if not args.no_export:
        V1_MODEL.write_text(
            json.dumps(model, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8"
    )
    draw(items, datasets, fixed, curves)
    print(f"结果：{OUT}" + ("" if args.no_export else f"；控制器：{V1_MODEL}"))


ADAPT_SIZES = (0, 10, 20, 50, 100)
ADAPT_REPEATS = 10


def adaptation(
    items: Sequence[R.Item], datasets: Sequence[str], names: Sequence[str], lam: float
) -> dict[str, Any]:
    """留出一个数据集，在另外两个上训练，再加上目标数据集里随机抽的 N 条（总权重是其他数据的一半，
    事先定的），在目标数据集其余的题上评测。重复 10 次取平均。"""
    out: dict[str, Any] = {}
    for ds in datasets:
        home = [i for i, it in enumerate(items) if it.home == ds]
        other = [i for i, it in enumerate(items) if it.home != ds]
        out[ds] = {}
        for n in ADAPT_SIZES:
            runs = []
            for rep in range(ADAPT_REPEATS if n else 1):
                rng = np.random.RandomState(rep)
                sample = set(rng.choice(home, n, replace=False).tolist()) if n else set()
                train = [items[i] for i in other] + [items[i] for i in sorted(sample)]
                weight = len(other) / (2 * n) if n else 1.0
                weights = [1.0] * len(other) + [weight] * len(sample)
                model = R.fit_two_stage(train, names, weights=weights)
                test = [items[i] for i in home if i not in sample]
                proba = model.predict(test)
                choices = []
                for it, p in zip(test, proba, strict=True):
                    costs = {a: model.costs.of(a) for a in p}
                    choices.append(R.take(it, R.decide(p, costs, lam, list(p))))
                policy = R.PolicyResult("adapted", choices, lam)
                mem = policy.subset([it.is_memory for it in test])
                runs.append(
                    {
                        "memory_accuracy": mem.accuracy,
                        "memory_l0": mem.share("L0"),
                        "memory_cost": mem.cost,
                        "none_l0": policy.subset([not it.is_memory for it in test]).share("L0"),
                    }
                )
            out[ds][str(n)] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    return out


def _l0(policy: R.PolicyResult) -> list[bool]:
    return [R.level_of(c.action) == "L0" for c in policy.choices]


def draw(
    items: Sequence[R.Item],
    datasets: Sequence[str],
    fixed: dict[str, R.PolicyResult],
    curves: dict[str, list[R.PolicyResult]],
) -> None:
    for mode in ("light", "dark"):
        panels = []
        for ds in datasets:
            mask = [it.dataset == ds for it in items]
            ds_items = [it for it in items if it.dataset == ds]
            keys = ("always-L1", "always-L2", "B6r", "B6z", "reactive", "cascade", "oracle")
            pts = {k: fixed[k].subset(mask) for k in keys}
            b3 = R.reference(ds_items, "B3")
            if b3 is not None:
                pts["B3"] = b3
            panels.append(
                (
                    DATASET_EN[ds],
                    {
                        "B6 learned (probe + cues)": [p.subset(mask) for p in curves["qp"]],
                        "B6h hybrid (+ LLM when unsure)": [
                            p.subset(mask) for p in curves["hybrid"]
                        ],
                        "B6 cues only": [p.subset(mask) for p in curves["q"]],
                    },
                    pts,
                )
            )
        plot_curves(
            panels,
            OUT / f"pareto-{mode}.png",
            mode=mode,
            title="Retrieval controller on memory questions: accuracy vs. cost",
            subtitle="Lines sweep λ (cross-fitted, 5 folds × 5 repeats); "
            "the controller may also choose L0 (no retrieval)",
            ylabel="Accuracy (judged)",
        )
        ratio = MIX_RATIOS[-1]
        pts = {
            k: _as_point(mixed_metrics(fixed[k], items, ratio))
            for k in ("always-L1", "always-L2", "B6r", "B6z", "cascade", "oracle")
        }
        curves_m = {
            "B6 learned (probe + cues)": [
                _as_point(mixed_metrics(p, items, ratio)) for p in curves["qp"]
            ],
            "B6h hybrid (+ LLM when unsure)": [
                _as_point(mixed_metrics(p, items, ratio)) for p in curves["hybrid"]
            ],
            "B6 cues only": [_as_point(mixed_metrics(p, items, ratio)) for p in curves["q"]],
        }
        plot_curves(
            [("Mixed stream: 20% memory questions, 80% general", curves_m, pts)],
            OUT / f"mixed-{mode}.png",
            mode=mode,
            title="Should it recall at all?",
            subtitle="y: accuracy on the memory questions; x: average cost over all queries",
            ylabel="Accuracy on memory questions",
        )


def _as_point(m: dict[str, float]) -> R.PolicyResult:
    """把混合流的（记忆题准确率，平均成本）包装成可以画图的点。"""
    return _Point(m["memory_accuracy"], m["cost"])


class _Point(R.PolicyResult):
    def __init__(self, accuracy: float, cost: float) -> None:
        super().__init__("mixed", [])
        self._accuracy = accuracy
        self._cost = cost

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @property
    def cost(self) -> float:
        return self._cost


if __name__ == "__main__":
    main()
