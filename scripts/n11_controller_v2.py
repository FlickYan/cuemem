"""N11：检索控制器 v2 的离线评测与导出（不调用任何 API）。

N9 在测试集上查出的两个控制器问题，在这里修：

1. **看图增益饱和**。v1 把再感知的增益加到"三个开发集合并"的准确率上再截断到 1：合并的准确率
   主要来自文字数据集，单点事实题 L1 / L2 都在 0.86 左右，加上 MemLens 的看图增益双双溢出成
   1.0，两档打平，代价项就总选便宜的 L1。v2 在有图的空间里直接查"有图数据"估出来的准确率表
   （routing.image_table），不再加增益、截断。门控与推理类型模型不变。
2. **准确率表随回答模型变**。选档的特征与回答模型无关，"这一档能答对多少"有关。
   ControllerModel.with_tables 换上新回答模型的表，门控与类型模型不变。

评测协议（D-052）：

- **开发集**（干净）：与 N7 相同的折与重复做交叉拟合，比较 v1 与 v2；v2 的默认 λ 按 N7 同一条
  规则（记忆题上准确率不低于 always-L1 的最大 λ）在开发集上选。
- **测试集**（事后）：v2 的改法是看了测试集的错题之后才做的，测试集上的数字只算事后诊断，
  不替代 N9 冻结评测的结论。
- **换回答模型**：qwen3.8-flash 用 N7 在开发集上的标注标定、在测试子集上评；qwen3.6-35b-a3b
  没有开发集标注，在测试子集上按对话历史 2 折交叉标定（事后）。

v2 是 N11 冻结评测用的那一版，复现脚本只核对、不覆盖它（--no-export --check，见
scripts/n7_controller.py）。

用法：
    uv run python scripts/n11_controller_v2.py            # 评测 + 导出 controller_model_v2.json
    uv run python scripts/n11_controller_v2.py --no-export
    uv run python scripts/n11_controller_v2.py --no-export --check  # 核对 v2 没有变
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

import n7_controller as N7
import n9_report as N9
import numpy as np

from cuemem.config import PROJECT_ROOT
from cuemem.evals import routing as R
from cuemem.evals.splits import load_split
from cuemem.read.controller import V1_MODEL, V2_MODEL, LearnedController
from cuemem.read.probe import scene_reference

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RUNS = PROJECT_ROOT / "runs" / "eval"
OUT = PROJECT_ROOT / "docs" / "results" / "n11-improvements"
LAM_V1 = 30.0
MIX_RATIO = 0.2
DEV_LABELS = {
    "longmemeval_s": "LongMemEval dev100",
    "locomo": "LoCoMo 前 3 段对话",
    "memlens_32k": "MemLens 32K dev100",
}
QFLASH_DEV = ("dev100-n7-qflash", "locomo-3conv-n7-qflash")


def pct(x: float) -> float:
    return round(100 * x, 1)


def row(policy: R.PolicyResult) -> dict[str, float]:
    see = float(np.mean([c.action.endswith(R.SEE) for c in policy.choices]))
    return {
        "accuracy": pct(policy.accuracy),
        "cost_per_1k": round(1000 * policy.cost, 3),
        "l0": pct(policy.share("L0")),
        "l2": pct(policy.share("L2")),
        "see": pct(see),
        "n": len(policy.choices),
    }


def versus(a: R.PolicyResult, b: R.PolicyResult) -> dict[str, float]:
    wins, losses, p = R.paired(a, b)
    return {"wins": wins, "losses": losses, "p": round(p, 4)}


def frozen(items: Sequence[R.Item], ctrl: LearnedController, lam: float) -> R.PolicyResult:
    return N9.frozen(items, ctrl, lam, "B6")


# ---------- 1. 开发集：交叉拟合比较 v1 与 v2 ----------

IMAGE_FLOOR = ["q_visual", "q_scene"]


def add_scene(items: Sequence[R.Item]) -> None:
    """已有的探针结果里没有 q_scene（N11 新加的问题线索）：按问题文本补上。"""
    for it in items:
        it.features = {**it.features, "q_scene": float(scene_reference(it.question))}


def floored(items: Sequence[R.Item], policy: R.PolicyResult) -> R.PolicyResult:
    """有图空间的 L0 保底（与 LearnedController.decide 相同的规则）：问画面细节就至少 L1 + 看图。"""
    choices = []
    for it, c in zip(items, policy.choices, strict=True):
        if (
            it.has_images
            and R.level_of(c.action) == "L0"
            and "L1+see" in it.actions
            and any(it.features.get(name, 0.0) > 0 for name in IMAGE_FLOOR)
        ):
            c = R.take(it, "L1+see")
        choices.append(c)
    return R.PolicyResult(policy.name, choices, policy.lam)


def dev_section(
    items: list[R.Item], names: list[str], repeats: int
) -> tuple[dict[str, Any], float]:
    preds = {
        "v1": R.crossfit(items, lambda tr: R.fit_two_stage(tr, names), repeats=repeats),
        "v2": R.crossfit(
            items, lambda tr: R.fit_two_stage(tr, names, by_images=True), repeats=repeats
        ),
    }
    is_mem = [it.is_memory for it in items]
    target = R.fixed(items, "L1").subset(is_mem).accuracy
    curves = {
        "v1": R.sweep(items, preds["v1"], name="B6 v1"),
        "v2 table only": R.sweep(items, preds["v2"], name="B6 v2 table only"),
    }
    curves["v2"] = [floored(items, c) for c in curves["v2 table only"]]
    lams = {
        version: float(N7.choose_lambda([c.subset(is_mem) for c in curve], target).lam or 0.0)
        for version, curve in curves.items()
    }
    policies = {
        "always-L1": R.fixed(items, "L1"),
        "always-L2": R.fixed(items, "L2"),
        "B6r": R.rule(items),
        "B6 v1": R.learned(items, preds["v1"], LAM_V1, name="B6 v1"),
        "B6 v2 table only": R.learned(items, preds["v2"], lams["v2"], name="table only"),
        "B6 v2": floored(items, R.learned(items, preds["v2"], lams["v2"], name="B6 v2")),
    }
    out: dict[str, Any] = {"lambda": lams, "combined": {}, "by_dataset": {}, "mixed": {}}
    for name, policy in policies.items():
        out["combined"][name] = row(policy.subset(is_mem))
        m = N7.mixed_metrics(policy, items, MIX_RATIO)
        out["mixed"][name] = {
            "memory_accuracy": pct(m["memory_accuracy"]),
            "cost_per_1k": round(1000 * m["cost"], 3),
            "false_recall": pct(m["false_recall"]),
            "memory_l0": pct(m["memory_l0"]),
        }
    for dataset in DEV_LABELS:
        mask = [it.dataset == dataset for it in items]
        sec = {name: row(policy.subset(mask)) for name, policy in policies.items()}
        v1, v2, l2 = (policies[k].subset(mask) for k in ("B6 v1", "B6 v2", "always-L2"))
        sec["tests"] = {"v2 vs v1": versus(v2, v1), "v2 vs always-L2": versus(v2, l2)}
        out["by_dataset"][dataset] = sec
    v1, v2, l2 = (policies[k].subset(is_mem) for k in ("B6 v1", "B6 v2", "always-L2"))
    out["tests"] = {"v2 vs v1": versus(v2, v1), "v2 vs always-L2": versus(v2, l2)}
    # 曲线：记忆题上各 λ 的准确率与成本（成本区间取几条曲线共同覆盖的部分）
    out["curves"] = {
        version: [{"lam": c.lam, **row(c.subset(is_mem))} for c in curve]
        for version, curve in curves.items()
    }
    low = max(min(c.subset(is_mem).cost for c in curve) for curve in curves.values())
    high = min(max(c.subset(is_mem).cost for c in curve) for curve in curves.values())
    out["auc"] = {
        version: round(100 * R.area_under_curve([c.subset(is_mem) for c in curve], low, high), 2)
        for version, curve in curves.items()
    }
    out["auc_range_per_1k"] = [round(1000 * low, 3), round(1000 * high, 3)]
    return out, lams["v2"]


def fit_v2(
    items: list[R.Item], names: list[str], lam: float, memory: list[R.Item]
) -> dict[str, Any]:
    """用全部开发数据训练 v2，返回要导出的 JSON。"""
    final = R.fit_two_stage(items, names, by_images=True)
    meta = {
        "trained_on": {
            "memory": dict(Counter(it.dataset for it in memory)),
            "no_memory": dict(Counter(it.dataset for it in items if not it.is_memory)),
        },
        "lambda_rule": "开发集记忆题合并后，准确率不低于 always-L1 的最大 λ（与 v1 相同的规则）",
        "hybrid_band": list(N7.BAND),
        "image_floor": IMAGE_FLOOR,
        "changes": (
            "v2（N11）：门控与推理类型模型与 v1 相同；有图的空间直接查有图数据的准确率表；"
            "有图的空间里问画面细节时不选 L0"
        ),
        "created": datetime.now().strftime("%Y-%m-%d"),
    }
    return final.to_json(lam=lam, meta=meta)


def export_v2(model: dict[str, Any]) -> None:
    V2_MODEL.write_text(json.dumps(model, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    v1 = json.loads(V1_MODEL.read_text(encoding="utf-8"))
    same = np.allclose(np.asarray(v1["coef"]), np.asarray(model["coef"]), atol=1e-5)
    print(f"导出 v2：{V2_MODEL}（λ = {model['lambda']:g}；类型模型与 v1 一致：{same}）")


# ---------- 2. 测试集：事后重放 ----------


def test_section(ctrl_v1: LearnedController, ctrl_v2: LearnedController) -> dict[str, Any]:
    out: dict[str, Any] = {"by_dataset": {}}
    # 只改查表、不加保底：看 N9 §5.3 归因于饱和的那几道题是不是被修回来、别处又丢了多少
    table_only = LearnedController(replace(ctrl_v2.model, image_floor=None), ctrl_v2.lam)
    memory: list[R.Item] = []
    for dataset, run in N9.TEST_RUNS.items():
        items = R.load_items(RUNS / run)
        if not items:
            continue
        add_scene(items)
        if dataset != "memlens_128k":
            memory.extend(items)
        policies = {
            "always-L1": R.fixed(items, "L1"),
            "always-L2": R.fixed(items, "L2"),
            "B6r": R.rule(items),
            "B6 v1": frozen(items, ctrl_v1, ctrl_v1.lam),
            "B6 v2 table only": frozen(items, table_only, table_only.lam),
            "B6 v2": frozen(items, ctrl_v2, ctrl_v2.lam),
        }
        sec: dict[str, Any] = {name: row(p) for name, p in policies.items()}
        sec["tests"] = {
            "table only vs v1": versus(policies["B6 v2 table only"], policies["B6 v1"]),
            "v2 vs v1": versus(policies["B6 v2"], policies["B6 v1"]),
            "v2 vs always-L2": versus(policies["B6 v2"], policies["always-L2"]),
            "v2 vs always-L1": versus(policies["B6 v2"], policies["always-L1"]),
        }
        sec["v2_vs_l2"] = N9.versus_l2(items, policies["B6 v2"])
        sec["v1_losses_to_l2"] = recovered(items, policies)
        out["by_dataset"][dataset] = sec
    mixed = R.load_mixed_items(RUNS / N9.MIXED_RUN, memory)
    add_scene(mixed)
    items = memory + mixed
    is_mem = [it.is_memory for it in items]
    policies = {
        "always-L1": R.fixed(items, "L1"),
        "always-L2": R.fixed(items, "L2"),
        "B6 v1": frozen(items, ctrl_v1, ctrl_v1.lam),
        "B6 v2": frozen(items, ctrl_v2, ctrl_v2.lam),
    }
    out["combined"] = {name: row(p.subset(is_mem)) for name, p in policies.items()}
    v1, v2, l2 = (policies[k].subset(is_mem) for k in ("B6 v1", "B6 v2", "always-L2"))
    out["tests"] = {"v2 vs v1": versus(v2, v1), "v2 vs always-L2": versus(v2, l2)}
    out["mixed"] = {}
    for name, policy in policies.items():
        m = N7.mixed_metrics(policy, items, MIX_RATIO)
        out["mixed"][name] = {
            "memory_accuracy": pct(m["memory_accuracy"]),
            "cost_per_1k": round(1000 * m["cost"], 3),
            "false_recall": pct(m["false_recall"]),
            "memory_l0": pct(m["memory_l0"]),
        }
    return out


def recovered(items: Sequence[R.Item], policies: dict[str, R.PolicyResult]) -> dict[str, int]:
    """v1 输给"总是 L2"的题：只改查表、再加保底之后各救回多少（按 v1 当时选的档分开）。"""
    out: Counter[str] = Counter()
    rows = zip(
        items,
        policies["B6 v1"].choices,
        policies["B6 v2 table only"].choices,
        policies["B6 v2"].choices,
        policies["always-L2"].choices,
        strict=True,
    )
    for _, v1, table, v2, deep in rows:
        if deep.correct and not v1.correct:
            level = R.level_of(v1.action)
            out[f"lost_at_{level}"] += 1
            out[f"lost_at_{level}_fixed_by_table"] += table.correct
            out[f"lost_at_{level}_fixed_by_v2"] += v2.correct
    return dict(sorted(out.items()))


# ---------- 3. 换回答模型：重新标定准确率表 ----------


def memory_rows(table: R.AccuracyTable) -> dict[str, dict[str, float]]:
    """新回答模型的表只换记忆题的行："与记忆无关"那一行按假设各档都答对，沿用原表。"""
    return {t: v for t, v in table.level.items() if t != "none"}


def reader_policies(
    items: Sequence[R.Item], ctrl: LearnedController, calibrated: dict[str, R.PolicyResult]
) -> dict[str, Any]:
    policies: dict[str, R.PolicyResult] = {
        "L1": R.fixed(items, "L1"),
        "L2": R.fixed(items, "L2"),
        "B6 v2": frozen(items, ctrl, ctrl.lam),
        **calibrated,
    }
    got = R.reference(items, "B3")
    if got is not None:
        policies["B3"] = got
    out: dict[str, Any] = {name: row(p) for name, p in policies.items()}
    tests = {}
    for name in calibrated:
        tests[f"{name} vs B6 v2"] = versus(policies[name], policies["B6 v2"])
        tests[f"{name} vs L2"] = versus(policies[name], policies["L2"])
    tests["B6 v2 vs L2"] = versus(policies["B6 v2"], policies["L2"])
    out["tests"] = tests
    return out


def crossfit_calibration(
    items: Sequence[R.Item], ctrl: LearnedController, seed: int = 0
) -> R.PolicyResult:
    """在这个回答模型的标注上按对话历史 2 折：一半标定准确率表，另一半上评（事后）。"""
    choices: list[R.Choice | None] = [None] * len(items)
    for train, test in R.group_folds(items, 2, seed):
        table = R.accuracy_table([items[i] for i in train], ctrl.model.types)
        model = ctrl.model.with_tables(memory_rows(table))
        calibrated = LearnedController(model, ctrl.lam)
        for i in test:
            it = items[i]
            d = calibrated.decide(it.features, has_images=it.has_images, lam=ctrl.lam)
            choices[i] = R.take(it, it.variant(d.level, d.see))
    return R.PolicyResult("B6 calibrated (2-fold)", [c for c in choices if c is not None])


def readers_section(ctrl: LearnedController) -> dict[str, Any]:
    out: dict[str, Any] = {}
    # qwen3.8-flash：开发集上的标注（N7 的 H3 运行）标定
    dev = [it for run in QFLASH_DEV for it in R.load_items(RUNS / run)]
    table = R.accuracy_table(dev, ctrl.model.types)
    qflash = LearnedController(ctrl.model.with_tables(memory_rows(table)), ctrl.lam)
    out["qflash_table"] = {
        t: {k: round(v, 3) for k, v in r.items()} for t, r in memory_rows(table).items()
    }
    for reader in ("qwen3.8-flash", "qwen3.6-35b-a3b", "deepseek-flash"):
        sec: dict[str, Any] = {}
        pooled: list[R.Item] = []
        for dataset, split in N9.READER_SPLITS.items():
            run = N9.READERS[reader].get(dataset)
            if run is None or not N9.available(run):
                continue
            ids = set(load_split(dataset, split))
            items = [
                it
                for it in N9.reader_items(dataset, run, ids)
                if all(m in it.references for m in ("B3",))
            ]
            if not items:
                continue
            pooled.extend(items)
            calibrated: dict[str, R.PolicyResult] = {}
            if reader == "qwen3.8-flash":
                calibrated["B6 v2 + dev calibration"] = frozen(items, qflash, ctrl.lam)
            sec[dataset] = reader_policies(items, ctrl, calibrated)
            # 更直接的旋钮：λ。回答模型越弱，L1 与 L2 的差距越大，调小 λ 就多走 L2
            l2 = R.fixed(items, "L2")
            sec[dataset]["lambda_sweep"] = [
                {"lam": lam, **row(p), "vs_L2": versus(p, l2)}
                for lam in (30.0, 15.0, 10.0, 5.0, 0.0)
                for p in [frozen(items, ctrl, lam)]
            ]
        if pooled:
            # 2 折交叉标定：先在两个数据集合并的标注上折，再按数据集拆开报告
            policy = crossfit_calibration(pooled, ctrl)
            for dataset in list(sec):
                mask = [it.dataset == dataset for it in pooled]
                sub = policy.subset(mask)
                items = [it for it, m in zip(pooled, mask, strict=True) if m]
                base = frozen(items, ctrl, ctrl.lam)
                sec[dataset]["B6 v2 + 2-fold calibration"] = row(sub)
                sec[dataset]["tests"]["B6 v2 + 2-fold calibration vs B6 v2"] = versus(sub, base)
                sec[dataset]["tests"]["B6 v2 + 2-fold calibration vs L2"] = versus(
                    sub, R.fixed(items, "L2")
                )
        out[reader] = sec
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    N7.add_export_flags(parser, V2_MODEL.name)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    memory = [it for name in N7.RUNS for it in R.load_items(RUNS / name)]
    mixed = R.load_mixed_items(RUNS / N7.MIXED_RUN, memory)
    items = memory + mixed
    names = R.feature_names(items, "qp")  # 分类器用的特征：先取，再补保底规则用的 q_scene
    add_scene(items)
    print(f"开发集：记忆题 {len(memory)}，无关查询 {len(mixed)}")
    dev, lam_v2 = dev_section(items, names, args.repeats)
    print(f"开发集 λ：{dev['lambda']}；AUC：{dev['auc']}")
    if args.check or not args.no_export:
        model = fit_v2(items, names, lam_v2, memory)
        if args.check:  # 在写任何结果之前：对不上时 docs/results 也不动
            N7.check_export(V2_MODEL, model)
        if not args.no_export:
            export_v2(model)
    ctrl_v1 = LearnedController.load(V1_MODEL)
    ctrl_v2 = LearnedController.load(V2_MODEL)
    results = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "dev": dev,
        "test_posthoc": test_section(ctrl_v1, ctrl_v2),
        "readers": readers_section(ctrl_v2),
    }
    (OUT / "controller_v2.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float) + "\n", encoding="utf-8"
    )
    for name, r in results["dev"]["combined"].items():
        print(f"开发集记忆题 {name}: {r['accuracy']}% ${r['cost_per_1k']}/千题")
    for name, r in results["test_posthoc"]["combined"].items():
        print(f"测试集记忆题（事后）{name}: {r['accuracy']}% ${r['cost_per_1k']}/千题")
    print(f"结果：{OUT / 'controller_v2.json'}")


if __name__ == "__main__":
    main()
