"""N11 之后：再感知的门槛（D-058，控制器 v3）。离线评测与导出，不调用任何 API。

问题：v2 在有图的空间里直接查"有图数据"的准确率表，表来自 MemLens——那里几乎每道题都问图，
看图总是更准，于是 v2 在有图的空间里对 96–98% 的问题都附上原图。真实对话不是这样：演示里
问预算、问尺码也附上了朋友那双鞋的照片，模型被图带偏，把上一句"找长得像的鞋"又答了一遍。

v3 = v2 + 一道门槛：问题带画面线索（q_visual / q_scene），或者图片描述的检索相似度
（p_caption_top1）不低于 τ，才考虑看图；否则只在不看图的动作里选。门控、类型模型、准确率表与
λ 都不变。

协议（与 D-052 相同）：τ 只在开发集（MemLens 32K dev100）上选——让开发集准确率不低于 v2 的
最大 τ；测试集只作事后重放。文字数据集的空间里没有图，结果应与 v2 逐题相同（顺带核对）。

v3 是默认的控制器，复现脚本只核对、不覆盖它（--no-export --check，见 scripts/n7_controller.py）。

用法：
    uv run python scripts/n11_see_gate.py            # 评测 + 导出 controller_model.json（v3）
    uv run python scripts/n11_see_gate.py --no-export
    uv run python scripts/n11_see_gate.py --no-export --check  # 核对 v3 没有变
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

import multiplicity
import n9_report as N9
from n7_controller import add_export_flags, finish_export
from n11_controller_v2 import IMAGE_FLOOR, OUT, RUNS, add_scene, frozen, pct, versus

from cuemem.evals import routing as R
from cuemem.read.controller import DEFAULT_MODEL, V2_MODEL, LearnedController

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEV_RUN = "memlens32-dev100-n7"
GRID = [round(0.30 + 0.02 * i, 2) for i in range(21)]  # 0.30 … 0.70


def gated(ctrl: LearnedController, tau: float) -> LearnedController:
    gate = {"cues": IMAGE_FLOOR, "caption_min": tau}
    return LearnedController(replace(ctrl.model, see_gate=gate), ctrl.lam)


def summary(items: Sequence[R.Item], policy: R.PolicyResult) -> dict[str, float]:
    see = [c.action.endswith(R.SEE) for c in policy.choices]
    return {
        "n": len(items),
        "accuracy": pct(policy.accuracy),
        "cost_per_1k": round(1000 * policy.cost, 3),
        "see_share": pct(sum(see) / len(see)) if see else 0.0,
    }


def choose_tau(v2: LearnedController) -> tuple[float, list[dict[str, Any]]]:
    items = R.load_items(RUNS / DEV_RUN)
    add_scene(items)
    base = frozen(items, v2, v2.lam)
    rows = []
    for tau in GRID:
        policy = frozen(items, gated(v2, tau), v2.lam)
        rows.append({"tau": tau, **summary(items, policy)})
    keep = [r["tau"] for r in rows if r["accuracy"] >= pct(base.accuracy)]
    tau = max(keep) if keep else GRID[0]
    print(f"开发集 {DEV_RUN}：v2 {pct(base.accuracy)}%；τ = {tau}")
    return tau, [{"tau": "v2", **summary(items, base)}, *rows]


def test_replay(v2: LearnedController, v3: LearnedController) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset, run in N9.TEST_RUNS.items():
        items = R.load_items(RUNS / run)
        if not items:
            continue
        add_scene(items)
        a, b = frozen(items, v2, v2.lam), frozen(items, v3, v3.lam)
        out[dataset] = {
            "v2": summary(items, a),
            "v3": summary(items, b),
            "changed": sum(x.action != y.action for x, y in zip(a.choices, b.choices, strict=True)),
            "v3 vs v2": versus(b, a),
        }
        print(f"{dataset:14} v2 {out[dataset]['v2']} → v3 {out[dataset]['v3']}")
    return out


def v3_model(v2_raw: dict[str, Any], tau: float) -> dict[str, Any]:
    """v3 = v2 + 再感知的门槛：要导出的 JSON。"""
    return {
        **v2_raw,
        "version": "n11-two-stage-v3",
        "see_gate": {"cues": IMAGE_FLOOR, "caption_min": tau},
        "changes": (
            v2_raw.get("changes", "")
            + "；v3（N11 之后，D-058）：有图的空间里，问题带画面线索或图片描述的检索相似度不低于"
            f" {tau} 才考虑看图"
        ),
        "created": datetime.now().strftime("%Y-%m-%d"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_export_flags(parser, DEFAULT_MODEL.name)
    args = parser.parse_args()
    v2 = LearnedController.load(V2_MODEL)
    tau, dev = choose_tau(v2)
    model = v3_model(json.loads(V2_MODEL.read_text(encoding="utf-8")), tau)
    if finish_export(args, DEFAULT_MODEL, model):  # 在写任何结果之前：对不上时都不动
        print(f"导出 v3：{DEFAULT_MODEL}")
    v3 = gated(v2, tau)
    result = {"rule": "开发集准确率不低于 v2 的最大 τ", "tau": tau, "dev": dev}
    result["test_post_hoc"] = test_replay(v2, v3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "see_gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"结果：{OUT / 'see_gate.json'}")
    multiplicity.update(OUT / "see_gate.json", multiplicity.n11_see_gate)  # → see_gate-holm.json


if __name__ == "__main__":
    main()
