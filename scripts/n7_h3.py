"""N7：换一个回答模型（reader），比较控制器与模型自主检索的稳定性（假设 H3）。

同一批开发题（LongMemEval dev100、LoCoMo 前 3 段对话）分别用 deepseek-flash（主实验）与
qwen3.8-flash 作回答模型，各跑 L0 / L1 / L2 与 B3（模型自主检索）。

- 控制器的选档只看探针特征，与回答模型无关：B6 用部署的模型（在 deepseek-flash 的标注上训练）
  直接选档，B6r 用规则选档，换了回答模型只换"这一档答没答对"；
- B3 的检索决定由回答模型自己做，换了模型，决定也跟着变。

注意：qwen3.8-flash 与判分模型 qwen3.8-max 同一家族（D-008 要求避开），同一判分模型下比较方法之间的
差异仍然有效，但两个回答模型之间的绝对准确率不能直接比。

用法（先跑完两组评测，探针特征与回答模型无关，直接复用主实验的）：
    uv run python scripts/n7_h3.py
"""

from __future__ import annotations

import json
import shutil
from typing import Any

from cuemem.config import PROJECT_ROOT
from cuemem.evals import routing as R
from cuemem.read.controller import V1_MODEL, LearnedController

RUNS = PROJECT_ROOT / "runs" / "eval"
OUT = PROJECT_ROOT / "docs" / "results" / "n7-controller"
READERS = {
    "deepseek-flash": {"longmemeval_s": "dev100-n7", "locomo": "locomo-3conv-n7"},
    "qwen3.8-flash": {"longmemeval_s": "dev100-n7-qflash", "locomo": "locomo-3conv-n7-qflash"},
}


def crossfit_b6(items: list[R.Item], actions: dict[str, str], lam: float) -> R.PolicyResult:
    """B6 的选档：主实验交叉拟合的结果（每道题都来自没见过它的模型，在 deepseek-flash 的标注上
    训练）。选档只看探针特征，与回答模型无关；换了回答模型只换"这一档答没答对"。"""
    return R.PolicyResult("B6", [R.take(it, actions[it.sample_id]) for it in items], lam)


def row(policy: R.PolicyResult) -> dict[str, float]:
    return {"accuracy": policy.accuracy, "cost": policy.cost, "l2": policy.share("L2")}


def main() -> None:
    lam = LearnedController.load(V1_MODEL).lam
    actions = json.loads((RUNS / "n7-controller" / "b6_crossfit.json").read_text("utf-8"))
    results: dict[str, Any] = {"lambda": lam, "readers": {}}
    per_reader: dict[str, dict[str, dict[str, R.PolicyResult]]] = {}
    loaded: dict[str, dict[str, list[R.Item]]] = {}
    for reader, runs in READERS.items():
        loaded[reader] = {}
        for dataset, name in runs.items():
            run_dir = RUNS / name
            probe = run_dir / "probe.jsonl"
            if not probe.exists():  # 探针特征与回答模型无关：复用主实验的
                shutil.copy(RUNS / READERS["deepseek-flash"][dataset] / "probe.jsonl", probe)
            loaded[reader][dataset] = R.load_items(run_dir)
    for reader, runs in READERS.items():
        per_reader[reader] = {}
        for dataset in runs:
            # 两个回答模型都跑完的题（配对比较要逐题对齐）
            common = set.intersection(
                *(
                    {it.sample_id for it in loaded[r][dataset] if "B3" in it.references}
                    for r in READERS
                )
            )
            items = [it for it in loaded[reader][dataset] if it.sample_id in common]
            policies = {
                "always-L1": R.fixed(items, "L1"),
                "always-L2": R.fixed(items, "L2"),
                "B6r": R.rule(items),
                "B6": crossfit_b6(items, actions, lam),
            }
            b3 = R.reference(items, "B3")
            if b3 is not None:
                policies["B3"] = b3
            per_reader[reader][dataset] = policies
            results["readers"].setdefault(reader, {})[dataset] = {
                k: row(v) | {"n": len(v.choices)} for k, v in policies.items()
            }
    # 换回答模型前后，同一道题、同一策略的配对比较
    changes: dict[str, Any] = {}
    for dataset in READERS["deepseek-flash"]:
        changes[dataset] = {}
        before = per_reader["deepseek-flash"][dataset]
        after = per_reader["qwen3.8-flash"][dataset]
        for name in before:
            if name in after:
                a, b = after[name], before[name]
                wins, losses, p = R.paired(a, b)
                changes[dataset][name] = {
                    "delta": a.accuracy - b.accuracy,
                    "wins": wins,
                    "losses": losses,
                    "p": p,
                }
        for name in ("B6", "B6r", "always-L2"):
            wins, losses, p = R.paired(after[name], after["B3"])
            changes[dataset][f"{name}_vs_B3_qwen"] = {"wins": wins, "losses": losses, "p": p}
    results["changes"] = changes
    (OUT / "h3.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    for dataset in READERS["deepseek-flash"]:
        print(f"== {dataset}")
        for name in ("always-L1", "always-L2", "B6r", "B6", "B3"):
            cells = []
            for reader in READERS:
                r = results["readers"][reader][dataset][name]
                cells.append(f"{reader} {r['accuracy']:.1%} ${r['cost'] * 1000:.2f}")
            c = changes[dataset][name]
            print(f"  {name:10s} " + " | ".join(cells) + f" | Δ {c['delta']:+.1%} "
                  f"(+{c['wins']} / −{c['losses']}, p = {c['p']:.3f})")  # fmt: skip
        for name in ("B6", "B6r", "always-L2"):
            c = changes[dataset][f"{name}_vs_B3_qwen"]
            print(f"  qwen: {name} vs B3 +{c['wins']} / −{c['losses']} p = {c['p']:.3f}")


if __name__ == "__main__":
    main()
