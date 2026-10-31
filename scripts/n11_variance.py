"""N11：同一配置再跑一遍，看结果有多少是随机波动。

N9 每个组合只跑了一次。这里关掉 LLM 缓存（不读也不写，N9 的缓存保持原样），在同样的记忆空间上
把几个主要方法重跑一遍：回答模型、L2 的规划、判分都重新调用。比较两次运行：

- 准确率之差；逐题翻转（这次对、上次错，或反过来）的题数与符号检验；
- 翻转里有多少是回答文字没变、只是判分变了（判分的随机性），多少是回答本身变了。

报告里方法之间差几个点的结论，要和这里的波动幅度对照着看。

运行（12–14 点，DeepSeek 非高峰；N9 用的是 v1 控制器）：
    CUEMEM_LLM_CACHE_MODE=off CUEMEM_RETRIEVAL__CONTROLLER=controller_model_v1.json \\
        uv run cuemem eval run --dataset longmemeval_s --split test400 \\
        --methods B4-nobm25,B6 --name test400-n11-rerun --workers 8 --max-cost 10
    CUEMEM_LLM_CACHE_MODE=off uv run cuemem eval run --dataset locomo --split test7 \\
        --methods B4-nobm25 --name locomo-test7-n11-rerun --workers 8 --max-cost 10
    uv run python scripts/n11_variance.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cuemem.config import load_settings
from cuemem.evals.report import load_rows
from cuemem.evals.routing import sign_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "n11-improvements" / "variance.json"
PAIRS = {  # 重跑的运行 → N9 的原运行
    "test400-n11-rerun": "test400-n9",
    "locomo-test7-n11-rerun": "locomo-test7-n9",
}


def latest(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["sample_id"], r["method"]): r for r in rows}


def compare(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(first) & set(second))
    methods = sorted({m for _, m in keys})
    out: dict[str, Any] = {}
    for method in methods:
        ids = [k for k in keys if k[1] == method]
        a = [bool(first[k]["correct"]) for k in ids]
        b = [bool(second[k]["correct"]) for k in ids]
        gained = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
        lost = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
        same_text = [
            k for k in ids if first[k]["response"].strip() == second[k]["response"].strip()
        ]
        judge_flips = sum(1 for k in same_text if first[k]["correct"] != second[k]["correct"])
        out[method] = {
            "n": len(ids),
            "n9": round(sum(a) / len(ids), 4),
            "rerun": round(sum(b) / len(ids), 4),
            "delta_pts": round((sum(b) - sum(a)) / len(ids) * 100, 1),
            "gained": gained,
            "lost": lost,
            "flip_rate": round((gained + lost) / len(ids), 4),
            "p": round(sign_test(gained, lost), 4),
            "same_response": len(same_text),
            "judge_only_flips": judge_flips,
        }
    return out


def main() -> None:
    runs = load_settings().paths.eval_runs
    rows = {
        name: latest(load_rows(runs / name, "results.jsonl"))
        for pair in PAIRS.items()
        for name in pair
    }
    if missing := [name for name, found in rows.items() if not found]:
        # 缺一个就不写：以前跳过缺的、照样写，可选的重跑没跑时把发表的结果清空成 {}
        raise SystemExit(
            f"还没有结果：{'、'.join(missing)}（重跑的命令见脚本说明）；没有写 {OUT.name}"
        )
    result: dict[str, Any] = {}
    for rerun, original in PAIRS.items():
        first, second = rows[original], rows[rerun]
        result[rerun] = {"original": original, "methods": compare(first, second)}
        for method, row in result[rerun]["methods"].items():
            print(
                f"{rerun} {method}: N9 {row['n9']:.1%} → 重跑 {row['rerun']:.1%} "
                f"（{row['delta_pts']:+.1f}），翻转 {row['gained']}+{row['lost']} 题"
                f"（{row['flip_rate']:.1%}），其中只是判分变了 {row['judge_only_flips']}"
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"→ {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
