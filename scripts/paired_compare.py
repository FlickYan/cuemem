"""配对比较：同一批题目上两种方法谁对谁错，并做精确符号检验（McNemar 精确版）。

准确率的置信区间是各自独立算的，两种方法在同一批题上比较时，配对检验更灵敏：
只看"一个对、一个错"的题，在"两种方法一样好"的假设下，这些题里谁赢应当各占一半。一次比较好几个
方法、好几种题型时，另列 Holm 校正后的 p（总表各方法是一族，分题型的表整张是一族）。

另外把知识更新题拆成"问现在的值"与"问以前的值"两组：后者只有保留了历史版本才答得出（H4）。

用法：
    uv run python scripts/paired_compare.py runs/eval/dev100-n3 --base B2 \\
        --others B2F,B2F-keys,B2F-current,B2F-plain,B2F-noprofile
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from cuemem.evals.routing import holm, sign_test

HISTORY_WORDS = re.compile(
    r"\b(previous|previously|before|used to|originally|at first|initially"
    r"|when I (just )?started)\b",
    re.IGNORECASE,
)


def load(run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[row["sample_id"]][row["method"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--base", default="B2")
    parser.add_argument("--others", default="B2F")
    args = parser.parse_args()
    rows = load(args.run_dir)
    others = [m.strip() for m in args.others.split(",") if m.strip()]

    # Holm 校正：总表里各方法对基准是一族，分题型的表整张是一族
    print(f"## 配对比较（基准 {args.base}）\n")
    print("| 方法 | 题数 | 基准对、它错 | 基准错、它对 | 净提升 | 符号检验 p | Holm 校正后 |")
    print("|---|---|---|---|---|---|---|")
    overall: dict[str, tuple[int, int, int]] = {}
    for method in others:
        pairs = [(r[args.base], r[method]) for r in rows.values() if args.base in r and method in r]
        lost = sum(1 for b, m in pairs if b["correct"] and not m["correct"])
        won = sum(1 for b, m in pairs if m["correct"] and not b["correct"])
        overall[method] = (len(pairs), won, lost)
    # 没有一题分出胜负的（+0 / −0）不算一次检验
    adjusted = holm({m: sign_test(w, lo) for m, (_, w, lo) in overall.items() if w + lo})
    for method, (n, won, lost) in overall.items():
        print(
            f"| {method} | {n} | {lost} | {won} | {won - lost:+d} "
            f"| {sign_test(won, lost):.3f} | {adjusted.get(method, 1.0):.3f} |"
        )

    print("\n## 分题型的配对结果（它对、基准错 / 基准对、它错，符号检验 p，整张表 Holm 校正后）\n")
    types = sorted({r[args.base]["question_type"] for r in rows.values() if args.base in r})
    print("| 题型 | " + " | ".join(others) + " |")
    print("|---|" + "---|" * len(others))
    cells: dict[tuple[str, str], tuple[int, int]] = {}
    for qtype in types:
        for method in others:
            pairs = [
                (r[args.base], r[method])
                for r in rows.values()
                if args.base in r and method in r and r[args.base]["question_type"] == qtype
            ]
            won = sum(1 for b, m in pairs if m["correct"] and not b["correct"])
            lost = sum(1 for b, m in pairs if b["correct"] and not m["correct"])
            cells[(qtype, method)] = (won, lost)
    decided = {f"{t}|{m}": sign_test(w, lo) for (t, m), (w, lo) in cells.items() if w + lo}
    adjusted = holm(decided)
    for qtype in types:
        row = []
        for method in others:
            won, lost = cells[(qtype, method)]
            p = sign_test(won, lost)
            holm_p = adjusted.get(f"{qtype}|{method}", 1.0)
            row.append(f"+{won} / −{lost}（p {p:.2f}，Holm {holm_p:.2f}）")
        print(f"| {qtype} | " + " | ".join(row) + " |")

    print("\n## 知识更新题：问现在的值 vs 问以前的值\n")
    methods = [args.base, *others]
    groups: dict[str, list[dict[str, dict[str, Any]]]] = {"现在": [], "以前": []}
    for r in rows.values():
        base = r.get(args.base)
        if base is None or base["question_type"] != "knowledge-update" or base["is_abstention"]:
            continue
        groups["以前" if HISTORY_WORDS.search(base["question"]) else "现在"].append(r)
    print("| 分组 | 题数 | " + " | ".join(methods) + " |")
    print("|---|---|" + "---|" * len(methods))
    for name, items in groups.items():
        cells = []
        for method in methods:
            scored = [r[method]["correct"] for r in items if method in r]
            cells.append(f"{sum(scored)}/{len(scored)}" if scored else "—")
        print(f"| {name} | {len(items)} | " + " | ".join(cells) + " |")
    for r in groups["以前"]:
        print(f"- 以前：{r[args.base]['question']}")


if __name__ == "__main__":
    main()
