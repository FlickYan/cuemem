"""N11：外部基线 Mem0 与 CueMem 同框对比（只读结果，不调用 API）。

Mem0 在 LoCoMo test200r 与 LongMemEval test100r 上各跑一次（方法 M0，见 evals/mem0_baseline.py）；
CueMem 各方法取 N9 测试集运行里同一批题的结果（同一个回答模型、同一个判分模型、同一套提示）。
每个运行里对 Mem0 的几个配对检验是一族，做 Holm 校正，写到 mem0-holm.json。

    MEM0="--with mem0ai==2.2.1 --with spacy --with fastembed --with en_core_web_sm@$SPACY_SM"
    uv run $MEM0 cuemem eval run --dataset locomo --split test200r --methods M0 \\
        --name locomo-test200r-mem0
    uv run $MEM0 cuemem eval run --dataset longmemeval_s --split test100r --methods M0 \\
        --name test100r-mem0
    uv run python scripts/n11_mem0.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from cuemem.config import load_settings
from cuemem.evals.report import load_rows
from cuemem.evals.routing import sign_test

sys.path.insert(0, str(Path(__file__).parent))
import multiplicity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "n11-improvements" / "mem0.json"
# Mem0 的运行 → （数据集，N9 的运行）
RUNS = {
    "locomo-test200r-mem0": ("locomo", "locomo-test7-n9"),
    "test100r-mem0": ("longmemeval_s", "test400-n9"),
}
CUEMEM = ["B1", "B2", "B4-nobm25", "B5", "B6"]
LABELS = {
    "M0": "Mem0 (OSS 2.2, hybrid search, top-30 per user)",
    "B1": "Full context",
    "B2": "Naive RAG",
    "B4-nobm25": "CueMem L1",
    "B5": "CueMem L2",
    "B6": "CueMem controller",
}


def latest(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["sample_id"], r["method"]): r for r in rows}


def write_cost(settings: Any, dataset: str, original: str, keys: set[str]) -> dict[str, Any]:
    """写入的花费：Mem0 取各空间 build.json；CueMem 取 N9 运行的写入与巩固统计（首次构建）。"""
    mem0 = {"cost": 0.0, "spent": 0.0, "memories": 0, "llm_calls": 0, "spaces": 0}
    root = settings.paths.eval_spaces / "mem0"
    for key in keys:
        path = root / re.sub(r"[^A-Za-z0-9_.-]", "_", f"{dataset}:{key}") / "build.json"
        if path.exists():
            build = json.loads(path.read_text("utf-8"))
            mem0["spaces"] += 1
            for field in ("cost", "spent", "memories", "llm_calls"):
                mem0[field] += build.get(field, 0)
    cuemem = {"cost": 0.0, "llm_calls": 0, "spaces": 0}
    stages: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_rows(settings.paths.eval_runs / original, "ingest.jsonl"):
        if row["space_key"] in keys:  # 续跑会把复用空间的统计再记一遍：每个空间每个阶段取最后一条
            stages[(row["space_key"], row["stage"])] = row
    for (_, stage), row in stages.items():
        cuemem["cost"] += row.get("cost", 0.0)
        cuemem["llm_calls"] += row.get("llm_calls", 0)
        cuemem["spaces"] += stage == "consolidate"
    return {"mem0": mem0, "cuemem": cuemem}


def main() -> None:
    settings = load_settings()
    result: dict[str, Any] = {"labels": LABELS}
    for run, (dataset, original) in RUNS.items():
        mem0 = latest(load_rows(settings.paths.eval_runs / run, "results.jsonl"))
        if not mem0:
            print(f"{run}：还没有结果，跳过")
            continue
        ids = sorted({sid for sid, _ in mem0})
        base = latest(load_rows(settings.paths.eval_runs / original, "results.jsonl"))
        cells = {**mem0, **{k: v for k, v in base.items() if k[0] in set(ids)}}
        methods: dict[str, Any] = {}
        for method in ["M0", *CUEMEM]:
            rows = [cells[(sid, method)] for sid in ids if (sid, method) in cells]
            if not rows:
                continue
            by_type: dict[str, list[bool]] = defaultdict(list)
            for row in rows:
                by_type[row["question_type"]].append(bool(row["correct"]))
            methods[method] = {
                "n": len(rows),
                "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 4),
                "cost_per_1k": round(sum(r["usage"]["cost"] for r in rows) / len(rows) * 1000, 2),
                "by_type": {t: round(sum(v) / len(v), 3) for t, v in sorted(by_type.items())},
            }
        paired = {}
        for other in ("B4-nobm25", "B2", "B6"):
            both = [sid for sid in ids if (sid, "M0") in cells and (sid, other) in cells]
            ours = {s: bool(cells[(s, other)]["correct"]) for s in both}
            theirs = {s: bool(cells[(s, "M0")]["correct"]) for s in both}
            wins = sum(1 for s in both if ours[s] and not theirs[s])
            losses = sum(1 for s in both if not ours[s] and theirs[s])
            paired[f"{other}_vs_M0"] = {
                "wins": wins,
                "losses": losses,
                "p": sign_test(wins, losses),
            }
        keys = {cells[(sid, "M0")]["space_key"] for sid in ids if (sid, "M0") in cells}
        recall = [
            r["retrieval"]["session_recall_all"]
            for (_, m), r in mem0.items()
            if m == "M0" and r.get("retrieval") and "session_recall_all" in r["retrieval"]
        ]
        result[run] = {
            "dataset": dataset,
            "questions": len(ids),
            "errors": sum(1 for _ in load_rows(settings.paths.eval_runs / run, "errors.jsonl")),
            "methods": methods,
            "paired": paired,
            "write": write_cost(settings, dataset, original, keys),
            "mem0_session_recall": round(sum(recall) / len(recall), 3) if recall else None,
        }
        print(f"\n{run}（{len(ids)} 题）")
        for method, row in methods.items():
            print(f"  {LABELS[method]:<40} {row['accuracy']:.1%}  ${row['cost_per_1k']:.2f} / 1k q")
        for name, row in paired.items():
            print(f"  {name}: +{row['wins']} / -{row['losses']}  p={row['p']:.3g}")
        print(f"  写入：{result[run]['write']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"→ {OUT.relative_to(ROOT)}")
    multiplicity.update(OUT, multiplicity.n11_mem0)  # Holm 校正 → mem0-holm.json


if __name__ == "__main__":
    main()
