"""N12：LongMemEval 会话 ID 泄漏有多大影响（只读结果，不调用 API）。

泄漏：LongMemEval 的证据会话 ID 都以 "answer_" 开头（拒答题还带 "_abs"），其他会话是
sharegpt_ / ultrachat_ / 哈希；回答模型看到的每条检索结果标题里写着 "session <ID>"。所有检索
方法（B2、CueMem L1 / L2 / 控制器、Mem0 与外部基线）都受影响，全上下文（B1）按序号给会话编号，
不受影响。修复（D-067）在加载时按日期改成 session_<i>。

同一批题（test100r）跑三臂，回答与判分相同：
- raw：原始 ID（CUEMEM_EVAL_RAW_SESSION_IDS=1，修复之前的条件，即 N9 / N11 发表数字的条件）；
- anon：中性 ID（修复后的默认）；
- label：在 anon 的空间上，只把证据标题里的 ID 换回原始 ID（CUEMEM_EVAL_RAW_SESSION_IDS=label）。

label 对 anon 只差标题里的字，是泄漏本身的效应；raw 对 label 标签相同、只差空间里的 ID
（同一会话内各轮的先后按含 ID 的哈希排，换了 ID 顺序就变），是换写法带来的噪声；raw 对 anon
两者都有。

协议：
- orig：原协议（默认配置：抽取 qwen3.8-flash、向量 qwen3.7-text-embedding、回答 deepseek-flash、
  判分 qwen3.8-max），写入在 data/eval-spaces-default；运行名 orig-lme-*；
- pq：只用通义（CUEMEM_PRESET=qwen，回答 qwen3.8-flash），写入在 data/eval-spaces-qwen；运行名
  pq-lme-*（raw 一臂用修复之前的代码跑，结果与 =1 相同，见 tests/test_eval_leak.py）。

    uv run python scripts/n12_leak.py

结果写到 docs/results/n12-leak/summary.json。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from cuemem.evals.routing import sign_test

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "eval"
OUT = ROOT / "docs" / "results" / "n12-leak" / "summary.json"

# 协议 → 臂 → 这一臂的运行（CueMem 与 Mem0 分开跑）
ARMS: dict[str, dict[str, list[str]]] = {
    "orig": {
        "raw": ["orig-lme-raw-cuemem", "orig-lme-raw-mem0"],
        "anon": ["orig-lme-cuemem", "orig-lme-mem0"],
        "label": ["orig-lme-label-cuemem", "orig-lme-label-mem0"],
    },
    "pq": {
        "raw": ["pq-lme-raw-cuemem", "pq-lme-warm-b1", "pq-lme-raw-mem0"],
        "anon": ["pq-lme-cuemem", "pq-lme-mem0"],
        "label": ["pq-lme-label-cuemem", "pq-lme-label-mem0"],
    },
}
RETRIEVAL = ["B2", "B4-nobm25", "B5", "B6", "M0"]
LABELS = {
    "B1": "Full context",
    "B2": "Naive RAG",
    "B4-nobm25": "CueMem L1",
    "B5": "CueMem L2",
    "B6": "CueMem controller",
    "M0": "Mem0 2.2.1",
}
COMPARISONS = [  # （甲，乙，含义）
    ("label", "anon", "leak only (same spaces, only the evidence-header IDs differ)"),
    ("raw", "label", "renaming noise only (same labels, rounds reordered)"),
    ("raw", "anon", "leak + renaming noise (pre-fix vs post-fix)"),
]


def load(run: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = RUNS / run / "results.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return {(r["method"], r["sample_id"]): r for r in rows}


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    return round(sum(bool(r["correct"]) for r in rows) / len(rows), 4) if rows else None


def main() -> None:
    summary: dict[str, Any] = {}
    for protocol, arms in ARMS.items():
        by_arm = {
            arm: {k: v for run in runs for k, v in load(run).items()} for arm, runs in arms.items()
        }
        methods = sorted({m for rows in by_arm.values() for m, _ in rows}, key=list(LABELS).index)
        section: dict[str, Any] = {"runs": arms, "methods": {}, "pooled": {}}
        pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for method in methods:
            entry: dict[str, Any] = {"label": LABELS.get(method, method), "arms": {}}
            for arm, rows in by_arm.items():
                mine = [r for (m, _), r in rows.items() if m == method]
                if not mine:
                    continue
                abst = [r for r in mine if r.get("is_abstention")]
                entry["arms"][arm] = {
                    "n": len(mine),
                    "accuracy": accuracy(mine),
                    "abstention_n": len(abst),
                    "abstention_accuracy": accuracy(abst),
                }
            entry["paired"] = {}
            for a, b, meaning in COMPARISONS:
                ra = {s: r for (m, s), r in by_arm[a].items() if m == method}
                rb = {s: r for (m, s), r in by_arm[b].items() if m == method}
                common = sorted(ra.keys() & rb.keys())
                if not common:
                    continue
                wins = sum(1 for s in common if ra[s]["correct"] and not rb[s]["correct"])
                losses = sum(1 for s in common if rb[s]["correct"] and not ra[s]["correct"])
                entry["paired"][f"{a}_vs_{b}"] = {
                    "n": len(common),
                    "wins": wins,
                    "losses": losses,
                    "p": round(sign_test(wins, losses), 4),
                    "meaning": meaning,
                }
                if method in RETRIEVAL:
                    pooled[f"{a}_vs_{b}"][0] += wins
                    pooled[f"{a}_vs_{b}"][1] += losses
            section["methods"][method] = entry
        for key, (wins, losses) in pooled.items():
            questions = sum(
                section["methods"][m]["paired"][key]["n"]
                for m in methods
                if m in RETRIEVAL and key in section["methods"][m]["paired"]
            )
            section["pooled"][key] = {
                "wins": wins,
                "losses": losses,
                "net_points": round(100 * (wins - losses) / questions, 2) if questions else None,
                "note": "summed over retrieval methods on the same questions (not independent)",
            }
        summary[protocol] = section
        print(f"\n== {protocol}")
        for entry in section["methods"].values():
            accs = "  ".join(
                f"{arm} {100 * v['accuracy']:.1f}%"
                for arm, v in entry["arms"].items()
                if v["accuracy"]
            )
            pairs = "  ".join(
                f"{k}: +{v['wins']}/-{v['losses']} p={v['p']}" for k, v in entry["paired"].items()
            )
            print(f"  {entry['label']:<18} {accs}\n  {'':<18} {pairs}")
        for key, v in section["pooled"].items():
            print(f"  pooled {key}: +{v['wins']}/-{v['losses']} (net {v['net_points']} pt)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"\n→ {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
