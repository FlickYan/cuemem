"""L2 纯检索评测：在开发集上比较 L2 各组件对召回的贡献，不调用 reader。

只用到通义的模型（规划与充分性检查用 qwen3.8-flash，重排用 qwen3-rerank，向量 qwen3.7），
不受 DeepSeek 高峰时段影响；规划、检查与重排的结果都进响应缓存，重跑几乎免费。
依赖评测运行留下的磁盘空间（paths.eval_spaces）。

用法：
    uv run python scripts/tune_l2.py --split dev100
    uv run python scripts/tune_l2.py --split dev100 --only "L1 6k" "L2 规划"  # 只跑部分配置
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from cuemem.config import load_settings
from cuemem.core.trace import start_trace
from cuemem.embed import build_embedder
from cuemem.evals.datasets import Sample, dataset_path, load_longmemeval
from cuemem.evals.runner import EvalRunner, space_dir, usage_from_trace
from cuemem.evals.splits import load_split
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.memory import MemorySystem, build_reranker
from cuemem.read.graph import GraphConfig
from cuemem.read.l1 import L1Config, estimate_tokens
from cuemem.read.l2 import L2Config
from cuemem.write.consolidate import scope

L1_6K = L1Config(budget_tokens=6000, max_rounds=60)
L1_6K_DENSE = replace(L1_6K, sparse="none")
FIRST_PASS = L2Config(ppr=False, iterations=0)  # 只有规划与多查询融合
VARIANTS: dict[str, L1Config | L2Config] = {
    "L1 4k（B4）": L1Config(),
    "L1 6k / 60 轮（对照）": L1_6K,
    "L2 规则计划，无子查询（应与对照相同）": replace(FIRST_PASS, plan=False),
    "L2 规划，max 融合": FIRST_PASS,
    "L2 规划，rrf 融合": replace(FIRST_PASS, fusion="rrf"),
    "L2 规划 + 图扩散": replace(FIRST_PASS, ppr=True),
    "L2 规划 + 图扩散（只用抽取出的边）": replace(
        FIRST_PASS, ppr=True, graph=GraphConfig(mention_edges=False, similar_edges=False)
    ),
    "L2 规划 + 图扩散（不连相似事实）": replace(
        FIRST_PASS, ppr=True, graph=GraphConfig(similar_edges=False)
    ),
    "L2 规划 + 图扩散 ×1.0": replace(FIRST_PASS, ppr=True, ppr_weight=1.0),
    "L2 规划，不用 BM25": replace(FIRST_PASS, l1=L1_6K_DENSE),
    "L2 规划 + 重排": replace(FIRST_PASS, rerank=True),
    "L2 规划 + 重排，不用 BM25": replace(FIRST_PASS, rerank=True, l1=L1_6K_DENSE),
    "L1 6k + 重排": replace(FIRST_PASS, plan=False, rerank=True),
    "L1 6k + 重排，不用 BM25": replace(FIRST_PASS, plan=False, rerank=True, l1=L1_6K_DENSE),
    "L1 4k，不用 BM25": replace(L1Config(), sparse="none"),
    "L1 6k，不用 BM25": L1_6K_DENSE,
    "L2 规划 + 补检索 ×2": replace(FIRST_PASS, iterations=2),
    "L2 规划 + 补检索 ×2，不用 BM25": replace(FIRST_PASS, iterations=2, l1=L1_6K_DENSE),
    "L2 完整（规划 + 图扩散 + 补检索 ×2）": L2Config(),
}
METRICS = ("turn_recall_any", "turn_recall_all", "session_recall_all", "turn_ndcg")
FOCUS_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev100")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", nargs="*", default=None, help="只跑名字包含这些词的配置")
    args = parser.parse_args()

    settings = load_settings()
    ids = load_split("longmemeval_s", args.split)
    evalset = load_longmemeval(dataset_path(settings.paths.datasets, "longmemeval_s"), ids=ids)
    samples = [s for s in evalset.samples if not s.is_abstention]
    ledger = Ledger(settings.paths.ledger, run_id=f"tune_l2:{args.split}")
    embedder = build_embedder(settings, ledger=ledger)
    llm = LLMClient(settings, cache=ResponseCache(settings.paths.llm_cache), ledger=ledger)
    reranker = build_reranker(settings, llm=llm, ledger=ledger)
    variants = {
        name: cfg
        for name, cfg in VARIANTS.items()
        if not args.only or any(word in name for word in args.only)
    }

    def evaluate(sample: Sample) -> dict[str, Any]:
        directory = space_dir(settings, "longmemeval_s", embedder, sample.space_key)
        system = MemorySystem.on_disk(
            settings, directory, embedder=embedder, llm=llm, ledger=ledger, reranker=reranker
        )
        space_id = f"longmemeval_s:{sample.space_key}"
        sessions = evalset.sessions_of(sample)
        out: dict[str, Any] = {"type": sample.question_type, "rows": {}}
        try:
            for name, cfg in variants.items():
                with start_trace("tune", method=name) as trace:
                    if isinstance(cfg, L1Config):
                        result = system.l1.search(
                            space_id, sample.question, sample.question_time, cfg
                        )
                        hits, details = result.rounds, {}
                    else:
                        deep = system.l2.search(
                            space_id,
                            sample.question,
                            sample.question_time,
                            cfg,
                            cache_scope=scope(space_id, "ask"),
                        )
                        hits, details = deep.rounds, deep.details()
                usage = usage_from_trace(trace)
                out["rows"][name] = {
                    "metrics": EvalRunner._retrieval(sample, sessions, hits) or {},
                    "rounds": len(hits),
                    "tokens": sum(estimate_tokens(h.unit.text) for h in hits),
                    "llm_calls": usage["llm_calls"],
                    "cost": usage["cost"],
                    "checks": len(details.get("checks", [])),
                    "followups": sum(1 for c in details.get("checks", []) if c.get("added_rounds")),
                    "ms": trace.total_ms or 0.0,
                }
        finally:
            system.close()
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, samples))

    print(f"## L2 纯检索指标（{args.split}，不含拒答题，{len(samples)} 题）\n")
    print(
        "| 配置 | 发言召回（任一） | 发言召回（全部） | 会话召回（全部） | 发言 NDCG "
        "| 平均轮数 | 平均 tokens | LLM 调用 / 题 | 名义成本 / 题 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for name in variants:
        rows = [r["rows"][name] for r in results]
        values = [statistics.mean(row["metrics"][m] for row in rows) for m in METRICS]
        print(
            f"| {name} | "
            + " | ".join(f"{v:.1%}" for v in values)
            + f" | {statistics.mean(r['rounds'] for r in rows):.1f}"
            + f" | {statistics.mean(r['tokens'] for r in rows):,.0f}"
            + f" | {statistics.mean(r['llm_calls'] for r in rows):.2f}"
            + f" | ${statistics.mean(r['cost'] for r in rows):.5f} |"
        )

    print("\n## 分题型的发言召回（全部）\n")
    print("| 配置 | " + " | ".join(FOCUS_TYPES) + " |")
    print("|---|" + "---|" * len(FOCUS_TYPES))
    for name in variants:
        cells = []
        for qtype in FOCUS_TYPES:
            values = [
                r["rows"][name]["metrics"]["turn_recall_all"] for r in results if r["type"] == qtype
            ]
            cells.append(f"{statistics.mean(values):.1%}" if values else "—")
        print(f"| {name} | " + " | ".join(cells) + " |")

    checks: dict[str, list[int]] = defaultdict(list)
    for r in results:
        for name, row in r["rows"].items():
            if row["checks"]:
                checks[name].append(row["followups"])
    if checks:
        print("\n## 充分性检查\n")
        for name, followups in checks.items():
            share = sum(1 for f in followups if f) / len(followups)
            mean_followups = statistics.mean(followups)
            print(f"- {name}：{share:.0%} 的题至少补检索了一次，平均 {mean_followups:.2f} 次")
    latency = {name: sorted(r["rows"][name]["ms"] for r in results) for name in variants}
    print("\n## 耗时（含 LLM 调用；命中缓存时很快）\n")
    for name, values in latency.items():
        print(f"- {name}：p50 {values[len(values) // 2]:.0f} ms")
    ledger.close()


if __name__ == "__main__":
    main()
