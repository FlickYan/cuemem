"""纯检索评测：在开发集上比较各种检索配置的召回与排序质量，不调用 reader（不花 LLM 的钱）。

依赖评测运行留下的磁盘空间（paths.eval_spaces，先跑一次 `cuemem eval run --methods B2F`）。
查询向量命中缓存；没缓存的查询会调用一次向量接口（通义，不受 DeepSeek 高峰影响）。

用法：
    uv run python scripts/tune_l1.py --split dev100              # 默认对比一组配置
    uv run python scripts/tune_l1.py --split dev100 --grid       # 网格搜索融合权重
    uv run python scripts/tune_l1.py --split dev100 --latency    # 测 L1 的检索延迟（不走向量缓存）
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from cuemem.config import load_settings
from cuemem.embed import build_embedder
from cuemem.evals.datasets import Sample, dataset_path, load_longmemeval
from cuemem.evals.runner import EvalRunner, space_dir
from cuemem.evals.splits import load_split
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.memory import MemorySystem
from cuemem.read.l1 import L1Config, L1Retriever

BASE = L1Config(
    sparse="bm25", sparse_weight=0.5, summary_weight=0.0, time_weight=0.0, order_weight=0.0
)
VARIANTS: dict[str, L1Config | str] = {
    "B2 dense 轮": "naive",
    "B2F 事实增强键": "facts",
    "按类型 RRF + BM25（对照）": L1Config(
        fusion="kind", sparse="bm25", time_weight=0, order_weight=0
    ),
    "L1 基础：模态内取最大 + BM25×0.5": BASE,
    "基础，不用稀疏路": replace(BASE, sparse="none"),
    "基础，稀疏改用 v4": replace(BASE, sparse="v4"),  # 只有向量模型带稀疏向量时才跑
    "基础 + 新旧顺序": replace(BASE, order_weight=0.5),
    "基础 + 时间区间（按宽度打折）": replace(BASE, time_weight=1.0),
    "基础 + 时间区间 ×2": replace(BASE, time_weight=2.0),
    "基础 + 时间区间 + 新旧顺序": replace(BASE, time_weight=1.0, order_weight=0.5),
    "基础 + 摘要": replace(BASE, summary_weight=0.5),
}
METRICS = ("turn_recall_any", "turn_recall_all", "session_recall_all", "turn_ndcg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev100")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--latency", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    ids = load_split("longmemeval_s", args.split)
    evalset = load_longmemeval(dataset_path(settings.paths.datasets, "longmemeval_s"), ids=ids)
    samples = [s for s in evalset.samples if not s.is_abstention]
    ledger = Ledger(None)
    embedder = build_embedder(settings, ledger=ledger, use_cache=not args.latency)
    llm = LLMClient(settings)  # 纯检索不会调用 LLM，只是组装 MemorySystem 需要

    # qwen3.7-text-embedding 只取 dense（D-014），空间里没有 v4 的词项向量，"v4"会退化成不用稀疏路
    has_sparse = settings.embedding.sparse
    variants: dict[str, L1Config | str] = {
        name: cfg
        for name, cfg in VARIANTS.items()
        if has_sparse or not (isinstance(cfg, L1Config) and cfg.sparse == "v4")
    }
    if args.grid:
        variants = {}
        for sparse, spw, fw, sw, tw in itertools.product(
            ("v4" if has_sparse else "none", "bm25"),
            (0.5, 1.0),
            (0.8, 1.0, 1.2),
            (0.0, 0.5),
            (0.0, 1.0, 2.0),
        ):
            name = f"{sparse} sparse={spw} fact={fw} summary={sw} time={tw}"
            variants[name] = L1Config(
                sparse=sparse,  # type: ignore[arg-type]
                sparse_weight=spw,
                fact_weight=fw,
                summary_weight=sw,
                time_weight=tw,
            )
    # 纯检索比较固定条数（前 k 轮）下的召回，不用 token 预算
    variants = {
        name: (replace(cfg, k=args.k, budget_tokens=None) if isinstance(cfg, L1Config) else cfg)
        for name, cfg in variants.items()
    }

    def evaluate(sample: Sample) -> dict[str, Any]:
        directory = space_dir(settings, "longmemeval_s", embedder, sample.space_key)
        system = MemorySystem.on_disk(
            settings, directory, embedder=embedder, llm=llm, ledger=ledger
        )
        space_id = f"longmemeval_s:{sample.space_key}"
        sessions = evalset.sessions_of(sample)
        out: dict[str, Any] = {"type": sample.question_type, "metrics": {}, "latency": {}}
        try:
            l1 = L1Retriever(system.db, system.vectors, system.embedder)
            for name, cfg in variants.items():
                if cfg == "naive":
                    hits = system.retriever.search(space_id, sample.question, args.k)
                elif cfg == "facts":
                    hits = system.fact_retriever.search(space_id, sample.question, k=args.k).rounds
                else:
                    assert isinstance(cfg, L1Config)
                    result = l1.search(space_id, sample.question, sample.question_time, cfg)
                    hits = result.rounds
                    out["latency"][name] = result.timings_ms
                out["metrics"][name] = EvalRunner._retrieval(sample, sessions, hits) or {}
        finally:
            system.close()
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, samples))

    print(f"## 纯检索指标（{args.split}，不含拒答题，{len(samples)} 题，前 k = {args.k} 轮）\n")
    print("| 配置 | 发言召回（任一） | 发言召回（全部） | 会话召回（全部） | 发言 NDCG |")
    print("|---|---|---|---|---|")
    rows = []
    for name in variants:
        values = [statistics.mean(r["metrics"][name][m] for r in results) for m in METRICS]
        rows.append((name, values))
    if args.grid:
        rows.sort(key=lambda item: -item[1][2])
        rows = rows[:15]
    for name, values in rows:
        print(f"| {name} | " + " | ".join(f"{v:.1%}" for v in values) + " |")

    if not args.grid:
        print("\n## 分题型的发言 NDCG\n")
        types = sorted({r["type"] for r in results})
        print("| 配置 | " + " | ".join(types) + " |")
        print("|---|" + "---|" * len(types))
        for name in variants:
            cells = []
            for qtype in types:
                values = [r["metrics"][name]["turn_ndcg"] for r in results if r["type"] == qtype]
                cells.append(f"{statistics.mean(values):.1%}")
            print(f"| {name} | " + " | ".join(cells) + " |")

    latency: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for name, timing in r["latency"].items():
            latency[name]["total"].append(timing["total"])
            latency[name]["embed"].append(timing["embed"])
            latency[name]["local"].append(timing["total"] - timing["embed"])
    if args.latency and latency:
        print("\n## L1 检索延迟（含一次不走缓存的查询向量化）\n")
        print("每道题依次跑各配置，第一个用到 BM25 的配置包含为该空间现建 BM25 索引的时间。\n")
        print("| 配置 | 总计 p50 / p95 | 其中查询向量化 p50 / p95 | 本地检索与融合 p50 / p95 |")
        print("|---|---|---|---|")
        for name, parts in latency.items():
            cells = [
                f"{_quantile(parts[part], 0.5):.0f} / {_quantile(parts[part], 0.95):.0f} ms"
                for part in ("total", "embed", "local")
            ]
            print(f"| {name} | " + " | ".join(cells) + " |")


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


if __name__ == "__main__":
    main()
