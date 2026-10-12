"""向量模型对比（D-014 的后续）：同一批开发集题目，只换向量模型，比较"轮"上 dense 检索的召回。

为了单独衡量向量模型本身，只用最朴素的检索（B2：轮的 dense top-k），不用事实、BM25 与重排。
默认从 dev100 分层抽 30 道题（不碰 test400）；v4 的向量都在缓存里，新模型需要实际向量化。

用法：
    uv run python scripts/compare_embeddings.py --questions 30 \\
        --models text-embedding-v4,qwen3.7-text-embedding
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from collections import defaultdict
from typing import Any

from cuemem.config import load_settings
from cuemem.embed import build_embedder
from cuemem.evals.datasets import dataset_path, load_longmemeval, to_conversation
from cuemem.evals.runner import EvalRunner
from cuemem.evals.splits import load_split
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.memory import MemorySystem

METRICS = ("turn_recall_any", "turn_recall_all", "session_recall_all", "turn_ndcg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--models", default="text-embedding-v4,qwen3.7-text-embedding")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = load_settings()
    ids = load_split("longmemeval_s", "dev100")
    evalset = load_longmemeval(dataset_path(base.paths.datasets, "longmemeval_s"), ids=ids)
    pool = [s for s in evalset.samples if not s.is_abstention]
    rng = random.Random(args.seed)
    by_type: dict[str, list[Any]] = defaultdict(list)
    for sample in pool:
        by_type[sample.question_type].append(sample)
    picked: list[Any] = []
    for items in by_type.values():
        rng.shuffle(items)
    while len(picked) < args.questions and any(by_type.values()):  # 按题型轮流抽，保持分层
        for items in by_type.values():
            if items and len(picked) < args.questions:
                picked.append(items.pop())

    print(f"{len(picked)} 道题，前 k = {args.k} 轮\n")
    print(
        "| 向量模型 | 发言召回（任一） | 发言召回（全部） | 会话召回（全部） | 发言 NDCG"
        " | 向量化 tokens | 耗时 |"
    )
    print("|---|---|---|---|---|---|---|")
    for model in [m.strip() for m in args.models.split(",")]:
        # 只比较 dense（语义检索质量）；稀疏向量在 N4 由 BM25 承担。v4 保持原配置以命中向量缓存
        sparse = base.embedding.sparse if model == base.embedding.model else False
        embedding = base.embedding.model_copy(update={"model": model, "sparse": sparse})
        settings = base.model_copy(update={"embedding": embedding})
        ledger = Ledger(None)
        embedder = build_embedder(settings, ledger=ledger)
        llm = LLMClient(settings)
        started = time.perf_counter()
        scores: dict[str, list[float]] = defaultdict(list)
        tokens = 0
        for sample in picked:
            space_id = f"cmp:{sample.space_key}"
            system = MemorySystem.ephemeral(settings, embedder=embedder, llm=llm, ledger=ledger)
            report = system.ingest(to_conversation(space_id, evalset.sessions_of(sample)))
            tokens += report.embed_tokens
            hits = system.search(space_id, sample.question, args.k)
            metrics = EvalRunner._retrieval(sample, evalset.sessions_of(sample), hits) or {}
            for name in METRICS:
                scores[name].append(metrics.get(name, 0.0))
            system.close()
        elapsed = time.perf_counter() - started
        cells = " | ".join(f"{statistics.mean(scores[m]):.1%}" for m in METRICS)
        print(f"| {model} | {cells} | {tokens:,} | {elapsed:.0f} 秒 |")


if __name__ == "__main__":
    main()
