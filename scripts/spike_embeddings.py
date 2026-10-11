"""N1 技术验证：通义向量接口的延迟、维度、稀疏向量与成本（需要 DASHSCOPE_API_KEY）。

用法：uv run python scripts/spike_embeddings.py [--models qwen3.7-text-embedding ...]
直接调用接口、不走缓存，所以每次运行都会产生极少量费用。
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from cuemem.config import load_settings
from cuemem.embed.dashscope import DashScopeTextEmbedder

DOCS = [
    "User: 我周末刚领养了一只金毛，取名叫 Max。\nAssistant: 恭喜！金毛性格温顺。",
    "User: I want a laptop under 1.5 kg with a budget of 8000 RMB.\nAssistant: Sure.",
    "User: 上个月我跳槽了，现在在新加坡一家电商公司做 AI 产品经理。",
    "User: Max 特别喜欢在东海岸公园散步。\nAssistant: 注意中午避开高温。",
] * 5  # 20 条，覆盖一个完整批次


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["text-embedding-v4", "qwen3.7-text-embedding"],
    )
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    settings = load_settings()
    api_key = settings.providers[settings.embedding.provider].api_key()
    if api_key is None:
        raise SystemExit("缺少 DASHSCOPE_API_KEY，请先写入 .env")

    print(
        "| 模型 | 单条查询 p50/p95 | 10 条文档一批 | dense 维度"
        " | sparse 平均非零项 | 查询-文档相似度 |"
    )
    print("|---|---|---|---|---|---|")
    for model in args.models:
        cfg = settings.embedding.model_copy(update={"model": model, "batch_size": 10})
        embedder = DashScopeTextEmbedder(cfg, api_key)
        try:
            single = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                query = embedder.embed(["Max 是什么品种的狗？"], "query")
                single.append((time.perf_counter() - started) * 1000)
            started = time.perf_counter()
            docs = embedder.embed(DOCS[:10], "document")
            batch_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            print(f"| {model} | 失败：{type(exc).__name__}: {str(exc)[:80]} | | | | |")
            continue
        sims = docs.dense @ query.dense[0]
        nnz = np.mean([len(s) for s in docs.sparse]) if docs.sparse else 0
        print(
            f"| {model} | {np.percentile(single, 50):.0f} / {np.percentile(single, 95):.0f} ms"
            f" | {batch_ms:.0f} ms | {docs.dense.shape[1]} | {nnz:.1f}"
            f" | 金毛 {sims[0]:.3f} vs 笔记本 {sims[1]:.3f} |"
        )


if __name__ == "__main__":
    main()
