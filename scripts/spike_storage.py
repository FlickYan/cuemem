"""N1 技术验证：Qdrant 本地模式在笔记本上的写入、检索延迟与内存占用。

用法：uv run python scripts/spike_storage.py [--sizes 1000 5000 20000] [--dim 1024]
不需要 API Key。使用随机向量，只测性能，不测检索质量。
"""

from __future__ import annotations

import argparse
import resource
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from cuemem.embed.base import SparseVector, l2_normalize
from cuemem.store.vector import VectorStore


def _timed_ms(fn: Callable[[], object], repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def _p(samples: list[float], q: float) -> float:
    return float(np.percentile(samples, q))


def _rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS 以字节为单位，Linux 以 KB 为单位
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def bench(n: int, dim: int, rng: np.random.Generator) -> dict[str, float]:
    dense = l2_normalize(rng.standard_normal((n, dim)).astype(np.float32))
    sparse = [
        SparseVector.from_pairs(
            zip(rng.integers(0, 250_000, 60).tolist(), rng.random(60).tolist(), strict=True)
        )
        for _ in range(n)
    ]
    ids = [f"u{i}" for i in range(n)]
    payloads = [{"kind": "round", "ts": float(i)} for i in range(n)]
    queries = l2_normalize(rng.standard_normal((30, dim)).astype(np.float32))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "qdrant"
        store = VectorStore(path, dim)
        started = time.perf_counter()
        for s in range(0, n, 256):
            store.upsert(
                "bench",
                ids[s : s + 256],
                dense[s : s + 256],
                sparse[s : s + 256],
                payloads[s : s + 256],
            )
        upsert_s = time.perf_counter() - started

        it = iter(range(10**9))
        dense_ms = _timed_ms(lambda: store.search_dense("bench", queries[next(it) % 30], 10), 30)
        sparse_ms = _timed_ms(lambda: store.search_sparse("bench", sparse[next(it) % n], 10), 30)
        filtered_ms = _timed_ms(
            lambda: store.search_dense(
                "bench", queries[next(it) % 30], 10, ts_range=(n * 0.25, n * 0.75)
            ),
            30,
        )
        store.close()

        started = time.perf_counter()
        reopened = VectorStore(path, dim)
        count = reopened.count("bench")
        reopen_s = time.perf_counter() - started
        reopened.close()
        assert count == n

    return {
        "n": n,
        "upsert_s": upsert_s,
        "dense_p50": _p(dense_ms, 50),
        "dense_p95": _p(dense_ms, 95),
        "sparse_p50": _p(sparse_ms, 50),
        "sparse_p95": _p(sparse_ms, 95),
        "filtered_p50": _p(filtered_ms, 50),
        "reopen_s": reopen_s,
        "peak_rss_mb": _rss_mb(),
        "mean_dense_ms": statistics.mean(dense_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 20000])
    parser.add_argument("--dim", type=int, default=1024)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    print(
        "| 点数 | 写入耗时 | dense p50/p95 | sparse p50/p95"
        " | 过滤 dense p50 | 重新打开 | 峰值内存 |"
    )
    print("|---|---|---|---|---|---|---|")
    for n in args.sizes:
        r = bench(n, args.dim, rng)
        print(
            f"| {n} | {r['upsert_s']:.1f} s | {r['dense_p50']:.1f} / {r['dense_p95']:.1f} ms"
            f" | {r['sparse_p50']:.1f} / {r['sparse_p95']:.1f} ms | {r['filtered_p50']:.1f} ms"
            f" | {r['reopen_s']:.1f} s | {r['peak_rss_mb']:.0f} MB |"
        )


if __name__ == "__main__":
    main()
