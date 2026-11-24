"""本地模式向量库里分数完全相同的点（第十轮体检 Q21）：关掉再打开之后，取到的前 k 个与写入它们
的那个进程里相同。qdrant-client 的本地模式按数组位置排相同的分数（np.argsort），位置就是读入的
顺序：1.19.1 按写入的顺序读回（SELECT point FROM points），1.20.0 改成倒着读
（ORDER BY rowid DESC），重启之后同分的点换了次序，连前 k 个里有谁都变了（重复的文本就是这种
情形），发表的结果按 1.19.1 重放。所以 uv.lock 停在 1.19.x（tests/test_ci_pins.py 核对，D-072）；
要升级，先让这里的比较通过，再做包含关掉重开的检索比较。只比 ID，不比浮点分数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cuemem.store import VectorStore

COPIES = 20  # 每个向量的份数：远多于 k，前 k 个全在同分的点里挑


def test_tie_order_survives_reopen(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(size=(8, 16)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    n = len(base) * COPIES
    ids = [f"u{i:03d}" for i in range(n)]
    dense = np.stack([base[i % len(base)] for i in range(n)])
    payloads = [{"kind": "round", "ts": float(i)} for i in range(n)]

    store = VectorStore(tmp_path / "qdrant", 16)
    store.upsert("s", ids, dense, None, payloads)
    written = [[hit.unit_id for hit in store.search_dense("s", query, 10)] for query in base]
    store.close()

    store = VectorStore(tmp_path / "qdrant", 16)
    try:
        reopened = [[hit.unit_id for hit in store.search_dense("s", query, 10)] for query in base]
    finally:
        store.close()
    # 每个查询的前 10 个都是与它相同的那 20 份里的（分数并列），不是别的向量
    for i, hits in enumerate(written):
        assert len(hits) == 10 and all(int(hit[1:]) % len(base) == i for hit in hits), hits
    assert reopened == written
