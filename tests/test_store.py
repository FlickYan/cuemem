from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fakes import TOY_CONVERSATION

from cuemem.core.models import Conversation
from cuemem.embed import SparseVector
from cuemem.embed.base import l2_normalize
from cuemem.errors import ConfigMismatchError
from cuemem.store import MemoryDB, VectorStore
from cuemem.write import build_rounds, build_turns


def test_db_writes_are_idempotent(tmp_path: Path) -> None:
    db = MemoryDB(tmp_path / "memory.sqlite")
    session = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8")).sessions[0]
    turns = build_turns("demo", session)
    units = build_rounds(turns)
    db.ensure_space("demo")
    db.add_session("demo", session.session_id, session.started_at)

    assert db.add_turns(turns) == len(turns)
    assert db.add_turns(turns) == 0
    assert db.add_units(units) == len(units)
    assert db.add_units(units) == 0
    assert db.missing_unit_ids("demo", [units[0].id, "nope"]) == ["nope"]
    fetched = db.get_units("demo", [units[1].id, units[0].id])
    assert [u.id for u in fetched] == [units[1].id, units[0].id]  # 保持请求的顺序
    assert db.get_units("other-space", [units[0].id]) == []  # 记忆空间之间互相隔离
    assert db.list_spaces()[0]["turns"] == len(turns)


def test_meta_guard_detects_embedder_change(tmp_path: Path) -> None:
    db = MemoryDB(tmp_path / "memory.sqlite")
    db.check_or_set_meta("embedder", "model-a:8")
    db.check_or_set_meta("embedder", "model-a:8")
    with pytest.raises(ConfigMismatchError):
        db.check_or_set_meta("embedder", "model-b:16")


def test_vector_store_search_and_filters(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "qdrant", dim=4)
    dense = l2_normalize(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0.9, 0.1, 0, 0]], dtype=np.float32))
    sparse = [
        SparseVector(indices=[1, 2], values=[1.0, 0.5]),
        SparseVector(indices=[3], values=[1.0]),
        SparseVector(indices=[], values=[]),  # 没有稀疏向量的点也能写入
    ]
    payloads = [
        {"kind": "round", "ts": 100.0},
        {"kind": "fact", "ts": 200.0},
        {"kind": "round", "ts": 300.0},
    ]
    store.upsert("s", ["a", "b", "c"], dense, sparse, payloads)

    assert [h.unit_id for h in store.search_dense("s", dense[0], 2)] == ["a", "c"]
    assert [h.unit_id for h in store.search_dense("s", dense[0], 3, kinds=["fact"])] == ["b"]
    in_range = store.search_dense("s", dense[0], 3, ts_range=(150.0, None))
    assert [h.unit_id for h in in_range] == ["c", "b"]
    assert [h.unit_id for h in store.search_sparse("s", sparse[1], 3)] == ["b"]
    assert store.search_dense("other", dense[0], 3) == []

    store.upsert("s", ["a"], dense[:1], sparse[:1], payloads[:1])  # 相同 ID 覆盖，不会重复
    assert store.count("s") == 3
    store.close()

    reopened = VectorStore(tmp_path / "qdrant", dim=4)  # 持久化
    assert reopened.count("s") == 3
    reopened.delete_space("s")
    assert reopened.count("s") == 0
    reopened.close()
