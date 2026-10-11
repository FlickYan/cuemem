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


def test_v1_database_is_migrated(tmp_path: Path) -> None:
    import sqlite3
    from datetime import datetime

    from cuemem.core.models import Turn

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE turns (id TEXT PRIMARY KEY, space_id TEXT NOT NULL, session_id TEXT NOT NULL,"
        " idx INTEGER NOT NULL, role TEXT NOT NULL, text TEXT NOT NULL, ts TEXT NOT NULL,"
        " media_ids TEXT NOT NULL DEFAULT '[]', UNIQUE (space_id, session_id, idx));"
        "PRAGMA user_version = 1;"
    )
    conn.close()

    db = MemoryDB(path)  # 打开时自动补上 speaker 列
    turn = Turn(
        id="t1",
        space_id="s",
        session_id="x",
        idx=0,
        role="user",
        text="hi",
        ts=datetime(2023, 1, 1),
        speaker="Alice",
    )
    assert db.add_turns([turn]) == 1
    assert db.get_turns("s")[0].speaker == "Alice"


def test_llm_cache_gains_scope_column(tmp_path: Path) -> None:
    """N3 之前建的响应缓存没有 scope 列：打开时自动补上，旧条目照常可读。"""
    import sqlite3

    from cuemem.llm.cache import ResponseCache

    path = tmp_path / "llm.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE llm_cache (key TEXT PRIMARY KEY, model TEXT NOT NULL,"
        " response TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute("INSERT INTO llm_cache (key, model, response) VALUES ('k', 'm', '{\"a\": 1}')")
    conn.commit()
    conn.close()

    cache = ResponseCache(path)
    assert cache.get("k") == {"a": 1}
    cache.put("k2", "m", {"b": 2}, scope="space|ask")
    assert cache.delete_scope("space|ask") == 1
    assert cache.get("k2") is None and cache.get("k") == {"a": 1}
