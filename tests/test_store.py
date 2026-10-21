from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from fakes import TOY_CONVERSATION

from cuemem.core.models import Conversation, Fact, MemoryUnit, SessionIn, Turn, TurnIn, TurnMeta
from cuemem.embed import SparseVector
from cuemem.embed.base import l2_normalize
from cuemem.errors import ConfigMismatchError, StorageLockedError
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


def test_nested_transactions_commit_only_at_the_outermost(tmp_path: Path) -> None:
    """嵌套的事务：外层出错时里层的写入一起撤销（以前里层自己提交了，撤不回来）；里层出错只撤销
    里层，外层接着写、照常提交。在线写入把发言、"轮"与巩固任务放在一个事务里（D-068 之后）。"""
    path = tmp_path / "memory.sqlite"
    db = MemoryDB(path)
    session = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8")).sessions[0]
    turns = build_turns("demo", session)
    with pytest.raises(RuntimeError, match="boom"), db.transaction():
        db.ensure_space("demo")
        db.add_session("demo", session.session_id, session.started_at)
        db.add_turns(turns)
        raise RuntimeError("boom")
    assert MemoryDB(path).list_spaces() == []  # 另一个连接也看不到：什么都没提交

    with db.transaction():
        db.ensure_space("demo")
        with pytest.raises(RuntimeError, match="inner"), db.transaction():
            db.add_session("demo", session.session_id, session.started_at)
            raise RuntimeError("inner")
        db.set_meta("k", "v")
    other = MemoryDB(path)
    assert [s["space_id"] for s in other.list_spaces()] == ["demo"]
    assert other.get_sessions("demo") == {} and other.get_meta("k") == "v"


def test_versions_change_with_every_write_that_matters(tmp_path: Path) -> None:
    """巩固跨运行复用载入的状态、不重算内容指纹，靠这两个版本发现库变了：派生记忆的版本在写
    事实、实体、实体检索单元，清空派生记忆，删除空间，事务回滚，以及别的连接（别的进程）提交时
    变；在线写入的发言与"轮"不动它。发言的版本只在这个会话写了发言、空间删过发言、回滚与别的
    连接提交时变。"""
    path = tmp_path / "memory.sqlite"
    db = MemoryDB(path)
    session = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8")).sessions[0]
    turns = build_turns("demo", session)
    rounds = build_rounds(turns)
    when = datetime(2026, 1, 1)
    fact = Fact(
        id="f1",
        space_id="demo",
        subject_id="u",
        subject="User",
        predicate="likes",
        object="tea",
        statement="The user likes tea.",
        recorded_at=when,
        session_id=session.session_id,
        source_turn_ids=[turns[0].id],
    )
    entity = MemoryUnit(
        id="e1",
        space_id="demo",
        kind="entity",
        session_id=session.session_id,
        text="Max (pet): a dog",
        ts=when,
        source_turn_ids=[],
    )
    derived, sid = db.derived_version("demo"), session.session_id
    changes: list[str] = []

    def check(step: str) -> None:
        nonlocal derived
        now = db.derived_version("demo")
        if now != derived:
            changes.append(step)
        derived = now

    db.add_turns(turns[:2])
    check("add_turns")
    db.add_units(rounds)
    check("add_units(round)")
    db.delete_units("demo", [rounds[0].id])
    check("delete_units(round)")
    db.apply_consolidation(facts=[fact], entities=[], episodes=[], units=[])
    check("apply_consolidation")
    db.apply_consolidation(facts=[], entities=[], episodes=[], units=[])
    check("apply_consolidation(nothing)")
    db.add_units([entity])
    check("add_units(entity)")
    db.delete_units("demo", [entity.id])
    check("delete_units(entity)")
    db.clear_derived("demo")
    check("clear_derived")
    with pytest.raises(RuntimeError), db.transaction():
        raise RuntimeError("rolled back")
    check("rollback")
    MemoryDB(path).apply_consolidation(facts=[fact], entities=[], episodes=[], units=[])
    check("other connection")
    db.delete_space("demo")
    check("delete_space")
    assert changes == [
        "apply_consolidation",
        "add_units(entity)",
        "delete_units(entity)",
        "clear_derived",
        "rollback",
        "other connection",
        "delete_space",
    ]

    turns_of = {s: db.turns_version("demo", s) for s in (sid, "other")}
    db.add_turns(turns[2:])
    assert db.turns_version("demo", sid) != turns_of[sid]
    assert db.turns_version("demo", "other") == turns_of["other"]  # 别的会话不受影响
    more = build_turns("demo", session, len(turns))[:1]  # 带附加信息（SDK 的 metadata）的发言
    before = db.turns_version("demo", sid)
    assert db.add_turns(more, {more[0].id: TurnMeta(metadata={"source": "crm"})}) == 1
    assert db.turns_version("demo", sid) != before
    for step in (
        lambda: db.delete_turns("demo", [turns[0].id]),
        lambda: MemoryDB(path).add_turns(turns[:1]),
        lambda: db.delete_space("demo"),
    ):
        before = db.turns_version("demo", "other")
        step()
        assert db.turns_version("demo", "other") != before


def test_scrub_leaves_no_deleted_text_without_vacuum(tmp_path: Path) -> None:
    """遗忘之后的 scrub 只把 WAL 写回并截断，不 VACUUM：删除时 secure_delete 已经覆写了
    被删的内容，长到溢出页里的发言也一样。"""
    path = tmp_path / "memory.sqlite"
    db = MemoryDB(path)
    session = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8")).sessions[0]
    turns = build_turns("demo", session)
    secret = "".join(f"ZQX{i:05d} " for i in range(3000))  # 约 27 KB，跨好几个溢出页
    turns[0] = turns[0].model_copy(update={"text": secret})
    db.ensure_space("demo")
    db.add_session("demo", session.session_id, session.started_at)
    db.add_turns(turns)
    db.scrub()  # 先写回数据库文件本身

    def on_disk() -> bytes:
        wal = path.with_name(path.name + "-wal")
        return path.read_bytes() + (wal.read_bytes() if wal.exists() else b"")

    assert b"ZQX02999" in on_disk()
    db.delete_turns("demo", [turns[0].id])
    db.scrub()
    raw = on_disk()
    assert b"ZQX" not in raw
    assert turns[1].text.encode() in raw  # 对照：没删的发言还在
    db.close()


def test_aware_times_are_stored_and_read_as_local_time(tmp_path: Path) -> None:
    """带时区的时间在入口换算成本机时间、去掉时区，不带时区的原样保留（评测数据都不带时区）。
    之前用带时区的时间写进去的行（'+00:00'），读出来也换算：否则与新写入的行混在一起无法比较。"""
    naive = datetime(2026, 8, 1, 10, 0)
    aware = naive.astimezone(UTC)  # 同一时刻，UTC 表示
    plain = SessionIn(session_id="s", started_at=naive, turns=[TurnIn(role="user", text="hi")])
    assert plain.started_at is naive
    assert TurnIn(role="user", text="hi", ts=naive).ts is naive
    converted = SessionIn(
        session_id="s", started_at=aware, turns=[TurnIn(role="user", text="hi", ts=aware)]
    )
    assert converted.started_at == naive and converted.started_at.tzinfo is None
    assert converted.turns[0].ts == naive and converted.turns[0].ts.tzinfo is None

    db = MemoryDB(tmp_path / "memory.sqlite")
    db.ensure_space("demo")
    db.add_session("demo", "old", aware)
    old = Turn(id="t", space_id="demo", session_id="old", idx=0, role="user", text="hi", ts=aware)
    db.add_turns([old])
    db.add_units(build_rounds([old]))
    assert db.get_sessions("demo") == {"old": naive}
    assert [(t.ts, t.ts.tzinfo) for t in db.get_turns("demo")] == [(naive, None)]
    assert [(u.ts, u.ts.tzinfo) for u in db.list_units("demo")] == [(naive, None)]
    db.close()


def test_meta_guard_detects_embedder_change(tmp_path: Path) -> None:
    db = MemoryDB(tmp_path / "memory.sqlite")
    db.check_or_set_meta("embedder", "model-a:8")
    db.check_or_set_meta("embedder", "model-a:8")
    with pytest.raises(ConfigMismatchError) as caught:
        db.check_or_set_meta("embedder", "model-b:16")
    # 报错指向新的数据目录（pip 安装后没有仓库里的 data/ 目录可删）
    message = str(caught.value)
    assert str(tmp_path / "memory.sqlite") in message and "model-a:8" in message
    assert "CUEMEM_HOME" in message and "Memory(home=...)" in message
    assert "data/ 目录" not in message


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


def test_locked_local_store_explains_itself(tmp_path: Path) -> None:
    """Qdrant 本地模式一次只允许一个进程打开：第二个打开者得到能看懂的报错，而不是库内部的异常。"""
    first = VectorStore(tmp_path / "qdrant", dim=4)
    with pytest.raises(StorageLockedError, match="CUEMEM_VECTOR_STORE__URL"):
        VectorStore(tmp_path / "qdrant", dim=4)
    first.close()
    VectorStore(tmp_path / "qdrant", dim=4).close()  # 锁随关闭释放


def test_copy_into_moves_every_collection(tmp_path: Path) -> None:
    """迁到服务端：文字与图片集合连同向量、payload 原样复制，检索结果不变。"""
    source = VectorStore(tmp_path / "local", dim=4)
    dense = l2_normalize(np.eye(4, dtype=np.float32)[:3] + 0.1)
    sparse = [SparseVector(indices=[i], values=[1.0]) for i in range(3)]
    payloads = [{"kind": "round", "ts": float(i)} for i in range(3)]
    source.upsert("s", ["a", "b", "c"], dense, sparse, payloads)
    source.upsert_images("s", ["m1"], l2_normalize(np.ones((1, 2), dtype=np.float32)), [{}])
    target = VectorStore(tmp_path / "server", dim=4)
    counts = source.copy_into(target)
    assert sorted(counts.values()) == [1, 3]
    assert target.count("s") == 3
    before = [h.unit_id for h in source.search_dense("s", dense[1], 3)]
    assert [h.unit_id for h in target.search_dense("s", dense[1], 3)] == before
    assert [h.unit_id for h in target.search_sparse("s", sparse[2], 1)] == ["c"]
    assert [h.unit_id for h in target.search_images("s", np.ones(2) / np.sqrt(2), 1)] == ["m1"]
    source.close()
    target.close()


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


def test_fact_indexes_are_added_to_existing_databases(tmp_path: Path) -> None:
    """每次读取都要沿版本链找被取代的旧版本、数当前有效的事实：以前只有 (space_id, subject_id,
    predicate) 一个索引，两个查询都扫整个空间。打开已有的库时自动补上两个索引，查询结果不变。"""
    import sqlite3

    from cuemem.core.models import Fact

    path = tmp_path / "memory.sqlite"
    db = MemoryDB(path)
    when = datetime(2025, 1, 1)
    facts = [
        Fact(
            id=f"f{i:03d}",
            space_id="s" if i % 3 else "other",
            subject_id="user",
            subject="User",
            predicate="likes",
            object=f"thing {i}",
            statement=f"The user likes thing {i}.",
            recorded_at=when.replace(day=1 + i % 28),
            invalidated_at=when if i % 4 == 0 else None,
            superseded_by=f"f{i + 1:03d}" if i % 4 == 0 else None,
            session_id="x",
            source_turn_ids=[],
        )
        for i in range(60)
    ]
    db.apply_consolidation(facts=facts, entities=[], episodes=[], units=[])

    def read() -> tuple[list[str], dict[str, int]]:
        old = db.get_facts("s", superseded_by=["f005", "f009", "f033", "nope"])
        return [f.id for f in old], db.space_stats("s")

    db.close()
    conn = sqlite3.connect(path)  # 模拟升级前的库：没有这两个索引
    conn.executescript(
        "DROP INDEX IF EXISTS idx_facts_superseded; DROP INDEX IF EXISTS idx_facts_current;"
    )
    conn.close()
    db = MemoryDB(path)
    with_indexes = read()
    names = {row[1] for row in db._conn.execute("PRAGMA index_list(facts)")}
    assert {"idx_facts_superseded", "idx_facts_current"} <= names
    plan = " ".join(
        str(row[-1])
        for row in db._conn.execute(
            "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM facts"
            " WHERE space_id = ? AND invalidated_at IS NULL",
            ("s",),
        )
    )
    assert "idx_facts_current" in plan
    for name in ("idx_facts_superseded", "idx_facts_current"):
        db._conn.execute(f"DROP INDEX {name}")
    assert read() == with_indexes  # 没有索引时（全表扫描）结果相同
    assert with_indexes[0] == ["f004", "f032", "f008"]  # 按 recorded_at、id 排序
    db.close()


def test_v5_database_gains_turn_meta(tmp_path: Path) -> None:
    """v6 加了 turn_meta（SDK 的 metadata、过期时间与更正目标）：v5 的库打开时自动补上这张表，
    已有的发言没有附加信息；附加信息只写给这次真正写进去的发言，随发言一起删。"""
    import sqlite3

    from cuemem.core.models import TurnMeta

    def turn(turn_id: str, idx: int) -> Turn:
        return Turn(
            id=turn_id,
            space_id="s",
            session_id="x",
            idx=idx,
            role="user",
            text=f"hi {idx}",
            ts=datetime(2025, 1, 1),
        )

    path = tmp_path / "memory.sqlite"
    db = MemoryDB(path)
    assert db.add_turns([turn("t0", 0)]) == 1
    db.close()
    conn = sqlite3.connect(path)  # 模拟 v5 的库：没有 turn_meta
    conn.executescript("DROP TABLE turn_meta; PRAGMA user_version = 5;")
    conn.close()

    db = MemoryDB(path)
    assert db._conn.execute("PRAGMA user_version").fetchone()[0] == 6
    assert db.get_turn_meta("s") == {} and [t.id for t in db.get_turns("s")] == ["t0"]
    tagged = TurnMeta(metadata={"topic": "food", "n": 1}, expires_at="2030-01-01")
    assert db.add_turns([turn("t1", 1)], {"t1": tagged}) == 1
    assert db.add_turns([turn("t0", 0)], {"t0": tagged}) == 0  # 已有的发言：不换上附加信息
    assert db.get_turn_meta("s") == {"t1": tagged}
    assert db.get_turn_meta("s", ["t0", "t1", "nope"]) == {"t1": tagged}
    assert db.get_turn_meta("s", []) == {} and db.get_turn_meta("other") == {}
    db.delete_turns("s", ["t1"])
    assert db.get_turn_meta("s") == {}
    db.close()


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


def test_llm_cache_hold_keeps_only_what_was_reused(tmp_path: Path) -> None:
    """遗忘时对账与消歧的缓存先标记（hold），重放之后只留下重放时命中过的（settle）。标记写在库
    里：进程在核对之前退出时，下次打开缓存就把标记着的条目全部删掉（删除时覆写）。"""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    from cuemem.llm.cache import ResponseCache

    path = tmp_path / "llm.sqlite"

    def on_disk() -> bytes:
        wal = path.with_name("llm.sqlite-wal")
        return path.read_bytes() + (wal.read_bytes() if wal.exists() else b"")

    cache = ResponseCache(path)
    for key in ("a", "b", "c"):
        cache.put(key, "m", {"text": f"secret-{key}"}, scope="s|consolidate")
    cache.put("x", "m", {"text": "other"}, scope="t|consolidate")
    assert cache.hold("s|consolidate") == 3
    with cache.recording() as hits:
        assert cache.get("a") is not None and cache.get("missing") is None
        with ThreadPoolExecutor(1) as pool:  # 复制了上下文的线程里的命中也记得到
            assert pool.submit(copy_context().run, cache.get, "b").result() is not None
        thread = threading.Thread(target=cache.get, args=("c",))  # 没复制的不算
        thread.start()
        thread.join()
    assert hits == {"a", "b"}
    assert cache.settle("s|consolidate", hits) == 1
    assert [cache.get(k) is not None for k in ("a", "b", "c", "x")] == [True, True, False, True]
    assert cache.settle("s|consolidate", set()) == 0  # 标记已经去掉，再核对不删东西

    assert cache.hold("s|consolidate") == 2
    cache.close()  # 没有核对：进程在遗忘重放时退出
    assert b"secret-a" in on_disk()
    cache = ResponseCache(path)
    assert cache.get("a") is None and cache.get("b") is None and cache.get("x") is not None
    assert cache._conn.execute("SELECT COUNT(*) FROM llm_cache_held").fetchone() == (0,)
    raw = on_disk()  # 还开着：删完就截断了 WAL
    assert b"secret-a" not in raw and b"secret-b" not in raw and b"other" in raw
    cache.close()
