from __future__ import annotations

import hashlib
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import TOY_CONVERSATION
from PIL import Image

from cuemem.core.models import Conversation, Fact, MemoryUnit, SessionIn, Turn, TurnIn, TurnMeta
from cuemem.embed import SparseVector
from cuemem.embed.base import l2_normalize
from cuemem.errors import ConfigMismatchError, StorageLockedError, VectorStoreOpenError
from cuemem.store import MemoryDB, VectorStore, media
from cuemem.store import sqlite as sqlite_module
from cuemem.store.media import MediaStore, image_data_url
from cuemem.store.sqlite import legacy_idx_key, next_idx_key
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


def test_point_reads_agree_with_the_whole_space_reads(tmp_path: Path) -> None:
    """按主键或 LIMIT 1 的读取（检索前判断空间是不是空的、add 查会话的开始时间、为几个会话登记
    任务、判断派生记忆是否过期）与读出整个空间再挑出来的结果相同：以前这些地方都读整个空间，
    耗时随会话数增长。"""
    db = MemoryDB(tmp_path / "memory.sqlite")
    when = datetime(2026, 1, 1, 10)
    sessions = {f"s{i:04d}": when.replace(minute=i % 60, second=i // 60) for i in range(1200)}
    for sid, started in sessions.items():
        db.add_session("demo", sid, started)
    db.add_session("other", "s0001", when)
    assert db.has_sessions("demo") and not db.has_sessions("empty")
    assert db.session_started("demo", "s0007") == sessions["s0007"]
    assert db.session_started("demo", "nope") is None
    wanted = ["s1100", "s0003", "nope", "s0003", *sorted(sessions)[::2]]  # 多于一批（500 个）
    assert db.sessions_of("demo", wanted) == {s: sessions[s] for s in wanted if s in sessions}
    assert db.sessions_of("demo", []) == {}
    assert db.turns_versions("demo", ["s0001", "nope"]) == {
        s: db.turns_version("demo", s) for s in ("s0001", "nope")
    }

    assert not db.has_jobs("demo") and not db.has_jobs_of_other_version("demo", "v2")
    db.enqueue_job("old", "demo", "s0001", "v1", "h0")
    assert db.has_jobs("demo") and not db.has_jobs("other")
    assert db.has_jobs_of_other_version("demo", "v2")
    assert not db.has_jobs_of_other_version("demo", "v1")
    # 一批登记与逐个登记相同：已有的任务不动，新任务取代同一会话还没执行的旧任务
    rows = [("a", "s0001", "v2", "h1"), ("old", "s0001", "v1", "h0"), ("a", "s0001", "v2", "h1")]
    rows += [(f"j{i}", f"s{i:04d}", "v2", "h") for i in range(2, 200)]
    assert db.enqueue_jobs("demo", rows) == 199
    jobs = {j["id"]: j["status"] for j in db.get_jobs("demo")}
    assert jobs["old"] == "superseded" and jobs["a"] == "pending" and len(jobs) == 200
    assert db.enqueue_jobs("demo", rows) == 0


def test_requeue_after_clearing_grows_linearly_with_the_sessions(tmp_path: Path) -> None:
    """遗忘、重建与重放清空派生层之后按会话重新登记任务：每登记一个都把同一会话还没执行的旧
    任务标记为 superseded。以前这一步只能走 (space_id, status) 的索引，扫一遍已经登记的全部
    任务，耗时随会话数的平方增长（2 万个会话约 20 秒，一直拿着整个库的锁）；现在有
    (space_id, session_id) 的索引，已有的库打开时自动补上。数 SQLite 执行的指令，不计时。"""
    import sqlite3

    path = tmp_path / "memory.sqlite"
    db = MemoryDB(path)
    db.close()
    conn = sqlite3.connect(path)  # 模拟升级前的库：没有这个索引
    conn.execute("DROP INDEX IF EXISTS idx_jobs_session")
    conn.close()

    def steps(space: str, n: int) -> int:
        db.ensure_space(space)
        db.enqueue_jobs(space, [(f"{space}-old{i}", f"s{i}", "v", "h") for i in range(n)])
        db.clear_derived(space)
        ticks = [0]

        def tick() -> int:
            ticks[0] += 1
            return 0

        db._conn.set_progress_handler(tick, 100)
        try:
            new = [(f"{space}-{i}", f"s{i}", "v", "h") for i in range(n)]
            assert db.enqueue_jobs(space, new) == n
        finally:
            db._conn.set_progress_handler(None, 0)
        assert {j["status"] for j in db.get_jobs(space)} == {"pending"}
        return ticks[0]

    db = MemoryDB(path)
    names = {row[1] for row in db._conn.execute("PRAGMA index_list(jobs)")}
    assert "idx_jobs_session" in names
    small, large = steps("a", 600), steps("b", 1200)
    assert large < 2.5 * small  # 线性：会话翻倍，指令数约翻倍（以前约是 4 倍）
    db.close()


def _one_turn(db: MemoryDB, space: str, session_id: str) -> Turn:
    session = SessionIn(
        session_id=session_id,
        started_at=datetime(2026, 8, 1, 10, 0),
        turns=[TurnIn(role="user", text=f"hello from {space}")],
    )
    db.ensure_space(space)
    db.add_session(space, session_id, session.started_at)
    [turn] = build_turns(space, session)
    db.add_turns([turn])
    return turn


def test_turn_numbers_of_sessions_without_turns_are_dropped_exactly(tmp_path: Path) -> None:
    """删掉空间、删光一个会话的发言时，它们的 next_idx 一起删（键里是用户 ID 与会话 ID）。以前的
    版本写下的键有歧义（next_idx:{空间}:{会话}）：空间 'acme' 的会话 'alice:x' 与空间 'acme:alice'
    的会话 'x' 是同一个键，按前缀删会删掉别人的；每种拆法都核对，还有会话在用的留着。"""
    db = MemoryDB(tmp_path / "memory.sqlite")
    kept = _one_turn(db, "acme:alice", "x")
    _one_turn(db, "acme", "y")
    gone = _one_turn(db, "acme", "z")
    for space, sid in [("acme", "y"), ("acme", "z"), ("acme:alice", "x")]:
        db.set_meta(next_idx_key(space, sid), "1")
    db.set_meta(legacy_idx_key("acme", "y"), "1")
    db.set_meta(legacy_idx_key("acme:alice", "x"), "1")  # 也是 legacy_idx_key("acme", "alice:x")
    db.set_meta(legacy_idx_key("acme", "left-by-an-old-forget"), "3")
    db.set_meta("consolidation:acme", "fingerprint")

    db.delete_turns("acme", [gone.id])  # 会话 z 没有发言了
    assert next_idx_key("acme", "z") not in db.meta_keys("next")
    assert legacy_idx_key("acme", "left-by-an-old-forget") not in db.meta_keys("next")
    assert db.get_meta(next_idx_key("acme", "y")) == "1"

    db.delete_space("acme")
    assert db.meta_keys("next") == sorted(
        [next_idx_key("acme:alice", "x"), legacy_idx_key("acme:alice", "x")]
    )
    assert db.get_meta("consolidation:acme") is None
    assert [t.id for t in db.get_turns("acme:alice")] == [kept.id]


def test_turn_write_counts_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """记着写入次数的会话数有上限（不给 run_id 的 add 每次都是一个新会话，常驻的服务以前一直
    记着）：到了上限就清空、换一代，清空之前读到的版本都算变过，巩固不会把清空后重新数到同一个
    次数的会话当成没变。delete_space 也不再记着这个空间的会话。"""
    monkeypatch.setattr(sqlite_module, "TURN_WRITES_KEPT", 2, raising=False)
    db = MemoryDB(tmp_path / "memory.sqlite")
    versions = {}
    for sid in ("a", "b", "c", "d"):
        _one_turn(db, "demo", sid)
        versions[sid] = db.turns_version("demo", sid)
    assert len(db._turn_writes) <= 2
    assert db.turns_version("demo", "a") != versions["a"]
    assert db.turns_version("demo", "d") == versions["d"]
    db.delete_space("demo")
    assert not [key for key in db._turn_writes if key[0] == "demo"]


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


@pytest.mark.skipif(sys.platform == "win32", reason="Windows 没有 RLIMIT_NOFILE")
def test_too_many_collections_explains_itself(tmp_path: Path) -> None:
    """本地模式每个集合一直开着一个文件、启动时全部打开：集合多过打开文件数的上限时，以前抛出
    sqlite3.OperationalError（不是 CueMemError：cuemem serve、proxy 与 migrate-vectors 都打印调用栈
    退出），现在是写明集合数、上限与办法的 VectorStoreOpenError。"""
    store = VectorStore(tmp_path / "qdrant", dim=4)
    vector = l2_normalize(np.ones((1, 4), dtype=np.float32))
    for i in range(80):
        store.upsert(f"user-{i}", ["u"], vector, None, [{"kind": "round", "ts": 0.0}])
    store.close()
    script = (
        "import resource, sys\n"
        "from pathlib import Path\n"
        "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))\n"
        "from cuemem.store.vector import VectorStore\n"
        "VectorStore(Path(sys.argv[1]), dim=4)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "qdrant")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode != 0
    assert "VectorStoreOpenError" in done.stderr and "OperationalError" in done.stderr
    assert "这里有 80 个集合，打开文件数的上限是 64（硬上限 64）" in done.stderr
    assert "CUEMEM_VECTOR_STORE__URL" in done.stderr


def test_local_store_errors_say_what_to_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本地向量库的文件打不开时（打开与建集合都是）抛出 VectorStoreOpenError（CueMemError：命令行
    提示原因，写入与后台线程的出错、HTTP 接口都看得到）；集合不多时说要检查权限与磁盘。"""
    import sqlite3

    import qdrant_client.local.persistence as persistence

    store = VectorStore(tmp_path / "qdrant", dim=4)
    vector = l2_normalize(np.ones((1, 4), dtype=np.float32))
    store.upsert("old", ["u"], vector, None, [{"kind": "round", "ts": 0.0}])
    connect = sqlite3.connect

    def refuse(location: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if location == ":memory:":
            return connect(location, *args, **kwargs)
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(persistence.sqlite3, "connect", refuse)
    with pytest.raises(VectorStoreOpenError, match="权限与磁盘空间"):
        store.upsert("new", ["u"], vector, None, [{"kind": "round", "ts": 0.0}])
    with pytest.raises(VectorStoreOpenError, match="权限与磁盘空间"):
        store.upsert_images("old", ["m"], l2_normalize(np.ones((1, 2), dtype=np.float32)), [{}])
    assert store.count("old") == 1  # 已有的集合照常可用
    store.close()
    with pytest.raises(VectorStoreOpenError, match="unable to open database file"):
        VectorStore(tmp_path / "qdrant", dim=4)


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


def test_opening_the_cache_keeps_holds_of_a_forget_still_running(tmp_path: Path) -> None:
    """遗忘在 hold 与 settle 之间时，别的进程（评测、CLI、另一个服务）或同一进程里别的缓存对象
    打开同一份缓存：这次遗忘的标记不动，重放照样用得上（以前打开缓存就删掉全部标记，还记一条
    "进程中途退出"的警告）。标记它的进程真的退出了（崩溃）时，下次打开照样删掉。"""
    import subprocess
    import sys

    from cuemem.llm.cache import ResponseCache

    path = tmp_path / "llm.sqlite"
    first = ResponseCache(path)
    for i in range(3):
        first.put(f"k{i}", "m", {"decisions": [i]}, scope="alice|consolidate")
    assert first.hold("alice|consolidate") == 3
    with first.recording() as hits:
        assert first.get("k0") is not None
        second = ResponseCache(path)  # 遗忘重放到一半时打开
        assert first.get("k1") is not None
    assert first.settle("alice|consolidate", hits) == 1
    assert [second.get(f"k{i}") is not None for i in range(3)] == [True, True, False]
    second.close()

    crash = (
        "import os, sys; from pathlib import Path; from cuemem.llm.cache import ResponseCache; "
        "c = ResponseCache(Path(sys.argv[1])); c.put('x', 'm', {'secret': 1}, scope='bob|c'); "
        "c.hold('bob|c'); os._exit(1)"
    )
    subprocess.run([sys.executable, "-c", crash, str(path)], check=False)
    assert first.get("x") is not None  # 还开着的对象不替别人核对
    reopened = ResponseCache(path)
    assert reopened.get("x") is None and reopened.get("k0") is not None
    assert list((tmp_path / "llm.sqlite.holds").iterdir()) == []
    reopened.close()
    first.close()


# ---------- 媒体库：解压炸弹 ----------


def _png(path: Path, side: int) -> Path:
    Image.new("RGB", (side, side), "white").save(path)
    return path


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_media_store_refuses_images_with_too_many_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """像素数超过 Pillow 的 Image.MAX_IMAGE_PIXELS 的图片不进媒体库，报 ValueError（在线写入因此
    什么都不留，记忆代理只丢这一次写回）。以前超过两倍的抛出 DecompressionBombError（不是
    ValueError），一到两倍之间的只有警告、照样存下，之后每次用到都解码一遍。上限在调用时读，
    CueMem 不改它（整个进程共用这个设置，SDK 用户的程序也在用）。"""
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)
    store = MediaStore(tmp_path / "media")
    for side in (200, 120):  # 两倍以上；一到两倍之间
        with pytest.raises(ValueError, match="too large"):
            store.put(_png(tmp_path / f"{side}.png", side))
    assert not (tmp_path / "media").exists()
    stored = store.put(_png(tmp_path / "50.png", 50))
    assert stored.path.exists() and (stored.width, stored.height) == (50, 50)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", None)  # 关掉了 Pillow 的上限：照常存
    assert store.put(tmp_path / "200.png").width == 200
    assert Image.MAX_IMAGE_PIXELS is None


@pytest.mark.parametrize("thumbnail", [False, True], ids=["data-url", "ui-thumbnail"])
def test_large_images_are_decoded_two_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, thumbnail: bool
) -> None:
    """解码大图（像素数超过 LARGE_PIXELS）的同时最多 LARGE_DECODES 张：几十 KB 的 PNG 可以有
    八千多万像素，解码要两三百 MB，几个请求同时解码就能撑爆共用的服务。小图不排队。"""
    if thumbnail:
        pytest.importorskip("gradio")
        from cuemem.api.ui import _thumbnail as decode
    else:
        decode = image_data_url
    path = str(_png(tmp_path / "a.png", 64))
    convert = Image.Image.convert
    lock = threading.Lock()
    active = peak = 0
    barrier: threading.Barrier | None = None

    def counted(self: Image.Image, *args: Any, **kwargs: Any) -> Image.Image:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if barrier is not None:
                barrier.wait()  # 小图：六个要同时在解码
            else:
                time.sleep(0.1)
            return convert(self, *args, **kwargs)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(Image.Image, "convert", counted)
    monkeypatch.setattr(media, "LARGE_PIXELS", 64 * 64 - 1)
    with ThreadPoolExecutor(6) as pool:
        outputs = list(pool.map(lambda _: decode(path), range(6)))
    assert peak == media.LARGE_DECODES == 2 and len(set(outputs)) == 1
    monkeypatch.setattr(media, "LARGE_PIXELS", 64 * 64)
    barrier, peak = threading.Barrier(6, timeout=10), 0
    with ThreadPoolExecutor(6) as pool:
        assert set(pool.map(lambda _: decode(path), range(6))) == set(outputs)
    assert peak == 6


@pytest.mark.parametrize("queued", [False, True], ids=["small", "large"])
def test_image_bytes_sent_to_models_are_unchanged(
    monkeypatch: pytest.MonkeyPatch, queued: bool
) -> None:
    """发给模型的图片字节（因而缓存的键）与以前逐字节相同，排不排队都一样（不用 draft()、
    thumbnail() 之类的捷径）。768 不缩放，512 缩放。"""
    if queued:
        monkeypatch.setattr(media, "LARGE_PIXELS", 0)
    photo = Path(__file__).parents[1] / "demo" / "shop" / "images" / "P01.jpg"

    def digest(**kwargs: Any) -> str:
        return hashlib.sha256(image_data_url(photo, **kwargs).encode()).hexdigest()

    assert digest() == "372131d73958967e1c0b7bfdccebc90b21457e8848456a5c10d15c9ab1f4ed84"
    assert digest(max_side=512) == (
        "48ef386cd0e7b9249270ccdb24f8893b05e666346a6982c5603881e90971af22"
    )
