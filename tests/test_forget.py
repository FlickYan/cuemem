"""遗忘（被遗忘权）：删除一段会话后，由它派生的一切都要消失，被它取代过的旧事实恢复为当前；
并且从数据库文件、向量库、两种缓存里都找不到被删的内容。"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from scripted import (
    EMPTY_EXTRACTION,
    TOY_EXTRACTIONS,
    TOY_UPDATE_KEYWORDS,
    ScriptedModel,
    reconcile_by_keyword,
)

from cuemem.config import Settings
from cuemem.core.models import Conversation, SessionIn, TurnIn
from cuemem.embed import CachedEmbedder
from cuemem.memory import MemorySystem

SECRET = "电商"  # 只出现在第 4 段会话（s4）里的词
PIN = "SECRET-1234"


@pytest.fixture
def system(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[MemorySystem]:
    fake_api.chat_replies = [
        ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    ]
    conv = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    memory = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    memory.ingest(conv, consolidate=True)
    yield memory
    memory.close()


def _on_disk(path: Path) -> bytes:
    """SQLite 数据库文件连同它的 WAL 的全部字节（取证用）。"""
    wal = path.with_name(path.name + "-wal")
    return path.read_bytes() + (wal.read_bytes() if wal.exists() else b"")


def test_forget_session_cascades_and_restores(system: MemorySystem, fake_api: FakeAPI) -> None:
    assert any(SECRET in f.statement for f in system.facts("demo"))
    s4_rounds = [u.id for u in system.db.list_units("demo", kinds=["round"], session_ids=["s4"])]
    s4_texts = [u.text for u in system.db.list_units("demo") if SECRET in u.text]
    assert isinstance(system.embedder, CachedEmbedder)
    assert system.embedder.cached(s4_texts, "document")
    # 向量缓存里没有原文，取证时找被删文本的缓存键与向量本身
    keys = [system.embedder._key(t, "document") for t in s4_texts]
    embedded = [v.tobytes()[:64] for v, _ in system.embedder.cache.get_many(keys).values()]
    assert len(embedded) == len(keys)
    extraction_requests = sum(
        1 for r in fake_api.chat_requests if "long-term memory of a personal" in str(r)
    )
    statements: list[str] = []
    system.db._conn.set_trace_callback(statements.append)

    report = system.forget("demo", session_ids=["s4"])
    system.db._conn.set_trace_callback(None)

    assert report.turns_deleted == 4 and report.sessions_affected == ["s4"]
    # 原始层：轮次与会话都没了
    assert all(t.session_id != "s4" for t in system.db.get_turns("demo"))
    assert "s4" not in system.db.get_sessions("demo")
    # 派生层：从 s4 来的事实消失，被它取代过的旧事实恢复为当前有效
    facts = system.facts("demo")
    assert not any(SECRET in f.statement for f in facts)
    current = {f.statement for f in system.facts("demo", current_only=True)}
    assert "用户在北京一家互联网公司工作。" in current
    assert "用户买笔记本的预算约 8000 元。" in current
    core = system.core_memory("demo")
    assert core is not None and SECRET not in core.text and "北京" in core.text
    assert all(e.session_id != "s4" for e in system.db.get_episodes("demo"))
    # 向量库：s4 的"轮"及派生单元的向量都删掉了，且与数据库一一对应
    remaining = set(system.vectors.unit_ids("demo"))
    assert not remaining & set(s4_rounds)
    assert remaining == {u.id for u in system.db.list_units("demo")}
    # 缓存：被删文本的向量、s4 的抽取结果都清掉了；其余会话的抽取仍可复用
    assert not system.embedder.cached(s4_texts, "document")
    assert system.llm.cache is not None
    assert not any(SECRET in r for r in system.llm.cache.responses())
    assert report.llm_cache_purged > 0 and report.embed_cache_purged > 0
    extraction_after = sum(
        1 for r in fake_api.chat_requests if "long-term memory of a personal" in str(r)
    )
    assert extraction_after == extraction_requests  # 重放时其余 4 段会话全部命中缓存

    # 取证：数据库与两种缓存的文件（连同 WAL）里也找不到被删的内容。删除时覆写
    # （secure_delete），删完把 WAL 写回并截断就够了，不用 VACUUM 重写整个库
    assert not any(s.lstrip().upper().startswith("VACUUM") for s in statements)
    paths = system.settings.paths
    raw = _on_disk(paths.db)
    assert SECRET.encode("utf-8") not in raw
    assert "北京".encode() in raw  # 对照：没被删的内容还在
    cached = _on_disk(paths.llm_cache)
    assert SECRET.encode("utf-8") not in cached and "北京".encode() in cached
    vectors = _on_disk(paths.embed_cache)
    assert not any(k.encode() in vectors for k in keys)
    assert not any(v in vectors for v in embedded)
    kept = next(u.text for u in system.db.list_units("demo") if "北京" in u.text)
    assert system.embedder._key(kept, "document").encode() in vectors


def test_forget_single_turn_rebuilds_its_round(system: MemorySystem) -> None:
    [turn] = [
        t for t in system.db.get_turns("demo", "s3") if "巴厘岛" in t.text and t.role == "user"
    ]
    report = system.forget("demo", turn_ids=[turn.id])
    assert report.turns_deleted == 1
    rounds = system.db.list_units("demo", kinds=["round"], session_ids=["s3"])
    assert rounds and all("巴厘岛" not in u.text for u in rounds)
    assert "s3" in system.db.get_sessions("demo")  # 会话里还有别的轮次，会话保留
    assert set(system.vectors.unit_ids("demo")) == {u.id for u in system.db.list_units("demo")}


def test_forget_unknown_target_is_noop(system: MemorySystem) -> None:
    before = len(system.facts("demo"))
    report = system.forget("demo", session_ids=["nope"])
    assert report.turns_deleted == 0
    assert len(system.facts("demo")) == before


def test_close_closes_its_own_caches(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """from_settings 建的系统关闭时连同两种缓存一起关（关掉最后一个连接时 SQLite 把 WAL 写回
    并删掉）；评测的系统（ephemeral / on_disk）共用传入的缓存，关闭时不能把它们关掉。"""
    fake_api.chat_replies = [ScriptedModel(TOY_EXTRACTIONS)]
    conv = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    system.ingest(conv, consolidate=True)
    llm, embedder = system.llm, system.embedder
    assert llm.cache is not None and isinstance(embedder, CachedEmbedder)
    texts = [u.text for u in system.db.list_units("demo")]
    MemorySystem.ephemeral(n3_settings, embedder=embedder, llm=llm, ledger=system.ledger).close()
    assert len(llm.cache) > 0 and embedder.cached(texts, "document")  # 共用的缓存还开着

    system.close()
    paths = n3_settings.paths
    for path in (paths.llm_cache, paths.embed_cache):
        assert not path.with_name(path.name + "-wal").exists()
    with pytest.raises(sqlite3.ProgrammingError):
        len(llm.cache)
    with pytest.raises(sqlite3.ProgrammingError):
        embedder.cached(texts, "document")


def _in_thread(target: Any, *args: Any, **kwargs: Any) -> tuple[threading.Thread, list[Any]]:
    """在另一个线程里调用（模拟另一个请求），返回线程与它的结果（或异常）。"""
    out: list[Any] = []

    def run() -> None:
        try:
            out.append(target(*args, **kwargs))
        except Exception as exc:
            out.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, out


@pytest.mark.parametrize("how", ["forget", "delete_space"])
def test_forget_waits_for_background_consolidation(
    n3_settings: Settings, fake_api: FakeAPI, how: str
) -> None:
    """后台线程正在抽取这段会话时遗忘它：抽取结果不能在遗忘之后写回事实、向量与缓存。"""
    entered, release = threading.Event(), threading.Event()

    def extract(prompt: str) -> dict[str, Any]:
        if PIN not in prompt:
            return EMPTY_EXTRACTION
        entered.set()
        release.wait(10)  # 抽取请求还在路上（慢的供应商）
        fact = {
            "statement": f"The user's PIN is {PIN}.",
            "subject": "User",
            "predicate": "pin",
            "object": PIN,
            "importance": 5,
            "turns": [0],
        }
        return {"summary": "", "facts": [fact], "entities": []}

    fake_api.chat_replies = [ScriptedModel(extract)]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        worker = system.start_worker()
        session = SessionIn(
            session_id="s1",
            started_at=datetime(2026, 9, 1, 10),
            turns=[TurnIn(role="user", text=f"My PIN is {PIN}, keep it safe")],
        )
        system.append("alice", session, 0)  # 登记任务并交给后台线程
        assert entered.wait(5)
        if how == "forget":
            thread, out = _in_thread(system.forget, "alice", session_ids=["s1"])
        else:
            thread, out = _in_thread(system.delete_space, "alice")
        thread.join(0.3)  # 不持锁时遗忘在这期间就删完了原始日志、清空了派生层
        release.set()
        thread.join(10)
        worker.wait()

        assert worker.errors == []
        [report] = out
        assert not isinstance(report, Exception) and report.turns_deleted == 1
        assert system.db.get_facts("alice") == []
        assert system.vectors.unit_ids("alice") == []
        assert system.llm.cache is not None
        assert not any(PIN in r for r in system.llm.cache.responses())


def test_forget_other_session_during_background_reconciliation(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """后台线程正在对账时遗忘另一段会话：巩固不能因为它读过的记录被删了而半途崩掉。"""
    model = ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    entered, release = threading.Event(), threading.Event()

    def reply(body: dict[str, Any]) -> Any:
        system_prompt = body["messages"][0]["content"]
        if system_prompt.startswith("You maintain a long-term memory") and not release.is_set():
            entered.set()
            release.wait(10)
        return model(body)

    fake_api.chat_replies = [reply]
    conv = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        worker = system.start_worker()
        system.ingest(conv)  # 交给后台巩固
        assert entered.wait(10)
        thread, out = _in_thread(system.forget, "demo", session_ids=["s1"])
        thread.join(0.3)
        release.set()
        thread.join(10)
        worker.wait()

        assert worker.errors == []
        [report] = out
        assert not isinstance(report, Exception) and report.turns_deleted > 0
        assert all(f.session_id != "s1" for f in system.facts("demo"))
        assert all(j["status"] == "done" for j in system.db.get_jobs("demo"))
        core = system.core_memory("demo")
        assert core is not None and "林晓" not in core.text and "新加坡" in core.text
