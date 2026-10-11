"""遗忘（被遗忘权）：删除一段会话后，由它派生的一切都要消失，被它取代过的旧事实恢复为当前；
并且从数据库文件、向量库、两种缓存里都找不到被删的内容。"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword

from cuemem.config import Settings
from cuemem.core.models import Conversation
from cuemem.embed import CachedEmbedder
from cuemem.memory import MemorySystem

SECRET = "电商"  # 只出现在第 4 段会话（s4）里的词


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


def test_forget_session_cascades_and_restores(system: MemorySystem, fake_api: FakeAPI) -> None:
    assert any(SECRET in f.statement for f in system.facts("demo"))
    s4_rounds = [u.id for u in system.db.list_units("demo", kinds=["round"], session_ids=["s4"])]
    s4_texts = [u.text for u in system.db.list_units("demo") if SECRET in u.text]
    assert isinstance(system.embedder, CachedEmbedder)
    assert system.embedder.cached(s4_texts, "document")
    extraction_requests = sum(
        1 for r in fake_api.chat_requests if "long-term memory of a personal" in str(r)
    )

    report = system.forget("demo", session_ids=["s4"])

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

    # 取证：数据库文件里也找不到被删的内容（secure_delete + VACUUM）
    db_path = system.settings.paths.db
    raw = db_path.read_bytes()
    wal = db_path.with_name(db_path.name + "-wal")
    raw += wal.read_bytes() if wal.exists() else b""
    assert SECRET.encode("utf-8") not in raw
    assert "北京".encode() in raw  # 对照：没被删的内容还在


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
