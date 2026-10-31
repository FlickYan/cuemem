"""遗忘（被遗忘权）：删除一段会话后，由它派生的一切都要消失，被它取代过的旧事实恢复为当前；
并且从数据库文件、向量库、两种缓存里都找不到被删的内容。"""

from __future__ import annotations

import errno
import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
from fakes import NO_BALANCE, REJECT, TOY_CONVERSATION, FakeAPI
from PIL import Image
from scripted import (
    EMPTY_EXTRACTION,
    TOY_EXTRACTIONS,
    TOY_UPDATE_KEYWORDS,
    ScriptedModel,
    echo_extraction,
    reconcile_by_keyword,
)
from scripted import _fact as fact_reply
from test_sdk import AUG, SEP, moved, where_i_live

import cuemem.llm.client as llm_client
from cuemem import Memory
from cuemem.config import Settings
from cuemem.core.ids import content_hash
from cuemem.core.models import Conversation, SessionIn, TurnIn
from cuemem.core.trace import Trace
from cuemem.embed import CachedEmbedder
from cuemem.embed.multimodal import DashScopeMultimodalEmbedder
from cuemem.errors import EmbeddingAPIError
from cuemem.memory import MemorySystem
from cuemem.write.consolidate import scope

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


@pytest.fixture
def requests_by_key(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """LLM 缓存键 → 算出它的请求（JSON）。对账与消歧的回复只有编号，被删的内容只在请求里，
    取证时要按请求找缓存条目。"""
    seen: dict[str, str] = {}
    real = llm_client.request_key

    def spy(request: dict[str, Any]) -> str:
        key = real(request)
        seen[key] = json.dumps(request, ensure_ascii=False)
        return key

    monkeypatch.setattr(llm_client, "request_key", spy)
    return seen


def _assert_no_cache_entry_with(
    system: MemorySystem, requests: dict[str, str], words: list[str]
) -> list[str]:
    """LLM 缓存里（连同文件与 WAL 的字节）没有请求里带着这些词的条目，返回这些条目的键。"""
    assert system.llm.cache is not None
    doomed = [k for k, body in requests.items() if any(w in body for w in words)]
    assert doomed
    conn = system.llm.cache._conn
    kept = {key for (key,) in conn.execute("SELECT key FROM llm_cache")}
    assert not kept & set(doomed)
    assert conn.execute("SELECT COUNT(*) FROM llm_cache_held").fetchone() == (0,)
    raw = _on_disk(system.settings.paths.llm_cache)
    assert not any(k.encode() in raw for k in doomed)
    assert not any(w.encode() in raw for w in words)
    return doomed


def test_forget_session_cascades_and_restores(
    requests_by_key: dict[str, str], system: MemorySystem, fake_api: FakeAPI
) -> None:
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
    _assert_no_cache_entry_with(system, requests_by_key, [SECRET])  # 连同请求里有它的条目的键
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


def _org(name: str, description: str, kind: str = "organization") -> dict[str, Any]:
    return {"name": name, "type": kind, "aliases": [], "description": description}


# b 是要遗忘的（Zurich、Bluebird 只在它里面）。c 的对账列着 b 的住址与 a 的公司，消歧列着
# b 的项目；d 的对账与消歧只列着 a 的职位与公司；e 的对账列着 c 的住址
SECRET_WORDS = ["Zurich", "Bluebird"]
SECRET_CHAT = [
    ("s-a", "2026-01-05", "I work at Acme as a designer."),
    ("s-b", "2026-02-05", "I moved to Zurich for Bluebird, a secret research project."),
    ("s-c", "2026-03-05", "I moved to Geneva and now work at Globex on Bluejay."),
    ("s-d", "2026-04-05", "I got promoted to lead designer; Initech hired our team."),
    ("s-e", "2026-05-05", "I moved to Bern last week."),
]
SECRET_EXTRACTIONS: dict[str, dict[str, Any]] = {
    "2026-01-05": {
        "summary": "",
        "facts": [
            fact_reply("The user works at Acme.", "works_at", "Acme", turns=[0]),
            fact_reply("The user is a designer.", "job_title", "designer", turns=[0]),
        ],
        "entities": [_org("Acme", "the company where the user works")],
    },
    "2026-02-05": {
        "summary": "",
        "facts": [
            fact_reply(
                "The user moved to Zurich for the Bluebird project.",
                "lives_in",
                "Zurich",
                turns=[0],
            )
        ],
        "entities": [
            _org(
                "Bluebird",
                "a secret research project the user joined in Zurich",
                kind="project",
            )
        ],
    },
    "2026-03-05": {
        "summary": "",
        "facts": [
            fact_reply("The user moved to Geneva.", "lives_in", "Geneva", turns=[0]),
            fact_reply("The user works at Globex.", "works_at", "Globex", turns=[0]),
        ],
        "entities": [
            _org("Bluejay", "a research project the user joined in Geneva", kind="project")
        ],
    },
    "2026-04-05": {
        "summary": "",
        "facts": [
            fact_reply(
                "The user was promoted to lead designer.", "job_title", "lead designer", turns=[0]
            )
        ],
        "entities": [_org("Initech", "the company that hired the user's team")],
    },
    "2026-05-05": {
        "summary": "",
        "facts": [fact_reply("The user moved to Bern.", "lives_in", "Bern", turns=[0])],
        "entities": [],
    },
}


def _secret_system(
    settings: Settings, fake_api: FakeAPI, **replies: Any
) -> tuple[MemorySystem, ScriptedModel]:
    """建好 SECRET_CHAT 的空间。实体描述相似度在 0.5–0.9 之间的交给模型消歧（哈希向量下
    Bluejay 与 Bluebird 约 0.75、Initech 与 Acme 约 0.63）。"""
    consolidation = settings.consolidation.model_copy(
        update={"entity_gray_threshold": 0.5, "entity_merge_threshold": 0.9}
    )
    model = ScriptedModel(
        SECRET_EXTRACTIONS, reconcile=reconcile_by_keyword("moved", "works", "designer"), **replies
    )
    fake_api.chat_replies = [model]
    system = MemorySystem.from_settings(
        settings.model_copy(update={"consolidation": consolidation}),
        http_client=fake_api.client(),
    )
    sessions = [
        SessionIn(
            session_id=sid,
            started_at=datetime.fromisoformat(f"{day}T10:00:00"),
            turns=[TurnIn(role="user", text=text)],
        )
        for sid, day, text in SECRET_CHAT
    ]
    system.ingest(Conversation(space_id="alice", sessions=sessions), consolidate=True)
    # 对账：c、d、e 各一次；消歧：c、d 各一次。列着 b 的内容的是 c 的两次
    assert model.calls["reconcile"] == 3 and model.calls["entity"] == 2
    prompts = [*model.prompts["reconcile"], *model.prompts["entity"]]
    assert sum(1 for p in prompts if any(w in p for w in SECRET_WORDS)) == 2
    return system, model


def _sent(model: ScriptedModel, kind: str, *words: str) -> int:
    """这类请求里带着这些词的发了几次（命中缓存的不算）。"""
    return sum(1 for p in model.prompts[kind] if all(w in p for w in words))


def test_forget_keeps_reconciliation_and_entity_cache_the_replay_reuses(
    n3_settings: Settings, fake_api: FakeAPI, requests_by_key: dict[str, str]
) -> None:
    """遗忘之后不再把整个空间的对账与消歧重新请求一遍（D-021 修订）：输入没变的（d、e 的）命中
    缓存，只有列过被删会话内容的（c 的）重新请求；请求里有被删内容的缓存条目一条不留，连同它的
    键。以前遗忘清掉整个空间的对账与消歧缓存，重放时全部重新请求。"""
    system, model = _secret_system(n3_settings, fake_api)
    with system:
        before = model.calls.copy()

        report = system.forget("alice", session_ids=["s-b"])

        assert report.sessions_affected == ["s-b"]
        # c 的对账少了 b 的住址，重新请求；c 的项目没有候选了，不用消歧；d、e 的都命中缓存
        assert model.calls["reconcile"] - before["reconcile"] == 1
        assert model.calls["entity"] == before["entity"]
        assert model.calls["extract"] == before["extract"]
        assert not any(w in model.prompts["reconcile"][-1] for w in SECRET_WORDS)
        assert _sent(model, "reconcile", "lead designer") == 1
        assert _sent(model, "entity", "Initech") == 1
        current = {f.statement for f in system.facts("alice", current_only=True)}
        assert {"The user moved to Bern.", "The user was promoted to lead designer."} <= current
        assert not any(w in f.statement for f in system.facts("alice") for w in SECRET_WORDS)
        # 取证：请求里有 Zurich、Bluebird 的条目（b 的抽取，c 的对账与消歧）连同键都删了，
        # 文件里也找不到；d 的对账与消歧（重放原样用到的）还在
        doomed = _assert_no_cache_entry_with(system, requests_by_key, SECRET_WORDS)
        assert len(doomed) == 3
        assert system.llm.cache is not None
        kept = {key for (key,) in system.llm.cache._conn.execute("SELECT key FROM llm_cache")}
        assert {k for k, body in requests_by_key.items() if "Initech" in body} <= kept


def _flag_once(word: str) -> Any:
    """兜底检查的替身：第一次在剩下的事实里看到带 word 的就说它还在透露，之后都说没有。"""
    flagged: list[int] = []

    def decide(prompt: str) -> dict[str, Any]:
        if "Remaining facts:" in prompt and not flagged:
            for n, line in re.findall(r"^(\d+)\. (.*)$", prompt, re.MULTILINE):
                if word in line:
                    flagged.append(int(n))
                    return {"same": [int(n)]}
        return {"same": []}

    return decide


def test_forget_memories_regrow_rounds_reuse_cache_and_leave_no_check_behind(
    n3_settings: Settings, fake_api: FakeAPI, requests_by_key: dict[str, str]
) -> None:
    """按事实遗忘、兜底时又追了一轮（D-051）：每一轮的重放都不再重新请求输入没变的对账与消歧
    （以前每轮都整个再付一遍）。找转述与兜底检查的请求里写着被删的事实，最后一次兜底检查之后没有
    forget 替它清理，以前它的缓存条目一直留着。"""
    system, model = _secret_system(n3_settings, fake_api, regrown=_flag_once("Geneva"))
    with system:
        [secret] = [f for f in system.facts("alice") if "Zurich" in f.statement]
        before = model.calls.copy()

        report = system.forget_memories("alice", fact_ids=[secret.id])

        # 第一轮删 b；兜底发现 c 的 "moved to Geneva" 还在透露，第二轮删 c 的出处
        assert report.regrown == 1 and report.sessions_affected == ["s-b", "s-c"]
        assert model.calls["regrown"] - before["regrown"] == 3  # 找转述 + 两次兜底
        assert model.calls["reconcile"] - before["reconcile"] == 1  # 只有第一轮 c 的
        assert model.calls["entity"] == before["entity"]
        assert _sent(model, "reconcile", "lead designer") == 1
        assert _sent(model, "entity", "Initech") == 1
        current = {f.statement for f in system.facts("alice", current_only=True)}
        assert current >= {"The user moved to Bern.", "The user works at Acme."}
        assert not any("Geneva" in f.statement for f in system.facts("alice"))
        _assert_no_cache_entry_with(system, requests_by_key, SECRET_WORDS)
        _assert_no_cache_entry_with(system, requests_by_key, ["Geneva", "Globex", "Bluejay"])


def _traces(system: MemorySystem) -> str:
    return "".join(p.read_text("utf-8") for p in system.settings.paths.traces.glob("*.jsonl"))


def test_forget_cleans_up_even_when_replay_fails(
    n3_settings: Settings, fake_api: FakeAPI, requests_by_key: dict[str, str]
) -> None:
    """重放半途失败（对账时余额不足）：原始日志已经删了，重试时找不到要删的发言，所以 trace、
    向量缓存与 LLM 缓存当场就要清理，异常照常抛出；没做完的巩固任务留着，下次巩固时补齐。"""
    model = ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    broke = threading.Event()

    def reply(body: dict[str, Any]) -> Any:
        reconciling = body["messages"][0]["content"].startswith("You maintain a long-term memory")
        return NO_BALANCE if broke.is_set() and reconciling else model(body)

    fake_api.chat_replies = [reply]
    conv = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        system.ingest(conv, consolidate=True)
        turn = system.db.get_turns("demo", "s2")[0]
        question = f"{turn.text} 我现在在哪工作？"  # 提问里带着要删的原话
        system.ask("demo", question, mode="l1")
        system.search("demo", turn.text)  # 原话本身做过检索查询（代理与 Agent 都这样）
        assert turn.text in _traces(system)
        assert isinstance(system.embedder, CachedEmbedder)
        assert system.embedder.cached([turn.text, question], "query")
        old_round = next(u.text for u in system.db.list_units("demo") if turn.text in u.text)

        broke.set()
        # s4 的对账列过 s2 的预算：遗忘 s2 之后它的输入变了，要重新请求（输入没变的对账命中缓存）
        with pytest.raises(openai.APIStatusError, match="Insufficient Balance"):
            system.forget("demo", session_ids=["s2"])
        assert all(t.id != turn.id for t in system.db.get_turns("demo"))
        assert turn.text not in _traces(system)
        assert not system.embedder.cached([old_round], "document")
        assert not system.embedder.cached([turn.text, question], "query")
        # s4 原来的对账列着 s2 的预算：重放没用到它就失败了，照样删掉（标记的条目都核对过）
        _assert_no_cache_entry_with(system, requests_by_key, [turn.text, "预算约 8000 元"])
        assert any(j["status"] == "pending" for j in system.db.get_jobs("demo"))

        broke.clear()
        system.consolidate("demo")
        assert all(j["status"] == "done" for j in system.db.get_jobs("demo"))
        core = system.core_memory("demo")
        assert core is not None and "新加坡" in core.text


SECRET_QUESTION = "我在新加坡哪家电商公司工作来着？"


def no_space_for_trace_rewrites(monkeypatch: pytest.MonkeyPatch) -> None:
    """重写 trace 文件时磁盘满：.part 写了一点就抛 ENOSPC。重写要先有这个文件那么大的空间，
    提交遗忘只要几 KB、追加一行 trace 只要一行，它们照常成功。"""
    real = Path.write_text

    def full(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.name.endswith(".jsonl.part"):
            real(self, "partial", encoding="utf-8")
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full)


def test_a_trace_file_that_cannot_be_rewritten_does_not_stop_the_forget(
    system: MemorySystem, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """遗忘提交之后重写 trace 文件失败（磁盘满）：以前 OSError 从提交之后抛出（HTTP 回 500），
    跳过了清理向量缓存，还留下一个 .part；重试找不到要删的发言，提问的查询向量、被删文本的向量与
    trace 里的原话都再也清不掉。现在整个文件删掉（删除不要空间），记一条警告，遗忘照常做完；同一
    天别的空间的 trace 也没了，它们提问的查询向量一起删（否则之后它们的遗忘再也找不到）。"""
    assert system.tracer is not None and isinstance(system.embedder, CachedEmbedder)
    traces = system.settings.paths.traces
    system.retrieve("demo", SECRET_QUESTION, mode="l1")
    other = Trace("retrieve", space="bob", question="Where did I park the car?")
    system.tracer.write(other)
    system.embedder.embed(["Where did I park the car?"], "query")
    s4_texts = [u.text for u in system.db.list_units("demo") if SECRET in u.text]
    assert system.embedder.cached(s4_texts, "document") == set(s4_texts)
    assert SECRET in _traces(system)

    no_space_for_trace_rewrites(monkeypatch)
    with caplog.at_level("WARNING", "cuemem.core.trace"):
        report = system.forget_memories("demo", session_ids=["s4"], report_rebuild_errors=True)
    monkeypatch.undo()

    assert report.turns_deleted == 4 and report.traces_removed >= 1
    assert [r.name for r in caplog.records] == ["cuemem.core.trace"]
    assert list(traces.glob("*.part")) == []
    assert SECRET not in _traces(system) and "park the car" not in _traces(system)
    assert system.embedder.cached([SECRET_QUESTION, "Where did I park the car?"], "query") == set()
    assert system.embedder.cached(s4_texts, "document") == set()


def embed_cache_full_once(system: MemorySystem, monkeypatch: pytest.MonkeyPatch) -> None:
    """向量缓存第一次删除时磁盘满（sqlite3 的 database or disk is full），之后照常。"""
    assert isinstance(system.embedder, CachedEmbedder)
    cache = system.embedder.cache
    real, calls = cache.delete_many, []

    def full(keys: Sequence[str]) -> int:
        calls.append(len(keys))
        if len(calls) == 1:
            raise sqlite3.OperationalError("database or disk is full")
        return real(keys)

    monkeypatch.setattr(cache, "delete_many", full)


def test_an_embedding_cache_that_cannot_be_purged_does_not_stop_the_forget(
    system: MemorySystem, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """遗忘提交之后清理向量缓存出错（磁盘满，或与评测共用的 embed.sqlite 被锁住）：以前从提交
    之后抛出，跳过了 scrub；重试找不到要删的发言，trace 又已经删了（提问只记在那里），提问的查询
    向量再也清不掉。现在记一条警告、在遗忘的 trace 里记下 embed_cache_left，遗忘照常返回。"""
    system.retrieve("demo", SECRET_QUESTION, mode="l1")
    embed_cache_full_once(system, monkeypatch)
    with caplog.at_level("WARNING", "cuemem.memory"):
        report = system.forget_memories("demo", session_ids=["s4"], report_rebuild_errors=True)
    assert report.turns_deleted == 4 and system.db.get_turns("demo", "s4") == []
    assert [r.name for r in caplog.records if r.levelname == "WARNING"] == ["cuemem.memory"]
    records = [json.loads(line) for line in _traces(system).splitlines()]
    [forget] = [r for r in records if r["name"] == "forget"]
    assert forget["attrs"]["embed_cache_left"] is True
    assert SECRET not in _traces(system)


def test_an_embedding_cache_failure_after_delete_space_does_not_raise(
    system: MemorySystem, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """删除空间删掉 trace 之后清理其中提问的查询向量出错：同样只记警告（以前抛出，delete_all 与
    reset 的重试找不到这些 trace）。"""
    assert isinstance(system.embedder, CachedEmbedder)
    system.retrieve("demo", SECRET_QUESTION, mode="l1")
    embed_cache_full_once(system, monkeypatch)
    with caplog.at_level("WARNING", "cuemem.memory"):
        assert system.purge_space_traces(["demo"]) >= 1
    assert [r.name for r in caplog.records if r.levelname == "WARNING"] == ["cuemem.memory"]
    assert SECRET_QUESTION not in _traces(system)
    assert system.embedder.cached([SECRET_QUESTION], "query") == {SECRET_QUESTION}  # 不重试


def test_a_trace_line_that_is_not_utf8_does_not_stop_the_forget(system: MemorySystem) -> None:
    """trace 文件里一行不是 UTF-8（这一版之前追加写到一半，断在一个汉字中间，下一次追加接在它
    后面）：以前遗忘读到它就抛 UnicodeDecodeError，从此这个服务上每次遗忘都在提交之后报错
    （HTTP 400），过期的 trace 也删不掉。现在这一行原样留着，遗忘照常删掉该删的，过期时整个文件
    照常删掉。"""
    assert system.tracer is not None and isinstance(system.embedder, CachedEmbedder)
    traces = system.settings.paths.traces
    system.retrieve("demo", SECRET_QUESTION, mode="l1")
    [day] = traces.glob("*.jsonl")
    with day.open("ab") as f:
        f.write(b'{"attrs": {"space": "bob", "question": "\xe6\x88')
        f.write(b'{"attrs": {"space": "carol", "question": "hi"}}\n')

    report = system.forget_memories("demo", session_ids=["s4"], report_rebuild_errors=True)
    assert report.turns_deleted == 4 and report.traces_removed >= 1
    raw = day.read_bytes()
    assert SECRET.encode() not in raw and b"\xe6\x88{" in raw
    assert system.embedder.cached([SECRET_QUESTION], "query") == set()
    report = system.forget_memories("demo", session_ids=["s1"], report_rebuild_errors=True)
    assert report.turns_deleted > 0
    files, _ = system.tracer.prune(date.today() + timedelta(days=2))
    assert files >= 1 and list(traces.glob("*.jsonl")) == []


PIN_QUESTION = "what is my card PIN?"
SUB_QUERY = "user's bank card PIN"
FOLLOW_UP = "bank card PIN 4321 Commerzbank"  # 检查的模型看过记忆，追加的查询复述了被删的 PIN


def _l2_model() -> ScriptedModel:
    """L2：规划给一个子查询；检查证据够不够的模型说不够，追加一个查询（第二次检查时它已经查过，
    没有新的查询，检查结束）。"""
    return ScriptedModel(
        echo_extraction,
        planner=lambda prompt: {"answer_type": "fact", "sub_queries": [SUB_QUERY]},
        checker=lambda prompt: {"sufficient": False, "missing": "bank", "queries": [FOLLOW_UP]},
    )


def _pin_and_ask(system: MemorySystem, space: str, text: str, images: Sequence[str] = ()) -> None:
    turn = TurnIn(role="user", text=text, ts=AUG, images=list(images))
    system.append(space, SessionIn(session_id="s1", started_at=AUG, turns=[turn]), 0)
    system.consolidate(space)
    system.retrieve(space, PIN_QUESTION, mode="l2")


def test_forget_purges_the_query_vectors_of_l2_sub_queries_and_follow_ups(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """L2 的子查询与检查之后追加的查询同样向量化成查询向量，追加的查询还可能复述被删的内容
    （检查的模型看过记忆）。以前遗忘、删除空间与过期只删提问的查询向量，这些谁也找不到；它们本来
    就记在 trace 里（plan 的 sub_queries、check 的 queries），现在按 trace 一起删。"""
    fake_api.chat_replies = [_l2_model()]
    queries = [PIN_QUESTION, SUB_QUERY, FOLLOW_UP]
    pin = "My Commerzbank card PIN is 4321, keep it safe"
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        assert isinstance(system.embedder, CachedEmbedder)
        _pin_and_ask(system, "eve", pin)
        assert system.embedder.cached(queries, "query") == set(queries)
        system.forget("eve", session_ids=["s1"])
        assert system.embedder.cached(queries, "query") == set()

        _pin_and_ask(system, "eve", pin)
        system.delete_space("eve")
        assert system.embedder.cached(queries, "query") == set()

        _pin_and_ask(system, "eve", pin)  # 过期：trace 文件改成很久以前的日期
        for n, path in enumerate(sorted(system.settings.paths.traces.glob("*.jsonl")), 1):
            path.rename(path.with_name(f"200001{n:02d}.jsonl"))
        files, purged = system.prune_traces(30)
        assert files >= 1 and purged == 3
        assert system.embedder.cached(queries, "query") == set()


def _shared_image_cache(system: MemorySystem) -> DashScopeMultimodalEmbedder:
    """换上 QwenCloud 的图片向量器（MockTransport 回假的向量），与文本向量共用一个缓存，与
    from_settings 相同（离线测试的哈希向量器没有缓存）。"""

    def handle(request: httpx2.Request) -> httpx2.Response:
        contents = json.loads(request.content)["input"]["contents"]
        rows = [[float(len(json.dumps(c)) % 7 + 1)] + [0.5] * 15 for c in contents]
        embeddings = [{"index": i, "embedding": row} for i, row in enumerate(rows)]
        return httpx2.Response(
            200, json={"output": {"embeddings": embeddings}, "usage": {"input_tokens": 3}}
        )

    assert isinstance(system.embedder, CachedEmbedder)
    cfg = system.settings.multimodal_embedding.model_copy(update={"dim": 16})
    transport = httpx2.Client(transport=httpx2.MockTransport(handle))
    mm = DashScopeMultimodalEmbedder(
        cfg, "sk-test", http_client=transport, cache=system.embedder.cache
    )
    system.image_embedder = system.writer.image_embedder = system.l1.image_embedder = mm
    return mm


def test_forget_purges_the_multimodal_query_and_image_vectors(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """有图的空间里 L1 把提问、子查询与追加的查询另外向量化成图片空间里的文字向量，写入时图片
    按内容向量化，都在同一个缓存文件里。以前遗忘、删除空间与过期只删文本向量器的（D-070 留待
    以后的第 6 项）。图片的向量与原图文件同一个规则：别的空间还有同一张图时留着，那个空间也删了
    才删。"""
    fake_api.chat_replies = [_l2_model()]
    card = tmp_path / "card.png"
    Image.new("RGB", (40, 30), "blue").save(card)
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        mm = _shared_image_cache(system)
        cache = mm.cache
        assert cache is not None
        text = "My bank card, PIN 4321 <image>"
        _pin_and_ask(system, "bob", text, [str(card)])  # 同一张图在两个空间里
        _pin_and_ask(system, "eve", text, [str(card)])
        texts = [mm._key("text", content_hash(q)) for q in (PIN_QUESTION, SUB_QUERY, FOLLOW_UP)]
        image = mm._key("image", hashlib.sha256(card.read_bytes()).hexdigest())
        assert set(cache.get_many([*texts, image])) == {*texts, image}

        report = system.forget("eve", session_ids=["s1"])
        assert set(cache.get_many([*texts, image])) == {image}  # bob 的还在
        assert report.embed_cache_purged >= 3
        system.delete_space("bob")
        assert cache.get_many([*texts, image]) == {}


def _sdk(settings: Settings, fake_api: FakeAPI, reply: Any = None, **kwargs: Any) -> Memory:
    fake_api.chat_replies = [reply or ScriptedModel(where_i_live, reconcile=moved)]
    return Memory(settings, http_client=fake_api.client(), **kwargs)


def _statements(memory: Memory) -> list[str]:
    return sorted(f["memory"] for f in memory.get_all(user_id="u")["results"])


def _consistent(system: MemorySystem) -> None:
    """向量库里的点与数据库里的记忆单元一一对应。"""
    assert set(system.vectors.unit_ids("u")) == {u.id for u in system.db.list_units("u")}


def _pin_chat(memory: Memory) -> str:
    """三段会话，最后一段（c）是一问一答，用户那句带着要删的 PINCODE，检索过一次（trace 里有）。
    返回这句的 ID。"""
    memory.add("I live in Berlin", user_id="u", run_id="a", timestamp=AUG, wait=True)
    day2 = AUG + timedelta(days=1)
    memory.add("I like jazz", user_id="u", run_id="b", timestamp=day2, wait=True)
    out = memory.add(
        [
            {"role": "user", "content": "I like PINCODE chess"},
            {"role": "assistant", "content": "Chess it is."},
        ],
        user_id="u",
        run_id="c",
        timestamp=SEP,
        wait=True,
    )
    memory.search("PINCODE chess?", user_id="u", mode="l1")
    return str(out["turn_ids"][0])


def _forgotten(memory: Memory) -> None:
    """_pin_chat 里用户那句删干净了：事实、trace 与"轮"里都没有它，向量与记录一一对应。"""
    system = memory.system
    assert _statements(memory) == ["I like jazz", "The user lives in Berlin."]
    rounds = system.db.list_units("u", kinds=["round"], session_ids=["c"])
    assert [u.text for u in rounds] == ["Assistant: Chess it is."]
    assert "PINCODE" not in _traces(system)
    _consistent(system)


@pytest.mark.parametrize("broken", ["upsert", "delete"])
def test_forget_vector_store_error_deletes_nothing(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch, broken: str
) -> None:
    """遗忘的第一步向量库出错（写新"轮"的向量，或删要失去记录的向量）：数据库与 LLM 缓存都还
    没动，什么都没删，异常照常抛出（report_rebuild_errors 也一样），原样重试就删干净。以前先删
    发言、各自提交，之后才动向量：出错时发言没了，由它抽出的事实还是当前有效的；删派生向量出错时
    任务也一个不剩。"""
    with _sdk(n3_settings, fake_api, background=False) as memory:
        system = memory.system
        user_turn = _pin_chat(memory)
        facts = {f.id for f in system.db.get_facts("u")}
        real = getattr(system.vectors, broken)

        def unreachable(space: str, ids: Sequence[str], *rest: Any) -> Any:
            if broken == "upsert" or facts & set(ids):
                raise ConnectionError("qdrant unreachable")
            return real(space, ids, *rest)

        assert system.llm.cache is not None
        cached = len(system.llm.cache)
        monkeypatch.setattr(system.vectors, broken, unreachable)
        with pytest.raises(ConnectionError):
            system.forget("u", turn_ids=[user_turn], report_rebuild_errors=True)
        monkeypatch.setattr(system.vectors, broken, real)

        assert len(system.db.get_turns("u")) == 4
        assert "I like PINCODE chess" in _statements(memory)
        assert [j["status"] for j in system.db.get_jobs("u")] == ["done"] * 3
        assert len(system.llm.cache) == cached
        report = system.forget("u", turn_ids=[user_turn], report_rebuild_errors=True)
        assert report.turns_deleted == 1 and report.rebuild_error is None
        _forgotten(memory)


def test_forget_during_an_embedding_outage_deletes_nothing(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """按发言删（HTTP 的 forget 带 turn_ids、CLI 的 --turn）时向量接口不可用：剩下的那句组成的
    "轮"文本变了、要向量化，在删任何东西之前就出错，原样重试就删干净。以前发言已经删了，由它
    抽出的事实还是当前有效的、还检索得到，剩下的那句再也没有"轮"，重试找不到要删的发言。"""
    with _sdk(n3_settings, fake_api, background=False) as memory:
        system = memory.system
        user_turn = _pin_chat(memory)
        assert isinstance(system.embedder, CachedEmbedder)
        inner = system.embedder.inner

        def down(texts: Sequence[str], kind: Any) -> Any:
            raise EmbeddingAPIError("向量接口返回 HTTP 503", status=503)

        monkeypatch.setattr(inner, "embed", down)
        with pytest.raises(EmbeddingAPIError):
            system.forget_memories("u", turn_ids=[user_turn], report_rebuild_errors=True)
        monkeypatch.undo()

        assert len(system.db.get_turns("u")) == 4
        assert "I like PINCODE chess" in _statements(memory)
        again = system.forget_memories("u", turn_ids=[user_turn], report_rebuild_errors=True)
        assert again.turns_deleted == 1 and again.rebuild_error is None
        _forgotten(memory)


def test_forget_deletes_and_clears_in_one_transaction(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """换上新的"轮"、删发言、清空派生层与重新登记任务在同一个事务里：其中一步出错（磁盘满）时
    一起撤销，什么都没删，原样重试就删干净（向量已经删掉的派生记忆在重放时重建）。以前删发言
    先提交了：之后出错时由被删发言抽出的事实还是当前有效的，按事实 ID 再删也找不到发言。"""
    with _sdk(n3_settings, fake_api, background=False) as memory:
        system = memory.system
        user_turn = _pin_chat(memory)

        def disk_full(space_id: str) -> list[str]:
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(system.db, "clear_derived", disk_full)
        with pytest.raises(sqlite3.OperationalError):
            system.forget("u", turn_ids=[user_turn], report_rebuild_errors=True)
        monkeypatch.undo()

        assert len(system.db.get_turns("u")) == 4
        rounds = system.db.list_units("u", kinds=["round"], session_ids=["c"])
        assert [u.text for u in rounds] == ["User: I like PINCODE chess\nAssistant: Chess it is."]
        assert "I like PINCODE chess" in _statements(memory)
        assert [j["status"] for j in system.db.get_jobs("u")] == ["done"] * 3
        assert system.forget("u", turn_ids=[user_turn]).turns_deleted == 1
        _forgotten(memory)


def test_a_forget_whose_transaction_fails_keeps_the_reconciliation_cache(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """遗忘的事务出错（磁盘满）时日志与派生层都没动：遗忘开始时标记的对账与消歧条目全部留下，
    不留标记。以前外层的 finally 照样核对，重放还没开始、一条都没用到，这个空间的对账与消歧
    缓存全部删掉，重试时每次对账都要重新付费。受影响会话的抽取与回答缓存、派生记忆的向量在事务
    之前已经删了（先删副本、后删日志），重试的重放补上。"""
    with _sdk(n3_settings, fake_api, background=False) as memory:
        system = memory.system
        user_turn = _pin_chat(memory)
        later = SEP + timedelta(days=1)
        memory.add("I moved to Munich", user_id="u", run_id="d", timestamp=later, wait=True)
        cache = system.llm.cache
        assert cache is not None

        def reconciliations() -> set[str]:
            rows = cache._conn.execute(
                "SELECT key FROM llm_cache WHERE scope = ?", (scope("u", "consolidate"),)
            )
            return {key for (key,) in rows}

        reconciled = reconciliations()
        assert reconciled  # d 的新住址与 a 的旧住址对过账

        def disk_full(space_id: str) -> list[str]:
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(system.db, "clear_derived", disk_full)
        with pytest.raises(sqlite3.OperationalError):
            system.forget("u", turn_ids=[user_turn], report_rebuild_errors=True)
        monkeypatch.undo()

        assert len(system.db.get_turns("u")) == 5
        assert reconciliations() == reconciled
        assert cache._conn.execute("SELECT COUNT(*) FROM llm_cache_held").fetchone() == (0,)
        report = system.forget("u", turn_ids=[user_turn], report_rebuild_errors=True)
        assert report.turns_deleted == 1 and report.rebuild_error is None
        assert _statements(memory) == ["I like jazz", "The user lives in Munich."]
        assert reconciliations() == reconciled  # 重放原样用到了它：没有重新请求
        _consistent(system)


def test_sdk_delete_says_so_when_nothing_was_deleted(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """事实出处的发言已经不在日志里（之前的版本遗忘到一半出错留下的：发言与"轮"删了，事实还是
    当前有效的）：按 ID 删它一句发言都删不到，回 "Nothing deleted" 并让调用方重建，不再回
    "Memory deleted"。"""
    with _sdk(n3_settings, fake_api, background=False) as memory:
        system = memory.system
        _pin_chat(memory)
        [fact] = [f for f in system.db.get_facts("u") if "PINCODE" in f.statement]
        rounds = system.db.list_units("u", kinds=["round"], session_ids=["c"])
        system.db.delete_units("u", [u.id for u in rounds])
        system.db.delete_turns("u", [t.id for t in system.db.get_turns("u", "c")])

        out = memory.delete(fact.id)

        assert out["message"] == "Nothing deleted" and out["turns_deleted"] == 0
        assert "consolidate(rebuild=True)" in out["warning"]
        memory.consolidate(user_id="u", rebuild=True)
        assert _statements(memory) == ["I like jazz", "The user lives in Berlin."]


def test_forget_does_not_extract_log_only_sessions(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """遗忘之后的重放只为原来有任务的会话登记（与会话又长了时的重放相同）：infer=False 只记了
    日志的会话（"之后调用 consolidate 再抽取"）不会因为删了别的记忆就被抽取。以前遗忘为每个
    会话都登记任务，删一条无关的事实，证件号就进了事实。显式的 consolidate 照旧抽取它。"""
    with _sdk(n3_settings, fake_api, background=True) as memory:
        memory.add("My passport number is X123", user_id="u", run_id="private", infer=False)
        memory.add("I live in Berlin", user_id="u", timestamp=AUG)
        memory.add("I moved to Munich", user_id="u", timestamp=SEP)
        memory.wait()
        munich = next(f for f in memory.get_all(user_id="u")["results"] if "Munich" in f["memory"])

        memory.delete(munich["id"])
        memory.wait()

        assert _statements(memory) == ["The user lives in Berlin."]
        assert "private" not in {j["session_id"] for j in memory.system.db.get_jobs("u")}
        memory.consolidate(user_id="u")
        assert "My passport number is X123" in _statements(memory)


def test_forget_keeps_moderated_and_failed_outcomes(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """遗忘之后的重放，内容没变的会话保留原来的结局：内容审核拒绝过的不再发给两家（拒绝不进
    缓存，以前每删一次都再请求一遍），失败的保留失败次数与退避时间，不在遗忘里同步再试一次。"""
    model = ScriptedModel(where_i_live, reconcile=moved)

    def reply(body: dict[str, Any]) -> Any:
        if body["messages"][0]["content"].startswith("You build the long-term memory"):
            said = body["messages"][-1]["content"]
            if "forbidden" in said:
                return REJECT
            if "BROKEN" in said:
                return NO_BALANCE
        return model(body)

    def extractions(word: str) -> int:  # 遗忘时找转述的请求里可以有它们，不算
        return sum(
            body["messages"][0]["content"].startswith("You build the long-term memory")
            and word in body["messages"][-1]["content"]
            for body in fake_api.chat_requests
        )

    with _sdk(n3_settings, fake_api, reply, background=True) as memory:
        memory.add("some forbidden content", user_id="u", run_id="mod", timestamp=AUG)
        memory.add("BROKEN extraction", user_id="u", run_id="broken", timestamp=AUG)
        memory.add("I live in Berlin", user_id="u", run_id="a", timestamp=AUG)
        memory.add("I moved to Munich", user_id="u", run_id="b", timestamp=SEP)
        memory.wait()

        def outcomes() -> dict[str, tuple[str, int, str | None]]:
            jobs = memory.system.db.get_jobs("u")
            return {
                j["session_id"]: (j["status"], j["attempts"], j["stats"].get("skipped"))
                for j in jobs
                if j["session_id"] in ("mod", "broken")
            }

        before = outcomes()
        assert before == {"mod": ("done", 0, "content_filter"), "broken": ("failed", 1, None)}
        sent = extractions("forbidden"), extractions("BROKEN")
        munich = next(f for f in memory.get_all(user_id="u")["results"] if "Munich" in f["memory"])

        memory.delete(munich["id"])

        assert (extractions("forbidden"), extractions("BROKEN")) == sent
        assert outcomes() == before
        assert _statements(memory) == ["The user lives in Berlin."]


def _fails_once(prefix: str) -> tuple[Any, dict[str, int]]:
    """回复函数：系统提示以 prefix 开头的请求，把 failures 设成 1 之后的第一次回余额不足（402），
    之后供应商恢复。"""
    model = ScriptedModel(where_i_live, reconcile=moved)
    left = {"failures": 0}

    def reply(body: dict[str, Any]) -> Any:
        if left["failures"] and body["messages"][0]["content"].startswith(prefix):
            left["failures"] -= 1
            return NO_BALANCE
        return model(body)

    return reply, left


def _settles(memory: Memory, expected: list[str]) -> list[str]:
    """等后台线程把事实补到 expected（最多 10 秒），返回最后看到的。"""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _statements(memory) != expected:
        time.sleep(0.05)
    return _statements(memory)


def test_a_replay_that_raised_is_handed_to_the_worker(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """遗忘之后的重放抛出了异常（对账时余额不足）：删除已经做了（warning），没做完的任务交给
    后台线程按退避补齐。以前任务留着却没人提交，要等这个空间的下一次写入或重启，在那之前被删
    事实之后的住址与喜好都不见了。"""
    reply, left = _fails_once("You maintain a long-term memory")
    with _sdk(n3_settings, fake_api, reply, background=True) as memory:
        assert memory.system.worker is not None
        memory.system.worker.retry_base_s = 0.05  # 后台线程的退避，测试里缩短
        memory.add("I live in Berlin", user_id="u", timestamp=AUG)
        memory.add("I moved to Munich", user_id="u", timestamp=SEP)
        memory.add("I moved to Paris", user_id="u", timestamp=SEP + timedelta(days=1))
        memory.add("I like jazz", user_id="u", timestamp=SEP + timedelta(days=2))
        memory.wait()
        everything = memory.get_all(user_id="u", include_superseded=True)["results"]
        munich = next(f for f in everything if "Munich" in f["memory"])

        left["failures"] = 1
        out = memory.delete(munich["id"])

        assert out["message"] == "Memory deleted" and out.get("warning")
        expected = ["I like jazz", "The user lives in Paris."]
        assert _settles(memory, expected) == expected


def test_a_failed_extraction_in_the_replay_is_retried_by_the_worker(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """遗忘之后的重放里一个会话抽取失败（不抛出，任务记为失败）：同样交给后台线程，按退避重试。
    以前没人提交，要等这个空间的下一次写入或重启，会话里没删的那句一直没有事实。"""
    reply, left = _fails_once("You build the long-term memory")
    with _sdk(n3_settings, fake_api, reply, background=True) as memory:
        assert memory.system.worker is not None
        memory.system.worker.retry_base_s = 0.05
        chat = [
            {"role": "user", "content": "I live in Berlin"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "I like jazz"},
            {"role": "assistant", "content": "nice"},
        ]
        memory.add(chat, user_id="u", run_id="r1", timestamp=AUG)
        memory.wait()
        berlin = next(f for f in memory.get_all(user_id="u")["results"] if "Berlin" in f["memory"])

        left["failures"] = 1
        memory.delete(berlin["id"])

        assert _settles(memory, ["I like jazz"]) == ["I like jazz"]


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

        assert not worker.errors
        [report] = out
        assert not isinstance(report, Exception) and report.turns_deleted == 1
        assert system.db.get_facts("alice") == []
        assert system.vectors.unit_ids("alice") == []
        assert system.llm.cache is not None
        assert not any(PIN in r for r in system.llm.cache.responses())


def test_forget_waits_for_an_answer_in_flight(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """回答模型还在生成（它看过要删的记忆）时遗忘：遗忘等这次回答写完缓存与 trace 再清理，
    不能先清理、再让回答带着被删的内容写进 LLM 缓存与 trace。"""
    entered, release = threading.Event(), threading.Event()

    def extract(prompt: str) -> dict[str, Any]:
        fact = {
            "statement": f"The user's PIN is {PIN}.",
            "subject": "User",
            "predicate": "pin",
            "object": PIN,
            "importance": 5,
            "turns": [0],
        }
        return {"summary": "", "facts": [fact] if PIN in prompt else [], "entities": []}

    def reader(prompt: str) -> dict[str, Any]:
        entered.set()
        release.wait(10)  # 回答模型还在生成
        return {"answer": f"Your PIN is {PIN}.", "citations": [], "evidence_sufficient": True}

    fake_api.chat_replies = [ScriptedModel(extract, reader=reader)]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        session = SessionIn(
            session_id="s1",
            started_at=datetime(2026, 9, 1, 10),
            turns=[TurnIn(role="user", text=f"My PIN is {PIN}, keep it safe")],
        )
        system.append("alice", session, 0)
        system.consolidate("alice")
        asking, answered = _in_thread(system.ask, "alice", "What is my PIN?", mode="l1")
        assert entered.wait(5)
        forgetting, out = _in_thread(system.forget, "alice", session_ids=["s1"])
        forgetting.join(0.3)
        assert forgetting.is_alive()  # 等着这次回答
        release.set()
        asking.join(10)
        forgetting.join(10)

        [answer] = answered
        assert not isinstance(answer, Exception) and PIN in answer.answer
        [report] = out
        assert not isinstance(report, Exception) and report.turns_deleted == 1
        assert system.llm.cache is not None
        assert not any(PIN in r for r in system.llm.cache.responses())
        traces = n3_settings.paths.traces.glob("*.jsonl")
        assert not any(PIN in p.read_text("utf-8") for p in traces)

        assert system.ask("alice", "What is my PIN?", mode="l1").answer  # 之后照常回答


@pytest.mark.parametrize("entry", ["sdk_search", "recall_listing"])
def test_forget_waits_for_a_raised_search_in_flight(
    n3_settings: Settings,
    fake_api: FakeAPI,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    """SDK 的 search 与 recall_listing（MCP、各框架集成的 recall）控制器选了 L0、改走 L1 时检索
    两次，第二次直接打包探针的候选：遗忘要等整次检索做完。以前两次各自持有读，探针做完、打包之前
    遗忘插了进来，返回的是已经删掉的事实与对话片段，第二次检索的 trace 也在遗忘清理之后才写下。"""
    from cuemem import Memory
    from cuemem.evidence import recall_listing

    fake_api.chat_replies = [ScriptedModel(where_i_live, reconcile=moved)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        system = memory.system
        memory.add("I like green tea", user_id="u", wait=True)
        memory.add("My PIN is 4321 and I like chess", user_id="u", run_id="pin", wait=True)
        assert memory.search("Hi there", user_id="u", min_level="l0")["level"] == "L0"
        doomed = {u.id for u in system.db.list_units("u", kinds=["round"]) if u.session_id == "pin"}
        doomed |= {f.id for f in system.db.get_facts("u") if f.session_id == "pin"}
        assert doomed

        run, pack = system.probe.run, system.l1.pack
        forgetting: list[tuple[threading.Thread, list[Any]]] = []
        at_pack: list[int] = []

        def probe_while_a_forget_arrives(*args: Any, **kwargs: Any) -> Any:
            out = run(*args, **kwargs)
            if not forgetting:  # 探针做完时来了一次遗忘，等着这次检索
                forgetting.append(_in_thread(system.forget, "u", session_ids=["pin"]))
                while system._gate("u")._waiting == 0:
                    time.sleep(0.001)
            return out

        def recording_pack(*args: Any, **kwargs: Any) -> Any:
            at_pack.append(len(system.db.get_turns("u", "pin")))
            return pack(*args, **kwargs)

        monkeypatch.setattr(system.probe, "run", probe_while_a_forget_arrives)
        monkeypatch.setattr(system.l1, "pack", recording_pack)
        if entry == "sdk_search":
            level = memory.search("Hi there", user_id="u")["level"]
        else:
            level = recall_listing(system, "u", "Hi there", at_least_l1=True)["level"]
        [(thread, out)] = forgetting
        thread.join(10)

        assert level == "L1"
        [report] = out
        assert not isinstance(report, Exception) and report.turns_deleted == 1
        assert at_pack == [1]  # 打包时被删的发言还在：遗忘在这次检索之后
        assert system.db.get_turns("u", "pin") == []
        for path in n3_settings.paths.traces.glob("*.jsonl"):
            for line in path.read_text("utf-8").splitlines():
                if json.loads(line)["attrs"].get("space") == "u":
                    assert "4321" not in line and not any(i in line for i in doomed)


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

        assert not worker.errors
        [report] = out
        assert not isinstance(report, Exception) and report.turns_deleted > 0
        assert all(f.session_id != "s1" for f in system.facts("demo"))
        assert all(j["status"] == "done" for j in system.db.get_jobs("demo"))
        core = system.core_memory("demo")
        assert core is not None and "林晓" not in core.text and "新加坡" in core.text


def _one_turn(session_id: str, text: str) -> SessionIn:
    return SessionIn(
        session_id=session_id,
        started_at=datetime(2026, 9, 1, 10),
        turns=[TurnIn(role="user", text=text)],
    )


@pytest.mark.parametrize("how", ["forget", "delete_space"])
def test_reads_go_on_while_a_forget_waits_for_background_consolidation(
    n3_settings: Settings, fake_api: FakeAPI, how: str
) -> None:
    """后台线程正在巩固（慢的抽取、限流退避、长的重放）时来了遗忘：遗忘等巩固锁的这段时间里，
    这个空间的检索照常进行。以前遗忘先拿读写门、再等巩固锁，新的读取排在它后面，要等整轮
    巩固做完。"""
    entered, release = threading.Event(), threading.Event()

    def extract(prompt: str) -> dict[str, Any]:
        if "slow" in prompt:
            entered.set()
            release.wait(10)  # 后台线程里慢的抽取
        return EMPTY_EXTRACTION

    fake_api.chat_replies = [ScriptedModel(extract)]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        system.append("alice", _one_turn("s1", f"My PIN is {PIN}"), 0)
        system.consolidate("alice")
        worker = system.start_worker()
        system.append("alice", _one_turn("s2", "a slow session to extract"), 0)
        assert entered.wait(5)
        if how == "forget":
            forgetting, out = _in_thread(system.forget, "alice", session_ids=["s1"])
        else:
            forgetting, out = _in_thread(system.delete_space, "alice")
        forgetting.join(0.3)
        assert forgetting.is_alive()  # 等着后台线程的巩固
        reading, found = _in_thread(system.retrieve, "alice", "What is my PIN?", mode="l1")
        reading.join(2)
        went_on = not reading.is_alive()
        release.set()
        forgetting.join(10)
        reading.join(10)
        worker.wait()

        assert went_on, "检索排在等巩固锁的遗忘后面"
        [recall] = found
        assert not isinstance(recall, Exception)
        [report] = out
        assert not isinstance(report, Exception) and report.turns_deleted >= 1
        assert PIN not in json.dumps([t.text for t in system.db.get_turns("alice")])


def test_forget_inside_a_read_is_refused_instead_of_deadlocking(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """持有读（reading）的线程里遗忘、删除空间或巩固：直接报错。遗忘先拿巩固锁、再等进行中的
    读取做完，读取里再去拿巩固锁就互相等死（两个各自持有读的线程同时遗忘，以前也会等死）。"""
    fake_api.chat_replies = [ScriptedModel(lambda prompt: EMPTY_EXTRACTION)]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        system.append("alice", _one_turn("s1", "a note to keep around"), 0)
        with system.reading("alice"):
            for call in (
                lambda: system.forget("alice", session_ids=["s1"]),
                lambda: system.delete_space("alice"),
                lambda: system.consolidate("alice"),
            ):
                with pytest.raises(RuntimeError, match="reading"):
                    call()
            assert len(system.db.get_turns("alice")) == 1
        assert system.forget("alice", session_ids=["s1"]).turns_deleted == 1
