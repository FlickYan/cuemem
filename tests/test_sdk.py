"""Python SDK（N12，D-061）：Mem0 形状的接口，CueMem 的语义（版本链、控制器、遗忘）。全部离线。"""

from __future__ import annotations

import asyncio
import base64
import io
import itertools
import json
import os
import re
import sqlite3
import tempfile
import threading
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pytest
from fakes import NO_BALANCE, FakeAPI
from PIL import Image
from scripted import ScriptedModel

from cuemem import AsyncMemory, Memory
from cuemem.config import PathsConfig, Settings
from cuemem.embed import CachedEmbedder
from cuemem.embed.base import EmbedKind
from cuemem.errors import ConcurrentWriteError, EmbeddingAPIError, IdempotencyConflictError
from cuemem.evidence import RAISED_REASON, format_profile, recall_listing, stored_text
from cuemem.memory import _open_mark
from cuemem.read.l1 import L1Config
from cuemem.read.l2 import L2Config
from cuemem.sdk import _key_hash, _matches, parse_messages, space_for
from cuemem.store.vector import VectorStore

AUG = datetime(2026, 8, 1, 10, 0)
SEP = datetime(2026, 9, 1, 10, 0)


def where_i_live(prompt: str) -> dict[str, Any]:
    """抽取替身：'I live in X' / 'I moved to X' → 单值事实 lives_in；其余发言各记一条多值事实。"""
    facts = []
    for n, text in re.findall(r"\[t(\d+)\] User: (.*)", prompt):
        place = re.search(r"(?:live in|moved to) (\w+)", text)
        if place:
            facts.append(
                {
                    "statement": f"The user lives in {place.group(1)}.",
                    "subject": "User",
                    "predicate": "lives_in",
                    "object": place.group(1),
                    "category": "attribute",
                    "cardinality": "single",
                    "importance": 5,
                    "turns": [int(n)],
                }
            )
        else:
            facts.append(
                {
                    "statement": text,
                    "predicate": "said",
                    "object": text[:40],
                    "cardinality": "multi",
                    "importance": 4,
                    "turns": [int(n)],
                }
            )
    return {"summary": "", "facts": facts, "entities": []}


def moved(prompt: str) -> dict[str, Any]:
    """对账替身：新的住址取代旧的（现实变了）。"""
    return {"decisions": [{"new": 1, "action": "update", "target": "e1"}]}


@pytest.fixture
def model() -> ScriptedModel:
    return ScriptedModel(
        where_i_live,
        reconcile=moved,
        reader=lambda prompt: {
            "answer": "You live in Munich.",
            "citations": ["f1"],
            "evidence_sufficient": True,
        },
    )


@pytest.fixture
def memory(n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel) -> Iterator[Memory]:
    fake_api.chat_replies = [model]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        yield m


def test_add_returns_mem0_style_events_and_keeps_history(memory: Memory) -> None:
    first = memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    assert first["status"] == "consolidated"
    assert [(e["event"], e["memory"]) for e in first["results"]] == [
        ("ADD", "The user lives in Berlin.")
    ]
    berlin = first["results"][0]["id"]

    second = memory.add(
        [
            {"role": "user", "content": "I moved to Munich"},
            {"role": "assistant", "content": "Nice!"},
        ],
        user_id="alice",
        timestamp=SEP,
        wait=True,
    )
    (event,) = second["results"]
    assert event["event"] == "UPDATE"
    assert event["memory"] == "The user lives in Munich."
    assert event["previous_memory"] == ["The user lives in Berlin."]
    assert event["reason"] == "changed"
    munich = event["id"]

    # 当前的事实只有新的；旧版本没有被删除，留在版本链里
    assert [f["memory"] for f in memory.get_all(user_id="alice")["results"]] == [
        "The user lives in Munich."
    ]
    everything = memory.get_all(user_id="alice", include_superseded=True)["results"]
    assert {f["id"] for f in everything} == {berlin, munich}
    assert [
        f["memory"] for f in memory.get_all(user_id="alice", valid_at="2026-08-15")["results"]
    ] == ["The user lives in Berlin."]
    old = memory.get(berlin)
    assert old is not None and old["status"].startswith("no longer true")
    assert old["superseded_by"] == munich and old["space_id"] == "alice"
    for start in (berlin, munich):  # 从链上任何一个版本都能取到整条链
        chain = memory.history(start)
        assert [(v["event"], v["id"]) for v in chain] == [("ADD", berlin), ("UPDATE", munich)]
    assert memory.get("no-such-id") is None and memory.history("no-such-id") == []
    assert "Munich" in memory.profile(user_id="alice")


def test_search_and_answer(memory: Memory) -> None:
    memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    found = memory.search("Where do I live?", user_id="alice", mode="l1", at=SEP)
    assert found["level"] == "L1"
    kinds = {r["type"] for r in found["results"]}
    assert kinds == {"fact", "excerpt"}
    fact = next(r for r in found["results"] if r["type"] == "fact")
    assert fact["memory"] == "The user lives in Berlin." and fact["user_id"] == "alice"
    assert "score" in fact
    assert (
        len(memory.search("Where do I live?", user_id="alice", mode="l1", limit=1)["results"]) == 1
    )

    # Mem0 v3 的写法：filters 与 top_k
    v3 = memory.search("Where do I live?", filters={"user_id": "alice"}, top_k=1, mode="l1")
    assert len(v3["results"]) == 1
    assert memory.get_all(filters={"user_id": "alice"})["results"][0]["object"] == "Berlin"
    # 其余的键是 metadata 的条件（没写过 metadata：什么都不给）；写错的运算符报错
    unknown = {"user_id": "alice", "topic": "home"}
    assert memory.search("Where do I live?", filters=unknown, mode="l1")["results"] == []
    with pytest.raises(ValueError, match="Unsupported filter operators"):
        memory.search("x", filters={"user_id": "alice", "topic": {"regex": "h.*"}})

    auto = memory.search("Where do I live?", user_id="alice", at=SEP)
    assert auto["level"] in {"L0", "L1", "L2"} and auto["reasons"]

    answer = memory.answer("Where do I live?", user_id="alice", mode="l1", at=SEP)
    assert answer["answer"] == "You live in Munich."
    assert answer["citations"][0]["ref"] == "f1"
    assert answer["citations"][0]["memory"] == "The user lives in Berlin."

    # 没有记忆的用户：不检索，直接 L0
    empty = memory.search("Where do I live?", user_id="bob")
    assert empty == {
        "results": [],
        "images": [],
        "profile": "",
        "level": "L0",
        "see": False,
        "reasons": ["no memories yet"],
    }


def test_search_does_not_fall_below_l1_unless_asked(memory: Memory) -> None:
    """调用 search 就是已经决定要查记忆：控制器选了 L0 时改走 L1，与 MCP、Agents SDK 与 LangGraph
    的 recall 相同（at_least_l1），reasons 写明。以前 SDK 的 search 这时返回空列表，同一句话经
    MCP 却查得到。min_level="l0" 照控制器的决定；mode="l0" 只要画像。"""
    memory.add("I like green tea", user_id="u", wait=True)
    memory.add("I like chess", user_id="u", wait=True)
    gate = memory.search("Hi there", user_id="u", min_level="l0")
    assert gate["level"] == "L0" and gate["results"] == []  # 控制器认为这句不需要回忆
    found = memory.search("Hi there", user_id="u")
    assert found["level"] == "L1" and found["reasons"] == [RAISED_REASON]
    listing = recall_listing(memory.system, "u", "Hi there", at_least_l1=True)
    assert listing["level"] == "L1"
    assert {r["id"] for r in found["results"] if r["type"] == "fact"} == {
        f["id"] for f in listing["facts"]
    }
    assert {r["id"] for r in found["results"] if r["type"] == "excerpt"} == {
        e["id"] for e in listing["excerpts"]
    }
    assert found["results"]
    assert memory.search("Hi there", user_id="u", mode="l0")["results"] == []
    deep = memory.search("What do I like?", user_id="u", mode="l1")
    assert deep["level"] == "L1" and deep["reasons"] == []  # 指定了档位：照办


def test_spaces_are_isolated_by_user_and_agent(memory: Memory) -> None:
    memory.add("I live in Berlin", user_id="alice", wait=True)
    memory.add("I live in Paris", user_id="alice", agent_id="travel-bot", wait=True)
    memory.add("I live in Rome", agent_id="travel-bot", wait=True)
    assert {s["space_id"] for s in memory.spaces()} == {
        "alice",
        "travel-bot/alice",
        "agent/travel-bot",
    }
    assert [f["object"] for f in memory.get_all(user_id="alice")["results"]] == ["Berlin"]
    assert [
        f["object"] for f in memory.get_all(user_id="alice", agent_id="travel-bot")["results"]
    ] == ["Paris"]
    with pytest.raises(ValueError, match="user_id"):
        space_for()
    # 分隔符转义：不同的 user / agent 组合不会落进同一个空间
    combos = [("agent/x", None), (None, "x"), ("a/b", None), ("b", "a"), ("a%2Fb", None)]
    assert len({space_for(user, agent) for user, agent in combos}) == len(combos)
    assert space_for("alice") == "alice" and space_for("alice", "bot") == "bot/alice"


def test_openai_messages_with_images(
    memory: Memory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    local = tmp_path / "shoe.png"
    Image.new("RGB", (16, 16), "blue").save(local)
    out = memory.add(
        [
            {"role": "system", "content": "You are helpful."},  # 不写入
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "These are my running shoes"},
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "image_url", "image_url": {"url": str(local)}},
                ],
            },
            {"role": "tool", "content": "ignored"},
            {"role": "assistant", "content": "They look great!", "name": "Cue"},
        ],
        user_id="alice",
        infer=False,
    )
    assert out["status"] == "logged" and len(out["turn_ids"]) == 2
    turns = memory.system.db.get_turns("alice")
    assert turns[0].text == "These are my running shoes <image> <image>"
    assert len(turns[0].media_ids) == 2 and turns[1].speaker == "Cue"
    stored = memory.system.db.get_media("alice")
    assert len(stored) == 2 and all(Path(m.path).exists() for m in stored)
    assert local.exists()  # 调用方的文件不动；data URL 的临时文件已删除
    assert not list(scratch.iterdir())
    with pytest.raises(ValueError, match="download"):
        memory.add(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": "https://x.test/a.png"}],
            },
            user_id="alice",
        )


def test_stored_text_matches_what_parse_messages_stores() -> None:
    """evidence.stored_text（记忆代理与各框架集成判断哪些轮已经在上下文里）与 SDK 写进记忆的
    文本一致：两份写法一旦走样，已经在请求里的轮又会被想起来。"""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    image = {"type": "image_url", "image_url": {"url": data_url}}
    cases: list[Any] = [
        "Plain words",
        None,
        [{"type": "text", "text": "My shoes"}, image],
        [{"type": "image_url", "image_url": data_url}, {"type": "text", "text": " then words "}],
        [{"type": "text", "text": ""}, {"type": "text"}, image, {"type": "text", "text": "a"}],
    ]
    for content in cases:
        message = {"role": "user", "content": content}
        turns, temporary = parse_messages([message], AUG)
        for path in temporary:
            path.unlink()
        assert stored_text(message) == (turns[0].text if turns else ""), content


def test_run_id_appends_and_infer_false_defers_extraction(memory: Memory) -> None:
    a = memory.add("I live in Berlin", user_id="alice", run_id="chat-1", infer=False)
    b = memory.add("I like jazz", user_id="alice", run_id="chat-1", infer=False)
    assert a["session_id"] == b["session_id"] == "chat-1"
    assert [t.idx for t in memory.system.db.get_turns("alice", "chat-1")] == [0, 1]
    assert memory.get_all(user_id="alice")["results"] == []  # 只记了日志
    # 原始日志立即可检索
    hits = memory.search("jazz", user_id="alice", mode="l1")["results"]
    assert any("jazz" in h["memory"] for h in hits)
    report = memory.consolidate(user_id="alice")
    assert report["sessions"] == 1
    assert {f["memory"] for f in memory.get_all(user_id="alice")["results"]} == {
        "The user lives in Berlin.",
        "I like jazz",
    }


def test_concurrent_adds_to_one_run_keep_every_turn(memory: Memory) -> None:
    """同一个 run_id 并发 add：每句都写进去，序号连续，返回的 turn_id 指向自己写的那句。以前序号
    在锁外分配，撞号的发言被 INSERT OR IGNORE 悄悄丢掉，返回的 ID 却指向别人的话。"""
    memory.add("seed", user_id="u", run_id="s", timestamp=AUG, infer=False)
    barrier = threading.Barrier(16)
    returned: dict[int, list[str]] = {}

    def add(i: int) -> None:
        barrier.wait()
        out = memory.add(f"msg {i}", user_id="u", run_id="s", timestamp=AUG, infer=False)
        returned[i] = out["turn_ids"]

    threads = [threading.Thread(target=add, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    turns = memory.system.db.get_turns("u", "s")
    assert sorted(t.idx for t in turns) == list(range(17))
    text = {t.id: t.text for t in turns}
    assert {i: [text.get(tid) for tid in ids] for i, ids in returned.items()} == {
        i: [f"msg {i}"] for i in range(16)
    }


def test_concurrent_async_adds_keep_every_turn(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    fake_api.chat_replies = [model]

    async def run() -> tuple[list[dict[str, Any]], dict[str, str]]:
        async with AsyncMemory(n3_settings, http_client=fake_api.client(), background=False) as m:
            await m.add("seed", user_id="u", run_id="s", timestamp=AUG, infer=False)
            outs = await asyncio.gather(
                *(
                    m.add(f"msg {i}", user_id="u", run_id="s", timestamp=AUG, infer=False)
                    for i in range(8)
                )
            )
            turns = m.sync.system.db.get_turns("u", "s")
            return outs, {t.id: t.text for t in turns}

    outs, text = asyncio.run(run())
    assert len(text) == 9
    assert [text.get(out["turn_ids"][0]) for out in outs] == [f"msg {i}" for i in range(8)]


def test_add_raises_when_a_turn_was_not_written(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """序号撞上已有的发言（INSERT OR IGNORE 没写进去）：报错，而不是返回没写进去的 turn_id。"""
    memory.add("I live in Berlin", user_id="alice", run_id="chat", infer=False)
    monkeypatch.setattr(Memory, "_next_idx", lambda self, space, session_id: 0)
    with pytest.raises(ConcurrentWriteError, match="chat"):
        memory.add("I like jazz", user_id="alice", run_id="chat", infer=False)
    assert [t.text for t in memory.system.db.get_turns("alice", "chat")] == ["I live in Berlin"]


def test_a_failed_add_leaves_nothing_and_its_retry_writes_once(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """向量接口出错：add 抛出，日志、"轮"、巩固任务与空间都不留；调用方原样重试时只写一遍。
    以前发言先提交了：重试之后同一句话在日志里有两遍，第一遍没有"轮"（检索不到），抽取两次。"""
    system = memory.system
    embed = system.embedder.embed
    outage = iter([True])

    def flaky(texts: Sequence[str], kind: EmbedKind) -> Any:
        if next(outage, False):
            raise EmbeddingAPIError("向量接口返回 HTTP 503", status=503)
        return embed(texts, kind)

    monkeypatch.setattr(system.embedder, "embed", flaky)
    with pytest.raises(EmbeddingAPIError):
        memory.add("I live in Berlin", user_id="alice", run_id="r1")
    assert system.db.list_spaces() == [] and system.db.get_jobs("alice") == []

    out = memory.add("I live in Berlin", user_id="alice", run_id="r1")
    turns = system.db.get_turns("alice")
    assert [(t.idx, t.text) for t in turns] == [(0, "I live in Berlin")]
    assert out["turn_ids"] == [turns[0].id]
    rounds = system.db.list_units("alice", kinds=["round"])
    assert [u.meta["turn_indices"] for u in rounds] == [[0]]
    assert [j["session_id"] for j in system.db.get_jobs("alice")] == ["r1"]


def test_a_write_failing_after_the_vectors_takes_them_back(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """数据库那一步出错时（向量已经写进向量库）：整个事务撤销，这次写的向量也删掉。以前发言已经
    提交，向量库里有点、数据库里却没有"轮"。发言的附加信息（metadata）在同一个事务里，一起撤销。"""
    system = memory.system

    def broken(units: Sequence[Any]) -> int:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(system.db, "add_units", broken)
    with pytest.raises(sqlite3.OperationalError):
        memory.add("I live in Berlin", user_id="alice", run_id="r1", metadata={"src": "crm"})
    assert system.db.get_turns("alice") == [] and system.vectors.count("alice") == 0
    assert system.db.get_turn_meta("alice") == {}


def _unwritable_traces(settings: Settings, tmp_path: Path) -> Settings:
    """trace 目录建不起来（放在一个普通文件下面）：每次写 trace 都抛 NotADirectoryError，
    与磁盘满、目录没有写权限时一样。打开 Memory 时不建目录，照常打开。"""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("x", encoding="utf-8")
    paths = settings.paths.model_copy(update={"traces": blocker / "traces"})
    return settings.model_copy(update={"paths": paths})


def test_a_trace_that_cannot_be_written_does_not_fail_a_committed_write(
    n3_settings: Settings,
    fake_api: FakeAPI,
    model: ScriptedModel,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """trace 在提交之后才写：写不进去时 add、update 与 delete 照常返回，记一条警告（接连失败只记
    一次）。以前它的 OSError 让已经提交的 add 抛出，调用方（记忆代理与 CueMemSession 也会）原样
    重试，同一句话在日志里有两遍；已经做完的遗忘也报成失败。"""
    fake_api.chat_replies = [model]
    settings = _unwritable_traces(n3_settings, tmp_path)
    with (
        Memory(settings, http_client=fake_api.client(), background=False) as m,
        caplog.at_level("WARNING", "cuemem.core.trace"),
    ):
        out = m.add("I live in Berlin", user_id="alice", run_id="r1", timestamp=AUG)
        turns = m.system.db.get_turns("alice", "r1")
        assert [(t.idx, t.text) for t in turns] == [(0, "I live in Berlin")]
        assert out["status"] == "queued" and out["turn_ids"] == [turns[0].id]

        jazz = m.add("I like jazz", user_id="alice", timestamp=AUG, wait=True)["results"][0]
        updated = m.update(jazz["id"], "I like blues")
        assert updated["message"] == "Memory updated"
        assert len(m.system.db.get_turns("alice", updated["session_id"])) == 1
        blues = updated["results"][0]["id"]
        assert m.delete(blues)["message"] == "Memory deleted"
        assert m.get(blues) is None
    warnings = [r for r in caplog.records if r.name == "cuemem.core.trace"]
    assert len(warnings) == 1 and "traces" in warnings[0].getMessage()


def test_a_trace_that_cannot_be_written_does_not_hide_the_error(
    n3_settings: Settings,
    fake_api: FakeAPI,
    model: ScriptedModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写入本身出错、trace 又写不进去：调用方拿到的是原来的异常。以前是 trace 的 OSError（原来的
    异常只在 __context__ 里），worth_retrying 据此把一次不该重试的写入（HTTP 400）当成可以重试。"""
    fake_api.chat_replies = [model]
    with Memory(
        _unwritable_traces(n3_settings, tmp_path), http_client=fake_api.client(), background=False
    ) as m:

        def rejected(texts: Sequence[str], kind: EmbedKind) -> Any:
            raise EmbeddingAPIError("向量接口返回 HTTP 400", status=400)

        monkeypatch.setattr(m.system.embedder, "embed", rejected)
        with pytest.raises(EmbeddingAPIError):
            m.add("I live in Berlin", user_id="alice", run_id="r1")
        assert m.system.db.get_turns("alice") == []


@pytest.mark.parametrize("call", ["add", "update"])
def test_next_turn_number_is_written_with_the_turns(
    memory: Memory, monkeypatch: pytest.MonkeyPatch, call: str
) -> None:
    """add 与 update 记下会话里下一句的序号（next_idx）与发言在同一个事务里：写它出错时发言也
    不留，调用方原样重试只写一遍。以前它在发言提交之后另起一个事务写，出错时（另一个进程长时间
    占着写锁的 "database is locked"、磁盘满）发言已经在日志里，重试又写一遍。"""
    system = memory.system
    jazz = memory.add("I like jazz", user_id="alice", timestamp=AUG, wait=True)["results"][0]
    sessions = set(system.db.get_sessions("alice"))
    jobs = len(system.db.get_jobs("alice"))
    set_meta = type(system.db).set_meta

    def locked(db: Any, key: str, value: str) -> None:
        if key.startswith("next-idx:"):
            raise sqlite3.OperationalError("database is locked")
        set_meta(db, key, value)

    def write() -> dict[str, Any]:
        if call == "add":
            return memory.add("I live in Berlin", user_id="alice", run_id="r1", timestamp=SEP)
        return memory.update(jazz["id"], "I like blues")

    monkeypatch.setattr(type(system.db), "set_meta", locked)
    with pytest.raises(sqlite3.OperationalError):
        write()
    assert set(system.db.get_sessions("alice")) == sessions
    assert len(system.db.get_jobs("alice")) == jobs

    monkeypatch.setattr(type(system.db), "set_meta", set_meta)
    out = write()
    turns = system.db.get_turns("alice", out["session_id"])
    assert [t.idx for t in turns] == [0] and [t.id for t in turns] == out["turn_ids"]
    assert set(system.db.get_sessions("alice")) == sessions | {out["session_id"]}


def whole_space_session_reads(statements: Sequence[str]) -> list[str]:
    """读出一个空间的全部会话（ID 与开始时间，不按会话 ID 查）的语句。"""
    pattern = r"SELECT id, started_at FROM sessions WHERE space_id = '[^']*'$"
    return [s for s in statements if re.search(pattern, s)]


def test_add_and_search_do_not_read_every_session_of_the_space(memory: Memory) -> None:
    """add 只按主键查它写的那个会话（开始时间、登记任务），search 只问空间里有没有会话。以前每次
    add 读两遍整个空间的会话、每次 search 读一遍：不带 run_id 的 add 每次都是一个新会话，2 万个
    会话时每遍约 20 毫秒，都拿着整个进程共用的数据库锁。"""
    system = memory.system
    for day in (1, 2, 3):
        memory.add(f"I like jazz no. {day}", user_id="alice", timestamp=AUG + timedelta(days=day))
    assert len(system.db.get_sessions("alice")) == 3
    statements: list[str] = []
    system.db._conn.set_trace_callback(statements.append)
    try:
        memory.add("I live in Berlin", user_id="alice", timestamp=SEP)
        memory.add("I play chess", user_id="alice", run_id="r1", timestamp=SEP)
        added = whole_space_session_reads(statements)
        assert any("FROM sessions" in s for s in statements)
        statements.clear()
        found = memory.search("Where do I live?", user_id="alice")
        searched = whole_space_session_reads(statements)
    finally:
        system.db._conn.set_trace_callback(None)
    assert found["results"] and any("FROM sessions" in s for s in statements)
    assert added == [] and searched == []


def test_a_failed_write_leaves_no_image_behind(
    memory: Memory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """带图的写入在向量化时出错：这次存进媒体库的原图也删掉。以前留在 data/media 里，却没有任何
    记录指向它，遗忘与 delete_all 都找不到它（例如向量接口不可用时发来的证件照片）。别的记录
    正在用的同一张图不删。"""
    system = memory.system
    passport, shared = tmp_path / "passport.png", tmp_path / "shared.png"
    Image.new("RGB", (16, 16), "red").save(passport)
    Image.new("RGB", (16, 16), "blue").save(shared)
    memory.add([photo("My cat", shared)], user_id="bob", infer=False)
    (kept,) = system.db.get_media("bob")
    embed = system.embedder.embed

    def documents_down(texts: Sequence[str], kind: EmbedKind) -> Any:
        if kind == "document":
            raise EmbeddingAPIError("向量接口返回 HTTP 503", status=503)
        return embed(texts, kind)

    monkeypatch.setattr(system.embedder, "embed", documents_down)
    for image in (passport, shared):
        with pytest.raises(EmbeddingAPIError):
            memory.add([photo("This is my passport", image)], user_id="alice", run_id="r1")
    assert system.db.get_media("alice") == []
    files = [p for p in system.settings.paths.media.rglob("*") if p.is_file()]
    assert files == [Path(kept.path)]

    monkeypatch.setattr(system.embedder, "embed", embed)  # 恢复之后重试：原图重新存进来
    memory.add([photo("This is my passport", passport)], user_id="alice", run_id="r1")
    (stored,) = system.db.get_media("alice")
    assert Path(stored.path).exists()
    memory.delete_all(user_id="alice")
    assert not Path(stored.path).exists() and Path(kept.path).exists()


def photo(text: str, image: Path) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": str(image)}],
    }


def without_balance_for_reconciling(model: ScriptedModel, fake_api: FakeAPI) -> dict[str, bool]:
    """对账时模型供应商余额不足（HTTP 402）；把返回的开关设成 False 就恢复。"""
    broke = {"on": True}

    def reply(body: dict[str, Any]) -> Any:
        system = body["messages"][0]["content"]
        if broke["on"] and system.startswith("You maintain a long-term memory"):
            return NO_BALANCE
        return model(body)

    fake_api.chat_replies = [reply]
    return broke


def test_extraction_errors_after_the_write_are_reported_not_raised(
    memory: Memory, model: ScriptedModel, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """add(wait=True) 写进发言之后对账时余额不足：不抛出，status 是 "queued"、error 写明，下次巩固
    时补上。以前抛出供应商的 402，调用方分不清写没写进去，重试就把同一句话写了两遍。update 记下
    更正之后抽取出错时同样不抛出。"""
    memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    broke = without_balance_for_reconciling(model, fake_api)
    out = memory.add("I moved to Munich", user_id="alice", run_id="r2", timestamp=SEP, wait=True)
    assert out["status"] == "queued" and out["results"] == []
    assert "Error code: 402" in out["error"] and "Do not add them again" in out["error"]
    assert [t.text for t in memory.system.db.get_turns("alice", "r2")] == ["I moved to Munich"]

    broke["on"] = False
    memory.consolidate(user_id="alice")
    (munich,) = memory.get_all(user_id="alice")["results"]
    assert munich["memory"] == "The user lives in Munich."

    def broken(space: str, **kwargs: Any) -> Any:
        raise RuntimeError("provider down")

    monkeypatch.setattr(memory.system, "consolidate", broken)
    update = memory.update(munich["id"], "I live in Hamburg")
    assert update["message"].startswith("Correction logged, but extraction failed")
    assert "provider down" in update["error"] and update["results"] == []
    corrections = [t for t in memory.system.db.get_turns("alice") if t.text.startswith("Corr")]
    assert [t.id for t in corrections] == update["turn_ids"]


def without_balance_for_extracting(model: ScriptedModel, fake_api: FakeAPI) -> dict[str, bool]:
    """抽取时模型供应商余额不足（HTTP 402；密钥不对、没充值时一样）；开关设成 False 就恢复。"""
    broke = {"on": True}

    def reply(body: dict[str, Any]) -> Any:
        system = body["messages"][0]["content"]
        if broke["on"] and system.startswith("You build the long-term memory"):
            return NO_BALANCE
        return model(body)

    fake_api.chat_replies = [reply]
    return broke


def test_add_wait_reports_a_failed_extraction(
    memory: Memory, model: ScriptedModel, fake_api: FakeAPI
) -> None:
    """add(wait=True) 时这个会话的抽取失败（不抛出，记在任务上）：status 是 "queued"、error 写明，
    与巩固抛出时相同。以前 status 是 "consolidated"、没有 error，调用方以为抽取做完了。update 的
    更正抽取失败时同样带上 error。没有后台线程时 status() 里的失败不会自动重试（retry_at 是
    None）。"""
    broke = without_balance_for_extracting(model, fake_api)
    out = memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    assert out["status"] == "queued" and out["results"] == [] and "consolidation" in out
    assert "Error code: 402" in out["error"] and "Do not add them again" in out["error"]
    (failed,) = memory.status(user_id="alice")["failed"]
    assert failed["session_id"] == out["session_id"] and failed["retry_at"] is None
    assert memory.status(user_id="alice")["background"] is False

    broke["on"] = False
    memory.consolidate(user_id="alice")
    (berlin,) = memory.get_all(user_id="alice")["results"]
    broke["on"] = True
    update = memory.update(berlin["id"], "I live in Rome")
    assert update["message"].startswith("Correction logged, but extraction failed")
    assert "Error code: 402" in update["error"]


def test_background_extraction_failure_is_logged(
    n3_settings: Settings,
    fake_api: FakeAPI,
    model: ScriptedModel,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """后台抽取失败（余额不足、密钥不对）时记一条警告日志，写明空间与原因。以前只记在
    worker.errors 里：add() 照常返回、wait() 正常结束，get_all() 一直是空的，什么都看不到。
    同一个空间同一种出错只记一次；这个空间之后有一次巩固顺利做完了，再出错时再记。"""
    broke = without_balance_for_extracting(model, fake_api)
    with (
        Memory(n3_settings, http_client=fake_api.client()) as m,
        caplog.at_level("WARNING", "cuemem.write.consolidate"),
    ):

        def warnings() -> list[str]:
            return [r.getMessage() for r in caplog.records if r.name == "cuemem.write.consolidate"]

        for text in ("I live in Berlin", "I like jazz"):
            assert m.add(text, user_id="alice")["status"] == "queued"
            m.wait()
        assert m.get_all(user_id="alice")["results"] == []
        (first,) = warnings()
        assert "'alice'" in first and "402" in first and "Memory.status()" in first
        state = m.status(user_id="alice")
        assert len(state["failed"]) == 2 and state["background"] is True
        assert all(f["attempts"] == 1 and f["retry_at"] for f in state["failed"])
        assert state["errors"] and all("402" in e for e in state["errors"])

        broke["on"] = False
        m.add("I like tea", user_id="alice")  # 顺利做完：之后同样的出错再记
        m.wait()
        broke["on"] = True
        m.add("I like rain", user_id="alice")
        m.wait()
        assert len(warnings()) == 2


def test_forgetting_a_session_whose_extraction_failed_leaves_no_trace_of_its_words(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """后台抽取失败时 trace 里记一条 consolidate_failed。结构化输出不合格时出错的后几行引着模型
    的输出（pydantic 的 input_value=...），也就是用户的原话；以前这条事件记出错的前 200 个字符，
    遗忘这段会话（delete_all(run_id=...)、按 ID 删除、/api 的遗忘）删不到它：它只有会话 ID 与
    出错，遗忘按属性里的原话、按记忆 ID 找。现在与日志、worker.errors 一样只记第一行。"""
    secret = "My bank PIN is 4711-SECRET"

    def quoting(prompt: str) -> dict[str, Any]:  # 不合格的输出（facts 不是列表），引着原话
        if "PIN" in prompt or "does not match the schema" in prompt:
            return {"summary": "", "facts": {"statement": secret}, "entities": []}
        return where_i_live(prompt)

    fake_api.chat_replies = [ScriptedModel(quoting)]
    with Memory(n3_settings, http_client=fake_api.client(), background=True) as m:
        m.add("I like tea", user_id="bob", run_id="keep")
        m.add(secret, user_id="bob", run_id="r1")
        m.wait()
        assert [f["session_id"] for f in m.status(user_id="bob")["failed"]] == ["r1"]
        events = [
            event
            for path in Path(n3_settings.paths.traces).glob("*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
            for event in json.loads(line)["events"]
            if event["kind"] == "consolidate_failed"
        ]
        assert events and all(e["sessions"] == ["r1"] for e in events)
        assert all("StructuredOutputError" in e["error"] for e in events)
        assert all("\n" not in e["error"] for e in events)
        m.delete_all(user_id="bob", run_id="r1")
        assert m.system.db.get_turns("bob", "r1") == []
        for path in Path(n3_settings.paths.traces).glob("*.jsonl"):
            assert "4711-SECRET" not in path.read_text(encoding="utf-8")


def test_a_delete_whose_rebuild_fails_still_reports_the_deletion(
    memory: Memory, model: ScriptedModel, fake_api: FakeAPI
) -> None:
    """delete 删掉原始日志之后、重建派生记忆时对账余额不足：已经删了，不抛出（/api/v1 是 200），
    warning 写明重建下次巩固时补上。以前抛出供应商的 402，调用方以为没删。"""
    memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    munich = memory.add("I moved to Munich", user_id="alice", timestamp=SEP, wait=True)
    memory.add("I moved to Paris", user_id="alice", timestamp=SEP + timedelta(days=1), wait=True)
    broke = without_balance_for_reconciling(model, fake_api)
    out = memory.delete(munich["results"][0]["id"])
    assert out["message"] == "Memory deleted" and out["turns_deleted"] == 1
    assert "Error code: 402" in out["warning"] and "next consolidation" in out["warning"]
    assert [t.text for t in memory.system.db.get_turns("alice")] == [
        "I live in Berlin",
        "I moved to Paris",
    ]
    broke["on"] = False
    memory.consolidate(user_id="alice")
    assert [f["memory"] for f in memory.get_all(user_id="alice")["results"]] == [
        "The user lives in Paris."
    ]


def _descending_uuids(monkeypatch: pytest.MonkeyPatch) -> None:
    """uuid4 依次变小：以前生成的会话 ID 以随机数结尾，开始时间相同时按 ID 排序，后写的排到前面。"""
    count = itertools.count()
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=(0xFFFFFF - next(count)) << 104))


def test_adds_with_the_same_timestamp_are_reconciled_in_order(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式给了同样的时间（只精确到天或秒的导入、Mem0 式的批量写入）：后写的才是当前的。巩固按
    （开始时间，会话 ID）排序，以前会话 ID 以随机数结尾，对账的先后是随机的，旧住址可能成了
    当前的。"""
    _descending_uuids(monkeypatch)
    first = memory.add("I live in Berlin", user_id="alice", timestamp=AUG, infer=False)
    second = memory.add("I moved to Munich", user_id="alice", timestamp=AUG, infer=False)
    assert first["session_id"] < second["session_id"]
    memory.consolidate(user_id="alice")
    assert [f["object"] for f in memory.get_all(user_id="alice")["results"]] == ["Munich"]


def test_adds_in_the_same_second_are_reconciled_in_order(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不给时间、后台抽取，连着几次 add（同一秒里）：后说的才是当前的。以前缺省时间截到秒，
    开始时间相同，排序落到随机的会话 ID 上。"""
    gate = threading.Event()

    def held(body: dict[str, Any]) -> Any:  # 第一次抽取等到三次 add 都写完：后两段一起巩固
        gate.wait(5)
        return model(body)

    fake_api.chat_replies = [held]
    _descending_uuids(monkeypatch)
    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        memory.add("Hi there", user_id="alice")
        memory.add("I live in Berlin", user_id="alice")
        memory.add("I moved to Munich", user_id="alice")
        gate.set()
        memory.wait()
        current = memory.get_all(user_id="alice")["results"]
        assert [f["object"] for f in current if f["predicate"] == "lives_in"] == ["Munich"]


def test_a_run_spanning_days_dates_facts_by_the_day_they_were_said(
    memory: Memory, model: ScriptedModel
) -> None:
    """一个 run_id 跨了好几天（复用的会话、LangGraph 的线程）：后来某天说的事实按那天记录，抽取
    提示里标出那天的日期。以前一律记成会话开始的那天：8 月 10 日说的"搬到慕尼黑"比 8 月 5 日另一段
    对话里的"住在柏林"还早，存成已被取代，画像里还是柏林。同一天里说的仍按会话开始时间记录。"""
    for text, run_id, at in [
        ("Hi, I'm planning a trip", "main", AUG),
        ("I like jazz", "main", AUG + timedelta(hours=2)),
        ("I live in Berlin", "side", AUG + timedelta(days=4)),
        ("I moved to Munich", "main", AUG + timedelta(days=9)),
    ]:
        memory.add(text, user_id="u", run_id=run_id, timestamp=at, infer=False)
    memory.consolidate(user_id="u")
    [prompt] = [p for p in model.prompts["extract"] if "Munich" in p]
    assert "[t1] User: I like jazz\n[Date: 2026-08-10 (Monday)]\n[t2] User: I moved" in prompt
    everything = memory.get_all(user_id="u", include_superseded=True)["results"]
    facts = {f["object"]: f for f in everything}
    assert facts["Munich"]["status"] == "current"
    assert facts["Munich"]["created_at"] == (AUG + timedelta(days=9)).isoformat()
    assert facts["Berlin"]["valid_to"] == "2026-08-10"
    assert facts["I like jazz"]["created_at"] == AUG.isoformat()  # 同一天：会话开始的时间
    assert "Munich" in memory.profile(user_id="u") and "Berlin" not in memory.profile(user_id="u")


def _utc_z(local: datetime) -> str:
    """本机时间 → 同一时刻的 UTC ISO 字符串，带 'Z'（JS 客户端与 pydantic 的常见写法）。"""
    return f"{local.astimezone(UTC):%Y-%m-%dT%H:%M:%S}Z"


def test_aware_and_naive_timestamps_mix(memory: Memory) -> None:
    """带时区的时间换算成本机时间。以前带时区与不带时区的发言混在一个空间里，巩固报 TypeError、
    任务一直卡在待处理，带时间的检索也报 TypeError。"""
    memory.add("I live in Berlin", user_id="alice", timestamp=_utc_z(AUG), wait=True)
    assert [t.ts for t in memory.system.db.get_turns("alice")] == [AUG]
    assert memory.search("What did I do last week?", user_id="alice", mode="l1")["level"] == "L1"
    memory.add("I moved to Munich", user_id="alice", timestamp=SEP, wait=True)
    assert [j["status"] for j in memory.system.db.get_jobs("alice")] == ["done", "done"]
    assert [f["object"] for f in memory.get_all(user_id="alice")["results"]] == ["Munich"]

    later = (SEP + timedelta(days=3)).astimezone(timezone(timedelta(hours=-12)))
    found = memory.search("Where did I live last month?", user_id="alice", mode="l1", at=later)
    assert any("Berlin" in r["memory"] for r in found["results"])
    assert memory.answer("Where do I live?", user_id="alice", mode="l1", at=later)["answer"]
    # 那一刻系统相信什么：同一时刻换个时区表示，结果相同
    known = (SEP + timedelta(hours=1)).astimezone(timezone(timedelta(hours=-12)))
    believed = memory.get_all(user_id="alice", known_at=known)["results"]
    assert [f["object"] for f in believed] == ["Munich"]


def test_aware_timestamp_does_not_stall_background_extraction(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    fake_api.chat_replies = [model]
    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        memory.add("I live in Berlin", user_id="alice", timestamp=AUG)
        memory.wait()
        memory.add("I moved to Munich", user_id="alice", timestamp=_utc_z(SEP))
        memory.wait()
        assert memory.system.worker is not None and list(memory.system.worker.errors) == []
        assert [j["status"] for j in memory.system.db.get_jobs("alice")] == ["done", "done"]
        assert [f["object"] for f in memory.get_all(user_id="alice")["results"]] == ["Munich"]


def test_run_id_scopes_search_get_all_and_delete_all(memory: Memory) -> None:
    """run_id（filters 里的写法与关键字参数）只取、只删这个会话的记忆。以前 filters 里的 run_id
    被悄悄忽略：照 Mem0 写的代码拿到同一个用户别的会话的记忆。"""
    memory.add("I like jazz", user_id="dan", run_id="r1", wait=True)
    memory.add("I like chess", user_id="dan", run_id="r2", wait=True)
    scoped = {"user_id": "dan", "run_id": "r1"}
    facts = memory.get_all(filters=scoped)["results"]
    assert [(f["memory"], f["session_id"]) for f in facts] == [("I like jazz", "r1")]
    assert memory.get_all(user_id="dan", run_id="r2")["results"][0]["memory"] == "I like chess"
    assert len(memory.get_all(user_id="dan")["results"]) == 2
    for found in (
        memory.search("chess", filters=scoped, mode="l1"),
        memory.search("chess", user_id="dan", run_id="r1", mode="l1"),
    ):
        assert found["results"] and {r["session_id"] for r in found["results"]} == {"r1"}
        assert all("chess" not in r["memory"] for r in found["results"])
    everywhere = memory.search("chess", user_id="dan", mode="l1")["results"]
    assert {r["session_id"] for r in everywhere} == {"r1", "r2"}
    assert memory.search("chess", user_id="dan", run_id="r9", mode="l1")["results"] == []

    report = memory.delete_all(user_id="dan", run_id="r1")
    assert report["sessions_affected"] == ["r1"]
    assert [f["memory"] for f in memory.get_all(user_id="dan")["results"]] == ["I like chess"]
    assert [s["space_id"] for s in memory.spaces()] == ["dan"]  # 空间还在，只删了这个会话


def test_run_scoped_search_keeps_only_that_runs_images(memory: Memory, tmp_path: Path) -> None:
    for run, color in (("r1", "red"), ("r2", "blue")):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (16, 16), color).save(path)
        content = [
            {"type": "text", "text": f"My {color} running shoes"},
            {"type": "image_url", "image_url": {"url": str(path)}},
        ]
        memory.add({"role": "user", "content": content}, user_id="eve", run_id=run, infer=False)
    assert len(memory.search("running shoes", user_id="eve", mode="l1")["images"]) == 2
    scoped = memory.search("running shoes", user_id="eve", run_id="r1", mode="l1")
    assert {r["session_id"] for r in scoped["results"]} == {"r1"}
    (image,) = scoped["images"]
    r1_turns = [t.id for t in memory.system.db.get_turns("eve", "r1")]
    (r1_image,) = memory.system.db.get_media("eve", turn_ids=r1_turns)
    assert image["id"] == r1_image.id


def test_metadata_is_kept_on_memories_and_filterable(memory: Memory, tmp_path: Path) -> None:
    """add(metadata=...)（Mem0 的写法）：记在发言上，由它们抽出的事实、对话片段与图片带上它，
    filters 里按它筛（Mem0 的运算符与 AND / OR / NOT）。以前 add 不认 metadata（TypeError），
    filters 里别的键都报错。"""
    tea = memory.add("I like green tea", user_id="dan", metadata={"topic": "food", "stars": 5})
    assert tea["status"] == "queued"
    memory.add("I like chess", user_id="dan", metadata={"topic": "games", "stars": 3})
    memory.add("I like rain", user_id="dan")
    memory.consolidate(user_id="dan")

    def facts(**where: Any) -> list[str]:
        found = memory.get_all(filters={"user_id": "dan", **where})["results"]
        return sorted(f["memory"] for f in found)

    every = {f["memory"]: f for f in memory.get_all(user_id="dan")["results"]}
    assert every["I like green tea"]["metadata"] == {"topic": "food", "stars": 5}
    assert (
        every["I like rain"]["metadata"] == {} and every["I like rain"]["expiration_date"] is None
    )
    tea_and_chess = ["I like chess", "I like green tea"]
    assert facts(topic="food") == ["I like green tea"]
    assert facts(topic="*") == tea_and_chess
    assert facts(topic={"ne": "food"}) == ["I like chess", "I like rain"]
    assert facts(topic={"in": ["food", "games"]}) == tea_and_chess
    assert facts(topic={"nin": ["food"]}) == ["I like chess", "I like rain"]
    assert facts(topic={"icontains": "FO"}) == ["I like green tea"]
    assert facts(stars={"gte": 4}) == ["I like green tea"]
    assert facts(stars={"gte": 3, "lte": 4}) == ["I like chess"]
    assert facts(stars={"gt": "x"}) == []  # 比不了大小（数字与字符串）：不算满足
    assert facts(OR=[{"topic": "food"}, {"stars": {"lt": 4}}]) == tea_and_chess
    assert facts(AND=[{"topic": "*"}, {"NOT": [{"topic": "food"}]}]) == ["I like chess"]
    assert facts(AND=[{"user_id": "dan"}, {"topic": "games"}]) == ["I like chess"]
    run = every["I like rain"]["session_id"]
    assert facts(run_id={"in": [run, "other"]}) == ["I like rain"]
    for bad in ({"topic": {"regex": "f.*"}}, {"OR": {"topic": "food"}}, {"topic": {"in": "food"}}):
        with pytest.raises(ValueError, match="filter"):
            memory.get_all(filters={"user_id": "dan", **bad})
    with pytest.raises(ValueError, match="single id"):
        memory.get_all(filters={"user_id": {"in": ["dan", "eve"]}})
    for bad_meta in ({"when": datetime.now()}, {1: "x"}):
        with pytest.raises(ValueError, match="metadata"):
            memory.add("I like snow", user_id="dan", metadata=bad_meta)
    assert len(memory.get_all(user_id="dan")["results"]) == 3  # 报错时什么都没写

    # search：事实、对话片段与图片都按 metadata 筛，都带着 metadata
    photo = tmp_path / "cup.png"
    Image.new("RGB", (16, 16), "green").save(photo)
    content = [
        {"type": "text", "text": "My green tea cup"},
        {"type": "image_url", "image_url": {"url": str(photo)}},
    ]
    memory.add({"role": "user", "content": content}, user_id="dan", metadata={"topic": "kitchen"})
    found = memory.search("green tea", filters={"user_id": "dan", "topic": "food"}, mode="l1")
    assert {r["type"] for r in found["results"]} == {"fact", "excerpt"}
    assert {r["metadata"]["topic"] for r in found["results"]} == {"food"}
    assert found["images"] == []
    kitchen = memory.search(
        "green tea cup", filters={"user_id": "dan", "topic": "kitchen"}, mode="l1"
    )
    assert [i["metadata"] for i in kitchen["images"]] == [{"topic": "kitchen"}]
    assert {r["metadata"]["topic"] for r in kitchen["results"]} == {"kitchen"}


RECORD = {"n": 5, "tags": ["Work", "home"], "s": "Hello World"}
FILTER_CASES: list[tuple[dict[str, Any], bool]] = [
    # 比较正好落在边界上：lt / lte、gt / gte 写反了才分得出来
    ({"n": {"lt": 5}}, False),
    ({"n": {"lte": 5}}, True),
    ({"n": {"gt": 5}}, False),
    ({"n": {"gte": 5}}, True),
    ({"n": {"gt": "x"}}, False),  # 不能比较的类型：不算满足
    # NOT 是"一个都不满足"：两个条件里有一个满足就是 False
    ({"NOT": [{"n": 5}, {"n": 6}]}, False),
    ({"NOT": [{"n": 6}, {"n": 7}]}, True),
    ({"OR": [{"n": 6}, {"n": 5}]}, True),
    ({"AND": [{"n": 5}, {"s": "*"}]}, True),
    # 列表：contains 认原样的一项，icontains 不分大小写；字符串里找子串
    ({"tags": {"contains": "Work"}}, True),
    ({"tags": {"contains": "work"}}, False),
    ({"tags": {"icontains": "work"}}, True),
    ({"s": {"icontains": "world"}}, True),
    ({"missing": {"ne": 1}}, True),
    ({"missing": {"nin": [1]}}, True),
    ({"missing": "*"}, False),
]


@pytest.mark.parametrize(("where", "expected"), FILTER_CASES, ids=[str(c) for c, _ in FILTER_CASES])
def test_filter_operators_at_their_edges(where: dict[str, Any], expected: bool) -> None:
    """filters 的运算符（Mem0 的写法，D-069）逐个钉住：以前没有一个值正好在边界上，NOT 只给过
    一个条件，lte 写成 <、lt 写成 <=、gt 写成 >=、NOT 写成"不全满足"都没有测试失败。"""
    assert _matches(where, RECORD) is expected


def test_mem0_reserved_filters(memory: Memory) -> None:
    """filters 里 Mem0 保留的 created_at、updated_at 与 categories 也对记忆自己的字段（照 Mem0
    写的筛选不再悄悄什么都不给）；以前只对 metadata，三个都是 0 条。"""
    memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    (fact,) = memory.get_all(user_id="alice")["results"]
    assert fact["category"] == "attribute"
    for where in (
        {"categories": {"contains": "attribute"}},
        {"created_at": {"gte": "2000-01-01"}},
        {"updated_at": {"gte": "2000-01-01"}},
    ):
        assert [f["id"] for f in memory.get_all(user_id="alice", filters=where)["results"]] == [
            fact["id"]
        ], where
        found = memory.search("Where do I live?", user_id="alice", mode="l1", filters=where)
        assert fact["id"] in {r["id"] for r in found["results"]}, where
    assert memory.get_all(user_id="alice", filters={"created_at": {"lt": "2000-01-01"}}) == {
        "results": []
    }
    early = memory.search(
        "Where do I live?", user_id="alice", mode="l1", filters={"created_at": {"lt": "2000"}}
    )
    assert early["results"] == [] and early["images"] == []
    assert memory.get_all(user_id="alice", filters={"categories": {"contains": "event"}}) == {
        "results": []
    }
    # 对话片段有 created_at，没有 categories
    excerpts = memory.search(
        "Where do I live?", user_id="alice", mode="l1", filters={"created_at": {"gte": "2000"}}
    )
    assert any(r["type"] == "excerpt" for r in excerpts["results"])
    only = memory.search(
        "Where do I live?", user_id="alice", mode="l1", filters={"categories": "*"}
    )
    assert {r["type"] for r in only["results"]} == {"fact"}

    # categories 是列表：eq、ne、in、nin 逐项对（以前 in 一条都不给，nin、ne 留下要排除的那条）
    def ids(where: dict[str, Any]) -> list[str]:
        return [f["id"] for f in memory.get_all(user_id="alice", filters=where)["results"]]

    for where in (
        {"categories": {"in": ["attribute", "event"]}},
        {"categories": {"eq": "attribute"}},
        {"categories": {"eq": ["attribute"]}},
        {"categories": "attribute"},
        {"categories": {"ne": "event"}},
        {"categories": {"nin": ["event"]}},
    ):
        assert ids(where) == [fact["id"]], where
    for where in (
        {"categories": {"in": ["event"]}},
        {"categories": {"nin": ["attribute"]}},
        {"categories": {"ne": "attribute"}},
        {"categories": {"eq": ["attribute", "event"]}},
        {"NOT": [{"categories": {"in": ["attribute"]}}]},
    ):
        assert ids(where) == [], where


def test_metadata_wins_over_the_reserved_filter_names(memory: Memory) -> None:
    """metadata 里有同名的键时以 metadata 为准：以前能筛出来的照样筛出来（category 本来就不是
    保留的名字，照旧只对 metadata）。"""
    memory.add("I like tea", user_id="bob", metadata={"category": "food"}, wait=True)
    (tea,) = memory.get_all(user_id="bob", filters={"category": "food"})["results"]
    assert tea["category"] != "food"  # 事实自己的 category 不同，按 metadata 筛出来
    assert memory.get_all(user_id="bob", filters={"category": tea["category"]}) == {"results": []}
    memory.add("I like chess", user_id="carol", metadata={"created_at": "x"}, wait=True)
    assert len(memory.get_all(user_id="carol", filters={"created_at": "x"})["results"]) == 1
    # 记忆自己的 created_at（2026-…）在这个范围里，metadata 的 "x" 不在：按 metadata 筛
    dated = {"created_at": {"gte": "2000", "lt": "3000"}}
    assert memory.get_all(user_id="carol", filters=dated) == {"results": []}


def test_metadata_and_expiration_stay_out_of_extraction(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel, tmp_path: Path
) -> None:
    """metadata 与过期时间不进抽取提示，也不进会话的内容指纹：抽取与对账的请求、任务键与不带
    它们时逐字相同（缓存照样命中），metadata 里的内容不会发给模型。"""
    fake_api.chat_replies = [model]

    def run(home: str, **extra: Any) -> tuple[list[dict[str, Any]], list[str]]:
        paths = PathsConfig().resolve_against(tmp_path / home)
        fake_api.chat_requests.clear()
        with Memory(
            n3_settings.model_copy(update={"paths": paths}),
            http_client=fake_api.client(),
            background=False,
        ) as m:
            for text, at in (("I live in Berlin", AUG), ("I moved to Munich", SEP)):
                m.add(text, user_id="alice", run_id=text[-6:], timestamp=at, wait=True, **extra)
            jobs = [j["id"] for j in m.system.db.get_jobs("alice")]
            return list(fake_api.chat_requests), jobs

    plain = run("plain")
    tagged = run("tagged", metadata={"crm": "TOP-SECRET-7"}, expiration_date="2099-01-01")
    assert len(plain[0]) == 3  # 两次抽取、一次对账
    assert tagged == plain
    assert "TOP-SECRET-7" not in str(tagged[0])


def test_a_run_that_grows_with_metadata_is_extracted_again(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel, tmp_path: Path
) -> None:
    """同一个 run_id 带着 metadata 再写一次：会话长了，巩固要重新抽取，与不带 metadata 时的请求
    与结果逐字相同。巩固按发言的版本（MemoryDB.turns_version）判断会话变没变，带附加信息的写入
    以前不让它变：第二句没有抽取，"搬到慕尼黑"一直不在记忆里。"""
    fake_api.chat_replies = [model]

    def run(home: str, **extra: Any) -> tuple[list[dict[str, Any]], list[str]]:
        paths = PathsConfig().resolve_against(tmp_path / home)
        fake_api.chat_requests.clear()
        with Memory(
            n3_settings.model_copy(update={"paths": paths}),
            http_client=fake_api.client(),
            background=False,
        ) as m:
            for text, at in (("I live in Berlin", AUG), ("I moved to Munich", SEP)):
                m.add(text, user_id="alice", run_id="r", timestamp=at, wait=True, **extra)
            facts = sorted(f["memory"] for f in m.get_all(user_id="alice")["results"])
            return list(fake_api.chat_requests), facts

    plain = run("plain")
    assert plain[1] == ["The user lives in Munich."]
    assert run("tagged", metadata={"crm": "TOP-SECRET-7"}) == plain


def test_expired_memories_leave_search_get_all_and_the_profile(memory: Memory) -> None:
    """expiration_date（Mem0 的写法）：过期之后 search、get_all 与画像里不再出现，get 与 history
    照样取得到（数据要 delete 才删）。只给日期时到那天结束为止；同一件事后来又说过一次、没给
    过期时间，它就不过期。"""
    today = date.today()
    berlin = memory.add(
        "I live in Berlin", user_id="u", expiration_date=today - timedelta(days=1), wait=True
    )["results"][0]["id"]
    memory.add("I like jazz", user_id="u", expiration_date=today.isoformat(), wait=True)
    soon = datetime.now() - timedelta(seconds=1)
    memory.add("I like chess", user_id="u", expiration_date=soon, wait=True)
    memory.add("I like rain", user_id="u", expiration_date="2020-01-01", wait=True)
    memory.add("I like rain", user_id="u", wait=True)  # 又说了一次，没给过期时间

    current = {f["memory"]: f for f in memory.get_all(user_id="u")["results"]}
    assert sorted(current) == ["I like jazz", "I like rain"]
    assert current["I like jazz"]["expiration_date"] == today.isoformat()
    assert current["I like rain"]["expiration_date"] is None
    everything = memory.get_all(user_id="u", include_superseded=True)["results"]
    assert "The user lives in Berlin." not in {f["memory"] for f in everything}

    found = memory.search("Where do I live? Do I like chess?", user_id="u", mode="l1")
    assert found["results"]
    assert not any("Berlin" in r["memory"] or "chess" in r["memory"] for r in found["results"])
    assert "Berlin" not in found["profile"] and "jazz" in found["profile"]
    assert "Berlin" not in memory.profile(user_id="u")
    assert "Berlin" in memory.system.core_memory("u").text  # type: ignore[union-attr]

    old = memory.get(berlin)
    assert old is not None and old["expiration_date"] == (today - timedelta(days=1)).isoformat()
    assert [v["id"] for v in memory.history(berlin)] == [berlin]
    with pytest.raises(ValueError, match="expiration date"):
        memory.add("I like snow", user_id="u", expiration_date="next week")


def _expired_berlin(memory: Memory) -> None:
    """过期的住址（昨天到期）与没有过期时间的爱好，都已巩固。"""
    yesterday = date.today() - timedelta(days=1)
    memory.add("I live in Berlin", user_id="u", expiration_date=yesterday, wait=True)
    memory.add("I like jazz", user_id="u", wait=True)
    assert "Berlin" in memory.system.core_memory("u").text  # type: ignore[union-attr]


@pytest.mark.parametrize("mode", ["auto", "l0", "l1", "l2"])
def test_answer_does_not_use_expired_memories(
    memory: Memory, model: ScriptedModel, mode: str
) -> None:
    """第五轮体检 P7：以前只有 search、get_all 与 profile 去掉过期的记忆，answer() 直接调用
    MemorySystem.ask，过期的事实、对话片段与画像里的那一行都进了回答模型的提示（L0 也带着
    画像）。"""
    _expired_berlin(memory)
    memory.answer("Where do I live? Do I like jazz?", user_id="u", mode=mode)  # type: ignore[arg-type]
    (prompt,) = model.prompts["read"]
    assert "Berlin" not in prompt and "jazz" in prompt


@pytest.mark.parametrize("mode", ["l1", "l2"])
def test_version_chain_does_not_bring_back_expired_facts(
    memory: Memory, model: ScriptedModel, mode: Literal["l1", "l2"]
) -> None:
    """第八轮体检 P10：hidden 挡住了各路检索，但版本链会把被取代的旧事实补回来（"已经不对了，
    见 [f2]"）。过期的旧住址要在 MemorySystem._without 里再去掉一次，不然它随新住址一起进了
    回答模型的提示与 recall 的清单（auto 选的层不补版本链，查不出来，所以只测 L1、L2）。"""
    memory.add(
        "I live in Berlin",
        user_id="u",
        expiration_date=date.today() - timedelta(days=1),
        wait=True,
    )
    memory.add("I moved to Munich", user_id="u", wait=True)
    memory.answer("Where do I live?", user_id="u", mode=mode)
    (prompt,) = model.prompts["read"]
    assert "Munich" in prompt and "Berlin" not in prompt
    listing = recall_listing(memory.system, "u", "Where do I live?", mode=mode, at_least_l1=True)
    assert "Munich" in json.dumps(listing) and "Berlin" not in json.dumps(listing)


def test_hatch_search_does_not_find_expired_memories(
    memory: Memory, model: ScriptedModel, fake_api: FakeAPI
) -> None:
    """hatch：控制器召回的记忆与模型自己用 memory_search 补查的都不带过期的。"""
    _expired_berlin(memory)
    steps: list[dict[str, Any]] = []

    def reply(body: dict[str, Any]) -> Any:
        if not any(t["function"]["name"] == "memory_search" for t in body.get("tools", [])):
            return model(body)
        steps.append(body)
        if len(steps) == 1:
            args = json.dumps({"query": "Where do I live?"})
            return {"tool_calls": [{"id": "c1", "name": "memory_search", "arguments": args}]}
        answer = json.dumps({"answer": "x", "citations": [], "evidence_sufficient": False})
        return {"tool_calls": [{"id": "c2", "name": "answer", "arguments": answer}]}

    fake_api.chat_replies = [reply]
    result = memory.answer("Where do I live? Do I like jazz?", user_id="u", mode="hatch")
    assert result["answer"] == "x" and len(steps) == 2
    sent = json.dumps(steps[1]["messages"], ensure_ascii=False)
    assert "Berlin" not in sent and "jazz" in sent


def test_recall_for_other_agents_drops_expired_memories(memory: Memory) -> None:
    """记忆代理与各框架集成的证据块（recall_evidence）、MCP 与 /recall 的清单（recall_listing）
    都不给过期的事实、对话片段与画像里的那一行；L0 只给画像时也一样。"""
    from cuemem.evidence import recall_evidence

    _expired_berlin(memory)
    question = "Where do I live? Do I like jazz?"
    for mode in ("auto", "l0", "l1", "l2"):
        evidence = recall_evidence(memory.system, "u", question, mode=mode)  # type: ignore[arg-type]
        assert "Berlin" not in evidence.block and "jazz" in evidence.block, mode
        listing = recall_listing(memory.system, "u", question, mode=mode, at_least_l1=True)  # type: ignore[arg-type]
        assert "Berlin" not in json.dumps(listing) and "jazz" in json.dumps(listing), mode
        assert listing["facts"] and "jazz" in listing["profile"], mode


def test_served_profile_and_chat_drop_expired_memories(
    memory: Memory, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """第五轮复审：MCP 的 profile 工具、GET /api/spaces/{space}/profile（界面的画像栏）与
    cuemem serve 的对话（画像与召回）以前都读原样的核心记忆，同一个 MCP Server 上 recall
    不给的过期住址 profile 照样给。"""
    from fastapi.testclient import TestClient

    from cuemem.agent import Agent
    from cuemem.api.app import create_app
    from cuemem.api.mcp import build_server
    from cuemem.api.service import ChatService

    _expired_berlin(memory)
    agent = Agent(memory.system, router="rule")
    service = ChatService(memory.system, agent)
    assert service.profile("u") == "- I like jazz"
    result = asyncio.run(build_server(service, "u").call_tool("profile", {}))
    assert "".join(getattr(c, "text", "") for c in result.content) == format_profile(
        "- I like jazz"
    )
    app = create_app(service, upload_dir=tmp_path / "up", ui=False, idle_minutes=None)
    with TestClient(app) as http:
        assert http.get("/api/spaces/u/profile").json() == {"text": "- I like jazz"}
    session = agent.start("u")
    assert [text for _, _, text in session.profile] == ["I like jazz"]
    before = len(fake_api.chat_requests)
    agent.chat(session, "Where do I live? Do I like jazz?", mode="l1")
    sent = json.dumps(fake_api.chat_requests[before:], ensure_ascii=False)
    assert "Berlin" not in sent and "jazz" in sent


def test_memory_browser_hides_expired_memories(memory: Memory, tmp_path: Path) -> None:
    """第七轮体检 P17：记忆浏览器的事实表、实体图与图库以前列出过期的记忆，同一个页面的画像
    栏却不给：过期的住址在事实表里、在实体图里算一条边。默认不列出，show_expired 时列出、
    标着 expired；遗忘的候选（界面的下拉框、演示脚本）照样列出：删除之前它还在日志里。"""
    from fastapi.testclient import TestClient

    from cuemem.agent import Agent
    from cuemem.api.app import create_app
    from cuemem.api.replay import _forget
    from cuemem.api.service import ChatService
    from cuemem.api.ui import _forget_choices

    _expired_berlin(memory)
    picture = tmp_path / "flat.png"
    Image.new("RGB", (16, 16), "green").save(picture)
    content = [
        {"type": "text", "text": "My old flat"},
        {"type": "image_url", "image_url": {"url": str(picture)}},
    ]
    memory.add(
        [{"role": "user", "content": content}],
        user_id="u",
        expiration_date=date.today() - timedelta(days=1),
        infer=False,
    )
    service = ChatService(memory.system, Agent(memory.system, router="rule"))
    berlin = "The user lives in Berlin."

    def statements(rows: list[dict[str, Any]]) -> list[str]:
        return [r["statement"] for r in rows]

    assert statements(service.facts("u")) == ["I like jazz"]
    shown = service.facts("u", show_expired=True)
    assert statements(shown) == [berlin, "I like jazz"]
    assert [r["expired"] for r in shown] == [True, False]
    assert [n["facts"] for n in service.graph("u")["nodes"]] == [1]
    assert [n["facts"] for n in service.graph("u", show_expired=True)["nodes"]] == [2]
    assert service.media("u") == []
    (image,) = service.media("u", show_expired=True)
    assert image["expired"] is True
    assert berlin in [
        label.split(" · ", 1)[1] for label, _ in _forget_choices(service, "u", "fact")
    ]
    assert [i for _, i in _forget_choices(service, "u", "image")] == [image["id"]]

    app = create_app(service, upload_dir=tmp_path / "up", ui=False, idle_minutes=None)
    with TestClient(app) as http:
        assert statements(http.get("/api/spaces/u/facts").json()) == ["I like jazz"]
        listed = http.get("/api/spaces/u/facts", params={"show_expired": "true"}).json()
        assert statements(listed) == [berlin, "I like jazz"]
        (user,) = http.get("/api/spaces/u/graph").json()["nodes"]
        assert user["facts"] == 1
        (user,) = http.get("/api/spaces/u/graph?show_expired=true").json()["nodes"]
        assert user["facts"] == 2
        assert http.get("/api/spaces/u/media").json() == []
        assert len(http.get("/api/spaces/u/media?show_expired=true").json()) == 1
    assert _forget(service, "u", "Berlin")["facts"] == [berlin]
    assert berlin not in statements(service.facts("u", show_expired=True))


@pytest.mark.parametrize("mode", ["l1", "l2"])
def test_expired_image_in_a_live_round_is_not_loaded(
    memory: Memory, tmp_path: Path, mode: Literal["l1", "l2"]
) -> None:
    """第八轮体检 P10：一轮里的用户发言带着图片、已经过期，同一轮里助手的回复没有过期——这一轮
    照样检索得到（出处各句都过期了才算过期），装入的轮里的图片却要在 MemorySystem._without
    里去掉：它不经过各路检索的 hidden，以前会作为描述或原图交给回答模型。

    两次 add 各成一轮；后写的回复并进前一轮的做法同 OpenAI Agents 集成（_merge_round）。"""
    picture = tmp_path / "flat.png"
    Image.new("RGB", (16, 16), "green").save(picture)
    content = [
        {"type": "text", "text": "My old flat"},
        {"type": "image_url", "image_url": {"url": str(picture)}},
    ]
    yesterday = date.today() - timedelta(days=1)
    memory.add(
        [{"role": "user", "content": content}],
        user_id="u",
        run_id="flat",
        expiration_date=yesterday,
        infer=False,
    )
    memory.add(
        [{"role": "assistant", "content": "What a bright flat"}],
        user_id="u",
        run_id="flat",
        infer=False,
    )
    system = memory.system
    stale = [u.id for u in system.db.list_units("u", kinds=["round"], session_ids=["flat"])]
    system.writer.replace_rounds("u", "flat", stale)
    (merged,) = system.db.list_units("u", kinds=["round"], session_ids=["flat"])
    (image,) = system.db.get_media("u")
    assert merged.meta["media_ids"] == [image.id] and "bright flat" in merged.text
    gone = system.expired_ids("u", datetime.now())
    assert image.id in gone and merged.id not in gone

    l1 = replace(L1Config(), attach_images=2)  # 附上原图：attach 也要去掉它
    l2 = replace(L2Config(), l1=l1)
    recall = system.retrieve("u", "my old flat", mode=mode, l1=l1, l2=l2, hide_expired=True)
    assert [h.unit.id for h in recall.view.rounds] == [merged.id]
    assert image.id not in recall.view.media and image.id not in recall.view.attach


def doctors(prompt: str) -> dict[str, Any]:
    """抽取替身：'I see Dr. X for my Y' → 宾语是医生的事实，外加医生的实体（带描述）；其余发言
    各记一条多值事实（陈述里提到医生的名字时，实体图与实体列表按名字找到它）。"""
    facts, entities = [], []
    for n, text in re.findall(r"\[t(\d+)\] User: (.*)", prompt):
        seen = re.search(r"I see (Dr\. \w+) .*for my (\w+)", text)
        if seen:
            doctor, issue = seen.groups()
            facts.append(
                {"statement": f"The user sees {doctor} for {issue}.", "subject": "User",
                 "predicate": "sees", "object": doctor, "cardinality": "multi",
                 "importance": 5, "turns": [int(n)]}
            )  # fmt: skip
            entities.append(
                {"name": doctor, "type": "person", "description": f"the user's {issue} doctor"}
            )
        else:
            facts.append(
                {"statement": text, "predicate": "said", "object": text[:40],
                 "cardinality": "multi", "importance": 4, "turns": [int(n)]}
            )  # fmt: skip
    return {"summary": "", "facts": facts, "entities": entities}


def test_entity_list_hides_entities_known_only_from_expired_facts(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """第八轮体检 P5：实体的检索单元没有出处，expired_ids 标不出它们；/api/spaces/{space}/entities
    以前照样列出过期会话里抽出的实体，连同描述（"the user's depression doctor"），事实表、
    实体图与检索都已经不给了。默认不列出，show_expired 时列出、标着 expired。还有一条没过期的
    事实（陈述里按名字提到它）的实体照旧列出；用户本人与没有事实提到的实体不受影响。"""
    from fastapi.testclient import TestClient

    from cuemem.agent import Agent
    from cuemem.api.app import create_app
    from cuemem.api.service import ChatService

    fake_api.chat_replies = [ScriptedModel(doctors)]
    yesterday = date.today() - timedelta(days=1)
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        for text in ("I see Dr. Kim every week for my depression", "I see Dr. Lee for my knee"):
            memory.add(text, user_id="u", expiration_date=yesterday, wait=True)
        memory.add("Dr. Lee fixed my knee last year", user_id="u", wait=True)
        memory.add("I like jazz", user_id="u", wait=True)
        service = ChatService(memory.system, Agent(memory.system, router="rule"))

        def listing(rows: list[dict[str, Any]]) -> dict[str, bool]:
            return {r["name"]: r["expired"] for r in rows}

        shown = listing(service.entities("u", show_expired=True))
        assert shown["Dr. Kim"] is True and shown["Dr. Lee"] is False
        assert not any(shown[name] for name in shown if name != "Dr. Kim")
        assert "Dr. Kim" not in listing(service.entities("u"))
        assert listing(service.entities("u")) == {k: v for k, v in shown.items() if k != "Dr. Kim"}
        app = create_app(service, upload_dir=tmp_path / "up", ui=False, idle_minutes=None)
        with TestClient(app) as http:
            rows = http.get("/api/spaces/u/entities").json()
            assert "Dr. Kim" not in listing(rows) and "Dr. Lee" in listing(rows)
            assert "depression" not in json.dumps(rows)
            rows = http.get("/api/spaces/u/entities", params={"show_expired": "true"}).json()
            assert listing(rows) == shown


def test_entity_list_is_unchanged_without_expiries(memory: Memory) -> None:
    """没有过期的发言时（评测的空间都是）照旧列出全部实体，只多了 expired: false。"""
    from cuemem.agent import Agent
    from cuemem.api.service import ChatService

    memory.add("I live in Berlin", user_id="u", wait=True)
    service = ChatService(memory.system, Agent(memory.system, router="rule"))
    rows = service.entities("u")
    assert [r["id"] for r in rows] == [e.id for e in memory.system.entities("u")]
    assert rows and all(r["expired"] is False for r in rows)


def test_purge_expired_deletes_expired_memories(memory: Memory, fake_api: FakeAPI) -> None:
    """第七轮体检 P20：expiration_date 以前只是读取时不给出，过期的发言、事实、向量与缓存一直
    留着，要逐条 delete。purge_expired 真的删掉过期的发言（遗忘），同一件事后来又说过一次、
    没给过期时间的留着、重放时重新抽出来；还没过期、过期没那么久的不动。"""
    yesterday = date.today() - timedelta(days=1)
    berlin = memory.add("I live in Berlin", user_id="u", expiration_date=yesterday, wait=True)
    berlin_id = berlin["results"][0]["id"]
    memory.add("I like rain", user_id="v", expiration_date="2020-01-01", wait=True)
    memory.add("I like rain", user_id="v", wait=True)  # 又说了一次，没给过期时间
    memory.add("I like jazz", user_id="v", expiration_date="2999-01-01", wait=True)
    system = memory.system
    kept = system.db.get_turns("v")[1:]

    # 过期 30 天以上的：只有 2020 年到期的那一句，昨天到期的住址还留着
    reports = system.purge_expired(older_than=timedelta(days=30))
    assert sorted(reports) == ["v"] and reports["v"].turns_deleted == 1
    assert memory.get(berlin_id) is not None
    reports = system.purge_expired(older_than=timedelta(0))
    assert sorted(reports) == ["u"] and reports["u"].turns_deleted == 1
    assert memory.get(berlin_id) is None
    assert memory.get_all(user_id="u", show_expired=True)["results"] == []
    assert system.db.expiring_turns("u") == {} and not system.db.get_turns("u")
    assert "Berlin" not in (system.core_memory("u").text if system.core_memory("u") else "")
    # 后来又说过一次的留着，那一句与没过期的一句原样不动
    assert system.db.get_turns("v") == kept
    assert sorted(f["memory"] for f in memory.get_all(user_id="v")["results"]) == [
        "I like jazz",
        "I like rain",
    ]
    assert list(system.db.expiring_turns("v").values()) == ["2999-01-01"]
    before = len(fake_api.chat_requests)
    assert system.purge_expired(older_than=timedelta(0)) == {}  # 都删过了：不重放、不发请求
    assert len(fake_api.chat_requests) == before


def test_purge_expired_leaves_spaces_without_expiry_alone(
    memory: Memory, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有带过期时间的发言的空间（评测的空间都是）：不遗忘、不发请求、库不动。"""
    memory.add("I live in Berlin", user_id="w", metadata={"k": "v"}, wait=True)
    system = memory.system
    turns, facts = system.db.get_turns("w"), system.facts("w")
    calls: list[str] = []
    monkeypatch.setattr(system, "forget", lambda space, **kw: calls.append(space))
    before = (len(fake_api.chat_requests), len(fake_api.embed_requests))
    assert system.purge_expired(older_than=timedelta(0)) == {}
    assert system.purge_expired("w", older_than=timedelta(0)) == {}
    assert calls == [] and (len(fake_api.chat_requests), len(fake_api.embed_requests)) == before
    assert system.db.get_turns("w") == turns and system.facts("w") == facts


def test_expired_retention_is_a_daily_opt_in_sweep_job(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUEMEM_EXPIRED_RETENTION_DAYS 没设时（默认）cuemem serve / proxy 的清理线程不删过期的
    记忆、要做的事不变；设了时启动时与之后每天删一遍过期这么多天以上的。"""
    from cuemem.api.app import _sweep_jobs

    system = memory.system
    assert system.settings.expired_retention_days is None
    assert len(_sweep_jobs(system)) == 1  # 只有收拾媒体库（第七轮体检 P8）
    calls: list[timedelta] = []
    monkeypatch.setattr(
        system, "purge_expired", lambda *a, older_than, **kw: calls.append(older_than) or {}
    )
    assert system.expire_memories() == {} and calls == []  # 没设：什么都不做
    system.settings = system.settings.model_copy(update={"expired_retention_days": 3})
    assert _sweep_jobs(system)[1:] == [system.expire_memories]
    today = date(2026, 11, 6)
    system.expire_memories(today=today)
    system.expire_memories(today=today)  # 同一天只删一次
    system.expire_memories(today=today + timedelta(days=1))
    assert calls == [timedelta(days=3)] * 2


def test_a_restated_memory_without_expiry_stays(memory: Memory, model: ScriptedModel) -> None:
    """同一件事后来又说过一次、没给过期时间：它一直有效（同 search 的规则）。"""
    memory.add("I like rain", user_id="u", expiration_date="2020-01-01", wait=True)
    memory.add("I like rain", user_id="u", wait=True)
    memory.answer("Do I like rain?", user_id="u", mode="l1")
    assert "I like rain" in model.prompts["read"][0]


def test_hiding_expired_memories_leaves_other_spaces_untouched(
    memory: Memory, model: ScriptedModel, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有过期记忆的空间（评测的空间从不写过期时间，metadata 与还没到期的也算）：打开
    hide_expired 时 exclude 还是调用方给的同一个对象，画像不动，请求逐字节相同。"""
    memory.add("I live in Berlin", user_id="u", metadata={"k": "v"}, wait=True)
    memory.add("I like jazz", user_id="u", expiration_date="2999-01-01", wait=True)
    system = memory.system
    excludes: list[Any] = []
    search = system.l1.search

    def spy(*args: Any, **kwargs: Any) -> Any:
        excludes.append(kwargs.get("exclude"))
        return search(*args, **kwargs)

    monkeypatch.setattr(system.l1, "search", spy)
    assert system.llm.cache is not None
    monkeypatch.setattr(system.llm.cache, "get", lambda key: None)  # 每遍都发出请求
    question = "Where do I live? Do I like jazz?"

    def run(mode: str, **hide: bool) -> tuple[list[str], str]:
        before = len(fake_api.chat_requests)
        answer = system.ask("u", question, mode=mode, **hide)  # type: ignore[arg-type]
        sent = [json.dumps(b, sort_keys=True) for b in fake_api.chat_requests[before:]]
        return sent, answer.answer

    mine = ("x", "y")
    for mode in ("auto", "l0", "l1", "l2"):
        plain = run(mode)
        assert plain[0] and run(mode, hide_expired=True) == plain, mode
        mine_before = len(excludes)
        system.retrieve("u", question, mode="l1", exclude=mine, hide_expired=True)
        assert excludes[mine_before:] == [mine]
    assert system.expired_ids("u", datetime.now()) == frozenset()


def test_forgetting_deletes_the_metadata_of_forgotten_turns(memory: Memory) -> None:
    """metadata 属于原始层，可能也有个人信息：遗忘一段发言时一起删，delete_all 全删。"""
    pin = memory.add("My PIN is 4321", user_id="u", metadata={"note": "PIN-NOTE-4321"}, wait=True)
    memory.add("I like hiking", user_id="u", metadata={"note": "hobby"}, wait=True)
    memory.delete(pin["results"][0]["id"])
    left = memory.system.db.get_turn_meta("u")
    assert [m.metadata for m in left.values()] == [{"note": "hobby"}]
    db_path = memory.system.settings.paths.db
    assert b"PIN-NOTE-4321" not in db_path.read_bytes()
    memory.delete_all(user_id="u")
    assert memory.system.db.get_turn_meta("u") == {}


def test_delete_restores_the_older_version_and_delete_all_wipes(memory: Memory) -> None:
    memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    munich = memory.add("I moved to Munich", user_id="alice", timestamp=SEP, wait=True)
    report = memory.delete(munich["results"][0]["id"])
    assert report["turns_deleted"] == 1 and report["facts_after"] == 1
    assert [f["memory"] for f in memory.get_all(user_id="alice")["results"]] == [
        "The user lives in Berlin."
    ]
    with pytest.raises(KeyError):
        memory.delete(munich["results"][0]["id"])

    memory.add("I live in Paris", user_id="bob", wait=True)
    memory.delete_all(user_id="alice")
    assert [s["space_id"] for s in memory.spaces()] == ["bob"]
    assert memory.search("Where do I live?", user_id="alice")["results"] == []
    # 删掉后重建的空间：会话从第 0 句开始
    memory.add("I live in Oslo", user_id="alice", run_id="r", infer=False)
    assert [t.idx for t in memory.system.db.get_turns("alice", "r")] == [0]


def current_memories(memory: Memory, user_id: str) -> list[str]:
    return sorted(f["memory"] for f in memory.get_all(user_id=user_id)["results"])


def test_update_is_a_logged_correction(memory: Memory, model: ScriptedModel) -> None:
    """update(memory_id, data)（Mem0 的写法）：不就地改写，而是记一句更正、同步巩固。新事实以
    "说错了"取代原来那条（不问 LLM；以前多值的谓词只会新增），history 里有前后两个版本，原来的
    metadata 与过期时间带过去；重放之后照旧；遗忘这句更正时原来那条恢复。以前没有 update。"""
    berlin = memory.add(
        "I live in Berlin",
        user_id="alice",
        timestamp=AUG,
        metadata={"source": "crm"},
        expiration_date="2099-01-01",
        wait=True,
    )["results"][0]
    jazz = memory.add("I like jazz", user_id="alice", timestamp=AUG, wait=True)["results"][0]
    asked = model.calls["reconcile"]

    out = memory.update(berlin["id"], "I live in Paris", metadata={"checked": True})
    assert out["message"] == "Memory updated"
    (event,) = out["results"]
    assert (event["event"], event["memory"]) == ("UPDATE", "The user lives in Paris.")
    assert event["previous_memory"] == ["The user lives in Berlin."]
    assert event["reason"] == "corrected" and event["space_id"] == "alice"
    assert event["metadata"] == {"source": "crm", "checked": True}
    assert event["expiration_date"] == "2099-01-01"
    assert model.calls["reconcile"] == asked  # 不问 LLM
    turns = memory.system.db.get_turns("alice", out["session_id"])
    assert [t.text for t in turns] == ["Correction: I live in Paris"]
    assert [t.id for t in turns] == out["turn_ids"]
    chain = memory.history(berlin["id"])
    assert [(v["event"], v["memory"]) for v in chain] == [
        ("ADD", "The user lives in Berlin."),
        ("UPDATE", "The user lives in Paris."),
    ]
    assert memory.get(berlin["id"])["superseded_by"] == event["id"]  # type: ignore[index]
    # 说错了的从未成立：按事件时间回放时也没有柏林
    assert "Berlin" not in str(memory.get_all(user_id="alice", valid_at="2026-08-15"))

    blues = memory.update(jazz["id"], "I like blues")["results"][0]
    assert blues["event"] == "UPDATE" and blues["previous_memory"] == ["I like jazz"]
    paris = ["Correction: I like blues", "The user lives in Paris."]
    assert current_memories(memory, "alice") == paris

    with pytest.raises(ValueError, match="already replaced"):
        memory.update(berlin["id"], "I live in Rome")
    with pytest.raises(ValueError, match="empty"):
        memory.update(event["id"], "  ")
    with pytest.raises(KeyError):
        memory.update("no-such-id", "I live in Rome")

    memory.consolidate(user_id="alice", rebuild=True)  # 从原始日志重放：更正照样取代原来那条
    assert current_memories(memory, "alice") == paris
    memory.delete(blues["id"])  # 遗忘这句更正：原来那条恢复
    assert current_memories(memory, "alice") == ["I like jazz", "The user lives in Paris."]


def test_update_that_extracts_nothing_leaves_the_memory(
    memory: Memory, model: ScriptedModel
) -> None:
    berlin = memory.add("I live in Berlin", user_id="alice", wait=True)["results"][0]
    model.extractions = lambda prompt: (
        {"summary": "", "facts": [], "entities": []}
        if "Correction" in prompt
        else where_i_live(prompt)
    )
    out = memory.update(berlin["id"], "I live in Hamburg")
    assert out["message"].startswith("No memory could be extracted") and out["results"] == []
    assert current_memories(memory, "alice") == ["The user lives in Berlin."]
    # 这句更正也不留在日志里：以前它作为对话片段与原来那条一起被检索出来、随导出带走
    assert out["turn_ids"] == [] and "error" not in out
    assert [t.text for t in memory.system.db.get_turns("alice")] == ["I live in Berlin"]
    found = memory.search("Hamburg", user_id="alice", mode="l1")["results"]
    assert not any("Hamburg" in hit["memory"] for hit in found)
    exported = memory.export(user_id="alice")["sessions"]
    assert [t["text"] for s in exported for t in s["turns"]] == ["I live in Berlin"]


def test_reset_deletes_every_space(memory: Memory) -> None:
    """reset()（Mem0 的写法）：删除这个数据目录里的全部记忆，还开着的 run 一并结束。"""
    memory.add("I live in Berlin", user_id="alice", metadata={"k": 1}, wait=True)
    memory.add("I live in Rome", agent_id="bot", wait=True)
    run = memory.open_run(user_id="carol")
    memory.add("Hi there", user_id="carol", run_id=run)
    out = memory.reset()
    assert out["message"] == "All memories deleted"
    assert sorted(out["spaces_deleted"]) == ["agent/bot", "alice", "carol"]
    assert out["turns_deleted"] == 3
    assert memory.spaces() == [] and memory.system.db.get_turn_meta("alice") == {}
    assert memory.system.open_sessions("carol") == set()
    assert memory.system.adopt_open("carol") == 0
    memory.add("I live in Oslo", user_id="alice", wait=True)  # 之后照常可用
    assert current_memories(memory, "alice") == ["The user lives in Oslo."]


def naming(memory: Memory, user: str) -> tuple[list[str], bool, int]:
    """还写着这个用户 ID 的：元信息的键、数据库文件里的字节、trace 的条数（attrs.space）。"""
    db, paths = memory.system.db, memory.system.settings.paths
    db.scrub()  # WAL 写回数据库文件
    keys = [k for (k,) in db._conn.execute("SELECT key FROM meta") if user in k]
    records = [
        json.loads(line)
        for path in paths.traces.glob("*.jsonl")
        for line in path.read_text("utf-8").splitlines()
    ]
    traces = [r for r in records if r["attrs"].get("space") == user]
    return keys, user.encode() in paths.db.read_bytes(), len(traces)


def test_deleting_a_user_with_more_turns_than_sqlite_takes_parameters(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第五轮复审：遗忘整个用户时 get_media(turn_ids=…) 与 citing_turns 一条 SQL 带上全部的轮、
    事实与图片，超过 SQLite 的参数上限（3.32 起 32766：几万句的用户）就 "too many SQL
    variables"、什么都没删。这里把上限降到 600（别处的分批是 500 个），批次降到 100。"""
    import cuemem.store.sqlite as sqlite_module

    memory.add("I live in Berlin", user_id="u", wait=True)
    talk = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(700)]
    memory.add(talk, user_id="u", infer=False)
    db = memory.system.db
    monkeypatch.setattr(sqlite_module, "IN_CHUNK", 100)
    db._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 600)
    many = [f"x{i}" for i in range(700)]
    assert db.get_media("u", turn_ids=many) == [] and db.citing_turns("u", many) == []
    assert db.get_facts("u", session_ids=many) == []
    memory.delete_all(user_id="u")
    assert memory.get_all(user_id="u") == {"results": []} and db.get_turns("u") == []


@pytest.mark.parametrize("how", ["delete_all", "reset"])
def test_delete_all_and_reset_leave_nothing_naming_the_user(memory: Memory, how: str) -> None:
    """delete_all 与 reset 之后，库里与 trace 里不再有写着这个用户 ID（常常是邮箱）的东西。以前
    留着：每次 add 一个写着用户 ID 与时间的 next_idx 键、巩固配置的指纹、崩溃时留下的进行中会话的
    标记，以及控制器选了 L0 的那次检索的 trace（没有列出记忆 ID，遗忘按 ID 找不到它）——里面有
    用户问过的话。别的用户的不动。"""
    user, system = "alice@example.com", memory.system
    for i in range(3):
        memory.add(f"note {i}", user_id=user, infer=False)
    memory.add("I live in Berlin", user_id=user, timestamp=AUG, wait=True)
    with memory.run(user_id=user) as rid:
        memory.add("I like jazz", user_id=user, run_id=rid)
    memory.search("Where do I live?", user_id=user)
    system.db.set_meta(_open_mark(user, "chat-1"), AUG.isoformat())  # 之前的进程崩溃时留下的
    memory.add("I like tea", user_id=user, infer=False, idempotency_key="k1")  # add_keys 的一行
    memory.add("I live in Paris", user_id="bob", run_id="r", infer=False)
    keys, in_db, traces = naming(memory, user)
    assert len(keys) >= 7 and in_db and traces >= 7

    if how == "delete_all":
        memory.delete_all(user_id=user)
    else:
        memory.reset()
    assert naming(memory, user) == ([], False, 0)
    assert system.db._conn.execute("SELECT COUNT(*) FROM add_keys").fetchone() == (0,)
    if how == "delete_all":
        assert naming(memory, "bob")[2] == 1  # bob 的写入的 trace 还在
        memory.add("I moved to Rome", user_id="bob", run_id="r", infer=False)
        assert [t.idx for t in system.db.get_turns("bob", "r")] == [0, 1]


def test_a_reset_that_fails_partway_still_purges_the_traces_of_what_it_deleted(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reset 删到一半出错（向量库在第二个空间连不上）：已经删掉的空间的 trace 也删掉。以前 trace
    放到全部空间删完之后一起删，中途出错就一条都没删；删掉了的空间不在空间列表里，重试的 reset
    看不到它们，用户 ID 与问过的话一直留在 trace 里。"""
    users = ["alice@example.com", "bob@example.com", "carol@example.com"]
    for user in users:
        memory.add("I live in Berlin", user_id=user, wait=True)
        memory.search("hello there, what is the weather?", user_id=user, min_level="l0")
    assert all(naming(memory, user)[2] for user in users)
    real = memory.system.vectors.delete_space
    calls: list[str] = []

    def flaky(space_id: str) -> None:
        calls.append(space_id)
        if len(calls) == 2:
            raise ConnectionError("qdrant unreachable")
        real(space_id)

    monkeypatch.setattr(memory.system.vectors, "delete_space", flaky)
    with pytest.raises(ConnectionError):
        memory.reset()
    monkeypatch.undo()
    first, failed, untouched = calls[0], calls[1], next(u for u in users if u not in calls)
    left = [s["space_id"] for s in memory.spaces()]
    assert first not in left and failed in left and untouched in left
    assert naming(memory, first)[2] == 0
    assert naming(memory, untouched)[2] > 0  # 还没开始删的空间不动
    memory.reset()  # 重试删掉剩下的
    assert memory.spaces() == [] and all(naming(memory, u)[2] == 0 for u in users)


# ---------- add 的 idempotency_key（第六轮体检） ----------


def _turn_count(memory: Memory, space: str = "alice") -> int:
    return len(memory.system.db.get_turns(space))


def test_a_retry_with_the_same_key_writes_once(memory: Memory) -> None:
    """响应丢了（MemoryClient 超时，sent=True）之后用同一个键重试：只写一次，返回 duplicate 与
    那次的 session_id、turn_ids。以前没有办法：add 默认 queued，get_all 看不到刚写的，盲目重试
    原始日志里就有两遍。不给键时与以前一样，每次都写。"""
    first = memory.add("I live in Berlin", user_id="alice", idempotency_key="k1")
    assert first["status"] == "queued" and "replayed" not in first
    again = memory.add("I live in Berlin", user_id="alice", idempotency_key="k1", wait=True)
    assert again == {
        "results": [],
        "session_id": first["session_id"],
        "turn_ids": first["turn_ids"],
        "status": "duplicate",
        "replayed": True,
    }
    assert _turn_count(memory) == 1
    # wait=True 的重试等抽取做完（第七轮体检）：以前立即返回，紧接着的 get_all 是空的
    assert memory.system.db.get_jobs("alice", status="pending") == []
    (berlin,) = memory.get_all(user_id="alice")["results"]
    assert berlin["memory"] == "The user lives in Berlin."
    memory.add("I live in Berlin", user_id="alice")
    assert _turn_count(memory) == 2


def test_a_waiting_retry_reports_a_failed_extraction(
    memory: Memory, model: ScriptedModel, fake_api: FakeAPI
) -> None:
    """wait=True 的重试遇上这次会话的抽取失败：仍是 duplicate、results 是空的，加上 error。"""
    first = memory.add("I live in Berlin", user_id="alice", idempotency_key="k1")
    without_balance_for_extracting(model, fake_api)
    again = memory.add("I live in Berlin", user_id="alice", idempotency_key="k1", wait=True)
    assert again["status"] == "duplicate" and again["replayed"] is True
    assert again["results"] == [] and again["session_id"] == first["session_id"]
    assert "Error code: 402" in again["error"] and "Do not add them again" in again["error"]


def test_a_waiting_retry_into_an_open_run_returns_the_duplicate(memory: Memory) -> None:
    """那次写进的是还开着的 run：wait=True 的重试不报 ValueError、不抽取，照旧返回 duplicate。"""
    memory.open_run("r", user_id="alice")
    first = memory.add("I live in Berlin", user_id="alice", run_id="r", idempotency_key="k1")
    again = memory.add(
        "I live in Berlin", user_id="alice", run_id="r", idempotency_key="k1", wait=True
    )
    assert again == {
        "results": [],
        "session_id": "r",
        "turn_ids": first["turn_ids"],
        "status": "duplicate",
        "replayed": True,
    }
    assert memory.system.db.get_jobs("alice") == []  # 结束时才抽取


def test_a_keyed_add_that_collides_writes_nothing_and_its_retry_writes(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """带键的 add 撞号（另一个进程先写了这些序号）：ConcurrentWriteError，键与发言都不留；
    原样重试写进这次的发言。以前提交之后才查：键已经记下，重试返回 duplicate 与别人的发言 ID，
    这次的发言丢了。"""
    db = memory.system.db
    memory.add("I like tea", user_id="alice", run_id="r", infer=False)
    real = Memory._next_idx
    monkeypatch.setattr(Memory, "_next_idx", lambda self, space, session_id: 0)
    with pytest.raises(ConcurrentWriteError, match="Nothing was written"):
        memory.add("I like coffee", user_id="alice", run_id="r", infer=False, idempotency_key="k")
    assert [t.text for t in db.get_turns("alice", "r")] == ["I like tea"]
    assert db.add_key_lookup("alice", _key_hash("k")) is None
    monkeypatch.setattr(Memory, "_next_idx", real)
    out = memory.add("I like coffee", user_id="alice", run_id="r", infer=False, idempotency_key="k")
    assert out["status"] == "logged"
    text = {t.id: t.text for t in db.get_turns("alice", "r")}
    assert [text[tid] for tid in out["turn_ids"]] == ["I like coffee"]


def test_the_same_key_for_another_request_is_refused(memory: Memory) -> None:
    """同一个键用于不同的请求（消息不同，或别的参数不同）：IdempotencyConflictError，什么都不写。
    wait 不算请求的一部分；本地的 datetime 与 ISO 文本算同一个时间。"""
    memory.add("I live in Berlin", user_id="alice", timestamp=AUG, idempotency_key="k1")
    for change in (
        {"messages": "I live in Rome"},
        {"metadata": {"k": 1}},
        {"infer": False},
        {"timestamp": SEP},
    ):
        args: dict[str, Any] = {"messages": "I live in Berlin", "timestamp": AUG, **change}
        with pytest.raises(IdempotencyConflictError, match="different add request"):
            memory.add(user_id="alice", idempotency_key="k1", **args)
    same = memory.add(
        "I live in Berlin", user_id="alice", timestamp=AUG.isoformat(), idempotency_key="k1"
    )
    assert same["status"] == "duplicate"
    assert _turn_count(memory) == 1


def test_keys_are_per_user_and_checked(memory: Memory) -> None:
    """键按用户（记忆空间）分开：bob 用 alice 用过的键照常写。键要是 1 到 255 个可打印的 ASCII
    字符（它也要能放进 HTTP 头），否则 ValueError，什么都不写。"""
    memory.add("I live in Berlin", user_id="alice", idempotency_key="k1")
    assert memory.add("I live in Berlin", user_id="bob", idempotency_key="k1")["status"] == "queued"
    for bad in ("", "x" * 256, "naïve", "a\nb"):
        with pytest.raises(ValueError, match="idempotency_key"):
            memory.add("hi", user_id="carol", idempotency_key=bad)
    assert memory.add("hi", user_id="carol", idempotency_key="x" * 255)["status"] == "queued"
    assert _turn_count(memory, "carol") == 1
    (row,) = memory.system.db._conn.execute(
        "SELECT key_hash FROM add_keys WHERE space_id = 'alice'"
    ).fetchall()
    assert "k1" not in row[0]  # 只存键的 sha256


def test_no_key_runs_no_extra_query(memory: Memory, monkeypatch: pytest.MonkeyPatch) -> None:
    """不给键时与以前完全相同：不查、不写 add_keys。"""
    db = memory.system.db

    def unused(*args: Any) -> Any:
        raise AssertionError("add_keys touched without an idempotency_key")

    monkeypatch.setattr(db, "add_key_lookup", unused)
    monkeypatch.setattr(db, "add_key_insert", unused)
    assert memory.add("I live in Berlin", user_id="alice", wait=True)["status"] == "consolidated"


def test_racing_retries_write_once_and_leave_no_vectors_behind(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """8 个线程同时用同一个键 add：只有一个写进去，别的在提交时撞上主键、整个事务回滚（连同
    发言与暂存的向量），按 duplicate 返回同一个 session_id。向量与只写一次的空间一样多。每个
    线程的第一次查询都当作没查到：8 个都走到写入，确定地走一遍回滚的路径。"""
    barrier = threading.Barrier(8)
    out: list[dict[str, Any]] = []
    looked = threading.local()
    real = Memory._replayed

    def first_misses(self: Memory, *args: str) -> dict[str, Any] | None:
        if not getattr(looked, "once", False):
            looked.once = True
            return None
        return real(self, *args)

    monkeypatch.setattr(Memory, "_replayed", first_misses)

    def go() -> None:
        barrier.wait()
        out.append(memory.add("I like tea", user_id="alice", idempotency_key="race"))

    threads = [threading.Thread(target=go) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(r["status"] for r in out) == ["duplicate"] * 7 + ["queued"]
    assert len({r["session_id"] for r in out}) == 1
    assert _turn_count(memory) == 1
    memory.add("I like tea", user_id="bob")
    vectors = memory.system.vectors
    assert vectors.count("alice") == vectors.count("bob") > 0


def test_a_forgotten_write_is_not_written_again_by_a_late_retry(memory: Memory) -> None:
    """遗忘之后键只留下墓碑：请求的摘要、会话与发言 ID 都置空（由被遗忘的内容算出，不能留着），
    晚到的重试返回 forgotten，不再写一遍；用这个键的不同请求也是 forgotten。没有被遗忘的发言的
    键照旧是 duplicate。"""
    first = memory.add("I live in Berlin", user_id="alice", idempotency_key="k1", wait=True)
    memory.add("I like jazz", user_id="alice", run_id="r", idempotency_key="k2", infer=False)
    memory.delete(first["results"][0]["id"])
    assert len(memory.system.db.get_turns("alice")) == 1
    rows = dict(
        memory.system.db._conn.execute(
            "SELECT request_hash IS NULL AND session_id IS NULL AND turn_ids IS NULL, COUNT(*)"
            " FROM add_keys GROUP BY 1"
        ).fetchall()
    )
    assert rows == {1: 1, 0: 1}
    forgotten = {
        "results": [],
        "session_id": None,
        "turn_ids": [],
        "status": "forgotten",
        "replayed": True,
    }
    assert memory.add("I live in Berlin", user_id="alice", idempotency_key="k1") == forgotten
    assert memory.add("Something else", user_id="alice", idempotency_key="k1") == forgotten
    kept = memory.add("I like jazz", user_id="alice", run_id="r", idempotency_key="k2", infer=False)
    assert kept["status"] == "duplicate"
    assert len(memory.system.db.get_turns("alice")) == 1


def test_keys_expire_after_a_day(memory: Memory) -> None:
    """键记 24 小时（与 Stripe 相同）：过期的不再算，下一次带键的写入顺带删掉过期的行。"""
    db = memory.system.db
    memory.add("I like tea", user_id="alice", idempotency_key="old")
    stale = (datetime.now() - timedelta(hours=25)).isoformat()
    with db.transaction() as conn:
        conn.execute("UPDATE add_keys SET created_at = ?", (stale,))
    assert memory.add("I like tea", user_id="alice", idempotency_key="old")["status"] == "queued"
    assert _turn_count(memory) == 2
    assert db._conn.execute("SELECT COUNT(*) FROM add_keys").fetchone() == (1,)


def test_async_memory_passes_the_key(memory: Memory) -> None:
    async def twice() -> list[str]:
        mem = AsyncMemory.from_system(memory.system)
        return [
            (await mem.add("I like tea", user_id="alice", idempotency_key="a1"))["status"]
            for _ in range(2)
        ]

    assert asyncio.run(twice()) == ["queued", "duplicate"]
    assert _turn_count(memory) == 1


def test_delete_all_keeps_the_mark_of_a_run_still_open(memory: Memory) -> None:
    """delete_all 删掉崩溃时留下的进行中会话的标记，本进程里还开着的 run 的标记留着：进程随后
    崩溃时，它之后写的发言还有人接手（adopt_open）。"""
    system = memory.system
    run = memory.open_run(user_id="alice")
    memory.add("I live in Berlin", user_id="alice", run_id=run)
    system.db.set_meta(_open_mark("alice", "chat-1"), AUG.isoformat())
    memory.delete_all(user_id="alice")
    assert system.db.meta_keys(_open_mark("alice")) == [_open_mark("alice", run)]
    memory.add("I live in Rome", user_id="alice", run_id=run)  # 序号从 0 重新开始
    assert [t.idx for t in system.db.get_turns("alice", run)] == [0]
    assert memory.end_run(run, user_id="alice")["turns"] == 1
    assert system.db.meta_keys(_open_mark("alice")) == []


def test_a_forget_keeps_the_next_turn_number_until_the_session_is_gone(memory: Memory) -> None:
    """遗忘会话末尾的发言之后，序号照旧不复用（D-059：下一句是 2，抽取时看得出中间删过）；会话的
    发言全删了，它的 next_idx 也删掉（以前一直留着，键里是用户 ID 与会话 ID）。"""
    system = memory.system
    both = [{"role": "user", "content": "first thing"}, {"role": "user", "content": "second one"}]
    memory.add(both, user_id="erin", run_id="r", infer=False)
    system.forget("erin", turn_ids=[system.db.get_turns("erin", "r")[1].id])
    memory.add("third thing", user_id="erin", run_id="r", infer=False)
    assert [t.idx for t in system.db.get_turns("erin", "r")] == [0, 2]
    assert [k for k in system.db.meta_keys("next") if "erin" in k]
    system.forget("erin", session_ids=["r"])
    assert [k for k in system.db.meta_keys("next") if "erin" in k] == []


def test_users_whose_ids_share_a_prefix_keep_their_turn_numbers(memory: Memory) -> None:
    """空间 'acme' 与 'acme:alice'（作用域 acme:* 的用户 ID 带冒号）：删掉 acme 的全部记忆不碰
    acme:alice 的会话的序号；acme 用 run_id 'alice:x' 写入也不让 acme:alice 的会话 'x' 跳号。
    以前两者的 next_idx 是同一行（next_idx:acme:alice:x）。"""
    system = memory.system
    two = [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]
    memory.add(two, user_id="acme:alice", run_id="x", infer=False)
    system.forget("acme:alice", turn_ids=[system.db.get_turns("acme:alice", "x")[1].id])
    six = [{"role": "user", "content": str(i)} for i in range(6)]
    memory.add(six, user_id="acme", run_id="alice:x", infer=False)
    memory.delete_all(user_id="acme")
    memory.add("three", user_id="acme:alice", run_id="x", infer=False)
    assert [t.idx for t in system.db.get_turns("acme:alice", "x")] == [0, 2]


def colliding_agents() -> tuple[str, str]:
    """两个 Agent：alice 在它们下面的两个空间，旧的集合名（前 40 个字符加 32 位摘要）相同。"""
    seen: dict[str, str] = {}
    for i in itertools.count():
        agent = f"tenant0customer0support0assistant0prod0x{i}"
        name = VectorStore.legacy_name(space_for("alice", agent))
        if name in seen:
            return seen[name], agent
        seen[name] = agent
    raise AssertionError("unreachable")


def test_spaces_never_share_a_vector_collection(memory: Memory) -> None:
    """集合名以前只带 32 位摘要：能自己起 agent_id 的令牌离线撞出与别人同名的集合，delete_all
    删掉别人的向量（检索从此什么都找不到），写入挤进别人的检索结果。现在新集合带 128 位摘要；
    升级之前的旧名字集合只给能证明是自己的空间（数据库里它的记忆单元在里面）用。"""
    victim, attacker = colliding_agents()
    mine, theirs = space_for("alice", victim), space_for("alice", attacker)
    assert mine != theirs and VectorStore.legacy_name(mine) == VectorStore.legacy_name(theirs)
    vectors = memory.system.vectors
    vectors._names[mine] = VectorStore.legacy_name(mine)  # 升级之前建的：旧名字的集合
    memory.add("I live in Berlin and love sailing", user_id="alice", agent_id=victim, infer=False)
    vectors._names.clear()  # 升级之后的新进程

    memory.system.db.ensure_space(theirs)  # 数据库里先有这个空间（例如 Claude 记忆工具）也一样
    memory.add("My password hint is purple", user_id="alice", agent_id=attacker, infer=False)
    assert vectors.collection_name(mine) == VectorStore.legacy_name(mine)
    assert vectors.collection_name(theirs) != vectors.collection_name(mine)
    assert vectors.count(mine) == vectors.count(theirs) == 1
    memory.delete_all(user_id="alice", agent_id=attacker)
    assert vectors.count(mine) == 1
    found = memory.search("sailing", user_id="alice", agent_id=victim, mode="l1")["results"]
    assert any("sailing" in hit["memory"] for hit in found)
    memory.delete_all(user_id="alice", agent_id=victim)  # 自己的旧集合照样删得掉
    names = [c.name for c in vectors._client.get_collections().collections]
    assert VectorStore.legacy_name(mine) not in names


def test_delete_leaves_no_vectors_or_traces_of_the_fact(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """删一条事实之后，它的陈述不在向量缓存里（找转述时算的查询向量、兜底时重新算的文档向量），
    看过它的回答的 trace 与那次提问的查询向量也都删了（回答复述了它，trace 里没有用户的原话）。"""

    def reworded(prompt: str) -> dict[str, Any]:  # 陈述与原话不同：分得清是谁的向量
        out = where_i_live(prompt)
        for fact in out["facts"]:
            fact["statement"] = f"The user said: {fact['statement']}"
        return out

    answer = "Ana's birthday is on October 3."
    reader = {"answer": answer, "citations": ["f1"], "evidence_sufficient": True}
    fake_api.chat_replies = [ScriptedModel(reworded, reader=lambda prompt: reader)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        memory.add("My sister Ana's birthday is October 3", user_id="u", timestamp=AUG, wait=True)
        memory.add("I like hiking", user_id="u", timestamp=SEP, wait=True)
        question = "When is Ana's birthday?"
        assert memory.answer(question, user_id="u", mode="l1")["answer"] == answer
        traces = memory.system.settings.paths.traces
        assert any("October 3" in p.read_text("utf-8") for p in traces.glob("*.jsonl"))
        [fact] = [f for f in memory.system.facts("u") if "Ana" in f.statement]

        memory.delete(fact.id)
        statements = [f.statement for f in memory.system.facts("u")]
        assert statements == ["The user said: I like hiking"]
        embedder = memory.system.embedder
        assert isinstance(embedder, CachedEmbedder)
        assert not embedder.cached([fact.statement], "document")
        assert not embedder.cached([fact.statement, question], "query")
        assert not any("October 3" in p.read_text("utf-8") for p in traces.glob("*.jsonl"))


def test_delete_purges_the_query_vectors_of_searches_that_found_it(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """search 只检索、不回答，以前不记 trace：检索到被删事实的提问（提问里常常就带着被删的
    信息）的查询向量一直留在缓存里。现在不在 trace 里的检索记一条 retrieve trace，遗忘时按
    检索记录找到这些提问，连同查询向量一起删掉。"""
    fake_api.chat_replies = [ScriptedModel(where_i_live)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        memory.add("My PIN is 4321", user_id="u", timestamp=AUG, wait=True)
        memory.add("I like hiking", user_id="u", timestamp=SEP, wait=True)
        questions = ["Is my PIN 4321?", "What is my PIN?"]
        for question in questions:
            assert memory.search(question, user_id="u", mode="l1")["results"]
        embedder = memory.system.embedder
        assert isinstance(embedder, CachedEmbedder) and embedder.cached(questions, "query")
        [fact] = [f for f in memory.system.facts("u") if "PIN" in f.statement]

        memory.delete(fact.id)
        assert [q for q in questions if embedder.cached([q], "query")] == []
        traces = memory.system.settings.paths.traces.glob("*.jsonl")
        assert not any("PIN" in p.read_text("utf-8") for p in traces)


@pytest.mark.parametrize("sep", ["\u2028", "\u2029", "\x85"])
def test_delete_purges_searches_whose_question_has_a_line_separator(
    n3_settings: Settings, fake_api: FakeAPI, sep: str
) -> None:
    """提问里有 U+2028、U+2029 或 U+0085（代理与 Agent 拿用户的原话做提问）：trace 照样删掉、
    提问的查询向量照样清掉。以前删 trace 时按 splitlines 分行，这样的 trace 断成几行解析不了的，
    原样留着，提问的查询向量也找不到。"""
    fake_api.chat_replies = [ScriptedModel(where_i_live)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        memory.add("My PIN is 4321", user_id="u", timestamp=AUG, wait=True)
        memory.add("I like hiking", user_id="u", timestamp=SEP, wait=True)
        question = f"What is my PIN?{sep}Is it 4321?"
        assert memory.search(question, user_id="u", mode="l1")["results"]
        embedder = memory.system.embedder
        assert isinstance(embedder, CachedEmbedder) and embedder.cached([question], "query")
        [fact] = [f for f in memory.system.facts("u") if "PIN" in f.statement]

        memory.delete(fact.id)
        assert embedder.cached([question], "query") == set()
        traces = memory.system.settings.paths.traces.glob("*.jsonl")
        assert not any("PIN" in p.read_text("utf-8") for p in traces)


def test_trace_retention_takes_the_query_vectors_of_expired_traces(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """traces_retention_days：删掉更早的 trace 文件，连同其中提问的查询向量（遗忘只能从 trace 里
    找到提问：trace 过期了、查询向量还在的话，遗忘就清不掉它）；没过期的 trace 与提问留着。没设
    时什么都不删；设了时每天只删一遍。"""
    fake_api.chat_replies = [ScriptedModel(where_i_live)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        system = memory.system
        directory = system.settings.paths.traces
        memory.add("My PIN is 4321", user_id="u", timestamp=AUG, wait=True)
        old, new = "What is my PIN?", "Is my PIN 4321?"
        assert memory.search(old, user_id="u", mode="l1")["results"]
        [today_file] = directory.glob("*.jsonl")
        today_file.rename(directory / "20200101.jsonl")  # 这些 trace 都算很久以前的
        assert memory.search(new, user_id="u", mode="l1")["results"]
        embedder = system.embedder
        assert isinstance(embedder, CachedEmbedder)
        assert embedder.cached([old, new], "query") == {old, new}

        assert system.expire_traces() == (0, 0)  # 没设保留天数
        assert (directory / "20200101.jsonl").exists()
        system.settings = system.settings.model_copy(update={"traces_retention_days": 30})
        files, purged = system.expire_traces()
        assert files == 1 and purged >= 1
        assert embedder.cached([old, new], "query") == {new}
        assert sorted(p.name for p in directory.glob("*.jsonl")) == [today_file.name]

        (directory / "20200102.jsonl").write_text("{}\n", "utf-8")
        assert system.expire_traces() == (0, 0)  # 今天已经删过
        tomorrow = date.today() + timedelta(days=1)
        assert system.expire_traces(today=tomorrow)[0] == 1
        assert not (directory / "20200102.jsonl").exists()


def test_a_run_is_extracted_once_when_it_ends(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """一段对话每轮 add 一次：放在 run() 里，结束时整段只抽取一次（原始日志照样立即可检索）；
    进行中不能 add(wait=True)。不放在 run 里的 add 照旧每次都抽取。"""
    fake_api.chat_replies = [model]
    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        likes = ["jazz", "tea", "rain", "chess", "sushi", "Paris", "running", "cats"]
        with memory.run(user_id="alice") as rid:
            for like in likes:
                turn = [
                    {"role": "user", "content": f"I like {like}"},
                    {"role": "assistant", "content": "Nice."},
                ]
                assert memory.add(turn, user_id="alice", run_id=rid)["status"] == "queued"
            memory.wait()
            assert model.calls["extract"] == 0 and memory.get_all(user_id="alice")["results"] == []
            hits = memory.search("chess", user_id="alice", mode="l1")["results"]
            assert any("chess" in h["memory"] for h in hits)
            with pytest.raises(ValueError, match="still open"):
                memory.add("I live in Berlin", user_id="alice", run_id=rid, wait=True)
        memory.wait()
        assert model.calls["extract"] == 1
        assert model.prompts["extract"][0].count("User: I like") == 8
        assert len(memory.get_all(user_id="alice")["results"]) == 8

        for text in ("I like tea", "I like rain"):
            memory.add(text, user_id="bob", run_id="chat")
            memory.wait()
        assert model.calls["extract"] == 3
        assert "I like tea" in model.prompts["extract"][-1]
        assert "I like rain" in model.prompts["extract"][-1]


def test_close_ends_open_runs(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """close() 结束还没结束的 run：登记抽取任务，下次启动后台线程时完成。"""
    fake_api.chat_replies = [model]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        assert memory.open_run("chat-1", user_id="alice") == "chat-1"
        memory.add("I live in Berlin", user_id="alice", run_id="chat-1")
        assert memory.system.db.get_jobs("alice") == []
    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        memory.wait()
        assert [f["memory"] for f in memory.get_all(user_id="alice")["results"]] == [
            "The user lives in Berlin."
        ]
        assert memory.system.adopt_open("alice") == 0  # 正常结束：没有留下进行中的标记
        ended = memory.end_run("chat-1", user_id="alice", wait=True)
        assert ended["turns"] == 1 and ended["consolidation"]["sessions"] == 0


@pytest.mark.parametrize("first", ["add", "open_run"])
def test_runs_left_open_by_a_crash_are_adopted(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel, first: str
) -> None:
    """进程崩溃时还开着的 run（没有 end_run、没有 close()）：下一个进程第一次碰到这个用户时
    交给抽取，标记随之去掉。用同一个 run_id 再 open_run 的那段接着写，不接手；infer=False
    写入的会话照样不抽取。"""
    fake_api.chat_replies = [model]
    crashed = Memory(n3_settings, http_client=fake_api.client(), background=False)
    left = crashed.open_run(user_id="alice")
    crashed.add("I live in Berlin", user_id="alice", run_id=left)
    crashed.open_run("again", user_id="alice")
    crashed.add("I like jazz", user_id="alice", run_id="again")
    crashed.add("I like tea", user_id="alice", infer=False)
    crashed.system.close()  # 进程被杀：Memory.close() 没来得及结束这两个 run

    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        if first == "add":
            memory.add("I like hiking", user_id="alice", timestamp=datetime(2030, 1, 1))
        else:
            assert memory.open_run("again", user_id="alice") == "again"
        memory.wait()
        jobs = {j["session_id"]: j["status"] for j in memory.system.db.get_jobs("alice")}
        facts = {f["memory"] for f in memory.get_all(user_id="alice")["results"]}
        assert "The user lives in Berlin." in facts
        if first == "add":  # 没有重新打开的 run 一起接手
            assert set(jobs.values()) == {"done"} and len(jobs) == 3 and "again" in jobs
        else:
            assert jobs == {left: "done"}
            assert memory.system.open_sessions("alice") == {"again"}
            memory.end_run("again", user_id="alice", wait=True)
            assert "I like jazz" in str(memory.get_all(user_id="alice")["results"])
        assert not any("tea" in f for f in facts)
        assert memory.system.adopt_open("alice") == 0  # 接手过的标记都去掉了


def test_background_extraction(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    fake_api.chat_replies = [model]
    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        out = memory.add("I live in Berlin", user_id="alice")
        assert out["status"] == "queued" and out["results"] == []
        memory.wait()
        assert [f["object"] for f in memory.get_all(user_id="alice")["results"]] == ["Berlin"]


def test_config_dict_and_home(
    settings: Settings, fake_api: FakeAPI, model: ScriptedModel, tmp_path: Path
) -> None:
    fake_api.chat_replies = [model]
    memory = Memory.from_config(
        {"embedding": {"backend": "hash", "dim": 64}},
        home=tmp_path / "home",
        http_client=fake_api.client(),
        background=False,
    )
    try:
        assert memory.settings.embedding.dim == 64
        assert memory.settings.embedding.model == "qwen3.7-text-embedding"  # 其余字段来自默认配置
        assert memory.settings.paths.db == (tmp_path / "home" / "data" / "cuemem.sqlite").resolve()
        memory.add("I live in Berlin", user_id="alice", wait=True)
        assert memory.get_all(user_id="alice")["results"]
    finally:
        memory.close()


MEM0_CONFIG = {
    "llm": {"provider": "openai", "config": {"model": "gpt-4.1-mini"}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    "vector_store": {"provider": "qdrant", "config": {"host": "localhost", "port": 6333}},
    "custom_instructions": "Only remember shoe sizes.",
    "history_db_path": "/tmp/history.db",
}


@pytest.mark.parametrize(
    ("config", "said"),
    [
        ({"consolidation": {"instruction": "x"}}, "consolidation.instructions"),
        ({"retreival": {}}, "did you mean retrieval?"),
        ({"retrieval": {"controler": "v1.json"}}, "retrieval.controller"),
        (MEM0_CONFIG, 'Memory(preset="openai")'),
        (MEM0_CONFIG, "Memory(instructions=...)"),
        (MEM0_CONFIG, '{"vector_store": {"url": "http://localhost:6333"}}'),
        ({"preset": "openai", "home": "/tmp/x"}, "Pass preset=, home= as keyword arguments"),
    ],
)
def test_unknown_config_keys_are_refused(
    clean_env: None,
    fake_api: FakeAPI,
    tmp_path: Path,
    config: dict[str, Any],
    said: str,
) -> None:
    """配置字典里不认识的键：ValueError，列出带点的路径与最像的已知键；Mem0 格式的配置附上换成
    CueMem 写法的提示。以前它们被悄悄丢掉：Mem0 的 llm: openai 照样用默认的千问与 DeepSeek（用户
    的话发给了没有配置过的供应商），custom_instructions、qdrant 服务器与写错的键都不起作用。
    报错在打开数据目录之前：什么都没建。"""
    for make in (Memory, Memory.from_config):
        with pytest.raises(ValueError, match=re.escape(said)):
            make(config, home=tmp_path / "home", http_client=fake_api.client(), background=False)
    with pytest.raises(ValueError, match="Unknown config keys"):
        AsyncMemory(config, home=tmp_path / "home", http_client=fake_api.client())
    assert not (tmp_path / "home").exists() and fake_api.chat_requests == []


HASH = {"embedding": {"backend": "hash", "dim": 64}}


def test_config_dict_takes_the_vector_store_api_key(
    clean_env: None, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """配置字典可以给 Qdrant 服务端的 API Key（vector_store.api_key）：以前这个键不认识，被
    _check_config 拒绝；本地模式不用它。"""
    config = {**HASH, "vector_store": {"api_key": "k-secret"}}
    memory = Memory(config, home=tmp_path / "home", http_client=fake_api.client(), background=False)
    try:
        assert memory.settings.vector_store.secret() == "k-secret"
        assert "k-secret" not in repr(memory.settings)
    finally:
        memory.close()


def test_sdk_does_not_inherit_the_batch_peak_policy(
    settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """SDK 是交互式的：配置字典（或不给配置）时缺省 peak_policy="warn"，不继承默认配置里批量任务
    用的 wait；显式给的值（字典里的或参数）照用；Settings 对象保留它自己的值。"""

    def policy(config: Any = None, **kwargs: Any) -> str:
        home = tmp_path / "home"  # 配置字典的数据目录；Settings 对象用它自己的 paths
        with Memory(
            config, home=home, http_client=fake_api.client(), background=False, **kwargs
        ) as m:
            assert m.system.llm.peak_policy == m.settings.peak_policy
            return m.settings.peak_policy

    assert policy(HASH) == "warn"
    assert policy({**HASH, "peak_policy": "wait"}) == "wait"
    assert policy(HASH, peak_policy="wait") == "wait"
    assert policy({**HASH, "peak_policy": "allow"}, peak_policy="wait") == "wait"
    assert policy(settings) == "allow"  # conftest 的 Settings：保留它自己的值
    assert policy(settings.model_copy(update={"peak_policy": "wait"})) == "wait"
    assert policy(settings, peak_policy="warn") == "warn"

    async def run() -> str:
        async with AsyncMemory(
            HASH,
            home=tmp_path / "home",
            http_client=fake_api.client(),
            background=False,
            peak_policy="allow",
        ) as m:
            return str(m.settings.peak_policy)

    assert asyncio.run(run()) == "allow"


def test_sdk_calls_are_not_paused_in_peak_hours(
    settings: Settings, fake_api: FakeAPI, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """DeepSeek 的高峰时段（周三 10:00，北京时间）：以前 SDK 继承 wait，answer() 与 L2 的
    search() 各自暂停到 12:00，看起来像卡死。现在照常调用；选了 wait 时开始暂停记一条警告。"""
    from zoneinfo import ZoneInfo

    fake_api.chat_replies = [ScriptedModel(where_i_live)]
    clock = [datetime(2026, 10, 14, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += timedelta(seconds=seconds)

    def at_peak(m: Memory) -> Memory:
        m.system.llm._clock = lambda: clock[0]
        m.system.llm._sleep = sleep
        return m

    def deepseek(**kwargs: Any) -> Memory:
        return Memory(
            HASH,
            preset="deepseek",
            home=tmp_path,
            http_client=fake_api.client(),
            background=False,
            **kwargs,
        )

    with at_peak(deepseek()) as m:
        m.add("I live in Berlin", user_id="alice", infer=False)
        m.answer("Where do I live?", user_id="alice", mode="l1")
        m.search("Where do I live?", user_id="alice", mode="l2")
        m.add("I moved to Munich", user_id="alice", wait=True)
    models = [r["model"] for r in fake_api.chat_requests]
    assert slept == [] and len(models) >= 3 and set(models) == {"deepseek-flash"}

    with at_peak(deepseek(peak_policy="wait")) as m, caplog.at_level("WARNING", "cuemem.sdk"):
        m.answer("Where else did I live?", user_id="alice", mode="l1")
    assert sum(slept) == 2 * 3600  # 选了 wait：暂停到 12:00
    assert "deepseek peak hours: pausing model calls until 2026-10-14T12:00" in caplog.text


def test_async_memory(n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel) -> None:
    fake_api.chat_replies = [model]

    async def run() -> list[str]:
        async with AsyncMemory(n3_settings, http_client=fake_api.client(), background=False) as m:
            await m.add("I live in Berlin", user_id="alice", wait=True)
            found = await m.get_all(user_id="alice")
            return [f["memory"] for f in found["results"]]

    assert asyncio.run(run()) == ["The user lives in Berlin."]


def test_async_run(n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel) -> None:
    fake_api.chat_replies = [model]

    async def run() -> tuple[list[Any], list[str]]:
        async with AsyncMemory(n3_settings, http_client=fake_api.client(), background=False) as m:
            async with m.run(user_id="alice", wait=True) as rid:
                await m.add("I live in Berlin", user_id="alice", run_id=rid)
                during = (await m.get_all(user_id="alice"))["results"]
            found = await m.get_all(user_id="alice")
            return during, [f["memory"] for f in found["results"]]

    assert asyncio.run(run()) == ([], ["The user lives in Berlin."])


def test_status_counts_jobs_without_reading_them_all(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """status() 只数待处理的、只读失败的任务（以前读出并解开每个任务：1 万个任务时 40 毫秒，界面
    每发一条消息查一次，第五轮体检）。结果与以前相同，失败的仍按以前的顺序（按会话，同一会话的按
    写入的先后）；HTTP 服务的状态同样只数。"""
    from cuemem.agent import Agent
    from cuemem.api.service import ChatService

    db = memory.system.db
    space = space_for("alice", None)
    db.ensure_space(space)
    db.enqueue_jobs(
        space,
        [
            ("j1", "zz", "v", "h1"),
            ("j2", "aa", "v", "h2"),
            ("j3", "mm", "v", "h3"),
            ("j4", "dd", "v", "h4"),
            ("j5", "ss", "v", "h5"),
            ("j6", "ss", "v", "h6"),  # j5 被取代
            ("j7", "aa", "old", "h7"),  # j2 被取代，下面再改回失败：同一会话两个失败的
        ],
    )
    with db.transaction() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'failed', attempts = 2, error = 'boom\nmore'"
            " WHERE id IN ('j1', 'j2', 'j7')"
        )
        conn.execute("UPDATE jobs SET status = 'done' WHERE id = 'j4'")

    def reference() -> dict[str, Any]:  # 以前的写法：读出全部任务再挑
        jobs = db.get_jobs(space)
        failed = [
            {
                "session_id": j["session_id"],
                "attempts": j["attempts"],
                "error": (j["error"] or "").splitlines()[0][:300],
                "retry_at": None,
            }
            for j in jobs
            if j["status"] == "failed"
        ]
        counts = {s: sum(1 for j in jobs if j["status"] == s) for s in ("pending", "done")}
        return {"failed": failed, **counts}

    expected = reference()
    assert [f["session_id"] for f in expected["failed"]] == ["aa", "aa", "zz"]
    chat = ChatService(memory.system, Agent(memory.system, router="rule"))
    calls: list[str | None] = []
    real = type(db).get_jobs

    def spy(self: Any, space_id: str, status: str | None = None) -> list[dict[str, Any]]:
        calls.append(status)
        return real(self, space_id, status)

    monkeypatch.setattr(type(db), "get_jobs", spy)
    state = memory.status(user_id="alice")
    served = chat.status(space)
    assert None not in calls
    assert state["pending"] == expected["pending"] == 2
    assert state["failed"] == expected["failed"]
    assert {k: served[k] for k in ("pending", "failed", "done")} == {
        "pending": 2,
        "failed": 3,
        "done": expected["done"],
    }
    assert db.job_counts(space) == {"pending": 2, "failed": 3, "done": 1, "superseded": 1}


def test_unopenable_database_raises_database_open_error(
    offline_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """数据库打不开（路径上是个目录）：以前抛出不带路径的 sqlite3.OperationalError，账本的连接也
    一直开着；现在 DatabaseOpenError 说明是哪个文件，它同时是 CueMemError 与
    sqlite3.OperationalError（按后者捕获的调用方照样捕获得到），已经打开的账本关掉。"""
    from cuemem.errors import CueMemError, DatabaseOpenError
    from cuemem.llm.ledger import Ledger

    offline_settings.paths.db.mkdir(parents=True)
    closed: list[str] = []
    original = Ledger.close

    def close(self: Ledger) -> None:
        closed.append("ledger")
        original(self)

    monkeypatch.setattr(Ledger, "close", close)
    with pytest.raises(DatabaseOpenError) as caught:
        Memory(offline_settings, background=False)
    assert isinstance(caught.value, sqlite3.OperationalError)
    assert isinstance(caught.value, CueMemError)
    assert str(offline_settings.paths.db) in str(caught.value)
    assert "打不开" in str(caught.value)
    assert closed == ["ledger"]


def test_data_path_blocked_by_a_file_raises_database_open_error(
    offline_settings: Settings,
) -> None:
    """数据库的上级目录是个文件：以前 mkdir 抛出 FileExistsError（不是 CueMemError），现在也是
    DatabaseOpenError。"""
    from cuemem.errors import DatabaseOpenError

    blocker = offline_settings.paths.db.parent / "plain-file"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("", encoding="utf-8")
    paths = offline_settings.paths.model_copy(update={"db": blocker / "cuemem.sqlite"})
    with pytest.raises(DatabaseOpenError, match="plain-file"):
        Memory(offline_settings.model_copy(update={"paths": paths}), background=False)


@pytest.mark.parametrize("which", ["db", "ledger", "llm_cache", "embed_cache"])
def test_corrupt_store_raises_database_open_error(
    offline_settings: Settings, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """数据库、账本或缓存损坏、不是 SQLite 数据库（崩溃之后、拷坏了）：以前抛出不带路径的
    sqlite3.DatabaseError（file is not a database），看不出是四个文件里的哪一个；现在
    DatabaseOpenError 写明文件与从备份恢复，数据库的锁放掉，坏文件的连接也关掉（第八轮体检）。"""
    from cuemem.errors import DatabaseOpenError
    from cuemem.store.files import lock_database, unlock_database

    path: Path = getattr(offline_settings.paths, which)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(8192))
    opened: list[sqlite3.Connection] = []
    connect = sqlite3.connect

    def tracking(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = connect(*args, **kwargs)
        if str(args[0]) == str(path):
            opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking)
    with pytest.raises(DatabaseOpenError) as caught:
        Memory(offline_settings, background=False)
    assert isinstance(caught.value, sqlite3.OperationalError)
    assert str(path) in str(caught.value)
    assert "不是 SQLite 数据库" in str(caught.value) and "从备份恢复" in str(caught.value)
    assert "检查目录权限" not in str(caught.value)
    assert opened
    for conn in opened:  # 已经关掉：再用就是 ProgrammingError
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
    unlock_database(lock_database(offline_settings.paths.db))  # 锁已经放掉


def _second_vector_dir(settings: Settings) -> Settings:
    """同一个数据库、另一个向量库目录：两个进程共用数据库、向量在 Qdrant 服务端时的样子
    （本地模式的 Qdrant 锁挡不住它们）。"""
    paths = settings.paths.model_copy(update={"qdrant": settings.paths.qdrant.parent / "qdrant-b"})
    return settings.model_copy(update={"paths": paths})


def test_second_system_on_one_database_is_refused(offline_settings: Settings) -> None:
    """一个数据库同一时间只能有一个进程写：以前向量在服务端时第二个 Memory() 照样打开，接手第一个
    还在写的会话（这里 B 的 add 把 A 进行中的 r1 登记成待抽取）。现在第二个打开者
    StorageLockedError，说明拿着锁的进程；第一个关掉之后就能打开。"""
    from cuemem.errors import StorageLockedError

    first = Memory(offline_settings, background=False)
    try:
        first.open_run("r1", user_id="u")
        first.add("I live in Berlin.", user_id="u", run_id="r1", infer=False)
        with pytest.raises(StorageLockedError) as caught:
            Memory(_second_vector_dir(offline_settings), background=False)
        message = str(caught.value)
        assert str(offline_settings.paths.db) in message and f"pid {os.getpid()}" in message
        assert "CUEMEM_ALLOW_SHARED_DB=1" in message
    finally:
        first.close()
    with Memory(_second_vector_dir(offline_settings), background=False) as second:
        second.add("Hello", user_id="u", infer=False)


def test_database_lock_holds_across_processes(offline_settings: Settings, tmp_path: Path) -> None:
    """另一个进程也拿不到这把锁（flock），报错里是这个进程的 pid。"""
    import subprocess
    import sys

    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from cuemem.errors import StorageLockedError\n"
        "from cuemem.store.files import lock_database\n"
        "try:\n"
        "    lock_database(Path(sys.argv[1]))\n"
        "except StorageLockedError as exc:\n"
        "    print(exc)\n"
        "else:\n"
        "    print('locked')\n"
    )
    with Memory(offline_settings, background=False):
        out = subprocess.run(
            [sys.executable, "-c", script, str(offline_settings.paths.db)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            check=True,
            env={k: v for k, v in os.environ.items() if k != "CUEMEM_ALLOW_SHARED_DB"},
        ).stdout
    assert "正被另一个进程使用" in out and f"pid {os.getpid()}" in out, out


def test_shared_database_opt_out_warns(
    offline_settings: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """CUEMEM_ALLOW_SHARED_DB=1：不加锁，两个都打开，并记一条警告说明风险。"""
    monkeypatch.setenv("CUEMEM_ALLOW_SHARED_DB", "1")
    with (
        caplog.at_level("WARNING", logger="cuemem.store.files"),
        Memory(offline_settings, background=False),
        Memory(_second_vector_dir(offline_settings), background=False),
    ):
        pass
    warned = [r.getMessage() for r in caplog.records if "CUEMEM_ALLOW_SHARED_DB" in r.getMessage()]
    assert len(warned) == 2 and "reindex" in warned[0]


def test_full_embedding_cache_does_not_break_search(
    memory: Memory, caplog: pytest.LogCaptureFixture
) -> None:
    """向量缓存所在的磁盘满了：以前每个没见过的问题都抛出 database or disk is full（检索用不了）；
    现在照样检索，只是问题的向量没有缓存，接连失败只记一条警告。"""
    from fakes import fill_to_capacity

    memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
    embedder = memory.system.embedder
    assert isinstance(embedder, CachedEmbedder)
    fill_to_capacity(embedder.cache._conn)
    with caplog.at_level("WARNING", logger="cuemem.store.files"):
        for question in ("Where do I live these days?", "Which city is my home now?"):
            found = memory.search(question, user_id="alice", mode="l1")
            assert any(r["memory"] == "The user lives in Berlin." for r in found["results"])
    warned = [r.getMessage() for r in caplog.records if "向量缓存写不进" in r.getMessage()]
    assert len(warned) == 1


def test_full_llm_cache_still_answers_and_accounts(
    n3_settings: Settings, fake_api: FakeAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """LLM 缓存所在的磁盘满了：以前付费拿到的回答写不进缓存就抛出，账本也没记这笔（先写缓存后
    记账）；现在回答照样返回、每次付费的调用都记了账，只记一条警告。"""
    from fakes import fill_to_capacity

    model = ScriptedModel(
        where_i_live,
        reader=lambda prompt: {
            "answer": "Berlin. " * 2500,
            "citations": ["f1"],
            "evidence_sufficient": True,
        },
    )
    fake_api.chat_replies = [model]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        memory.add("I live in Berlin", user_id="alice", timestamp=AUG, wait=True)
        system = memory.system
        assert system.llm.cache is not None
        fill_to_capacity(system.llm.cache._conn)

        def rows() -> int:
            return int(system.ledger._conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0])

        before, paid = rows(), len(fake_api.chat_requests)
        with caplog.at_level("WARNING", logger="cuemem.store.files"):
            for question in ("Where do I live?", "Which city do I live in?"):
                answer = memory.answer(question, user_id="alice", mode="l1")
                assert answer["answer"].startswith("Berlin.")
        assert len(fake_api.chat_requests) - paid == 2
        assert rows() - before == 2
    warned = [r.getMessage() for r in caplog.records if "LLM 缓存写不进" in r.getMessage()]
    assert len(warned) == 1
