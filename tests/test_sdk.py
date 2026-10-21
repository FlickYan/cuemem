"""Python SDK（N12，D-061）：Mem0 形状的接口，CueMem 的语义（版本链、控制器、遗忘）。全部离线。"""

from __future__ import annotations

import asyncio
import base64
import io
import itertools
import re
import sqlite3
import tempfile
import threading
import uuid
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI
from PIL import Image
from scripted import ScriptedModel

from cuemem import AsyncMemory, Memory
from cuemem.config import Settings
from cuemem.embed import CachedEmbedder
from cuemem.embed.base import EmbedKind
from cuemem.errors import ConcurrentWriteError, EmbeddingAPIError
from cuemem.evidence import stored_text
from cuemem.sdk import parse_messages, space_for

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
    with pytest.raises(ValueError, match="Unsupported filters"):
        memory.search("x", filters={"user_id": "alice", "categories": ["home"]})

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
    提交，向量库里有点、数据库里却没有"轮"。"""
    system = memory.system

    def broken(units: Sequence[Any]) -> int:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(system.db, "add_units", broken)
    with pytest.raises(sqlite3.OperationalError):
        memory.add("I live in Berlin", user_id="alice", run_id="r1")
    assert system.db.get_turns("alice") == [] and system.vectors.count("alice") == 0


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


HASH = {"embedding": {"backend": "hash", "dim": 64}}


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
