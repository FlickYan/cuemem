"""Python SDK（N12，D-061）：Mem0 形状的接口，CueMem 的语义（版本链、控制器、遗忘）。全部离线。"""

from __future__ import annotations

import asyncio
import base64
import io
import re
import tempfile
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI
from PIL import Image
from scripted import ScriptedModel

from cuemem import AsyncMemory, Memory
from cuemem.config import Settings
from cuemem.embed import CachedEmbedder
from cuemem.sdk import space_for

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


def test_async_memory(n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel) -> None:
    fake_api.chat_replies = [model]

    async def run() -> list[str]:
        async with AsyncMemory(n3_settings, http_client=fake_api.client(), background=False) as m:
            await m.add("I live in Berlin", user_id="alice", wait=True)
            found = await m.get_all(user_id="alice")
            return [f["memory"] for f in found["results"]]

    assert asyncio.run(run()) == ["The user lives in Berlin."]
