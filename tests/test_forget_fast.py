"""遗忘的快速路径（第七轮体检 P21）：整段删掉的会话什么都没留下（没有事实、没有新建或关联的
实体）时，遗忘只删它们的发言、片段、摘要与任务，不再清空整个空间重放。以前 update() 的更正
抽不出事实时也重放整个空间（2 万轮约 59 秒，期间这个用户的读取都在等）。

等价性：同一份数据复制两份，一份走快速路径、一份强制重放，派生层（事实、实体、片段、核心
记忆、检索单元、任务）与向量库的点都相同。全部离线。"""

from __future__ import annotations

import re
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI
from scripted import ScriptedModel

from cuemem import Memory
from cuemem.config import PathsConfig, Settings

SPACE = "alice"
START = datetime(2026, 8, 1, 10, 0)


def extract(prompt: str) -> dict[str, Any]:
    """抽取替身：'I live in X' → 单值事实；'I like X' → 多值事实；闲聊（smalltalk）与含糊的
    更正（not true）什么都抽不出，带 'chat' 的有摘要。"""
    summary = f"Chat {len(prompt) % 97}." if "chat" in prompt else ""
    facts = []
    for n, text in re.findall(r"\[t(\d+)\] User: (.*)", prompt):
        if "smalltalk" in text or "not true" in text:
            continue
        place = re.search(r"live in (\w+)", text)
        liked = re.search(r"like (\w+)", text)
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
        elif liked:
            facts.append(
                {
                    "statement": f"The user likes {liked.group(1)}.",
                    "subject": "User",
                    "predicate": "likes",
                    "object": liked.group(1),
                    "cardinality": "multi",
                    "importance": 4,
                    "turns": [int(n)],
                }
            )
    return {"summary": summary, "facts": facts, "entities": []}


@pytest.fixture
def model() -> ScriptedModel:
    return ScriptedModel(
        extract, reconcile=lambda prompt: {"decisions": [{"new": 1, "action": "update"}]}
    )


def _settings(base: Settings, root: Path) -> Settings:
    return base.model_copy(update={"paths": PathsConfig().resolve_against(root)})


@contextmanager
def opened(settings: Settings, fake_api: FakeAPI) -> Iterator[Memory]:
    with Memory(settings, http_client=fake_api.client(), background=False) as m:
        yield m


def history(m: Memory, texts: list[str]) -> list[str]:
    """每句一段会话（按天排开），同步巩固；返回各段的会话 ID。"""
    sessions = []
    for i, text in enumerate(texts):
        out = m.add(text, user_id=SPACE, timestamp=START + timedelta(days=i), wait=True)
        sessions.append(out["session_id"])
    return sessions


def snapshot(m: Memory) -> dict[str, Any]:
    """派生层与原始层（不比时间戳：核心记忆与任务的 created_at、updated_at）与向量库的点。"""
    conn = m.system.db._conn

    def rows(sql: str) -> list[tuple[Any, ...]]:
        return sorted(tuple(r) for r in conn.execute(sql, (SPACE,)))

    return {
        "turns": rows("SELECT * FROM turns WHERE space_id = ?"),
        "facts": rows("SELECT * FROM facts WHERE space_id = ?"),
        "entities": rows("SELECT * FROM entities WHERE space_id = ?"),
        "episodes": rows("SELECT * FROM episodes WHERE space_id = ?"),
        "core": rows("SELECT text, fact_ids FROM core_memory WHERE space_id = ?"),
        "units": rows("SELECT * FROM units WHERE space_id = ?"),
        "jobs": rows(
            "SELECT id, session_id, version, content_hash, status, attempts, error, stats"
            " FROM jobs WHERE space_id = ?"
        ),
        "points": sorted(m.system.vectors.unit_ids(SPACE)),
    }


def runs(m: Memory) -> list[int]:
    """数 consolidator.run 被调用了几次（重放会调用它）。"""
    calls = [0]
    original = m.system.consolidator.run

    def counting(space_id: str) -> Any:
        calls[0] += 1
        return original(space_id)

    m.system.consolidator.run = counting  # type: ignore[method-assign]
    return calls


def fast_vs_replay(
    tmp_path: Path,
    n3_settings: Settings,
    fake_api: FakeAPI,
    model: ScriptedModel,
    texts: list[str],
    act: Callable[[Memory, list[str]], Any],
) -> tuple[dict[str, Any], dict[str, Any], list[int], list[int]]:
    """同一段历史复制两份：A 照常（快速路径），B 强制重放。返回两边的快照与 run 的次数。"""
    fake_api.chat_replies = [model]
    a, b = _settings(n3_settings, tmp_path / "a"), _settings(n3_settings, tmp_path / "b")
    with opened(a, fake_api) as m:
        sessions = history(m, texts)
    shutil.copytree(tmp_path / "a", tmp_path / "b")
    with opened(a, fake_api) as m:
        fast_runs = runs(m)
        act(m, sessions)
        fast = snapshot(m)
    with opened(b, fake_api) as m:
        m.system._inert_sessions = lambda *args, **kwargs: False  # type: ignore[method-assign]
        slow_runs = runs(m)
        act(m, sessions)
        slow = snapshot(m)
    return fast, slow, fast_runs, slow_runs


def correct(text: str) -> Callable[[Memory, list[str]], Any]:
    def act(m: Memory, sessions: list[str]) -> Any:
        (berlin,) = [f for f in m.get_all(user_id=SPACE)["results"] if "Berlin" in f["memory"]]
        out = m.update(berlin["id"], text)
        assert out["message"].startswith("No memory could be extracted")
        assert out["turn_ids"] == []

    return act


HISTORY = ["I live in Berlin", "smalltalk chat about rain", "I like jazz", "I like tea"]


@pytest.mark.parametrize("text", ["not true anymore", "not true anymore, chat"])
def test_a_correction_that_extracts_nothing_matches_the_replay(
    tmp_path: Path, n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel, text: str
) -> None:
    """更正抽不出事实：没有摘要的与有摘要的（它的片段、摘要单元与向量两边都删了）。"""
    fast, slow, fast_runs, slow_runs = fast_vs_replay(
        tmp_path, n3_settings, fake_api, model, HISTORY, correct(text)
    )
    assert fast == slow
    assert fast_runs == [1] and slow_runs == [2]  # update 自己的巩固一次；快速路径不再重放
    assert not any("Correction" in t[5] for t in fast["turns"])
    assert len(fast["episodes"]) == 1 and len(fast["facts"]) == 3


def test_forgetting_an_inert_session_in_the_middle_matches_the_replay(
    tmp_path: Path, n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """历史中间一段什么都没留下的闲聊（有摘要）：删掉它也不重放，结果与重放相同。"""

    def act(m: Memory, sessions: list[str]) -> None:
        report = m.system.forget(SPACE, session_ids=[sessions[1]])
        assert report.turns_deleted == 1

    fast, slow, fast_runs, slow_runs = fast_vs_replay(
        tmp_path, n3_settings, fake_api, model, HISTORY, act
    )
    assert fast == slow
    assert fast_runs == [0] and slow_runs == [1]
    assert fast["episodes"] == [] and not any("smalltalk" in t[5] for t in fast["turns"])


def test_update_that_extracts_nothing_does_not_replay_the_space(
    tmp_path: Path, n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """50 段会话的空间里 update 一句抽不出事实的更正：除了它自己的巩固不再调用 run；结果与以前
    相同（原来那条不变，更正不留在日志、检索与导出里）。"""
    fake_api.chat_replies = [model]
    with opened(_settings(n3_settings, tmp_path / "a"), fake_api) as m:
        history(m, ["I live in Berlin", *[f"I like thing{i}" for i in range(49)]])
        (berlin,) = [f for f in m.get_all(user_id=SPACE)["results"] if "Berlin" in f["memory"]]
        calls = runs(m)
        started = time.perf_counter()
        out = m.update(berlin["id"], "not true anymore")
        elapsed = time.perf_counter() - started
        assert calls == [1], elapsed
        assert out["message"].startswith("No memory could be extracted") and out["turn_ids"] == []
        assert "The user lives in Berlin." in [
            f["memory"] for f in m.get_all(user_id=SPACE)["results"]
        ]
        assert len(m.system.db.get_turns(SPACE)) == 50
        found = m.search("not true anymore", user_id=SPACE, mode="l1")["results"]
        assert not any("Correction" in hit["memory"] for hit in found)
        exported = m.export(user_id=SPACE)["sessions"]
        assert not any("Correction" in t["text"] for s in exported for t in s["turns"])


# ---------- 不走快速路径的情形：照旧重放 ----------


def replays(m: Memory, act: Callable[[], Any]) -> bool:
    calls = runs(m)
    act()
    return calls[0] > 0


def test_a_session_that_repeats_a_fact_still_replays(
    tmp_path: Path, n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """抽出的事实与已有的相同（重复，没有新增）：任务统计里 facts 不是 0，照旧重放。"""
    fake_api.chat_replies = [model]
    with opened(_settings(n3_settings, tmp_path / "a"), fake_api) as m:
        history(m, HISTORY)
        out = m.add("I like jazz", user_id=SPACE, timestamp=START + timedelta(days=30), wait=True)
        (job,) = [j for j in m.system.db.get_jobs(SPACE) if j["session_id"] == out["session_id"]]
        assert job["stats"]["facts"] == 1 and job["stats"]["added"] == 0
        assert replays(m, lambda: m.system.forget(SPACE, session_ids=[out["session_id"]]))


def test_a_session_that_linked_an_entity_still_replays(
    tmp_path: Path, n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    fake_api.chat_replies = [model]
    with opened(_settings(n3_settings, tmp_path / "a"), fake_api) as m:
        sessions = history(m, HISTORY)
        conn = m.system.db._conn
        conn.execute(
            "UPDATE jobs SET stats = json_set(stats, '$.entities_linked', 1)"
            " WHERE space_id = ? AND session_id = ?",
            (SPACE, sessions[1]),
        )
        conn.commit()
        assert replays(m, lambda: m.system.forget(SPACE, session_ids=[sessions[1]]))


def test_forgetting_part_of_a_session_still_replays(
    tmp_path: Path, n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    fake_api.chat_replies = [model]
    with opened(_settings(n3_settings, tmp_path / "a"), fake_api) as m:
        history(m, HISTORY)
        out = m.add(
            [
                {"role": "user", "content": "smalltalk one"},
                {"role": "user", "content": "smalltalk two"},
            ],
            user_id=SPACE,
            timestamp=START + timedelta(days=30),
            wait=True,
        )
        first = m.system.db.get_turns(SPACE, out["session_id"])[0].id
        assert replays(m, lambda: m.system.forget(SPACE, turn_ids=[first]))


def test_a_space_with_a_pending_job_still_replays(
    tmp_path: Path, n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    fake_api.chat_replies = [model]
    with opened(_settings(n3_settings, tmp_path / "a"), fake_api) as m:
        sessions = history(m, HISTORY)
        m.add("I like cake", user_id=SPACE, timestamp=START + timedelta(days=30))  # 不等巩固
        assert m.system.db.job_counts(SPACE).get("pending") == 1
        assert replays(m, lambda: m.system.forget(SPACE, session_ids=[sessions[1]]))
