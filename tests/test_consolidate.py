"""巩固的端到端场景：用示例对话（林晓换工作、搬到新加坡、笔记本预算变化）检验
知识更新、历史状态查询、双时间轴、幂等与重放、迟到消息、核心记忆和事实增强回答。"""

from __future__ import annotations

import inspect
import itertools
import json
import logging
import re
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import numpy as np
import pytest
from fakes import NO_BALANCE, TOY_CONVERSATION, FakeAPI, chat_completion_body
from scripted import (
    TOY_EXTRACTIONS,
    TOY_UPDATE_KEYWORDS,
    ScriptedModel,
    echo_extraction,
    growing_extraction,
    reconcile_by_keyword,
)
from test_sdk import AUG, SEP, moved, where_i_live

from cuemem import Memory
from cuemem import sdk as sdk_module
from cuemem.config import PathsConfig, Settings
from cuemem.core.ids import stable_id
from cuemem.core.models import Conversation, Entity, Fact, SessionIn, Turn, TurnIn
from cuemem.core.trace import start_trace
from cuemem.embed import CachedEmbedder
from cuemem.embed.base import EmbeddingBatch, EmbedKind
from cuemem.embed.multimodal import DashScopeMultimodalEmbedder
from cuemem.memory import MemorySystem
from cuemem.sdk import space_for
from cuemem.store import MemoryDB
from cuemem.write import consolidate as consolidate_module
from cuemem.write.consolidate import MAX_ATTEMPTS, REBUILD_PENDING, Consolidator
from cuemem.write.extract import PIPELINE_VERSION, Extractor, session_content_hash

EXTRACT = "You build the long-term memory"  # 抽取请求的系统提示开头
RECONCILE = "You maintain a long-term memory"  # 对账请求的系统提示开头


def toy() -> Conversation:
    return Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))


def by_statement(facts: list[Fact], text: str) -> Fact:
    [fact] = [f for f in facts if text in f.statement]
    return fact


@pytest.fixture
def model(fake_api: FakeAPI) -> ScriptedModel:
    scripted = ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    fake_api.chat_replies = [scripted]
    return scripted


@pytest.fixture
def system(n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel) -> MemorySystem:
    memory = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    memory.ingest(toy())
    report = memory.consolidate("demo")
    assert report.sessions == 5 and not report.failed
    yield memory
    memory.close()


def test_knowledge_update_keeps_history(system: MemorySystem, model: ScriptedModel) -> None:
    facts = system.facts("demo")
    beijing = by_statement(facts, "北京一家互联网公司")
    singapore = by_statement(facts, "新加坡一家电商公司")
    # 单值事实换了新值：LLM 确认后取代。现实变了，旧事实有结束时间，但没有被删除
    assert singapore.is_current and not beijing.is_current
    assert beijing.superseded_by == singapore.id
    assert beijing.supersede_reason == "changed"
    assert beijing.valid_to == "2025-08"  # 新工作的开始时间
    assert beijing.invalidated_at == datetime(2025, 9, 10, 19, 0)  # 系统得知的时间

    # 同一会话的灰区事实（工作、职位、预算）合并成一次 LLM 调用
    old_budget = by_statement(facts, "8000")
    new_budget = by_statement(facts, "6000")
    assert old_budget.superseded_by == new_budget.id and new_budget.is_current
    assert model.calls["reconcile"] == 1
    prompt = model.prompts["reconcile"][0]
    assert "8000" in prompt and "北京一家互联网公司" in prompt and "New fact 3" in prompt

    # 多值事实累积；关于 Max 的事实挂在 Max 这个实体上
    current = {f.statement for f in system.facts("demo", current_only=True)}
    assert {
        "Max 喜欢在东海岸公园散步。",
        "用户在 2025 年 2 月底领养了一只名叫 Max 的金毛。",
    } <= current
    max_entity = next(e for e in system.entities("demo") if e.name == "Max")
    assert max_entity.type == "pet" and max_entity.mentions >= 3
    assert by_statement(facts, "鸡肉味").subject_id == max_entity.id
    assert "我的狗" not in max_entity.aliases  # 泛称不作为别名


def test_bitemporal_queries(system: MemorySystem) -> None:
    def works(**kwargs: object) -> set[str]:
        return {f.object for f in system.facts("demo", **kwargs) if f.predicate == "works_at"}  # type: ignore[arg-type]

    # 事件时间轴："2025 年 6 月，现实中用户在哪工作"
    assert works(valid_at="2025-06-01") == {"北京一家互联网公司"}
    assert works(valid_at="2025-08-15") == {"新加坡一家电商公司"}
    # 记录时间轴："2025 年 9 月 1 日那天，系统认为用户在哪工作"——那时还没听说跳槽
    assert works(known_at=datetime(2025, 9, 1)) == {"北京一家互联网公司"}
    assert works(known_at=datetime(2025, 10, 1)) == {"新加坡一家电商公司"}
    assert works(current_only=True) == {"新加坡一家电商公司"}


def test_core_memory_is_a_view_of_current_facts(system: MemorySystem) -> None:
    core = system.core_memory("demo")
    assert core is not None
    assert "新加坡一家电商公司" in core.text and "林晓" in core.text
    assert "北京" not in core.text  # 被取代的事实不进画像
    assert "鸡肉味" not in core.text  # 重要度不够
    assert set(core.fact_ids) <= {f.id for f in system.facts("demo", current_only=True)}


def test_consolidation_is_idempotent_and_replayable(
    system: MemorySystem, fake_api: FakeAPI
) -> None:
    before = {(f.id, f.statement, f.is_current) for f in system.facts("demo")}
    requests = len(fake_api.chat_requests)

    again = system.consolidate("demo")  # 任务都已完成：什么也不做
    assert again.sessions == 0
    assert len(fake_api.chat_requests) == requests

    system.ingest(toy())  # 重复导入同一份对话不会登记新任务，也不会重放（会话没长）
    assert system.consolidate("demo").sessions == 0

    rebuilt = system.consolidate("demo", rebuild=True)  # 清空派生记忆后从原始日志重放
    assert rebuilt.sessions == 5
    assert {(f.id, f.statement, f.is_current) for f in system.facts("demo")} == before
    assert len(fake_api.chat_requests) == requests  # 全部命中响应缓存，不花钱
    jobs = system.db.get_jobs("demo")
    assert len(jobs) == 5 and {j["status"] for j in jobs} == {"done"}
    assert jobs[0]["stats"]["facts"] > 0


def test_index_contains_facts_summaries_and_entities(system: MemorySystem) -> None:
    kinds = {
        kind: system.vectors.count("demo", kinds=[kind])
        for kind in ("round", "fact", "summary", "entity")
    }
    assert kinds["round"] == 9
    assert kinds["fact"] == len(system.facts("demo"))
    assert kinds["summary"] == 5
    assert kinds["entity"] == 3  # Max、巴厘岛、东海岸公园（用户本人不进向量索引）
    # 向量库与数据库一一对应
    assert set(system.vectors.unit_ids("demo")) == {u.id for u in system.db.list_units("demo")}


def test_ask_with_facts_shows_validity(system: MemorySystem, model: ScriptedModel) -> None:
    answer = system.ask("demo", "我现在在哪工作？", question_time=datetime(2025, 10, 10))
    prompt = model.prompts["read"][-1]
    assert "About the user (core memory" in prompt
    assert "Notes from memory" in prompt
    # 命中旧事实时，取代它的新事实沿版本链一起带出，并标注状态
    assert "用户于 2025 年 8 月跳槽到新加坡一家电商公司。" in prompt
    assert "no longer true from 2025-08" in prompt
    assert "current) 用户于 2025 年 8 月跳槽" in prompt
    assert answer.facts and all(ref in answer.refs for ref in ("m1", "f1"))

    naive = system.ask(
        "demo", "我现在在哪工作？", question_time=datetime(2025, 10, 10), mode="naive"
    )
    assert "Notes from memory" not in model.prompts["read"][-1]
    assert naive.facts == [] and all(h.unit.kind == "round" for h in naive.hits)


def test_late_arriving_old_session_does_not_override(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    conv = toy()
    newer = conv.model_copy(update={"sessions": conv.sessions[2:]})
    older = conv.model_copy(update={"sessions": conv.sessions[:2]})
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as memory:
        memory.ingest(newer, consolidate=True)
        memory.ingest(older, consolidate=True)  # 后导入更早的两段对话
        facts = memory.facts("demo")
        beijing = by_statement(facts, "北京一家互联网公司")
        singapore = by_statement(facts, "新加坡一家电商公司")
        assert singapore.is_current and not beijing.is_current
        assert beijing.superseded_by == singapore.id and beijing.valid_to == "2025-08"
        # 与按时间顺序处理的结果一致：系统在 2025-09-10 才得知旧工作不再成立
        assert beijing.invalidated_at == datetime(2025, 9, 10, 19, 0)
        assert by_statement(facts, "6000").is_current
        assert not by_statement(facts, "8000").is_current
        current = {f.statement for f in memory.facts("demo", current_only=True)}
    # 对照：另一个独立的存储里按时间顺序导入同一份对话
    fresh_paths = n3_settings.paths.model_copy(
        update={
            "db": n3_settings.paths.db.with_name("in-order.sqlite"),
            "qdrant": n3_settings.paths.qdrant.with_name("qdrant-in-order"),
        }
    )
    with MemorySystem.from_settings(
        n3_settings.model_copy(update={"paths": fresh_paths}), http_client=fake_api.client()
    ) as in_order:
        in_order.ingest(conv, consolidate=True)
        assert {f.statement for f in in_order.facts("demo", current_only=True)} == current


def test_background_worker(n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel) -> None:
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as memory:
        worker = memory.start_worker()
        memory.ingest(toy())  # 只登记任务，立即返回
        worker.wait()
        assert not worker.errors
        assert worker.reports and worker.reports[0].sessions == 5
        assert memory.facts("demo", current_only=True)


def outage(model: ScriptedModel) -> tuple[dict[str, int], Any]:
    """供应商故障的替身：left 还剩几次时，抽取请求返回余额不足（HTTP 402，SDK 不重试）。"""
    left = {"left": 0}

    def reply(body: dict[str, Any]) -> Any:
        if left["left"] and body["messages"][0]["content"].startswith(EXTRACT):
            left["left"] -= 1
            return NO_BALANCE
        return model(body)

    return left, reply


def statuses(memory: Memory, space: str) -> list[tuple[str, int]]:
    return sorted((j["status"], j["attempts"]) for j in memory.system.db.get_jobs(space))


def test_worker_retries_failed_job_after_outage(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """后台抽取遇上供应商故障：不用等显式的巩固，后台线程按退避时间自动重试；失败 3 次之后
    不再自动重试（之后的会话照常抽取），显式的 consolidate 仍能补上。"""
    down, reply = outage(ScriptedModel(where_i_live))
    fake_api.chat_replies = [reply]
    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        worker = memory.system.worker
        assert worker is not None
        worker.retry_base_s = 0.0  # 不等退避时间：失败之后马上重试
        down["left"] = 2  # 前两次失败，第三次成功
        memory.add("I live in Berlin", user_id="alice")
        memory.wait()
        assert statuses(memory, "alice") == [("done", 2)]
        assert [f["memory"] for f in memory.get_all(user_id="alice")["results"]] == [
            "The user lives in Berlin."
        ]
        assert len(worker.errors) == 2 and "1 个会话抽取失败" in worker.errors[0]
        assert "Insufficient Balance" in worker.errors[0]

        down["left"] = 99  # 一直失败
        memory.add("I live in Paris", user_id="bob", timestamp=datetime(2026, 8, 1))
        memory.wait()
        assert statuses(memory, "bob") == [("failed", 3)]
        down["left"] = 0
        memory.add("I like jazz", user_id="bob", timestamp=datetime(2026, 9, 1))
        memory.wait()
        assert statuses(memory, "bob") == [("done", 0), ("failed", 3)]
        memory.consolidate(user_id="bob")  # 显式的巩固重试全部失败的任务
        assert [s for s, _ in statuses(memory, "bob")] == ["done", "done"]
        assert len(memory.get_all(user_id="bob")["results"]) == 2


def test_worker_retries_a_space_whose_consolidation_raised(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """后台巩固时对账（不是抽取）遇上余额不足：异常从 run() 抛出，任务保持待处理。后台线程按
    同样的退避时间把这个空间再做一遍，供应商恢复之后不用等新的写入就补上；一直抛出时最多做
    MAX_ATTEMPTS 次。"""
    model = ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    down = {"left": 0}

    def reply(body: dict[str, Any]) -> Any:
        if down["left"] and body["messages"][0]["content"].startswith(RECONCILE):
            down["left"] -= 1
            return NO_BALANCE
        return model(body)

    fake_api.chat_replies = [reply]
    conv = toy()
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        system.ingest(conv.model_copy(update={"sessions": conv.sessions[:1]}), consolidate=True)
        worker = system.start_worker()
        worker.retry_base_s = 0.2  # 退避缩短：第一次之后等 0.2 秒（缺省 1 分钟）
        down["left"] = 1  # 对账失败一次，之后恢复
        system.ingest(conv)  # 后面几个会话交给后台线程：抽取成功，对账时 402
        worker.wait()
        assert len(worker.errors) == 1 and "Insufficient Balance" in worker.errors[0]
        deadline = time.monotonic() + 10  # 没有新的提交：后台线程自己到时间再做
        while {j["status"] for j in system.db.get_jobs("demo")} != {"done"}:
            assert time.monotonic() < deadline, system.db.get_jobs("demo")
            time.sleep(0.05)
        assert len(worker.errors) == 1
        assert system.facts("demo", current_only=True)

        calls: list[str] = []

        def broken(space_id: str) -> None:
            calls.append(space_id)
            raise RuntimeError("boom")

        monkeypatch.setattr(system.consolidator, "run", broken)
        worker.retry_base_s = 0.0
        worker.submit("demo")
        worker.wait()  # 退避时间为 0：到期的重试在这一轮清出 _running 之前排上，wait() 一并等
        assert calls == ["demo"] * MAX_ATTEMPTS and worker._until_due() is None


def test_failed_jobs_back_off(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """自动重试的退避：第一次失败之后等 1 分钟，第二次之后等 4 分钟，失败 3 次就不再自动重试。"""
    down, reply = outage(ScriptedModel(where_i_live))
    fake_api.chat_replies = [reply]
    down["left"] = 99
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        memory.add("I live in Berlin", user_id="alice")
        consolidator = memory.system.consolidator
        for attempts, wait in ((1, 60), (2, 240)):
            assert consolidator.run("alice").failed
            assert statuses(memory, "alice") == [("failed", attempts)]
            left = consolidator.next_retry("alice")
            assert left is not None and wait - 5 < left <= wait
            now = datetime.now()
            assert consolidator.retry_due("alice", now=now + timedelta(seconds=wait - 5)) == 0
            assert consolidator.retry_due("alice", now=now + timedelta(seconds=wait + 1)) == 1
            assert statuses(memory, "alice") == [("pending", attempts)]
        assert consolidator.run("alice").failed
        assert consolidator.next_retry("alice") is None
        assert consolidator.retry_due("alice", now=datetime.now() + timedelta(days=1)) == 0


def test_worker_resumes_pending_jobs_on_start(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """进程退出时还没做完的抽取：下次启动后台线程时接着做，不用等这个用户再写入。"""
    fake_api.chat_replies = [ScriptedModel(where_i_live)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        assert memory.add("I live in Berlin", user_id="alice")["status"] == "queued"
        assert statuses(memory, "alice") == [("pending", 0)]
    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        memory.wait()
        assert statuses(memory, "alice") == [("done", 0)]
        assert [f["memory"] for f in memory.get_all(user_id="alice")["results"]] == [
            "The user lives in Berlin."
        ]


def test_close_timeout_does_not_record_closed_db_error(
    n3_settings: Settings, fake_api: FakeAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """关闭时后台线程还在抽取、等不到它做完：照常关闭，线程随后碰到已关闭的数据库时不记报错；
    任务留在待处理，下次启动时接着做。"""
    model = ScriptedModel(where_i_live)
    entered, release = threading.Event(), threading.Event()

    def reply(body: dict[str, Any]) -> Any:
        if body["messages"][0]["content"].startswith(EXTRACT) and not release.is_set():
            entered.set()
            release.wait(10)
        return model(body)

    fake_api.chat_replies = [reply]
    memory = Memory(n3_settings, http_client=fake_api.client())
    worker = memory.system.worker
    assert worker is not None
    memory.add("I live in Berlin", user_id="alice")
    assert entered.wait(10)
    worker.stop_timeout_s = 0.2
    with caplog.at_level(logging.WARNING, logger="cuemem.write.consolidate"):
        memory.close()
    assert worker.closing and "下次启动时接着做" in caplog.text
    release.set()
    for thread in worker._threads:
        thread.join(10)
    assert not any(t.is_alive() for t in worker._threads) and not worker.errors

    with Memory(n3_settings, http_client=fake_api.client()) as again:
        again.wait()
        assert statuses(again, "alice") == [("done", 0)]
        assert [f["memory"] for f in again.get_all(user_id="alice")["results"]] == [
            "The user lives in Berlin."
        ]


def test_one_slow_space_does_not_hold_back_another(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """一个用户的巩固很慢（几百个会话的重放、供应商限流）时，别的用户的抽取照常开始：后台有几个
    线程（consolidation.background_workers，缺省 4）。以前只有一个线程，别的用户都排在它后面
    （离线复现：600 个会话的重放期间，另一个用户一个会话的抽取等了近 20 秒，单独只要 0.01 秒）。"""
    model = ScriptedModel(echo_extraction)
    alice_in, bob_in = threading.Event(), threading.Event()

    def reply(body: dict[str, Any]) -> Any:
        text = str(body["messages"])
        if "ALICE" in text:
            alice_in.set()
            bob_in.wait(10)
        if "BOB" in text:
            bob_in.set()
        return model(body)

    fake_api.chat_replies = [reply]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        worker = system.start_worker()
        assert len(worker._threads) == n3_settings.consolidation.background_workers == 4
        system.append("alice", said_once("s1", "ALICE lives in Rome", 1), 0)
        assert alice_in.wait(10)  # alice 的抽取卡在供应商那里
        system.append("bob", said_once("s1", "BOB lives in Oslo", 1), 0)
        assert bob_in.wait(5), "bob 的抽取排在 alice 的后面"
        worker.wait()
        assert not worker.errors
        assert [f.statement for f in system.facts("bob")] == ["BOB lives in Oslo"]
        assert [f.statement for f in system.facts("alice")] == ["ALICE lives in Rome"]


def test_repeated_submits_of_one_space_are_merged(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """一个空间在队列里最多排一次：线程都忙时同一个用户连着写 30 次（每次都提交），轮到它时只
    巩固一轮。以前提交几次就巩固几轮，其余的什么都不做、各写一条 trace（离线复现：30 次后台的
    add 巩固了 30 轮，24 轮一个会话都没有）。"""
    model = ScriptedModel(echo_extraction)
    held, release = threading.Event(), threading.Event()

    def reply(body: dict[str, Any]) -> Any:
        if "HOLD" in str(body["messages"]):
            held.set()
            release.wait(10)
        return model(body)

    fake_api.chat_replies = [reply]
    one = n3_settings.consolidation.model_copy(update={"background_workers": 1})
    settings = n3_settings.model_copy(update={"consolidation": one})
    with MemorySystem.from_settings(settings, http_client=fake_api.client()) as system:
        worker = system.start_worker()
        assert len(worker._threads) == 1
        system.append("hold", said_once("s1", "HOLD the only thread", 1), 0)
        assert held.wait(10)
        system.append("dave", said_once("s1", "I bought a bike", 1), 0)
        for _ in range(29):
            worker.submit("dave")
        release.set()
        worker.wait()
        runs = [r for r in worker.reports if r.space_id == "dave"]
        assert [r.sessions for r in runs] == [1]


def test_a_space_submitted_while_it_consolidates_runs_again(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """正在巩固的空间又提交了（期间又写了一个会话）：这一轮做完再做一轮，新会话不会因为合并
    提交而漏掉，wait() 也等到它做完。"""
    model = ScriptedModel(echo_extraction)
    entered, release = threading.Event(), threading.Event()

    def reply(body: dict[str, Any]) -> Any:
        if "FIRST" in str(body["messages"]) and not release.is_set():
            entered.set()
            release.wait(10)
        return model(body)

    fake_api.chat_replies = [reply]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        worker = system.start_worker()
        system.append("erin", said_once("s1", "FIRST I live in Lima", 1), 0)
        assert entered.wait(10)
        system.append("erin", said_once("s2", "I work as a nurse", 2), 0)
        release.set()
        worker.wait()
        assert {j["status"] for j in system.db.get_jobs("erin")} == {"done"}
        assert sorted(f.statement for f in system.facts("erin")) == [
            "FIRST I live in Lima",
            "I work as a nurse",
        ]
        assert [r.sessions for r in worker.reports if r.space_id == "erin"] == [1, 1]


def said_once(session_id: str, text: str, day: int) -> SessionIn:
    return SessionIn(
        session_id=session_id,
        started_at=datetime(2026, 9, day, 10),
        turns=[TurnIn(role="user", text=text)],
    )


@pytest.mark.parametrize("rebuild", [True, False], ids=["rebuild", "plain"])
def test_consolidate_reads_open_sessions_once_it_holds_the_lock(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch, rebuild: bool
) -> None:
    """显式的巩固要等后台线程做完手上这一轮才拿到巩固锁。等锁期间结束的会话照样被抽取（重建时
    它刚登记的任务被清空，要重新登记）；等锁期间新开、已经说了几句的会话不被抽取半段。"""
    model = ScriptedModel(echo_extraction)
    entered, release, asked = threading.Event(), threading.Event(), threading.Event()

    def reply(body: dict[str, Any]) -> Any:
        if "SLOW" in str(body["messages"]) and not release.is_set():
            entered.set()
            release.wait(10)
        return model(body)

    fake_api.chat_replies = [reply]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        worker = system.start_worker()
        if rebuild:
            system.open_session("u", "x")
            system.append("u", said_once("x", "I live in Paris", 2), 0)  # 进行中：不登记
        system.append("u", said_once("slow", "SLOW extraction in flight", 1), 0)
        assert entered.wait(10)  # 后台线程拿着巩固锁，卡在抽取里

        exclusive = system.consolidator.exclusive

        def announced(space_id: str) -> Any:
            asked.set()
            return exclusive(space_id)

        monkeypatch.setattr(system.consolidator, "exclusive", announced)
        waiting = threading.Thread(target=lambda: system.consolidate("u", rebuild=rebuild))
        waiting.start()
        assert asked.wait(10)  # 显式的巩固在等锁
        if rebuild:
            system.close_session("u", "x", submit=False)  # 会话结束：登记任务
        else:
            system.open_session("u", "y")  # 新的一段对话开始，已经说了一句
            system.append("u", said_once("y", "I am planning a trip", 3), 0)
        release.set()
        waiting.join(10)
        worker.wait()
        assert not worker.errors
        jobs = {j["session_id"]: j["status"] for j in system.db.get_jobs("u")}
        facts = sorted(f.statement for f in system.facts("u"))
        if rebuild:
            assert jobs == {"slow": "done", "x": "done"}
            assert facts == ["I live in Paris", "SLOW extraction in flight"]
        else:
            assert jobs == {"slow": "done"} and facts == ["SLOW extraction in flight"]
            system.close_session("u", "y", wait=True)
            assert "I am planning a trip" in [f.statement for f in system.facts("u")]


def test_stale_enqueue_and_run_do_not_read_the_whole_space(
    system: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """判断派生记忆是否过期（add(wait=True) 与 update 每次都有）不再读出全部任务；为几个会话登记
    任务（每次 add）与执行任务不再读出全部会话。以前这几处都随空间里的会话数变慢：2 万个会话时
    读全部任务约 0.1 秒、读全部会话约 20 毫秒，都拿着数据库的锁。"""
    whole: list[str] = []
    get_jobs, get_sessions = MemoryDB.get_jobs, MemoryDB.get_sessions

    def jobs_spy(db: MemoryDB, space_id: str, status: str | None = None) -> Any:
        if status is None:
            whole.append(f"get_jobs({space_id})")
        return get_jobs(db, space_id, status)

    def sessions_spy(db: MemoryDB, space_id: str) -> Any:
        whole.append(f"get_sessions({space_id})")
        return get_sessions(db, space_id)

    monkeypatch.setattr(MemoryDB, "get_jobs", jobs_spy)
    monkeypatch.setattr(MemoryDB, "get_sessions", sessions_spy)
    consolidator = system.consolidator
    assert not consolidator.stale("demo")
    system.append("demo", said_once("s9", "I bought a bike", 9), 0, enqueue=False)
    assert consolidator.enqueue("demo", ["s9", "nope", "s9"]) == 1
    assert consolidator.enqueue("demo", ["s9"]) == 0
    assert consolidator.run("demo").sessions == 1
    system.db.set_meta("consolidation:empty", "another fingerprint")
    assert not consolidator.stale("empty")  # 没有任务的空间：没有要重建的
    system.db.set_meta("consolidation:demo", "another fingerprint")
    assert consolidator.stale("demo")
    system.db.set_meta("consolidation:demo", consolidator.fingerprint())
    assert not consolidator.stale("demo")
    system.db.enqueue_job("old", "demo", "s9", "an older pipeline", "h")
    assert consolidator.stale("demo")
    assert whole == []


def reference_enqueue(
    db: MemoryDB, space_id: str, session_ids: Sequence[str] | None, skip: set[str]
) -> int:
    """以前登记任务的做法：读出全部会话，逐个会话读发言、算指纹，各自 INSERT OR IGNORE，新登记
    时取代同一会话还没执行的旧任务。"""
    sessions = db.get_sessions(space_id)
    added = 0
    for session_id in session_ids if session_ids is not None else sorted(sessions):
        if session_id not in sessions or session_id in skip:
            continue
        turns = db.get_turns(space_id, session_id)
        if not turns:
            continue
        digest = session_content_hash(turns, sessions[session_id])
        job_id = stable_id(space_id, "consolidate", session_id, digest, PIPELINE_VERSION)
        now = datetime.now().isoformat()
        with db.transaction() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO jobs (id, space_id, session_id, version, content_hash,"
                " status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)",
                (job_id, space_id, session_id, PIPELINE_VERSION, digest, now, now),
            )
            if cursor.rowcount > 0:
                added += 1
                conn.execute(
                    "UPDATE jobs SET status = 'superseded', updated_at = ?"
                    " WHERE space_id = ? AND session_id = ? AND id != ?"
                    " AND status IN ('pending', 'failed')",
                    (now, space_id, session_id, job_id),
                )
    return added


@pytest.mark.parametrize("listed", [False, True], ids=["all", "listed"])
def test_batched_enqueue_matches_enqueueing_one_session_at_a_time(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path, listed: bool
) -> None:
    """登记任务改成只读列出的会话、一次取发言的版本、一个事务批量插入（MemoryDB.enqueue_jobs，
    任务多时先读出已有的任务 ID）之后，任务表（按 rowid 的顺序）与返回的个数与以前逐个会话登记
    时逐行相同：巩固过、没变的，巩固过又长了的，已经登记、还没执行的（没变的与又长了的），新的，
    只记了日志的（infer=False），没有发言的，重复列出的，库里没有的，以及跳过的会话。"""
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as system:
        db = system.db
        system.append("u", said_once("done", "I live in Paris", 1), 0)
        system.append("u", said_once("grown", "I like jazz", 2), 0)
        system.append("u", said_once("skipped", "I like tea", 3), 0)
        system.consolidate("u")
        system.append("u", said_once("pending", "I am planning a trip", 4), 0)
        bulk = [f"bulk-{i:03d}" for i in range(120)]  # 任务多于 100 个：先读出已有的任务 ID
        for i, sid in enumerate(bulk):
            started = datetime(2026, 10, 1) + timedelta(minutes=i)
            db.add_session("u", sid, started)
            db.add_turns(
                [
                    Turn(
                        id=f"{sid}-0",
                        space_id="u",
                        session_id=sid,
                        idx=0,
                        role="user",
                        text=f"note {i}",
                        ts=started,
                    )
                ]
            )
        # 已经登记的会话：同时记下它们的内容指纹（之后的登记先查它，发言变了才重算）
        system.consolidator.enqueue("u", ["done", "grown", "skipped", "pending", *bulk[::2]])
        for sid, text, day in (("grown", "and blues", 2), ("skipped", "and coffee", 3)):
            system.append("u", said_once(sid, text, day), 1, enqueue=False)
        system.append("u", said_once("pending", "to Rome", 4), 1, enqueue=False)
        system.append("u", said_once("new", "I bought a bike", 5), 0, enqueue=False)
        system.append("u", said_once("logged", "I only log this", 6), 0, enqueue=False)
        db.add_session("u", "empty", datetime(2026, 9, 7, 10))
        copy = tmp_path / "copy.sqlite"
        with sqlite3.connect(copy) as target:
            db._conn.backup(target)
        target.close()

        ids = None
        if listed:
            ids = ["new", "grown", "new", "missing", "skipped", "empty", "done", "pending"]
            ids += bulk[::-1]
        skip = {"skipped"}
        added = system.consolidator.enqueue("u", ids, skip=skip)
        reference = MemoryDB(copy)
        try:
            expected = reference_enqueue(reference, "u", ids, skip)
            rows = "SELECT id, session_id, status, content_hash FROM jobs ORDER BY rowid"
            assert db._conn.execute(rows).fetchall() == reference._conn.execute(rows).fetchall()
        finally:
            reference.close()
        assert added == expected == (63 if listed else 64)


def test_content_filter_skips_session_or_falls_back_to_rules(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """内容审核误拒：抽取被两家都拒的会话跳过（不算失败），灰区对账被拒时退回"按新增处理"。"""
    from fakes import REJECT

    scripted = ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))

    def reply(body: dict) -> str:  # type: ignore[type-arg]
        prompt = body["messages"][-1]["content"]
        if "Conversation date: 2025-06-20" in prompt or prompt.startswith("New fact"):
            return REJECT
        return scripted(body)

    fake_api.chat_replies = [reply]
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as memory:
        memory.ingest(toy())
        report = memory.consolidate("demo", strict=True)  # 跳过的会话不让严格模式失败
        assert report.skipped == ["s3"] and not report.failed
        assert report.rules["add:llm_unavailable"] == 3  # 工作、职位、预算都没能裁决
        facts = memory.facts("demo")
        assert not any("巴厘岛" in f.statement for f in facts)
        assert by_statement(facts, "8000").is_current  # 没能裁决，保守地都保留为当前
        assert by_statement(facts, "6000").is_current
        jobs = {j["session_id"]: j for j in memory.db.get_jobs("demo")}
        assert jobs["s3"]["status"] == "done" and jobs["s3"]["stats"]["skipped"] == "content_filter"
        models = [r["model"] for r in fake_api.chat_requests]
        assert "deepseek-flash" in models  # 被拒后确实尝试了另一家


def test_config_change_triggers_rebuild(
    n3_settings: Settings, fake_api: FakeAPI, model: ScriptedModel
) -> None:
    """影响巩固结果的配置（如阈值）变了：派生记忆自动重建；只改并发数（同一空间里并行抽取的
    会话数、后台同时巩固的空间数）不算。"""
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as memory:
        memory.ingest(toy(), consolidate=True)
        before = memory.consolidator.fingerprint()
        assert not memory.consolidator.stale("demo")
    faster = n3_settings.consolidation.model_copy(update={"workers": 7, "background_workers": 2})
    with MemorySystem.from_settings(
        n3_settings.model_copy(update={"consolidation": faster}), http_client=fake_api.client()
    ) as memory:
        assert memory.consolidator.fingerprint() == before
        assert not memory.consolidator.stale("demo")
    changed = n3_settings.consolidation.model_copy(update={"profile_min_importance": 5})
    with MemorySystem.from_settings(
        n3_settings.model_copy(update={"consolidation": changed}), http_client=fake_api.client()
    ) as memory:
        assert memory.consolidator.stale("demo")
        report = memory.consolidate("demo")  # 自动重建：5 个会话全部重放（命中缓存）
        assert report.sessions == 5 and not memory.consolidator.stale("demo")


HIKING = [{"role": "user", "content": "I like hiking"}, {"role": "assistant", "content": "Nice!"}]
ALPS = [
    {"role": "user", "content": "Mostly in the Alps"},
    {"role": "assistant", "content": "Cool."},
]


def statements(memory: Memory, space: str = "u") -> list[str]:
    return sorted(f.statement for f in memory.system.facts(space))


def test_a_session_that_grows_is_replayed_from_the_log(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """巩固过的会话又追加了发言（SDK 每轮 add 同一个 run_id）：整段重新抽取之后，之前那次抽取的
    说法（"The user likes hiking."）不再留着，派生记忆与从原始日志重建的一样。重放只重新抽取
    长了的会话，其余会话命中缓存。"""
    model = ScriptedModel(growing_extraction)
    fake_api.chat_replies = [model]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        add = memory.add
        add("I live in Berlin", user_id="u", run_id="before", timestamp=datetime(2026, 7, 1))
        add(HIKING, user_id="u", run_id="chat", timestamp=datetime(2026, 8, 1), wait=True)
        add("I like jazz", user_id="u", run_id="after", timestamp=datetime(2026, 9, 1), wait=True)
        assert "The user likes hiking." in statements(memory)
        requests = len(fake_api.chat_requests)

        add(ALPS, user_id="u", run_id="chat", timestamp=datetime(2026, 8, 1, 0, 5), wait=True)
        assert len(fake_api.chat_requests) == requests + 1  # 只有长了的会话重新抽取
        assert "Mostly in the Alps" in model.prompts["extract"][-1]
        grown = statements(memory)
        assert grown == [
            "The user enjoys hiking in the Alps.",
            "The user said: I like jazz",
            "The user said: I live in Berlin",
            "The user said: Mostly in the Alps",
        ]
        assert [j["status"] for j in memory.system.db.get_jobs("u")] == ["done"] * 3
        traces = memory.system.settings.paths.traces.glob("*.jsonl")
        assert any("consolidate_regrown_replay" in p.read_text("utf-8") for p in traces)

        memory.consolidate(user_id="u", rebuild=True)
        assert statements(memory) == grown


def test_a_grown_session_is_not_replayed_without_a_writable_cache(
    n3_settings: Settings, fake_api: FakeAPI, caplog: pytest.LogCaptureFixture
) -> None:
    """LLM 缓存不可写时重放会把所有会话重新抽取一遍：照旧增量巩固，并给出警告。"""
    fake_api.chat_replies = [ScriptedModel(growing_extraction)]
    settings = n3_settings.model_copy(update={"llm_cache_mode": "off"})
    with Memory(settings, http_client=fake_api.client(), background=False) as memory:
        memory.add(HIKING, user_id="u", run_id="chat", wait=True)
        with caplog.at_level(logging.WARNING, logger="cuemem.write.consolidate"):
            memory.add(ALPS, user_id="u", run_id="chat", wait=True)
        assert "不从原始日志重放" in caplog.text
        assert "The user likes hiking." in statements(memory)


def test_a_replay_extracts_only_what_was_extracted_before(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """会话又长了时的重放只重新抽取原来就有任务的会话：infer=False 只记了日志的会话照旧不抽取；
    内容审核拒绝过、内容没变的会话不再请求（拒绝不进缓存）；失败次数用完的会话不因重放清零、
    重新开始自动重试。显式的 consolidate 照旧抽取 infer=False 的会话。"""
    from fakes import REJECT

    model = ScriptedModel(echo_extraction)
    sent: Counter[str] = Counter()

    def reply(body: dict[str, Any]) -> Any:
        text = str(body["messages"])
        sent.update(mark for mark in ("DIARY", "FILTERED", "BROKEN") if mark in text)
        if "FILTERED" in text:
            return REJECT
        if "BROKEN" in text and body["messages"][0]["content"].startswith(EXTRACT):
            return NO_BALANCE
        return model(body)

    fake_api.chat_replies = [reply]
    t0 = datetime(2026, 9, 1, 10)
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        memory.add("DIARY raw log only", user_id="u", infer=False, timestamp=t0)
        memory.add("FILTERED by moderation", user_id="u", timestamp=t0 + timedelta(hours=1))
        memory.add("BROKEN extraction", user_id="u", timestamp=t0 + timedelta(hours=2))
        memory.add("I like tea", user_id="u", run_id="chat", timestamp=t0 + timedelta(hours=3))
        consolidator = memory.system.consolidator
        consolidator.run("u")  # 后台线程的做法：只执行待处理的任务
        for _ in range(MAX_ATTEMPTS - 1):  # 退避时间到了就重试，直到用完次数
            assert consolidator.retry_due("u", base_s=0) == 1
            consolidator.run("u")

        def outcomes() -> dict[str, tuple[str, int, str | None]]:
            jobs = memory.system.db.get_jobs("u")
            return {
                memory.system.db.get_turns("u", j["session_id"])[0].text.split()[0]: (
                    j["status"],
                    j["attempts"],
                    j["stats"].get("skipped"),
                )
                for j in jobs
                if j["status"] != "superseded"
            }

        before = outcomes()
        assert before == {
            "FILTERED": ("done", 0, "content_filter"),
            "BROKEN": ("failed", MAX_ATTEMPTS, None),
            "I": ("done", 0, None),
        }
        requests = dict(sent)

        memory.add("I like jazz", user_id="u", run_id="chat", timestamp=t0 + timedelta(hours=4))
        assert consolidator.retry_due("u", base_s=0) == 0
        with start_trace("consolidate", memory.system.tracer, space="u"):
            report = consolidator.run("u")  # chat 又长了：从原始日志重放
        assert report.sessions == 1 and not report.failed
        assert outcomes() == before
        assert dict(sent) == requests  # 没有新的抽取请求发给这三个会话
        assert statements(memory) == ["I like jazz", "I like tea"]
        traces = memory.system.settings.paths.traces.glob("*.jsonl")
        assert any("consolidate_regrown_replay" in p.read_text("utf-8") for p in traces)

        memory.consolidate(user_id="u")  # 显式的巩固：infer=False 的会话这时才抽取
        assert "DIARY raw log only" in statements(memory)


def test_a_replay_survives_a_vector_store_error(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """会话又长了时的重放，删派生记忆的向量时向量库出错（Qdrant 服务一时连不上）：任务都还在，
    后台线程的重试把重放做完。以前先提交清空（任务也一起清空了）再删向量，出错时任务一个不剩，
    重试与重启都找不到要做的事，整个空间的事实都没了。"""
    fake_api.chat_replies = [ScriptedModel(where_i_live, reconcile=moved)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        system = memory.system
        memory.add("I live in Berlin", user_id="u", run_id="a", timestamp=AUG, wait=True)
        day2 = AUG + timedelta(days=1)
        memory.add("I like jazz", user_id="u", run_id="b", timestamp=day2, wait=True)
        memory.add("I like chess", user_id="u", run_id="b", timestamp=SEP)  # b 又长了，待处理
        delete, calls = system.vectors.delete, Counter[str]()

        def unreachable_once(space: str, ids: Sequence[str]) -> Any:
            calls["delete"] += 1
            if calls["delete"] == 1:
                raise ConnectionError("qdrant unreachable")
            return delete(space, ids)

        monkeypatch.setattr(system.vectors, "delete", unreachable_once)
        with pytest.raises(ConnectionError):
            system.consolidator.run("u")  # 后台线程的做法
        assert len(statements(memory)) == 2  # 什么都没清空
        assert system.db.spaces_with_work(PIPELINE_VERSION, MAX_ATTEMPTS) == ["u"]
        system.consolidator.run("u")  # 后台线程按退避重试
        assert statements(memory) == ["I like chess", "I like jazz", "The user lives in Berlin."]
        assert set(system.vectors.unit_ids("u")) == {u.id for u in system.db.list_units("u")}


def committed(db: MemoryDB, space: str) -> tuple[list[tuple[str, str]], list[str]]:
    """另开一个连接读到的任务与事实：只有已经提交的。进程这时被杀掉，重启之后看到的就是这些。"""
    assert db.path is not None
    other = sqlite3.connect(db.path)
    try:
        jobs = other.execute(
            "SELECT session_id, status FROM jobs WHERE space_id = ? ORDER BY session_id", (space,)
        ).fetchall()
        facts = other.execute(
            "SELECT statement FROM facts WHERE space_id = ? ORDER BY statement", (space,)
        ).fetchall()
    finally:
        other.close()
    return [(s, status) for s, status in jobs], [f for (f,) in facts]


@pytest.mark.parametrize("replay", [True, False], ids=["regrown-replay", "rebuild"])
def test_a_rebuild_clears_and_requeues_in_one_transaction(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch, replay: bool
) -> None:
    """重建（会话又长了时的重放，与 consolidate(rebuild=True)）的清空与重新登记在一个事务里：
    重新登记时出错（另一个进程拿着数据库，"database is locked"）或进程被杀，事实与任务都还在，
    后台线程照样能把重放做完。以前清空单独提交、之后才重新登记：两者之间空间既没有事实也没有
    任务，后台线程的重试与重启都找不到要做的事，事实要等这个用户再写一次才回来（第五轮体检）。"""
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        system = memory.system
        memory.add("I live in Paris", user_id="u", run_id="s1", timestamp=AUG, wait=True)
        day2 = AUG + timedelta(days=1)
        memory.add("I play violin", user_id="u", run_id="s2", timestamp=day2, wait=True)
        if replay:
            memory.add("I also play piano", user_id="u", run_id="s1", timestamp=SEP)  # s1 又长了
        jobs = sorted((j["session_id"], j["status"]) for j in system.db.get_jobs("u"))
        facts = statements(memory)
        assert facts == ["I live in Paris", "I play violin"]
        requeue, seen = system.consolidator.requeue, []

        def killed_once(space: str, before: Any, skip: Any) -> int:
            if not seen:
                seen.append(committed(system.db, space))  # 进程在这里被杀掉时留下的
                raise sqlite3.OperationalError("database is locked")
            return requeue(space, before, skip)

        monkeypatch.setattr(system.consolidator, "requeue", killed_once)
        with pytest.raises(sqlite3.OperationalError):
            if replay:
                system.consolidator.run("u")  # 后台线程的做法
            else:
                system.consolidate("u", rebuild=True)
        assert seen == [(jobs, facts)]  # 清空没有单独提交
        assert sorted((j["session_id"], j["status"]) for j in system.db.get_jobs("u")) == jobs
        assert statements(memory) == facts
        # 清空重建出错之后：要么向量与数据库一致，要么看得出要重建（第五轮复审：派生记忆的向量
        # 删了、事务却回滚时，以前 stale() 是 False，事实列得出来、检索却永远找不到）。重放靠留着
        # 的那对任务（下面的 spaces_with_work）
        units = {u.id for u in system.db.list_units("u")}
        if not replay:
            assert set(system.vectors.unit_ids("u")) == units or system.consolidator.stale("u")
        if replay:
            assert system.db.spaces_with_work(PIPELINE_VERSION, MAX_ATTEMPTS) == ["u"]
            system.consolidator.run("u")  # 后台线程按退避重试
            assert statements(memory) == ["I also play piano", "I live in Paris", "I play violin"]
        else:
            # serve、代理与 MCP 的写法：不等巩固，后台线程接着 run。它要先把没做完的重建做完
            # （第六轮体检：以前 run 最后照样写这次的指纹，标记被抹掉，缺的向量再没人补）
            memory.add("I like tea", user_id="u", run_id="s3", timestamp=SEP)
            system.consolidator.run("u")
            assert statements(memory) == ["I like tea", *facts]
            assert not system.consolidator.stale("u")
            hits = memory.search("Paris", user_id="u", mode="l1")["results"]
            assert "I live in Paris" in [h["memory"] for h in hits]
        assert set(system.vectors.unit_ids("u")) == {u.id for u in system.db.list_units("u")}


def test_a_worker_run_without_jobs_finishes_an_interrupted_rebuild(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """清空重建出错之后，没有新任务的一次后台 run 也把重建做完（第六轮体检）：以前标记留着、
    向量缺着，要等显式的 consolidate。"""
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        system = memory.system
        memory.add("I live in Paris", user_id="u", run_id="s1", timestamp=AUG, wait=True)
        requeue = system.consolidator.requeue

        def locked(space: str, before: Any, skip: Any) -> int:
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(system.consolidator, "requeue", locked)
        with pytest.raises(sqlite3.OperationalError):
            system.consolidate("u", rebuild=True)
        monkeypatch.setattr(system.consolidator, "requeue", requeue)
        assert system.consolidator.stale("u")
        system.consolidator.run("u")
        assert not system.consolidator.stale("u")
        assert statements(memory) == ["I live in Paris"]
        assert set(system.vectors.unit_ids("u")) == {u.id for u in system.db.list_units("u")}


def _interrupt_rebuild(system: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """consolidate(rebuild=True) 在重新登记时出错：标记 REBUILD_PENDING 留在库里。"""
    requeue = system.consolidator.requeue

    def locked(space: str, before: Any, skip: Any) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(system.consolidator, "requeue", locked)
    with pytest.raises(sqlite3.OperationalError):
        system.consolidate("u", rebuild=True)
    monkeypatch.setattr(system.consolidator, "requeue", requeue)
    assert system.db.get_meta("consolidation:u") == REBUILD_PENDING


def test_a_resumed_rebuild_drops_the_marker_once_it_commits(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """接着做的重建一提交，标记就删了（第六轮复审）：之后的重放途中进程被杀，下一次不会从头
    再重建一遍。以前事务里把标记原样写回，要等 _run 做完整个空间才换掉。"""
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        system = memory.system
        memory.add("I live in Paris", user_id="u", run_id="s1", timestamp=AUG, wait=True)
        _interrupt_rebuild(system, monkeypatch)
        replay = system.consolidator._replay_regrown

        def killed(space: str) -> None:
            raise KeyboardInterrupt  # 进程在接着做的重建提交之后被杀

        monkeypatch.setattr(system.consolidator, "_replay_regrown", killed)
        with pytest.raises(KeyboardInterrupt):
            system.consolidator.run("u")
        assert system.db.get_meta("consolidation:u") != REBUILD_PENDING
        assert system.db.get_jobs("u", status="pending")  # 剩下的事在任务里
        monkeypatch.setattr(system.consolidator, "_replay_regrown", replay)
        system.consolidator.run("u")
        assert not system.consolidator.stale("u")
        assert statements(memory) == ["I live in Paris"]
        assert set(system.vectors.unit_ids("u")) == {u.id for u in system.db.list_units("u")}


def test_consolidate_after_an_interrupted_rebuild_rebuilds_once(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中断之后的 consolidate() 只重建一次（第六轮复审）：以前 stale() 重建一次、写回标记，
    run 看见标记又重建一次。"""
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        system = memory.system
        memory.add("I live in Paris", user_id="u", run_id="s1", timestamp=AUG, wait=True)
        _interrupt_rebuild(system, monkeypatch)
        rebuild, calls = system.consolidator.rebuild, []

        def counted(space: str, *, replay: bool = False) -> None:
            calls.append(replay)
            rebuild(space, replay=replay)

        monkeypatch.setattr(system.consolidator, "rebuild", counted)
        system.consolidate("u")
        assert calls == [False]
        assert not system.consolidator.stale("u")
        assert statements(memory) == ["I live in Paris"]


def said(
    statement: str, subject: str, predicate: str, obj: str, turn: int, cardinality: str = "single"
) -> dict[str, Any]:
    return {
        "statement": statement,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "category": "attribute",
        "cardinality": cardinality,
        "importance": 4,
        "turns": [turn],
    }


# 抽取提示要求用英文写记忆，名字与专名保留原来的文字
NATIVE_NAMES: dict[str, Any] = {
    "2026-01-10": {
        "summary": "",
        "facts": [
            said("Анна lives in Moscow.", "Анна", "lives_in", "Moscow", 0),
            said("Иван works as a doctor.", "Иван", "job_title", "doctor", 1),
        ],
        "entities": [],
    },
    "2026-02-01": {
        "summary": "",
        "facts": [
            said("The user has a dog named 초코.", "User", "has_pet", "초코", 0, "multi"),
            said("The user has a dog named 보리.", "User", "has_pet", "보리", 0, "multi"),
            said("The user lives in 서울.", "User", "lives_in", "서울", 0),
        ],
        "entities": [],
    },
    "2026-03-01": {
        "summary": "",
        "facts": [said("The user moved to 부산.", "User", "lives_in", "부산", 0)],
        "entities": [],
    },
}


def test_names_and_values_in_other_scripts(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """名字与宾语是西里尔、韩文等文字时（以前规范化成 ''）：两位俄文名字的说话人是两个实体，
    关于他们的事实都留着；同一会话里两只韩文名字的狗都留着；从 서울 搬到 부산 经过对账，
    서울 不再是当前的（以前当成同一个值，부산 被当作重复丢掉）。"""
    model = ScriptedModel(
        NATIVE_NAMES,
        reconcile=lambda prompt: {"decisions": [{"new": 1, "action": "update", "target": "e1"}]},
    )
    fake_api.chat_replies = [model]
    speakers = SessionIn(
        session_id="g",
        started_at=datetime(2026, 1, 10),
        turns=[
            TurnIn(role="user", speaker="Анна", text="Я живу в Москве."),
            TurnIn(role="user", speaker="Иван", text="А я работаю врачом."),
        ],
    )
    pets = SessionIn(
        session_id="a",
        started_at=datetime(2026, 2, 1),
        turns=[TurnIn(role="user", text="강아지 초코랑 보리를 키워요. 서울에 살아요.")],
    )
    moved = SessionIn(
        session_id="b",
        started_at=datetime(2026, 3, 1),
        turns=[TurnIn(role="user", text="부산으로 이사했어요.")],
    )
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as memory:
        memory.ingest(Conversation(space_id="ru", sessions=[speakers]), consolidate=True)
        people = {e.name: e.id for e in memory.entities("ru")}
        assert set(people) == {"Анна", "Иван"}
        facts = memory.facts("ru")
        assert by_statement(facts, "Moscow").subject_id == people["Анна"]
        assert by_statement(facts, "doctor").subject_id == people["Иван"]

        memory.ingest(Conversation(space_id="ko", sessions=[pets, moved]), consolidate=True)
        facts = memory.facts("ko")
        assert {f.object for f in facts if f.predicate == "has_pet"} == {"초코", "보리"}
        seoul, busan = by_statement(facts, "서울"), by_statement(facts, "부산")
        assert busan.is_current and not seoul.is_current
        assert seoul.superseded_by == busan.id
        assert model.calls["reconcile"] == 1 and "서울" in model.prompts["reconcile"][0]


ACCENTS: dict[str, Any] = {
    "2026-01-10": {
        "summary": "",
        "facts": [
            said("The user's budget is ８０００ yuan.", "User", "budget", "８０００ yuan", 0),
            said("Müller likes chess.", "Müller", "likes", "chess", 0, "multi"),
            said("Möller likes tennis.", "Möller", "likes", "tennis", 1, "multi"),
        ],
        "entities": [],
    },
    "2026-02-01": {
        "summary": "",
        "facts": [
            said("The user's budget is ６０００ yuan.", "User", "budget", "６０００ yuan", 0),
            said("The user has a friend named Zoë.", "User", "has_friend", "Zoë", 0, "multi"),
            said("The user has a friend named Zoé.", "User", "has_friend", "Zoé", 0, "multi"),
            said("The user has a friend named Zoà.", "User", "has_friend", "Zoà", 0, "multi"),
        ],
        "entities": [],
    },
}


def test_accented_letters_and_fullwidth_digits(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """原来的键删掉带重音的拉丁字母与全角数字：'Müller' 与 'Möller' 两位说话人成了一个实体，
    预算从 ８０００ 改成 ６０００ 被当成重复丢掉，同一会话里 'Zoë' 与 'Zoà' 两句的事实 ID 相同、
    后一句被丢掉。现在都分开；折叠重音后相同的 'Zoé' 仍与 'Zoë' 是同一个值。"""
    model = ScriptedModel(
        ACCENTS,
        reconcile=lambda prompt: {"decisions": [{"new": 1, "action": "update", "target": "e1"}]},
    )
    fake_api.chat_replies = [model]
    first = SessionIn(
        session_id="a",
        started_at=datetime(2026, 1, 10),
        turns=[
            TurnIn(
                role="user", speaker="Müller", text="Mein Budget: ８０００ yuan. Ich mag Schach."
            ),
            TurnIn(role="user", speaker="Möller", text="Ich spiele Tennis."),
        ],
    )
    second = SessionIn(
        session_id="b",
        started_at=datetime(2026, 2, 1),
        turns=[TurnIn(role="user", text="Budget now ６０００ yuan. Friends: Zoë, Zoé, Zoà.")],
    )
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as memory:
        memory.ingest(Conversation(space_id="de", sessions=[first, second]), consolidate=True)
        people = {e.name: e.id for e in memory.entities("de") if e.type == "person"}
        assert {"Müller", "Möller"} <= set(people) and people["Müller"] != people["Möller"]
        assert people["Müller"] == stable_id("de", "entity", "m ller", "person")  # 原来的 ID
        facts = memory.facts("de")
        assert by_statement(facts, "chess").subject_id == people["Müller"]
        assert by_statement(facts, "tennis").subject_id == people["Möller"]
        old, new = by_statement(facts, "８０００"), by_statement(facts, "６０００")
        assert new.is_current and old.superseded_by == new.id
        assert model.calls["reconcile"] == 1
        friends = [f for f in facts if f.predicate == "has_friend" and f.is_current]
        assert sorted(f.object for f in friends) == ["Zoà", "Zoë"]
        zoe = by_statement(facts, "Zoë")
        assert zoe.id == stable_id("de", "fact", "b", "user has a friend named zo")  # 原来的 ID


# ---------- 跨巩固复用载入的状态（R6） ----------


def life_story(prompt: str) -> dict[str, Any]:
    """抽取替身：住址、喜好、最喜欢的颜色、家人与宠物各抽成一条事实；提到的宠物与人记成实体，
    宠物的描述后来才说（实体的描述变了，描述向量要重新查）；喜好的陈述两边带空白（库里存的是
    去掉空白的，载入时查的是另一段文本）。"""
    facts: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []

    def fact(statement: str, predicate: str, obj: str, cardinality: str, turn: str) -> None:
        subject = statement.split()[0] if predicate == "is_a" else "User"
        facts.append(
            {
                "statement": statement,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "cardinality": cardinality,
                "importance": 4,
                "turns": [int(turn)],
            }
        )

    for n, text in re.findall(r"\[t(\d+)\] User: (.*)", prompt):
        if m := re.search(r"(?:live in|moved to) (\w+)", text):
            fact(f"The user lives in {m[1]}.", "lives_in", m[1], "single", n)
        elif m := re.search(r"like (\w+)", text):
            fact(f" The user likes {m[1]}. ", "likes", m[1], "multi", n)
        elif m := re.search(r"favorite color is (\w+)", text):
            fact(f"The user's favorite color is {m[1]}.", "favorite_color", m[1], "single", n)
        elif m := re.search(r"(\w+) is my (dog|sister)", text):
            pet = m[2] == "dog"
            fact(
                f"{m[1]} is the user's {m[2]}.",
                "has_pet" if pet else "has_family_member",
                m[1],
                "multi",
                n,
            )
            kind = "pet" if pet else "person"
            entities.append({"name": m[1], "type": kind, "aliases": [], "description": ""})
        elif m := re.search(r"(\w+) is an? (.*)", text):
            fact(f"{m[1]} is a {m[2]}", "is_a", m[2], "single", n)
            # 只说了是什么，没说是宠物：类型记成 other，链接到已有的宠物时补上描述
            entities.append({"name": m[1], "type": "other", "aliases": [], "description": m[2]})
    return {"summary": "", "facts": facts, "entities": entities}


@pytest.fixture
def story_settings(offline_settings: Settings) -> Settings:
    """哈希向量有字面相似度：阈值放低，让对账与消歧的灰区真的要问模型；抽取逐个会话做，
    请求的先后确定。"""
    consolidation = offline_settings.consolidation.model_copy(
        update={
            "duplicate_threshold": 0.95,
            "conflict_threshold": 0.4,
            "multi_conflict_threshold": 0.8,
            "entity_merge_threshold": 0.95,
            "entity_gray_threshold": 0.3,
            "workers": 1,
        }
    )
    return offline_settings.model_copy(update={"consolidation": consolidation})


def story(settings: Settings, fake_api: FakeAPI, root: Path) -> dict[str, Any]:
    """SDK 的一串写入、遗忘、导入、重建与删除空间，返回发给模型的请求、交给向量器的文本
    （缓存未命中）、add 返回的事件、最后的事实与实体，以及从库里整个载入状态的次数。"""
    fake_api.chat_replies = [
        ScriptedModel(
            life_story,
            reconcile=reconcile_by_keyword("lives in", "favorite color"),
            entity=lambda prompt: {"matches": [{"new": 1, "same_as": "c1"}]},
        )
    ]
    fake_api.chat_requests.clear()
    paths = PathsConfig().resolve_against(root)
    settings = settings.model_copy(update={"paths": paths})
    out: dict[str, Any] = {"events": [], "misses": [], "loads": 0}
    with Memory(settings, http_client=fake_api.client(), background=False) as m:
        db = m.system.db
        assert isinstance(m.system.embedder, CachedEmbedder)
        inner = m.system.embedder.inner
        embed, get_entities = inner.embed, db.get_entities

        def recording(texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
            out["misses"].extend(f"{kind}:{t}" for t in texts)
            return embed(texts, kind)

        def counting(space_id: str, ids: Sequence[str] | None = None) -> list[Entity]:
            out["loads"] += ids is None
            return get_entities(space_id, ids)

        inner.embed = recording  # type: ignore[method-assign]
        db.get_entities = counting  # type: ignore[method-assign]
        day = itertools.count()

        def add(text: str, run: str, user: str = "u", **kw: Any) -> dict[str, Any]:
            when = datetime(2026, 1, 1) + timedelta(days=next(day))
            result = m.add(text, user_id=user, run_id=run, timestamp=when, **kw)
            out["events"].append(result["results"])
            return result

        add("I live in Berlin", "s1", wait=True)
        add("I like hiking", "s2", wait=True)
        add("Max is my dog", "s3", wait=True)
        add("My favorite color is blue", "s4", wait=True)
        add("Max is a golden retriever", "s5", wait=True)  # 描述填上了：描述向量要重新查
        add("Rex is my dog", "s5b", wait=True)  # 实体的灰区：与 Max 的描述有几分像
        add("I moved to Munich", "s6", wait=True)  # 灰区：取代柏林
        add("I like chess", "s7", wait=True)
        add("My favorite color is green", "s7", wait=True)  # 会话又长了：清空重放
        add("I like painting", "s8", infer=False)
        out["events"].append(m.consolidate(user_id="u"))
        chess = next(f for f in m.get_all(user_id="u")["results"] if "chess" in f["memory"])
        m.delete(chess["id"])  # 遗忘：清空重放
        add("I like swimming", "s9", wait=True)
        add("Mia is my sister", "s10", wait=True)
        m.load(m.export(user_id="u"), user_id="v", wait=True)  # 导入到另一个空间
        add("I moved to Paris", "s11", wait=True)
        add("I live in Rome", "v1", user="v", wait=True)
        m.delete_all(user_id="u", run_id="s2")  # 按会话遗忘
        out["events"].append(m.consolidate(user_id="u", rebuild=True))
        add("I like rowing", "s12", wait=True)
        m.delete_all(user_id="v")  # 删除空间之后同一个空间重新写入
        add("I live in Oslo", "v2", user="v", wait=True)
        add("Bella is my dog", "v3", user="v", wait=True)
        for user in ("u", "v"):
            space = space_for(user)
            out[f"facts:{user}"] = [f.model_dump() for f in db.get_facts(space)]
            out[f"entities:{user}"] = [e.model_dump() for e in get_entities(space)]
        out["units"] = sorted((u.id, u.kind, u.text) for u in db.list_units(space_for("u")))
    for report in out["events"]:
        if isinstance(report, dict):
            report.pop("elapsed_ms", None)
    out["requests"] = [json.dumps(r, sort_keys=True) for r in fake_api.chat_requests]
    return out


def test_reused_state_matches_loading_from_the_database(
    story_settings: Settings,
    fake_api: FakeAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """以前每次巩固都从库里读全部事实与实体、逐条查向量缓存；现在上一次巩固之后留下镜像，
    派生记忆没变过就直接用。经过遗忘、会话又长了的重放、导入、重建与删除空间，发给模型的请求、
    向量缓存的未命中、add 的事件与最后的事实和实体，与每次都从库里读完全相同。"""
    # 只撤销这一处：以前 monkeypatch.undo() 把这个测试的环境变量也全撤了（settings 夹具设的假 Key、
    # clean_env 指到临时目录的 .env），第二遍只在环境里本来就有真 Key 时才跑得通（CI 没有 Key：
    # 抽取失败，找不到 chess 那条事实）
    with monkeypatch.context() as patch:
        patch.setattr(consolidate_module, "MIRRORED_SPACES", 0)  # 不留镜像：每次都从库里读
        cold = story(story_settings, fake_api, tmp_path / "cold")
    warm = story(story_settings, fake_api, tmp_path / "warm")
    assert warm["requests"] == cold["requests"]
    assert warm["misses"] == cold["misses"]
    assert warm["events"] == cold["events"]
    for key in ("facts:u", "entities:u", "facts:v", "entities:v", "units"):
        assert warm[key] == cold[key], key
    # 场景确实走到了要问模型的灰区与要重新查的向量，镜像也确实用上了
    assert sum(RECONCILE in r for r in cold["requests"]) >= 2
    assert any("You link entity mentions" in r for r in cold["requests"])
    assert "document:Max (pet): golden retriever" in cold["misses"]  # 载入时才查的描述
    assert any(f["supersede_reason"] == "changed" for f in cold["facts:u"])
    assert warm["loads"] < cold["loads"] // 2


def corrections_story(settings: Settings, fake_api: FakeAPI, root: Path) -> dict[str, Any]:
    """SDK 的 metadata、update（记进原始日志的更正）、带 metadata 又长了的会话、导出导入到另一个
    空间（version 2，带更正目标），最后遗忘那句更正。返回发给模型的请求、各步的结果、更正之后与
    最后的事实和实体，以及从库里整个载入状态的次数。"""
    fake_api.chat_replies = [
        ScriptedModel(life_story, reconcile=reconcile_by_keyword("lives in", "favorite color"))
    ]
    fake_api.chat_requests.clear()
    paths = PathsConfig().resolve_against(root)
    out: dict[str, Any] = {"events": [], "loads": 0}
    with Memory(
        settings.model_copy(update={"paths": paths}),
        http_client=fake_api.client(),
        background=False,
    ) as m:
        db = m.system.db
        get_entities = db.get_entities

        def counting(space_id: str, ids: Sequence[str] | None = None) -> list[Entity]:
            out["loads"] += ids is None
            return get_entities(space_id, ids)

        db.get_entities = counting  # type: ignore[method-assign]
        day = itertools.count()

        def add(text: str, run: str, **kw: Any) -> None:
            # 写在将来：update 给更正的会话取的时间是原来那条之后一点，不是当前时间，两次相同
            when = datetime(2030, 1, 1) + timedelta(days=next(day))
            result = m.add(text, user_id="u", run_id=run, timestamp=when, wait=True, **kw)
            out["events"].append(result["results"])

        add("I live in Berlin", "s1", metadata={"source": "crm"})
        add("I like swimming", "s2", metadata={"source": "chat"})
        add("Max is my dog", "s3")
        swim = next(f for f in m.get_all(user_id="u")["results"] if "swimming" in f["memory"])
        updated = m.update(swim["id"], "I like diving", metadata={"checked": True})
        out["events"].append(updated)
        add("I moved to Munich", "s4")
        add("My favorite color is blue", "s4", metadata={"source": "chat"})  # 会话又长了
        out["events"].append(m.get_all(user_id="u", filters={"source": "chat"})["results"])
        m.load(m.export(user_id="u"), user_id="v", wait=True)
        (diving,) = updated["results"]
        out["corrected:u"] = [f.model_dump() for f in db.get_facts(space_for("u"))]
        m.delete(diving["id"])  # 遗忘那句更正：原来那条恢复
        for user in ("u", "v"):
            space = space_for(user)
            out[f"facts:{user}"] = [f.model_dump() for f in db.get_facts(space)]
            out[f"entities:{user}"] = [e.model_dump() for e in get_entities(space)]
    for event in out["events"]:
        if isinstance(event, dict):
            event["consolidation"].pop("elapsed_ms", None)
    out["requests"] = [json.dumps(r, sort_keys=True) for r in fake_api.chat_requests]
    return out


def test_reused_state_matches_loading_from_the_database_with_corrections(
    story_settings: Settings,
    fake_api: FakeAPI,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SDK 的 update 与 metadata（R10）经过留下的镜像（R6）：更正以"说错了"取代原来那条、带着
    metadata 又长了的会话、导入与遗忘更正，发给模型的请求、事件与最后的事实和实体，与每次都从
    库里读完全相同。更正的会话 ID 带计数与随机数，这里固定住，两次的事实 ID 才相同。"""
    monkeypatch.setattr(
        sdk_module, "sortable_session_id", lambda prefix, when: f"{prefix}-{when:%Y%m%d-%H%M%S-%f}"
    )
    monkeypatch.setattr(consolidate_module, "MIRRORED_SPACES", 0)  # 不留镜像：每次都从库里读
    cold = corrections_story(story_settings, fake_api, tmp_path / "cold")
    monkeypatch.setattr(consolidate_module, "MIRRORED_SPACES", 4)
    warm = corrections_story(story_settings, fake_api, tmp_path / "warm")
    for key in ("requests", "events", "corrected:u", "facts:u", "entities:u", "facts:v"):
        assert warm[key] == cold[key], key
    assert any(f["supersede_reason"] == "corrected" for f in cold["corrected:u"])
    assert not any(f["supersede_reason"] == "corrected" for f in cold["facts:u"])
    assert {f["object"] for f in cold["facts:u"] if f["superseded_by"] is None} >= {"swimming"}
    assert warm["loads"] < cold["loads"]


def test_add_with_wait_does_not_reload_the_whole_space(
    n3_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SDK 的 add(wait=True)：以前前后各读一遍全部事实、巩固时再从库里读一遍全部事实与实体、
    逐条查向量缓存，还把每个会话的发言读出来重算内容指纹（9,600 条事实、800 个会话时每次
    1–1.5 秒）。现在只读新会话的发言、新的与被取代的事实。"""
    fake_api.chat_replies = [ScriptedModel(where_i_live, reconcile=lambda p: {"decisions": []})]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        for i in range(12):
            m.add(f"I like topic {i}", user_id="u", run_id=f"s{i}", wait=True)
        db, embedder = m.system.db, m.system.embedder
        assert isinstance(embedder, CachedEmbedder)
        calls: Counter[str] = Counter()

        def spy(name: str, target: Any) -> None:
            real = getattr(target, name)

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls[name] += 1
                if name == "get_facts" and kwargs.get("ids") is None:
                    calls["get_facts(all)"] += 1
                if name == "get_turns":
                    calls[f"get_turns:{args[1] if len(args) > 1 else 'all'}"] += 1
                if name == "get_many":
                    calls["cached texts"] += len(args[0])
                return real(*args, **kwargs)

            monkeypatch.setattr(target, name, wrapper)

        for name in ("get_facts", "get_entities", "get_turns"):
            spy(name, db)
        spy("get_many", embedder.cache)
        out = m.add("I moved to Munich", user_id="u", run_id="s12", wait=True)
    assert [e["memory"] for e in out["results"]] == ["The user lives in Munich."]
    assert calls["get_facts(all)"] == 0 and calls["get_entities"] == 0
    # 只读新会话的发言（分配序号、登记任务、抽取），其余 12 个会话的内容指纹不重算
    assert calls["get_turns"] == calls["get_turns:s12"] == 3
    assert calls["cached texts"] <= 3  # 只查这次新抽取的陈述


def test_image_and_text_embeddings_share_one_cache_connection(
    settings: Settings, fake_api: FakeAPI
) -> None:
    """默认配置（通义的多模态向量）下图片与文本向量器共用一个向量缓存连接，图片向量写缓存时文本
    向量器的版本不变，巩固留着的镜像照样能用。以前图片的另开一个连接：以文搜图遇到没见过的问题、
    写入带图的发言都写缓存，文本那边读到的 PRAGMA data_version 就变了，巩固当缓存被别的进程改过，
    丢掉所有空间的镜像（9,000 条事实时每次巩固多花约 0.3 秒）。"""
    with MemorySystem.from_settings(settings, http_client=fake_api.client()) as system:
        embedder, image = system.embedder, system.image_embedder
        assert isinstance(embedder, CachedEmbedder)
        assert isinstance(image, DashScopeMultimodalEmbedder) and image.cache is not None
        assert image.cache is embedder.cache
        before = embedder.version()
        image.cache.put_many([("mm-key", np.zeros(4, dtype=np.float32), None)])
        assert embedder.version() == before


def test_default_consolidation_fingerprint_is_unchanged(
    settings: Settings, fake_api: FakeAPI
) -> None:
    """默认配置的巩固配置指纹（第三轮体检之前的值）。新加的配置字段计入指纹时，每个已有空间都被
    判为过期：显式的巩固把它重建一遍，评测复用的空间报 ConfigMismatchError。并发数这类不影响结果
    的字段（workers、background_workers）要放进 Consolidator.fingerprint 的 exclude。"""
    with MemorySystem.from_settings(settings, http_client=fake_api.client()) as system:
        assert system.consolidator.fingerprint() == "2f0adc7890b6a52e"


def test_fingerprint_function_matches_the_method(settings: Settings, fake_api: FakeAPI) -> None:
    """cuemem doctor 不组装 Consolidator、按配置算指纹（consolidation_fingerprint，第八轮体检
    P16）：与 Consolidator.fingerprint 逐字节相同，有没有补充说明都是，否则 doctor 把每个空间都
    报成过期。"""
    from cuemem.write.consolidate import consolidation_fingerprint

    model = settings.roles["extractor"].model
    for instructions in (None, "只记饮食偏好"):
        config = settings.consolidation.model_copy(update={"instructions": instructions})
        current = settings.model_copy(update={"consolidation": config})
        with MemorySystem.from_settings(current, http_client=fake_api.client()) as system:
            assert consolidation_fingerprint(model, config) == system.consolidator.fingerprint()
    assert consolidation_fingerprint(model, settings.consolidation) == "2f0adc7890b6a52e"


def test_reused_state_notices_writes_from_another_connection(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """别的进程（库的另一个连接）改了事实时不用留下的镜像，照样从库里读：这里另一个连接把
    "住在柏林"标成了不再有效，搬到慕尼黑就没有可以取代的旧住址，不该再问模型。"""
    model = ScriptedModel(where_i_live)
    fake_api.chat_replies = [model]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        m.add("I live in Berlin", user_id="u", run_id="s1", wait=True)
        [berlin] = m.system.db.get_facts(space_for("u"))
        assert m.system.db.path is not None
        other = MemoryDB(m.system.db.path)
        retired = berlin.model_copy(update={"invalidated_at": berlin.recorded_at})
        other.apply_consolidation(facts=[retired], entities=[], episodes=[], units=[])
        other.close()
        m.add("I moved to Munich", user_id="u", run_id="s2", wait=True)
    assert model.calls["reconcile"] == 0


def test_state_is_not_reused_after_another_connection_wrote_during_the_run(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """巩固进行中别的连接改了库（这里在对账问模型时把"住在柏林"标成不再有效）：这次巩固的
    状态是改动之前载入的，做完之后不能留给下一次，下一次照样从库里读。"""
    path: list[Path] = []
    prompts: list[str] = []

    def decide(prompt: str) -> dict[str, Any]:
        prompts.append(prompt)
        if len(prompts) == 1:  # 搬到慕尼黑的对账进行中：另一个连接改了柏林那条
            other = MemoryDB(path[0])
            [berlin] = [f for f in other.get_facts(space_for("u")) if "Berlin" in f.statement]
            retired = berlin.model_copy(update={"invalidated_at": berlin.recorded_at})
            other.apply_consolidation(facts=[retired], entities=[], episodes=[], units=[])
            other.close()
        return {"decisions": []}  # 都按新增处理：这次巩固不会改写柏林那条

    fake_api.chat_replies = [ScriptedModel(where_i_live, reconcile=decide)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        assert m.system.db.path is not None
        path.append(m.system.db.path)
        m.add("I live in Berlin", user_id="u", run_id="s1", wait=True)
        m.add("I moved to Munich", user_id="u", run_id="s2", wait=True)
        m.add("I moved to Paris", user_id="u", run_id="s3", wait=True)
    assert len(prompts) == 2
    assert "Munich" in prompts[1] and "Berlin" not in prompts[1]


@pytest.mark.parametrize("later", [timedelta(hours=5), timedelta(days=2)], ids=["same-day", "days"])
def test_facts_reconcile_in_the_order_they_were_said(
    n3_settings: Settings, fake_api: FakeAPI, later: timedelta
) -> None:
    """一个会话里先说住在 Berlin、几小时后说搬到 Munich，抽取模型却把 Munich 列在前面：Munich
    是当前的、取代了 Berlin（第六轮体检）。以前同一天里的事实记录时间都是会话开始的时间，对账按
    模型列出的顺序走，Berlin 反倒成了当前的；跨了天时以前就对，现在照样对。"""

    def reversed_order(prompt: str) -> dict[str, Any]:
        out = where_i_live(prompt)
        out["facts"] = out["facts"][::-1]  # 模型先列后说的
        return out

    fake_api.chat_replies = [ScriptedModel(reversed_order, reconcile=moved)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as memory:
        with memory.run(user_id="u", wait=True) as run_id:
            memory.add("I live in Berlin", user_id="u", run_id=run_id, timestamp=AUG)
            memory.add("I moved to Munich", user_id="u", run_id=run_id, timestamp=AUG + later)
        facts = {f.object: f for f in memory.system.facts("u")}
        assert facts.keys() == {"Berlin", "Munich"}
        assert facts["Munich"].is_current and not facts["Berlin"].is_current
        assert facts["Berlin"].superseded_by == facts["Munich"].id


CUT = 6000  # 模拟的输出上限（字符）：echo_extraction 每条事实约 250 个字符


def truncating(fake_api: FakeAPI, calls: list[dict[str, Any]]) -> httpx2.Client:
    """抽取的回复超过 CUT 个字符时截断、finish_reason 为 length；别的请求照 FakeAPI。calls 记下
    每个抽取请求的请求体。"""

    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content or b"{}")
        messages = body.get("messages") or [{}]
        if not str(messages[0].get("content", "")).startswith(EXTRACT):
            return fake_api._handle(request)
        calls.append(body)
        text = json.dumps(echo_extraction(messages[1]["content"]), ensure_ascii=False)
        reason = "length" if len(text) > CUT else "stop"
        reply = chat_completion_body(text[:CUT], deepseek=False, finish_reason=reason)
        return httpx2.Response(200, json=reply)

    return httpx2.Client(transport=httpx2.MockTransport(handle))


def notes(n: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"Note {i}: my item number {i} is called thing-{i}."}
        for i in range(n)
    ]


def shown_turns(call: dict[str, Any]) -> int:
    """一个抽取请求里展示了几句发言。"""
    return len(re.findall(r"^\[t\d+\]", call["messages"][1]["content"], re.M))


HALVES = [60, 30, 15, 15, 30, 15, 15]  # 60 句的一半怎样分到回复完整（先序）


def test_truncated_extraction_splits_the_window(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """一次写入 120 句各带一条事实的短发言，抽取的回复在输出上限处被截断（第九轮体检）：把窗口
    对半分开再抽，直到回复完整，120 条事实都在、出处是各自的发言。以前把截断的 JSON 发回去要求
    改正，三次都被截断，任务失败；三个回复进了缓存，之后每次重试都重放同样的失败。"""
    calls: list[dict[str, Any]] = []
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=truncating(fake_api, calls), background=False) as m:
        with start_trace("add") as trace:
            result = m.add(notes(120), user_id="alice", timestamp=AUG, wait=True)
        assert result["status"] == "consolidated"
        # 每个请求只有原来的两条消息，没有"改正"的反馈
        assert [shown_turns(c) for c in calls] == [120, *HALVES, *HALVES]
        assert all(len(c["messages"]) == 2 for c in calls)
        splits = [e["turns"] for e in trace.events if e["kind"] == "extract_split"]
        assert splits == [120, 60, 30, 30, 60, 30, 30]

        space = space_for("alice")
        turns = {t.id: t for t in m.system.db.get_turns(space)}
        facts = m.system.db.get_facts(space)
        assert len(facts) == 120
        for fact in facts:
            [source] = fact.source_turn_ids
            assert turns[source].text == fact.statement
        assert sorted(turns[f.source_turn_ids[0]].idx for f in facts) == list(range(120))
        assert m.status(user_id="alice")["failed"] == []


def test_truncated_extraction_is_retried_as_before_when_not_enabled(
    n3_settings: Settings, fake_api: FakeAPI
) -> None:
    """split_truncated 没开（评测直接构造 MemorySystem；这里把开关关掉）时与以前一样：三个请求，
    后两个带着截断的回复与校验错误，任务失败。之后打开开关再巩固：缓存里第一个回复带着
    finish_reason=length，不重发它，直接从对半分开始，这个会话恢复了。"""
    calls: list[dict[str, Any]] = []
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=truncating(fake_api, calls), background=False) as m:
        extractor = m.system.consolidator.extractor
        assert extractor.split_truncated  # from_settings 打开
        extractor.split_truncated = False
        result = m.add(notes(120), user_id="alice", timestamp=AUG, wait=True)
        assert result["status"] == "queued" and "StructuredOutputError" in result["error"]
        assert [len(c["messages"]) for c in calls] == [2, 4, 6]
        first, second, third = (c["messages"] for c in calls)
        assert second[:2] == first and third[:4] == second
        assert second[2]["role"] == "assistant" and len(second[2]["content"]) == CUT
        assert second[3]["content"].startswith("That JSON does not match the schema: ")
        assert m.get_all(user_id="alice")["results"] == []

        m.consolidate(user_id="alice")  # 重试：三个回复都来自缓存，照样失败
        assert len(calls) == 3
        assert len(m.status(user_id="alice")["failed"]) == 1

        extractor.split_truncated = True
        m.consolidate(user_id="alice")
        assert [shown_turns(c) for c in calls[3:]] == [*HALVES, *HALVES]
        assert len(m.system.db.get_facts(space_for("alice"))) == 120
        assert m.status(user_id="alice")["failed"] == []


def test_one_truncated_turn_fails_after_one_call(n3_settings: Settings, fake_api: FakeAPI) -> None:
    """只有一句发言、回复还是被截断：没法再分，任务照旧失败，但只发一次请求（以前三次）。"""
    calls: list[dict[str, Any]] = []
    with Memory(n3_settings, http_client=truncating(fake_api, calls), background=False) as m:
        long = [{"role": "user", "content": "I keep a list: " + "item, " * 1200}]
        result = m.add(long, user_id="alice", timestamp=AUG, wait=True)
        assert result["status"] == "queued"
        assert "(OutputTruncatedError: " in result["error"]
        assert "item" not in result["error"]
        assert len(calls) == 1
        assert len(m.status(user_id="alice")["failed"]) == 1


def test_splitting_is_off_unless_from_settings_turns_it_on() -> None:
    """评测直接构造 MemorySystem、Consolidator、Extractor：开关默认关，请求照旧。"""
    for cls in (MemorySystem, Consolidator, Extractor):
        assert inspect.signature(cls.__init__).parameters["split_truncated"].default is False
