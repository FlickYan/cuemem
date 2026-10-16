"""巩固的端到端场景：用示例对话（林晓换工作、搬到新加坡、笔记本预算变化）检验
知识更新、历史状态查询、双时间轴、幂等与重放、迟到消息、核心记忆和事实增强回答。"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any

import pytest
from fakes import NO_BALANCE, TOY_CONVERSATION, FakeAPI
from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword
from test_sdk import where_i_live

from cuemem import Memory
from cuemem.config import Settings
from cuemem.core.models import Conversation, Fact
from cuemem.memory import MemorySystem

EXTRACT = "You build the long-term memory"  # 抽取请求的系统提示开头


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

    system.ingest(toy())  # 重复导入同一份对话不会登记新任务
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
    worker._thread.join(10)
    assert not worker._thread.is_alive() and not worker.errors

    with Memory(n3_settings, http_client=fake_api.client()) as again:
        again.wait()
        assert statuses(again, "alice") == [("done", 0)]
        assert [f["memory"] for f in again.get_all(user_id="alice")["results"]] == [
            "The user lives in Berlin."
        ]


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
    """影响巩固结果的配置（如阈值）变了：派生记忆自动重建；只改并发数不算。"""
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()) as memory:
        memory.ingest(toy(), consolidate=True)
        before = memory.consolidator.fingerprint()
        assert not memory.consolidator.stale("demo")
    faster = n3_settings.consolidation.model_copy(update={"workers": 7})
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
