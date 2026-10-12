"""巩固的端到端场景：用示例对话（林晓换工作、搬到新加坡、笔记本预算变化）检验
知识更新、历史状态查询、双时间轴、幂等与重放、迟到消息、核心记忆和事实增强回答。"""

from __future__ import annotations

from datetime import datetime

import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword

from cuemem.config import Settings
from cuemem.core.models import Conversation, Fact
from cuemem.memory import MemorySystem


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
