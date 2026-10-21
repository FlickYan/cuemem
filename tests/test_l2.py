"""L2 深度检索：查询规划、图扩散、多查询融合、充分性检查与补检索、笔记式阅读、重排、
规则版控制器与模型自主检索（B3）。全部离线：脚本化模型 + 哈希向量。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from typing import Any

import numpy as np
import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from scipy import sparse
from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword

from cuemem.config import Settings
from cuemem.core.models import Conversation
from cuemem.core.trace import start_trace
from cuemem.evals.methods import L1RAG, build_methods
from cuemem.evals.runner import usage_from_trace
from cuemem.memory import MemorySystem
from cuemem.read.agent import date_diff
from cuemem.read.controller import RuleController
from cuemem.read.graph import GraphConfig, MemoryGraph
from cuemem.read.l1 import L1Config
from cuemem.read.l2 import L2Config
from cuemem.read.plan import PlanOutput, from_output, range_text, rule_plan
from cuemem.write.consolidate import scope

NOW = datetime(2023, 5, 30, 22, 0)  # 周二
ASKED = datetime(2025, 10, 10)


# ---------- 查询规划 ----------


def test_plan_output_is_parsed_leniently() -> None:
    out = PlanOutput.model_validate(
        {
            "answer_type": "How-Many",  # 不认识的类型归为 other
            "exhaustive": True,
            "sub_queries": "I bought earrings",  # 单个字符串也接受
            "time_start": "2023-05-01T00:00",
            "time_end": "not a date",
            "entities": None,
        }
    )
    assert out.answer_type == "other" and out.sub_queries == ["I bought earrings"]
    assert out.time_start == "2023-05-01" and out.time_end is None and out.entities == []


def test_plan_prefers_rule_time_range_and_validates_model_range() -> None:
    question = "What did I buy two weeks ago?"
    output = PlanOutput(
        answer_type="list",
        sub_queries=["I bought", "what did I buy two weeks ago?", "I bought"],
        time_start="2023-05-16",
        time_end="2023-05-16",
    )
    plan = from_output(question, NOW, output)
    # 规则能解析"两周前"时用规则的区间（带容差），不用模型给的单日
    assert range_text(plan.time_range) == "2023-05-13–2023-05-19"
    assert plan.sub_queries == ["I bought"]  # 与原问题重复、彼此重复的子查询去掉
    assert plan.aggregate and plan.exhaustive  # 列举类问题需要穷尽
    assert plan.queries == [question, "I bought"]

    # 规则解析不了时用模型的区间：起止颠倒会纠正，落在提问之后的忽略，太宽的忽略
    vague = "Where did I travel around the holidays?"
    swapped = from_output(vague, NOW, PlanOutput(time_start="2022-12-31", time_end="2022-12-20"))
    assert range_text(swapped.time_range) == "2022-12-20–2022-12-31"
    future = from_output(vague, NOW, PlanOutput(time_start="2023-07-01", time_end="2023-07-02"))
    assert future.time_range is None and future.notes
    wide = from_output(vague, NOW, PlanOutput(time_start="2010-01-01", time_end="2023-01-01"))
    assert wide.time_range is None


@pytest.mark.parametrize(
    ("question", "answer_type"),
    [
        ("How many fitness classes do I attend in a typical week?", "count"),
        ("How much did I spend on groceries in total?", "sum"),
        ("How long have I been working at my company?", "duration"),
        ("Which did I visit first, the zoo or the museum?", "order"),
        ("When did I adopt Max?", "date"),
        ("Can you recommend a show for tonight?", "advice"),
        ("Did I go to the concert with a friend?", "yes_no"),
        ("Where do I work?", "fact"),
    ],
)
def test_rule_plan_answer_types(question: str, answer_type: str) -> None:
    plan = rule_plan(question, NOW)
    assert plan.answer_type == answer_type and plan.source == "rules" and not plan.sub_queries


# ---------- 图扩散 ----------


def test_personalized_pagerank_on_a_chain() -> None:
    # a — b — c：从 a 出发，越远分数越低，总和为 1
    adjacency = sparse.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    rows = np.asarray(adjacency.sum(axis=1)).ravel()
    graph = MemoryGraph(
        ids=["a", "b", "c"],
        kinds=["entity", "fact", "round"],
        transition=sparse.csr_matrix(sparse.diags(1 / rows) @ adjacency),
        index={"a": 0, "b": 1, "c": 2},
        alias_pattern=None,
        alias_to_entity={},
        entity_degree={},
        hubs=set(),
    )
    scores = graph.personalized_pagerank({"a": 1.0}, GraphConfig())
    assert scores["a"] > scores["b"] > scores["c"] > 0
    assert sum(scores.values()) == pytest.approx(1.0)
    assert graph.ranked(scores, "round") == [("c", scores["c"])]
    assert graph.personalized_pagerank({"missing": 1.0}, GraphConfig()) == {}


@pytest.fixture
def toy(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[tuple[MemorySystem, ScriptedModel]]:
    model = ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    fake_api.chat_replies = [model]
    conv = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    system.ingest(conv, consolidate=True)
    yield system, model
    system.close()


def test_memory_graph_links_entities_and_mentions(
    toy: tuple[MemorySystem, ScriptedModel],
) -> None:
    system, _ = toy
    graph = MemoryGraph.build(system.db, system.vectors, "demo", GraphConfig())
    max_id = graph.link_entities(["Max 喜欢去哪里散步？"])
    assert len(max_id) == 1
    (entity,) = max_id
    # "Max" 出现在三段对话的原话里：实体—轮的提及边把它们连起来
    neighbours = graph.transition[graph.index[entity]].indices
    rounds = [graph.ids[i] for i in neighbours if graph.kinds[i] == "round"]
    assert len(rounds) >= 3
    assert graph.stats["entity_round"] >= 3
    scores = graph.personalized_pagerank(max_id, GraphConfig())
    assert graph.ranked(scores, "round")  # 从实体出发能走到轮


# ---------- L2 检索 ----------


def test_l2_sub_queries_and_follow_up_retrieval(
    toy: tuple[MemorySystem, ScriptedModel],
) -> None:
    system, model = toy
    model.planner = lambda prompt: {
        "answer_type": "list",
        "exhaustive": True,
        "sub_queries": ["领养了一只金毛", "笔记本预算"],
        "entities": ["Max"],
    }
    checks = iter(
        [
            {"sufficient": False, "missing": "旅行计划", "queries": ["去巴厘岛旅行"]},
            {"sufficient": True},
        ]
    )
    model.checker = lambda prompt: next(checks)
    cfg = L2Config(l1=L1Config(budget_tokens=60, max_rounds=60), followup_budget=60, ppr=True)
    result = system.l2.search(
        "demo", "我都跟你说过哪些事？", ASKED, cfg, cache_scope=scope("demo", "ask")
    )
    assert result.plan.source == "llm" and result.plan.aggregate
    assert [c.sufficient for c in result.checks] == [False, True]
    assert result.checks[0].added_rounds >= 1
    assert any("巴厘岛" in h.unit.text for h in result.rounds)  # 补检索找到的
    assert result.rounds[-1].route == "l2_followup"
    assert len(result.facts) > 10 or len(result.facts) == len(system.db.get_facts("demo"))
    details = result.details()
    assert details["plan"]["sub_queries"] == ["领养了一只金毛", "笔记本预算"]
    assert details["ppr"]["graph"]["nodes"] > 0
    assert model.calls["plan"] == 1 and model.calls["check"] == 2

    # 第二次相同的检索全部命中缓存：不再调用规划与检查
    system.l2.search("demo", "我都跟你说过哪些事？", ASKED, cfg, cache_scope=scope("demo", "ask"))
    assert model.calls["plan"] == 1 and model.calls["check"] == 2


def test_l2_without_planner_equals_l1_ranking(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    system, model = toy
    l1_cfg = L1Config(budget_tokens=200, max_rounds=60)
    question = "我现在在哪工作？"
    l1 = system.l1.search("demo", question, ASKED, l1_cfg)
    l2 = system.l2.search("demo", question, ASKED, L2Config(l1=l1_cfg, plan=False, iterations=0))
    assert [h.unit.id for h in l2.rounds] == [h.unit.id for h in l1.rounds]
    assert l2.plan.source == "rules" and model.calls["plan"] == 0


# ---------- 本地 Qdrant 的检索次数 ----------


def count_searches(system: MemorySystem, monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """记下每次向量检索按哪种单元过滤：本地 Qdrant 每次按类型过滤的检索都要扫一遍整个集合。"""
    kinds: list[str | None] = []
    real = system.vectors._query

    def query(space_id: str, vector: Any, using: str, k: int, flt: Any) -> Any:
        kinds.append(flt.must[0].match.any[0] if flt is not None else None)
        return real(space_id, vector, using, k, flt)

    monkeypatch.setattr(system.vectors, "_query", query)
    return kinds


def test_l1_skips_routes_that_cannot_change_scores(
    toy: tuple[MemorySystem, ScriptedModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    """纯文字空间里不检索图片描述（结果一定是空的）；摘要的权重为 0 时不检索摘要（乘 0 不改
    分数），只有探针为了 p_summary_* 特征检索它。以前每次都是四次检索。"""
    system, _ = toy
    question = "Max 喜欢吃什么？"
    kinds = count_searches(system, monkeypatch)
    cfg = L1Config()
    cands = system.l1.candidates("demo", question, ASKED, cfg)
    assert kinds == ["round", "fact"]
    assert cands.routes[("image", "dense")] == [] and ("summary", "dense") not in cands.routes
    kinds.clear()
    probe = system.probe.run("demo", question, ASKED, cfg)
    assert kinds == ["round", "fact", "summary"]
    assert probe.candidates.routes[("summary", "dense")]
    assert probe.features["p_summary_top1"] > 0 and probe.features["p_caption_top1"] == 0.0
    # 摘要这一路在权重为 0 时不改任何分数：加上它重新打分，排名与分数逐位相同
    summary = {k: v for k, v in probe.candidates.routes.items() if k[0] == "summary"}
    rescored = system.l1._score("demo", cands.query, {**cands.routes, **summary}, cfg)
    assert rescored.ranked == cands.ranked and rescored.fact_score == cands.fact_score
    kinds.clear()
    system.l1.candidates("demo", question, ASKED, L1Config(summary_weight=0.5))
    assert kinds == ["round", "fact", "summary"]
    kinds.clear()
    bm25 = system.l1.candidates("demo", question, ASKED, L1Config(sparse="bm25"))
    assert kinds == ["round", "fact"] and bm25.routes[("image", "sparse")] == []


def test_l2_reuses_the_probe_routes_for_the_question(
    toy: tuple[MemorySystem, ScriptedModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    """控制器选了 L2 时，原问题的召回直接用探针的各路结果、按计划重新打分；子查询照常检索。
    以前探针检索四次之后，L2 对原问题再检索四次。"""
    system, model = toy
    question = "我一共养了几只狗？"
    kinds = count_searches(system, monkeypatch)
    cfg = L2Config(plan=False, iterations=0)
    recall = system.retrieve("demo", question, question_time=ASKED, lam=-1e9, l2=cfg)
    assert recall.level == "l2" and recall.candidates is not None
    assert kinds == ["round", "fact", "summary"]  # 只有探针的检索
    kinds.clear()
    fresh = system.l2.search("demo", question, ASKED, cfg)  # 没有探针：自己检索
    assert kinds == ["round", "fact"]
    assert [(h.unit.id, h.score) for h in recall.view.rounds] == [
        (h.unit.id, h.score) for h in fresh.rounds
    ]
    assert [(f.fact.id, f.score) for f in recall.view.facts] == [
        (f.fact.id, f.score) for f in fresh.facts
    ]

    # 有子查询时：原问题用探针的结果，两个子查询各检索两次（轮、事实）
    model.planner = lambda prompt: {"answer_type": "count", "sub_queries": ["狗", "领养"]}
    kinds.clear()
    planned = system.retrieve("demo", question, question_time=ASKED, lam=-1e9, l2=L2Config())
    assert planned.level == "l2" and planned.plan is not None and planned.plan.sub_queries
    assert kinds == ["round", "fact", "summary", "round", "fact", "round", "fact"]


def test_probe_routes_are_reused_only_for_the_same_input(
    toy: tuple[MemorySystem, ScriptedModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    """只有问题、逐位相同的查询向量、每路条数、稀疏路与排除的 ID 都相同时才用探针的结果。"""
    system, _ = toy
    question = "Max 喜欢吃什么？"
    probe = system.l1.candidates("demo", question, ASKED, L1Config(), probe=True)
    assert probe.recall_key is not None
    vector = np.frombuffer(probe.recall_key.vector, dtype=np.float32).copy()
    kinds = count_searches(system, monkeypatch)
    same = system.l1.candidates(
        "demo", question, ASKED, L1Config(), dense_vec=vector.copy(), reuse=probe
    )
    assert kinds == []
    assert same.routes == {k: v for k, v in probe.routes.items() if k[0] != "summary"}
    nudged = vector.copy()
    nudged[0] = np.nextafter(nudged[0], np.float32(1))
    some_round = system.db.list_units("demo", kinds=["round"])[0].id
    for text, cfg, kwargs in (
        (question, L1Config(), {"dense_vec": nudged}),
        (question, L1Config(), {"dense_vec": vector, "exclude": [some_round]}),
        (question, L1Config(pool=10), {"dense_vec": vector}),
        (question, L1Config(sparse="bm25"), {"dense_vec": vector}),
        ("Max 爱吃什么？", L1Config(), {"dense_vec": vector}),
    ):
        kinds.clear()
        system.l1.candidates("demo", text, ASKED, cfg, reuse=probe, **kwargs)
        assert kinds == ["round", "fact"], (text, cfg, kwargs)


def test_recall_listing_raised_from_l0_packs_the_probe(
    toy: tuple[MemorySystem, ScriptedModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP、Agents 与 LangGraph 的 recall（at_least_l1）：控制器选了 L0 时直接打包探针的候选，
    结果与直接走 L1 相同。以前改走 L1 时把探针做过的四次检索再做一遍。"""
    from cuemem.evidence import RAISED_REASON, recall_listing

    system, _ = toy
    question = "Max 喜欢吃什么？"
    kinds = count_searches(system, monkeypatch)
    raised = recall_listing(system, "demo", question, at=ASKED, at_least_l1=True, lam=1e9)
    assert raised["level"] == "L1" and raised["reasons"] == [RAISED_REASON]
    assert kinds == ["round", "fact", "summary"]
    kinds.clear()
    plain = recall_listing(system, "demo", question, at=ASKED, mode="l1")
    assert kinds == ["round", "fact"]
    assert raised["excerpts"] and raised["excerpts"] == plain["excerpts"]
    assert raised["facts"] == plain["facts"]


@pytest.mark.parametrize("at", [ASKED, None])
def test_sdk_search_raised_from_l0_packs_the_probe(
    toy: tuple[MemorySystem, ScriptedModel], monkeypatch: pytest.MonkeyPatch, at: datetime | None
) -> None:
    """SDK 的 search（缺省不低于 L1）与 recall_listing 相同：控制器选了 L0 时直接打包探针的候选
    （两次用同一个问题时间），结果与直接走 L1 相同。以前改走 L1 时把探针做过的检索再做一遍。"""
    from cuemem.evidence import RAISED_REASON
    from cuemem.sdk import Memory

    system, _ = toy
    memory = Memory.from_system(system)
    question = "Max 喜欢吃什么？"
    kinds = count_searches(system, monkeypatch)
    raised = memory.search(question, user_id="demo", at=at, lam=1e9)
    assert raised["level"] == "L1" and raised["reasons"] == [RAISED_REASON]
    assert kinds == ["round", "fact", "summary"]
    kinds.clear()
    plain = memory.search(question, user_id="demo", at=at, mode="l1")
    assert kinds == ["round", "fact"]
    assert raised["results"] and raised["results"] == plain["results"]


def test_planner_and_checker_failures_fall_back(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    system, model = toy
    model.planner = lambda prompt: {"answer_type": ["not", "a", "string"], "exhaustive": "maybe"}
    model.checker = lambda prompt: {"sufficient": "perhaps"}
    result = system.l2.search("demo", "我现在在哪工作？", ASKED, L2Config())
    assert result.plan.source == "rules" and "规划失败" in result.plan.notes[0]
    assert result.checks[0].error == "StructuredOutputError" and result.checks[0].sufficient
    assert result.rounds


def test_rerank_reorders_and_is_cached(
    toy: tuple[MemorySystem, ScriptedModel], fake_api: FakeAPI
) -> None:
    system, _ = toy
    cfg = L2Config(plan=False, iterations=0, rerank=True)
    with start_trace("t") as trace:
        first = system.l2.search("demo", "Max 狗粮", ASKED, cfg, cache_scope="demo|ask")
    usage = usage_from_trace(trace)
    assert len(fake_api.rerank_requests) == 1
    assert usage["rerank_calls"] == 1 and usage["cost"] > 0 and usage["spent"] > 0
    assert first.debug["rerank"]["pool"] > 0
    with start_trace("t") as trace:
        system.l2.search("demo", "Max 狗粮", ASKED, cfg, cache_scope="demo|ask")
    cached = usage_from_trace(trace)
    assert len(fake_api.rerank_requests) == 1  # 命中缓存
    assert cached["cost"] == usage["cost"] and cached["spent"] == 0


# ---------- 回答 ----------


def test_deep_answer_uses_notes_and_reading_guide(
    toy: tuple[MemorySystem, ScriptedModel],
) -> None:
    system, model = toy
    model.planner = lambda prompt: {
        "answer_type": "count",
        "exhaustive": True,
        "sub_queries": ["狗"],
        "time_start": "2025-06-01",
        "time_end": "2025-06-30",
    }
    model.reader = lambda prompt: {
        "notes": ["[m1] 2025-06-20: Max 挑食"],
        "answer": "一次",
        "citations": ["m1"],
        "evidence_sufficient": True,
    }
    answer = system.ask("demo", "我六月提过几次 Max？", question_time=ASKED, mode="l2")
    prompt = model.prompts["read"][-1]
    assert "Reading guide:" in prompt and "count it once" in prompt
    assert "2025-06-01–2025-06-30" in prompt and "★" in prompt
    assert answer.notes == ["[m1] 2025-06-20: Max 挑食"] and answer.answer == "一次"
    assert answer.details["plan"]["answer_type"] == "count"

    # 不记笔记：阅读方式与 L1 相同
    system.ask("demo", "我六月提过几次 Max？", question_time=ASKED, mode="l2", notes=False)
    assert "Reading guide:" not in model.prompts["read"][-1]


def test_rule_controller_routes() -> None:
    controller = RuleController()
    to_l2 = [
        "How many properties did I view before making an offer?",
        "How long had I been using my phone when it broke?",
        "Which streaming service did I start using most recently?",
        "What did I buy first, the bike or the helmet?",
        "我一共去过几次健身房？",
    ]
    to_l1 = [
        "Where do I work?",
        "What did I do last weekend?",  # "last weekend" 是时间表达，不是先后
        "Can you recommend a restaurant for tonight?",
    ]
    for question in to_l2:
        assert controller.decide(question, NOW).level == "L2", question
    for question in to_l1:
        assert controller.decide(question, NOW).level == "L1", question
    assert RuleController().decide(to_l2[0], NOW).reasons[:2] == ["aggregation", "order"]


def test_auto_mode_records_route(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    system, model = toy
    simple = system.ask("demo", "我在哪工作？", question_time=ASKED, mode="auto", router="rule")
    assert simple.details["route"]["level"] == "L1" and model.calls["plan"] == 0
    deep = system.ask("demo", "我一共养了几只狗？", question_time=ASKED, mode="auto", router="rule")
    assert deep.details["route"] == {
        "router": "rule",
        "level": "L2",
        "see": False,
        "reasons": ["aggregation"],
    }
    assert model.calls["plan"] == 1


def test_agent_searches_then_answers(
    toy: tuple[MemorySystem, ScriptedModel], fake_api: FakeAPI
) -> None:
    system, _ = toy
    steps: list[dict[str, Any]] = []

    def agent(body: dict[str, Any]) -> dict[str, Any]:
        steps.append(body)
        tools = [t["function"]["name"] for t in body.get("tools", [])]
        if len(steps) == 1:
            args = json.dumps({"query": "Max 狗粮"}, ensure_ascii=False)
            calls = [{"id": "c1", "name": "memory_search", "arguments": args}]
            calls.append({"id": "c2", "name": "date_diff", "arguments": '{"start": "2025-06-20", '
                          '"end": "2025-10-10"}'})  # fmt: skip
            return {"tool_calls": calls}
        assert "answer" in tools
        answer = {"answer": "鸡肉味", "citations": ["m1", "f1", "x9"], "evidence_sufficient": True}
        return {"tool_calls": [{"id": "c3", "name": "answer", "arguments": json.dumps(answer)}]}

    fake_api.chat_replies = [agent]
    result = system.ask("demo", "Max 爱吃什么口味的狗粮？", question_time=ASKED, mode="agent")
    assert result.answer == "鸡肉味" and result.citations == ["m1", "f1"]  # 编造的编号去掉
    assert result.details["searches"] == ["Max 狗粮"] and result.hits
    tool_messages = [m for m in steps[1]["messages"] if m["role"] == "tool"]
    assert "鸡肉" in tool_messages[0]["content"] and "112 days" in tool_messages[1]["content"]


def test_hatch_gives_recalled_memories_then_lets_the_model_search(
    toy: tuple[MemorySystem, ScriptedModel], fake_api: FakeAPI
) -> None:
    system, _ = toy
    steps: list[dict[str, Any]] = []

    def agent(body: dict[str, Any]) -> dict[str, Any]:
        steps.append(body)
        if len(steps) == 1:
            args = json.dumps({"query": "Max 狗粮"}, ensure_ascii=False)
            return {"tool_calls": [{"id": "c1", "name": "memory_search", "arguments": args}]}
        answer = {"answer": "鸡肉味", "citations": ["m1", "f1"], "evidence_sufficient": True}
        return {"tool_calls": [{"id": "c2", "name": "answer", "arguments": json.dumps(answer)}]}

    fake_api.chat_replies = [agent]
    result = system.ask(
        "demo", "Max 爱吃什么口味的狗粮？", question_time=ASKED, mode="hatch", router="rule"
    )
    first = steps[0]["messages"]
    assert "Memories recalled for this question are given with it" in first[0]["content"]
    assert "Retrieved conversation excerpts:" in first[1]["content"]  # 控制器召回的记忆随问题给出
    assert result.details["route"]["router"] == "rule"
    assert result.details["searches"] == ["Max 狗粮"]
    shown = first[1]["content"].count("\n[m")
    tool = next(m for m in steps[1]["messages"] if m["role"] == "tool")
    if "[m" in tool["content"]:  # 补查到的新片段编号接着往下排
        assert f"[m{shown + 1}]" in tool["content"]
    assert result.answer == "鸡肉味" and set(result.citations) <= {"m1", "f1"}


def test_b3_request_is_unchanged_without_recall(
    toy: tuple[MemorySystem, ScriptedModel], fake_api: FakeAPI
) -> None:
    system, _ = toy
    bodies: list[dict[str, Any]] = []

    def agent(body: dict[str, Any]) -> dict[str, Any]:
        bodies.append(body)
        answer = {"answer": "x", "citations": [], "evidence_sufficient": False}
        return {"tool_calls": [{"id": "c1", "name": "answer", "arguments": json.dumps(answer)}]}

    fake_api.chat_replies = [agent]
    system.ask("demo", "Max 是什么品种？", question_time=ASKED, mode="agent")
    messages = bodies[0]["messages"]
    assert messages[0]["content"].endswith("call the answer tool exactly once.")
    assert "You can search your memory" in messages[0]["content"]
    assert messages[1]["content"].endswith("Question: Max 是什么品种？")
    assert "Retrieved conversation excerpts" not in messages[1]["content"]


def test_date_diff() -> None:
    assert date_diff("2023-05-01", "2023-05-30").startswith("29 days")
    assert date_diff("2023-05-30", "2023-05-01").startswith("-29 days")
    assert date_diff("May 1", "2023-05-30") == "Invalid date: use YYYY-MM-DD."


def test_new_methods_build() -> None:
    names = ["B4-b6k", "B4-note", "B5", "B5-noplan", "B5-noiter", "B5-nonote", "B5-ppr"]
    methods = build_methods([*names, "B6r", "B3"])
    assert [m.name for m in methods] == [*names, "B6r", "B3"]
    b5 = next(m for m in methods if m.name == "B5")
    assert b5.needs_consolidation
    # B4 固定为 N4 的配置（带 BM25）；N5 起 L1 的默认不用 BM25
    b4 = build_methods(["B4"])[0]
    assert isinstance(b4, L1RAG) and b4.config.sparse == "bm25" and L1Config().sparse == "none"
    assert replace(L2Config(), ppr=True).ppr and not L2Config().ppr


def test_report_l2_section_and_false_abstentions() -> None:
    from cuemem.evals.report import _l2_section, false_abstentions, method_stats

    def row(method: str, sample: str, *, correct: bool, response: str, **details: Any) -> Any:
        return {
            "sample_id": sample,
            "method": method,
            "correct": correct,
            "is_abstention": False,
            "question_type": "multi-session",
            "response": response,
            "retrieval": None,
            "retrieved": [],
            "retrieved_facts": [],
            "details": details or None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "llm_calls": 3,
                "llm_cache_hits": 0,
                "cost": 0.001,
                "latency_ms": 10.0,
                "llm_calls_by_role": {"planner": 2, "reader": 1},
            },
            "judge_usage": {"cost": 0.0},
        }

    plan = {"source": "llm", "sub_queries": ["a", "b"], "exhaustive": True, "time_range": None}
    checks = [{"added_rounds": 2}, {"added_rounds": 0}]
    rows = [
        row("B5", "q1", correct=True, response="3", plan=plan, checks=checks, evidence_tokens=600),
        row("B5", "q2", correct=False, response="I don't know.", plan=plan, checks=[]),
        row("B6r", "q1", correct=True, response="3", route={"level": "L2", "reasons": ["order"]}),
        row("B6r", "q2", correct=False, response="Paris", route={"level": "L1", "reasons": []}),
    ]
    text = "\n".join(_l2_section(rows, method_stats(rows)))
    assert "| 平均子查询数 | 2.00 |" in text and "| 至少补检索一次 | 50% |" in text
    assert "| 每题规划与检查的调用次数 | 2.00 |" in text
    assert "B6r：1 / 2 题走 L2（50%）；触发的线索：order 1。" in text
    assert false_abstentions(rows) == {"B5": 1}
