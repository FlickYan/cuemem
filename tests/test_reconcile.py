"""事实对账与实体消歧的单元测试：用手工构造的向量，逐条检验规则与灰区的 LLM 裁决。"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import numpy as np
from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.core.models import Fact
from cuemem.llm import LLMClient
from cuemem.write.entities import EntityResolver, EntityState, ResolutionReport, user_entity_id
from cuemem.write.extract import ExtractedEntity
from cuemem.write.reconcile import FactState, Reconciler

USER = "u"


def vec(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def fact(
    fid: str,
    statement: str,
    predicate: str,
    obj: str,
    when: str,
    *,
    date: str | None = None,
    cardinality: str = "single",
    negated: bool = False,
    subject: str = USER,
) -> Fact:
    return Fact(
        id=fid,
        space_id="s",
        subject_id=subject,
        subject="User",
        predicate=predicate,
        object=obj,
        statement=statement,
        cardinality=cardinality,  # type: ignore[arg-type]
        negated=negated,
        valid_from=date,
        recorded_at=datetime.fromisoformat(when),
        session_id=f"session-{when}",
        source_turn_ids=[f"t-{fid}"],
    )


def reconciler(
    settings: Settings, fake_api: FakeAPI, decisions: list[dict[str, Any]]
) -> Reconciler:
    fake_api.chat_replies = [json.dumps({"decisions": decisions})]
    return Reconciler(LLMClient(settings, http_client=fake_api.client()))


def test_single_valued_conflict_is_confirmed_by_llm(settings: Settings, fake_api: FakeAPI) -> None:
    """登记过的单值谓词换了新值，也要由 LLM 确认：规则取代在 dev100 上的精确率只有约 10%。"""
    state = FactState()
    rec = reconciler(settings, fake_api, [{"new": 1, "action": "update", "target": "e1"}])
    old = fact("a", "User lives in Beijing.", "lives_in", "Beijing", "2025-03-01")
    new = fact(
        "b", "User moved to Singapore.", "lives_in", "Singapore", "2025-09-10", date="2025-08"
    )
    rec.reconcile(state, [(old, vec(1, 0))])
    result = rec.reconcile(state, [(new, vec(0, 1))])
    assert [op.action for op in result.operations] == ["supersede"]
    assert len(fake_api.chat_requests) == 1
    retired = state.facts["a"]
    assert retired.valid_to == "2025-08" and retired.supersede_reason == "changed"
    assert retired.invalidated_at == datetime(2025, 9, 10)


def test_event_date_is_not_compared_with_said_date(settings: Settings, fake_api: FakeAPI) -> None:
    """dev100 的真实错误："参加了五次（2022 年的事）"比"参加了三次"说得晚，是更新的计数；
    旧事实没有事件时间时，应按说出的先后判断，而不是拿 2022 去比 2023-05-11。"""
    state = FactState()
    rec = reconciler(settings, fake_api, [{"new": 1, "action": "update", "target": "e1"}])
    three = fact(
        "a", "User attended a support group three times.", "attended_count", "3", "2023-05-11"
    )
    five = fact(
        "b", "User attended five sessions.", "attended_count", "5", "2023-10-30", date="2022"
    )
    rec.reconcile(state, [(three, vec(1, 0))])
    rec.reconcile(state, [(five, vec(0.6, 0.8))])
    assert state.facts["b"].is_current and not state.facts["a"].is_current


def test_duplicate_merges_provenance(settings: Settings, fake_api: FakeAPI) -> None:
    state = FactState()
    rec = reconciler(settings, fake_api, [])
    first = fact("a", "User likes hiking.", "likes", "hiking", "2025-03-01", cardinality="multi")
    again = fact("b", "User enjoys hiking.", "likes", "Hiking", "2025-04-01", cardinality="multi")
    rec.reconcile(state, [(first, vec(1, 0))])
    result = rec.reconcile(state, [(again, vec(0.9, 0.1))])
    assert result.operations[0].action == "merge"
    assert state.facts["a"].source_turn_ids == ["t-a", "t-b"]
    assert "b" not in state.facts
    # 同一会话重放同一条事实（ID 相同）：什么也不做
    assert rec.reconcile(state, [(first, vec(1, 0))]).operations[0].action == "noop"


def test_negation_retires_positive_fact(settings: Settings, fake_api: FakeAPI) -> None:
    state = FactState()
    rec = reconciler(settings, fake_api, [])
    likes = fact("a", "User likes cheese.", "likes", "cheese", "2025-03-01", cardinality="multi")
    stops = fact("b", "User no longer likes cheese.", "likes", "cheese", "2025-05-01", negated=True)
    rec.reconcile(state, [(likes, vec(1, 0))])
    rec.reconcile(state, [(stops, vec(0, 1))])
    assert not state.facts["a"].is_current and state.facts["a"].superseded_by == "b"
    assert state.facts["b"].is_current


def test_gray_zone_goes_to_llm_in_one_call(settings: Settings, fake_api: FakeAPI) -> None:
    state = FactState()
    rec = reconciler(
        settings,
        fake_api,
        [
            {"new": 1, "action": "update", "target": "e1"},
            {"new": 2, "action": "correct", "target": "[e2]"},
        ],
    )
    pb = fact("a", "User's 5K best is 27:45.", "5k_personal_best", "27:45", "2025-04-11")
    dog = fact("c", "User's dog is a retriever.", "dog_breed", "retriever", "2025-04-11")
    rec.reconcile(state, [(pb, vec(1, 0, 0)), (dog, vec(0, 1, 0))])
    new_pb = fact("b", "User's 5K best is 26:30.", "5k_personal_best", "26:30", "2025-07-30")
    new_dog = fact("d", "User's dog is a Labrador.", "dog_breed", "Labrador", "2025-07-30")
    result = rec.reconcile(state, [(new_pb, vec(0.8, 0, 0.6)), (new_dog, vec(0, 0.8, 0.6))])
    assert len(fake_api.chat_requests) == 1  # 两条灰区事实合并成一次调用
    prompt = fake_api.chat_requests[0]["messages"][-1]["content"]
    assert "New fact 1: User's 5K best is 26:30." in prompt and "[e1] User's 5K best" in prompt
    assert result.llm_calls == 1
    # 现实变了：旧事实有结束时间
    assert state.facts["a"].supersede_reason == "changed" and state.facts["a"].valid_to
    # 认知变了（更正）：旧事实被替换，但事件时间轴上没有"结束"
    assert state.facts["c"].supersede_reason == "corrected"
    assert state.facts["c"].valid_to is None and state.facts["c"].invalidated_at is not None


def test_invalid_llm_target_falls_back_to_add(settings: Settings, fake_api: FakeAPI) -> None:
    state = FactState()
    rec = reconciler(settings, fake_api, [{"new": 1, "action": "update", "target": "e9"}])
    rec.reconcile(state, [(fact("a", "x1", "score", "1", "2025-01-01"), vec(1, 0))])
    rec.reconcile(state, [(fact("b", "x2", "score", "2", "2025-02-01"), vec(1, 0.1))])
    assert state.facts["a"].is_current and state.facts["b"].is_current


def test_partial_dates_compare_at_common_precision(settings: Settings, fake_api: FakeAPI) -> None:
    """'2025-05'（只知道月份）与 '2025-05-07' 比不出先后，按被说出的时间判断，不能误判为迟到。"""
    state = FactState()
    rec = reconciler(settings, fake_api, [{"new": 1, "action": "update", "target": "e1"}])
    old = fact("a", "User works at A.", "works_at", "A", "2025-05-07", date="2025-05-07")
    new = fact("b", "User joined B this month.", "works_at", "B", "2025-05-20", date="2025-05")
    rec.reconcile(state, [(old, vec(1, 0))])
    rec.reconcile(state, [(new, vec(0, 1))])
    assert state.facts["b"].is_current and not state.facts["a"].is_current


def test_registered_multi_valued_never_calls_llm(settings: Settings, fake_api: FakeAPI) -> None:
    state = FactState()
    rec = reconciler(settings, fake_api, [])
    a = fact("a", "User visited Paris.", "visited", "Paris", "2025-01-01", cardinality="multi")
    b = fact("b", "User visited Rome.", "visited", "Rome", "2025-02-01", cardinality="multi")
    rec.reconcile(state, [(a, vec(1, 0))])
    rec.reconcile(state, [(b, vec(0.99, 0.05))])
    assert state.facts["a"].is_current and state.facts["b"].is_current
    assert fake_api.chat_requests == []


# ---------- 实体消歧 ----------


def resolver(settings: Settings, fake_api: FakeAPI, reply: dict[str, Any]) -> EntityResolver:
    fake_api.chat_replies = [json.dumps(reply)]
    return EntityResolver(LLMClient(settings, http_client=fake_api.client()))


def test_entity_resolution_levels(settings: Settings, fake_api: FakeAPI) -> None:
    state = EntityState("s")
    res = resolver(settings, fake_api, {"matches": [{"new": 1, "same_as": "c1"}]})
    when = datetime(2025, 3, 1)
    report = ResolutionReport()
    res.ensure_people(state, ["User", "Assistant"], when, report)
    assert state.lookup("I") == user_entity_id("s") and state.lookup("assistant") is None

    max_dog = ExtractedEntity(name="Max", type="pet", aliases=["my dog"], description="dog")
    res.resolve(state, [max_dog], {"Max (pet): dog": vec(1, 0, 0)}, when, report)
    max_id = state.lookup("Max")
    assert max_id is not None and state.lookup("my dog") is None  # 泛称不进别名索引

    # 1. 名字精确匹配
    report = ResolutionReport()
    res.resolve(state, [ExtractedEntity(name="max", type="pet")], {}, when, report)
    assert report.linked_exact == 1 and state.entities[max_id].mentions == 2
    # 2. 向量足够相似：直接合并，并记下新别名
    report = ResolutionReport()
    maxie = ExtractedEntity(name="Maxie", type="pet", description="the dog")
    res.resolve(state, [maxie], {"Maxie (pet): the dog": vec(0.99, 0.1, 0)}, when, report)
    assert report.linked_vector == 1 and state.lookup("Maxie") == max_id
    # 3. 灰区：交给 LLM 确认
    report = ResolutionReport()
    buddy = ExtractedEntity(name="Buddy", type="pet", description="a puppy")
    res.resolve(state, [buddy], {"Buddy (pet): a puppy": vec(0.8, 0.6, 0)}, when, report)
    assert report.linked_llm == 1 and state.lookup("Buddy") == max_id
    # 类型不兼容的不会合并
    report = ResolutionReport()
    place = ExtractedEntity(name="Max Bar", type="place", description="a bar")
    res.resolve(state, [place], {"Max Bar (place): a bar": vec(1, 0, 0)}, when, report)
    assert report.created == 1 and state.lookup("Max Bar") != max_id
    assert len(fake_api.chat_requests) == 1


def test_user_name_becomes_alias(settings: Settings, fake_api: FakeAPI) -> None:
    state = EntityState("s")
    res = resolver(settings, fake_api, {"matches": []})
    when = datetime(2025, 3, 1)
    report = ResolutionReport()
    res.name_user(state, "林晓", when, report)
    res.resolve(state, [ExtractedEntity(name="林晓", type="person")], {}, when, report)
    assert state.lookup("林晓") == user_entity_id("s")
    assert "林晓" in state.entities[user_entity_id("s")].aliases
    assert report.created == 1  # 只有用户本人这一个实体


def test_plans_do_not_trigger_single_valued_rule(settings: Settings, fake_api: FakeAPI) -> None:
    """ "旅行期间住在某酒店"被标成 lives_in 时，不能按规则把用户的住址取代掉。"""
    state = FactState()
    rec = reconciler(settings, fake_api, [{"new": 1, "action": "add"}])
    home = fact("a", "User lives in Seattle.", "lives_in", "Seattle", "2025-01-01")
    trip = fact("b", "User will stay at Park Hyatt Tokyo.", "lives_in", "Park Hyatt", "2025-05-22")
    trip = trip.model_copy(update={"category": "plan"})
    rec.reconcile(state, [(home, vec(1, 0))])
    rec.reconcile(state, [(trip, vec(0, 1))])
    assert state.facts["a"].is_current and state.facts["b"].is_current
    assert len(fake_api.chat_requests) == 1  # 交给 LLM 判断，而不是直接取代
