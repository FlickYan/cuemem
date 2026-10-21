"""事实对账与实体消歧的单元测试：用手工构造的向量，逐条检验规则与灰区的 LLM 裁决。"""

from __future__ import annotations

import json
import random
from datetime import datetime
from typing import Any

import numpy as np
import pytest
from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.core.models import Fact
from cuemem.llm import LLMClient
from cuemem.write import reconcile as reconcile_module
from cuemem.write.entities import EntityResolver, EntityState, ResolutionReport, user_entity_id
from cuemem.write.extract import ExtractedEntity
from cuemem.write.predicates import normalize_value
from cuemem.write.reconcile import FactState, Reconciler, _by_similarity

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


def test_gray_zone_prompt_quotes_the_source_message(settings: Settings, fake_api: FakeAPI) -> None:
    """抽取出的"Mia 31 岁"看不出是更正还是过了生日：灰区的提示附上它出自的那句话。"""
    state = FactState()
    rec = reconciler(settings, fake_api, [{"new": 1, "action": "correct", "target": "e1"}])
    rec.reconcile(state, [(fact("a", "Mia is 30.", "age", "30", "2025-03-03"), vec(1, 0))])
    new = fact("b", "Mia is 31.", "age", "31", "2025-06-20")
    said = {"t-b": "Oops, I got that wrong before:\n\nMia is 31, not 30.", "t-x": "unrelated"}
    rec.reconcile(state, [(new, vec(0.8, 0.6))], sources=said)
    prompt = fake_api.chat_requests[0]["messages"][-1]["content"]
    assert 'From the message: "Oops, I got that wrong before: Mia is 31, not 30."' in prompt
    assert "unrelated" not in prompt
    assert state.facts["a"].supersede_reason == "corrected"


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


def test_logged_correction_replaces_its_target(settings: Settings, fake_api: FakeAPI) -> None:
    """SDK 的 update 记下的更正（corrections：轮 ID → 要取代的事实）：由那句抽出的事实里，同谓词
    的、其次陈述最相近的那条以"说错了"取代目标，不问 LLM；同一句抽出的其他事实照常对账。没有
    更正时规则不变（多值的谓词只会新增）。"""
    state = FactState()
    rec = reconciler(settings, fake_api, [])
    jazz = fact("a", "User likes jazz.", "likes", "jazz", "2025-01-01", cardinality="multi")
    rec.reconcile(state, [(jazz, vec(1, 0, 0))])
    edit = ["t-edit"]
    trip = fact(
        "b", "User visited a jazz bar.", "visited", "bar", "2025-02-01", cardinality="multi"
    )
    blues = fact("c", "User likes blues.", "likes", "blues", "2025-02-01", cardinality="multi")
    soul = fact("d", "User likes soul.", "likes", "soul", "2025-02-01", cardinality="multi")
    new = [
        (f.model_copy(update={"source_turn_ids": edit}), v)
        for f, v in (
            (trip, vec(0.9, 0.1, 0)),  # 陈述最相近，但谓词不同
            (blues, vec(0, 1, 0)),
            (soul, vec(0.3, 0.7, 0)),  # 同谓词里与 jazz 更近的这条被选中
        )
    ]
    result = rec.reconcile(state, new, corrections={"t-edit": "a"})
    retired = state.facts["a"]
    assert retired.superseded_by == "d" and retired.supersede_reason == "corrected"
    assert retired.valid_to is None
    assert [(op.action, op.fact_id, op.rule) for op in result.operations] == [
        ("add", "b", "multi_valued"),
        ("add", "c", "multi_valued"),
        ("supersede", "d", "edit"),
    ]
    assert fake_api.chat_requests == []

    # 目标已经不在（被遗忘、换了抽取模型）或不是当前的：照常对账
    later = fact("e", "User likes funk.", "likes", "funk", "2025-03-01", cardinality="multi")
    funk = later.model_copy(update={"source_turn_ids": ["t-edit2"]})
    rec.reconcile(state, [(funk, vec(0, 0, 1))], corrections={"t-edit2": "a"})
    assert state.facts["e"].is_current and state.facts["a"].superseded_by == "d"
    ghost = fact("f", "User likes pop.", "likes", "pop", "2025-03-02", cardinality="multi")
    ghost = ghost.model_copy(update={"source_turn_ids": ["t-edit3"]})
    rec.reconcile(state, [(ghost, vec(0, 0, 1))], corrections={"t-edit3": "gone"})
    assert state.facts["f"].is_current


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


def test_speakers_in_other_scripts_are_separate_people(
    settings: Settings, fake_api: FakeAPI
) -> None:
    """说话人名字是西里尔文或只有 emoji：以前名字键都是 ''，所有这样的说话人是同一个实体，
    别名索引里还有 '' 这个键（任何规范化成 '' 的宾语都会指向这个人）。"""
    state = EntityState("s")
    res = resolver(settings, fake_api, {"matches": []})
    report = ResolutionReport()
    res.ensure_people(state, ["Анна", "Иван", "🙂", "🙃"], datetime(2026, 1, 10), report)
    assert report.created == 4 and len(state.entities) == 4
    assert "" not in state.alias_index
    assert state.lookup("Анна") != state.lookup("Иван")
    assert state.lookup("анна") == state.lookup("Анна") is not None
    assert state.lookup("—") is None


def test_reconcile_cost_is_not_quadratic(
    settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """遗忘与重建都按会话把整个空间重放一遍：以前每条新事实都把这个主语的全部当前事实规范化
    一遍、把它们的向量 np.vstack 一遍，n 条事实要 n² 次（9600 条要 6 分多钟）。现在候选从
    FactState 的索引里取，宾语的规范形式按文本只算一次。"""
    calls = 0
    real = reconcile_module.normalize_value

    def counting(text: str) -> str:
        nonlocal calls
        calls += 1
        return real(text)

    monkeypatch.setattr(reconcile_module, "normalize_value", counting)
    state = FactState()
    rec = reconciler(settings, fake_api, [])
    n = 300
    for i in range(n):  # 互相正交的向量：每条都是登记过的多值谓词，直接新增
        vector = np.zeros(n, dtype=np.float32)
        vector[i] = 1.0
        statement = f"User likes thing {i}."
        new = fact(f"f{i}", statement, "likes", f"thing {i}", "2025-01-01", cardinality="multi")
        rec.reconcile(state, [(new, vector)])
    assert len(state.current_of(USER)) == n
    assert calls <= 2 * n  # 以前约 n² 次（9 万次）
    assert fake_api.chat_requests == []


def test_fact_state_index_matches_a_full_scan() -> None:
    """FactState 的索引与逐条扫描当前事实的结果相同：先后顺序、同值事实里的第一条、每行的
    谓词，相似度与把向量 np.vstack 起来再相乘逐位相同。覆盖合并、取代、迟到的旧版本、同一个
    ID 换了主语、没有向量的事实，以及向量不是 float32 的主语（照原样 np.vstack）。"""
    rng = random.Random(7)
    vectors = np.random.default_rng(7)
    dim = 1024
    state = FactState()
    subjects = ["u", "mia", "odd"]
    predicates = ["likes", "lives_in", "age", "has_pet"]
    objects = ["Cats", "cats", "the cats", "Berlin", "Paris", "31", "a dog"]

    def unit(subject: str) -> np.ndarray:
        v = vectors.standard_normal(dim)
        v = v / np.linalg.norm(v)
        return v if subject == "odd" else v.astype(np.float32)  # "odd" 的向量是 float64

    def new_fact(i: int) -> Fact:
        return fact(
            f"f{i}",
            f"statement {i}",
            rng.choice(predicates),
            rng.choice(objects),
            "2025-01-01",
            negated=rng.random() < 0.2,
            subject=rng.choice(subjects),
        )

    when = datetime(2025, 6, 1)
    for i in range(700):
        current = [f for f in state.facts.values() if f.is_current]
        roll = rng.random()
        if roll < 0.55 or not current:  # 新事实（偶尔没有向量，偶尔是迟到的旧版本）
            new = new_fact(i)
            if rng.random() < 0.05:
                new = new.model_copy(update={"invalidated_at": when, "superseded_by": "x"})
            state.put(new, None if rng.random() < 0.1 else unit(new.subject_id))
        elif roll < 0.75:  # 合并：只追加溯源
            old = rng.choice(current)
            state.put(old.model_copy(update={"source_turn_ids": [*old.source_turn_ids, "t"]}))
        elif roll < 0.9:  # 取代：旧事实不再是当前的
            old = rng.choice(current)
            state.put(old.model_copy(update={"invalidated_at": when, "superseded_by": f"f{i}"}))
        else:  # 同一个 ID 写进别的内容（同一会话里陈述相同的两条事实）
            old = rng.choice(list(state.facts.values()))
            other = new_fact(i).model_copy(update={"id": old.id})
            state.put(other, unit(other.subject_id) if rng.random() < 0.7 else None)
        if i % 7:
            continue
        for subject in subjects:
            scan = [f for f in state.facts.values() if f.subject_id == subject and f.is_current]
            assert state.current_of(subject) == scan
            with_vectors = [f for f in scan if f.id in state.vectors]
            query = unit(subject)
            ids, sims = state.similarities(subject, query)
            assert ids == [f.id for f in with_vectors]
            if with_vectors:
                expected = np.vstack([state.vectors[f.id] for f in with_vectors]) @ query
                assert sims.dtype == expected.dtype and sims.tobytes() == expected.tobytes()
                for predicate in predicates:
                    rows = state.predicate_rows(subject, predicate).tolist()
                    assert rows == [f.predicate == predicate for f in with_vectors]
            for predicate in predicates:
                for obj in objects:
                    for negated in (False, True):
                        value = normalize_value(obj)
                        first = next(
                            (
                                f
                                for f in scan
                                if f.predicate == predicate
                                and normalize_value(f.object) == value
                                and f.negated == negated
                            ),
                            None,
                        )
                        assert state.first_current(subject, predicate, value, negated) == first


def test_similarity_ranking_matches_sorted() -> None:
    """相似度排序与以前对全部 (事实, 相似度) 做 sorted(key=-相似度) 再筛选的结果相同：相同的
    相似度保持原来的先后，-0.0 与 0.0 算相同，有 NaN 时照 sorted 的结果。"""
    rng = random.Random(3)
    pool = [0.9, 0.9, 0.5, 0.0, -0.0, -0.3, 0.75, float("nan")]
    for trial in range(400):
        n = rng.randint(0, 12)
        scores = [rng.choice(pool[:-1] if trial % 2 else pool) for _ in range(n)]
        wanted = [rng.random() < 0.6 for _ in range(n)]
        expected = [i for i in sorted(range(n), key=lambda i: -scores[i]) if wanted[i]]
        got = _by_similarity(np.array(scores, dtype=np.float64), np.array(wanted, dtype=bool))
        assert got == expected


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
