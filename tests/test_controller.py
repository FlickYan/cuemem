"""检索控制器（N7）：探针特征、两段式控制器的决策与 λ、训练时与运行时两套实现一致、
离线重放的各个策略、零样本 LLM 路由，以及 ask(mode="auto") 的三种控制器。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword

from cuemem.api.service import english_reason
from cuemem.config import Settings
from cuemem.core.models import Conversation
from cuemem.evals import routing as R
from cuemem.memory import MemorySystem
from cuemem.read.controller import ControllerModel, LearnedController, LLMRouter
from cuemem.read.probe import query_features

ASKED = datetime(2025, 10, 10, 20, 0)

# 一个手写的小模型：问题里有聚合线索就偏"多会话"，检索分数低就偏"与记忆无关"
TOY_MODEL: dict[str, Any] = {
    "features": ["q_aggregation", "p_round_top1"],
    "mean": [0.0, 0.0],
    "scale": [1.0, 1.0],
    "types": ["multi", "single", "none"],
    "coef": [[2.0, 0.0], [-2.0, 0.0]],
    "intercept": [0.0, 0.0],
    "gate": {"coef": [0.0, -20.0], "intercept": 5.0},
    "accuracy": {
        "multi": {"L0": 0.0, "L1": 0.5, "L2": 0.9},
        "single": {"L0": 0.0, "L1": 0.9, "L2": 0.9},
        "none": {"L0": 1.0, "L1": 1.0, "L2": 1.0},
    },
    "see_gain": {
        "multi": {"L1": 0.2, "L2": 0.2},
        "single": {"L1": 0.0, "L2": 0.0},
        "none": {"L1": 0.0, "L2": 0.0},
    },
    "cost": {"L0": 0.0003, "L1": 0.0015, "L2": 0.0035, "see": 0.0002},
    "lambda": 30.0,
}


def toy_controller(tmp_path: Path, **overrides: Any) -> LearnedController:
    path = tmp_path / "controller.json"
    path.write_text(json.dumps({**TOY_MODEL, **overrides}), encoding="utf-8")
    return LearnedController.load(path)


@pytest.fixture
def toy(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[tuple[MemorySystem, ScriptedModel]]:
    model = ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    fake_api.chat_replies = [model]
    conv = Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8"))
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    system.ingest(conv, consolidate=True)
    yield system, model
    system.close()


# ---------- 探针 ----------


def test_query_features_pick_up_cues() -> None:
    f = query_features("How many days did it take me to finish the Lego set I mentioned?")
    assert f["q_first_person"] == f["q_aggregation"] == f["q_duration"] == f["q_recall"] == 1
    assert f["q_visual"] == 0
    g = query_features("What color was the bus in the photo?")
    assert g["q_visual"] == 1 and g["q_first_person"] == 0
    h = query_features("我上次说的那家店叫什么？")
    assert h["q_first_person"] == h["q_recall"] == 1
    assert query_features("Did I buy the blue one?")["q_past_self"] == 1


def test_probe_is_one_retrieval_without_llm(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    system, model = toy
    before = dict(model.calls)
    result = system.probe.run("demo", "Max 喜欢吃什么？", ASKED)
    assert model.calls == before  # 探针不调用 LLM
    groups = {name.split("_")[0] for name in result.features}
    assert groups == {"q", "p", "m"}
    assert result.features["p_round_top1"] > 0 and result.features["m_rounds"] > 0
    assert result.candidates.ranked and result.candidates.routes[("round", "dense")]
    # 探针的候选与 L1 自己检索的一样：控制器选了 L1 时直接打包，结果相同
    packed = system.l1.pack("demo", result.candidates, system_l1_config())
    searched = system.l1.search("demo", "Max 喜欢吃什么？", ASKED, system_l1_config())
    assert [h.unit.id for h in packed.rounds] == [h.unit.id for h in searched.rounds]


def system_l1_config() -> Any:
    from cuemem.read.l1 import L1Config

    return L1Config()


# ---------- 两段式控制器 ----------


def test_learned_controller_trades_accuracy_for_cost(tmp_path: Path) -> None:
    ctrl = toy_controller(tmp_path)
    memory = {"q_aggregation": 1.0, "p_round_top1": 0.6}  # 聚合题、记忆里有强命中
    assert ctrl.decide(memory, has_images=False).level == "L2"
    assert ctrl.decide(memory, has_images=False, lam=300).level == "L1"  # λ 大：省钱
    assert ctrl.decide(memory, has_images=False, lam=2000).level == "L0"
    # 有图的空间：多会话题上再感知有增益，值得多花一点
    see = ctrl.decide(memory, has_images=True)
    assert (see.level, see.see) == ("L2", True)
    general = {"q_aggregation": 0.0, "p_round_top1": 0.0}  # 检索不到东西：与记忆无关
    decision = ctrl.decide(general, has_images=False)
    assert decision.level == "L0" and decision.scores["p_none"] > 0.99
    assert decision.reasons[0].startswith("与记忆无关")
    # 混合方案：LLM 判"与记忆无关"时用它的概率代替门控
    assert ctrl.decide(memory, has_images=False, p_none=0.95).level == "L0"


def test_outlier_features_are_clipped(tmp_path: Path) -> None:
    """分布外的特征（比如换了向量模型）被截断，不会外推出极端的概率。"""
    ctrl = toy_controller(tmp_path, clip=1.0)
    _, p_none = ctrl.model.parts({"q_aggregation": 0.0, "p_round_top1": -50.0})
    assert p_none == pytest.approx(1 / (1 + np.exp(-(5.0 + 20.0))))  # 相当于 z = −1


def synthetic_items(n: int = 90, seed: int = 0) -> list[R.Item]:
    """三类题：单点事实（L1 就够）、多会话（要 L2）、与记忆无关（L0 就够）；特征与类型相关。"""
    rng = np.random.RandomState(seed)
    items = []
    for i in range(n):
        kind = ("single", "multi", "none")[i % 3]
        features = {
            "q_aggregation": float(kind == "multi") + rng.normal(0, 0.1),
            "p_round_top1": (0.1 if kind == "none" else 0.6) + rng.normal(0, 0.05),
            "m_media": float(i % 2),
        }
        right = {
            "single": {"L0": False, "L1": True, "L2": True},
            "multi": {"L0": False, "L1": i % 2 == 0, "L2": True},
            "none": {"L0": True, "L1": True, "L2": True},
        }[kind]
        cost = {"L0": 0.0003, "L1": 0.0015, "L2": 0.0035}
        actions = {a: R.Outcome(right[a], cost[a]) for a in R.LEVELS}
        if features["m_media"]:
            for lv in ("L1", "L2"):
                actions[f"{lv}+see"] = R.Outcome(right[lv] or kind == "multi", cost[lv] + 0.0002)
        items.append(
            R.Item(
                sample_id=f"q{i}",
                dataset="toy",
                group=f"g{i}",
                question=f"question {i}",
                question_type=kind,
                rtype=kind,
                features=features,
                actions=actions,
                home="toy",
            )
        )
    return items


def test_training_and_runtime_controllers_agree(tmp_path: Path) -> None:
    """离线训练（evals/routing.py）导出的 JSON，运行时（read/controller.py）做出的选择完全相同。"""
    items = synthetic_items()
    model = R.fit_two_stage(items, ["q_aggregation", "p_round_top1", "m_media"])
    assert model.types[-1] == "none" and model.gate_coef is not None
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model.to_json(lam=20.0, meta={})), encoding="utf-8")
    runtime = LearnedController.load(path)
    offline = model.predict(items)
    for lam in (0.0, 20.0, 300.0):
        for it, proba in zip(items, offline, strict=True):
            costs = {a: model.costs.of(a) for a in proba}
            want = R.decide(proba, costs, lam, list(proba))
            got = runtime.decide(it.features, has_images=it.has_images, lam=lam)
            assert got.action == want


def test_v2_training_and_runtime_agree_with_the_image_table(tmp_path: Path) -> None:
    """v2（有图的空间查有图的准确率表）：离线导出的 JSON 与运行时的选择同样逐题一致。"""
    items = synthetic_items()
    model = R.fit_two_stage(items, ["q_aggregation", "p_round_top1", "m_media"], by_images=True)
    assert model.accuracy.image is not None
    exported = model.to_json(lam=20.0, meta={})
    assert exported["version"] == "n11-two-stage-v2" and "accuracy_images" in exported
    path = tmp_path / "model.json"
    path.write_text(json.dumps(exported), encoding="utf-8")
    runtime = LearnedController.load(path)
    offline = model.predict(items)
    for lam in (0.0, 20.0, 300.0):
        for it, proba in zip(items, offline, strict=True):
            costs = {a: model.costs.of(a) for a in proba}
            want = R.decide(proba, costs, lam, list(proba))
            got = runtime.decide(it.features, has_images=it.has_images, lam=lam)
            assert got.action == want


def test_v2_image_table_does_not_saturate(tmp_path: Path) -> None:
    """v1 把看图增益加到合并的准确率上再截断：L1、L2 都到 1.0、打平，总选便宜的 L1。
    v2 直接查有图的表，L2 + 看图确实更准时（且值这个价）就选它。"""
    single = {"single": 1.0}
    v1 = toy_controller(
        tmp_path,
        see_gain={**TOY_MODEL["see_gain"], "single": {"L1": 0.3, "L2": 0.3}},
    )
    p1 = v1.model.action_proba(single, has_images=True)
    assert p1["L1+see"] == p1["L2+see"] == 1.0
    images = {
        t: {"L0": 0.0, "L1": 0.4, "L2": 0.45, "L1+see": 0.65, "L2+see": 0.8}
        for t in ("multi", "single")
    }
    images["none"] = dict.fromkeys(("L0", "L1", "L2", "L1+see", "L2+see"), 1.0)
    v2 = toy_controller(tmp_path, accuracy_images=images)
    p2 = v2.model.action_proba(single, has_images=True)
    assert p2["L1+see"] == 0.65 and p2["L2+see"] == 0.8
    # 没有图的空间不受影响
    assert v2.model.action_proba(single, has_images=False) == v1.model.action_proba(
        single, has_images=False
    )


def test_image_floor_lifts_visual_questions_out_of_l0(tmp_path: Path) -> None:
    """保底（D-052）：有图的空间里问画面细节，门控再确信"与记忆无关"也至少 L1 + 看图。"""
    images = {
        t: {"L0": 0.0, "L1": 0.4, "L2": 0.45, "L1+see": 0.65, "L2+see": 0.8}
        for t in ("multi", "single")
    }
    images["none"] = dict.fromkeys(("L0", "L1", "L2", "L1+see", "L2+see"), 1.0)
    ctrl = toy_controller(tmp_path, accuracy_images=images, image_floor=["q_visual", "q_scene"])
    unrelated = {"q_aggregation": 0.0, "p_round_top1": 0.0}  # 门控：几乎肯定与记忆无关
    assert ctrl.decide(unrelated, has_images=True).level == "L0"
    visual = {**unrelated, "q_scene": 1.0}
    decision = ctrl.decide(visual, has_images=True)
    assert decision.action == "L1+see" and "画面细节" in decision.reasons[-1]
    assert english_reason(decision.reasons[-1]).startswith("asks about details in a photo")
    assert ctrl.decide(visual, has_images=False).level == "L0"  # 没有图的空间不保底
    # v1 的模型文件没有 image_floor：行为不变
    assert toy_controller(tmp_path).decide(visual, has_images=True).level == "L0"


def test_scene_reference_cue() -> None:
    assert query_features("What color is the metal structure on the far left?")["q_scene"] == 1
    assert query_features("Examine the readout shown in the photo.")["q_scene"] == 1
    assert query_features("How do I make sourdough bread?")["q_scene"] == 0


def test_two_stage_learns_the_types() -> None:
    items = synthetic_items(150)
    preds = R.crossfit(items, lambda tr: R.fit_two_stage(tr, ["q_aggregation", "p_round_top1"]))
    careful, cheap = R.learned(items, preds, 5.0), R.learned(items, preds, 100.0)

    def actions(policy: R.PolicyResult, kind: str, images: bool) -> set[str]:
        return {
            c.action
            for c, it in zip(policy.choices, items, strict=True)
            if it.rtype == kind and it.has_images == images
        }

    # 多会话题：没有图时走 L2；有图时"L1 + 看原图"就能答对，选这个更便宜的动作
    assert actions(careful, "multi", images=False) == {"L2"}
    assert actions(careful, "multi", images=True) == {"L1+see"}
    assert careful.accuracy == 1.0
    # λ 大：无关查询都不检索，单点事实题仍然检索
    assert actions(cheap, "none", images=False) | actions(cheap, "none", images=True) == {"L0"}
    assert actions(cheap, "single", images=False) == {"L1"}
    assert cheap.cost < careful.cost < R.fixed(items, "L2").cost


# ---------- 离线重放的策略 ----------


def tiny_item(**outcomes: tuple[bool, float, bool]) -> R.Item:
    """outcomes：动作 → （答对，成本，回答是否自认不确定）。"""
    actions = {
        a: R.Outcome(ok, cost, sufficient=not unsure) for a, (ok, cost, unsure) in outcomes.items()
    }
    return R.Item("q", "toy", "g", "q?", "single", "single", {}, actions)


def test_replayed_policies() -> None:
    hard = tiny_item(L0=(False, 1, True), L1=(False, 2, True), L2=(True, 5, False))
    easy = tiny_item(L0=(True, 1, False), L1=(True, 2, False), L2=(True, 5, False))
    confident_wrong = tiny_item(L0=(False, 1, False), L1=(True, 2, False), L2=(True, 5, False))
    items = [hard, easy, confident_wrong]
    assert [c.action for c in R.oracle(items).choices] == ["L2", "L0", "L1"]
    reactive = R.reactive(items)
    assert [(c.action, c.cost, c.escalated) for c in reactive.choices] == [
        ("L2", 7, True),  # L1 不确定 → 升级，两档的钱都花了
        ("L1", 2, False),
        ("L1", 2, False),
    ]
    cascade = R.cascade(items)
    # 模型在 L0 自信地答错时，级联就停在那里：按不确定度触发的弱点
    assert [(c.action, c.correct) for c in cascade.choices] == [
        ("L2", True),
        ("L0", True),
        ("L0", False),
    ]
    assert R.fixed(items, "L1").accuracy == pytest.approx(2 / 3)
    wins, losses, p = R.paired(R.oracle(items), R.fixed(items, "L1"))
    assert (wins, losses) == (1, 0) and p == 1.0


def test_hybrid_replaces_uncertain_gate_with_llm() -> None:
    items = synthetic_items(30)
    preds = R.crossfit(items, lambda tr: R.fit_two_stage(tr, ["q_aggregation", "p_round_top1"]))
    preds.p_none[0] = 0.5  # 门控拿不准
    llm_none = [True] * len(items)
    hybrid, asked = R.hybrid(preds, llm_none, band=(0.1, 0.9), confidence=0.95)
    assert asked[0] and hybrid.p_none[0] == 0.95
    assert sum(asked) == sum(0.1 < p < 0.9 for p in preds.p_none)


# ---------- 零样本 LLM 路由与 ask(mode="auto") ----------


def test_llm_router_levels_and_fallback(toy: tuple[MemorySystem, ScriptedModel]) -> None:
    system, model = toy
    router = LLMRouter(system.llm)
    assert router.decide("我一共养了几只狗？").level == "L2"
    assert router.decide("法国的首都是哪里？").level == "L0"
    assert router.decide("Max 喜欢吃什么？").level == "L1"
    model.router = lambda prompt: {"level": "MAYBE"}  # 不合法的输出：退回 L1
    fallback = router.decide("随便问问")
    assert fallback.level == "L1" and "路由失败" in fallback.reasons[0]


def test_ask_auto_with_learned_and_hybrid_controllers(
    toy: tuple[MemorySystem, ScriptedModel], tmp_path: Path
) -> None:
    system, model = toy
    system.learned = toy_controller(tmp_path)
    answer = system.ask("demo", "Max 喜欢吃什么？", question_time=ASKED, mode="auto")
    route = answer.details["route"]
    assert route["router"] == "learned" and "probe_ms" in route["scores"]
    assert route["level"] in ("L0", "L1", "L2")
    # λ 很大时只用常驻上下文：不检索，回答模型看不到对话片段
    reads = model.calls["read"]
    cheap = system.ask("demo", "Max 喜欢吃什么？", question_time=ASKED, mode="auto", lam=1e6)
    assert cheap.details["route"]["level"] == "L0" and cheap.hits == []
    assert model.calls["read"] == reads + 1
    assert "Retrieved conversation excerpts:\n(none)" in model.prompts["read"][-1]
    # 混合方案：门控拿不准时才问 LLM
    system.learned = toy_controller(tmp_path, gate={"coef": [0.0, 0.0], "intercept": 0.0})
    routes = model.calls["route"]
    unsure = system.ask("demo", "Max 喜欢吃什么？", question_time=ASKED, mode="auto",
                        router="hybrid")  # fmt: skip
    assert model.calls["route"] == routes + 1 and unsure.details["route"]["scores"]["asked_llm"]
    llm = system.ask("demo", "法国的首都是哪里？", question_time=ASKED, mode="auto", router="llm")
    assert llm.details["route"] == {
        "router": "llm",
        "level": "L0",
        "see": False,
        "reasons": ["llm:NONE"],
    }


def test_controller_model_file_matches_its_schema() -> None:
    """仓库里导出的控制器（scripts/n7_controller.py）能被运行时读进来，且维度一致。"""
    model = ControllerModel.load(
        Path(__file__).parents[1] / "src/cuemem/read/controller_model.json"
    )
    assert model.coef.shape == (len(model.memory_types), len(model.features))
    assert model.gate_coef is not None and model.types[-1] == "none"
    assert set(model.accuracy) == set(model.types)
    assert model.lam > 0
