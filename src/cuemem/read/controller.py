"""检索控制器（设计文档 §6.3）：决定这一问用哪一档检索——L0 只用常驻上下文、L1 快速召回、
L2 深度检索——以及要不要再看原图（再感知）。

三种实现，同一个接口（RouteDecision）：

- **RuleController**（B6r，N5 的基线）：只看问题里的线索，不调用 LLM、不看检索结果。聚合
  （"多少、几次、一共、哪些"）、时间推算与先后（"多久、哪个先、第一次、之前"）、比较（"更多、
  最多"）走 L2，其余走 L1。规则在看到 L2 的逐题结果之前写定，避免照着结果调规则；
- **LLMRouter**（B6z，N7 的对照）：零样本让便宜的模型（qwen3.8-flash）只看问题文本选一档；
- **LearnedController**（B6，N7）：两段式。先用探针特征（read/probe.py）预测这道题属于哪类推理
  （多分类逻辑回归），再从离线全信息标注的表里查各档在这类题上的准确率，按
  P̂(答对 | x, a) − λ · ĉ(a) 选动作。λ 是运行时旋钮：调大更省钱，调小更准；不用重新训练。
  模型参数是一个 JSON 文件（evals/routing.py 训练并导出），运行时只用 numpy。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from cuemem.core.trace import record_event
from cuemem.errors import ContentFilteredError, StructuredOutputError
from cuemem.llm.client import LLMClient, Message
from cuemem.read.query import COMPARE_WORDS, DURATION_WORDS, ORDER_WORDS, analyze

Level = Literal["L0", "L1", "L2"]
Router = Literal["learned", "hybrid", "rule", "llm"]
# 默认是 v3（N11 之后：v2 + 只在问到图时看图，D-058）；v2 是 N11 导出的那一版（有图的空间直接查
# 有图的准确率表），v1 是 N7 导出、N9 测试集上冻结评测的那一版。复现某个节点的数字时用当时的文件
# （配置 retrieval.controller，scripts/reproduce.py 会自动指定）
DEFAULT_MODEL = Path(__file__).with_name("controller_model.json")
V1_MODEL = Path(__file__).with_name("controller_model_v1.json")
V2_MODEL = Path(__file__).with_name("controller_model_v2.json")
# 混合方案（B6h）：门控估计的 P(none) 落在这个区间里才问 LLM；LLM 判"无关"时 P(none) 取
# HYBRID_CONFIDENCE，判"需要记忆"时取 1 − HYBRID_CONFIDENCE。两个数都是事先定的，没有在评测数据上调
HYBRID_BAND = (0.1, 0.9)
HYBRID_CONFIDENCE = 0.95


@dataclass(frozen=True)
class RouteDecision:
    level: Level
    reasons: list[str] = field(default_factory=list)
    see: bool = False  # 回答时附上原图（再感知）
    scores: dict[str, float] = field(default_factory=dict)  # 各动作的答对概率与效用（可解释）

    @property
    def action(self) -> str:
        return self.level + ("+see" if self.see else "")


class RuleController:
    def decide(self, question: str, question_time: datetime) -> RouteDecision:
        query = analyze(question, question_time)
        reasons: list[str] = []
        if query.aggregation:
            reasons.append("aggregation")
        if DURATION_WORDS.search(question):
            reasons.append("duration")
        if ORDER_WORDS.search(question):
            reasons.append("order")
        if COMPARE_WORDS.search(question):
            reasons.append("compare")
        return RouteDecision("L2" if reasons else "L1", reasons)


ROUTER_SYSTEM = """You decide how much memory retrieval a personal assistant needs before it answers a user's message. The assistant keeps a long-term memory of its past conversations with this user, including photos the user shared.

Levels:
- NONE: the message does not depend on the user's past conversations (general knowledge, how-to, creative or brainstorming requests, chit-chat). The assistant answers from its own knowledge.
- FAST: one quick search of past conversations is enough: the answer is a single fact or detail mentioned before.
- DEEP: a thorough multi-step search: combine details from several past conversations, count or list things, compare, order events in time, or work out dates and durations.

Also say whether answering needs a fresh look at photos the user shared (visual details such as colors, objects, or text in an image)."""  # noqa: E501

_ROUTER_LEVELS: dict[str, Level] = {"NONE": "L0", "FAST": "L1", "DEEP": "L2"}


class RouterOutput(BaseModel):
    level: Literal["NONE", "FAST", "DEEP"]
    look_at_images: bool = False


class LLMRouter:
    """零样本 LLM 路由（N7 的对照）：只看问题文本，不看记忆；失败时退回 L1。"""

    def __init__(self, llm: LLMClient, role: str = "planner") -> None:
        self.llm = llm
        self.role = role

    def decide(self, question: str, *, cache_scope: str | None = None) -> RouteDecision:
        messages: list[Message] = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": f"User message:\n{question}"},
        ]
        try:
            out = self.llm.structured(
                self.role, messages, RouterOutput, max_tokens=60, cache_scope=cache_scope
            ).value
        except (StructuredOutputError, ContentFilteredError) as exc:
            record_event("router_fallback", error=type(exc).__name__)
            return RouteDecision("L1", [f"路由失败（{type(exc).__name__}），默认 L1"])
        return RouteDecision(
            _ROUTER_LEVELS[out.level], [f"llm:{out.level}"], see=out.look_at_images
        )


@dataclass
class ControllerModel:
    """两段式控制器的参数（evals/routing.py 的 TwoStageModel.to_json 导出）。

    有门控时分两层：门控（二分类）估计"这一问与记忆无关"的概率 P(none | x)，推理类型的多分类
    只管记忆题；P(t | x) = (1 − P(none | x)) · P(t | x, 记忆题)。
    """

    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    types: list[str]  # 有门控时最后一个是 "none"
    coef: np.ndarray  # (记忆题类型数, 特征数)
    intercept: np.ndarray
    accuracy: dict[str, dict[str, float]]  # 推理类型 → 档 → 准确率
    see_gain: dict[str, dict[str, float]]  # 推理类型 → 档 → 再感知的增益
    cost: dict[str, float]  # 档 → 平均名义成本（美元）；"see" 是再感知多出来的成本
    lam: float  # 默认的 λ
    gate_coef: np.ndarray | None = None
    # v2：有图的空间里各动作（含再感知）的准确率；为空时退回 v1 的"加增益再截断"
    accuracy_images: dict[str, dict[str, float]] | None = None
    # v2：有图的空间里，这些问题线索之一成立时不选 L0，至少快速召回并看图（D-052）
    image_floor: list[str] | None = None
    # v3：再感知的门槛（D-058）。有图的空间里，问题带画面线索（cues 里的特征之一成立），或者图片
    # 描述的检索相似度不低于 caption_min，才考虑看图；为空时不设门槛（v2）
    see_gate: dict[str, Any] | None = None
    gate_intercept: float = 0.0
    clip: float = 5.0  # 标准化后的特征截断在 ±clip 以内（与训练时一致）
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> ControllerModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        known = {
            "features",
            "mean",
            "scale",
            "types",
            "coef",
            "intercept",
            "gate",
            "accuracy",
            "see_gain",
            "cost",
            "lambda",
            "clip",
            "accuracy_images",
            "image_floor",
            "see_gate",
        }
        gate = raw.get("gate")
        return cls(
            features=list(raw["features"]),
            mean=np.asarray(raw["mean"], dtype=float),
            scale=np.asarray(raw["scale"], dtype=float),
            types=list(raw["types"]),
            coef=np.asarray(raw["coef"], dtype=float),
            intercept=np.asarray(raw["intercept"], dtype=float),
            accuracy=raw["accuracy"],
            see_gain=raw["see_gain"],
            cost=raw["cost"],
            lam=float(raw["lambda"]),
            gate_coef=np.asarray(gate["coef"], dtype=float) if gate else None,
            gate_intercept=float(gate["intercept"]) if gate else 0.0,
            clip=float(raw.get("clip", 5.0)),
            accuracy_images=raw.get("accuracy_images"),
            image_floor=raw.get("image_floor"),
            see_gate=raw.get("see_gate"),
            meta={k: v for k, v in raw.items() if k not in known},
        )

    def with_tables(
        self,
        accuracy: dict[str, dict[str, float]],
        accuracy_images: dict[str, dict[str, float]] | None = None,
    ) -> ControllerModel:
        """换一个回答模型时重新标定（N11）：选档用的特征与回答模型无关，"这一档能答对多少"有关。
        门控与类型模型不变，只换准确率表；表里缺的类型沿用原表。"""
        merged = {t: {**row, **accuracy.get(t, {})} for t, row in self.accuracy.items()}
        images = self.accuracy_images
        if accuracy_images is not None and images is not None:
            images = {t: {**row, **accuracy_images.get(t, {})} for t, row in images.items()}
        elif accuracy_images is not None:
            images = accuracy_images
        return replace(self, accuracy=merged, accuracy_images=images)

    @property
    def memory_types(self) -> list[str]:
        return self.types[:-1] if self.gate_coef is not None else self.types

    def parts(self, features: dict[str, float]) -> tuple[dict[str, float], float]:
        """（记忆题条件下各推理类型的概率，与记忆无关的概率）。没有门控时后者为 0。"""
        x = np.array([features.get(n, 0.0) for n in self.features])
        z = np.clip((x - self.mean) / self.scale, -self.clip, self.clip)  # 分布外的输入不外推
        logits = z @ self.coef.T + self.intercept
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()
        types = {t: float(v) for t, v in zip(self.memory_types, p, strict=True)}
        if self.gate_coef is None:
            return types, 0.0
        return types, float(1.0 / (1.0 + np.exp(-(z @ self.gate_coef + self.gate_intercept))))

    def type_proba(
        self, features: dict[str, float], *, p_none: float | None = None
    ) -> dict[str, float]:
        """各推理类型的概率；p_none 可以换掉门控的估计（混合方案用 LLM 的判断）。"""
        types, gate = self.parts(features)
        none = gate if p_none is None else p_none
        if self.gate_coef is None:
            return types
        return {**{t: (1 - none) * v for t, v in types.items()}, "none": none}

    def action_proba(self, types: dict[str, float], *, has_images: bool) -> dict[str, float]:
        if has_images and self.accuracy_images is not None:
            actions = next(iter(self.accuracy_images.values()))
            return {
                a: sum(p * self.accuracy_images[t][a] for t, p in types.items()) for a in actions
            }
        out = {
            lv: sum(p * self.accuracy[t][lv] for t, p in types.items()) for lv in ("L0", "L1", "L2")
        }
        if has_images:
            for lv in ("L1", "L2"):
                gain = sum(p * self.see_gain[t][lv] for t, p in types.items())
                out[f"{lv}+see"] = min(1.0, max(0.0, out[lv] + gain))
        return out

    def about_images(self, features: dict[str, float]) -> bool:
        """这条消息是不是在问之前分享过的图（再感知的门槛）。没有门槛时总是 True。"""
        gate = self.see_gate
        if not gate:
            return True
        if any(features.get(name, 0.0) > 0 for name in gate.get("cues", [])):
            return True
        return features.get("p_caption_top1", 0.0) >= float(gate.get("caption_min", 0.0))

    def action_cost(self, action: str) -> float:
        level = action.removesuffix("+see")
        return self.cost[level] + (self.cost.get("see", 0.0) if action.endswith("+see") else 0.0)


IMAGE_FLOOR_REASON = "有图的空间里问画面细节：至少快速召回并看图"
TYPE_LABELS = {
    "single": "单点事实",
    "multi": "多会话 / 多跳",
    "temporal": "时间推理",
    "update": "知识更新",
    "preference": "偏好与建议",
    "open": "开放推断",
    "abstention": "拒答",
    "none": "与记忆无关",
}


class LearnedController:
    """B6：两段式检索控制器（N7）。features 来自记忆探针；lam 越大越省钱。"""

    def __init__(self, model: ControllerModel, lam: float | None = None) -> None:
        self.model = model
        self.lam = model.lam if lam is None else lam

    @classmethod
    def load(cls, path: Path | None = None, lam: float | None = None) -> LearnedController:
        return cls(ControllerModel.load(path or DEFAULT_MODEL), lam)

    def gate(self, features: dict[str, float]) -> float:
        """门控估计的"与记忆无关"的概率（混合方案据此决定要不要问 LLM）。"""
        return self.model.parts(features)[1]

    def decide(
        self,
        features: dict[str, float],
        *,
        has_images: bool,
        lam: float | None = None,
        p_none: float | None = None,
    ) -> RouteDecision:
        lam = self.lam if lam is None else lam
        types = self.model.type_proba(features, p_none=p_none)
        proba = self.model.action_proba(types, has_images=has_images)
        if has_images and not self.model.about_images(features):
            # 门槛（D-058）：有图的准确率表来自 MemLens，那里几乎每道题都问图，看图总是更准；
            # 照搬到真实对话，有图的空间里问预算、问尺码也会附上原图，模型会被图带偏
            proba = {a: p for a, p in proba.items() if not a.endswith("+see")}
        # 舍到 9 位小数再比较：与离线训练（evals/routing.py 的 decide）的平局规则逐位一致
        utility = {a: round(p - lam * self.model.action_cost(a), 9) for a, p in proba.items()}
        action = max(utility, key=lambda a: (utility[a], -self.model.action_cost(a)))
        top = sorted(types.items(), key=lambda kv: -kv[1])[:2]
        reasons = [f"{TYPE_LABELS.get(t, t)} {p:.0%}" for t, p in top if p >= 0.1]
        floor = self.model.image_floor
        if (
            has_images
            and floor
            and action.startswith("L0")
            and "L1+see" in proba
            and any(features.get(name, 0.0) > 0 for name in floor)
        ):
            # 保底（D-052）：门控对"问已分享照片里的细节"的问题常判成无关；在有图的空间里，
            # 问画面细节而这条消息没有附图，问的多半是之前的照片
            action = "L1+see"
            reasons.append(IMAGE_FLOOR_REASON)
        scores = {f"p:{a}": round(p, 4) for a, p in proba.items()}
        scores.update({f"u:{a}": round(u, 4) for a, u in utility.items()})
        scores["p_none"] = round(types.get("none", 0.0), 4)
        level: Level = (
            "L0" if action.startswith("L0") else "L1" if action.startswith("L1") else "L2"
        )
        return RouteDecision(level, reasons, see=action.endswith("+see"), scores=scores)
