"""检索控制器的离线训练与评测（N7，设计文档 §6.3）。

**全信息标注**：离线时每道题的每个动作（L0 / L1 / L2；MemLens 上另有再感知开 / 关）都跑过、
判过分，正确性与名义成本记在评测运行的 results.jsonl 里，探针特征记在 probe.jsonl 里。
所有策略（固定动作、规则、零样本 LLM 路由、学出来的控制器、反应式升级）都在这张表上重放：
选了哪个动作，就取那个动作的正确性与成本。回答都命中缓存、温度为 0，重放与真实运行一致。

**两段式控制器**（D-038）：直接拟合"每个动作答不答得对"，标签里的噪声太大——同一道题 L1、L2
的证据只差一点，回答就可能翻转（LoCoMo 单跳题上 L2 对 L1 是 +6 / −5）。所以换成更稳的标签：

1. 用探针特征预测这道题属于哪类推理（单点事实、多会话 / 多跳、时间推理、知识更新、偏好、
   开放推断、拒答；混合流里还有"与记忆无关"），多分类逻辑回归；
2. 各动作在各类题上的准确率从训练数据查表：P̂(a | x) = Σ_t P(t | x) · acc(a | t)；
3. 按 P̂(a | x) − λ · ĉ(a) 选动作，λ 是运行时的成本旋钮。

直接拟合每个动作答对概率的做法（设计文档原方案）保留为对照（direct）。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cuemem.read.controller import RuleController
from cuemem.read.probe import DONT_KNOW

FloatArray = NDArray[np.float64]

LEVELS = ("L0", "L1", "L2")
SEE = "+see"

# 数据集 → 动作 → 评测方法。MemLens 的 L1、L2 不带再感知，"+see" 是同一档再附 3 张原图
ACTION_METHODS: dict[str, dict[str, str]] = {
    "longmemeval_s": {"L0": "L0", "L1": "B4-nobm25", "L2": "B5"},
    "locomo": {"L0": "L0", "L1": "B4-nobm25", "L2": "B5"},
    "memlens_32k": {
        "L0": "L0",
        "L1": "B4-mm",
        "L1+see": "B4-see",
        "L2": "B5",
        "L2+see": "B5-see",
    },
}
ACTION_METHODS["memlens_128k"] = ACTION_METHODS["memlens_32k"]
# 作为对照的方法（有就读进来）：模型自主检索、规则控制器、零样本 LLM 路由、学出来的控制器……
REFERENCE_METHODS = (
    "B0",
    "B1",
    "B2",
    "B3",
    "B6r",
    "B6z",
    "B6",
    "B6h",
    "B6a",
    "B4-note",
    "B5-ppr",
    "ORACLE",
)

# 各数据集的题型 → 统一的推理类型（拒答题另算）。"none" 是混合流里与记忆无关的查询
REASONING_TYPES = (
    "single",
    "multi",
    "temporal",
    "update",
    "preference",
    "open",
    "abstention",
    "none",
)
TYPE_OF = {
    "single-session-user": "single",
    "single-session-assistant": "single",
    "single-session-preference": "preference",
    "multi-session": "multi",
    "temporal-reasoning": "temporal",
    "knowledge-update": "update",
    "single-hop": "single",
    "multi-hop": "multi",
    "temporal": "temporal",
    "open-domain": "open",
    "information_extraction": "single",
    "multi_session_reasoning": "multi",
    "temporal_reasoning": "temporal",
    "knowledge_update": "update",
    "answer_refusal": "abstention",
    "no-memory": "none",
}
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
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "q": ("q_",),  # 只看问题文本的线索
    "p": ("p_", "m_"),  # 只看探针反馈与记忆规模
    "qp": ("q_", "p_", "m_"),  # 两者都用（默认）
}
SHRINK = 5.0  # 查表准确率向总体均值收缩的先验强度（某类题在训练集里很少时不至于走极端）
# 标准化后的特征截断在 ±CLIP 以内：分布外的输入（换了向量模型、换了语言）不会被线性模型外推到极端
CLIP = 5.0


def reasoning_type(question_type: str, is_abstention: bool) -> str:
    if is_abstention:
        return "abstention"
    return TYPE_OF.get(question_type, "single")


def level_of(action: str) -> str:
    return action.removesuffix(SEE)


# ---------- 数据：每道题、每个动作的结果 ----------


@dataclass
class Outcome:
    correct: bool
    cost: float  # 名义成本（美元）
    sufficient: bool | None = None  # 回答模型自认证据是否充分
    dont_know: bool = False  # 回答像"不知道"
    route: str | None = None  # 路由类方法（B6r、B6z、B6）实际选了哪一档

    @property
    def unsure(self) -> bool:
        """反应式升级的触发条件：自认证据不足，或回答像"不知道"。"""
        return self.sufficient is False or self.dont_know


@dataclass
class Item:
    sample_id: str
    dataset: str
    group: str  # 交叉验证分组：同一份对话历史的题放在同一折（LoCoMo 一段对话约 130 题）
    question: str
    question_type: str
    rtype: str
    features: dict[str, float]
    actions: dict[str, Outcome]
    references: dict[str, Outcome] = field(default_factory=dict)
    home: str = ""  # 记忆空间所在的数据集（混合流的查询借用开发集的空间）

    @property
    def is_memory(self) -> bool:
        return self.rtype != "none"

    @property
    def has_images(self) -> bool:
        return self.features.get("m_media", 0.0) > 0

    def variant(self, level: str, see: bool = True) -> str:
        """某一档的动作名：空间里有图且要再感知时用带原图的版本。"""
        name = f"{level}{SEE}"
        return name if see and name in self.actions else level


def _usage_cost(row: dict[str, Any]) -> float:
    usage = row["usage"]
    if isinstance(usage, str):  # 早期结果把用量存成了 repr 字符串
        import ast

        usage = ast.literal_eval(usage)
    return float(usage["cost"])


def _outcome(row: dict[str, Any]) -> Outcome:
    details = row.get("details") or {}
    route = details.get("route", {}).get("level") if isinstance(details, dict) else None
    return Outcome(
        correct=bool(row["correct"]),
        cost=_usage_cost(row),
        sufficient=row.get("evidence_sufficient"),
        dont_know=bool(DONT_KNOW.search(row.get("response") or "")),
        route=route,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # 被中断的最后一行
            continue
    return rows


def load_items(run_dir: Path) -> list[Item]:
    """读一次评测运行：每道题各动作的结果 + 探针特征。缺动作或缺特征的题跳过。"""
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    dataset = info["config"]["dataset"]
    methods = ACTION_METHODS[dataset]
    rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(run_dir / "results.jsonl"):
        rows[row["sample_id"]][row["method"]] = row
    probes = {p["sample_id"]: p for p in _read_jsonl(run_dir / "probe.jsonl")}
    items = []
    for sample_id, by_method in rows.items():
        probe = probes.get(sample_id)
        if probe is None or any(m not in by_method for m in methods.values()):
            continue
        first = by_method[methods["L1"]]
        group = f"{dataset}:{probe['space_key']}"
        items.append(
            Item(
                sample_id=sample_id,
                dataset=dataset,
                group=group,
                question=first["question"],
                question_type=first["question_type"],
                rtype=reasoning_type(first["question_type"], bool(first["is_abstention"])),
                features=probe["features"],
                actions={a: _outcome(by_method[m]) for a, m in methods.items()},
                references={m: _outcome(by_method[m]) for m in REFERENCE_METHODS if m in by_method},
                home=dataset,
            )
        )
    return sorted(items, key=lambda it: it.sample_id)


def load_mixed_items(run_dir: Path, memory: Sequence[Item]) -> list[Item]:
    """混合流里与记忆无关的查询（scripts/n7_mixed.py 构建）：推理类型记为 "none"。

    它们不需要记忆：假设任何一档都能答好，各档成本取所在数据集记忆题的平均值；回答模型对它们
    不会自称"不知道"（对反应式升级与级联最有利的假设）。难负例与它对应的记忆题放在同一组，
    交叉验证时不会一个在训练集、一个在测试集。
    """
    groups = {it.sample_id: it.group for it in memory}
    by_dataset: dict[str, list[Item]] = defaultdict(list)
    for it in memory:
        by_dataset[it.dataset].append(it)
    mean_cost = {
        ds: {a: float(np.mean([it.actions[a].cost for it in its])) for a in its[0].actions}
        for ds, its in by_dataset.items()
    }
    items = []
    for row in _read_jsonl(run_dir / "probe.jsonl"):
        dataset = row["space_dataset"]
        if dataset not in mean_cost:
            continue
        paired = row.get("paired", "")
        group = (
            groups.get(paired, row["sample_id"]) if row["source"] == "topic" else row["sample_id"]
        )
        items.append(
            Item(
                sample_id=row["sample_id"],
                dataset=f"mixed:{row['source']}",
                group=group,
                question=row["question"],
                question_type="no-memory",
                rtype="none",
                features=row["features"],
                home=dataset,
                actions={
                    a: Outcome(True, c, sufficient=True) for a, c in mean_cost[dataset].items()
                },
            )
        )
    return sorted(items, key=lambda it: it.sample_id)


def feature_names(items: Sequence[Item], feature_set: str = "qp") -> list[str]:
    prefixes = FEATURE_SETS[feature_set]
    names = {n for it in items for n in it.features if n.startswith(prefixes)}
    return sorted(names)


def matrix(items: Sequence[Item], names: Sequence[str]) -> FloatArray:
    return np.array([[it.features.get(n, 0.0) for n in names] for it in items], dtype=float)


# ---------- 控制器：两段式（推理类型 → 查表）与直接拟合 ----------


@dataclass
class CostTable:
    level: dict[str, float]  # 各档的平均名义成本
    see: float  # 再感知多出来的成本（只在有图的空间里）

    def of(self, action: str) -> float:
        return self.level[level_of(action)] + (self.see if action.endswith(SEE) else 0.0)


def cost_table(items: Sequence[Item]) -> CostTable:
    level = {lv: float(np.mean([it.actions[lv].cost for it in items])) for lv in LEVELS}
    extra = [
        it.actions[f"{lv}{SEE}"].cost - it.actions[lv].cost
        for it in items
        for lv in ("L1", "L2")
        if f"{lv}{SEE}" in it.actions
    ]
    return CostTable(level, float(np.mean(extra)) if extra else 0.0)


@dataclass
class AccuracyTable:
    """各类题上各档的准确率（向总体收缩），以及再感知在各类题上的增益（只来自有图的数据）。"""

    level: dict[str, dict[str, float]]  # 推理类型 → 档 → 准确率
    see_gain: dict[str, dict[str, float]]  # 推理类型 → 档 → 再感知的增益


def accuracy_table(items: Sequence[Item], types: Sequence[str] = REASONING_TYPES) -> AccuracyTable:
    level: dict[str, dict[str, float]] = {}
    see_gain: dict[str, dict[str, float]] = {}
    # 收缩的目标是记忆题的总体均值：无关查询各档都算"答对"，混进来会把小类的 L0 准确率拉高
    base = [it for it in items if it.is_memory] or list(items)
    overall = {lv: float(np.mean([it.actions[lv].correct for it in base])) for lv in LEVELS}
    gains_all = {
        lv: [
            float(it.actions[f"{lv}{SEE}"].correct) - float(it.actions[lv].correct)
            for it in base
            if f"{lv}{SEE}" in it.actions
        ]
        for lv in ("L1", "L2")
    }
    see_overall = {lv: float(np.mean(g)) if g else 0.0 for lv, g in gains_all.items()}
    for t in types:
        members = [it for it in items if it.rtype == t]
        # 与记忆无关的查询按假设各档都答对：直接取均值，不向记忆题的总体收缩
        shrink = 0.0 if t == "none" and members else SHRINK
        level[t] = {
            lv: (sum(it.actions[lv].correct for it in members) + shrink * overall[lv])
            / (len(members) + shrink)
            for lv in LEVELS
        }
        see_gain[t] = {}
        for lv in ("L1", "L2"):
            gains = [
                float(it.actions[f"{lv}{SEE}"].correct) - float(it.actions[lv].correct)
                for it in members
                if f"{lv}{SEE}" in it.actions
            ]
            if t == "none":  # 无关查询看不看图都一样
                see_gain[t][lv] = 0.0
                continue
            see_gain[t][lv] = (sum(gains) + SHRINK * see_overall[lv]) / (len(gains) + SHRINK)
    return AccuracyTable(level, see_gain)


def combine(
    type_rows: FloatArray, types: Sequence[str], table: AccuracyTable, has_images: Sequence[bool]
) -> list[dict[str, float]]:
    """第二段：P̂(a | x) = Σ_t P(t | x) · acc(a | t)；有图的空间另算再感知的两个动作。"""
    out = []
    for row, images in zip(type_rows, has_images, strict=True):
        p = {
            lv: float(sum(row[i] * table.level[t][lv] for i, t in enumerate(types)))
            for lv in LEVELS
        }
        if images:
            for lv in ("L1", "L2"):
                gain = float(sum(row[i] * table.see_gain[t][lv] for i, t in enumerate(types)))
                p[f"{lv}{SEE}"] = min(1.0, max(0.0, p[lv] + gain))
        out.append(p)
    return out


@dataclass
class TwoStageModel:
    """两段式控制器的全部参数：可以导出成 JSON，运行时只需要 numpy（read/controller.py）。

    有无关查询参与训练时分两层：门控（二分类）先估计"这一问与记忆无关"的概率，推理类型的多分类
    只在记忆题上训练，P(t | x) = (1 − P(none | x)) · P(t | x, 记忆题)。两个问题分开学，
    无关查询的数量（训练时是记忆题的两倍）不会挤占推理类型的分类。
    """

    features: list[str]
    mean: FloatArray
    scale: FloatArray
    types: list[str]  # 门控时最后一个是 "none"
    coef: FloatArray  # (记忆题类型数, 特征数)
    intercept: FloatArray
    accuracy: AccuracyTable
    costs: CostTable
    gate_coef: FloatArray | None = None
    gate_intercept: float = 0.0

    def type_proba(self, x: FloatArray) -> FloatArray:
        z = np.clip((x - self.mean) / self.scale, -CLIP, CLIP)
        logits = z @ self.coef.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        if self.gate_coef is None:
            return np.asarray(p, dtype=float)
        none = 1.0 / (1.0 + np.exp(-(z @ self.gate_coef + self.gate_intercept)))
        return np.asarray(np.hstack([p * (1 - none)[:, None], none[:, None]]), dtype=float)

    def predict_types(self, items: Sequence[Item]) -> FloatArray:
        return self.type_proba(matrix(items, self.features))

    def predict(self, items: Sequence[Item]) -> list[dict[str, float]]:
        rows = self.predict_types(items)
        return combine(rows, self.types, self.accuracy, [it.has_images for it in items])

    def predict_parts(
        self, items: Sequence[Item]
    ) -> tuple[FloatArray, list[dict[str, float]]] | None:
        return _parts(self.predict_types(items), self.types, self.accuracy, items)

    def to_json(self, *, lam: float, meta: dict[str, Any]) -> dict[str, Any]:
        def rounded(table: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
            return {
                t: {k: round(v, 4) for k, v in row.items()}
                for t, row in table.items()
                if t in self.types
            }

        gate = None
        if self.gate_coef is not None:
            gate = {
                "coef": [round(float(v), 6) for v in self.gate_coef],
                "intercept": round(self.gate_intercept, 6),
            }
        return {
            "version": "n7-two-stage-v1",
            "clip": CLIP,
            "features": self.features,
            "mean": [round(float(v), 6) for v in self.mean],
            "scale": [round(float(v), 6) for v in self.scale],
            "types": self.types,
            "coef": [[round(float(v), 6) for v in row] for row in self.coef],
            "intercept": [round(float(v), 6) for v in self.intercept],
            "gate": gate,
            "accuracy": rounded(self.accuracy.level),
            "see_gain": rounded(self.accuracy.see_gain),
            "cost": {
                **{lv: round(v, 6) for lv, v in self.costs.level.items()},
                "see": round(self.costs.see, 6),
            },
            "lambda": lam,
            **meta,
        }


def _standardize(x: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return mean, scale


def _softmax_form(model: Any) -> tuple[list[str], FloatArray, FloatArray]:
    types = [str(t) for t in model.classes_]
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(model.intercept_, dtype=float)
    if len(types) == 2:  # 二分类时 sklearn 只给一行系数：补成两类的 softmax 形式
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.array([0.0, intercept[0]])
    return types, coef, intercept


def fit_two_stage(
    items: Sequence[Item],
    names: Sequence[str],
    *,
    c: float = 0.3,
    c_gate: float = 1.0,
    label: Callable[[Item], str] | None = None,
    weights: Sequence[float] | None = None,
) -> TwoStageModel:
    """label 可以换掉推理类型（消融：只分"深 / 浅"两类）。

    weights 是样本权重：少量目标场景的数据加了权，才不会被其他场景的大量数据淹没。
    """
    from sklearn.linear_model import LogisticRegression

    x = matrix(items, names)
    mean, scale = _standardize(x)
    z = np.clip((x - mean) / scale, -CLIP, CLIP)
    labels = [label(it) if label else it.rtype for it in items]
    w = np.ones(len(items)) if weights is None else np.asarray(weights, dtype=float)
    is_none = np.array([lb == "none" for lb in labels])
    gate_coef: FloatArray | None = None
    gate_intercept = 0.0
    if is_none.any() and not is_none.all():
        gate = LogisticRegression(C=c_gate, max_iter=5000).fit(z, is_none, sample_weight=w)
        gate_coef = np.asarray(gate.coef_[0], dtype=float)
        gate_intercept = float(gate.intercept_[0])
        z_types, w_types = z[~is_none], w[~is_none]
        type_labels = [lb for lb, n in zip(labels, is_none, strict=True) if not n]
    else:
        z_types, w_types, type_labels = z, w, labels
    model = LogisticRegression(C=c, max_iter=5000).fit(z_types, type_labels, sample_weight=w_types)
    types, coef, intercept = _softmax_form(model)
    if gate_coef is not None:
        types = [*types, "none"]
    relabeled = [_relabel(it, lb) for it, lb in zip(items, labels, strict=True)]
    return TwoStageModel(
        features=list(names),
        mean=mean,
        scale=scale,
        types=types,
        coef=coef,
        intercept=intercept,
        accuracy=accuracy_table(relabeled, types),
        costs=cost_table(items),
        gate_coef=gate_coef,
        gate_intercept=gate_intercept,
    )


def _relabel(item: Item, rtype: str) -> Item:
    if item.rtype == rtype:
        return item
    return Item(
        item.sample_id,
        item.dataset,
        item.group,
        item.question,
        item.question_type,
        rtype,
        item.features,
        item.actions,
        item.references,
        item.home,
    )


@dataclass
class TextModel:
    """对照：只看问题文本（词袋 TF-IDF + 逻辑回归），结构与两段式相同（门控 + 推理类型）。"""

    vectorizer: Any
    gate: Any | None
    classifier: Any
    types: list[str]
    accuracy: AccuracyTable
    costs: CostTable

    def predict_types(self, items: Sequence[Item]) -> FloatArray:
        x = self.vectorizer.transform([it.question for it in items])
        p = np.asarray(self.classifier.predict_proba(x), dtype=float)
        if self.gate is None:
            return p
        none = np.asarray(self.gate.predict_proba(x)[:, 1], dtype=float)
        return np.asarray(np.hstack([p * (1 - none)[:, None], none[:, None]]), dtype=float)

    def predict(self, items: Sequence[Item]) -> list[dict[str, float]]:
        rows = self.predict_types(items)
        return combine(rows, self.types, self.accuracy, [it.has_images for it in items])

    def predict_parts(
        self, items: Sequence[Item]
    ) -> tuple[FloatArray, list[dict[str, float]]] | None:
        return _parts(self.predict_types(items), self.types, self.accuracy, items)


def _parts(
    rows: FloatArray, types: Sequence[str], table: AccuracyTable, items: Sequence[Item]
) -> tuple[FloatArray, list[dict[str, float]]] | None:
    """门控模型的两部分：P(none | x)，以及"是记忆题"条件下各动作的答对概率。"""
    if not types or types[-1] != "none":
        return None
    p_none = rows[:, -1]
    memory = rows[:, :-1] / np.clip(1 - p_none, 1e-9, None)[:, None]
    return p_none, combine(memory, types[:-1], table, [it.has_images for it in items])


def fit_text(items: Sequence[Item], *, c: float = 3.0) -> TextModel:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x = vectorizer.fit_transform([it.question for it in items])
    is_none = np.array([not it.is_memory for it in items])
    gate = None
    if is_none.any() and not is_none.all():
        gate = LogisticRegression(C=c, max_iter=5000).fit(x, is_none)
        idx = np.where(~is_none)[0]
        classifier = LogisticRegression(C=c, max_iter=5000).fit(
            x[idx], [items[i].rtype for i in idx]
        )
        types = [*(str(t) for t in classifier.classes_), "none"]
    else:
        classifier = LogisticRegression(C=c, max_iter=5000).fit(x, [it.rtype for it in items])
        types = [str(t) for t in classifier.classes_]
    return TextModel(
        vectorizer, gate, classifier, types, accuracy_table(items, types), cost_table(items)
    )


@dataclass
class DirectModel:
    """对照：每一档各拟合一个"答对概率"的逻辑回归（设计文档 §6.3 原方案）；再感知用查表增益。"""

    features: list[str]
    mean: FloatArray
    scale: FloatArray
    models: dict[str, Any]
    costs: CostTable
    two_stage: TwoStageModel  # 只用它来算再感知的增益

    def predict(self, items: Sequence[Item]) -> list[dict[str, float]]:
        z = np.clip((matrix(items, self.features) - self.mean) / self.scale, -CLIP, CLIP)
        level = {lv: m.predict_proba(z)[:, 1] for lv, m in self.models.items()}
        base = self.two_stage.predict(items)
        out = []
        for i, it in enumerate(items):
            p = {lv: float(level[lv][i]) for lv in LEVELS}
            if it.has_images:
                for lv in ("L1", "L2"):
                    gain = base[i][f"{lv}{SEE}"] - base[i][lv]
                    p[f"{lv}{SEE}"] = min(1.0, max(0.0, p[lv] + gain))
            out.append(p)
        return out


def fit_direct(items: Sequence[Item], names: Sequence[str], *, c: float = 0.1) -> DirectModel:
    from sklearn.linear_model import LogisticRegression

    x = matrix(items, names)
    mean, scale = _standardize(x)
    z = np.clip((x - mean) / scale, -CLIP, CLIP)
    models = {}
    for lv in LEVELS:
        y = [int(it.actions[lv].correct) for it in items]
        if len(set(y)) < 2:  # 训练集里这一档全对或全错：常数模型
            models[lv] = _Constant(float(y[0]))
            continue
        models[lv] = LogisticRegression(C=c, max_iter=5000).fit(z, y)
    return DirectModel(
        features=list(names),
        mean=mean,
        scale=scale,
        models=models,
        costs=cost_table(items),
        two_stage=fit_two_stage(items, names),
    )


@dataclass
class _Constant:
    p: float

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return np.tile([1 - self.p, self.p], (len(x), 1))


# ---------- 交叉拟合：每道题的预测都来自没见过它的模型 ----------

Fit = Callable[[Sequence[Item]], Any]


@dataclass
class Predictions:
    """每道题各动作的答对概率与成本估计（交叉拟合，多次重复取平均）。"""

    proba: list[dict[str, float]]
    costs: list[dict[str, float]]
    type_proba: list[dict[str, float]] = field(default_factory=list)
    # 门控模型才有：P(none | x)、记忆题条件下各动作的答对概率、无关查询各动作的答对概率
    p_none: list[float] = field(default_factory=list)
    proba_memory: list[dict[str, float]] = field(default_factory=list)
    proba_none: list[dict[str, float]] = field(default_factory=list)

    def with_none(self, p_none: Sequence[float]) -> Predictions:
        """换掉"与记忆无关"的概率（混合方案：门控拿不准时用 LLM 的判断），重算各动作的答对概率。"""
        proba = []
        for p, mem, none in zip(p_none, self.proba_memory, self.proba_none, strict=True):
            proba.append({a: p * none.get(a, 1.0) + (1 - p) * v for a, v in mem.items()})
        return Predictions(
            proba,
            self.costs,
            self.type_proba,
            list(p_none),
            self.proba_memory,
            self.proba_none,
        )


def group_folds(
    items: Sequence[Item], n_folds: int, seed: int
) -> list[tuple[list[int], list[int]]]:
    """按分组随机分折：同一份对话历史的题在同一折里。"""
    rng = np.random.RandomState(seed)
    groups = sorted({it.group for it in items})
    order = list(rng.permutation(len(groups)))
    fold_of = {groups[g]: i % n_folds for i, g in enumerate(order)}
    folds = []
    for k in range(n_folds):
        test = [i for i, it in enumerate(items) if fold_of[it.group] == k]
        train = [i for i, it in enumerate(items) if fold_of[it.group] != k]
        if test and train:
            folds.append((train, test))
    return folds


def crossfit(
    items: Sequence[Item],
    fit: Fit,
    *,
    n_folds: int = 5,
    repeats: int = 5,
    seed: int = 0,
    splits: Sequence[tuple[list[int], list[int]]] | None = None,
) -> Predictions:
    """splits 给定时只按它分一次（跨数据集：在两个数据集上训练、第三个上测试）。"""
    proba: list[dict[str, float]] = [defaultdict(float) for _ in items]
    costs: list[dict[str, float]] = [defaultdict(float) for _ in items]
    types: list[dict[str, float]] = [defaultdict(float) for _ in items]
    p_none = [0.0] * len(items)
    memory: list[dict[str, float]] = [defaultdict(float) for _ in items]
    none_acc: list[dict[str, float]] = [defaultdict(float) for _ in items]
    gated = True
    rounds = (
        [list(splits)]
        if splits is not None
        else [group_folds(items, n_folds, seed + r) for r in range(repeats)]
    )
    for folds in rounds:
        weight = 1.0 / len(rounds)
        for train, test in folds:
            model = fit([items[i] for i in train])
            test_items = [items[i] for i in test]
            predicted = model.predict(test_items)
            type_rows = model.predict_types(test_items) if hasattr(model, "predict_types") else None
            parts = model.predict_parts(test_items) if hasattr(model, "predict_parts") else None
            gated = gated and parts is not None
            if parts is not None:
                for j, i in enumerate(test):
                    p_none[i] += weight * float(parts[0][j])
                    for action, v in parts[1][j].items():
                        memory[i][action] += weight * v
                    for action in parts[1][j]:
                        level = model.accuracy.level["none"][level_of(action)]
                        none_acc[i][action] += weight * level
            for j, i in enumerate(test):
                for action, p in predicted[j].items():
                    proba[i][action] += weight * p
                for action in items[i].actions:
                    costs[i][action] += weight * model.costs.of(action)
                if type_rows is not None:
                    for k, t in enumerate(model.types):
                        types[i][t] += weight * float(type_rows[j][k])
    if not gated:
        return Predictions(
            [dict(p) for p in proba], [dict(c) for c in costs], [dict(t) for t in types]
        )
    return Predictions(
        [dict(p) for p in proba],
        [dict(c) for c in costs],
        [dict(t) for t in types],
        p_none,
        [dict(m) for m in memory],
        [dict(n) for n in none_acc],
    )


# ---------- 策略与重放 ----------


@dataclass
class Choice:
    action: str  # 最终用来回答的动作
    correct: bool
    cost: float  # 实际花费：反应式升级时是两档之和
    escalated: bool = False


@dataclass
class PolicyResult:
    name: str
    choices: list[Choice]
    lam: float | None = None

    @property
    def accuracy(self) -> float:
        return float(np.mean([c.correct for c in self.choices])) if self.choices else 0.0

    @property
    def cost(self) -> float:
        return float(np.mean([c.cost for c in self.choices])) if self.choices else 0.0

    def share(self, level: str) -> float:
        return float(np.mean([level_of(c.action) == level for c in self.choices]))

    @property
    def escalation(self) -> float:
        return float(np.mean([c.escalated for c in self.choices]))

    def subset(self, mask: Sequence[bool]) -> PolicyResult:
        return PolicyResult(
            self.name, [c for c, m in zip(self.choices, mask, strict=True) if m], self.lam
        )


def take(item: Item, action: str) -> Choice:
    outcome = item.actions[action]
    return Choice(action, outcome.correct, outcome.cost)


def fixed(items: Sequence[Item], level: str, *, see: bool = True) -> PolicyResult:
    return PolicyResult(f"always-{level}", [take(it, it.variant(level, see)) for it in items])


def oracle(items: Sequence[Item]) -> PolicyResult:
    """事后挑最便宜且答对的动作（都答错时付最便宜的）：上限参照，会"吃进"回答的随机翻转。"""
    choices = []
    for it in items:
        right = [a for a, o in it.actions.items() if o.correct]
        pool = right or list(it.actions)
        action = min(pool, key=lambda a: (it.actions[a].cost, a))
        choices.append(take(it, action))
    return PolicyResult("oracle", choices)


def type_oracle(items: Sequence[Item], deep: Sequence[str] = ("multi", "temporal")) -> PolicyResult:
    """已知真实题型时的路由：多会话 / 多跳与时间题走 L2，其余走 L1（不可部署，参照用）。"""
    return PolicyResult(
        "type-oracle",
        [take(it, it.variant("L2" if it.rtype in deep else "L1")) for it in items],
    )


def rule(items: Sequence[Item]) -> PolicyResult:
    """B6r 的规则（N5）：问题里有聚合、时长、先后、比较线索就走 L2；MemLens 上带再感知。"""
    controller = RuleController()
    now = datetime(2026, 1, 1)  # 规则只看线索词，与提问时间无关
    choices = []
    for it in items:
        level = controller.decide(it.question, now).level
        choices.append(take(it, it.variant(level)))
    return PolicyResult("B6r", choices)


def reactive(items: Sequence[Item]) -> PolicyResult:
    """反应式升级：先走 L1，回答模型自认证据不足或说"不知道"时再走 L2（两档的钱都花了）。"""
    choices = []
    for it in items:
        first = it.variant("L1")
        if it.actions[first].unsure:
            second = take(it, it.variant("L2"))
            second.cost += it.actions[first].cost
            second.escalated = True
            choices.append(second)
        else:
            choices.append(take(it, first))
    return PolicyResult("reactive", choices)


def cascade(items: Sequence[Item]) -> PolicyResult:
    """按模型不确定度逐级升级：L0 → L1 → L2，每一级"不确定"才往下走（FLARE 式触发的离线版）。"""
    choices = []
    for it in items:
        spent = 0.0
        for level in LEVELS:
            action = it.variant(level)
            outcome = it.actions[action]
            spent += outcome.cost
            if level == "L2" or not outcome.unsure:
                choices.append(Choice(action, outcome.correct, spent, escalated=level != "L0"))
                break
    return PolicyResult("cascade", choices)


def reference(items: Sequence[Item], method: str) -> PolicyResult | None:
    """直接用某个对照方法的真实运行结果（B3、B6z、B6……）。"""
    if not all(method in it.references for it in items):
        return None
    choices = []
    for it in items:
        o = it.references[method]
        choices.append(Choice(o.route or method, o.correct, o.cost))
    return PolicyResult(method, choices)


def decide(
    proba: dict[str, float], costs: dict[str, float], lam: float, allowed: Sequence[str]
) -> str:
    """效用最大化：P̂(a) − λ · ĉ(a)；平局时选便宜的（先舍到 9 位小数，浮点误差不影响平局）。"""
    return max(allowed, key=lambda a: (round(proba[a] - lam * costs[a], 9), -costs[a]))


def hybrid(
    preds: Predictions,
    llm_none: Sequence[bool],
    *,
    band: tuple[float, float] = (0.1, 0.9),
    confidence: float = 0.95,
) -> tuple[Predictions, list[bool]]:
    """混合方案：门控拿不准（P(none) 落在 band 里）时才问零样本 LLM 路由器，把它的判断
    （无关 / 需要记忆）换成固定的概率（confidence / 1 − confidence）代替门控的估计。

    返回新的预测，以及每条查询是否调用了 LLM。confidence 是事先定的，不在评测数据上调。
    """
    asked = [band[0] < p < band[1] for p in preds.p_none]
    p_none = [
        (confidence if none else 1 - confidence) if ask else p
        for p, ask, none in zip(preds.p_none, asked, llm_none, strict=True)
    ]
    return preds.with_none(p_none), asked


def learned(
    items: Sequence[Item],
    preds: Predictions,
    lam: float,
    *,
    name: str = "learned",
    escalate: bool = False,
    extra: Sequence[float] | None = None,
) -> PolicyResult:
    """学出来的控制器在 λ 下的选择；escalate=True 时叠加反应式升级（选了 L1 且回答不确定 → L2）。

    extra：每条查询额外的成本（混合方案里调用 LLM 路由器的那一次）。
    """
    choices = []
    extra = extra or [0.0] * len(items)
    for it, proba, costs, more in zip(items, preds.proba, preds.costs, extra, strict=True):
        allowed = [a for a in it.actions if a in proba]
        action = decide(proba, costs, lam, allowed)
        choice = take(it, action)
        if escalate and level_of(action) == "L1" and it.actions[action].unsure:
            target = "L2" + (SEE if action.endswith(SEE) else "")
            upgraded = take(it, target)
            upgraded.cost += choice.cost
            upgraded.escalated = True
            choice = upgraded
        choice.cost += more
        choices.append(choice)
    return PolicyResult(name, choices, lam)


LAMBDAS = (0.0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000, 3000)


def sweep(
    items: Sequence[Item],
    preds: Predictions,
    *,
    name: str,
    lams: Sequence[float] = LAMBDAS,
    escalate: bool = False,
    extra: Sequence[float] | None = None,
) -> list[PolicyResult]:
    return [learned(items, preds, lam, name=name, escalate=escalate, extra=extra) for lam in lams]


def pareto_front(points: Sequence[PolicyResult]) -> list[PolicyResult]:
    """按成本排序，只保留"没有更便宜且更准"的点。"""
    front: list[PolicyResult] = []
    for p in sorted(points, key=lambda r: (r.cost, -r.accuracy)):
        if not front or p.accuracy > front[-1].accuracy + 1e-12:
            front.append(p)
    return front


def area_under_curve(points: Sequence[PolicyResult], low: float, high: float) -> float:
    """成本区间 [low, high] 上帕累托前沿（阶梯）的平均准确率：比较两条曲线的单个数字。"""
    front = pareto_front(points)
    if not front:
        return 0.0
    grid = np.linspace(low, high, 200)
    best = []
    for g in grid:
        affordable = [p.accuracy for p in front if p.cost <= g + 1e-15]
        best.append(max(affordable) if affordable else 0.0)
    return float(np.mean(best))


# ---------- 统计 ----------


def sign_test(wins: int, losses: int) -> float:
    """双侧精确符号检验的 p 值。"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(a: PolicyResult, b: PolicyResult) -> tuple[int, int, float]:
    """a 对 b：a 对 b 错的题数、a 错 b 对的题数、符号检验 p 值。"""
    wins = sum(1 for x, y in zip(a.choices, b.choices, strict=True) if x.correct and not y.correct)
    losses = sum(
        1 for x, y in zip(a.choices, b.choices, strict=True) if not x.correct and y.correct
    )
    return wins, losses, sign_test(wins, losses)


def routing_agreement(policy: PolicyResult, items: Sequence[Item]) -> float:
    """与"最便宜且答对的动作"一致的比例（只算至少有一个动作答对的题）。"""
    best = oracle(items)
    pairs = [
        (c.action, o.action)
        for c, o, it in zip(policy.choices, best.choices, items, strict=True)
        if any(x.correct for x in it.actions.values())
    ]
    return float(np.mean([level_of(a) == level_of(b) for a, b in pairs])) if pairs else 0.0


def type_confusion(items: Sequence[Item], preds: Predictions) -> dict[str, Counter[str]]:
    """真实推理类型 → 预测最可能的类型的计数（看类型分类器错在哪）。"""
    out: dict[str, Counter[str]] = defaultdict(Counter)
    for it, tp in zip(items, preds.type_proba, strict=True):
        if tp:
            out[it.rtype][max(tp, key=lambda t: tp[t])] += 1
    return out
