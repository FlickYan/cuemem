"""事实对账：新事实与已有记忆比较，执行 ADD / SUPERSEDE / MERGE / NOOP（D-019）。

规则优先，只有规则判断不了的"灰区"才调用 LLM（同一会话的灰区事实合并成一次调用）：

| 情况 | 动作 | 是否调用 LLM |
|---|---|---|
| 同一会话重放出同一条事实（ID 相同） | NOOP | 否 |
| 同主语、同谓词、同宾语，或陈述几乎相同且宾语一致 | MERGE：追加溯源 | 否 |
| 否定已有的肯定事实（"不再喜欢……"） | SUPERSEDE（现实变了） | 否 |
| 登记过的多值谓词（喜欢、去过……） | ADD | 否 |
| 单值事实遇到同谓词或相似的事实，多值事实遇到高度相似的事实 | 交给 LLM 判断 | 是 |
| 其他 | ADD | 否 |

取代分两种，这正是需要两条时间轴的原因：
- **现实变了**（update，搬家、换工作）：旧事实 valid_to = 新事实的开始时间，
  invalidated_at = 得知时间；
- **认知变了**（correct，用户更正或补充细节）：旧事实只设 invalidated_at，valid_to 保持为空——
  它在现实中并没有"结束"，只是系统不再采信这个版本。

灰区里的每条新事实附上它出自的那句话（N11）：抽取出的事实只剩"Mia 31 岁"，看不出用户是在
更正（"之前说错了"）还是现实变了（过了生日），这要看原话。

时间规则：迟到的旧消息不会覆盖新状态。如果新事实的事件时间早于当前事实（例如后导入了一段
更早的对话），新事实直接作为历史版本写入，当前事实保持不变。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, field_validator

from cuemem.core.models import Fact, SupersedeReason
from cuemem.core.timeutil import day
from cuemem.core.trace import span
from cuemem.embed.base import FloatArray
from cuemem.errors import ContentFilteredError
from cuemem.llm.client import LLMClient
from cuemem.write.predicates import normalize_value, registered_cardinality

Action = Literal["add", "supersede", "merge", "noop"]

RECONCILE_SYSTEM = """You maintain a long-term memory of facts about people. New facts were just learned from a conversation. For each new fact, compare it with the existing facts listed under it and choose one action:
- "update": the new fact gives a new value for the same attribute of the same person or thing, and the two cannot both be true now (a different home city, a new job, a new count or total, a new personal best, a schedule changed for the same days). Give the replaced fact's id as "target".
- "correct": the user says the existing fact was a mistake ("sorry, I meant...", "actually it was..."), or the new fact restates the same information more precisely. A situation that changed (moving, a new job) is an "update", not a correction. Give its id as "target".
- "duplicate": the new fact says the same thing as an existing fact. Give its id as "target".
- "add": anything else, including facts on the same topic that can both be true (another project, another purchase, another event, more details about something) and facts with a different or narrower scope (weekdays vs. Fridays, at work vs. at home, one trip vs. another), even when the new fact is a recent change: "I work from home on Mondays now" does not replace "I work at the office", because the general fact still holds on the other days.
A new fact may come with the message it was taken from: use it to tell a correction of an earlier statement from a change in the world.
When unsure, choose "add". Return one decision for every new fact."""  # noqa: E501
SOURCE_CHARS = 400  # 灰区里新事实附上的原话最多这么多字符


class LLMDecision(BaseModel):
    new: int
    action: Literal["add", "update", "correct", "duplicate"] = "add"
    target: str | None = None

    @field_validator("action", mode="before")
    @classmethod
    def _action(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        return text if text in {"add", "update", "correct", "duplicate"} else "add"

    @field_validator("target", mode="before")
    @classmethod
    def _target(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().strip("[]").lower()
        return text or None


class LLMDecisions(BaseModel):
    decisions: list[LLMDecision] = []


@dataclass
class Operation:
    """对账的一条结果，记录在 trace 和任务统计里，便于审计"为什么这样处理"。"""

    action: Action
    fact_id: str
    target_id: str | None = None
    reason: SupersedeReason | None = None
    rule: str = ""  # duplicate / similar / single_valued / negation / llm / late_arrival / default


@dataclass
class FactState:
    """一个空间的事实快照（含历史版本）与当前事实的陈述向量。巩固过程中随会话逐步更新。"""

    facts: dict[str, Fact] = field(default_factory=dict)
    vectors: dict[str, FloatArray] = field(default_factory=dict)

    def current_of(self, subject_id: str) -> list[Fact]:
        return [f for f in self.facts.values() if f.subject_id == subject_id and f.is_current]


@dataclass
class ReconcileResult:
    operations: list[Operation] = field(default_factory=list)
    changed: dict[str, Fact] = field(default_factory=dict)  # 本次新增或修改过的事实
    llm_calls: int = 0

    def count(self, action: Action) -> int:
        return sum(1 for op in self.operations if op.action == action)


class Reconciler:
    def __init__(
        self,
        llm: LLMClient,
        *,
        duplicate_threshold: float = 0.93,
        conflict_threshold: float = 0.75,
        multi_conflict_threshold: float = 0.88,
        role: str = "extractor",
        fallback_model: str | None = None,
    ) -> None:
        self.llm = llm
        self.fallback_model = fallback_model
        self.duplicate_threshold = duplicate_threshold
        self.conflict_threshold = conflict_threshold  # 单值事实：相似度超过它就可能冲突
        self.multi_conflict_threshold = multi_conflict_threshold  # 多值事实要更相似才送 LLM
        self.role = role

    def reconcile(
        self,
        state: FactState,
        new_facts: Sequence[tuple[Fact, FloatArray]],
        *,
        cache_scope: str | None = None,
        sources: Mapping[str, str] | None = None,
    ) -> ReconcileResult:
        """按顺序处理一个会话的新事实，原地更新 state，返回操作列表与改动过的事实。

        sources：轮 ID → 这一轮的原话；灰区交给 LLM 时，每条新事实附上它出自的那句话。
        """
        result = ReconcileResult()
        gray: list[tuple[Fact, FloatArray, list[Fact]]] = []
        for fact, vector in new_facts:
            if fact.id in state.facts:
                result.operations.append(Operation("noop", fact.id, fact.id, rule="replay"))
                continue
            current = state.current_of(fact.subject_id)
            sims = self._similarities(state, current, vector)
            same_value = next(
                (
                    c
                    for c in current
                    if c.predicate == fact.predicate
                    and normalize_value(c.object) == normalize_value(fact.object)
                    and c.negated == fact.negated
                ),
                None,
            )
            if same_value is not None:
                self._merge(state, result, same_value, fact, rule="duplicate")
                continue
            # 陈述几乎相同、且宾语一致（或一个包含另一个）才算重复：向量对数字和专名不敏感，
            # "预算 8000 元"与"预算 6000 元"的余弦相似度可能很高，不能只看相似度就合并
            similar = next(
                (
                    c
                    for c, sim in sorted(sims, key=lambda pair: -pair[1])
                    if sim >= self.duplicate_threshold
                    and c.negated == fact.negated
                    and _same_object(c.object, fact.object)
                ),
                None,
            )
            if similar is not None:
                self._merge(state, result, similar, fact, rule="similar")
                continue
            if fact.negated:
                positive = next(
                    (
                        c
                        for c in current
                        if c.predicate == fact.predicate
                        and not c.negated
                        and normalize_value(c.object) == normalize_value(fact.object)
                    ),
                    None,
                )
                if positive is not None:
                    self._supersede(state, result, positive, fact, vector, "changed", "negation")
                    continue
            registered = registered_cardinality(fact.predicate)
            if registered == "multi":  # 登记过的多值谓词语义明确，不需要 LLM
                self._add(state, result, fact, vector, rule="multi_valued")
                continue
            # 单值事实（登记过的或模型标注的）遇到同谓词或相似的已有事实：交给 LLM 确认是否取代。
            # 不按规则直接取代：dev100 抽样显示规则取代的精确率只有约 10%（D-019）
            single = registered == "single" or fact.cardinality == "single"
            threshold = self.conflict_threshold if single else self.multi_conflict_threshold
            related = [
                c
                for c, sim in sorted(sims, key=lambda pair: -pair[1])
                if (single and c.predicate == fact.predicate) or sim >= threshold
            ][:3]
            if related:
                gray.append((fact, vector, related))
                continue
            self._add(state, result, fact, vector, rule="default")
        if gray:
            self._resolve_gray(state, result, gray, cache_scope, sources or {})
        return result

    # ---------- 动作 ----------

    def _add(
        self,
        state: FactState,
        result: ReconcileResult,
        fact: Fact,
        vector: FloatArray,
        *,
        rule: str,
        record: bool = True,
    ) -> None:
        state.facts[fact.id] = fact
        state.vectors[fact.id] = vector
        result.changed[fact.id] = fact
        if record:
            result.operations.append(Operation("add", fact.id, rule=rule))

    def _merge(
        self, state: FactState, result: ReconcileResult, existing: Fact, new: Fact, *, rule: str
    ) -> None:
        """同一信息再次出现：只追加溯源（更多轮次支持这条事实），陈述保持不变。"""
        sources = list(dict.fromkeys([*existing.source_turn_ids, *new.source_turn_ids]))
        merged = existing.model_copy(
            update={
                "source_turn_ids": sources,
                "importance": max(existing.importance, new.importance),
                "entity_ids": list(dict.fromkeys([*existing.entity_ids, *new.entity_ids])),
                "valid_from": existing.valid_from or new.valid_from,
            }
        )
        state.facts[existing.id] = merged
        result.changed[existing.id] = merged
        result.operations.append(Operation("merge", new.id, existing.id, rule=rule))

    def _supersede(
        self,
        state: FactState,
        result: ReconcileResult,
        old: Fact,
        new: Fact,
        vector: FloatArray,
        reason: SupersedeReason,
        rule: str,
    ) -> None:
        if reason == "changed" and _earlier(new, old):
            # 迟到的旧消息：新事实直接作为历史版本写入。失效的记录时间取取代它的那条事实
            # 被说出的时间，与按时间顺序处理的结果一致——处理顺序不影响最终状态
            historical = new.model_copy(
                update={
                    "valid_to": _effective(old),
                    "invalidated_at": max(new.recorded_at, old.recorded_at),
                    "superseded_by": old.id,
                    "supersede_reason": "changed",
                }
            )
            state.facts[new.id] = historical
            state.vectors[new.id] = vector
            result.changed[new.id] = historical
            result.operations.append(Operation("add", new.id, old.id, reason, "late_arrival"))
            return
        retired = old.model_copy(
            update={
                "valid_to": _effective(new) if reason == "changed" else None,
                "invalidated_at": max(new.recorded_at, old.recorded_at),
                "superseded_by": new.id,
                "supersede_reason": reason,
            }
        )
        state.facts[old.id] = retired
        result.changed[old.id] = retired
        self._add(state, result, new, vector, rule=rule, record=False)
        result.operations.append(Operation("supersede", new.id, old.id, reason, rule))

    # ---------- 灰区：交给 LLM ----------

    def _resolve_gray(
        self,
        state: FactState,
        result: ReconcileResult,
        gray: list[tuple[Fact, FloatArray, list[Fact]]],
        cache_scope: str | None,
        sources: Mapping[str, str],
    ) -> None:
        labels: dict[str, str] = {}
        blocks: list[str] = []
        for i, (fact, _, related) in enumerate(gray, start=1):
            lines = [f"New fact {i}: {fact.statement} (said on {day(fact.recorded_at)})"]
            said = " / ".join(sources[t] for t in fact.source_turn_ids if t in sources)
            said = " ".join(said.split())  # 压掉换行：提示按空行分块
            if said:
                cut = said[:SOURCE_CHARS] + ("…" if len(said) > SOURCE_CHARS else "")
                lines.append(f'From the message: "{cut}"')
            lines.append("Existing facts:")
            for existing in related:
                label = f"e{len(labels) + 1}"
                labels[label] = existing.id
                lines.append(
                    f"  [{label}] {existing.statement} (said on {day(existing.recorded_at)})"
                )
            blocks.append("\n".join(lines))
        messages = [
            {"role": "system", "content": RECONCILE_SYSTEM},
            {"role": "user", "content": "\n\n".join(blocks)},
        ]
        try:
            with span("reconcile", items=len(gray)):
                output = self.llm.structured(
                    self.role,
                    messages,
                    LLMDecisions,
                    cache_scope=cache_scope,
                    fallback_model=self.fallback_model,
                ).value
        except ContentFilteredError:  # 两家的内容审核都拒绝：退回默认动作，全部按新增处理
            for fact, vector, _ in gray:
                self._add(state, result, fact, vector, rule="llm_unavailable")
            return
        result.llm_calls += 1
        decisions = {d.new: d for d in output.decisions}
        for i, (fact, vector, related) in enumerate(gray, start=1):
            decision = decisions.get(i)
            allowed = {c.id for c in related}
            target_id = labels.get(decision.target or "") if decision else None
            target = state.facts.get(target_id) if target_id in allowed else None
            # 目标在同一批里已被别的新事实取代时，也按新增处理
            if (
                decision is None
                or decision.action == "add"
                or target is None
                or not target.is_current
            ):
                self._add(state, result, fact, vector, rule="llm")
            elif decision.action == "duplicate":
                self._merge(state, result, target, fact, rule="llm")
            elif decision.action == "update":
                self._supersede(state, result, target, fact, vector, "changed", "llm")
            else:
                self._supersede(state, result, target, fact, vector, "corrected", "llm")

    def _similarities(
        self, state: FactState, current: Sequence[Fact], vector: FloatArray
    ) -> list[tuple[Fact, float]]:
        with_vectors = [c for c in current if c.id in state.vectors]
        if not with_vectors:
            return []
        matrix = np.vstack([state.vectors[c.id] for c in with_vectors])
        sims = matrix @ vector
        return [(c, float(s)) for c, s in zip(with_vectors, sims, strict=True)]


def _same_object(a: str, b: str) -> bool:
    na, nb = normalize_value(a), normalize_value(b)
    return na == nb or bool(na and nb and (na in nb or nb in na))


def _effective(fact: Fact) -> str:
    """事实生效的时间：有事件时间用事件时间，否则用被说出的那天。"""
    return fact.valid_from or day(fact.recorded_at)


def _earlier(a: Fact, b: Fact) -> bool:
    """a 是否早于 b 生效。

    只有两条事实都带明确的事件时间时才按事件时间比较（部分日期按共同精度，'2023-05' 与
    '2023-05-07' 只比到月）；否则按被说出的先后。不能拿一条事实的事件时间去比另一条的说出时间：
    "参加了五次互助小组（2022 年的事）"比"参加了三次（2023-05 说的）"说得晚，是更新的计数，
    按事件时间比却会被误判成迟到的旧消息（dev100 的真实错误）。
    """
    if a.valid_from and b.valid_from:
        n = min(len(a.valid_from), len(b.valid_from))
        if a.valid_from[:n] != b.valid_from[:n]:
            return a.valid_from[:n] < b.valid_from[:n]
    return a.recorded_at < b.recorded_at
