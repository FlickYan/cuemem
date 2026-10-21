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

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
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


ValueKey = tuple[str, str, str, bool]  # （主语，谓词，宾语的规范形式，是否否定）


@dataclass
class FactState:
    """一个空间的事实快照（含历史版本）与当前事实的陈述向量。巩固过程中随会话逐步更新。

    另有几个随写入更新的索引，对账查候选时不用每条新事实都扫一遍全部事实（以前遗忘与重建的
    重放在 CPU 上是平方级）：每个主语的当前事实（按 facts 的插入顺序）与它们的向量矩阵、
    按（主语，谓词，规范化的宾语，否定）分组的当前事实、宾语规范形式的备忘。查出来的候选、
    先后与相似度都与逐条扫描相同。所以 facts 与 vectors 只能经 put 改动。
    """

    facts: dict[str, Fact] = field(default_factory=dict)
    vectors: dict[str, FloatArray] = field(default_factory=dict)
    _pos: dict[str, int] = field(default_factory=dict, repr=False)  # 事实 ID → 在 facts 里的位置
    _subjects: dict[str, _Current] = field(default_factory=dict, repr=False)
    _values: dict[ValueKey, dict[str, int]] = field(default_factory=dict, repr=False)
    _codes: dict[str, int] = field(default_factory=dict, repr=False)  # 谓词 → 编号
    _norms: dict[str, str] = field(default_factory=dict, repr=False)  # 宾语 → normalize_value

    def put(self, fact: Fact, vector: FloatArray | None = None) -> None:
        """写入或替换一条事实；vector 不为 None 时同时设它的陈述向量。索引随之更新。"""
        old = self.facts.get(fact.id)
        pos = self._pos.setdefault(fact.id, len(self._pos))  # 替换不改变在字典里的位置
        self.facts[fact.id] = fact
        if vector is not None:
            self.vectors[fact.id] = vector
        if old is not None and old.is_current:
            if fact.is_current and vector is None and _same_keys(old, fact):
                return  # 合并：只追加了溯源，索引不变
            self._unindex(old)
        if fact.is_current:
            self._index(fact, pos)

    def current_of(self, subject_id: str) -> list[Fact]:
        """这个主语的当前事实，按 facts 的插入顺序。"""
        current = self._subjects.get(subject_id)
        return [self.facts[i] for i in current.order] if current else []

    def first_current(
        self, subject_id: str, predicate: str, value: str, negated: bool
    ) -> Fact | None:
        """current_of 里第一条谓词相同、宾语的规范形式（normalize_value）是 value、否定与否
        相同的事实。"""
        ids = self._values.get((subject_id, predicate, value, negated))
        return self.facts[next(iter(ids))] if ids else None

    def similarities(self, subject_id: str, vector: FloatArray) -> tuple[list[str], FloatArray]:
        """这个主语有向量的当前事实（按 current_of 的顺序）与它们的陈述和 vector 的相似度。
        与把这些向量 np.vstack 起来再乘 vector 是同一次矩阵运算，结果逐位相同。"""
        current = self._subjects.get(subject_id)
        if current is None or not current.rows:
            return [], np.zeros(0, dtype=np.float32)
        return list(current.rows), current.matrix() @ vector

    def predicate_rows(self, subject_id: str, predicate: str) -> npt.NDArray[np.bool_]:
        """similarities 的各行是不是这个谓词。"""
        current = self._subjects[subject_id]
        code = self._codes.get(predicate, -1)
        return current.codes[: len(current.rows)] == code

    def norm(self, text: str) -> str:
        """normalize_value(text)，算过的记下来：一条事实的宾语不会变，每次对账却都要比。"""
        value = self._norms.get(text)
        if value is None:
            value = self._norms[text] = normalize_value(text)
        return value

    def _value_key(self, fact: Fact) -> ValueKey:
        return (fact.subject_id, fact.predicate, self.norm(fact.object), fact.negated)

    def _index(self, fact: Fact, pos: int) -> None:
        current = self._subjects.setdefault(fact.subject_id, _Current())
        code = self._codes.setdefault(fact.predicate, len(self._codes))
        current.add(fact.id, pos, self.vectors.get(fact.id), code)
        _insert(self._values.setdefault(self._value_key(fact), {}), fact.id, pos)

    def _unindex(self, fact: Fact) -> None:
        self._subjects[fact.subject_id].remove(fact.id)
        key = self._value_key(fact)
        ids = self._values[key]
        del ids[fact.id]
        if not ids:
            del self._values[key]


class _Current:
    """一个主语的当前事实：ID（按在 facts 里的位置排）；有向量的那些另按同样的顺序存成一个
    可增长的 float32 矩阵（每条新事实不用再 np.vstack 一遍）与每行谓词的编号。

    向量不是一维 float32、或维数不一样时不用矩阵（regular=False），查询时照原样 np.vstack。"""

    def __init__(self) -> None:
        self.order: dict[str, int] = {}  # 当前事实 ID → 位置，按位置排
        self.rows: list[str] = []  # 有向量的当前事实，按位置排
        self.row_pos: list[int] = []
        self.vecs: list[FloatArray] = []
        self.codes: npt.NDArray[np.int32] = np.zeros(8, dtype=np.int32)
        self.buf: FloatArray | None = None  # 前 len(rows) 行是 vecs
        self.regular = True

    def add(self, fact_id: str, pos: int, vector: FloatArray | None, code: int) -> None:
        _insert(self.order, fact_id, pos)
        if vector is None:
            return
        n = len(self.rows)
        k = bisect.bisect_left(self.row_pos, pos)
        self.rows.insert(k, fact_id)
        self.row_pos.insert(k, pos)
        self.vecs.insert(k, vector)
        if n + 1 > len(self.codes):
            self.codes = np.concatenate([self.codes, np.zeros_like(self.codes)])
        self.codes[k + 1 : n + 1] = self.codes[k:n]
        self.codes[k] = code
        if not self.regular:
            return
        if not (vector.dtype == np.float32 and vector.ndim == 1) or (
            self.buf is not None and vector.shape[0] != self.buf.shape[1]
        ):
            self.regular, self.buf = False, None
            return
        if self.buf is None or n + 1 > len(self.buf):
            grown = np.empty((max(8, 2 * (n + 1)), vector.shape[0]), dtype=np.float32)
            if self.buf is not None:
                grown[:n] = self.buf[:n]
            self.buf = grown
        self.buf[k + 1 : n + 1] = self.buf[k:n]
        self.buf[k] = vector

    def remove(self, fact_id: str) -> None:
        pos = self.order.pop(fact_id)
        k = bisect.bisect_left(self.row_pos, pos)
        if k == len(self.rows) or self.rows[k] != fact_id:
            return  # 没有向量
        n = len(self.rows)
        del self.rows[k], self.row_pos[k], self.vecs[k]
        self.codes[k : n - 1] = self.codes[k + 1 : n]
        if self.buf is not None:
            self.buf[k : n - 1] = self.buf[k + 1 : n]

    def matrix(self) -> FloatArray:
        if self.buf is not None:
            return self.buf[: len(self.rows)]
        return np.vstack(self.vecs)


def _insert(index: dict[str, int], key: str, pos: int) -> None:
    """往按位置排的 {ID: 位置} 里加一项（几乎总是加在最后）。"""
    if index and pos < next(reversed(index.values())):
        index[key] = pos
        ordered = sorted(index.items(), key=lambda item: item[1])
        index.clear()
        index.update(ordered)
        return
    index[key] = pos


def _same_keys(a: Fact, b: Fact) -> bool:
    return (a.subject_id, a.predicate, a.object, a.negated) == (
        b.subject_id,
        b.predicate,
        b.object,
        b.negated,
    )


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
            # 候选都从 state 的索引里取（见 FactState）：与按 current_of 逐条比较的结果相同
            subject, value = fact.subject_id, state.norm(fact.object)
            same_value = state.first_current(subject, fact.predicate, value, fact.negated)
            if same_value is not None:
                self._merge(state, result, same_value, fact, rule="duplicate")
                continue
            ids, sims = state.similarities(subject, vector)
            scores = sims.astype(np.float64)  # 与阈值比较、排序都按 float(sim)，与以前相同
            # 陈述几乎相同、且宾语一致（或一个包含另一个）才算重复：向量对数字和专名不敏感，
            # "预算 8000 元"与"预算 6000 元"的余弦相似度可能很高，不能只看相似度就合并
            similar = next(
                (
                    c
                    for c in (
                        state.facts[ids[i]]
                        for i in _by_similarity(scores, scores >= self.duplicate_threshold)
                    )
                    if c.negated == fact.negated and _same_object(c.object, fact.object, state.norm)
                ),
                None,
            )
            if similar is not None:
                self._merge(state, result, similar, fact, rule="similar")
                continue
            if fact.negated:
                positive = state.first_current(subject, fact.predicate, value, False)
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
            wanted = scores >= threshold
            if single and ids:
                wanted |= state.predicate_rows(subject, fact.predicate)
            related = [state.facts[ids[i]] for i in _by_similarity(scores, wanted)[:3]]
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
        state.put(fact, vector)
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
        state.put(merged)
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
            state.put(historical, vector)
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
        state.put(retired)
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


def _by_similarity(scores: npt.NDArray[np.float64], wanted: npt.NDArray[np.bool_]) -> list[int]:
    """wanted 为真的行号，按相似度从高到低、相似度相同的保持原来的先后：与对全部 (事实, 相似度)
    sorted(key=-相似度) 之后再筛出这些行的结果相同。有 NaN 时 sorted 的结果与 numpy 的排序
    不同，照原来的写法对全部行排序。"""
    if np.isnan(scores).any():
        return [i for i in sorted(range(len(scores)), key=lambda i: -scores[i]) if wanted[i]]
    rows = np.flatnonzero(wanted)
    return [int(i) for i in rows[np.argsort(-scores[rows], kind="stable")]]


def _same_object(a: str, b: str, norm: Callable[[str], str] = normalize_value) -> bool:
    na, nb = norm(a), norm(b)
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
