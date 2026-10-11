"""评测方法：B0 无记忆、B1 全上下文、B2 朴素检索、B2F 朴素检索 + 事实记忆（N3），
以及"只给证据会话"的上限参照。

所有方法共用同一个 Reader（同一个回答模型、同一套提示词），区别只在给它看什么。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from cuemem.core.models import FactHit, Hit
from cuemem.evals.datasets import EvalSession, Sample, to_session_ins
from cuemem.memory import MemorySystem
from cuemem.read.facts import Validity
from cuemem.read.reader import Reader, format_history


@dataclass
class MethodContext:
    sample: Sample
    sessions: Sequence[EvalSession]  # 这道题的完整对话历史
    reader: Reader
    system: MemorySystem | None  # 只有需要索引的方法才有
    space_id: str


@dataclass
class MethodOutput:
    response: str
    evidence_sufficient: bool | None
    hits: list[Hit] = field(default_factory=list)
    facts: list[FactHit] = field(default_factory=list)


class Method(Protocol):
    name: str
    label: str
    needs_index: bool  # 是否需要先把对话历史写入记忆（向量化）
    needs_consolidation: bool  # 是否还需要巩固（抽取事实、实体、摘要），N3 起

    def answer(self, ctx: MethodContext) -> MethodOutput: ...


class NoMemory:
    """B0：不给任何记忆，衡量"不靠记忆能答对多少"（主要是拒答题与常识题）。"""

    name, label, needs_index, needs_consolidation = "B0", "无记忆", False, False

    def answer(self, ctx: MethodContext) -> MethodOutput:
        a = ctx.reader.answer(ctx.sample.question, [], question_time=ctx.sample.question_time)
        return MethodOutput(a.answer, a.evidence_sufficient)


class FullContext:
    """B1：把整份对话历史塞进上下文（LongMemEval_S 每题约 12 万 tokens），最贵。"""

    name, label, needs_index, needs_consolidation = "B1", "全上下文", False, False

    def answer(self, ctx: MethodContext) -> MethodOutput:
        history = format_history(to_session_ins(ctx.sessions))
        a = ctx.reader.answer_from_history(
            ctx.sample.question, history, question_time=ctx.sample.question_time
        )
        return MethodOutput(a.answer, a.evidence_sufficient)


class NaiveRAG:
    """B2：N1 的朴素检索——问题向量化后在"轮"上做 dense top-k。"""

    label, needs_index, needs_consolidation = "朴素检索", True, False

    def __init__(self, k: int = 10) -> None:
        self.k = k
        self.name = "B2"

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id,
            ctx.sample.question,
            question_time=ctx.sample.question_time,
            k=self.k,
            mode="naive",
        )
        return MethodOutput(a.answer, a.evidence_sufficient, hits=a.hits)


class FactRAG:
    """B2F：检索算法与 B2 相同（dense top-k），但用上了写入路径（N3）的产出：
    事实增强键、带有效期的事实、核心记忆。各个变体用来做消融。"""

    needs_index, needs_consolidation = True, True

    def __init__(
        self,
        name: str,
        label: str,
        *,
        k: int = 10,
        validity: Validity = "full",
        show_facts: bool = True,
        profile: bool = True,
        fact_keys: bool = True,
    ) -> None:
        self.name = name
        self.label = label
        self.k = k
        self.validity: Validity = validity
        self.show_facts = show_facts
        self.profile = profile
        self.fact_keys = fact_keys

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id,
            ctx.sample.question,
            question_time=ctx.sample.question_time,
            k=self.k,
            mode="facts",
            validity=self.validity,
            show_facts=self.show_facts,
            profile=self.profile,
            fact_keys=self.fact_keys,
        )
        return MethodOutput(a.answer, a.evidence_sufficient, hits=a.hits, facts=a.facts)


class OracleSessions:
    """上限参照：只给标注的证据会话。它衡量"检索完美时 reader 能答对多少"，不是可部署的方法。"""

    name, label, needs_index = "ORACLE", "只给证据会话（上限参照）", False
    needs_consolidation = False

    def answer(self, ctx: MethodContext) -> MethodOutput:
        wanted = set(ctx.sample.evidence_session_ids)
        evidence = [s for s in ctx.sessions if s.session_id in wanted]
        history = format_history(to_session_ins(evidence))
        a = ctx.reader.answer_from_history(
            ctx.sample.question, history, question_time=ctx.sample.question_time
        )
        return MethodOutput(a.answer, a.evidence_sufficient)


METHOD_NAMES = (
    "B0",
    "B1",
    "B2",
    "B2F",
    "B2F-keys",
    "B2F-current",
    "B2F-plain",
    "B2F-noprofile",
    "ORACLE",
)


def build_methods(names: Sequence[str], *, k: int = 10) -> list[Method]:
    factories: dict[str, Callable[[], Method]] = {
        "B0": NoMemory,
        "B1": FullContext,
        "B2": lambda: NaiveRAG(k),
        "B2F": lambda: FactRAG("B2F", "朴素检索 + 事实记忆", k=k),
        # 消融：只用事实增强键改善"轮"的排序，不把事实与画像放进上下文（LongMemEval 论文的做法）
        "B2F-keys": lambda: FactRAG(
            "B2F-keys", "只用事实增强键", k=k, show_facts=False, profile=False
        ),
        # 消融：只保留当前有效的事实，相当于"覆盖式更新"（Mem0 式 UPDATE / DELETE）
        "B2F-current": lambda: FactRAG(
            "B2F-current", "只给当前事实（覆盖式更新）", k=k, validity="current"
        ),
        # 消融：事实照给，但不标注有效状态、不沿版本链补齐
        "B2F-plain": lambda: FactRAG("B2F-plain", "事实不标有效期", k=k, validity="none"),
        "B2F-noprofile": lambda: FactRAG("B2F-noprofile", "不给核心记忆", k=k, profile=False),
        "ORACLE": OracleSessions,
    }
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"未知方法 {unknown}，可选：{list(METHOD_NAMES)}")
    return [factories[n]() for n in names]
