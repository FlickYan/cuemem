"""评测方法：B0 无记忆、B1 全上下文、B2 朴素检索、B2F 朴素检索 + 事实记忆（N3）、
B4 L1 快速召回（N4）、B5 L2 深度检索与 B6r 规则版控制器（N5）、B3 模型自主检索（N5），
以及"只给证据会话"的上限参照。

所有方法共用同一个回答模型和同一套系统提示，区别只在给它看什么、怎么检索。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from cuemem.core.models import FactHit, Hit
from cuemem.evals.datasets import EvalSession, Sample, to_session_ins
from cuemem.memory import MemorySystem
from cuemem.read.facts import Validity
from cuemem.read.l1 import L1Config
from cuemem.read.l2 import L2Config
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
    details: dict[str, Any] = field(default_factory=dict)  # 检索过程（L2 的计划、控制器的选择等）


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


class L1RAG:
    """B4：L1 快速召回（N4）——轮、事实、摘要的 dense + BM25 混合检索，模态内取最大后 RRF 融合，
    时间感知加分与时间感知打包；外加 N3 的带有效期事实与核心记忆。不调用额外的 LLM。"""

    needs_index, needs_consolidation = True, True

    def __init__(
        self,
        name: str,
        label: str,
        config: L1Config,
        *,
        relative_time: bool = True,
        notes: bool = False,
    ) -> None:
        self.name = name
        self.label = label
        self.config = config
        self.relative_time = relative_time
        self.notes = notes

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id,
            ctx.sample.question,
            question_time=ctx.sample.question_time,
            mode="l1",
            l1=self.config,
            relative_time=self.relative_time,
            notes=self.notes,
        )
        return MethodOutput(a.answer, a.evidence_sufficient, hits=a.hits, facts=a.facts)


class L2RAG:
    """B5：L2 深度检索（N5）——查询规划、多查询召回、图扩散、充分性检查与补检索，
    reader 先记笔记再回答。每题多 2–4 次便宜模型（qwen3.8-flash）调用。"""

    needs_index, needs_consolidation = True, True

    def __init__(self, name: str, label: str, config: L2Config, *, notes: bool = True) -> None:
        self.name = name
        self.label = label
        self.config = config
        self.notes = notes

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id,
            ctx.sample.question,
            question_time=ctx.sample.question_time,
            mode="l2",
            l2=self.config,
            notes=self.notes,
        )
        return MethodOutput(
            a.answer, a.evidence_sufficient, hits=a.hits, facts=a.facts, details=a.details
        )


class RoutedRAG:
    """B6r：规则版检索控制器（N7 的基线）——按问题线索（聚合、时间推算、先后、比较）选 L2，
    其余走 L1。两条路径与 B4、B5 完全相同，所以回答直接命中它们的缓存。"""

    needs_index, needs_consolidation = True, True

    def __init__(self, name: str, label: str, l1: L1Config, l2: L2Config) -> None:
        self.name = name
        self.label = label
        self.l1 = l1
        self.l2 = l2

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id,
            ctx.sample.question,
            question_time=ctx.sample.question_time,
            mode="auto",
            l1=self.l1,
            l2=self.l2,
        )
        return MethodOutput(
            a.answer, a.evidence_sufficient, hits=a.hits, facts=a.facts, details=a.details
        )


class AgentRAG:
    """B3：模型自主检索（ReAct）——reader 带 memory_search 与 date_diff 工具，自己决定
    要不要检索、检索什么、检索几次。"让 LLM 自己决定"的代表做法，N7 与控制器对比。"""

    name, label = "B3", "模型自主检索（ReAct）"
    needs_index, needs_consolidation = True, True

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id, ctx.sample.question, question_time=ctx.sample.question_time, mode="agent"
        )
        return MethodOutput(
            a.answer, a.evidence_sufficient, hits=a.hits, facts=a.facts, details=a.details
        )


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
    "B4",
    "B4-k10",
    "B4-k5",
    "B4-b2k",
    "B4-notime",
    "B4-norel",
    "B4-nobm25",
    "B4-b6k",
    "B4-note",
    "B5",
    "B5-noplan",
    "B5-ppr",
    "B5-noiter",
    "B5-nonote",
    "B6r",
    "B3",
    "ORACLE",
)


# N4 定稿时的 L1 配置（带 BM25）：B4 及其消融固定用它，保证 N4 的结果可以复现
N4_L1 = L1Config(sparse="bm25")


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
        # B4：N4 定稿时的 L1（带 BM25）；其余是消融。N5 起 L1 默认不用 BM25（D-026 修订），
        # 等同于 B4-nobm25
        "B4": lambda: L1RAG("B4", "L1 快速召回", N4_L1),
        "B4-k10": lambda: L1RAG("B4-k10", "固定取 10 轮", replace(N4_L1, k=10, budget_tokens=None)),
        "B4-k5": lambda: L1RAG("B4-k5", "固定取 5 轮", replace(N4_L1, k=5, budget_tokens=None)),
        "B4-b2k": lambda: L1RAG("B4-b2k", "预算 2k tokens", replace(N4_L1, budget_tokens=2000)),
        "B4-notime": lambda: L1RAG("B4-notime", "不做时间加分", replace(N4_L1, time_weight=0)),
        "B4-norel": lambda: L1RAG("B4-norel", "不标相对时间", N4_L1, relative_time=False),
        "B4-nobm25": lambda: L1RAG("B4-nobm25", "去掉 BM25", replace(N4_L1, sparse="none")),
        # N5 的两个对照：给 L1 与 L2 相同的预算；L1 也先记笔记再回答（规则版阅读提示）
        "B4-b6k": lambda: L1RAG("B4-b6k", "预算 6k tokens（与 L2 相同）", L2Config().l1),
        "B4-note": lambda: L1RAG("B4-note", "先记笔记再回答", L1Config(), notes=True),
        # B5：L2 的最终配置；其余是消融
        "B5": lambda: L2RAG("B5", "L2 深度检索", L2Config()),
        "B5-noplan": lambda: L2RAG("B5-noplan", "不做查询规划", replace(L2Config(), plan=False)),
        "B5-ppr": lambda: L2RAG("B5-ppr", "加图扩散（PPR）", replace(L2Config(), ppr=True)),
        "B5-noiter": lambda: L2RAG(
            "B5-noiter", "不做充分性检查与补检索", replace(L2Config(), iterations=0)
        ),
        "B5-nonote": lambda: L2RAG("B5-nonote", "不记笔记", L2Config(), notes=False),
        "B6r": lambda: RoutedRAG("B6r", "规则版控制器（L1 / L2）", L1Config(), L2Config()),
        "B3": AgentRAG,
        "ORACLE": OracleSessions,
    }
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"未知方法 {unknown}，可选：{list(METHOD_NAMES)}")
    return [factories[n]() for n in names]
