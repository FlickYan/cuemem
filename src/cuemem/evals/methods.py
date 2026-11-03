"""评测方法：B0 无记忆、L0 只给核心记忆（N7）、B1 全上下文、B2 朴素检索、
B2F 朴素检索 + 事实记忆（N3）、B4 L1 快速召回（N4）、B5 L2 深度检索与 B6r 规则版控制器（N5）、
B3 模型自主检索（N5）、B6a 控制器先选、模型兜底（N9），以及"只给证据会话"的上限参照。

所有方法共用同一个回答模型和同一套系统提示，区别只在给它看什么、怎么检索。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from cuemem.core.models import FactHit, Hit
from cuemem.evals.datasets import EvalSession, Sample, SessionIds, to_session_ins
from cuemem.memory import MemorySystem
from cuemem.read.controller import Router
from cuemem.read.facts import Validity
from cuemem.read.l1 import L1Config
from cuemem.read.l2 import L2Config
from cuemem.read.reader import Reader, format_history, history_blocks


@dataclass
class MethodContext:
    sample: Sample
    sessions: Sequence[EvalSession]  # 这道题的完整对话历史
    reader: Reader
    system: MemorySystem | None  # 只有需要索引的方法才有
    space_id: str
    # 会话 ID 的写法（见 datasets 的模块说明）：自己存写入的方法（M0 与外部基线）按它分开存
    session_ids: SessionIds = "native"


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


class ResidentContext:
    """L0：只给常驻上下文（核心记忆，即用户画像），不检索——N7 控制器最便宜的一档。
    与 B0 的区别：B0 什么都不给；L0 给每轮都在上下文里的那部分（稳定前缀，可命中 prompt cache）。"""

    name, label = "L0", "L0 常驻上下文（只给核心记忆）"
    needs_index, needs_consolidation = True, True

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id, ctx.sample.question, question_time=ctx.sample.question_time, mode="l0"
        )
        return MethodOutput(a.answer, a.evidence_sufficient, details=a.details)


class FullContext:
    """B1：把整份对话历史塞进上下文（LongMemEval_S 每题约 12 万 tokens），最贵。
    带图的对话（MemLens）把图片按原位插在对话里，即"长上下文模型直接看图"。"""

    name, label, needs_index, needs_consolidation = "B1", "全上下文", False, False

    def answer(self, ctx: MethodContext) -> MethodOutput:
        a = ctx.reader.answer_from_history(
            ctx.sample.question, _history(ctx.sessions), question_time=ctx.sample.question_time
        )
        return MethodOutput(a.answer, a.evidence_sufficient)


def _history(sessions: Sequence[EvalSession]) -> str | list[dict[str, Any]]:
    session_ins = to_session_ins(sessions)
    if any(t.images for s in session_ins for t in s.turns):
        return history_blocks(session_ins)
    return format_history(session_ins)


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
        show_facts: bool = True,
    ) -> None:
        self.name = name
        self.label = label
        self.config = config
        self.relative_time = relative_time
        self.notes = notes
        self.show_facts = show_facts  # False：不给事实与核心记忆（只看对话片段）

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
            show_facts=self.show_facts,
            profile=self.show_facts,
        )
        return MethodOutput(
            a.answer, a.evidence_sufficient, hits=a.hits, facts=a.facts, details=a.details
        )


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
    """检索控制器选档后回答：B6r 规则版（N5，只在 L1 / L2 之间选）；N7 的 B6 学到的控制器
    （探针 + 两段式，L0 / L1 / L2 与是否再感知）、B6h 混合方案（门控拿不准时问 LLM）、
    B6z 零样本 LLM 路由。各条路径与 L0、B4、B5（及 -see）完全相同，回答直接命中它们的缓存。"""

    needs_index, needs_consolidation = True, True

    def __init__(
        self, name: str, label: str, l1: L1Config, l2: L2Config, *, router: Router = "rule"
    ) -> None:
        self.name = name
        self.label = label
        self.l1 = l1
        self.l2 = l2
        self.router: Router = router

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id,
            ctx.sample.question,
            question_time=ctx.sample.question_time,
            mode="auto",
            l1=self.l1,
            l2=self.l2,
            router=self.router,
        )
        return MethodOutput(
            a.answer, a.evidence_sufficient, hits=a.hits, facts=a.facts, details=a.details
        )


class AgentRAG:
    """B3：模型自主检索（ReAct）——reader 带 memory_search 与 date_diff 工具，自己决定
    要不要检索、检索什么、检索几次。"让 LLM 自己决定"的代表做法，N7 与控制器对比。

    B6a（N9）：控制器先选档召回，召回的记忆随问题给出，模型缺什么再用 memory_search 补查——
    N8 Agent 的做法（D-042）。"""

    needs_index, needs_consolidation = True, True

    def __init__(self, name: str = "B3", *, hatch: bool = False) -> None:
        self.name = name
        self.hatch = hatch
        self.label = "控制器先选、模型兜底" if hatch else "模型自主检索（ReAct）"

    def answer(self, ctx: MethodContext) -> MethodOutput:
        assert ctx.system is not None
        a = ctx.system.ask(
            ctx.space_id,
            ctx.sample.question,
            question_time=ctx.sample.question_time,
            mode="hatch" if self.hatch else "agent",
            router="learned",
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
        a = ctx.reader.answer_from_history(
            ctx.sample.question, _history(evidence), question_time=ctx.sample.question_time
        )
        return MethodOutput(a.answer, a.evidence_sufficient)


METHOD_NAMES = (
    "B0",
    "L0",
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
    "B6",
    "B6h",
    "B6z",
    "B3",
    "B6a",
    "B4-text",
    "B4-cap",
    "B4-iso",
    "B4-mm",
    "B4-see",
    "B4-novis",
    "B4-see-novis",
    "B5-see",
    "ORACLE",
    "M0",
    "MG",
    "ML",
    "MC",
    "MA",
)

# N12：更多外部基线（evals/baselines/），都要用 uv 临时装上对应的包，见各模块说明
EXTERNAL = {
    "MG": ("cuemem.evals.baselines.graphiti", "GraphitiBaseline"),
    "ML": ("cuemem.evals.baselines.langmem", "LangMemBaseline"),
    "MC": ("cuemem.evals.baselines.cognee", "CogneeBaseline"),
    "MA": ("cuemem.evals.baselines.amem", "AMemBaseline"),
}


# N4 定稿时的 L1 配置（带 BM25）：B4 及其消融固定用它，保证 N4 的结果可以复现
N4_L1 = L1Config(sparse="bm25")
# N6：只用对话与图片描述、不用事实的两个配置（事实是从带描述的会话里抽取的，会泄露图片内容）
# 对比图片描述时不用以文搜图，只看描述本身的作用
CAPTIONS = L1Config(fact_weight=0.0, visual_weight=0.0)
TEXT_ONLY = replace(CAPTIONS, image_kind=None)


def _mem0() -> Method:
    from cuemem.evals.mem0_baseline import Mem0Baseline

    return Mem0Baseline()


def _external(name: str) -> Callable[[], Method]:
    def make() -> Method:
        import importlib

        module, cls = EXTERNAL[name]
        method: Method = getattr(importlib.import_module(module), cls)()
        return method

    return make


def build_methods(names: Sequence[str], *, k: int = 10) -> list[Method]:
    factories: dict[str, Callable[[], Method]] = {
        "B0": NoMemory,
        "L0": ResidentContext,
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
        # N7：学到的控制器（默认 λ 取模型文件里的值）、混合方案、零样本 LLM 路由
        "B6": lambda: RoutedRAG(
            "B6", "学到的控制器（探针 + 两段式）", L1Config(), L2Config(), router="learned"
        ),
        "B6h": lambda: RoutedRAG(
            "B6h", "混合控制器（拿不准时问 LLM）", L1Config(), L2Config(), router="hybrid"
        ),
        "B6z": lambda: RoutedRAG("B6z", "零样本 LLM 路由", L1Config(), L2Config(), router="llm"),
        "B3": AgentRAG,
        # N9：控制器先选档召回、模型可用 memory_search 兜底（N8 的 Agent 做法）
        "B6a": lambda: AgentRAG("B6a", hatch=True),
        # N6 多模态（MemLens）：从纯文本记忆逐级加上图片描述、事实、再感知
        "B4-text": lambda: L1RAG(
            "B4-text", "纯文本记忆（图片只留占位）", TEXT_ONLY, show_facts=False
        ),
        "B4-cap": lambda: L1RAG(
            "B4-cap", "对话 + 上下文感知的图片描述", CAPTIONS, show_facts=False
        ),
        "B4-iso": lambda: L1RAG(
            "B4-iso",
            "对话 + 孤立的图片描述",
            replace(CAPTIONS, image_kind="image_iso"),
            show_facts=False,
        ),
        "B4-mm": lambda: L1RAG(
            "B4-mm", "L1：对话 + 图片描述 + 以文搜图 + 事实与核心记忆", L1Config()
        ),
        "B4-see": lambda: L1RAG(
            "B4-see", "B4-mm + 再感知 3 张原图", replace(L1Config(), attach_images=3)
        ),
        "B4-novis": lambda: L1RAG(
            "B4-novis", "B4-mm 去掉以文搜图", replace(L1Config(), visual_weight=0.0)
        ),
        "B4-see-novis": lambda: L1RAG(
            "B4-see-novis",
            "B4-see 去掉以文搜图",
            replace(L1Config(), visual_weight=0.0, attach_images=3),
        ),
        "B5-see": lambda: L2RAG(
            "B5-see",
            "L2 + 再感知 3 张原图",
            replace(L2Config(), l1=replace(L2Config().l1, attach_images=3)),
        ),
        "ORACLE": OracleSessions,
        # N11：外部基线 Mem0（需要临时装上 mem0ai，见 evals/mem0_baseline.py）
        "M0": _mem0,
        # N12：Graphiti、LangMem、Cognee、A-MEM（见 evals/external.py 与 evals/baselines/）
        **{name: _external(name) for name in EXTERNAL},
    }
    unknown = [n for n in names if n not in factories]
    if unknown:
        raise ValueError(f"未知方法 {unknown}，可选：{list(METHOD_NAMES)}")
    return [factories[n]() for n in names]
