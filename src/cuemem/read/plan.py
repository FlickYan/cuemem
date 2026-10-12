"""查询规划（L2 第一步，设计文档 §6.5）：一次便宜模型的调用，把问题拆成检索计划。

- 答案类型：计数、求和、列举、日期、时长、先后、比较、是非、单个事实、建议……
- 是否需要"穷尽"：计数、求和、列举类问题要找全分散在各个会话里的每一项；
- 子查询：每条只找一块信息，措辞贴近用户当时会怎么说（"我看了一套房子"而不是"看房次数"）；
- 时间约束：换算成绝对日期区间（以提问时间为参照）；
- 目标实体：问题点名的人、宠物、地点、物品、事件，用于实体图扩散的种子。

规划失败（输出不合格、被内容审核拒绝）时退回规则版计划（rule_plan），检索照常进行。
规则版也用于消融"去掉查询规划"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, get_args

from pydantic import BaseModel, field_validator

from cuemem.core.trace import record_event, span
from cuemem.errors import ContentFilteredError, StructuredOutputError
from cuemem.llm.client import LLMClient
from cuemem.read.query import QueryInfo, TimeRange, analyze

AnswerType = Literal[
    "count",
    "sum",
    "list",
    "date",
    "duration",
    "order",
    "compare",
    "yes_no",
    "fact",
    "advice",
    "other",
]
ANSWER_TYPES = set(get_args(AnswerType))
AGGREGATE_TYPES = frozenset({"count", "sum", "list"})
TEMPORAL_TYPES = frozenset({"date", "duration", "order"})
MAX_SUB_QUERIES = 4
MAX_ENTITIES = 5

PLANNER_SYSTEM = """You plan memory retrieval for a personal assistant that remembers its past conversations with the user. Given a question and the date it is asked, describe what has to be looked up in those conversations. Do not answer the question.

- answer_type: "count" (how many), "sum" (a total amount), "list" (several items), "date" (when something happened), "duration" (how long, or how much time between two events), "order" (which happened first, or a sequence), "compare" (which one is more, most, larger...), "yes_no", "fact" (one specific detail), "advice" (a recommendation or suggestion that should use what is known about the user), or "other".
- exhaustive: true if the answer needs every matching item from all past conversations (counts, totals, lists, "every time..."), otherwise false.
- sub_queries: 1 to 4 short search queries, each looking for one piece of information that is needed, phrased the way the user would have mentioned it in a conversation (for example "I viewed a house" rather than "number of house viewings"). For exhaustive questions, cover the different ways or occasions in which the items may have come up. For questions that combine several facts (an age and a date, two events to compare), write one query per fact.
- time_start, time_end: if the question refers to a particular time ("two weeks ago", "last Saturday", "in March", "on Valentine's Day", "the past month"), the date range it refers to as YYYY-MM-DD, end inclusive, computed from the question date. For approximate expressions widen the range (about 3 days either side for "two weeks ago", about 10 days for "two months ago"). Otherwise null.
- entities: names of specific people, pets, places, organizations, products or events in the question, not the user ("I", "me")."""  # noqa: E501

_AGGREGATE_WORDS = re.compile(
    r"\b(how many|how much|total|in total|altogether|combined|list|count|sum)\b"
    r"|多少|几次|几个|几种|总共|一共|哪些",
    re.I,
)
_DURATION_WORDS = re.compile(
    r"\bhow (long|many (days|weeks|months|years|hours|minutes))\b|多久|多长时间|几天|几周",
    re.I,
)
_DATE_WORDS = re.compile(r"\b(when|what (date|day|time))\b|什么时候|哪天|几号", re.I)
_ORDER_WORDS = re.compile(
    r"\b(first|earliest|which (came|happened|did i do) first|before|after|order)\b|先|最早|第一",
    re.I,
)
_COMPARE_WORDS = re.compile(r"\b(more|most|less|least|larger|bigger|better|compared)\b|更|最", re.I)
_ADVICE_WORDS = re.compile(
    r"\b(recommend|suggest(ion)?s?|tips|advice|ideas?|should i|can you help)\b|推荐|建议",
    re.I,
)
_YES_NO = re.compile(r"^(is|are|was|were|do|does|did|have|has|had|can|could|will|would)\b", re.I)


class PlanOutput(BaseModel):
    """规划模型的输出格式（宽松解析：类型不认识就归为 other，日期不合法就丢掉）。"""

    answer_type: AnswerType = "other"
    exhaustive: bool = False
    sub_queries: list[str] = []
    time_start: str | None = None
    time_end: str | None = None
    entities: list[str] = []

    @field_validator("answer_type", mode="before")
    @classmethod
    def _answer_type(cls, value: Any) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        return text if text in ANSWER_TYPES else "other"

    @field_validator("sub_queries", "entities", mode="before")
    @classmethod
    def _strings(cls, value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        return [str(v).strip() for v in items if str(v or "").strip()]

    @field_validator("time_start", "time_end", mode="before")
    @classmethod
    def _date(cls, value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip() or value.strip().lower() == "null":
            return None
        try:
            return datetime.fromisoformat(value.strip()[:10]).strftime("%Y-%m-%d")
        except ValueError:
            return None


@dataclass(frozen=True)
class QueryPlan:
    question: str
    answer_type: AnswerType
    exhaustive: bool
    sub_queries: list[str]
    time_range: TimeRange | None
    entities: list[str]
    source: Literal["llm", "rules"]
    query: QueryInfo  # 规则版的问题分析（时间区间已按计划替换）
    notes: list[str] = field(default_factory=list)  # 规划过程中的说明，例如"规划失败，改用规则"

    @property
    def aggregate(self) -> bool:
        return self.exhaustive or self.answer_type in AGGREGATE_TYPES

    @property
    def queries(self) -> list[str]:
        """检索用的全部查询：原问题在前，子查询在后。"""
        return [self.question, *self.sub_queries]

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "answer_type": self.answer_type,
            "exhaustive": self.exhaustive,
            "sub_queries": self.sub_queries,
            "time_range": range_text(self.time_range),
            "entities": self.entities,
            "notes": self.notes,
        }


class QueryPlanner:
    def __init__(self, llm: LLMClient, *, role: str = "planner") -> None:
        self.llm = llm
        self.role = role

    def plan(
        self, question: str, question_time: datetime, *, cache_scope: str | None = None
    ) -> QueryPlan:
        prompt = f"Question date: {question_time:%Y-%m-%d (%A)}\nQuestion: {question}"
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        with span("plan") as record:
            try:
                output = self.llm.structured(
                    self.role, messages, PlanOutput, cache_scope=cache_scope
                ).value
            except (StructuredOutputError, ContentFilteredError) as exc:
                record_event("plan_fallback", error=type(exc).__name__)
                fallback = rule_plan(question, question_time)
                return _with_note(fallback, f"规划失败（{type(exc).__name__}），改用规则版计划")
            plan = from_output(question, question_time, output)
            record.update(plan.as_dict())
        return plan


def from_output(question: str, question_time: datetime, output: PlanOutput) -> QueryPlan:
    """把模型输出整理成检索计划：去掉重复的子查询，校验时间区间。

    时间区间优先用规则解析的结果（"两周前"这类常见说法有经过测试的容差）；
    规则解析不了时才用模型给的区间，并检查它没有落在提问之后、宽度合理。
    """
    query = analyze(question, question_time)
    notes: list[str] = []
    seen = {_norm(question)}
    sub_queries: list[str] = []
    for text in output.sub_queries:
        if _norm(text) not in seen:
            seen.add(_norm(text))
            sub_queries.append(text)
    time_range = query.time_range
    if time_range is None:
        time_range = _plan_range(output, question_time, notes)
        if time_range is not None:
            query = _replace_range(query, time_range)
    answer_type = output.answer_type
    exhaustive = output.exhaustive or answer_type in AGGREGATE_TYPES
    return QueryPlan(
        question=question,
        answer_type=answer_type,
        exhaustive=exhaustive,
        sub_queries=sub_queries[:MAX_SUB_QUERIES],
        time_range=time_range,
        entities=list(dict.fromkeys(output.entities))[:MAX_ENTITIES],
        source="llm",
        query=query,
        notes=notes,
    )


def rule_plan(question: str, question_time: datetime) -> QueryPlan:
    """不调用 LLM 的计划：答案类型按关键词猜，没有子查询，时间区间来自规则解析。"""
    query = analyze(question, question_time)
    answer_type: AnswerType = "fact"
    if _AGGREGATE_WORDS.search(question):
        answer_type = "sum" if re.search(r"\bhow much\b|总共花|一共花", question, re.I) else "count"
    elif _DURATION_WORDS.search(question):
        answer_type = "duration"
    elif _ORDER_WORDS.search(question):
        answer_type = "order"
    elif _DATE_WORDS.search(question):
        answer_type = "date"
    elif _ADVICE_WORDS.search(question):
        answer_type = "advice"
    elif _COMPARE_WORDS.search(question):
        answer_type = "compare"
    elif _YES_NO.search(question.strip()):
        answer_type = "yes_no"
    return QueryPlan(
        question=question,
        answer_type=answer_type,
        exhaustive=answer_type in AGGREGATE_TYPES or query.aggregation,
        sub_queries=[],
        time_range=query.time_range,
        entities=[],
        source="rules",
        query=query,
    )


def _plan_range(output: PlanOutput, question_time: datetime, notes: list[str]) -> TimeRange | None:
    if output.time_start is None and output.time_end is None:
        return None
    start_text = output.time_start or output.time_end
    end_text = output.time_end or output.time_start
    assert start_text is not None and end_text is not None
    first, last = sorted([datetime.fromisoformat(start_text), datetime.fromisoformat(end_text)])
    start, end = first, last + timedelta(days=1)  # 计划里的结束日期包含当天
    latest = question_time + timedelta(days=1)
    if start >= latest:
        notes.append(f"模型给的时间区间 {start_text}–{end_text} 在提问之后，已忽略")
        return None
    end = min(end, latest)
    if (end - start).days > 3 * 366:
        notes.append(f"模型给的时间区间 {start_text}–{end_text} 太宽，已忽略")
        return None
    return TimeRange(start, end, f"{start_text}–{end_text}")


def _replace_range(query: QueryInfo, time_range: TimeRange) -> QueryInfo:
    return QueryInfo(
        text=query.text,
        question_time=query.question_time,
        time_range=time_range,
        wants_first=query.wants_first,
        wants_latest=query.wants_latest,
        wants_history=query.wants_history,
        aggregation=query.aggregation,
        temporal=query.temporal,
    )


def _with_note(plan: QueryPlan, note: str) -> QueryPlan:
    return QueryPlan(
        question=plan.question,
        answer_type=plan.answer_type,
        exhaustive=plan.exhaustive,
        sub_queries=plan.sub_queries,
        time_range=plan.time_range,
        entities=plan.entities,
        source=plan.source,
        query=plan.query,
        notes=[*plan.notes, note],
    )


def range_text(time_range: TimeRange | None) -> str | None:
    """区间的日期写法（结束日包含在内），如 2023-05-13–2023-05-19。"""
    if time_range is None:
        return None
    last = time_range.end - timedelta(seconds=1)
    return f"{time_range.start:%Y-%m-%d}–{last:%Y-%m-%d}"


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()
