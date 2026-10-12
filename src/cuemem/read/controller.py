"""规则版检索控制器：N7 可学习控制器的基线（设计文档 §6.3）。

只看问题里的线索，不调用 LLM，也不看检索结果，决定走 L1 还是 L2：

- 聚合：问"多少、几次、一共、哪些、列出……"——证据散落在多个会话，需要多查询召回、多给事实、
  补检索；
- 时间推算与先后：问"多久、几天、哪个先、第一次、最近一次、之前、之后……"——要比较多条证据的日期；
- 比较：问"更多、最多、比……"——要把几个候选都找全再比较；
其余（问一个具体事实、要建议）走 L1。

规则在看到 L2 的逐题结果之前写定，避免照着结果调规则；N7 用探针特征与离线标签训练的控制器
和它比较。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from cuemem.read.query import analyze

Level = Literal["L1", "L2"]

_NOT_ORDER = r"(?!\s+(?:week|month|year|weekend|night|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)"  # noqa: E501
_ORDER = re.compile(
    rf"\b(first|earliest|last{_NOT_ORDER}|latest|most recent(?:ly)?|before|after|order"
    r"|sequence)\b|第一次|最早|最后一次|最近一次|先后|之前|之后",
    re.I,
)
_DURATION = re.compile(
    r"\bhow (?:long|many (?:days|weeks|months|years|hours|minutes))\b"
    r"|多久|多长时间|几天|几周|几个月|几年",
    re.I,
)
_COMPARE = re.compile(
    r"\b(more|most|less|least|fewer|larger|largest|bigger|biggest|compared?|difference)\b"
    r"|更多|最多|最少|比较",
    re.I,
)


@dataclass(frozen=True)
class RouteDecision:
    level: Level
    reasons: list[str] = field(default_factory=list)


class RuleController:
    def decide(self, question: str, question_time: datetime) -> RouteDecision:
        query = analyze(question, question_time)
        reasons: list[str] = []
        if query.aggregation:
            reasons.append("aggregation")
        if _DURATION.search(question):
            reasons.append("duration")
        if _ORDER.search(question):
            reasons.append("order")
        if _COMPARE.search(question):
            reasons.append("compare")
        return RouteDecision("L2" if reasons else "L1", reasons)
