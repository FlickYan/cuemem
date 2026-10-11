"""核心数据模型（N1 范围）：会话、轮次、记忆单元、检索结果与回答。

事实（Fact）、实体（Entity）、图片（MediaItem）等在 N3 / N6 加入。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]
UnitKind = Literal["round", "fact", "summary", "image"]


class TurnIn(BaseModel):
    """导入格式中的一次发言。"""

    role: Role
    text: str
    ts: datetime | None = None  # 缺省时使用会话开始时间
    speaker: str | None = None  # 说话人名字；多人对话（如 LoCoMo）里用来区分是谁说的


class SessionIn(BaseModel):
    session_id: str
    started_at: datetime
    turns: list[TurnIn]


class Conversation(BaseModel):
    """导入格式：一个记忆空间下的若干会话。"""

    space_id: str
    sessions: list[SessionIn]


class Turn(BaseModel):
    """原始轮次：唯一的事实来源，写入后不再修改。"""

    id: str
    space_id: str
    session_id: str
    idx: int
    role: Role
    text: str
    ts: datetime
    media_ids: list[str] = Field(default_factory=list)
    speaker: str | None = None


class MemoryUnit(BaseModel):
    """可被检索的记忆单元。N1 只有"轮"（一问一答）；N3 起增加事实、摘要等。"""

    id: str
    space_id: str
    kind: UnitKind
    session_id: str
    text: str
    ts: datetime
    source_turn_ids: list[str]
    meta: dict[str, Any] = Field(default_factory=dict)


class Hit(BaseModel):
    """一条检索结果。"""

    unit: MemoryUnit
    score: float
    route: str  # 来自哪一路检索，如 dense / sparse
    rank: int  # 在该路中的排名，从 1 开始


class Answer(BaseModel):
    question: str
    answer: str
    evidence_sufficient: bool
    citations: list[str]  # 引用编号，如 ["m1", "m3"]
    refs: dict[str, str]  # 引用编号 → 记忆单元 ID
    hits: list[Hit]
    question_time: datetime
