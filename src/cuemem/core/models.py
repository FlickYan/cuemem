"""核心数据模型。

- 原始层：会话、轮次（Turn）——唯一的事实来源，写入后不再修改；
- 派生层（N3）：实体（Entity）、事实（Fact）、会话片段（Episode）、核心记忆（CoreMemory），
  全部可以从原始层重建；
- 检索层：记忆单元（MemoryUnit）是可被检索的统一视图，轮、事实、摘要、实体都是它的一种；
- 图片（MediaItem）的结构在 N3 定下，N6 实现写入与检索。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant"]
UnitKind = Literal["round", "fact", "summary", "entity", "image"]
EntityType = Literal["person", "pet", "place", "organization", "product", "event", "other"]
FactCategory = Literal[
    "attribute", "preference", "possession", "relationship", "event", "plan", "other"
]
Cardinality = Literal["single", "multi"]
# 取代的两种原因：现实变了（搬家、换工作），或认知变了（说错后更正、补充细节）
SupersedeReason = Literal["changed", "corrected"]


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
    """可被检索的记忆单元：轮（一问一答）、事实、会话摘要、实体，N6 起还有图片。"""

    id: str
    space_id: str
    kind: UnitKind
    session_id: str
    text: str
    ts: datetime
    source_turn_ids: list[str]
    meta: dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """人、宠物、地点、组织、物品、事件……别名用于消歧（"Max""那只金毛"）。"""

    id: str
    space_id: str
    name: str
    type: EntityType
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    first_seen: datetime
    last_seen: datetime
    mentions: int = 1

    @property
    def descriptor(self) -> str:
        """用于向量化与消歧的一行描述。"""
        return f"{self.name} ({self.type}): {self.description}".strip()


class Fact(BaseModel):
    """原子事实（语义记忆），带双时间轴。

    - 事件时间 valid_from / valid_to：现实中何时开始、何时不再成立。用 ISO 日期字符串，
      可以只精确到月或年（"2023-05"），因为对话里常常只说"五月""去年"；
    - 记录时间 recorded_at / invalidated_at：系统何时得知、何时得知它不再是最新认知。
      这里取对话发生的时间，而不是程序运行的时间，保证重放结果确定。

    取代有两种：现实变了（valid_to 有值）与认知变了（只有 invalidated_at，valid_to 为空）。
    """

    id: str
    space_id: str
    subject_id: str
    subject: str  # 主语实体的名字（冗余存一份，便于展示）
    predicate: str  # snake_case，如 lives_in、job_title、5k_personal_best
    object: str
    statement: str  # 自包含的一句话：已消解代词、已换算为绝对日期
    category: FactCategory = "other"
    cardinality: Cardinality = "multi"
    negated: bool = False  # "不再喜欢……"之类的否定事实
    valid_from: str | None = None
    valid_to: str | None = None
    recorded_at: datetime
    invalidated_at: datetime | None = None
    superseded_by: str | None = None
    supersede_reason: SupersedeReason | None = None
    importance: int = 3  # 1–5
    entity_ids: list[str] = Field(default_factory=list)
    session_id: str
    source_turn_ids: list[str]

    @property
    def is_current(self) -> bool:
        return self.invalidated_at is None


class Episode(BaseModel):
    """会话片段：提供"摘要"这一粒度。目前一个会话一个片段。"""

    id: str
    space_id: str
    session_id: str
    start_ts: datetime
    end_ts: datetime
    summary: str
    turn_ids: list[str]


class CoreMemory(BaseModel):
    """核心记忆（用户画像）：重要且当前有效的事实的物化视图，每轮常驻上下文。"""

    space_id: str
    text: str
    fact_ids: list[str]
    updated_at: datetime


class MediaItem(BaseModel):
    """图片（视频 / 音频为扩展）。N3 只定义结构，N6 实现写入、描述与检索。"""

    id: str
    space_id: str
    sha256: str
    path: str
    caption: str = ""
    ocr_text: str = ""
    tags: list[str] = Field(default_factory=list)
    entity_ids: list[str] = Field(default_factory=list)
    source_turn_id: str
    ts: datetime


class Hit(BaseModel):
    """一条检索结果。"""

    unit: MemoryUnit
    score: float
    route: str  # 来自哪一路检索，如 dense / sparse / fact_key
    rank: int  # 在该路中的排名，从 1 开始


class FactHit(BaseModel):
    """检索到的一条事实；via_chain 表示它是沿版本链补进来的（新旧版本互相带出）。"""

    fact: Fact
    score: float
    via_chain: bool = False


class Answer(BaseModel):
    question: str
    answer: str
    evidence_sufficient: bool
    citations: list[str]  # 引用编号，如 ["m1", "f3"]
    refs: dict[str, str]  # 引用编号 → 记忆单元 ID（轮）或事实 ID
    hits: list[Hit]
    question_time: datetime
    facts: list[FactHit] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # L2 的阅读笔记（先记笔记再回答）
    details: dict[str, Any] = Field(default_factory=dict)  # 检索过程：L2 的计划、充分性检查等
