"""抽取：一次 LLM 调用把一个会话整理成事实、实体与摘要（结构化输出）。

抽取是**无状态**的：输入只有这个会话本身（和会话日期），不依赖记忆空间里已有的内容。
因此同一空间的各个会话可以并行抽取；相同的会话（评测中不同题目共用的会话）命中响应缓存，
只付一次钱。和已有记忆的比较（消歧、对账）放在之后串行进行（D-020）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, get_args

from pydantic import BaseModel, field_validator

from cuemem.core.ids import content_hash
from cuemem.core.models import Cardinality, EntityType, FactCategory, Turn
from cuemem.core.timeutil import normalize_partial_date
from cuemem.core.trace import span
from cuemem.llm.client import LLMClient, Message
from cuemem.write.caption import inline_images
from cuemem.write.predicates import SUGGESTED

# 抽取逻辑（提示词、输出结构、后处理）的版本。变化后，已巩固的空间需要重建派生记忆
PIPELINE_VERSION = "n3-v1"

EXTRACTION_SYSTEM = f"""You build the long-term memory of a personal assistant. Read one conversation and extract what is worth remembering about {{focus}}, so that questions about them can be answered weeks or months later.

Extract facts about {{focus}} and the people, pets, places, things and events in their life:
- personal details: home, work, studies, family, health, routines, finances;
- preferences, opinions, habits, likes and dislikes;
- possessions and purchases (brands, models, prices when stated);
- experiences and events: what happened, where, with whom, and when;
- plans, goals and upcoming events;
- numbers that may change later: counts, amounts, times, scores, frequencies, durations.
Details mentioned in passing (for example "by the way, I've been doing yoga twice a week") are often exactly what is asked about later. Do not skip them.

Do not extract general knowledge, generic advice or explanations from the assistant, the assistant's own traits, small talk, or hypotheticals. Include something the assistant said only if the user accepted it or acted on it.

For each fact:
- statement: one self-contained sentence that a stranger could understand. Use names instead of pronouns and "the user" for the user. Keep specifics (names, numbers, brands, places). Convert relative times ("yesterday", "last month", "two weeks ago") into absolute dates using the conversation date. Write it in the language named in the conversation header.
- subject: who or what the fact is about: {{subject_hint}}, or the name of another person, pet or thing.
- predicate: short snake_case. Prefer one of: {SUGGESTED}. For a single-valued attribute make the predicate name the exact attribute, following patterns such as <item>_budget, <event>_personal_best or <activity>_frequency.
- object: the value, kept short.
- category: attribute, preference, possession, relationship, event, plan or other.
- cardinality: "single" if a newer value would replace an older one (where someone lives, their job, a current count or amount, a personal best, a budget, how often they do something); "multi" if values accumulate (events, things they like, places visited, items owned).
- negated: true if the fact says something is no longer, or not, the case.
- date: when the event happened or the state began, as YYYY-MM-DD, YYYY-MM or YYYY; null if unknown. Use the conversation date only if it happened that day. If the exact day is uncertain, give a coarser date instead of guessing, and keep the statement free of hedging.
- importance: 1 (trivial detail) to 5 (core identity, major life event, strong lasting preference).
- turns: the numbers of the turns that state the fact.

Also list the named entities that appear in the facts (people, pets, places, organizations, products, events) with their type, other names used for them in this conversation, and a short description; and summarize the conversation in one or two sentences. Write the descriptions and the summary in the same language as the statements.
Return empty lists if nothing is worth remembering."""  # noqa: E501

# 应用给抽取的补充说明（consolidation.instructions，N12）放在系统提示的最后
INSTRUCTIONS_HEADER = (
    "Additional instructions from the application (they take precedence over the defaults "
    "above when they conflict):"
)

# 遗忘删掉了会话里的发言时（D-021，发言编号不复用，留下空缺）才加的说明。剩下的发言可能用代词
# 指着被删的内容（"那副登山杖正合适，我给她买了"）；不说明的话，模型会从剩下的人里挑一个顶上
# （演示里成了"给 Lena 买"）。没有空缺的会话渲染结果不变，已有的抽取缓存照常命中（D-059）
DELETED_NOTE = (
    "Some messages in this conversation were deleted at the user's request; each gap is marked "
    '[deleted]. Do not reconstruct or guess what they said. A pronoun such as "her", "him", '
    '"them" or "those" next to a deleted gap usually pointed to something in the deleted '
    "message: do not resolve it to a person or thing named elsewhere in the conversation. Write "
    'such a fact without the name (for example "someone"), or leave it out. This overrides the '
    "rule to use names instead of pronouns."
)

_ENTITY_TYPES = set(get_args(EntityType))
_CATEGORIES = set(get_args(FactCategory))
_TURN_NUMBER = re.compile(r"\d+")


class ExtractedEntity(BaseModel):
    name: str
    type: EntityType = "other"
    aliases: list[str] = []
    description: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def _type(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        return text if text in _ENTITY_TYPES else "other"

    @field_validator("aliases", mode="before")
    @classmethod
    def _aliases(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        return [str(v).strip() for v in value or [] if str(v).strip()]


class ExtractedFact(BaseModel):
    statement: str
    subject: str = "User"
    predicate: str = "related_to"
    object: str = ""
    category: FactCategory = "other"
    cardinality: Cardinality = "multi"
    negated: bool = False
    date: str | None = None
    importance: int = 3
    turns: list[int] = []

    @field_validator("category", mode="before")
    @classmethod
    def _category(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        return text if text in _CATEGORIES else "other"

    @field_validator("cardinality", mode="before")
    @classmethod
    def _cardinality(cls, value: Any) -> str:
        return "single" if str(value or "").strip().lower() == "single" else "multi"

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, value: Any) -> str | None:
        return normalize_partial_date(value)

    @field_validator("importance", mode="before")
    @classmethod
    def _importance(cls, value: Any) -> int:
        try:
            number = round(float(value))
        except (TypeError, ValueError):
            return 3
        return min(5, max(1, number))

    @field_validator("turns", mode="before")
    @classmethod
    def _turns(cls, value: Any) -> list[int]:
        if value is None:
            return []
        items = value if isinstance(value, list) else [value]
        out: list[int] = []
        for item in items:  # 兼容 3、"3"、"t3"、"[t3]"
            match = _TURN_NUMBER.search(str(item))
            if match:
                out.append(int(match.group()))
        return out

    @field_validator("statement", "object", "subject", "predicate", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()


class ExtractionOutput(BaseModel):
    summary: str = ""
    facts: list[ExtractedFact] = []
    entities: list[ExtractedEntity] = []

    @field_validator("summary", mode="before")
    @classmethod
    def _summary(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()


@dataclass
class CandidateFact:
    """抽取出、尚未与已有记忆比较的事实。"""

    raw: ExtractedFact
    source_turn_ids: list[str]


@dataclass
class SessionExtraction:
    session_id: str
    started_at: datetime
    summary: str
    facts: list[CandidateFact]
    entities: list[ExtractedEntity]
    turn_ids: list[str]
    attempts: int = 1
    dropped_turn_refs: int = 0  # 引用了不存在的轮次编号的次数（模型质量指标）
    meta: dict[str, Any] = field(default_factory=dict)


def participants(turns: Sequence[Turn]) -> list[str]:
    """参与者：多人对话（如 LoCoMo）用说话人名字，否则是 User 与 Assistant。"""
    names = list(dict.fromkeys(t.speaker for t in turns if t.speaker))
    return names if names else ["User", "Assistant"]


def render_session(
    turns: Sequence[Turn],
    started_at: datetime,
    *,
    assistant_chars: int | None,
    image_notes: Mapping[str, Sequence[str]] | None = None,
    deleted_from: int | None = None,
) -> str:
    """把会话渲染成带轮次编号的文本。

    助手回复只保留开头 assistant_chars 个字符：记忆主要关于用户，LongMemEval 里助手回复
    占对话文本的 87%，截断后抽取的输入 token 少一大半；完整的助手回复仍保存在原始日志和"轮"
    索引里，"助手说过什么"这类问题照样能检索到。

    image_notes（N6）：发言 ID → 这条发言里各张图的描述，填回 "<image>" 的位置。只有带图的
    会话才多一行说明，不带图的会话渲染结果与之前完全相同（抽取缓存照常命中）。

    deleted_from：会话里有发言被遗忘删掉时，这一段（超长会话切成的窗口）第一句应有的编号；
    空缺标成 [deleted]，并加一行说明（DELETED_NOTE）。为 None（没删过）时渲染结果不变。
    """
    names = participants(turns)
    who = ", ".join(names) if names != ["User", "Assistant"] else "User, Assistant (AI)"
    lines = [
        f"Conversation date: {started_at:%Y-%m-%d (%A)}",
        f"Participants: {who}",
        f"Language: {conversation_language(turns)}",
    ]
    notes = image_notes or {}
    if any(notes.get(t.id) for t in turns):
        lines.append(
            "Images shared in the conversation are shown as [Image: description]; treat what "
            "they show as information the speaker shared."
        )
    body: list[str] = []
    expected = deleted_from
    for turn in turns:
        if expected is not None:
            if turn.idx > expected:
                body.append("[deleted]")
            expected = turn.idx + 1
        label = turn.speaker or ("User" if turn.role == "user" else "Assistant")
        text = turn.text.strip()
        if turn.role == "assistant" and assistant_chars is not None and len(text) > assistant_chars:
            text = text[:assistant_chars].rstrip() + " […]"
        if notes.get(turn.id):
            text = inline_images(text, notes[turn.id])
        body.append(f"[t{turn.idx}] {label}: {text}")
    if "[deleted]" in body:
        lines.append(DELETED_NOTE)
    return "\n".join([*lines, "", *body])


def gaps(turns: Sequence[Turn]) -> bool:
    """发言编号有没有空缺（遗忘删掉了其中的发言）。编号从 0 起连续时没有。"""
    return any(t.idx != i for i, t in enumerate(turns))


def conversation_language(turns: Sequence[Turn]) -> str:
    """对话语言（按用户发言里汉字的比例判断）。写进提示词，避免模型把中文对话的事实写成英文。"""
    text = "".join(t.text for t in turns if t.role == "user") or "".join(t.text for t in turns)
    letters = [c for c in text if c.isalpha()]
    cjk = sum(1 for c in letters if "\u4e00" <= c <= "\u9fff")
    return "Chinese" if letters and cjk / len(letters) >= 0.2 else "English"


class Extractor:
    def __init__(
        self,
        llm: LLMClient,
        *,
        role: str = "extractor",
        model: str | None = None,
        assistant_chars: int | None = 600,
        max_chars: int = 24_000,
        fallback_model: str | None = None,
        instructions: str | None = None,
    ) -> None:
        self.llm = llm
        self.role = role
        self.model = model
        self.assistant_chars = assistant_chars
        self.max_chars = max_chars
        self.fallback_model = fallback_model
        self.instructions = instructions.strip() if instructions else None

    def extract(
        self,
        session_id: str,
        turns: Sequence[Turn],
        started_at: datetime,
        *,
        cache_scope: str | None = None,
        image_notes: Mapping[str, Sequence[str]] | None = None,
    ) -> SessionExtraction:
        """抽取一个会话。超长会话按轮次切成几个窗口分别抽取，再合并结果。"""
        by_idx = {t.idx: t.id for t in turns}
        result = SessionExtraction(
            session_id=session_id,
            started_at=started_at,
            summary="",
            facts=[],
            entities=[],
            turn_ids=[t.id for t in turns],
            attempts=0,
        )
        summaries: list[str] = []
        deleted = gaps(turns)
        start = 0
        for window in self._windows(turns):
            messages = self.messages(
                window,
                started_at,
                image_notes=image_notes,
                deleted_from=start if deleted else None,
            )
            start = window[-1].idx + 1
            with span("extract", session=session_id, turns=len(window)):
                structured = self.llm.structured(
                    self.role,
                    messages,
                    ExtractionOutput,
                    model=self.model,
                    cache_scope=cache_scope,
                    fallback_model=self.fallback_model,
                )
            output = structured.value
            result.attempts += structured.attempts
            if output.summary:
                summaries.append(output.summary)
            result.entities.extend(output.entities)
            for fact in output.facts:
                if not fact.statement.strip():
                    continue
                ids = [by_idx[i] for i in dict.fromkeys(fact.turns) if i in by_idx]
                result.dropped_turn_refs += len(set(fact.turns)) - len(ids)
                result.facts.append(CandidateFact(raw=fact, source_turn_ids=ids))
        result.summary = " ".join(summaries)
        return result

    def messages(
        self,
        turns: Sequence[Turn],
        started_at: datetime,
        *,
        image_notes: Mapping[str, Sequence[str]] | None = None,
        deleted_from: int | None = None,
    ) -> list[Message]:
        names = participants(turns)
        if names == ["User", "Assistant"]:
            focus, subject_hint = "the user", '"User" for the user'
        else:
            focus = "the speakers (" + ", ".join(names) + ")"
            subject_hint = "the speaker's name (" + ", ".join(names) + ")"
        system = EXTRACTION_SYSTEM.format(focus=focus, subject_hint=subject_hint)
        if self.instructions:  # 没有补充说明时提示词逐字不变，已有的抽取缓存照常命中
            system += f"\n\n{INSTRUCTIONS_HEADER}\n{self.instructions}"
        text = render_session(
            turns,
            started_at,
            assistant_chars=self.assistant_chars,
            image_notes=image_notes,
            deleted_from=deleted_from,
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": text}]

    def _windows(self, turns: Sequence[Turn]) -> list[list[Turn]]:
        windows: list[list[Turn]] = [[]]
        size = 0
        for turn in turns:
            length = len(turn.text)
            if turn.role == "assistant" and self.assistant_chars is not None:
                length = min(length, self.assistant_chars)
            if windows[-1] and size + length > self.max_chars:
                windows.append([])
                size = 0
            windows[-1].append(turn)
            size += length
        return [w for w in windows if w]


def session_content_hash(turns: Sequence[Turn], started_at: datetime) -> str:
    """会话内容指纹：任务键的一部分。会话追加了新轮次或轮次被删除时，指纹随之改变。"""
    parts = [started_at.isoformat()]
    parts += [f"{t.idx}\x1e{t.role}\x1e{t.speaker or ''}\x1e{t.text}" for t in turns]
    # 带图的发言再加上图片 ID（不带图的会话指纹与之前相同，已有的任务键不变）
    parts += [f"{t.idx}\x1e{','.join(t.media_ids)}" for t in turns if t.media_ids]
    return content_hash("\x1f".join(parts))
