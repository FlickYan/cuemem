"""Reader：把检索到的记忆按时间顺序编号放进上下文，让 LLM 带引用地回答。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from cuemem.core.models import Answer, FactHit, Hit, MediaItem, SessionIn
from cuemem.core.trace import span
from cuemem.llm.client import LLMClient, Message
from cuemem.read.facts import ImageMode, MemoryView, Validity, format_facts
from cuemem.read.plan import TEMPORAL_TYPES, QueryPlan, range_text
from cuemem.read.query import TimeRange
from cuemem.store.media import image_data_url
from cuemem.write.caption import fill_images, image_note

# 评测基准以英文为主，提示词用英文；回答语言跟随问题
READER_SYSTEM = """You are a personal assistant with long-term memory of past conversations with the user.
You will receive either memories retrieved from earlier conversations (each with an id such as [m1] and the date it happened) or the full history of earlier conversations.

Rules:
1. Base your answer on the memories. If memories conflict, trust the most recent one unless the question asks about the past.
2. Interpret relative time expressions (such as "last week" or "two months ago") relative to the current date given below.
3. When the memories contain the answer, state it directly and concisely; do not hedge.
4. If the question asks for a specific fact and the memories do not contain it, say that you don't know. Never invent details.
5. If the question asks for advice or recommendations, give helpful, specific suggestions that make use of what the memories say about the user (their preferences, plans, and the things they own or use).
6. Answer in the same language as the question.
7. If the memories have ids, list in "citations" the ids of the memories you actually used."""  # noqa: E501


class ReaderOutput(BaseModel):
    answer: str = Field(description="final answer to the question")
    citations: list[str] = Field(
        default_factory=list, description="ids of the memories used, e.g. ['m1', 'm3']"
    )
    evidence_sufficient: bool = Field(
        description="true if the retrieved memories were sufficient to answer"
    )


class NotedReaderOutput(BaseModel):
    """L2 的回答格式：先记笔记（Chain-of-Note 式阅读），再回答。字段顺序就是生成顺序。"""

    notes: list[str] = Field(
        default_factory=list,
        description="written before answering: one short note per relevant memory, "
        "with its id, its date and the relevant detail",
    )
    answer: str = Field(description="final answer to the question")
    citations: list[str] = Field(
        default_factory=list, description="ids of the memories used, e.g. ['m1', 'f3']"
    )
    evidence_sufficient: bool = Field(
        description="true if the retrieved memories were sufficient to answer"
    )


IN_RANGE_MARK = "★"


def reading_guide(plan: QueryPlan) -> list[str]:
    """按检索计划给 reader 的阅读提示：先记笔记；计数类逐项列出再数；时间类显式算日期差。"""
    lines = [
        "Before answering, write notes: for each memory relevant to the question, one short "
        "note with its id, its date and the relevant detail. Skip unrelated memories."
    ]
    if plan.aggregate:
        lines.append(
            "This question needs every matching item. In the notes, list each distinct item "
            "that matches, with its memory id. The same item is often mentioned in several "
            "conversations: count it once. Items mentioned only in passing count too. Then "
            "count or add up the items in your notes."
        )
    if plan.answer_type in TEMPORAL_TYPES:
        lines.append(
            'Work out dates from the memory headers. "N days before the question" gives each '
            "memory's distance from the question date, so the time between two events is the "
            "difference of these numbers. Write the calculation in your notes."
        )
    if plan.time_range is not None:
        lines.append(
            f"The question refers to {range_text(plan.time_range)}; memories from that period "
            f"are marked with {IN_RANGE_MARK}."
        )
    return lines


def format_memories(
    hits: Sequence[Hit],
    *,
    question_time: datetime | None = None,
    highlight: TimeRange | None = None,
    media: Mapping[str, MediaItem] | None = None,
    image_mode: ImageMode = "context",
    attach: Sequence[str] = (),
) -> tuple[str, dict[str, str]]:
    """按时间顺序排列证据并编号，返回（上下文文本，编号 → 记忆单元 ID）。

    给了 question_time 时，每条再标上"距提问几天"（时间感知打包，N4）：
    让 reader 少做日期运算，"几天前""哪个先发生"这类问题直接有据可依。
    给了 highlight（问题所指的时间区间，L2）时，区间内的证据标上 ★。
    轮里的图片（N6）在原位写成 [Image i3: 描述]；附了原图的注明是第几张图。
    """
    ordered = sorted(hits, key=lambda h: (h.unit.ts, h.unit.id))
    refs: dict[str, str] = {}
    blocks: list[str] = []
    images = media or {}
    image_count = 0
    for i, hit in enumerate(ordered, start=1):
        ref = f"m{i}"
        refs[ref] = hit.unit.id
        header = f"[{ref}] {hit.unit.ts:%Y-%m-%d (%a) %H:%M}"
        if question_time is not None:
            header += f" · {relative_days(hit.unit.ts, question_time)}"
        header += f" · session {hit.unit.session_id}"
        if highlight is not None and highlight.contains(hit.unit.ts):
            header += f" {IN_RANGE_MARK}"
        text = hit.unit.text
        rendered: list[str] = []
        for media_id in hit.unit.meta.get("media_ids", []):
            image_count += 1
            label = f"i{image_count}"
            refs[label] = media_id
            rendered.append(_image_text(label, images.get(media_id), image_mode, attach))
        if rendered or "<image>" in text:
            text = fill_images(text, rendered)
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks), refs


def _image_text(label: str, item: MediaItem | None, mode: ImageMode, attach: Sequence[str]) -> str:
    if item is None or mode == "none":
        return "[Image]"
    note = image_note(item, isolated=mode == "isolated")
    if item.id in attach:
        return f"[Image {label}, see picture {attach.index(item.id) + 1} below: {note}]"
    return f"[Image {label}: {note}]"


def relative_days(moment: datetime, question_time: datetime) -> str:
    days = (question_time.date() - moment.date()).days
    if days == 0:
        return "the same day as the question"
    if days < 0:
        return f"{-days} days after the question"
    unit = "day" if days == 1 else "days"
    if days < 14:
        return f"{days} {unit} before the question"
    if days < 60:
        return f"{days} days (about {round(days / 7)} weeks) before the question"
    return f"{days} days (about {round(days / 30.4)} months) before the question"


def memory_blocks(
    view: MemoryView,
    question_time: datetime,
    *,
    validity: Validity = "full",
    show_facts: bool = True,
    profile: bool = True,
    relative_time: bool = False,
    highlight: TimeRange | None = None,
) -> tuple[list[str], dict[str, str], list[FactHit]]:
    """把一次检索的材料排成上下文：核心记忆、带有效期的事实、按时间排列的对话片段。

    返回（各块文本，编号 → 记忆 ID，实际展示的事实）。Reader 与 Agent（N8）共用。
    """
    blocks: list[str] = []
    if profile and view.core is not None and view.core.text:
        # 多人对话（如 LoCoMo）的核心记忆按人分组，每组以"名字:"开头
        who = "the user" if view.core.text.startswith("- ") else "the people in these chats"
        blocks.append(f"About {who} (core memory, current facts):\n{view.core.text}")
    facts = [fh for fh in view.facts if fh.fact.is_current or validity != "current"]
    fact_refs: dict[str, str] = {}
    when = question_time if relative_time else None
    if show_facts and facts:
        text, fact_refs = format_facts(
            facts, validity=validity, question_time=when, highlight=highlight
        )
        blocks.append(text)
    context, round_refs = format_memories(
        view.rounds,
        question_time=when,
        highlight=highlight,
        media=view.media,
        image_mode=view.image_mode,
        attach=view.attach,
    )
    blocks.append(f"Retrieved conversation excerpts:\n{context or '(none)'}")
    return blocks, {**round_refs, **fact_refs}, facts if show_facts else []


def attached_pictures(view: MemoryView) -> list[tuple[str, str]]:
    """要附上的原图（再感知）：（说明，data URL）。"""
    out = []
    for k, media_id in enumerate(view.attach, start=1):
        item = view.media.get(media_id)
        if item is not None:
            out.append((f"Picture {k} (shared on {item.ts:%Y-%m-%d})", image_data_url(item.path)))
    return out


class Reader:
    def __init__(self, llm: LLMClient, role: str = "reader") -> None:
        self.llm = llm
        self.role = role

    def answer(
        self,
        question: str,
        hits: Sequence[Hit],
        *,
        question_time: datetime,
        cache_scope: str | None = None,
    ) -> Answer:
        context, refs = format_memories(hits)
        output = self._read(
            question,
            f"Retrieved memories:\n{context or '(none)'}",
            question_time,
            cache_scope=cache_scope,
            memories=len(hits),
        )
        # 只保留真实存在的引用编号，防止模型编造
        citations = [c.strip("[] ") for c in output.citations if c.strip("[] ") in refs]
        return Answer(
            question=question,
            answer=output.answer,
            evidence_sufficient=output.evidence_sufficient,
            citations=citations,
            refs=refs,
            hits=list(hits),
            question_time=question_time,
        )

    def answer_with_memory(
        self,
        question: str,
        view: MemoryView,
        *,
        question_time: datetime,
        validity: Validity = "full",
        show_facts: bool = True,
        profile: bool = True,
        relative_time: bool = False,
        cache_scope: str | None = None,
    ) -> Answer:
        """事实增强检索（B2F）与 L1 的回答：核心记忆 + 带有效期的事实 + 原始对话片段。

        relative_time=True 时（L1 的时间感知打包），对话片段与事实都标上距提问几天。

        系统提示与其他方法完全相同，事实的格式说明放在上下文里，保证方法之间只差"给它看什么"。
        """
        blocks, refs, facts = memory_blocks(
            view,
            question_time,
            validity=validity,
            show_facts=show_facts,
            profile=profile,
            relative_time=relative_time,
        )
        output = self._read(
            question,
            "\n\n".join(blocks),
            question_time,
            cache_scope=cache_scope,
            pictures=attached_pictures(view),
            memories=len(view.rounds),
            facts=len(facts),
        )
        return self._answer(question, output, refs, view, facts, question_time)

    def answer_deep(
        self,
        question: str,
        view: MemoryView,
        *,
        plan: QueryPlan,
        question_time: datetime,
        cache_scope: str | None = None,
    ) -> Answer:
        """L2 的回答：证据与 L1 同样排版，另加按检索计划生成的阅读提示，并要求先记笔记再回答。

        系统提示仍与其他方法相同；阅读提示放在上下文末尾、问题之前。
        """
        blocks, refs, facts = memory_blocks(
            view, question_time, relative_time=True, highlight=plan.time_range
        )
        guide = "\n".join(f"- {line}" for line in reading_guide(plan))
        blocks.append(f"Reading guide:\n{guide}")
        output = self._read(
            question,
            "\n\n".join(blocks),
            question_time,
            cache_scope=cache_scope,
            schema=NotedReaderOutput,
            pictures=attached_pictures(view),
            memories=len(view.rounds),
            facts=len(facts),
        )
        answer = self._answer(question, output, refs, view, facts, question_time)
        answer.notes = list(output.notes) if isinstance(output, NotedReaderOutput) else []
        return answer

    @staticmethod
    def _answer(
        question: str,
        output: ReaderOutput | NotedReaderOutput,
        refs: dict[str, str],
        view: MemoryView,
        facts: list[FactHit],
        question_time: datetime,
    ) -> Answer:
        citations = [c.strip("[] ") for c in output.citations if c.strip("[] ") in refs]
        return Answer(
            question=question,
            answer=output.answer,
            evidence_sufficient=output.evidence_sufficient,
            citations=citations,
            refs=refs,
            hits=list(view.rounds),
            question_time=question_time,
            facts=facts,
        )

    def answer_from_history(
        self, question: str, history: str | list[dict[str, Any]], *, question_time: datetime
    ) -> Answer:
        """不检索，直接读给定的对话历史（全上下文基线与"只给证据会话"的上限参照用）。

        history 是内容块列表时（带图的对话，见 history_blocks），图片按原位插在对话里。
        """
        if isinstance(history, list):
            output = self._read(
                question,
                [{"type": "text", "text": "Conversation history:\n"}, *history],
                question_time,
                memories=0,
            )
        else:
            output = self._read(
                question,
                f"Conversation history:\n{history or '(none)'}",
                question_time,
                memories=0,
            )
        return Answer(
            question=question,
            answer=output.answer,
            evidence_sufficient=output.evidence_sufficient,
            citations=[],
            refs={},
            hits=[],
            question_time=question_time,
        )

    def _read(
        self,
        question: str,
        context_block: str | list[dict[str, Any]],
        question_time: datetime,
        *,
        cache_scope: str | None = None,
        schema: type[ReaderOutput] | type[NotedReaderOutput] = ReaderOutput,
        pictures: Sequence[tuple[str, str]] = (),
        **span_attrs: object,
    ) -> ReaderOutput | NotedReaderOutput:
        """组装请求。纯文本时 user 消息是一个字符串（与 N4 之前的请求逐字相同，缓存照常命中）；
        带图时改用内容块：对话里的图按原位插入，再感知的原图附在证据之后、问题之前。"""
        head = f"Current date: {question_time:%Y-%m-%d (%a) %H:%M}\n\n"
        tail = f"Question: {question}"
        content: str | list[dict[str, Any]]
        if isinstance(context_block, str) and not pictures:
            content = f"{head}{context_block}\n\n{tail}"
        else:
            blocks: list[dict[str, Any]] = [{"type": "text", "text": head}]
            if isinstance(context_block, str):
                blocks.append({"type": "text", "text": context_block})
            else:
                blocks.extend(context_block)
            if pictures:
                blocks.append(
                    {
                        "type": "text",
                        "text": "\n\nAttached pictures: the original images of the memories "
                        'marked "see picture k" above. Look at them for visual details the '
                        "descriptions may leave out.",
                    }
                )
                for label, url in pictures:
                    blocks.append({"type": "text", "text": f"\n{label}:"})
                    blocks.append({"type": "image_url", "image_url": {"url": url}})
            blocks.append({"type": "text", "text": f"\n\n{tail}"})
            content = blocks
            span_attrs = {**span_attrs, "pictures": len(pictures)}
        messages: list[Message] = [
            {"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": content},
        ]
        with span("read", **span_attrs):
            return self.llm.structured(self.role, messages, schema, cache_scope=cache_scope).value


def history_blocks(sessions: Sequence[SessionIn], *, max_side: int = 768) -> list[dict[str, Any]]:
    """带图的对话历史：文本与图片按原位交替排成内容块（全上下文基线直接看图用）。"""
    blocks: list[dict[str, Any]] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            blocks.append({"type": "text", "text": "".join(buffer)})
            buffer.clear()

    for i, session in enumerate(sorted(sessions, key=lambda s: s.started_at), start=1):
        buffer.append(f"\n[Session {i} · {session.started_at:%Y-%m-%d (%a) %H:%M}]\n")
        for turn in session.turns:
            label = turn.speaker or ("User" if turn.role == "user" else "Assistant")
            parts = turn.text.split("<image>")
            buffer.append(f"{label}: {parts[0]}")
            images = list(turn.images)
            for j, part in enumerate(parts[1:]):
                if j < len(images):
                    flush()
                    url = image_data_url(images[j], max_side=max_side)
                    blocks.append({"type": "image_url", "image_url": {"url": url}})
                buffer.append(part)
            for path in images[len(parts) - 1 :]:  # 没有占位标记的图片接在发言末尾
                flush()
                url = image_data_url(path, max_side=max_side)
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            buffer.append("\n")
    flush()
    return blocks


def format_history(sessions: Sequence[SessionIn]) -> str:
    """把若干会话按时间顺序拼成纯文本历史，每个会话标注日期。"""
    blocks: list[str] = []
    for i, session in enumerate(sorted(sessions, key=lambda s: s.started_at), start=1):
        lines = [f"[Session {i} · {session.started_at:%Y-%m-%d (%a) %H:%M}]"]
        for turn in session.turns:
            label = turn.speaker or ("User" if turn.role == "user" else "Assistant")
            lines.append(f"{label}: {turn.text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
