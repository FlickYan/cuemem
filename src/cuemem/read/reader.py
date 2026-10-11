"""Reader：把检索到的记忆按时间顺序编号放进上下文，让 LLM 带引用地回答。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, Field

from cuemem.core.models import Answer, Hit, SessionIn
from cuemem.core.trace import span
from cuemem.llm.client import LLMClient
from cuemem.read.facts import MemoryView, Validity, format_facts

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


def format_memories(hits: Sequence[Hit]) -> tuple[str, dict[str, str]]:
    """按时间顺序排列证据并编号，返回（上下文文本，编号 → 记忆单元 ID）。"""
    ordered = sorted(hits, key=lambda h: (h.unit.ts, h.unit.id))
    refs: dict[str, str] = {}
    blocks: list[str] = []
    for i, hit in enumerate(ordered, start=1):
        ref = f"m{i}"
        refs[ref] = hit.unit.id
        header = f"[{ref}] {hit.unit.ts:%Y-%m-%d (%a) %H:%M} · session {hit.unit.session_id}"
        blocks.append(f"{header}\n{hit.unit.text}")
    return "\n\n".join(blocks), refs


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
        cache_scope: str | None = None,
    ) -> Answer:
        """事实增强检索（B2F）的回答：核心记忆 + 带有效期的事实 + 原始对话片段。

        系统提示与其他方法完全相同，事实的格式说明放在上下文里，保证方法之间只差"给它看什么"。
        """
        blocks: list[str] = []
        if profile and view.core is not None and view.core.text:
            # 多人对话（如 LoCoMo）的核心记忆按人分组，每组以"名字:"开头
            who = "the user" if view.core.text.startswith("- ") else "the people in these chats"
            blocks.append(f"About {who} (core memory, current facts):\n{view.core.text}")
        facts = [fh for fh in view.facts if fh.fact.is_current or validity != "current"]
        fact_refs: dict[str, str] = {}
        if show_facts and facts:
            text, fact_refs = format_facts(facts, validity=validity)
            blocks.append(text)
        context, round_refs = format_memories(view.rounds)
        blocks.append(f"Retrieved conversation excerpts:\n{context or '(none)'}")
        output = self._read(
            question,
            "\n\n".join(blocks),
            question_time,
            cache_scope=cache_scope,
            memories=len(view.rounds),
            facts=len(fact_refs),
        )
        refs = {**round_refs, **fact_refs}
        citations = [c.strip("[] ") for c in output.citations if c.strip("[] ") in refs]
        return Answer(
            question=question,
            answer=output.answer,
            evidence_sufficient=output.evidence_sufficient,
            citations=citations,
            refs=refs,
            hits=list(view.rounds),
            question_time=question_time,
            facts=facts if show_facts else [],
        )

    def answer_from_history(
        self, question: str, history: str, *, question_time: datetime
    ) -> Answer:
        """不检索，直接读给定的对话历史（全上下文基线与"只给证据会话"的上限参照用）。"""
        output = self._read(
            question, f"Conversation history:\n{history or '(none)'}", question_time, memories=0
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
        context_block: str,
        question_time: datetime,
        *,
        cache_scope: str | None = None,
        **span_attrs: object,
    ) -> ReaderOutput:
        prompt = (
            f"Current date: {question_time:%Y-%m-%d (%a) %H:%M}\n\n"
            f"{context_block}\n\n"
            f"Question: {question}"
        )
        messages = [
            {"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        with span("read", **span_attrs):
            return self.llm.structured(
                self.role, messages, ReaderOutput, cache_scope=cache_scope
            ).value


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
