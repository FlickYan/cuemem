"""Reader：把检索到的记忆按时间顺序编号放进上下文，让 LLM 带引用地回答。"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, Field

from cuemem.core.models import Answer, Hit
from cuemem.core.trace import span
from cuemem.llm.client import LLMClient

# 评测基准以英文为主，提示词用英文；回答语言跟随问题
READER_SYSTEM = """You are a personal assistant with long-term memory of past conversations with the user.
You will receive memories retrieved from earlier conversations. Each memory has an id such as [m1] and the date it happened.

Rules:
1. Base your answer on the memories. If memories conflict, trust the most recent one unless the question asks about the past.
2. Interpret relative time expressions ("last week", "上个月") relative to the current date given below.
3. If the memories do not contain the answer, say that you don't know. Never invent details.
4. Answer in the same language as the question, concisely.
5. In "citations", list the ids of the memories you actually used."""  # noqa: E501


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

    def answer(self, question: str, hits: Sequence[Hit], *, question_time: datetime) -> Answer:
        context, refs = format_memories(hits)
        prompt = (
            f"Current date: {question_time:%Y-%m-%d (%a) %H:%M}\n\n"
            f"Retrieved memories:\n{context or '(none)'}\n\n"
            f"Question: {question}"
        )
        messages = [
            {"role": "system", "content": READER_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        with span("read", memories=len(hits)):
            output = self.llm.structured(self.role, messages, ReaderOutput).value
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
