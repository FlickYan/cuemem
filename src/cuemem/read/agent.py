"""记忆工具与"模型自主检索"（基线 B3，设计文档 §8.3）。

reader 拿到两个工具，自己决定要不要检索、检索什么、检索几次（ReAct），最后调用 answer 交卷：

- memory_search(query)：在记忆里检索（L1 快速召回），返回带编号与日期的事实和对话片段；
- date_diff(start, end)：两个日期相差几天（避免模型算错日期）。

这是"让 LLM 自己决定检索"的代表做法。检索控制器（N7）要回答的问题正是：把这个决定交给一个
便宜、可校准的外部策略，是否比交给 LLM 更划算、更稳定。这里的工具接口也是之后 Agent Runtime
（N8）与 MCP Server 暴露给其他 Agent 的接口。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any

from pydantic import ValidationError

from cuemem.core.models import Answer, CoreMemory, FactHit, Hit
from cuemem.core.trace import record_event, span
from cuemem.llm.client import LLMClient, Message
from cuemem.read.facts import MemoryView, format_facts
from cuemem.read.l1 import L1Config, L1Result
from cuemem.read.reader import READER_SYSTEM, Reader, ReaderOutput, relative_days

AGENT_RULES = """You can search your memory of earlier conversations with the memory_search tool before answering. Search when the question may depend on what the user told you before; you may search several times with different queries (for example once per item or event the question involves), but one to three searches are usually enough. Use date_diff for date arithmetic. When you are ready, call the answer tool exactly once."""  # noqa: E501

MEMORY_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Search the memory of earlier conversations with the user. Returns short facts "
            "and conversation excerpts with their dates and ids."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "what to look for, phrased the way the user would have said it",
                }
            },
            "required": ["query"],
        },
    },
}
DATE_DIFF_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "date_diff",
        "description": "Number of days from start to end (dates as YYYY-MM-DD).",
        "parameters": {
            "type": "object",
            "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
            "required": ["start", "end"],
        },
    },
}
ANSWER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "Give the final answer to the question.",
        "parameters": ReaderOutput.model_json_schema(),
    },
}
TOOLS = [MEMORY_SEARCH_TOOL, DATE_DIFF_TOOL, ANSWER_TOOL]

SearchFn = Callable[[str, L1Config], L1Result]


def date_diff(start: str, end: str) -> str:
    """两个日期相差的天数，附带周与月的近似值。"""
    try:
        first, last = date.fromisoformat(start.strip()[:10]), date.fromisoformat(end.strip()[:10])
    except ValueError:
        return "Invalid date: use YYYY-MM-DD."
    days = (last - first).days
    weeks, rest = divmod(abs(days), 7)
    return (
        f"{days} days from {first} to {last} "
        f"({weeks} weeks and {rest} days; about {abs(days) / 30.4:.1f} months)"
    )


@dataclass
class SearchSession:
    """一次回答过程中的检索记录：编号跨多次检索连续，已经给过的记忆不再重复给。"""

    question_time: datetime
    search: SearchFn
    config: L1Config
    rounds: list[Hit] = field(default_factory=list)
    facts: list[FactHit] = field(default_factory=list)
    refs: dict[str, str] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)

    def run(self, query: str) -> str:
        self.queries.append(query)
        result = self.search(query, self.config)
        shown_rounds = {h.unit.id for h in self.rounds}
        shown_facts = {fh.fact.id for fh in self.facts}
        new_rounds = [h for h in result.rounds if h.unit.id not in shown_rounds]
        new_facts = [fh for fh in result.facts if fh.fact.id not in shown_facts]
        blocks: list[str] = []
        if new_facts:
            text, fact_refs = format_facts(new_facts, question_time=self.question_time)
            offset = len(self.facts)
            renumber = {ref: f"f{offset + i}" for i, ref in enumerate(fact_refs, start=1)}
            # 一次性替换编号（逐个 replace 会把刚改过的编号再改一遍）
            text = re.sub(r"\[(f\d+)\]", lambda m: f"[{renumber.get(m[1], m[1])}]", text)
            for old, new in renumber.items():
                self.refs[new] = fact_refs[old]
            blocks.append(text)
            self.facts.extend(new_facts)
        for hit in sorted(new_rounds, key=lambda h: (h.unit.ts, h.unit.id)):
            ref = f"m{len(self.rounds) + 1}"
            self.refs[ref] = hit.unit.id
            self.rounds.append(hit)
            header = (
                f"[{ref}] {hit.unit.ts:%Y-%m-%d (%a) %H:%M} · "
                f"{relative_days(hit.unit.ts, self.question_time)}"
            )
            blocks.append(f"{header}\n{hit.unit.text}")
        if not blocks:
            return "No new memories found for this query (earlier results are still valid)."
        return "\n\n".join(blocks)


class ToolReader:
    def __init__(self, llm: LLMClient, *, role: str = "reader", max_steps: int = 5) -> None:
        self.llm = llm
        self.role = role
        self.max_steps = max_steps
        self.reader = Reader(llm, role=role)

    def answer(
        self,
        question: str,
        *,
        question_time: datetime,
        search: SearchFn,
        core: CoreMemory | None,
        config: L1Config | None = None,
        cache_scope: str | None = None,
    ) -> Answer:
        session = SearchSession(
            question_time=question_time,
            search=search,
            config=config or replace(L1Config(), budget_tokens=2000),
        )
        context = f"Current date: {question_time:%Y-%m-%d (%a) %H:%M}"
        if core is not None and core.text:
            context += f"\n\nAbout the user (core memory, current facts):\n{core.text}"
        messages: list[Message] = [
            {"role": "system", "content": f"{READER_SYSTEM}\n\n{AGENT_RULES}"},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ]
        output: ReaderOutput | None = None
        for step in range(1, self.max_steps + 1):
            final = step == self.max_steps
            with span("agent_step", step=step):
                # 最后一步只给 answer 并点名要求调用它（只限制可用工具不够：模型会照着
                # 历史里出现过的工具名继续检索）
                result = self.llm.chat(
                    self.role,
                    messages,
                    tools=[ANSWER_TOOL] if final else TOOLS,
                    tool_choice=(
                        {"type": "function", "function": {"name": "answer"}}
                        if final
                        else "required"
                    ),
                    cache_scope=cache_scope,
                )
            if not result.tool_calls:  # 模型没有调用工具，直接给了文字：当作回答
                output = ReaderOutput(answer=result.content or "", evidence_sufficient=True)
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                        for call in result.tool_calls
                    ],
                }
            )
            for call in result.tool_calls:
                if call.name == "answer":
                    output = _parse_answer(call.arguments)
                    break
                reply = self._run_tool(call.name, call.arguments, session)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
            if output is not None:
                break
        record_event("agent", searches=len(session.queries), queries=session.queries)
        details: dict[str, Any] = {"searches": session.queries}
        if output is None:
            # 步数用完还没交卷：用标准 reader 读它检索到的全部记忆作答（并在结果里注明）
            details["forced_answer"] = True
            answer = self.reader.answer_with_memory(
                question,
                MemoryView(rounds=session.rounds, facts=session.facts, core=core),
                question_time=question_time,
                relative_time=True,
                cache_scope=cache_scope,
            )
            answer.details = details
            return answer
        citations = [c.strip("[] ") for c in output.citations if c.strip("[] ") in session.refs]
        return Answer(
            question=question,
            answer=output.answer,
            evidence_sufficient=output.evidence_sufficient,
            citations=citations,
            refs=session.refs,
            hits=session.rounds,
            question_time=question_time,
            facts=session.facts,
            details=details,
        )

    @staticmethod
    def _run_tool(name: str, arguments: str, session: SearchSession) -> str:
        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return "Invalid arguments: expected a JSON object."
        if name == "memory_search" and isinstance(args.get("query"), str):
            return session.run(args["query"])
        if name == "date_diff":
            return date_diff(str(args.get("start", "")), str(args.get("end", "")))
        return f"Unknown tool or missing arguments: {name}"


def _parse_answer(arguments: str) -> ReaderOutput:
    try:
        return ReaderOutput.model_validate_json(arguments or "{}")
    except ValidationError:
        try:
            raw = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            raw = {}
        text = raw.get("answer") if isinstance(raw, dict) else None
        return ReaderOutput(answer=str(text or arguments), evidence_sufficient=True)
