"""召回的证据（N12）：记忆代理、MCP、HTTP 接口与各框架集成共用的"想起 → 排成证据 → 注入"。

两种形态：

- **证据块**（recall_evidence / inject_evidence）：控制器选档后，把画像、带有效期的事实、按时间
  排列的对话片段排成与回答模型相同的文字（read/reader.py 的 memory_blocks），包在 <memory> 里
  放在最后一句用户消息的开头。记忆代理与框架集成的"自动想起"都用它；
- **清单**（recall_listing / format_listing）：给另一个 Agent 自己推理用的结构化结果，每条带 ID，
  可以按 ID 遗忘（resolve_ids）。MCP 的 recall 工具与 HTTP 的 /recall 都用它。

以前这几段在记忆代理、服务层与三个集成里各写了一份（集成评审时查出），这里收拢成一份。
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from cuemem.core.models import Fact
from cuemem.core.timeutil import naive_local
from cuemem.memory import MemorySystem, Recall, RecallMode
from cuemem.read.controller import Router, english_reason
from cuemem.read.reader import memory_blocks

MEMORY_NOTE = (
    "Memories about the user from earlier conversations, recalled by the memory system. Use "
    "them when they help and ignore the rest. If a fact was superseded, rely on the newer one. "
    "Don't mention this block unless the user asks how you know something."
)
RAISED_REASON = "raised from L0: the caller asked to recall"


@dataclass
class Evidence:
    """一次自动想起的结果。block 为空表示没有要注入的内容（L0 且没有画像，或什么都没取到）。"""

    query: str
    level: str  # L0 / L1 / L2
    reasons: list[str]
    recalled: int  # 注入的条数：画像算一条，加上事实与对话片段
    block: str
    see: bool = False  # 控制器想看原图（证据块只有描述，原图由调用方决定要不要附上）
    recall: Recall | None = None


def recall_evidence(
    system: MemorySystem,
    space: str,
    query: str,
    *,
    at: datetime | None = None,
    mode: RecallMode = "auto",
    router: Router = "learned",
    lam: float | None = None,
    exclude: Collection[str] = (),
) -> Evidence:
    """控制器选档、取回记忆、排成证据块。exclude：已经原样在上下文里的轮与图片的 ID。"""
    now = naive_local(at) if at is not None else datetime.now()  # 证据块里的"几天前"也按本机时间
    recall = system.retrieve(
        space, query, question_time=now, mode=mode, router=router, lam=lam, exclude=exclude
    )
    return evidence_from(recall, query, now)


def evidence_from(recall: Recall, query: str, now: datetime) -> Evidence:
    view = recall.view
    if recall.level == "l0":  # 不需要回忆：只给画像（没有画像就什么都不加）
        view = replace(view, rounds=[], facts=[], attach=[])
    blocks, refs, _ = memory_blocks(view, now, relative_time=True)
    if not view.rounds:  # 没取到对话片段时不留一个空的"(none)"块
        blocks = blocks[:-1]
    has_profile = view.core is not None and bool(view.core.text)
    decision = recall.decision
    return Evidence(
        query=query,
        level=recall.level.upper(),
        reasons=[english_reason(r) for r in decision.reasons] if decision else [],
        recalled=len(refs) + int(has_profile),
        block="\n\n".join([MEMORY_NOTE, *blocks]) if blocks else "",
        see=bool(decision and decision.see),
        recall=recall,
    )


def inject_evidence(messages: Sequence[dict[str, Any]], block: str) -> list[dict[str, Any]]:
    """把证据块放在最后一句消息的开头（OpenAI 格式；不改调用方的列表与消息）。

    放在最后而不是系统提示里：系统提示与之前的消息不动，供应商的前缀缓存照常命中。"""
    if not block or not messages:
        return list(messages)
    last = dict(messages[-1])
    note = f"<memory>\n{block}\n</memory>\n\n"
    content = last.get("content")
    if isinstance(content, list):
        last["content"] = [{"type": "text", "text": note}, *content]
    else:
        last["content"] = note + (content or "")
    return [*messages[:-1], last]


def message_text(content: Any) -> str:
    """OpenAI 格式的消息内容（字符串或内容块）→ 文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if b.get("type") == "text")
    return ""


# ---------- 清单：给另一个 Agent 自己推理 ----------


def recall_listing(
    system: MemorySystem,
    space: str,
    query: str,
    *,
    mode: RecallMode = "auto",
    router: Router = "learned",
    lam: float | None = None,
    at: datetime | None = None,
    at_least_l1: bool = False,
) -> dict[str, Any]:
    """检索控制器选档、取回记忆，原样交给调用方（另一个 Agent）去推理，不调用回答模型。

    at_least_l1：调用方已经决定要查记忆（例如 MCP 的 recall），控制器选了 L0 时改走 L1，
    控制器只决定要不要查得更深。
    """
    recall = system.retrieve(space, query, question_time=at, mode=mode, router=router, lam=lam)
    raised = at_least_l1 and recall.level == "l0"
    if raised:
        recall = system.retrieve(space, query, question_time=at, mode="l1")
    view = recall.view
    decision = recall.decision
    return {
        "level": recall.level.upper(),
        "see": bool(decision and decision.see),
        "reasons": (
            [RAISED_REASON]
            if raised
            else [english_reason(r) for r in decision.reasons]
            if decision
            else []
        ),
        "profile": view.core.text if view.core is not None else "",
        "facts": [
            {
                "id": fh.fact.id,
                "statement": fh.fact.statement,
                "said_on": fh.fact.recorded_at.isoformat(),
                "status": fact_state(fh.fact),
            }
            for fh in view.facts
        ],
        "excerpts": [
            {
                "id": hit.unit.id,
                "session_id": hit.unit.session_id,
                "ts": hit.unit.ts.isoformat(),
                "text": hit.unit.text,
                "turn_ids": hit.unit.source_turn_ids,
            }
            for hit in sorted(view.rounds, key=lambda h: h.unit.ts)
        ],
        "images": [
            {"id": m.id, "caption": m.caption, "path": m.path, "attached": m.id in view.attach}
            for m in view.media.values()
        ],
    }


def format_listing(found: dict[str, Any]) -> str:
    """recall_listing 的结果 → 给模型看的文字（MCP 与各框架的 recall 工具共用）。"""
    why = f" ({', '.join(found['reasons'])})" if found["reasons"] else ""
    lines = [f"Memory controller: {found['level']}{why}"]
    if found["profile"]:
        lines += ["", "Profile:", found["profile"]]
    if found["facts"]:
        lines += ["", "Facts:"]
        lines += [
            f"[{f['id']}] (said {f['said_on'][:10]}; {f['status']}) {f['statement']}"
            for f in found["facts"]
        ]
    if found["excerpts"]:
        lines += ["", "Conversation excerpts:"]
        for e in found["excerpts"]:
            lines.append(f"[{e['id']}] {e['ts'][:16].replace('T', ' ')}\n{e['text']}")
    if not found["facts"] and not found["excerpts"]:
        lines += ["", "No memories were retrieved for this query."]
    return "\n".join(lines)


def resolve_ids(
    system: MemorySystem, space: str, ids: Sequence[str]
) -> tuple[list[str], list[str]]:
    """recall 给出的 ID → （事实 ID，对话片段出自的发言 ID）。不认识的 ID 忽略。"""
    fact_ids = [f.id for f in system.db.get_facts(space, ids=list(ids))]
    rounds = system.db.get_units(space, list(ids))
    turns = [t for unit in rounds if unit.kind == "round" for t in unit.source_turn_ids]
    return fact_ids, turns


def fact_state(fact: Fact) -> str:
    """界面与清单里的有效状态：当前有效 / 何时因现实变化而失效 / 何时被更正。"""
    if fact.is_current:
        return "current"
    when = f"{fact.invalidated_at:%Y-%m-%d}" if fact.invalidated_at else "?"
    if fact.supersede_reason == "corrected":
        return f"corrected on {when}"
    return f"changed on {when}" + (f" (until {fact.valid_to})" if fact.valid_to else "")
