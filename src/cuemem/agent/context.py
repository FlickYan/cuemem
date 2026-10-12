"""上下文工程（设计文档 §6.6，N8）：一轮对话交给模型的内容分区装配，各区有 token 预算。

分区（从前到后）：

1. 系统提示：角色、规则、引用格式、领域说明——所有会话都相同；
2. 核心记忆（用户画像）：会话开始时取一次快照，整个会话不变；每条带编号（p1……），可以引用；
3. 工作记忆：当前会话最近几轮的原话（含图片），超出预算时成段淘汰；
4. 本轮召回的记忆：检索控制器选档后取到的事实、对话片段与图片（再感知时附原图）；
5. 当前消息：日期 + 用户的话（+ 图片）。

1–3 组成稳定前缀：同一会话里，每一轮的请求都以上一轮请求的前缀开头，只在末尾追加，
供应商的前缀缓存（DeepSeek 与通义的隐式缓存）就能持续命中。易变的内容（召回的记忆、日期）
只放在最后一条消息里，下一轮不再保留——召回每轮重算，不污染前缀。工作记忆超出预算时
一次淘汰到预算的一半，而不是每轮挤掉最早的一句：前缀只在淘汰的那一轮变一次。
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from cuemem.core.models import FactHit
from cuemem.llm.client import Message
from cuemem.read.facts import MemoryView
from cuemem.read.l1 import estimate_tokens
from cuemem.read.reader import attached_pictures, memory_blocks
from cuemem.store.media import image_data_url

Layout = Literal["stable", "naive"]

# 回答里的引用编号：[f2]、[m3]、[i1]，也接受 [f2, m3] 这样写在一起的；编号不存在的
# （模型编造的）不算引用，但同样从正文里去掉
_CITATION = re.compile(r"\[\s*([a-z]\d+(?:\s*[,，;；]\s*[a-z]\d+)*)\s*\]")
_REF = re.compile(r"[a-z]\d+")


@dataclass(frozen=True)
class ContextBudget:
    """各分区的 token 上限（估算值：汉字约 1 个 token，其他文字约 4 个字符 1 个 token）。"""

    core: int = 600  # 核心记忆（巩固时上限约 400 tokens，这里留些余量）
    working: int = 3000  # 工作记忆：当前会话最近几轮的原话
    recalled: int = 4000  # 召回的对话片段（L1 打包预算；L2 用自己的 6k）
    evict_to: float = 0.5  # 工作记忆超出预算时，一次淘汰到预算的这个比例
    image_tokens: int = 800  # 一张图按多少 token 计（缩放到最长边 768 像素之后的量级）

    def as_dict(self) -> dict[str, int]:
        return {"core": self.core, "working": self.working, "recalled": self.recalled}


@dataclass
class Exchange:
    """工作记忆里的一轮：用户的话（可带图）与助手的回复（去掉了引用编号）。"""

    user: str
    assistant: str
    at: datetime
    images: list[tuple[str, str]] = field(default_factory=list)  # （编号 u1，本地路径）
    user_idx: int = 0  # 用户这句在会话里的发言序号（写入记忆时的轮次编号）

    def tokens(self, budget: ContextBudget) -> int:
        return (
            estimate_tokens(self.user)
            + estimate_tokens(self.assistant)
            + budget.image_tokens * len(self.images)
        )


@dataclass
class Context:
    """装配好的一轮请求。"""

    messages: list[Message]
    refs: dict[str, str]  # 编号 → 记忆 ID：p1、f1 → 事实，m1 → 轮，i1 → 图片
    facts: list[FactHit]  # 实际展示的事实（编号接着往下排时要用）
    partitions: dict[str, int]  # 各分区的 token 估算
    prefix: int  # 稳定前缀的消息条数（系统提示 + 工作记忆）


def truncate_lines(text: str, budget: int) -> tuple[str, bool]:
    """按行截断到预算以内（核心记忆一行一条事实，整行保留或整行丢弃）。"""
    kept: list[str] = []
    used = 0
    for line in text.splitlines():
        cost = estimate_tokens(line)
        if kept and used + cost > budget:
            return "\n".join(kept), True
        kept.append(line)
        used += cost
    return "\n".join(kept), False


def system_prompt(
    persona: str, profile: Sequence[tuple[str, str]], budget: ContextBudget
) -> tuple[str, int, list[str]]:
    """系统提示 + 用户画像快照（核心记忆，每条带编号 p1、p2……，与召回的记忆一样可以引用）。

    返回（文本，画像的 token 数，放进去的画像编号）。
    """
    if not profile:
        return f"{persona}\n\nAbout the user: nothing is known yet.", 0, []
    lines = [f"[{ref}] {statement}" for ref, statement in profile]
    text, cut = truncate_lines("\n".join(lines), budget.core)
    shown = [ref for ref, _ in profile[: len(text.splitlines())]]
    note = "; truncated" if cut else ""
    block = f"About the user (profile from earlier conversations{note}):\n{text}"
    return f"{persona}\n\n{block}", estimate_tokens(text), shown


def history(window: Sequence[Exchange]) -> list[Message]:
    """工作记忆排成真实的多轮对话。带图的用户消息用内容块，图片按原样附上（同一张图每次编码
    结果相同，前缀不变）。"""
    messages: list[Message] = []
    for exchange in window:
        if exchange.images:
            blocks: list[dict[str, Any]] = [{"type": "text", "text": exchange.user}]
            for label, path in exchange.images:
                blocks.append({"type": "text", "text": f"\n[Picture {label}]"})
                blocks.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
            messages.append({"role": "user", "content": blocks})
        else:
            messages.append({"role": "user", "content": exchange.user})
        messages.append({"role": "assistant", "content": exchange.assistant})
    return messages


def turn_message(
    text: str,
    when: datetime,
    view: MemoryView | None,
    uploads: Sequence[tuple[str, str]],
) -> tuple[Message, dict[str, str], list[FactHit], int, int]:
    """这一轮的用户消息：日期、召回的记忆（与评测里的 reader 同一种排版）、用户的话与图片。

    返回（消息，编号 → 记忆 ID，展示的事实，召回部分的 token 数，再感知的原图张数）。
    """
    head = f"Current date: {when:%Y-%m-%d (%a) %H:%M}"
    refs: dict[str, str] = {}
    facts: list[FactHit] = []
    recalled = ""
    pictures: list[tuple[str, str]] = []
    if view is not None and (view.rounds or view.facts):
        blocks, refs, facts = memory_blocks(view, when, profile=False, relative_time=True)
        recalled = (
            "Recalled memories for this message (from earlier conversations; "
            "cite their ids when you use them):\n\n" + "\n\n".join(blocks)
        )
        pictures = attached_pictures(view)
    else:
        recalled = "Recalled memories for this message: none."
    tail = f"User message:\n{text}"
    if not pictures and not uploads:
        content: str | list[dict[str, Any]] = f"{head}\n\n{recalled}\n\n{tail}"
    else:
        parts: list[dict[str, Any]] = [{"type": "text", "text": f"{head}\n\n{recalled}"}]
        if pictures:
            parts.append(
                {
                    "type": "text",
                    "text": "\n\nAttached pictures: the original images of the memories "
                    'marked "see picture k" above.',
                }
            )
            for label, url in pictures:
                parts.append({"type": "text", "text": f"\n{label}:"})
                parts.append({"type": "image_url", "image_url": {"url": url}})
        parts.append({"type": "text", "text": f"\n\n{tail}"})
        for label, path in uploads:
            parts.append({"type": "text", "text": f"\n[Picture {label}]"})
            parts.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
        content = parts
    message: Message = {"role": "user", "content": content}
    return message, refs, facts, estimate_tokens(recalled), len(pictures)


def build_context(
    *,
    persona: str,
    profile: Sequence[tuple[str, str, str]],
    window: Sequence[Exchange],
    text: str,
    when: datetime,
    view: MemoryView | None,
    uploads: Sequence[tuple[str, str]],
    budget: ContextBudget,
    tools: Sequence[dict[str, Any]] = (),
    layout: Layout = "stable",
) -> Context:
    """layout="naive" 只用于对照实验：把日期与召回的记忆放进系统提示（很多实现的写法），
    每轮的请求从第一条消息起就不同，供应商的前缀缓存只能命中角色说明那一小段。"""
    system, core_tokens, shown = system_prompt(
        persona, [(ref, statement) for ref, _, statement in profile], budget
    )
    message, refs, facts, recalled_tokens, pictures = turn_message(text, when, view, uploads)
    if layout == "naive":
        message, system = _naive(message, system)
    messages: list[Message] = [{"role": "system", "content": system}, *history(window)]
    prefix = len(messages) if layout == "stable" else 0
    messages.append(message)
    profile_ids = {ref: fact_id for ref, fact_id, _ in profile}
    refs = {**{ref: profile_ids[ref] for ref in shown}, **refs}
    partitions = {
        "system": estimate_tokens(persona),
        "tools": estimate_tokens(json.dumps(list(tools), ensure_ascii=False)) if tools else 0,
        "core": core_tokens,
        "working": sum(e.tokens(budget) for e in window),
        "recalled": recalled_tokens + budget.image_tokens * pictures,
        "message": estimate_tokens(text) + budget.image_tokens * len(uploads),
    }
    return Context(messages, refs, facts, partitions, prefix)


def _naive(message: Message, system: str) -> tuple[Message, str]:
    """对照布局：最后一条消息里"用户的话"之前的部分（日期、召回的记忆、附图说明）挪进系统提示，
    图片仍留在用户消息里。"""
    content = message["content"]
    if isinstance(content, str):
        head, _, tail = content.partition("User message:\n")
        return {"role": "user", "content": tail}, f"{system}\n\n{head.strip()}"
    texts: list[str] = []
    rest: list[dict[str, Any]] = []
    moved = True
    for block in content:
        text = block.get("text", "") if block.get("type") == "text" else ""
        if moved and "User message:\n" in text:
            before, _, after = text.partition("User message:\n")
            texts.append(before)
            rest.append({"type": "text", "text": after})
            moved = False
        elif moved and block.get("type") == "text":
            texts.append(text)
        else:
            rest.append(block)  # 再感知的原图在"用户的话"之前：作为用户消息的一部分保留
    return {"role": "user", "content": rest}, f"{system}\n\n{''.join(texts).strip()}"


def evict(window: list[Exchange], budget: ContextBudget) -> int:
    """工作记忆超出预算时，从最早的一轮开始淘汰到预算的 evict_to 以内（至少留下最近一轮）。

    返回淘汰的轮数；被淘汰的内容已经写入长期记忆，之后可以被检索到。
    """
    sizes = [e.tokens(budget) for e in window]
    total = sum(sizes)
    if total <= budget.working:
        return 0
    target = budget.working * budget.evict_to
    dropped = 0
    while dropped < len(window) - 1 and total > target:
        total -= sizes[dropped]
        dropped += 1
    del window[:dropped]
    return dropped


def citations(answer: str, refs: dict[str, str]) -> list[str]:
    """回答里引用了哪些记忆（只保留真实存在的编号，按首次出现的顺序）。"""
    found: list[str] = []
    for group in _CITATION.findall(answer):
        for ref in _REF.findall(group):
            if ref in refs and ref not in found:
                found.append(ref)
    return found


def strip_citations(answer: str) -> str:
    """去掉引用编号：写入记忆与放进工作记忆的是用户看到的话，编号只在当轮有意义。"""
    text = _CITATION.sub("", answer)
    text = re.sub(r"[ \t]+([,.;:!?，。；：！？])", r"\1", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
