"""图片描述（巩固阶段，N6）：把图片压缩成可检索、可阅读的文字。

两种描述，用来检验"上下文感知的描述优于孤立描述"（假设 H6）：

- **上下文感知**（默认）：模型同时看图和这张图前后的对话。图片的含义常常取决于对话：同一张
  收据，可能是"用户买咖啡机的发票"，也可能只是在问怎么报销；
- **孤立**：只看图。

描述之外还要图中的文字（收据金额、招牌、标签）与几个关键词。描述只写看得见的东西：
回答时还可以把原图取回来再看一眼（再感知），压缩丢掉的细节可以在那时补回。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, field_validator

from cuemem.core.models import MediaItem, Turn
from cuemem.core.trace import span
from cuemem.llm.client import LLMClient, Message
from cuemem.store.media import image_data_url

CaptionMode = Literal["context", "isolated"]

CAPTION_SYSTEM = """You describe an image that was shared in a conversation with a personal assistant, for the assistant's long-term memory. Questions asked weeks later may be about any visible detail, so be concrete and complete.

- caption: 2-4 sentences. Say what the image shows and the details someone might later ask about: the objects and how many there are, colors, brands and labels, people or animals and what they are doing, the setting, and anything unusual. If the conversation around the image is given, also say in a few words what the image is in that conversation (for example "the receipt for the coffee maker the user bought"), but describe only what is visible; do not invent details.
- text_in_image: all readable text in the image (signs, labels, receipts, prices, dates), verbatim, or an empty string if there is none.
- tags: 3 to 8 short keywords."""  # noqa: E501

CONTEXT_TURN_CHARS = 1200  # 带图的那条发言最多保留多少字符
NEIGHBOR_CHARS = 500  # 前后各一条发言最多保留多少字符


class CaptionOutput(BaseModel):
    caption: str = ""
    text_in_image: str = ""
    tags: list[str] = []

    @field_validator("caption", "text_in_image", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        if isinstance(value, list):
            return " ".join(str(v) for v in value)
        return "" if value is None else str(value).strip()

    @field_validator("tags", mode="before")
    @classmethod
    def _tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        items = value if isinstance(value, list) else str(value).split(",")
        return [str(v).strip() for v in items if str(v or "").strip()][:8]


class Captioner:
    def __init__(
        self,
        llm: LLMClient,
        *,
        role: str = "captioner",
        max_side: int = 768,
        fallback_model: str | None = None,
    ) -> None:
        self.llm = llm
        self.role = role
        self.max_side = max_side
        self.fallback_model = fallback_model

    def caption(
        self,
        item: MediaItem,
        turns: Sequence[Turn],
        *,
        mode: CaptionMode,
        cache_scope: str | None = None,
    ) -> CaptionOutput:
        with span("caption", mode=mode):
            return self.llm.structured(
                self.role,
                self.messages(item, turns, mode=mode),
                CaptionOutput,
                cache_scope=cache_scope,
                fallback_model=self.fallback_model,
            ).value

    def messages(
        self, item: MediaItem, turns: Sequence[Turn], *, mode: CaptionMode
    ) -> list[Message]:
        if mode == "context":
            instruction = (
                f"{render_context(item, turns)}\n\n"
                "Describe the image attached to the message marked >>>."
            )
        else:
            instruction = "Describe this image."
        content: list[dict[str, Any]] = [
            {"type": "text", "text": instruction},
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(item.path, max_side=self.max_side)},
            },
        ]
        return [
            {"role": "system", "content": CAPTION_SYSTEM},
            {"role": "user", "content": content},
        ]


def render_context(item: MediaItem, turns: Sequence[Turn]) -> str:
    """带图发言及其前后各一条发言；带图的那条用 >>> 标出，同一条里有几张图时说明是第几张。"""
    ordered = sorted(turns, key=lambda t: t.idx)
    position = next((i for i, t in enumerate(ordered) if t.id == item.source_turn_id), None)
    if position is None:
        return "Conversation: (not available)"
    lines = [f"Conversation date: {ordered[position].ts:%Y-%m-%d (%A)}"]
    for i in range(max(0, position - 1), min(len(ordered), position + 2)):
        turn = ordered[i]
        limit = CONTEXT_TURN_CHARS if i == position else NEIGHBOR_CHARS
        text = turn.text.strip()
        if len(text) > limit:
            text = text[:limit].rstrip() + " […]"
        label = turn.speaker or ("User" if turn.role == "user" else "Assistant")
        marker = ">>> " if i == position else ""
        lines.append(f"{marker}{label}: {text}")
    current = ordered[position]
    if len(current.media_ids) > 1 and item.id in current.media_ids:
        number = current.media_ids.index(item.id) + 1
        count = len(current.media_ids)
        lines.append(f"(The marked message has {count} images; this is image {number}.)")
    return "\n".join(lines)


def image_note(item: MediaItem, *, isolated: bool = False) -> str:
    """一张图在文本里的写法：描述加图中文字。用于抽取的输入、检索单元与回答时的证据。

    孤立描述在生成时已经把图中文字并进去了（caption_iso）。
    """
    if isolated:
        return item.caption_iso.strip() or "an image"
    note = item.caption.strip() or "an image"
    if item.ocr_text.strip():
        note += f" Text in the image: {item.ocr_text.strip()}"
    return note


def with_text(output: CaptionOutput) -> str:
    """描述与图中文字合成一段（孤立描述按这种形式存）。"""
    if output.text_in_image:
        return f"{output.caption} Text in the image: {output.text_in_image}".strip()
    return output.caption


def inline_images(text: str, notes: Sequence[str]) -> str:
    """把文本里的 "<image>" 标记依次换成 [Image: 描述]；标记不够时把多出的描述接在末尾。"""
    return fill_images(text, [f"[Image: {n}]" for n in notes])


def fill_images(text: str, rendered: Sequence[str]) -> str:
    """把 "<image>" 标记依次换成 rendered 里的文字（多出来的接在末尾，不够的写成 [Image]）。"""
    parts = text.split("<image>")
    out = [parts[0]]
    for i, part in enumerate(parts[1:]):
        out.append(rendered[i] if i < len(rendered) else "[Image]")
        out.append(part)
    filled = "".join(out)
    extra = rendered[len(parts) - 1 :]
    if extra:
        filled = filled.rstrip() + " " + " ".join(extra)
    return filled
