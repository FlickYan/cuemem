"""Agent 的工具：统一的定义方式，加上两个记忆工具。

- memory_search：逃生通道。检索控制器每轮已经按需召回了记忆；模型觉得还缺什么时可以自己再查
  （与基线 B3 的工具相同，编号接着本轮已召回的往下排，已经给过的不再重复给）；
- date_diff：日期相减，避免模型算错天数。

领域工具（例如电商演示的商品检索，见 agent/shop.py）用同样的方式定义，交给 Agent 即可。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cuemem.read.agent import DATE_DIFF_TOOL, MEMORY_SEARCH_TOOL, SearchSession, date_diff

if TYPE_CHECKING:
    from cuemem.memory import MemorySystem


@dataclass
class ToolResult:
    text: str  # 给模型看的结果
    data: dict[str, Any] = field(default_factory=dict)  # 结构化结果（界面展示用，例如商品卡片）


@dataclass
class ToolContext:
    """工具执行时能用到的东西：记忆系统、当前空间与时间、本轮的检索记录、可引用的图片。"""

    system: MemorySystem
    space_id: str
    when: datetime
    memory: SearchSession  # 本轮已经给模型看过的记忆（编号 → 记忆 ID 在 memory.refs 里）
    uploads: dict[str, str] = field(default_factory=dict)  # 用户在本会话发的图：u1 → 本地路径

    def image_path(self, label: str) -> Path | None:
        """模型说的图片编号 → 本地文件：u1 是用户在本会话发的图，i1 是召回的记忆里的图。"""
        label = label.strip().strip("[]")
        if label in self.uploads:
            return Path(self.uploads[label])
        media_id = self.memory.refs.get(label)
        if media_id is None:
            return None
        items = self.system.db.get_media(self.space_id, ids=[media_id])
        return Path(items[0].path) if items else None


ToolFn = Callable[[dict[str, Any], ToolContext], ToolResult]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    run: ToolFn

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _memory_search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    query = str(args.get("query", "")).strip()
    if not query:
        return ToolResult("Give a non-empty query.")
    return ToolResult(ctx.memory.run(query))


def memory_tools() -> list[Tool]:
    search = MEMORY_SEARCH_TOOL["function"]
    diff = DATE_DIFF_TOOL["function"]
    return [
        Tool(search["name"], search["description"], search["parameters"], _memory_search),
        Tool(
            diff["name"],
            diff["description"],
            diff["parameters"],
            lambda args, ctx: ToolResult(
                date_diff(str(args.get("start", "")), str(args.get("end", "")))
            ),
        ),
    ]
