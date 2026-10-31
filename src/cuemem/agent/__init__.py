"""Agent Runtime（N8）：自研的对话循环——控制器按需想起、分区装配上下文、工具调用、在线写回。

按需导入（与 cuemem/__init__.py 相同）：只用 agent.shop、agent.context 这类模块时不连带加载
整个记忆系统。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cuemem.agent.context import ContextBudget
    from cuemem.agent.runtime import Agent, ChatSession, Persona, Reply
    from cuemem.agent.tools import Tool, ToolContext, ToolResult

__all__ = [
    "Agent",
    "ChatSession",
    "ContextBudget",
    "Persona",
    "Reply",
    "Tool",
    "ToolContext",
    "ToolResult",
]
_SOURCES = {
    "Agent": "cuemem.agent.runtime",
    "ChatSession": "cuemem.agent.runtime",
    "ContextBudget": "cuemem.agent.context",
    "Persona": "cuemem.agent.runtime",
    "Reply": "cuemem.agent.runtime",
    "Tool": "cuemem.agent.tools",
    "ToolContext": "cuemem.agent.tools",
    "ToolResult": "cuemem.agent.tools",
}


def __getattr__(name: str) -> Any:
    if name in _SOURCES:
        return getattr(import_module(_SOURCES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
