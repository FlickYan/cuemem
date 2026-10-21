"""LLM 访问层：OpenAI 兼容客户端、响应缓存、用量账本。

按需导入（与 cuemem/__init__.py 相同）：只用账本、缓存或分时价格时不加载 openai（约 0.5 秒）。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cuemem.llm.cache import ResponseCache
    from cuemem.llm.client import LLMClient, Structured
    from cuemem.llm.ledger import Ledger, UsageRecord, chat_cost
    from cuemem.llm.types import ChatResult, ToolCall, Usage

__all__ = [
    "ChatResult",
    "LLMClient",
    "Ledger",
    "ResponseCache",
    "Structured",
    "ToolCall",
    "Usage",
    "UsageRecord",
    "chat_cost",
]
_SOURCES = {
    "ChatResult": "cuemem.llm.types",
    "LLMClient": "cuemem.llm.client",
    "Ledger": "cuemem.llm.ledger",
    "ResponseCache": "cuemem.llm.cache",
    "Structured": "cuemem.llm.client",
    "ToolCall": "cuemem.llm.types",
    "Usage": "cuemem.llm.types",
    "UsageRecord": "cuemem.llm.ledger",
    "chat_cost": "cuemem.llm.ledger",
}


def __getattr__(name: str) -> Any:
    if name in _SOURCES:
        return getattr(import_module(_SOURCES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
