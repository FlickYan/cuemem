"""读取路径：朴素检索（B2）、事实增强检索（B2F）、L1 快速召回、L2 深度检索、规则版控制器、
记忆工具（B3）与 Reader。

按需导入（与 cuemem/__init__.py 相同）：只用 read.query、read.controller 这类模块时不连带加载
向量库与模型客户端。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cuemem.read.facts import FactAugmentedRetriever, MemoryView, format_facts
    from cuemem.read.naive import NaiveRetriever
    from cuemem.read.reader import Reader, ReaderOutput, format_history, format_memories

__all__ = [
    "FactAugmentedRetriever",
    "MemoryView",
    "NaiveRetriever",
    "Reader",
    "ReaderOutput",
    "format_facts",
    "format_history",
    "format_memories",
]
_SOURCES = {
    "FactAugmentedRetriever": "cuemem.read.facts",
    "MemoryView": "cuemem.read.facts",
    "format_facts": "cuemem.read.facts",
    "NaiveRetriever": "cuemem.read.naive",
    "Reader": "cuemem.read.reader",
    "ReaderOutput": "cuemem.read.reader",
    "format_history": "cuemem.read.reader",
    "format_memories": "cuemem.read.reader",
}


def __getattr__(name: str) -> Any:
    if name in _SOURCES:
        return getattr(import_module(_SOURCES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
