"""存储层：SQLite 结构化数据 + Qdrant 向量索引。

按需导入（与 cuemem/__init__.py 相同）：只用 store.sqlite、store.media 时不加载 Qdrant 客户端
（约 1 秒）。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cuemem.store.sqlite import MemoryDB
    from cuemem.store.vector import VectorHit, VectorStore

__all__ = ["MemoryDB", "VectorHit", "VectorStore"]
_SOURCES = {
    "MemoryDB": "cuemem.store.sqlite",
    "VectorHit": "cuemem.store.vector",
    "VectorStore": "cuemem.store.vector",
}


def __getattr__(name: str) -> Any:
    if name in _SOURCES:
        return getattr(import_module(_SOURCES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
