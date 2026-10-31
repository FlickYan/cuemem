"""写入路径：在线写入（N1）与异步巩固（N3）。

按需导入（与 cuemem/__init__.py 相同）：只用 write.extract、write.caption 这类模块时不连带加载
向量库。
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cuemem.write.online import IngestReport, OnlineWriter, build_rounds, build_turns

__all__ = ["IngestReport", "OnlineWriter", "build_rounds", "build_turns"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(import_module("cuemem.write.online"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
