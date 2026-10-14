"""CueMem：线索驱动、可自我调控的多模态 Agent 长期记忆系统。

from cuemem import Memory      # Python SDK（N12）：add / search / answer / history / delete
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"
__all__ = ["AsyncMemory", "Memory", "__version__"]

if TYPE_CHECKING:
    from cuemem.sdk import AsyncMemory, Memory


def __getattr__(name: str) -> Any:
    # 按需导入：import cuemem 只取版本号时不加载向量库、模型客户端等重的依赖
    if name in ("Memory", "AsyncMemory"):
        from cuemem import sdk

        return getattr(sdk, name)
    raise AttributeError(f"module 'cuemem' has no attribute {name!r}")
