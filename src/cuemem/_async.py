"""AsyncMemory 与 AsyncMemoryClient 共用的异步方法（第三轮体检）：每个方法在线程池里调用同步对象
（self.sync，Memory 或 MemoryClient）上的同名方法，参数与类型注解与 Memory 逐字相同
（tests/test_memory_api.py 核对）。

以前两个类各有一份几乎一样的代码，方法经 __getattr__ 转发：类型检查器看到的都是 Any，
await m.serach(...)、search(user_id=42)、mode="l9" 一个都查不出来。__getattr__ 现在只在运行时
存在（if not TYPE_CHECKING），把别的属性（.system、.settings、.base_url……）交给同步对象，写错的
方法名类型检查器照样报错。

只用标准库（MemoryClient 不加载记忆系统，见 cuemem/client.py）。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Literal, Self

if TYPE_CHECKING:  # 只用于类型注解
    from cuemem.client import MemoryClient
    from cuemem.read.controller import Router
    from cuemem.sdk import AnswerMode, Memory, MessagesIn, SearchMode


class AsyncMethods:
    """异步方法的公共部分；子类在 __init__ 里设置 self.sync。"""

    sync: Memory | MemoryClient

    # ---------- 写入 ----------

    async def add(
        self,
        messages: MessagesIn,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        expiration_date: date | datetime | str | None = None,
        timestamp: datetime | str | float | None = None,
        infer: bool = True,
        wait: bool = False,
    ) -> dict[str, Any]:
        """见 Memory.add。"""
        return await asyncio.to_thread(
            self.sync.add,
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            expiration_date=expiration_date,
            timestamp=timestamp,
            infer=infer,
            wait=wait,
        )

    async def open_run(
        self, run_id: str | None = None, *, user_id: str | None = None, agent_id: str | None = None
    ) -> str:
        """见 Memory.open_run。"""
        return await asyncio.to_thread(
            self.sync.open_run, run_id, user_id=user_id, agent_id=agent_id
        )

    async def end_run(
        self,
        run_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """见 Memory.end_run。"""
        return await asyncio.to_thread(
            self.sync.end_run, run_id, user_id=user_id, agent_id=agent_id, wait=wait
        )

    @asynccontextmanager
    async def run(
        self,
        run_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> AsyncIterator[str]:
        """async with m.run(user_id="alice") as run_id: ...（见 Memory.run）。"""
        rid = await self.open_run(run_id, user_id=user_id, agent_id=agent_id)
        try:
            yield rid
        finally:
            await self.end_run(rid, user_id=user_id, agent_id=agent_id, wait=wait)

    async def update(
        self,
        memory_id: str,
        text: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        data: str | None = None,
    ) -> dict[str, Any]:
        """见 Memory.update。"""
        return await asyncio.to_thread(
            self.sync.update, memory_id, text, metadata=metadata, data=data
        )

    async def consolidate(
        self, *, user_id: str | None = None, agent_id: str | None = None, rebuild: bool = False
    ) -> dict[str, Any]:
        """见 Memory.consolidate。"""
        return await asyncio.to_thread(
            self.sync.consolidate, user_id=user_id, agent_id=agent_id, rebuild=rebuild
        )

    async def wait(self) -> None:
        """见 Memory.wait。"""
        await asyncio.to_thread(self.sync.wait)

    # ---------- 读取 ----------

    async def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
        mode: SearchMode = "auto",
        at: datetime | None = None,
        router: Router = "learned",
        lam: float | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        min_level: Literal["l0", "l1"] = "l1",
        show_expired: bool = False,
    ) -> dict[str, Any]:
        """见 Memory.search。"""
        return await asyncio.to_thread(
            self.sync.search,
            query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
            mode=mode,
            at=at,
            router=router,
            lam=lam,
            filters=filters,
            top_k=top_k,
            min_level=min_level,
            show_expired=show_expired,
        )

    async def answer(
        self,
        question: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: AnswerMode = "auto",
        at: datetime | None = None,
        router: Router = "learned",
        lam: float | None = None,
    ) -> dict[str, Any]:
        """见 Memory.answer。"""
        return await asyncio.to_thread(
            self.sync.answer,
            question,
            user_id=user_id,
            agent_id=agent_id,
            mode=mode,
            at=at,
            router=router,
            lam=lam,
        )

    async def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_superseded: bool = False,
        valid_at: str | None = None,
        known_at: datetime | None = None,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        show_expired: bool = False,
    ) -> dict[str, Any]:
        """见 Memory.get_all。"""
        return await asyncio.to_thread(
            self.sync.get_all,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            include_superseded=include_superseded,
            valid_at=valid_at,
            known_at=known_at,
            limit=limit,
            filters=filters,
            top_k=top_k,
            show_expired=show_expired,
        )

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """见 Memory.get。"""
        return await asyncio.to_thread(self.sync.get, memory_id)

    async def media(self, media_id: str) -> bytes | None:
        """见 Memory.media。"""
        return await asyncio.to_thread(self.sync.media, media_id)

    async def history(self, memory_id: str) -> list[dict[str, Any]]:
        """见 Memory.history。"""
        return await asyncio.to_thread(self.sync.history, memory_id)

    async def profile(self, *, user_id: str | None = None, agent_id: str | None = None) -> str:
        """见 Memory.profile。"""
        return await asyncio.to_thread(self.sync.profile, user_id=user_id, agent_id=agent_id)

    # ---------- 遗忘 ----------

    async def delete(self, memory_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        """见 Memory.delete。"""
        return await asyncio.to_thread(self.sync.delete, memory_id, dry_run=dry_run)

    async def delete_all(
        self, *, user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        """见 Memory.delete_all。"""
        return await asyncio.to_thread(
            self.sync.delete_all, user_id=user_id, agent_id=agent_id, run_id=run_id
        )

    async def reset(self) -> dict[str, Any]:
        """见 Memory.reset。"""
        return await asyncio.to_thread(self.sync.reset)

    # ---------- 导出与导入 ----------

    async def export(
        self, *, user_id: str | None = None, agent_id: str | None = None, derived: bool = True
    ) -> dict[str, Any]:
        """见 Memory.export。"""
        return await asyncio.to_thread(
            self.sync.export, user_id=user_id, agent_id=agent_id, derived=derived
        )

    async def load(
        self,
        data: Mapping[str, Any],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """见 Memory.load。"""
        return await asyncio.to_thread(
            self.sync.load, data, user_id=user_id, agent_id=agent_id, wait=wait
        )

    # ---------- 其他 ----------

    async def spaces(self) -> list[dict[str, Any]]:
        """见 Memory.spaces。"""
        return await asyncio.to_thread(self.sync.spaces)

    async def close(self) -> None:
        """见 Memory.close。"""
        await asyncio.to_thread(self.sync.close)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    if not TYPE_CHECKING:  # 只在运行时：类型检查器照样报出写错的方法名

        def __getattr__(self, name: str) -> Any:
            """别的属性（.system、.settings、.base_url……）取同步对象上的。"""
            if name == "sync":  # __init__ 没做完（例如构造 Memory 时出错）：不无限递归
                raise AttributeError(name)
            return getattr(self.sync, name)
