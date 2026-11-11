"""远程的 Python SDK：MemoryClient 的方法与参数和 cuemem.sdk.Memory 完全相同，调用的是
cuemem serve 的 /api/v1 记忆接口（api/memories.py）。本地与远程之间只换一行：

    from cuemem import Memory, MemoryClient

    m = Memory()                                                   # 本地：直接打开数据目录
    m = MemoryClient("http://127.0.0.1:7860", api_key="<CUEMEM_API_TOKEN>")   # 远程
    m.add("I moved to Berlin last month", user_id="alice", metadata={"source": "crm"})
    m.search("Where do I live?", user_id="alice")

几个进程（或别的机器上的应用）可以共用一个 cuemem serve：本地模式的向量库同一时间只能被一个进程
打开，SDK 不能与正在运行的服务同时打开同一个数据目录。

- 返回值与 Memory 相同（服务端就是用 Memory 算出来的）；出错时抛出与 Memory 相同的异常
  （KeyError、ValueError、TypeError、cuemem.errors 里的类型），其余的 HTTP 错误（令牌不对、连不上
  的服务）是 cuemem.errors.MemoryAPIError；
- add 的图片：本地路径（或 file://）在客户端读成 data URL 再发，服务器不读它自己磁盘上的文件；
- 时间参数（timestamp、at、known_at、expiration_date）按 ISO 格式发送：不带时区的时间按服务器的
  本机时间理解，客户端与服务器不在一个时区时请带上时区；
- 这个模块只依赖 httpx2，不加载向量库、模型客户端等（import 很快）。
"""

from __future__ import annotations

import base64
import contextlib
import json
import math
import mimetypes
import os
import random
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import quote

import httpx2

from cuemem import errors
from cuemem._async import AsyncMethods
from cuemem.errors import CueMemError, MemoryAPIError

if TYPE_CHECKING:  # 只用于类型注解：签名与 Memory 逐字相同（tests/test_memory_api.py 核对）
    from cuemem.read.controller import Router
    from cuemem.sdk import AnswerMode, MessagesIn, SearchMode

DEFAULT_URL = "http://127.0.0.1:7860"  # cuemem serve 的默认地址
TOKEN_ENV = "CUEMEM_API_TOKEN"
# wait() 让服务端比客户端的读超时早这么多秒放弃：先收到服务端的 504（TimeoutError），而不是
# 客户端自己的读超时（见 MemoryClient.wait）
WAIT_MARGIN_S = 5.0
# max_retries > 0 时的自动重试（MemoryClient._send）：429、503 的 Retry-After 不超过这么多秒才等
# 它再重发（会话数上限的 429 是 60 秒，不重试）；没有拿到响应时的退避从 RETRY_BASE_S 起每次翻倍、
# 最多 RETRY_CAP_S 秒，再乘 0.5 到 1 之间的随机数
RETRY_AFTER_MAX_S = 30.0
RETRY_BASE_S = 0.5
RETRY_CAP_S = 8.0
_BUILTIN_ERRORS: dict[str, type[Exception]] = {
    "KeyError": KeyError,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "TimeoutError": TimeoutError,  # POST /wait 的 timeout 到了（504），同本地的 Memory.wait
}


class MemoryClient:
    """cuemem serve 的记忆接口（/api/v1）的客户端，方法与 cuemem.sdk.Memory 相同。线程安全；
    用完调用 close()，或用 with 语句。

    base_url：服务的地址（默认 cuemem serve 的 http://127.0.0.1:7860），要以 http:// 或 https://
    开头（ValueError），末尾的 /api/v1 去掉；host 是它的别名（Mem0 的 MemoryClient 这样叫），
    两个不能都给。api_key：访问令牌，不填时读环境变量 CUEMEM_API_TOKEN（与服务端同名）。限定
    作用域的令牌（服务端的 CUEMEM_API_KEYS_FILE）只能访问它的用户：之外的 user_id、agent_id 报
    MemoryAPIError（status 403），别人的记忆 ID 与不存在的一样；timeout：单次请求的秒数
    （wait=True 的 add、update 要等抽取做完）；http_client：自带的 httpx2.Client（测试里传
    FastAPI 的 TestClient；超时照它自己的设置，timeout 不再起作用）。

    max_retries（默认 0：每个调用只发一次，请求与以前逐字节相同）：大于 0 时一个调用最多再重发
    这么多次（第九轮体检）。重发的情形只有三种：连不上（unreachable，请求没有送到）；429 或 503
    带着不超过 30 秒的数字 Retry-After（限流、模型供应商限流、向量库暂时不可用），等这么久再
    发；只读的调用（get_all、get、media、history、profile、status、export、spaces、search、
    context）与 add 在超时、连接中途断开之后。add 没给 idempotency_key 时为这次调用生成一个
    （uuid4），每次重发都带着它，服务端只写一次：重发得到的是 status 为 duplicate、results 为空
    的结果（wait=True 时等那次的抽取做完），要事件的话再读 get_all；服务端要认 Idempotency-Key
    头（第六轮体检之前的不认，会写两次）。update、delete、delete_many、delete_all、reset、
    consolidate、open_run、end_run、load、answer、wait 在超时、断开之后不重发（可能已经做了）。
    别的状态码都不重发：502、504（wait 的 TimeoutError）、409、4xx，没有 Retry-After 或超过
    30 秒的 429、503（会话数上限、向量库满了）。没有拿到响应时的退避是 0.5 秒起每次翻倍、最多
    8 秒，带随机抖动。最后抛出的是最后一次的异常，它的 attempts 是一共发了几次；一个调用最多
    占 max_retries + 1 个读超时再加上等待的时间（wait=True 的 add 尤其要算上）。
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        host: str | None = None,
        timeout: float = 300.0,
        http_client: httpx2.Client | None = None,
        max_retries: int = 0,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(f"max_retries must be a non-negative integer (got {max_retries!r}).")
        if base_url is not None and host is not None:
            raise ValueError("Give base_url or host (the same argument), not both.")
        url = base_url if base_url is not None else host if host is not None else DEFAULT_URL
        # 以前 Mem0 的写法 MemoryClient("m0-…")（第一个参数是 API Key）在第一次调用时才报错。
        # 报错里不带这个值：它可能就是令牌
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ValueError(
                "base_url must start with http:// or https:// (cuemem serve listens on "
                f"{DEFAULT_URL} by default); the token goes in api_key=, not the first argument."
            )
        # 文档里 curl 的写法以 /api/v1 结尾：照样认，以前每个调用都是 404 "Not Found"
        self.base_url = url.rstrip("/").removesuffix("/api/v1")
        token = api_key or os.environ.get(TOKEN_ENV) or None
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._owns_http = http_client is None
        self._http = http_client or httpx2.Client(timeout=timeout)
        self._runs: set[tuple[str, str | None, str | None]] = set()  # 这个客户端开始的 run
        self._runs_lock = threading.Lock()
        self._max_retries = max_retries
        self._sleep = time.sleep  # 测试里换掉这两个
        self._random = random.random

    # ---------- 写入 ----------

    def add(
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
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """写入对话（见 Memory.add）。内容块里的本地图片先读成 data URL。idempotency_key 作为
        Idempotency-Key 头发送（不给时不发，请求与以前相同）：超时或连接断开（MemoryAPIError 的
        code 为 timeout、disconnected）之后用同一个键原样重发，服务端只写一次；wait=True 的
        重发等那次的抽取做完再返回 duplicate（第七轮体检）。max_retries 大于 0 时不给键的 add
        在这里生成一个（uuid4），这次调用的每次重发都带着它，超时、断开之后自动重发。"""
        body = {
            "messages": _inline_images(messages),
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "metadata": _metadata(metadata),
            "expiration_date": _iso(expiration_date),
            "timestamp": _iso(timestamp),
            "infer": infer,
            "wait": wait,
        }
        if idempotency_key is None and self._max_retries > 0:
            idempotency_key = str(uuid.uuid4())
        if idempotency_key is None:
            return self._call("POST", "/memories", body=body)
        _check_key(idempotency_key)
        headers = {"Idempotency-Key": idempotency_key}
        return self._call("POST", "/memories", body=body, headers=headers, retry_sent=True)

    def open_run(
        self, run_id: str | None = None, *, user_id: str | None = None, agent_id: str | None = None
    ) -> str:
        """开始一段对话（见 Memory.open_run）。客户端异常退出、没有 end_run 时，服务端在这段对话
        闲置 idle_minutes（cuemem serve --idle-minutes，默认 30）分钟之后把它交给抽取；之后再往
        这个 run_id 里 add 时重新打开，end_run 时整段再抽取一次。"""
        body = {"run_id": run_id, "user_id": user_id, "agent_id": agent_id}
        rid = str(self._call("POST", "/runs", body=body)["run_id"])
        with self._runs_lock:
            self._runs.add((rid, user_id, agent_id))
        return rid

    def end_run(
        self,
        run_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> dict[str, Any]:
        """结束一段对话，整段交给抽取（见 Memory.end_run）。"""
        with self._runs_lock:
            self._runs.discard((run_id, user_id, agent_id))
        body = {"run_id": run_id, "user_id": user_id, "agent_id": agent_id, "wait": wait}
        return self._call("POST", "/runs/end", body=body)

    @contextmanager
    def run(
        self,
        run_id: str | None = None,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
    ) -> Iterator[str]:
        """with m.run(user_id="alice") as run_id: ...（见 Memory.run）。"""
        rid = self.open_run(run_id, user_id=user_id, agent_id=agent_id)
        try:
            yield rid
        finally:
            self.end_run(rid, user_id=user_id, agent_id=agent_id, wait=wait)

    def update(
        self,
        memory_id: str,
        text: str | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        data: str | None = None,
    ) -> dict[str, Any]:
        """改一条记忆：记进原始日志的一句更正（见 Memory.update）。text 与 data 在这里合成一个，
        照旧以 "data" 发送：以前的服务端只认这个键。都没给时交给服务端报 ValueError。"""
        if text is not None and data is not None and text != data:
            raise ValueError("Give text or data (the same argument), not two different values.")
        chosen = text if text is not None else data
        body = {"data": chosen, "metadata": _metadata(metadata)}
        return self._call("PUT", f"/memories/{_id(memory_id)}", body=body)

    def consolidate(
        self, *, user_id: str | None = None, agent_id: str | None = None, rebuild: bool = False
    ) -> dict[str, Any]:
        body = {"user_id": user_id, "agent_id": agent_id, "rebuild": rebuild}
        return self._call("POST", "/consolidate", body=body)

    def wait(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """等服务端的后台线程把已登记的抽取做完（见 Memory.wait）。限定作用域的令牌不给 user_id、
        agent_id 时只等它能访问的用户。服务端最多等 300 秒（WAIT_TIMEOUT_S），没做完时抛
        TimeoutError，同本地的 timeout。

        发给服务端的等待时间比这个客户端的读超时短 WAIT_MARGIN_S 秒（不给 timeout 时也是）：
        以前两边都是 300 秒，服务端等满了才回 504，客户端的读超时差不多同时到，抛的多半是
        MemoryAPIError（timeout，"may have been processed"），不是 TimeoutError。"""
        read = getattr(getattr(self._http, "timeout", None), "read", None)
        if isinstance(read, int | float):  # 读超时为 None（不限）时照旧
            cap = max(read - WAIT_MARGIN_S, read / 2)
            timeout = cap if timeout is None else min(timeout, cap)
        body = {"user_id": user_id, "agent_id": agent_id, "timeout": timeout}
        self._call("POST", "/wait", body=body)

    def status(self, *, user_id: str | None = None, agent_id: str | None = None) -> dict[str, Any]:
        """抽取的状态：待处理与失败的会话、失败的原因（见 Memory.status）。"""
        params = {"user_id": user_id, "agent_id": agent_id}
        return self._call("GET", "/status", params=params, retry_sent=True)

    # ---------- 读取 ----------

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
        mode: SearchMode = "auto",
        at: datetime | date | str | None = None,
        router: Router = "learned",
        lam: float | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        min_level: Literal["l0", "l1"] = "l1",
        show_expired: bool = False,
    ) -> dict[str, Any]:
        """检索（见 Memory.search）。"""
        _count("limit", limit)
        _count("top_k", top_k)
        body = {
            "query": query,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "limit": limit,
            "mode": mode,
            "at": _iso(at),
            "router": router,
            "lam": lam,
            "filters": dict(filters) if filters is not None else None,
            "top_k": top_k,
            "min_level": min_level,
            "show_expired": show_expired,
        }
        return self._call("POST", "/memories/search", body=body, retry_sent=True)

    def context(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        at: datetime | date | str | None = None,
        mode: SearchMode = "auto",
        router: Router = "learned",
        lam: float | None = None,
        min_level: Literal["l0", "l1"] = "l1",
        wrap: bool = True,
    ) -> dict[str, Any]:
        """放进提示词的记忆（见 Memory.context）。"""
        body = {
            "query": query,
            "user_id": user_id,
            "agent_id": agent_id,
            "at": _iso(at),
            "mode": mode,
            "router": router,
            "lam": lam,
            "min_level": min_level,
            "wrap": wrap,
        }
        return self._call("POST", "/context", body=body, retry_sent=True)

    def answer(
        self,
        question: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        mode: AnswerMode = "auto",
        at: datetime | date | str | None = None,
        router: Router = "learned",
        lam: float | None = None,
    ) -> dict[str, Any]:
        body = {
            "question": question,
            "user_id": user_id,
            "agent_id": agent_id,
            "mode": mode,
            "at": _iso(at),
            "router": router,
            "lam": lam,
        }
        return self._call("POST", "/answer", body=body)

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_superseded: bool = False,
        valid_at: str | date | None = None,
        known_at: datetime | date | str | None = None,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        top_k: int | None = None,
        show_expired: bool = False,
    ) -> dict[str, Any]:
        """全部事实（见 Memory.get_all）。filters 以 JSON 字符串放在查询参数里。"""
        _count("limit", limit)
        _count("top_k", top_k)
        params = {
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
            "include_superseded": include_superseded,
            "valid_at": _iso(valid_at),
            "known_at": _iso(known_at),
            "limit": limit,
            "filters": _json_param(filters),
            "top_k": top_k,
            "show_expired": show_expired,
        }
        return self._call("GET", "/memories", params=params, retry_sent=True)

    def get(self, memory_id: str) -> dict[str, Any] | None:
        try:
            return self._call("GET", f"/memories/{_id(memory_id)}", retry_sent=True)
        except KeyError:
            return None

    def media(self, media_id: str) -> bytes | None:
        """一张图片的原文件（见 Memory.media）：图片项里的 path 是服务器上的路径，读不到。"""
        try:
            return self._send("GET", f"/media/{_id(media_id)}", retry_sent=True).content
        except KeyError:
            return None

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        return self._call("GET", f"/memories/{_id(memory_id)}/history", retry_sent=True)

    def profile(self, *, user_id: str | None = None, agent_id: str | None = None) -> str:
        params = {"user_id": user_id, "agent_id": agent_id}
        return str(self._call("GET", "/profile", params=params, retry_sent=True)["profile"])

    # ---------- 遗忘 ----------

    def delete(self, memory_id: str, *, dry_run: bool = False) -> dict[str, Any]:
        params = {"dry_run": dry_run}
        return self._call("DELETE", f"/memories/{_id(memory_id)}", params=params)

    def delete_many(self, memory_ids: Sequence[str], *, dry_run: bool = False) -> dict[str, Any]:
        """一次遗忘好几条记忆（见 Memory.delete_many）：POST /api/v1/memories/batch-delete，算一次
        写入。单个字符串在发送之前 TypeError，与本地相同；去重与上限由服务端检查。"""
        if isinstance(memory_ids, str | bytes) or not isinstance(memory_ids, Sequence):
            raise TypeError("memory_ids must be a list of memory ids, not a single id.")
        body = {"memory_ids": list(memory_ids), "dry_run": dry_run}
        return self._call("POST", "/memories/batch-delete", body=body)

    def batch_delete(self, memories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Mem0 的写法：memories=[{"memory_id": ...}, ...]，与 delete_many 相同（见
        Memory.batch_delete）。"""
        if isinstance(memories, str | Mapping):
            raise TypeError('memories must be a list of {"memory_id": ...} objects.')
        ids: list[str] = []
        for item in memories:
            if not isinstance(item, Mapping) or "memory_id" not in item:
                raise ValueError('Each item of memories must be an object with "memory_id".')
            ids.append(item["memory_id"])
        return self.delete_many(ids)

    def delete_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """见 Memory.delete_all。dry_run 只在为 True 时发送：不预演的请求与以前逐字相同。"""
        params: dict[str, Any] = {"user_id": user_id, "agent_id": agent_id, "run_id": run_id}
        if dry_run:
            params["dry_run"] = True
        return self._call("DELETE", "/memories", params=params)

    def reset(self) -> dict[str, Any]:
        """删除服务端数据目录里的全部记忆（见 Memory.reset）。不可恢复。限定作用域的令牌只删它
        作用域里的用户（每个都能用 delete_all 删），管理令牌删全部。"""
        return self._call("POST", "/reset")

    # ---------- 导出与导入 ----------

    def export(
        self, *, user_id: str | None = None, agent_id: str | None = None, derived: bool = True
    ) -> dict[str, Any]:
        params = {"user_id": user_id, "agent_id": agent_id, "derived": derived}
        return self._call("GET", "/export", params=params, retry_sent=True)

    def load(
        self,
        data: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        wait: bool = False,
        format: Literal["cuemem", "mem0"] = "cuemem",
    ) -> dict[str, Any]:
        """见 Memory.load；format="mem0" 时由服务端换算（较旧的服务端不认这个键）。"""
        if format not in ("cuemem", "mem0"):  # 与本地的 Memory.load 相同，发送之前报错
            raise ValueError(f"format must be 'cuemem' or 'mem0', not {format!r}")
        payload: dict[str, Any] | list[Any] = (
            dict(data)
            if isinstance(data, Mapping)
            else [dict(item) if isinstance(item, Mapping) else item for item in data]
        )
        body = {
            "data": payload,
            "user_id": user_id,
            "agent_id": agent_id,
            "wait": wait,
            "format": format,
        }
        return self._call("POST", "/import", body=body)

    # ---------- 其他 ----------

    def spaces(self) -> list[dict[str, Any]]:
        return self._call("GET", "/spaces", retry_sent=True)

    def close(self) -> None:
        """关闭。这个客户端开始、还没结束的 run 先结束（交给服务端抽取）；自己建的连接关掉。"""
        with self._runs_lock:
            runs = sorted(self._runs, key=str)
            self._runs.clear()
        for rid, user_id, agent_id in runs:
            # 服务已经停了：这些 run 留给服务端重启时接手
            with contextlib.suppress(CueMemError, httpx2.HTTPError):
                self.end_run(rid, user_id=user_id, agent_id=agent_id)
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> MemoryClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _call(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        retry_sent: bool = False,
    ) -> Any:
        response = self._send(
            method, path, body=body, params=params, headers=headers, retry_sent=retry_sent
        )
        return response.json()

    def _send(
        self,
        method: str,
        path: str,
        *,
        body: Any = None,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        retry_sent: bool = False,
    ) -> httpx2.Response:
        """headers：这次请求另加的头（add 的 Idempotency-Key）；不给时与以前完全相同。

        max_retries 大于 0 时在这里重发（规则见类的说明）：看的是原始的响应，不是换算之后的
        异常——会话数上限的 429 换成 SessionLimitError 之后就没有 Retry-After 了。retry_sent：
        请求可能已经送到（超时、连接中途断开）之后也重发，只给只读的调用与带着幂等键的 add。"""
        query = {k: v for k, v in (params or {}).items() if v is not None}  # None 不发
        for attempt in range(self._max_retries + 1):
            last = attempt == self._max_retries
            try:
                response = self._http.request(
                    method,
                    f"{self.base_url}/api/v1{path}",
                    json=body,
                    params=query,
                    headers={**self._headers, **headers} if headers else self._headers,
                )
            except httpx2.HTTPError as exc:
                err = self._transport_error(exc)
                resend = err.sent is False or (
                    retry_sent and err.code in ("timeout", "disconnected")
                )
                if last or not resend:
                    err.attempts = attempt + 1
                    raise err from exc
                self._sleep(
                    min(RETRY_BASE_S * 2**attempt, RETRY_CAP_S) * (0.5 + self._random() / 2)
                )
                continue
            if response.status_code < 400:
                return response
            pause = _retry_after(response) if response.status_code in (429, 503) else None
            if last or pause is None:
                error = _error(response, self.base_url)
                # 也可能是 KeyError、cuemem.errors 里的类型：都记上一共发了几次
                error.attempts = attempt + 1  # type: ignore[attr-defined]
                raise error
            self._sleep(pause)
        raise AssertionError("unreachable")  # 最后一次不是返回就是抛出

    def _transport_error(self, exc: httpx2.HTTPError) -> MemoryAPIError:
        """没有拿到响应时的出错，按请求有没有送到服务端分开（code 与 sent）：连不上（unreachable，
        sent=False）时写入一定没做，可以重发；超时（timeout）与连接中途断开（disconnected）时
        sent=True，写入可能已经提交（add(wait=True) 要等抽取做完，超时多半在发言写下之后），
        不要盲目重发。以前一律是 "Cannot reach"，只能看 __cause__ 分辨。"""
        if isinstance(exc, httpx2.ConnectError | httpx2.ConnectTimeout | httpx2.PoolTimeout):
            hint = (
                "; is `cuemem serve` running there?" if isinstance(exc, httpx2.ConnectError) else ""
            )
            return MemoryAPIError(
                f"Cannot reach {self.base_url}: {exc}{hint}", code="unreachable", sent=False
            )
        if isinstance(exc, httpx2.ReadTimeout | httpx2.WriteTimeout):
            limit = getattr(self._http, "timeout", None)
            seconds = getattr(
                limit, "read" if isinstance(exc, httpx2.ReadTimeout) else "write", None
            )
            within = f" within {seconds:g}s" if isinstance(seconds, int | float) else ""
            return MemoryAPIError(
                f"No response from {self.base_url}{within}; the request may have been processed "
                "(repeat it with the same idempotency_key, or check export)",
                code="timeout",
                sent=True,
            )
        if isinstance(exc, httpx2.RemoteProtocolError | httpx2.ReadError | httpx2.WriteError):
            return MemoryAPIError(
                f"Connection to {self.base_url} broke off: {exc}; the request may have been "
                "processed (repeat it with the same idempotency_key, or check export)",
                code="disconnected",
                sent=True,
            )
        return MemoryAPIError(
            f"Request to {self.base_url} failed: {exc}", code="transport", sent=True
        )


class AsyncMemoryClient(AsyncMethods):
    """MemoryClient 的异步版本（与 AsyncMemory 对应）：同样的方法与参数，在线程池里执行
    （cuemem/_async.py）。self.sync 是包着的 MemoryClient；别的属性（.base_url）取它的。
    取消一次调用停不下已经发出的请求，见 AsyncMethods。max_retries 的重试在线程池的那个线程里
    等待，不占事件循环。"""

    sync: MemoryClient

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        host: str | None = None,
        timeout: float = 300.0,
        http_client: httpx2.Client | None = None,
        max_retries: int = 0,
    ) -> None:
        self.sync = MemoryClient(
            base_url,
            api_key=api_key,
            host=host,
            timeout=timeout,
            http_client=http_client,
            max_retries=max_retries,
        )


def _retry_after(response: httpx2.Response) -> float | None:
    """429、503 的 Retry-After：不超过 RETRY_AFTER_MAX_S 的非负秒数才值得等（返回这个数）；没有
    这个头、是 HTTP 日期、或者更长时返回 None（不重发，照旧抛出）。"""
    try:
        seconds = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and 0 <= seconds <= RETRY_AFTER_MAX_S else None


def _check_key(key: str) -> None:
    """idempotency_key 写错时在发送之前 ValueError，与本地的 Memory.add 相同（sdk._key_hash；
    这里不加载 SDK）：1 到 255 个可打印的 ASCII 字符，要能放进 HTTP 头。"""
    if not isinstance(key, str) or not 0 < len(key) <= 255 or not all(" " <= c <= "~" for c in key):
        raise ValueError("idempotency_key must be 1 to 255 printable ASCII characters.")


def _error(response: httpx2.Response, base_url: str) -> Exception:
    """/api/v1 的出错 → 与本地 Memory 相同的异常；别的（401、找不到接口、代理返回的 HTML、模型
    供应商限流）是 MemoryAPIError，带着状态码、错误码、服务端的异常类型与 Retry-After（以前
    没有 retry_after，按它退避的代码换成 MemoryClient 就不退避了）。404 却没有异常类型的是找不到
    接口：多半 base_url 不是 cuemem serve 的根地址（记忆代理的端口、别的服务），报错里写明。"""
    try:
        body = response.json()
    except ValueError:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message, kind, code = str(error.get("message")), error.get("type"), error.get("code")
    else:
        detail = body.get("detail") if isinstance(body, dict) else None
        message, kind, code = str(detail or response.text[:300]), None, None
    if isinstance(kind, str):
        builtin = _BUILTIN_ERRORS.get(kind)
        if builtin is not None:
            return builtin(message)
        if kind == "EmbeddingAPIError":  # 与本地一样带着上游的状态码与 Retry-After（据此退避）
            upstream = error.get("upstream_status") if isinstance(error, dict) else None
            return errors.EmbeddingAPIError(
                message,
                status=upstream if isinstance(upstream, int) else None,
                retry_after=response.headers.get("retry-after"),
            )
        known = getattr(errors, kind, None)  # 只认 cuemem.errors 里的异常类型
        if (
            isinstance(known, type)
            and issubclass(known, CueMemError)
            and known is not MemoryAPIError
        ):
            return known(message)
    if response.status_code == 404 and not isinstance(kind, str):
        message += (
            f": base_url must be the root of `cuemem serve`, e.g. {DEFAULT_URL} (got {base_url}); "
            "an older server may also lack this endpoint"
        )
    return MemoryAPIError(
        message,
        status=response.status_code,
        code=code if isinstance(code, str) else None,
        retry_after=response.headers.get("retry-after"),
        type=kind if isinstance(kind, str) else None,
    )


def _iso(value: date | datetime | str | float | None) -> str | float | None:
    """时间参数 → JSON：datetime 与 date 写成 ISO 格式（带不带时区原样保留），其余原样。"""
    if isinstance(value, date):  # datetime 也是 date
        return value.isoformat()
    return value


def _metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """与 Memory 相同的检查与报错（sdk._turn_meta）：不先查的话，JSON 会把非字符串的键悄悄变成
    字符串，写不成 JSON 的值在发送时报别的错。"""
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a dict.")
    if any(not isinstance(key, str) for key in metadata):
        raise ValueError("metadata keys must be strings.")
    try:
        return dict(json.loads(json.dumps(dict(metadata), allow_nan=False)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metadata must be JSON-serializable: {exc}") from exc


def _count(name: str, value: int | None) -> None:
    """与 Memory 相同的检查（sdk._count）：不先查的话，JSON 里的 true 到服务端成了 1。"""
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{name} must be a positive integer (got {value!r}).")


def _json_param(value: Mapping[str, Any] | None) -> str | None:
    return json.dumps(dict(value)) if value is not None else None


def _id(memory_id: str) -> str:
    return quote(memory_id, safe="")


def _inline_images(messages: MessagesIn) -> Any:
    """消息里的本地图片（路径或 file://）→ data URL：图片在客户端这台机器上，服务器读不到。
    data URL 与网络地址原样发送（网络图片照 SDK 的规矩由服务端报错）。"""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, Mapping):
        return _inline_message(messages)
    return [_inline_message(m) for m in messages]


def _inline_message(message: Mapping[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, str):
        return dict(message)
    return {**message, "content": [_inline_block(block) for block in content]}


def _inline_block(block: Any) -> Any:
    if not isinstance(block, Mapping) or block.get("type") != "image_url":
        return block
    value = block.get("image_url")
    url = value.get("url") if isinstance(value, Mapping) else value
    if not isinstance(url, str) or url.startswith(("data:", "http://", "https://")):
        return block
    path = Path(url.removeprefix("file://")).expanduser()
    if not path.is_file():
        raise ValueError(f"Image not found: {url}")  # 与 Memory 相同
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data_url = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"
    image = {**value, "url": data_url} if isinstance(value, Mapping) else {"url": data_url}
    return {**block, "image_url": image}
