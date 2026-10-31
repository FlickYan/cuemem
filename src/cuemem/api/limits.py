"""请求体的上限（第三轮体检）：cuemem serve 与 cuemem proxy 的应用都在读请求体之前检查。

FastAPI 先把整个请求体读完、解析完，才轮到鉴权（/api 与 /api/v1 现在先认令牌，见
api.auth.authenticated_route），界面（Gradio）的登录与事件接口也一样：没有上限时，不带令牌的一个
大请求就能占掉几百 MB 内存（9 MB 的 JSON 数组解析出来是两百多 MB），/login 能缓冲几百 MB 的表单。
按路径分三档，先匹配的算：

- 导入（/api/spaces/import、/api/v1/import）：max_import_bytes，默认 1 GiB，环境变量
  CUEMEM_MAX_IMPORT_BYTES（导出的文件带着 base64 的原图）；
- /api/、/v1/ 与界面的上传（/gradio_api/upload）：max_body_bytes，默认 64 MiB，环境变量
  CUEMEM_MAX_BODY_BYTES（OpenAI 格式的图片是 data URL）；
- 别的路径（界面的登录、事件等）：8 MiB。界面的文件经上传接口另传，事件的 JSON 很小。

超过上限时 413：/api/v1 下是它的出错格式（code 为 payload_too_large），别处是 {"detail": …}。

限定作用域的令牌（第三轮复审，keys 给了时）：上限管的是字节，JSON 解析出来的对象却是请求体的二十
多倍（[{},{},…] 约 24 倍），在接口处理完之前一直占着。以前一个令牌导入 1 GiB 要约 25 GB 内存，并发
几个 64 MiB 的请求也能撑爆共用的服务。现在：

- 导入那一档只给管理令牌（与没配令牌的本机演示）；限定作用域的令牌导入也是 max_body_bytes；
- 一个限定作用域的令牌同时在读、在处理的请求体加起来不超过 max_body_bytes，超过时 429（带
  Retry-After: 1，/api/v1 下 code 为 rate_limited）。一个令牌解析出来的对象因此最多约是
  max_body_bytes 的 25 倍（默认约 1.6 GB）。

管不到的：好几个令牌一起来（每个各算各的）；界面的事件接口（登录的会话不带 Bearer 令牌，固定
8 MiB 那一档）。共用的服务要按令牌数与这个倍数留内存，或者调小 CUEMEM_MAX_BODY_BYTES。

请求的次数（第四轮体检）：限定作用域的令牌写了 rate_per_minute 时（api/auth.py），它在 /api/、
/v1/ 与 /mcp 的 POST 每分钟最多这么多个：令牌桶，容量就是 rate_per_minute，每秒补
rate_per_minute / 60 个。超过时 429，Retry-After 是等到下一个名额的秒数（/api/v1 下 code 为
rate_limited）。不写的令牌、管理令牌不限；GET 不算。管不到界面（登录的会话不带 Bearer 令牌，用
agent 角色固定的模型与 max_tokens）；一次写入有多大（抽取约按每字节两个字符发给上游）由
CUEMEM_MAX_BODY_BYTES 管。
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from cuemem.api.auth import Keys, Principal, bearer_principal
from cuemem.api.memories import PREFIX, _error
from cuemem.errors import CueMemError

MAX_BODY_ENV = "CUEMEM_MAX_BODY_BYTES"
MAX_IMPORT_ENV = "CUEMEM_MAX_IMPORT_BYTES"
MAX_BODY_BYTES = 64 << 20
MAX_IMPORT_BYTES = 1 << 30
OTHER_BODY_BYTES = 8 << 20  # 界面的登录与事件（不可配置）
IMPORT_PATHS = ("/api/spaces/import", "/api/v1/import")
BODY_PATHS = ("/api/", "/v1/", "/gradio_api/upload")
RATE_PATHS = ("/api/", "/v1/", "/mcp")  # rate_per_minute 管的路径（只算 POST）


class BodyLimit:
    """整个应用的中间件：请求体超过这个路径的上限时 413（档位见模块说明）。带 Content-Length
    时在应用读请求体之前就回；分块传输时边读边数，超过时抛出 HTTPException(413)：FastAPI 读请求体
    时遇到 HTTPException 原样抛出，由应用（或挂载的 Gradio）的出错处理写成响应。

    keys：配置的令牌（cuemem serve 与 proxy 传入）。给了时按 Bearer 令牌认出限定作用域的令牌：导入
    不用 max_import_bytes，同时在处理的请求体加起来有上限，写了 rate_per_minute 的还有次数的上限
    （见模块说明）。令牌不对的请求不算：接口在读请求体之前就 401（auth.authenticated_route）。
    clock：次数上限用的时钟（秒，测试里换掉）。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int = MAX_BODY_BYTES,
        max_import_bytes: int = MAX_IMPORT_BYTES,
        keys: Keys | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_import_bytes = max_import_bytes
        self.keys = keys
        self.clock = clock
        # 限定作用域的令牌（名字）→ 正在读、在处理的请求体的字节数。只在事件循环里读写
        self.in_flight: dict[str, int] = {}
        # 写了 rate_per_minute 的令牌（名字）→ 令牌桶记下的时间（见 _wait）。只在事件循环里读写
        self.next_free: dict[str, float] = {}

    def limit(self, path: str, tenant: str | None = None) -> tuple[int, str | None]:
        """这个路径的（上限，设置它的环境变量）；固定的那一档没有环境变量。tenant：限定作用域的
        令牌的名字（导入也用 max_body_bytes）。"""
        if path in IMPORT_PATHS and tenant is None:
            return self.max_import_bytes, MAX_IMPORT_ENV
        if path in IMPORT_PATHS or path.startswith(BODY_PATHS):
            return self.max_body_bytes, MAX_BODY_ENV
        return OTHER_BODY_BYTES, None

    def _tenant(self, scope: Scope) -> Principal | None:
        """请求带的是哪个限定作用域的令牌；管理令牌、令牌不对、没有配令牌时 None。"""
        if self.keys is None or not self.keys.scoped:
            return None
        principal = bearer_principal(self.keys, Headers(scope=scope).get("authorization"))
        return None if principal is None or principal.admin else principal

    def _wait(self, principal: Principal) -> float:
        """令牌桶（容量 rate_per_minute，每秒补 rate_per_minute / 60 个）：这个请求能不能过。能过
        时拿走一个名额、返回 0；不能过时返回等到下一个名额的秒数（不拿）。写法是 GCRA：每个令牌
        只记一个时刻 next_free，桶到这个时刻才重新装满（每用一个名额推后 60 / rate 秒）；它比现在
        晚出 rate - 1 个间隔以上时桶里不到一个名额。与记着桶里还剩几个等价，不用累加浮点数。"""
        rate = principal.rate_per_minute
        assert rate is not None
        interval = 60.0 / rate
        now = self.clock()
        due = max(self.next_free.get(principal.name, now), now)
        wait = due - now - (rate - 1) * interval  # 等到下一个名额的秒数；<= 0 时有名额
        if wait > 1e-9:
            return wait
        self.next_free[principal.name] = due + interval
        return 0.0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope["path"]
        principal = self._tenant(scope)
        tenant = principal.name if principal is not None else None
        if (
            principal is not None
            and principal.rate_per_minute is not None
            and scope["method"] == "POST"
            and path.startswith(RATE_PATHS)
        ):
            wait = self._wait(principal)
            if wait:
                message = _too_often(path, principal.rate_per_minute, wait)
                retry = {"Retry-After": str(max(1, math.ceil(wait - 1e-6)))}
                response = (
                    _error(429, message, None, retry)
                    if _v1(path)
                    else JSONResponse({"detail": message}, 429, retry)
                )
                await response(scope, receive, send)
                return
        cap, env = self.limit(path, tenant)
        length = Headers(scope=scope).get("content-length", "")
        declared = int(length) if length.isascii() and length.isdigit() else 0
        if declared > cap:
            message = _too_large(path, cap, env, tenant)
            response = (
                _error(413, message, None) if _v1(path) else JSONResponse({"detail": message}, 413)
            )
            await response(scope, receive, send)
            return
        if tenant is not None and self.in_flight.get(tenant, 0) + declared > self.max_body_bytes:
            message = _too_many(path, self.max_body_bytes)
            response = (
                _error(429, message, None, RETRY)
                if _v1(path)
                else JSONResponse({"detail": message}, 429, RETRY)
            )
            await response(scope, receive, send)
            return
        received = 0
        held = 0  # 这个请求记在 in_flight 里的字节数

        def hold(size: int) -> None:
            nonlocal held
            if tenant is not None and size > held:
                if self.in_flight.get(tenant, 0) + size - held > self.max_body_bytes:
                    raise HTTPException(429, _too_many(path, self.max_body_bytes), RETRY)
                self.in_flight[tenant] = self.in_flight.get(tenant, 0) + size - held
                held = size

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > cap:
                    raise HTTPException(413, _too_large(path, cap, env, tenant))
                hold(received)  # 分块传输，或者比声明的长
            return message

        hold(declared)
        try:
            await self.app(scope, counted, send)
        finally:
            if tenant is not None and held:
                left = self.in_flight.get(tenant, 0) - held
                if left > 0:
                    self.in_flight[tenant] = left
                else:
                    self.in_flight.pop(tenant, None)


RETRY = {"Retry-After": "1"}


def _v1(path: str) -> bool:
    return path == PREFIX or path.startswith(PREFIX + "/")


def _too_large(path: str, cap: int, env: str | None, tenant: str | None = None) -> str:
    """/api（/api/v1 除外）的出错是中文的；/api/v1、记忆代理与界面是英文的。限定作用域的令牌
    导入时另外说明更大的导入要用管理令牌。"""
    imported = tenant is not None and path in IMPORT_PATHS
    if path.startswith("/api/") and not _v1(path):
        return (
            f"请求体太大：这里的上限是 {cap} 字节"
            + (f"（环境变量 {env}）" if env else "")
            + ("；更大的导入要用管理令牌" if imported else "")
        )
    return (
        f"Request body too large: the limit here is {cap} bytes"
        + (f" (set with {env})" if env else "")
        + ("; larger imports need the admin token." if imported else ".")
    )


def _too_many(path: str, budget: int) -> str:
    """一个限定作用域的令牌同时在处理的请求体加起来超过上限（429）。"""
    if path.startswith("/api/") and not _v1(path):
        return (
            f"这个访问令牌同时在处理的请求体太多：加起来最多 {budget} 字节"
            f"（环境变量 {MAX_BODY_ENV}），请稍后重试"
        )
    return (
        f"Too many request bodies in flight for this access key: at most {budget} bytes at "
        f"once (set with {MAX_BODY_ENV}); retry shortly."
    )


def _too_often(path: str, rate: int, wait: float) -> str:
    """一个限定作用域的令牌超过了 rate_per_minute（429）。"""
    seconds = max(1, math.ceil(wait - 1e-6))
    if path.startswith("/api/") and not _v1(path):
        return f"这个访问令牌的请求太频繁：每分钟最多 {rate} 个，请 {seconds} 秒后重试"
    return (
        f"Too many requests for this access key: at most {rate} requests per minute; "
        f"retry in {seconds} s."
    )


def body_limits(environ: Mapping[str, str] | None = None) -> tuple[int, int]:
    """环境变量给出的（max_body_bytes，max_import_bytes），单位字节；没设或是空的时用默认值
    （64 MiB、1 GiB）。不是正整数时 CueMemError（cuemem serve / proxy 打印原因、退出码 1）。"""
    env = os.environ if environ is None else environ
    values: list[int] = []
    for name, default in ((MAX_BODY_ENV, MAX_BODY_BYTES), (MAX_IMPORT_ENV, MAX_IMPORT_BYTES)):
        raw = (env.get(name) or "").strip()
        if not raw:
            values.append(default)
            continue
        if not (raw.isascii() and raw.isdigit()) or int(raw) <= 0:
            raise CueMemError(f"{name} 要是正整数（字节数，默认 {default}），现在是 {raw!r}")
        values.append(int(raw))
    return values[0], values[1]
