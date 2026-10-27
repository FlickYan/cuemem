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
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from cuemem.api.memories import PREFIX, _error
from cuemem.errors import CueMemError

MAX_BODY_ENV = "CUEMEM_MAX_BODY_BYTES"
MAX_IMPORT_ENV = "CUEMEM_MAX_IMPORT_BYTES"
MAX_BODY_BYTES = 64 << 20
MAX_IMPORT_BYTES = 1 << 30
OTHER_BODY_BYTES = 8 << 20  # 界面的登录与事件（不可配置）
IMPORT_PATHS = ("/api/spaces/import", "/api/v1/import")
BODY_PATHS = ("/api/", "/v1/", "/gradio_api/upload")


class BodyLimit:
    """整个应用的中间件：请求体超过这个路径的上限时 413（档位见模块说明）。带 Content-Length
    时在应用读请求体之前就回；分块传输时边读边数，超过时抛出 HTTPException(413)：FastAPI 读请求体
    时遇到 HTTPException 原样抛出，由应用（或挂载的 Gradio）的出错处理写成响应。"""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int = MAX_BODY_BYTES,
        max_import_bytes: int = MAX_IMPORT_BYTES,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.max_import_bytes = max_import_bytes

    def limit(self, path: str) -> tuple[int, str | None]:
        """这个路径的（上限，设置它的环境变量）；固定的那一档没有环境变量。"""
        if path in IMPORT_PATHS:
            return self.max_import_bytes, MAX_IMPORT_ENV
        if path.startswith(BODY_PATHS):
            return self.max_body_bytes, MAX_BODY_ENV
        return OTHER_BODY_BYTES, None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path: str = scope["path"]
        cap, env = self.limit(path)
        length = Headers(scope=scope).get("content-length", "")
        if length.isascii() and length.isdigit() and int(length) > cap:
            message = _too_large(path, cap, env)
            response = (
                _error(413, message, None) if _v1(path) else JSONResponse({"detail": message}, 413)
            )
            await response(scope, receive, send)
            return
        received = 0

        async def counted() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > cap:
                    raise HTTPException(413, _too_large(path, cap, env))
            return message

        await self.app(scope, counted, send)


def _v1(path: str) -> bool:
    return path == PREFIX or path.startswith(PREFIX + "/")


def _too_large(path: str, cap: int, env: str | None) -> str:
    """/api（/api/v1 除外）的出错是中文的；/api/v1、记忆代理与界面是英文的。"""
    if path.startswith("/api/") and not _v1(path):
        return f"请求体太大：这里的上限是 {cap} 字节" + (f"（环境变量 {env}）" if env else "")
    return f"Request body too large: the limit here is {cap} bytes" + (
        f" (set with {env})." if env else "."
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
