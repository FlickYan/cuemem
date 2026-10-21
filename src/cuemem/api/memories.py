"""Mem0 形状的 HTTP 记忆接口（/api/v1，N12 之后）：Python SDK（cuemem.sdk.Memory）的每个方法一个
接口，cuemem.client.MemoryClient 照同样的方法签名调用它们，代码在本地与远程之间只换一行。

    POST   /api/v1/memories                  add（messages、user_id、agent_id、run_id、metadata、
                                             expiration_date、timestamp、infer、wait）
    POST   /api/v1/memories/search           search（query、filters、top_k、mode、min_level…）
    GET    /api/v1/memories                  get_all（filters 是 JSON 字符串）
    GET    /api/v1/memories/{id}             get（不存在时 404）
    PUT    /api/v1/memories/{id}             update（data、metadata）
    GET    /api/v1/memories/{id}/history     history
    DELETE /api/v1/memories/{id}             delete（dry_run）
    DELETE /api/v1/memories                  delete_all（user_id、agent_id、run_id）
    POST   /api/v1/reset                     reset
    以及 answer、profile、runs（open_run）、runs/end（end_run）、consolidate、wait、export、
    import（load）、spaces。

作用域与 SDK 相同：user_id、agent_id 合起来定一个记忆空间，不碰原始的空间 ID（"/" 不用转义）。
鉴权与 /api 的其他接口相同（CUEMEM_API_TOKEN）。

报错是英文的 {"error": {"code", "type", "message"}}：type 是 SDK 抛出的异常类型（KeyError、
ValueError、TypeError 或 cuemem.errors 里的类型），MemoryClient 据此抛出同样的异常。请求体不合
格式（422）、令牌不对（401）在 /api/v1 下也写成这个形状（install_error_format）。

图片只收 data URL：服务器不读自己磁盘上的文件（与记忆代理相同），MemoryClient 把本地路径先读成
data URL 再发；网络图片照 SDK 的规矩不下载。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from cuemem.errors import ConcurrentWriteError, CueMemError
from cuemem.read.controller import Router
from cuemem.sdk import AnswerMode, Memory, SearchMode

PREFIX = "/api/v1"
_CODES = {
    400: "invalid_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    415: "unsupported_media_type",
    422: "invalid_request",
    502: "upstream_error",
}


class V1Error(HTTPException):
    """/api/v1 的出错：状态码、错误码与 SDK 的异常类型（MemoryClient 据此抛出同样的异常）。"""

    def __init__(self, status: int, message: str, *, error_type: str | None = None) -> None:
        super().__init__(status, message)
        self.error_type = error_type


class AddRequest(BaseModel):
    messages: str | dict[str, Any] | list[dict[str, Any]]
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] | None = None
    expiration_date: str | None = Field(None, description="ISO date or datetime")
    timestamp: str | float | None = Field(None, description="ISO datetime or Unix time")
    infer: bool = True
    wait: bool = False


class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    limit: int | None = None
    mode: SearchMode = "auto"
    at: datetime | None = None
    router: Router = "learned"
    lam: float | None = None
    filters: dict[str, Any] | None = None
    top_k: int | None = None
    min_level: Literal["l0", "l1"] = "l1"


class AnswerRequest(BaseModel):
    question: str
    user_id: str | None = None
    agent_id: str | None = None
    mode: AnswerMode = "auto"
    at: datetime | None = None
    router: Router = "learned"
    lam: float | None = None


class UpdateRequest(BaseModel):
    data: str
    metadata: dict[str, Any] | None = None


class RunRequest(BaseModel):
    run_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None


class EndRunRequest(BaseModel):
    run_id: str
    user_id: str | None = None
    agent_id: str | None = None
    wait: bool = False


class ConsolidateRequest(BaseModel):
    user_id: str | None = None
    agent_id: str | None = None
    rebuild: bool = False


class ImportRequest(BaseModel):
    data: dict[str, Any]
    user_id: str | None = None
    agent_id: str | None = None
    wait: bool = False


@contextmanager
def _sdk_errors(*, by_id: bool = False) -> Iterator[None]:
    """SDK 的异常 → /api/v1 的出错。KeyError 只在按 ID 找记忆的接口上算"没有这条"（404），
    别处的 KeyError 是程序错误，照常 500。"""
    try:
        yield
    except KeyError as exc:
        if not by_id:
            raise
        raise V1Error(404, str(exc.args[0] if exc.args else exc), error_type="KeyError") from exc
    except (ValueError, TypeError) as exc:
        raise V1Error(400, str(exc), error_type=type(exc).__name__) from exc
    except ConcurrentWriteError as exc:
        raise V1Error(409, str(exc), error_type=type(exc).__name__) from exc
    except CueMemError as exc:
        raise V1Error(502, str(exc), error_type=type(exc).__name__) from exc


def _no_server_files(messages: str | dict[str, Any] | list[dict[str, Any]]) -> None:
    """图片只收 data URL（网络图片交给 SDK 报它自己的错）：本地路径与 file:// 指向的是服务器上的
    文件，远程调用方不能借此把服务器上的文件写进记忆。内容块写错时报 ValueError，不让 SDK 在
    取 image_url 时抛 KeyError。"""
    items = [messages] if isinstance(messages, dict) else messages
    for message in items if isinstance(items, list) else []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("Each content block must be an object.")
            if block.get("type") != "image_url":
                continue
            value = block.get("image_url")
            url = value.get("url") if isinstance(value, dict) else value
            if not isinstance(url, str) or not url.startswith(("data:", "http://", "https://")):
                raise ValueError(
                    "Send images as data URLs: the server does not read files from its own disk "
                    "(MemoryClient converts local paths for you)."
                )


def _filters(raw: str | None) -> dict[str, Any] | None:
    """GET 的 filters 是 JSON 字符串（嵌套的条件放不进普通的查询参数）。"""
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, dict):
        raise V1Error(400, "filters must be a JSON object.", error_type="ValueError")
    return value


def memories_router(memory: Memory, *, dependencies: list[Any] | None = None) -> APIRouter:
    """/api/v1 下的记忆接口，包住一个 Memory（服务里是 Memory.from_system，与界面、记忆代理共用
    同一个记忆系统）。同步的接口：FastAPI 在线程池里执行，Memory 本身线程安全。"""
    router = APIRouter(prefix=PREFIX, dependencies=dependencies or [], tags=["memories (v1)"])

    @router.post("/memories")
    def add(request: AddRequest) -> dict[str, Any]:
        with _sdk_errors():
            _no_server_files(request.messages)
            return memory.add(
                request.messages,
                user_id=request.user_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                metadata=request.metadata,
                expiration_date=request.expiration_date,
                timestamp=request.timestamp,
                infer=request.infer,
                wait=request.wait,
            )

    @router.post("/memories/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        with _sdk_errors():
            return memory.search(
                request.query,
                user_id=request.user_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                limit=request.limit,
                mode=request.mode,
                at=request.at,
                router=request.router,
                lam=request.lam,
                filters=request.filters,
                top_k=request.top_k,
                min_level=request.min_level,
            )

    @router.get("/memories")
    def get_all(
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_superseded: bool = False,
        valid_at: str | None = None,
        known_at: datetime | None = None,
        limit: int | None = None,
        filters: str | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        where = _filters(filters)
        with _sdk_errors():
            return memory.get_all(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                include_superseded=include_superseded,
                valid_at=valid_at,
                known_at=known_at,
                limit=limit,
                filters=where,
                top_k=top_k,
            )

    @router.delete("/memories")
    def delete_all(
        user_id: str | None = None, agent_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        with _sdk_errors():
            return memory.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)

    @router.get("/memories/{memory_id}")
    def get(memory_id: str) -> dict[str, Any]:
        with _sdk_errors(by_id=True):
            item = memory.get(memory_id)
        if item is None:
            raise V1Error(404, f"No memory with id {memory_id}", error_type="KeyError")
        return item

    @router.put("/memories/{memory_id}")
    def update(memory_id: str, request: UpdateRequest) -> dict[str, Any]:
        with _sdk_errors(by_id=True):
            return memory.update(memory_id, request.data, metadata=request.metadata)

    @router.get("/memories/{memory_id}/history")
    def history(memory_id: str) -> list[dict[str, Any]]:
        with _sdk_errors(by_id=True):
            return memory.history(memory_id)

    @router.delete("/memories/{memory_id}")
    def delete(memory_id: str, dry_run: bool = False) -> dict[str, Any]:
        with _sdk_errors(by_id=True):
            return memory.delete(memory_id, dry_run=dry_run)

    @router.post("/reset")
    def reset() -> dict[str, Any]:
        with _sdk_errors():
            return memory.reset()

    @router.post("/answer")
    def answer(request: AnswerRequest) -> dict[str, Any]:
        with _sdk_errors():
            return memory.answer(
                request.question,
                user_id=request.user_id,
                agent_id=request.agent_id,
                mode=request.mode,
                at=request.at,
                router=request.router,
                lam=request.lam,
            )

    @router.get("/profile")
    def profile(user_id: str | None = None, agent_id: str | None = None) -> dict[str, str]:
        with _sdk_errors():
            return {"profile": memory.profile(user_id=user_id, agent_id=agent_id)}

    @router.post("/runs")
    def open_run(request: RunRequest) -> dict[str, str]:
        with _sdk_errors():
            run_id = memory.open_run(
                request.run_id, user_id=request.user_id, agent_id=request.agent_id
            )
        return {"run_id": run_id}

    @router.post("/runs/end")
    def end_run(request: EndRunRequest) -> dict[str, Any]:
        with _sdk_errors():
            return memory.end_run(
                request.run_id,
                user_id=request.user_id,
                agent_id=request.agent_id,
                wait=request.wait,
            )

    @router.post("/consolidate")
    def consolidate(request: ConsolidateRequest) -> dict[str, Any]:
        with _sdk_errors():
            return memory.consolidate(
                user_id=request.user_id, agent_id=request.agent_id, rebuild=request.rebuild
            )

    @router.post("/wait")
    def wait() -> dict[str, bool]:
        memory.wait()
        return {"ok": True}

    @router.get("/export")
    def export(
        user_id: str | None = None, agent_id: str | None = None, derived: bool = True
    ) -> dict[str, Any]:
        with _sdk_errors():
            return memory.export(user_id=user_id, agent_id=agent_id, derived=derived)

    @router.post("/import")
    def load(request: ImportRequest) -> dict[str, Any]:
        with _sdk_errors():
            return memory.load(
                request.data,
                user_id=request.user_id,
                agent_id=request.agent_id,
                wait=request.wait,
            )

    @router.get("/spaces")
    def spaces() -> list[dict[str, Any]]:
        return memory.spaces()

    return router


def install_error_format(app: FastAPI) -> None:
    """/api/v1 下的出错（包括 401、422、找不到接口）都写成 {"error": {"code", "type", "message"}}；
    别的路径照 FastAPI 的默认（{"detail": …}）。"""

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        if not _v1(request):
            return await http_exception_handler(request, exc)
        return _error(
            exc.status_code, str(exc.detail), getattr(exc, "error_type", None), exc.headers
        )

    async def validation_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        if not _v1(request):
            return await request_validation_exception_handler(request, exc)
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}"
            for e in exc.errors()
        )
        return _error(422, f"Invalid request: {problems}", "ValueError")

    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(RequestValidationError, validation_error)


def _v1(request: Request) -> bool:
    return request.url.path.startswith(PREFIX + "/") or request.url.path == PREFIX


def _error(
    status: int, message: str, error_type: str | None, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    body = {"code": _CODES.get(status, "error"), "type": error_type, "message": message}
    return JSONResponse({"error": body}, status, dict(headers or {}))
