"""Mem0 形状的 HTTP 记忆接口（/api/v1，N12 之后）：Python SDK（cuemem.sdk.Memory）的每个方法一个
接口，cuemem.client.MemoryClient 照同样的方法签名调用它们，代码在本地与远程之间只换一行。

    POST   /api/v1/memories                  add（messages、user_id、agent_id、run_id、metadata、
                                             expiration_date、timestamp、infer、wait）
    POST   /api/v1/memories/search           search（query、filters、top_k、mode、min_level、
                                             show_expired…）
    GET    /api/v1/memories                  get_all（filters 是 JSON 字符串）
    GET    /api/v1/memories/{id}             get（不存在时 404）
    PUT    /api/v1/memories/{id}             update（text 或 data、metadata）
    GET    /api/v1/memories/{id}/history     history
    DELETE /api/v1/memories/{id}             delete（dry_run）
    DELETE /api/v1/memories                  delete_all（user_id、agent_id、run_id）
    POST   /api/v1/reset                     reset
    GET    /api/v1/status                    status（抽取的状态：待处理、失败的会话与原因）
    以及 answer、profile、runs（open_run）、runs/end（end_run）、consolidate、wait、export、
    import（load）、spaces。

作用域与 SDK 相同：user_id、agent_id 合起来定一个记忆空间，不碰原始的空间 ID（"/" 不用转义）。
鉴权与 /api 的其他接口相同（CUEMEM_API_TOKEN）；限定作用域的令牌（api/auth.py）只能访问自己的
用户：点名了作用域之外的 user_id、agent_id（包括写在 filters 里的）时 403，按 ID 找的记忆在作用域
之外时与不存在的一样（404；history 是空列表），spaces 只列出作用域里的，reset 只有管理令牌能用。

报错是英文的 {"error": {"code", "type", "message"}}：type 是 SDK 抛出的异常类型（KeyError、
ValueError、TypeError 或 cuemem.errors 里的类型），MemoryClient 据此抛出同样的异常。请求体不合
格式（422）、令牌不对（401）在 /api/v1 下也写成这个形状（install_error_format）。模型与向量供应商
出错时的状态码与 /api 相同（api/failures.py：限流 429 带 Retry-After、余额不足 402、超时 504……），
code 是出错的类别（rate_limited、no_balance、timeout 等）；向量接口的出错另带 upstream_status
（上游原来的状态码），MemoryClient 抛出的 EmbeddingAPIError 与本地一样带着 status 与 retry_after。

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
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from cuemem.api.auth import Principal, principal_of
from cuemem.api.failures import provider_failure
from cuemem.errors import ConcurrentWriteError, CueMemError, EmbeddingAPIError
from cuemem.read.controller import Router
from cuemem.sdk import AnswerMode, Memory, SearchMode, _scope, space_for

PREFIX = "/api/v1"
_CODES = {
    400: "invalid_request",
    401: "unauthorized",
    402: "no_balance",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "invalid_request",
    429: "rate_limited",
    502: "upstream_error",
    503: "unavailable",
    504: "timeout",
}


class V1Error(HTTPException):
    """/api/v1 的出错：状态码、错误码与 SDK 的异常类型（MemoryClient 据此抛出同样的异常）。
    code 不给时按状态码取（_CODES）；headers：例如限流时上游的 Retry-After。upstream_status：
    向量接口的出错（EmbeddingAPIError）原来的状态码，写进出错的 upstream_status，MemoryClient
    据此还原异常的 status（回给调用方的状态码可能不同：上游的 500 是 502）。"""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: str | None = None,
        code: str | None = None,
        headers: Mapping[str, str] | None = None,
        upstream_status: int | None = None,
    ) -> None:
        super().__init__(status, message, headers=dict(headers) if headers else None)
        self.error_type = error_type
        self.code = code
        self.upstream_status = upstream_status


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
    show_expired: bool = False


class AnswerRequest(BaseModel):
    question: str
    user_id: str | None = None
    agent_id: str | None = None
    mode: AnswerMode = "auto"
    at: datetime | None = None
    router: Router = "learned"
    lam: float | None = None


class UpdateRequest(BaseModel):
    # text 与 data 是同一个参数（Mem0 2.x 的新旧两个名字，见 Memory.update）；MemoryClient 发 data
    text: str | None = None
    data: str | None = None
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
    别处的 KeyError 是程序错误，照常 500。模型与向量供应商的出错与 /api 相同（api/failures.py）：
    限流 429 与暂时不可用 503 带上游的 Retry-After，余额不足 402，超时 504，内容审核拒绝 422，
    别的 502；以前 openai 的异常是没有说明的 500，向量接口限流、内容审核拒绝都是 502。"""
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
    except Exception as exc:
        failure = provider_failure(exc)
        if failure is not None:
            raise V1Error(
                failure.status,
                failure.for_model(),
                error_type=type(exc).__name__,
                code=failure.kind,
                headers={"Retry-After": failure.retry_after} if failure.retry_after else None,
                upstream_status=exc.status if isinstance(exc, EmbeddingAPIError) else None,
            ) from exc
        if isinstance(exc, CueMemError):
            raise V1Error(502, str(exc), error_type=type(exc).__name__) from exc
        raise


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


def _allow(who: Principal, space: str | None) -> None:
    """请求点名的空间在令牌的作用域之外：403。"""
    if not who.allows(space):
        raise V1Error(403, f"This API key cannot access the memory space {space!r}.")


def memories_router(
    memory: Memory,
    *,
    dependencies: list[Any] | None = None,
    route_class: type[APIRoute] = APIRoute,
) -> APIRouter:
    """/api/v1 下的记忆接口，包住一个 Memory（服务里是 Memory.from_system，与界面、记忆代理共用
    同一个记忆系统）。同步的接口：FastAPI 在线程池里执行，Memory 本身线程安全。

    dependencies 要有 api.auth.guard（认出令牌）：每个接口按它的作用域检查，没有时报错（500），
    不当成管理令牌放行。route_class：api.auth.authenticated_route，令牌不对时不读请求体。"""
    router = APIRouter(
        prefix=PREFIX,
        dependencies=dependencies or [],
        tags=["memories (v1)"],
        route_class=route_class,
    )

    def allow(
        http: Request,
        user_id: str | None,
        agent_id: str | None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """user_id、agent_id 定下的空间（filters 里的 user_id、agent_id 优先）要在作用域里。与 SDK
        用同一个函数解析 filters：检查的空间就是 SDK 要读的空间。都没给时报 SDK 的 ValueError。"""
        user_id, agent_id, _, _ = _scope(user_id, agent_id, None, filters)
        _allow(principal_of(http), space_for(user_id, agent_id))

    def visible(http: Request, memory_id: str) -> bool:
        """按 ID 找的记忆：不存在的，或者在作用域里的。作用域之外的与不存在的一样处理。"""
        space = memory.system.db.fact_space(memory_id)
        return space is None or principal_of(http).allows(space)

    def missing(memory_id: str) -> V1Error:
        return V1Error(404, f"No memory with id {memory_id}", error_type="KeyError")

    @router.post("/memories")
    def add(http: Request, request: AddRequest) -> dict[str, Any]:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id)
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
    def search(http: Request, request: SearchRequest) -> dict[str, Any]:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id, request.filters)
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
                show_expired=request.show_expired,
            )

    @router.get("/memories")
    def get_all(
        http: Request,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_superseded: bool = False,
        valid_at: str | None = None,
        known_at: datetime | None = None,
        limit: int | None = None,
        filters: str | None = None,
        top_k: int | None = None,
        show_expired: bool = False,
    ) -> dict[str, Any]:
        where = _filters(filters)
        with _sdk_errors():
            allow(http, user_id, agent_id, where)
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
                show_expired=show_expired,
            )

    @router.delete("/memories")
    def delete_all(
        http: Request,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        with _sdk_errors():
            allow(http, user_id, agent_id)
            return memory.delete_all(user_id=user_id, agent_id=agent_id, run_id=run_id)

    @router.get("/memories/{memory_id}")
    def get(http: Request, memory_id: str) -> dict[str, Any]:
        if not visible(http, memory_id):
            raise missing(memory_id)
        with _sdk_errors(by_id=True):
            item = memory.get(memory_id)
        if item is None:
            raise missing(memory_id)
        return item

    @router.put("/memories/{memory_id}")
    def update(http: Request, memory_id: str, request: UpdateRequest) -> dict[str, Any]:
        if not visible(http, memory_id):
            raise missing(memory_id)
        with _sdk_errors(by_id=True):
            return memory.update(
                memory_id, request.text, metadata=request.metadata, data=request.data
            )

    @router.get("/memories/{memory_id}/history")
    def history(http: Request, memory_id: str) -> list[dict[str, Any]]:
        if not visible(http, memory_id):
            return []  # 与不存在的 ID 相同
        with _sdk_errors(by_id=True):
            return memory.history(memory_id)

    @router.delete("/memories/{memory_id}")
    def delete(http: Request, memory_id: str, dry_run: bool = False) -> dict[str, Any]:
        if not visible(http, memory_id):
            raise missing(memory_id)
        with _sdk_errors(by_id=True):
            return memory.delete(memory_id, dry_run=dry_run)

    @router.post("/reset")
    def reset(http: Request) -> dict[str, Any]:
        if not principal_of(http).admin:  # 删的是每个用户的记忆
            raise V1Error(403, "Only the admin token (CUEMEM_API_TOKEN) can reset all memories.")
        with _sdk_errors():
            return memory.reset()

    @router.post("/answer")
    def answer(http: Request, request: AnswerRequest) -> dict[str, Any]:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id)
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
    def profile(
        http: Request, user_id: str | None = None, agent_id: str | None = None
    ) -> dict[str, str]:
        with _sdk_errors():
            allow(http, user_id, agent_id)
            return {"profile": memory.profile(user_id=user_id, agent_id=agent_id)}

    @router.post("/runs")
    def open_run(http: Request, request: RunRequest) -> dict[str, str]:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id)
            run_id = memory.open_run(
                request.run_id, user_id=request.user_id, agent_id=request.agent_id
            )
        return {"run_id": run_id}

    @router.post("/runs/end")
    def end_run(http: Request, request: EndRunRequest) -> dict[str, Any]:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id)
            return memory.end_run(
                request.run_id,
                user_id=request.user_id,
                agent_id=request.agent_id,
                wait=request.wait,
            )

    @router.post("/consolidate")
    def consolidate(http: Request, request: ConsolidateRequest) -> dict[str, Any]:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id)
            return memory.consolidate(
                user_id=request.user_id, agent_id=request.agent_id, rebuild=request.rebuild
            )

    @router.post("/wait")
    def wait() -> dict[str, bool]:
        memory.wait()
        return {"ok": True}

    @router.get("/status")
    def status(
        http: Request, user_id: str | None = None, agent_id: str | None = None
    ) -> dict[str, Any]:
        with _sdk_errors():
            allow(http, user_id, agent_id)
            return memory.status(user_id=user_id, agent_id=agent_id)

    @router.get("/export")
    def export(
        http: Request, user_id: str | None = None, agent_id: str | None = None, derived: bool = True
    ) -> dict[str, Any]:
        with _sdk_errors(by_id=True):  # 没有这个空间：KeyError（404），与本地相同
            allow(http, user_id, agent_id)
            return memory.export(user_id=user_id, agent_id=agent_id, derived=derived)

    @router.post("/import")
    def load(http: Request, request: ImportRequest) -> dict[str, Any]:
        # 写进哪个空间与 Memory.load 的取法相同：给了 user_id、agent_id 时是它们的空间，否则是
        # 文件里的 space_id
        if request.user_id or request.agent_id:
            target: str | None = space_for(request.user_id, request.agent_id)
        else:
            target = str(request.data["space_id"]) if "space_id" in request.data else None
        _allow(principal_of(http), target)
        with _sdk_errors():
            return memory.load(
                request.data,
                user_id=request.user_id,
                agent_id=request.agent_id,
                wait=request.wait,
            )

    @router.get("/spaces")
    def spaces(http: Request) -> list[dict[str, Any]]:
        who = principal_of(http)
        return [row for row in memory.spaces() if who.allows(row["space_id"])]

    return router


def install_error_format(app: FastAPI) -> None:
    """/api/v1 下的出错（包括 401、422、找不到接口）都写成 {"error": {"code", "type", "message"}}；
    别的路径照 FastAPI 的默认（{"detail": …}）。"""

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        if not _v1(request):
            return await http_exception_handler(request, exc)
        return _error(
            exc.status_code,
            str(exc.detail),
            getattr(exc, "error_type", None),
            exc.headers,
            getattr(exc, "code", None),
            getattr(exc, "upstream_status", None),
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
    status: int,
    message: str,
    error_type: str | None,
    headers: Mapping[str, str] | None = None,
    code: str | None = None,
    upstream_status: int | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "code": code or _CODES.get(status, "error"),
        "type": error_type,
        "message": message,
    }
    if upstream_status is not None:
        body["upstream_status"] = upstream_status
    return JSONResponse({"error": body}, status, dict(headers or {}))
