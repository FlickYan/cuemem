"""HTTP 接口（FastAPI，N8）：会话与对话、记忆浏览、遗忘、巩固，以及"只检索不回答"的记忆服务。

界面（Gradio）挂在同一个应用的根路径上，与接口共用一个服务对象（api/ui.py）。
接口文档：启动后打开 /docs。

访问令牌（N11）：设置环境变量 CUEMEM_API_TOKEN 后，/api 下除健康检查外的接口都要带
Authorization: Bearer <令牌>，界面先要求登录（用户名任意，密码填令牌）。不设置时不鉴权，
只适合在本机演示：cuemem serve 默认只监听 127.0.0.1，监听别的地址又没有令牌时拒绝启动
（加 --insecure-no-auth 才照常启动）。

浏览器（N12）：/api、/v1 与界面的 /gradio_api 不接受别的网站发来的请求（SameOriginOnly），否则
任何网页都能借访问者的浏览器读写本机上不鉴权的记忆、花上游的额度；allowed_hosts 给出时只接受
这些 Host（cuemem serve / proxy 监听本机地址时设置，防 DNS 重绑定：恶意域名解析到 127.0.0.1 后，
它的网页就算同源）。
"""

from __future__ import annotations

import secrets
import shutil
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, get_args
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from cuemem import __version__
from cuemem.agent import Reply
from cuemem.api.failures import provider_failure
from cuemem.api.service import ChatService
from cuemem.errors import CueMemError
from cuemem.memory import RecallMode
from cuemem.portable import export_space, import_space
from cuemem.read.controller import Router

if TYPE_CHECKING:
    from cuemem.api.proxy import MemoryProxy

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
TOKEN_ENV = "CUEMEM_API_TOKEN"
# 额外接受的主机名（逗号分隔，"*" 表示不限）：监听本机地址时追加进 Host 白名单，也作为 Origin 放行
HOSTS_ENV = "CUEMEM_ALLOWED_HOSTS"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")  # Host 头里的 IPv6 地址带方括号
# 放在模块级：注解是字符串（from __future__ import annotations），FastAPI 解析不到函数里的局部变量
_BEARER = HTTPBearer(auto_error=False)


class SameOriginOnly:
    """/api、/v1 与界面自己的接口（/gradio_api）不接受别的网站发来的浏览器请求：带 Origin 头时，
    Origin 的主机与端口要与 Host 头相同（同源，例如 /docs 的 Try it out、界面本身），或者 Origin
    的主机名在 hosts 里（反向代理改写了 Host 时）；Origin: null 一律拒绝。不带 Origin 的请求（SDK、
    curl、gradio_client）不受影响。

    是整个应用的中间件，不是接口的依赖：预检请求（OPTIONS）匹配不到接口，会落到根路径上的
    Gradio。Gradio 只在 Host 正好是 localhost、127.0.0.1、0.0.0.0 时限制来源，别的 Host（监听
    ::1、127.0.0.2 等本机地址，Docker、局域网）放行任何来源并允许带凭据：它的 /gradio_api 能读出
    任何空间的记忆、遗忘、对话，上传接口不经预检就会写文件。"""

    PATHS = ("/api/", "/v1/", "/gradio_api/")

    def __init__(self, app: ASGIApp, hosts: Sequence[str] = ()) -> None:
        self.app = app
        self.hosts = {h.lower() for h in hosts}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(self.PATHS):
            headers = Headers(scope=scope)
            origin = headers.get("origin")
            if origin is not None and not self._allowed(origin, headers.get("host", "")):
                response = JSONResponse({"detail": "不接受别的网站发来的浏览器请求"}, 403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

    def _allowed(self, origin: str, host: str) -> bool:
        if "*" in self.hosts:
            return True
        try:
            parts = urlsplit(origin)
            name = parts.hostname or ""
        except ValueError:
            return False
        if parts.scheme not in ("http", "https") or not name:
            return False
        name = f"[{name}]" if ":" in name else name
        return parts.netloc.lower() == host.lower() or name in self.hosts


def _guard(app: FastAPI, allowed_hosts: Sequence[str] | None, origin_hosts: Sequence[str]) -> None:
    """浏览器侧的防护（create_app 与 create_proxy_app 共用）；Host 检查在最外层，先于 Origin。"""
    app.add_middleware(SameOriginOnly, hosts=origin_hosts)
    if allowed_hosts is not None:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(allowed_hosts))


class StartRequest(BaseModel):
    space: str | None = None
    at: datetime | None = Field(None, description="模拟的会话开始时间；不填为现在")


class SessionOut(BaseModel):
    session_id: str
    space: str
    started_at: datetime


class RecallRequest(BaseModel):
    query: str
    mode: str = "auto"
    router: str | None = None
    lam: float | None = None
    at: datetime | None = None


class ForgetRequest(BaseModel):
    session_ids: list[str] = Field(default_factory=list)
    turn_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    media_ids: list[str] = Field(default_factory=list)
    dry_run: bool = Field(False, description="只预演：返回会删哪些发言与事实，什么都不删")


@contextmanager
def _errors() -> Iterator[None]:
    """把服务层的已知错误转成 HTTP 状态码。模型与向量供应商的错误各有状态码与说明：限流 429、
    余额不足 402、暂时不可用 503（这两个带上游的 Retry-After）、超时 504、内容审核拒绝 422、
    其余 502（见 api/failures.py）。以前 OpenAI SDK 的异常是没有说明的 500。"""
    try:
        yield
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0] if exc.args else exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        failure = provider_failure(exc)
        if failure is not None:
            headers = {"Retry-After": failure.retry_after} if failure.retry_after else None
            raise HTTPException(failure.status, failure.message(), headers=headers) from exc
        if isinstance(exc, CueMemError):
            raise HTTPException(502, str(exc)) from exc
        raise


def _mode(value: str) -> RecallMode:
    if value not in get_args(RecallMode):
        raise HTTPException(422, f"mode 只能是 {'、'.join(get_args(RecallMode))}")
    return value  # type: ignore[return-value]


def _router(value: str | None) -> Router | None:
    if value is not None and value not in get_args(Router):
        raise HTTPException(422, f"router 只能是 {'、'.join(get_args(Router))}")
    return value  # type: ignore[return-value]


def _same(given: str, token: str) -> bool:
    return secrets.compare_digest(given.encode(), token.encode())


def _token_check(token: str) -> Callable[..., None]:
    def check(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
    ) -> None:
        if credentials is None or not _same(credentials.credentials, token):
            raise HTTPException(401, "缺少或错误的访问令牌", headers={"WWW-Authenticate": "Bearer"})

    return check


def _ui_login(token: str | None) -> Callable[[str, str], bool] | None:
    """界面的登录检查：用户名任意，密码是访问令牌。"""
    if not token:
        return None
    return lambda _user, password: _same(password, token)


def create_app(
    service: ChatService,
    *,
    upload_dir: Path,
    ui: bool = True,
    token: str | None = None,
    proxy: MemoryProxy | None = None,
    allowed_hosts: Sequence[str] | None = None,
    origin_hosts: Sequence[str] = (),
) -> FastAPI:
    """token 非空时启用访问令牌：接口要带 Bearer 令牌，界面要先登录。

    proxy：同时提供 OpenAI 兼容的记忆代理（/v1/chat/completions，N12），与界面共用一个记忆系统，
    经代理写入的对话可以在界面里浏览。

    allowed_hosts：只接受这些 Host（接口与界面；None 不检查）。origin_hosts：除同源外放行的
    Origin 主机名（"*" 不限）。见模块说明。

    upload_dir：上传的图片只在请求期间放在这里，写进记忆之后就删（媒体库里另存一份）。建应用时
    先清掉里面留下的图片：以前的版本一直留着上传的原图，遗忘之后磁盘上还有一份（D-021）；
    会话不跨进程，这时没有会话还在读它们。
    """
    _clear_uploads(upload_dir)
    app = FastAPI(
        title="CueMem",
        version=__version__,
        description="Long-term multimodal memory for LLM agents: chat, browse, forget, recall.",
    )
    api = APIRouter(dependencies=[Depends(_token_check(token))] if token else [])

    @app.get("/api/health")  # 不鉴权：给负载均衡与监控探活用
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "open_sessions": len(service.sessions)}

    @api.get("/api/spaces")
    def spaces() -> list[dict[str, Any]]:
        return service.spaces()

    @api.post("/api/sessions")
    def start(request: StartRequest) -> SessionOut:
        session = service.start(request.space, at=request.at)
        return SessionOut(
            session_id=session.session_id, space=session.space_id, started_at=session.started_at
        )

    @api.post("/api/sessions/{session_id}/messages")
    def message(
        session_id: str,
        text: Annotated[str, Form()],
        images: Annotated[list[UploadFile] | None, File()] = None,
        mode: Annotated[str, Form()] = "auto",
        router: Annotated[str | None, Form()] = None,
        lam: Annotated[float | None, Form()] = None,
    ) -> Reply:
        paths: list[str] = []
        try:
            for upload in images or []:
                paths.append(_save_upload(upload, upload_dir))
            with _errors():
                return service.chat(
                    session_id,
                    text,
                    images=paths,
                    mode=_mode(mode),
                    router=_router(router),
                    lam=lam,
                )
        finally:  # 上传的文件只用这一轮：图片已经存进记忆，会话改读那份；不留副本（遗忘删不到）
            _discard(paths, service, session_id)

    @api.post("/api/sessions/{session_id}/end")
    def end(session_id: str, wait: bool = False) -> dict[str, Any]:
        with _errors():
            return service.end(session_id, wait=wait)

    # 空间 ID 里可以有 "/"（sdk.space_for：bot/alice、agent/shop），所以用 {space:path}。
    # media 列表排在 media/{media_id} 前面：…/x/media/media 是空间 x/media 的列表（图片 ID 是哈希）
    @api.post("/api/spaces/{space:path}/recall")
    def recall(space: str, request: RecallRequest) -> dict[str, Any]:
        with _errors():
            return service.recall(
                space,
                request.query,
                mode=_mode(request.mode),
                router=_router(request.router),
                lam=request.lam,
                at=request.at,
            )

    @api.get("/api/spaces/{space:path}/profile")
    def profile(space: str) -> dict[str, str]:
        return {"text": service.profile(space)}

    @api.get("/api/spaces/{space:path}/facts")
    def facts(space: str, current_only: bool = False) -> list[dict[str, Any]]:
        return service.facts(space, current_only=current_only)

    @api.get("/api/spaces/{space:path}/entities")
    def entities(space: str) -> list[dict[str, Any]]:
        return service.entities(space)

    @api.get("/api/spaces/{space:path}/graph")
    def graph(space: str) -> dict[str, Any]:
        return service.graph(space)

    @api.get("/api/spaces/{space:path}/media")
    def media(space: str) -> list[dict[str, Any]]:
        return service.media(space)

    @api.get("/api/spaces/{space:path}/media/{media_id}")
    def media_file(space: str, media_id: str) -> FileResponse:
        items = service.system.db.get_media(space, ids=[media_id])
        if not items or not Path(items[0].path).exists():
            raise HTTPException(404, "没有这张图片")
        return FileResponse(items[0].path)

    @api.get("/api/spaces/{space:path}/sessions")
    def sessions(space: str) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "started_at": s.started_at.isoformat(),
                "open": s.open,
                "turns": s.turns,
            }
            for s in service.session_list(space)
        ]

    @api.get("/api/spaces/{space:path}/status")
    def status(space: str) -> dict[str, Any]:
        return service.status(space)

    @api.post("/api/spaces/{space:path}/forget")
    def forget(space: str, request: ForgetRequest) -> dict[str, Any]:
        if request.dry_run:
            with _errors():
                return service.system.preview_forget(
                    space,
                    session_ids=request.session_ids,
                    turn_ids=request.turn_ids,
                    fact_ids=request.fact_ids,
                    media_ids=request.media_ids,
                )
        with _errors():
            report = service.forget(
                space,
                session_ids=request.session_ids,
                turn_ids=request.turn_ids,
                fact_ids=request.fact_ids,
                media_ids=request.media_ids,
            )
        return {
            "turns_deleted": report.turns_deleted,
            "sessions_affected": report.sessions_affected,
            "facts_before": report.facts_before,
            "facts_after": report.facts_after,
            "images_deleted": report.images_deleted,
        }

    @api.get("/api/spaces/{space:path}/export")
    def export(space: str, derived: bool = True) -> dict[str, Any]:
        """导出记忆空间（N12）：原始日志、图片原文件与派生记忆的快照。"""
        with _errors():
            return export_space(service.system, space, derived=derived)

    @api.post("/api/spaces/import")
    def import_(data: dict[str, Any], space: str | None = None) -> dict[str, Any]:
        """导入 export 的结果（N12）：写回原始日志，派生记忆在后台重新抽取。"""
        with _errors():
            report = import_space(service.system, data, space_id=space)
        if service.system.worker is not None:
            service.system.worker.submit(report["space_id"])
        return report

    @api.post("/api/spaces/{space:path}/consolidate")
    def consolidate(space: str) -> dict[str, Any]:
        with _errors():
            return service.consolidate(space)

    app.include_router(api)
    if proxy is not None:
        from cuemem.api.proxy import proxy_router

        app.include_router(
            proxy_router(proxy, dependencies=[Depends(_token_check(token))] if token else [])
        )
    if ui:
        import gradio as gr

        from cuemem.api.ui import CSS, THEME, build_ui

        app = gr.mount_gradio_app(
            app,
            build_ui(service),
            path="/",
            theme=THEME,
            css=CSS,
            allowed_paths=[str(service.system.settings.paths.media)],  # 界面不读上传目录
            footer_links=[],  # 接口文档在 /docs
            auth=_ui_login(token),
            auth_message="Sign in with any username and the access token as the password.",
        )
    _guard(app, allowed_hosts, origin_hosts)  # 中间件：落到界面的预检请求也先过这里
    return app


def _clear_uploads(directory: Path) -> None:
    """删掉上传目录里直接放着的图片（只认 _save_upload 会写的扩展名，别的文件不动）。"""
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if path.suffix.lower() in IMAGE_TYPES and path.is_file():
            path.unlink(missing_ok=True)


def _save_upload(upload: UploadFile, directory: Path) -> str:
    suffix = Path(upload.filename or "").suffix.lower() or ".jpg"
    if suffix not in IMAGE_TYPES:
        raise HTTPException(415, f"只支持图片：{', '.join(sorted(IMAGE_TYPES))}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return str(target)


def _discard(paths: Sequence[str], service: ChatService, session_id: str) -> None:
    """删掉这一轮上传的文件。会话还在读的不删：记忆系统没有媒体库时只能读原文件。"""
    try:
        kept = set(service.get(session_id).uploads.values())
    except KeyError:
        kept = set()
    for path in paths:
        if path not in kept:
            Path(path).unlink(missing_ok=True)
