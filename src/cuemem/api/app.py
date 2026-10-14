"""HTTP 接口（FastAPI，N8）：会话与对话、记忆浏览、遗忘、巩固，以及"只检索不回答"的记忆服务。

界面（Gradio）挂在同一个应用的根路径上，与接口共用一个服务对象（api/ui.py）。
接口文档：启动后打开 /docs。

访问令牌（可选，N11）：设置环境变量 CUEMEM_API_TOKEN 后，/api 下除健康检查外的接口都要带
Authorization: Bearer <令牌>，界面先要求登录（用户名任意，密码填令牌）。不设置时不鉴权，
只适合在本机演示（cuemem serve 默认只监听 127.0.0.1）。
"""

from __future__ import annotations

import secrets
import shutil
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, get_args

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from cuemem import __version__
from cuemem.agent import Reply
from cuemem.api.service import ChatService
from cuemem.errors import CueMemError
from cuemem.memory import RecallMode
from cuemem.read.controller import Router

if TYPE_CHECKING:
    from cuemem.api.proxy import MemoryProxy

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
TOKEN_ENV = "CUEMEM_API_TOKEN"
# 放在模块级：注解是字符串（from __future__ import annotations），FastAPI 解析不到函数里的局部变量
_BEARER = HTTPBearer(auto_error=False)


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


@contextmanager
def _errors() -> Iterator[None]:
    """把服务层的已知错误转成 HTTP 状态码。"""
    try:
        yield
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0] if exc.args else exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except CueMemError as exc:
        raise HTTPException(502, str(exc)) from exc


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
) -> FastAPI:
    """token 非空时启用访问令牌：接口要带 Bearer 令牌，界面要先登录。

    proxy：同时提供 OpenAI 兼容的记忆代理（/v1/chat/completions，N12），与界面共用一个记忆系统，
    经代理写入的对话可以在界面里浏览。
    """
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
        paths = [_save_upload(upload, upload_dir) for upload in images or []]
        with _errors():
            return service.chat(
                session_id, text, images=paths, mode=_mode(mode), router=_router(router), lam=lam
            )

    @api.post("/api/sessions/{session_id}/end")
    def end(session_id: str, wait: bool = False) -> dict[str, Any]:
        with _errors():
            return service.end(session_id, wait=wait)

    @api.post("/api/spaces/{space}/recall")
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

    @api.get("/api/spaces/{space}/profile")
    def profile(space: str) -> dict[str, str]:
        return {"text": service.profile(space)}

    @api.get("/api/spaces/{space}/facts")
    def facts(space: str, current_only: bool = False) -> list[dict[str, Any]]:
        return service.facts(space, current_only=current_only)

    @api.get("/api/spaces/{space}/entities")
    def entities(space: str) -> list[dict[str, Any]]:
        return service.entities(space)

    @api.get("/api/spaces/{space}/graph")
    def graph(space: str) -> dict[str, Any]:
        return service.graph(space)

    @api.get("/api/spaces/{space}/media")
    def media(space: str) -> list[dict[str, Any]]:
        return service.media(space)

    @api.get("/api/spaces/{space}/media/{media_id}")
    def media_file(space: str, media_id: str) -> FileResponse:
        items = service.system.db.get_media(space, ids=[media_id])
        if not items or not Path(items[0].path).exists():
            raise HTTPException(404, "没有这张图片")
        return FileResponse(items[0].path)

    @api.get("/api/spaces/{space}/sessions")
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

    @api.get("/api/spaces/{space}/status")
    def status(space: str) -> dict[str, Any]:
        return service.status(space)

    @api.post("/api/spaces/{space}/forget")
    def forget(space: str, request: ForgetRequest) -> dict[str, Any]:
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

    @api.post("/api/spaces/{space}/consolidate")
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
            allowed_paths=[str(upload_dir), str(service.system.settings.paths.media)],
            footer_links=[],  # 接口文档在 /docs
            auth=_ui_login(token),
            auth_message="Sign in with any username and the access token as the password.",
        )
    return app


def _save_upload(upload: UploadFile, directory: Path) -> str:
    suffix = Path(upload.filename or "").suffix.lower() or ".jpg"
    if suffix not in IMAGE_TYPES:
        raise HTTPException(415, f"只支持图片：{', '.join(sorted(IMAGE_TYPES))}")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    return str(target)
