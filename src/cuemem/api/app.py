"""HTTP 接口（FastAPI，N8）：会话与对话、记忆浏览、遗忘、巩固，以及"只检索不回答"的记忆服务。

界面（Gradio）挂在同一个应用的根路径上，与接口共用一个服务对象（api/ui.py）。
接口文档：启动后打开 /docs。/api/v1 下是 Mem0 形状的记忆接口，与 Python SDK 一一对应
（api/memories.py，远程客户端是 cuemem.client.MemoryClient）。

访问令牌（N11）：设置环境变量 CUEMEM_API_TOKEN 后，/api 下除健康检查外的接口都要带
Authorization: Bearer <令牌>，界面先要求登录（用户名任意，密码填令牌）。不设置时不鉴权，
只适合在本机演示：cuemem serve 默认只监听 127.0.0.1，监听别的地址又没有令牌时拒绝启动
（加 --insecure-no-auth 才照常启动）。

多租户（N12 之后）：CUEMEM_API_KEYS_FILE 另外给出限定作用域的令牌，每个令牌只能访问列出的用户
（与 Agent）的记忆，/api、/api/v1、界面与记忆代理都按它检查（格式与规则见 api/auth.py）。界面里
用这种令牌登录时，用户名填令牌的名字。

浏览器（N12）：/api、/v1 与界面的 /gradio_api 不接受别的网站发来的请求（SameOriginOnly），否则
任何网页都能借访问者的浏览器读写本机上不鉴权的记忆、花上游的额度；allowed_hosts 给出时只接受
这些 Host（cuemem serve / proxy 监听本机地址时设置，防 DNS 重绑定：恶意域名解析到 127.0.0.1 后，
它的网页就算同源）。

请求体的上限（第三轮体检，api/limits.py）：/api、/v1 与界面的上传默认 64 MiB，导入 1 GiB，界面
别的接口 8 MiB，超过时 413，在读请求体之前；/api 与 /api/v1 也先认令牌再读请求体
（auth.authenticated_route）。
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, get_args
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from cuemem import __version__
from cuemem.agent import Reply
from cuemem.api.auth import Keys, Principal, authenticated_route, guard, principal_of, ui_login
from cuemem.api.failures import provider_failure
from cuemem.api.limits import MAX_BODY_BYTES, MAX_IMPORT_BYTES, BodyLimit
from cuemem.api.memories import install_error_format, memories_router
from cuemem.api.service import ChatService
from cuemem.config import extra_remedy
from cuemem.errors import CueMemError
from cuemem.memory import MemorySystem, RecallMode
from cuemem.portable import export_space, import_space
from cuemem.read.controller import Router
from cuemem.sdk import Memory

if TYPE_CHECKING:
    from cuemem.api.proxy import MemoryProxy

logger = logging.getLogger(__name__)

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
# 额外接受的主机名（逗号分隔，"*" 表示不限）：监听本机地址时追加进 Host 白名单，也作为 Origin 放行
HOSTS_ENV = "CUEMEM_ALLOWED_HOSTS"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]")  # Host 头里的 IPv6 地址带方括号


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


def _guard(
    app: FastAPI,
    allowed_hosts: Sequence[str] | None,
    origin_hosts: Sequence[str],
    *,
    max_body_bytes: int = MAX_BODY_BYTES,
    max_import_bytes: int = MAX_IMPORT_BYTES,
) -> None:
    """浏览器侧的防护与请求体的上限（create_app 与 create_proxy_app 共用）；Host 检查在最外层，
    先于 Origin，再是请求体的上限（api/limits.py）。这三项都不读请求体。"""
    app.add_middleware(BodyLimit, max_body_bytes=max_body_bytes, max_import_bytes=max_import_bytes)
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


def _check_space(who: Principal, space: str) -> str:
    """令牌的作用域之外的空间：403（空间是请求自己点名的，不涉及别人有没有数据）。"""
    if not who.allows(space):
        raise HTTPException(403, f"这个访问令牌不能访问记忆空间 {space}")
    return space


def _import_target(data: Mapping[str, Any], space: str | None) -> str | None:
    """导入写进哪个空间：与 import_space 的取法相同（参数优先，否则是文件里的 space_id）。"""
    return space or (str(data["space_id"]) if "space_id" in data else None)


def _allowed_space(space: str, http: Request) -> str:
    """/api/spaces/{space}/… 的路径参数，先过作用域检查（路由的依赖，在 guard 之后执行）。"""
    return _check_space(principal_of(http), space)


AllowedSpace = Annotated[str, Depends(_allowed_space)]


SWEEP_INTERVAL_S = 60.0  # 清理线程多久查一次闲置的会话与过期的 trace（见 _sweep_jobs）


def _sweep(stop: threading.Event, interval_s: float, *jobs: Callable[[], object]) -> None:
    """清理线程的循环：先依次做一遍 jobs（结束闲置的会话、删过期的 trace），之后每 interval_s
    秒一遍，直到 stop。一个出错不影响别的、也不让线程退出（记一条警告）。以前先等 interval_s
    秒再做第一遍：过期的 trace 要在启动时就删。"""
    while True:
        for job in jobs:
            try:
                job()
            except Exception:
                logger.warning("清理线程（结束闲置的会话、删过期的 trace）出错", exc_info=True)
        if stop.wait(interval_s):
            return


def _sweep_jobs(system: MemorySystem, *idle: Callable[[], object]) -> list[Callable[[], object]]:
    """清理线程要做的事：结束闲置的会话（idle），设了 traces_retention_days 时再加上删过期的
    trace（启动时与之后每天一次，见 MemorySystem.expire_traces）。cuemem serve 与 cuemem proxy
    共用。"""
    jobs = list(idle)
    if system.settings.traces_retention_days is not None:
        jobs.append(system.expire_traces)
    return jobs


def create_app(
    service: ChatService,
    *,
    upload_dir: Path,
    ui: bool = True,
    token: str | None = None,
    keys: Keys | None = None,
    proxy: MemoryProxy | None = None,
    allowed_hosts: Sequence[str] | None = None,
    origin_hosts: Sequence[str] = (),
    idle_minutes: float | None = 30.0,
    max_body_bytes: int = MAX_BODY_BYTES,
    max_import_bytes: int = MAX_IMPORT_BYTES,
) -> FastAPI:
    """token 非空时启用访问令牌：接口要带 Bearer 令牌，界面要先登录。keys：管理令牌之外还有限定
    作用域的令牌（Keys.from_env，见 api/auth.py；给了时不看 token）：每个令牌只能访问自己作用域
    里的空间，/api、/api/v1、界面与记忆代理都检查。

    proxy：同时提供 OpenAI 兼容的记忆代理（/v1/chat/completions，N12），与界面共用一个记忆系统，
    经代理写入的对话可以在界面里浏览。

    allowed_hosts：只接受这些 Host（接口与界面；None 不检查）。origin_hosts：除同源外放行的
    Origin 主机名（"*" 不限）。见模块说明。

    max_body_bytes、max_import_bytes：请求体的上限（字节，超过时 413；导入用后者，/api、/v1 与
    界面的上传用前者，界面别的接口固定 8 MiB，见 api/limits.py）。cuemem serve 从环境变量
    CUEMEM_MAX_BODY_BYTES、CUEMEM_MAX_IMPORT_BYTES 读（limits.body_limits）。

    upload_dir：上传的图片只在请求期间放在这里，写进记忆之后就删（媒体库里另存一份）。建应用时
    先清掉里面留下的图片：以前的版本一直留着上传的原图，遗忘之后磁盘上还有一份（D-021）；
    会话不跨进程，这时没有会话还在读它们。

    idle_minutes：闲置这么久的会话结束、交给后台抽取（None 不结束）。应用运行期间（lifespan）
    一个清理线程每分钟查一遍：/api 与界面的会话（ChatService.close_idle）、/api/v1 的 run
    （Memory.close_idle_runs）与记忆代理的会话（MemoryProxy.close_idle，按它自己的 idle_minutes）。
    以前关掉的标签页、崩溃的客户端留下的会话一直开着，服务运行期间一直不抽取；与记忆代理和各个
    框架集成一样，默认 30 分钟。设了 traces_retention_days 时这个线程还删过期的 trace（启动时与
    之后每天一次，见 _sweep_jobs），idle_minutes 为 None 时也起。
    """
    _clear_uploads(upload_dir)
    v1 = Memory.from_system(service.system)  # /api/v1 的 run 由清理线程结束闲置的

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        idle: list[Callable[[], object]] = []
        if idle_minutes is not None:
            service.idle_minutes = idle_minutes  # 404 与界面的说明里写上
            idle_s = idle_minutes * 60
            idle += [
                lambda: service.close_idle(idle_s),
                lambda: v1.close_idle_runs(idle_s),
            ]
            # cuemem serve 里的记忆代理：以前要等下一个 /v1 请求才结束闲置的会话
            if proxy is not None:
                idle.append(proxy.close_idle)
        jobs = _sweep_jobs(service.system, *idle)  # 只设了 traces_retention_days 时也要起线程
        if not jobs:
            yield
            return
        stop = threading.Event()
        args = (stop, SWEEP_INTERVAL_S, *jobs)
        sweeper = threading.Thread(target=_sweep, args=args, name="cuemem-api-sweep", daemon=True)
        sweeper.start()
        try:
            yield
        finally:  # 退出时还开着的会话由 cuemem serve 结束（ChatService.close、close_all）
            stop.set()
            sweeper.join(timeout=5)

    app = FastAPI(
        title="CueMem",
        version=__version__,
        description="Long-term multimodal memory for LLM agents: chat, browse, forget, recall.",
        lifespan=lifespan,
    )
    keys = keys if keys is not None else Keys.from_token(token)
    guarded = guard(keys)  # 认出令牌（principal_of）；没有配令牌时不鉴权
    checked = authenticated_route(keys)  # 令牌不对时在读请求体之前就 401
    api = APIRouter(dependencies=guarded, route_class=checked)

    def own_session(http: Request, session_id: str) -> None:
        """会话属于作用域之外的空间时，与不存在的会话一样 404（会话 ID 不是探测的途径）。"""
        try:
            space = service.get(session_id).space_id
        except KeyError:
            return  # 由服务层照常报 404
        if not principal_of(http).allows(space):
            raise HTTPException(404, f"没有进行中的会话 {session_id}")

    @app.get("/api/health")  # 不鉴权：给负载均衡与监控探活用
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "open_sessions": len(service.sessions)}

    @api.get("/api/spaces")
    def spaces(http: Request) -> list[dict[str, Any]]:
        who = principal_of(http)
        return [row for row in service.spaces() if who.allows(row["space_id"])]

    @api.post("/api/sessions")
    def start(http: Request, request: StartRequest) -> SessionOut:
        _check_space(principal_of(http), request.space or service.default_space)
        session = service.start(request.space, at=request.at)
        return SessionOut(
            session_id=session.session_id, space=session.space_id, started_at=session.started_at
        )

    @api.post("/api/sessions/{session_id}/messages")
    def message(
        http: Request,
        session_id: str,
        text: Annotated[str, Form()],
        images: Annotated[list[UploadFile] | None, File()] = None,
        mode: Annotated[str, Form()] = "auto",
        router: Annotated[str | None, Form()] = None,
        lam: Annotated[float | None, Form()] = None,
    ) -> Reply:
        own_session(http, session_id)  # 在存下上传的图片之前
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
    def end(http: Request, session_id: str, wait: bool = False) -> dict[str, Any]:
        own_session(http, session_id)
        with _errors():
            return service.end(session_id, wait=wait)

    # 空间 ID 里可以有 "/"（sdk.space_for：bot/alice、agent/shop），所以用 {space:path}；路径里的
    # %2F 在路由之前已经解成 "/"。AllowedSpace 检查令牌的作用域（之外 403）。
    # media 列表排在 media/{media_id} 前面：…/x/media/media 是空间 x/media 的列表（图片 ID 是哈希）
    @api.post("/api/spaces/{space:path}/recall")
    def recall(space: AllowedSpace, request: RecallRequest) -> dict[str, Any]:
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
    def profile(space: AllowedSpace) -> dict[str, str]:
        return {"text": service.profile(space)}

    @api.get("/api/spaces/{space:path}/facts")
    def facts(space: AllowedSpace, current_only: bool = False) -> list[dict[str, Any]]:
        return service.facts(space, current_only=current_only)

    @api.get("/api/spaces/{space:path}/entities")
    def entities(space: AllowedSpace) -> list[dict[str, Any]]:
        return service.entities(space)

    @api.get("/api/spaces/{space:path}/graph")
    def graph(space: AllowedSpace) -> dict[str, Any]:
        return service.graph(space)

    @api.get("/api/spaces/{space:path}/media")
    def media(space: AllowedSpace) -> list[dict[str, Any]]:
        return service.media(space)

    @api.get("/api/spaces/{space:path}/media/{media_id}")
    def media_file(space: AllowedSpace, media_id: str) -> FileResponse:
        items = service.system.db.get_media(space, ids=[media_id])
        if not items or not Path(items[0].path).exists():
            raise HTTPException(404, "没有这张图片")
        return FileResponse(items[0].path)

    @api.get("/api/spaces/{space:path}/sessions")
    def sessions(space: AllowedSpace) -> list[dict[str, Any]]:
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
    def status(space: AllowedSpace) -> dict[str, Any]:
        return service.status(space)

    @api.post("/api/spaces/{space:path}/forget")
    def forget(space: AllowedSpace, request: ForgetRequest) -> dict[str, Any]:
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
        out: dict[str, Any] = {
            "turns_deleted": report.turns_deleted,
            "sessions_affected": report.sessions_affected,
            "facts_before": report.facts_before,
            "facts_after": report.facts_after,
            "images_deleted": report.images_deleted,
        }
        if report.rebuild_error:  # 删除已经做了：不是出错，以前回 402 之类、调用方以为没删
            out["warning"] = (
                f"已经删除；用剩下的日志重建派生记忆时出错（{report.rebuild_error}），"
                "下次巩固时补上，在那之前事实可能不全"
            )
        return out

    @api.get("/api/spaces/{space:path}/export")
    def export(space: AllowedSpace, derived: bool = True) -> dict[str, Any]:
        """导出记忆空间（N12）：原始日志、图片原文件与派生记忆的快照。"""
        with _errors():
            return export_space(service.system, space, derived=derived)

    @api.post("/api/spaces/import")
    def import_(http: Request, data: dict[str, Any], space: str | None = None) -> dict[str, Any]:
        """导入 export 的结果（N12）：写回原始日志，派生记忆在后台重新抽取。写进作用域之外的
        空间（参数，或文件里的 space_id）时 403，什么都不写。"""
        target = _import_target(data, space)
        if not principal_of(http).allows(target):
            raise HTTPException(403, f"这个访问令牌不能导入到记忆空间 {target}")
        with _errors():
            report = import_space(service.system, data, space_id=space)
        if service.system.worker is not None:
            service.system.worker.submit(report["space_id"])
        return report

    @api.post("/api/spaces/{space:path}/consolidate")
    def consolidate(space: AllowedSpace) -> dict[str, Any]:
        with _errors():
            return service.consolidate(space)

    app.include_router(api)
    # Mem0 形状的记忆接口（/api/v1）：与界面、记忆代理共用同一个记忆系统，鉴权与作用域相同
    app.include_router(memories_router(v1, dependencies=guarded, route_class=checked))
    install_error_format(app)
    if proxy is not None:
        from cuemem.api.proxy import proxy_router

        app.include_router(proxy_router(proxy, dependencies=guarded))
    if ui and importlib.util.find_spec("gradio") is None:
        # 界面是可选依赖 cuemem[ui]：没装时照常提供接口与记忆代理
        logger.warning("演示界面要用可选依赖 ui（Gradio），没有安装：%s", extra_remedy("ui"))
        ui = False
    if ui:
        import gradio as gr

        from cuemem.api.ui import CSS, THEME, build_ui

        demo = build_ui(service, access=keys.by_username)  # 界面里登录的是哪个令牌
        app = gr.mount_gradio_app(
            app,
            demo,
            path="/",
            theme=THEME,
            css=CSS,
            allowed_paths=[str(service.system.settings.paths.media)],  # 界面不读上传目录
            max_file_size=max_body_bytes,  # 上传的图片与 /api 同一个上限（页面里先查一遍）
            footer_links=[],  # 接口文档在 /docs
            auth=ui_login(keys),
            auth_message=(
                "Sign in with the access token as the password: any username for the admin "
                "token, the key's name for a scoped key."
            ),
        )
        # Gradio 只在 Blocks.launch() 之后检查请求里的文件路径（FileData.path 要是经它上传的）；
        # 挂载的界面从不 launch。不打开时，登录的用户（限定作用域的也行）点名服务器上的任何文件，
        # Gradio 都在事件函数之前把它复制进缓存，再经 /gradio_api/file= 交出去。不在 ui.py 的事件
        # 函数里查：它拿到的已经是缓存里的副本，聊天记录这个输入也不经过它
        demo.has_launched = True
    # 中间件：落到界面的预检请求也先过这里；请求体的上限也管界面的登录与事件接口
    _guard(
        app,
        allowed_hosts,
        origin_hosts,
        max_body_bytes=max_body_bytes,
        max_import_bytes=max_import_bytes,
    )
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
