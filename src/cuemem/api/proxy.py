"""OpenAI 兼容的记忆代理（N12，D-062）：任何用 OpenAI SDK 的应用，把 base_url 指过来就有了长期记忆。

    client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="<CUEMEM_API_TOKEN 或任意值>")
    client.chat.completions.create(model="gpt-5.4-mini", messages=[...], user="alice")

对标 Mem0 的 proxy（OpenAI 兼容的包装）与 Supermemory 的 Memory Router（改 base URL 的透明代理）。
它们每一轮都检索；这里每一轮先过检索控制器（N7）：

1. **想起**：对最后一句用户消息跑控制器，选 L0（不检索，只给画像）/ L1 / L2；当前会话里
   已经写进记忆、并且原样还在请求里的轮，检索时排除；
2. **注入**：取到的画像、带有效期的事实与对话片段排成和 reader 相同的证据块，放在最后一句
   用户消息的前面（与 Mem0 的 proxy 相同的位置）；系统提示与之前的消息不动，
   供应商的前缀缓存照常命中；
3. **转发**：按配置里的模型目录转发给上游（model 是配置里的别名或模型 ID，
   否则用 agent 角色的模型），流式与非流式都支持，响应原样返回；
   控制器的决定放在响应头 X-CueMem-* 里；
4. **写回**：用户的话（含图片）与回答在线写入原始日志（不调用 LLM）；上游出错时不写，流式
   回答中途断开（客户端停下）只写用户的话。会话进行中不抽取事实，
   空闲超过 idle_minutes 后整个会话交给后台抽取（与 Agent 的做法相同，D-045）。

身份：请求体里的 user 字段，或请求头 X-CueMem-User（可加 X-CueMem-Agent）；会话：请求头
X-CueMem-Session，不给时自动切分：请求里没有之前的消息（调用方开始了新的一段对话）或者空闲
超过 idle_minutes 时开新的会话。并行的多段对话要各带自己的会话头，否则会混进同一个会话。
都没有用户、又没配默认用户时，请求原样转发、不碰记忆。

请求体必须是 Content-Type: application/json，别的网站发来的浏览器请求（Origin 与 Host 不同）一律
403（api.app.SameOriginOnly）：否则任何网页都能借访问者的浏览器往本机的代理写记忆、花上游的额度。
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any

import anyio
import anyio.to_thread
import httpx2
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from cuemem import __version__
from cuemem.config import ModelSpec
from cuemem.core.ids import sortable_session_id
from cuemem.core.trace import record_event
from cuemem.evidence import (
    in_context_rounds,
    inject_evidence,
    message_text,
    recall_evidence,
    stored_text,
)
from cuemem.memory import MemorySystem, RecallMode
from cuemem.sdk import Memory, space_for

_SESSION_ID = re.compile(r"[^A-Za-z0-9_.:-]")


@dataclass
class Run:
    """进行中的会话（一段对话）。"""

    space: str
    session_id: str
    last_seen: float
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Turn:
    """一次请求在记忆侧的准备结果。"""

    space: str | None  # None：不碰记忆，原样转发
    user_id: str | None = None
    agent_id: str | None = None
    run: Run | None = None
    write_user: dict[str, Any] | None = None  # 要写回的用户消息（OpenAI 格式）
    level: str = "none"
    reasons: list[str] = field(default_factory=list)
    recalled: int = 0  # 注入的记忆条数（画像、事实、对话片段）
    block: str = ""  # 注入的证据块


class MemoryProxy:
    def __init__(
        self,
        system: MemorySystem,
        *,
        default_user: str | None = None,
        role: str = "agent",
        mode: RecallMode = "auto",
        idle_minutes: float = 30.0,
        http_client: httpx2.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.system = system
        self.memory = Memory.from_system(system)
        self.default_user = default_user
        self.role = role
        self.mode: RecallMode = mode
        self.idle_s = idle_minutes * 60
        self._http = http_client or httpx2.Client(timeout=300.0)
        self._clock = clock
        self._runs: dict[tuple[str, str | None], Run] = {}
        self._adopted: set[str] = set()  # 接手过崩溃前留下的会话的空间
        self._lock = threading.Lock()
        system.start_worker()  # 会话结束后的抽取在后台完成

    # ---------- 请求 ----------

    def models(self) -> dict[str, Any]:
        data = [
            {"id": alias, "object": "model", "created": 0, "owned_by": spec.provider}
            for alias, spec in self.system.settings.models.items()
        ]
        return {"object": "list", "data": data}

    def complete(
        self, body: dict[str, Any], headers: Mapping[str, str]
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        """非流式：返回（状态码，响应体，附加响应头）。上游出错时原样返回（连同它的
        Retry-After）、不写记忆；连不上或超时时回 OpenAI 格式的 502 / 504。"""
        turn, upstream, url, auth = self._prepare(body, headers)
        try:
            response = self._http.post(url, json=upstream, headers=auth)
        except httpx2.TransportError as exc:
            status, error = _unreachable(exc)
            return status, error, _headers(turn)
        data = _json(response)
        if response.status_code == 200:
            choices = data.get("choices") or [{}]
            message = choices[0].get("message") or {}
            self._write_back(turn, message.get("content"))
            return 200, data, _headers(turn)
        return response.status_code, data, {**_headers(turn), **_retry_headers(response)}

    def stream(
        self, body: dict[str, Any], headers: Mapping[str, str]
    ) -> tuple[int, Iterator[bytes], dict[str, str]]:
        """流式：先拿到上游的状态码，再把 SSE 原样转发；流读完后写回回答。

        状态码是 200 时返回的迭代器（_Relay）要么读完，要么调用 close()：它占着上游的连接，
        中途不要的（客户端断开、用户停下）只写回用户的话；上游中途出错时什么也不写。
        上游一开始就出错时与非流式相同。"""
        turn, upstream, url, auth = self._prepare(body, headers)
        request = self._http.build_request("POST", url, json=upstream, headers=auth)
        try:
            response = self._http.send(request, stream=True)
        except httpx2.TransportError as exc:
            status, error = _unreachable(exc)
            return status, iter([json.dumps(error).encode()]), _headers(turn)
        if response.status_code != 200:
            content = response.read()
            response.close()
            extra = {**_headers(turn), **_retry_headers(response)}
            return response.status_code, iter([content]), extra
        return 200, _Relay(self, turn, response), _headers(turn)

    # ---------- 会话 ----------

    def close_idle(self) -> int:
        """空闲超过 idle_minutes 的会话交给后台抽取；返回关掉的会话数。"""
        now = self._clock()
        with self._lock:
            stale = [k for k, r in self._runs.items() if now - r.last_seen >= self.idle_s]
            runs = [self._runs.pop(k) for k in stale]
        for run in runs:
            self.system.close_session(run.space, run.session_id)
        return len(runs)

    def open_runs(self) -> int:
        with self._lock:
            return len(self._runs)

    def close_all(self) -> None:
        with self._lock:
            runs = list(self._runs.values())
            self._runs.clear()
        for run in runs:
            self.system.close_session(run.space, run.session_id)

    def _run(self, space: str, explicit: str | None, *, fresh: bool = False) -> Run:
        """这次请求属于哪个会话。fresh：请求里没有之前的对话（调用方开始了新的一段），
        没指定会话时结束上一段、开新的一段——会话的边界跟着调用方的对话走。"""
        now = self._clock()
        self.close_idle()
        self._adopt(space, explicit)
        key = (space, explicit)
        with self._lock:
            run = self._runs.get(key)
            if run is not None and fresh and explicit is None:
                self._runs.pop(key)
                ended: Run | None = run
                run = None
            else:
                ended = None
            if run is None:
                when = datetime.now()
                sid = explicit or sortable_session_id("proxy", when)
                run = Run(space, sid, now)
                self._runs[key] = run
                self.system.open_session(space, sid)  # 进行中：每轮照常写入，但不抽取
            run.last_seen = now
        if ended is not None:
            self.system.close_session(ended.space, ended.session_id)
        return run

    def _adopt(self, space: str, explicit: str | None) -> None:
        """这个进程第一次碰到这个空间：之前的进程崩溃时还在进行中的会话（正常退出时 close_all
        已经交出去了）交给后台抽取。调用方指定的会话除外：它马上要接着写。"""
        with self._lock:
            if space in self._adopted:
                return
            self._adopted.add(space)
        self.system.adopt_open(space, keep=[explicit] if explicit else [])

    # ---------- 内部 ----------

    def _prepare(
        self, body: dict[str, Any], headers: Mapping[str, str]
    ) -> tuple[Turn, dict[str, Any], str, dict[str, str]]:
        turn = self._recall(body, headers)
        upstream, url, auth = self._upstream(body)
        if turn.write_user is not None and turn.block:
            upstream["messages"] = inject_evidence(upstream["messages"], turn.block)
        return turn, upstream, url, auth

    def _recall(self, body: dict[str, Any], headers: Mapping[str, str]) -> Turn:
        user_id = headers.get("x-cuemem-user") or body.get("user") or self.default_user
        agent_id = headers.get("x-cuemem-agent") or None
        messages = body.get("messages") or []
        if user_id is None or not messages:
            return Turn(space=None)
        space = space_for(user_id, agent_id)
        explicit = headers.get("x-cuemem-session")
        session = _SESSION_ID.sub("_", explicit)[:80] if explicit else None
        earlier = [m for m in messages[:-1] if m.get("role") in ("user", "assistant")]
        run = self._run(space, session, fresh=not earlier)
        turn = Turn(space=space, user_id=user_id, agent_id=agent_id, run=run)
        last = messages[-1]
        if last.get("role") != "user":  # 工具调用的后续请求：只写回回答
            return turn
        turn.write_user = last
        query = message_text(last.get("content"))
        if not query.strip() or not self.system.db.get_sessions(space):
            return turn
        try:
            evidence = recall_evidence(
                self.system,
                space,
                query,
                mode=self.mode,
                exclude=self._in_context(space, run.session_id, earlier),
            )
        except Exception as exc:  # 想不起来不该让这一轮失败：不注入记忆，照常转发
            record_event("proxy_recall_failed", space=space, error=f"{type(exc).__name__}: {exc}")
            turn.level = "error"
            return turn
        turn.level, turn.reasons = evidence.level, evidence.reasons
        turn.recalled, turn.block = evidence.recalled, evidence.block
        record_event("proxy_recall", space=space, level=turn.level, recalled=turn.recalled)
        return turn

    def _in_context(self, space: str, session_id: str, earlier: list[dict[str, Any]]) -> list[str]:
        """当前会话里已经原样在请求中的轮（检索时排除，名额留给请求里没有的）。之前的每条消息
        按写回时的样子比对：先过 _storable（只有 data URL 的图片会存下、记作 <image>）。"""
        present = {stored_text(_storable(m)).strip() for m in earlier}
        return in_context_rounds(self.system, space, [session_id], present)

    def _upstream(self, body: dict[str, Any]) -> tuple[dict[str, Any], str, dict[str, str]]:
        """请求体 → 上游的请求体、地址与鉴权头。model 是配置里的别名或模型 ID，否则用默认模型。"""
        settings = self.system.settings
        requested = body.get("model")
        alias = requested if requested in settings.models else None
        if alias is None:
            alias = next(
                (a for a, s in settings.models.items() if s.model_id == requested),
                settings.roles[self.role].model,
            )
        spec = settings.models[alias]
        provider = settings.providers[spec.provider]
        upstream = {k: v for k, v in body.items() if k != "user"}
        upstream["model"] = spec.model_id
        _adapt(upstream, spec)
        key = provider.api_key()
        auth = {"Authorization": f"Bearer {key}"} if key else {}
        return upstream, provider.base_url.rstrip("/") + "/chat/completions", auth

    def _write_back(self, turn: Turn, answer: str | None) -> None:
        if turn.space is None or turn.run is None:
            return
        messages = [_storable(turn.write_user)] if turn.write_user is not None else []
        if answer and answer.strip():
            messages.append({"role": "assistant", "content": answer})
        if not messages:
            return
        try:
            with turn.run.lock:  # 同一会话的写回按顺序进行（发言序号只增不减）
                self.memory.add(
                    messages,
                    user_id=turn.user_id,
                    agent_id=turn.agent_id,
                    run_id=turn.run.session_id,
                )
        except Exception as exc:  # 写回失败不影响已经返回给调用方的回答
            record_event("proxy_write_failed", error=f"{type(exc).__name__}: {exc}")


class _Relay(Iterator[bytes]):
    """把上游的 SSE 一行行原样转发；流怎么结束决定写回什么（D-062：上游出错不写记忆）。

    - 读完（done）：写回用户的话与回答。不要求 [DONE] 或 finish_reason：有的上游不发
      [DONE] 就关流，finish_reason 为 length 的回答也是完整的；
    - 上游中途出错（failed，含流里的 {"error": ...}）：什么也不写，与非流式的出错相同；
    - 调用方中途 close()（stopped，客户端断开、用户停下）：只写回用户的话。半截的回答不算
      回答（调用方下一次请求的历史里有它）；用户的话却要写：之后的请求只写各自最后一句
      用户消息，这句不写就永远丢了。

    写成类而不是生成器：还没开始读就 close() 的生成器不执行收尾，上游的连接一直占着。"""

    def __init__(self, proxy: MemoryProxy, turn: Turn, response: httpx2.Response) -> None:
        self._proxy = proxy
        self._turn = turn
        self._response = response
        self._lines = response.iter_lines()
        self._pieces: list[str] = []
        self._error = False
        self._ended: str | None = None

    def __next__(self) -> bytes:
        if self._ended is not None:
            raise StopIteration
        try:
            line = next(self._lines)
        except StopIteration:
            self._end("failed" if self._error else "done")
            raise
        except BaseException:
            self._end("failed")
            raise
        payload = line[5:].strip() if line.startswith("data:") else ""
        if payload and payload != "[DONE]":
            chunk = _sse_json(payload)
            self._error = self._error or bool(chunk.get("error"))
            self._pieces.append(_delta_text(chunk))
        return (line + "\n").encode()

    def close(self) -> None:
        self._end("stopped")

    def _end(self, ended: str) -> None:
        if self._ended is not None:
            return
        self._ended = ended
        self._response.close()
        answer = "".join(self._pieces)
        if ended == "done":
            self._proxy._write_back(self._turn, answer)
            return
        record_event(
            "proxy_stream_incomplete", space=self._turn.space, ended=ended, chars=len(answer)
        )
        if ended == "stopped":
            self._proxy._write_back(self._turn, None)


class _RelayResponse(StreamingResponse):
    """流式响应一结束（读完、出错、客户端断开）就关掉 _Relay。

    Starlette 在客户端断开时不关同步迭代器，收尾要等垃圾回收：期间占着上游的连接，写回落在
    任意线程上、排在下一轮之后，在同一线程上还会等自己持有的会话锁，死锁。"""

    def __init__(self, chunks: Iterator[bytes], **kwargs: Any) -> None:
        super().__init__(chunks, **kwargs)
        self._chunks = chunks

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self._chunks, "close", None)
            if close is not None:  # 写回要访问数据库与嵌入：放到工作线程，取消时也做完
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(close)


def _adapt(upstream: dict[str, Any], spec: ModelSpec) -> None:
    """按模型目录补上供应商特有的参数（例如关掉思考模式），并处理推理模型的参数差异。"""
    for key, value in spec.extra_body.items():
        upstream.setdefault(key, value)
    if spec.fixed_temperature:
        upstream.pop("temperature", None)
        upstream.pop("top_p", None)
    if spec.max_tokens_param == "max_completion_tokens" and "max_tokens" in upstream:
        upstream.setdefault("max_completion_tokens", upstream.pop("max_tokens"))


def _storable(message: dict[str, Any]) -> dict[str, Any]:
    """写回记忆的用户消息：图片只存请求里自带的 data URL，其余原样。

    网络图片不下载；本地路径与 file:// 一律不收——它们指向的是代理所在机器上的文件，
    远程调用方不能借此把服务器上的文件写进自己的记忆。"""
    content = message.get("content")
    if not isinstance(content, list):
        return message
    kept = [
        block
        for block in content
        if block.get("type") != "image_url" or _image_url(block).startswith("data:")
    ]
    return {**message, "content": kept}


def _image_url(block: dict[str, Any]) -> str:
    value = block.get("image_url")
    return str(value.get("url", "") if isinstance(value, dict) else value or "")


def _sse_json(payload: str) -> dict[str, Any]:
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return chunk if isinstance(chunk, dict) else {}


def _delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    delta = choices[0].get("delta") if choices else None
    return (delta or {}).get("content") or ""


def _json(response: httpx2.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {"error": {"message": response.text[:500], "type": "upstream_error"}}
    return data if isinstance(data, dict) else {"data": data}


def _unreachable(exc: httpx2.TransportError) -> tuple[int, dict[str, Any]]:
    """上游连不上或超时：OpenAI 格式的 502 / 504（以前是没有说明的 500）。"""
    if isinstance(exc, httpx2.TimeoutException):
        message = f"The upstream model provider did not respond in time ({type(exc).__name__})."
        return 504, {"error": {"message": message, "type": "timeout", "code": "upstream_timeout"}}
    message = f"Could not reach the upstream model provider ({type(exc).__name__}: {exc})."
    return 502, {"error": {"message": message, "type": "upstream_error", "code": "upstream_error"}}


def _retry_headers(response: httpx2.Response) -> dict[str, str]:
    """上游出错时转给调用方的重试提示：OpenAI SDK 照 retry-after(-ms) 等待、照 x-should-retry
    决定重不重试。以前只转发了响应体，限流时调用方不知道该等多久。"""
    names = ("retry-after", "retry-after-ms", "x-should-retry")
    return {name: response.headers[name] for name in names if name in response.headers}


def _headers(turn: Turn) -> dict[str, str]:
    if turn.space is None:
        return {"X-CueMem-Level": "none"}
    out = {
        "X-CueMem-Level": turn.level,
        "X-CueMem-Recalled": str(turn.recalled),
        "X-CueMem-Reasons": "; ".join(turn.reasons),
    }
    if turn.run is not None:
        out["X-CueMem-Session"] = turn.run.session_id
    # 响应头只能是 latin-1：理由里偶尔有别的字符
    return {k: v.encode("ascii", "replace").decode() for k, v in out.items()}


def proxy_router(proxy: MemoryProxy, *, dependencies: list[Any] | None = None) -> APIRouter:
    """/v1/models 与 /v1/chat/completions；可以挂在 cuemem serve 的应用上，也可以单独运行。"""
    router = APIRouter(dependencies=dependencies or [])

    @router.get("/v1/models")
    def models() -> dict[str, Any]:
        return proxy.models()

    @router.post("/v1/chat/completions")
    def completions(body: Annotated[dict[str, Any], Depends(_body)], request: Request) -> Response:
        headers = {k.lower(): v for k, v in request.headers.items()}
        if body.get("stream"):
            status, chunks, extra = proxy.stream(body, headers)
            if status != 200:
                return Response(b"".join(chunks), status, extra, media_type="application/json")
            return _RelayResponse(chunks, headers=extra, media_type="text/event-stream")
        status, data, extra = proxy.complete(body, headers)
        return JSONResponse(data, status, extra)

    return router


async def _body(request: Request) -> dict[str, Any]:
    # 只收 JSON：text/plain、表单与不带 Content-Type 的 POST 是浏览器不经预检就能跨站发的请求
    media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media != "application/json" and not (
        media.startswith("application/") and media.endswith("+json")
    ):
        raise HTTPException(415, "Content-Type must be application/json.")
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "The request body must be JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "The request body must be a JSON object.")
    return body


def create_proxy_app(
    proxy: MemoryProxy,
    *,
    token: str | None = None,
    allowed_hosts: Sequence[str] | None = None,
    origin_hosts: Sequence[str] = (),
) -> FastAPI:
    """单独运行的记忆代理（cuemem proxy）。token 非空时要求 Authorization: Bearer <token>：
    OpenAI SDK 的 api_key 填这个令牌即可（上游的 Key 由 CueMem 的配置提供）。

    不接受别的网站发来的浏览器请求；allowed_hosts、origin_hosts 同 api.app.create_app。"""
    from cuemem.api.app import _guard, _token_check

    stop = threading.Event()

    def sweep() -> None:  # 每分钟把空闲的会话交给后台抽取
        while not stop.wait(60):
            proxy.close_idle()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        threading.Thread(target=sweep, name="cuemem-proxy-sweep", daemon=True).start()
        try:
            yield
        finally:  # 退出时把进行中的会话都登记抽取（后台没做完的，下次巩固时完成）
            stop.set()
            proxy.close_all()

    app = FastAPI(
        title="CueMem memory proxy",
        version=__version__,
        description="OpenAI-compatible chat completions with long-term memory.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "open_runs": proxy.open_runs()}

    deps = [Depends(_token_check(token))] if token else []
    app.include_router(proxy_router(proxy, dependencies=deps))
    _guard(app, allowed_hosts, origin_hosts)
    return app
