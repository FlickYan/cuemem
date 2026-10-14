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
4. **写回**：用户的话（含图片）与回答在线写入原始日志（不调用 LLM）。会话进行中不抽取事实，
   空闲超过 idle_minutes 后整个会话交给后台抽取（与 Agent 的做法相同，D-045）。

身份：请求体里的 user 字段，或请求头 X-CueMem-User（可加 X-CueMem-Agent）；会话：请求头
X-CueMem-Session，不给时自动切分：请求里没有之前的消息（调用方开始了新的一段对话）或者空闲
超过 idle_minutes 时开新的会话。并行的多段对话要各带自己的会话头，否则会混进同一个会话。
都没有用户、又没配默认用户时，请求原样转发、不碰记忆。
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any

import httpx2
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from cuemem import __version__
from cuemem.config import ModelSpec
from cuemem.core.trace import record_event
from cuemem.evidence import MEMORY_NOTE as MEMORY_NOTE  # 兼容旧的导入位置（集成改完后删掉）
from cuemem.evidence import inject_evidence, message_text, recall_evidence
from cuemem.memory import MemorySystem, RecallMode
from cuemem.sdk import Memory, space_for

_inject = inject_evidence  # 兼容旧的导入位置（集成改完后删掉）
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
        """非流式：返回（状态码，响应体，附加响应头）。上游出错时原样返回、不写记忆。"""
        turn, upstream, url, auth = self._prepare(body, headers)
        response = self._http.post(url, json=upstream, headers=auth)
        data = _json(response)
        if response.status_code == 200:
            choices = data.get("choices") or [{}]
            message = choices[0].get("message") or {}
            self._write_back(turn, message.get("content"))
        return response.status_code, data, _headers(turn)

    def stream(
        self, body: dict[str, Any], headers: Mapping[str, str]
    ) -> tuple[int, Iterator[bytes], dict[str, str]]:
        """流式：先拿到上游的状态码，再把 SSE 原样转发；流结束后写回回答。"""
        turn, upstream, url, auth = self._prepare(body, headers)
        request = self._http.build_request("POST", url, json=upstream, headers=auth)
        response = self._http.send(request, stream=True)
        if response.status_code != 200:
            content = response.read()
            response.close()
            return response.status_code, iter([content]), _headers(turn)

        def relay() -> Iterator[bytes]:
            pieces: list[str] = []
            try:
                for line in response.iter_lines():
                    yield (line + "\n").encode()
                    if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                        pieces.append(_deltamessage_text(line[5:]))
            finally:
                response.close()
                self._write_back(turn, "".join(pieces))

        return 200, relay(), _headers(turn)

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
                sid = explicit or f"proxy-{when:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
                run = Run(space, sid, now)
                self._runs[key] = run
                self.system.open_session(space, sid)  # 进行中：每轮照常写入，但不抽取
            run.last_seen = now
        if ended is not None:
            self.system.close_session(ended.space, ended.session_id)
        return run

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
        """当前会话里已经原样在请求中的轮（检索时排除，名额留给请求里没有的）。按内容比对：
        调用方可能删减了历史，或者在同一会话里开了别的对话，不能假定写进去的都还在。"""
        present = {message_text(m.get("content")).strip() for m in earlier}
        rounds = self.system.db.list_units(space, kinds=["round"], session_ids=[session_id])
        turns = {t.id: t for t in self.system.db.get_turns(space, session_id)}
        return [
            u.id
            for u in rounds
            if all(t in turns and turns[t].text.strip() in present for t in u.source_turn_ids)
        ]

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


def _deltamessage_text(payload: str) -> str:
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return ""
    choices = chunk.get("choices") or []
    delta = choices[0].get("delta") if choices else None
    return (delta or {}).get("content") or ""


def _json(response: httpx2.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {"error": {"message": response.text[:500], "type": "upstream_error"}}
    return data if isinstance(data, dict) else {"data": data}


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
            return StreamingResponse(chunks, headers=extra, media_type="text/event-stream")
        status, data, extra = proxy.complete(body, headers)
        return JSONResponse(data, status, extra)

    return router


async def _body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "The request body must be JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "The request body must be a JSON object.")
    return body


def create_proxy_app(proxy: MemoryProxy, *, token: str | None = None) -> FastAPI:
    """单独运行的记忆代理（cuemem proxy）。token 非空时要求 Authorization: Bearer <token>：
    OpenAI SDK 的 api_key 填这个令牌即可（上游的 Key 由 CueMem 的配置提供）。"""
    from cuemem.api.app import _token_check

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
    return app
