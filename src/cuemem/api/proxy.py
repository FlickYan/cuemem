"""OpenAI 兼容的记忆代理（N12，D-062）：任何用 OpenAI SDK 的应用，把 base_url 指过来就有了长期记忆。

    client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="<CUEMEM_API_TOKEN 或任意值>")
    client.chat.completions.create(model="gpt-5.4-mini", messages=[...], user="alice")

对标 Mem0 的 proxy（OpenAI 兼容的包装）与 Supermemory 的 Memory Router（改 base URL 的透明代理）。
它们每一轮都检索；这里每一轮先过检索控制器（N7）：

1. **想起**：对最后一句用户消息跑控制器，选 L0（不检索，只给画像）/ L1 / L2；当前会话里
   已经写进记忆、并且原样还在请求里的轮，检索时排除。请求不是以用户消息结尾时（工具调用的
   后续请求、用户消息之后还有系统消息）不再检索一次：这个会话上一次为同一句用户消息想起的
   证据块原样再用（见 Kept），响应头照它写；没有留着的（重启、停放过的会话、用户消息改过）
   或者之后这个空间遗忘过时，对那句用户消息重新想起；
2. **注入**：取到的画像、带有效期的事实与对话片段排成和 reader 相同的证据块，放在最后一句
   用户消息的前面（与 Mem0 的 proxy 相同的位置）；系统提示与之前的消息不动，
   供应商的前缀缓存照常命中。Chat Completions 没有状态，工具调用的后续请求里调用方重发的
   用户消息不带证据块：照上一步再放进同一句用户消息，转发的请求到这句为止与第一次逐字节
   相同，一轮之内的工具循环前缀缓存接着命中（下一句用户消息的请求里，之前的用户消息照旧
   不带证据块）；
3. **转发**：按配置里的模型目录转发给上游（model 是配置里的别名或模型 ID；不认识的模型默认
   改用 agent 角色的模型并记一条警告，proxy.unknown_model: reject 时返回 400；目录里供应商的
   Key 没设置的模型返回 503 model_not_available，GET /v1/models 也不列它们），流式与非流式
   都支持，响应原样返回；控制器的决定与实际用的模型（X-CueMem-Model）放在响应头里；
4. **写回**：用户的话（含图片）与回答在线写入原始日志（不调用 LLM）；上游出错时不写，流式
   回答中途断开（客户端停下）只写用户的话。写不进去（向量接口暂时不可用等）时留着，这个会话
   下一次写回或结束之前再写（见 Run；一个会话最多留 MAX_UNSAVED 次、MAX_UNSAVED_BYTES，超出的
   丢掉最早的、记警告）。会话进行中不抽取事实，
   空闲超过 idle_minutes 后整个会话交给后台抽取（与 Agent 的做法相同，D-045）。

身份：请求体里的 user 字段，或请求头 X-CueMem-User（可加 X-CueMem-Agent；不是 ASCII 的 ID 写成
utf-8''%E5%BC%A0%E4%B8%89 这样的 UTF-8 百分号编码，见 auth.header_id，或者放进 user 字段）；
会话：请求头 X-CueMem-Session，不给时自动切分：请求里没有之前的消息（调用方开始了新的一段对话）
或者空闲超过 idle_minutes 时开新的会话。并行的多段对话要各带自己的会话头，否则会混进同一个会话。
一个用户（记忆空间）内存里同时进行中的会话最多 CUEMEM_MAX_OPEN_SESSIONS 个（默认 64）：再开新的时
先停放最久没有请求的那个（会话照旧进行中、不抽取，调用方回来时接着写，闲置太久时照常交给抽取；
整个进程最多停放 PARKED_KEPT 个，再多时最早停放的交给抽取）；都留着没写进去的写回时 429
（见 _make_room）。
都没有用户、又没配默认用户时，请求原样转发、不碰记忆。限定作用域的令牌（api/auth.py）只能用
作用域里的用户与 Agent（包括配置的默认用户）：之外的 403（OpenAI 格式），在检索、开会话与转发
之前。

作用域不是花费的上限：请求里的 model、n、max_tokens 与供应商的开关（enable_thinking 等）原样
转发。限定作用域的令牌可以另外写（第四轮体检，见 api/auth.py；不写时转发的请求逐字节不变）：
models（只能用这些模型，之外 403；目录里的 extra_body 盖过调用方给的；列出的别名只放行它自己，
不放行同一个模型 ID 的别的别名，见 _upstream）与 max_output_tokens（max_tokens、
max_completion_tokens 不超过它，并且总按这个模型在目录里的写法 max_tokens_param 写上一份；n 只能
是 1，否则 400）。都在检索、开会话与转发之前查。max_output_tokens 管不住调用方的开关
（enable_thinking、thinking），只有 models 管得住。

另外两个 OpenAI 兼容的接口：
- /v1/responses：Responses API 的一个子集——非流式、纯文字输入（字符串或 input_text 的消息），
  instructions、max_output_tokens、temperature、top_p。换成 Chat Completions 走上面同样的想起、
  注入、转发与写回，回答写成 Responses 的格式。流式、工具、图片与 previous_response_id 等返回
  501（OpenAI 格式的出错，带 x-should-retry: false），用 /v1/chat/completions；
- /v1/embeddings：交给配置里的向量模型（与记忆共用一个，不碰记忆、不进向量缓存），支持
  encoding_format 的 float 与 base64。

请求体必须是 Content-Type: application/json，别的网站发来的浏览器请求（Origin 与 Host 不同）一律
403（api.app.SameOriginOnly）：否则任何网页都能借访问者的浏览器往本机的代理写记忆、花上游的额度。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any

import anyio
import anyio.to_thread
import httpx2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from cuemem import __version__
from cuemem.api.auth import (
    AnyKeys,
    Keys,
    Principal,
    guard,
    identity_headers,
    principal_of,
    space_cap_message,
    spaces_over_cap,
)
from cuemem.api.failures import provider_failure
from cuemem.api.limits import MAX_BODY_BYTES, MAX_OPEN_SESSIONS_ENV, open_session_limit
from cuemem.config import ModelSpec, UnknownModel
from cuemem.core.ids import sortable_session_id
from cuemem.core.trace import record_event
from cuemem.embed import CachedEmbedder, TextEmbedder
from cuemem.errors import EmbeddingAPIError, VectorStoreFullError
from cuemem.evidence import (
    in_context_rounds,
    inject_evidence,
    message_text,
    payload_size,
    recall_evidence,
    stored_text,
)
from cuemem.memory import MemorySystem, RecallMode
from cuemem.sdk import Memory, space_for
from cuemem.write.online import worth_retrying

logger = logging.getLogger(__name__)

_SESSION_ID = re.compile(r"[^A-Za-z0-9_.:-]")
MAX_EMBED_INPUTS = 2048  # 与 OpenAI 的上限相同
# 一个会话写不进去、留着重试的写回最多这么多次、这么大（文字与图片 data URL 的字符数，
# evidence.payload_size）：长时间故障而调用方一直在说话时，超出的丢掉最早的，不让内存一直长
MAX_UNSAVED = 100
MAX_UNSAVED_BYTES = 32 * 1024 * 1024
# 为腾出名额而停放的会话（见 MemoryProxy._make_room）最多记这么多个（每个空间各自的
# CUEMEM_MAX_OPEN_SESSIONS 之外，整个进程合计）：再多时最早停放的那个照闲置结束交给抽取。
# 停放的会话照旧进行中（数据库里的进行中标记与 MemorySystem 的登记都留着），每个只占一个键与
# 一个时间
PARKED_KEPT = 4096


class ProxyError(Exception):
    """请求不能转发：状态码与 OpenAI 格式的出错（{"error": {...}}），不碰记忆、不调用上游。"""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        kind: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = {"error": {"message": message, "type": kind, "param": param, "code": code}}
        self.headers = dict(headers or {})

    def response(self) -> JSONResponse:
        return JSONResponse(self.body, self.status, self.headers)


@dataclass
class Unsaved:
    """没写进记忆的一次写回：消息、当时的时间（重试时照这个时间写），与当时这个空间删过几次发言
    （MemoryDB.turn_epoch：之后遗忘过就不再写，它们是遗忘之前说的）。"""

    messages: list[dict[str, Any]]
    when: datetime
    epoch: int
    size: int = 0  # evidence.payload_size(messages)


@dataclass
class Kept:
    """一个会话最近一次想起的结果，留给这一轮里不以用户消息结尾的请求（工具调用的后续）再用。

    digest：注入的那句用户消息写回时的样子（stored_text(_storable(...))，与 _in_context 的
    比对相同）的 SHA-256：调用方重发的还是这一句才再用。epoch：想起之前这个空间删过几次发言
    （MemoryDB.turn_epoch）：之后遗忘过就不再用，重新想起。block 为空的也留着（没有画像的 L0
    不用再想一遍）。证据块里的相对时间与它一起不变：再放进去的要与第一次逐字节相同。期间后台
    抽取出的新事实、刚过期的事实不会进来，最多留到会话结束（idle_minutes）。"""

    digest: str
    block: str
    level: str
    reasons: list[str]
    recalled: int
    epoch: int


@dataclass
class Run:
    """进行中的会话（一段对话）。

    unsaved：写回失败（向量接口暂时不可用等，见 write.online.worth_retrying）、留着重试的写回，
    按先后排。之后每次写回这个会话时先写它们，会话结束（空闲、开了新的一段、close_all）之前
    再试一次，还是写不进去才丢掉、记警告。在线写入要么全写、要么不写（R1），以前丢下它们就永远
    丢了：之后的请求只写各自最后一句用户消息。最多留 MAX_UNSAVED 次、MAX_UNSAVED_BYTES
    （unsaved_bytes 是它们的 size 之和），超出时丢掉最早的、记警告（MemoryProxy._trim）。

    kept：最近一次想起的结果（见 Kept），一个会话只留一份，随 Run 一起没有（空闲结束、停放、
    调用方开了新的一段、重启）。"""

    space: str
    session_id: str
    last_seen: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    user_id: str | None = None
    agent_id: str | None = None
    unsaved: list[Unsaved] = field(default_factory=list)
    unsaved_bytes: int = 0
    kept: Kept | None = None

    def keep(self, unsaved: Unsaved) -> None:
        self.unsaved.append(unsaved)
        self.unsaved_bytes += unsaved.size

    def drop(self, count: int = 1) -> None:
        """丢掉最早的 count 次写回。"""
        self.unsaved_bytes -= sum(u.size for u in self.unsaved[:count])
        del self.unsaved[:count]


@dataclass
class Turn:
    """一次请求在记忆侧的准备结果。"""

    space: str | None  # None：不碰记忆，原样转发
    user_id: str | None = None
    agent_id: str | None = None
    run: Run | None = None
    write_user: dict[str, Any] | None = None  # 要写回的用户消息（OpenAI 格式）
    anchor: int | None = None  # 最后一句用户消息在 messages 里的位置：证据块放进这一句
    level: str = "none"
    reasons: list[str] = field(default_factory=list)
    recalled: int = 0  # 注入的记忆条数（画像、事实、对话片段）
    block: str = ""  # 注入的证据块
    epoch: int | None = None  # 想起之前这个空间删过几次发言（MemoryDB.turn_epoch，见 _write_back）


class MemoryProxy:
    def __init__(
        self,
        system: MemorySystem,
        *,
        default_user: str | None = None,
        role: str = "agent",
        mode: RecallMode = "auto",
        idle_minutes: float = 30.0,
        unknown_model: UnknownModel | None = None,
        http_client: httpx2.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_open_runs: int | None = None,
    ) -> None:
        self.system = system
        # 一个记忆空间同时进行中的会话数的上限（见 _make_room）；不给时读 CUEMEM_MAX_OPEN_SESSIONS
        self.max_open_runs = max_open_runs if max_open_runs is not None else open_session_limit()
        self.memory = Memory.from_system(system)
        self.default_user = default_user
        self.role = role
        self.mode: RecallMode = mode
        self.idle_s = idle_minutes * 60
        # 不认识的模型：不给时照配置（proxy.unknown_model，默认 remap）
        self.unknown_model: UnknownModel = unknown_model or system.settings.proxy.unknown_model
        # /v1/embeddings 直接用里面的向量器：调用方的文本不是记忆，不写进向量缓存（遗忘清不到）
        embedder = system.embedder
        self._embedder: TextEmbedder = (
            embedder.inner if isinstance(embedder, CachedEmbedder) else embedder
        )
        self._http = http_client or httpx2.Client(timeout=300.0)
        self._clock = clock
        self._runs: dict[tuple[str, str | None], Run] = {}
        # 停放的会话（_make_room）：键 -> (会话 ID, 最后一次请求的时间)，按停放的先后
        self._parked: OrderedDict[tuple[str, str | None], tuple[str, float]] = OrderedDict()
        self._adopted: set[str] = set()  # 接手过崩溃前留下的会话的空间
        self._warned: set[str] = set()  # 已经警告过的不认识的模型名
        self._lock = threading.Lock()
        system.start_worker()  # 会话结束后的抽取在后台完成

    # ---------- 请求 ----------

    def models(self) -> dict[str, Any]:
        """GET /v1/models：目录里供应商的 Key 已设置的别名（_available_aliases），加上向量模型。"""
        catalog = self.system.settings.models
        available = set(self._available_aliases())
        data = [
            {"id": alias, "object": "model", "created": 0, "owned_by": spec.provider}
            for alias, spec in catalog.items()
            if alias in available
        ]
        cfg = self.system.settings.embedding  # /v1/embeddings 的模型
        owner = cfg.provider if cfg.backend in ("dashscope", "openai") else "cuemem"
        data.append({"id": self._embedder.name, "object": "model", "created": 0, "owned_by": owner})
        return {"object": "list", "data": data}

    def complete(
        self,
        body: dict[str, Any],
        headers: Mapping[str, str],
        principal: Principal | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        """非流式：返回（状态码，响应体，附加响应头）。上游出错时原样返回（连同它的
        Retry-After）、不写记忆；连不上或超时时回 OpenAI 格式的 502 / 504。
        principal：请求的令牌，用户在它的作用域之外时 ProxyError（403）；None 不限制。"""
        turn, upstream, url, auth = self._prepare(body, headers, principal)
        extra = _headers(turn, upstream["model"])
        try:
            response = self._http.post(url, json=upstream, headers=auth)
        except httpx2.TransportError as exc:
            status, error = _unreachable(exc)
            return status, error, extra
        data = _json(response)
        if response.status_code == 200:
            choices = data.get("choices") or [{}]
            message = choices[0].get("message") or {}
            self._write_back(turn, message.get("content"))
            return 200, data, extra
        return response.status_code, data, {**extra, **_retry_headers(response)}

    def stream(
        self,
        body: dict[str, Any],
        headers: Mapping[str, str],
        principal: Principal | None = None,
    ) -> tuple[int, Iterator[bytes], dict[str, str]]:
        """流式：先拿到上游的状态码，再把 SSE 原样转发；流读完后写回回答。

        状态码是 200 时返回的迭代器（_Relay）要么读完，要么调用 close()：它占着上游的连接，
        中途不要的（客户端断开、用户停下）只写回用户的话；上游中途出错时什么也不写。
        上游一开始就出错时与非流式相同。"""
        turn, upstream, url, auth = self._prepare(body, headers, principal)
        request = self._http.build_request("POST", url, json=upstream, headers=auth)
        extra = _headers(turn, upstream["model"])
        try:
            response = self._http.send(request, stream=True)
        except httpx2.TransportError as exc:
            status, error = _unreachable(exc)
            return status, iter([json.dumps(error).encode()]), extra
        if response.status_code != 200:
            content = response.read()
            response.close()
            return response.status_code, iter([content]), {**extra, **_retry_headers(response)}
        return 200, _Relay(self, turn, response), extra

    def respond(
        self,
        body: dict[str, Any],
        headers: Mapping[str, str],
        principal: Principal | None = None,
    ) -> tuple[int, dict[str, Any], dict[str, str]]:
        """/v1/responses 的子集（见模块说明）：换成 Chat Completions，走 complete 同样的想起、注入、
        转发与写回，再把回答写成 Responses 的格式。不支持的参数报 ProxyError（501）。"""
        status, data, extra = self.complete(_responses_to_chat(body), headers, principal)
        if status != 200:  # 上游的出错原样返回（OpenAI 两种接口的出错格式相同）
            return status, data, extra
        return status, _chat_to_response(body, data, extra["X-CueMem-Model"]), extra

    def embeddings(self, body: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """/v1/embeddings：交给配置里的向量模型（文档向量，不加查询指令）。返回（响应体，响应头）。"""
        embedder = self._embedder
        name = embedder.name  # 实际用的模型（离线的 hash 后端也照实写）
        self._check_model(body.get("model"), {name, self.system.settings.embedding.model}, name)
        texts = _embedding_input(body.get("input"))
        dimensions = body.get("dimensions")
        if dimensions is not None and dimensions != embedder.dim:
            raise ProxyError(400, f"{name} returns {embedder.dim} dimensions.", param="dimensions")
        encoding = body.get("encoding_format") or "float"
        if encoding not in ("float", "base64"):
            raise ProxyError(
                400, "encoding_format must be float or base64.", param="encoding_format"
            )
        try:
            batch = embedder.embed(texts, "document")
        except EmbeddingAPIError as exc:  # 向量器自己已经按退避重试过
            # 状态码与 HTTP 接口相同（api/failures.py）：限流 429、暂时不可用 503 带 Retry-After，
            # 余额不足 402，超时 504，别的 502
            failure = provider_failure(exc)
            assert failure is not None  # 向量接口的出错都有对应的状态码
            raise ProxyError(
                failure.status,
                str(exc),
                kind=_ERROR_TYPES.get(failure.kind, "upstream_error"),
                code=failure.kind,
                headers={"Retry-After": failure.retry_after} if failure.retry_after else None,
            ) from exc
        data = [
            {"object": "embedding", "index": i, "embedding": _vector(row, encoding)}
            for i, row in enumerate(batch.dense)
        ]
        usage = {"prompt_tokens": batch.tokens, "total_tokens": batch.tokens}
        return (
            {"object": "list", "data": data, "model": name, "usage": usage},
            {"X-CueMem-Model": name},
        )

    # ---------- 会话 ----------

    def close_idle(self) -> int:
        """空闲超过 idle_minutes 的会话（包括停放的，见 _make_room）交给后台抽取；返回关掉的
        会话数。"""
        now = self._clock()
        with self._lock:
            stale = [k for k, r in self._runs.items() if now - r.last_seen >= self.idle_s]
            runs = [self._runs.pop(k) for k in stale]
            parked = [k for k, (_, seen) in self._parked.items() if now - seen >= self.idle_s]
            runs += [self._unpark(k) for k in parked]
        for run in runs:
            self._finish(run)
        return len(runs)

    def open_runs(self) -> int:
        """内存里进行中的会话数（不含停放的，见 parked_runs）。"""
        with self._lock:
            return len(self._runs)

    def parked_runs(self) -> int:
        """停放的会话数（见 _make_room）。"""
        with self._lock:
            return len(self._parked)

    def close_all(self) -> None:
        with self._lock:
            runs = list(self._runs.values())
            self._runs.clear()
            runs += [self._unpark(k) for k in list(self._parked)]
        for run in runs:
            self._finish(run)

    def _unpark(self, key: tuple[str, str | None]) -> Run:
        """取走一个停放的会话（调用方持有 self._lock），还原成 Run 交给 _finish：它没有留着的
        写回（停放时就没有），_finish 只是把它交给抽取。"""
        sid, seen = self._parked.pop(key)
        return Run(key[0], sid, seen)

    def _finish(self, run: Run) -> None:
        """会话结束：没写进去的写回最后再写一次（见 Run），再交给后台抽取。"""
        with run.lock:
            self._save(run, final=True)
        self.system.close_session(run.space, run.session_id)

    def _run(
        self,
        space: str,
        explicit: str | None,
        *,
        fresh: bool = False,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> Run:
        """这次请求属于哪个会话。fresh：请求里没有之前的对话（调用方开始了新的一段），
        没指定会话时结束上一段、开新的一段——会话的边界跟着调用方的对话走。user_id、agent_id：
        这个空间是谁的（space_for 的参数，写回时用）。"""
        now = self._clock()
        self.close_idle()
        self._adopt(space, explicit)
        key = (space, explicit)
        ended: list[Run] = []
        try:
            with self._lock:
                run = self._runs.get(key)
                if run is not None and fresh and explicit is None:
                    ended.append(self._runs.pop(key))
                    run = None
                # 停放的会话（_make_room）：接着用原来的会话 ID，没交给过抽取；没指定会话、调用方
                # 开始了新的一段时与进行中的相同，先结束它
                parked = self._parked.pop(key, None) if run is None else None
                if parked is not None and fresh and explicit is None:
                    ended.append(Run(space, parked[0], parked[1]))
                    parked = None
                if run is None:
                    try:
                        evicted = self._make_room(space)
                    except ProxyError:
                        # 第八轮复审：腾不出名额（429）时停放的会话放回去，照旧由 close_idle、
                        # close_all 交给抽取；以前它两边都不在，一直进行中
                        if parked is not None:
                            self._parked[key] = parked
                        raise
                    if evicted is not None:
                        ended.append(evicted)
                    when = datetime.now()
                    sid = (
                        parked[0]
                        if parked is not None
                        else explicit or sortable_session_id("proxy", when)
                    )
                    run = Run(space, sid, now, user_id=user_id, agent_id=agent_id)
                    self._runs[key] = run
                    # 进行中：每轮照常写入，但不抽取（停放过的会话还开着，open_session 什么也不做）
                    self.system.open_session(space, sid)
                run.last_seen = now
        finally:  # 已经取走、要结束的会话 429 时也交给抽取
            for done in ended:
                self._finish(done)
        return run

    def _make_room(self, space: str) -> Run | None:
        """_run 开新会话之前（调用方持有 self._lock）：这个空间内存里进行中的会话没到 max_open_runs
        个时 None；到了时把其中最久没有请求、没有留着重试的写回、也没在写回的那个停放，没有这样的
        会话时 ProxyError 429。以前不限：调用方每换一个 X-CueMem-Session 就多一个会话，故障期间每个
        会话还能留 MAX_UNSAVED_BYTES 的写回，内存跟着会话数一直长。

        停放只丢掉内存里的 Run（它没有留着的写回，丢掉不丢东西），只记下会话 ID 与最后一次请求的
        时间（_parked）；会话照旧进行中：不交给抽取，数据库里的进行中标记与 MemorySystem 的登记
        都留着（巩固照旧跳过它，崩溃后照旧由 adopt_open 接手）。调用方再用这个会话时接着写，
        闲置太久时由 close_idle 交给抽取，退出时由 close_all。第八轮体检 P1：以前这里直接结束它
        （交给抽取），调用方回来接着写时会话又打开、长了，下次结束时整个空间重放
        （Consolidator._replay_regrown）：一个用户的并行会话多于上限时每次腾名额都重放一遍。
        停放的会话超过 PARKED_KEPT 个时返回最早停放的那个（调用方结束它，交给抽取）。"""
        mine = [k for k, r in self._runs.items() if r.space == space]
        if len(mine) < self.max_open_runs:
            return None
        idle = [k for k in mine if not self._runs[k].unsaved and not self._runs[k].lock.locked()]
        if not idle:
            raise ProxyError(
                429,
                f"This user already has {len(mine)} open memory sessions (the limit is "
                f"{self.max_open_runs}, set by {MAX_OPEN_SESSIONS_ENV}) and none can be ended "
                "now: retry later.",
                kind="rate_limit_error",
                code="rate_limited",
                headers={"Retry-After": "60"},
            )
        oldest = min(idle, key=lambda k: self._runs[k].last_seen)
        run = self._runs.pop(oldest)
        self._parked[oldest] = (run.session_id, run.last_seen)
        if len(self._parked) > PARKED_KEPT:
            return self._unpark(next(iter(self._parked)))
        return None

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
        self, body: dict[str, Any], headers: Mapping[str, str], principal: Principal | None
    ) -> tuple[Turn, dict[str, Any], str, dict[str, str]]:
        # 先认模型与令牌的用量限制：被拒绝的请求不开会话、不检索
        try:
            upstream, url, auth = self._upstream(body, principal)
        except ProxyError as exc:
            if exc.status == 503:  # 缺 Key 排在令牌的作用域之后：越权的请求照旧先 403
                self._identity(body, headers, principal)
            raise
        turn = self._recall(body, headers, principal)
        if turn.anchor is not None and turn.block:
            # 放进最后一句用户消息；它之后的消息（工具调用与结果等）原样接在后面。请求以用户
            # 消息结尾时后面没有东西，与以前逐字节相同
            messages, cut = upstream["messages"], turn.anchor + 1
            upstream["messages"] = inject_evidence(messages[:cut], turn.block) + messages[cut:]
        return turn, upstream, url, auth

    def _identity(
        self, body: dict[str, Any], headers: Mapping[str, str], principal: Principal | None
    ) -> tuple[Any, str | None]:
        """请求的用户与 Agent；用户不在令牌的作用域里时 ProxyError（403）。"""
        try:  # 不是 ASCII 的 ID：原样的 UTF-8 或 utf-8''<百分号编码>（auth.header_id）
            header_user, agent_id = identity_headers(headers)
        except ValueError as exc:
            raise ProxyError(400, str(exc), param="user") from None
        user_id = header_user or body.get("user") or self.default_user
        if (  # 在开会话、检索之前；管理令牌不限制
            user_id is not None
            and principal is not None
            and not principal.admin
            and (not isinstance(user_id, str) or not principal.allows(space_for(user_id, agent_id)))
        ):
            raise ProxyError(
                403,
                f"This API key cannot access the memories of user {user_id!r}"
                + (f" with agent {agent_id!r}" if agent_id else "")
                + ": use a user in its scope (the user field or the X-CueMem-User header).",
                kind="permission_error",
                param="user",
                code="forbidden",
            )
        return user_id, agent_id

    def _recall(
        self, body: dict[str, Any], headers: Mapping[str, str], principal: Principal | None
    ) -> Turn:
        user_id, agent_id = self._identity(body, headers, principal)
        messages = body.get("messages") or []
        if user_id is None or not messages:
            return Turn(space=None)
        space = space_for(user_id, agent_id)
        held = spaces_over_cap(principal, space, self.system.db.space_ids)
        if held is not None and principal is not None:  # 令牌的 max_spaces（api/auth.py）
            raise ProxyError(
                429,
                space_cap_message(principal, held),
                kind="rate_limit_error",
                code="rate_limited",
                headers={"x-should-retry": "false"},  # 等也没用：要删空间或调高上限
            )
        # 新的空间、本地向量库又没有余量建它：在开会话、转发之前 503，不让这一轮写不进去。空间的
        # 第一张图片要另建图片集合，同样先查（第九轮体检 P3）：以前转发、回答之后写回才失败
        try:
            self.system.vectors.check_room(space, images=self._stores_image(messages[-1]))
        except VectorStoreFullError as exc:
            raise ProxyError(
                503,
                exc.for_client(),
                kind="server_error",
                code="capacity",
                headers={"x-should-retry": "false"},  # 要运营者处理，自动重试也一样
            ) from exc
        explicit = headers.get("x-cuemem-session")
        session = _SESSION_ID.sub("_", explicit)[:80] if explicit else None
        earlier = [m for m in messages[:-1] if m.get("role") in ("user", "assistant")]
        run = self._run(space, session, fresh=not earlier, user_id=user_id, agent_id=agent_id)
        turn = Turn(space=space, user_id=user_id, agent_id=agent_id, run=run)
        users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not users:
            return turn
        turn.anchor = users[-1]
        asked = messages[turn.anchor]
        # 不以用户消息结尾：工具调用的后续请求（或用户消息之后还有系统消息），只写回回答
        follow_up = turn.anchor != len(messages) - 1
        if not follow_up:
            turn.write_user = asked
            run.kept = None  # 新的一轮：上一轮想起的不再用（这次想不起来时也不用）
        query = message_text(asked.get("content"))
        if not query.strip() or not self.system.db.has_sessions(space):
            return turn
        digest = _digest(asked)
        epoch = self.system.db.turn_epoch(space)  # 在想起之前记：之后的遗忘都比得出来
        kept = run.kept
        if follow_up and kept is not None and kept.digest == digest and kept.epoch == epoch:
            # 这一轮已经想起过、之后没有遗忘过：原样再用，不再检索
            turn.level, turn.reasons = kept.level, list(kept.reasons)
            turn.recalled, turn.block, turn.epoch = kept.recalled, kept.block, kept.epoch
            record_event(
                "proxy_recall", space=space, level=turn.level, recalled=turn.recalled, reused=True
            )
            return turn
        turn.epoch = epoch
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
        run.kept = Kept(digest, turn.block, turn.level, list(turn.reasons), turn.recalled, epoch)
        record_event("proxy_recall", space=space, level=turn.level, recalled=turn.recalled)
        return turn

    def _stores_image(self, message: Any) -> bool:
        """这句消息写回时会不会存图片（要用图片集合）：配了图片库与图片向量、是用户消息、带着
        data URL 的图片（与 _storable 相同：网络图片与本地路径不存，不该因此 503）。"""
        if self.system.media is None or self.system.image_embedder is None:
            return False
        if not isinstance(message, dict) or message.get("role") != "user":
            return False
        content = message.get("content")
        return isinstance(content, list) and any(
            isinstance(block, dict)
            and block.get("type") == "image_url"
            and _image_url(block).startswith("data:")
            for block in content
        )

    def _in_context(self, space: str, session_id: str, earlier: list[dict[str, Any]]) -> list[str]:
        """当前会话里已经原样在请求中的轮（检索时排除，名额留给请求里没有的）。之前的每条消息
        按写回时的样子比对：先过 _storable（只有 data URL 的图片会存下、记作 <image>）。"""
        present = {stored_text(_storable(m)).strip() for m in earlier}
        return in_context_rounds(self.system, space, [session_id], present)

    def _upstream(
        self, body: dict[str, Any], principal: Principal | None = None
    ) -> tuple[dict[str, Any], str, dict[str, str]]:
        """请求体 → 上游的请求体、地址与鉴权头。principal 写了 models、max_output_tokens 时按它们
        查、改（见模块说明）；没写时与以前逐字节相同。"""
        settings = self.system.settings
        spec = self._model(body.get("model"))  # 改用默认模型之后的
        models = principal.models if principal is not None else None
        cap = principal.max_output_tokens if principal is not None else None
        if models is not None:
            # 列出的别名只放行这个别名的一项（连同它的 extra_body）：同一个模型 ID 可以有好几个
            # 别名（deepseek-flash 与打开思考的 deepseek-flash-thinking），以前按模型 ID 一并放行。
            # 列出的是模型 ID、又不是哪个别名时，放行这个 ID 的各项
            aliases = {alias for alias, s in settings.models.items() if s == spec}
            bare = spec.model_id in models and spec.model_id not in settings.models
            if aliases.isdisjoint(models) and not bare:
                raise ProxyError(
                    403,
                    f"This API key may use only these models: {', '.join(models)} "
                    f"(the request resolves to {spec.model_id!r}).",
                    kind="permission_error",
                    param="model",
                    code="forbidden",
                )
        if cap is not None and body.get("n") not in (None, 1):
            raise ProxyError(400, "This API key may request one choice only (n = 1).", param="n")
        provider = settings.providers[spec.provider]
        key = provider.api_key()
        if key is None:
            # 预设叠在默认配置上，目录里还有没设 Key 的供应商的别名（--preset ollama 下的
            # qwen3.8-flash 等）。以前照样转发、不带鉴权头：用户的话连同想起的记忆发给了
            # 没配置 Key 的供应商，再被 401 拒绝。现在在检索、开会话之前 503
            requested = body.get("model")
            named = isinstance(requested, str) and (
                requested in settings.models or requested == spec.model_id
            )
            name = requested if named else spec.model_id  # 改用 agent 角色的模型时写它
            raise ProxyError(
                503,
                f"Model {name!r} needs {provider.api_key_env}, "
                "which is not set on this server. Use one of: "
                f"{', '.join(self._available_aliases())} (see GET /v1/models).",
                kind="server_error",
                param="model",
                code="model_not_available",
                headers={"x-should-retry": "false"},  # 缺的是服务端的配置，重试也一样
            )
        upstream = {k: v for k, v in body.items() if k != "user"}
        upstream["model"] = spec.model_id
        _adapt(upstream, spec, override=models is not None)
        if cap is not None:
            _cap_output(upstream, spec, cap)
        auth = {"Authorization": f"Bearer {key}"} if key else {}
        return upstream, provider.base_url.rstrip("/") + "/chat/completions", auth

    def _model(self, requested: Any) -> ModelSpec:
        """请求的 model → 模型目录里的一项：配置里的别名或模型 ID；没给时是 role 的模型；
        不认识的照 unknown_model 处理（见 _check_model）。"""
        models = self.system.settings.models
        if isinstance(requested, str):
            if requested in models:
                return models[requested]
            for spec in models.values():
                if spec.model_id == requested:
                    return spec
        fallback = models[self.system.settings.roles[self.role].model]
        self._check_model(requested, models.keys(), fallback.model_id)
        return fallback

    def _available_aliases(self) -> list[str]:
        """目录里供应商的 Key 已设置（或不需要 Key）的别名，按字母排。别的别名转发不了（503）。"""
        settings = self.system.settings
        return sorted(
            alias
            for alias, spec in settings.models.items()
            if settings.providers[spec.provider].api_key() is not None
        )

    def _check_model(self, requested: Any, known: Collection[str], used: str) -> None:
        """请求的 model 不在 known 里时：unknown_model 是 reject 时报 400（OpenAI 的
        model_not_found，列出转发得了的别名），否则照旧改用 used，每个模型名记一次警告（响应头
        X-CueMem-Model 写着实际用的模型）。没给 model 的请求不算：没有要求哪个模型。"""
        if requested is None or (isinstance(requested, str) and requested in known):
            return
        if self.unknown_model == "reject":
            raise ProxyError(
                400,
                f"The model {requested!r} is not served by this proxy. Use one of: "
                f"{', '.join(self._available_aliases())} (see GET /v1/models).",
                param="model",
                code="model_not_found",
            )
        name = str(requested)
        with self._lock:
            first = name not in self._warned
            if first and len(self._warned) < 1000:  # 调用方乱写的模型名不能让它无限长大
                self._warned.add(name)
        if first:
            logger.warning(
                "Model %r is not in the CueMem model catalog: using %s instead (see the "
                "X-CueMem-Model response header; set proxy.unknown_model to 'reject' to refuse).",
                name,
                used,
            )

    def _write_back(self, turn: Turn, answer: str | None) -> None:
        """把这一轮写进记忆：用户的话与回答。

        想起之后、写回之前这个空间遗忘过（等上游回答的时候别的调用方删了这个用户的记忆）时，注入了
        记忆的回答不写：它是照想起的记忆写的，可能复述了刚删的内容，写回去下次抽取时又长出来。以前
        写回时才记遗忘次数，比不出这段时间里的遗忘（D-068 记下的剩余限制，流式时这段时间更长）。
        用户的话照写：它不是从记忆来的，与这次遗忘无关时丢掉就再也没有了（之后的请求只写各自最后
        一句用户消息）。没有注入记忆的回答（想起失败、空间里还没有记忆）照写。工具调用的后续请求
        也注入了这一轮的证据块（见 _recall），同样照这条规则：期间遗忘过时它的回答不写。"""
        if turn.space is None or turn.run is None:
            return
        user = [_storable(turn.write_user)] if turn.write_user is not None else []
        answered = bool(answer and answer.strip())
        if not user and not answered:
            return
        run = turn.run
        with run.lock:  # 同一会话的写回按顺序进行（发言序号只增不减）
            epoch = self.system.db.turn_epoch(run.space)
            messages = list(user)
            if answered and turn.block and turn.epoch is not None and turn.epoch != epoch:
                record_event(
                    "proxy_write_dropped", space=run.space, reason="forgotten", part="answer"
                )
            elif answered:
                messages.append({"role": "assistant", "content": answer})
            if not messages:
                return
            run.keep(Unsaved(messages, datetime.now(), epoch, payload_size(messages)))
            self._save(run)
            self._trim(run)  # 写得进去时先写：只限写完之后还留着的

    @staticmethod
    def _trim(run: Run) -> None:
        """留着重试的写回超过 MAX_UNSAVED 次或 MAX_UNSAVED_BYTES 时丢掉最早的（调用方持有
        run.lock）。最新的这一次总留着：单个请求已经由 MAX_BODY_BYTES 限住。"""
        dropped: list[Unsaved] = []
        while len(run.unsaved) > 1 and (
            len(run.unsaved) > MAX_UNSAVED or run.unsaved_bytes > MAX_UNSAVED_BYTES
        ):
            dropped.append(run.unsaved[0])
            run.drop()
            record_event("proxy_write_dropped", space=run.space, reason="backlog")
        if dropped:
            logger.warning(
                "CueMem could not store %d message(s) of proxy session %s for a long time and "
                "dropped them (more than %d write-backs or %d bytes were waiting).",
                sum(len(u.messages) for u in dropped),
                run.session_id,
                MAX_UNSAVED,
                MAX_UNSAVED_BYTES,
            )

    def _save(self, run: Run, *, final: bool = False) -> None:
        """把这个会话还没写进去的写回按先后写进记忆，各用各自的时间（调用方持有 run.lock）。
        写回失败不影响已经返回给调用方的回答：重试有用的（worth_retrying）这一次与之后的都留着
        （见 Run），final（会话要结束了）时丢掉；重试也没用的只丢这一次，接着写后面的。"""
        while run.unsaved:
            head = run.unsaved[0]
            if head.epoch != self.system.db.turn_epoch(run.space):
                # 之后这个空间遗忘过发言（例如调用方删了这个用户的记忆）：这是遗忘之前说的话
                record_event("proxy_write_dropped", space=run.space, reason="forgotten")
                run.drop()
                continue
            try:
                self.memory.add(
                    head.messages,
                    user_id=run.user_id,
                    agent_id=run.agent_id,
                    run_id=run.session_id,
                    timestamp=head.when,
                )
            except Exception as exc:
                keep = worth_retrying(exc) and not final
                error = f"{type(exc).__name__}: {exc}"
                record_event("proxy_write_failed", space=run.space, error=error, kept=keep)
                if keep:
                    return
                lost = run.unsaved if worth_retrying(exc) else [head]
                logger.warning(
                    "CueMem could not store %d message(s) of proxy session %s and dropped "
                    "them (%s).",
                    sum(len(u.messages) for u in lost),
                    run.session_id,
                    error,
                )
                run.drop(len(lost))
                continue
            run.drop()


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


def _adapt(upstream: dict[str, Any], spec: ModelSpec, *, override: bool = False) -> None:
    """按模型目录补上供应商特有的参数（例如关掉思考模式），并处理推理模型的参数差异。调用方给了
    同名参数时用调用方的；override（令牌限定了 models）时用目录的：限定模型就是限定花费，调用方
    不能借 enable_thinking 之类的开关把便宜的模型变贵。"""
    for key, value in spec.extra_body.items():
        if override:
            upstream[key] = value
        else:
            upstream.setdefault(key, value)
    if spec.fixed_temperature:
        upstream.pop("temperature", None)
        upstream.pop("top_p", None)
    if spec.max_tokens_param == "max_completion_tokens" and "max_tokens" in upstream:
        upstream.setdefault("max_completion_tokens", upstream.pop("max_tokens"))


def _cap_output(upstream: dict[str, Any], spec: ModelSpec, cap: int) -> None:
    """令牌的 max_output_tokens：max_tokens 与 max_completion_tokens 不超过它（不是整数的也换成
    它），并且总按这个模型的写法（max_tokens_param）写上一份：没给的时候是 cap，只给了另一个名字
    时取给了的（已经压过的）值。以前只给 max_completion_tokens 时目录里写 max_tokens 的模型
    （DeepSeek、Qwen）收不到 max_tokens，供应商不认这个名字时上限就不起作用（第五轮体检）。"""
    given = [key for key in ("max_tokens", "max_completion_tokens") if key in upstream]
    for key in given:
        value = upstream[key]
        if not isinstance(value, int) or isinstance(value, bool) or value > cap:
            upstream[key] = cap
    if spec.max_tokens_param not in upstream:
        upstream[spec.max_tokens_param] = min((upstream[key] for key in given), default=cap)


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


def _digest(message: dict[str, Any]) -> str:
    """用户消息写回时的样子的 SHA-256（见 Kept）：与 _in_context 比对时的写法相同。"""
    return hashlib.sha256(
        stored_text(_storable(message)).strip().encode("utf-8", "surrogatepass")
    ).hexdigest()


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


# ---------- /v1/embeddings ----------

# 向量接口出错（api/failures.py 的分类）→ OpenAI 格式里的 type（与对话转发的 _unreachable 相同）
_ERROR_TYPES = {"rate_limited": "rate_limit_error", "timeout": "timeout"}


def _embedding_input(raw: Any) -> list[str]:
    """input：一段文字或文字的列表。token ID 的数组不收（向量模型的分词这里不知道）。"""
    texts = [raw] if isinstance(raw, str) else raw
    if not isinstance(texts, list) or not texts:
        raise ProxyError(
            400, "input must be a string or a non-empty list of strings.", param="input"
        )
    if not all(isinstance(text, str) for text in texts):
        raise ProxyError(400, "input must be text: token arrays are not supported.", param="input")
    if not all(texts):
        raise ProxyError(400, "input must not contain empty strings.", param="input")
    if len(texts) > MAX_EMBED_INPUTS:
        raise ProxyError(400, f"input can have at most {MAX_EMBED_INPUTS} items.", param="input")
    return texts


def _vector(row: Any, encoding: str) -> list[float] | str:
    """一行向量 → float 列表，或 base64 编码的小端 float32（官方 OpenAI SDK 缺省请求 base64）。"""
    vector = np.asarray(row, dtype="<f4")
    if encoding == "base64":
        return base64.b64encode(vector.tobytes()).decode()
    return list(vector.tolist())


# ---------- /v1/responses ----------

# 照着转换的参数，与不影响回答、照单全收的参数（store：这里不保存回答，响应里写 store: false）
_RESPONSES_MAPPED = frozenset(
    {"model", "input", "instructions", "max_output_tokens", "temperature", "top_p", "user"}
)
_RESPONSES_IGNORED = frozenset(
    {"metadata", "store", "parallel_tool_calls", "service_tier", "prompt_cache_key"}
    | {"safety_identifier"}
)
# 写成这些值时等于没给的参数；别的值（流式、工具、结构化输出等）不支持
_RESPONSES_DEFAULTS: dict[str, tuple[Any, ...]] = {
    "stream": (False,),
    "background": (False,),
    "tools": ([],),
    "tool_choice": ("auto", "none"),
    "include": ([],),
    "truncation": ("disabled",),
    "text": ({}, {"format": {"type": "text"}}),
    "reasoning": ({},),
}


def _unsupported(what: str) -> ProxyError:
    """501：OpenAI 的 SDK 遇到 5xx 会重试，x-should-retry: false 让它不重试。"""
    return ProxyError(
        501,
        "The CueMem proxy supports a subset of the Responses API (non-streaming text input, "
        f"no tools, no stored responses). Not supported: {what}. "
        "Use /v1/chat/completions for the rest.",
        code="not_implemented",
        headers={"x-should-retry": "false"},
    )


def _responses_to_chat(body: Mapping[str, Any]) -> dict[str, Any]:
    """Responses 的请求 → Chat Completions 的请求：instructions 是第一条系统消息，input 的每条
    消息照抄（developer 写成 system，内容块只收 input_text / output_text，拼成一段文字）。"""
    unsupported = sorted(
        key
        for key, value in body.items()
        if value is not None
        and key not in _RESPONSES_MAPPED | _RESPONSES_IGNORED
        and value not in _RESPONSES_DEFAULTS.get(key, ())
    )
    if unsupported:
        raise _unsupported(", ".join(unsupported))
    instructions = body.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise ProxyError(400, "instructions must be a string.", param="instructions")
    raw = body.get("input")
    items = [{"role": "user", "content": raw}] if isinstance(raw, str) else raw
    if not isinstance(items, list) or not items:
        raise ProxyError(
            400, "input must be a string or a non-empty list of messages.", param="input"
        )
    messages = [{"role": "system", "content": instructions}] if instructions else []
    messages += [_input_message(item) for item in items]
    chat = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_output_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "user": body.get("user"),  # 与 Chat Completions 一样：记忆属于哪个用户
    }
    return {key: value for key, value in chat.items() if value is not None}


def _input_message(item: Any) -> dict[str, str]:
    if not isinstance(item, dict):
        raise ProxyError(400, "Each input item must be an object.", param="input")
    kind = item.get("type", "message")
    if kind != "message":
        raise _unsupported(f"input items of type {kind!r}")
    role = item.get("role")
    if role not in ("user", "assistant", "system", "developer"):
        raise ProxyError(400, f"Unknown message role {role!r}.", param="input")
    content = item.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type not in ("input_text", "output_text"):
                raise _unsupported(f"{part_type or 'non-text'} content (only input_text)")
            parts.append(str(part.get("text") or ""))
        content = "\n".join(parts)
    elif not isinstance(content, str):
        raise ProxyError(400, "Message content must be a string or a list.", param="input")
    return {"role": "system" if role == "developer" else role, "content": content}


def _chat_to_response(
    body: Mapping[str, Any], data: Mapping[str, Any], model: str
) -> dict[str, Any]:
    """Chat Completions 的回答 → Responses 的响应对象（一条 assistant 消息，一段 output_text）。
    回答被截断（length）或被内容审核拦下时 status 是 incomplete。"""
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    reason = {"length": "max_output_tokens", "content_filter": "content_filter"}.get(
        choice.get("finish_reason")
    )
    status = "incomplete" if reason else "completed"
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    cached = cached if cached is not None else usage.get("prompt_cache_hit_tokens")  # DeepSeek
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    now = int(time.time())
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": data.get("created") or now,
        "completed_at": now if status == "completed" else None,
        "status": status,
        "background": False,
        "error": None,
        "incomplete_details": {"reason": reason} if reason else None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "metadata": body.get("metadata") or {},
        "model": data.get("model") or model,
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": status,
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
                ],
            }
        ],
        "parallel_tool_calls": body.get("parallel_tool_calls") is not False,
        "previous_response_id": None,
        "store": False,
        "temperature": body.get("temperature"),
        "text": {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice") or "auto",
        "tools": [],
        "top_p": body.get("top_p"),
        "truncation": "disabled",
        "usage": {
            "input_tokens": prompt,
            "input_tokens_details": {"cached_tokens": int(cached or 0), "cache_write_tokens": 0},
            "output_tokens": completion,
            "output_tokens_details": {"reasoning_tokens": int(reasoning or 0)},
            "total_tokens": int(usage.get("total_tokens") or prompt + completion),
        },
        "user": body.get("user"),
    }


def _headers(turn: Turn, model: str) -> dict[str, str]:
    """响应头：控制器的决定（X-CueMem-Level / -Recalled / -Reasons / -Session）与实际转发给上游的
    模型 ID（X-CueMem-Model：请求的模型不认识、改用了默认模型时，调用方由此知道）。"""
    if turn.space is None:
        return {"X-CueMem-Level": "none", "X-CueMem-Model": model}
    out = {
        "X-CueMem-Level": turn.level,
        "X-CueMem-Recalled": str(turn.recalled),
        "X-CueMem-Reasons": "; ".join(turn.reasons),
        "X-CueMem-Model": model,
    }
    if turn.run is not None:
        out["X-CueMem-Session"] = turn.run.session_id
    # 响应头只能是 latin-1：理由里偶尔有别的字符
    return {k: v.encode("ascii", "replace").decode() for k, v in out.items()}


def proxy_router(proxy: MemoryProxy, *, dependencies: list[Any] | None = None) -> APIRouter:
    """/v1/models、/v1/chat/completions、/v1/responses 与 /v1/embeddings；可以挂在 cuemem serve 的
    应用上，也可以单独运行。dependencies 要有 api.auth.guard（认出令牌，按它的作用域检查用户）。"""
    router = APIRouter(dependencies=dependencies or [])

    @router.get("/v1/models")
    def models() -> dict[str, Any]:
        return proxy.models()

    @router.post("/v1/chat/completions")
    def completions(body: Annotated[dict[str, Any], Depends(_body)], request: Request) -> Response:
        headers = {k.lower(): v for k, v in request.headers.items()}
        who = principal_of(request)
        try:
            if body.get("stream"):
                status, chunks, extra = proxy.stream(body, headers, who)
                if status != 200:
                    return Response(b"".join(chunks), status, extra, media_type="application/json")
                return _RelayResponse(chunks, headers=extra, media_type="text/event-stream")
            status, data, extra = proxy.complete(body, headers, who)
        except ProxyError as exc:
            return exc.response()
        return JSONResponse(data, status, extra)

    @router.post("/v1/responses")
    def responses(body: Annotated[dict[str, Any], Depends(_body)], request: Request) -> Response:
        headers = {k.lower(): v for k, v in request.headers.items()}
        try:
            status, data, extra = proxy.respond(body, headers, principal_of(request))
        except ProxyError as exc:
            return exc.response()
        return JSONResponse(data, status, extra)

    @router.post("/v1/embeddings")
    def embeddings(body: Annotated[dict[str, Any], Depends(_body)]) -> Response:
        try:
            data, extra = proxy.embeddings(body)
        except ProxyError as exc:
            return exc.response()
        return JSONResponse(data, 200, extra)

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
    keys: AnyKeys | None = None,
    allowed_hosts: Sequence[str] | None = None,
    origin_hosts: Sequence[str] = (),
    allow_ip_hosts: bool = False,
    max_body_bytes: int = MAX_BODY_BYTES,
) -> FastAPI:
    """单独运行的记忆代理（cuemem proxy）。token 非空时要求 Authorization: Bearer <token>：
    OpenAI SDK 的 api_key 填这个令牌即可（上游的 Key 由 CueMem 的配置提供）。keys：另有限定作用域
    的令牌（Keys.from_env；给了时不看 token），每个令牌只能用作用域里的用户。

    不接受别的网站发来的浏览器请求；allowed_hosts、origin_hosts、allow_ip_hosts 同
    api.app.create_app。
    max_body_bytes：请求体的上限（字节，超过时 413，见 api/limits.py），cuemem proxy 从环境变量
    CUEMEM_MAX_BODY_BYTES 读。"""
    from cuemem.api.app import (
        SWEEP_INTERVAL_S,
        _adopt_after,
        _guard,
        _readiness,
        _sweep,
        _sweep_jobs,
    )

    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # 每分钟把空闲的会话交给后台抽取；启动一个闲置期之后，把之前的进程（崩溃时）留下、没人
        # 接着写的会话交给抽取（一次）；设了 traces_retention_days、expired_retention_days 时再删
        # 过期的 trace、过期的记忆
        adopt = _adopt_after(proxy.system, proxy.idle_s)
        jobs = _sweep_jobs(proxy.system, proxy.close_idle, adopt)
        threading.Thread(
            target=_sweep,
            args=(stop, SWEEP_INTERVAL_S, *jobs),
            name="cuemem-proxy-sweep",
            daemon=True,
        ).start()
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

    @app.get("/health")  # async：不排在默认线程池后面（open_runs 只拿一下锁）
    async def health() -> dict[str, Any]:
        return {"ok": True, "version": __version__, "open_runs": proxy.open_runs()}

    ready_limiter = anyio.CapacityLimiter(2)

    @app.get("/ready")  # 向量库与后台抽取能不能用，同 serve 的 /api/ready
    async def ready() -> JSONResponse:
        return await _readiness(proxy.system, ready_limiter)

    keys = keys if keys is not None else Keys.from_token(token)
    app.include_router(proxy_router(proxy, dependencies=guard(keys)))
    _guard(
        app,
        allowed_hosts,
        origin_hosts,
        max_body_bytes=max_body_bytes,
        keys=keys,
        allow_ip_hosts=allow_ip_hosts,
    )
    return app
