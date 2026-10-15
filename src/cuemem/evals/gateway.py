"""评测网关（N12）：把 CueMem 的 LLM 客户端与向量器包成 OpenAI 兼容的 HTTP 接口，给外部基线用。

外部的记忆系统（Graphiti、LangMem、Cognee、A-MEM……）各自带着 OpenAI 兼容的客户端。让它们照
自己的方式调用（`base_url` 指到这里），而不是把它们的模型接口一个个改写，好处是：

- **同样的模型**：聊天一律交给 extractor 角色的模型（与 CueMem 写入路径相同，N11 的 Mem0 基线
  也是这样），向量一律交给同一个向量器；请求里的 model 名只记下来，不改变路由；
- **同一本账、同一个缓存**：每次调用都走 LLMClient（记账、响应缓存、高峰暂停、内容审核误拒时
  换备用模型）；重跑评测时命中缓存，几乎不花钱；
- **按"桶"统计**：地址里带上桶名（`/b/<桶>/v1`），每个桶分别累计调用次数、tokens 与花费，
  外部系统写入一份对话历史花了多少，直接从桶里读。

与 OpenAI 接口的差异（都记在响应之外，调用方无感）：
- `response_format` 是 json_schema 时改成 json_object，并把 schema 追加到系统提示里（DeepSeek 与
  通义的 JSON 模式不保证支持 json_schema）；
- `stream=true` 时先拿到完整结果，再按 SSE 一次发出（外部系统只是读完整个流）；
- `n` 只支持 1；`seed`、`logprobs` 之类的参数忽略。

为了让外部系统得到它们假定的 OpenAI 语义：
- 带工具的请求没写 `parallel_tool_calls` 时按 OpenAI 的默认（true）发出：通义不设为 true 时
  一次只回一个工具调用，而 LangMem 等要求"一次并行调用多个工具"；写了就照原样转发；
- 检索阶段的桶（`<系统>.search.<空间>`，见 ExternalBaseline.bucket）里的向量请求按查询
  向量化（kind="query"），与 CueMem 自己的检索相同；其余按文档。

每个桶另外统计失败的请求与其中被内容审核拒绝的、带工具的请求、返回的工具调用、一次回多个
工具调用的回复、带工具却没调用的回复、因长度上限被截断的回复：外部系统吞掉或静默丢掉的东西
（如 trustcall 把模型调用的错误改写成提示重试）在写入记录里看得到。

出错时照 OpenAI 的语义回给外部系统，由它们按自己的策略处理：
- **限流**：上游的 429 在 LLM 客户端里已经由 SDK 退避重试过（providers.*.max_retries 次，照上游
  的 Retry-After 等），仍然限流时回 429 并带上上游的 Retry-After，由外部系统自己退避重试，网关
  不再加一层重试；向量请求同样（向量器自己重试 embedding.max_attempts 次）。按桶计入
  rate_limited。以前它是一个没有说明的 500：Cognee 之类按"服务器出错"处理（它对 429 会放慢
  派发，对 500 不会），写入记录里也看不出是限流。
- **其他上游错误**（5xx、连不上、余额不足等）回 502，消息里带上上游的报错；内容审核拒绝与
  请求不合法回 400。
- **解析不了的 JSON**：JSON 模式的回复解析不了时，外部系统多半把同样的请求再发一次；LLM 客户端
  记下这个请求，再来时不读缓存、重新调用（retry_bad_json，见 LLMClient.chat），否则每次重试都
  从缓存里拿到同一个坏回复。按桶计入 bad_json_replies 与 json_refreshes。
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from openai import APIError, APITimeoutError, BadRequestError, RateLimitError

from cuemem.core.trace import start_trace
from cuemem.embed.base import EmbedKind, TextEmbedder
from cuemem.errors import ContentFilteredError, CueMemError, EmbeddingAPIError
from cuemem.llm.client import LLMClient, Message, unparsable_json
from cuemem.llm.types import ChatResult

DEFAULT_BUCKET = "default"
SEARCH_STAGE = "search"  # ExternalBaseline.bucket(key, "search")：检索时的请求


@dataclass
class BucketUsage:
    """一个桶的累计用量。cost 是名义成本，spent 是实际花费（缓存命中不花钱）。"""

    chat_calls: int = 0
    cache_hits: int = 0
    fallbacks: int = 0
    errors: int = 0  # 最终失败的聊天请求（换了备用模型也失败的算一次）
    refusals: int = 0  # 其中被内容审核拒绝的（重试也没用）
    rate_limited: int = 0  # 回了 429 的请求（聊天与向量）：上游重试之后仍然限流
    tool_requests: int = 0  # 带工具的请求
    tool_calls: int = 0  # 返回的工具调用
    multi_tool_replies: int = 0  # 一次回了多个工具调用的回复（并行工具调用生效）
    no_tool_replies: int = 0  # 带工具却没有调用的回复（例如通义没照 tool_choice=required 做）
    truncated_replies: int = 0  # finish_reason 为 length：输出被长度上限截断
    bad_json_replies: int = 0  # JSON 模式的回复解析不了（外部系统多半会重试）
    json_refreshes: int = 0  # 其后同样的请求再来：不读缓存、重新调用的次数
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    embedding_calls: int = 0
    embedded_texts: int = 0
    embedding_tokens: int = 0
    embedding_errors: int = 0  # 最终失败的向量请求（含限流）
    cost: float = 0.0
    spent: float = 0.0
    latency_ms: float = 0.0
    models_requested: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["cost"] = round(self.cost, 6)
        out["spent"] = round(self.spent, 6)
        out["latency_ms"] = round(self.latency_ms, 1)
        return out


class Gateway:
    """OpenAI 兼容的 /chat/completions、/embeddings、/models，背后是 CueMem 的客户端。

    role：聊天交给哪个角色的模型（默认 extractor）。fallback_model：内容审核误拒、超时或
    JSON 生成被中止时改用的模型，默认取巩固配置里的备用模型。
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: TextEmbedder,
        *,
        role: str = "extractor",
        fallback_model: str | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.role = role
        self.fallback = (
            fallback_model
            if fallback_model is not None
            else llm.settings.consolidation.fallback_model
        )
        self._usage: dict[str, BucketUsage] = {}
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    # ---------- 运行 ----------

    def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        """在后台线程里启动（port=0 时自动选一个空闲端口），返回根地址。"""
        if self._server is not None:
            return self.base_url
        if port == 0:
            with socket.socket() as probe:
                probe.bind((host, 0))
                port = probe.getsockname()[1]
        config = uvicorn.Config(self.app(), host=host, port=port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="cuemem-eval-gateway", daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + 30
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("评测网关没能启动")
            time.sleep(0.05)
        self.base_url = f"http://{host}:{port}"
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=10)
        self._server = self._thread = None

    def url(self, bucket: str = DEFAULT_BUCKET) -> str:
        """外部系统用的 base_url（OpenAI SDK 的写法，以 /v1 结尾）。"""
        if not self.base_url:
            raise RuntimeError("评测网关还没有启动")
        return f"{self.base_url}/b/{bucket}/v1"

    def usage(self, bucket: str = DEFAULT_BUCKET) -> dict[str, Any]:
        with self._lock:
            return self._bucket(bucket).as_dict()

    # ---------- 接口 ----------

    def app(self) -> FastAPI:
        app = FastAPI(title="CueMem eval gateway")

        @app.post("/b/{bucket}/v1/chat/completions", response_model=None)
        async def chat(bucket: str, request: Request) -> JSONResponse | StreamingResponse:
            body = await _json_body(request)
            return await _in_thread(self._chat_response, bucket, body)

        @app.post("/v1/chat/completions", response_model=None)
        async def chat_default(request: Request) -> JSONResponse | StreamingResponse:
            body = await _json_body(request)
            return await _in_thread(self._chat_response, DEFAULT_BUCKET, body)

        @app.post("/b/{bucket}/v1/embeddings")
        async def embeddings(bucket: str, request: Request) -> JSONResponse:
            body = await _json_body(request)
            return await _in_thread(self._embeddings_response, bucket, body)

        @app.post("/v1/embeddings")
        async def embeddings_default(request: Request) -> JSONResponse:
            body = await _json_body(request)
            return await _in_thread(self._embeddings_response, DEFAULT_BUCKET, body)

        @app.get("/b/{bucket}/v1/models")
        @app.get("/v1/models")
        def models(bucket: str = DEFAULT_BUCKET) -> dict[str, Any]:
            alias = self.llm.settings.roles[self.role].model
            return {
                "object": "list",
                "data": [
                    {"id": alias, "object": "model", "created": 0, "owned_by": "cuemem"},
                    {
                        "id": self.embedder.name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "cuemem",
                    },
                ],
            }

        return app

    # ---------- 聊天 ----------

    def chat(self, bucket: str, body: dict[str, Any]) -> tuple[ChatResult, str]:
        """处理一次聊天请求，返回（结果，实际用的模型别名）。出错时抛出（由接口转成 HTTP 错误）。"""
        if int(body.get("n") or 1) != 1:
            raise HTTPException(400, "Only n=1 is supported.")
        messages: list[Message] = list(body.get("messages") or [])
        if not messages:
            raise HTTPException(400, "messages is required.")
        response_format = body.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            messages = _with_schema(messages, response_format.get("json_schema") or {})
            response_format = {"type": "json_object"}
        max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
        primary = self.llm.settings.roles[self.role].model
        tools = body.get("tools") or None
        # 没写时按 OpenAI 的默认（true）：外部系统都按它写；通义不设为 true 时一次只回一个工具调用
        parallel = body.get("parallel_tool_calls")
        kwargs: dict[str, Any] = {
            "temperature": body.get("temperature"),
            "max_tokens": int(max_tokens) if max_tokens else None,
            "response_format": response_format,
            "tools": tools,
            "tool_choice": body.get("tool_choice") if tools else None,
            "parallel_tool_calls": (parallel is None or bool(parallel)) if tools else None,
            "cache_scope": f"gateway|{bucket}",
            # 解析不了的 JSON 回复之后，同样的请求（外部系统的重试）不读缓存
            "retry_bad_json": True,
        }
        json_mode = isinstance(response_format, dict) and response_format.get("type") in (
            "json_object",
            "json_schema",
        )
        requested = str(body.get("model") or "")
        with start_trace("gateway", bucket=bucket) as trace:
            try:
                result, alias, fallback_used = self._call(messages, kwargs, primary)
            except Exception as exc:  # 包括备用模型也失败的情况
                self._count_error(
                    bucket,
                    requested,
                    refused=isinstance(exc, ContentFilteredError),
                    rate_limited=isinstance(exc, RateLimitError),
                )
                raise
        self._count_chat(bucket, requested, trace, fallback_used)
        self._count_reply(bucket, result, tools is not None, json_mode)
        return result, alias

    def _call(
        self, messages: list[Message], kwargs: dict[str, Any], primary: str
    ) -> tuple[ChatResult, str, bool]:
        """先用主模型；内容审核误拒、超时或 JSON 生成被中止时换备用模型。

        限流不换模型（RateLimitError 原样抛出，由接口回 429）：换了模型就不是同一个协议了。"""
        try:
            return self.llm.chat(self.role, messages, **kwargs), primary, False
        except (ContentFilteredError, APITimeoutError, BadRequestError) as exc:
            aborted = "Model output became abnormal" in str(exc)
            usable = self.fallback is not None and self.fallback != primary
            if not usable or (isinstance(exc, BadRequestError) and not aborted):
                raise
        result = self.llm.chat(self.role, messages, model=self.fallback, **kwargs)
        return result, str(self.fallback), True

    def _chat_response(self, bucket: str, body: dict[str, Any]) -> JSONResponse | StreamingResponse:
        try:
            result, alias = self.chat(bucket, body)
        except HTTPException:
            raise
        except RateLimitError as exc:  # LLM 客户端已经按 SDK 的退避重试过：网关不再重试
            return _rate_limited(str(exc), _retry_after(exc.response.headers))
        except (ContentFilteredError, BadRequestError) as exc:
            return _error(400, str(exc), "invalid_request_error")
        except (APITimeoutError, CueMemError) as exc:
            return _error(502, f"{type(exc).__name__}: {exc}", "upstream_error")
        except APIError as exc:  # 其余的上游错误：5xx、连不上、余额不足（402）等
            return _error(502, f"{type(exc).__name__}: {exc}", "upstream_error")
        completion = _completion(result, alias)
        if body.get("stream"):
            return StreamingResponse(_as_stream(completion), media_type="text/event-stream")
        return JSONResponse(completion)

    # ---------- 向量 ----------

    def _embeddings_response(self, bucket: str, body: dict[str, Any]) -> JSONResponse:
        raw = body.get("input")
        texts = [raw] if isinstance(raw, str) else list(raw or [])
        if not all(isinstance(t, str) for t in texts):
            raise HTTPException(400, "input must be a string or a list of strings (no token ids).")
        failure: EmbeddingAPIError | None = None
        with start_trace("gateway", bucket=bucket) as trace:
            try:
                batch = self.embedder.embed(texts, embed_kind(bucket))
            except EmbeddingAPIError as exc:  # 向量器自己已经按退避重试过：网关不再重试
                failure = exc
        from cuemem.evals.runner import usage_from_trace

        usage = usage_from_trace(trace)  # 失败之前已经完成的分批也花了钱
        with self._lock:
            entry = self._bucket(bucket)
            entry.embedding_calls += 1
            entry.embedded_texts += len(texts) if failure is None else 0
            entry.embedding_tokens += int(usage["embedding_tokens"])
            entry.embedding_errors += int(failure is not None)
            entry.rate_limited += int(failure is not None and failure.status == 429)
            entry.cost += float(usage["cost"])
            entry.spent += float(usage["spent"])
        if failure is not None:
            if failure.status == 429:
                return _rate_limited(str(failure), failure.retry_after)
            return _error(502, f"{type(failure).__name__}: {failure}", "upstream_error")
        dimensions = body.get("dimensions")
        data = []
        for i, row in enumerate(batch.dense):
            vector = row.tolist()
            if dimensions and int(dimensions) < len(vector):  # 截断后不再归一化：调用方自己处理
                vector = vector[: int(dimensions)]
            data.append({"object": "embedding", "index": i, "embedding": vector})
        tokens = int(usage["embedding_tokens"])
        return JSONResponse(
            {
                "object": "list",
                "data": data,
                "model": self.embedder.name,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
            }
        )

    # ---------- 统计 ----------

    def _bucket(self, bucket: str) -> BucketUsage:
        return self._usage.setdefault(bucket, BucketUsage())

    def _count_chat(self, bucket: str, requested: str, trace: Any, fallback: bool) -> None:
        from cuemem.evals.runner import usage_from_trace

        usage = usage_from_trace(trace)
        refreshes = sum(1 for e in trace.events if e["kind"] == "llm_cache_refresh")
        with self._lock:
            entry = self._bucket(bucket)
            entry.chat_calls += int(usage["llm_calls"])
            entry.cache_hits += int(usage["llm_cache_hits"])
            entry.fallbacks += int(fallback)
            entry.json_refreshes += refreshes
            entry.input_tokens += int(usage["input_tokens"])
            entry.output_tokens += int(usage["output_tokens"])
            entry.cached_input_tokens += int(usage["cached_input_tokens"])
            entry.cost += float(usage["cost"])
            entry.spent += float(usage["spent"])
            entry.latency_ms += float(usage["latency_ms"] or 0.0)
            entry.models_requested[requested] = entry.models_requested.get(requested, 0) + 1

    def _count_reply(
        self, bucket: str, result: ChatResult, with_tools: bool, json_mode: bool = False
    ) -> None:
        calls = len(result.tool_calls)
        with self._lock:
            entry = self._bucket(bucket)
            entry.truncated_replies += int(result.finish_reason == "length")
            entry.bad_json_replies += int(json_mode and unparsable_json(result))
            if with_tools:
                entry.tool_requests += 1
                entry.tool_calls += calls
                entry.multi_tool_replies += int(calls > 1)
                entry.no_tool_replies += int(calls == 0)

    def _count_error(
        self, bucket: str, requested: str, *, refused: bool, rate_limited: bool = False
    ) -> None:
        with self._lock:
            entry = self._bucket(bucket)
            entry.errors += 1
            entry.refusals += int(refused)
            entry.rate_limited += int(rate_limited)
            entry.models_requested[requested] = entry.models_requested.get(requested, 0) + 1


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "The request body must be JSON.") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "The request body must be a JSON object.")
    return body


async def _in_thread(function: Any, *args: Any) -> Any:
    """同步的 LLM 客户端放到线程池里跑，不阻塞事件循环（并发由客户端按供应商限流）。"""
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(function, *args)


def embed_kind(bucket: str) -> EmbedKind:
    """桶里的向量请求按查询还是文档向量化：检索阶段的桶（<系统>.search.<空间>）是查询。

    通义的向量模型分 text_type=query / document（本机的 bge 没有检索指令时两者相同）；外部系统
    检索时只向量化问题，与 CueMem 的检索、M0 的做法一致。写入阶段（包括写入时查已有记忆）按文档。"""
    parts = bucket.split(".")
    return "query" if len(parts) > 2 and parts[1] == SEARCH_STAGE else "document"


def _with_schema(messages: list[Message], json_schema: dict[str, Any]) -> list[Message]:
    """json_schema → json_object：把 schema 写进系统提示
    （DeepSeek 的 JSON 模式要求提示里有 json 字样）。"""
    schema = json_schema.get("schema") or json_schema
    instruction = (
        "Output format: reply with one JSON object only, without markdown fences, conforming to "
        f"this JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    if (
        messages
        and messages[0].get("role") == "system"
        and isinstance(messages[0].get("content"), str)
    ):
        first = {**messages[0], "content": f"{messages[0]['content']}\n\n{instruction}"}
        return [first, *messages[1:]]
    return [{"role": "system", "content": instruction}, *messages]


def _completion(result: ChatResult, alias: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": result.content}
    if result.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in result.tool_calls
        ]
    usage = result.usage
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": alias,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason
                or ("tool_calls" if result.tool_calls else "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens},
        },
    }


def _as_stream(completion: dict[str, Any]) -> Iterator[bytes]:
    """完整结果按 SSE 发出：一个带全部内容的 delta 块、一个带用量的结束块，然后 [DONE]。"""
    choice = completion["choices"][0]
    delta: dict[str, Any] = {"role": "assistant", "content": choice["message"].get("content")}
    if choice["message"].get("tool_calls"):
        delta["tool_calls"] = [
            {**call, "index": i} for i, call in enumerate(choice["message"]["tool_calls"])
        ]
    base = {k: completion[k] for k in ("id", "created", "model")}
    chunks = [
        {**base, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}]},
        {
            **base,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}],
            "usage": completion["usage"],
        },
    ]
    for chunk in chunks:
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message[:1000], "type": kind}}, status)


def _rate_limited(message: str, retry_after: str | None) -> JSONResponse:
    """429，照 OpenAI 的格式；上游给了 Retry-After 就带上（OpenAI SDK 照它等待后重试）。"""
    headers = {"Retry-After": retry_after} if retry_after else None
    body = {"error": {"message": message[:1000], "type": "rate_limit_error", "code": "rate_limit"}}
    return JSONResponse(body, 429, headers=headers)


def _retry_after(headers: Any) -> str | None:
    """上游响应的 Retry-After（秒数或 HTTP 日期，原样转发）；只有 retry-after-ms 时换算成秒。"""
    value = headers.get("retry-after")
    if value:
        return str(value)
    millis = headers.get("retry-after-ms")
    try:
        return str(max(1, math.ceil(float(millis) / 1000))) if millis else None
    except ValueError:
        return None
