"""OpenAI 兼容的 LLM 客户端：一套代码接入 DeepSeek 与通义千问。

职责：
- 按角色选模型（roles → models），拼请求参数，处理供应商差异（例如关闭思考模式）；
- 按供应商限制并发；
- 分时价格：高峰时段需要真实调用时，按策略暂停到高峰结束（见 llm/peak.py）；
- 本地响应缓存：相同请求只调用一次 API，既省钱也让实验可复现；
- 用量记账与 trace 事件；
- 结构化输出：JSON 模式 + Schema 提示 + pydantic 校验，不合格时把错误反馈给模型重试。
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast

import httpx2
from openai import APITimeoutError, BadRequestError, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from cuemem.config import ModelSpec, Settings
from cuemem.core.trace import record_event
from cuemem.errors import (
    CacheMissError,
    ContentFilteredError,
    MissingAPIKeyError,
    StructuredOutputError,
)
from cuemem.llm.cache import ResponseCache, request_key
from cuemem.llm.ledger import Ledger, UsageRecord, chat_cost
from cuemem.llm.peak import in_peak, peak_end
from cuemem.llm.types import ChatResult, ToolCall, Usage

Message = dict[str, Any]
PeakPolicy = Literal["wait", "warn", "allow"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass
class Structured[T: BaseModel]:
    """结构化输出的结果：解析后的对象、最后一次调用的原始结果、尝试次数。"""

    value: T
    result: ChatResult
    attempts: int


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        cache: ResponseCache | None = None,
        ledger: Ledger | None = None,
        http_client: httpx2.Client | None = None,
        peak_policy: PeakPolicy | None = None,
        on_peak_wait: Callable[[str, datetime], None] | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.ledger = ledger
        self.peak_policy: PeakPolicy = peak_policy or settings.peak_policy
        self._on_peak_wait = on_peak_wait  # 开始等待时通知调用方（每段高峰每个供应商一次）
        self._clock = clock  # 测试时注入假时钟；None 表示真实时间
        self._sleep = sleep
        self._http_client = http_client  # 测试时注入模拟的传输层
        self._clients: dict[str, OpenAI] = {}
        self._semaphores = {
            name: threading.BoundedSemaphore(p.max_concurrency)
            for name, p in settings.providers.items()
        }
        self._lock = threading.Lock()
        self._notified: set[tuple[str, datetime]] = set()
        self._inflight: dict[str, threading.Event] = {}  # 正在请求中的缓存键（相同请求合并）
        self._bad_json: set[str] = set()  # 上次拿到解析不了的 JSON 的缓存键（见 retry_bad_json）

    def chat(
        self,
        role: str,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        use_cache: bool = True,
        cache_scope: str | None = None,
        retry_bad_json: bool = False,
    ) -> ChatResult:
        """调用一次聊天补全。role 决定默认模型与参数，model 可显式指定模型别名。

        parallel_tool_calls 只在带工具时发出（通义不设为 true 时一次只回一个工具调用，N12）。
        cache_scope 标记缓存条目所属的记忆空间，遗忘时据此清理。

        retry_bad_json：给照 OpenAI 的方式重试的外部系统用（评测网关，N12）。JSON 模式的回复解析
        不了时记下这个请求；同样的请求再来（外部系统的重试）时不读缓存，重新调用并覆盖缓存里的
        坏回复——否则每次重试都从缓存里拿到同一个坏回复。缓存键不变，能解析的回复照常命中缓存；
        解析不了的回复照常缓存（外部系统换一个请求接着修时，重跑能原样复现）。CueMem 自己的
        结构化输出（structured）重试时带上校验错误，请求本来就不同，不用它。
        """
        alias, spec, params = self._prepare(
            role,
            model,
            temperature,
            max_tokens,
            response_format,
            tools,
            tool_choice,
            parallel_tool_calls,
        )
        caching = use_cache and self.cache is not None and self.settings.llm_cache_mode != "off"
        json_guard = retry_bad_json and _json_mode(params)
        key = ""
        if caching:
            assert self.cache is not None
            key = request_key(
                {
                    "provider": spec.provider,
                    "model": spec.model_id,
                    "messages": messages,
                    "params": params,
                    "extra_body": spec.extra_body,
                }
            )
            # 上次同样的请求拿到的是解析不了的 JSON：这是外部系统的重试，不读缓存
            # （只读缓存模式下不调用 API，照常读缓存）
            refresh = (
                json_guard
                and self.settings.llm_cache_mode != "read_only"
                and self._bad_json_seen(key)
            )
            while True:
                if not refresh:
                    hit = self._from_cache(key, role, alias, spec)
                    if hit is not None:
                        if json_guard:
                            self._note_json(key, hit)
                        return hit
                    if self.settings.llm_cache_mode == "read_only":
                        raise CacheMissError(
                            f"read_only 模式下缓存未命中：role={role}, model={alias}"
                        )
                # 相同请求合并（single-flight）：别的线程正在请求同样的内容时，等它的结果，
                # 不重复花钱；它失败了（没写进缓存）就由本线程接着请求
                waiting = self._claim(key)
                if waiting is None:
                    break
                waiting.wait()
                refresh = False  # 别的线程刚请求过同样的内容：缓存里是新的回复，直接用
            # 真的要重新调用时才记：同时到的同样重试里，等别的线程结果的那些读的是缓存
            if refresh:
                record_event("llm_cache_refresh", role=role, model=alias)
        try:
            result = self._call_api(
                role, alias, spec, messages, params, key if caching else "", cache_scope
            )
        finally:
            if caching:
                self._release(key)
        if json_guard and caching:
            self._note_json(key, result)
        return result

    def _bad_json_seen(self, key: str) -> bool:
        with self._lock:
            return key in self._bad_json

    def _note_json(self, key: str, result: ChatResult) -> None:
        """记下（或清掉）这个请求"上次拿到的是解析不了的 JSON"。"""
        with self._lock:
            if unparsable_json(result):
                self._bad_json.add(key)
            else:
                self._bad_json.discard(key)

    def _from_cache(self, key: str, role: str, alias: str, spec: ModelSpec) -> ChatResult | None:
        assert self.cache is not None
        cached = self.cache.get(key)
        if cached is None:
            return None
        result = ChatResult.model_validate(
            {**cached, "role": role, "from_cache": True, "latency_ms": 0.0}
        )
        self._account(role, alias, spec, result)
        return result

    def _claim(self, key: str) -> threading.Event | None:
        """登记"本线程正在请求 key"；已有别的线程在请求时，返回它完成时会触发的事件。"""
        with self._lock:
            event = self._inflight.get(key)
            if event is None:
                self._inflight[key] = threading.Event()
            return event

    def _release(self, key: str) -> None:
        with self._lock:
            event = self._inflight.pop(key, None)
        if event is not None:
            event.set()

    def _call_api(
        self,
        role: str,
        alias: str,
        spec: ModelSpec,
        messages: list[Message],
        params: dict[str, Any],
        key: str,
        cache_scope: str | None,
    ) -> ChatResult:
        self._peak_gate(spec.provider, role)
        create = cast(Any, self._client(spec.provider).chat.completions.create)
        started = time.perf_counter()
        with self._semaphores[spec.provider]:
            try:
                completion = create(
                    model=spec.model_id,
                    messages=messages,
                    extra_body=spec.extra_body or None,
                    **params,
                )
            except BadRequestError as exc:
                if _is_content_filter(exc):
                    record_event("llm_filtered", role=role, model=alias)
                    raise ContentFilteredError(f"{alias} 的内容审核拒绝了请求：{exc}") from exc
                raise
        latency_ms = (time.perf_counter() - started) * 1000
        result = _parse_completion(
            cast(ChatCompletion, completion),
            alias=alias,
            model_id=spec.model_id,
            role=role,
            latency_ms=latency_ms,
        )
        # 空回复不缓存：它往往是偶发问题，缓存后会被永久复现
        if key and (result.content or result.tool_calls):
            assert self.cache is not None
            self.cache.put(
                key,
                alias,
                result.model_dump(mode="json", exclude={"role", "from_cache", "latency_ms"}),
                scope=cache_scope,
            )
        self._account(role, alias, spec, result)
        return result

    def structured[T: BaseModel](
        self,
        role: str,
        messages: list[Message],
        schema: type[T],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        max_attempts: int = 3,
        cache_scope: str | None = None,
        fallback_model: str | None = None,
    ) -> Structured[T]:
        """让模型输出符合 schema 的 JSON 并解析；不合格时把校验错误反馈给模型重试。

        fallback_model：内容审核拒绝或请求超时时改用的模型（通常是另一家供应商的）。
        超时多半不是网络问题：模型在某条输入上生成失控、迟迟不结束，重发同样的请求还会超时。
        """
        try:
            return self._structured(
                role, messages, schema, model, max_tokens, max_attempts, cache_scope
            )
        except (ContentFilteredError, APITimeoutError) as exc:
            primary = model or (
                self.settings.roles[role].model if role in self.settings.roles else None
            )
            if fallback_model is None or fallback_model == primary:
                raise
            record_event("llm_fallback", role=role, model=fallback_model, reason=type(exc).__name__)
            return self._structured(
                role, messages, schema, fallback_model, max_tokens, max_attempts, cache_scope
            )

    def _structured[T: BaseModel](
        self,
        role: str,
        messages: list[Message],
        schema: type[T],
        model: str | None,
        max_tokens: int | None,
        max_attempts: int,
        cache_scope: str | None,
    ) -> Structured[T]:
        msgs = _with_schema_instruction(messages, schema)
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            try:
                result = self.chat(
                    role,
                    msgs,
                    model=model,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    cache_scope=cache_scope,
                )
            except BadRequestError as exc:
                # QwenCloud 在 JSON 模式下偶尔中止生成（"Model output became abnormal ..."），
                # 同样的请求重发多半还会中止：补一句提示，换一个请求再试
                if not _is_aborted_json(exc):
                    raise
                record_event("llm_json_aborted", role=role, attempt=attempt)
                last_error = "供应商中止了 JSON 生成"
                msgs = [
                    *msgs,
                    {"role": "user", "content": "Reply with one short, valid JSON object."},
                ]
                continue
            text = _strip_code_fence(result.content or "")
            if not text:
                last_error = "模型返回了空内容"
                msgs = [
                    *msgs,
                    {
                        "role": "user",
                        "content": "Your reply was empty. Reply with the JSON object.",
                    },
                ]
                continue
            try:
                return Structured(
                    value=schema.model_validate_json(text), result=result, attempts=attempt
                )
            except ValidationError as exc:
                last_error = str(exc)
                feedback = (
                    f"That JSON does not match the schema: {_brief_errors(exc)}. "
                    "Reply with a corrected JSON object only."
                )
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": feedback},
                ]
        raise StructuredOutputError(f"{max_attempts} 次尝试后仍未得到合法 JSON：{last_error}")

    def vision(self, role: str, model: str | None = None) -> bool:
        """这个角色（或指定的模型）能不能看图。不能时调用方不附图片（本地纯文本模型，N12）。"""
        role_cfg = self.settings.roles.get(role)
        alias = model or (role_cfg.model if role_cfg else None)
        spec = self.settings.models.get(alias) if alias is not None else None
        return spec is not None and spec.vision

    def list_models(self, provider: str) -> list[str]:
        """列出供应商可用的模型 ID，用于核对配置（不是所有供应商都支持）。"""
        return sorted(m.id for m in self._client(provider).models.list())

    def _prepare(
        self,
        role: str,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        parallel_tool_calls: bool | None = None,
    ) -> tuple[str, ModelSpec, dict[str, Any]]:
        role_cfg = self.settings.roles.get(role)
        alias = model or (role_cfg.model if role_cfg else None)
        if alias is None:
            raise KeyError(f"角色 {role} 没有配置模型，调用时也没有指定 model")
        spec = self.settings.models[alias]
        params: dict[str, Any] = {}
        if not spec.fixed_temperature:  # 推理模型只接受默认温度，不带这个参数（N12）
            params["temperature"] = (
                temperature
                if temperature is not None
                else (role_cfg.temperature if role_cfg else 0.0)
            )
        limit = (
            max_tokens if max_tokens is not None else (role_cfg.max_tokens if role_cfg else None)
        )
        if limit is not None:
            params[spec.max_tokens_param] = limit
        if response_format is not None:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if tools and parallel_tool_calls is not None:
            params["parallel_tool_calls"] = parallel_tool_calls
        return alias, spec, params

    def _client(self, provider: str) -> OpenAI:
        with self._lock:
            if provider not in self._clients:
                cfg = self.settings.providers[provider]
                api_key = cfg.api_key()
                if api_key is None:
                    raise MissingAPIKeyError(
                        f"缺少环境变量 {cfg.api_key_env}（{provider}），"
                        "请运行 uv run cuemem configure"
                    )
                self._clients[provider] = OpenAI(
                    api_key=api_key,
                    base_url=cfg.base_url,
                    timeout=cfg.timeout_s,
                    max_retries=cfg.max_retries,  # SDK 自带对 429 / 5xx / 网络错误的指数退避重试
                    http_client=self._http_client,
                )
            return self._clients[provider]

    def _now(self) -> datetime:
        return self._clock() if self._clock is not None else datetime.now().astimezone()

    def _peak_gate(self, provider: str, role: str) -> None:
        """高峰时段的真实调用：wait 策略下暂停到高峰结束；命中缓存的请求不会走到这里。"""
        cfg = self.settings.providers[provider].peak_hours
        if cfg is None or self.peak_policy == "allow":
            return
        end = peak_end(cfg, self._now())
        if end is None:
            return
        if self.peak_policy == "warn":
            record_event("peak_call", provider=provider, role=role)
            return
        with self._lock:
            first = (provider, end) not in self._notified
            self._notified.add((provider, end))
        if first and self._on_peak_wait is not None:
            self._on_peak_wait(provider, end)
        record_event("peak_wait", provider=provider, role=role, until=end.isoformat())
        while True:  # 分段睡眠：时钟被调整或高峰配置变化时也能正确醒来
            now = self._now()
            remaining = (end - now).total_seconds()
            if remaining <= 0 or not in_peak(cfg, now):
                return
            self._sleep(min(remaining, 60.0))

    def _account(self, role: str, alias: str, spec: ModelSpec, result: ChatResult) -> None:
        nominal, discounted = chat_cost(spec.price, result.usage)
        peak = self.settings.providers[spec.provider].peak_hours
        if peak is not None and not result.from_cache and not in_peak(peak, self._now()):
            discounted *= peak.off_peak_factor  # 非高峰时段的实际价格（标价按高峰计）
        usage = result.usage
        if self.ledger is not None:
            self.ledger.record(
                UsageRecord(
                    ts=datetime.now(),
                    run_id=self.ledger.run_id,
                    kind="chat",
                    role=role,
                    provider=spec.provider,
                    model=alias,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    cost=nominal,
                    cost_cached=discounted,
                    currency=spec.currency,
                    from_cache=result.from_cache,
                    latency_ms=result.latency_ms,
                )
            )
        record_event(
            "llm",
            role=role,
            model=alias,
            from_cache=result.from_cache,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            cost=round(nominal, 6),
            spent=0.0 if result.from_cache else round(discounted, 6),
            latency_ms=round(result.latency_ms, 1),
        )


def _json_mode(params: dict[str, Any]) -> bool:
    response_format = params.get("response_format")
    return isinstance(response_format, dict) and response_format.get("type") in (
        "json_object",
        "json_schema",
    )


def unparsable_json(result: ChatResult) -> bool:
    """JSON 模式的回复解析不了：去掉 markdown 围栏后仍不是合法 JSON（空回复也算）。

    按 json.loads 的默认（严格）规则：字符串里有原样的换行等控制字符也算解析不了，外部系统
    多数照这个规则解析。带工具调用的回复不算（内容不是给调用方解析的）。"""
    if result.tool_calls:
        return False
    try:
        json.loads(_strip_code_fence(result.content or ""))
    except ValueError:  # 包括 json.JSONDecodeError
        return True
    return False


def _is_aborted_json(exc: BadRequestError) -> bool:
    """QwenCloud 在 JSON 模式下中止生成时的报错（400 invalid_parameter_error）。"""
    return "Model output became abnormal" in str(exc)


def _is_content_filter(exc: BadRequestError) -> bool:
    """内容审核拒绝的特征。

    QwenCloud 返回 data_inspection_failed；DeepSeek 返回 Content Exists Risk。
    """
    text = str(exc)
    return any(
        marker in text
        for marker in ("data_inspection_failed", "inappropriate content", "Content Exists Risk")
    )


def _parse_completion(
    completion: ChatCompletion, *, alias: str, model_id: str, role: str, latency_ms: float
) -> ChatResult:
    choice = completion.choices[0] if completion.choices else None
    message = choice.message if choice else None
    tool_calls: list[ToolCall] = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        if function is not None:
            tool_calls.append(
                ToolCall(id=call.id, name=function.name, arguments=function.arguments or "")
            )
    return ChatResult(
        content=getattr(message, "content", None),
        reasoning_content=getattr(message, "reasoning_content", None),
        tool_calls=tool_calls,
        finish_reason=getattr(choice, "finish_reason", None),
        usage=_parse_usage(completion.usage),
        model_alias=alias,
        model_id=model_id,
        role=role,
        latency_ms=latency_ms,
    )


def _parse_usage(usage: Any) -> Usage:
    """统一用量字段：DeepSeek 报 prompt_cache_hit_tokens，通义报 prompt_tokens_details。"""
    if usage is None:
        return Usage()
    cached = getattr(usage, "prompt_cache_hit_tokens", None)
    if cached is None:
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
    return Usage(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_input_tokens=cached or 0,
        reasoning_tokens=reasoning or 0,
    )


def _with_schema_instruction(messages: list[Message], schema: type[BaseModel]) -> list[Message]:
    """把输出格式要求追加到系统提示中（DeepSeek 的 JSON 模式要求提示里出现 "json" 字样）。"""
    schema_text = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    instruction = (
        "Output format: reply with one JSON object only, without markdown fences, "
        f"conforming to this JSON Schema:\n{schema_text}"
    )
    if messages and messages[0].get("role") == "system":
        first = {**messages[0], "content": f"{messages[0]['content']}\n\n{instruction}"}
        return [first, *messages[1:]]
    return [{"role": "system", "content": instruction}, *messages]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    match = _FENCE_RE.match(text)
    return match.group(1).strip() if match else text


def _brief_errors(exc: ValidationError, limit: int = 3) -> str:
    parts = []
    for err in exc.errors()[:limit]:
        location = ".".join(str(p) for p in err["loc"]) or "<root>"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)
