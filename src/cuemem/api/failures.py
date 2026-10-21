"""模型与向量供应商出错时回给调用方的说明（HTTP 接口与 MCP 共用）。

限流、余额不足、暂时不可用、超时、连不上、内容审核拒绝，各给一个状态码与一句说明；限流与暂时
不可用时带上上游的 Retry-After。以前 HTTP 接口把 OpenAI SDK 的异常变成没有说明的 500，MCP 只回
"Error executing tool <name>"：调用方分不清是 CueMem 坏了，还是该等一会儿再试。

LLM 客户端与向量器在抛出之前已经按各自的规则重试过（providers.*.max_retries、
embedding.max_attempts），这里不再重试。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import httpx2
from openai import APIConnectionError, APIStatusError, APITimeoutError

from cuemem.errors import ContentFilteredError, EmbeddingAPIError

Kind = Literal[
    "rate_limited", "no_balance", "unavailable", "timeout", "upstream_error", "content_filtered"
]

_ZH: dict[Kind, str] = {
    "rate_limited": "模型供应商限流，重试之后仍然被拒（HTTP 429），请稍后再试",
    "no_balance": "模型供应商的账户余额不足（HTTP 402），充值之后再试",
    "unavailable": "模型供应商暂时不可用（HTTP 503），请稍后再试",
    "timeout": "模型供应商迟迟没有回应（请求超时），请稍后再试",
    "upstream_error": "模型供应商出错",
    "content_filtered": "模型供应商的内容审核拒绝了这次请求，换个说法再试",
}
_EN: dict[Kind, str] = {
    "rate_limited": "The model provider is rate-limiting requests (HTTP 429). Try again later",
    "no_balance": "The model provider account has run out of balance (HTTP 402). Retrying will "
    "not help until it is topped up",
    "unavailable": "The model provider is temporarily unavailable (HTTP 503). Try again later",
    "timeout": "The model provider did not respond in time. Try again later",
    "upstream_error": "The model provider returned an error",
    "content_filtered": "The model provider's content filter rejected this request. Rephrase it "
    "and try again",
}


@dataclass(frozen=True)
class ProviderFailure:
    kind: Kind
    status: int  # 回给调用方的 HTTP 状态码
    detail: str  # 上游的报错
    retry_after: str | None = None  # 上游的 Retry-After（秒数或 HTTP 日期）

    def message(self) -> str:
        """HTTP 接口的说明（与别的报错一样是中文），后面附上上游的原话。"""
        wait = f"（约 {self.retry_after} 秒后）" if _seconds(self.retry_after) else ""
        return f"{_ZH[self.kind]}{wait}：{self.detail[:500]}"

    def for_model(self) -> str:
        """MCP 的说明（给调用方的模型看，英文）。"""
        wait = f" (in about {self.retry_after} s)" if _seconds(self.retry_after) else ""
        detail = f": {self.detail[:300]}" if self.kind == "upstream_error" else ""
        return f"{_EN[self.kind]}{wait}{detail}."


def provider_failure(exc: BaseException) -> ProviderFailure | None:
    """exc 是模型或向量供应商的错误时，回给调用方的状态码与说明；别的异常返回 None。"""
    detail = str(exc)
    if isinstance(exc, ContentFilteredError):
        return ProviderFailure("content_filtered", 422, detail)
    if isinstance(exc, APITimeoutError):
        return ProviderFailure("timeout", 504, detail)
    if isinstance(exc, APIConnectionError):
        return ProviderFailure("upstream_error", 502, detail)
    if isinstance(exc, APIStatusError):
        return _by_status(exc.status_code, detail, retry_after_header(exc.response.headers))
    if isinstance(exc, EmbeddingAPIError):
        if exc.status is not None:
            return _by_status(exc.status, detail, exc.retry_after)
        if isinstance(exc.__cause__, httpx2.TimeoutException):
            return ProviderFailure("timeout", 504, detail)
        return ProviderFailure("upstream_error", 502, detail)  # 连不上、返回的格式不对
    return None


def _by_status(status: int, detail: str, retry_after: str | None) -> ProviderFailure:
    if status == 429:
        return ProviderFailure("rate_limited", 429, detail, retry_after)
    if status == 402:
        return ProviderFailure("no_balance", 402, detail)
    if status == 503:
        return ProviderFailure("unavailable", 503, detail, retry_after)
    if status in (408, 504):
        return ProviderFailure("timeout", 504, detail)
    return ProviderFailure("upstream_error", 502, detail)  # 5xx、鉴权失败、请求不合法……


def retry_after_header(headers: Mapping[str, str]) -> str | None:
    """上游响应的 Retry-After（秒数或 HTTP 日期，原样转发）；只有 retry-after-ms 时换算成秒。"""
    value = headers.get("retry-after")
    if value:
        return str(value)
    millis = headers.get("retry-after-ms")
    try:
        return str(max(1, math.ceil(float(millis) / 1000))) if millis else None
    except (ValueError, OverflowError):
        return None


def _seconds(value: str | None) -> bool:
    return value is not None and value.isdigit()
