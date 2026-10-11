"""测试替身：模拟 OpenAI 兼容的聊天接口与 DashScope 原生向量接口，并记录收到的请求。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOY_CONVERSATION = PROJECT_ROOT / "examples" / "toy_conversation.json"

ChatReply = str | Callable[[dict[str, Any]], str]


def chat_completion_body(
    content: str | None,
    *,
    deepseek: bool,
    prompt_tokens: int = 120,
    completion_tokens: int = 30,
    cached_tokens: int = 0,
) -> dict[str, Any]:
    """按两家供应商各自的 usage 字段格式构造聊天补全响应。"""
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if deepseek:
        usage["prompt_cache_hit_tokens"] = cached_tokens
        usage["prompt_cache_miss_tokens"] = prompt_tokens - cached_tokens
    else:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


class FakeAPI:
    """chat_replies 按顺序消费，元素可以是字符串或"由请求体生成回复"的函数；只剩一个时重复使用。"""

    def __init__(self) -> None:
        self.chat_replies: list[ChatReply] = ["OK"]
        self.chat_requests: list[dict[str, Any]] = []
        self.embed_requests: list[dict[str, Any]] = []
        self.embed_status: list[int] = []  # 预设的状态码序列，用于模拟限流和错误
        self.embed_retry_after: str | None = None  # 出错响应附带的 Retry-After 头
        self.cached_tokens = 0
        self.embed_dim = 8

    def client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content or b"{}")
        if request.url.path.endswith("/chat/completions"):
            self.chat_requests.append(body)
            reply = self.chat_replies.pop(0) if len(self.chat_replies) > 1 else self.chat_replies[0]
            content = reply(body) if callable(reply) else reply
            deepseek = "deepseek" in request.url.host
            return httpx2.Response(
                200,
                json=chat_completion_body(
                    content, deepseek=deepseek, cached_tokens=self.cached_tokens
                ),
            )
        if "text-embedding" in request.url.path:
            self.embed_requests.append(body)
            if self.embed_status:
                status = self.embed_status.pop(0)
                if status != 200:
                    headers = (
                        {"Retry-After": self.embed_retry_after} if self.embed_retry_after else {}
                    )
                    return httpx2.Response(
                        status, json={"code": "Error", "message": "simulated"}, headers=headers
                    )
            texts = body["input"]["texts"]
            embeddings = [
                {
                    "text_index": i,
                    "embedding": [float(len(text) % 7 + 1)] + [0.5] * (self.embed_dim - 1),
                    # 故意包含重复下标，检验合并逻辑
                    "sparse_embedding": [
                        {"index": 7, "value": 0.2, "token": "a"},
                        {"index": 7, "value": 0.9, "token": "a"},
                        {"index": 3, "value": 0.4, "token": "b"},
                    ],
                }
                for i, text in enumerate(texts)
            ]
            return httpx2.Response(
                200,
                json={
                    "output": {"embeddings": embeddings},
                    "usage": {"total_tokens": 5 * len(texts)},
                },
            )
        return httpx2.Response(404, json={"error": "not found"})
