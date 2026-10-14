"""测试替身：模拟 OpenAI 兼容的聊天与向量接口、DashScope 原生向量接口，并记录收到的请求。"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOY_CONVERSATION = PROJECT_ROOT / "examples" / "toy_conversation.json"

# 回复可以是字符串，也可以是"由请求体生成回复"的函数；函数返回 dict 时表示工具调用：
# {"tool_calls": [{"id": "c1", "name": "memory_search", "arguments": "{...}"}]}
ChatReply = str | Callable[[dict[str, Any]], str | dict[str, Any]]
REJECT = "__REJECT__"  # 回复函数返回它时，模拟内容审核拒绝（HTTP 400）
ABORT_JSON = "__ABORT_JSON__"  # 模拟 QwenCloud 在 JSON 模式下中止生成（HTTP 400）
TIMEOUT = "__TIMEOUT__"  # 模拟请求超时（生成失控、迟迟不结束）


def chat_completion_body(
    content: str | None,
    *,
    deepseek: bool,
    prompt_tokens: int = 120,
    completion_tokens: int = 30,
    cached_tokens: int = 0,
    tool_calls: list[dict[str, Any]] | None = None,
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
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in tool_calls
        ]
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": usage,
    }


def sse_body(content: str) -> bytes:
    """流式聊天补全的响应：每个词一个 data 块，最后是用量与 [DONE]。"""
    chunks = []
    for i, word in enumerate(content.split(" ")):
        piece = word if i == 0 else " " + word
        delta = {"role": "assistant", "content": piece} if i == 0 else {"content": piece}
        chunks.append(
            {
                "id": "c",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta}],
            }
        )
    chunks.append(
        {
            "id": "c",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }
    )
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


class FakeAPI:
    """chat_replies 按顺序消费，元素可以是字符串或"由请求体生成回复"的函数；只剩一个时重复使用。"""

    def __init__(self) -> None:
        self.chat_replies: list[ChatReply] = ["OK"]
        self.chat_requests: list[dict[str, Any]] = []
        self.embed_requests: list[dict[str, Any]] = []
        self.rerank_requests: list[dict[str, Any]] = []
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
            if content == TIMEOUT:
                raise httpx2.ReadTimeout("timed out", request=request)
            deepseek = "deepseek" in request.url.host
            if isinstance(content, dict):  # 工具调用
                return httpx2.Response(
                    200,
                    json=chat_completion_body(
                        content.get("content"), deepseek=deepseek, tool_calls=content["tool_calls"]
                    ),
                )
            if content == ABORT_JSON:
                return httpx2.Response(
                    400,
                    json={
                        "error": {
                            "code": "invalid_parameter_error",
                            "message": "Model output became abnormal while generating a JSON "
                            "response for response_format.",
                            "type": "invalid_request_error",
                        }
                    },
                )
            if content == REJECT:  # 模拟供应商的内容审核拒绝（QwenCloud 的报错格式）
                return httpx2.Response(
                    400,
                    json={
                        "error": {
                            "code": "data_inspection_failed",
                            "message": "Input text data may contain inappropriate content.",
                            "type": "data_inspection_failed",
                        }
                    },
                )
            if body.get("stream"):  # 流式（N12 的记忆代理）：按词切成几个 SSE 块
                return httpx2.Response(
                    200,
                    content=sse_body(content),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx2.Response(
                200,
                json=chat_completion_body(
                    content, deepseek=deepseek, cached_tokens=self.cached_tokens
                ),
            )
        if "text-rerank" in request.url.path:
            # 重排：按问题与文本共有的词数打分，结果确定、可预期
            self.rerank_requests.append(body)
            query = set(body["input"]["query"].lower().split())
            docs = body["input"]["documents"]
            results = [
                {"index": i, "relevance_score": len(query & set(doc.lower().split())) / 10}
                for i, doc in enumerate(docs)
            ]
            results.sort(key=lambda r: -r["relevance_score"])
            tokens = sum(len(d.split()) for d in docs)
            return httpx2.Response(
                200, json={"output": {"results": results}, "usage": {"total_tokens": tokens}}
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
        if request.url.path.endswith("/embeddings"):  # OpenAI 兼容的向量接口（N12）
            self.embed_requests.append(body)
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            data = [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": [float(len(text) % 7 + 1)] + [0.5] * (self.embed_dim - 1),
                }
                for i, text in enumerate(texts)
            ]
            data.reverse()  # 故意打乱顺序，检验按 index 排序
            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "data": data,
                    "model": body["model"],
                    "usage": {"prompt_tokens": 5 * len(texts), "total_tokens": 5 * len(texts)},
                },
            )
        return httpx2.Response(404, json={"error": "not found"})
