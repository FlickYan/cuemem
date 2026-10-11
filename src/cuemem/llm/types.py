"""LLM 调用相关的数据结构。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0  # 供应商侧 prompt cache 命中的输入 token
    reasoning_tokens: int = 0  # 思考模式下的推理 token（已包含在 output_tokens 中）


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，由调用方解析


class ChatResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    content: str | None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model_alias: str
    model_id: str
    role: str | None = None
    from_cache: bool = False  # 是否命中本地响应缓存（没有实际调用 API）
    latency_ms: float = 0.0
