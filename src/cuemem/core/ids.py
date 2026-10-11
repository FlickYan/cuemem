"""确定性 ID 与内容哈希：同样的输入永远得到同样的 ID，写入因此是幂等的。"""

from __future__ import annotations

import hashlib
import uuid

_POINT_NAMESPACE = uuid.UUID("6f1c5a3e-8d2b-4f7a-9c41-2b8e5d7a0c13")


def stable_id(*parts: object, length: int = 16) -> str:
    """由若干字段拼出稳定的短 ID。"""
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def point_uuid(unit_id: str) -> str:
    """向量库要求点 ID 是 UUID 或整数，这里由记忆单元 ID 确定性地派生。"""
    return str(uuid.uuid5(_POINT_NAMESPACE, unit_id))
