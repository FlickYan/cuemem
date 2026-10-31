"""确定性 ID 与内容哈希：同样的输入永远得到同样的 ID，写入因此是幂等的。"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from datetime import datetime

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


class _Ticker:
    """严格递增的纳秒计数：时钟粒度粗（Windows）、同一纳秒里连着取时也不重复、不倒退。"""

    def __init__(self) -> None:
        self._last = 0
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            self._last = max(self._last + 1, time.time_ns())
            return self._last


_tick = _Ticker()


def sortable_session_id(prefix: str, when: datetime) -> str:
    """生成的会话 ID：前缀、开始时间、严格递增的计数，最后 4 位随机数（几个进程同时写也不撞）。

    巩固按（开始时间，会话 ID）的先后对账。开始时间相同（显式给了同样的时间）时，同一进程里
    先生成的会话排在前面，后写的才是最新的；以前 ID 以随机数结尾，先后是随机的。"""
    return f"{prefix}-{when:%Y%m%d-%H%M%S}-{_tick():020d}-{uuid.uuid4().hex[:4]}"
