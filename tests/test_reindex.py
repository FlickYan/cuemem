"""向量库按数据库补齐（cuemem reindex、doctor --vectors，第五轮体检 P21）。"""

from __future__ import annotations

from datetime import datetime

import pytest

from cuemem.core.models import Fact, MemoryUnit
from cuemem.core.timeutil import to_epoch
from cuemem.write.consolidate import unit_payload


def test_unit_payload_keeps_each_write_path_shape() -> None:
    """在线写入与巩固以前各写一份 payload、键的顺序各不相同：合成 unit_payload 之后一点都不变
    （键的顺序进了本地向量库的 pickle）。"""
    ts = datetime(2025, 3, 1, 9, 30)
    epoch = to_epoch(ts)

    def unit(kind: str) -> MemoryUnit:
        return MemoryUnit(
            id=f"u-{kind}",
            space_id="s",
            kind=kind,
            session_id="s1",
            text="t",
            ts=ts,
            source_turn_ids=[],
        )

    for kind in ("round", "image", "image_iso"):
        assert list(unit_payload(unit(kind)).items()) == [
            ("kind", kind),
            ("session_id", "s1"),
            ("ts", epoch),
        ]
    for kind in ("summary", "entity"):
        assert list(unit_payload(unit(kind)).items()) == [
            ("session_id", "s1"),
            ("ts", epoch),
            ("kind", kind),
        ]
    fact = Fact(
        id="u-fact",
        space_id="s",
        subject_id="e1",
        subject="User",
        predicate="said",
        statement="t",
        object="x",
        recorded_at=ts,
        session_id="s1",
        source_turn_ids=[],
        invalidated_at=ts,
    )
    assert list(unit_payload(unit("fact"), fact).items()) == [
        ("session_id", "s1"),
        ("ts", epoch),
        ("kind", "fact"),
        ("current", False),
        ("subject_id", "e1"),
    ]
    with pytest.raises(ValueError, match="u-fact"):
        unit_payload(unit("fact"))
