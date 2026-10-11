"""事实在上下文里的呈现：有效状态、版本链指向、消融模式。"""

from __future__ import annotations

from datetime import datetime

from cuemem.core.models import Fact, FactHit
from cuemem.read.facts import format_facts


def make(fid: str, statement: str, **extra: object) -> FactHit:
    fact = Fact(
        id=fid,
        space_id="s",
        subject_id="u",
        subject="User",
        predicate="works_at",
        object="x",
        statement=statement,
        recorded_at=extra.pop("recorded_at", datetime(2025, 3, 1)),  # type: ignore[arg-type]
        session_id="s1",
        source_turn_ids=[],
        **extra,  # type: ignore[arg-type]
    )
    return FactHit(fact=fact, score=0.5)


def test_full_validity_marks_status_and_successor() -> None:
    old = make(
        "a",
        "User works in Beijing.",
        valid_to="2025-08",
        invalidated_at=datetime(2025, 9, 10),
        superseded_by="b",
        supersede_reason="changed",
    )
    new = make(
        "b", "User works in Singapore.", recorded_at=datetime(2025, 9, 10), valid_from="2025-08"
    )
    wrong = make(
        "c",
        "User's dog is a retriever.",
        invalidated_at=datetime(2025, 4, 1),
        superseded_by="zz",  # 取代它的事实不在上下文里
        supersede_reason="corrected",
    )
    text, refs = format_facts([new, old, wrong], validity="full")
    assert refs == {"f1": "a", "f2": "c", "f3": "b"}  # 按时间排列
    assert (
        "[f1] (said on 2025-03-01; no longer true from 2025-08, see [f3]) User works in Beijing."
        in text
    )
    assert "[f3] (said on 2025-09-10; since 2025-08; current) User works in Singapore." in text
    assert "[f2] (said on 2025-03-01; corrected on 2025-04-01) User's dog" in text
    assert text.startswith("Notes from memory")


def test_plain_mode_hides_validity() -> None:
    old = make(
        "a", "User works in Beijing.", invalidated_at=datetime(2025, 9, 10), superseded_by="b"
    )
    text, _ = format_facts([old], validity="none")
    assert "no longer true" not in text and "current" not in text
    assert "[f1] (said on 2025-03-01) User works in Beijing." in text
