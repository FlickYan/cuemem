from __future__ import annotations

import math
from datetime import datetime

import pytest

from cuemem.core.models import Hit, MemoryUnit
from cuemem.evals.datasets import Sample
from cuemem.evals.metrics import bootstrap_ci, retrieval_metrics
from cuemem.evals.splits import stratified_ids


def make_hit(session_id: str, turn_indices: list[int], rank: int) -> Hit:
    unit = MemoryUnit(
        id=f"{session_id}-{turn_indices[0]}",
        space_id="s",
        kind="round",
        session_id=session_id,
        text="...",
        ts=datetime(2023, 1, 1),
        source_turn_ids=[],
        meta={"turn_indices": turn_indices},
    )
    return Hit(unit=unit, score=1.0 / rank, route="dense", rank=rank)


def test_bootstrap_ci_is_deterministic_and_sane() -> None:
    values = [1.0] * 70 + [0.0] * 30
    low, high = bootstrap_ci(values)
    assert (low, high) == bootstrap_ci(values)  # 固定种子
    assert low < 0.7 < high
    assert bootstrap_ci([1.0, 1.0, 1.0]) == (1.0, 1.0)


def test_retrieval_metrics() -> None:
    hits = [make_hit("s1", [0, 1], 1), make_hit("s2", [4, 5], 2), make_hit("s3", [0, 1], 3)]
    metrics = retrieval_metrics(
        hits,
        evidence_sessions=["s2", "s9"],
        evidence_turns={("s2", 4)},
        relevant_units=1,
    )
    assert metrics["session_recall_any"] == 1.0
    assert metrics["session_recall_all"] == 0.5  # s9 没有被命中
    assert metrics["turn_recall_any"] == 1.0
    # 唯一的相关项排在第 2 位：DCG = 1 / log2(3)，理想情况排在第 1 位：IDCG = 1
    assert metrics["turn_ndcg"] == pytest.approx(1 / math.log2(3))


def test_stratified_split_is_proportional_and_reproducible() -> None:
    # a 题型 60 道、b 题型 40 道，各有 10% 拒答题；抽 20 道时各层名额没有并列的余数
    samples = []
    for qtype, count in [("a", 60), ("b", 40)]:
        for i in range(count):
            is_abs = i < count // 10
            samples.append(
                Sample(
                    dataset="d",
                    sample_id=f"{qtype}{i}" + ("_abs" if is_abs else ""),
                    space_key=f"{qtype}{i}",
                    question="?",
                    answer="!",
                    question_type=qtype,
                    ability=qtype,
                    is_abstention=is_abs,
                    question_time=datetime(2023, 1, 1),
                )
            )
    chosen = stratified_ids(samples, 20, seed=0)
    assert len(chosen) == 20
    assert chosen == stratified_ids(samples, 20, seed=0)  # 固定种子，可复现
    assert chosen != stratified_ids(samples, 20, seed=1)
    counts = {q: sum(1 for c in chosen if c.startswith(q)) for q in "ab"}
    assert counts == {"a": 12, "b": 8}
    assert sum(1 for c in chosen if c.endswith("_abs")) == 2  # 拒答题按比例抽到 10%


def test_audit_export_and_score(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from cuemem.evals.audit import export_audit, pick_rows, score_audit

    rows = [
        {
            "sample_id": f"q{i}",
            "method": "B2",
            "question_type": "multi-session",
            "question": "Q?",
            "answer": "A",
            "response": "R",
            "correct": i % 3 == 0,
            "judge_raw": "yes",
        }
        for i in range(30)
    ]
    picked = pick_rows(rows, 10, seed=0)
    assert len(picked) == 10
    assert sum(r["correct"] for r in picked) == 5  # 判对、判错各一半

    path = tmp_path / "audit.md"
    assert export_audit(rows, path, n=10) == 10
    text = path.read_text(encoding="utf-8")
    text = text.replace("- [ ] 同意", "- [x] 同意", 7)
    path.write_text(text.replace("- [ ] 不同意", "- [x] 不同意", 1), encoding="utf-8")
    # 第 1 条同时勾了同意和不同意，视为无效；其余 6 条同意、0 条不同意、3 条未勾
    assert score_audit(path) == (6, 0, 4)
