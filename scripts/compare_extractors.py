"""抽取模型对比：同一批会话分别用两个模型抽取，比较质量、成本与速度（D-018）。

样本来自开发集 dev100（不碰 test400）：
- 证据会话：按题型分层抽 N 道"答案是一个具体信息"的题（单会话用户信息、偏好、知识更新），
  取它们的证据会话；
- 干扰会话：再随机抽同样数量的非证据会话，看模型会不会从无关对话里抽出一堆琐碎事实。

自动指标：
- 证据覆盖：标注为证据的发言（has_answer）至少被一条事实引用的比例；
- 答案召回：标准答案的实词有多少出现在证据会话抽出的事实里（取整题平均）；
- 每会话事实数、带日期的事实比例、JSON 重试次数、引用了不存在轮次的次数；
- 成本（名义）、延迟。

用法：
    uv run python scripts/compare_extractors.py --questions 30 \
        --models qwen3.8-flash,deepseek-flash
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from cuemem.config import PROJECT_ROOT, load_settings
from cuemem.core.trace import start_trace
from cuemem.evals.datasets import EvalSession, dataset_path, load_longmemeval, to_session_ins
from cuemem.evals.runner import usage_from_trace
from cuemem.evals.splits import load_split
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.write.extract import Extractor, SessionExtraction
from cuemem.write.online import build_turns

TYPES = ("single-session-user", "single-session-preference", "knowledge-update")
STOP = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "is",
    "was",
    "my",
    "i",
    "you",
    "your",
    "it",
    "that",
    "with",
    "by",
    "from",
    "as",
    "be",
    "are",
    "were",
}


def content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--models", default="qwen3.8-flash,deepseek-flash")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", default="runs/extractor-comparison")
    args = parser.parse_args()

    settings = load_settings()
    ids = load_split("longmemeval_s", "dev100")
    evalset = load_longmemeval(dataset_path(settings.paths.datasets, "longmemeval_s"), ids=ids)
    rng = random.Random(args.seed)
    pool = [s for s in evalset.samples if s.question_type in TYPES and not s.is_abstention]
    by_type: dict[str, list[Any]] = defaultdict(list)
    for sample in pool:
        by_type[sample.question_type].append(sample)
    picked: list[Any] = []
    for qtype in TYPES:  # 各题型轮流抽，保持分层
        rng.shuffle(by_type[qtype])
    while len(picked) < args.questions and any(by_type.values()):
        for qtype in TYPES:
            if by_type[qtype] and len(picked) < args.questions:
                picked.append(by_type[qtype].pop())

    evidence: dict[str, tuple[EvalSession, str]] = {}  # 会话 ID → (会话, 题目 ID)
    fillers: dict[str, EvalSession] = {}
    for sample in picked:
        sessions = evalset.sessions_of(sample)
        for s in sessions:
            if s.session_id in sample.evidence_session_ids:
                evidence[s.session_id] = (s, sample.sample_id)
        others = [s for s in sessions if s.session_id not in sample.evidence_session_ids]
        filler = rng.choice(others)
        fillers.setdefault(filler.session_id, filler)
    sessions = {sid: s for sid, (s, _) in evidence.items()} | fillers
    print(f"{len(picked)} 道题：{len(evidence)} 个证据会话 + {len(fillers)} 个干扰会话")

    ledger = Ledger(None, run_id="compare-extractors")
    llm = LLMClient(settings, cache=ResponseCache(settings.paths.llm_cache), ledger=ledger)
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, Any]] = {}
    for model in [m.strip() for m in args.models.split(",")]:
        extractor = Extractor(
            llm, model=model, assistant_chars=settings.consolidation.assistant_chars
        )

        def run(
            item: tuple[str, EvalSession], extractor: Extractor = extractor
        ) -> tuple[str, SessionExtraction, dict[str, Any], float]:
            sid, session = item
            [session_in] = to_session_ins([session])
            turns = build_turns("cmp", session_in)
            started = time.perf_counter()
            with start_trace("extract") as trace:
                result = extractor.extract(sid, turns, session.started_at)
            return sid, result, usage_from_trace(trace), time.perf_counter() - started

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(run, sessions.items()))
        summary[model] = evaluate(results, evidence, fillers, picked)
        dump = {
            sid: {
                "summary": r.summary,
                "facts": [
                    {
                        "statement": c.raw.statement,
                        "predicate": c.raw.predicate,
                        "date": c.raw.date,
                        "turns": c.raw.turns,
                    }
                    for c in r.facts
                ],
            }
            for sid, r, _, _ in results
        }
        (out_dir / f"{model}.json").write_text(
            json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print_table(summary)


def evaluate(
    results: list[tuple[str, SessionExtraction, dict[str, Any], float]],
    evidence: dict[str, tuple[EvalSession, str]],
    fillers: dict[str, EvalSession],
    picked: list[Any],
) -> dict[str, Any]:
    by_sid = {sid: r for sid, r, _, _ in results}
    covered = total_evidence_turns = 0
    for sid, (session, _) in evidence.items():
        turn_ids = build_turns("cmp", to_session_ins([session])[0])
        wanted = {turn_ids[i].id for i, t in enumerate(session.turns) if t.has_answer}
        cited = {tid for c in by_sid[sid].facts for tid in c.source_turn_ids}
        total_evidence_turns += len(wanted)
        covered += len(wanted & cited)
    recalls = []
    for sample in picked:
        answer = content_tokens(sample.answer)
        if not answer or len(answer) > 8:  # 只看短答案题
            continue
        text = " ".join(
            c.raw.statement
            for sid in sample.evidence_session_ids
            if sid in by_sid
            for c in by_sid[sid].facts
        )
        recalls.append(len(answer & content_tokens(text)) / len(answer))
    facts_evidence = [len(by_sid[sid].facts) for sid in evidence]
    facts_filler = [len(by_sid[sid].facts) for sid in fillers]
    all_facts = [c for _, r, _, _ in results for c in r.facts]
    usage = [u for _, _, u, _ in results]
    latency = sorted(t for _, _, _, t in results)
    return {
        "sessions": len(results),
        "evidence_turn_coverage": covered / max(1, total_evidence_turns),
        "answer_token_recall": statistics.mean(recalls) if recalls else None,
        "answer_questions": len(recalls),
        "facts_per_evidence_session": statistics.mean(facts_evidence),
        "facts_per_filler_session": statistics.mean(facts_filler) if facts_filler else 0,
        "dated_fact_share": sum(1 for c in all_facts if c.raw.date) / max(1, len(all_facts)),
        "retries": sum(r.attempts - 1 for _, r, _, _ in results),
        "dropped_turn_refs": sum(r.dropped_turn_refs for _, r, _, _ in results),
        "input_tokens": sum(u["input_tokens"] for u in usage),
        "output_tokens": sum(u["output_tokens"] for u in usage),
        "nominal_cost": sum(u["cost"] for u in usage),
        "cost_per_session": sum(u["cost"] for u in usage) / max(1, len(results)),
        "latency_p50_s": latency[len(latency) // 2],
        "latency_p95_s": latency[min(len(latency) - 1, int(len(latency) * 0.95))],
        "from_cache": sum(u["llm_cache_hits"] for u in usage),
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }


def print_table(summary: dict[str, dict[str, Any]]) -> None:
    rows = [
        ("证据发言覆盖", "evidence_turn_coverage", "{:.1%}"),
        ("答案实词召回", "answer_token_recall", "{:.1%}"),
        ("每个证据会话的事实数", "facts_per_evidence_session", "{:.1f}"),
        ("每个干扰会话的事实数", "facts_per_filler_session", "{:.1f}"),
        ("带日期的事实比例", "dated_fact_share", "{:.1%}"),
        ("JSON 重试次数", "retries", "{}"),
        ("引用不存在的轮次", "dropped_turn_refs", "{}"),
        ("每会话名义成本", "cost_per_session", "${:.5f}"),
        ("延迟 p50 / 秒", "latency_p50_s", "{:.1f}"),
        ("延迟 p95 / 秒", "latency_p95_s", "{:.1f}"),
    ]
    models = list(summary)
    print("| 指标 | " + " | ".join(models) + " |")
    print("|---|" + "---|" * len(models))
    for label, key, fmt in rows:
        cells = [
            fmt.format(summary[m][key]) if summary[m][key] is not None else "—" for m in models
        ]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
