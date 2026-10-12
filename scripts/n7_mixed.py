"""构建混合流（Mixed-Stream，N7）：记忆相关的问题与"不需要记忆"的查询混在一起，评测"该不该想起"。

与记忆无关的查询有两种：

- **通用请求**：databricks-dolly-15k（CC BY-SA 3.0）里不带参考材料的四类——开放问答、一般问答、
  头脑风暴、创作；按类别比例抽 585 条（与开发集的记忆题一样多），只记行号，不把原文提交进仓库；
- **同话题的常识问题**（难负例）：对开发集的每道记忆题，让 qwen3.8-flash 写一个话题相同、但与
  用户本人无关的问题（"我那套乐高拼了几天？" → "世界上零件最多的乐高套装有多少块？"）。生成结果
  提交在 evals/mixed/topic_negatives.jsonl，温度 0、有缓存，可以复现。

每条查询配一个开发集的记忆空间：难负例用它所对应记忆题的空间（同一个话题，最难分辨）；通用请求
随机配一道记忆题的空间与提问时间。然后在那个空间上跑探针（不调用 LLM），特征写入
runs/eval/mixed-n7/probe.jsonl。

用法：
    uv run python scripts/n7_mixed.py    # 生成难负例（只用通义，不受 DeepSeek 高峰影响）并跑探针
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field

from cuemem.config import PROJECT_ROOT, load_settings
from cuemem.errors import ContentFilteredError, StructuredOutputError
from cuemem.evals.datasets import Sample, dataset_path
from cuemem.evals.datasets import load as load_dataset
from cuemem.evals.runner import EvalRunner, ProbeQuery, RunConfig
from cuemem.evals.splits import load_split
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient, Message
from cuemem.llm.ledger import Ledger

MIXED_DIR = PROJECT_ROOT / "evals" / "mixed"
DOLLY_CATEGORIES = ("open_qa", "general_qa", "brainstorming", "creative_writing")
# 开发集：LongMemEval dev100、LoCoMo 前 3 段对话（385 题）、MemLens 32K dev100
DEV_SETS = (
    ("longmemeval_s", "dev100", None),
    ("locomo", None, 385),
    ("memlens_32k", "dev100", None),
)

NEGATIVE_SYSTEM = """You write test questions for a personal assistant that has long-term memory of its past conversations with the user.
Given a question that depends on the user's own past (their conversations, plans, purchases, people they know, photos they shared), write ONE question on the same topic that anyone could ask a general-purpose assistant and that does NOT depend on the user's past.

Rules:
- Same topic and similar length; it may be a factual question, a how-to, or a request for ideas.
- Do not refer to the user's own life: no "I", "my", "me", "we", "our"; no names of their friends, family, pets, or the people in their chats; no "the photo I sent".
- It must be answerable from general knowledge alone."""  # noqa: E501


class Negative(BaseModel):
    question: str = Field(description="the new general-knowledge question")


def dev_samples(settings_datasets: Path) -> list[Sample]:
    samples: list[Sample] = []
    for dataset, split, limit in DEV_SETS:
        ids = load_split(dataset, split) if split else None
        evalset = load_dataset(dataset, dataset_path(settings_datasets, dataset), ids=ids)
        chosen = [s for s in evalset.samples if s.question_type != "adversarial"]
        samples.extend(chosen[:limit] if limit else chosen)
    return samples


def sample_dolly(path: Path, n: int, seed: int = 0) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    pool: dict[str, list[int]] = {c: [] for c in DOLLY_CATEGORIES}
    for i, row in enumerate(rows):
        text = row["instruction"].strip()
        if row["category"] in pool and not row["context"].strip() and 10 <= len(text) <= 300:
            pool[row["category"]].append(i)
    total = sum(len(v) for v in pool.values())
    rng = random.Random(seed)
    chosen: list[int] = []
    quotas = {c: round(n * len(v) / total) for c, v in pool.items()}
    quotas[DOLLY_CATEGORIES[0]] += n - sum(quotas.values())
    for category, indices in pool.items():
        chosen.extend(rng.sample(indices, quotas[category]))
    return [
        {"line": i, "category": rows[i]["category"], "text": rows[i]["instruction"].strip()}
        for i in sorted(chosen)
    ]


def topic_negatives(llm: LLMClient, samples: list[Sample], path: Path) -> dict[str, str]:
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["sample_id"]] = row["negative"]
    todo = [s for s in samples if s.sample_id not in existing]

    def generate(sample: Sample) -> tuple[Sample, str | None]:
        messages: list[Message] = [
            {"role": "system", "content": NEGATIVE_SYSTEM},
            {"role": "user", "content": f"Question about the user's past:\n{sample.question}"},
        ]
        try:
            out = llm.structured("planner", messages, Negative, max_tokens=400).value
        except (StructuredOutputError, ContentFilteredError) as exc:  # 个别题生成失败就跳过
            print(f"跳过 {sample.sample_id}：{type(exc).__name__}")
            return sample, None
        return sample, out.question.strip()

    with ThreadPoolExecutor(max_workers=8) as pool:
        for sample, negative in pool.map(generate, todo):
            if negative:
                existing[sample.sample_id] = negative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            if s.sample_id not in existing:
                continue
            # 只存题号与生成的问题：原题按题号从数据集里查（LoCoMo 是 CC BY-NC，不复制进仓库）
            row = {
                "sample_id": s.sample_id,
                "dataset": s.dataset,
                "negative": existing[s.sample_id],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return existing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mixed-n7")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    settings = load_settings()
    samples = dev_samples(settings.paths.datasets)
    ledger = Ledger(settings.paths.ledger, run_id="n7-mixed")
    llm = LLMClient(settings, cache=ResponseCache(settings.paths.llm_cache), ledger=ledger)
    negatives = topic_negatives(llm, samples, MIXED_DIR / "topic_negatives.jsonl")

    dolly = sample_dolly(dataset_path(settings.paths.datasets, "dolly"), len(samples))
    rng = random.Random(1)
    hosts = list(samples)
    rng.shuffle(hosts)
    (MIXED_DIR / "dolly_lines.json").write_text(
        json.dumps(
            {
                "source": "databricks-dolly-15k (CC BY-SA 3.0)",
                "categories": list(DOLLY_CATEGORIES),
                "seed": 0,
                "lines": [d["line"] for d in dolly],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    queries: list[ProbeQuery] = []
    for s in samples:
        if s.sample_id not in negatives:
            continue
        queries.append(
            ProbeQuery(
                f"topic:{s.sample_id}",
                s.dataset,
                s.space_key,
                negatives[s.sample_id],
                s.question_time,
                {
                    "dataset": "mixed",
                    "source": "topic",
                    "space_dataset": s.dataset,
                    "space_key": s.space_key,
                    "paired": s.sample_id,
                    "question": negatives[s.sample_id],
                    "question_type": "no-memory",
                    "is_abstention": False,
                },
            )
        )
    for d, host in zip(dolly, hosts, strict=True):
        queries.append(
            ProbeQuery(
                f"dolly:{d['line']}",
                host.dataset,
                host.space_key,
                str(d["text"]),
                host.question_time,
                {
                    "dataset": "mixed",
                    "source": "dolly",
                    "category": d["category"],
                    "space_dataset": host.dataset,
                    "space_key": host.space_key,
                    "paired": host.sample_id,
                    "question": d["text"],
                    "question_type": "no-memory",
                    "is_abstention": False,
                },
            )
        )
    config = RunConfig(
        name=args.name, dataset="mixed", split=None, methods=[], workers=args.workers
    )
    runner = EvalRunner(settings, config)
    done, errors = runner.probe_queries(queries)
    ledger.close()
    print(f"难负例 {len(negatives)} 条，通用请求 {len(dolly)} 条")
    print(f"探针完成 {done} 条，出错 {errors} 条")


if __name__ == "__main__":
    main()
