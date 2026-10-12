"""N9：测试集上的混合流——冻结的控制器（N7 训练、导出）在没见过的数据上判断"该不该想起"。

做法与 N7 的开发集混合流（scripts/n7_mixed.py）相同，只换成测试集：

- 记忆题：LongMemEval test400、LoCoMo 后 7 段对话（非对抗题）、MemLens 32K 官方 195 题子集；
- 通用请求：databricks-dolly-15k 里与 N7 不重叠的行，条数与记忆题相同，只记行号；
- 同话题的常识问题（难负例）：每道测试题生成一个，结果提交在
  evals/mixed/topic_negatives_test.jsonl（只存题号与生成的问题）。

每条查询配一个测试集的记忆空间，在那个空间上跑探针（不调用 LLM），写到
runs/eval/mixed-n9/probe.jsonl。需要先建好测试集的记忆空间（--build-only）。

用法：
    uv run python scripts/n9_mixed.py    # 只用通义，不受 DeepSeek 高峰影响
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from cuemem.config import PROJECT_ROOT, load_settings
from cuemem.evals.datasets import Sample, dataset_path
from cuemem.evals.datasets import load as load_dataset
from cuemem.evals.runner import EvalRunner, ProbeQuery, RunConfig
from cuemem.evals.splits import load_split
from cuemem.llm.cache import ResponseCache
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger

sys.path.insert(0, str(Path(__file__).parent))
from n7_mixed import DOLLY_CATEGORIES, topic_negatives

MIXED_DIR = PROJECT_ROOT / "evals" / "mixed"
TEST_SETS = (
    ("longmemeval_s", "test400"),
    ("locomo", "test7"),
    ("memlens_32k", "agent195"),
)


def test_samples(datasets: Path) -> list[Sample]:
    samples: list[Sample] = []
    for dataset, split in TEST_SETS:
        evalset = load_dataset(
            dataset, dataset_path(datasets, dataset), ids=load_split(dataset, split)
        )
        samples.extend(s for s in evalset.samples if s.question_type != "adversarial")
    return samples


def sample_dolly(path: Path, n: int, exclude: set[int], seed: int = 9) -> list[dict[str, object]]:
    """与 N7 同样的筛选与按类别配额，排除 N7 用过的行。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    pool: dict[str, list[int]] = {c: [] for c in DOLLY_CATEGORIES}
    for i, row in enumerate(rows):
        text = row["instruction"].strip()
        if (
            row["category"] in pool
            and not row["context"].strip()
            and 10 <= len(text) <= 300
            and i not in exclude
        ):
            pool[row["category"]].append(i)
    total = sum(len(v) for v in pool.values())
    rng = random.Random(seed)
    quotas = {c: round(n * len(v) / total) for c, v in pool.items()}
    quotas[DOLLY_CATEGORIES[0]] += n - sum(quotas.values())
    chosen: list[int] = []
    for category, indices in pool.items():
        chosen.extend(rng.sample(indices, quotas[category]))
    return [
        {"line": i, "category": rows[i]["category"], "text": rows[i]["instruction"].strip()}
        for i in sorted(chosen)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mixed-n9")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--generate-only", action="store_true", help="只生成难负例（不需要记忆空间），不跑探针"
    )
    args = parser.parse_args()
    settings = load_settings()
    samples = test_samples(settings.paths.datasets)
    ledger = Ledger(settings.paths.ledger, run_id="n9-mixed")
    llm = LLMClient(settings, cache=ResponseCache(settings.paths.llm_cache), ledger=ledger)
    negatives = topic_negatives(llm, samples, MIXED_DIR / "topic_negatives_test.jsonl")
    if args.generate_only:
        ledger.close()
        print(f"测试记忆题 {len(samples)} 道，难负例 {len(negatives)} 条")
        return

    dev_lines = set(json.loads((MIXED_DIR / "dolly_lines.json").read_text("utf-8"))["lines"])
    dolly = sample_dolly(dataset_path(settings.paths.datasets, "dolly"), len(samples), dev_lines)
    assert not dev_lines & {int(str(d["line"])) for d in dolly}
    (MIXED_DIR / "dolly_lines_test.json").write_text(
        json.dumps(
            {
                "source": "databricks-dolly-15k (CC BY-SA 3.0)",
                "categories": list(DOLLY_CATEGORIES),
                "seed": 9,
                "excludes": "dolly_lines.json（N7 开发集用过的行）",
                "lines": [d["line"] for d in dolly],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    rng = random.Random(1)
    hosts = list(samples)
    rng.shuffle(hosts)
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
    done, errors = EvalRunner(settings, config).probe_queries(queries)
    ledger.close()
    print(f"测试记忆题 {len(samples)} 道；难负例 {len(negatives)} 条，通用请求 {len(dolly)} 条")
    print(f"探针完成 {done} 条，出错 {errors} 条")


if __name__ == "__main__":
    main()
