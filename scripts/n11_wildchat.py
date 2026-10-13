"""N11：门控在真实流量上的误召回——用 WildChat 里真实的聊天机器人用户消息代替构造的无关查询。

N7 / N9 的混合流里，"不需要记忆"的查询来自 Dolly 的通用请求与生成的同话题问题，是构造的。
这里换成真实用户发给聊天机器人的话：WildChat-1M 第 1 个分片里，英文、无毒、没有打码的会话的
**第一句**（会话的开头不可能指向之前的会话，所以都不需要长期记忆）。条数与测试集记忆题相同
（1,750 条），每条配一个测试集的记忆空间跑探针，然后用冻结的控制器（v1 与 v2）离线判断：

- 门控把多少条真实消息送去检索（误召回），与 Dolly、同话题问题比；
- 把混合流里的无关查询换成这些真实消息后，"比总是检索省多少"还成不成立。

仓库里只存会话的哈希（evals/mixed/wildchat_test.json），原文在本地的数据集文件里。

用法：
    uv run cuemem data download wildchat                      # 约 231 MB，ODC-BY
    uv run --with pyarrow python scripts/n11_wildchat.py prepare   # 筛选、抽样（离线）
    uv run python scripts/n11_wildchat.py probe               # 探针：查询向量化要调用向量接口
    uv run python scripts/n11_wildchat.py report              # 离线判定与统计
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import n7_controller as N7
import n9_mixed as M9
import n9_report as N9

from cuemem.config import PROJECT_ROOT, load_settings
from cuemem.evals import routing as R
from cuemem.evals.datasets import dataset_path
from cuemem.evals.runner import EvalRunner, ProbeQuery, RunConfig
from cuemem.read.controller import DEFAULT_MODEL, V1_MODEL, LearnedController

IDS = PROJECT_ROOT / "evals" / "mixed" / "wildchat_test.json"
OUT = PROJECT_ROOT / "docs" / "results" / "n11-improvements" / "wildchat.json"
RUN = "mixed-n11-wildchat"
SEED = 11
MIX_RATIO = 0.2


def local_texts(settings: Any) -> Path:
    return dataset_path(settings.paths.datasets, "wildchat").with_name("first_turns_test.jsonl")


def prepare(settings: Any) -> None:
    """筛选第一句：英文、无毒、没有打码、10–300 个字符（与 Dolly 的筛选一致）、按文字去重；
    抽出与测试集记忆题同样多的条数。按批读取，只留第一句，不把整个分片读进内存。"""
    import pyarrow.parquet as pq

    source = pq.ParquetFile(dataset_path(settings.paths.datasets, "wildchat"))
    columns = ["conversation_hash", "conversation", "language", "toxic", "redacted"]
    seen: set[str] = set()
    pool: dict[str, str] = {}
    for batch in source.iter_batches(batch_size=2000, columns=columns):
        for row in batch.to_pylist():
            if row["language"] != "English" or row["toxic"] or row["redacted"]:
                continue
            first = next((t for t in row["conversation"] if t["role"] == "user"), None)
            text = " ".join(str(first["content"]).split()) if first else ""
            if 10 <= len(text) <= 300 and text.casefold() not in seen:
                seen.add(text.casefold())
                pool[row["conversation_hash"]] = text
    samples = M9.test_samples(settings.paths.datasets)
    chosen = sorted(random.Random(SEED).sample(sorted(pool), len(samples)))
    IDS.write_text(
        json.dumps(
            {
                "source": "allenai/WildChat-1M@7d6490e, data/train-00000-of-00014.parquet (ODC-BY)",
                "filter": "English, not toxic, not redacted; first user turn, 10-300 characters; "
                "deduplicated by text",
                "pool": len(pool),
                "seed": SEED,
                "conversation_hashes": chosen,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    path = local_texts(settings)
    path.write_text(
        "".join(
            json.dumps({"hash": h, "text": pool[h]}, ensure_ascii=False) + "\n" for h in chosen
        ),
        encoding="utf-8",
    )
    print(f"候选 {len(pool)} 条，抽 {len(chosen)} 条 → {IDS.relative_to(PROJECT_ROOT)}、{path}")


def probe(settings: Any) -> None:
    ids = json.loads(IDS.read_text("utf-8"))["conversation_hashes"]
    texts = {
        row["hash"]: row["text"]
        for row in map(json.loads, local_texts(settings).read_text("utf-8").splitlines())
    }
    missing = [h for h in ids if h not in texts]
    if missing:
        raise SystemExit(f"本地缺少 {len(missing)} 条原文：先运行 prepare")
    samples = M9.test_samples(settings.paths.datasets)
    hosts = list(samples)
    random.Random(SEED).shuffle(hosts)
    queries = [
        ProbeQuery(
            f"wildchat:{h}",
            host.dataset,
            host.space_key,
            texts[h],
            host.question_time,
            {
                "dataset": "mixed",
                "source": "wildchat",
                "space_dataset": host.dataset,
                "space_key": host.space_key,
                "paired": host.sample_id,
                "question": texts[h],
                "question_type": "no-memory",
                "is_abstention": False,
            },
        )
        for h, host in zip(ids, hosts, strict=True)
    ]
    config = RunConfig(name=RUN, dataset="mixed", split=None, methods=[], workers=8)
    done, errors = EvalRunner(settings, config).probe_queries(queries)
    print(f"探针完成 {done} 条，出错 {errors} 条", flush=True)
    # 零样本 LLM 路由（B6z，qwen3.8-flash 只看消息本身）对同样的消息怎么判：每条一次便宜的调用
    runs = settings.paths.eval_runs
    N7.CACHE = runs / RUN
    routed = N7.llm_routes(R.load_mixed_items(runs / RUN, memory_items(runs)))
    print(f"零样本 LLM 路由：{len(routed)} 条", flush=True)


def memory_items(runs: Path) -> list[R.Item]:
    """测试集记忆题（与 N9 混合流相同：LongMemEval、LoCoMo、MemLens 32K）。"""
    return [
        it
        for dataset, run in N9.TEST_RUNS.items()
        if dataset != "memlens_128k"
        for it in R.load_items(runs / run)
    ]


def cached_routes(*paths: Path) -> dict[str, dict[str, Any]]:
    """只读已有的路由结果，不调用 API（report 可能在联网时段之外运行）。"""
    routes: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                routes[row["sample_id"]] = row
    return routes


def report(settings: Any) -> None:
    runs = settings.paths.eval_runs
    memory = memory_items(runs)
    routes = cached_routes(runs / "n9-final" / "router_llm.jsonl", runs / RUN / "router_llm.jsonl")
    sources = {
        "wildchat": R.load_mixed_items(runs / RUN, memory),
        **{
            name: [
                it
                for it in R.load_mixed_items(runs / N9.MIXED_RUN, memory)
                if it.dataset == f"mixed:{name}"
            ]
            for name in ("dolly", "topic")
        },
    }
    if not sources["wildchat"]:
        raise SystemExit(f"{RUN} 还没有探针结果：先运行 probe")
    result: dict[str, Any] = {"n_memory": len(memory)}
    for version, path in (("v1", V1_MODEL), ("v2", DEFAULT_MODEL)):
        ctrl = LearnedController.load(path)
        out: dict[str, Any] = {}
        for name, items in sources.items():
            mixed = memory + items
            policy = N9.frozen(mixed, ctrl, ctrl.lam, f"B6 {version}")
            metrics = N7.mixed_metrics(policy, mixed, MIX_RATIO)
            always = N7.mixed_metrics(R.fixed(mixed, "L1"), mixed, MIX_RATIO)
            p_none = [ctrl.model.type_proba(it.features).get("none", 0.0) for it in mixed]
            labels = [not it.is_memory for it in mixed]
            levels = Counter(
                R.level_of(c.action)
                for c, it in zip(policy.choices, mixed, strict=True)
                if not it.is_memory
            )
            out[name] = {
                "n": len(items),
                "false_recall": round(100 * metrics["false_recall"], 1),
                "routed": dict(sorted(levels.items())),
                "gate_auc": round(N7.auc(labels, p_none), 4),
                "memory_accuracy": round(100 * metrics["memory_accuracy"], 1),
                "cost_per_1k": round(1000 * metrics["cost"], 3),
                "always_l1_cost_per_1k": round(1000 * always["cost"], 3),
                "saving_vs_always_l1": round(100 * (1 - metrics["cost"] / always["cost"]), 1),
            }
            if all(it.sample_id in routes for it in mixed):  # 零样本 LLM 路由与混合方案（对照）
                b6h, asked = N9.hybrid(mixed, ctrl, routes, ctrl.lam)
                for label, other in (("B6z", N7.llm_policy(mixed, routes)), ("B6h", b6h)):
                    m = N7.mixed_metrics(other, mixed, MIX_RATIO)
                    out[name][label] = {
                        "false_recall": round(100 * m["false_recall"], 1),
                        "memory_l0": round(100 * m["memory_l0"], 1),
                        "memory_accuracy": round(100 * m["memory_accuracy"], 1),
                        "cost_per_1k": round(1000 * m["cost"], 3),
                    }
                out[name]["B6h"]["asked_llm"] = round(100 * float(np.mean(asked)), 1)
            print(
                f"{version} {name}: 误召回 {out[name]['false_recall']}%，门控 AUC "
                f"{out[name]['gate_auc']}，比总是 L1 省 {out[name]['saving_vs_always_l1']}%"
            )
        result[version] = out
    ctrl = LearnedController.load(DEFAULT_MODEL)
    recalled = [
        (round(ctrl.model.type_proba(it.features).get("none", 0.0), 3), it.sample_id, it.question)
        for it in sources["wildchat"]
    ]
    recalled.sort()
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(recalled), size=10, replace=False)
    examples = {
        "least_none": recalled[:15],  # 门控最确信"需要记忆"的真实消息
        "random": [recalled[int(i)] for i in sorted(picks)],
    }
    # 原文只写在本地（runs/ 不进仓库）；仓库里的结果只留会话哈希（D-057）
    local = runs / RUN / "examples.json"
    local.write_text(json.dumps(examples, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    result["examples"] = {k: [[p, sid] for p, sid, _ in v] for k, v in examples.items()}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"→ {OUT.relative_to(PROJECT_ROOT)}（样例原文在 {local}）")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["prepare", "probe", "report"])
    args = parser.parse_args()
    settings = load_settings()
    {"prepare": prepare, "probe": probe, "report": report}[args.action](settings)


if __name__ == "__main__":
    main()
