"""N11：换一个判分模型重判 N9 测试集上的回答，看结论是否依赖判分模型。

主判分是 qwen3.8-max。第二判分用 deepseek-flash（另一家的模型，非思考模式），提示与解析
和主判分完全相同（LongMemEval 官方模板；MemLens 官方提示）。重判 N9 四个测试集上主要方法的
全部回答，然后统计：

- 两个判分的一致率与 Cohen's κ（按数据集）；
- 每个方法在两种判分下的准确率，方法之间的排序是否改变（Kendall τ）；
- 报告里的关键对比（L1 对全上下文、控制器对 L1 / L2、ReAct 对 L2……）胜负是否改变。

注意 deepseek-flash 也是主回答模型：判自己家族的回答可能偏宽，报告里要一并说明。

用法：
    uv run python scripts/n11_second_judge.py run      # 调用 API（约 1.3 万次判分，几美元）
    uv run python scripts/n11_second_judge.py report   # 只读结果，写 second_judge.json

逐条判分写在 runs/eval/n11-judge2/（不进仓库），中断后重跑会接着做。
"""

from __future__ import annotations

import argparse
import json
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cuemem.config import RoleConfig, Settings, load_settings
from cuemem.evals.datasets import Sample, dataset_path
from cuemem.evals.datasets import load as load_dataset
from cuemem.evals.judge import Judge
from cuemem.evals.report import load_rows
from cuemem.evals.routing import sign_test
from cuemem.evals.splits import load_split
from cuemem.llm import Ledger, LLMClient
from cuemem.llm.cache import ResponseCache

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "n11-improvements" / "second_judge.json"
JUDGE2 = "deepseek-flash"
WORKERS = 16
# 数据集 → （N9 的运行名，划分）
RUNS = {
    "longmemeval_s": ("test400-n9", "test400"),
    "locomo": ("locomo-test7-n9", "test7"),
    "memlens_32k": ("memlens32-test-n9", "agent195"),
    "memlens_128k": ("memlens128-test-n9", "agent195"),
}
# 结果表里的主要方法（消融与参照点不重判）
TEXT_METHODS = ["B1", "B2", "B4-nobm25", "B5", "B6", "B3", "B6a"]
METHODS = {
    "longmemeval_s": TEXT_METHODS,
    "locomo": TEXT_METHODS,
    "memlens_32k": ["B1", "B4-see", "B5-see", "B6", "B3", "B6a"],
    "memlens_128k": ["B1", "B4-see", "B5-see", "B6"],
}
# 报告里的关键对比：（甲，乙）——甲对乙的胜负
TEXT_PAIRS = [("B4-nobm25", "B1"), ("B6", "B4-nobm25"), ("B6", "B5"), ("B3", "B5"), ("B6a", "B5")]
PAIRS = {
    "longmemeval_s": TEXT_PAIRS,
    "locomo": TEXT_PAIRS,
    "memlens_32k": [("B4-see", "B1"), ("B6", "B4-see"), ("B6", "B5-see"), ("B3", "B5-see")],
    "memlens_128k": [("B4-see", "B1"), ("B6", "B4-see"), ("B6", "B5-see")],
}


def primary_rows(settings: Settings, dataset: str) -> list[dict[str, Any]]:
    """N9 运行里主要方法的回答（续跑造成的重复行取最后一条）。"""
    run, _ = RUNS[dataset]
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_rows(settings.paths.eval_runs / run, "results.jsonl"):
        if row["method"] in METHODS[dataset]:
            latest[(row["sample_id"], row["method"])] = row
    return list(latest.values())


def samples(settings: Settings, dataset: str) -> dict[str, Sample]:
    _, split = RUNS[dataset]
    path = dataset_path(settings.paths.datasets, dataset)
    evalset = load_dataset(dataset, path, ids=load_split(dataset, split))
    return {s.sample_id: s for s in evalset.samples}


def run(settings: Settings) -> None:
    out_dir = settings.paths.eval_runs / "n11-judge2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "verdicts.jsonl"
    done = {(r["dataset"], r["sample_id"], r["method"]) for r in load_rows(out_dir, path.name)}
    roles = {**settings.roles, "judge2": RoleConfig(model=JUDGE2)}
    deepseek = settings.providers["deepseek"].model_copy(update={"max_concurrency": WORKERS})
    settings = settings.model_copy(
        update={"roles": roles, "providers": {**settings.providers, "deepseek": deepseek}}
    )
    llm = LLMClient(
        settings,
        cache=ResponseCache(settings.paths.llm_cache),
        ledger=Ledger(settings.paths.ledger, run_id="eval:n11-judge2"),
    )
    judge = Judge(llm, role="judge2")
    lock = threading.Lock()
    errors: list[str] = []
    with path.open("a", encoding="utf-8") as out:
        for dataset in RUNS:
            by_id = samples(settings, dataset)
            todo = [
                row
                for row in primary_rows(settings, dataset)
                if (dataset, row["sample_id"], row["method"]) not in done
            ]
            print(f"{dataset}: {len(todo)} 条待判", flush=True)

            def one(
                row: dict[str, Any], dataset: str = dataset, by_id: dict[str, Sample] = by_id
            ) -> None:
                try:
                    verdict = judge.judge(by_id[row["sample_id"]], row["response"])
                except Exception as exc:  # 一条判分失败不影响其他的；重跑时会补上
                    with lock:
                        errors.append(f"{row['sample_id']} {row['method']}: {exc}"[:200])
                    return
                record = {
                    "dataset": dataset,
                    "sample_id": row["sample_id"],
                    "method": row["method"],
                    "primary": row["correct"],
                    "second": verdict.correct,
                    "raw": verdict.raw[-200:],
                }
                with lock:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()

            with ThreadPoolExecutor(WORKERS) as pool:
                futures = [pool.submit(one, row) for row in todo]
                for i, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    if i % 1000 == 0:
                        print(f"  {i} / {len(todo)}", flush=True)
    if errors:
        print(f"判分失败 {len(errors)} 条（重跑会补上），例如：{errors[:3]}")


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (agree - expected) / (1 - expected) if expected < 1 else 1.0


def kendall_tau(x: list[float], y: list[float]) -> float:
    concordant = discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            s = (x[i] - x[j]) * (y[i] - y[j])
            concordant += s > 0
            discordant += s < 0
    total = math.comb(len(x), 2)
    return (concordant - discordant) / total if total else 1.0


def report(settings: Settings) -> None:
    verdicts = load_rows(settings.paths.eval_runs / "n11-judge2", "verdicts.jsonl")
    latest = {(v["dataset"], v["sample_id"], v["method"]): v for v in verdicts}
    result: dict[str, Any] = {"judge": {"primary": "qwen3.8-max", "second": JUDGE2}}
    disagreements: list[dict[str, Any]] = []
    for dataset in RUNS:
        rows = [v for (d, _, _), v in latest.items() if d == dataset]
        if not rows:
            continue
        pairs = [(bool(v["primary"]), bool(v["second"])) for v in rows]
        by_method: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        cell = {(v["sample_id"], v["method"]): v for v in rows}
        for v in rows:
            by_method[v["method"]].append((bool(v["primary"]), bool(v["second"])))
        methods = [m for m in METHODS[dataset] if m in by_method]
        acc1 = [sum(a for a, _ in by_method[m]) / len(by_method[m]) for m in methods]
        acc2 = [sum(b for _, b in by_method[m]) / len(by_method[m]) for m in methods]
        comparisons = []
        for a, b in PAIRS[dataset]:
            ids = sorted({sid for sid, m in cell if m == a} & {sid for sid, m in cell if m == b})
            entry: dict[str, Any] = {"a": a, "b": b, "n": len(ids)}
            for key in ("primary", "second"):
                wins = sum(1 for s in ids if cell[(s, a)][key] and not cell[(s, b)][key])
                losses = sum(1 for s in ids if not cell[(s, a)][key] and cell[(s, b)][key])
                entry[key] = {"wins": wins, "losses": losses, "p": sign_test(wins, losses)}
            same = (entry["primary"]["wins"] - entry["primary"]["losses"]) * (
                entry["second"]["wins"] - entry["second"]["losses"]
            )
            entry["same_direction"] = same > 0
            comparisons.append(entry)
        result[dataset] = {
            "n": len(pairs),
            "agreement": round(sum(1 for a, b in pairs if a == b) / len(pairs), 4),
            "kappa": round(kappa(pairs), 3),
            "primary_yes": round(sum(a for a, _ in pairs) / len(pairs), 4),
            "second_yes": round(sum(b for _, b in pairs) / len(pairs), 4),
            "methods": {
                m: {"n": len(by_method[m]), "primary": round(x, 4), "second": round(y, 4)}
                for m, x, y in zip(methods, acc1, acc2, strict=True)
            },
            "rank_primary": [m for _, m in sorted(zip(acc1, methods, strict=True), reverse=True)],
            "rank_second": [m for _, m in sorted(zip(acc2, methods, strict=True), reverse=True)],
            "kendall_tau": round(kendall_tau(acc1, acc2), 3),
            "comparisons": comparisons,
        }
        disagreements += [v for v in rows if bool(v["primary"]) != bool(v["second"])]
        print(
            f"{dataset}: n={len(pairs)} 一致 {result[dataset]['agreement']:.1%} "
            f"κ={result[dataset]['kappa']} τ={result[dataset]['kendall_tau']}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    path = settings.paths.eval_runs / "n11-judge2" / "disagreements.jsonl"
    path.write_text(
        "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in disagreements), "utf-8"
    )
    print(f"→ {OUT.relative_to(ROOT)}；两个判分不一致的 {len(disagreements)} 条 → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["run", "report"])
    args = parser.parse_args()
    settings = load_settings()
    if args.action == "run":
        run(settings)
    report(settings)


if __name__ == "__main__":
    main()
