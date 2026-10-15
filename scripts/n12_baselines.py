"""N12：更多外部基线与 CueMem 同框对比（只读结果，不调用 API）。

同一个协议下的运行按名字归到一起：`<协议>-<数据集>-<系统>`，例如 pds-locomo-cuemem、
pds-locomo-mem0、pds-locomo-graphiti。脚本把这些运行里同一批题的结果放在一起：准确率与
95% 置信区间、各题型、与 CueMem L1 / L2 的配对符号检验、回答成本，以及每个系统写入的成本、
调用次数与耗时。

协议（D-065）：
- pds：只用 DeepSeek。抽取、规划、回答、判分与外部系统内部的调用都是 deepseek-flash；
  向量是本机的 BAAI/bge-small-en-v1.5（fastembed）。CUEMEM_PRESET=deepseek。
  写入在 data/eval-spaces。
- pq：只用通义（DeepSeek 没有余额时）。抽取、规划、回答与外部系统内部的调用都是 qwen3.8-flash，
  判分 qwen3.8-max，向量 qwen3.7-text-embedding（查询与文档分开向量化）、重排 qwen3-rerank。
  CUEMEM_PRESET=qwen，要 DASHSCOPE_API_KEY；写入放在另一个目录，免得复用 pds 的写入：
  CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen。
- qwen：N9 / N11 的原协议（抽取 qwen3.8-flash、向量 qwen3.7-text-embedding、回答 deepseek-flash、
  判分 qwen3.8-max）。抽取与向量与 pq 相同，写入也在 data/eval-spaces-qwen。

运行（LoCoMo test200r；外部系统用 uv 临时装上，命令见 evals/baselines/ 各模块与
evals/mem0_baseline.py，--methods M0 / MG / ML / MC / MA，运行名 `<协议>-locomo-<系统>`）：

    # pds 协议
    export CUEMEM_PRESET=deepseek
    uv run --extra local cuemem eval run --dataset locomo --split test200r \\
        --methods B1,B2,B4-nobm25,B5,B6 --name pds-locomo-cuemem
    uv run --extra local --with mem0ai==2.2.1 --with spacy --with fastembed \\
        --with "en_core_web_sm @ $SPACY_SM" cuemem eval run --dataset locomo --split test200r \\
        --methods M0 --name pds-locomo-mem0
    uv run python scripts/n12_baselines.py --protocol pds --dataset locomo

    # pq 协议
    export CUEMEM_PRESET=qwen CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen
    uv run cuemem eval run --dataset locomo --split test200r --peak warn \\
        --methods B1,B2,B4-nobm25,B5,B6 --name pq-locomo-cuemem
    uv run --with mem0ai==2.2.1 --with spacy --with fastembed \\
        --with "en_core_web_sm @ $SPACY_SM" cuemem eval run --dataset locomo --split test200r \\
        --peak warn --methods M0 --name pq-locomo-mem0
    uv run python scripts/n12_baselines.py --protocol pq --dataset locomo

写入成本与 report.md 同一口径（cuemem.evals.report.write_cost）：CueMem 取运行里的写入与巩固
记录，外部基线取各自的 build.json。外部基线的写入目录按协议选（PROTOCOL_SPACES，与当前的
CUEMEM_PATHS__EVAL_SPACES 无关；--eval-spaces 可以另行指定），build.json 记的协议与运行的抽取
模型、向量模型对不上的不算（计入 mismatched）。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from cuemem.config import home_dir, load_settings
from cuemem.evals.report import EXTERNAL_SYSTEMS, load_rows, run_protocol, write_cost
from cuemem.evals.routing import sign_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "n12-baselines"
LABELS = {
    "B1": "Full context",
    "B2": "Naive RAG (dense top-10)",
    "B4-nobm25": "CueMem L1",
    "B5": "CueMem L2",
    "B6": "CueMem controller",
    "M0": "Mem0 2.2.1 (OSS)",
    "MG": "Graphiti (graphiti-core 0.30.2)",
    "ML": "LangMem 0.0.30",
    "MC": "Cognee 1.6.4",
    "MA": "A-MEM (agiresearch/A-mem@ceffb86)",
}
# 外部系统的写入统计放在写入目录的哪个子目录（方法 → 系统名），与报告共用
SYSTEMS = EXTERNAL_SYSTEMS
# 各协议的写入目录（相对数据目录）：pq 与 qwen 的抽取模型、向量模型相同，共用一份写入
PROTOCOL_SPACES = {
    "pds": Path("data/eval-spaces"),
    "pq": Path("data/eval-spaces-qwen"),
    "qwen": Path("data/eval-spaces-qwen"),
}
REFERENCES = ("B4-nobm25", "B5")


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(centre - half, 4), round(centre + half, 4)


def load_cells(
    runs_dir: Path, prefix: str
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[str], dict[str, str]]:
    """同一协议、同一数据集的全部运行：（题号，方法）→ 最后一条结果；另外返回运行名列表、
    每个方法的结果最后来自哪次运行（读它的写入协议）。"""
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    sources: dict[str, str] = {}
    runs = sorted(p.name for p in runs_dir.iterdir() if p.is_dir() and p.name.startswith(prefix))
    for run in runs:
        for row in load_rows(runs_dir / run, "results.jsonl"):
            cells[(row["sample_id"], row["method"])] = row
            sources[row["method"]] = run
    return cells, runs, sources


def eval_spaces_for(protocol: str, settings: Any, override: Path | None = None) -> Path:
    """外部基线的写入目录：--eval-spaces > 协议对应的目录（PROTOCOL_SPACES）> 当前配置的
    paths.eval_spaces。不跟着当前配置走：pq 的写入与 pds 的目录名相同（如 mem0/locomo_conv-42），
    忘了设 CUEMEM_PATHS__EVAL_SPACES 时会悄悄读到 pds 的写入。相对路径相对数据目录。"""
    if override is not None:
        return override if override.is_absolute() else home_dir() / override
    if protocol in PROTOCOL_SPACES:
        return home_dir() / PROTOCOL_SPACES[protocol]
    return Path(settings.paths.eval_spaces)


def run_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    info: dict[str, Any] = json.loads(path.read_text("utf-8")) if path.exists() else {}
    return info


def write_stats(
    settings: Any,
    dataset: str,
    method: str,
    keys: set[str],
    runs_dir: Path,
    runs: list[str],
    *,
    eval_spaces: Path | None = None,
    protocol: str | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """写入的成本，与 report.md 同一口径（cuemem.evals.report.write_cost）：外部系统取各空间的
    build.json（eval_spaces 下，默认是当前配置的 paths.eval_spaces；协议对不上的不算）；CueMem
    取这些运行里的写入与巩固统计。

    全上下文（B1）不写入；朴素检索（B2）只建索引（向量化，不调用 LLM）；L1 / L2 / 控制器用同一份
    巩固后的记忆，写入成本相同。"""
    ingests = [row for run in runs for row in load_rows(runs_dir / run, "ingest.jsonl")]
    return write_cost(
        method,
        keys,
        dataset=dataset,
        ingests=ingests,
        eval_spaces=eval_spaces if eval_spaces is not None else settings.paths.eval_spaces,
        protocol=protocol,
        rows=rows or [],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="pds")
    parser.add_argument("--dataset", default="locomo")
    parser.add_argument(
        "--eval-spaces",
        type=Path,
        default=None,
        help="外部基线的写入目录（相对数据目录）；默认按协议选，见 PROTOCOL_SPACES",
    )
    args = parser.parse_args()
    settings = load_settings()
    runs_dir = settings.paths.eval_runs
    prefix = f"{args.protocol}-{args.dataset}-"
    cells, runs, sources = load_cells(runs_dir, prefix)
    if not cells:
        print(f"没有 {prefix}* 的运行")
        return
    eval_spaces = eval_spaces_for(args.protocol, settings, args.eval_spaces)
    protocols = {run: run_protocol(run_info(runs_dir / run)) for run in runs}
    methods_present = sorted(
        {m for _, m in cells}, key=lambda m: list(LABELS).index(m) if m in LABELS else 99
    )
    # 只比较每个方法都有结果的题（某个系统中途失败的题不计入任何方法）
    by_method: dict[str, set[str]] = defaultdict(set)
    for sid, method in cells:
        by_method[method].add(sid)
    common = sorted(set.intersection(*by_method.values()))
    out: dict[str, Any] = {
        "protocol": args.protocol,
        "dataset": args.dataset,
        "runs": runs,
        "questions": len(common),
        "dropped": {
            m: len(ids) - len(common) for m, ids in by_method.items() if len(ids) > len(common)
        },
        "labels": {m: LABELS.get(m, m) for m in methods_present},
        "methods": {},
        "paired": {},
    }
    for method in methods_present:
        rows = [cells[(sid, method)] for sid in common]
        correct = sum(bool(r["correct"]) for r in rows)
        by_type: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            by_type[row["question_type"]].append(bool(row["correct"]))
        keys = {r["space_key"] for r in rows}
        out["methods"][method] = {
            "n": len(rows),
            "accuracy": round(correct / len(rows), 4),
            "ci95": wilson(correct, len(rows)),
            "answer_cost_per_1k": round(
                sum(r["usage"]["cost"] for r in rows) / len(rows) * 1000, 3
            ),
            "answer_tokens_mean": round(
                sum(r["usage"].get("input_tokens", 0) for r in rows) / len(rows)
            ),
            "by_type": {
                t: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)}
                for t, v in sorted(by_type.items())
            },
            "write": write_stats(
                settings,
                args.dataset,
                method,
                keys,
                runs_dir,
                runs,
                eval_spaces=eval_spaces,
                protocol=protocols.get(sources[method]),
                rows=[r for (_, m), r in cells.items() if m == method],
            ),
        }
    for ref in REFERENCES:
        if ref not in methods_present:
            continue
        for other in methods_present:
            if other == ref:
                continue
            wins = sum(
                1 for s in common if cells[(s, ref)]["correct"] and not cells[(s, other)]["correct"]
            )
            losses = sum(
                1 for s in common if not cells[(s, ref)]["correct"] and cells[(s, other)]["correct"]
            )
            out["paired"][f"{ref}_vs_{other}"] = {
                "wins": wins,
                "losses": losses,
                "p": round(sign_test(wins, losses), 4),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{args.protocol}-{args.dataset}.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"{args.protocol} / {args.dataset}：{len(common)} 题，运行 {runs}")
    print(f"外部基线的写入目录：{eval_spaces}")
    for method, row in out["methods"].items():
        lo, hi = row["ci95"]
        write = row["write"]
        if write.get("mismatched"):
            print(f"  ! {method}：{write['mismatched']:.0f} 份写入记录的协议与运行不符，没有计入")
        print(
            f"  {LABELS.get(method, method):<34} {row['accuracy']:.1%} [{lo:.1%}, {hi:.1%}]"
            f"  回答 ${row['answer_cost_per_1k']:.2f}/1k q"
            f"  写入 ${write.get('spent', write.get('cost', 0)):.3f}"
            f", {write.get('seconds', 0):.0f} s"
        )
    for name, row in out["paired"].items():
        print(f"  {name}: +{row['wins']} / -{row['losses']}  p={row['p']:.3g}")
    print(f"→ {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
