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
  CUEMEM_PRESET=qwen，要 DASHSCOPE_API_KEY；写入放在另一个目录：
  CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen。外部基线（M0、MG、ML、MC、MA）的写入目录
  不分协议，build.json 记下协议（抽取模型|向量模型:维度）：忘了设这个变量时，MG、ML、MC、MA
  遇到 pds 的写入既不复用、也不删掉重建，每道题报 ConfigMismatchError（运行结束时归并列出，见
  evals/external.py），设好目录重跑即可。M0 的写入记录早先不记协议（新写的才记，记了的同样
  报错），data/eval-spaces 与 data/eval-spaces-qwen 下现存的 M0 写入都没记，按当前协议处理，
  **会被悄悄复用**：跑 M0 前务必确认这个变量。
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
记录，外部基线取各自的写入记录（运行里触发写入的那道题记下的统计，或写入目录下的 build.json；
协议对得上的优先）。外部基线的写入目录按协议选（PROTOCOL_SPACES，与当前的
CUEMEM_PATHS__EVAL_SPACES 无关；--eval-spaces 可以另行指定），记的协议与运行的抽取模型、向量
模型对不上的不算（都对不上时计入 mismatched）。

会话 ID（N12 修复的泄漏，见 evals/datasets.py）：LongMemEval / MemLens 的写入按运行自己记下的
写法找（run.json，datasets.run_session_ids；修复之前的运行没记，是原始 ID），不看当前的
CUEMEM_EVAL_RAW_SESSION_IDS：改写过的写入在 `<数据集>-anon_<历史>` 下，记的写法对不上的同样
不算。写法不同的运行（修复前后、label 消融）不放在一起比。LoCoMo 不受影响。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from cuemem.config import home_dir, load_settings
from cuemem.evals.datasets import SessionIds, run_raw_labels, run_session_ids
from cuemem.evals.report import (
    EXTERNAL_SYSTEMS,
    PROTOCOL_SPACES,
    load_rows,
    run_protocol,
    write_cost,
)
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
# 敏感性运行：运行名以 <标签>-<协议>-<数据集>- 开头（不会被归进主结果），复用同一份写入，
# 只换检索设置。名字 → （运行名的标签，方法，对照的主方法）
SENSITIVITY = {"MA-k40": ("k40", "MA", "MA")}  # A-MEM 作者给 GPT-4o 级模型的 k = 40
# 各协议的写入目录（相对数据目录）：PROTOCOL_SPACES，与报告共用（报告给没记写入目录的旧运行
# 按运行名推断）；pq 与 qwen 的抽取模型、向量模型相同，共用一份写入
REFERENCES = ("B4-nobm25", "B5")


def paired(
    a: tuple[dict[tuple[str, str], dict[str, Any]], str],
    b: tuple[dict[tuple[str, str], dict[str, Any]], str],
    questions: list[str],
) -> dict[str, Any]:
    """同一批题上甲对乙的赢 / 输题数（甲对、乙错算赢）与精确符号检验。a、b 是（结果表，方法）。"""
    (ca, ma), (cb, mb) = a, b
    wins = sum(1 for s in questions if ca[(s, ma)]["correct"] and not cb[(s, mb)]["correct"])
    losses = sum(1 for s in questions if not ca[(s, ma)]["correct"] and cb[(s, mb)]["correct"])
    return {"wins": wins, "losses": losses, "p": round(sign_test(wins, losses), 4)}


def accuracy_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """准确率、Wilson 区间、回答成本（每千题）、每题输入 tokens 与分题型准确率。"""
    correct = sum(bool(r["correct"]) for r in rows)
    by_type: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(bool(row["correct"]))
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "ci95": wilson(correct, len(rows)),
        "answer_cost_per_1k": round(sum(r["usage"]["cost"] for r in rows) / len(rows) * 1000, 3),
        "answer_tokens_mean": round(
            sum(r["usage"].get("input_tokens", 0) for r in rows) / len(rows)
        ),
        "by_type": {
            t: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)}
            for t, v in sorted(by_type.items())
        },
    }


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
    忘了设 CUEMEM_PATHS__EVAL_SPACES 时会读到 pds 的写入：记了协议的对不上、计入 mismatched，
    Mem0 早先的写入没记协议，会被悄悄算进来。相对路径相对数据目录。"""
    if override is not None:
        return override if override.is_absolute() else home_dir() / override
    if protocol in PROTOCOL_SPACES:
        return home_dir() / PROTOCOL_SPACES[protocol]
    return Path(settings.paths.eval_spaces)


def run_info(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run.json"
    info: dict[str, Any] = json.loads(path.read_text("utf-8")) if path.exists() else {}
    return info


def runs_session_ids(runs_dir: Path, runs: list[str]) -> SessionIds:
    """这些运行的会话 ID 写法（各自 run.json 里记的，不看当前的环境变量）。混用了不同写法（或
    label 消融）的运行不能放在一起比：修复泄漏之前的结果与之后的混在一起，直接报错。"""
    modes: dict[str, tuple[SessionIds, bool]] = {}
    for run in runs:
        info = run_info(runs_dir / run)
        modes[run] = (run_session_ids(info), run_raw_labels(info))
    if len(set(modes.values())) > 1:
        detail = "；".join(
            f"{run}: {ids}{'+label' if labels else ''}" for run, (ids, labels) in modes.items()
        )
        raise SystemExit(f"这些运行的会话 ID 写法不同，不能放在一起比（{detail}）")
    return next(iter(modes.values()))[0] if modes else "native"


def write_stats(
    settings: Any,
    dataset: str,
    method: str,
    keys: set[str],
    runs_dir: Path,
    runs: list[str],
    scheme: SessionIds | None = None,
    *,
    eval_spaces: Path | None = None,
    protocol: str | None = None,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """写入的成本，与 report.md 同一口径（cuemem.evals.report.write_cost）：外部系统取 rows 里
    触发写入的那道题记下的统计或各空间的 build.json（eval_spaces 下，默认是当前配置的
    paths.eval_spaces；协议对得上的优先，对不上的不算）；CueMem 取这些运行里的写入与巩固统计。

    全上下文（B1）不写入；朴素检索（B2）只建索引（向量化，不调用 LLM）；L1 / L2 / 控制器用同一份
    巩固后的记忆，写入成本相同。scheme 是这些运行的会话 ID 写法（runs_session_ids；None 时按
    没记写法的运行处理：LongMemEval / MemLens 是原始 ID）：改写过的写入在 <数据集>-anon_<历史>
    下，记的写法或协议对不上的计入 mismatched；外部系统找不到任何写入记录的历史计入 missing，
    都不算进成本。"""
    ingests = [row for run in runs for row in load_rows(runs_dir / run, "ingest.jsonl")]
    stats = write_cost(
        method,
        keys,
        dataset=dataset,
        ingests=ingests,
        eval_spaces=eval_spaces if eval_spaces is not None else settings.paths.eval_spaces,
        protocol=protocol,
        rows=rows or [],
        session_ids=scheme,
    )
    if method in SYSTEMS:
        missing = len(keys) - stats.get("spaces", 0) - stats.get("mismatched", 0)
        if missing:
            stats["missing"] = float(missing)
    return stats


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
    # 会话 ID 的写法、写入协议与数据集名都按运行自己的记录（不看当前的环境变量与配置）：
    # 运行名里的数据集只是简写（如 pq-lme-…），写入目录用的是数据集的全名
    scheme = runs_session_ids(runs_dir, runs)
    eval_spaces = eval_spaces_for(args.protocol, settings, args.eval_spaces)
    infos = {run: run_info(runs_dir / run) for run in runs}
    protocols = {run: run_protocol(info) for run, info in infos.items()}
    datasets = {
        run: str((info.get("config") or {}).get("dataset") or args.dataset)
        for run, info in infos.items()
    }
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
        **({"session_ids": scheme} if scheme != "native" else {}),
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
        keys = {r["space_key"] for r in rows}
        out["methods"][method] = {
            **accuracy_stats(rows),
            "write": write_stats(
                settings,
                datasets.get(sources[method], args.dataset),
                method,
                keys,
                runs_dir,
                runs,
                scheme,
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
            out["paired"][f"{ref}_vs_{other}"] = paired((cells, ref), (cells, other), common)
    # 敏感性运行：同一批题，写入与主方法相同（成本见主方法），只看准确率与回答成本
    for name, (tag, method, base) in SENSITIVITY.items():
        variant, variant_runs, _ = load_cells(runs_dir, f"{tag}-{prefix}")
        if base not in methods_present or not all((s, method) in variant for s in common):
            continue
        entry: dict[str, Any] = {
            "runs": variant_runs,
            **accuracy_stats([variant[(s, method)] for s in common]),
            "write": f"same as {base}",
            "paired": {f"{name}_vs_{base}": paired((variant, method), (cells, base), common)},
        }
        for ref in REFERENCES:
            if ref in methods_present:
                entry["paired"][f"{ref}_vs_{name}"] = paired(
                    (cells, ref), (variant, method), common
                )
        out.setdefault("sensitivity", {})[name] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{args.protocol}-{args.dataset}.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"{args.protocol} / {args.dataset}：{len(common)} 题，运行 {runs}")
    print(f"外部基线的写入目录：{eval_spaces}（会话 ID：{scheme}）")
    for method, row in out["methods"].items():
        lo, hi = row["ci95"]
        write = row["write"]
        for problem, what in (
            ("missing", "找不到写入记录"),
            ("mismatched", "的写入记录与运行的协议或会话 ID 写法不符"),
        ):
            if write.get(problem):
                print(f"  ! {method}：{write[problem]:.0f} 份对话历史{what}，没有计入写入成本")
        print(
            f"  {LABELS.get(method, method):<34} {row['accuracy']:.1%} [{lo:.1%}, {hi:.1%}]"
            f"  回答 ${row['answer_cost_per_1k']:.2f}/1k q"
            f"  写入 ${write.get('spent', write.get('cost', 0)):.3f}"
            f", {write.get('seconds', 0):.0f} s"
        )
    for name, row in out["paired"].items():
        print(f"  {name}: +{row['wins']} / -{row['losses']}  p={row['p']:.3g}")
    for name, entry in out.get("sensitivity", {}).items():
        lo, hi = entry["ci95"]
        print(
            f"  敏感性 {name}（{', '.join(entry['runs'])}）："
            f"{entry['accuracy']:.1%} [{lo:.1%}, {hi:.1%}]"
            f"  回答 ${entry['answer_cost_per_1k']:.2f}/1k q"
            f"  {entry['answer_tokens_mean']:,} tokens"
        )
        for pair, row in entry["paired"].items():
            print(f"    {pair}: +{row['wins']} / -{row['losses']}  p={row['p']:.3g}")
    print(f"→ {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
