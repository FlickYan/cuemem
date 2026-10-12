"""N9：错误分析。

两步：

1. dump：从某次运行里按题型分层抽主方法答错的题，连同证据发言、控制器选的档、检索是否命中、
   证据图片有没有附上（MemLens）、ORACLE 是否答对，导出成便于人工阅读的 Markdown
   （写到 runs/eval/n9-final/，不提交：LoCoMo 是 CC BY-NC）；
2. summarize：先按规则自动分类（拒答题作答 / 控制器没检索 / 误拒答 / 检索未命中 /
   命中但答错且 ORACLE 答对 / ORACLE 也答错），再读人工标注
   （docs/results/n9-final/error_labels.json，只存题号与标签），
   汇总成 docs/results/n9-final/errors.json。

人工标注的标签：
- missing：证据没有进上下文（检索没取到，或取到的轮里不含关键的那句）；
- outdated：用了过时的版本（知识更新），或没注意到后来的更正；
- temporal：日期推算错（相对时间、间隔、先后）；
- aggregation：计数、求和、列举漏了或重复了；
- false_refusal：证据在上下文里，却说"不知道"；
- hallucination：编了证据里没有的内容（包括拒答题该说"不知道"却作答）；
- reasoning：证据都在、也没漏，推理或理解错了；
- perception：图片附上了，却看错或没看出细节（计数、颜色、读数、位置）；
- routing：控制器选错了档：该检索却选了 L0；或选了 L1、换 L2（附图与否不变）就能答对；
  或该附原图却没附（反过来 L2 答错而 L1 答对不算，那是 L2 自身的波动）；
- judge：回答其实对（同一件东西换了个叫法，或只是不如标准答案具体），判分判错；
- dataset：标准答案与对话本身矛盾或推不出来（如 MemLens 知识更新题里会话日期与叙述顺序
  不一致，按日期最新的说法作答反被判错；LoCoMo 标准答案的年份或天数算错、答案只在图片里）。

自动分类里的 routed_l0 指控制器选了 L0（只给画像、不检索）。

用法：
    uv run python scripts/n9_errors.py dump --run test400-n9 --method B6a --n 40
    uv run python scripts/n9_errors.py summarize
"""

from __future__ import annotations

import argparse
import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cuemem.config import PROJECT_ROOT, load_settings
from cuemem.core.ids import stable_id
from cuemem.evals.datasets import EvalSet, Sample, dataset_path, sample_evidence_turns
from cuemem.evals.datasets import load as load_dataset
from cuemem.evals.report import load_rows
from cuemem.read.probe import DONT_KNOW
from cuemem.write.online import media_id

RUNS = PROJECT_ROOT / "runs" / "eval"
OUT = PROJECT_ROOT / "docs" / "results" / "n9-final"
LABELS = OUT / "error_labels.json"
LABEL_NAMES = (
    "missing",
    "outdated",
    "temporal",
    "aggregation",
    "false_refusal",
    "hallucination",
    "reasoning",
    "perception",
    "routing",
    "judge",
    "dataset",
)


def _field(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if isinstance(value, str) and value[:1] in "{[":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def route_of(row: dict[str, Any]) -> dict[str, Any]:
    details = _field(row, "details") or {}
    return details.get("route") or {} if isinstance(details, dict) else {}


def auto_category(row: dict[str, Any], oracle: dict[str, bool]) -> str:
    """按规则给一道错题分类（与评测报告的错误归因同口径，另加拒答、误拒答与控制器没检索）。"""
    if row["is_abstention"]:
        return "answered_unanswerable"
    if route_of(row).get("level") == "L0":
        return "routed_l0"
    if DONT_KNOW.search(row.get("response") or ""):
        return "false_refusal"
    retrieval = _field(row, "retrieval") or {}
    if not retrieval:
        return "no_retrieval"
    hit = retrieval.get("turn_recall_any", retrieval.get("session_recall_any", 0.0)) > 0
    if not hit:
        return "retrieval_miss"
    if row["sample_id"] not in oracle:
        return "hit_but_wrong"
    return "reading" if not oracle[row["sample_id"]] else "noisy_or_incomplete"


def wrong_rows(run: str, method: str) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows = load_rows(RUNS / run, "results.jsonl")
    oracle = {r["sample_id"]: bool(r["correct"]) for r in rows if r["method"] == "ORACLE"}
    wrong = [r for r in rows if r["method"] == method and not r["correct"]]
    return sorted(wrong, key=lambda r: r["sample_id"]), oracle


def load_evalset(run: str) -> tuple[str, EvalSet]:
    info = json.loads((RUNS / run / "run.json").read_text("utf-8"))
    dataset = str(info["config"]["dataset"])
    settings = load_settings()
    return dataset, load_dataset(dataset, dataset_path(settings.paths.datasets, dataset))


def needle_images(dataset: str, evalset: EvalSet, sample: Sample) -> set[str]:
    """MemLens：证据会话里关键图片（needle_images/）的记录 ID，与回答时附上的原图 ID 同一算法。"""
    space_id = f"{dataset}:{sample.space_key}"
    return {
        media_id(space_id, stable_id(space_id, s.session_id, idx), i)
        for s in evalset.sessions_of(sample)
        for idx, t in enumerate(s.turns)
        if t.has_answer
        for i, image in enumerate(t.images)
        if "needle_images/" in image
    }


def photo_status(row: dict[str, Any], needles: set[str]) -> str:
    """回答时附原图的情况：没附 / 附了但不含证据图 / 附上了证据图。"""
    details = _field(row, "details") or {}
    attached = set(details.get("attached") or []) if isinstance(details, dict) else set()
    if not attached:
        return "none"
    return "needle" if attached & needles else "other"


def dump(run: str, method: str, n: int, seed: int) -> Path:
    wrong, oracle = wrong_rows(run, method)
    dataset, evalset = load_evalset(run)
    samples = {s.sample_id: s for s in evalset.samples}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in wrong:
        by_type["abstention" if r["is_abstention"] else r["question_type"]].append(r)
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    quotas = {t: max(1, round(n * len(v) / len(wrong))) for t, v in by_type.items()}
    for t, rows in sorted(by_type.items()):
        chosen.extend(rng.sample(rows, min(len(rows), quotas[t])))
    lines = [f"# {run} · {method} · 错题 {len(wrong)} 道，抽 {len(chosen)} 道\n"]
    for r in chosen:
        sample = samples[r["sample_id"]]
        sessions = evalset.sessions_of(sample)
        marks = sample_evidence_turns(sample, sessions)
        evidence = [
            f"  - [{s.session_id} · {s.started_at:%Y-%m-%d}] {t.speaker or t.role}: {t.text[:300]}"
            + (f"（附图 {len(t.images)} 张）" if t.images else "")
            for s in sessions
            for i, t in enumerate(s.turns)
            if (s.session_id, i) in marks
        ][:6]
        route = route_of(r)
        retrieval = _field(r, "retrieval") or {}
        hit = retrieval.get("turn_recall_any", retrieval.get("session_recall_any"))
        lines += [
            f"## {r['sample_id']}（{r['question_type']}；自动分类 {auto_category(r, oracle)}；"
            f"ORACLE {'对' if oracle.get(r['sample_id']) else '错'}）",
            f"- 提问时间：{sample.question_time:%Y-%m-%d}",
            f"- 问题：{r['question']}",
            f"- 标准答案：{r['answer']}",
            f"- 回答：{(r.get('response') or '')[:600]}",
            f"- 控制器：{route.get('level', '—')}{'+see' if route.get('see') else ''}"
            f"（{'；'.join(route.get('reasons') or []) or '—'}）；检索命中证据：{hit}",
        ]
        if dataset.startswith("memlens"):
            status = photo_status(r, needle_images(dataset, evalset, sample))
            lines.append(f"- 附原图：{status}")
        lines += ["- 证据发言：", *(evidence or ["  - （数据集没有标到发言）"]), ""]
    path = RUNS / "n9-final" / f"errors-{run}-{method}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def summarize() -> dict[str, Any]:
    labels: dict[str, dict[str, str]] = {}
    if LABELS.exists():
        labels = json.loads(LABELS.read_text("utf-8"))
    out: dict[str, Any] = {}
    for key, rows_labels in labels.items():
        run, method = key.split(":")
        wrong, oracle = wrong_rows(run, method)
        auto = Counter(auto_category(r, oracle) for r in wrong)
        types = {r["sample_id"]: r["question_type"] for r in wrong}
        photos: Counter[str] = Counter()
        if run.startswith("memlens"):
            dataset, evalset = load_evalset(run)
            samples = {s.sample_id: s for s in evalset.samples}
            for r in wrong:
                needles = needle_images(dataset, evalset, samples[r["sample_id"]])
                photos[photo_status(r, needles)] += 1
        manual = Counter(rows_labels.values())
        by_type: dict[str, Counter[str]] = defaultdict(Counter)
        for sample_id, label in rows_labels.items():
            by_type[types.get(sample_id, "?")][label] += 1
        out[key] = {
            "wrong": len(wrong),
            "auto": dict(auto),
            **({"photos": dict(photos)} if photos else {}),
            "reviewed": len(rows_labels),
            "manual": dict(manual),
            "manual_by_type": {t: dict(c) for t, c in by_type.items()},
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "errors.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--method", required=True)
    d.add_argument("--n", type=int, default=40)
    d.add_argument("--seed", type=int, default=0)
    sub.add_parser("summarize")
    args = parser.parse_args()
    if args.command == "dump":
        print(dump(args.run, args.method, args.n, args.seed))
    else:
        print(json.dumps(summarize(), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
