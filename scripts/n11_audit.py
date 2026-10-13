"""N11：为人工判分抽检准备 50 条——N9 测试集上主要方法的回答。

N2 留下的"人工抽检 50 条判分"一直没做。最终结论来自 N9 的测试集，所以直接抽 N9 的判分：
按数据集分层（LongMemEval 15、LoCoMo 15、MemLens 32K 10、MemLens 128K 10），每层判对、判错各半。
导出到 runs/eval/n9-final/judge-audit.md（不进仓库）。逐条勾选"同意 / 不同意"后统计：

    uv run python scripts/n11_audit.py            # 导出（已存在时不覆盖）
    uv run cuemem eval audit n9-final --score     # 勾选完后统计一致率
"""

from __future__ import annotations

from pathlib import Path

from cuemem.config import load_settings
from cuemem.evals.audit import export_audit, pick_rows
from cuemem.evals.report import load_rows

# 数据集 → （N9 的运行名，抽几条）
STRATA = {
    "longmemeval_s": ("test400-n9", 15),
    "locomo": ("locomo-test7-n9", 15),
    "memlens_32k": ("memlens32-test-n9", 10),
    "memlens_128k": ("memlens128-test-n9", 10),
}
METHODS = {"B1", "B2", "B4-nobm25", "B4-see", "B5", "B5-see", "B6", "B3", "B6a"}
SEED = 11


def main() -> None:
    settings = load_settings()
    path: Path = settings.paths.eval_runs / "n9-final" / "judge-audit.md"
    if path.exists():
        print(f"{path} 已存在：为避免覆盖你的勾选，没有重新导出")
        return
    picked = []
    for dataset, (run, n) in STRATA.items():
        rows = [
            row
            for row in load_rows(settings.paths.eval_runs / run, "results.jsonl")
            if row["method"] in METHODS and not str(row["judge_raw"]).startswith("auto_zero")
        ]
        chosen = pick_rows(rows, n, seed=SEED)
        print(f"{dataset}: {len(chosen)} 条（判对 {sum(r['correct'] for r in chosen)}）")
        picked += chosen
    count = export_audit(picked, path, n=len(picked), seed=SEED)
    print(f"已导出 {count} 条 → {path}")


if __name__ == "__main__":
    main()
