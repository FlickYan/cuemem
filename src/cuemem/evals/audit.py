"""判分抽检：随机抽取若干条判分结果，导出成可勾选的 Markdown，人工核对后统计一致率。

判对与判错各抽一半：两类错误（把错的判成对、把对的判成错）都需要检查。
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_AGREE = "- [x] 同意"
_DISAGREE = "- [x] 不同意"


def pick_rows(rows: Sequence[dict[str, Any]], n: int, *, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    positive = [r for r in rows if r["correct"]]
    negative = [r for r in rows if not r["correct"]]
    rng.shuffle(positive)
    rng.shuffle(negative)
    half = n // 2
    take_neg = min(len(negative), half)
    take_pos = min(len(positive), n - take_neg)
    take_neg = min(len(negative), n - take_pos)  # 正例不够时用反例补足
    picked = positive[:take_pos] + negative[:take_neg]
    rng.shuffle(picked)
    return picked


def export_audit(rows: Sequence[dict[str, Any]], path: Path, *, n: int = 50, seed: int = 0) -> int:
    picked = pick_rows(rows, n, seed=seed)
    lines = [
        "# 判分抽检",
        "",
        '逐条检查 LLM 判分是否正确：同意就把"同意"前的 `[ ]` 改成 `[x]`，不同意就勾"不同意"。',
        "完成后运行 `uv run cuemem eval audit <运行名> --score` 统计一致率。",
        "",
    ]
    for i, row in enumerate(picked, start=1):
        verdict = "正确" if row["correct"] else "错误"
        lines += [
            f"## {i}. `{row['sample_id']}` · {row['method']} · {row['question_type']}",
            "",
            f"- **问题**：{_one_line(row['question'])}",
            f"- **标准答案**：{_one_line(row['answer'])}",
            f"- **模型回答**：{_one_line(row['response'])}",
            f"- **判分**：{verdict}（judge 原始回复：`{_one_line(row['judge_raw'])}`）",
            "",
            "- [ ] 同意",
            "- [ ] 不同意",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(picked)


def score_audit(path: Path) -> tuple[int, int, int]:
    """返回（同意数，不同意数，未勾选数）。"""
    text = path.read_text(encoding="utf-8")
    items = re.split(r"\n## \d+\. ", text)[1:]
    agree = sum(1 for item in items if _AGREE in item and _DISAGREE not in item)
    disagree = sum(1 for item in items if _DISAGREE in item and _AGREE not in item)
    return agree, disagree, len(items) - agree - disagree


def _one_line(text: str, limit: int = 600) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"
