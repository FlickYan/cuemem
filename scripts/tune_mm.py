"""多模态纯检索评测（MemLens）：比较图片描述、孤立描述、以文搜图对召回的影响，不调用 reader。

MemLens 的证据只标到会话；另外每道题的关键图片放在 release_images/needle_images/ 下，
据此多看两个指标：关键图片所在的轮有没有装进上下文（needle 轮召回）、附上的原图里有没有关键图片
（附图命中）。只用到通义的向量接口，不受 DeepSeek 高峰影响。依赖构建好的记忆空间：
    uv run cuemem eval run --dataset memlens_32k --split dev100 --methods B4 --build-only

用法：
    uv run python scripts/tune_mm.py --dataset memlens_32k --split dev100
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from cuemem.config import load_settings
from cuemem.embed import build_embedder, build_image_embedder
from cuemem.evals.datasets import Sample, dataset_path
from cuemem.evals.datasets import load as load_dataset
from cuemem.evals.runner import space_dir
from cuemem.evals.splits import load_split
from cuemem.llm.client import LLMClient
from cuemem.llm.ledger import Ledger
from cuemem.memory import MemorySystem
from cuemem.read.l1 import L1Config

CAPTIONS = L1Config(fact_weight=0.0, attach_images=3)
VARIANTS: dict[str, L1Config] = {
    "纯文本（图片只留占位）": replace(CAPTIONS, image_kind=None),
    "对话 + 上下文感知描述": CAPTIONS,
    "对话 + 孤立描述": replace(CAPTIONS, image_kind="image_iso"),
    "对话 + 描述 + 事实（B4-mm）": replace(CAPTIONS, fact_weight=1.0),
    "B4-mm + 以文搜图 ×0.5": replace(CAPTIONS, fact_weight=1.0, visual_weight=0.5),
    "B4-mm + 以文搜图 ×1.0": replace(CAPTIONS, fact_weight=1.0, visual_weight=1.0),
    "只用以文搜图（不用描述键）": replace(CAPTIONS, image_kind=None, visual_weight=1.0),
    "对话 + 描述 + 以文搜图 ×0.5（不用事实）": replace(CAPTIONS, visual_weight=0.5),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="memlens_32k")
    parser.add_argument("--split", default="dev100")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    settings = load_settings()
    path = dataset_path(settings.paths.datasets, args.dataset)
    evalset = load_dataset(args.dataset, path, ids=load_split(args.dataset, args.split))
    needles = {
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (path.parent / "release_images" / "needle_images").iterdir()
    }
    ledger = Ledger(None)
    embedder = build_embedder(settings, ledger=ledger)
    image_embedder = build_image_embedder(settings, ledger=ledger)
    llm = LLMClient(settings)
    samples = [s for s in evalset.samples if not s.is_abstention]

    def evaluate(sample: Sample) -> dict[str, Any] | None:
        directory = space_dir(settings, args.dataset, embedder, sample.space_key)
        if not (directory / "build.json").exists():
            return None  # 还没构建完（或正被构建进程占用）：跳过
        try:
            system = MemorySystem.on_disk(
                settings,
                directory,
                embedder=embedder,
                llm=llm,
                ledger=ledger,
                image_embedder=image_embedder,
            )
        except RuntimeError:
            return None
        space_id = f"{args.dataset}:{sample.space_key}"
        wanted = set(sample.evidence_session_ids)
        out: dict[str, Any] = {"type": sample.question_type, "rows": {}}
        try:
            media = system.db.get_media(space_id)
            # 这道题的关键图片：证据会话里、放在 needle_images 下的图（别的题的关键图片
            # 也可能作为干扰出现在这份历史里，不算）
            key_media = [m for m in media if m.sha256 in needles and _in(m, wanted, system)]
            needle_turns = {m.source_turn_id for m in key_media}
            key_shas = {m.sha256 for m in key_media}
            for name, cfg in VARIANTS.items():
                result = system.l1.search(space_id, sample.question, sample.question_time, cfg)
                sessions = {h.unit.session_id for h in result.rounds}
                packed_turns = {t for h in result.rounds for t in h.unit.source_turn_ids}
                attached = {result.media[m].sha256 for m in result.attach if m in result.media}
                out["rows"][name] = {
                    "session_recall_all": len(sessions & wanted) / len(wanted) if wanted else 1.0,
                    "needle_round": (
                        len(needle_turns & packed_turns) / len(needle_turns)
                        if needle_turns
                        else None
                    ),
                    "needle_attached": (float(bool(attached & key_shas)) if needle_turns else None),
                    "rounds": len(result.rounds),
                }
        finally:
            system.close()
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [r for r in pool.map(evaluate, samples) if r is not None]
    samples = samples[: len(results)] if len(results) < len(samples) else samples

    print(f"## 多模态纯检索指标（{args.dataset} · {args.split}，不含拒答题，{len(results)} 题）\n")
    columns = ["配置", "会话召回（全部）", "关键图片所在轮的召回", "附图命中关键图片（前 3 张）"]
    print("| " + " | ".join([*columns, "平均轮数"]) + " |")
    print("|---|---|---|---|---|")
    for name in VARIANTS:
        rows = [r["rows"][name] for r in results]
        needle = [r["needle_round"] for r in rows if r["needle_round"] is not None]
        attach = [r["needle_attached"] for r in rows if r["needle_attached"] is not None]
        print(
            f"| {name} | {statistics.mean(r['session_recall_all'] for r in rows):.1%}"
            f" | {statistics.mean(needle):.1%}（{len(needle)} 题）"
            f" | {statistics.mean(attach):.1%}"
            f" | {statistics.mean(r['rounds'] for r in rows):.1f} |"
        )
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for name, row in r["rows"].items():
            if row["needle_round"] is not None:
                by_type[r["type"]][name].append(row["needle_round"])
    print("\n## 分题型：关键图片所在轮的召回\n")
    print("| 配置 | " + " | ".join(sorted(by_type)) + " |")
    print("|---|" + "---|" * len(by_type))
    for name in VARIANTS:
        cells = [
            f"{statistics.mean(by_type[t][name]):.0%}" if by_type[t][name] else "—"
            for t in sorted(by_type)
        ]
        print(f"| {name} | " + " | ".join(cells) + " |")


def _in(item: Any, sessions: set[str], system: MemorySystem) -> bool:
    turn = system.db.get_turns(item.space_id, ids=[item.source_turn_id])
    return bool(turn) and turn[0].session_id in sessions


if __name__ == "__main__":
    main()
