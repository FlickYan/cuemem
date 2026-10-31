"""N9：把 docs/results/n9-final/results.json 与 errors.json（人工错误标注的汇总）渲染成表格。

结果页与技术报告里的数字都从这里来，避免手抄出错。配对比较旁边的"Holm 校正后"取自同目录的
results-holm.json（scripts/multiplicity.py）。由 scripts/n9_report.py 在最后调用，也可以单独
运行（只读这三个 JSON、只写 tables.md，不需要评测运行）：
    uv run python scripts/n9_tables.py
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cuemem.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "results" / "n9-final"

NAMES = {
    "B0": "B0 无记忆",
    "L0": "L0 只给核心记忆",
    "B1": "B1 全上下文",
    "B2": "B2 朴素检索",
    "B2F": "B2F 朴素检索 + 写入路径",
    "B2F-current": "B2F 只给当前事实",
    "B2F-plain": "B2F 事实不标有效期",
    "B4-mm": "L1（不附原图）",
    "B4-nobm25": "L1 快速召回",
    "B4-see": "L1 + 再感知",
    "B4-note": "L1 + 先记笔记",
    "B5": "L2 深度检索",
    "B5-see": "L2 + 再感知",
    "B5-nonote": "L2 去掉笔记",
    "B5-noiter": "L2 去掉补检索",
    "B5-ppr": "L2 + 图扩散",
    "B6r": "B6r 规则版控制器",
    "B6": "B6 学到的控制器",
    "B6h": "B6h 拿不准时问 LLM",
    "B6z": "B6z 零样本 LLM 路由",
    "B3": "B3 模型自主检索",
    "B6a": "B6a 控制器先选、模型兜底",
    "ORACLE": "ORACLE 只给证据会话",
}
ORDER = list(NAMES)
DATASETS = {
    "longmemeval_s": "LongMemEval test400",
    "locomo": "LoCoMo 后 7 段对话",
    "memlens_32k": "MemLens 32K",
    "memlens_128k": "MemLens 128K",
}
POLICY_NAMES = {
    "always-L0": "总是 L0",
    "always-L1": "总是 L1",
    "always-L2": "总是 L2",
    "always-L1 (no photos)": "总是 L1（不附原图）",
    "always-L2 (no photos)": "总是 L2（不附原图）",
    "B6 (frozen)": "**B6 冻结的控制器（λ = 30）**",
    "type-oracle": "已知真实题型的路由（参照）",
    "oracle": "事后理想路由（参照）",
}
READER_ORDER = ("qwen3.6-35b-a3b", "qwen3.8-flash", "deepseek-flash", "deepseek-v4-pro")
LABELS = {
    "missing": "证据没进上下文",
    "outdated": "用了过时的版本",
    "temporal": "日期推算",
    "aggregation": "计数 / 求和 / 列举",
    "false_refusal": "误拒答",
    "hallucination": "编造",
    "reasoning": "推理或理解",
    "perception": "看错图",
    "routing": "控制器选错档",
    "judge": "判分过严",
    "dataset": "标准答案有误或推不出",
}
AUTO = {
    "answered_unanswerable": "拒答题作答",
    "routed_l0": "控制器没检索",
    "false_refusal": "误拒答",
    "no_retrieval": "没有检索",
    "retrieval_miss": "检索未命中",
    "noisy_or_incomplete": "命中、ORACLE 答对",
    "reading": "命中、ORACLE 也错",
    "hit_but_wrong": "命中但答错",
}


def pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return [*lines, ""]


def p_value(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def holm_families() -> dict[str, Any]:
    """results-holm.json 里的族；还没有这个文件时为空（表里的校正列都写"—"）。"""
    path = OUT / "results-holm.json"
    if not path.exists():
        return {}
    families: dict[str, Any] = json.loads(path.read_text("utf-8"))["families"]
    return families


def holm_cell(families: dict[str, Any], family: str, key: str, mirror: str | None = None) -> str:
    """一个比较在族内 Holm 校正后的 p。族里同一对方法只留一个方向（Families 跳过镜像），所以
    表里反过来写的那一行要用 mirror 查；+0 / −0 不算检验、不在族里，写"—"。"""
    tests = families.get(family, {}).get("tests", {})
    test = tests.get(key) or (tests.get(mirror) if mirror is not None else None)
    return "—" if test is None else p_value(test["p_holm"])


def main_tables(results: dict[str, Any]) -> list[str]:
    out = ["## 主表", ""]
    for dataset, label in DATASETS.items():
        stats = results["main"].get(dataset)
        if not stats:
            continue
        out += [f"### {label}", ""]
        rows = []
        for method in [m for m in ORDER if m in stats]:
            s = stats[method]
            lo, hi = s["ci"]
            rows.append(
                [
                    NAMES[method],
                    s["n"],
                    pct(s["accuracy"]),
                    f"{100 * lo:.1f}–{100 * hi:.1f}",
                    f"${1000 * s['cost']:.2f}",
                    f"{s['input_tokens']:,.0f}",
                ]
            )
        out += table(
            ["方法", "题数", "准确率", "95% 置信区间", "每千题名义成本", "每题输入 tokens"], rows
        )
        types = sorted({t for s in stats.values() for t in (s.get("by_type") or {})})
        if any(s.get("abstention") for s in stats.values()):
            types.append("abstention")
        key = [
            m
            for m in ("B1", "B4-nobm25", "B4-see", "B5", "B5-see", "B6", "B3", "B6a")
            if m in stats
        ]
        if types and key:
            rows = []
            for method in key:
                by = dict(stats[method].get("by_type") or {})
                if stats[method].get("abstention"):
                    by["abstention"] = stats[method]["abstention"]
                rows.append([NAMES[method]] + [pct(by[t][1]) if t in by else "—" for t in types])
            first = dict(stats[key[0]].get("by_type") or {})
            if stats[key[0]].get("abstention"):
                first["abstention"] = stats[key[0]]["abstention"]
            counts = [str(first.get(t, [0])[0]) for t in types]
            out += table(
                ["分题型"] + [f"{t}（{c}）" for t, c in zip(types, counts, strict=True)], rows
            )
    return out


def controller_tables(results: dict[str, Any]) -> list[str]:
    out = ["## 冻结的控制器", ""]
    holm = holm_families()
    for dataset, label in DATASETS.items():
        sec = results["controller"].get(dataset)
        if not sec:
            continue
        agree = sec["live_replay_agreement"]
        out += [
            f"### {label}（{sec['n']} 题；B6 真实运行与冻结模型重放逐题一致 "
            f"{agree[0]} / {agree[1]}；"
            f"探针 p50 {sec['probe_ms']['p50']:.0f} ms、p95 {sec['probe_ms']['p95']:.0f} ms）",
            "",
        ]
        rows = []
        for name, p in sec["policies"].items():
            rows.append(
                [
                    POLICY_NAMES.get(name, NAMES.get(name, name)),
                    f"{p['accuracy']:.1f}%",
                    f"${p['cost_per_1k']:.2f}",
                    f"{p['l0']:.0f}%",
                    f"{p['l2']:.0f}%",
                ]
            )
        out += table(["策略", "准确率", "每千题名义成本", "走 L0", "走 L2"], rows)
        vs = sec.get("b6_vs_l2") or {}
        if vs:
            parts = [
                f"{k.replace('lost_at_', '输在 ').replace('won_at_', '赢在 ')} {v}"
                for k, v in vs.items()
            ]
            out += [f"B6 与总是 L2 逐题比：{'，'.join(parts)}（按 B6 选的档）。", ""]
        rows = []
        for a, tests in sec["tests"].items():
            for b, t in tests.items():
                rows.append(
                    [
                        f"{a} 对 {b}",
                        f"+{t['wins']} / −{t['losses']}",
                        p_value(t["p"]),
                        holm_cell(holm, f"{dataset}：控制器", f"{a} vs {b}", f"{b} vs {a}"),
                    ]
                )
        out += table(["配对比较", "胜 / 负", "符号检验", "Holm 校正后"], rows)
    return out


def mixed_tables(results: dict[str, Any]) -> list[str]:
    mixed = results.get("mixed")
    if not mixed:
        return []
    no_memory = sum(mixed["no_memory"].values())
    out = [
        "## 混合流（测试集版）",
        "",
        f"记忆题 {mixed['memory']} 道，与记忆无关的查询 {no_memory} 条（同话题难负例 "
        f"{mixed['no_memory'].get('mixed:topic', 0)}、Dolly 通用请求 "
        f"{mixed['no_memory'].get('mixed:dolly', 0)}），"
        f"按记忆题占 {100 * mixed['ratio']:.0f}% 加权。"
        f"B6h 对 {mixed['hybrid_asked']:.1f}% 的查询问了 LLM。",
        "",
    ]
    rows = []
    for name in ("always-L1", "always-L2", "B6r", "B6z", "B6h", "B6"):
        m = mixed[name]
        rows.append(
            [
                POLICY_NAMES.get(name, NAMES.get(name, name)),
                f"{m['memory_accuracy']:.1f}%",
                f"${m['cost_per_1k']:.2f}",
                f"{m['false_recall']:.1f}%",
                f"{m['memory_l0']:.1f}%",
            ]
        )
    out += table(
        [
            "策略",
            "记忆题准确率",
            "每千条名义成本",
            "误召回（无关查询走了检索）",
            "漏召回（记忆题走了 L0）",
        ],
        rows,
    )
    auc = mixed["gate_auc"]
    parts = [f"{k} {v:.3f}" for k, v in auc.items()]
    out += [f"冻结的门控区分记忆题与无关查询的 AUC：{'，'.join(parts)}。", ""]
    rows = [
        [
            DATASETS.get(home, home),
            f"{d['B6_memory_accuracy']:.1f}% / {d['L1_memory_accuracy']:.1f}%",
            f"${d['B6_cost_per_1k']:.2f} / ${d['L1_cost_per_1k']:.2f}",
            f"{d['false_recall']:.1f}%",
            f"{d['memory_l0']:.1f}%",
        ]
        for home, d in mixed["by_dataset"].items()
    ]
    out += table(
        ["记忆空间", "记忆题准确率 B6 / 总是 L1", "每千条成本 B6 / 总是 L1", "误召回", "漏召回"],
        rows,
    )
    rows = [
        [
            f"{c['lam']:g}",
            f"{c['memory_accuracy']:.1f}%",
            f"${c['cost_per_1k']:.2f}",
            f"{c['false_recall']:.1f}%",
        ]
        for c in mixed["curve"]
    ]
    out += ["λ 扫描（混合流）：", "", *table(["λ", "记忆题准确率", "每千条成本", "误召回"], rows)]
    return out


def comparison_tables(results: dict[str, Any]) -> list[str]:
    out = ["## 配对比较", ""]
    holm = holm_families()
    for dataset, label in DATASETS.items():
        comps = results["comparisons"].get(dataset)
        if not comps:
            continue
        out += [f"### {label}", ""]
        rows = []
        for name, c in comps.items():
            a, b = name.split(" vs ")
            for group in sorted(c, key=lambda k: k != "all"):  # "全部"那一行放在最前
                g = c[group]
                if group != "all" and g["n"] < 10:
                    continue
                rows.append(
                    [
                        f"{NAMES.get(a, a)} 对 {NAMES.get(b, b)}" if group == "all" else "",
                        "全部" if group == "all" else group,
                        g["n"],
                        f"{g[a]:.1f}% / {g[b]:.1f}%",
                        f"+{g['wins']} / −{g['losses']}",
                        p_value(g["p"]),
                        holm_cell(holm, f"{dataset}：方法对比（总体）", name)
                        if group == "all"
                        else holm_cell(holm, f"{dataset}：{name}（分题型）", group),
                    ]
                )
        out += table(["比较", "题型", "题数", "准确率", "胜 / 负", "符号检验", "Holm 校正后"], rows)
    return out


def reader_tables(results: dict[str, Any]) -> list[str]:
    readers = results.get("readers") or {}
    if not readers:
        return []
    out = ["## 换回答模型（H3）", ""]
    methods = ("L0", "B4-nobm25", "B5", "B6r", "B6", "B3", "B6a")
    holm = holm_families()
    for dataset, sec in readers.items():
        out += [f"### {DATASETS[dataset]}（{sec['n']} 题）", ""]
        rows = []
        for reader in [r for r in READER_ORDER if r in sec["readers"]]:
            entry = sec["readers"][reader]
            rows.append(
                [reader]
                + [
                    f"{entry[m]['accuracy']:.1f}%（${entry[m]['cost_per_1k']:.2f}）"
                    if m in entry
                    else "—"
                    for m in methods
                ]
            )
        out += table(["回答模型"] + [NAMES[m] for m in methods], rows)
        rows = []
        for reader in [r for r in READER_ORDER if r in sec["readers"]]:
            for name, t in sec["readers"][reader]["tests"].items():
                rows.append(
                    [
                        reader,
                        name,
                        f"+{t['wins']} / −{t['losses']}",
                        p_value(t["p"]),
                        holm_cell(holm, f"{dataset}：换回答模型（{reader}）", name),
                    ]
                )
        out += table(["回答模型", "配对比较", "胜 / 负", "符号检验", "Holm 校正后"], rows)
    return out


def photo_tables(results: dict[str, Any]) -> list[str]:
    photos = results.get("photos") or {}
    if not photos:
        return []
    out = ["## MemLens：回答时附的原图里有没有证据图", ""]
    rows = []
    for dataset, by_method in photos.items():
        for method, p in by_method.items():
            rows.append(
                [
                    DATASETS[dataset],
                    NAMES.get(method, method),
                    p["image_questions"],
                    f"{p['needle_attached']:.1f}%",
                    f"{p['needle_first']:.1f}%",
                    "—" if p["accuracy_attached"] is None else f"{p['accuracy_attached']:.1f}%",
                    (
                        "—"
                        if p["accuracy_not_attached"] is None
                        else f"{p['accuracy_not_attached']:.1f}%（{p['not_attached']} 题）"
                    ),
                ]
            )
    out += table(
        [
            "数据集",
            "方法",
            "看图题",
            "附上了证据图",
            "排第一的是证据图",
            "附上时准确率",
            "没附上时准确率",
        ],
        rows,
    )
    return out


def error_tables(results: dict[str, Any]) -> list[str]:
    out = ["## 错误分析", ""]
    rows = []
    for dataset, by_method in results.get("errors", {}).items():
        for method, e in by_method.items():
            attribution = "，".join(f"{AUTO.get(k, k)} {v}" for k, v in e["attribution"].items())
            rows.append(
                [
                    DATASETS[dataset],
                    NAMES.get(method, method),
                    f"{e['wrong']} / {e['n']}",
                    attribution or "—",
                    e["false_abstention"],
                    e["answered_unanswerable"],
                ]
            )
    out += table(["数据集", "方法", "错题", "自动归因", "误拒答", "拒答题作答"], rows)
    path = OUT / "errors.json"
    if path.exists():
        manual = json.loads(path.read_text("utf-8"))
        for key, e in manual.items():
            run, method = key.split(":")
            out += [
                f"### {run} · {NAMES.get(method, method)}："
                f"人工看了 {e['reviewed']} / {e['wrong']} 道错题",
                "",
            ]
            rows = [
                [LABELS.get(k, k), v, f"{100 * v / e['reviewed']:.0f}%"]
                for k, v in sorted(e["manual"].items(), key=lambda kv: -kv[1])
            ]
            out += table(["人工标注", "题数", "占比"], rows)
            auto = "，".join(
                f"{AUTO.get(k, k)} {v}" for k, v in sorted(e["auto"].items(), key=lambda kv: -kv[1])
            )
            out += [f"全部 {e['wrong']} 道错题的自动分类：{auto}。", ""]
    return out


def render(results: dict[str, Any] | None = None) -> str:
    """tables.md 的全文；results 为 None 时读 OUT 里的 results.json。"""
    if results is None:
        results = json.loads((OUT / "results.json").read_text("utf-8"))
    lines = [
        "# N9 结果表（自动生成）",
        "",
        "由 `scripts/n9_report.py`（调用 `scripts/n9_tables.py`）"
        "从 [results.json](results.json) 与 [results-holm.json](results-holm.json) 生成，"
        "不要手改。",
        "",
        "- 成本是名义成本（高峰标价、未命中缓存，D-010），每题只算回答。一次性的写入（向量化与"
        "巩固；每千份历史 LongMemEval 约 $43、LoCoMo 约 $35、MemLens 32K 约 $15、128K 约 $54）"
        "不在内，写入的成本表与算上写入之后的对比见 [N9 结果](README.md) 主结果表后的说明"
        "（D-070）。",
        '- "符号检验"是单个比较的 p；"Holm 校正后"是按 [results-holm.json](results-holm.json) '
        '的族（`scripts/multiplicity.py`）做 Holm 校正之后的 p。+0 / −0 不算检验，写"—"。',
        "- LongMemEval 的各节里，B0、L0、B1 与 ORACLE 之外的方法都是在 N12 修好会话 ID 泄漏"
        "（D-067）之前测的，见 [泄漏对照](../n12-leak/README.md)；数字照原样保留。",
        "",
    ]
    for part in (
        main_tables,
        controller_tables,
        mixed_tables,
        comparison_tables,
        reader_tables,
        photo_tables,
        error_tables,
    ):
        lines += part(results)
    return "\n".join(lines)


def write(results: dict[str, Any] | None = None) -> Path:
    path = OUT / "tables.md"
    path.write_text(render(results), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write())
