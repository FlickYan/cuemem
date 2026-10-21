"""多重比较校正（Holm–Bonferroni）：发表结果里的配对符号检验按"族"校正 p 值。

一份结果里做了很多个检验时，单看每个 p < 0.05，总有几个是碰巧的。这里把每份结果里的检验分成族，
族内做 Holm 校正（cuemem.evals.routing.holm_family），写到结果旁边的 <文件名>-holm.json；结果本身
（准确率、胜负、原始 p）一字不动。每一族包含哪些检验：

- N9 docs/results/n9-final/results.json：
  - "<数据集>：方法对比（总体）"：comparisons 里每个对比的总体一行（L2 对 L1、对全上下文……）；
  - "<数据集>：<甲> vs <乙>（分题型）"：同一个对比在各题型上（"四种题型都显著"这类说法）；
  - "<数据集>：控制器"：冻结的控制器、B6a、B3 与各固定档位、规则版和彼此之间的配对；
  - "<数据集>：换回答模型（<模型>）"：同一个回答模型下的几个对比。
- N12 docs/results/n12-baselines/<协议>-<数据集>.json，每个协议一份：
  - "L1、L2 对外部系统"：CueMem L1、L2 各对 Mem0、Graphiti、LangMem、Cognee、A-MEM；
  - "CueMem 与参照"：L1、L2 对全上下文、朴素检索、控制器与彼此；
  - "敏感性（<名字>）"：敏感性运行（A-MEM k = 40）的配对。
- N12 docs/results/n12-leak/summary.json，每个协议：泄漏的逐方法对照、合并各检索方法的对照、
  每一臂里对 Mem0 的比较，各是一族。
- N11 docs/results/n11-improvements/：mem0.json 每个运行的"对 Mem0"；second_judge.json 每个数据集
  在每个判分模型下的关键对比；controller_v2.json 开发集与测试集事后分析（总体、各数据集）、换回答
  模型后的对比与 λ 扫描；see_gate.json 测试集事后分析。
- N7 docs/results/n7-controller/：results.json 每个范围（全部、各数据集）里控制器的配对；h3.json
  每个数据集里换回答模型前后的配对、换成通义之后对 B3 的配对。

一份结果有不止一族时另有一族"全文"：这份结果里全部的检验放在一起校正，是最保守的上界（各族问的
不是同一个问题，通常不该放在一起校正，只用来看哪些结论在最严的口径下也站得住；同一个比较在两节里
各报一次的按两次算）。同一对方法互为镜像（甲对乙、乙对甲）的只算一次；没有一题分出胜负的
（+0 / −0，p = 1）不算一次检验。

输出的每个检验带 wins、losses、p（按胜负重新算的精确符号检验）、p_holm（族内校正后）、nominal
（p < α）与 holm（p_holm < α）；lost 列出名义上显著、校正后不再显著的检验。

用法：
    uv run python scripts/multiplicity.py   # 按 docs/results 里已发表的结果重算全部 *-holm.json

生成这些结果的脚本（n9_report、n12_baselines、n12_leak、n11_mem0、n11_second_judge、
n11_controller_v2、n11_see_gate、n7_controller、n7_h3）写完自己的 JSON 之后调用 update()，
重跑时一并更新、打印校正后的结论。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from cuemem.evals.report import EXTERNAL_SYSTEMS
from cuemem.evals.routing import holm_family

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
ALPHA = 0.05
CUEMEM_REFERENCES = ("B4-nobm25", "B5")  # 与 n12_baselines.REFERENCES 相同：L1、L2
EVERYTHING = "全文"

Test = Mapping[str, Any]


class Families:
    """按族收集一份结果里的检验。pair 给出（甲，乙）时，同一族里已有（乙，甲）的镜像就跳过。"""

    def __init__(self) -> None:
        self.groups: dict[str, tuple[str, dict[str, Test]]] = {}
        self._pairs: dict[str, set[tuple[str, str]]] = {}

    def add(
        self,
        family: str,
        description: str,
        key: str,
        test: Test,
        *,
        pair: tuple[str, str] | None = None,
    ) -> None:
        seen = self._pairs.setdefault(family, set())
        if pair is not None:
            if (pair[1], pair[0]) in seen:
                return
            seen.add(pair)
        self.groups.setdefault(family, (description, {}))[1][key] = test


def report(source: str, families: Families) -> dict[str, Any]:
    """每一族做 Holm 校正；不止一族时另加"全文"一族。"""
    groups = dict(families.groups)
    if len(groups) > 1:
        everything = {
            f"{name} | {key}": test
            for name, (_, tests) in groups.items()
            for key, test in tests.items()
        }
        groups[EVERYTHING] = ("这份结果里全部的检验放在一起校正（最保守的上界）", everything)
    out: dict[str, Any] = {
        "source": source,
        "method": "Holm–Bonferroni（族内校正，族内错判一次以上的概率不超过 alpha）",
        "alpha": ALPHA,
        "families": {},
        "lost": [],
    }
    for name, (description, tests) in groups.items():
        decided = {k: t for k, t in tests.items() if int(t["wins"]) + int(t["losses"]) > 0}
        if not decided:
            continue
        rows = holm_family(decided, ALPHA)
        out["families"][name] = {"description": description, "m": len(rows), "tests": rows}
        out["lost"] += [f"{name} | {k}" for k, r in rows.items() if r["nominal"] and not r["holm"]]
    return out


def holm_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}-holm.json")


def update(path: Path, families: Callable[[dict[str, Any]], Families]) -> dict[str, Any]:
    """按 path 里的结果写 <文件名>-holm.json，打印每一族校正前后显著的个数与不再显著的检验。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    result = report(path.name, families(data))
    target = holm_path(path)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Holm 校正（α = {ALPHA}）→ {target.name}")
    for name, family in result["families"].items():
        rows = family["tests"].values()
        nominal, kept = sum(r["nominal"] for r in rows), sum(r["holm"] for r in rows)
        print(f"  {name}（{family['m']} 个检验）：名义上显著 {nominal} 个，校正后 {kept} 个")
        for key, row in family["tests"].items():
            if name != EVERYTHING and row["nominal"] and not row["holm"]:
                print(
                    f"    不再显著：{key} +{row['wins']} / −{row['losses']} "
                    f"p = {row['p']:.4f} → {row['p_holm']:.4f}"
                )
    return result


# ---------- 各份结果的族 ----------


def n9(results: dict[str, Any]) -> Families:
    fam = Families()
    for dataset, comparisons in results.get("comparisons", {}).items():
        for name, groups in comparisons.items():
            if "all" in groups:
                fam.add(
                    f"{dataset}：方法对比（总体）",
                    "comparisons 里每个对比的总体一行",
                    name,
                    groups["all"],
                )
            for group, test in groups.items():
                if group != "all":
                    fam.add(
                        f"{dataset}：{name}（分题型）",
                        "同一个对比在各题型上（拒答题单列）",
                        group,
                        test,
                    )
    for dataset, section in results.get("controller", {}).items():
        for a, row in section.get("tests", {}).items():
            for b, test in row.items():
                fam.add(
                    f"{dataset}：控制器",
                    "冻结的控制器、B6a、B3 与固定档位、规则版和彼此之间的配对",
                    f"{a} vs {b}",
                    test,
                    pair=(a, b),
                )
    for dataset, section in results.get("readers", {}).items():
        for reader, entry in section.get("readers", {}).items():
            for name, test in entry.get("tests", {}).items():
                fam.add(
                    f"{dataset}：换回答模型（{reader}）",
                    "同一个回答模型下的几个对比",
                    name,
                    test,
                )
    return fam


def n12_baselines(out: dict[str, Any]) -> Families:
    fam = Families()
    for key, test in out.get("paired", {}).items():
        a, _, b = key.partition("_vs_")
        if a in CUEMEM_REFERENCES and b in EXTERNAL_SYSTEMS:
            fam.add(
                "L1、L2 对外部系统",
                "CueMem L1、L2 各对 Mem0、Graphiti、LangMem、Cognee、A-MEM",
                key,
                test,
            )
        else:
            fam.add(
                "CueMem 与参照",
                "L1、L2 对全上下文、朴素检索、控制器与彼此（L1 对 L2 只算一次）",
                key,
                test,
                pair=(a, b),
            )
    for name, entry in out.get("sensitivity", {}).items():
        for key, test in entry.get("paired", {}).items():
            fam.add(f"敏感性（{name}）", "敏感性运行与主方法、L1、L2 的配对", key, test)
    return fam


def n12_leak(summary: dict[str, Any]) -> Families:
    fam = Families()
    for protocol, section in summary.items():
        for method, entry in section.get("methods", {}).items():
            for key, test in entry.get("paired", {}).items():
                fam.add(
                    f"{protocol}：泄漏（逐方法）",
                    "每个方法的三组对照（label 对 anon、raw 对 label、raw 对 anon）",
                    f"{method} {key}",
                    test,
                )
        for key, test in section.get("pooled", {}).items():
            fam.add(f"{protocol}：泄漏（合并）", "合并各检索方法的三组对照", key, test)
        for arm, tests in section.get("vs_mem0", {}).items():
            for key, test in tests.items():
                fam.add(
                    f"{protocol}：对 Mem0（{arm}）",
                    "同一臂里朴素检索与 CueMem 各方法对 Mem0",
                    key,
                    test,
                )
    return fam


def n11_mem0(result: dict[str, Any]) -> Families:
    fam = Families()
    for run, section in result.items():
        if isinstance(section, dict) and "paired" in section:
            for key, test in section["paired"].items():
                fam.add(f"{run}：对 Mem0", "CueMem 各方法与朴素检索对 Mem0", key, test)
    return fam


def n11_second_judge(result: dict[str, Any]) -> Families:
    fam = Families()
    for dataset, section in result.items():
        if not isinstance(section, dict) or "comparisons" not in section:
            continue
        for comparison in section["comparisons"]:
            for judge, label in (("primary", "主判分"), ("second", "第二判分")):
                fam.add(
                    f"{dataset}：关键对比（{label}）",
                    "报告里的关键对比在这个判分模型下的胜负",
                    f"{comparison['a']} vs {comparison['b']}",
                    comparison[judge],
                )
    return fam


def n11_controller_v2(result: dict[str, Any]) -> Families:
    fam = Families()
    for stage, label in (("dev", "开发集"), ("test_posthoc", "测试集事后分析")):
        section = result.get(stage, {})
        for key, test in section.get("tests", {}).items():
            fam.add(f"{label}：总体", "v2 控制器对 v1 与固定档位", key, test)
        for dataset, entry in section.get("by_dataset", {}).items():
            for key, test in entry.get("tests", {}).items():
                fam.add(f"{label}：{dataset}", "v2 控制器对 v1 与固定档位", key, test)
    for reader, datasets in result.get("readers", {}).items():
        for dataset, entry in datasets.items():
            if not isinstance(entry, dict) or "tests" not in entry:
                continue
            for key, test in entry["tests"].items():
                fam.add(
                    f"换回答模型（{reader}）：{dataset}",
                    "换回答模型后控制器与校准的对比",
                    key,
                    test,
                )
            for row in entry.get("lambda_sweep", []):
                fam.add(
                    f"换回答模型（{reader}）：{dataset} λ 扫描",
                    "各个 λ 下控制器对 L2",
                    f"λ={row['lam']:g}",
                    row["vs_L2"],
                )
    return fam


def n11_see_gate(result: dict[str, Any]) -> Families:
    fam = Families()
    for dataset, entry in result.get("test_post_hoc", {}).items():
        for key, test in entry.items():
            if isinstance(test, dict) and "wins" in test:
                fam.add("测试集事后分析", "各数据集上 v3 对 v2", f"{dataset} {key}", test)
    return fam


def n7_controller(results: dict[str, Any]) -> Families:
    fam = Families()
    for scope, table in results.get("memory", {}).items():
        for a, row in table.get("tests", {}).items():
            for b, test in row.items():
                fam.add(
                    f"{scope}：控制器的配对",
                    "B6、B6h 对固定档位、规则版、零样本路由与 B3",
                    f"{a} vs {b}",
                    test,
                    pair=(a, b),
                )
    return fam


def n7_h3(results: dict[str, Any]) -> Families:
    fam = Families()
    for dataset, changes in results.get("changes", {}).items():
        for name, test in changes.items():
            if name.endswith("_vs_B3_qwen"):
                fam.add(
                    f"{dataset}：换成通义之后对 B3", "回答模型换成通义之后各策略对 B3", name, test
                )
            else:
                fam.add(
                    f"{dataset}：换回答模型前后",
                    "同一策略、同一道题，通义对 DeepSeek",
                    name,
                    test,
                )
    return fam


# 已发表的结果（相对 docs/results 的通配）→ 它的族
DOCUMENTS: tuple[tuple[str, Callable[[dict[str, Any]], Families]], ...] = (
    ("n9-final/results.json", n9),
    ("n12-baselines/*-*.json", n12_baselines),
    ("n12-leak/summary.json", n12_leak),
    ("n11-improvements/mem0.json", n11_mem0),
    ("n11-improvements/second_judge.json", n11_second_judge),
    ("n11-improvements/controller_v2.json", n11_controller_v2),
    ("n11-improvements/see_gate.json", n11_see_gate),
    ("n7-controller/results.json", n7_controller),
    ("n7-controller/h3.json", n7_h3),
)


def published(root: Path = RESULTS) -> list[tuple[Path, Callable[[dict[str, Any]], Families]]]:
    return [
        (path, families)
        for pattern, families in DOCUMENTS
        for path in sorted(root.glob(pattern))
        if not path.name.endswith("-holm.json")
    ]


def main() -> None:
    for path, families in published():
        print(f"\n== {path.relative_to(ROOT)}")
        update(path, families)


if __name__ == "__main__":
    main()
