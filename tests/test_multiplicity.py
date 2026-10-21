"""多重比较校正（Holm–Bonferroni）：routing.holm 与发表结果旁边的 *-holm.json。"""

from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any

import pytest

from cuemem.evals.routing import holm, holm_family, sign_test

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"


@pytest.fixture
def multiplicity(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("multiplicity")


def test_holm_adjusted_p_values() -> None:
    """从小到大第 i 个乘以 m − i、再取前面的最大值（单调），不超过 1；键的顺序不变。"""
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert list(adjusted) == ["a", "b", "c", "d"]
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02})
    assert holm({"x": 0.6, "y": 0.7}) == {"x": 1.0, "y": 1.0}
    assert holm({"only": 0.03}) == {"only": 0.03}
    assert holm({}) == {}


def test_holm_agrees_with_the_step_down_procedure() -> None:
    """校正后的 p < α 恰好等于 Holm 的逐步检验：按 p 从小到大，第 i 个（从 1 数）与 α / (m − i + 1)
    比，一直拒绝到第一个比不过的为止。"""
    rng = random.Random(0)
    for _ in range(300):
        m = rng.randint(1, 12)
        pvalues = {f"t{i}": rng.choice([rng.random(), rng.random() / 50]) for i in range(m)}
        adjusted = holm(pvalues)
        for alpha in (0.01, 0.05, 0.1):
            rejected: set[str] = set()
            for i, key in enumerate(sorted(pvalues, key=lambda k: pvalues[k])):
                if pvalues[key] > alpha / (m - i):
                    break
                rejected.add(key)
            assert {k for k, p in adjusted.items() if p <= alpha} == rejected


def test_holm_family_recomputes_p_from_wins_and_losses() -> None:
    """发表的 JSON 里 p 多半四舍五入到 4 位（极小的写成 0）：按胜负重新算，比较用未舍入的值。"""
    tests = {
        "L2 vs Mem0": {"wins": 17, "losses": 5, "p": 0.0169},
        "L2 vs Graphiti": {"wins": 62, "losses": 6, "p": 0.0},
        "L1 vs Mem0": {"wins": 14, "losses": 11, "p": 0.69},
    }
    rows = holm_family(tests)
    assert rows["L2 vs Graphiti"]["p"] == round(sign_test(62, 6), 4) == 0.0
    assert rows["L2 vs Mem0"] == {
        "wins": 17,
        "losses": 5,
        "p": 0.0169,
        "p_holm": round(2 * sign_test(17, 5), 4),
        "nominal": True,
        "holm": True,
    }
    assert rows["L1 vs Mem0"]["p_holm"] == 0.69 and not rows["L1 vs Mem0"]["nominal"]


def test_families_skip_mirrors_and_undecided_tests(multiplicity: Any) -> None:
    """同一族里甲对乙、乙对甲只算一次；+0 / −0 不算一次检验；不止一族时另有"全文"。"""
    out = {
        "paired": {
            "B4-nobm25_vs_B5": {"wins": 6, "losses": 15, "p": 0.0784},
            "B5_vs_B4-nobm25": {"wins": 15, "losses": 6, "p": 0.0784},
            "B5_vs_B1": {"wins": 0, "losses": 0, "p": 1.0},
            "B5_vs_M0": {"wins": 17, "losses": 5, "p": 0.0169},
            "B4-nobm25_vs_M0": {"wins": 14, "losses": 11, "p": 0.69},
        }
    }
    report = multiplicity.report("x.json", multiplicity.n12_baselines(out))
    internal = report["families"]["CueMem 与参照"]
    assert list(internal["tests"]) == ["B4-nobm25_vs_B5"] and internal["m"] == 1
    external = report["families"]["L1、L2 对外部系统"]
    assert external["m"] == 2 and external["tests"]["B5_vs_M0"]["holm"]
    assert report["families"]["全文"]["m"] == 3  # 0.0169 × 3 > 0.05：最严的口径下不再显著
    assert report["lost"] == ["全文 | L1、L2 对外部系统 | B5_vs_M0"]


def test_published_holm_files_are_up_to_date(multiplicity: Any) -> None:
    """docs/results 里每份带配对检验的结果旁边都有 Holm 校正的 <文件名>-holm.json，内容与按结果
    重算的一致（结果本身不动）。新的结果里有检验却没有登记族时，这里会报出来。"""
    covered = {path for path, _ in multiplicity.published()}
    for path in sorted(RESULTS.rglob("*.json")):
        if path.name.endswith("-holm.json"):
            continue
        text = path.read_text(encoding="utf-8")
        has_tests = '"wins"' in text and '"losses"' in text
        assert has_tests == (path in covered), path
    for path, families in multiplicity.published():
        expected = multiplicity.report(path.name, families(json.loads(path.read_text("utf-8"))))
        committed = json.loads(multiplicity.holm_path(path).read_text(encoding="utf-8"))
        assert committed == expected, path


def test_holm_changes_the_borderline_claims(multiplicity: Any) -> None:
    """审计复现的几条：N12 pds 下 L2 对 Mem0（p = 0.0169）在"L1、L2 对外部系统"一族里不再显著；
    N9 LoCoMo 上 L2 对 L1 的开放域（+8 / −1）与时间题（+15 / −5）在分题型的一族里不再显著，
    L2 对全上下文（+89 / −60）在总体对比的一族里不再显著；L2 对 L1 的总体仍然显著。"""

    def load(name: str) -> dict[str, Any]:
        return json.loads((RESULTS / name).read_text(encoding="utf-8"))

    pds = load("n12-baselines/pds-locomo-holm.json")["families"]["L1、L2 对外部系统"]["tests"]
    assert pds["B5_vs_M0"]["nominal"] and not pds["B5_vs_M0"]["holm"]
    assert pds["B5_vs_MC"]["holm"] and pds["B5_vs_ML"]["holm"]
    n9 = load("n9-final/results-holm.json")["families"]
    by_type = n9["locomo：B5 vs B4-nobm25（分题型）"]["tests"]
    assert {k for k, r in by_type.items() if r["nominal"] and not r["holm"]} == {
        "open-domain",
        "temporal",
    }
    overall = n9["locomo：方法对比（总体）"]["tests"]
    assert not overall["B5 vs B1"]["holm"] and overall["B5 vs B4-nobm25"]["holm"]


def test_n12_leak_writes_its_holm_file(
    multiplicity: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """生成结果的脚本写完自己的 JSON 之后，旁边的 Holm 校正一并更新（这里用 n12_leak）。"""
    leak = importlib.import_module("n12_leak")
    runs = tmp_path / "runs"
    rows = {
        "orig-lme-cuemem": [("B4-nobm25", f"q{i}", i < 9) for i in range(12)],
        "orig-lme-mem0": [("M0", f"q{i}", i >= 9) for i in range(12)],
    }
    for run, results in rows.items():
        (runs / run).mkdir(parents=True)
        lines = [
            json.dumps({"method": m, "sample_id": s, "correct": c, "is_abstention": False})
            for m, s, c in results
        ]
        (runs / run / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(leak, "RUNS", runs)
    monkeypatch.setattr(leak, "ROOT", tmp_path)
    monkeypatch.setattr(leak, "OUT", tmp_path / "out" / "summary.json")
    leak.main()
    holm_file = json.loads((tmp_path / "out" / "summary-holm.json").read_text("utf-8"))
    test = holm_file["families"]["orig：对 Mem0（anon）"]["tests"]["B4-nobm25_vs_M0"]
    assert (test["wins"], test["losses"]) == (9, 3) and test["p"] == round(sign_test(9, 3), 4)
