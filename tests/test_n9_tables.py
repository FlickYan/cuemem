"""N9 发表的表与图：tables.md 由 scripts/n9_tables.py 从 results.json 与 results-holm.json 生成，
Pareto 图由 scripts/n9_report.py 画。都只读发表的 JSON，不需要评测运行。"""

from __future__ import annotations

import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
N9 = ROOT / "docs" / "results" / "n9-final"


@pytest.fixture
def n9_tables(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    module = importlib.import_module("n9_tables")
    monkeypatch.setattr(module, "OUT", N9)
    return module


@pytest.fixture
def n9_report(monkeypatch: pytest.MonkeyPatch) -> Any:
    pytest.importorskip("matplotlib")
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("n9_report")


def test_tables_md_is_up_to_date(n9_tables: Any) -> None:
    """提交的 tables.md 就是脚本按提交的 JSON 生成的：改了生成脚本或 JSON 却没重新生成时失败。"""
    assert n9_tables.render() == (N9 / "tables.md").read_text("utf-8")


def test_tables_md_states_cost_scope_holm_and_leak(n9_tables: Any) -> None:
    """表头写明成本只算回答（写入另算，D-070）、Holm 校正那一列的意思、LongMemEval 检索类方法
    是在会话 ID 泄漏修复之前测的（D-067）；每个名义上显著、校正后不再显著的检验，表里那一行都
    写着校正后的 p。以前表头只有"名义成本"，249 个 p 都是单个比较的，没有一个 Holm。"""
    text = n9_tables.render()
    header = text[: text.index("## 主表")]
    for phrase in ("只算回答", "写入", "D-070", "Holm", "results-holm.json", "D-067"):
        assert phrase in header, phrase
    assert "../n12-leak/README.md" in header

    holm = json.loads((N9 / "results-holm.json").read_text("utf-8"))
    lines = text.splitlines()
    checked = 0
    for entry in holm["lost"]:
        family, key = entry.split(" | ", 1)
        if family in ("全文", "三个记忆数据集合并：控制器"):
            continue
        test = holm["families"][family]["tests"][key]
        adjusted = f"| {n9_tables.p_value(test['p_holm'])} |"
        # 控制器那几族只留一个方向，表里反过来写的那一行胜负对调
        tallies = {
            f"| +{test['wins']} / −{test['losses']} |",
            f"| +{test['losses']} / −{test['wins']} |",
        }
        assert any(line.endswith(adjusted) and any(t in line for t in tallies) for line in lines), (
            entry
        )
        checked += 1
    assert checked == 21
    assert "| B6 (frozen) 对 always-L2 | +6 / −17 | p = 0.035 | p = 0.382 |" in lines
    assert (
        "| L1 快速召回 对 B1 全上下文 | 全部 | 1155 | 79.7% / 82.0% | +65 / −92 | p = 0.038 "
        "| p = 0.151 |" in lines
    )
    assert "| qwen3.6-35b-a3b | B6 vs B5 | +0 / −7 | p = 0.016 | p = 0.078 |" in lines


def test_holm_cell_finds_mirrored_pairs_and_skips_untested_rows(n9_tables: Any) -> None:
    """族里同一对方法只留一个方向（scripts/multiplicity.Families）：表里反过来写的那一行也要查到；
    +0 / −0 不算检验、不在族里，写"—"。"""
    families = {"f": {"tests": {"A vs B": {"p_holm": 0.0123}, "C vs D": {"p_holm": 0.0004}}}}
    assert n9_tables.holm_cell(families, "f", "A vs B") == "p = 0.012"
    assert n9_tables.holm_cell(families, "f", "B vs A", "A vs B") == "p = 0.012"
    assert n9_tables.holm_cell(families, "f", "C vs D") == "p < 0.001"
    assert n9_tables.holm_cell(families, "f", "E vs F", "F vs E") == "—"
    assert n9_tables.holm_cell(families, "missing", "A vs B") == "—"
    assert n9_tables.holm_cell({}, "f", "A vs B") == "—"


def test_pareto_figure_shows_full_context_and_answer_only_axis(
    n9_report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pareto 图：横轴写明是回答成本、不含一次性的写入；全上下文在 LongMemEval（$31.88）与
    MemLens 128K（$40.60）上画在右边缘、标着真实的数（以前这两个面板不画它，图例里却有），另外
    两个面板照常画在真实的位置；字体固定为 DejaVu Sans，不跟着本机的 matplotlibrc 变。"""
    import matplotlib

    results = json.loads((N9 / "results.json").read_text("utf-8"))
    monkeypatch.setitem(matplotlib.rcParams, "font.family", ["cursive"])  # 本机的设置
    figures: list[Any] = []
    close = n9_report.plt.close
    monkeypatch.setattr(n9_report.plt, "close", figures.append)  # 留着画好的图来检查
    n9_report.plot_pareto(results["controller"], lambda mode: tmp_path / f"pareto-{mode}.png")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pareto-dark.png", "pareto-light.png"]
    assert len(figures) == 2
    for fig in figures:
        _check_pareto(fig, results["controller"], list(n9_report.DATASET_EN.values()))
        close(fig)


def _check_pareto(fig: Any, sections: dict[str, Any], titles: list[str]) -> None:
    from matplotlib.collections import PathCollection
    from matplotlib.text import Text

    def plotted(ax: Any, x: float, y: float) -> bool:
        points = [
            (float(px), float(py)) for c in ax.collections if isinstance(c, PathCollection)
            for px, py in c.get_offsets()
        ]  # fmt: skip
        return any(math.isclose(px, x) and math.isclose(py, y) for px, py in points)

    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert [ax.get_title(loc="left") for ax in axes] == titles
    for ax, (dataset, sec) in zip(axes, sections.items(), strict=True):
        assert (
            ax.get_xlabel() == "nominal answer cost per 1k questions (log; one-off write excluded)"
        )
        b1 = sec["policies"]["B1"]
        right = ax.get_xlim()[1]
        if dataset in ("longmemeval_s", "memlens_128k"):
            assert b1["cost_per_1k"] > right
            assert plotted(ax, right, b1["accuracy"])  # 右边缘
            label = rf"B1 {b1['accuracy']:.1f}% · \${b1['cost_per_1k']:.2f} →"
            assert [t.get_text() for t in ax.texts] == [label]
        else:
            assert b1["cost_per_1k"] < right
            assert plotted(ax, b1["cost_per_1k"], b1["accuracy"])
            assert list(ax.texts) == []
    assert [t.get_text() for t in axes[0].texts] == ["B1 73.0% · \\$31.88 →"]
    assert [t.get_text() for t in axes[3].texts] == ["B1 46.2% · \\$40.60 →"]
    texts = [t for t in fig.findobj(Text) if t.get_text()]
    assert texts and {tuple(t.get_fontfamily()) for t in texts} == {("DejaVu Sans",)}
    assert "B1 full context" in [t.get_text() for t in fig.legends[0].get_texts()]


def test_figures_only_redraws_from_the_published_results(
    n9_report: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--figures-only 只按发表的 results.json 重画四张图，不需要评测运行，也不动 JSON 与表。"""
    out = tmp_path / "n9-final"
    out.mkdir()
    for name in ("results.json", "results-holm.json", "tables.md"):
        shutil.copy(N9 / name, out / name)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    monkeypatch.setattr(n9_report, "OUT", out)
    monkeypatch.setattr(n9_report, "RUNS_DIR", tmp_path / "eval")
    monkeypatch.setattr(n9_report, "CACHE", tmp_path / "eval" / "n9-final")
    monkeypatch.setattr(sys, "argv", ["n9_report.py", "--figures-only"])
    n9_report.main()
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [*before, "pareto-dark.png", "pareto-light.png", "readers-dark.png", "readers-light.png"]
    )
    assert {name: (out / name).read_bytes() for name in before} == before
    assert not (tmp_path / "eval").exists()
