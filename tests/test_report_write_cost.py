"""写入成本（N12）：report.md 与 scripts/n12_baselines.py 用同一个口径（write_cost）。

CueMem 的写入取运行里的 ingest.jsonl，外部基线取它们各自的 build.json；pq 协议的写入在另一个目录。
全部离线，运行目录与写入记录都是手写的。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from cuemem.evals.report import build_report, run_protocol, write_cost

QWEN = "qwen3.8-flash|qwen3.7-text-embedding:1024"
DEEPSEEK = "deepseek-flash|BAAI/bge-small-en-v1.5:384"
KEYS = {"conv-1", "conv-2"}


def result(method: str, sample_id: str, space_key: str, **details: Any) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "dataset": "locomo",
        "space_key": space_key,
        "question_type": "single-hop",
        "ability": "single-hop",
        "is_abstention": False,
        "method": method,
        "question": "Where does Ana live?",
        "answer": "Berlin",
        "response": "Berlin",
        "evidence_sufficient": True,
        "correct": True,
        "judge_raw": "yes",
        "f1": None,
        "retrieval": None,
        "retrieved": [],
        "retrieved_facts": [],
        "details": details,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "llm_calls": 1,
            "llm_cache_hits": 0,
            "cost": 0.001,
            "latency_ms": 100.0,
        },
        "judge_usage": {"cost": 0.0001},
    }


def ingest(space: str, stage: str, cost: float, **extra: Any) -> dict[str, Any]:
    row = {"space_key": space, "stage": stage, "cost": cost, "spent": cost / 2, "llm_calls": 0}
    row.update(extra)
    if stage == "consolidate":  # 巩固记录里报告要用到的统计
        row.update(
            turns=10,
            sessions=2,
            facts_extracted=5,
            added=5,
            superseded=0,
            merged=0,
            noop=0,
            input_tokens=1000,
            output_tokens=100,
            facts_current=5,
            facts_total=5,
            entities=2,
        )
    else:
        row.setdefault("embedding_tokens", 100)
    return row


INGESTS = [
    ingest("conv-1", "consolidate", 0.5, llm_calls=99, elapsed_ms=1000.0),  # 之后又重新巩固过
    ingest("conv-1", "ingest", 0.001, latency_ms=1000.0),
    ingest("conv-2", "ingest", 0.001, latency_ms=1000.0),
    ingest("conv-1", "consolidate", 0.03, llm_calls=50, elapsed_ms=60000.0),
    ingest("conv-2", "consolidate", 0.03, llm_calls=50, elapsed_ms=60000.0),
]


def build(spaces: Path, system: str, key: str, **stats: Any) -> None:
    path = spaces / system / f"locomo_{key}" / "build.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stats), "utf-8")


@pytest.fixture
def spaces(tmp_path: Path) -> Path:
    """pq 协议的写入目录：M0 两份（旧格式，没记协议），MC 一份协议对、一份是别的协议的。"""
    root = tmp_path / "data" / "eval-spaces-qwen"
    for key, cost in (("conv-1", 0.1), ("conv-2", 0.2)):
        build(
            root,
            "mem0",
            key,
            cost=cost,
            spent=cost / 2,
            llm_calls=58,
            latency_ms=500000.0,
            memories=400,
        )
    build(
        root,
        "cognee",
        "conv-1",
        protocol=f"{QWEN}|cognee==1.6.4|beam-turn-pair",
        cost=0.2,
        spent=0.1,
        chat_calls=600,
        seconds=600.0,
    )
    build(root, "cognee", "conv-2", protocol=DEEPSEEK, cost=9.0, chat_calls=1, seconds=1.0)
    return root


def test_cuemem_write_cost_by_method() -> None:
    """B0 / B1 / ORACLE 不写入；B2 只算向量化；其余算向量化 + 巩固（同一份历史只取最后一次）。"""
    for method in ("B0", "B1", "ORACLE"):
        assert write_cost(method, KEYS, dataset="locomo", ingests=INGESTS) == {}
    b2 = write_cost("B2", KEYS, dataset="locomo", ingests=INGESTS)
    assert b2 == {"cost": 0.002, "spent": 0.001, "llm_calls": 0, "seconds": 2.0, "spaces": 0}
    l1 = write_cost("B4-nobm25", KEYS, dataset="locomo", ingests=INGESTS)
    assert l1["cost"] == pytest.approx(0.062) and l1["llm_calls"] == 100
    assert l1["seconds"] == 122.0 and l1["spaces"] == 2
    assert write_cost("B5", KEYS, dataset="locomo", ingests=INGESTS) == l1  # 共用一份记忆
    only_one = write_cost("B5", {"conv-2"}, dataset="locomo", ingests=INGESTS)
    assert only_one["cost"] == pytest.approx(0.031)


def test_external_write_cost_reads_build_records(spaces: Path) -> None:
    m0 = write_cost("M0", KEYS, dataset="locomo", eval_spaces=spaces, protocol=QWEN)
    assert m0 == {
        "spaces": 2,
        "cost": 0.3,
        "spent": 0.15,
        "llm_calls": 116,
        "seconds": 1000.0,  # M0 没有墙钟耗时：取延迟之和
        "memories": 800,
    }
    # 别的协议的写入记录不算（例如 pq 的运行读到了 pds 的目录）
    mc = write_cost("MC", KEYS, dataset="locomo", eval_spaces=spaces, protocol=QWEN)
    assert mc["spaces"] == 1 and mc["cost"] == 0.2 and mc["mismatched"] == 1
    assert mc["llm_calls"] == 600 and mc["seconds"] == 600.0
    # 不核对协议时照算；找不到写入目录时取触发写入的那道题记下的统计
    assert write_cost("MC", KEYS, dataset="locomo", eval_spaces=spaces)["cost"] == 9.2
    rows = [result("MC", "q2", "conv-2", cognee_build={"protocol": QWEN, "cost": 0.25})]
    mc = write_cost("MC", KEYS, dataset="locomo", eval_spaces=spaces, protocol=QWEN, rows=rows)
    assert mc["spaces"] == 2 and mc["cost"] == 0.45 and "mismatched" not in mc
    alone = write_cost("MC", KEYS, dataset="locomo", protocol=QWEN, rows=rows)
    assert alone["spaces"] == 1 and alone["cost"] == 0.25
    assert write_cost("MG", KEYS, dataset="locomo", eval_spaces=spaces) == {}


def run_info(name: str, methods: list[str], **extra: Any) -> dict[str, Any]:
    model = {"model": "qwen3.8-flash", "model_id": "qwen3.8-flash"}
    return {
        "config": {
            "name": name,
            "dataset": "locomo",
            "split": "test200r",
            "methods": methods,
            "k": 10,
            "limit": None,
        },
        "samples": 2,
        "roles": {"reader": model, "extractor": model, "judge": model},
        "embedding": {"model": "qwen3.7-text-embedding", "dim": 1024, "backend": "dashscope"},
        "git": "abc1234",
        "created": "2026-10-15T00:00:00",
        "resumed": [],
        **extra,
    }


def write_run(
    run_dir: Path,
    rows: list[dict[str, Any]],
    ingests: list[dict[str, Any]] = (),  # type: ignore[assignment]
    **extra: Any,
) -> None:
    run_dir.mkdir(parents=True)
    info = run_info(run_dir.name, sorted({r["method"] for r in rows}), **extra)
    (run_dir / "run.json").write_text(json.dumps(info), "utf-8")
    lines = "".join(json.dumps(r) + "\n" for r in rows)
    (run_dir / "results.jsonl").write_text(lines, "utf-8")
    if ingests:
        lines = "".join(json.dumps(r) + "\n" for r in ingests)
        (run_dir / "ingest.jsonl").write_text(lines, "utf-8")


def test_run_protocol_matches_external_build_records() -> None:
    assert run_protocol(run_info("x", [])) == QWEN
    assert run_protocol({"config": {}}) is None


def test_report_counts_external_write_cost(tmp_path: Path, spaces: Path) -> None:
    """report.md 里外部基线的写入成本与 CueMem 的写入、巩固同一口径，计入合计；以前外部基线
    只有回答成本，看上去比 CueMem 便宜得多。"""
    methods = ("B1", "B2", "B4-nobm25", "M0", "MC")
    rows = [
        result(method, f"q{i}", key)
        for method in methods
        for i, key in enumerate(sorted(KEYS), start=1)
    ]
    run_dir = tmp_path / "runs" / "pq-locomo-all"
    # 运行记录里记了写入目录：调用方给的（当前配置，可能是 pds 的）不用
    write_run(run_dir, rows, INGESTS[1:], eval_spaces=str(spaces))
    report = build_report(run_dir, eval_spaces=tmp_path / "elsewhere").read_text("utf-8")

    assert "## 写入成本（一次性，按方法）" in report
    assert "| B1 | 不写入（直接读原文） | — | — | — | — | — | $0.00100 |" in report
    assert "| B2 | 向量化（建索引） | 2 / 2 | $0.0020 |" in report
    assert "| B4-nobm25 | 向量化 + 巩固（L1 / L2 / 控制器共用一份） | 2 / 2 | $0.0620 |" in report
    # M0：两份写入共 $0.3，200 题里这里只答了 2 题，分摊到每题 $0.15
    assert "| M0 | 它自己的写入（build.json） | 2 / 2 | $0.3000 | 116 | 1,000s " in report
    assert "| $0.15000 | $0.15100 |" in report
    assert "| MC | 它自己的写入（build.json） | 1 / 2 | $0.2000 |" in report
    assert "- MC：1 份对话历史找不到写入记录" in report
    assert "- MC：1 份对话历史的写入记录属于别的协议" in report
    assert "| 写入 · M0（mem0） | $0.3000 |" in report
    assert "| 写入 · MC（cognee） | $0.2000 |" in report
    # 合计：回答 10 × 0.001 + 向量化 0.002 + 巩固 0.06 + 外部 0.5 + 判分 10 × 0.0001
    assert "| **合计** | **$0.5730** | |" in report
    assert "| M0 外部基线：Mem0 2.2.1（开源版） |" in report


@pytest.mark.filterwarnings("ignore:No artists with labels")  # 散点图只画 CueMem 的主线
def test_report_without_build_records_uses_build_details(tmp_path: Path) -> None:
    """找不到写入目录时，取触发写入的那道题记下的统计（<系统>_build）。"""
    rows = [
        result("MC", "q1", "conv-1", cognee_build={"protocol": QWEN, "cost": 0.25}),
        result("MC", "q2", "conv-1"),
    ]
    run_dir = tmp_path / "pq-pilot-cognee"
    write_run(run_dir, rows)
    report = build_report(run_dir).read_text("utf-8")
    assert "| MC | 它自己的写入（build.json） | 1 / 1 | $0.2500 |" in report
    assert "| 写入 · MC（cognee） | $0.2500 |" in report
    assert "| 写入（向量化） |" not in report  # 只有外部基线的运行：没有 CueMem 的写入


def load_script() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "n12_baselines.py"
    spec = importlib.util.spec_from_file_location("n12_baselines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["n12_baselines"] = module
    spec.loader.exec_module(module)
    return module


def test_n12_pq_reads_its_own_build_records(
    clean_env: None, tmp_path: Path, spaces: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pq 的运行（pq-locomo-*）读 data/eval-spaces-qwen 的写入记录，哪怕当前配置没设
    CUEMEM_PATHS__EVAL_SPACES（以前会悄悄读到 pds 的同名写入）；写入成本与报告同一口径。"""
    pds = tmp_path / "data" / "eval-spaces"
    build(pds, "mem0", "conv-1", cost=5.0, llm_calls=58)
    build(pds, "mem0", "conv-2", cost=5.0, llm_calls=58)
    runs = tmp_path / "runs" / "eval"
    cuemem_rows = [
        result(m, f"q{i}", key)
        for m in ("B1", "B4-nobm25")
        for i, key in enumerate(sorted(KEYS), start=1)
    ]
    write_run(runs / "pq-locomo-cuemem", cuemem_rows, INGESTS)
    for method, system in (("M0", "mem0"), ("MC", "cognee")):
        rows = [result(method, "q1", "conv-1"), result(method, "q2", "conv-2")]
        write_run(runs / f"pq-locomo-{system}", rows)
    write_run(runs / "pq-pilot-mem0", [result("M0", "q9", "conv-9")])  # 别的前缀不算

    module = load_script()
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["n12_baselines.py", "--protocol", "pq"])
    module.main()

    out = json.loads((tmp_path / "out" / "pq-locomo.json").read_text("utf-8"))
    assert out["runs"] == ["pq-locomo-cognee", "pq-locomo-cuemem", "pq-locomo-mem0"]
    assert out["questions"] == 2
    write = {m: row["write"] for m, row in out["methods"].items()}
    assert write["M0"]["cost"] == 0.3 and write["M0"]["spaces"] == 2  # 不是 pds 的 10.0
    assert write["MC"]["cost"] == 0.2 and write["MC"]["mismatched"] == 1
    assert write["B1"] == {}
    assert write["B4-nobm25"] == write_cost("B4-nobm25", KEYS, dataset="locomo", ingests=INGESTS)
    assert out["labels"]["M0"] == "Mem0 2.2.1 (OSS)" and out["labels"]["MC"] == "Cognee 1.6.4"
    assert module.LABELS["MG"] == "Graphiti (graphiti-core 0.30.2)"
    assert module.LABELS["ML"] == "LangMem 0.0.30"
    assert module.LABELS["MA"] == "A-MEM (agiresearch/A-mem@ceffb86)"

    # 显式指定写入目录时照用（相对数据目录）
    monkeypatch.setattr(
        sys, "argv", ["n12_baselines.py", "--protocol", "pq", "--eval-spaces", "data/eval-spaces"]
    )
    module.main()
    out = json.loads((tmp_path / "out" / "pq-locomo.json").read_text("utf-8"))
    assert out["methods"]["M0"]["write"]["cost"] == 10.0
