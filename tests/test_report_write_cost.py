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


def test_write_records_are_ranked_by_protocol(tmp_path: Path) -> None:
    """同一份对话历史两处写入记录（运行自己的 <系统>_build、写入目录下的 build.json）：协议对得上
    的优先（运行自己的在前），再取没记协议的；旧的 pds 写入（没有 protocol，只有 extractor 与
    embedder）也按这两项核对，不会当成没记协议的混进 pq 的成本。"""
    pds = tmp_path / "data" / "eval-spaces"
    # pds 的旧写入：Graphiti 的 embedder 带维度，LangMem 的不带
    build(
        pds,
        "graphiti",
        "conv-1",
        extractor="deepseek-flash",
        embedder="BAAI/bge-small-en-v1.5:384",
        cost=1.4177,
        chat_calls=2014,
    )
    build(
        pds,
        "langmem",
        "conv-1",
        extractor="deepseek-flash",
        embedder="BAAI/bge-small-en-v1.5",
        cost=0.0816,
        chat_calls=33,
    )
    build(pds, "mem0", "conv-1", cost=0.2, llm_calls=58)  # Mem0 从来不记协议
    own = {"protocol": QWEN, "cost": 0.6325, "chat_calls": 1688}
    rows = [result("MG", "q1", "conv-1", graphiti_build=own)]
    one = {"conv-1"}

    # pq 的运行用了 pds 的写入目录（旧运行没记 eval_spaces、当前配置是 pds 的）：取运行自己的记录
    mg = write_cost("MG", one, dataset="locomo", eval_spaces=pds, protocol=QWEN, rows=rows)
    assert mg["cost"] == 0.6325 and mg["llm_calls"] == 1688 and "mismatched" not in mg
    # 没有运行自己的记录时，旧的 pds 写入按 extractor / embedder 判为别的协议，不算
    for method in ("MG", "ML"):
        other = write_cost(method, one, dataset="locomo", eval_spaces=pds, protocol=QWEN)
        assert other == {"mismatched": 1}
    # pds 的运行照常认它们（LangMem 的 embedder 没带维度：只比模型名）
    assert write_cost("MG", one, dataset="locomo", eval_spaces=pds, protocol=DEEPSEEK)["cost"]
    ml = write_cost("ML", one, dataset="locomo", eval_spaces=pds, protocol=DEEPSEEK)
    assert ml["cost"] == 0.0816 and ml["spaces"] == 1

    # 都没记协议（Mem0）：运行自己的记录优先于写入目录里的
    rows = [result("M0", "q1", "conv-1", mem0_build={"cost": 0.1, "llm_calls": 58})]
    m0 = write_cost("M0", one, dataset="locomo", eval_spaces=pds, protocol=QWEN, rows=rows)
    assert m0["cost"] == 0.1
    assert write_cost("M0", one, dataset="locomo", eval_spaces=pds, protocol=QWEN)["cost"] == 0.2

    # 协议对得上的 build.json 优先于运行自己的、没记协议的记录；对不上的运行自己的记录不取
    qwen = tmp_path / "data" / "eval-spaces-qwen"
    build(qwen, "graphiti", "conv-1", protocol=QWEN, cost=0.6)
    unstamped = [result("MG", "q1", "conv-1", graphiti_build={"cost": 9.0})]
    mg = write_cost("MG", one, dataset="locomo", eval_spaces=qwen, protocol=QWEN, rows=unstamped)
    assert mg["cost"] == 0.6
    stale = [result("MG", "q1", "conv-1", graphiti_build={"protocol": DEEPSEEK, "cost": 9.0})]
    mg = write_cost("MG", one, dataset="locomo", eval_spaces=qwen, protocol=QWEN, rows=stale)
    assert mg["cost"] == 0.6


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
    # conv-2 有写入记录，只是属于别的协议：只算 mismatched，不再同时算作找不到
    assert "- MC：1 份对话历史的写入记录属于别的协议" in report
    assert "找不到写入记录" not in report
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


@pytest.mark.filterwarnings("ignore:No artists with labels")
def test_report_of_old_run_infers_build_directory_from_run_name(
    clean_env: None, tmp_path: Path, spaces: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N12 之前的运行没在 run.json 里记 eval_spaces：报告按运行名的协议前缀找写入目录
    （pq-… → data/eval-spaces-qwen），不跟着当前配置（这里给的是 pds 的目录）读到 pds 的同名写入；
    推断不了时用当前配置的，并在报告里注明。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path))
    pds = tmp_path / "data" / "eval-spaces"
    build(pds, "mem0", "conv-1", cost=5.0, llm_calls=58)
    build(pds, "mem0", "conv-2", cost=5.0, llm_calls=58)
    rows = [result("M0", "q1", "conv-1"), result("M0", "q2", "conv-2")]

    old_pq = tmp_path / "runs" / "pq-locomo-mem0"
    write_run(old_pq, rows)
    report = build_report(old_pq, eval_spaces=pds).read_text("utf-8")
    assert "| M0 | 它自己的写入（build.json） | 2 / 2 | $0.3000 |" in report
    assert f"- 外部基线的写入记录取自 `{spaces}`：运行记录里没有记写入目录" in report

    unknown = tmp_path / "runs" / "smoke-mem0"
    write_run(unknown, rows)
    report = build_report(unknown, eval_spaces=pds).read_text("utf-8")
    assert "| M0 | 它自己的写入（build.json） | 2 / 2 | $10.0000 |" in report
    assert f"- 外部基线的写入记录取自 `{pds}`：运行记录里没有记写入目录，用的是当前配置" in report

    recorded = tmp_path / "runs" / "pq-locomo-mem0-new"
    write_run(recorded, rows, eval_spaces=str(spaces))
    report = build_report(recorded, eval_spaces=pds).read_text("utf-8")
    assert "| M0 | 它自己的写入（build.json） | 2 / 2 | $0.3000 |" in report
    assert "- 外部基线的写入记录取自" not in report


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


# ---------- 会话 ID 的写法（N12 修复的泄漏，见 evals/datasets.py 与 test_eval_leak.py） ----------

LME = "longmemeval_s"


def lme_build(spaces: Path, system: str, folder: str, **stats: Any) -> None:
    """写入目录下的 build.json；folder 是 <写入用的数据集名>_<题号>（改写过的带 -anon）。"""
    path = spaces / system / folder / "build.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stats), "utf-8")


@pytest.fixture
def lme_spaces(tmp_path: Path) -> Path:
    """同一个写入目录里两种写法的写入并存：修复之前的（原始 ID，在原来的位置、没记写法）与
    改写过会话 ID 的（<数据集>-anon_<题号>，记着 session_ids）。"""
    root = tmp_path / "data" / "eval-spaces-qwen"
    lme_build(root, "mem0", f"{LME}_q1", cost=1.0, llm_calls=50)
    lme_build(root, "mem0", f"{LME}_q2", cost=2.0, llm_calls=50)
    lme_build(root, "mem0", f"{LME}-anon_q1", cost=0.5, llm_calls=40, session_ids="anon")
    # 放错了地方的写入：在 -anon 下却没记写法（= 原始 ID），不能算到改写后的运行上
    lme_build(root, "mem0", f"{LME}-anon_q2", cost=0.25, llm_calls=40)
    lme_build(root, "graphiti", f"{LME}_q1", protocol=QWEN, cost=0.9, seconds=90.0)
    lme_build(
        root, "graphiti", f"{LME}-anon_q1", protocol=QWEN, cost=0.6, seconds=60, session_ids="anon"
    )
    return root


def test_external_write_cost_follows_the_runs_session_ids(lme_spaces: Path) -> None:
    """改写过会话 ID 的运行只算 -anon 下、记着 anon 的写入；原始 ID 的运行（含修复之前、没记写法的
    运行：session_ids=None）只算原来位置上的写入；写法对不上的计入 mismatched。"""
    keys = {"q1", "q2"}
    anon = write_cost("M0", keys, dataset=LME, eval_spaces=lme_spaces, session_ids="anon")
    assert (anon["spaces"], anon["cost"], anon["mismatched"]) == (1, 0.5, 1)
    raw = write_cost("M0", keys, dataset=LME, eval_spaces=lme_spaces, session_ids="raw")
    assert (raw["spaces"], raw["cost"]) == (2, 3.0) and "mismatched" not in raw
    assert write_cost("M0", keys, dataset=LME, eval_spaces=lme_spaces) == raw  # 没记写法 = raw

    # 协议的核对照旧：两种写法各有一份协议对得上的 Graphiti 写入，各取各的
    one = {"q1"}
    kwargs: dict[str, Any] = {"dataset": LME, "eval_spaces": lme_spaces, "protocol": QWEN}
    mg = write_cost("MG", one, session_ids="anon", **kwargs)
    assert (mg["cost"], mg["seconds"]) == (0.6, 60)
    mg = write_cost("MG", one, session_ids="raw", **kwargs)
    assert (mg["cost"], mg["seconds"]) == (0.9, 90)
    other = write_cost("MG", one, dataset=LME, eval_spaces=lme_spaces, protocol=DEEPSEEK)
    assert other == {"mismatched": 1}


def test_run_details_are_ranked_after_the_session_id_check(lme_spaces: Path) -> None:
    """运行自己的写入记录（<系统>_build）照样先核对会话 ID 的写法，对得上的再按协议排（运行自己的
    优先）；写法对不上的不会因为排在前面就被取。"""
    one = {"q1"}
    own_anon = {"protocol": QWEN, "cost": 0.7, "session_ids": "anon"}
    rows = [{**result("MG", "q1", "q1", graphiti_build=own_anon), "dataset": LME}]
    kwargs: dict[str, Any] = {"dataset": LME, "eval_spaces": lme_spaces, "protocol": QWEN}
    assert write_cost("MG", one, rows=rows, session_ids="anon", **kwargs)["cost"] == 0.7
    # 原始 ID 的运行读到一条改写后的记录：剔除，取原来位置上协议对得上的 build.json
    raw = write_cost("MG", one, rows=rows, session_ids="raw", **kwargs)
    assert raw["cost"] == 0.9 and "mismatched" not in raw
    # 只有写法对不上的记录：计入 mismatched，不算成本
    alone = write_cost("MG", one, dataset=LME, protocol=QWEN, rows=rows, session_ids="raw")
    assert alone == {"mismatched": 1}


def test_locomo_write_cost_ignores_the_session_id_scheme(spaces: Path) -> None:
    """LoCoMo 的会话 ID 本来就是中性的（native）：写入的位置与核对都与修复之前相同，-anon 的目录
    不会被读到。"""
    (spaces / "mem0" / "locomo-anon_conv-1").mkdir()
    (spaces / "mem0" / "locomo-anon_conv-1" / "build.json").write_text('{"cost": 99}', "utf-8")
    keys = {"conv-1", "conv-2"}
    before = write_cost("M0", keys, dataset="locomo", eval_spaces=spaces, protocol=QWEN)
    native = write_cost(
        "M0", keys, dataset="locomo", eval_spaces=spaces, protocol=QWEN, session_ids="native"
    )
    assert native == before and native["cost"] == 0.3 and native["spaces"] == 2


def write_lme_run(run_dir: Path, rows: list[dict[str, Any]], spaces: Path, **extra: Any) -> None:
    """一次 LongMemEval 的运行；extra 里给 session_ids 时记进 run.json（修复之后的运行都记）。"""
    write_run(run_dir, [{**r, "dataset": LME} for r in rows], eval_spaces=str(spaces), **extra)
    info = json.loads((run_dir / "run.json").read_text("utf-8"))
    info["config"].update(dataset=LME, split="test100r")
    (run_dir / "run.json").write_text(json.dumps(info), "utf-8")


@pytest.mark.filterwarnings("ignore:No artists with labels")
@pytest.mark.parametrize("env", ["", "1", "label"])
def test_report_reads_the_build_records_of_the_runs_session_ids(
    tmp_path: Path, lme_spaces: Path, monkeypatch: pytest.MonkeyPatch, env: str
) -> None:
    """report.md 按运行记下的写法找外部基线的写入：改写过的运行读 -anon 下的，没记写法的旧运行
    （修复之前，原始 ID）读原来位置上的；当前的 CUEMEM_EVAL_RAW_SESSION_IDS 不起作用。"""
    monkeypatch.setenv("CUEMEM_EVAL_RAW_SESSION_IDS", env)
    rows = [result("M0", "q1", "q1"), result("M0", "q2", "q2")]

    anon = tmp_path / "runs" / "pq-lme-mem0"
    write_lme_run(anon, rows, lme_spaces, session_ids="anon")
    report = build_report(anon, eval_spaces=tmp_path / "elsewhere").read_text("utf-8")
    assert "| M0 | 它自己的写入（build.json） | 1 / 2 | $0.5000 | 40 |" in report
    assert "- M0：1 份对话历史的写入记录属于别的协议或会话 ID 写法" in report
    assert "找不到写入记录" not in report  # q2 有原始 ID 的写入，写法对不上，不是找不到
    assert "- 这次运行的会话 ID 写法是 anon（run.json 记的）" in report
    assert "| 写入 · M0（mem0） | $0.5000 |" in report

    legacy = tmp_path / "runs" / "pq-lme-raw-mem0"
    write_lme_run(legacy, rows, lme_spaces)  # 修复之前的运行：run.json 没记写法
    report = build_report(legacy, eval_spaces=tmp_path / "elsewhere").read_text("utf-8")
    assert "| M0 | 它自己的写入（build.json） | 2 / 2 | $3.0000 | 100 |" in report
    assert "属于别的协议" not in report
    assert "- 这次运行的会话 ID 写法是 raw（run.json 没有记：修复泄漏之前的运行）" in report


@pytest.mark.filterwarnings("ignore:No artists with labels")
def test_report_of_locomo_run_has_no_session_id_note(tmp_path: Path, spaces: Path) -> None:
    run_dir = tmp_path / "runs" / "pq-locomo-mem0"
    write_run(run_dir, [result("M0", "q1", "conv-1")], eval_spaces=str(spaces))
    report = build_report(run_dir).read_text("utf-8")
    assert "| M0 | 它自己的写入（build.json） | 1 / 1 | $0.1000 |" in report
    assert "会话 ID 写法" not in report


@pytest.mark.filterwarnings("ignore:No artists with labels")
@pytest.mark.parametrize("scheme", ["anon", "raw"])
def test_n12_reads_the_build_records_of_the_runs_session_ids(
    clean_env: None,
    tmp_path: Path,
    lme_spaces: Path,
    monkeypatch: pytest.MonkeyPatch,
    scheme: str,
) -> None:
    """N12 对比脚本把运行记下的写法交给 write_cost（不看环境变量），写入目录用运行记录里的数据集
    全名（运行名里只是简写 lme）；找不到写入记录的历史单独计数，与 report.md 的说明同一口径。"""
    runs = tmp_path / "runs" / "eval"
    rows = [result("M0", f"q{i}", f"q{i}") for i in (1, 2, 3)]
    extra = {"session_ids": scheme} if scheme == "anon" else {}  # raw：修复之前的运行没记
    write_lme_run(runs / "pq-lme-mem0", rows, lme_spaces, **extra)

    module = load_script()
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path))
    monkeypatch.setenv("CUEMEM_EVAL_RAW_SESSION_IDS", "1" if scheme == "anon" else "")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["n12_baselines.py", "--protocol", "pq", "--dataset", "lme"])
    module.main()

    out = json.loads((tmp_path / "out" / "pq-lme.json").read_text("utf-8"))
    assert out["session_ids"] == scheme
    write = out["methods"]["M0"]["write"]
    if scheme == "anon":
        counts = (write["spaces"], write["cost"], write["mismatched"], write["missing"])
        assert counts == (1, 0.5, 1, 1)
    else:
        assert (write["spaces"], write["cost"], write["missing"]) == (2, 3.0, 1)
        assert "mismatched" not in write

    # 同一次运行的 report.md 与 N12 对比脚本数得一样：找到记录但写法对不上的只算 mismatched，
    # 不再同时算作找不到（以前 report.md 把 q2 两边都算了一次）
    report = build_report(runs / "pq-lme-mem0").read_text("utf-8")
    assert f"| M0 | 它自己的写入（build.json） | {write['spaces']:.0f} / 3 |" in report
    assert f"- M0：{write['missing']:.0f} 份对话历史找不到写入记录" in report
    if scheme == "anon":
        assert f"- M0：{write['mismatched']:.0f} 份对话历史的写入记录属于别的协议" in report
    else:
        assert "属于别的协议" not in report
