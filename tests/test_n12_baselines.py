"""N12 外部基线的可复现性（第四轮体检）：对比脚本把文档引用的检索召回与写入规模写进 JSON（以前只在
文档里，没有出处）；装外部系统的命令固定写入时的版本。全部离线，运行目录与写入记录都是手写的。"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from fakes import PROJECT_ROOT

QWEN = "qwen3.8-flash|qwen3.7-text-embedding:1024"
DEEPSEEK = "deepseek-flash|BAAI/bge-small-en-v1.5:384"
TARGET = "WARNING: Target entity not found in nodes for edge relation: %s"


def _script() -> Any:
    path = PROJECT_ROOT / "scripts" / "n12_baselines.py"
    spec = importlib.util.spec_from_file_location("n12_baselines", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["n12_baselines"] = module
    spec.loader.exec_module(module)
    return module


def _recall(session: float, turn: float) -> dict[str, float]:
    return {
        "session_recall_any": session,
        "session_recall_all": session / 2,
        "turn_recall_any": turn,
        "turn_recall_all": turn / 2,
        "turn_ndcg": turn,
    }


def _row(
    method: str,
    sample_id: str,
    space: str,
    retrieval: dict[str, float] | None,
    turns: list[str],
    **details: Any,
) -> dict[str, Any]:
    """一道题的结果；retrieved 的每一项是（会话 ID，发言 ID 列表，分数）。"""
    return {
        "sample_id": sample_id,
        "dataset": "locomo",
        "space_key": space,
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
        "retrieval": retrieval,
        "retrieved": [["session_1", turns, 0.5]] if retrieval else [],
        "retrieved_facts": [],
        "details": details,
        "usage": {"input_tokens": 100, "output_tokens": 10, "llm_calls": 1, "cost": 0.001},
        "judge_usage": {"cost": 0.0001},
    }


def _write_run(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """一次 pq 协议的运行：run.json 记下抽取模型与向量模型（写入记录按它核对协议）。"""
    run_dir.mkdir(parents=True)
    info = {
        "config": {"name": run_dir.name, "dataset": "locomo", "split": "test200r"},
        "roles": {"extractor": {"model": "qwen3.8-flash"}},
        "embedding": {"model": "qwen3.7-text-embedding", "dim": 1024},
    }
    (run_dir / "run.json").write_text(json.dumps(info), "utf-8")
    lines = "".join(json.dumps(row) + "\n" for row in rows)
    (run_dir / "results.jsonl").write_text(lines, "utf-8")


def _build(spaces: Path, system: str, key: str, **stats: Any) -> None:
    path = spaces / system / f"locomo_{key}" / "build.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(stats), "utf-8")


def test_n12_records_retrieval_recall_and_build_size(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """对比 JSON 的每个方法另有检索召回（retrieval），外部系统另有写入的规模（build）：文档引用的
    发言召回、Graphiti 的实体与事实、LangMem 丢掉的会话等以前不在任何 JSON 里。召回不指向原文发言的
    方法（ML、M0）发言召回是 null，不是 0；没有检索的 B1 没有这个键。写入规模与写入成本取同一批
    写入记录：运行自己的优先、协议对不上的不算。原有的键与值不变。"""
    runs = tmp_path / "runs" / "eval"
    cuemem = [_row("B1", q, s, None, []) for q, s in (("q1", "conv-1"), ("q2", "conv-2"))]
    cuemem += [
        _row("B4-nobm25", "q1", "conv-1", _recall(1.0, 1.0), ["D1:3"]),
        _row("B4-nobm25", "q2", "conv-2", _recall(1.0, 0.0), ["D2:1"]),
    ]
    _write_run(runs / "pq-locomo-cuemem", cuemem)
    # LangMem 的记忆不带发言 ID：发言召回按构造是 0
    _write_run(
        runs / "pq-locomo-langmem",
        [
            _row("ML", "q1", "conv-1", _recall(1.0, 0.0), []),
            _row("ML", "q2", "conv-2", _recall(0.0, 0.0), []),
        ],
    )
    # conv-1 的写入由这次运行触发：它记下的统计（details）优先于写入目录里的 build.json
    own = {"protocol": QWEN, "entities": 10, "facts": 30, "failed_episodes": ["e1"]}
    own["warnings"] = {TARGET: 3}
    _write_run(
        runs / "pq-locomo-graphiti",
        [
            _row("MG", "q1", "conv-1", _recall(1.0, 1.0), ["D1:3"], graphiti_build=own),
            _row("MG", "q2", "conv-2", _recall(1.0, 1.0), ["D2:1"]),
        ],
    )
    spaces = tmp_path / "data" / "eval-spaces-qwen"
    _build(spaces, "graphiti", "conv-1", **{**own, "entities": 999, "warnings": {TARGET: 70}})
    warnings = {TARGET: 2, "WARNING: Source entity not found": 5}
    _build(spaces, "graphiti", "conv-2", protocol=QWEN, entities=5, facts=12, warnings=warnings)
    _build(
        spaces,
        "langmem",
        "conv-1",
        protocol=QWEN,
        sessions=3,
        empty_sessions=["session_2"],
        memories_over_embed_limit=1,
        longest_memory_chars=900,
    )
    _build(spaces, "langmem", "conv-2", protocol=DEEPSEEK, sessions=9, longest_memory_chars=99999)

    module = _script()
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "OUT", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["n12_baselines.py", "--protocol", "pq"])
    module.main()

    out = json.loads((tmp_path / "out" / "pq-locomo.json").read_text("utf-8"))
    methods = out["methods"]
    assert "retrieval" not in methods["B1"] and "build" not in methods["B4-nobm25"]
    assert methods["B4-nobm25"]["retrieval"] == {
        "session_recall_any": 1.0,
        "session_recall_all": 0.5,
        "turn_recall_any": 0.5,
        "turn_recall_all": 0.25,
    }
    assert methods["ML"]["retrieval"] == {
        "session_recall_any": 0.5,
        "session_recall_all": 0.25,
        "turn_recall_any": None,
        "turn_recall_all": None,
    }
    assert methods["MG"]["retrieval"]["turn_recall_any"] == 1.0
    assert methods["MG"]["build"] == {
        "entities": 15,
        "facts": 42,
        "failed_episodes": 1,
        "target_entity_not_found_per_space": [2, 3],
    }
    assert methods["ML"]["build"] == {
        "sessions": 3,
        "empty_sessions": 1,
        "memories_over_embed_limit": 1,
        "longest_memory_chars": 900,
    }
    assert methods["ML"]["write"]["mismatched"] == 1  # 另一个协议的写入两边都不算
    # 原有的键与顺序不变：新键只是插进来（去掉之后与以前的输出相同）
    assert list(methods["ML"]) == [
        "n",
        "accuracy",
        "ci95",
        "answer_cost_per_1k",
        "answer_tokens_mean",
        "by_type",
        "retrieval",
        "write",
        "build",
    ]


# 外部系统另装的包按 N12 期间构建机 uv 缓存里的版本固定（docs/results/n12-baselines/environments/）
PINS = {"spacy": "3.8.16", "fastembed": "0.9.0", "chromadb": "1.5.9"}
MEM0_COMMANDS = (
    ".github/workflows/baselines.yml",
    "scripts/reproduce.py",
    "scripts/n11_mem0.py",
    "scripts/n12_baselines.py",
    "src/cuemem/evals/mem0_baseline.py",
    "tests/test_mem0_baseline.py",
)
AMEM_COMMANDS = (
    ".github/workflows/baselines.yml",
    "src/cuemem/evals/baselines/amem.py",
    "tests/test_baseline_amem.py",
)


def test_external_install_commands_pin_the_build_time_versions() -> None:
    """装外部系统的命令固定写入时的版本：Mem0 带的 spaCy 与 fastembed 以前不固定，每次重新解析
    （已经解析到 fastembed 0.9.1）；A-MEM 只要求 chromadb>=0.4.22，以前评测命令里没有固定，只有
    测试与 CI 固定了 1.5.9。列表写法（scripts/reproduce.py）与命令行写法都查。"""
    for rel in sorted({*MEM0_COMMANDS, *AMEM_COMMANDS}):
        text = (PROJECT_ROOT / rel).read_text("utf-8")
        loose = re.findall(r'--with[",\s]+(spacy|fastembed|chromadb)(?![=\w-])', text)
        assert not loose, (rel, loose)
        for name, version in re.findall(r'--with[",\s]+(spacy|fastembed|chromadb)==([\d.]+)', text):
            assert version == PINS[name], (rel, name, version)
        amem = re.findall(r'--with "agentic-memory @ [^"]+"', text)
        pinned = re.findall(r'--with "agentic-memory @ [^"]+"[\s\\\'"]*--with chromadb==', text)
        assert len(pinned) == len(amem), rel
        assert amem or rel not in AMEM_COMMANDS, rel
        for name in ("spacy", "fastembed"):
            assert f"{name}=={PINS[name]}" in text or rel not in MEM0_COMMANDS, (rel, name)
