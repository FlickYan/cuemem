"""评测运行器的端到端测试：迷你 LongMemEval + 模拟接口 + 离线哈希向量器。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.errors import EmbeddingAPIError
from cuemem.evals.datasets import load_longmemeval
from cuemem.evals.methods import MethodContext, MethodOutput
from cuemem.evals.report import build_report, load_rows
from cuemem.evals.runner import EvalRunner, RunConfig

FIXTURES = Path(__file__).parent / "fixtures"


def fake_model(body: dict[str, Any]) -> str:
    """同时扮演 reader 与 judge：
    - reader：上下文里出现 golden retriever 就答出来，否则说不知道；
    - judge：模型回答包含正确答案（或拒答题里说了不知道）就判 yes。
    """
    prompt = body["messages"][-1]["content"]
    if prompt.startswith("I will give you"):
        answer = re.search(
            r"(?:Correct Answer|Rubric|Explanation): (.*)\n\nModel Response", prompt, re.S
        )
        response = re.search(r"Model Response: (.*)\n\n", prompt, re.S)
        assert answer and response
        if "unanswerable" in prompt:
            return "yes" if "don't know" in response.group(1) else "no"
        return "yes" if answer.group(1).lower() in response.group(1).lower() else "no"
    found = "golden retriever" in prompt.split("Question:")[0]
    answer = "golden retriever" if found else "I don't know."
    return json.dumps({"answer": answer, "citations": [], "evidence_sufficient": found})


@pytest.fixture
def eval_settings(offline_settings: Settings, tmp_path: Path) -> Settings:
    paths = offline_settings.paths.model_copy(update={"eval_runs": tmp_path / "eval"})
    return offline_settings.model_copy(update={"paths": paths})


def run(settings: Settings, fake_api: FakeAPI, **overrides: Any) -> Any:
    fake_api.chat_replies = [fake_model]
    config = RunConfig(
        name=overrides.pop("name", "test-run"),
        dataset="longmemeval_s",
        split=None,
        methods=overrides.pop("methods", ["B0", "B1", "B2", "ORACLE"]),
        workers=overrides.pop("workers", 2),
        **overrides,
    )
    runner = EvalRunner(settings, config, http_client=fake_api.client())
    return runner, runner.run(load_longmemeval(FIXTURES / "longmemeval_mini.json"))


def test_end_to_end_resume_and_report(eval_settings: Settings, fake_api: FakeAPI) -> None:
    _, summary = run(eval_settings, fake_api)
    assert (summary.planned, summary.completed, summary.errors) == (12, 12, 0)

    rows = load_rows(summary.run_dir, "results.jsonl")
    correct = {(r["sample_id"], r["method"]): r["correct"] for r in rows}
    # 无记忆答不出事实题，但拒答题天然答对；全上下文、检索与上限参照都能答对事实题
    assert not correct[("q-ssu", "B0")]
    assert correct[("q-cat_abs", "B0")]
    assert correct[("q-ssu", "B1")] and correct[("q-ssu", "B2")] and correct[("q-ssu", "ORACLE")]

    b2 = next(r for r in rows if r["sample_id"] == "q-ssu" and r["method"] == "B2")
    assert b2["retrieval"]["session_recall_any"] == 1.0
    assert b2["retrieved"]  # 记录了命中的轮次
    b1 = next(r for r in rows if r["sample_id"] == "q-ssu" and r["method"] == "B1")
    assert b1["usage"]["llm_calls"] == 1
    assert b1["usage"]["cost"] > 0
    assert b1["judge_usage"]["llm_calls"] == 1  # 判分单独统计

    # 断点续跑：已完成的组合全部跳过
    _, again = run(eval_settings, fake_api)
    assert again.planned == 0
    assert len(load_rows(summary.run_dir, "results.jsonl")) == 12

    report = build_report(summary.run_dir).read_text(encoding="utf-8")
    assert "## 总体结果" in report and "| B2 朴素检索 |" in report
    assert "## 检索质量" in report
    # 时间题（答案 12 天）连 ORACLE 也答错，应归为阅读与推理问题
    assert "## 错误归因" in report
    assert "| ORACLE 也答错：阅读与推理问题 | 1 |" in report
    assert (summary.run_dir / "pareto-light.png").exists()
    assert (summary.run_dir / "pareto-dark.png").exists()


def test_error_attribution() -> None:
    from cuemem.evals.report import error_attribution

    def row(
        sample_id: str, method: str, correct: bool, recall: float | None = None
    ) -> dict[str, Any]:
        retrieval = None if recall is None else {"turn_recall_any": recall}
        return {
            "sample_id": sample_id,
            "method": method,
            "correct": correct,
            "is_abstention": False,
            "retrieval": retrieval,
        }

    rows = [
        row("q1", "B2", False, recall=0.0),
        row("q1", "ORACLE", True),  # 检索没命中
        row("q2", "B2", False, recall=1.0),
        row("q2", "ORACLE", True),  # 命中但干扰多
        row("q3", "B2", False, recall=1.0),
        row("q3", "ORACLE", False),  # 阅读与推理问题
        row("q4", "B2", True, recall=1.0),
        row("q4", "ORACLE", True),
    ]
    assert error_attribution(rows) == {
        "B2": {"retrieval_miss": 1, "noisy_or_incomplete": 1, "reading": 1}
    }


def test_budget_guard_stops_early(eval_settings: Settings, fake_api: FakeAPI) -> None:
    _, summary = run(
        eval_settings, fake_api, name="budget", methods=["B1"], workers=1, max_cost=1e-9
    )
    assert summary.stopped_by_budget
    assert summary.completed == 1  # 第一个组合花掉预算后就停了
    assert summary.spent > 0


class BoomMethod:
    name, label, needs_index = "BOOM", "总是出错", False

    def answer(self, ctx: MethodContext) -> MethodOutput:
        raise RuntimeError("故意出错")


def test_errors_are_recorded_and_retried(eval_settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = [fake_model]
    config = RunConfig(name="errors", dataset="longmemeval_s", split=None, methods=["B0"])
    runner = EvalRunner(eval_settings, config, http_client=fake_api.client())
    runner.methods = [BoomMethod()]
    summary = runner.run(load_longmemeval(FIXTURES / "longmemeval_mini.json"))
    assert (summary.completed, summary.errors) == (0, 3)
    errors = load_rows(summary.run_dir, "errors.jsonl")
    assert {e["stage"] for e in errors} == {"answer"}
    assert "故意出错" in errors[0]["error"]
    assert not (summary.run_dir / "results.jsonl").exists()  # 出错的组合不算完成，下次会重试


class FailingEmbedder:
    name, dim, supports_sparse = "failing", 256, True

    def embed(self, texts: Any, kind: Any) -> Any:
        raise EmbeddingAPIError("HTTP 429 模拟限流")


def test_ingest_failure_only_affects_index_methods(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    fake_api.chat_replies = [fake_model]
    config = RunConfig(
        name="ingest-fail", dataset="longmemeval_s", split=None, methods=["B0", "B2"]
    )
    runner = EvalRunner(eval_settings, config, http_client=fake_api.client())
    runner._embedder = FailingEmbedder()
    summary = runner.run(load_longmemeval(FIXTURES / "longmemeval_mini.json"))
    assert (summary.completed, summary.errors) == (3, 3)
    assert {r["method"] for r in load_rows(summary.run_dir, "results.jsonl")} == {"B0"}
    errors = load_rows(summary.run_dir, "errors.jsonl")
    assert {(e["method"], e["stage"]) for e in errors} == {("B2", "ingest")}
