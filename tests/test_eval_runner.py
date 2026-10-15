"""评测运行器的端到端测试：迷你 LongMemEval + 模拟接口 + 离线哈希向量器。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI

from cuemem.config import Settings
from cuemem.errors import ConfigMismatchError, EmbeddingAPIError
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


MISMATCH = (
    "/data/eval-spaces/toy/longmemeval_s_q 是另一个协议（a|e:8）的 toy 写入，这次运行的协议是 "
    "b|e:8：不复用，也不覆盖。……设 CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen"
)


class MismatchedMethod:
    """像写入目录里是另一个协议的写入的外部基线：每道题都报同一个配置错误，不花钱。"""

    name, label, needs_index, needs_consolidation = "MX", "协议对不上", False, False

    def answer(self, ctx: MethodContext) -> MethodOutput:
        raise ConfigMismatchError(MISMATCH)


def test_config_mismatch_from_a_baseline_is_reported_once(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    """外部基线的协议报错每道题各记一次（errors.jsonl），其余方法照常；报告与命令行按报错归并，
    只列一次，带上解决办法。"""
    from cuemem.evals.report import error_groups, pending_errors

    fake_api.chat_replies = [fake_model]
    config = RunConfig(name="mismatch", dataset="longmemeval_s", split=None, methods=["B0"])
    runner = EvalRunner(eval_settings, config, http_client=fake_api.client())
    runner.methods = [*runner.methods, MismatchedMethod()]
    summary = runner.run(load_longmemeval(FIXTURES / "longmemeval_mini.json"))
    assert (summary.completed, summary.errors) == (3, 3)
    assert error_groups(pending_errors(summary.run_dir)) == [
        ("MX", f"ConfigMismatchError: {MISMATCH}", 3)
    ]
    report = build_report(summary.run_dir).read_text(encoding="utf-8")
    assert "有 3 个组合出错未计入结果" in report
    assert f"  - MX × 3：ConfigMismatchError: {MISMATCH}" in report


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


def with_extraction(extract: Any) -> Any:
    """在 fake_model 的基础上，把巩固阶段的抽取请求交给 extract 处理。"""

    def reply(body: dict[str, Any]) -> str:
        first = body["messages"][0]
        if first["role"] == "system" and first["content"].startswith("You build the long-term"):
            return str(extract(body))
        return fake_model(body)

    return reply


def test_fact_methods_consolidate_once_per_history(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    from scripted import ScriptedModel, echo_extraction

    fake_api.chat_replies = [with_extraction(ScriptedModel(echo_extraction))]
    config = RunConfig(
        name="facts", dataset="longmemeval_s", split=None, methods=["B2", "B2F", "B2F-current"]
    )
    runner = EvalRunner(eval_settings, config, http_client=fake_api.client())
    summary = runner.run(load_longmemeval(FIXTURES / "longmemeval_mini.json"))
    assert (summary.completed, summary.errors) == (9, 0)

    stages = load_rows(summary.run_dir, "ingest.jsonl")
    consolidated = [r for r in stages if r["stage"] == "consolidate"]
    assert len(consolidated) == 3  # 每份对话历史巩固一次，三个方法共用
    assert all(r["sessions"] == 3 and r["facts_extracted"] == 3 for r in consolidated)
    assert all(r["llm_calls"] == 3 and r["turns"] == 6 for r in consolidated)

    rows = load_rows(summary.run_dir, "results.jsonl")
    b2f = next(r for r in rows if r["sample_id"] == "q-ssu" and r["method"] == "B2F")
    assert b2f["correct"] and b2f["retrieved_facts"]
    assert b2f["retrieval"]["turn_recall_any"] == 1.0
    b2 = next(r for r in rows if r["sample_id"] == "q-ssu" and r["method"] == "B2")
    assert b2["retrieved_facts"] == []
    assert b2["usage"]["llm_calls"] == 1  # 巩固的调用不计入回答成本

    report = build_report(summary.run_dir).read_text(encoding="utf-8")
    assert "## 写入路径：巩固（N3）" in report
    assert "| 巩固（抽取、对账、摘要） |" in report
    assert "| B2F 朴素检索 + 事实记忆 |" in report


def test_consolidation_failure_only_affects_fact_methods(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    fake_api.chat_replies = [with_extraction(lambda body: "not json")]
    config = RunConfig(name="cfail", dataset="longmemeval_s", split=None, methods=["B2", "B2F"])
    runner = EvalRunner(eval_settings, config, http_client=fake_api.client())
    summary = runner.run(load_longmemeval(FIXTURES / "longmemeval_mini.json"))
    assert (summary.completed, summary.errors) == (3, 3)
    errors = load_rows(summary.run_dir, "errors.jsonl")
    assert {(e["method"], e["stage"]) for e in errors} == {("B2F", "consolidate")}


def test_spaces_are_reused_across_runs(eval_settings: Settings, fake_api: FakeAPI) -> None:
    """第二次运行（换个运行名）直接复用磁盘上的记忆空间：不再抽取，写入统计沿用首次构建的。"""
    from scripted import ScriptedModel, echo_extraction

    model = ScriptedModel(echo_extraction)
    fake_api.chat_replies = [with_extraction(model)]
    data = load_longmemeval(FIXTURES / "longmemeval_mini.json")
    first = EvalRunner(
        eval_settings,
        RunConfig(name="first", dataset="longmemeval_s", split=None, methods=["B2F"], workers=1),
        http_client=fake_api.client(),
    ).run(data)
    extractions = model.calls["extract"]
    # 三份迷你历史的会话内容相同，相同的抽取请求命中响应缓存（单线程，结果确定）
    assert extractions == 3
    second = EvalRunner(
        eval_settings,
        RunConfig(name="second", dataset="longmemeval_s", split=None, methods=["B2F"], workers=1),
        http_client=fake_api.client(),
    ).run(data)
    assert model.calls["extract"] == extractions  # 没有重新抽取
    rows = load_rows(second.run_dir, "ingest.jsonl")
    assert all(r.get("reused") for r in rows) and len(rows) == 6
    built = {(r["space_key"], r["stage"]): r for r in load_rows(first.run_dir, "ingest.jsonl")}
    for r in rows:
        assert (
            r["facts_extracted" if r["stage"] == "consolidate" else "units"]
            == built[(r["space_key"], r["stage"])][
                "facts_extracted" if r["stage"] == "consolidate" else "units"
            ]
        )
    assert (second.completed, second.errors) == (3, 0)


def test_resume_with_different_embedding_is_refused(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    """同名运行换了向量模型：拒绝续跑，避免新旧结果混在一个目录里。"""
    from cuemem.errors import ConfigMismatchError

    run(eval_settings, fake_api, name="same", methods=["B0"])
    other = eval_settings.model_copy(
        update={"embedding": eval_settings.embedding.model_copy(update={"model": "other-model"})}
    )
    with pytest.raises(ConfigMismatchError, match="向量模型"):
        run(other, fake_api, name="same", methods=["B0"])


def test_build_only_builds_spaces_without_answering(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    from scripted import ScriptedModel, echo_extraction

    fake_api.chat_replies = [with_extraction(ScriptedModel(echo_extraction))]
    config = RunConfig(
        name="build", dataset="longmemeval_s", split=None, methods=["B4"], build_only=True
    )
    runner = EvalRunner(eval_settings, config, http_client=fake_api.client())
    summary = runner.run(load_longmemeval(FIXTURES / "longmemeval_mini.json"))
    assert (summary.planned, summary.completed, summary.errors) == (3, 3, 0)
    assert load_rows(summary.run_dir, "results.jsonl") == []  # 没有回答、没有判分
    stages = {r["stage"] for r in load_rows(summary.run_dir, "ingest.jsonl")}
    assert stages == {"ingest", "consolidate"}
    # 之后正式运行时直接复用构建好的空间：不再抽取
    calls = len(fake_api.chat_requests)
    config = RunConfig(name="answer", dataset="longmemeval_s", split=None, methods=["B4"])
    EvalRunner(eval_settings, config, http_client=fake_api.client()).run(
        load_longmemeval(FIXTURES / "longmemeval_mini.json")
    )
    extracts = [
        r
        for r in fake_api.chat_requests[calls:]
        if r["messages"][0]["content"].startswith("You build the long-term")
    ]
    assert extracts == []


def test_probe_writes_controller_features(eval_settings: Settings, fake_api: FakeAPI) -> None:
    """eval probe：在构建好的空间上为每道题跑一次探针（不调用 LLM），特征写入 probe.jsonl。"""
    from scripted import ScriptedModel, echo_extraction

    fake_api.chat_replies = [with_extraction(ScriptedModel(echo_extraction))]
    evalset = load_longmemeval(FIXTURES / "longmemeval_mini.json")
    config = RunConfig(
        name="probe", dataset="longmemeval_s", split=None, methods=["B4"], build_only=True
    )
    EvalRunner(eval_settings, config, http_client=fake_api.client()).run(evalset)
    calls = len(fake_api.chat_requests)
    runner = EvalRunner(eval_settings, config, http_client=fake_api.client())
    done, errors = runner.probe(evalset)
    assert (done, errors) == (len(evalset.samples), 0)
    assert len(fake_api.chat_requests) == calls  # 探针不调用 LLM
    rows = load_rows(runner.run_dir, "probe.jsonl")
    assert {r["sample_id"] for r in rows} == {s.sample_id for s in evalset.samples}
    assert all(r["features"]["m_rounds"] > 0 and "q_first_person" in r["features"] for r in rows)
    assert runner.probe(evalset) == (0, 0)  # 已经算过的跳过


def test_reproduce_steps_use_known_methods() -> None:
    """复现脚本里每次评测运行的方法名都能被评测框架识别（拼错要几个小时后才会暴露）。"""
    import importlib.util
    import sys
    from pathlib import Path

    from cuemem.evals.methods import build_methods

    path = Path(__file__).resolve().parents[1] / "scripts" / "reproduce.py"
    spec = importlib.util.spec_from_file_location("reproduce", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reproduce"] = module  # dataclass 需要在 sys.modules 里找到所在模块
    spec.loader.exec_module(module)
    runs = [s for s in module.STEPS if s.command[3:5] == ["eval", "run"]]
    assert len(runs) >= 20
    for step in runs:
        methods = step.command[step.command.index("--methods") + 1].split(",")
        assert [m.name for m in build_methods(methods)] == methods
