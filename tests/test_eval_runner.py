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


class MismatchedMethod:
    """像写入目录里是另一个协议的写入的外部基线：每道题按自己的写入目录（<eval_spaces>/graphiti/
    <空间>）报协议错误（与 ExternalBaseline 同一个 check_protocol），不花钱。"""

    name, label, needs_index, needs_consolidation = "MX", "协议对不上", False, False

    def __init__(self, root: Path) -> None:
        self.root = root

    def answer(self, ctx: MethodContext) -> MethodOutput:
        from cuemem.evals.external import check_protocol, space_key

        directory = self.root / "graphiti" / space_key(ctx.space_id)
        check_protocol({"protocol": "a|e:8"}, "b|e:8", directory, "graphiti")
        raise AssertionError("协议对不上时应当报错")


def test_config_mismatch_from_a_baseline_is_reported_once(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    """外部基线的协议报错每道题各记一次（errors.jsonl，每道题的写入目录不同），其余方法照常；
    报告与命令行把写入目录里的空间名换成占位符归并，只列一次，带上解决办法。"""
    from cuemem.evals.report import error_groups, pending_errors

    fake_api.chat_replies = [fake_model]
    config = RunConfig(name="mismatch", dataset="longmemeval_s", split=None, methods=["B0"])
    runner = EvalRunner(eval_settings, config, http_client=fake_api.client())
    root = eval_settings.paths.eval_spaces
    runner.methods = [*runner.methods, MismatchedMethod(root)]
    summary = runner.run(load_longmemeval(FIXTURES / "longmemeval_mini.json"))
    assert (summary.completed, summary.errors) == (3, 3)
    raw = {e["error"] for e in load_rows(summary.run_dir, "errors.jsonl")}
    assert len(raw) == 3  # 每道题一个空间、一个写入目录
    [(method, error, n)] = error_groups(pending_errors(summary.run_dir))
    assert (method, n) == ("MX", 3)
    assert error.startswith(f"ConfigMismatchError: {root / 'graphiti' / '<空间>'} 是另一个协议")
    assert "CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen" in error
    report = build_report(summary.run_dir).read_text(encoding="utf-8")
    assert "有 3 个组合出错未计入结果" in report
    assert f"  - MX × 3：{error}" in report and "另有" not in report


def test_error_groups_ignore_what_differs_per_question() -> None:
    """归并时只把因题而异的部分换成占位符：外部基线写入目录里的空间名与上游的请求 ID；
    外部系统自己的包路径、别的报错照原样，不同的报错不并在一起。"""
    from cuemem.evals.report import error_groups

    def limited(request_id: str) -> str:
        return f"RateLimitError: Error code: 429 - Too many requests. (request_id: {request_id})"

    errors = [
        {"method": "MC", "error": limited("0d00bab8-57d0-4752-910e-6ea9b3ca5b33")},
        {"method": "MC", "error": limited("3df0f52c-30f4-427a-880a-75aab843b077")},
        {"method": "ML", "error": "ConfigMismatchError: /d/langmem/longmemeval_s-anon_q1 是另一个"},
        {"method": "ML", "error": "ConfigMismatchError: /d/langmem/longmemeval_s-anon_q2 是另一个"},
        {"method": "M0", "error": "ImportError: /venv/site-packages/mem0/memory/main.py"},
        {"method": "M0", "error": "ImportError: /venv/site-packages/mem0/vector_stores/qdrant.py"},
    ]
    assert error_groups(errors) == [
        ("MC", limited("<id>"), 2),
        ("ML", "ConfigMismatchError: /d/langmem/<空间> 是另一个", 2),
        ("M0", "ImportError: /venv/site-packages/mem0/memory/main.py", 1),
        ("M0", "ImportError: /venv/site-packages/mem0/vector_stores/qdrant.py", 1),
    ]


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


def _consolidation_variant(settings: Settings, fake_api: FakeAPI, name: str, **config: Any) -> Any:
    """LoCoMo 迷你数据上跑 B1 与 B4（B4 要巩固；Alice 说的每句话记成一条事实）。"""
    from scripted import ScriptedModel, echo_extraction

    from cuemem.evals.datasets import load_locomo

    def extract(prompt: str) -> Any:
        return echo_extraction(prompt.replace("] Alice: ", "] User: "))

    fake_api.chat_replies = [with_extraction(ScriptedModel(extract))]
    run_config = RunConfig(
        name=name, dataset="locomo", split=None, methods=["B1", "B4"], workers=1, **config
    )
    runner = EvalRunner(settings, run_config, http_client=fake_api.client())
    return runner, runner.run(load_locomo(FIXTURES / "locomo_mini.json"))


def test_spaces_built_with_another_consolidation_config_are_not_rebuilt(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    """评测空间的目录不分巩固配置：指纹对不上的空间（如另一个协议的）不悄悄重建，需要事实记忆的
    方法报一条归并后的配置错误，其余方法照常；指纹相同的照常复用；--rebuild-spaces 时照旧重建。"""
    from cuemem.evals.report import error_groups, pending_errors
    from cuemem.evals.runner import space_dir
    from cuemem.store.sqlite import MemoryDB

    runner, first = _consolidation_variant(eval_settings, fake_api, "first")
    assert first.errors == 0
    directory = space_dir(eval_settings, "locomo", runner.embedder, "conv-test")
    build = (directory / "build.json").read_text("utf-8")

    def facts() -> int:
        db = MemoryDB(directory / "memory.sqlite")
        try:
            return len(db.get_facts("locomo:conv-test"))
        finally:
            db.close()

    built = facts()
    assert built > 0

    # 同一套配置：复用，不报错
    _, same = _consolidation_variant(eval_settings, fake_api, "same")
    assert same.errors == 0 and same.completed == first.completed
    assert all(r.get("reused") for r in load_rows(same.run_dir, "ingest.jsonl"))

    # 只换了备用模型（原协议与 pq 协议的差别）：指纹不同，不重建
    other = eval_settings.model_copy(
        update={
            "consolidation": eval_settings.consolidation.model_copy(
                update={"fallback_model": "qwen3.7-plus"}
            )
        }
    )
    _, mismatched = _consolidation_variant(other, fake_api, "mismatched")
    errors = load_rows(mismatched.run_dir, "errors.jsonl")
    assert {(e["method"], e["stage"]) for e in errors} == {("B4", "consolidate")}
    assert mismatched.errors == len(errors) > 1
    assert {r["method"] for r in load_rows(mismatched.run_dir, "results.jsonl")} == {"B1"}
    [(method, error, n)] = error_groups(pending_errors(mismatched.run_dir))
    assert (method, n) == ("B4", len(errors))
    assert error.startswith("ConfigMismatchError: ")
    assert "CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen" in error
    assert "--rebuild-spaces" in error
    assert (directory / "build.json").read_text("utf-8") == build
    assert facts() == built

    # 明确要求重建：照旧清空派生记忆，按这次的配置重新巩固
    _, rebuilt = _consolidation_variant(other, fake_api, "rebuilt", rebuild_spaces=True)
    assert rebuilt.errors == 0
    [row] = [r for r in load_rows(rebuilt.run_dir, "ingest.jsonl") if r["stage"] == "consolidate"]
    assert row["sessions"] > 0 and not row.get("reused")


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


def _with_role(settings: Settings, role: str, model: str) -> Settings:
    roles = dict(settings.roles)
    roles[role] = roles[role].model_copy(update={"model": model})
    return settings.model_copy(update={"roles": roles})


def _with_controller(settings: Settings, name: str) -> Settings:
    retrieval = settings.retrieval.model_copy(update={"controller": name})
    return settings.model_copy(update={"retrieval": retrieval})


def test_resume_refuses_other_settings_while_questions_remain(
    eval_settings: Settings, fake_api: FakeAPI
) -> None:
    """还有没答完的题时，判分模型、--k、--limit、要回答的题（--types）与控制器也要与首次一致；
    全部答完的运行不追加结果，配置变了也照常（复现脚本只是重新生成报告）。每次续跑都记下代码
    版本、命令行与环境变量。"""
    from cuemem.errors import ConfigMismatchError

    _, first = run(eval_settings, fake_api, name="r", methods=["B0"], limit=2)
    assert (first.planned, first.completed) == (2, 2)
    path = first.run_dir / "run.json"
    info = json.loads(path.read_text("utf-8"))
    assert info["retrieval"]["controller_version"] == "n11-two-stage-v3"
    assert len(info["retrieval"]["controller_sha256"]) == 64
    assert len(info["sample_ids_sha256"]) == 64 and info["samples"] == 3
    assert info["argv"] and isinstance(info["env"], dict) and info["resumes"] == []

    judged = _with_role(eval_settings, "judge", "deepseek-flash")
    _, again = run(judged, fake_api, name="r", methods=["B0"], limit=2)  # 全部答完：照常
    assert again.planned == 0 and len(load_rows(first.run_dir, "results.jsonl")) == 2
    info = json.loads(path.read_text("utf-8"))
    assert len(info["resumed"]) == 1 and isinstance(info["resumed"][0], str)
    [resume] = info["resumes"]
    assert resume["at"] == info["resumed"][0] and resume["git"] and "argv" in resume
    assert "env" in resume

    judge = re.escape("判分模型：qwen3.8-max → deepseek-flash")
    with pytest.raises(ConfigMismatchError, match=judge):
        run(judged, fake_api, name="r", methods=["B0", "B1"], limit=2)
    with pytest.raises(ConfigMismatchError, match="只跑前 N 题（--limit）：2 → None"):
        run(eval_settings, fake_api, name="r", methods=["B0"])
    with pytest.raises(ConfigMismatchError, match="检索条数（--k）：10 → 5"):
        run(eval_settings, fake_api, name="r", methods=["B0", "B2"], limit=2, k=5)
    with pytest.raises(ConfigMismatchError, match="控制器：n11-two-stage-v3 → n7-two-stage-v1"):
        run(_with_controller(eval_settings, "controller_model_v1.json"), fake_api, name="r",
            methods=["B6"], limit=2)  # fmt: skip
    evalset = load_longmemeval(FIXTURES / "longmemeval_mini.json")
    evalset.samples = evalset.samples[1:]  # 像 --types 筛掉了一种题型
    config = RunConfig(name="r", dataset="longmemeval_s", split=None, methods=["B0", "B1"], limit=2)
    with pytest.raises(ConfigMismatchError, match="要回答的题"):
        EvalRunner(eval_settings, config, http_client=fake_api.client()).run(evalset)
    assert len(json.loads(path.read_text("utf-8"))["resumes"]) == 1  # 拒绝的续跑什么都不写

    _, more = run(eval_settings, fake_api, name="r", methods=["B0", "B1"], limit=2)
    assert (more.planned, more.completed, more.errors) == (2, 2, 0)
    assert len(json.loads(path.read_text("utf-8"))["resumes"]) == 2


def test_resume_infers_what_old_runs_did_not_record(
    eval_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """旧的 run.json 没记题号的哈希时比题数；没记控制器（ad56fb0 之前，只有 v1）时按 v1 比，记的是
    带版本号的文件名时读那个文件，记的是 controller_model.json（先后指过 v2、v3）时提醒、不比；
    没有用控制器的方法时不比控制器。"""
    from cuemem.errors import ConfigMismatchError
    from cuemem.evals.runner import _check_resume_extended

    _, first = run(eval_settings, fake_api, name="old", methods=["B0"])
    info = json.loads((first.run_dir / "run.json").read_text("utf-8"))
    evalset = load_longmemeval(FIXTURES / "longmemeval_mini.json")
    retrieval = {k: v for k, v in info["retrieval"].items() if not k.startswith("controller")}
    legacy = {
        k: v for k, v in info.items() if k not in ("sample_ids_sha256", "env", "argv", "resumes")
    }
    legacy["retrieval"] = retrieval
    b6 = RunConfig(name="old", dataset="longmemeval_s", split=None, methods=["B6"])
    v1 = _with_controller(eval_settings, "controller_model_v1.json")
    v2 = _with_controller(eval_settings, "controller_model_v2.json")
    _check_resume_extended(info, eval_settings, b6, evalset)
    retrained = {**info, "retrieval": {**info["retrieval"], "controller_sha256": "0" * 64}}
    with pytest.raises(ConfigMismatchError, match="n11-two-stage-v3 的模型文件内容变了"):
        _check_resume_extended(retrained, eval_settings, b6, evalset)

    with pytest.raises(ConfigMismatchError, match="控制器：n7-two-stage-v1 → n11-two-stage-v3"):
        _check_resume_extended(legacy, eval_settings, b6, evalset)
    _check_resume_extended(legacy, v1, b6, evalset)
    named = {**legacy, "retrieval": {**retrieval, "controller": "controller_model_v2.json"}}
    with pytest.raises(ConfigMismatchError, match="n11-two-stage-v2 → n7-two-stage-v1"):
        _check_resume_extended(named, v1, b6, evalset)
    _check_resume_extended(named, v2, b6, evalset)
    ambiguous = {**legacy, "retrieval": {**retrieval, "controller": "controller_model.json"}}
    with pytest.warns(UserWarning, match="看不出是哪一版"):
        _check_resume_extended(ambiguous, v1, b6, evalset)
    b4 = RunConfig(name="old", dataset="longmemeval_s", split=None, methods=["B4"])
    _check_resume_extended(legacy, v1, b4, evalset)

    fewer = load_longmemeval(FIXTURES / "longmemeval_mini.json")
    fewer.samples = fewer.samples[1:]
    with pytest.raises(ConfigMismatchError, match="题数（划分与 --types 筛出的）：3 → 2"):
        _check_resume_extended(legacy, eval_settings, b4, fewer)
    # 巩固配置只影响用 CueMem 事实记忆的方法：外部基线与 B1 不比它
    other = eval_settings.model_copy(
        update={
            "consolidation": eval_settings.consolidation.model_copy(
                update={"fallback_model": "qwen3.7-plus"}
            )
        }
    )
    with pytest.raises(ConfigMismatchError, match=re.escape("consolidation.fallback_model")):
        _check_resume_extended(legacy, other, b4, evalset)
    external = {**legacy, "config": {**legacy["config"], "methods": ["MG"]}}
    b1 = RunConfig(name="old", dataset="longmemeval_s", split=None, methods=["B1"])
    _check_resume_extended(external, other, b1, evalset)
    # A-MEM 的检索条数只在环境变量里：记了环境变量的运行才比
    ma = RunConfig(name="old", dataset="longmemeval_s", split=None, methods=["MA"])
    _check_resume_extended(info, eval_settings, ma, evalset)
    monkeypatch.setenv("CUEMEM_AMEM_K", "40")
    with pytest.raises(ConfigMismatchError, match="CUEMEM_AMEM_K）：默认 → 40"):
        _check_resume_extended(info, eval_settings, ma, evalset)
    _check_resume_extended(legacy, eval_settings, ma, evalset)


def test_published_runs_pass_the_resume_checks(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docs/results 里的每个 run.json，在复现脚本给那一步的环境变量下（N2、N3 换回
    text-embedding-v4）都通过续跑的核对：复现脚本续跑这些运行不会被拦下（题数要数据集，
    这里按记下的题数，只核对配置）。"""
    import importlib.util
    import sys
    from types import SimpleNamespace

    from cuemem.config import load_settings
    from cuemem.evals.datasets import session_id_scheme
    from cuemem.evals.runner import _check_resume_compatible, _check_resume_extended

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("reproduce", root / "scripts" / "reproduce.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reproduce"] = module
    spec.loader.exec_module(module)
    steps = {step.copy[1]: step for step in module.STEPS if step.copy}
    paths = sorted((root / "docs" / "results").glob("**/run.json"))
    assert len(paths) >= 10
    for path in paths:
        step = steps[str(path.parent.relative_to(root / "docs" / "results"))]
        with monkeypatch.context() as patch:
            for name, value in step.environment().items():
                patch.setenv(name, value)
            settings = load_settings()
            info = json.loads(path.read_text("utf-8"))
            command = step.command
            config = RunConfig(
                name=info["config"]["name"],
                dataset=info["config"]["dataset"],
                split=info["config"]["split"],
                methods=command[command.index("--methods") + 1].split(","),
                limit=int(command[command.index("--limit") + 1]) if "--limit" in command else None,
            )
            scheme = session_id_scheme(config.dataset)
            samples = [SimpleNamespace(sample_id=str(i)) for i in range(info["samples"])]
            evalset: Any = SimpleNamespace(samples=samples, session_ids=scheme, raw_labels=False)
            _check_resume_compatible(info, settings, config, scheme)
            _check_resume_extended(info, settings, config, evalset)


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


def _reproduce() -> Any:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "reproduce.py"
    spec = importlib.util.spec_from_file_location("reproduce", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reproduce"] = module  # dataclass 需要在 sys.modules 里找到所在模块
    spec.loader.exec_module(module)
    return module


def _is_eval_run(command: list[str]) -> bool:
    i = command.index("cuemem") if "cuemem" in command else -1
    return i >= 0 and command[i + 1 : i + 3] == ["eval", "run"]


TRAINING_SCRIPTS = (
    "scripts/n7_controller.py",
    "scripts/n11_controller_v2.py",
    "scripts/n11_see_gate.py",
)


def test_reproduce_does_not_overwrite_frozen_artifacts() -> None:
    """复现脚本只核对、不覆盖控制器的三个模型文件（v1、v2 是 N9、N11 冻结评测用的，v3 是默认
    的）；每次评测运行都加 --fail-on-error（出错或提前停止时不复制报告）；Mem0 固定在 N11、N12
    用的 2.2.1。"""
    module = _reproduce()
    trainers = [s for s in module.STEPS if any(p in s.command for p in TRAINING_SCRIPTS)]
    assert len(trainers) == len(TRAINING_SCRIPTS)
    assert all("--no-export" in s.command and "--check" in s.command for s in trainers)
    runs = [s for s in module.STEPS if _is_eval_run(s.command)]
    assert len(runs) >= 25 and all("--fail-on-error" in s.command for s in runs)
    assert all(s.run_name() for s in runs)
    mem0 = [s for s in runs if "M0" in s.command]
    assert len(mem0) == 2 and all("mem0ai==2.2.1" in s.command for s in mem0)


@pytest.mark.parametrize("script", TRAINING_SCRIPTS)
def test_training_scripts_accept_the_verify_flags(script: str) -> None:
    """复现脚本传给训练脚本的 --no-export 与 --check，三个脚本都认。"""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [sys.executable, script, "--help"], cwd=root, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert "--no-export" in done.stdout and "--check" in done.stdout


def test_training_scripts_check_before_they_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """三个训练脚本共用的 finish_export：--no-export --check 从不写；对不上时在写任何东西之前以
    非零退出码结束；不带参数时写入。三个脚本都只经它写控制器文件，并且在写结果之前调用。"""
    import argparse
    import importlib
    import re

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    n7 = importlib.import_module("n7_controller")
    parser = argparse.ArgumentParser()
    n7.add_export_flags(parser, "controller_model.json")
    frozen = {"lambda": 0.1, "coef": [[1.0, 2.0]], "created": "2026-01-01"}
    path = tmp_path / "controller_model.json"
    path.write_text(json.dumps(frozen), encoding="utf-8")
    committed = path.read_bytes()

    same = {**frozen, "coef": [[1.0, 2.000001]], "created": "2026-10-21"}  # 浮点噪声、导出日期
    assert not n7.finish_export(parser.parse_args(["--no-export", "--check"]), path, same)
    drifted = {**frozen, "coef": [[1.0, 2.5]]}
    for argv in (["--no-export", "--check"], ["--check"]):
        with pytest.raises(SystemExit, match="对不上"):
            n7.finish_export(parser.parse_args(argv), path, drifted)
    assert path.read_bytes() == committed
    missing = tmp_path / "missing.json"
    assert not n7.finish_export(parser.parse_args(["--no-export"]), missing, drifted)
    assert not missing.exists()
    assert n7.finish_export(parser.parse_args([]), path, drifted)  # 有意更新：直接运行时照旧导出
    assert path.read_text("utf-8") == json.dumps(drifted, ensure_ascii=False, indent=1) + "\n"

    for script in TRAINING_SCRIPTS:
        source = (root / script).read_text("utf-8")
        assert not re.search(r"\b(V1_MODEL|V2_MODEL|DEFAULT_MODEL)\.write_text", source), script
        main = source[source.index("\ndef main(") :]  # docs/results 下的结果在 OUT 里
        assert 0 <= main.index("finish_export(") < main.index("(OUT / "), script


class _FakeSteps:
    """替身的 subprocess.run：评测运行按 plan 依次往运行目录里追加结果与报错，返回给定的退出码。"""

    def __init__(self, runs: Path) -> None:
        self.runs = runs
        self.plan: dict[str, list[tuple[int, list[str], list[str]]]] = {}
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        from types import SimpleNamespace

        self.calls.append((command, kwargs["env"]))
        if "--name" not in command:
            return SimpleNamespace(returncode=int("fail" in command[-1]))
        name = command[command.index("--name") + 1]
        code, answered, failed = self.plan[name].pop(0)
        run_dir = self.runs / name
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "results.jsonl").open("a", encoding="utf-8") as f:
            f.writelines(json.dumps({"sample_id": q, "method": "B1"}) + "\n" for q in answered)
        with (run_dir / "errors.jsonl").open("a", encoding="utf-8") as f:
            for q in failed:
                f.write(json.dumps({"sample_id": q, "method": "B1", "error": "X: boom"}) + "\n")
        (run_dir / "report.md").write_text(f"report {len(self.calls)}", encoding="utf-8")
        (run_dir / "run.json").write_text(f"run {len(self.calls)}", encoding="utf-8")
        return SimpleNamespace(returncode=code)


@pytest.fixture
def reproduce(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _FakeSteps]:
    module = _reproduce()
    monkeypatch.setattr(module, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(module, "RESULTS", tmp_path / "results")
    fake = _FakeSteps(tmp_path / "runs")
    monkeypatch.setattr(module.subprocess, "run", fake)
    monkeypatch.delenv("CUEMEM_LLM_CACHE_MODE", raising=False)
    return module, fake


def test_reproduce_copies_only_complete_runs(
    reproduce: tuple[Any, _FakeSteps], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """复制报告的步骤：出错（退出码 3）或提前停止（4）时脚本停下，docs/results 不动；续跑没有
    新答的题时报告照常复制，run.json 留着原来的（续跑只在里面追加时间与环境）。"""
    module, fake = reproduce
    monkeypatch.setattr(
        module, "STEPS", [module.run("N2", "n2-run", "longmemeval_s", "dev100", "B1", copy="n2")]
    )
    dest = tmp_path / "results" / "n2"
    fake.plan["n2-run"] = [(3, ["q1"], ["q2"]), (4, ["q2"], []), (0, ["q3"], [])]
    for _ in range(2):
        with pytest.raises(SystemExit):
            module.main([])
        assert not dest.exists()
    module.main([])
    assert (dest / "run.json").read_text("utf-8") == "run 3"
    assert (dest / "report.md").read_text("utf-8") == "report 3"
    fake.plan["n2-run"] = [(0, [], [])]  # 全部答完：续跑没有新答的题
    module.main([])
    assert (dest / "report.md").read_text("utf-8") == "report 4"
    assert (dest / "run.json").read_text("utf-8") == "run 3"
    assert "--fail-on-error" in fake.calls[0][0]
    assert fake.calls[0][1].get("CUEMEM_LLM_CACHE_MODE") is None  # 默认照常读写缓存


def test_reproduce_tolerates_only_the_errors_it_already_had(
    reproduce: tuple[Any, _FakeSteps], monkeypatch: pytest.MonkeyPatch
) -> None:
    """不复制报告的评测运行：出错的组合与这一步之前完全相同（发表时就有、每次重跑都一样的出错，
    如内容审核拒绝的题）时继续往下走；第一次出错、多了新的出错或提前停止时停下。"""
    module, fake = reproduce
    after = module.script("N9", "scripts/after.py")
    steps = [module.run("N9", "n9-run", "longmemeval_s", "test400", "B1"), after]
    monkeypatch.setattr(module, "STEPS", steps)
    fake.plan["n9-run"] = [(3, ["q1"], ["q2"]), (3, [], ["q2"]), (3, [], ["q2", "q3"]), (4, [], [])]
    with pytest.raises(SystemExit, match="再运行一次"):
        module.main([])
    module.main([])  # 与上次相同的出错：照常继续
    assert fake.calls[-1][0] == after.command
    for _ in range(2):  # 多了 q3；提前停止
        with pytest.raises(SystemExit):
            module.main([])
        assert fake.calls[-1][0] != after.command


def test_reproduce_stops_when_a_probe_fails(
    reproduce: tuple[Any, _FakeSteps], monkeypatch: pytest.MonkeyPatch
) -> None:
    """探针有题出错（cuemem eval probe --fail-on-error 的退出码 3）时脚本停下：之后的分析
    （n9_mixed、n9_report 读 routing.load_items）会悄悄略过没有探针特征的题，用少了的题重写
    docs/results。探针出错不按"与之前相同的出错"放过。"""
    module, fake = reproduce
    probes = [s for s in module.STEPS if s.command[3:5] == ["eval", "probe"]]
    assert len(probes) == 7 and all("--fail-on-error" in s.command for s in probes)
    assert all(s.run_name() is None for s in probes)
    probe = module.probe("N9", "n9-run", "longmemeval_s", "test400")
    after = module.script("N9", "scripts/n9_report.py")
    monkeypatch.setattr(module, "STEPS", [probe, after])
    fake.plan["n9-run"] = [(3, [], []), (3, [], []), (0, [], [])]
    for _ in range(2):
        with pytest.raises(SystemExit):
            module.main([])
        assert fake.calls[-1][0] == probe.command
    module.main([])
    assert fake.calls[-1][0] == after.command


def test_reproduce_cache_only(
    reproduce: tuple[Any, _FakeSteps], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--cache-only：每一步都只读缓存；关掉缓存的步骤跳过；演示回放没有复现时只跳过它所在的
    节点，不加 --cache-only 时照旧停下。"""
    module, fake = reproduce
    steps = [
        module.run("N8", "nocache", "locomo", "test7", "B1", env=module.NO_CACHE, optional=True),
        module.Step("N8", ["uv", "run", "cuemem", "demo", "fail"], best_effort=True),
        module.script("N8", "scripts/n8_summary.py"),
        module.script("N9", "scripts/n9_report.py"),
    ]
    monkeypatch.setattr(module, "STEPS", steps)
    module.main(["--cache-only", "--with-optional"])
    commands = [command for command, _ in fake.calls]
    assert commands == [steps[1].command, steps[3].command]
    assert all(env["CUEMEM_LLM_CACHE_MODE"] == "read_only" for _, env in fake.calls)
    with pytest.raises(SystemExit):
        module.main([])


def test_reproduce_skips_analyses_whose_runs_are_missing(
    reproduce: tuple[Any, _FakeSteps], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读别的运行的分析（Step.requires）：执行到这一步时还缺运行的结果就跳过，不拿缺了的数据
    重写 docs/results。以前可选的重跑没跑时照样执行 n11_variance，把 variance.json 写成 {}。"""
    module, fake = reproduce
    (variance,) = [s for s in module.STEPS if "scripts/n11_variance.py" in s.command]
    reruns = [s.run_name() for s in module.STEPS if s.optional and s.node == "N11"]
    assert set(variance.requires) == {*reruns, "test400-n9", "locomo-test7-n9"}

    rerun = module.run("N11", "rerun", "locomo", "test7", "B1", env=module.NO_CACHE, optional=True)
    analysis = module.script("N11", "scripts/n11_variance.py", requires=("rerun", "n9"))
    monkeypatch.setattr(module, "STEPS", [rerun, analysis])
    module.main(["--dry-run"])
    module.main([])
    assert fake.calls == []
    for name in ("rerun", "n9"):
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / "results.jsonl").write_text("{}\n", "utf-8")
    module.main([])
    assert [command for command, _ in fake.calls] == [analysis.command]


def test_reproduce_writes_machine_dependent_results_under_runs() -> None:
    """延迟基准与演示摘要随机器、随缓存而变：复现脚本让它们写到 runs/ 下，不覆盖发表的
    latency.json（另一台机器测的）与 demo.json（从缓存回放时花费与模型耗时接近 0）。"""
    module = _reproduce()
    for script in ("scripts/n11_latency.py", "scripts/n8_summary.py"):
        (step,) = [s for s in module.STEPS if script in s.command]
        out = step.command[step.command.index("--out") + 1]
        assert out.startswith("runs/"), script


def _script_module(name: str) -> Any:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_n11_variance_does_not_write_without_every_run(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺一个重跑或原运行就退出码非零、不写：以前打印"还没有结果，跳过"后照样写 {}，把发表的
    variance.json 清空。"""
    module = _script_module("n11_variance")
    out = tmp_path / "variance.json"
    out.write_text("published\n", "utf-8")
    monkeypatch.setattr(module, "OUT", out)
    runs = tmp_path / "eval"
    (runs / "test400-n9").mkdir(parents=True)
    row = {"sample_id": "q1", "method": "B6", "correct": True, "response": "x"}
    (runs / "test400-n9" / "results.jsonl").write_text(json.dumps(row) + "\n", "utf-8")
    monkeypatch.setenv("CUEMEM_PATHS__EVAL_RUNS", str(runs))
    with pytest.raises(SystemExit, match="test400-n11-rerun"):
        module.main()
    assert out.read_text("utf-8") == "published\n"


def _published(tmp_path: Path, module: Any, name: str) -> Path:
    """脚本要写的结果换成 tmp_path 下一份"发表的"文件（只有一行 published），连同 Holm 那份。"""
    out = tmp_path / "results" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("published\n", "utf-8")
    module.OUT = out
    return out


def _run_rows(runs: Path, name: str, *rows: dict[str, Any]) -> None:
    (runs / name).mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row) + "\n" for row in rows)
    (runs / name / "results.jsonl").write_text(text, "utf-8")


def test_reproduce_analyses_require_the_runs_they_read() -> None:
    """复现脚本里 Mem0 对比与第二判分这两步写明要读的运行（Step.requires），与脚本自己读的
    运行一致：以前没写，缺了 N9 的运行（只拷了缓存的机器、--only N11）照样执行、重写发表的结果。"""
    module = _reproduce()
    for script in ("n11_mem0", "n11_second_judge"):
        (step,) = [s for s in module.STEPS if f"scripts/{script}.py" in s.command]
        reads = _script_module(script).RUNS
        if script == "n11_mem0":  # Mem0 的运行 → （数据集，N9 的运行）
            wanted = {*reads, *(original for _, original in reads.values())}
        else:  # 数据集 → （N9 的运行，划分）
            wanted = {run for run, _ in reads.values()}
        assert set(step.requires) == wanted, script


def test_reproduce_n9_report_requires_the_test_runs() -> None:
    """复现脚本里 N9 汇总这一步写明要四个测试集的运行（Step.requires），与脚本自己读的
    TEST_RUNS 一致：以前没写，只拷了缓存、缺了运行的机器上照样执行，重写发表的结果。"""
    module = _reproduce()
    (step,) = [s for s in module.STEPS if "scripts/n9_report.py" in s.command]
    assert set(step.requires) == set(_script_module("n9_report").TEST_RUNS.values())


def test_n11_mem0_does_not_write_without_every_run(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺一个 Mem0 的运行或 N9 的运行就退出码非零、不写 mem0.json 与 mem0-holm.json：以前跳过缺
    的 Mem0 运行、把缺的 N9 运行当作空的，照样重写（只有 Mem0 的准确率，配对都是 +0 / −0）。"""
    module = _script_module("n11_mem0")
    out = _published(tmp_path, module, "mem0.json")
    runs = tmp_path / "eval"
    row = {"sample_id": "q1", "method": "M0", "correct": True, "question_type": "single"}
    _run_rows(runs, "locomo-test200r-mem0", row)
    monkeypatch.setenv("CUEMEM_PATHS__EVAL_RUNS", str(runs))
    with pytest.raises(SystemExit, match="test100r-mem0") as raised:
        module.main()
    assert "locomo-test7-n9" in str(raised.value) and "test400-n9" in str(raised.value)
    assert out.read_text("utf-8") == "published\n"
    assert [p.name for p in out.parent.iterdir()] == ["mem0.json"]


def test_n11_second_judge_does_not_write_without_every_dataset(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """report 缺一个数据集的重判就退出码非零、不写：以前跳过它照样重写 second_judge.json（没有
    runs/eval 时只剩判分模型的名字）。"""
    from cuemem.config import load_settings

    module = _script_module("n11_second_judge")
    out = _published(tmp_path, module, "second_judge.json")
    runs = tmp_path / "eval"
    (runs / "n11-judge2").mkdir(parents=True)
    verdict = {"dataset": "longmemeval_s", "sample_id": "q1", "method": "B6"}
    line = json.dumps({**verdict, "primary": True, "second": True}) + "\n"
    (runs / "n11-judge2" / "verdicts.jsonl").write_text(line, "utf-8")
    monkeypatch.setenv("CUEMEM_PATHS__EVAL_RUNS", str(runs))
    with pytest.raises(SystemExit, match="locomo") as raised:
        module.report(load_settings())
    assert "memlens32-test-n9" in str(raised.value)
    assert out.read_text("utf-8") == "published\n"
    assert [p.name for p in out.parent.iterdir()] == ["second_judge.json"]


def test_n12_leak_does_not_write_without_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺一个运行就退出码非零、不写 summary.json 与 summary-holm.json：以前缺的当作空的，没有
    runs/eval 时把发表的 summary.json 从 1.6 万字节写成几百字节。没有跑过的三个 Mem0 运行
    （NOT_RUN，README §3.3）不算缺。"""
    module = _script_module("n12_leak")
    out = _published(tmp_path, module, "summary.json")
    monkeypatch.setattr(module, "RUNS", tmp_path / "eval")
    every = [run for arms in module.ARMS.values() for runs in arms.values() for run in runs]
    assert set(every) > module.NOT_RUN
    for run in every:
        if run not in module.NOT_RUN:
            _run_rows(module.RUNS, run, {"sample_id": "q1", "method": "B1", "correct": True})
    assert module.missing() == []
    (module.RUNS / "orig-lme-label-cuemem" / "results.jsonl").unlink()
    with pytest.raises(SystemExit, match="orig-lme-label-cuemem"):
        module.main()
    assert out.read_text("utf-8") == "published\n"
    assert [p.name for p in out.parent.iterdir()] == ["summary.json"]


def test_n9_report_does_not_write_without_every_test_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺 N9 四个测试集的任何一个运行就退出码非零、什么都不写，也不建缓存目录：runs/ 不在仓库
    里，以前在没有运行的克隆里照着结果页的命令运行，退出码 0，把发表的 results.json 从 12.7 万
    字节写成 138 字节，results-holm.json 与 tables.md 也几乎清空。"""
    import sys

    module = _script_module("n9_report")
    out = tmp_path / "results" / "n9-final"
    out.mkdir(parents=True)
    published = {name: f"published {name}\n" for name in ("results.json", "results-holm.json")}
    published["tables.md"] = "published tables\n"
    for name, text in published.items():
        (out / name).write_text(text, "utf-8")
    runs = tmp_path / "eval"
    monkeypatch.setattr(module, "OUT", out)
    monkeypatch.setattr(module, "RUNS_DIR", runs)
    monkeypatch.setattr(module, "CACHE", runs / "n9-final")
    monkeypatch.setattr(module.n9_tables, "OUT", out)
    monkeypatch.setattr(sys, "argv", ["n9_report.py"])
    test_runs = list(module.TEST_RUNS.values())
    with pytest.raises(SystemExit, match="什么都没有写") as raised:
        module.main()
    assert all(run in str(raised.value) for run in test_runs)
    assert not runs.exists()
    for run in test_runs[:-1]:
        _run_rows(runs, run, {"sample_id": "q1", "method": "B1", "correct": True})
    with pytest.raises(SystemExit, match=test_runs[-1]) as raised:
        module.main()
    assert test_runs[0] not in str(raised.value)
    assert {p.name: p.read_text("utf-8") for p in out.iterdir()} == published
    assert sorted(p.name for p in runs.iterdir()) == sorted(test_runs[:-1])


def test_n8_summary_does_not_publish_a_cached_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回放里有命中本地缓存的模型调用（花费记 0、几乎不耗时）时不写发表的 demo.json；--out
    另写、--allow-cached 照写。以前从缓存回放就把发表的花费与耗时改成接近 0。"""
    module = _script_module("n8_summary")
    out = tmp_path / "demo.json"
    out.write_text("published\n", "utf-8")
    monkeypatch.setattr(module, "OUT", out)
    usage = {
        "llm_calls": 2.0,
        "cache_hits": 2.0,
        "input_tokens": 10.0,
        "cached_input_tokens": 0.0,
        "output_tokens": 5.0,
        "spent": 0.0,
    }
    turn = {
        "session": "s1",
        "index": 0,
        "text": "hi",
        "passed": True,
        "failures": [],
        "level": "L0",
        "expected_level": "L0",
        "reply": {
            "trace": {
                "usage": usage,
                "scores": {},
                "tools": [],
                "recalled": [],
                "partitions": {},
                "timings": {"total": 5.0, "llm": 1.0, "write": 1.0},
            },
            "citations": [],
            "plain": "hello",
        },
    }
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"turns": [turn]}), "utf-8")
    with pytest.raises(SystemExit, match="--allow-cached"):
        module.main([str(report)])
    assert out.read_text("utf-8") == "published\n"
    elsewhere = tmp_path / "runs" / "summary.json"
    module.main([str(report), "--out", str(elsewhere)])
    assert json.loads(elsewhere.read_text("utf-8"))["summary"]["spent_per_turn"] == 0.0
    module.main([str(report), "--allow-cached"])
    assert json.loads(out.read_text("utf-8"))["summary"]["chat_turns"] == 1
