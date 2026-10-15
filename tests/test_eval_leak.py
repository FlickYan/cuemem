"""N12 修复的金标泄漏：LongMemEval_S 的证据会话 ID 都以 answer_ 开头，而回答模型看到的每条检索
证据都标着会话 ID。加载时按时间顺序改写成 session_<i>（与 LoCoMo 相同的写法）；
CUEMEM_EVAL_RAW_SESSION_IDS=1 恢复原始 ID（复现泄漏的消融）；两种写法的持久化写入分开存、
不混用；LoCoMo 逐字节不变。全部离线。"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI
from scripted import ScriptedModel, echo_extraction

from cuemem.config import Settings
from cuemem.core.models import Hit
from cuemem.errors import ConfigMismatchError
from cuemem.evals.datasets import (
    RAW_SESSION_IDS_ENV,
    EvalSet,
    build_matches_session_ids,
    evidence_turns,
    load,
    load_locomo,
    load_longmemeval,
    sample_evidence_turns,
    session_id_scheme,
    storage_name,
    storage_space_id,
    to_conversation,
    to_session_ins,
)
from cuemem.evals.report import load_rows
from cuemem.evals.runner import EvalRunner, RunConfig, _check_resume_compatible, space_dir
from cuemem.read.reader import format_memories
from cuemem.store.sqlite import MemoryDB
from cuemem.write.online import build_rounds, build_turns

FIXTURES = Path(__file__).parent / "fixtures"

# 照 LongMemEval_S 的样子：证据会话 answer_…，其余 sharegpt_…、ultrachat_… 或十六进制；
# 顺序与时间对不上，有两个会话同一时间，还有一个会话出现两次（内容相同、时间不同）
HAYSTACK = [
    ("sharegpt_yywfIrx_0", "2023/05/25 (Thu) 09:00", "Any tips for a lightweight laptop?", False),
    ("answer_280352e9_2", "2023/05/22 (Mon) 09:00", "I hike in the Alps every weekend.", True),
    ("85a1be56_1", "2023/05/20 (Sat) 09:00", "I love baking sourdough bread.", False),
    ("ultrachat_412345", "2023/05/22 (Mon) 09:00", "What is the capital of Peru?", False),
    ("answer_280352e9_1", "2023/05/18 (Thu) 09:00", "I adopted a golden retriever, Max.", True),
    ("sharegpt_yywfIrx_0", "2023/05/28 (Sun) 09:00", "Any tips for a lightweight laptop?", False),
]
RAW_IDS = [sid for sid, *_ in HAYSTACK]
# 时间顺序：answer_1（5/18）、85a1be56（5/20）、answer_2 与 ultrachat 同在 5/22（照原顺序）、
# sharegpt（5/25；5/28 的重复会话沿用同一个 ID）
ANON_IDS = ["session_5", "session_3", "session_2", "session_4", "session_1", "session_5"]
RAW_MARKERS = ("answer_", "sharegpt_", "ultrachat_", "85a1be56", "280352e9")


def _session(text: str, evidence: bool) -> list[dict[str, Any]]:
    user: dict[str, Any] = {"role": "user", "content": text}
    if evidence:
        user["has_answer"] = True
    return [user, {"role": "assistant", "content": "Noted, thanks for sharing."}]


@pytest.fixture
def lme_path(tmp_path: Path) -> Path:
    item = {
        "question_id": "lk-1",
        "question_type": "multi-session",
        "question": "Which dog breed do I have, and where do I hike?",
        "question_date": "2023/06/01 (Thu) 10:00",
        "answer": "A golden retriever; the Alps.",
        "answer_session_ids": ["answer_280352e9_1", "answer_280352e9_2"],
        "haystack_session_ids": RAW_IDS,
        "haystack_dates": [date for _, date, _, _ in HAYSTACK],
        "haystack_sessions": [_session(text, ev) for _, _, text, ev in HAYSTACK],
    }
    path = tmp_path / "longmemeval_leak.json"
    path.write_text(json.dumps([item]), encoding="utf-8")
    return path


@pytest.fixture
def anon_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RAW_SESSION_IDS_ENV, raising=False)


# ---------- 加载 ----------


def test_session_ids_are_neutral_and_chronological(lme_path: Path, anon_env: None) -> None:
    evalset = load_longmemeval(lme_path)
    sessions = evalset.haystacks["lk-1"]
    assert evalset.session_ids == "anon"
    assert [s.session_id for s in sessions] == ANON_IDS
    assert [s.source_id for s in sessions] == RAW_IDS  # 原 ID 只留作追溯
    sample = evalset.samples[0]
    assert sample.evidence_session_ids == ["session_1", "session_3"]
    # 证据发言（has_answer）随会话改写：指标按改写后的 ID 算
    assert evidence_turns(sessions) == {("session_1", 0), ("session_3", 0)}
    assert sample_evidence_turns(sample, sessions) == {("session_1", 0), ("session_3", 0)}
    # 交给记忆系统的对话里没有原 ID（source_id 不往下传）
    dumped = to_conversation("space", sessions).model_dump_json()
    assert not any(marker in dumped for marker in RAW_MARKERS)


def test_raw_switch_restores_original_ids(lme_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RAW_SESSION_IDS_ENV, "1")
    evalset = load("longmemeval_s", lme_path)
    sessions = evalset.haystacks["lk-1"]
    assert evalset.session_ids == "raw"
    assert [s.session_id for s in sessions] == RAW_IDS
    assert evalset.samples[0].evidence_session_ids == ["answer_280352e9_1", "answer_280352e9_2"]
    assert evidence_turns(sessions) == {("answer_280352e9_1", 0), ("answer_280352e9_2", 0)}
    # 显式参数优先于环境变量
    assert load_longmemeval(lme_path, raw_ids=False).session_ids == "anon"
    for value in ("0", "", "no"):
        monkeypatch.setenv(RAW_SESSION_IDS_ENV, value)
        assert load_longmemeval(lme_path).session_ids == "anon"
    assert load_longmemeval(lme_path, raw_ids=True).session_ids == "raw"


def _hits(evalset: EvalSet, positions: list[int]) -> list[Hit]:
    """haystack 里这几个位置的会话各取第一轮，按给定顺序排成检索结果（两种写法下是同样的会话）。"""
    sessions = evalset.haystacks["lk-1"]
    hits = []
    for rank, pos in enumerate(positions, start=1):
        session = to_session_ins([sessions[pos]])[0]
        unit = build_rounds(build_turns("longmemeval_s:lk-1", session))[0]
        hits.append(Hit(unit=unit, score=1.0 / rank, route="dense", rank=rank))
    return hits


def test_retrieval_metrics_do_not_depend_on_the_id_scheme(lme_path: Path) -> None:
    anon = load_longmemeval(lme_path, raw_ids=False)
    raw = load_longmemeval(lme_path, raw_ids=True)
    for picks in ([4, 0, 2], [3, 1, 4], [0, 5], [1]):  # 证据、干扰、重复会话的各种组合
        metrics = [
            EvalRunner._retrieval(data.samples[0], data.haystacks["lk-1"], _hits(data, picks))
            for data in (anon, raw)
        ]
        assert metrics[0] == metrics[1], picks
    first = EvalRunner._retrieval(anon.samples[0], anon.haystacks["lk-1"], _hits(anon, [4, 0, 2]))
    assert first is not None
    assert first["session_recall_all"] == 0.5 and first["turn_recall_any"] == 1.0


def test_reader_context_shows_no_raw_session_ids(lme_path: Path) -> None:
    """回答模型看到的证据抬头：改写后只有 session_<i>；原始 ID 时证据会话一眼可见（泄漏）。"""
    anon = load_longmemeval(lme_path, raw_ids=False)
    context, _ = format_memories(_hits(anon, list(range(6))))
    assert "· session session_1" in context
    assert not any(marker in context for marker in RAW_MARKERS)
    raw = load_longmemeval(lme_path, raw_ids=True)
    leaked, _ = format_memories(_hits(raw, list(range(6))))
    assert "· session answer_280352e9_1" in leaked


# ---------- 端到端：交给模型的全部请求 ----------


def _reply(model: ScriptedModel) -> Any:
    """reader 固定回答、judge 判对，其余（抽取、对账、规划……）交给脚本化的模型。"""

    def reply(body: dict[str, Any]) -> str:
        first = body["messages"][0]
        content = body["messages"][-1]["content"]
        if isinstance(content, str) and content.startswith("I will give you"):
            return "yes"
        if first["role"] == "system" and first["content"].startswith(
            "You are a personal assistant"
        ):
            return json.dumps(
                {"answer": "golden retriever", "citations": [], "evidence_sufficient": True}
            )
        return str(model(body))

    return reply


@pytest.fixture
def eval_settings(offline_settings: Settings, tmp_path: Path) -> Settings:
    paths = offline_settings.paths.model_copy(update={"eval_runs": tmp_path / "eval"})
    return offline_settings.model_copy(update={"paths": paths})


def _run(
    settings: Settings, fake_api: FakeAPI, evalset: EvalSet, name: str, methods: list[str]
) -> Any:
    fake_api.chat_replies = [_reply(ScriptedModel(echo_extraction))]
    config = RunConfig(name=name, dataset=evalset.dataset, split=None, methods=methods, workers=1)
    runner = EvalRunner(settings, config, http_client=fake_api.client())
    return runner, runner.run(evalset)


def _reader_prompts(fake_api: FakeAPI) -> list[str]:
    return [
        body["messages"][-1]["content"]
        for body in fake_api.chat_requests
        if body["messages"][0]["content"].startswith("You are a personal assistant")
    ]


def test_no_model_request_contains_raw_session_ids(
    eval_settings: Settings, fake_api: FakeAPI, lme_path: Path, anon_env: None
) -> None:
    """全上下文、朴素检索、L1、L2、上限参照：抽取、回答、规划、判分的请求里都没有原始 ID。"""
    methods = ["B1", "B2", "B4", "B5", "ORACLE"]
    _, summary = _run(eval_settings, fake_api, load_longmemeval(lme_path), "anon", methods)
    assert (summary.completed, summary.errors) == (5, 0)
    every = json.dumps(fake_api.chat_requests, ensure_ascii=False)
    # （规划的系统提示里有 "answer_type" 这个字段名，所以整体只查具体的原始 ID）
    assert not any(marker in every for marker in RAW_IDS)
    prompts = _reader_prompts(fake_api)
    assert len(prompts) == 5
    assert not any("answer_" in p for p in prompts)
    assert sum("· session session_" in p for p in prompts) == 3  # B2、B4、B5 的证据带会话 ID
    rows = load_rows(summary.run_dir, "results.jsonl")
    b2 = next(r for r in rows if r["method"] == "B2")
    assert {sid for sid, _, _ in b2["retrieved"]} <= set(ANON_IDS)
    assert b2["retrieval"]["session_recall_all"] == 1.0  # 小历史全部召回：指标按改写后的 ID 对上

    # 消融：原始 ID 时证据会话的 ID 出现在回答模型的上下文里
    fake_api.chat_requests.clear()
    raw = load_longmemeval(lme_path, raw_ids=True)
    _run(eval_settings, fake_api, raw, "raw", ["B2"])
    assert any("· session answer_280352e9_1" in p for p in _reader_prompts(fake_api))


# ---------- 持久化：两种写法不混用 ----------


def _space_sessions(directory: Path, space_id: str = "longmemeval_s:lk-1") -> set[str]:
    db = MemoryDB(directory / "memory.sqlite")
    try:
        return set(db.get_sessions(space_id))
    finally:
        db.close()


def _space_meta(directory: Path) -> str | None:
    db = MemoryDB(directory / "memory.sqlite")
    try:
        return db.get_meta("session_ids")
    finally:
        db.close()


def test_spaces_with_different_id_schemes_are_kept_apart(
    eval_settings: Settings, fake_api: FakeAPI, lme_path: Path
) -> None:
    anon = load_longmemeval(lme_path, raw_ids=False)
    raw = load_longmemeval(lme_path, raw_ids=True)
    runner, _ = _run(eval_settings, fake_api, anon, "anon", ["B2"])
    anon_dir = space_dir(eval_settings, "longmemeval_s", runner.embedder, "lk-1", "anon")
    raw_dir = space_dir(eval_settings, "longmemeval_s", runner.embedder, "lk-1", "raw")
    assert anon_dir.parent.parent.name == "longmemeval_s-anon"
    assert raw_dir.parent.parent.name == "longmemeval_s"  # 原始 ID 沿用改写之前的位置
    assert _space_sessions(anon_dir) == set(ANON_IDS) and _space_meta(anon_dir) == "anon"
    assert not raw_dir.exists()

    _run(eval_settings, fake_api, raw, "raw", ["B2"])
    assert _space_sessions(raw_dir) == set(RAW_IDS) and _space_meta(raw_dir) == "raw"
    assert _space_sessions(anon_dir) == set(ANON_IDS)  # 互不影响

    # 放错了地方的空间（原始 ID 的空间出现在改写后的位置）：按记下的写法认出来，重建
    shutil.rmtree(anon_dir)
    shutil.copytree(raw_dir, anon_dir)
    _run(eval_settings, fake_api, anon, "anon-again", ["B2"])
    assert _space_sessions(anon_dir) == set(ANON_IDS) and _space_meta(anon_dir) == "anon"

    # 改写之前建的空间没有记下写法：原始 ID 的运行打开时记为 raw，照常复用
    db = MemoryDB(raw_dir / "memory.sqlite")
    db._conn.execute("DELETE FROM meta WHERE key = 'session_ids'")
    db._conn.commit()
    db.close()
    _, again = _run(eval_settings, fake_api, raw, "raw-legacy", ["B2"])
    assert all(r.get("reused") for r in load_rows(again.run_dir, "ingest.jsonl"))
    assert _space_meta(raw_dir) == "raw"


def test_resume_refuses_to_mix_id_schemes(
    eval_settings: Settings, fake_api: FakeAPI, lme_path: Path
) -> None:
    runner, summary = _run(
        eval_settings, fake_api, load_longmemeval(lme_path, raw_ids=False), "same", ["B0"]
    )
    info = json.loads((summary.run_dir / "run.json").read_text("utf-8"))
    assert info["session_ids"] == "anon"
    with pytest.raises(ConfigMismatchError, match="会话 ID"):
        _run(eval_settings, fake_api, load_longmemeval(lme_path, raw_ids=True), "same", ["B0"])
    # 改写之前的运行没有记下写法：LongMemEval 的按原始 ID 算，LoCoMo 的不受影响
    legacy = {k: v for k, v in info.items() if k != "session_ids"}
    with pytest.raises(ConfigMismatchError, match="会话 ID"):
        _check_resume_compatible(legacy, eval_settings, runner.config, "anon")
    _check_resume_compatible(legacy, eval_settings, runner.config, "raw")
    locomo = {**legacy, "config": {**legacy["config"], "dataset": "locomo"}}
    config = RunConfig(name="l", dataset="locomo", split=None, methods=["B0"])
    _check_resume_compatible(locomo, eval_settings, config, "native")


def test_mem0_stores_are_keyed_by_id_scheme(tmp_path: Path, lme_path: Path) -> None:
    """M0 的记忆元数据里存着会话 ID：改写过的写在 longmemeval_s-anon_<题号> 下并记下写法；
    改写之前的写入（没记写法）只给原始 ID 的运行复用；LoCoMo 的位置与记录不变。
    （不装 mem0ai：写入与检索换成替身，只测 Mem0Baseline 自己的流程。）"""
    from types import SimpleNamespace

    from cuemem.evals.mem0_baseline import Mem0Baseline
    from cuemem.evals.methods import MethodContext

    built: list[tuple[str, list[str]]] = []

    class Stub(Mem0Baseline):
        def _setup(self, llm: Any) -> None:
            self.root = tmp_path / "mem0"

        def _build(self, key: str, sessions: Any) -> dict[str, Any]:
            (self.root / key).mkdir(parents=True)  # 真的写入时由 Mem0 的存储建目录
            built.append((key, [s.session_id for s in sessions]))
            return {"memories": 0, "spent": 0.0}

        def _memory(self, key: str) -> Any:
            return SimpleNamespace(search=lambda *a, **k: {"results": []})

    reader = SimpleNamespace(
        llm=None, answer=lambda *a, **k: SimpleNamespace(answer="ok", evidence_sufficient=False)
    )
    method = Stub()

    def ask(evalset: EvalSet, scheme: Any = None) -> Any:
        sample = evalset.samples[0]
        args = [sample, evalset.haystacks["lk-1"], reader, None, "longmemeval_s:lk-1"]
        ctx = MethodContext(*args, scheme) if scheme else MethodContext(*args)
        return method.answer(ctx)

    anon = load_longmemeval(lme_path, raw_ids=False)
    raw = load_longmemeval(lme_path, raw_ids=True)
    ask(raw)  # 改写之前的写入：与原始 ID 同一个位置，没有记写法
    legacy = tmp_path / "mem0" / "longmemeval_s_lk-1" / "build.json"
    assert "session_ids" not in json.loads(legacy.read_text("utf-8"))
    assert "mem0_build" not in ask(raw, "raw").details  # 原始 ID 的运行复用
    assert "mem0_build" in ask(anon, "anon").details  # 改写后的运行另写一份
    assert "mem0_build" not in ask(anon, "anon").details
    record = tmp_path / "mem0" / "longmemeval_s-anon_lk-1" / "build.json"
    assert json.loads(record.read_text("utf-8"))["session_ids"] == "anon"
    assert built == [
        ("longmemeval_s_lk-1", RAW_IDS),
        ("longmemeval_s-anon_lk-1", ANON_IDS),
    ]
    record.write_text(json.dumps({"memories": 0}), "utf-8")  # 记的写法对不上：重建
    assert "mem0_build" in ask(anon, "anon").details and len(built) == 3


def test_build_records_are_matched_by_id_scheme() -> None:
    assert build_matches_session_ids({}, "native")  # LoCoMo：不核对
    assert build_matches_session_ids({}, "raw")  # 改写之前的写入都是原始 ID
    assert not build_matches_session_ids({}, "anon")
    assert build_matches_session_ids({"session_ids": "anon"}, "anon")
    assert not build_matches_session_ids({"session_ids": "anon"}, "raw")
    assert storage_space_id("longmemeval_s:q1_abs", "anon") == "longmemeval_s-anon:q1_abs"
    assert storage_space_id("longmemeval_s:q1_abs", "raw") == "longmemeval_s:q1_abs"
    assert storage_space_id("memlens_32k:q_1", "anon") == "memlens_32k-anon:q_1"


# ---------- LoCoMo 逐字节不变 ----------

# 基线提交 e7064ef 上同样跑一遍（B1、B2、B4，单线程）得到的全部请求消息的哈希：抽取、对账、
# 回答与判分的提示都与改写之前逐字相同，已有的响应缓存与评测空间照常命中。
# 有意修改提示词时（不是会话 ID 的事）重新生成这个快照。
LOCOMO_REQUESTS = "5766d428fa04c25a7cf5f755264da6f0f8b78925c898c7f24c96c1a888ce0d4d"
LOCOMO_B2_PROMPT = (
    "Current date: 2023-05-20 (Sat) 10:00\n\nRetrieved memories:\n"
    "[m1] 2023-05-08 (Mon) 13:56 · session session_1\nAlice: I adopted a puppy named Max.\n\n"
    "[m2] 2023-05-08 (Mon) 13:56 · session session_1\nBob: Cute! What breed?\n\n"
    "[m3] 2023-05-08 (Mon) 13:56 · session session_1\n"
    "Alice: A golden retriever. [shares a photo: a photo of a golden retriever puppy]\n\n"
    "[m4] 2023-05-20 (Sat) 10:00 · session session_2\nAlice: Wow, congrats!\n\n"
    "[m5] 2023-05-20 (Sat) 10:00 · session session_2\nBob: I ran a marathon yesterday.\n\n"
    "Question: What breed is Alice's puppy?"
)


def _locomo_view(evalset: EvalSet) -> Any:
    return (
        evalset.session_ids,
        {
            key: [(s.session_id, s.started_at, [t.has_answer for t in s.turns]) for s in v]
            for key, v in evalset.haystacks.items()
        },
        [(s.sample_id, s.evidence_session_ids, s.meta) for s in evalset.samples],
    )


@pytest.mark.parametrize("raw", ["", "1"])
def test_locomo_is_byte_for_byte_unchanged(
    eval_settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """会话 ID 本来就是 session_<n>：不改写、不受开关影响；空间位置、提示、运行记录都不变。"""
    monkeypatch.setenv(RAW_SESSION_IDS_ENV, raw)
    evalset = load_locomo(FIXTURES / "locomo_mini.json")
    view = _locomo_view(evalset)
    assert view[0] == "native" and session_id_scheme("locomo") == "native"
    assert [s.session_id for s in evalset.haystacks["conv-test"]] == ["session_1", "session_2"]
    assert all(s.source_id == "" for s in evalset.haystacks["conv-test"])
    assert view == _locomo_view(load("locomo", FIXTURES / "locomo_mini.json", raw_ids=True))
    assert storage_name("locomo") == "locomo"
    assert storage_space_id("locomo:conv-42", "native") == "locomo:conv-42"

    runner, summary = _run(eval_settings, fake_api, evalset, "locomo", ["B1", "B2", "B4"])
    assert summary.errors == 0
    expected = eval_settings.paths.eval_spaces / "locomo" / "hash-256-256" / "conv-test"
    assert space_dir(eval_settings, "locomo", runner.embedder, "conv-test") == expected
    assert expected.exists() and _space_meta(expected) is None  # LoCoMo 的空间不记写法
    info = json.loads((summary.run_dir / "run.json").read_text("utf-8"))
    assert "session_ids" not in info

    digests = sorted(
        hashlib.sha256(
            json.dumps(body["messages"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        for body in fake_api.chat_requests
    )
    assert len(digests) == 18
    assert hashlib.sha256("\n".join(digests).encode()).hexdigest() == LOCOMO_REQUESTS
    assert LOCOMO_B2_PROMPT in _reader_prompts(fake_api)
