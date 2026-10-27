"""外部基线 MA（A-MEM）：笔记格式、链接解析、检索结果到证据的转换（含检索指标）、持久化、
调用失败的处理、k 与写入缓存的协议。

不装 A-MEM 也能跑：A-MEM 只在 setup / build 里导入，这里测纯函数与替换进 A-MEM 的两个方法；
需要 A-MEM 的端到端写入与检索用 importorskip，装上评测用的版本（固定的提交与试过的 chromadb，
见 evals/baselines/amem.py）再跑：
    AMEM=git+https://github.com/agiresearch/A-mem@ceffb860f0712bbae97b184d440df62bc910ca8d
    uv run --with "agentic-memory @ $AMEM" --with chromadb==1.5.9 pytest tests/test_baseline_amem.py
它会顺带装上 sentence-transformers 与 torch（几 GB）。CI 的"外部基线"
（.github/workflows/baselines.yml）照这个命令跑。"""

from __future__ import annotations

import contextlib
import json
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from fakes import ABORT_JSON, REJECT, FakeAPI
from openai import APIConnectionError, BadRequestError, InternalServerError

from cuemem.errors import ConfigMismatchError
from cuemem.evals.baselines.amem import (
    BUILD_VERSION,
    DEFAULT_K,
    K_ENV,
    MAX_TOKENS,
    NOTES,
    AMemBaseline,
    Note,
    _consolidate_memories,
    _Controller,
    _find_related_memories,
    _Store,
    clean_analysis,
    consolidated_document,
    expand,
    fits,
    json_schema_of,
    link_targets,
    load_notes,
    merge_rounds,
    note_content,
    note_document,
    note_text,
    parse_keywords,
    round_starts,
    save_notes,
    strings,
    to_hits,
    top_k,
    when_text,
)
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.methods import build_methods
from cuemem.evals.metrics import retrieval_metrics

MAY8 = datetime(2023, 5, 8, 13, 56)
MAY25 = datetime(2023, 5, 25, 9, 5)
REQUEST = httpx2.Request("POST", "http://127.0.0.1/b/amem.build.x/v1/chat/completions")


def _locomo() -> list[EvalSession]:
    later = EvalSession(
        "session_2",
        MAY25,
        [
            EvalTurn("user", "I ran a charity race last Saturday.", "Melanie"),
            EvalTurn("user", "  ", "Caroline"),  # 空话跳过（但占着下标）
            EvalTurn("user", "Amazing! I joined a support group.", "Caroline"),
        ],
    )
    first = EvalSession(
        "session_1",
        MAY8,
        [
            EvalTurn("user", "I went to a support group yesterday.", "Caroline"),
            EvalTurn("user", "How was it?", "Melanie"),
        ],
    )
    return [later, first]  # 故意乱序：写入时按时间排


def _note(i: int, *, links: list[Any] | None = None, when: datetime = MAY8, turn: int = -1) -> Note:
    return Note(
        id=f"id-{i}",
        content=f"Speaker A says : line {i}",
        timestamp=when_text(when),
        session_id="session_1",
        session_time=when,
        turn_index=turn,
        keywords=[f"k{i}"],
        context=f"context {i}",
        tags=[f"t{i}"],
        links=links or [],
    )


def _error(kind: type[Exception], status: int = 500) -> Exception:
    if kind is APIConnectionError:
        return APIConnectionError(request=REQUEST)
    response = httpx2.Response(status, request=REQUEST)
    return kind("simulated", response=response, body=None)


def test_note_format_follows_the_paper() -> None:
    assert when_text(MAY8) == "1:56 pm on 8 May, 2023"  # LoCoMo 的 date_time 格式
    assert when_text(datetime(2023, 1, 2, 0, 5)) == "12:05 am on 2 January, 2023"
    assert when_text(MAY25) == "9:05 am on 25 May, 2023"
    turn = EvalTurn("user", " I went to a support group. ", "Caroline")
    assert note_content(turn) == "Speaker Caroline says : I went to a support group."
    assert note_content(EvalTurn("assistant", "Sure.")) == "Speaker assistant says : Sure."
    document = note_document("hi", "a chat", ["x", "y"], "t1, t2")
    assert document == "content:hi context:a chat keywords: x, y tags: t1, t2"
    assert note_document("hi", None, None, []) == "content:hi context:General keywords:  tags: "
    # 论文代码整理时的格式（memory_layer.py 的 consolidate_memories）：不带字段名，空格分隔
    document = consolidated_document("hi", "a chat", ["x", "y"], "t1, t2")
    assert document == "hi , a chat x y t1 t2"
    assert consolidated_document("hi", None, None, []) == "hi , General  "


def test_clean_analysis_keeps_usable_fields() -> None:
    raw = {"keywords": ["support group", " ", 3], "context": ["Talk", "about it"], "tags": "a, b"}
    assert clean_analysis(raw) == {
        "keywords": ["support group", "3"],
        "context": "Talk about it",
        "tags": ["a", "b"],
    }
    assert clean_analysis("not a dict") == {"keywords": [], "context": "General", "tags": []}
    assert strings(None) == [] and strings(7) == ["7"]


def test_link_targets_resolve_positions_and_ids() -> None:
    ids = {"id-0": 0, "id-1": 1, "id-2": 2, "id-3": 3}
    links = [2, "1", "memory index:3", "Memory Index: 2", "id-0", "id-9", 99, -1, True, None, {}]
    assert link_targets(links, ids, 4, own=1) == [2, 3, 0]  # 自己、重复、越界、对不上的都丢掉


def test_expand_follows_links_and_dedupes() -> None:
    notes = [_note(0), _note(1, links=["memory index:3", 0]), _note(2, links=[1]), _note(3)]
    ids = {n.id: i for i, n in enumerate(notes)}
    assert expand([1, 2, 3], notes, ids, k=3) == [(1, False), (3, True), (0, True), (2, False)]
    # 只取前 k 条，每条最多带 k 条链接
    assert expand([2, 1], notes, ids, k=1) == [(2, False), (1, True)]
    assert expand([], notes, ids, k=5) == []


def test_hits_carry_session_dates_turns_and_paper_format() -> None:
    notes = [_note(0), _note(1, when=MAY25, turn=4)]
    hits = to_hits("locomo_conv-1", notes, [(1, False), (0, True)], {1: 0.8})
    assert [h.unit.id for h in hits] == ["id-1", "id-0"]
    assert [h.route for h in hits] == ["amem", "amem_link"]
    assert hits[0].unit.ts == MAY25 and hits[0].unit.session_id == "session_1"
    assert hits[0].score == 0.8 and hits[1].score == 0.0 and hits[1].rank == 2
    # 一条笔记就是一句话：发言下标交给检索指标；旧写入（-1）不知道是哪一句，不标
    assert hits[0].unit.meta["turn_indices"] == [4] and "turn_indices" not in hits[1].unit.meta
    assert hits[0].unit.text == note_text(notes[1])
    assert note_text(notes[1]).splitlines() == [
        "memory content: Speaker A says : line 1",
        "memory context: context 1",
        "memory keywords: k1",
        "memory tags: t1",
    ]


def _chat_session() -> EvalSession:
    """LongMemEval 式的会话：用户发言与其后的助手回复是同一轮。"""
    texts = [("user", "q0"), ("assistant", "a1"), ("assistant", "a2"), ("user", "q3")]
    return EvalSession("s1", MAY8, [EvalTurn(role, text) for role, text in texts])


def test_retrieved_notes_count_for_turn_recall() -> None:
    """取到证据那句话的笔记，发言召回就是 1（以前不标下标，发言召回恒为 0）。"""
    sessions = _locomo()
    notes = [
        Note(f"n{i}", "c", "", sid, MAY8, turn_index=turn)
        for i, (sid, turn) in enumerate([("session_1", 0), ("session_2", 2), ("session_2", 0)])
    ]
    hits = to_hits("locomo_conv-1", notes, [(2, False), (1, False), (0, True)], {})
    merge_rounds(hits, sessions)  # LoCoMo 每句各成一轮：不变
    assert [h.unit.meta["turn_indices"] for h in hits] == [[0], [2], [0]]
    metrics = retrieval_metrics(
        hits,
        evidence_sessions={"session_2"},
        evidence_turns={("session_2", 2)},
        relevant_units=1,
    )
    assert metrics["turn_recall_any"] == 1.0 and metrics["turn_recall_all"] == 1.0
    assert metrics["turn_ndcg"] == pytest.approx(1 / 1.584962500721156)  # 排第 2


def test_notes_of_one_round_count_once() -> None:
    """同一轮的几句话各是一条笔记：发言下标并到排得最前的那条上，NDCG 不会超过 1。"""
    session = _chat_session()
    assert round_starts(session) == [0, 0, 0, 3]
    notes = [Note(f"n{i}", "c", "", "s1", MAY8, turn_index=i) for i in range(4)]
    hits = to_hits("lme_q1", notes, [(1, False), (3, False), (0, True), (2, True)], {})
    merge_rounds(hits, [session])
    assert [h.unit.meta["turn_indices"] for h in hits] == [[0, 1, 2], [3], [], []]
    metrics = retrieval_metrics(
        hits,
        evidence_sessions={"s1"},
        evidence_turns={("s1", 0), ("s1", 1)},  # 一轮里两句都是证据
        relevant_units=1,
    )
    assert metrics["turn_ndcg"] == 1.0 and metrics["turn_recall_all"] == 1.0
    # 只取到这一轮的用户发言、证据在助手回复里：不算命中（没取到的半轮不并进来）
    hits = to_hits("lme_q1", notes, [(0, False)], {})
    merge_rounds(hits, [session])
    assert hits[0].unit.meta["turn_indices"] == [0]


def test_parse_keywords() -> None:
    assert parse_keywords('{"keywords": "support group, Caro"}', "q") == "support group, Caro"
    assert parse_keywords('{"keywords": ["a", "b"]}', "q") == "a, b"
    assert parse_keywords("support cosmos group", "q") == "support cosmos group"  # 不是 JSON
    assert parse_keywords('{"keywords": ""}', "When?") == "When?"
    assert parse_keywords(None, "When?") == "When?"


def test_notes_round_trip(tmp_path: Path) -> None:
    notes = [_note(0, links=["memory index:1", 1], turn=3), _note(1, when=MAY25, turn=0)]
    save_notes(tmp_path / NOTES, notes)
    assert load_notes(tmp_path / NOTES) == notes
    legacy = notes[0].as_json()
    del legacy["turn_index"]  # 第一版写入没有这个字段
    assert Note.from_json(legacy).turn_index == -1


def test_find_related_memories_labels_real_positions() -> None:
    """邻居标上在记忆库里的真实序号：A-MEM 的 process_memory 按这个序号改邻居、记链接。"""

    class Retriever:
        def __init__(self) -> None:
            self.upserted: list[tuple[str, str, dict[str, Any]]] = []
            self.consolidations = 0

        def search(self, query: str, k: int = 5) -> dict[str, Any]:
            return {"ids": [["n2", "gone", "n0"][:k]], "distances": [[0.1, 0.2, 0.3][:k]]}

        def upsert(self, rows: list[tuple[str, str, dict[str, Any]]]) -> None:
            self.upserted += rows

    memories = OrderedDict(
        (
            f"n{i}",
            SimpleNamespace(
                id=f"n{i}",
                content=f"c{i}",
                timestamp="1:56 pm on 8 May, 2023",
                context=f"x{i}",
                keywords=[f"k{i}"],
                tags=[f"t{i}"],
            ),
        )
        for i in range(3)
    )
    system = SimpleNamespace(memories=memories, retriever=Retriever())
    text, indices = _find_related_memories(system, "query", k=5)
    assert indices == [2, 0]
    assert text.splitlines()[0] == (
        "memory index:2\ttalk start time:1:56 pm on 8 May, 2023\tmemory content: c2\t"
        "memory context: x2\tmemory keywords: ['k2']\tmemory tags: ['t2']"
    )
    assert _find_related_memories(SimpleNamespace(memories={}), "q") == ("", [])
    _consolidate_memories(system)
    assert [row[0] for row in system.retriever.upserted] == ["n0", "n1", "n2"]
    assert system.retriever.upserted[1][1] == "c1 , x1 k1 t1"  # 按当前的上下文、关键词与标签
    assert system.retriever.upserted[1][2] == {"timestamp": "1:56 pm on 8 May, 2023"}
    assert system.retriever.consolidations == 1


def test_store_embeds_the_add_and_consolidation_formats(tmp_path: Path) -> None:
    """写入时与整理时向量化的文本格式不同（都照论文代码）；整理按 ID 覆盖原来的向量。"""
    pytest.importorskip("chromadb")
    embedded: list[str] = []

    def embed(texts: list[str]) -> list[list[float]]:
        embedded.extend(texts)
        return [[1.0, float(len(text))] for text in texts]

    store = _Store(tmp_path / "chroma", embed)
    try:
        meta = {"context": "c", "keywords": ["k1", "k2"], "tags": ["t"], "timestamp": "x"}
        store.add_document("hi", meta, "n0")
        assert embedded == ["content:hi context:c keywords: k1, k2 tags: t"]
        system = SimpleNamespace(
            memories={
                "n0": SimpleNamespace(
                    id="n0", content="hi", context="c2", keywords=["k1"], tags=["t"], timestamp="x"
                )
            },
            retriever=store,
        )
        _consolidate_memories(system)
        assert embedded[-1] == "hi , c2 k1 t" and store.consolidations == 1
        stored = store.collection.get(ids=["n0"], include=["documents", "metadatas"])
        assert stored["documents"] == ["hi , c2 k1 t"]
        assert stored["metadatas"] == [{"timestamp": "x"}]
        assert int(store.collection.count()) == 1
    finally:
        store.close()


def _client(replies: list[Any], sent: list[dict[str, Any]]) -> Any:
    def create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content, finish = reply
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_controller_sends_amem_request_and_counts_problems() -> None:
    """请求照 A-MEM 的 OpenAIController（系统提示、temperature），回复上限放宽；
    数被截断、不是 JSON 与被拒绝（400，不重试）的回复。"""
    replies: list[Any] = [
        ('{"a": 1}', "stop"),
        ('{"a": "cut', "length"),
        _error(BadRequestError, 400),
    ]
    sent: list[dict[str, Any]] = []
    controller = _Controller(_client(replies, sent), "deepseek-flash", waits=(0.0,), resamples=0)
    schema = {"type": "json_object"}
    assert controller.get_completion("p", schema) == '{"a": 1}'
    assert sent[0]["messages"] == [
        {"role": "system", "content": "You must respond with a JSON object."},
        {"role": "user", "content": "p"},
    ]
    assert sent[0]["temperature"] == 0.7 and sent[0]["max_tokens"] == MAX_TOKENS == 4096
    assert sent[0]["response_format"] == schema and sent[0]["model"] == "deepseek-flash"
    controller.get_completion("p", schema, temperature=0.5)
    assert sent[1]["temperature"] == 0.5
    with pytest.raises(BadRequestError):
        controller.get_completion("p")
    assert "response_format" not in sent[2] and len(sent) == 3  # 400 不重试
    assert (controller.calls, controller.failed, controller.rejected) == (3, 0, 1)
    assert (controller.truncated, controller.invalid_json) == (1, 1)
    assert controller.errors == ["BadRequestError: simulated"]
    controller.check()  # 被拒绝的少数几次照 A-MEM 的做法跳过，写入继续


ANALYSIS: dict[str, Any] = {  # A-MEM 的 analyze_content 发的 response_format（没有 required）
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "context": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}
ECHO = json.dumps(ANALYSIS["json_schema"]["schema"])  # 通义 JSON 模式偶尔把 schema 原样回显
WRAPPED = '{"type": "object", "properties": {"keywords": ["a"], "context": "c", "tags": ["t"]}}'
VALID = '{"keywords": ["a"], "context": "c", "tags": ["t"], "extra": 1}'


def test_fits_requires_an_object_with_every_schema_field() -> None:
    schema = json_schema_of(ANALYSIS)
    assert schema is not None and fits(VALID, schema)  # 多出的字段无妨
    assert not fits(ECHO, schema) and not fits(WRAPPED, schema)
    assert not fits('{"keywords": ["a"], "context": "c"}', schema)  # 缺字段
    assert not fits('[{"keywords": [], "context": "", "tags": []}]', schema)  # 列表
    assert not fits('{"keywords": ["a", "b', schema) and not fits(None, schema)  # 截断
    assert fits('{"keywords": "a, b", "context": ["x"], "tags": []}', schema)  # 类型不细查
    assert json_schema_of({"type": "json_object"}) == {} and json_schema_of(None) is None
    assert fits("{}", {}) and not fits("[]", {})


def test_controller_resamples_replies_that_do_not_fit_the_schema() -> None:
    """回复不合 schema 时重新采样，每次把回复上限减 1（落到网关缓存的新键上）；最多两次，
    仍不合就照原样交给 A-MEM（它放弃这次分析或演化），不算调用失败。"""
    replies: list[Any] = [(ECHO, "stop"), (WRAPPED, "stop"), (VALID, "stop")]
    sent: list[dict[str, Any]] = []
    controller = _Controller(_client(replies, sent), "m", waits=())
    assert controller.get_completion("p", ANALYSIS) == VALID
    assert [request["max_tokens"] for request in sent] == [MAX_TOKENS, MAX_TOKENS - 1, 4094]
    assert all(request["response_format"] == ANALYSIS for request in sent)
    assert (controller.calls, controller.resampled, controller.unusable) == (1, 2, 0)
    assert controller.schema_mismatch == 2 and controller.invalid_json == 0

    cut = '{"keywords": ["a", "b'
    replies[:] = [(cut, "length"), ("[]", "stop"), (cut, "length")]
    assert controller.get_completion("p", ANALYSIS) == cut
    assert (controller.calls, controller.resampled, controller.unusable) == (2, 4, 1)
    assert (controller.truncated, controller.invalid_json, controller.schema_mismatch) == (2, 2, 3)
    controller.check()  # 不能用的回复照 A-MEM 的做法放弃，写入继续

    replies[:] = [("not json", "stop")]
    assert controller.get_completion("p") == "not json"  # 没有 response_format：不查
    assert controller.resampled == 4 and controller.invalid_json == 2


def test_controller_retries_transient_errors_then_aborts_the_build() -> None:
    """临时错误（5xx、连不上）等一会儿重试；用完重试仍失败就记进 failed，写入循环据此中止。"""
    waited: list[float] = []
    replies: list[Any] = [_error(InternalServerError), _error(APIConnectionError), ("{}", "stop")]
    sent: list[dict[str, Any]] = []
    controller = _Controller(
        _client(replies, sent), "m", waits=(30.0, 120.0, 300.0), sleep=waited.append
    )
    assert controller.get_completion("p", {"type": "json_object"}) == "{}"
    assert waited == [30.0, 120.0] and controller.retries == 2 and controller.failed == 0
    controller.check()

    # 空回复（DeepSeek 挂起很久后回空）也重试，不交给 A-MEM 当成"分析为空"
    replies[:] = [(" ", "stop"), ('{"a": 1}', "stop")]
    assert controller.get_completion("p", {"type": "json_object"}) == '{"a": 1}'
    assert controller.empty_replies == 1 and controller.retries == 3
    assert controller.invalid_json == 0 and waited[2:] == [30.0]

    replies[:] = [_error(InternalServerError)] * 4
    with pytest.raises(InternalServerError):
        controller.get_completion("p")
    assert waited[3:] == [30.0, 120.0, 300.0] and controller.failed == 1
    with pytest.raises(RuntimeError, match="整份写入作废"):
        controller.check()


def test_controller_aborts_when_most_calls_are_rejected() -> None:
    replies: list[Any] = [_error(BadRequestError, 400)] * 11 + [("{}", "stop")] * 9
    controller = _Controller(_client(replies, []), "m", waits=())
    for _ in range(20):
        with contextlib.suppress(BadRequestError):
            controller.get_completion("p")
    with pytest.raises(RuntimeError, match="一多半被拒绝"):
        controller.check()


def test_k_defaults_to_the_paper_and_can_be_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(K_ENV, raising=False)
    (method,) = build_methods(["MA"])  # 构造时不导入 A-MEM
    assert isinstance(method, AMemBaseline)
    assert method.name == "MA" and method.system == "amem"
    assert method.k == DEFAULT_K == 10 and "k=" not in method.label
    monkeypatch.setenv(K_ENV, "40")  # 论文表 8 给 GPT-4o 级模型的 k
    assert top_k() == 40
    method = AMemBaseline()
    assert method.k == 40 and method.label.endswith("k=40")
    for bad in ("0", "-5", "ten"):
        monkeypatch.setenv(K_ENV, bad)
        with pytest.raises(ValueError, match=K_ENV):
            AMemBaseline()


def test_query_keywords_pick_the_model_like_the_gateway(
    offline_settings: Any, fake_api: FakeAPI
) -> None:
    """检索的关键词调用在回答线程里发（用量进这道题），选模型与网关相同：JSON 生成失控时换备用
    模型；备用模型也被内容审核拒绝时用问题原文检索。不经网关的桶。"""
    from cuemem.core.trace import start_trace
    from cuemem.embed import build_embedder
    from cuemem.evals.gateway import Gateway
    from cuemem.evals.runner import usage_from_trace
    from cuemem.llm import Ledger, LLMClient, ResponseCache

    llm = LLMClient(
        offline_settings,
        cache=ResponseCache(None),
        ledger=Ledger(None),
        http_client=fake_api.client(),
    )
    baseline = AMemBaseline()
    baseline.llm = llm
    baseline.gateway = Gateway(llm, build_embedder(offline_settings, use_cache=False))
    question = "When did Caroline go to the support group?"
    fake_api.chat_replies = [lambda body: '{"keywords": "Caroline, support group"}']
    with start_trace("answer") as trace:
        assert baseline._keywords("c1", question) == ("Caroline, support group", "qwen3.8-flash")
    assert usage_from_trace(trace)["llm_calls"] == 1
    sent = fake_api.chat_requests[-1]
    assert sent["temperature"] == 0.7 and sent["max_tokens"] == MAX_TOKENS
    assert sent["response_format"] == {"type": "json_object"}
    fake_api.chat_replies = [
        lambda body: ABORT_JSON if body["model"] == "qwen3.8-flash" else '{"keywords": "a, b"}'
    ]
    # 换个问题：同样的请求会命中缓存
    assert baseline._keywords("c1", "Who?") == ("a, b", "deepseek-flash")  # 备用模型
    # 回显 schema：重新采样（上限减 1，落到缓存的新键上）
    echo = json.dumps({"type": "object", "properties": {"keywords": {"type": "string"}}})
    fake_api.chat_replies = [
        lambda body: echo if body["max_tokens"] == MAX_TOKENS else '{"keywords": "falls"}'
    ]
    assert baseline._keywords("c1", "Where did she hike?") == ("falls", "qwen3.8-flash")
    assert [r["max_tokens"] for r in fake_api.chat_requests[-2:]] == [MAX_TOKENS, MAX_TOKENS - 1]
    fake_api.chat_replies = [lambda body: REJECT]
    assert baseline._keywords("c1", "Where?") == ("Where?", "")
    assert baseline.gateway.usage(baseline.bucket("c1", "search"))["chat_calls"] == 0


def test_build_of_another_protocol_is_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """换了抽取模型或向量模型（pds → qwen）：不删另一个协议的写入，报错让换目录；
    只是 A-MEM 的写入参数变了（旧版写入）才重建。早先写入标记接在 protocol 后面，照样认得。"""
    pds = "deepseek-flash|BAAI/bge-small-en-v1.5:384"
    legacy = f"{pds}|amem@ceffb86|max_tokens={MAX_TOKENS}|v{BUILD_VERSION}"  # 现存写入的写法
    baseline = AMemBaseline()
    assert f"{pds}|{baseline.build_tag}" == legacy
    baseline.root, baseline.protocol = tmp_path, pds
    baseline.gateway = SimpleNamespace(usage=lambda bucket: {})  # type: ignore[assignment]
    built: list[str] = []
    monkeypatch.setattr(baseline, "build", lambda key, directory, sessions: built.append(key) or {})
    key = "locomo_conv-1"
    directory = tmp_path / key
    directory.mkdir()
    record = directory / "build.json"
    record.write_text(json.dumps({"protocol": legacy}), "utf-8")
    assert baseline._ensure_built(key, directory, _locomo()) is None and built == []
    baseline.protocol = "qwen3.8-flash|qwen:1024"
    (directory / NOTES).write_text("[]", "utf-8")
    with pytest.raises(ConfigMismatchError, match="CUEMEM_PATHS__EVAL_SPACES"):
        baseline._ensure_built(key, directory, _locomo())
    assert json.loads(record.read_text("utf-8")) == {"protocol": legacy}
    assert (directory / NOTES).exists() and built == []
    # 同一协议的旧版写入（回复上限 2000、v1）：重建，记下换掉的是哪一版
    baseline.protocol = pds
    record.write_text(json.dumps({"protocol": f"{pds}|amem@ceffb86|max_tokens=2000|v1"}), "utf-8")
    stats = baseline._ensure_built(key, directory, _locomo())
    assert stats is not None and built == [key]
    assert stats["protocol"] == pds and stats["build_tag"] == baseline.build_tag
    assert stats["replaced"] == {"build_tag": "amem@ceffb86|max_tokens=2000|v1"}
    assert baseline._ensure_built(key, directory, _locomo()) is None  # 新记录照常复用


# ---------- 装了 A-MEM 时：经网关端到端写入与检索 ----------


def _fake_model(body: dict[str, Any]) -> str:
    """假的抽取模型：按提示回 A-MEM 的笔记分析、演化决定，或检索用的关键词。"""
    prompt = body["messages"][-1]["content"]
    if "Generate a structured analysis" in prompt:
        if "How was it?" in prompt and body.get("max_tokens") == MAX_TOKENS:  # 回显 schema
            return json.dumps({"type": "object", "properties": {"keywords": {"type": "array"}}})
        content = prompt.rsplit("Content for analysis:", 1)[-1]
        keyword = "support group" if "support group" in content else "chat"
        return json.dumps({"keywords": [keyword], "context": "A talk.", "tags": ["life"]})
    if "memory evolution agent" in prompt:
        neighbors = re.findall(r"memory index:(\d+)", prompt)
        return json.dumps(
            {
                "should_evolve": True,
                "actions": ["strengthen", "update_neighbor"],
                "suggested_connections": [f"memory index:{neighbors[0]}"],
                "tags_to_update": ["linked"],
                "new_context_neighborhood": ["Updated context."] * len(neighbors),
                "new_tags_neighborhood": [["evolved"]] * len(neighbors),
            }
        )
    if "generate several keywords" in prompt:
        return json.dumps({"keywords": "support group"})
    return "{}"


def _baseline(offline_settings: Any, fake_api: Any) -> AMemBaseline:
    from cuemem.llm import Ledger, LLMClient, ResponseCache

    llm = LLMClient(
        offline_settings,
        cache=ResponseCache(None),
        ledger=Ledger(None),
        http_client=fake_api.client(),
    )
    baseline = AMemBaseline()
    baseline._setup(llm)
    return baseline


def test_build_and_retrieve_through_the_gateway(
    offline_settings: Any, fake_api: Any, tmp_path: Path
) -> None:
    """装了 A-MEM 时：每句一条笔记，分析与演化经网关调用，写完存盘；重新打开后检索带出链接。"""
    pytest.importorskip("agentic_memory.memory_system")
    pytest.importorskip("chromadb")
    from cuemem.core.trace import start_trace
    from cuemem.evals.runner import usage_from_trace

    fake_api.chat_replies = [_fake_model]
    baseline = _baseline(offline_settings, fake_api)
    assert baseline.build_tag == f"amem@ceffb86|max_tokens=4096|v{BUILD_VERSION}"
    assert baseline.protocol == f"qwen3.8-flash|{baseline.embedder.name}:{baseline.embedder.dim}"
    key, sessions = "locomo_conv-1", _locomo()
    directory = baseline.root / key
    question = "When did Caroline go to a support group?"
    sample = Sample("locomo", "q1", "conv-1", question, "", "x", "x", False, MAY25)
    try:
        built = baseline._ensure_built(key, directory, sessions)
        assert built is not None and built["memories"] == 4
        assert built["rejected_calls"] == 0 and built["invalid_json"] == 0
        assert built["truncated"] == 0 and built["max_tokens"] == MAX_TOKENS
        assert built["empty_analyses"] == 0 and built["retries"] == 0
        assert fake_api.chat_requests[0]["max_tokens"] == MAX_TOKENS  # 网关照传给上游
        assert built["amem_calls"] == 7  # 4 次分析 + 3 次演化
        # 第 2 条的分析先回显了 schema：重新采样一次（上限减 1），之后合规
        assert built["chat_calls"] == 8 and built["resampled"] == 1
        assert built["schema_mismatch"] == 1 and built["unusable_replies"] == 0
        assert built["evolved"] == 3 and built["links"] >= 3 and built["unresolved_links"] == 0
        notes = load_notes(directory / NOTES)
        assert [n.session_id for n in notes] == ["session_1"] * 2 + ["session_2"] * 2
        assert [n.turn_index for n in notes] == [0, 1, 0, 2]  # 空话占着下标
        assert notes[0].content == "Speaker Caroline says : I went to a support group yesterday."
        assert notes[0].timestamp == "1:56 pm on 8 May, 2023"
        assert notes[0].tags == ["evolved"] and notes[0].context == "Updated context."
        assert notes[3].keywords == ["support group"] and notes[3].tags == ["linked"]

        baseline.close(key)  # 模拟重新运行：从磁盘打开
        before = len(fake_api.chat_requests)
        with start_trace("answer") as trace:  # 关键词那次调用与查询向量算进这道题的用量
            hits, details = baseline.retrieve(key, directory, sample, sessions)
        assert details["amem_query"] == "support group" and details["amem_k"] == DEFAULT_K
        assert details["amem_query_model"] == "qwen3.8-flash"
        assert details["amem_retrieved"] == 4 and len(hits) == 4  # 只有 4 条，全取到
        usage = usage_from_trace(trace)
        assert usage["llm_calls"] == 1 and usage["cost"] > 0
        assert len(fake_api.chat_requests) == before + 1
        sent = fake_api.chat_requests[-1]
        assert sent["temperature"] == 0.7 and sent["response_format"] == {"type": "json_object"}
        assert sent["messages"][0]["content"].startswith("You must respond with a JSON object.")
        assert '"keywords"' in sent["messages"][0]["content"]  # schema 照网关的做法写进系统提示
        search = baseline.gateway.usage(baseline.bucket(key, "search"))
        assert search["chat_calls"] == 0 and search["embedding_calls"] == 0  # 不再经网关
        by_id = {h.unit.id: h for h in hits}
        assert by_id[notes[0].id].unit.ts == MAY8 and by_id[notes[3].id].unit.ts == MAY25
        assert by_id[notes[3].id].unit.meta["turn_indices"] == [2]
        assert by_id[notes[0].id].unit.text.startswith("memory content: Speaker Caroline says")
    finally:
        baseline.close(key)
        baseline.gateway.stop()


def test_failed_call_aborts_the_build(offline_settings: Any, fake_api: Any) -> None:
    """A-MEM 吞掉了演化调用的异常：写入中止，不留 build.json（下次运行重建）。"""
    pytest.importorskip("agentic_memory.memory_system")
    pytest.importorskip("chromadb")
    sent: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        sent.append(kwargs)
        content = _fake_model({"messages": kwargs["messages"]})
        if "memory evolution agent" in kwargs["messages"][-1]["content"]:
            raise _error(InternalServerError)
        message = SimpleNamespace(content=content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    baseline = _baseline(offline_settings, fake_api)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    baseline._controller = lambda url: _Controller(client, "m", waits=())
    key = "locomo_conv-1"
    directory = baseline.root / key
    try:
        with pytest.raises(RuntimeError, match="整份写入作废"):
            baseline._ensure_built(key, directory, _locomo())
        assert len(sent) == 3  # 第 1 条分析、第 2 条分析、第 2 条的演化（失败）后就停了
        assert not (directory / "build.json").exists() and key in baseline._failed
    finally:
        baseline.gateway.stop()
