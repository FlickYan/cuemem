"""向量库按数据库补齐（cuemem reindex、doctor --vectors，第五轮体检 P21）：向量库丢了、坏了，或者
与数据库不是同一时刻的备份时，检索悄悄少了结果。全部离线：哈希向量 + 脚本化模型。"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from PIL import Image
from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword
from typer.testing import CliRunner

from cuemem.cli import app
from cuemem.config import Settings, load_settings
from cuemem.core.models import Conversation, Fact, MemoryUnit, SessionIn, TurnIn
from cuemem.core.timeutil import to_epoch
from cuemem.memory import MemorySystem
from cuemem.store.vector import IMAGE_SUFFIX, VectorStore
from cuemem.write.consolidate import unit_payload
from cuemem.write.online import OnlineWriter

runner = CliRunner()
QUERIES = ["用户在哪里工作？", "笔记本电脑的预算是多少？", "Max 喜欢什么？"]


def _flat(text: str) -> str:
    return "".join(text.split())


def points(vectors: VectorStore, space_id: str) -> dict[str, dict[str, Any]]:
    """这个空间两个集合里的全部点：ID → 集合、payload（键的顺序也比）与向量。

    sparse 的权重按 float32 比：向量缓存存的是 float32，第一次向量化（没命中缓存）时写进向量库的
    是算出来的 float64，之后任何一次写入（包括 reindex）命中缓存，写的都是缓存里的那份。dense
    本来就是 float32，逐位比。"""
    out: dict[str, dict[str, Any]] = {}
    name = vectors.collection_name(space_id)
    for collection in (name, name + IMAGE_SUFFIX):
        if not vectors._client.collection_exists(collection):
            continue
        records, _ = vectors._client.scroll(
            collection, limit=10_000, with_payload=True, with_vectors=True
        )
        for r in records:
            assert r.payload is not None and isinstance(r.vector, dict)
            out[str(r.id)] = {
                "collection": collection,
                "payload": list(r.payload.items()),
                "vector": {k: _vector(v) for k, v in sorted(r.vector.items())},
            }
    return out


def _vector(value: Any) -> Any:
    if isinstance(value, list):
        return value
    return (list(value.indices), np.asarray(value.values, dtype=np.float32).tolist())


def wipe(system: MemorySystem, space_id: str, *, kinds: list[str] | None = None) -> None:
    """模拟向量库丢了：kinds 为空时删掉这个空间的两个集合（文字与图片），否则只删这几种点。"""
    if kinds is None:
        system.vectors.delete_space(space_id)
    else:
        system.vectors.delete(space_id, system.vectors.unit_ids(space_id, kinds=kinds))


def results(system: MemorySystem, space_id: str) -> tuple[list[Any], list[float]]:
    """检索结果（ID、路线与顺序）与分数：分数只差 sparse 权重的 float32 舍入（见 points）。"""
    ids: list[Any] = []
    scores: list[float] = []
    for query in QUERIES:
        hits = system.search(space_id, query)
        ids.append([(h.unit.id, h.route) for h in hits])
        scores.extend(h.score for h in hits)
        view = system.recall(space_id, query, current_only=True)
        ids.append([f.fact.id for f in view.facts])
        scores.extend(f.score for f in view.facts)
    return ids, scores


@pytest.fixture
def toy(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[MemorySystem]:
    fake_api.chat_replies = [
        ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    ]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    system.ingest(Conversation.model_validate_json(TOY_CONVERSATION.read_text("utf-8")))
    assert not system.consolidate("demo").failed
    yield system
    system.close()


def test_unit_payload_keeps_each_write_path_shape() -> None:
    """在线写入与巩固以前各写一份 payload、键的顺序各不相同：合成 unit_payload 之后一点都不变
    （键的顺序进了本地向量库的 pickle）。"""
    ts = datetime(2025, 3, 1, 9, 30)
    epoch = to_epoch(ts)

    def unit(kind: str) -> MemoryUnit:
        return MemoryUnit(
            id=f"u-{kind}",
            space_id="s",
            kind=kind,
            session_id="s1",
            text="t",
            ts=ts,
            source_turn_ids=[],
        )

    for kind in ("round", "image", "image_iso"):
        assert list(unit_payload(unit(kind)).items()) == [
            ("kind", kind),
            ("session_id", "s1"),
            ("ts", epoch),
        ]
    for kind in ("summary", "entity"):
        assert list(unit_payload(unit(kind)).items()) == [
            ("session_id", "s1"),
            ("ts", epoch),
            ("kind", kind),
        ]
    fact = Fact(
        id="u-fact",
        space_id="s",
        subject_id="e1",
        subject="User",
        predicate="said",
        statement="t",
        object="x",
        recorded_at=ts,
        session_id="s1",
        source_turn_ids=[],
        invalidated_at=ts,
    )
    assert list(unit_payload(unit("fact"), fact).items()) == [
        ("session_id", "s1"),
        ("ts", epoch),
        ("kind", "fact"),
        ("current", False),
        ("subject_id", "e1"),
    ]
    with pytest.raises(ValueError, match="u-fact"):
        unit_payload(unit("fact"))


def test_reindex_restores_every_point_and_search(toy: MemorySystem, fake_api: FakeAPI) -> None:
    """向量全丢了：get_all 照样列得出，检索却悄悄是空的。reindex 按数据库补回来：每个点的
    payload（包括被取代的事实的 current=False）、向量与检索结果都与丢之前相同，不调用 LLM。"""
    before = points(toy.vectors, "demo")
    searched = results(toy, "demo")
    assert any(("current", False) in p["payload"] for p in before.values())
    chat, embed = len(fake_api.chat_requests), len(fake_api.embed_requests)

    wipe(toy, "demo")
    assert toy.vectors.count("demo") == 0 and toy.search("demo", QUERIES[0]) == []
    report = toy.reindex("demo")

    assert points(toy.vectors, "demo") == before
    ids, scores = results(toy, "demo")
    assert ids == searched[0] and scores == pytest.approx(searched[1], rel=1e-6)
    counts = toy.db.unit_counts("demo")
    assert {k: report[k] for k in counts} == counts and report["orphans"] == 0
    for kind in counts:
        expected = {u.id for u in toy.db.list_units("demo", kinds=[kind])}
        assert set(toy.vectors.unit_ids("demo", kinds=[kind])) == expected
    assert (len(fake_api.chat_requests), len(fake_api.embed_requests)) == (chat, embed)
    # 什么都不缺时什么都不写；--full 全部重写一遍，结果相同
    assert toy.reindex("demo") == {
        "orphans": 0,
        "missing_files": 0,
        "images": 0,
        "image_orphans": 0,
    }
    assert sum(toy.reindex("demo", only_missing=False)[k] for k in counts) == sum(counts.values())
    # 覆盖已有的点时 qdrant-client 的本地模式按 float64 归一化（新点按 float32），dense 可能差
    # 最后一位：只比 payload
    rewritten = points(toy.vectors, "demo")
    assert {k: p["payload"] for k, p in rewritten.items()} == {
        k: p["payload"] for k, p in before.items()
    }


def test_rebuild_leaves_rounds_missing_reindex_restores_them(toy: MemorySystem) -> None:
    """consolidate(rebuild=True) 只重建派生记忆："轮"的向量丢了补不回来（数据库里有记录的"轮"
    在线写入不再写）。reindex 补上。"""
    rounds = set(toy.vectors.unit_ids("demo", kinds=["round"]))
    assert rounds
    wipe(toy, "demo", kinds=["round"])
    toy.consolidate("demo", rebuild=True)
    assert toy.vectors.unit_ids("demo", kinds=["round"]) == []
    report = toy.reindex("demo")
    assert report["round"] == len(rounds) and "fact" not in report
    assert set(toy.vectors.unit_ids("demo", kinds=["round"])) == rounds


def test_reindex_deletes_orphan_points(toy: MemorySystem) -> None:
    """向量库里有、数据库里没有的点（比如与数据库不是同一时刻的备份）删掉。"""
    [unit] = toy.db.list_units("demo", kinds=["summary"])[:1]
    stray = unit.model_copy(update={"id": "stray-unit"})
    batch = toy.embedder.embed([stray.text], "document")
    toy.vectors.upsert("demo", [stray.id], batch.dense, batch.sparse, [unit_payload(stray)])
    assert "stray-unit" in toy.vectors.unit_ids("demo")
    assert toy.reindex("demo")["orphans"] == 1
    assert "stray-unit" not in toy.vectors.unit_ids("demo")


def test_reindex_skips_fact_units_without_a_fact_row(toy: MemorySystem) -> None:
    """事实单元在事实表里找不到时补不上（payload 的 current 与 subject_id 不能猜），报
    facts_missing；别的照常补上。"""
    facts = toy.db.list_units("demo", kinds=["fact"])
    assert len(facts) > 1
    with toy.db.transaction():
        toy.db._conn.execute("DELETE FROM facts WHERE id = ?", (facts[0].id,))
    wipe(toy, "demo", kinds=["fact"])
    report = toy.reindex("demo")
    assert report["facts_missing"] == 1 and report["fact"] == len(facts) - 1
    assert set(toy.vectors.unit_ids("demo", kinds=["fact"])) == {u.id for u in facts[1:]}


def test_reindex_keeps_points_of_a_write_in_flight(
    toy: MemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """在线写入不拿空间的锁、先写向量后提交：写了向量、还没提交时 reindex 不能把它们当作多余的
    删掉，否则提交之后这些"轮"没有向量，检索永远找不到。"""
    stage = OnlineWriter._stage
    reports: list[dict[str, int]] = []

    def stage_then_reindex(self: OnlineWriter, *args: Any, **kwargs: Any) -> None:
        stage(self, *args, **kwargs)
        reports.append(toy.reindex("demo"))

    monkeypatch.setattr(OnlineWriter, "_stage", stage_then_reindex)
    session = SessionIn(
        session_id="late",
        started_at=datetime(2025, 10, 1, 9),
        turns=[TurnIn(role="user", text="我开始学大提琴了。")],
    )
    toy.append("demo", session, 0)
    assert reports == [{"orphans": 0, "missing_files": 0, "images": 0, "image_orphans": 0}]
    [late] = toy.db.list_units("demo", kinds=["round"], session_ids=["late"])
    assert late.id in toy.vectors.unit_ids("demo")
    assert toy.writer.in_flight("demo") == set()


def _image(path: Path, color: str) -> Path:
    Image.new("RGB", (64, 48), color).save(path)
    return path


@pytest.fixture
def pics(offline_settings: Settings, fake_api: FakeAPI, tmp_path: Path) -> Iterator[MemorySystem]:
    fake_api.chat_replies = [
        ScriptedModel(lambda prompt: {"summary": "photos", "facts": [], "entities": []})
    ]
    system = MemorySystem.from_settings(offline_settings, http_client=fake_api.client())
    red, blue = _image(tmp_path / "red.png", "red"), _image(tmp_path / "blue.png", "blue")
    conv = Conversation(
        space_id="pics",
        sessions=[
            SessionIn(
                session_id="s1",
                started_at=datetime(2024, 5, 1, 10),
                turns=[
                    TurnIn(role="user", text="Here is the sign <image> I saw", images=[str(red)]),
                    TurnIn(role="assistant", text="Nice sign!"),
                    TurnIn(role="user", text="And my car", images=[str(blue)]),
                ],
            )
        ],
    )
    assert system.ingest(conv, consolidate=True).images_added == 2
    yield system
    system.close()


def test_reindex_restores_image_vectors(pics: MemorySystem) -> None:
    """图片向量（另一个集合，payload 是 turn_id 与 ts）与图片描述单元（image、image_iso）同样
    补回来；多余的图片点删掉。"""
    before = points(pics.vectors, "pics")
    assert sum(p["collection"].endswith(IMAGE_SUFFIX) for p in before.values()) == 2
    assert pics.db.unit_counts("pics").get("image") == 2
    wipe(pics, "pics")
    report = pics.reindex("pics")
    assert report["images"] == 2 and report["image"] == 2 and report["image_iso"] == 2
    assert points(pics.vectors, "pics") == before

    assert pics.image_embedder is not None
    stray = pics.image_embedder.embed_images([Path(pics.db.get_media("pics")[0].path)])
    pics.vectors.upsert_images("pics", ["stray-image"], stray, [{"turn_id": "x", "ts": 0.0}])
    assert pics.reindex("pics")["image_orphans"] == 1
    assert points(pics.vectors, "pics") == before


# ---------- 命令行 ----------


@pytest.fixture
def cli_env(clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("db", "qdrant", "llm_cache", "embed_cache", "ledger", "traces", "media"):
        monkeypatch.setenv(f"CUEMEM_PATHS__{name.upper()}", str(tmp_path / name))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    assert runner.invoke(app, ["ingest", str(TOY_CONVERSATION)]).exit_code == 0


def test_doctor_vectors_and_reindex_cli(cli_env: None) -> None:
    """doctor --vectors 逐个空间比点数：一致时只有一行，对不上时写明空间与 reindex 的命令；
    reindex --space / --all 补齐。"""
    result = runner.invoke(app, ["doctor", "--vectors"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "1个空间的点数与数据库一致" in flat and "reindex" not in flat

    system = MemorySystem.from_settings(load_settings())
    rounds = len(system.vectors.unit_ids("demo"))
    wipe(system, "demo")
    system.close()

    result = runner.invoke(app, ["doctor", "--vectors"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert f"空间demo的向量库与数据库对不上（数据库→向量库：round{rounds}→0）" in flat
    assert "reindex--spacedemo" in flat and "reindex--all" in flat

    assert runner.invoke(app, ["reindex"]).exit_code == 2
    assert runner.invoke(app, ["reindex", "--space", "demo", "--all"]).exit_code == 2
    result = runner.invoke(app, ["reindex", "--space", "nobody"])
    assert result.exit_code == 1 and "没有记忆空间nobody" in _flat(result.output)
    result = runner.invoke(app, ["reindex", "--space", "demo"])
    assert result.exit_code == 0, result.output
    assert f"记忆空间demo：补上round{rounds}；删掉多余的点0个" in _flat(result.output)
    result = runner.invoke(app, ["reindex", "--all"])
    assert result.exit_code == 0
    assert "记忆空间demo：没有缺的；删掉多余的点0个" in _flat(result.output)
    assert "一致" in _flat(runner.invoke(app, ["doctor", "--vectors"]).output)


def test_doctor_vectors_skips_while_the_store_is_in_use(cli_env: None) -> None:
    """本地向量库同一时间只能一个进程打开：serve 开着时 doctor --vectors 跳过、写明原因，
    doctor 的其余检查照常（以前的 doctor 不打开向量库，这个检查要显式加 --vectors）。"""
    system = MemorySystem.from_settings(load_settings())
    try:
        result = runner.invoke(app, ["doctor", "--vectors"])
    finally:
        system.close()
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "向量库检查跳过" in flat and "cuememserve" in flat and "剩余空间" in flat
