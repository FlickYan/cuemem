"""多模态记忆（N6）：媒体库、带图写入、图片描述、图片检索键、再感知、全上下文看图、
MemLens 加载与判分、遗忘。全部离线：哈希向量 + 脚本化模型 + 临时生成的小图。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeAPI
from PIL import Image
from scripted import ScriptedModel

from cuemem.config import Settings
from cuemem.core.models import Conversation, SessionIn, TurnIn
from cuemem.evals.datasets import load_memlens
from cuemem.evals.judge import Judge, memlens_prompt, parse_answer_score
from cuemem.memory import MemorySystem
from cuemem.read.l1 import L1Config
from cuemem.read.reader import history_blocks
from cuemem.store.media import MediaStore, image_data_url
from cuemem.write.caption import fill_images, inline_images

ASKED = datetime(2024, 6, 1)


def _image(path: Path, color: str, size: tuple[int, int] = (64, 48)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture
def images(tmp_path: Path) -> dict[str, Path]:
    return {
        "red": _image(tmp_path / "red.png", "red"),
        "blue": _image(tmp_path / "blue.png", "blue"),
        "big": _image(tmp_path / "big.png", "green", size=(2000, 1000)),
    }


def test_media_store_is_content_addressed(tmp_path: Path, images: dict[str, Path]) -> None:
    store = MediaStore(tmp_path / "media")
    first = store.put(images["red"])
    copy = _image(tmp_path / "copy.png", "red")
    assert store.put(copy).path == first.path  # 同样的内容只存一份
    assert first.path.read_bytes() == images["red"].read_bytes()
    assert (first.width, first.height) == (64, 48)
    small = image_data_url(images["big"], max_side=200)
    assert small.startswith("data:image/jpeg;base64,") and len(small) < 20_000


def test_fill_images() -> None:
    assert inline_images("look <image> here", ["a dog"]) == "look [Image: a dog] here"
    assert fill_images("no marker", ["[A]"]) == "no marker [A]"
    assert fill_images("<image> and <image>", ["[A]"]) == "[A] and [Image]"


@pytest.fixture
def mm(
    offline_settings: Settings, fake_api: FakeAPI, images: dict[str, Path]
) -> Iterator[tuple[MemorySystem, ScriptedModel]]:
    model = ScriptedModel(
        lambda prompt: {
            "summary": "photos",
            "facts": [
                {
                    "statement": "The user showed a red sign that says CUE 42.",
                    "predicate": "showed",
                    "object": "red sign",
                    "turns": [0],
                    "cardinality": "multi",
                }
            ]
            if "[Image:" in prompt
            else [],
            "entities": [],
        }
    )
    fake_api.chat_replies = [model]
    system = MemorySystem.from_settings(offline_settings, http_client=fake_api.client())
    conv = Conversation(
        space_id="pics",
        sessions=[
            SessionIn(
                session_id="s1",
                started_at=datetime(2024, 5, 1, 10),
                turns=[
                    TurnIn(
                        role="user",
                        text="Here is the sign <image> I saw",
                        images=[str(images["red"])],
                    ),
                    TurnIn(role="assistant", text="Nice sign!"),
                    TurnIn(role="user", text="And my car", images=[str(images["blue"])]),
                ],
            ),
            SessionIn(
                session_id="s2",
                started_at=datetime(2024, 5, 20, 9),
                turns=[TurnIn(role="user", text="I like hiking in the mountains.")],
            ),
        ],
    )
    report = system.ingest(conv, consolidate=True)
    assert report.images_added == 2
    yield system, model
    system.close()


def test_ingest_stores_images_and_vectors(mm: tuple[MemorySystem, ScriptedModel]) -> None:
    system, _ = mm
    media = system.db.get_media("pics")
    assert len(media) == 2 and all(Path(m.path).exists() for m in media)
    turns = {t.idx: t for t in system.db.get_turns("pics", "s1")}
    assert turns[0].media_ids == [media[0].id] or turns[0].media_ids == [media[1].id]
    rounds = system.db.list_units("pics", kinds=["round"], session_ids=["s1"])
    assert sorted(len(r.meta.get("media_ids", [])) for r in rounds) == [1, 1]
    assert system.image_embedder is not None
    vector = system.image_embedder.embed_images([Path(media[0].path)])[0]
    assert system.vectors.search_images("pics", vector, 1)[0].unit_id == media[0].id


def test_consolidation_captions_images(mm: tuple[MemorySystem, ScriptedModel]) -> None:
    system, model = mm
    media = system.db.get_media("pics")
    assert all(m.caption == "a red sign" and m.ocr_text == "CUE 42" for m in media)
    assert all(m.caption_iso.startswith("an isolated red sign Text in the image") for m in media)
    kinds = {u.kind for u in system.db.list_units("pics")}
    assert {"image", "image_iso"} <= kinds
    assert model.calls["caption"] == 4  # 两张图 × 两种描述
    extract = [p for p in model.prompts["extract"] if "[Image:" in p]
    assert extract and "a red sign Text in the image: CUE 42" in extract[0]
    # 描述进入了事实（实体绑定）：抽取模型看到了图片里的内容
    assert any("CUE 42" in f.statement for f in system.db.get_facts("pics"))
    assert system.consolidate("pics").sessions == 0  # 幂等：再跑不会重新描述


def test_l1_uses_captions_and_attaches_originals(
    mm: tuple[MemorySystem, ScriptedModel],
) -> None:
    system, model = mm
    cfg = L1Config(attach_images=1, budget_tokens=None, k=1)
    result = system.l1.search("pics", "an isolated red sign CUE 42", ASKED, cfg)
    assert result.rounds[0].unit.session_id == "s1"
    assert len(result.attach) == 1 and result.attach[0] in result.media
    answer = system.ask("pics", "What did the sign say?", question_time=ASKED, l1=cfg)
    prompt = model.prompts["read"][-1]
    assert "[Image i1, see picture 1 below: a red sign Text in the image: CUE 42]" in prompt
    assert "Attached pictures" in prompt and model.images[-1] == 1
    assert answer.details["attached"] == result.attach
    # 不用描述（纯文本记忆）时只留占位；孤立描述时展示孤立描述
    system.ask("pics", "sign?", question_time=ASKED, l1=replace(cfg, image_kind=None))
    assert "[Image]" in model.prompts["read"][-1]
    system.ask("pics", "sign?", question_time=ASKED, l1=replace(cfg, image_kind="image_iso"))
    assert "an isolated red sign" in model.prompts["read"][-1]


def test_text_only_requests_stay_strings(
    mm: tuple[MemorySystem, ScriptedModel], fake_api: FakeAPI
) -> None:
    """没有附图时 user 消息仍是一个字符串：与 N6 之前的请求逐字相同，缓存照常命中。"""
    system, _ = mm
    system.ask("pics", "Where do I hike?", question_time=ASKED, l1=L1Config(image_kind=None))
    assert isinstance(fake_api.chat_requests[-1]["messages"][-1]["content"], str)


def test_visual_route_finds_images(mm: tuple[MemorySystem, ScriptedModel]) -> None:
    system, _ = mm
    cfg = L1Config(image_kind=None, visual_weight=1.0, attach_images=2)
    result = system.l1.search("pics", "car", ASKED, cfg)
    assert result.media and len(result.attach) >= 1


def test_visual_route_reads_only_the_hit_turns(
    mm: tuple[MemorySystem, ScriptedModel], monkeypatch: pytest.MonkeyPatch
) -> None:
    """以文搜图命中的图片按发言找所在的轮：只取命中图片的那几句发言。以前每次检索都载入并解析
    整个空间的原始日志（1.6 万轮时约 250 毫秒）。"""
    system, _ = mm
    cfg = L1Config(image_kind=None, visual_weight=1.0, attach_images=2)
    media = system.db.get_media("pics")
    rounds = system.db.list_units("pics", kinds=["round"])
    owner = {
        m.id: next(r.id for r in rounds if m.source_turn_id in r.source_turn_ids) for m in media
    }
    real = system.db.get_turns
    asked: list[list[str] | None] = []

    def get_turns(space_id: str, session_id: str | None = None, **kwargs: Any) -> Any:
        assert session_id is not None or kwargs.get("ids") is not None, "载入了整个原始日志"
        asked.append(kwargs.get("ids"))
        return real(space_id, session_id, **kwargs)

    monkeypatch.setattr(system.db, "get_turns", get_turns)
    cands = system.l1.candidates("pics", "car", ASKED, cfg)
    visual = cands.routes[("media", "visual")]
    assert {mid for mid, _ in visual} == set(owner)
    assert sorted(m.source_turn_id for m in media) in asked
    ranked = dict(cands.ranked)
    assert all(owner[mid] in ranked for mid in owner)  # 每张命中的图片都映射回了所在的轮
    assert set(cands.media_score) == set(owner)


def test_history_blocks_interleave_images(images: dict[str, Path]) -> None:
    session = SessionIn(
        session_id="s",
        started_at=datetime(2024, 5, 1),
        turns=[
            TurnIn(role="user", text="before <image> after", images=[str(images["red"])]),
            TurnIn(role="assistant", text="ok", images=[str(images["blue"])]),
        ],
    )
    blocks = history_blocks([session])
    kinds = [b["type"] for b in blocks]
    assert kinds == ["text", "image_url", "text", "image_url", "text"]
    assert blocks[0]["text"].endswith("User: before ") and blocks[2]["text"].startswith(" after")


def test_forget_deletes_images(mm: tuple[MemorySystem, ScriptedModel]) -> None:
    system, _ = mm
    paths = [Path(m.path) for m in system.db.get_media("pics")]
    report = system.forget("pics", session_ids=["s1"])
    assert report.images_deleted == 2 and not any(p.exists() for p in paths)
    assert system.db.get_media("pics") == []


def test_memlens_loader_and_judge(tmp_path: Path, images: dict[str, Path], fake_api: FakeAPI,
                                  settings: Settings) -> None:  # fmt: skip
    (tmp_path / "release_images" / "needle_images").mkdir(parents=True)
    target = tmp_path / "release_images" / "needle_images" / "a.png"
    target.write_bytes(images["red"].read_bytes())
    item = {
        "question_id": "q_1",
        "question_type": "multi_session_reasoning",
        "question_subtype": "arithmetic",
        "question": "How much did I spend?",
        "answer": "$10.00",
        "question_date": "2024/05/31 (Fri) 07:58",
        "haystack_dates": ["2024/05/06 (Mon) 17:17"],
        "haystack_session_ids": ["s1"],
        "answer_session_ids": ["s1"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "receipt <image>",
                    "images": [{"file": "needle_images/a.png"}],
                },
                {"role": "assistant", "content": "ok", "images": []},
            ]
        ],
    }  # fmt: skip
    path = tmp_path / "dataset_32k.json"
    path.write_text(json.dumps([item]))
    evalset = load_memlens(path)
    sample = evalset.samples[0]
    turn = evalset.haystacks["q_1"][0].turns[0]
    assert turn.images == [str(target)] and sample.meta["question_subtype"] == "arithmetic"
    assert turn.has_answer  # 证据会话里带关键图片的发言是证据发言
    # 会话 ID 按时间顺序改写成 session_<i>，证据随之改写，原 ID 只留作追溯
    assert not sample.is_abstention and sample.evidence_session_ids == ["session_1"]
    assert evalset.session_ids == "anon"
    assert evalset.haystacks["q_1"][0].session_id == "session_1"
    assert evalset.haystacks["q_1"][0].source_id == "s1"

    prompt = memlens_prompt(sample, "$10")
    assert "[MSR — Arithmetic]" in prompt and prompt.endswith("[Scoring Rationale]:")
    assert parse_answer_score('{"answer_score": 1}') == 1
    assert parse_answer_score("In summary, the student's answer deserves 0 points") == 0
    assert parse_answer_score("no verdict") == 0

    from cuemem.llm.client import LLMClient

    replies: list[Any] = ['[Scoring Rationale]: ok\n[JSON]:\n{"answer_score": 1}']
    fake_api.chat_replies = replies
    judge = Judge(LLMClient(settings, http_client=fake_api.client()))
    assert judge.judge(sample, "$10").correct
    assert not judge.judge(sample, "word " * 600).correct  # 官方：超过 500 词直接判 0
    assert len(fake_api.chat_requests) == 1


def test_search_by_image(mm: tuple[MemorySystem, ScriptedModel], images: dict[str, Path]) -> None:
    system, _ = mm
    found = system.search_by_image("pics", images["blue"], k=2)
    assert len(found) == 2
    best, score = found[0]
    assert Path(best.path).read_bytes() == images["blue"].read_bytes() and score > 0.99


def test_every_method_builds() -> None:
    """登记的每个评测方法都能构造出来（防止工厂函数写错参数，只有运行到那个方法才报错）。"""
    from cuemem.evals.methods import METHOD_NAMES, build_methods

    methods = build_methods(list(METHOD_NAMES))
    assert [m.name for m in methods] == list(METHOD_NAMES)
