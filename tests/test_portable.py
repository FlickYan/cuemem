"""对标同类项目补上的三件事（N12）：抽取的补充说明、记忆空间的导出与导入、遗忘的预演。全部离线。"""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from PIL import Image
from scripted import ScriptedModel, echo_extraction
from test_cli import _flat, runner

from cuemem import Memory
from cuemem.cli import app
from cuemem.config import PathsConfig, Settings
from cuemem.core.models import Turn
from cuemem.embed.base import EmbedKind
from cuemem.errors import EmbeddingAPIError
from cuemem.portable import FORMAT, import_space
from cuemem.write.extract import INSTRUCTIONS_HEADER, Extractor

DAY = datetime(2026, 8, 1, 10, 0)


@pytest.fixture
def memory(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[Memory]:
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        yield m


def test_extraction_instructions(n3_settings: Settings, fake_api: FakeAPI) -> None:
    model = ScriptedModel(echo_extraction)
    fake_api.chat_replies = [model]
    plain = Extractor(None, instructions=None).messages  # type: ignore[arg-type]
    custom = Extractor(None, instructions="  Track shoe sizes.  ").messages  # type: ignore[arg-type]
    turns = [Turn(id="t", space_id="s", session_id="x", idx=0, role="user", text="hi", ts=DAY)]
    assert INSTRUCTIONS_HEADER not in plain(turns, DAY)[0]["content"]
    assert (
        plain(turns, DAY)[0]["content"] + f"\n\n{INSTRUCTIONS_HEADER}\nTrack shoe sizes."
        == (custom(turns, DAY)[0]["content"])
    )

    with Memory(
        n3_settings, http_client=fake_api.client(), background=False, instructions="Track sizes."
    ) as m:
        assert m.settings.consolidation.instructions == "Track sizes."
        m.add("My shoe size is EU 40", user_id="alice", wait=True)
        extraction = fake_api.chat_requests[0]["messages"][0]["content"]
        assert extraction.startswith("You build the long-term memory")
        assert f"{INSTRUCTIONS_HEADER}\nTrack sizes." in extraction
        before = m.system.consolidator.fingerprint()
    # 没有补充说明时指纹与加这个字段之前相同；加了就变（已巩固的空间会重建）
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        assert m.system.consolidator.fingerprint() != before
        assert m.system.consolidator.stale("alice")


def test_export_and_import_round_trip(
    memory: Memory, n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    photo = tmp_path / "shoe.png"
    Image.new("RGB", (12, 12), "green").save(photo)
    memory.add(
        [
            {"role": "user", "content": "I run every Sunday"},
            {"role": "assistant", "content": "Great habit!"},
            {"role": "user", "content": "My birthday is in May"},
            {"role": "assistant", "content": "Noted."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "These are my shoes"},
                    {"type": "image_url", "image_url": {"url": str(photo)}},
                ],
            },
        ],
        user_id="alice",
        run_id="s1",
        timestamp=DAY,
        wait=True,
    )
    # 遗忘删掉中间一轮：留下序号空缺
    birthday = next(
        f["id"] for f in memory.get_all(user_id="alice")["results"] if "birthday" in f["memory"]
    )
    memory.delete(birthday)
    assert [t.idx for t in memory.system.db.get_turns("alice", "s1")] == [0, 1, 4]
    # 抽取前图片换成了描述（脚本化的描述是 "a red sign"）
    remaining = {f["memory"] for f in memory.get_all(user_id="alice")["results"]}
    assert len(remaining) == 2 and "I run every Sunday" in remaining

    data = json.loads(json.dumps(memory.export(user_id="alice")))  # 能原样写成 JSON
    assert data["format"] == FORMAT and data["version"] == 1
    (session,) = data["sessions"]
    assert [t["idx"] for t in session["turns"]] == [0, 1, 4]
    assert session["turns"][2]["text"] == "These are my shoes <image>"
    assert len(session["turns"][2]["images"]) == 1
    assert {f["statement"] for f in data["derived"]["facts"]} == remaining

    # 导入到另一台"机器"（另一个数据目录）的另一个空间，重新巩固
    other = n3_settings.model_copy(
        update={"paths": PathsConfig().resolve_against(tmp_path / "other")}
    )
    with Memory(other, http_client=fake_api.client(), background=False) as fresh:
        report = fresh.load(data, user_id="bob", wait=True)
        assert report["turns_added"] == 3 and report["images_added"] == 1
        turns = fresh.system.db.get_turns("bob", "s1")
        assert [(t.idx, t.role, t.ts) for t in turns] == [
            (0, "user", DAY),
            (1, "assistant", DAY),
            (4, "user", DAY),
        ]
        (image,) = fresh.system.db.get_media("bob")
        assert image.sha256 == session["turns"][2]["images"][0]["sha256"]
        assert Path(image.path).read_bytes() == photo.read_bytes()
        assert {f["memory"] for f in fresh.get_all(user_id="bob")["results"]} == remaining
        # 重新抽取时，遗忘留下的空缺照样标出来（D-059）
        assert any("[deleted]" in str(r) for r in fake_api.chat_requests)
        # 导入是幂等的
        again = import_space(fresh.system, data, space_id="bob")
        assert again["turns_added"] == 0

    with pytest.raises(ValueError, match="Not a CueMem export"):
        memory.load({"format": "something-else"})
    with pytest.raises(KeyError):
        memory.export(user_id="nobody")


def _png(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _export(*images: dict[str, Any]) -> dict[str, Any]:
    """一个会话的导出文件：先一句纯文本，再一句带图片的发言。"""
    turns = [
        {"idx": 0, "role": "user", "text": "hi", "ts": DAY.isoformat(), "images": []},
        {"idx": 1, "role": "user", "text": "look <image>", "ts": DAY.isoformat(), "images": images},
    ]
    session = {"session_id": "s1", "started_at": DAY.isoformat(), "turns": turns}
    return {"format": FORMAT, "version": 1, "space_id": "evil", "sessions": [session]}


def _image(data: bytes, mime: str = "image/png", sha256: str | None = None) -> dict[str, Any]:
    return {"sha256": sha256, "mime": mime, "data": base64.b64encode(data).decode("ascii")}


def test_an_import_that_fails_midway_writes_nothing(
    memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """导入到一半向量接口出错：一条发言都不写（以前前面的会话已经提交了，只有一半的空间），
    重试时完整导入，每个会话一个巩固任务。"""
    sessions = [
        {
            "session_id": sid,
            "started_at": DAY.isoformat(),
            "turns": [{"idx": 0, "role": "user", "text": text, "ts": DAY.isoformat()}],
        }
        for sid, text in (("s1", "I run every Sunday"), ("s2", "My birthday is in May"))
    ]
    data = {"format": FORMAT, "version": 1, "space_id": "x", "sessions": sessions}
    system = memory.system
    embed = system.embedder.embed

    def flaky(texts: Sequence[str], kind: EmbedKind) -> Any:
        if any("birthday" in t for t in texts):
            raise EmbeddingAPIError("向量接口返回 HTTP 503", status=503)
        return embed(texts, kind)

    monkeypatch.setattr(system.embedder, "embed", flaky)
    with pytest.raises(EmbeddingAPIError):
        memory.load(data, user_id="bob")
    assert system.db.list_spaces() == [] and system.vectors.count("bob") == 0

    monkeypatch.setattr(system.embedder, "embed", embed)
    assert memory.load(data, user_id="bob")["turns_added"] == 2
    assert sorted(j["session_id"] for j in system.db.get_jobs("bob")) == ["s1", "s2"]
    assert len(system.db.list_units("bob", kinds=["round"])) == 2


def test_import_never_writes_outside_its_scratch_dir(memory: Memory, tmp_path: Path) -> None:
    """图片的 sha256 与 mime 来自导入的文件：不能拿来拼文件名（绝对路径、"../" 都能逃出临时目录，
    mime 决定后缀，能写出 .py、.sqlite）。"""
    victim = tmp_path / "victim"
    victim.mkdir()
    planted = _image(_png(), "text/x-python", sha256=str(victim / "sitecustomize"))
    with pytest.raises(ValueError, match="sha256 does not match"):
        memory.load(_export(planted))
    assert list(victim.iterdir()) == []

    escape = f"cuemem-escape-{uuid.uuid4().hex}"
    with pytest.raises(ValueError, match="sha256 does not match"):
        memory.load(_export(_image(_png(), sha256=f"../{escape}")))
    assert not (Path(tempfile.gettempdir()) / f"{escape}.png").exists()
    assert memory.system.db.list_spaces() == []


def test_import_rejects_non_images_before_writing_anything(memory: Memory) -> None:
    """不是图片的附件：整个导入报 ValueError（HTTP 是 409），一条发言也不写，也不进媒体库。"""
    media_root = memory.system.settings.paths.media
    png = _png()
    for bad in (
        _image(b"<script>alert(1)</script>", "text/html"),
        _image(b"not a png at all", "image/png"),
        _image(png[:-20] + bytes([png[-20] ^ 0xFF]) + png[-19:]),  # 校验和不对（SyntaxError）
        _image(png[: len(png) // 2]),  # 截断（OSError）
        {"mime": "image/png", "data": "@@not base64@@"},
        {"mime": "image/png"},
    ):
        with pytest.raises(ValueError, match="not a valid image"):
            memory.load(_export(bad))
    assert memory.system.db.get_turns("evil") == []
    assert not media_root.exists() or not any(p.is_file() for p in media_root.rglob("*"))


def test_import_keeps_formats_the_export_mislabels(memory: Memory, tmp_path: Path) -> None:
    """后缀按 PIL 认出的格式取，不按声明的 mime：SDK 收下的 BMP、未知后缀（导出成 image/jpeg）
    照样能导回来；PNG、JPEG 的后缀与以前相同。"""
    bmp = tmp_path / "scan.bmp"
    Image.new("RGB", (6, 6), "blue").save(bmp)
    photo = tmp_path / "shot.jpg"
    Image.new("RGB", (6, 6), "white").save(photo)
    content = [
        {"type": "text", "text": "My scans"},
        {"type": "image_url", "image_url": {"url": str(bmp)}},
        {"type": "image_url", "image_url": {"url": str(photo)}},
    ]
    memory.add([{"role": "user", "content": content}], user_id="alice", run_id="s1", infer=False)
    data = memory.export(user_id="alice", derived=False)
    data["sessions"][0]["turns"][0]["images"][0]["mime"] = "image/jpeg"  # 声明错了也没关系
    report = memory.load(data, user_id="bob")
    assert report["images_added"] == 2
    stored = sorted(Path(m.path) for m in memory.system.db.get_media("bob"))
    assert sorted(p.suffix for p in stored) == [".bmp", ".jpg"]
    assert {p.read_bytes() for p in stored} == {bmp.read_bytes(), photo.read_bytes()}


def test_forget_dry_run_deletes_nothing(memory: Memory) -> None:
    out = memory.add(
        [
            {"role": "user", "content": "I live in Berlin"},
            {"role": "assistant", "content": "Berlin is lovely."},
            {"role": "user", "content": "I like jazz"},
        ],
        user_id="alice",
        wait=True,
    )
    berlin = next(e["id"] for e in out["results"] if "Berlin" in e["memory"])
    turns_before = memory.system.db.get_turns("alice")

    plan = memory.delete(berlin, dry_run=True)
    assert plan["message"].startswith("Dry run")
    assert [t["text"] for t in plan["turns"]] == ["I live in Berlin", "Berlin is lovely."]
    assert [f["memory"] for f in plan["facts"]] == ["I live in Berlin"]
    assert memory.system.db.get_turns("alice") == turns_before
    assert memory.get(berlin) is not None


def test_cli_export_import_and_preset(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    monkeypatch.setenv("CUEMEM_PRESET", "")  # --preset 会写这个变量：登记一下，测试结束时还原

    assert runner.invoke(app, ["ingest", str(TOY_CONVERSATION)]).exit_code == 0
    out = tmp_path / "demo.json"
    result = runner.invoke(app, ["export", "--space", "demo", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "已导出5个会话、18条发言" in _flat(result.output)
    result = runner.invoke(app, ["import", str(out), "--space", "copy"])
    assert result.exit_code == 0, result.output
    assert "新增18条发言" in _flat(result.output)
    assert "copy" in runner.invoke(app, ["spaces"]).output

    result = runner.invoke(app, ["--preset", "ollama", "doctor"])
    assert result.exit_code == 0, result.output
    assert os.environ["CUEMEM_PRESET"] == "ollama"
    assert "不需要Key" in _flat(result.output) and "llama3.1-8b" in result.output
