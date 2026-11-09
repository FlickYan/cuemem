"""对标同类项目补上的三件事（N12）：抽取的补充说明、记忆空间的导出与导入、遗忘的预演。全部离线。"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import runpy
import tempfile
import uuid
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import typer
from fakes import PROJECT_ROOT, TOY_CONVERSATION, FakeAPI
from PIL import Image
from scripted import ScriptedModel, echo_extraction
from test_cli import _flat, runner

from cuemem import Memory
from cuemem.cli import app
from cuemem.config import PathsConfig, Settings, load_settings, presets
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
    assert data["format"] == FORMAT and data["version"] == 2
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


def test_export_v2_carries_metadata_and_v1_files_still_import(memory: Memory) -> None:
    """导出格式 version 2：发言带 SDK 写入时给的 metadata 与过期时间（没有时省略），导入后照样
    能按 metadata 筛、照样过期。version 1 的文件（这个版本之前导出的）照样能导入；附加信息的
    类型不对时整个导入报错、一条发言也不写；更新的版本拒绝导入。"""
    memory.add(
        "I like green tea",
        user_id="alice",
        run_id="s1",
        timestamp=DAY,
        metadata={"topic": "food", "tags": ["drink"]},
        expiration_date="2099-12-31",
    )
    memory.add("I like rain", user_id="alice", run_id="s1", timestamp=DAY)
    memory.add("I like chess", user_id="alice", run_id="s2", timestamp=DAY, expiration_date=DAY)
    data = json.loads(json.dumps(memory.export(user_id="alice", derived=False)))
    assert data["version"] == 2
    turns = {t["text"]: t for s in data["sessions"] for t in s["turns"]}
    assert turns["I like green tea"]["metadata"] == {"topic": "food", "tags": ["drink"]}
    assert turns["I like green tea"]["expires_at"] == "2099-12-31"
    assert turns["I like chess"]["expires_at"] == DAY.isoformat()
    assert not {"metadata", "expires_at", "edit_of"} & set(turns["I like rain"])

    memory.load(data, user_id="bob", wait=True)
    tea = memory.get_all(filters={"user_id": "bob", "tags": {"contains": "drink"}})["results"]
    assert [(f["memory"], f["expiration_date"]) for f in tea] == [
        ("I like green tea", "2099-12-31")
    ]
    assert {f["memory"] for f in memory.get_all(user_id="bob")["results"]} == {
        "I like green tea",
        "I like rain",
    }

    # 这个版本之前导出的文件（version 1，发言没有附加信息）
    report = memory.load(_export(), user_id="carol", wait=True)
    assert report["turns_added"] == 2 and memory.system.db.get_turn_meta("carol") == {}
    assert len(memory.get_all(user_id="carol")["results"]) == 2

    for field, value in (("metadata", ["not", "an", "object"]), ("expires_at", "soon")):
        bad = _export()
        bad["version"] = 2
        bad["sessions"][0]["turns"][1][field] = value
        with pytest.raises(ValueError):
            memory.load(bad, user_id="dave")
        assert memory.system.db.get_turns("dave") == []
    newer = {**_export(), "version": 3}
    with pytest.raises(ValueError, match="newer"):
        memory.load(newer, user_id="dave")


def test_infer_false_survives_export_and_import(
    memory: Memory, n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """add(infer=False) 的会话没有巩固任务：导出时带 "infer": false，导入时同样不登记，后台线程
    不去抽取它（以前导入时为每个会话登记任务，多花 LLM 请求）；显式的巩固照样抽取，同源空间。
    还开着的 run 不带这个键，导入时照常登记任务。"""
    memory.add("raw diary line", user_id="alice", run_id="log", timestamp=DAY, infer=False)
    memory.add("I like green tea", user_id="alice", run_id="chat", timestamp=DAY)
    memory.open_run("live", user_id="alice")
    memory.add("half a conversation", user_id="alice", run_id="live", timestamp=DAY)
    data = json.loads(json.dumps(memory.export(user_id="alice", derived=False)))
    flags = {s["session_id"]: s.get("infer") for s in data["sessions"]}
    assert flags == {"log": False, "chat": None, "live": None}
    assert "infer" not in next(s for s in data["sessions"] if s["session_id"] == "chat")

    other = n3_settings.model_copy(
        update={"paths": PathsConfig().resolve_against(tmp_path / "other")}
    )
    fake_api.chat_requests.clear()
    with Memory(other, http_client=fake_api.client(), background=True) as fresh:
        fresh.load(data, user_id="bob")
        fresh.wait()
        assert {j["session_id"] for j in fresh.system.db.get_jobs("bob")} == {"chat", "live"}
        assert "raw diary line" not in json.dumps(fake_api.chat_requests)
        assert "green tea" in json.dumps(fake_api.chat_requests)
        # 显式的巩固照样抽取它（同源空间）
        fresh.consolidate(user_id="bob")
        assert "raw diary line" in {f["memory"] for f in fresh.get_all(user_id="bob")["results"]}
        # 目标空间里已有任务的会话：再导入时照常登记
        log = next(s for s in data["sessions"] if s["session_id"] == "log")
        log["turns"].append({**log["turns"][0], "idx": 5, "text": "another raw line"})
        before = len(fresh.system.db.get_jobs("bob"))
        assert import_space(fresh.system, data, space_id="bob")["turns_added"] == 1
        assert len(fresh.system.db.get_jobs("bob")) == before + 1

    bad = json.loads(json.dumps(data))
    next(s for s in bad["sessions"] if s["session_id"] == "log")["infer"] = "no"
    with pytest.raises(ValueError, match="Malformed export"):
        memory.load(bad, user_id="dave")
    assert memory.system.db.get_turns("dave") == []


def test_cli_export_reads_infer_false_from_the_database_alone(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cuemem export 只打开数据库（向量库坏了也能导出）：只写了日志的会话照样带 "infer": false，
    上一个进程没结束的 run 凭数据库里的标记认出来、不带这个键；导入后只给后者与普通会话登记任务。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    settings = load_settings()
    with Memory(settings, background=False) as memory:
        memory.add("raw diary line", user_id="alice", run_id="log", timestamp=DAY, infer=False)
        memory.add("I like green tea", user_id="alice", run_id="chat", timestamp=DAY)
        memory.open_run("live", user_id="alice")  # 进程结束时还开着：数据库里留着标记
        memory.add("half a conversation", user_id="alice", run_id="live", timestamp=DAY)
    (settings.paths.qdrant / "meta.json").write_text("", "utf-8")

    out = tmp_path / "alice.json"
    result = runner.invoke(app, ["export", "--space", "alice", "-o", str(out)])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text("utf-8"))
    assert {s["session_id"]: s.get("infer") for s in data["sessions"]} == {
        "log": False,
        "chat": None,
        "live": None,
    }
    assert runner.invoke(app, ["doctor", "--repair-vectors"]).exit_code == 0
    result = runner.invoke(app, ["import", str(out), "--space", "copy"])
    assert result.exit_code == 0, result.output
    with Memory(load_settings(), background=False) as memory:
        assert {j["session_id"] for j in memory.system.db.get_jobs("copy")} == {"chat", "live"}


def test_updates_survive_export_to_the_same_space(
    memory: Memory, n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """update 写下的更正（edit_of）随发言导出：导入到另一台机器上同名的空间时，更正照样取代原来
    那条；导入到别的空间时事实 ID 对不上，那句更正按普通的话对账（这里是多值的事实：两条都在）。"""
    jazz = memory.add("I like jazz", user_id="alice", timestamp=DAY, wait=True)["results"][0]
    edit = memory.update(jazz["id"], "I like blues")
    assert edit["message"] == "Memory updated"
    data = json.loads(json.dumps(memory.export(user_id="alice", derived=False)))
    (edit_turn,) = [t for s in data["sessions"] for t in s["turns"] if "edit_of" in t]
    assert edit_turn["edit_of"] == jazz["id"] and edit_turn["text"] == "Correction: I like blues"

    other = n3_settings.model_copy(
        update={"paths": PathsConfig().resolve_against(tmp_path / "other")}
    )
    with Memory(other, http_client=fake_api.client(), background=False) as fresh:
        fresh.load(data, wait=True)
        assert [f["memory"] for f in fresh.get_all(user_id="alice")["results"]] == [
            "Correction: I like blues"
        ]
        assert [v["memory"] for v in fresh.history(jazz["id"])] == [
            "I like jazz",
            "Correction: I like blues",
        ]
        fresh.load(data, user_id="bob", wait=True)
        assert len(fresh.get_all(user_id="bob")["results"]) == 2


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
    """不是图片的附件：整个导入报 ValueError（HTTP 是 409），一条发言也不写，也不进媒体库。媒体库
    不收的格式（store/media.py 的 ACCEPTED_FORMATS 之外，比如 XBM）也算：以前照样导进来。"""
    media_root = memory.system.settings.paths.media
    png = _png()
    xbm = io.BytesIO()
    Image.new("1", (8, 8)).save(xbm, format="XBM")
    for bad in (
        _image(xbm.getvalue(), "image/x-xbitmap"),
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


def test_preset_help_names_every_shipped_preset() -> None:
    """--preset 的帮助列出随包发布的每个预设（以前漏了 deepseek 与 qwen）。"""
    [option] = [p for p in typer.main.get_command(app).params if "--preset" in p.opts]
    text = getattr(option, "help", None)
    assert isinstance(text, str)
    assert presets() and all(name in text for name in presets()), text


def test_cli_export_keeps_the_old_backup_when_the_write_fails(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """export -o 先写同一目录下的临时文件再换名：写失败（磁盘满）时旧的备份原样留着、临时文件
    删掉，只打印一行原因、以 1 退出。以前 write_text 先截断旧文件，调用栈之后新旧两份都没了。
    原来是 0644 的备份换成 0600（命令行收紧了 umask）；-o 是符号链接时替换它指向的文件。"""
    import errno

    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    assert runner.invoke(app, ["ingest", str(TOY_CONVERSATION)]).exit_code == 0
    backups = tmp_path / "backups"
    backups.mkdir()
    out = backups / "out.json"
    out.write_text("{}", encoding="utf-8")
    out.chmod(0o644)
    assert runner.invoke(app, ["export", "--space", "demo", "-o", str(out)]).exit_code == 0
    assert out.stat().st_mode & 0o777 == 0o600
    good = out.read_bytes()
    json.loads(good)

    real_replace = os.replace

    def full_disk(src: Any, dst: Any) -> None:
        if Path(dst) == out:
            raise OSError(errno.ENOSPC, "No space left on device")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", full_disk)
    result = runner.invoke(app, ["export", "--space", "demo", "-o", str(out)])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    flat = _flat(result.output)
    assert "导出失败" in flat and "Nospaceleftondevice" in flat and "未改动" in flat
    assert out.read_bytes() == good and sorted(backups.iterdir()) == [out]
    monkeypatch.setattr(os, "replace", real_replace)

    link = tmp_path / "link.json"
    link.symlink_to(out)
    out.write_text("{}", encoding="utf-8")
    assert runner.invoke(app, ["export", "--space", "demo", "-o", str(link)]).exit_code == 0
    sessions = json.loads(good)["sessions"]
    assert link.is_symlink() and json.loads(out.read_bytes())["sessions"] == sessions
    # 不是普通文件（/dev/stdout、FIFO、/dev/null）的照旧直接写：不在 /dev 里建临时文件
    assert runner.invoke(app, ["export", "--space", "demo", "-o", os.devnull]).exit_code == 0


def _mem0_export() -> dict[str, Any]:
    """Mem0 get_all(show_expired=True) 的结果（mem0ai 2.2.1 的形状）：时间带时区（Mem0 写美西
    时间），一条带 metadata，一条带过期日期。"""
    return {
        "results": [
            {
                "id": "6f1c0e3a-sushi",
                "memory": "Likes sushi",
                "hash": "h1",
                "created_at": "2026-05-01T09:00:00.123456-07:00",
                "updated_at": "2026-06-01T10:00:00-07:00",
                "user_id": "alice",
                "metadata": {"source": "chat"},
            },
            {
                "id": "9a7d2b11-berlin",
                "memory": "Lives in Berlin",
                "hash": "h2",
                "created_at": "2026-05-02T09:00:00-07:00",
                "updated_at": None,
                "user_id": "alice",
                "run_id": "trip",
                "expiration_date": "2030-01-01",
            },
        ]
    }


def test_mem0_import_is_idempotent(memory: Memory, fake_api: FakeAPI) -> None:
    """load(..., format="mem0")（第八轮体检）：一条 Mem0 记忆一个会话，metadata、过期日期与
    mem0_id 随事实带出，事实引用导入的发言；再导入一次什么都不加、不再调用模型；Mem0 里改过的
    那条（更新时间不同）作为新会话进来，只加 1 条。带时区的时间换算成本机时间。"""
    from cuemem.core.timeutil import naive_local

    data = _mem0_export()
    report = memory.load(data, format="mem0", wait=True)
    assert report["space_id"] == "alice" and report["turns_added"] == 2
    assert report["skipped"] == 0
    turns = {t.text: t for t in memory.system.db.get_turns("alice")}
    assert turns["Likes sushi"].ts == naive_local(
        datetime.fromisoformat("2026-06-01T10:00:00-07:00")
    )
    assert {t.session_id[:5] for t in turns.values()} == {"mem0-"}
    facts = {f["memory"]: f for f in memory.get_all(user_id="alice")["results"]}
    assert facts["Likes sushi"]["metadata"] == {
        "source": "chat",
        "mem0_id": "6f1c0e3a-sushi",
        "mem0_updated_at": "2026-06-01T10:00:00-07:00",
    }
    assert facts["Lives in Berlin"]["metadata"] == {
        "mem0_id": "9a7d2b11-berlin",
        "mem0_run_id": "trip",
    }
    assert facts["Lives in Berlin"]["expiration_date"] == "2030-01-01"
    assert facts["Likes sushi"]["source_turn_ids"] == [turns["Likes sushi"].id]

    calls = len(fake_api.chat_requests)
    again = memory.load(data, format="mem0", wait=True)
    assert again["turns_added"] == 0 and len(fake_api.chat_requests) == calls
    assert len(memory.system.db.get_turns("alice")) == 2

    edited = _mem0_export()
    edited["results"][0].update(memory="Loves sushi", updated_at="2026-07-01T10:00:00-07:00")
    assert memory.load(edited, format="mem0", wait=True)["turns_added"] == 1
    assert memory.load(edited, format="mem0")["turns_added"] == 0


def test_mem0_import_refuses_mixed_users_and_malformed_items(memory: Memory) -> None:
    """几个用户混在一起又没给 user_id 时 ValueError、什么都不写（给了 user_id 时都进那个空间）；
    哪一条不合格式（memory 不是文字、metadata 不是对象、时间认不出）都整个拒绝，什么都不写。文字
    为空的记忆跳过、记数。format 写错时 ValueError。"""
    mixed = _mem0_export()
    mixed["results"][1]["user_id"] = "bob"
    with pytest.raises(ValueError, match="mixes several users"):
        memory.load(mixed, format="mem0")
    assert memory.spaces() == []
    report = memory.load(mixed["results"], format="mem0", user_id="carol")
    assert report["space_id"] == "carol" and report["turns_added"] == 2

    for field, value in (
        ("memory", None),
        ("metadata", [1]),
        ("created_at", "yesterday"),
        # 带时区、在日期范围边上：换算成本机时间时 OverflowError，以前原样抛出（第八轮复审）
        ("created_at", "0001-01-01T00:00:00+05:00"),
        ("updated_at", "9999-12-31T23:59:59-05:00"),
        ("expiration_date", "9999-12-31T23:59:59-05:00"),
        ("expiration_date", "soon"),
        ("user_id", 42),
    ):
        bad = _mem0_export()
        bad["results"][1][field] = value
        with pytest.raises(ValueError, match="Malformed Mem0 export"):
            memory.load(bad, format="mem0")
    with pytest.raises(ValueError, match="Malformed Mem0 export"):
        memory.load({"memories": []}, format="mem0", user_id="dave")
    assert [s["space_id"] for s in memory.spaces()] == ["carol"]

    blank = _mem0_export()
    blank["results"][0]["memory"] = "  "
    report = memory.load(blank, format="mem0", user_id="erin")
    assert report["turns_added"] == 1 and report["skipped"] == 1
    with pytest.raises(ValueError, match="format"):
        memory.load(blank, format="letta")  # type: ignore[arg-type]
    # 不是对象的文件当作 CueMem 导出：与 Memory.load 相同的说明，以前说 AttributeError
    with pytest.raises(ValueError, match="is a JSON object"):
        import_space(memory.system, blank["results"])  # type: ignore[arg-type]


def test_cli_import_from_mem0(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cuemem import --from mem0：不给 --user 时按 user_id 分组，每个用户导入到自己的空间，逐个
    报告；再运行一次什么都不加。--user 时都导入到那个用户。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    data = _mem0_export()
    data["results"][1]["user_id"] = "bob"
    path = tmp_path / "mem0.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(app, ["import", str(path), "--from", "mem0"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "记忆空间alice：1个会话，新增1条发言" in flat
    assert "记忆空间bob：1个会话，新增1条发言" in flat
    assert "各是一个事务" in flat
    result = runner.invoke(app, ["import", str(path), "--from", "mem0"])
    assert result.exit_code == 0 and _flat(result.output).count("新增0条发言") == 2
    result = runner.invoke(app, ["import", str(path), "--from", "mem0", "--user", "carol"])
    assert result.exit_code == 0 and "记忆空间carol：2个会话，新增2条发言" in _flat(result.output)
    spaces = runner.invoke(app, ["spaces"]).output
    assert all(name in spaces for name in ("alice", "bob", "carol"))

    data["results"][0]["user_id"] = None
    path.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(app, ["import", str(path), "--from", "mem0"])
    assert result.exit_code == 1 and "nouser_idoragent_id" in _flat(result.output)
    result = runner.invoke(app, ["import", str(path), "--from", "letta"])
    assert result.exit_code == 1 and "--from只能是" in _flat(result.output)

    # 忘了 --from mem0（第八轮复审）：两种 Mem0 的形状都一行提示加上 --from mem0，什么都不写；
    # 以前列表说 AttributeError，{"results": [...]} 说不是 CueMem 导出
    for shape in (data["results"], data):
        path.write_text(json.dumps(shape), encoding="utf-8")
        result = runner.invoke(app, ["import", str(path)])
        flat = _flat(result.output)
        assert result.exit_code == 1 and "--frommem0" in flat and "Error" not in flat, flat


@pytest.mark.parametrize("damage", ["truncated", "not-utf8"])
def test_cli_import_of_a_damaged_file_is_one_line(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    """导入截断的或不是 UTF-8 的文件：一行原因、以 1 退出（以前是 JSONDecodeError /
    UnicodeDecodeError 的调用栈），什么都不写。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    assert runner.invoke(app, ["ingest", str(TOY_CONVERSATION)]).exit_code == 0
    out = tmp_path / "demo.json"
    assert runner.invoke(app, ["export", "--space", "demo", "-o", str(out)]).exit_code == 0
    whole = out.read_bytes()
    bad = tmp_path / "bad.json"
    bad.write_bytes(whole[: len(whole) // 2] if damage == "truncated" else b"\xff\xfe" + whole)
    result = runner.invoke(app, ["import", str(bad), "--space", "copy"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    assert "不是有效的CueMem导出文件" in _flat(result.output)
    assert "copy" not in runner.invoke(app, ["spaces"]).output


@pytest.mark.parametrize(
    ("env", "command", "expected"),
    [
        ({}, ["--preset", "opneai", "doctor"], ["没有预设opneai；可用的有："]),
        (  # docker run --env-file 把行尾的注释算进值里
            {"CUEMEM_EMBEDDING__BACKEND": "hash   # 行尾注释"},
            ["proxy"],
            ["embedding.backend：Inputshouldbe", "收到'hash#行尾注释'"],
        ),
        (  # 令牌够长（至少 16 个字符），拦下它的只是配置
            {"CUEMEM_API_TOKEN": "t" * 16, "CUEMEM_PRESET": "opneai"},
            ["serve", "--host", "0.0.0.0"],
            ["没有预设opneai"],
        ),
        ({"CUEMEM_PRESET": "opneai"}, ["ask", "q", "--space", "s"], ["没有预设opneai"]),
        ({"CUEMEM_PRESET": "opneai"}, ["mcp"], ["没有预设opneai"]),
    ],
)
def test_config_errors_are_one_line_not_a_traceback(
    clean_env: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    command: list[str],
    expected: list[str],
) -> None:
    """预设名写错、配置项的值不合法时，各命令都只打印一行原因、以 1 退出。以前 load_settings 的
    ValueError（预设）与 pydantic 的 ValidationError（取值）在 doctor、proxy、serve、ask、mcp 等
    命令里都是几十行调用栈：Docker 里同样以 1 退出、反复重启，原因埋在调用栈里。mcp 写到 stderr。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_PRESET", "")  # --preset 会写这个变量：登记一下，测试结束时还原
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    served: list[Any] = []
    monkeypatch.setattr("uvicorn.run", lambda app_, **kwargs: served.append(app_))
    result = runner.invoke(app, command)
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    flat = _flat(result.output)
    assert "错误：配置有误：" in flat and all(part in flat for part in expected), result.output
    assert "errors.pydantic.dev" not in result.output and served == []
    if command == ["mcp"]:  # stdout 是 MCP 的通道
        assert "配置有误" in result.stderr and result.stdout == ""


@pytest.mark.parametrize(
    ("variable", "yaml", "expected"),
    [
        (  # 取值不合法：Key 写成了数字
            "CUEMEM_PRESET",
            "vector_store:\n  api_key: 4815162342\n",
            "vector_store.api_key：Inputshouldbeavalidstring",
        ),
        (  # 整份配置的引用检查：收到的是整份配置，里面带着 Key
            "CUEMEM_CONFIG",
            "vector_store:\n  api_key: qdrant-secret-0123456789\n"
            "roles:\n  agent:\n    model: nope\n",
            "角色agent引用了未定义的模型nope",
        ),
    ],
)
def test_config_errors_do_not_echo_the_vector_store_key(
    clean_env: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    yaml: str,
    expected: str,
) -> None:
    """配置写错时的一行原因不回显 Key：vector_store.api_key（Qdrant 服务端的 Key）是配置里的
    密钥，这一行会进 docker compose logs。以前名字像密钥的字段与整份配置都照样回显收到的值。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_PRESET", "")
    config = tmp_path / "config.yaml"
    config.write_text(yaml, encoding="utf-8")
    monkeypatch.setenv(variable, str(config))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    assert expected in _flat(result.output), result.output
    assert "4815162342" not in result.output and "qdrant-secret" not in result.output
    assert "收到" not in result.output


ENV_ASSIGNMENT = re.compile(r"^#?\s*([A-Z][A-Z0-9_]*)=(.*)$")


def test_env_example_lines_survive_docker_env_file(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """docker run --env-file 不认行尾注释：值原样取到行尾，连同空格与 # 之后的说明。.env.example
    里的每一行赋值（注释掉的取消注释之后）都没有这些，cuemem 也读得进去。以前三行把说明写在值的
    后面，取消注释之后预设成了 "openai   # 全部换成 OpenAI…"，哈希向量的那一行同样读不进去
    （compose 的 env_file 与 python-dotenv 会去掉行尾注释，只有这条路出错）。"""
    lines = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    found = [m.groups() for line in lines if (m := ENV_ASSIGNMENT.match(line))]
    names = {name for name, _ in found}
    assert {"CUEMEM_PRESET", "CUEMEM_EMBEDDING__BACKEND", "CUEMEM_API_TOKEN"} <= names
    for name, value in found:
        assert "#" not in value and value == value.strip(), (name, value)
        if name.startswith("CUEMEM_") and value:
            with monkeypatch.context() as patch:
                patch.setenv(name, value)
                load_settings(home=tmp_path)  # 预设名与取值都合法


def test_keyless_example_runs_offline(clean_env: None, capsys: pytest.CaptureFixture[str]) -> None:
    """examples/keyless.py（不要 Key 的试用）：不设任何 Key、不调用模型，用哈希向量写入一句话、
    再检索出来；数据在临时目录里。"""
    runpy.run_path(str(PROJECT_ROOT / "examples" / "keyless.py"), run_name="__main__")
    assert capsys.readouterr().out == "L1 User: I just started a new job at Acme in Berlin.\n"
