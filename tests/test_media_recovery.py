"""进程被杀掉时媒体库里的原图（第七轮体检 P8）：写入存下原图、图片记录还没提交时 kill -9，或者遗忘
提交之后、删原图之前 kill -9，以前留下一个没有任何记录指向的文件，遗忘、delete_all 与 reset 都找
不到它。现在两种操作先写标记，MemorySystem.recover_media（cuemem serve / proxy 启动时、cuemem
reindex）按标记删掉。媒体库与评测空间共用：没有标记的文件（评测空间的图片）一张都不删。

被杀掉的进程是真的子进程（os._exit，什么清理都不做）。"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
from fakes import FakeAPI
from PIL import Image
from typer.testing import CliRunner

from cuemem.cli import app
from cuemem.config import load_settings
from cuemem.core.models import MediaItem
from cuemem.memory import MemorySystem
from cuemem.store.media import MediaStore
from cuemem.store.sqlite import MemoryDB

TESTS = Path(__file__).parent
WHEN = datetime(2026, 11, 6, 9, 0)

# 子进程：用环境变量里的配置打开 Memory，写一张图（infer=False：不调用 LLM），在 crash 指定的
# 地方 os._exit(9)，与 kill -9 一样什么清理都不做
CHILD = """
import base64, io, os, sys
sys.path.insert(0, sys.argv[1])
from fakes import FakeAPI
from PIL import Image
from cuemem.config import load_settings
from cuemem.sdk import Memory

crash = sys.argv[2]
buffer = io.BytesIO()
Image.new("RGB", (24, 24), sys.argv[3]).save(buffer, format="PNG")
url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
memory = Memory(load_settings(), http_client=FakeAPI().client(), background=False)
system = memory.system
if crash == "write":
    system.writer._embed = lambda units: os._exit(9)
content = [{"type": "text", "text": "My passport photo"}, {"type": "image_url", "image_url": url}]
memory.add({"role": "user", "content": content}, user_id="alice", infer=False)
if crash == "forget":
    system._delete_image_files = lambda items: os._exit(9)
    memory.delete_all(user_id="alice")
os._exit(0)
"""


@pytest.fixture
def env(clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """数据都在 tmp_path 里（父子进程读同一份环境变量），离线的哈希向量。"""
    for name in ("db", "qdrant", "llm_cache", "embed_cache", "ledger", "traces", "media"):
        monkeypatch.setenv(f"CUEMEM_PATHS__{name.upper()}", str(tmp_path / name))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    scratch = tmp_path / "tmp"  # SDK 把 data URL 写成临时文件：不落到系统的 /tmp
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    return dict(os.environ)


def _killed(env: dict[str, str], crash: str, color: str = "red") -> None:
    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(TESTS), crash, color],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 9, done.stderr


def _images(media: Path) -> list[Path]:
    """媒体库里的原图与复制了一半的 .part（不算 .journal 里的标记）。"""
    return sorted(
        p for p in media.rglob("*") if p.is_file() and ".journal" not in p.relative_to(media).parts
    )


def _system() -> MemorySystem:
    return MemorySystem.from_settings(load_settings(), http_client=FakeAPI().client())


def test_write_killed_before_commit_leaves_no_photo(env: dict[str, str], tmp_path: Path) -> None:
    """向量化时被杀掉：原图已经存进媒体库，图片记录没有提交。以前之后的 delete_all 删不掉它
    （images_deleted 0），文件一直留着；现在 recover_media 按标记删掉，不留 .part。"""
    _killed(env, "write")
    media = tmp_path / "media"
    assert len(_images(media)) == 1  # 被杀掉时留下的
    system = _system()
    try:
        assert system.db.get_media("alice") == []  # 没有记录指向它
        assert system.recover_media() == (1, 0)
        assert _images(media) == []
        assert list((media / ".journal").iterdir()) == []
        assert system.recover_media() == (0, 0)
    finally:
        system.close()


def test_forget_killed_after_commit_leaves_no_photo(env: dict[str, str], tmp_path: Path) -> None:
    """遗忘提交之后、删原图之前被杀掉：记录已经删了，以前之后的遗忘再也找不到这个文件。"""
    _killed(env, "forget")
    media = tmp_path / "media"
    assert len(_images(media)) == 1
    system = _system()
    try:
        assert system.db.get_media("alice") == [] and system.db.get_turns("alice") == []
        assert system.recover_media() == (1, 0)
        assert _images(media) == []
    finally:
        system.close()


def test_serve_recovers_at_start_up(env: dict[str, str], tmp_path: Path) -> None:
    """cuemem serve / proxy 的清理线程一启动就收拾（之后每小时一遍）。"""
    from fastapi.testclient import TestClient

    from cuemem.agent import Agent
    from cuemem.api.app import create_app
    from cuemem.api.service import ChatService

    _killed(env, "write")
    system = _system()
    service = ChatService(system, Agent(system, router="rule"))
    try:
        with TestClient(create_app(service, upload_dir=tmp_path / "up", ui=False)):
            deadline = time.monotonic() + 10
            while _images(tmp_path / "media"):
                assert time.monotonic() < deadline
                time.sleep(0.01)
    finally:
        service.close()


def test_recovery_never_touches_unmarked_or_committed_files(
    env: dict[str, str], tmp_path: Path
) -> None:
    """媒体库与评测空间共用（MemorySystem.on_disk / ephemeral 的 MediaStore 不写标记，图片记录在
    评测空间自己的数据库里）：主数据库查不到它的记录，没有标记就不删，cuemem reindex --all 也
    不删。写入提交了、release 之前被杀掉的（标记还在，记录也在）：文件留着、只删标记。别的数据库
    的标记不碰。"""
    settings = load_settings()
    media = settings.paths.media
    eval_db = MemoryDB(tmp_path / "eval" / "memory.sqlite")
    picture = tmp_path / "eval.png"
    Image.new("RGB", (8, 8), "green").save(picture)
    eval_store = MediaStore(media)  # 与 on_disk、ephemeral 相同：没有 owner
    stored = eval_store.put(picture)
    eval_db.ensure_space("eval-q1")
    eval_db.add_media(
        [
            MediaItem(
                id="m1",
                space_id="eval-q1",
                sha256=stored.sha256,
                path=str(stored.path),
                source_turn_id="t1",
                ts=WHEN,
            )
        ]
    )
    eval_store.release(stored)
    assert stored.marker is None and not (media / ".journal").exists()

    # 主数据库里提交了、标记还在（release 之前被杀掉）：子进程写图后在 release 之前退出
    _killed(env, "write", color="blue")  # 先造一个被杀掉的写入，再造一个提交了的
    system = _system()
    try:
        committed = tmp_path / "committed.png"
        Image.new("RGB", (8, 8), "yellow").save(committed)
        assert system.media is not None
        kept = system.media.put(committed)
        assert kept.marker is not None and kept.marker.exists()
        system.db.ensure_space("bob")
        system.db.add_media(
            [
                MediaItem(
                    id="m2",
                    space_id="bob",
                    sha256=kept.sha256,
                    path=str(kept.path),
                    source_turn_id="t2",
                    ts=WHEN,
                )
            ]
        )
        # 进程被杀掉、换了一个同号的进程（容器重启后的 1 号进程）：标记与正在写入的计数都不在
        # 本进程里了，写下标记的进程号就是本进程，算"不在了"
        MediaStore._unmark(kept.marker, delete=False)
        system.media._done(kept.sha256)
        # 别的数据库（同一个媒体库的另一个部署）留下的标记：不碰它点名的文件
        foreign = media / ".journal" / f"{stored.sha256}.foreign.json"
        record = {"kind": "put", "sha256": stored.sha256, "name": stored.path.name}
        record |= {"owner": "/elsewhere/memory.sqlite", "pid": 1, "host": "x", "ts": 0}
        foreign.write_text(json.dumps(record), "utf-8")

        assert system.recover_media() == (1, 1)  # 被杀掉的写入那张删掉，提交了的留着
    finally:
        system.close()
    assert stored.path.exists() and kept.path.exists() and foreign.exists()
    assert not kept.marker.exists()
    runner = CliRunner()
    result = runner.invoke(app, ["reindex", "--all"])
    assert result.exit_code == 0, result.output
    assert stored.path.exists() and kept.path.exists()
    assert {p.name for p in _images(media)} == {stored.path.name, kept.path.name}
    eval_db.close()


def test_live_operations_are_left_alone(env: dict[str, str], tmp_path: Path) -> None:
    """本进程里还没结束的写入（标记在 _live_markers 里）不收拾，哪怕已经写下很久；doctor 只数
    该收拾的。写入失败时（_discard）照旧删原图，连同标记。"""
    system = _system()
    try:
        assert system.media is not None
        picture = tmp_path / "live.png"
        Image.new("RGB", (8, 8), "purple").save(picture)
        live = system.media.put(picture)
        assert live.marker is not None
        assert system.media.recover(system.db.media_sha_in_use, older_than_s=0) == (0, 0)
        assert system.media.pending(older_than_s=0) == 0 and live.path.exists()
        assert system.media.release(live, in_use=system.db.media_sha_in_use)
        assert not live.path.exists() and not live.marker.exists()
    finally:
        system.close()


def test_doctor_reports_interrupted_operations(env: dict[str, str]) -> None:
    """cuemem doctor 提示被中断的写入与遗忘，指向 cuemem reindex。"""
    _killed(env, "write")
    result = CliRunner().invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    flat = "".join(result.output.split())
    assert "1个被中断的写入或遗忘" in flat and "cuememreindex--all" in flat
    result = CliRunner().invoke(app, ["reindex", "--all"])
    assert result.exit_code == 0, result.output
    assert "删掉1张" in "".join(result.output.split())
    assert "被中断" not in CliRunner().invoke(app, ["doctor"]).output


def test_data_url_image_bytes_are_unchanged(env: dict[str, str]) -> None:
    """标记只是媒体库里的元信息文件：存下的原图逐字节就是传进来的图。"""
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(buffer, format="PNG")
    raw = buffer.getvalue()
    system = _system()
    try:
        assert system.media is not None
        source = Path(env["TMPDIR"]) / "white.png"
        source.write_bytes(raw)
        stored = system.media.put(source)
        assert stored.path.read_bytes() == raw
        system.media.release(stored)
        assert base64.b64encode(stored.path.read_bytes()) == base64.b64encode(raw)
    finally:
        system.close()
