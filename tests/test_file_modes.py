"""数据与密钥文件的权限（第四轮体检）：以前按常见的 umask 022 建成目录 0755、文件 0644，同一台
机器上别的账户能读对话原文、LLM 缓存、trace、导出与明文的令牌。新建的存储目录只给本人（0700），
命令行建的文件 0600；已经存在的目录一律不改（运营者的 data/、runs/），doctor 只提醒。"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

import pytest
from fakes import TOY_CONVERSATION, FakeAPI
from typer.testing import CliRunner

from cuemem.cli import app
from cuemem.config import Settings
from cuemem.envfile import update_env
from cuemem.memory import MemorySystem

pytestmark = pytest.mark.skipif(os.name != "posix", reason="只有 POSIX 有这些权限位")
runner = CliRunner()


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _flat(text: str) -> str:
    return "".join(text.split())


@pytest.fixture
def cli_home(clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """命令行的数据目录放在临时目录里（不碰仓库里的 data/、runs/），用离线的哈希向量。"""
    home = tmp_path / "home"
    monkeypatch.setenv("CUEMEM_HOME", str(home))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    return home


def test_new_storage_directories_are_private(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """MemorySystem.from_settings 新建的数据目录是 0700（SDK 也一样，不改进程的 umask）：以前是
    0755，里面的数据库、缓存与 trace 是 0644。"""
    os.umask(0o022)
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()):
        pass
    for name in ("data", "data/cache", "data/qdrant", "data/media", "runs", "runs/traces"):
        assert _mode(tmp_path / name) == 0o700, name
    assert _umask() == 0o022


def test_existing_directories_are_left_alone(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path
) -> None:
    """已经存在的目录不改权限（运营者的 data/ 可能有意给备份用户留了读权限）：只有新建的是 0700。
    向量库在 Qdrant 服务端时不建本地向量库的目录。"""
    os.umask(0o022)
    (tmp_path / "data").mkdir()
    os.chmod(tmp_path / "data", 0o755)
    with MemorySystem.from_settings(n3_settings, http_client=fake_api.client()):
        pass
    assert _mode(tmp_path / "data") == 0o755
    assert _mode(tmp_path / "data" / "cache") == 0o700 and _mode(tmp_path / "runs") == 0o700
    from cuemem.store.files import private_dirs, storage_dirs

    remote = n3_settings.model_copy(
        update={"vector_store": n3_settings.vector_store.model_copy(update={"url": "http://q"})}
    )
    assert tmp_path / "data" / "qdrant" not in storage_dirs(remote)
    nested = tmp_path / "a" / "b" / "c"
    private_dirs(nested, nested)
    assert [_mode(p) for p in (nested.parents[1], nested.parent, nested)] == [0o700] * 3
    (tmp_path / "file").write_text("x")
    private_dirs(tmp_path / "file" / "sub", tmp_path / "after")  # 建不了的留给各组件自己报错
    assert (tmp_path / "file").is_file() and _mode(tmp_path / "after") == 0o700


def test_cli_tightens_the_umask_and_never_loosens_it(cli_home: Path, tmp_path: Path) -> None:
    """命令行一启动就把 umask 收紧到至少 077：导出的文件（原始日志与图片）、缓存、trace 都是
    0600。以前是 0644。已经更严的 umask 照旧（从不放宽）。"""
    os.umask(0o022)
    assert runner.invoke(app, ["doctor"]).exit_code == 0
    assert _umask() & 0o077 == 0o077
    os.umask(0o022)
    assert runner.invoke(app, ["ingest", str(TOY_CONVERSATION)]).exit_code == 0
    out = tmp_path / "demo.json"
    result = runner.invoke(app, ["export", "--space", "demo", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert _mode(out) == 0o600
    assert _mode(cli_home / "data" / "cuemem.sqlite") == 0o600
    assert _mode(cli_home / "data") == 0o700
    from cuemem.cli import _private_umask

    os.umask(0o027)
    assert _private_umask() == 0o077 and _umask() == 0o077
    os.umask(0o277)
    assert _private_umask() == 0o277 and _umask() == 0o277


def test_update_env_never_exposes_keys_through_an_earlier_descriptor(tmp_path: Path) -> None:
    """update_env 把新内容写进 0600 的临时文件再换名：以前就地重写 0644 的 .env（cp .env.example
    .env 得到的）之后才 chmod，之前打开的文件描述符照样读到新写的 Key。.env 是符号链接时改它
    指向的文件，链接留着。"""
    os.umask(0o022)
    env = tmp_path / ".env"
    env.write_text("# 备注\nDEEPSEEK_API_KEY=\nOTHER=1\n", encoding="utf-8")
    assert _mode(env) == 0o644
    with env.open(encoding="utf-8") as held:
        update_env(env, {"DEEPSEEK_API_KEY": "sk-secret-123456"})
        assert "sk-secret" not in held.read()
    assert _mode(env) == 0o600
    assert env.read_text(encoding="utf-8") == "# 备注\nDEEPSEEK_API_KEY=sk-secret-123456\nOTHER=1\n"
    assert [p.name for p in tmp_path.iterdir()] == [".env"]  # 没有留下临时文件

    real = tmp_path / "secrets" / "cuemem.env"
    real.parent.mkdir()
    real.write_text("OTHER=1\n", encoding="utf-8")
    link = tmp_path / "linked.env"
    link.symlink_to(real)
    update_env(link, {"DASHSCOPE_API_KEY": "sk-ws-1"})
    assert link.is_symlink() and link.resolve() == real.resolve()
    assert real.read_text(encoding="utf-8") == "OTHER=1\nDASHSCOPE_API_KEY=sk-ws-1\n"
    assert _mode(real) == 0o600


def test_doctor_warns_about_readable_data_and_secrets(
    cli_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """doctor 提醒别的本地用户能读的数据目录（chmod 700）与 .env、keys 文件（chmod 600），只提醒、
    不改。数据目录下的子目录由上一级管着，不重复提醒。"""
    env = Path(os.environ["CUEMEM_ENV_FILE"])
    env.write_text("OTHER=1\n", encoding="utf-8")
    keys = tmp_path / "keys.yaml"
    keys.write_text("keys: []\n", encoding="utf-8")
    monkeypatch.setenv("CUEMEM_API_KEYS_FILE", str(keys))
    for directory in (cli_home / "data" / "cache", cli_home / "runs" / "traces"):
        directory.mkdir(parents=True)
    for path, mode in ((env, 0o644), (keys, 0o640), (cli_home / "data", 0o755)):
        os.chmod(path, mode)
    os.chmod(cli_home / "data" / "cache", 0o755)
    os.chmod(cli_home / "runs" / "traces", 0o750)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert f"chmod600{env}" in flat and f"chmod600{keys}" in flat
    data, traces = cli_home / "data", cli_home / "runs" / "traces"
    assert f"chmod700{data}" in flat and f"chmod700{traces}" in flat
    assert f"chmod700{data / 'cache'}" not in flat
    assert _mode(env) == 0o644 and _mode(cli_home / "data") == 0o755  # 不自动改

    for path, mode in ((env, 0o600), (keys, 0o600), (cli_home / "data", 0o700)):
        os.chmod(path, mode)
    os.chmod(cli_home / "runs" / "traces", 0o700)
    assert "chmod" not in runner.invoke(app, ["doctor"]).output


def test_proxy_warns_about_a_readable_keys_file(
    cli_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cuemem proxy / serve 启动时同样提醒别的本地用户能读的 .env 与 keys 文件（里面可能有明文的
    token:）。"""
    calls: list[Any] = []
    monkeypatch.setattr("uvicorn.run", lambda app_, **kwargs: calls.append(app_))
    keys = tmp_path / "keys.yaml"
    keys.write_text(
        "keys:\n  - name: alice-app\n    token: alice-token-0123456789\n"
        "    scopes: [{user: alice}]\n",
        encoding="utf-8",
    )
    os.chmod(keys, 0o644)
    monkeypatch.setenv("CUEMEM_API_KEYS_FILE", str(keys))
    result = runner.invoke(app, ["proxy"])
    assert result.exit_code == 0, result.output
    assert f"chmod600{keys}" in _flat(result.output) and len(calls) == 1
    os.chmod(keys, 0o600)
    result = runner.invoke(app, ["serve", "--no-ui", "--scenario", "plain"])
    assert result.exit_code == 0, result.output
    assert "chmod" not in result.output and len(calls) == 2


def test_gradio_temp_dir_is_private_and_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    """界面的临时目录：没设 GRADIO_TEMP_DIR 时换成这次启动自己的 0700 目录、退出时删掉（以前是
    整台机器共用、0755 的 /tmp/gradio，界面的图库把原图复制进去）；运营者设了的照旧。"""
    from cuemem.cli import _gradio_temp_dir

    monkeypatch.delenv("GRADIO_TEMP_DIR", raising=False)
    with _gradio_temp_dir(True):
        directory = Path(os.environ["GRADIO_TEMP_DIR"])
        assert directory.is_dir() and _mode(directory) == 0o700
        (directory / "photo.jpg").write_bytes(b"x")
    assert not directory.exists() and "GRADIO_TEMP_DIR" not in os.environ
    with _gradio_temp_dir(False):  # 没有界面
        assert "GRADIO_TEMP_DIR" not in os.environ
    monkeypatch.setenv("GRADIO_TEMP_DIR", "/srv/gradio")
    with _gradio_temp_dir(True):
        assert os.environ["GRADIO_TEMP_DIR"] == "/srv/gradio"
    assert os.environ["GRADIO_TEMP_DIR"] == "/srv/gradio"


def test_serve_gives_the_ui_a_private_temp_dir(
    cli_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cuemem serve 带界面时，Gradio 在这次启动自己的 0700 目录里放上传与图库的副本，停下时删掉。"""
    monkeypatch.delenv("GRADIO_TEMP_DIR", raising=False)
    seen: list[Path] = []

    def run(app_: Any, **kwargs: Any) -> None:
        directory = Path(os.environ["GRADIO_TEMP_DIR"])
        assert directory.is_dir() and _mode(directory) == 0o700
        seen.append(directory)

    monkeypatch.setattr("uvicorn.run", run)
    result = runner.invoke(app, ["serve", "--scenario", "plain"])
    assert result.exit_code == 0, result.output
    (directory,) = seen
    assert not directory.exists() and "GRADIO_TEMP_DIR" not in os.environ
