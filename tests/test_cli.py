from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from dotenv import dotenv_values
from fakes import TOY_CONVERSATION
from typer.testing import CliRunner

from cuemem.cli import app
from cuemem.config import load_settings

runner = CliRunner()


def _flat(text: str) -> str:
    """去掉所有空白，避免终端宽度导致的换行影响断言。"""
    return "".join(text.split())


@pytest.fixture
def isolated_paths(clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("db", "qdrant", "llm_cache", "embed_cache", "ledger", "traces"):
        monkeypatch.setenv(f"CUEMEM_PATHS__{name.upper()}", str(tmp_path / name))


@pytest.fixture
def env_path(clean_env: None) -> Path:
    """configure 写入的 .env：clean_env 已把它指向临时目录。"""
    return Path(os.environ["CUEMEM_ENV_FILE"])


def test_offline_ingest_spaces_doctor_usage(
    isolated_paths: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")

    result = runner.invoke(app, ["ingest", str(TOY_CONVERSATION)])
    assert result.exit_code == 0, result.output
    assert "新增18条轮次、9个记忆单元" in _flat(result.output)

    result = runner.invoke(app, ["spaces"])
    assert result.exit_code == 0, result.output
    assert "demo" in result.output

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "未设置Key" in _flat(result.output)

    result = runner.invoke(app, ["usage"])
    assert result.exit_code == 0, result.output


def test_missing_key_gives_friendly_error(isolated_paths: None) -> None:
    result = runner.invoke(app, ["ingest", str(TOY_CONVERSATION)])
    assert result.exit_code == 1
    assert "DASHSCOPE_API_KEY" in result.output


def test_configure_writes_keys(env_path: Path) -> None:
    env_path.write_text("# 我的备注\nOTHER=1\nDEEPSEEK_API_KEY=\n", encoding="utf-8")
    answers = "fake-deepseek-1234567890\nfake-qwencloud-0987654321\n"
    result = runner.invoke(app, ["configure"], input=answers)
    assert result.exit_code == 0, result.output

    values = dotenv_values(env_path)
    assert values["DEEPSEEK_API_KEY"] == "fake-deepseek-1234567890"
    assert values["DASHSCOPE_API_KEY"] == "fake-qwencloud-0987654321"
    assert values["OTHER"] == "1"  # 其他内容原样保留
    assert "# 我的备注" in env_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    # Key 不回显，只显示首尾几位
    assert "fake-deepseek-1234567890" not in result.output
    assert "fak…7890" in _flat(result.output)

    loaded = load_settings()
    assert loaded.providers["qwen"].api_key() == "fake-qwencloud-0987654321"


def test_configure_keeps_existing_values(env_path: Path) -> None:
    env_path.write_text("DEEPSEEK_API_KEY=fake-old-111111111\n", encoding="utf-8")
    result = runner.invoke(app, ["configure"], input="\n\n")  # 两次回车：保留、跳过
    assert result.exit_code == 0, result.output

    values = dotenv_values(env_path)
    assert values["DEEPSEEK_API_KEY"] == "fake-old-111111111"
    assert "DASHSCOPE_API_KEY" not in values  # 没填就不写入，并提示未填写
    assert "未填写" in result.output


def test_consolidate_inspect_and_forget(
    isolated_paths: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import FakeAPI
    from scripted import TOY_EXTRACTIONS, TOY_UPDATE_KEYWORDS, ScriptedModel, reconcile_by_keyword

    from cuemem import cli
    from cuemem.memory import MemorySystem

    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-qwen")
    for name in ("DUPLICATE", "CONFLICT", "MULTI_CONFLICT", "ENTITY_MERGE", "ENTITY_GRAY"):
        monkeypatch.setenv(f"CUEMEM_CONSOLIDATION__{name}_THRESHOLD", "0.9999")
    api = FakeAPI()
    api.chat_replies = [
        ScriptedModel(TOY_EXTRACTIONS, reconcile=reconcile_by_keyword(*TOY_UPDATE_KEYWORDS))
    ]
    original = MemorySystem.from_settings
    monkeypatch.setattr(
        cli.MemorySystem,
        "from_settings",
        lambda settings: original(settings, http_client=api.client()),
    )
    monkeypatch.setattr(cli.console, "width", 300)  # 表格不折行，便于断言

    result = runner.invoke(app, ["ingest", str(TOY_CONVERSATION), "--consolidate"])
    assert result.exit_code == 0, result.output
    assert "巩固5个会话" in _flat(result.output)

    result = runner.invoke(app, ["inspect", "facts", "--space", "demo", "--subject", "User"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "已变化" in flat and "当前" in flat and "新加坡一家电商公司" in flat

    result = runner.invoke(
        app, ["inspect", "facts", "--space", "demo", "--valid-at", "2025-06-01", "--current"]
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, ["inspect", "entities", "--space", "demo"])
    assert result.exit_code == 0 and "Max" in result.output

    result = runner.invoke(app, ["inspect", "profile", "--space", "demo"])
    assert result.exit_code == 0 and "林晓" in result.output

    result = runner.invoke(app, ["spaces"])
    assert result.exit_code == 0 and "demo" in result.output

    result = runner.invoke(app, ["forget", "--space", "demo", "--session", "s4"], input="n\n")
    assert result.exit_code == 1  # 没有确认，不删除
    result = runner.invoke(app, ["forget", "--space", "demo", "--session", "s4", "--yes"])
    assert result.exit_code == 0, result.output
    assert "已删除4条轮次" in _flat(result.output)
    result = runner.invoke(app, ["inspect", "facts", "--space", "demo", "--current"])
    assert "新加坡一家电商公司" not in _flat(result.output)
