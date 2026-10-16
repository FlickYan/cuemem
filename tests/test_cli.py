from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

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

    # 规则版控制器把聚合题送到 L2：显示控制器的选择、检索计划与充分性检查
    question = "我一共养了几只狗？"
    args = ["ask", question, "--space", "demo", "--at", "2025-10-10", "--show-context"]
    result = runner.invoke(app, [*args, "--mode", "auto", "--router", "rule"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "走L2" in flat and "检索计划（LLM）" in flat and "充分性检查1：够了" in flat
    assert runner.invoke(app, [*args, "--mode", "bogus"]).exit_code == 1

    result = runner.invoke(app, ["forget", "--space", "demo", "--session", "s4"], input="n\n")
    assert result.exit_code == 1  # 没有确认，不删除
    result = runner.invoke(app, ["forget", "--space", "demo", "--session", "s4", "--yes"])
    assert result.exit_code == 0, result.output
    assert "已删除4条轮次" in _flat(result.output)
    result = runner.invoke(app, ["inspect", "facts", "--space", "demo", "--current"])
    assert "新加坡一家电商公司" not in _flat(result.output)


def test_demo_and_chat_commands(
    isolated_paths: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json

    from fakes import FakeAPI
    from PIL import Image
    from scripted import ScriptedModel, echo_extraction

    from cuemem import cli
    from cuemem.memory import MemorySystem

    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    monkeypatch.setenv("CUEMEM_PATHS__MEDIA", str(tmp_path / "media"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-qwen")
    api = FakeAPI()
    api.chat_replies = [ScriptedModel(echo_extraction, agent=lambda body: "Hi Maya, noted.")]
    original = MemorySystem.from_settings
    monkeypatch.setattr(
        cli.MemorySystem,
        "from_settings",
        lambda settings, **kwargs: original(settings, http_client=api.client()),
    )
    monkeypatch.setattr(cli.console, "width", 300)
    Image.new("RGB", (32, 32), "white").save(tmp_path / "shoes.jpg")
    script = tmp_path / "script.json"
    script.write_text(
        json.dumps(
            {
                "space": "cli-demo",
                "sessions": [
                    {
                        "at": "2026-08-28T19:30",
                        "messages": [
                            {"text": "Hi, I'm Maya.", "expect": [["Maya"]]},
                            {"text": "Like these?", "images": ["shoes.jpg"], "expect_not": ["$"]},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    result = runner.invoke(app, ["demo", str(script), "--fresh", "--report", str(report)])
    assert result.exit_code == 0, result.output
    assert "通过2/2" in _flat(result.output)
    turns = json.loads(report.read_text("utf-8"))["turns"]
    assert [t["passed"] for t in turns] == [True, True]

    result = runner.invoke(app, ["chat", "--space", "cli-chat"], input="hello\n/end\n/quit\n")
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "HiMaya,noted." in flat and "会话结束" in flat


@pytest.mark.parametrize(
    ("host", "token", "insecure", "starts"),
    [
        ("0.0.0.0", None, False, False),
        ("192.168.1.10", None, False, False),
        ("::", None, False, False),
        ("", None, False, False),
        ("127.0.0.1", None, False, True),
        ("127.0.0.2", None, False, True),
        ("::1", None, False, True),
        ("localhost", None, False, True),
        ("0.0.0.0", "fake-token", False, True),
        ("0.0.0.0", None, True, True),
    ],
)
def test_require_token(host: str, token: str | None, insecure: bool, starts: bool) -> None:
    import typer

    from cuemem.cli import _require_token

    if starts:
        _require_token(host, token, insecure)
    else:
        with pytest.raises(typer.Exit):
            _require_token(host, token, insecure)


@pytest.fixture
def served(isolated_paths: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Any]:
    """cuemem serve / proxy 不真的监听：记下交给 uvicorn.run 的应用。"""
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    monkeypatch.setenv("CUEMEM_PATHS__MEDIA", str(tmp_path / "media"))
    calls: list[Any] = []
    monkeypatch.setattr("uvicorn.run", lambda app_, **kwargs: calls.append(app_))
    return calls


@pytest.mark.parametrize("command", [["serve", "--no-ui", "--scenario", "plain"], ["proxy"]])
def test_non_loopback_bind_needs_a_token(
    served: list[Any], monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    """监听非本机地址又没有访问令牌：拒绝启动（Docker 镜像监听 0.0.0.0），
    除非加 --insecure-no-auth。"""
    result = runner.invoke(app, [*command, "--host", "0.0.0.0"])
    assert result.exit_code == 2 and served == []
    assert "CUEMEM_API_TOKEN" in result.output and "--insecure-no-auth" in _flat(result.output)
    result = runner.invoke(app, [*command, "--host", "0.0.0.0", "--insecure-no-auth"])
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, [*command, "--host", "127.0.0.1"]).exit_code == 0
    monkeypatch.setenv("CUEMEM_API_TOKEN", "fake-token")
    assert runner.invoke(app, [*command, "--host", "0.0.0.0"]).exit_code == 0
    assert len(served) == 3


def test_data_download_skips_optional_datasets(
    isolated_paths: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """不点名时只下载评测必需的数据集；WildChat（231 MB，N11）要点名才下载。"""
    fetched: list[str] = []

    def fake_download(name: str, directory: Path) -> Path:
        fetched.append(name)
        path = tmp_path / f"{name}.bin"
        path.write_bytes(b"x")
        return path

    monkeypatch.setattr("cuemem.cli.download_dataset", fake_download)
    assert runner.invoke(app, ["data", "download"]).exit_code == 0
    assert "wildchat" not in fetched and "longmemeval_s" in fetched
    assert runner.invoke(app, ["data", "download", "wildchat"]).exit_code == 0
    assert fetched[-1] == "wildchat"


def test_eval_errors_are_listed_once_per_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """评测结束时出错的组合按报错归并列出：外部基线的协议报错每道题各记一次，这里只列一次；
    后来重跑成功的不算；报错里的方括号原样显示（不当作终端标记）。"""
    import json

    from cuemem.cli import _print_errors

    mismatch = "ConfigMismatchError: x 是另一个协议的 mem0 写入……CUEMEM_PATHS__EVAL_SPACES=y"
    errors = [{"sample_id": f"q{i}", "method": "M0", "error": mismatch} for i in range(3)]
    errors.append({"sample_id": "q9", "method": "B4", "error": "RuntimeError: [boom]"})
    errors.append({"sample_id": "q8", "method": "B4", "error": "RuntimeError: fixed later"})
    (tmp_path / "errors.jsonl").write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors), "utf-8"
    )
    done = {"sample_id": "q8", "method": "B4"}
    (tmp_path / "results.jsonl").write_text(json.dumps(done) + "\n", "utf-8")
    _print_errors(tmp_path, limit=1)
    out = _flat(capsys.readouterr().out)
    assert out.count("CUEMEM_PATHS__EVAL_SPACES") == 1 and "M0×3" in out
    assert "另有1种报错" in out and "fixed" not in out
    _print_errors(tmp_path)
    assert "B4×1：RuntimeError:[boom]" in _flat(capsys.readouterr().out)
