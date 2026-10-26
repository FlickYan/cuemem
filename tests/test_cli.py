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
        MemorySystem,
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
        MemorySystem,
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

    from cuemem.api.auth import Keys
    from cuemem.cli import _require_token

    keys = Keys.from_token(token)
    if starts:
        _require_token(host, keys, insecure)
    else:
        with pytest.raises(typer.Exit):
            _require_token(host, keys, insecure)


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
    除非加 --insecure-no-auth。被拒绝时不打开记忆系统：不接着做上次没做完的抽取（缺 Key 时
    也不白白记一次失败）。"""
    import sqlite3

    def jobs() -> list[tuple[Any, ...]]:
        with sqlite3.connect(os.environ["CUEMEM_PATHS__DB"]) as conn:
            return conn.execute(
                "SELECT id, status, attempts, error FROM jobs ORDER BY id"
            ).fetchall()

    assert runner.invoke(app, ["ingest", str(TOY_CONVERSATION)]).exit_code == 0  # 留下待处理的任务
    pending = jobs()
    assert pending and {status for _, status, _, _ in pending} == {"pending"}
    result = runner.invoke(app, [*command, "--host", "0.0.0.0"])
    assert result.exit_code == 2 and served == []
    assert "CUEMEM_API_TOKEN" in result.output and "--insecure-no-auth" in _flat(result.output)
    assert jobs() == pending
    result = runner.invoke(app, [*command, "--host", "0.0.0.0", "--insecure-no-auth"])
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, [*command, "--host", "127.0.0.1"]).exit_code == 0
    monkeypatch.setenv("CUEMEM_API_TOKEN", "fake-token")
    assert runner.invoke(app, [*command, "--host", "0.0.0.0"]).exit_code == 0
    assert len(served) == 3


@pytest.mark.parametrize("command", [["serve", "--no-ui", "--scenario", "plain"], ["proxy"]])
def test_loopback_bind_checks_the_host(
    served: list[Any], monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    """监听本机地址时只接受 Host 为本机的请求（DNS 重绑定）；监听别的地址时不查 Host（Docker、
    局域网）。CUEMEM_ALLOWED_HOSTS 追加的主机名作为 Host 与 Origin 都放行。"""
    from fastapi.testclient import TestClient

    assert runner.invoke(app, command).exit_code == 0
    assert runner.invoke(app, [*command, "--host", "0.0.0.0", "--insecure-no-auth"]).exit_code == 0
    monkeypatch.setenv("CUEMEM_ALLOWED_HOSTS", "mem.example, cuemem.lan")
    assert runner.invoke(app, command).exit_code == 0
    local, lan, extended = (TestClient(app_) for app_ in served)
    for host in ("127.0.0.1:7860", "localhost:7860", "[::1]:7860"):
        assert local.get("/v1/models", headers={"Host": host}).status_code == 200
    rebound = {"Host": "attacker.example:7860", "Origin": "http://attacker.example:7860"}
    assert local.get("/v1/models", headers=rebound).status_code == 400
    assert lan.get("/v1/models", headers={"Host": "192.168.1.10:7860"}).status_code == 200
    assert extended.get("/v1/models", headers={"Host": "cuemem.lan"}).status_code == 200
    proxied = {"Host": "127.0.0.1:7860", "Origin": "https://mem.example"}
    assert extended.get("/v1/models", headers=proxied).status_code == 200
    assert local.get("/v1/models", headers=proxied).status_code == 403


@pytest.mark.parametrize("command", [["serve", "--no-ui", "--scenario", "plain"], ["proxy"]])
def test_unknown_models_are_rejected_on_request(
    served: list[Any], monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    """--unknown-model reject 或配置 proxy.unknown_model（环境变量 CUEMEM_PROXY__UNKNOWN_MODEL）：
    记忆代理对不认识的模型返回 400；缺省照旧改用 agent 角色的模型。写错的值在启动之前报错。"""
    from fastapi.testclient import TestClient

    from cuemem.api import proxy as proxy_module

    made: list[Any] = []

    class Recorded(proxy_module.MemoryProxy):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(proxy_module, "MemoryProxy", Recorded)
    result = runner.invoke(app, [*command, "--unknown-model", "nope"])
    assert result.exit_code == 1 and "--unknown-model" in result.output and served == []
    assert runner.invoke(app, [*command, "--unknown-model", "reject"]).exit_code == 0
    monkeypatch.setenv("CUEMEM_PROXY__UNKNOWN_MODEL", "reject")
    assert runner.invoke(app, command).exit_code == 0
    assert runner.invoke(app, [*command, "--unknown-model", "remap"]).exit_code == 0  # 参数优先
    assert [p.unknown_model for p in made] == ["reject", "reject", "remap"]
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    for app_ in served[:2]:  # 被拒绝的请求不转发（改用默认模型的那个会真的发给上游，不在这里发）
        client = TestClient(app_, base_url="http://127.0.0.1:7860")
        response = client.post("/v1/chat/completions", json=body)
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"


def test_serve_ends_idle_sessions(served: list[Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """cuemem serve --idle-minutes（默认 30）：界面与 /api 的会话、/api/v1 的 run 与 /v1 记忆代理的
    会话闲置这么久就结束、交给抽取。不大于 0 时在打开记忆系统之前报错。"""
    from cuemem.api import app as app_module
    from cuemem.api import proxy as proxy_module

    idle: list[Any] = []
    create_app = app_module.create_app

    def recorded(*args: Any, **kwargs: Any) -> Any:
        idle.append(kwargs["idle_minutes"])
        return create_app(*args, **kwargs)

    class Recorded(proxy_module.MemoryProxy):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            idle.append(self.idle_s)

    monkeypatch.setattr(app_module, "create_app", recorded)
    monkeypatch.setattr(proxy_module, "MemoryProxy", Recorded)
    command = ["serve", "--no-ui", "--scenario", "plain"]
    result = runner.invoke(app, [*command, "--idle-minutes", "0"])
    assert result.exit_code == 1 and "--idle-minutes" in result.output and served == []
    assert runner.invoke(app, command).exit_code == 0
    assert runner.invoke(app, [*command, "--idle-minutes", "5"]).exit_code == 0
    assert idle == [1800.0, 30.0, 300.0, 5.0] and len(served) == 2


@pytest.mark.parametrize("command", [["serve", "--no-ui", "--scenario", "plain"], ["proxy"]])
def test_scoped_keys_from_the_environment(
    served: list[Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: list[str]
) -> None:
    """CUEMEM_API_KEYS_FILE（或 CUEMEM_API_KEYS）给出限定作用域的令牌：只有它们、没有
    CUEMEM_API_TOKEN 也算设了令牌，可以监听非本机地址；每个令牌只能访问自己的用户。文件写错时
    在打开记忆系统之前报错（退出码 1），不带令牌本身。"""
    import sqlite3

    from fastapi.testclient import TestClient

    keys = tmp_path / "keys.yaml"
    keys.write_text(
        "keys:\n  - {name: alice-app, token: alice-0123456789, scopes: [{user: alice}]}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CUEMEM_API_KEYS_FILE", str(keys))
    result = runner.invoke(app, [*command, "--host", "0.0.0.0"])
    assert result.exit_code == 0, result.output
    assert "alice-app" in result.output and "alice-0123456789" not in result.output
    # 命令已经退出、记忆系统已关：只看在碰记忆之前就做完的检查
    client = TestClient(served[-1])
    alice = {"Authorization": "Bearer alice-0123456789"}
    body = {"messages": [{"role": "user", "content": "hi"}], "user": "bob"}
    response = client.post("/v1/chat/completions", json=body, headers=alice)
    assert response.status_code == 403 and response.json()["error"]["code"] == "forbidden"
    assert client.get("/v1/models", headers={"Authorization": "Bearer x"}).status_code == 401
    if command[0] == "serve":
        assert client.get("/api/spaces/bob/facts", headers=alice).status_code == 403

    assert runner.invoke(app, ["ingest", str(TOY_CONVERSATION)]).exit_code == 0  # 留下待处理的任务
    with sqlite3.connect(os.environ["CUEMEM_PATHS__DB"]) as conn:
        pending = conn.execute("SELECT id, status FROM jobs ORDER BY id").fetchall()
    keys.write_text("keys:\n  - {name: a, token: alice-0123456789, scopes: []}\n", encoding="utf-8")
    result = runner.invoke(app, command)
    assert result.exit_code == 1 and "错误" in result.output and "scopes" in result.output
    assert "alice-0123456789" not in result.output and len(served) == 1
    with sqlite3.connect(os.environ["CUEMEM_PATHS__DB"]) as conn:
        assert conn.execute("SELECT id, status FROM jobs ORDER BY id").fetchall() == pending
    monkeypatch.setenv("CUEMEM_API_KEYS", "keys: []")
    result = runner.invoke(app, command)
    assert result.exit_code == 1 and "只能设一个" in result.output


def test_serve_falls_back_to_the_plain_assistant_outside_a_checkout(
    served: list[Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """导购演示的商品目录只在源码仓库里：pip 安装后 cuemem serve 默认改用通用助手照常起来（以前
    直接崩在 FileNotFoundError 上）；明说 --scenario shop 时给出可读的报错。仓库里默认仍是导购。"""
    from cuemem import cli

    assert cli._default_scenario() == "shop"
    missing = tmp_path / "site-packages" / "demo" / "shop" / "catalog.json"
    monkeypatch.setattr("cuemem.api.service.SHOP_CATALOG", missing)
    result = runner.invoke(app, ["serve", "--no-ui"])
    assert result.exit_code == 0, result.output
    assert len(served) == 1 and "通用助手" in result.output
    result = runner.invoke(app, ["serve", "--no-ui", "--scenario", "shop"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "错误" in result.output and "--scenarioplain" in _flat(result.output)
    assert len(served) == 1


def test_serve_without_the_ui_extra_serves_the_api(
    served: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """界面是可选依赖 ui：没装 Gradio 时 cuemem serve 照常提供接口与记忆代理，并说明怎么装
    （以前 gradio 是核心依赖，没装时 serve 崩在 ImportError 上）。"""
    import sys

    from fastapi.testclient import TestClient

    monkeypatch.setitem(sys.modules, "gradio", None)  # 当作没装
    result = runner.invoke(app, ["serve", "--scenario", "plain"])
    assert result.exit_code == 0, result.output
    assert "uvsync--extraui" in _flat(result.output) and "界面：" not in result.output
    (app_,) = served
    client = TestClient(app_, headers={"Host": "127.0.0.1:7860"})
    assert client.get("/api/health").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.get("/").status_code == 404  # 根路径上没有界面


def test_eval_commands_say_which_extra_to_install(
    isolated_paths: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """评测报告要画图：没装可选依赖 eval（matplotlib）时，eval run 与 eval report 一开始就说明
    要装什么，不白跑一遍评测；只构建空间、判分抽检不需要它。"""
    import json
    import sys

    monkeypatch.setitem(sys.modules, "matplotlib", None)  # 当作没装
    for args in (["eval", "report", "x"], ["eval", "run", "--name", "x", "--methods", "B1"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 1, result.output
        assert "uvsync--extraeval" in _flat(result.output), result.output
    monkeypatch.setattr("cuemem.config.source_checkout", lambda: False)
    result = runner.invoke(app, ["eval", "report", "x"])
    assert 'pipinstall"cuemem[eval]"' in _flat(result.output), result.output
    result = runner.invoke(app, ["eval", "run", "--name", "x", "--build-only"])
    assert "cuemem[eval]" not in result.output and "缺少数据文件" in result.output
    row = {
        "sample_id": "q1",
        "method": "B1",
        "question_type": "single-hop",
        "correct": True,
        "question": "?",
        "answer": "a",
        "response": "a",
        "judge_raw": "yes",
    }
    (tmp_path / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (tmp_path / "run.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["eval", "audit", str(tmp_path), "--n", "1"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "command", [["serve", "--no-ui", "--scenario", "plain"], ["chat", "--space", "s"], ["proxy"]]
)
def test_missing_key_is_a_friendly_error_for_servers(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: list[str]
) -> None:
    """缺 Key 时 serve、chat 与 proxy 只打印原因并以 1 退出，不再抛出调用栈（Docker 里会反复
    重启）；pip 安装后的提示是 cuemem configure。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr("cuemem.config.source_checkout", lambda: False)
    calls: list[Any] = []
    monkeypatch.setattr("uvicorn.run", lambda app_, **kwargs: calls.append(app_))
    result = runner.invoke(app, command, input="/quit\n")
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    flat = _flat(result.output)
    assert "错误" in flat and "DASHSCOPE_API_KEY" in flat and "请运行cuememconfigure" in flat
    assert calls == []


def test_demo_fresh_waits_for_resumed_consolidation(
    isolated_paths: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """上一次演示中途退出、留下待处理的任务：后台线程一启动就接着巩固这个空间。--fresh 等它做完
    再清空，它不会在清空之后把事实与向量写回来。"""
    import threading
    from datetime import datetime

    from fakes import FakeAPI
    from scripted import ScriptedModel, echo_extraction

    from cuemem.api.service import ChatService
    from cuemem.core.models import SessionIn, TurnIn
    from cuemem.memory import MemorySystem

    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    monkeypatch.setenv("CUEMEM_PATHS__MEDIA", str(tmp_path / "media"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "fake-qwen")
    entered, release = threading.Event(), threading.Event()
    model = ScriptedModel(echo_extraction)

    def reply(body: dict[str, Any]) -> Any:
        if "Berlin" in str(body["messages"]) and not release.is_set():
            entered.set()
            release.wait(10)  # 接着做的抽取还在路上
        return model(body)

    api = FakeAPI()
    api.chat_replies = [reply]
    with MemorySystem.from_settings(load_settings(), http_client=api.client()) as before:
        session = SessionIn(
            session_id="s1",
            started_at=datetime(2026, 9, 1, 10),
            turns=[TurnIn(role="user", text="I live in Berlin")],
        )
        before.append("cli-demo", session, 0)  # 登记了任务，没来得及巩固就退出了
    original = MemorySystem.from_settings
    monkeypatch.setattr(
        MemorySystem,
        "from_settings",
        lambda settings, **kwargs: original(settings, http_client=api.client()),
    )
    create = ChatService.create

    def resumed(settings: Any, **kwargs: Any) -> Any:
        service = create(settings, **kwargs)
        assert entered.wait(10)  # 后台线程正在巩固 cli-demo
        threading.Timer(0.3, release.set).start()
        return service

    monkeypatch.setattr(ChatService, "create", resumed)
    script = tmp_path / "script.json"
    script.write_text('{"space": "cli-demo", "sessions": []}', encoding="utf-8")
    result = runner.invoke(app, ["demo", str(script), "--fresh", "--report", str(tmp_path / "r")])
    assert result.exit_code == 0, result.output
    with MemorySystem.from_settings(load_settings(), http_client=api.client()) as after:
        assert after.db.get_facts("cli-demo") == []
        assert after.vectors.unit_ids("cli-demo") == []
        assert after.db.get_jobs("cli-demo") == []


def test_demo_uses_the_catalog_next_to_its_script(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cuemem demo 用剧本旁边的商品目录（demo/shop/ 里就是这样）：不在源码仓库里也能回放；
    旁边没有时才用仓库里的。"""
    from cuemem.api.service import ChatService
    from cuemem.errors import CueMemError

    seen: list[Path | None] = []

    def fake_create(settings: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["catalog_path"])
        raise CueMemError("stop")

    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(ChatService, "create", fake_create)
    script = tmp_path / "shop" / "script.json"
    script.parent.mkdir()
    script.write_text('{"space": "demo", "sessions": []}', encoding="utf-8")
    result = runner.invoke(app, ["demo", str(script)])
    assert result.exit_code == 1 and "stop" in result.output
    (script.parent / "catalog.json").write_text("{}", encoding="utf-8")
    assert runner.invoke(app, ["demo", str(script)]).exit_code == 1
    assert seen == [None, script.parent / "catalog.json"]


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

    monkeypatch.setattr("cuemem.evals.datasets.download", fake_download)
    assert runner.invoke(app, ["data", "download"]).exit_code == 0
    assert "wildchat" not in fetched and "longmemeval_s" in fetched
    assert runner.invoke(app, ["data", "download", "wildchat"]).exit_code == 0
    assert fetched[-1] == "wildchat"


def test_data_download_reports_a_checksum_mismatch(
    isolated_paths: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """下载的文件与固定的版本不同时，打印"错误：原因"、退出码 1，不打印调用栈。"""
    from cuemem.errors import DatasetChecksumError

    def fake_download(name: str, directory: Path) -> Path:
        raise DatasetChecksumError("下载的 locomo10.json 与固定的版本不同：sha256 应为 [abc]")

    monkeypatch.setattr("cuemem.evals.datasets.download", fake_download)
    result = runner.invoke(app, ["data", "download", "locomo"])
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit), result.exception
    assert "错误：下载的locomo10.json与固定的版本不同：sha256应为[abc]" in _flat(result.output)


@pytest.mark.parametrize(
    ("errors", "stopped", "code"), [(0, False, 0), (2, False, 3), (0, True, 4), (1, True, 4)]
)
def test_eval_run_fail_on_error(
    isolated_paths: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    errors: int,
    stopped: bool,
    code: int,
) -> None:
    """eval run --fail-on-error（复现脚本用）：有组合出错时退出码 3、达到花费上限提前停止时 4，
    报告照常生成；不加这个参数时退出码照旧是 0。"""
    from types import SimpleNamespace

    from cuemem import cli
    from cuemem.evals.runner import RunSummary

    class FakeRunner:
        def __init__(self, settings: Any, config: Any, **kwargs: Any) -> None:
            pass

        def run(self, evalset: Any) -> RunSummary:
            return RunSummary(
                run_dir=tmp_path,
                planned=3,
                completed=3 - errors,
                errors=errors,
                spent=0.0,
                nominal_cost=0.0,
                elapsed_s=0.0,
                stopped_by_budget=stopped,
            )

    reports: list[Path] = []

    def fake_report(run_dir: Path, **kwargs: Any) -> Path:
        reports.append(run_dir)
        return run_dir / "report.md"

    monkeypatch.setattr("cuemem.evals.runner.EvalRunner", FakeRunner)
    evalset = SimpleNamespace(dataset="longmemeval_s", samples=[])
    monkeypatch.setattr(cli, "_load_evalset", lambda *args: evalset)
    monkeypatch.setattr("cuemem.evals.report.build_report", fake_report)
    args = ["eval", "run", "--name", "x", "--methods", "B1"]
    assert runner.invoke(app, args).exit_code == 0
    result = runner.invoke(app, [*args, "--fail-on-error"])
    assert result.exit_code == code, result.output
    assert len(reports) == 2  # 报告照常生成
    assert ("--fail-on-error" in result.output) == bool(code)
    assert runner.invoke(app, [*args, "--build-only"]).exit_code == 0
    assert runner.invoke(app, [*args, "--build-only", "--fail-on-error"]).exit_code == code


def _reader_and_judge(rejected: set[str]) -> Any:
    """模拟模型，同时扮演回答与判分：回答时上下文里有 golden retriever 就答出来、否则说不知道；
    问题在 rejected 里时像内容审核拒绝（这个组合记为出错）。判分时回答里有 golden retriever
    就判对。"""
    import json

    from fakes import REJECT

    def reply(body: dict[str, Any]) -> str:
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("I will give you"):
            return "yes" if "golden retriever" in prompt.split("Model Response:")[-1] else "no"
        context, _, question = prompt.rpartition("Question: ")
        if question.strip() in rejected:
            return REJECT
        found = "golden retriever" in context
        answer = "golden retriever" if found else "I don't know."
        return json.dumps({"answer": answer, "citations": [], "evidence_sufficient": found})

    return reply


@pytest.fixture
def locomo_eval(clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """离线的 cuemem eval run：迷你 LoCoMo（4 道题，第 3 道 conv-test-q2 是对抗题）放进数据目录，
    评测运行器照旧，只把模型请求发给模拟接口（离线哈希向量器，不访问网络）。
    返回要模拟内容审核拒绝的问题（集合，测试里随时增删）。"""
    import shutil

    from fakes import FakeAPI

    from cuemem.evals.runner import EvalRunner

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "256")
    datasets = load_settings().paths.datasets
    datasets.mkdir(parents=True)
    shutil.copy(Path(__file__).parent / "fixtures" / "locomo_mini.json", datasets / "locomo10.json")
    rejected: set[str] = set()
    fake_api = FakeAPI()
    fake_api.chat_replies = [_reader_and_judge(rejected)]

    class OfflineRunner(EvalRunner):
        def __init__(self, settings: Any, config: Any, **kwargs: Any) -> None:
            super().__init__(settings, config, http_client=fake_api.client(), **kwargs)

    monkeypatch.setattr("cuemem.evals.runner.EvalRunner", OfflineRunner)
    return rejected


def _eval_rows(name: str, filename: str = "results.jsonl") -> list[dict[str, Any]]:
    from cuemem.evals.report import load_rows

    return load_rows(load_settings().paths.eval_runs / name, filename)


def _answered(name: str) -> set[tuple[str, str]]:
    return {(r["sample_id"].removeprefix("conv-test-"), r["method"]) for r in _eval_rows(name)}


LOCOMO_RUN = ["eval", "run", "--dataset", "locomo", "--split", "all", "--peak", "allow"]


def test_eval_run_selects_the_protocol_questions(locomo_eval: set[str]) -> None:
    """cuemem eval run 走真实的评测运行器（迷你 LoCoMo、模拟模型）：只评点名的方法；LoCoMo 默认
    不评对抗题，--types 点名时照评；--limit 只取前 N 题；eval probe 与评测选同一批题；run.json、
    结果与报告都写出来。规则在 datasets.protocol_samples，四处入口共用。"""
    import json

    result = runner.invoke(app, [*LOCOMO_RUN, "--name", "r", "--methods", "B0, B1,B2,"])
    assert result.exit_code == 0, result.output
    assert "3题×['B0','B1','B2']" in _flat(result.output)
    assert _answered("r") == {(q, m) for q in ("q0", "q1", "q3") for m in ("B0", "B1", "B2")}
    run_dir = load_settings().paths.eval_runs / "r"
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["samples"] == 3 and info["config"]["methods"] == ["B0", "B1", "B2"]
    assert (run_dir / "report.md").exists() and f"报告：{run_dir / 'report.md'}" in _flat(
        result.output
    )
    correct = {(r["sample_id"], r["method"]): r["correct"] for r in _eval_rows("r")}
    assert correct[("conv-test-q0", "B1")] and not correct[("conv-test-q0", "B0")]

    result = runner.invoke(
        app, [*LOCOMO_RUN, "--name", "adv", "--methods", "B1", "--types", "adversarial"]
    )
    assert result.exit_code == 0, result.output
    assert _answered("adv") == {("q2", "B1")}
    args = ["--name", "two", "--methods", "B1", "--types", "single-hop, adversarial"]
    assert runner.invoke(app, [*LOCOMO_RUN, *args]).exit_code == 0
    assert _answered("two") == {("q0", "B1"), ("q2", "B1")}
    args = ["--name", "lim", "--methods", "B0", "--limit", "2"]
    assert runner.invoke(app, [*LOCOMO_RUN, *args]).exit_code == 0
    assert _answered("lim") == {("q0", "B0"), ("q1", "B0")}  # 对抗题筛掉之后的前 2 题

    # 探针：r 的 B2 已经建好了空间，探针与评测选同一批题（不含对抗题）
    probe = ["eval", "probe", "--dataset", "locomo", "--split", "all", "--name", "r"]
    result = runner.invoke(app, probe)
    assert result.exit_code == 0 and "完成3题，出错0题" in _flat(result.output), result.output
    probed = {row["sample_id"] for row in _eval_rows("r", "probe.jsonl")}
    assert probed == {"conv-test-q0", "conv-test-q1", "conv-test-q3"}


def test_eval_run_resumes_and_reports_incomplete_runs(locomo_eval: set[str]) -> None:
    """出错的组合不算完成，--fail-on-error 时退出码 3；同名再跑只补出错的那一个，答完之后再跑什么
    都不做。达到花费上限提前停止时退出码 4。配置错误（未知的数据集、方法、--peak，缺数据文件）
    退出码 1，什么都不写。"""
    import json

    locomo_eval.add("When did Bob run a marathon?")
    args = [*LOCOMO_RUN, "--name", "r", "--methods", "B1", "--fail-on-error"]
    result = runner.invoke(app, args)
    assert result.exit_code == 3, result.output
    assert "完成2/3个组合，出错1个" in _flat(result.output)
    assert _answered("r") == {("q0", "B1"), ("q3", "B1")}
    assert [e["sample_id"] for e in _eval_rows("r", "errors.jsonl")] == ["conv-test-q1"]

    locomo_eval.clear()
    result = runner.invoke(app, args)  # 续跑：只补上次出错的组合
    assert result.exit_code == 0, result.output
    assert "完成1/1个组合，出错0个" in _flat(result.output)
    assert _answered("r") == {(q, "B1") for q in ("q0", "q1", "q3")}
    assert len(_eval_rows("r")) == 3
    result = runner.invoke(app, args)
    assert result.exit_code == 0 and "完成0/0个组合" in _flat(result.output), result.output
    info = json.loads((load_settings().paths.eval_runs / "r" / "run.json").read_text("utf-8"))
    assert len(info["resumed"]) == 2

    # 花费上限：第一个组合（B0 还没有缓存，真的调用了模拟接口）花掉预算之后不再开始新题
    args = [*LOCOMO_RUN, "--name", "b", "--methods", "B0", "--workers", "1", "--max-cost", "1e-9"]
    result = runner.invoke(app, [*args, "--fail-on-error"])
    assert result.exit_code == 4, result.output
    assert "已达到花费上限，提前停止" in _flat(result.output) and len(_eval_rows("b")) == 1
    args[args.index("b")] = "b2"
    assert runner.invoke(app, args).exit_code == 0  # 不加 --fail-on-error 时退出码照旧是 0

    runs = load_settings().paths.eval_runs
    before = sorted(p.name for p in runs.iterdir())
    for bad, message in (
        (["--dataset", "nope"], "未知数据集nope"),
        (["--peak", "sometimes"], "--peak只能是wait、warn或allow"),
        (["--dataset", "longmemeval_s"], "缺少数据文件"),
    ):
        result = runner.invoke(app, ["eval", "run", "--split", "all", "--name", "x", *bad])
        assert result.exit_code == 1 and message in _flat(result.output), result.output
    result = runner.invoke(app, [*LOCOMO_RUN, "--name", "x", "--methods", "B1,B99"])
    assert result.exit_code == 1 and "未知方法" in str(result.exception)
    assert sorted(p.name for p in runs.iterdir()) == before  # 都没有建运行目录


def test_eval_probe_fail_on_error(
    clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """eval probe --fail-on-error（复现脚本用）：有题出错（空间没建好）时退出码 3；不加时照旧 0。
    以前探针出错照样退出码 0，复现脚本接着用少了的题重写 docs/results。"""
    import shutil

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-dashscope")
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "256")
    datasets = load_settings().paths.datasets
    datasets.mkdir(parents=True)
    fixtures = Path(__file__).parent / "fixtures"
    shutil.copy(fixtures / "locomo_mini.json", datasets / "locomo10.json")
    args = ["eval", "probe", "--dataset", "locomo", "--split", "all", "--name", "p"]
    result = runner.invoke(app, args)  # 空间都没有构建：每道题都出错
    assert result.exit_code == 0 and "出错" in result.output
    result = runner.invoke(app, [*args, "--fail-on-error"])
    assert result.exit_code == 3 and "--fail-on-error" in result.output


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
