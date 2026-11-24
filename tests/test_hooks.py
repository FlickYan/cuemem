"""Claude Code 的钩子（cuemem hook profile | recall | capture，第十轮体检 Q14）：把 Claude Code 文档
里每个事件的示例输入喂给命令，核对标准输出与写进记忆的东西。全部离线：钩子的 MemoryClient
接在同一个进程里的 /api/v1 应用上（TestClient）。"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fakes import PROJECT_ROOT, FakeAPI
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripted import ScriptedModel
from test_sdk import AUG, moved, where_i_live
from typer.testing import CliRunner

import cuemem.evidence as evidence
from cuemem import Memory, hooks
from cuemem.api.auth import Keys, guard
from cuemem.api.memories import install_error_format, memories_router
from cuemem.cli import app
from cuemem.config import Settings
from cuemem.memory import MemorySystem

runner = CliRunner()
URL = "http://testserver"
SESSION = "00893aaf-19fa-41d2-8238-13269b9b3ca0"
RUN = f"cc-{SESSION}"
PROMPT_ID = "550e8400-e29b-41d4-a716-446655440000"
# https://code.claude.com/docs/en/hooks 里各个事件的示例输入（2026-11 核对），session_id 换成了
# 一个真的会话 ID；prompt_id 是所有事件共有的字段，第一句话之前没有
COMMON = {
    "session_id": SESSION,
    "transcript_path": f"/Users/dev/.claude/projects/-Users-dev-shop/{SESSION}.jsonl",
    "cwd": "/Users/dev/shop",
}
SESSION_START = {
    **COMMON,
    "hook_event_name": "SessionStart",
    "source": "resume",
    "model": "claude-model-id",
    "seconds_since_last_response": 5400,
    "context_tokens": 182340,
    "prompt_cache_likely_expired": True,
    "estimated_cache_write_usd": 1.1396,
}
USER_PROMPT = {
    **COMMON,
    "prompt_id": PROMPT_ID,
    "permission_mode": "default",
    "hook_event_name": "UserPromptSubmit",
    "prompt": "Write a function to calculate the factorial of a number",
}
STOP = {
    **COMMON,
    "prompt_id": PROMPT_ID,
    "permission_mode": "default",
    "hook_event_name": "Stop",
    "stop_hook_active": True,
    "last_assistant_message": "I've completed the refactoring. Here's a summary...",
    "background_tasks": [
        {
            "id": "task-001",
            "type": "shell",
            "status": "running",
            "description": "tail logs",
            "command": "tail -f /var/log/syslog",
        }
    ],
    "session_crons": [
        {"id": "cron-001", "schedule": "0 9 * * 1-5", "recurring": True, "prompt": "check"}
    ],
}
SESSION_END = {**COMMON, "hook_event_name": "SessionEnd", "reason": "other"}


@pytest.fixture
def system(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[MemorySystem]:
    fake_api.chat_replies = [ScriptedModel(where_i_live, reconcile=moved)]
    opened = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    yield opened
    opened.close()


class _Wire:
    """钩子拿到的 HTTP 连接：接在应用上的 TestClient，记下每个请求；钩子关它时不真的关。"""

    def __init__(self, http: TestClient) -> None:
        self.http = http
        self.timeout: Any = None
        self.sent: list[tuple[str, str]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Any:
        self.sent.append((method, url.removeprefix(URL + "/api/v1")))
        return self.http.request(method, url, **kwargs)

    def close(self) -> None:
        pass


@pytest.fixture
def served(system: MemorySystem, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Memory, _Wire]]:
    """一个开着的 /api/v1（不要令牌），钩子的连接接在它上面。给出服务端的 Memory 与连接。"""
    for name in (hooks.USER_ENV, hooks.AGENT_ENV, hooks.URL_ENV, "CUEMEM_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    memory = Memory.from_system(system)
    api = FastAPI()
    api.include_router(memories_router(memory, dependencies=guard(Keys())))
    install_error_format(api)
    with TestClient(api) as http:
        wire = _Wire(http)
        monkeypatch.setattr(hooks, "http_client", lambda timeout: wire)
        yield memory, wire


def hook(*args: str, event: Any = None, user: str | None = "dev", raw: str | None = None) -> Any:
    """运行 cuemem hook …，标准输入是事件的 JSON（raw 时原样）。退出码总是 0。"""
    scope = ["--user", user, "--url", URL] if user else ["--url", URL]
    text = raw if raw is not None else json.dumps(event)
    result = runner.invoke(app, ["hook", *args, *scope], input=text)
    assert result.exit_code == 0, (result.stdout, result.stderr)
    return result


def turns(memory: Memory, run_id: str = RUN, user: str = "dev") -> list[tuple[str, str]]:
    return [(t.role, t.text) for t in memory.system.db.get_turns(user, run_id)]


# ---------- 文档里的示例输入 ----------


def test_profile_prints_the_profile_block(served: tuple[Memory, _Wire]) -> None:
    """SessionStart：标准输出就是 <memory> 块（说明、画像、<memory 转义），与 context 在 L0 时
    给的逐字相同；空间里还没有画像时什么都不输出。只读，不开 run。"""
    memory, wire = served
    assert hook("profile", event=SESSION_START).stdout == ""
    memory.add("I live in Berlin", user_id="dev", timestamp=AUG, wait=True)
    wire.sent.clear()
    out = hook("profile", event=SESSION_START).stdout
    assert out == (
        f"<memory>\n{evidence.MEMORY_NOTE}\n\n"
        "About the user (core memory, current facts):\n- The user lives in Berlin.\n</memory>\n"
    )
    assert out == memory.context("x", user_id="dev", mode="l0")["context"] + "\n"
    assert wire.sent == [("GET", "/profile")]
    assert hook("profile", event={**SESSION_START, "source": "fork"}).stdout == out
    assert hook("profile", event=USER_PROMPT).stdout == ""  # 别的事件不归它管


def test_recall_prints_the_evidence_block(served: tuple[Memory, _Wire]) -> None:
    """UserPromptSubmit：标准输出是 context 给的证据块（纯文本，不是 JSON），预算 8000 个字符、
    排除本次会话的 run；不带 --capture 时什么都不写。"""
    memory, wire = served
    assert hook("recall", event=USER_PROMPT).stdout == ""  # 还没有记忆
    note = "The factorial function lives in mathutils.py and must stay iterative."
    memory.add(note, user_id="dev", run_id="earlier", timestamp=AUG, infer=False)
    wire.sent.clear()
    out = hook("recall", "--mode", "l1", event=USER_PROMPT).stdout
    expected = memory.context(
        USER_PROMPT["prompt"], user_id="dev", mode="l1", max_chars=8000, exclude_run_id=RUN
    )
    assert out == expected["context"] + "\n" and note in out
    assert out.startswith(f"<memory>\n{evidence.MEMORY_NOTE}") and out.endswith("\n</memory>\n")
    assert wire.sent == [("POST", "/context")]
    assert turns(memory) == [] and (("dev", RUN) not in memory._runs)
    # 缺省的 --mode auto、--min-level l0：照控制器的决定，与记忆代理相同
    plain = memory.context(
        USER_PROMPT["prompt"], user_id="dev", min_level="l0", max_chars=8000, exclude_run_id=RUN
    )
    assert hook("recall", event=USER_PROMPT).stdout == (
        plain["context"] + "\n" if plain["context"] else ""
    )


def test_capture_never_prints(served: tuple[Memory, _Wire]) -> None:
    """Stop：last_assistant_message 作为助手的一句写进这个会话的 run，标准输出为空。同一个事件
    送来两次（Stop 会重复触发）只写一次。SessionEnd 带 --end 时结束 run，整段交给抽取。"""
    memory, wire = served
    for _ in range(2):
        assert hook("capture", event=STOP).stdout == ""
    assert turns(memory) == [("assistant", STOP["last_assistant_message"])]
    assert ("dev", RUN) in memory._runs
    assert wire.sent == [("POST", "/runs"), ("POST", "/memories")] * 2
    assert hook("capture", event=SESSION_END).stdout == ""  # 没有 --end：run 留着
    assert ("dev", RUN) in memory._runs
    assert hook("capture", "--end", event=SESSION_END).stdout == ""
    assert ("dev", RUN) not in memory._runs
    assert wire.sent[-1] == ("POST", "/runs/end")


def test_a_very_long_session_id_still_fits_the_run_id_limit(served: tuple[Memory, _Wire]) -> None:
    """run_id 最多 sdk.MAX_ID_CHARS 个字符（第十轮体检 Q7）：钩子自己拼的 "cc-" + session_id 超过
    时换成散列，不然这个会话的每一句都被服务端拒绝、钩子又不出声。不超过的原样。"""
    from cuemem.sdk import MAX_ID_CHARS

    memory, _ = served
    assert hooks.MAX_RUN_CHARS == MAX_ID_CHARS
    fits = "s" * (MAX_ID_CHARS - len(hooks.RUN_PREFIX))
    assert hooks._run_id(fits) == hooks.RUN_PREFIX + fits
    long_id = "s" * (MAX_ID_CHARS + 44)
    run_id = hooks._run_id(long_id)
    assert run_id.startswith(hooks.RUN_PREFIX) and len(run_id) <= MAX_ID_CHARS
    assert run_id != hooks._run_id(long_id + "x")
    stop = {**STOP, "session_id": long_id}
    assert hook("capture", event=stop).stdout == ""
    assert turns(memory, run_id) == [("assistant", STOP["last_assistant_message"])]
    assert ("dev", run_id) in memory._runs
    assert hook("capture", "--end", event={**SESSION_END, "session_id": long_id}).stdout == ""
    assert ("dev", run_id) not in memory._runs


def test_capture_options(served: tuple[Memory, _Wire]) -> None:
    """--no-assistant 不写回答（回答里可能有仓库里的代码与密钥）；--max-turn-chars 截断；空的
    回答、PreCompact 等别的事件什么都不做（压缩之后对话还在继续，run 不能结束）。"""
    memory, wire = served
    hook("capture", "--no-assistant", event=STOP)
    hook("capture", event={**STOP, "last_assistant_message": "  "})
    hook("capture", event={**STOP, "last_assistant_message": None})
    hook("capture", "--end", event={**COMMON, "hook_event_name": "PreCompact", "trigger": "auto"})
    hook("capture", event={**STOP, "hook_event_name": "SubagentStop"})
    assert wire.sent == [] and turns(memory) == []
    hook("capture", "--max-turn-chars", "12", event=STOP)
    assert turns(memory) == [("assistant", "I've complet")]


# ---------- 想起与写入 ----------


def test_recall_leaves_out_the_session_itself(served: tuple[Memory, _Wire]) -> None:
    """--capture：先想起、再把这句话写进这个会话的 run。同一个会话的三句话，第三次想起的块里
    没有前两句（它们已经在模型的上下文里），另一个会话里说过的照常给出。以前的设想只靠“先
    想起后写入”，那只挡得住当前这一句。没有 SessionStart 也行：写入之前自己 open_run，不抽取。"""
    memory, wire = served
    old = "The zebra flux capacitor timeout was 30 seconds in the old deployment."
    memory.add(old, user_id="dev", run_id="cc-older", timestamp=AUG, infer=False)
    prompts = [
        "How do I configure the zebra flux capacitor timeout?",
        "Raise the zebra flux capacitor timeout to 45 seconds please",
        "What is the zebra flux capacitor timeout now?",
    ]
    outs = []
    for i, prompt in enumerate(prompts):
        event = {**USER_PROMPT, "prompt": prompt, "prompt_id": f"p{i}"}
        outs.append(hook("recall", "--capture", "--mode", "l1", event=event).stdout)
        hook("capture", event={**STOP, "prompt_id": f"p{i}", "last_assistant_message": f"Done {i}"})
    for out in outs:
        assert old in out and "[m2]" not in out
        assert not any(prompt in out for prompt in prompts)
    assert turns(memory) == [
        ("user", prompts[0]),
        ("assistant", "Done 0"),
        ("user", prompts[1]),
        ("assistant", "Done 1"),
        ("user", prompts[2]),
        ("assistant", "Done 2"),
    ]
    # 不排除的话它们是搜得到的：这个 run 开着，写进去的马上可以检索
    plain = memory.context(prompts[2], user_id="dev", mode="l1")["context"]
    assert prompts[0] in plain and prompts[1] in plain
    # 整段对话只在结束时抽取一次：开着的时候没有事实，也没有等着抽取的会话
    assert ("dev", RUN) in memory._runs
    assert memory.get_all(user_id="dev")["results"] == []
    assert memory.status(user_id="dev")["pending"] == 0
    assert wire.sent.count(("POST", "/runs")) == 6
    hook("capture", "--end", event=SESSION_END)
    assert ("dev", RUN) not in memory._runs
    assert memory.status(user_id="dev")["pending"] == 1  # 测试里没有后台线程：等着抽取
    memory.consolidate(user_id="dev")
    facts = {fact["memory"] for fact in memory.get_all(user_id="dev")["results"]}
    assert facts == {old, *prompts}  # 抽取替身把用户的每句话记成一条事实


def test_recall_and_capture_survive_retries_and_odd_prompts(
    served: tuple[Memory, _Wire],
) -> None:
    """同一句话送来两次只写一次（幂等键：session_id、prompt_id）；没有 prompt_id 时按内容；
    空的话与 "/" 开头的命令不查也不写；--max-turn-chars 截断写下的话；--agent 分开空间。"""
    memory, wire = served
    for _ in range(2):
        hook("recall", "--capture", event=USER_PROMPT)
    assert turns(memory) == [("user", USER_PROMPT["prompt"])]
    bare = {k: v for k, v in USER_PROMPT.items() if k != "prompt_id"}
    for _ in range(2):
        hook("recall", "--capture", event={**bare, "prompt": "no id here"})
    assert [text for _, text in turns(memory)] == [USER_PROMPT["prompt"], "no id here"]
    wire.sent.clear()
    for prompt in ("", "   ", "/compact", " /deploy staging", None, 7):
        hook("recall", "--capture", event={**USER_PROMPT, "prompt": prompt})
    assert wire.sent == [] and len(turns(memory)) == 2
    event = {**USER_PROMPT, "prompt": "abcdefghij" * 3, "prompt_id": "long"}
    hook("recall", "--capture", "--max-turn-chars", "10", "--agent", "shop", event=event)
    assert turns(memory, user="shop/dev") == [("user", "abcdefghij")]  # agent_id/user_id
    assert len(turns(memory)) == 2


def test_recall_stays_under_the_output_cap(served: tuple[Memory, _Wire]) -> None:
    """60 段对话的空间：不限时证据块有两三万个字符，钩子的输出不超过 Claude Code 的 10,000；
    --max-chars 给得再大也一样，给得太小按 1000 算。"""
    memory, _ = served
    memory.add("I live in Berlin", user_id="dev", timestamp=AUG, wait=True)
    for i in range(60):
        said = f"Note {i}: I worked on the postgres migration step {i} and kept detail {i}. " * 5
        reply = f"Understood, postgres step {i} noted. " * 10
        chat = [{"role": "user", "content": said}, {"role": "assistant", "content": reply}]
        memory.add(chat, user_id="dev", timestamp=AUG + timedelta(hours=i), infer=False)
    event = {**USER_PROMPT, "prompt": "what did I keep about the postgres migration step?"}
    assert len(memory.context(event["prompt"], user_id="dev", mode="l1")["context"]) > 10_000
    out = hook("recall", "--mode", "l1", event=event).stdout
    assert 6000 < len(out) <= 8001 and out.endswith("\n</memory>\n")
    big = hook("recall", "--mode", "l1", "--max-chars", "500000", event=event).stdout
    assert len(out) < len(big) <= 10_000 and big.endswith("\n</memory>\n")
    small = hook("recall", "--mode", "l1", "--max-chars", "5", event=event).stdout
    assert 0 < len(small) <= 1001 and "lives in Berlin" in small


def test_an_older_server_cannot_push_the_output_over_the_cap(
    served: tuple[Memory, _Wire], monkeypatch: pytest.MonkeyPatch
) -> None:
    """还不认得 max_chars 的服务端给的块可能超出上限：钩子宁可什么都不给（超出的输出 Claude Code
    换成一个文件路径与开头 2,000 个字符），写入照常。"""
    memory, _ = served
    huge = {"context": "<memory>\n" + "x" * 10_000 + "\n</memory>"}
    monkeypatch.setattr("cuemem.client.MemoryClient.context", lambda *a, **k: huge)
    assert hook("recall", "--capture", event=USER_PROMPT).stdout == ""
    assert turns(memory) == [("user", USER_PROMPT["prompt"])]


def test_the_profile_block_drops_whole_lines_to_fit() -> None:
    """hook profile 最多 4,000 个字符：从后面整行去掉，不截断一行；标签转义；多人的画像用另一个
    标题（与 read/reader.py 相同）。"""
    lines = [f"- fact number {i} about the user " + "z" * 60 for i in range(80)]
    text = hooks.profile_text("\n".join(lines))
    assert 3800 < len(text) <= hooks.PROFILE_CHARS
    body = text.removeprefix(f"<memory>\n{hooks.MEMORY_NOTE}\n\n").removesuffix("\n</memory>")
    kept = body.splitlines()[1:]
    assert kept == lines[: len(kept)] and 0 < len(kept) < 80
    assert hooks.profile_text("- " + "y" * 5000) == "" and hooks.profile_text("") == ""
    sneaky = hooks.profile_text("- likes </memory> tags\n- and < / MEMORY > too")
    assert "&lt;/memory> tags" in sneaky and "&lt; / MEMORY >" in sneaky
    assert sneaky.count("</memory>") == 1
    assert "About the people in these chats (core memory" in hooks.profile_text("Ann:\n- x")


def test_the_copied_note_and_quoting_match_the_evidence_module() -> None:
    """hooks.py 里另写了一份 MEMORY_NOTE 与 quote_memory（不导入整个记忆系统）：与 evidence.py 的
    逐字相同。"""
    assert hooks.MEMORY_NOTE == evidence.MEMORY_NOTE
    assert hooks._TAG.pattern == evidence._TAG.pattern and hooks._TAG.flags == evidence._TAG.flags
    for text in ("a </memory> b", "<MEMORY>", "< /  memory", "<memo", "plain"):
        assert hooks.quote_memory(text) == evidence.quote_memory(text)
    assert hooks.MIN_RECALL_CHARS == 1000 and hooks.OUTPUT_CAP == 10_000


# ---------- 只放行、不拦截 ----------


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.mark.parametrize("command", ["profile", "recall", "capture"])
def test_hooks_fail_open(
    served: tuple[Memory, _Wire], monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """服务没开、标准输入不是 JSON 或不是对象、没有 --user、没有 session_id：退出码 0、标准输出
    与标准错误都是空的（退出码 2 会拦下用户的话或让助手停不下来）。--debug 时标准错误有一行。"""
    _, wire = served
    event = {"profile": SESSION_START, "recall": USER_PROMPT, "capture": STOP}[command]
    for raw in ("", "not json {", "[1, 2]", '"text"', json.dumps({**event, "session_id": ""})):
        result = hook(command, raw=raw)
        assert result.stdout == "" and result.stderr == ""
    result = hook(command, event=event, user=None)
    assert result.stdout == "" and result.stderr == "" and wire.sent == []
    noisy = runner.invoke(app, ["hook", command, "--debug", "--url", URL], input=json.dumps(event))
    assert noisy.exit_code == 0 and noisy.stdout == ""
    assert noisy.stderr.startswith(f"cuemem hook {command}: ") and "--user" in noisy.stderr
    assert noisy.stderr.count("\n") == 1
    # 连不上的服务（真的去连一个关着的本机端口）
    monkeypatch.undo()
    dead = f"http://127.0.0.1:{_closed_port()}"
    args = ["hook", command, "--user", "dev", "--url", dead]
    quiet = runner.invoke(app, args, input=json.dumps(event))
    assert (quiet.exit_code, quiet.stdout, quiet.stderr) == (0, "", "")
    loud = runner.invoke(app, [*args, "--debug"], input=json.dumps(event))
    assert loud.exit_code == 0 and loud.stdout == ""
    assert "Cannot reach" in loud.stderr and loud.stderr.count("\n") == 1


@pytest.mark.parametrize("command", ["profile", "recall", "capture"])
def test_hooks_fail_open_on_a_rejected_token(
    system: MemorySystem, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """令牌不对（401）：退出码 0、不输出；--debug 的那一行里没有令牌。"""
    token = "wrong-token-0123456789"
    monkeypatch.setenv("CUEMEM_API_TOKEN", token)
    api = FastAPI()
    keys = Keys.parse("keys: []\n", "admin-token-0123456789")
    api.include_router(memories_router(Memory.from_system(system), dependencies=guard(keys)))
    install_error_format(api)
    event = {"profile": SESSION_START, "recall": USER_PROMPT, "capture": STOP}[command]
    with TestClient(api) as http:
        monkeypatch.setattr(hooks, "http_client", lambda timeout: _Wire(http))
        extra = ["--capture"] if command == "recall" else []
        result = hook(command, *extra, "--debug", event=event)
    assert result.stdout == "" and token not in result.stderr
    assert result.stderr.startswith(f"cuemem hook {command}: MemoryAPIError: ")
    assert result.stderr.count("\n") == 1  # recall --capture 两步都被拒绝，也只写一行
    assert system.db.get_turns("dev", RUN) == []


def test_the_debug_line_hides_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--debug 的那一行：令牌与地址里的用户名、密码换成 ***，多行的出错收成一行、最多 300 字。"""
    monkeypatch.setenv("CUEMEM_API_TOKEN", "s3cret-token-0123456789")
    problem = RuntimeError(
        "Cannot reach https://dev:hunter2@mem.example.com/x\nBearer s3cret-token-0123456789 "
        + "pad " * 200
    )
    hooks._note("recall", hooks.HookOptions("dev", debug=True), problem)
    hooks._note("recall", hooks.HookOptions("dev"), problem)  # 没有 --debug：不写
    err = capsys.readouterr().err
    assert err.startswith("cuemem hook recall: RuntimeError: Cannot reach https://***@mem.example")
    assert "hunter2" not in err and "s3cret" not in err and "Bearer ***" in err
    assert err.count("\n") == 1 and len(err) < 340


def test_a_slow_step_does_not_outlive_the_deadline(
    served: tuple[Memory, _Wire], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--timeout 是整个钩子的时间：想起用完了它，就不再发写入的请求；已经写出的证据块还在
    （超时的钩子 Claude Code 会取消并丢掉全部输出）。每个请求的超时是剩下的时间。"""
    memory, wire = served
    memory.add("The factorial helper is in mathutils.py", user_id="dev", run_id="x", infer=False)
    clock = [1000.0]
    monkeypatch.setattr(hooks.time, "monotonic", lambda: clock[0])
    real = wire.request

    def slow(method: str, url: str, **kwargs: Any) -> Any:
        assert wire.timeout == pytest.approx(8.0)
        clock[0] += 9.0
        return real(method, url, **kwargs)

    monkeypatch.setattr(wire, "request", slow)
    out = hook("recall", "--capture", "--mode", "l1", event=USER_PROMPT).stdout
    assert "mathutils.py" in out
    assert wire.sent == [("POST", "/context")] and turns(memory) == []


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["bogus"],
        ["recall", "--max-chars", "abc"],
        ["recall", "--no-such-option"],
        ["capture", "--timeout", "soon"],
        ["profile", "extra-argument"],
    ],
)
def test_a_mistyped_hook_command_does_not_block(args: list[str]) -> None:
    """写错的选项、不认识的子命令：命令行解析出错的退出码是 2，Claude Code 把钩子的退出码 2 当作
    拦下（UserPromptSubmit 上用户的每句话都发不出去）。cuemem hook 下面一律退出码 0、标准输出
    为空，原因在标准错误写一行；--help 照常，别的命令组不变。"""
    result = runner.invoke(app, ["hook", *args], input=json.dumps(USER_PROMPT))
    assert result.exit_code == 0 and result.stdout == ""
    assert result.stderr.startswith("cuemem hook：") and "退出码 0" in result.stderr
    shown = runner.invoke(app, ["hook", "recall", "--help"])
    assert shown.exit_code == 0 and "--capture" in shown.stdout
    assert runner.invoke(app, ["keys", "--no-such-option"]).exit_code == 2


def test_the_hook_does_not_load_the_memory_system() -> None:
    """cuemem hook 每句话运行一次：只导入 cuemem.client，不加载配置、向量库、模型客户端与界面。"""
    script = (
        "import json, sys\n"
        "from cuemem.cli import app\n"
        "try:\n"
        "    app(['hook', 'recall', '--user', 'dev', '--url', sys.argv[1], '--capture'])\n"
        "except SystemExit as stop:\n"
        "    code = stop.code\n"
        "heavy = ('cuemem.memory', 'cuemem.sdk', 'cuemem.evidence', 'cuemem.api', 'qdrant_client',"
        " 'openai', 'numpy', 'fastapi', 'gradio')\n"
        "print(json.dumps([code, [m for m in heavy if m in sys.modules],"
        " 'cuemem.client' in sys.modules]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, f"http://127.0.0.1:{_closed_port()}"],
        input=json.dumps(USER_PROMPT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [0, [], True]


# ---------- 例子里的配置 ----------


def test_the_example_settings_run_the_hooks(served: tuple[Memory, _Wire]) -> None:
    """examples/claude-code-hooks.json：四个事件各一个 command 钩子，timeout 比 --timeout 的默认值
    长（SessionEnd 的 5 秒也让 Claude Code 把它的 1.5 秒预算放宽），不用 async；每条命令原样
    交给 cuemem 都能解析、退出码 0。"""
    memory, _ = served
    path = Path(PROJECT_ROOT) / "examples" / "claude-code-hooks.json"
    settings = json.loads(path.read_text("utf-8"))
    assert set(settings) == {"hooks"}
    events = {
        "SessionStart": SESSION_START,
        "UserPromptSubmit": USER_PROMPT,
        "Stop": STOP,
        "SessionEnd": SESSION_END,
    }
    assert set(settings["hooks"]) == set(events)
    memory.add("I live in Berlin", user_id="dev", timestamp=AUG, wait=True)
    for name, event in events.items():
        (group,) = settings["hooks"][name]
        (handler,) = group["hooks"]
        assert set(group) == {"hooks"} and set(handler) == {"type", "command", "timeout"}
        assert handler["type"] == "command"
        assert handler["timeout"] == (5 if name == "SessionEnd" else 10)
        words = handler["command"].split()
        assert words[:2] == ["cuemem", "hook"] and "--user" not in words
        result = runner.invoke(
            app, [*words[1:], "--url", URL], input=json.dumps(event), env={hooks.USER_ENV: "dev"}
        )
        assert result.exit_code == 0 and result.stderr == ""
        if name == "SessionStart":
            assert result.stdout.startswith("<memory>\n")
        elif name != "UserPromptSubmit":
            assert result.stdout == ""
    assert turns(memory) == [
        ("user", USER_PROMPT["prompt"]),
        ("assistant", STOP["last_assistant_message"]),
    ]
    assert ("dev", RUN) not in memory._runs
