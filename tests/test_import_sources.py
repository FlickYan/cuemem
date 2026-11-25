"""cuemem import 的两件事（第十轮体检 Q17）：经正在运行的服务导入、分批（--url、--batch、
portable.split_document），以及 ChatGPT 与 Claude 的数据导出（--from chatgpt | claude）。

导出文件都是这里按公开资料里的结构现造的（没有真实的对话）：ChatGPT 的 mapping / current_node
与 conversations-N.json 对照 difegam.github.io/chatgpt-to-markdown 的 Export Format 与 Data Models
两页，Claude 的 chat_messages / content 对照 docs.rs 上 pond-db 0.3.2 的 claude_ai_export 适配器
（2026-11-24 读的）。全部离线：--url 的 MemoryClient 接在同一个进程里的 /api/v1 应用上。"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fakes import NO_BALANCE, FakeAPI
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scripted import ScriptedModel, echo_extraction
from test_cli import _flat, runner

import cuemem.cli as cli
from cuemem import Memory, MemoryClient
from cuemem.api.auth import Keys, guard
from cuemem.api.memories import install_error_format, memories_router
from cuemem.cli import app
from cuemem.config import PathsConfig, Settings, load_settings
from cuemem.core.timeutil import naive_local
from cuemem.embed.base import EmbedKind
from cuemem.embed.hashing import HashEmbedder
from cuemem.errors import EmbeddingAPIError
from cuemem.portable import (
    FORMAT,
    chatgpt_conversations,
    claude_conversations,
    diverged_sessions,
    from_chatgpt,
    from_claude,
    from_mem0,
    import_space,
    split_document,
    stored_source_ids,
)

URL = "http://testserver"
T0 = 1767268800.0  # 2026-01-01T12:00:00Z：ChatGPT 的时间是 Unix 秒


# ---------- 现造的导出文件 ----------


def _gpt_message(
    mid: str, role: str, text: str, when: float | None, **extra: Any
) -> dict[str, Any]:
    return {
        "id": mid,
        "author": {"role": role, "name": None, "metadata": {}},
        "create_time": when,
        "update_time": None,
        "content": {"content_type": "text", "parts": [text]},
        "status": "finished_successfully",
        "end_turn": True,
        "weight": 1.0,
        "metadata": {},
        "recipient": "all",
        "channel": None,
        **extra,
    }


def _gpt_conversation(
    cid: str,
    texts: Sequence[str],
    *,
    title: str = "Shoe sizes",
    regenerated: dict[int, str] | None = None,
) -> dict[str, Any]:
    """一个 ChatGPT 对话：根、一条隐藏的 system 消息，然后用户与助手轮流说 texts。regenerated
    里的序号处再分出一支（新的消息 id 与文字），current_node 走新的那一支；旧的那一支留在
    mapping 里，与真实的导出一样。"""
    mapping: dict[str, Any] = {
        "root": {"id": "root", "message": None, "parent": None, "children": ["sys"]},
        "sys": {
            "id": "sys",
            "parent": "root",
            "children": [],
            "message": _gpt_message(
                "sys",
                "system",
                "",
                None,
                metadata={"is_visually_hidden_from_conversation": True},
                weight=0.0,
            ),
        },
    }
    parent = "sys"
    regenerated = regenerated or {}
    for i, text in enumerate(texts):
        role = "user" if i % 2 == 0 else "assistant"
        if i in regenerated:  # 旧的那一句成了死掉的分支
            old = f"{cid}-m{i}"
            mapping[old] = {
                "id": old,
                "parent": parent,
                "children": [],
                "message": _gpt_message(old, role, text, T0 + i),
            }
            mapping[parent]["children"].append(old)
        node = f"{cid}-m{i}" + ("-v2" if i in regenerated else "")
        said = regenerated.get(i, text)
        mapping[node] = {
            "id": node,
            "parent": parent,
            "children": [],
            "message": _gpt_message(node, role, said, T0 + i),
        }
        mapping[parent]["children"].append(node)
        parent = node
    return {
        "title": title,
        "create_time": T0,
        "update_time": T0 + len(texts),
        "mapping": mapping,
        "current_node": parent,
        "conversation_id": cid,
        "id": cid,
        "default_model_slug": "auto",
        "is_archived": False,
    }


def _claude_conversation(
    uuid: str, texts: Sequence[str], *, name: str = "Trip to Lisbon", day: int = 1
) -> dict[str, Any]:
    """一个 Claude 对话：human 与 assistant 轮流说 texts，每条带 content 的 text 块。"""
    stamp = f"2026-02-{day:02d}T08:00:{{:02d}}.000000Z"
    messages = [
        {
            "uuid": f"{uuid}-m{i}",
            "text": text,
            "content": [{"type": "text", "text": text, "citations": []}],
            "sender": "human" if i % 2 == 0 else "assistant",
            "created_at": stamp.format(i),
            "updated_at": stamp.format(i),
            "attachments": [],
            "files": [],
        }
        for i, text in enumerate(texts)
    ]
    return {
        "uuid": uuid,
        "name": name,
        "summary": "",
        "created_at": stamp.format(0),
        "updated_at": stamp.format(len(texts)),
        "account": {"uuid": "00000000-aaaa-bbbb-cccc-000000000000"},
        "chat_messages": messages,
    }


BASE = ["I wear EU 40 shoes", "Noted.", "I live in Lyon", "Good to know."]


def _sid(source: str, conversation_id: str) -> str:
    return f"{source}-{hashlib.sha256(conversation_id.encode()).hexdigest()[:16]}"


def _texts(memory: Memory, user: str, session_id: str) -> list[str]:
    exported = memory.export(user_id=user, derived=False)
    (session,) = [s for s in exported["sessions"] if s["session_id"] == session_id]
    return [turn["text"] for turn in session["turns"]]


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------- 夹具 ----------


@pytest.fixture
def memory(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[Memory]:
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        yield m


@pytest.fixture
def served(memory: Memory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Memory]:
    """一个开着的 /api/v1（不要令牌）：cuemem import --url 的 MemoryClient 接在它上面。命令行
    自己的数据目录指到一个空的临时目录，核对 --url 时它不打开本地的数据库。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "cli-home"))
    monkeypatch.delenv("CUEMEM_API_TOKEN", raising=False)
    api = FastAPI()
    api.include_router(memories_router(memory, dependencies=guard(Keys())))
    install_error_format(api)
    with TestClient(api) as http:
        monkeypatch.setattr(cli, "_memory_client", lambda url: MemoryClient(url, http_client=http))
        yield memory


@pytest.fixture
def cli_home(clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """命令行自己打开数据目录（本地导入）：离线的哈希向量，不要 Key。"""
    home = tmp_path / "home"
    monkeypatch.setenv("CUEMEM_HOME", str(home))
    monkeypatch.setenv("CUEMEM_EMBEDDING__BACKEND", "hash")
    monkeypatch.setenv("CUEMEM_EMBEDDING__DIM", "128")
    return home


def _local_export(tmp_path: Path, space: str) -> dict[str, Any]:
    out = tmp_path / f"{space}-out.json"
    result = runner.invoke(app, ["export", "--space", space, "--no-derived", "-o", str(out)])
    assert result.exit_code == 0, result.output
    data: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
    return data


def _flaky(monkeypatch: pytest.MonkeyPatch, needle: str) -> Callable[[], None]:
    """让哈希向量器在文字里有 needle 时出错（模拟向量接口的 503）；返回恢复它的函数。"""
    embed = HashEmbedder.embed

    def failing(self: HashEmbedder, texts: Sequence[str], kind: EmbedKind) -> Any:
        if any(needle in t for t in texts):
            raise EmbeddingAPIError("向量接口返回 HTTP 503", status=503)
        return embed(self, texts, kind)

    monkeypatch.setattr(HashEmbedder, "embed", failing)
    return lambda: monkeypatch.setattr(HashEmbedder, "embed", embed)


def _mem0(n: int, user: str = "alice") -> list[dict[str, Any]]:
    return [
        {
            "id": f"mem-{user}-{i}",
            "memory": f"Fact number {i} about tea",
            "created_at": f"2026-05-{i + 1:02d}T09:00:00+00:00",
            "user_id": user,
        }
        for i in range(n)
    ]


# ---------- 分批：split_document ----------


def test_split_document_pieces_import_to_the_same_log(memory: Memory) -> None:
    """按会话切开的几份逐个导入，写下的日志与整份导入相同；每份带同样的开头、不带 derived，
    会话从不拆开。"""
    doc = from_mem0(_mem0(5), user_id="whole")
    doc["exported_at"] = "2026-06-01T00:00:00"
    doc["derived"] = {"facts": [], "profile": "x"}
    pieces = list(split_document(doc, max_sessions=2))
    assert [len(p["sessions"]) for p in pieces] == [2, 2, 1]
    assert all(
        list(p) == ["format", "version", "exported_at", "space_id", "sessions"] for p in pieces
    )
    assert [s for p in pieces for s in p["sessions"]] == doc["sessions"]

    assert memory.load(doc, user_id="whole")["turns_added"] == 5
    added = [import_space(memory.system, p, space_id="pieces")["turns_added"] for p in pieces]
    assert added == [2, 2, 1]
    whole = memory.export(user_id="whole", derived=False)["sessions"]
    assert memory.export(user_id="pieces", derived=False)["sessions"] == whole
    # 同样的几份再导入一次：什么都不加
    assert [import_space(memory.system, p, space_id="pieces")["turns_added"] for p in pieces] == [
        0,
        0,
        0,
    ]


def test_split_document_by_bytes_and_its_errors() -> None:
    """max_bytes 按各会话紧凑 JSON 的字节数算；一个会话自己就超过时 ValueError（写明用管理
    令牌），oversize_alone 时它单独成一份；不切时原样一份，没有会话时一份空的；形状不对时
    ValueError。"""
    doc = from_mem0(_mem0(4), user_id="bob")
    one = len(json.dumps(doc["sessions"][0], ensure_ascii=False, separators=(",", ":")).encode())
    assert [len(p["sessions"]) for p in split_document(doc, max_bytes=one * 2)] == [2, 2]
    assert [len(p["sessions"]) for p in split_document(doc, max_bytes=one * 2, max_sessions=1)] == [
        1,
        1,
        1,
        1,
    ]
    assert [len(p["sessions"]) for p in split_document(doc)] == [4]
    assert [p["sessions"] for p in split_document({**doc, "sessions": []}, max_sessions=3)] == [[]]

    doc["sessions"][1]["turns"][0]["text"] = "很长的一句话 " * 200
    with pytest.raises(ValueError, match=r"Session 'mem0-\w+' is larger than the server's body"):
        list(split_document(doc, max_bytes=one * 2))
    alone = list(split_document(doc, max_bytes=one * 2, oversize_alone=True))
    assert [len(p["sessions"]) for p in alone] == [1, 1, 2]
    for bad in ({"sessions": "x"}, {"sessions": [1]}, [1]):
        with pytest.raises(ValueError, match="Malformed export"):
            list(split_document(bad, max_sessions=1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_sessions"):
        list(split_document(doc, max_sessions=0))


# ---------- 经正在运行的服务导入：--url ----------


def test_cli_import_through_a_running_server(served: Memory, tmp_path: Path) -> None:
    """cuemem import --url：cuemem 与 mem0 两种文件都经 /api/v1/import 写进服务端，本地的数据
    目录没有被打开；--space、--user 决定写进哪个空间；再运行一次什么都不加。以前没有这个
    选项（No such option），serve 开着时本地导入只会报数据库被占用。"""
    served.add("I like green tea", user_id="alice", infer=False)
    exported = _write(tmp_path / "alice.json", served.export(user_id="alice"))
    result = runner.invoke(app, ["import", str(exported), "--url", URL, "--space", "copy"])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert f"经{URL}导入" in flat and "记忆空间copy：1个会话，新增1条发言" in flat
    assert "后台线程会抽取" in flat
    assert [t.text for t in served.system.db.get_turns("copy")] == ["I like green tea"]
    again = runner.invoke(app, ["import", str(exported), "--url", URL, "--space", "copy"])
    assert again.exit_code == 0 and "新增0条发言" in _flat(again.output)

    mem0 = _mem0(3) + _mem0(2, user="bob")[1:]
    path = _write(tmp_path / "mem0.json", mem0)
    result = runner.invoke(
        app, ["import", str(path), "--from", "mem0", "--url", URL, "--batch", "2"]
    )
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "记忆空间alice：3个会话，新增3条发言、0张图片（分2批）" in flat
    assert "记忆空间bob：1个会话，新增1条发言" in flat
    assert len(served.system.db.get_turns("alice")) == 4  # 原来的一句加 Mem0 的三条
    result = runner.invoke(
        app, ["import", str(path), "--from", "mem0", "--url", URL, "--user", "carol"]
    )
    assert result.exit_code == 0 and "记忆空间carol：4个会话，新增4条发言" in _flat(result.output)
    assert not (tmp_path / "cli-home").exists()  # 命令行没有碰本地的数据目录

    bad = runner.invoke(app, ["import", str(path), "--from", "mem0", "--url", "localhost:7860"])
    assert bad.exit_code == 1 and "--url" in bad.output and "Traceback" not in bad.output


def test_a_failed_batch_over_url_keeps_the_earlier_ones(
    served: Memory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第 2 批在服务端向量化出错：一行说明（第几批、前面写了几批），第 1 批已经写入；修好之后
    再运行同样的命令，只补上没写的。每个请求都是 wait=false（不在请求里等抽取）。"""
    path = _write(tmp_path / "mem0.json", _mem0(3))
    restore = _flaky(monkeypatch, "number 1")
    args = ["import", str(path), "--from", "mem0", "--url", URL, "--batch", "1"]
    result = runner.invoke(app, args)
    flat = _flat(result.output)
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "第2批（共3批）导入失败" in flat and "503" in flat
    assert "前面1批已经写入" in flat and "只补上没写的" in flat
    assert [t.text for t in served.system.db.get_turns("alice")] == ["Fact number 0 about tea"]

    restore()
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "记忆空间alice：3个会话，新增2条发言、0张图片（分3批）" in _flat(result.output)
    assert len(served.system.db.get_turns("alice")) == 3


def test_consolidate_over_url_waits_for_the_server(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--url --consolidate：各批照样是 wait=false，写完之后等服务端的后台线程抽取（/wait），
    不在导入的请求里等（一批 200 个会话的抽取超过客户端 300 秒的读超时）。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "cli-home"))
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    sent: list[tuple[str, Any]] = []
    with Memory(n3_settings, http_client=fake_api.client()) as memory:
        api = FastAPI()
        api.include_router(memories_router(memory, dependencies=guard(Keys())))
        install_error_format(api)
        with TestClient(api) as http:

            class Wire:
                timeout = http.timeout

                def request(self, method: str, url: str, **kwargs: Any) -> Any:
                    sent.append((url.removeprefix(URL + "/api/v1"), kwargs.get("json")))
                    return http.request(method, url, **kwargs)

            monkeypatch.setattr(
                cli,
                "_memory_client",
                lambda url: MemoryClient(url, http_client=Wire()),  # type: ignore[arg-type]
            )
            path = _write(tmp_path / "mem0.json", _mem0(3))
            result = runner.invoke(
                app,
                [
                    "import",
                    str(path),
                    "--from",
                    "mem0",
                    "--url",
                    URL,
                    "--batch",
                    "2",
                    "--consolidate",
                ],
            )
        assert result.exit_code == 0, result.output
        assert "alice：抽取完成" in _flat(result.output)
        assert [path for path, _ in sent] == ["/import", "/import", "/wait", "/status"]
        assert [body["wait"] for path, body in sent if path == "/import"] == [False, False]
        assert len(memory.get_all(user_id="alice")["results"]) == 3


def test_consolidate_over_url_exits_1_when_extraction_is_not_done(
    n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """第十轮复审（Q17）：--url --consolidate 之后服务端有会话抽取失败，或者服务端没有后台线程、
    会话还排着：写出来、退出码 1（发言已经写入）。以前是 0，而本地的 --consolidate 抽取失败时
    是 1，脚本没法照退出码判断。"""
    monkeypatch.setenv("CUEMEM_HOME", str(tmp_path / "cli-home"))
    path = _write(tmp_path / "mem0.json", _mem0(3))
    args = ["import", str(path), "--from", "mem0", "--url", URL, "--consolidate"]
    fake_api.chat_replies = [NO_BALANCE]  # 抽取的每个请求 402
    for background, said in ((True, "alice：3个会话抽取失败"), (False, "服务端没有后台线程")):
        paths = PathsConfig().resolve_against(tmp_path / f"server-{background}")
        home = n3_settings.model_copy(update={"paths": paths})
        with Memory(home, http_client=fake_api.client(), background=background) as memory:
            api = FastAPI()
            api.include_router(memories_router(memory, dependencies=guard(Keys())))
            install_error_format(api)
            with TestClient(api) as http:
                monkeypatch.setattr(
                    cli,
                    "_memory_client",
                    lambda url, http=http: MemoryClient(url, http_client=http),
                )
                result = runner.invoke(app, args)
            assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
            assert said in _flat(result.output), result.output
            assert "Traceback" not in result.output
            assert len(memory.system.db.get_turns("alice")) == 3  # 发言已经写入


# ---------- 本地：--batch 与数据库被占用时的提示 ----------


def test_local_batches_resume_and_cuemem_stays_all_or_nothing(
    cli_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本地导入：--from cuemem 不给 --batch 时仍是一个事务（中途出错一条都不写）；--batch 1 时
    各批各是一个事务，第 2 批出错时第 1 批留着，再运行一次只补上剩下的。"""
    doc = {**from_mem0(_mem0(3), user_id="bob"), "space_id": "bob"}
    del doc["skipped"]
    path = _write(tmp_path / "bob.json", doc)
    restore = _flaky(monkeypatch, "number 1")
    result = runner.invoke(app, ["import", str(path)])
    assert result.exit_code == 1 and "503" in result.output
    assert "已经写入" not in result.output
    assert "bob" not in runner.invoke(app, ["spaces"]).output

    result = runner.invoke(app, ["import", str(path), "--batch", "1"])
    flat = _flat(result.output)
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "第2批（共3批）导入失败" in flat and "前面1批已经写入" in flat
    assert len(_local_export(tmp_path, "bob")["sessions"]) == 1

    restore()
    result = runner.invoke(app, ["import", str(path), "--batch", "1"])
    assert result.exit_code == 0, result.output
    assert "记忆空间bob：3个会话，新增2条发言、0张图片（分3批）" in _flat(result.output)
    assert _local_export(tmp_path, "bob")["sessions"] == doc["sessions"]


def test_import_names_url_when_the_database_is_in_use(
    cli_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """另一个进程（cuemem serve）开着数据库时，import 的说明里写着 --url：以前只说"通过
    MemoryClient、HTTP 连到那个进程"，这个命令却没有这样的选项。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    path = _write(tmp_path / "mem0.json", _mem0(1))
    with Memory(load_settings(), background=False):
        result = runner.invoke(app, ["import", str(path), "--from", "mem0"])
    flat = _flat(result.output)
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "正被另一个进程使用" in flat and "--urlhttp://127.0.0.1:7860" in flat


# ---------- ChatGPT 与 Claude 的导出：换算 ----------


def test_chatgpt_keeps_the_active_branch_and_counts_what_it_skips() -> None:
    """只导入 current_node 这一支（重新生成过的那一句取新的）；隐藏的、system、tool、代码、
    思考、发给工具的消息与空的消息跳过并按原因记数，图片只记数；没有时间的消息用上一句的。"""
    conv = _gpt_conversation("c1", BASE, regenerated={2: "I live in Nice"})
    mapping, leaf = conv["mapping"], conv["current_node"]
    extras = [
        _gpt_message("tool-1", "tool", "42", T0 + 10, author={"role": "tool", "name": "python"}),
        _gpt_message(
            "code-1", "assistant", "", T0 + 11, content={"content_type": "code", "text": "1+1"}
        ),
        _gpt_message(
            "think-1",
            "assistant",
            "",
            T0 + 12,
            content={"content_type": "thoughts", "thoughts": []},
        ),
        _gpt_message(
            "call-1", "assistant", '{"prompt": "a cat"}', T0 + 13, recipient="dalle.text2im"
        ),
        _gpt_message("blank-1", "assistant", "  ", T0 + 14),
        _gpt_message(
            "pic-1",
            "user",
            "",
            None,
            content={
                "content_type": "multimodal_text",
                "parts": [
                    {
                        "content_type": "image_asset_pointer",
                        "asset_pointer": "file-service://file-x",
                    },
                    "What breed is this?",
                ],
            },
        ),
    ]
    for message in extras:
        mapping[message["id"]] = {
            "id": message["id"],
            "parent": leaf,
            "children": [],
            "message": message,
        }
        mapping[leaf]["children"].append(message["id"])
        leaf = message["id"]
    conv["current_node"] = leaf

    doc = from_chatgpt([conv], user_id="bob")
    assert doc["format"] == FORMAT and doc["space_id"] == "bob"
    (session,) = doc["sessions"]
    assert session["session_id"] == _sid("chatgpt", "c1")
    assert session["started_at"] == naive_local(datetime.fromtimestamp(T0, tz=UTC)).isoformat()
    turns = session["turns"]
    assert [t["text"] for t in turns] == [
        "I wear EU 40 shoes",
        "Noted.",
        "I live in Nice",
        "Good to know.",
        "What breed is this?",
    ]
    assert [t["idx"] for t in turns] == [0, 1, 2, 3, 4]
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert turns[2]["metadata"] == {
        "source": "chatgpt",
        "source_message_id": "c1-m2-v2",
        "source_title": "Shoe sizes",
    }
    assert turns[4]["ts"] == turns[3]["ts"]  # create_time 为 null：用上一句留下的发言的
    assert doc["skipped"] == {  # 隐藏的那条是 system：按 role 记
        "attachments": 1,
        "code": 1,
        "empty": 1,
        "system": 1,
        "thoughts": 1,
        "tool": 1,
        "tool_call": 1,
    }

    hidden = _gpt_conversation("c2", BASE)
    hidden["mapping"]["c2-m1"]["message"]["metadata"] = {
        "is_visually_hidden_from_conversation": True
    }
    doc = from_chatgpt([hidden], agent_id="helper")
    assert doc["space_id"] == "agent/helper" and doc["skipped"]["hidden"] == 1
    assert [t["text"] for t in doc["sessions"][0]["turns"]] == [BASE[0], BASE[2], BASE[3]]


def test_chat_exports_in_zips_and_folders(tmp_path: Path) -> None:
    """ChatGPT 的导出：分成几个文件的 zip（按编号读）、多包了一层目录的 zip、解压出的目录与
    单独的文件都能读；zip 里别的成员不读。Claude 的也一样。超过大小上限的成员拒绝，没有
    对话文件、不是 JSON 数组、几个目录里都有时一行 ValueError。"""
    convs = [_gpt_conversation(f"c{i}", [f"topic {i}", "ok"]) for i in range(4)]
    split = tmp_path / "export.zip"
    with zipfile.ZipFile(split, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("conversations-10.json", json.dumps(convs[3:]))
        archive.writestr("conversations-0.json", json.dumps(convs[:2]))
        archive.writestr("conversations-2.json", json.dumps(convs[2:3]))
        archive.writestr("user.json", json.dumps({"email": "nobody@example.invalid"}))
        archive.writestr("chat.html", "<html>" + "x" * 5000)
        archive.writestr("a/b/conversations.json", "not read: too deep")
    assert [c["id"] for c in chatgpt_conversations(split)] == ["c0", "c1", "c2", "c3"]

    nested = tmp_path / "nested.zip"
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("chatgpt-export-2026/conversations.json", json.dumps(convs[:2]))
        archive.writestr("chatgpt-export-2026/user.json", "{}")
    assert [c["id"] for c in chatgpt_conversations(nested)] == ["c0", "c1"]

    folder = tmp_path / "unzipped"
    folder.mkdir()
    with zipfile.ZipFile(split) as archive:
        for name in ("conversations-0.json", "conversations-2.json", "conversations-10.json"):
            (folder / name).write_bytes(archive.read(name))
    assert [c["id"] for c in chatgpt_conversations(folder)] == ["c0", "c1", "c2", "c3"]
    assert [c["id"] for c in chatgpt_conversations(folder / "conversations-2.json")] == ["c2"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.zip", "nested.zip", "unzipped"]

    claude = tmp_path / "claude.zip"
    with zipfile.ZipFile(claude, "w") as archive:
        archive.writestr(
            "conversations.json", json.dumps([_claude_conversation("u1", ["hi", "hello"])])
        )
        archive.writestr("users.json", "[]")
        archive.writestr("projects.json", "[]")
    assert [c["uuid"] for c in claude_conversations(claude)] == ["u1"]

    with pytest.raises(ValueError, match=r"conversations-0\.json is larger than 100 bytes"):
        list(chatgpt_conversations(split, max_file_bytes=100))
    with pytest.raises(ValueError, match=r"conversations-0\.json is larger than 100 bytes"):
        list(chatgpt_conversations(folder, max_file_bytes=100))
    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("user.json", "{}")
    with pytest.raises(
        ValueError, match=r"Malformed Claude export: empty\.zip has no conversations"
    ):
        list(claude_conversations(empty))
    two = tmp_path / "two.zip"
    with zipfile.ZipFile(two, "w") as archive:
        archive.writestr("a/conversations.json", "[]")
        archive.writestr("b/conversations.json", "[]")
    with pytest.raises(ValueError, match="several folders"):
        list(chatgpt_conversations(two))
    with pytest.raises(ValueError, match="is not a JSON array"):
        list(chatgpt_conversations(_write(tmp_path / "object.json", {"conversations": []})))
    broken = tmp_path / "broken.json"
    broken.write_text('[{"id": "c1", "mapping": ', encoding="utf-8")
    with pytest.raises(
        ValueError, match=r"Malformed ChatGPT export: broken\.json is not valid JSON"
    ):
        list(chatgpt_conversations(broken))
    cut = tmp_path / "cut.zip"
    cut.write_bytes(split.read_bytes()[:-40])
    with pytest.raises(ValueError, match="Malformed ChatGPT export"):
        list(chatgpt_conversations(cut))


def test_claude_text_blocks_and_the_text_fallback() -> None:
    """Claude 的消息：文字取 content 里的 text 块（thinking、tool_use、tool_result 跳过、记数），
    没有 content 的旧导出用消息的 text；human 是 user；附件只记数；没有 uuid 的消息用位置当 id。"""
    conv = _claude_conversation("u1", ["I moved to Lisbon", "Nice city.", "old style", "unused"])
    messages = conv["chat_messages"]
    messages[1]["content"] = [
        {"type": "thinking", "thinking": "they moved"},
        {"type": "text", "text": "Nice city."},
        {"type": "tool_use", "name": "web_search", "input": {"query": "Lisbon"}},
        {"type": "tool_result", "name": "web_search", "content": []},
        {"type": "text", "text": "It has trams."},
    ]
    messages[1]["text"] = "ignored when content has blocks"
    del messages[2]["content"], messages[2]["uuid"]
    messages[2]["attachments"] = [{"file_name": "notes.txt", "extracted_content": "secret"}]
    messages[3]["content"] = [{"type": "tool_result", "name": "web_search", "content": []}]
    messages[3]["text"] = ""
    messages.append({**messages[0], "uuid": "u1-sys", "sender": "system"})

    doc = from_claude([conv], user_id="bob")
    (session,) = doc["sessions"]
    assert session["session_id"] == _sid("claude", "u1")
    turns = session["turns"]
    assert [(t["role"], t["text"]) for t in turns] == [
        ("user", "I moved to Lisbon"),
        ("assistant", "Nice city.\n\nIt has trams."),
        ("user", "old style"),
    ]
    assert (
        turns[0]["ts"]
        == naive_local(datetime.fromisoformat("2026-02-01T08:00:00+00:00")).isoformat()
    )
    assert turns[1]["metadata"] == {
        "source": "claude",
        "source_message_id": "u1-m1",
        "source_title": "Trip to Lisbon",
    }
    assert turns[2]["metadata"]["source_message_id"] == "u1:2"
    assert doc["skipped"] == {
        "attachments": 1,
        "empty": 1,
        "system": 1,
        "thinking": 1,
        "tool_result": 2,
        "tool_use": 1,
    }
    assert "secret" not in json.dumps(doc)


def test_since_limit_empty_and_repeated_conversations() -> None:
    old, new = (
        _claude_conversation("old", ["a", "b"], day=1),
        _claude_conversation("new", ["c"], day=9),
    )
    blank = _claude_conversation("blank", [])
    doc = from_claude([old, new, blank, new], user_id="bob", since=datetime(2026, 2, 5))
    assert [s["session_id"] for s in doc["sessions"]] == [_sid("claude", "new")]
    assert doc["skipped"] == {
        "before_since": 1,
        "duplicate_conversations": 1,
        "empty_conversations": 1,
    }
    # limit：只换算前 N 个有内容的对话，之后的不读（第三项不合格式也不报错）
    doc = from_chatgpt(
        iter([_gpt_conversation("c1", ["x"]), _gpt_conversation("c2", ["y"]), "junk"]),
        user_id="bob",
        limit=2,
    )
    assert len(doc["sessions"]) == 2


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda c: c.pop("mapping"), "has no mapping"),
        (lambda c: c.update(current_node="gone"), "has no current_node in its mapping"),
        (lambda c: c["mapping"]["c1-m0"].update(parent="lost"), "points at a missing node"),
        (lambda c: c["mapping"]["sys"].update(parent="c1-m1"), "has a cycle"),
        (lambda c: c["mapping"]["c1-m1"]["message"].pop("author"), "without author or content"),
        (lambda c: c["mapping"]["c1-m1"]["message"]["content"].pop("parts"), "without parts"),
        (lambda c: c["mapping"]["c1-m1"]["message"].update(create_time="noon"), "bad time 'noon'"),
        (lambda c: c.update(create_time=1e30), "bad time"),
        (lambda c: (c.pop("conversation_id"), c.pop("id")), "conversation 1 has no id"),
    ],
)
def test_a_malformed_chatgpt_conversation_refuses_the_whole_file(
    damage: Callable[[dict[str, Any]], Any], message: str
) -> None:
    good, bad = _gpt_conversation("c0", BASE), _gpt_conversation("c1", BASE)
    damage(bad)
    with pytest.raises(ValueError, match="Malformed ChatGPT export: .*" + message):
        from_chatgpt([good, bad], user_id="bob")


def test_malformed_claude_exports_and_missing_users() -> None:
    conv = _claude_conversation("u1", ["hi"])
    for change, message in (
        ({"uuid": None}, "conversation 0 has no uuid"),
        ({"chat_messages": {}}, "has no chat_messages"),
        ({"chat_messages": ["hi"]}, "a message that is no object"),
        ({"created_at": "yesterday"}, "bad time 'yesterday'"),
        (
            {"created_at": None, "chat_messages": [{"sender": "human", "text": "x"}]},
            "no usable time",
        ),
    ):
        with pytest.raises(ValueError, match="Malformed Claude export: .*" + message):
            from_claude([{**conv, **change}], user_id="bob")
    with pytest.raises(ValueError, match="Malformed Claude export: conversation 0 is not a JSON"):
        from_claude(["hi"], user_id="bob")
    # 两种导出拿错了格式：说明里写着该用哪一个
    with pytest.raises(ValueError, match="--from claude"):
        from_chatgpt([conv], user_id="bob")
    with pytest.raises(ValueError, match="--from chatgpt"):
        from_claude([{"title": "t", "mapping": {}, "current_node": "x"}], user_id="bob")
    for convert in (from_chatgpt, from_claude):
        with pytest.raises(ValueError, match="pass user_id or agent_id"):
            convert([])


def test_lone_surrogates_in_a_chat_export_are_replaced(memory: Memory) -> None:
    """导出里截断的表情符号（JSON 的 \\ud83d，落单的代理项）写不成 UTF-8：换成 U+FFFD，照常导入。"""
    conv = json.loads(
        json.dumps(_gpt_conversation("c1", ["cut emoji X", "ok"])).replace("X", "\\ud83d")
    )
    conv["title"] = "t\udc00"
    doc = from_chatgpt([conv], user_id="bob")
    turn = doc["sessions"][0]["turns"][0]
    assert turn["text"] == "cut emoji �" and turn["metadata"]["source_title"] == "t�"
    assert import_space(memory.system, doc)["turns_added"] == 2


# ---------- 导入、再导入、多了几句、改写过 ----------


def test_chat_import_is_idempotent_and_grows(memory: Memory, fake_api: FakeAPI) -> None:
    """同一个导出再导入：0 条发言、不调用模型。对话后来多了两句：只加这两句（一次重新抽取），
    会话里没有重复。Claude 的也一样。"""
    for source, build, convert in (
        ("chatgpt", _gpt_conversation, from_chatgpt),
        ("claude", _claude_conversation, from_claude),
    ):
        user = f"{source}-user"
        first = import_space(
            memory.system, convert([build("c1", BASE)], user_id=user), consolidate=True
        )
        assert first["turns_added"] == 4
        calls = len(fake_api.chat_requests)
        again = import_space(
            memory.system, convert([build("c1", BASE)], user_id=user), consolidate=True
        )
        assert again["turns_added"] == 0 and len(fake_api.chat_requests) == calls

        longer = [*BASE, "I moved to Paris", "Ok."]
        grown = convert([build("c1", longer)], user_id=user)
        assert (
            diverged_sessions(
                grown, stored_source_ids(memory.system.db, user, [_sid(source, "c1")])
            )
            == []
        )
        assert import_space(memory.system, grown, consolidate=True)["turns_added"] == 2
        assert len(fake_api.chat_requests) == calls + 1
        assert _texts(memory, user, _sid(source, "c1")) == longer
        facts = {f["memory"] for f in memory.get_all(user_id=user)["results"]}
        assert facts == {"I wear EU 40 shoes", "I live in Lyon", "I moved to Paris"}
        assert memory.get_all(user_id=user)["results"][0]["metadata"]["source"] == source


def test_cli_chatgpt_import_reports_edited_conversations(
    served: Memory, tmp_path: Path, fake_api: FakeAPI
) -> None:
    """cuemem import --from chatgpt --url：第一次导入之后在 ChatGPT 里改写了一句（当前这一支
    换了）。再导入时这个对话被报告为改写过、不写，存下的文字不动，别的对话照常导入（以前的
    原型里新的一支被悄悄丢掉）；--dry-run 只报告；--replace-edited 先遗忘这个会话再导入，
    存下的是新的文字。"""
    first = _write(tmp_path / "v1.json", [_gpt_conversation("c1", BASE)])
    args = ["--from", "chatgpt", "--user", "bob", "--url", URL]
    result = runner.invoke(app, ["import", str(first), *args])
    assert result.exit_code == 0, result.output
    flat = _flat(result.output)
    assert "ChatGPT导出：1个对话、4条发言" in flat and "跳过：system1" in flat
    assert "记忆空间bob：1个会话，新增4条发言" in flat and "改写过" not in flat
    served.consolidate(user_id="bob")
    sid = _sid("chatgpt", "c1")
    assert _texts(served, "bob", sid) == BASE

    edited = [*BASE[:2], "I live in Nice", "Good to know.", "And I like trams", "Ok."]
    second = _write(
        tmp_path / "v2.json",
        [
            _gpt_conversation("c1", edited, regenerated={2: "I live in Nice"}),
            _gpt_conversation("c2", ["I play the cello", "Lovely."], title="Music"),
        ],
    )
    dry = runner.invoke(app, ["import", str(second), *args, "--dry-run"])
    flat = _flat(dry.output)
    assert dry.exit_code == 0, dry.output
    assert "1个对话在上次导入之后改写过" in flat and f"{sid}Shoesizes" in flat
    assert "--dry-run：什么都没有写" in flat and "记忆空间bob：" not in flat
    assert len(served.export(user_id="bob", derived=False)["sessions"]) == 1

    result = runner.invoke(app, ["import", str(second), *args])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "1个对话在上次导入之后改写过" in flat and "加--replace-edited" in flat
    assert "记忆空间bob：1个会话，新增2条发言" in flat
    assert _texts(served, "bob", sid) == BASE  # 存下的不动，后面多出的两句也不接上去
    assert _texts(served, "bob", _sid("chatgpt", "c2")) == ["I play the cello", "Lovely."]

    result = runner.invoke(app, ["import", str(second), *args, "--replace-edited"])
    flat = _flat(result.output)
    assert result.exit_code == 0, result.output
    assert "已遗忘这1个会话" in flat and "记忆空间bob：2个会话，新增6条发言" in flat
    assert _texts(served, "bob", sid) == edited
    served.consolidate(user_id="bob")
    facts = {f["memory"] for f in served.get_all(user_id="bob")["results"]}
    assert "I live in Nice" in facts and "I live in Lyon" not in facts
    again = runner.invoke(app, ["import", str(second), *args])
    assert "改写过" not in again.output and "新增0条发言" in _flat(again.output)


def test_cli_claude_import_locally(cli_home: Path, tmp_path: Path) -> None:
    """本地的 cuemem import --from claude：zip 直接导入；--dry-run 不建数据库；改写过的对话
    （这里是换了一条消息的 uuid）不写，--replace-edited 之后是新的；--limit、--since 起作用。"""
    export = tmp_path / "claude.zip"
    convs = [
        _claude_conversation("u1", ["I moved to Lisbon", "Nice city."], day=1),
        _claude_conversation("u2", ["I have a cat named Miso", "Cute."], name="Pets", day=9),
    ]
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("conversations.json", json.dumps(convs))
    args = ["import", str(export), "--from", "claude", "--user", "bob"]

    dry = runner.invoke(app, [*args, "--dry-run"])
    flat = _flat(dry.output)
    assert dry.exit_code == 0, dry.output
    assert "Claude导出：2个对话、4条发言" in flat and "→记忆空间bob" in flat
    assert "抽取各要调用LLM" in flat and "什么都没有写" in flat
    assert not (cli_home / "cuemem.sqlite").exists() and not list(cli_home.rglob("*.sqlite"))

    result = runner.invoke(app, [*args, "--limit", "1"])
    assert result.exit_code == 0 and "记忆空间bob：1个会话，新增2条发言" in _flat(result.output)
    result = runner.invoke(app, [*args, "--since", "2026-02-05"])
    flat = _flat(result.output)
    assert "before_since1" in flat and "记忆空间bob：1个会话，新增2条发言" in flat
    assert "运行cuememconsolidate" in flat
    result = runner.invoke(app, args)
    assert result.exit_code == 0 and "记忆空间bob：2个会话，新增0条发言" in _flat(result.output)

    convs[0]["chat_messages"][1].update(uuid="u1-m1-retry", text="Great food there.")
    convs[0]["chat_messages"][1]["content"][0]["text"] = "Great food there."
    with zipfile.ZipFile(export, "w") as archive:
        archive.writestr("conversations.json", json.dumps(convs))
    dry = runner.invoke(app, [*args, "--dry-run"])
    assert "1个对话在上次导入之后改写过" in _flat(dry.output)
    result = runner.invoke(app, args)
    assert result.exit_code == 0 and "记忆空间bob：1个会话，新增0条发言" in _flat(result.output)
    stored = {s["session_id"]: s for s in _local_export(tmp_path, "bob")["sessions"]}
    assert [t["text"] for t in stored[_sid("claude", "u1")]["turns"]][1] == "Nice city."
    result = runner.invoke(app, [*args, "--replace-edited"])
    assert result.exit_code == 0, result.output
    stored = {s["session_id"]: s for s in _local_export(tmp_path, "bob")["sessions"]}
    assert [t["text"] for t in stored[_sid("claude", "u1")]["turns"]] == [
        "I moved to Lisbon",
        "Great food there.",
    ]


def test_cli_chat_import_errors_are_one_line_and_write_nothing(
    cli_home: Path, tmp_path: Path
) -> None:
    """不合格式的导出、没给 --user、忘了 --from、拿错了格式、选项用错了地方：都是一行说明、
    以 1 退出，没有调用栈，什么都不写（数据目录里没有数据库）。"""
    good = [_gpt_conversation("c0", BASE), _gpt_conversation("c1", BASE)]
    bad = json.loads(json.dumps(good))
    del bad[1]["mapping"]["c1-m2"]["message"]["content"]["parts"]
    chatgpt = _write(tmp_path / "conversations.json", good)
    broken = _write(tmp_path / "broken.json", bad)
    claude = _write(tmp_path / "claude.json", [_claude_conversation("u1", ["hi", "hello"])])
    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("conversations-0.json", json.dumps(good))
    folder = tmp_path / "folder"
    folder.mkdir()
    for args, expected in (
        (
            [str(broken), "--from", "chatgpt", "--user", "bob"],
            "MalformedChatGPTexport:conversation'c1'",
        ),
        ([str(chatgpt), "--from", "chatgpt"], "用--user或--agent指定导入到谁的空间"),
        ([str(chatgpt)], "像是ChatGPT的数据导出：加上--fromchatgpt--userU"),
        ([str(claude)], "像是Claude的数据导出：加上--fromclaude--userU"),
        ([str(claude), "--from", "mem0"], "--fromclaude"),
        ([str(archive)], "是zip：ChatGPT、Claude的数据导出加上--fromchatgpt或--fromclaude"),
        (
            [str(claude), "--from", "chatgpt", "--user", "bob"],
            "itlookslikeaClaudeexport:use--fromclaude",
        ),
        ([str(chatgpt), "--from", "claude", "--user", "bob"], "use--fromchatgpt"),
        ([str(folder), "--from", "claude", "--user", "bob"], "folderhasnoconversations.json"),
        ([str(folder)], "是目录"),
        (
            [str(archive), "--from", "chatgpt", "--user", "bob", "--max-file-gb", "0.000001"],
            "islargerthan",
        ),
        ([str(chatgpt), "--dry-run"], "只用于--fromchatgpt、claude"),
        ([str(chatgpt), "--from", "chatgpt", "--user", "bob", "--space", "x"], "只给一种"),
    ):
        result = runner.invoke(app, ["import", *args])
        flat = _flat(result.output)
        assert result.exit_code == 1 and isinstance(result.exception, SystemExit), (args, flat)
        assert expected in flat and "Traceback" not in flat, (args, flat)
    assert not list(cli_home.rglob("*.sqlite"))
    assert runner.invoke(app, ["import", str(chatgpt), "--batch", "-1"]).exit_code == 2
