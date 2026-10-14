"""Claude 记忆工具（memory_20250818）的 CueMem 后端（N12）。全部离线：命令直接调用工具的 call，
tool_runner 用 SDK 自己的客户端配上模拟的传输层，CueMem 用脚本化的模拟模型。"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from datetime import datetime
from typing import Any

import pytest

anthropic = pytest.importorskip("anthropic")

import httpx2  # noqa: E402
from anthropic.lib.tools import ToolError  # noqa: E402
from fakes import FakeAPI  # noqa: E402
from scripted import ScriptedModel  # noqa: E402

from cuemem import AsyncMemory, Memory  # noqa: E402
from cuemem.api.proxy import MEMORY_NOTE  # noqa: E402
from cuemem.config import Settings  # noqa: E402
from cuemem.integrations.claude import (  # noqa: E402
    AsyncCueMemMemoryTool,
    CueMemMemoryTool,
    with_recall,
)

LISTING_HEADER = (
    "Here're the files and directories up to 2 levels deep in /memories, excluding hidden items "
    "and node_modules:"
)


def note_facts(prompt: str) -> dict[str, Any]:
    """抽取替身：笔记里每个 "- " 开头的行记一条事实（"lives in X" 是单值事实 lives_in）；
    普通发言整句记一条。"""
    facts = []
    for n, text in re.findall(r"\[t(\d+)\] User: (.*?)(?=\n\[t\d+\] |\Z)", prompt, re.S):
        lines = text.strip().split("\n")
        if lines[0].startswith("Memory note"):
            items = [x[2:].strip() for x in lines[1:] if x.startswith("- ")]
        else:
            items = [text.strip()]
        for item in items:
            place = re.search(r"lives? in (\w+)", item)
            facts.append(
                {
                    "statement": f"The user lives in {place.group(1)}." if place else item,
                    "subject": "User",
                    "predicate": "lives_in" if place else "likes",
                    "object": place.group(1) if place else item[:40],
                    "category": "attribute" if place else "preference",
                    "cardinality": "single" if place else "multi",
                    "importance": 5,
                    "turns": [int(n)],
                }
            )
    return {"summary": "", "facts": facts, "entities": []}


def moved(prompt: str) -> dict[str, Any]:
    """对账替身：新的住址取代旧的（现实变了）。"""
    return {"decisions": [{"new": 1, "action": "update", "target": "e1"}]}


@pytest.fixture
def memory(n3_settings: Settings, fake_api: FakeAPI) -> Iterator[Memory]:
    fake_api.chat_replies = [ScriptedModel(note_facts, reconcile=moved)]
    with Memory(n3_settings, http_client=fake_api.client(), background=False) as m:
        yield m


@pytest.fixture
def tool(memory: Memory) -> Iterator[CueMemMemoryTool]:
    with CueMemMemoryTool(memory, user_id="alice", wait=True) as t:
        yield t


def run(tool: CueMemMemoryTool, **command: Any) -> str:
    """像 tool_runner 那样把 tool_use 的 input 交给工具。"""
    out = tool.call(command)
    assert isinstance(out, str)
    return out


def listed(listing: str) -> list[str]:
    return [line.split("\t")[1] for line in listing.split("\n")[2:] if "\t" in line]


def facts(memory: Memory) -> dict[str, str]:
    return {f["memory"]: f["id"] for f in memory.get_all(user_id="alice")["results"]}


def notes(memory: Memory) -> list[str]:
    return [t.text for t in memory.system.db.get_turns("alice")]


def today() -> str:
    return f"{datetime.now():%Y-%m-%d}"


def test_empty_directory_shows_the_generated_files(tool: CueMemMemoryTool) -> None:
    out = run(tool, command="view", path="/memories")
    lines = out.split("\n")
    assert lines[0] == LISTING_HEADER
    assert re.fullmatch(r"\d+(\.\d)?[BKMG]\t/memories", lines[1])
    assert listed(out) == ["/memories/facts.md", "/memories/history/", "/memories/profile.md"]
    assert out.endswith("deleting a file makes the memory system forget what that file added.")
    assert "read-only" in out
    assert run(tool, command="view", path="/memories/") == out

    profile = run(tool, command="view", path="/memories/profile.md")
    assert profile.startswith(
        "Here's the content of /memories/profile.md with line numbers:\n     1\t# Profile\n"
    )
    assert "Nothing is known about the user yet." in profile
    assert "No facts yet." in run(tool, command="view", path="/memories/facts.md")
    history = run(tool, command="view", path="/memories/history")
    assert history.split("\n")[1:] == ["0B\t/memories/history"]


def test_create_saves_the_file_and_a_note_that_becomes_facts(
    tool: CueMemMemoryTool, memory: Memory
) -> None:
    text = "# About\n- The user lives in Berlin.\n- The user likes jazz.\n"
    assert (
        run(tool, command="create", path="/memories/about_me.md", file_text=text)
        == "File created successfully at: /memories/about_me.md"
    )
    assert notes(memory) == [f"Memory note /memories/about_me.md:\n{text}".strip()]
    found = facts(memory)
    assert set(found) == {"The user lives in Berlin.", "The user likes jazz."}

    shown = run(tool, command="view", path="/memories/about_me.md")
    assert shown == (
        "Here's the content of /memories/about_me.md with line numbers:\n"
        "     1\t# About\n     2\t- The user lives in Berlin.\n     3\t- The user likes jazz."
    )
    assert run(tool, command="view", path="/memories/about_me.md", view_range=[2, -1]).endswith(
        "with line numbers:\n     2\t- The user lives in Berlin.\n     3\t- The user likes jazz."
    )
    assert run(tool, command="view", path="/memories/about_me.md", view_range=[3, 3]).endswith(
        "line numbers:\n     3\t- The user likes jazz."
    )
    with pytest.raises(ToolError, match="Invalid `view_range`"):
        run(tool, command="view", path="/memories/about_me.md", view_range=[5, 6])

    facts_md = run(tool, command="view", path="/memories/facts.md")
    berlin = found["The user lives in Berlin."]
    assert (
        f"[{berlin}] (said on {today()}; current) The user lives in Berlin. "
        "(from /memories/about_me.md)"
    ) in facts_md
    assert "The user lives in Berlin." in run(tool, command="view", path="/memories/profile.md")
    root = run(tool, command="view", path="/memories")
    assert "/memories/about_me.md" in listed(root)
    assert f"{len(text.encode())}B\t/memories/about_me.md" in root


def test_edits_supersede_facts_and_history_keeps_every_version(
    tool: CueMemMemoryTool, memory: Memory
) -> None:
    text = "# About\n- The user lives in Berlin.\n- The user likes jazz.\n"
    run(tool, command="create", path="/memories/about_me.md", file_text=text)
    berlin = facts(memory)["The user lives in Berlin."]

    out = run(
        tool,
        command="str_replace",
        path="/memories/about_me.md",
        old_str="Berlin",
        new_str="Munich",
    )
    assert out == (
        "The memory file has been edited. Here is the snippet showing the change (with line "
        "numbers):\n     1\t# About\n     2\t- The user lives in Munich.\n"
        "     3\t- The user likes jazz.\n     4\t"
    )
    # 笔记只写改动的那一行（连同上方的标题）和改之前的样子，不是整个文件
    assert notes(memory)[-1] == (
        "Memory note /memories/about_me.md (edited):\n# About\n- The user lives in Munich.\n"
        '(This replaces: "- The user lives in Berlin.")'
    )
    current = facts(memory)
    assert set(current) == {"The user lives in Munich.", "The user likes jazz."}
    munich = current["The user lives in Munich."]
    old = memory.get(berlin)
    assert old is not None and old["superseded_by"] == munich

    facts_md = run(tool, command="view", path="/memories/facts.md")
    assert f"earlier versions: /memories/history/{munich}.md" in facts_md
    assert "Berlin" not in facts_md  # 只列当前的事实
    assert f"/memories/history/{munich}.md" in listed(run(tool, command="view", path="/memories"))
    history = run(tool, command="view", path=f"/memories/history/{munich}.md")
    assert re.search(
        rf"1\. \[{berlin}\] \(said on {today()}; no longer true.*see \[{munich}\]\) "
        r"The user lives in Berlin\.",
        history,
    )
    assert f"2. [{munich}] (said on {today()}; current) The user lives in Munich." in history
    assert run(tool, command="view", path=f"/memories/history/{berlin}.md") == history.replace(
        munich + ".md with", berlin + ".md with"
    )
    with pytest.raises(ToolError, match="does not exist"):
        run(tool, command="view", path="/memories/history/0123456789abcdef.md")

    out = run(
        tool,
        command="insert",
        path="/memories/about_me.md",
        insert_line=3,
        insert_text="- The user plays chess.\n",
    )
    assert out == "The file /memories/about_me.md has been edited."
    assert notes(memory)[-1] == (
        "Memory note /memories/about_me.md (added):\n# About\n- The user plays chess."
    )
    assert "The user plays chess." in facts(memory)
    assert run(tool, command="view", path="/memories/about_me.md").endswith(
        "     4\t- The user plays chess."
    )

    # new_str 可以省略：删掉 old_str；笔记写成删掉了什么
    run(
        tool,
        command="str_replace",
        path="/memories/about_me.md",
        old_str="- The user likes jazz.\n",
    )
    assert notes(memory)[-1] == (
        'Memory note /memories/about_me.md (edited):\n(Removed: "- The user likes jazz.")'
    )
    assert "jazz" not in run(tool, command="view", path="/memories/about_me.md")


def test_history_only_shows_facts_of_this_user(memory: Memory, tool: CueMemMemoryTool) -> None:
    with CueMemMemoryTool(memory, user_id="bob", wait=True) as bob:
        run(bob, command="create", path="/memories/me.md", file_text="- The user likes tea.\n")
    tea = memory.get_all(user_id="bob")["results"][0]["id"]
    with pytest.raises(ToolError, match="does not exist"):
        run(tool, command="view", path=f"/memories/history/{tea}.md")
    assert listed(run(tool, command="view", path="/memories")) == [
        "/memories/facts.md",
        "/memories/history/",
        "/memories/profile.md",
    ]


def test_generated_files_are_read_only(tool: CueMemMemoryTool, memory: Memory) -> None:
    run(tool, command="create", path="/memories/me.md", file_text="- The user likes jazz.\n")
    for path in ("/memories/profile.md", "/memories/facts.md", "/memories/history/abc.md"):
        attempts: list[dict[str, Any]] = [
            {"command": "create", "path": path, "file_text": "x"},
            {"command": "str_replace", "path": path, "old_str": "Facts", "new_str": "X"},
            {"command": "insert", "path": path, "insert_line": 0, "insert_text": "x"},
            {"command": "delete", "path": path},
            {"command": "rename", "old_path": path, "new_path": "/memories/x.md"},
            {"command": "rename", "old_path": "/memories/me.md", "new_path": path},
        ]
        for attempt in attempts:
            with pytest.raises(ToolError, match="generated by the memory system and is read-only"):
                tool.call(attempt)
    with pytest.raises(ToolError, match="read-only"):
        run(tool, command="delete", path="/memories/history")
    with pytest.raises(ToolError, match="is a file"):
        run(tool, command="create", path="/memories/facts.md/x.md", file_text="x")
    assert run(tool, command="view", path="/memories/me.md").endswith("- The user likes jazz.")
    assert len(notes(memory)) == 1


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "memories/x.md",
        "x.md",
        "/memoriesx/a.md",
        "/memories/../secrets.env",
        "/memories/a/../../etc/passwd",
        "/memories/./a.md",
        "/memories/.hidden",
        "/memories//a.md",
        "/memories/a\\..\\b.md",
        "/memories/%2e%2e/secrets.env",
        "/memories/a%2fb.md",
        "/memories/a\x00.md",
        "/memories/a\nb.md",
        "/memories/ a.md",
        "/memories/" + "a" * 101,
        "/memories/" + "/".join("d" * 9),
        "",
    ],
)
def test_paths_outside_memories_or_ambiguous_are_rejected(
    tool: CueMemMemoryTool, memory: Memory, path: str
) -> None:
    for command in (
        {"command": "view", "path": path},
        {"command": "create", "path": path, "file_text": "- The user likes jazz."},
        {"command": "delete", "path": path},
        {"command": "rename", "old_path": "/memories/a.md", "new_path": path},
    ):
        with pytest.raises(ToolError):
            tool.call(command)
    assert tool.store.files("alice") == {} and notes(memory) == []


def test_errors_follow_the_tool_contract(tool: CueMemMemoryTool) -> None:
    def error(**command: Any) -> str:
        with pytest.raises(ToolError) as caught:
            tool.call(command)
        return str(caught.value)

    assert (
        error(command="view", path="/memories/nope.md")
        == "The path /memories/nope.md does not exist. Please provide a valid path."
    )
    run(tool, command="create", path="/memories/a.md", file_text="one\ntwo\none\n")
    assert error(command="str_replace", path="/memories/b.md", old_str="x", new_str="y") == (
        "Error: The path /memories/b.md does not exist. Please provide a valid path."
    )
    assert error(command="str_replace", path="/memories/a.md", old_str="three", new_str="y") == (
        "No replacement was performed, old_str `three` did not appear verbatim in /memories/a.md."
    )
    assert error(command="str_replace", path="/memories/a.md", old_str="one", new_str="y") == (
        "No replacement was performed. Multiple occurrences of old_str `one` in lines: 1, 3. "
        "Please ensure it is unique"
    )
    assert error(command="insert", path="/memories/a.md", insert_line=9, insert_text="x") == (
        "Error: Invalid `insert_line` parameter: 9. It should be within the range of lines of "
        "the file: [0, 3]"
    )
    assert (
        error(command="insert", path="/memories/b.md", insert_line=0, insert_text="x")
        == "Error: The path /memories/b.md does not exist"
    )
    assert error(command="delete", path="/memories/b.md") == (
        "Error: The path /memories/b.md does not exist"
    )
    assert error(command="delete", path="/memories") == (
        "Error: Cannot delete the /memories directory itself"
    )
    assert error(command="rename", old_path="/memories/b.md", new_path="/memories/c.md") == (
        "Error: The path /memories/b.md does not exist"
    )
    run(tool, command="create", path="/memories/c.md", file_text="three\n")
    assert error(command="rename", old_path="/memories/a.md", new_path="/memories/c.md") == (
        "Error: The destination /memories/c.md already exists"
    )
    assert error(command="rename", old_path="/memories", new_path="/memories/x") == (
        "Error: Cannot rename the /memories directory itself"
    )
    run(tool, command="create", path="/memories/dir/x.md", file_text="x\n")
    assert error(command="create", path="/memories/dir", file_text="x") == (
        "Error: /memories/dir is a directory"
    )
    assert error(command="create", path="/memories/a.md/x.md", file_text="x") == (
        "Error: /memories/a.md is a file, not a directory"
    )
    assert error(command="str_replace", path="/memories/dir", old_str="x", new_str="y") == (
        "Error: The path /memories/dir does not exist. Please provide a valid path."
    )


def test_size_limits(memory: Memory) -> None:
    with CueMemMemoryTool(
        memory, user_id="alice", max_file_chars=50, max_files=2, max_total_chars=80
    ) as tool:
        with pytest.raises(ToolError, match="at most 50"):
            run(tool, command="create", path="/memories/a.md", file_text="x" * 51)
        run(tool, command="create", path="/memories/a.md", file_text="x" * 40)
        with pytest.raises(ToolError, match="limit of 80"):
            run(tool, command="create", path="/memories/b.md", file_text="y" * 41)
        run(tool, command="create", path="/memories/b.md", file_text="y" * 10)
        with pytest.raises(ToolError, match="already 2 memory files"):
            run(tool, command="create", path="/memories/c.md", file_text="z")
        with pytest.raises(ToolError, match="at most 50"):
            run(
                tool,
                command="insert",
                path="/memories/a.md",
                insert_line=1,
                insert_text="x" * 20,
            )
        assert tool.store.files("alice") == {"/memories/a.md": "x" * 40, "/memories/b.md": "y" * 10}


def test_long_files_are_viewed_in_pages(memory: Memory) -> None:
    text = "".join(f"line {n:04d} {'x' * 40}\n" for n in range(1, 801))  # 约 40K 字符
    with CueMemMemoryTool(memory, user_id="alice", max_file_chars=50_000) as tool:
        run(tool, command="create", path="/memories/log.md", file_text=text)
        first = run(tool, command="view", path="/memories/log.md")
        assert len(first) < 16_500
        last = int(re.findall(r"^\s+(\d+)\t", first, re.M)[-1])
        assert first.endswith(
            f"[Truncated: showing lines 1-{last} of 800. View the rest with view_range "
            f"[{last + 1}, -1].]"
        )
        rest = run(tool, command="view", path="/memories/log.md", view_range=[last + 1, -1])
        assert f"{last + 1:>6}\tline {last + 1:04d}" in rest


def test_delete_forgets_exactly_what_the_file_added(tool: CueMemMemoryTool, memory: Memory) -> None:
    memory.add("I live in Berlin", user_id="alice", wait=True)  # 对话里说的，不是文件写的
    run(
        tool,
        command="create",
        path="/memories/me.md",
        file_text="- The user lives in Munich.\n- The user likes jazz.\n",
    )
    run(tool, command="create", path="/memories/games.md", file_text="- The user plays chess.\n")
    assert set(facts(memory)) == {
        "The user lives in Munich.",
        "The user likes jazz.",
        "The user plays chess.",
    }

    out = run(tool, command="delete", path="/memories/me.md")
    head, gone, back = re.split(r"\nNo longer remembered:\n|\nCurrent again[^\n]*\n", out)
    assert head == (
        "Successfully deleted /memories/me.md\nForgotten from long-term memory: the 1 saved "
        "change(s) of this file; everything derived only from them was removed."
    )
    assert set(gone.split("\n")) == {"- The user lives in Munich.", "- The user likes jazz."}
    assert back == "- The user lives in Berlin."
    assert set(facts(memory)) == {"The user lives in Berlin.", "The user plays chess."}
    assert not any("/memories/me.md" in t for t in notes(memory))  # 日志里的笔记删掉了
    assert "I live in Berlin" in notes(memory)  # 对话本身不动
    with pytest.raises(ToolError, match="does not exist"):
        run(tool, command="view", path="/memories/me.md")
    assert listed(run(tool, command="view", path="/memories")) == [
        "/memories/facts.md",
        "/memories/games.md",
        "/memories/history/",
        "/memories/profile.md",
    ]


def test_rename_moves_the_file_and_its_notes(tool: CueMemMemoryTool, memory: Memory) -> None:
    run(
        tool, command="create", path="/memories/projects/a.md", file_text="- The user likes jazz.\n"
    )
    run(
        tool,
        command="create",
        path="/memories/projects/deep/b.md",
        file_text="- The user plays chess.\n",
    )
    root = listed(run(tool, command="view", path="/memories"))
    assert "/memories/projects/" in root and "/memories/projects/deep/" in root
    assert "/memories/projects/deep/b.md" not in root  # 只列两层
    assert listed(run(tool, command="view", path="/memories/projects")) == [
        "/memories/projects/a.md",
        "/memories/projects/deep/",
        "/memories/projects/deep/b.md",
    ]

    logged = notes(memory)
    with pytest.raises(ToolError, match="into itself"):
        run(tool, command="rename", old_path="/memories/projects", new_path="/memories/projects/x")
    assert (
        run(tool, command="rename", old_path="/memories/projects", new_path="/memories/work")
        == "Successfully renamed /memories/projects to /memories/work"
    )
    assert notes(memory) == logged  # 改名没有新信息，不写笔记
    assert run(tool, command="view", path="/memories/work/deep/b.md").endswith("chess.")
    with pytest.raises(ToolError, match="does not exist"):
        run(tool, command="view", path="/memories/projects/a.md")
    assert "from /memories/work/a.md" in run(tool, command="view", path="/memories/facts.md")

    out = run(tool, command="delete", path="/memories/work")
    assert "the 2 saved change(s) of these 2 files" in out
    assert facts(memory) == {}
    assert tool.store.files("alice") == {} and tool.store.notes("alice") == []


def test_forgetting_elsewhere_removes_the_file(tool: CueMemMemoryTool, memory: Memory) -> None:
    run(tool, command="create", path="/memories/me.md", file_text="- The user likes jazz.\n")
    run(tool, command="create", path="/memories/games.md", file_text="- The user plays chess.\n")
    memory.delete(facts(memory)["The user likes jazz."])  # 例如 MCP 的 forget
    assert "/memories/me.md" not in listed(run(tool, command="view", path="/memories"))
    with pytest.raises(ToolError, match="does not exist"):
        run(tool, command="view", path="/memories/me.md")
    assert list(tool.store.files("alice")) == ["/memories/games.md"]


def test_create_overwrites_and_logs_the_new_version(tool: CueMemMemoryTool, memory: Memory) -> None:
    run(tool, command="create", path="/memories/me.md", file_text="- The user likes jazz.\n")
    out = run(tool, command="create", path="/memories/me.md", file_text="- The user likes blues.\n")
    assert out == "File overwritten successfully at: /memories/me.md"
    # 覆盖时只写改了的行，以及原来的样子
    assert notes(memory)[-1] == (
        "Memory note /memories/me.md (rewritten):\n- The user likes blues.\n"
        '(This replaces: "- The user likes jazz.")'
    )
    run(tool, command="create", path="/memories/me.md", file_text="- The user likes blues.\n")
    assert len(notes(memory)) == 2  # 内容没变：不再写笔记
    # 只是挪了位置的行不算改动
    reordered = "# Music\n- The user likes jazz.\n# Games\n- The user plays chess.\n"
    run(tool, command="create", path="/memories/me.md", file_text=reordered)
    run(
        tool,
        command="create",
        path="/memories/me.md",
        file_text="# Games\n- The user plays chess.\n# Music\n- The user likes jazz.\n"
        "- The user likes tea.\n",
    )
    assert notes(memory)[-1] == (
        "Memory note /memories/me.md (rewritten):\n# Music\n- The user likes tea."
    )
    assert tool.clear_all_memory().startswith("All memory cleared\nForgotten from long-term memory")
    assert tool.store.files("alice") == {} and facts(memory) == {}


def test_blank_changes_write_no_note(tool: CueMemMemoryTool, memory: Memory) -> None:
    run(tool, command="create", path="/memories/todo.md", file_text="")
    run(tool, command="insert", path="/memories/todo.md", insert_line=0, insert_text="\n")
    assert notes(memory) == []
    assert run(tool, command="delete", path="/memories/todo.md") == (
        "Successfully deleted /memories/todo.md\nIt had added nothing to long-term memory."
    )


def test_a_change_that_cannot_be_logged_leaves_the_file_unchanged(
    tool: CueMemMemoryTool, memory: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    run(tool, command="create", path="/memories/me.md", file_text="- The user likes jazz.\n")

    def broken(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(memory, "add", broken)
    with pytest.raises(ToolError, match=r"\(RuntimeError\); /memories/me.md was not changed"):
        run(tool, command="str_replace", path="/memories/me.md", old_str="jazz", new_str="blues")
    with pytest.raises(ToolError, match="was not changed"):
        run(tool, command="create", path="/memories/new.md", file_text="- The user likes tea.\n")
    assert tool.store.files("alice") == {"/memories/me.md": "- The user likes jazz.\n"}


def message(content: list[dict[str, Any]], stop: str) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


TOOL_CALLS = [
    {
        "type": "tool_use",
        "id": "t1",
        "name": "memory",
        "input": {"command": "view", "path": "/memories"},
    },
    {
        "type": "tool_use",
        "id": "t2",
        "name": "memory",
        "input": {
            "command": "create",
            "path": "/memories/me.md",
            "file_text": "- The user lives in Berlin.\n",
        },
    },
    {
        "type": "tool_use",
        "id": "t3",
        "name": "memory",
        "input": {"command": "view", "path": "/etc/passwd"},
    },
]
REPLIES = [
    message(TOOL_CALLS, "tool_use"),
    message([{"type": "text", "text": "Noted: you live in Berlin."}], "end_turn"),
]


def check_round_trip(requests: list[dict[str, Any]], memory: Memory) -> None:
    assert requests[0]["tools"] == [{"type": "memory_20250818", "name": "memory"}]
    results = requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2", "t3"]
    assert results[0]["content"].startswith(LISTING_HEADER) and "is_error" not in results[0]
    assert results[1]["content"] == "File created successfully at: /memories/me.md"
    assert results[2]["is_error"] is True
    assert "Path must start with /memories" in json.dumps(results[2]["content"])
    assert set(facts(memory)) == {"The user lives in Berlin."}


def test_tool_runner_round_trip(tool: CueMemMemoryTool, memory: Memory) -> None:
    requests: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=REPLIES[len(requests) - 1])

    client = anthropic.Anthropic(
        api_key="test",
        base_url="http://anthropic.test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handle)),
        max_retries=0,
    )
    runner = client.beta.messages.tool_runner(
        model="claude-opus-5-5",
        max_tokens=1024,
        tools=[tool],
        messages=[{"role": "user", "content": "I live in Berlin. Please remember that."}],
    )
    final = runner.until_done()
    assert final.content[0].type == "text" and len(requests) == 2
    check_round_trip(requests, memory)


def test_async_tool_with_the_async_runner(n3_settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = [ScriptedModel(note_facts, reconcile=moved)]
    requests: list[dict[str, Any]] = []

    async def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(await request.aread()))
        return httpx2.Response(200, json=REPLIES[len(requests) - 1])

    async def main() -> None:
        async with AsyncMemory(n3_settings, http_client=fake_api.client(), background=False) as m:
            tool = AsyncCueMemMemoryTool(m, user_id="alice", wait=True)
            client = anthropic.AsyncAnthropic(
                api_key="test",
                base_url="http://anthropic.test",
                http_client=anthropic.DefaultAsyncHttpxClient(
                    transport=httpx2.MockTransport(handle)
                ),
                max_retries=0,
            )
            runner = client.beta.messages.tool_runner(
                model="claude-opus-5-5",
                max_tokens=1024,
                tools=[tool],
                messages=[{"role": "user", "content": "I live in Berlin. Please remember that."}],
            )
            await runner.until_done()
            check_round_trip(requests, m.sync)
            assert "Berlin" in await tool.call({"command": "view", "path": "/memories/me.md"})
            tool.close()

    asyncio.run(main())


def test_with_recall_puts_the_evidence_before_the_last_user_message(
    tool: CueMemMemoryTool, memory: Memory
) -> None:
    question = [{"role": "user", "content": "Where do I live?"}]
    same, evidence = with_recall(question, memory, user_id="alice")
    assert same == question and evidence.level == "none"  # 还没有记忆

    run(tool, command="create", path="/memories/me.md", file_text="- The user lives in Berlin.\n")
    out, evidence = with_recall(question, memory, user_id="alice", mode="l1")
    assert question == [{"role": "user", "content": "Where do I live?"}]  # 原列表不动
    content = out[-1]["content"]
    assert content.startswith(f"<memory>\n{MEMORY_NOTE}\n\n")
    assert content.endswith("</memory>\n\nWhere do I live?")
    assert "The user lives in Berlin." in content
    assert evidence.level == "L1" and evidence.recalled >= 1 and evidence.block in content

    blocks = [{"role": "user", "content": [{"type": "text", "text": "Where do I live?"}]}]
    out, _ = with_recall(blocks, memory, user_id="alice", mode="l1")
    assert out[-1]["content"][0]["text"].startswith("<memory>")
    assert out[-1]["content"][1] == {"type": "text", "text": "Where do I live?"}

    profile_only, evidence = with_recall(question, memory, user_id="alice", mode="l0")
    assert evidence.level == "L0"
    assert "About the user (core memory" in profile_only[-1]["content"]
    assert "Retrieved conversation excerpts" not in profile_only[-1]["content"]

    tool_turn = [
        *question,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "ok"}]},
    ]
    unchanged, evidence = with_recall(tool_turn, memory, user_id="alice")
    assert unchanged == tool_turn and evidence.level == "none"
