"""MCP Server（N8）：把 CueMem 的记忆当作工具交给其他 Agent（Claude Code、Claude Desktop……）。

四个工具：

- remember(text)：记一条笔记（偏好、事实、待办……），立即可检索，后台抽取成事实；
- recall(query, deep)：至少做一次快速召回（调用方已经决定要查记忆），控制器决定要不要查得更深；
  把带编号与日期的事实和对话片段原样交给调用方的模型去推理——这里不调用回答模型；
- forget(ids)：按 recall 给出的编号遗忘，删掉它们出自的对话，再重建派生记忆；
- profile()：用户画像（核心记忆）。

在 Claude Code 里注册（stdio）：
    claude mcp add cuemem -- uv --directory /path/to/rag run cuemem mcp --space me
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from cuemem.api.service import ChatService
from cuemem.evidence import format_listing

INSTRUCTIONS = """CueMem is a long-term memory of past conversations with this user.
Call recall before answering anything that may depend on the user's past (preferences, plans, people, things they own or did). Call remember when the user shares something worth keeping. Cite memory ids when you rely on them, and use forget when the user asks you to delete something."""  # noqa: E501


def build_server(service: ChatService, space_id: str) -> MCPServer[Any]:
    # 日志只要警告以上（stdout 是协议通道，日志走 stderr，但每个 HTTP 请求一行太吵）
    server: MCPServer[Any] = MCPServer("cuemem", instructions=INSTRUCTIONS, log_level="WARNING")

    @server.tool()
    def remember(text: str) -> str:
        """Save something worth remembering about the user (a preference, a fact, a plan).
        It can be recalled right away and is turned into structured facts in the background."""
        saved = service.remember(space_id, text)
        return f"Saved as {saved['turn_id']}."

    @server.tool()
    def recall(query: str, deep: bool = False) -> str:
        """Recall memories relevant to the query. The memory controller decides whether a fast
        search is enough or a deeper one is needed; deep=true forces the deep search. Returns the
        user's profile, facts with validity, and conversation excerpts, each with an id and a
        date."""
        found = service.recall(space_id, query, mode="l2" if deep else "auto", at_least_l1=True)
        return format_recall(found)

    @server.tool()
    def forget(ids: list[str]) -> str:
        """Forget memories by the ids that recall returned (fact ids or excerpt ids). Deletes
        the conversation they came from and rebuilds everything derived from it."""
        report = service.forget_ids(space_id, ids)
        return (
            f"Deleted {report.turns_deleted} conversation turn(s); "
            f"facts {report.facts_before} -> {report.facts_after}."
        )

    @server.tool()
    def profile() -> str:
        """The user's profile: the most important facts that are currently true."""
        return service.profile(space_id) or "Nothing is known about the user yet."

    return server


# MCP 的 recall 与各框架集成的 recall 工具用同一份文字格式（见 evidence.py）
format_recall = format_listing
