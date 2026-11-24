"""集成文档里 MCP 客户端的配置文件示例不走样（第十轮体检 Q16）。

不启动进程、不联网：代码块都是合法的 JSON，两种语言逐字相同；HTTP 的写法指向 /mcp 并带着认证
与用户的请求头；stdio 的写法里 cuemem 之后的参数是命令行真的认的。客户端自己的格式变了这里
查不出来，只管我们这一侧。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from cuemem.cli import app

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = {
    "zh": (DOCS / "integrations.md", "## MCP 客户端的配置文件"),
    "en": (DOCS / "en" / "integrations.md", "## MCP client configuration files"),
}


def _blocks(lang: str) -> list[Any]:
    """这一节（到下一个二级标题为止）里的 json 代码块，解析之后的样子。"""
    path, heading = PAGES[lang]
    lines = path.read_text("utf-8").splitlines()
    assert heading in lines, f"{path.name} 里没有“{heading}”一节"
    section: list[str] = []
    for line in lines[lines.index(heading) + 1 :]:
        if line.startswith("## "):
            break
        section.append(line)
    found = re.findall(r"```json\n(.*?)```", "\n".join(section), flags=re.S)
    assert found, f"{path.name} 的这一节里没有 json 代码块"
    return [json.loads(text) for text in found]


def _entries(blocks: list[Any]) -> list[dict[str, Any]]:
    """每个代码块里登记的 Server：mcpServers（Claude Code、Cursor、Claude Desktop）或 servers
    （VS Code）下面的各项。"""
    entries: list[dict[str, Any]] = []
    for block in blocks:
        servers = block.get("mcpServers") or block.get("servers")
        assert servers, block
        entries += servers.values()
    return entries


def test_snippets_parse_and_match_between_languages() -> None:
    assert _blocks("zh") == _blocks("en")


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_http_entries_point_at_mcp_with_identity_headers(lang: str) -> None:
    """管理员令牌、不设认证时 /mcp 没有 X-CueMem-User 就是 400：示例都带着它与 Authorization。"""
    remote = [entry for entry in _entries(_blocks(lang)) if "url" in entry]
    assert remote
    for entry in remote:
        assert entry["url"].endswith("/mcp"), entry
        assert {"Authorization", "X-CueMem-User"} <= set(entry["headers"]), entry
        assert entry["headers"]["Authorization"].startswith("Bearer $"), entry  # 令牌不写在文件里


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_stdio_entries_use_real_cli_arguments(lang: str) -> None:
    """stdio 的写法：cuemem 之后的参数加上 --help 交给命令行，认得才是 0（改了命令名或选项名，
    文档里的示例跟着失败）。"""
    local = [entry for entry in _entries(_blocks(lang)) if "command" in entry]
    assert local
    for entry in local:
        argv = [entry["command"], *entry["args"]]
        assert "cuemem" in argv, entry
        args = argv[argv.index("cuemem") + 1 :]
        assert args[0] == "mcp", entry
        result = CliRunner().invoke(app, [*args, "--help"])
        assert result.exit_code == 0, result.output
