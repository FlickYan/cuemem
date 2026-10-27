"""MCP Server（N8）：把 CueMem 的记忆当作工具交给其他 Agent（Claude Code、Claude Desktop……）。

四个工具：

- remember(text)：记一条笔记（偏好、事实、待办……），立即可检索，后台抽取成事实；
- recall(query, deep)：至少做一次快速召回（调用方已经决定要查记忆），控制器决定要不要查得更深；
  把带编号与日期的事实和对话片段原样交给调用方的模型去推理——这里不调用回答模型；
- forget(ids)：按 recall 给出的编号遗忘，删掉它们出自的对话，再重建派生记忆；
- profile()：用户画像（核心记忆）。

两种接法，工具与返回的文字相同：

- stdio（cuemem mcp --space me）：一个进程一个空间，独占数据目录（与 cuemem serve 不能同时开）：
    claude mcp add cuemem -- uv --directory /path/to/rag run cuemem mcp --space me
- HTTP（cuemem serve 的 /mcp，第三轮体检）：与界面、接口共用一个记忆系统，鉴权与作用域与 /api
  相同（McpEndpoint）；请求头 X-CueMem-User（可加 X-CueMem-Agent）选用户：
    claude mcp add --transport http cuemem http://127.0.0.1:7860/mcp \\
        --header "Authorization: Bearer $KEY" --header "X-CueMem-User: alice"
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from cuemem.api.auth import UNAUTHORIZED, Keys, bearer_principal
from cuemem.api.failures import provider_failure
from cuemem.api.service import ChatService
from cuemem.evidence import forget_summary, format_listing
from cuemem.sdk import space_for

INSTRUCTIONS = """CueMem is a long-term memory of past conversations with this user.
Call recall before answering anything that may depend on the user's past (preferences, plans, people, things they own or did). Call remember when the user shares something worth keeping. Cite memory ids when you rely on them, and use forget when the user asks you to delete something."""  # noqa: E501

# /mcp 上这次请求的记忆空间（McpEndpoint 设置）。同步的工具在线程池里执行，anyio 把上下文带过去
_SPACE: ContextVar[str] = ContextVar("cuemem_mcp_space")


def build_server(service: ChatService, space_id: str | None = None) -> MCPServer[Any]:
    """space_id：stdio 的那一个空间；None 时每次调用的空间由 McpEndpoint 按请求设置（_SPACE）。"""
    # 日志只要警告以上（stdout 是协议通道，日志走 stderr，但每个 HTTP 请求一行太吵）
    server: MCPServer[Any] = MCPServer("cuemem", instructions=INSTRUCTIONS, log_level="WARNING")

    def space() -> str:
        return space_id if space_id is not None else _SPACE.get()

    @server.tool()
    def remember(text: str) -> str:
        """Save something worth remembering about the user (a preference, a fact, a plan).
        It can be recalled right away and is turned into structured facts in the background."""
        with _provider_errors():
            saved = service.remember(space(), text)
        return f"Saved as {saved['turn_id']}."

    @server.tool()
    def recall(query: str, deep: bool = False) -> str:
        """Recall memories relevant to the query. The memory controller decides whether a fast
        search is enough or a deeper one is needed; deep=true forces the deep search. Returns the
        user's profile, facts with validity, and conversation excerpts, each with an id and a
        date."""
        with _provider_errors():
            found = service.recall(space(), query, mode="l2" if deep else "auto", at_least_l1=True)
        return format_recall(found)

    @server.tool()
    def forget(ids: list[str]) -> str:
        """Forget memories by the ids that recall returned (fact ids or excerpt ids). Deletes
        the conversation they came from and rebuilds everything derived from it."""
        with _provider_errors():  # 删除之后重建出错时不算出错，见 forget_summary
            report = service.forget_ids(space(), ids)
        return forget_summary(report)

    @server.tool()
    def profile() -> str:
        """The user's profile: the most important facts that are currently true."""
        return service.profile(space()) or "Nothing is known about the user yet."

    return server


class McpEndpoint:
    """cuemem serve 的 /mcp（api/app.py 挂成 Starlette 的 Route）：MCP 的 Streamable HTTP，无状态、
    JSON 响应，工具与 stdio 相同。以前 MCP 只有 stdio，独占数据目录：开着 cuemem serve 时 Claude
    Code 用不上同一份记忆，也没有多租户的 MCP。

    每个请求在交给 MCP 之前：
    - 认令牌（与 /api 相同的 Bearer 令牌，auth.bearer_principal），不对时 401；
    - 定空间：X-CueMem-User、X-CueMem-Agent 头（与记忆代理相同，sdk.space_for）；都没给时是
      令牌作用域里的第一个用户（Principal.home），管理令牌、不鉴权与全是通配的令牌没有，400；
    - 空间在令牌的作用域之外时 403，什么都不做；
    - 空间放进 _SPACE，工具读它。

    Host 与 Origin 由整个应用的 TrustedHostMiddleware 与 SameOriginOnly 检查（与 /api 相同），
    MCP SDK 自己的 Host 检查关掉：它只认 127.0.0.1 等带端口的 Host，Docker、局域网的地址与不带
    端口的 Host 都是 421。是 ASGI 的可调用对象（不是函数）：Starlette 把普通函数当成请求处理函数。
    """

    def __init__(self, service: ChatService, keys: Keys) -> None:
        self.server = build_server(service)
        self.keys = keys
        self._manager: StreamableHTTPSessionManager | None = None

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """应用的 lifespan 里进入（create_app 接在原来的 lifespan 上）。会话管理器只能 run 一次：
        每次进入都新建一个（streamable_http_app 新建它，记在 server.session_manager 上）。"""
        self.server.streamable_http_app(
            stateless_http=True,
            json_response=True,
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        manager = self.server.session_manager
        async with manager.run():
            self._manager = manager
            try:
                yield
            finally:
                self._manager = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        space = self._space(Headers(scope=scope))
        if isinstance(space, JSONResponse):
            await space(scope, receive, send)
            return
        if self._manager is None:  # 应用的 lifespan 没有运行（uvicorn 总会运行它）
            refused = {"detail": "MCP is not running: the app's lifespan has not started."}
            await JSONResponse(refused, 503)(scope, receive, send)
            return
        token = _SPACE.set(space)
        try:
            await self._manager.handle_request(scope, receive, send)
        finally:
            _SPACE.reset(token)

    def _space(self, headers: Headers) -> str | JSONResponse:
        """这个请求的记忆空间，或者拒绝它的响应（401、400、403；见类的说明）。"""
        principal = bearer_principal(self.keys, headers.get("authorization"))
        if principal is None:
            return JSONResponse({"detail": UNAUTHORIZED}, 401, {"WWW-Authenticate": "Bearer"})
        user_id = headers.get("x-cuemem-user") or None
        agent_id = headers.get("x-cuemem-agent") or None
        space = space_for(user_id, agent_id) if user_id or agent_id else principal.home()
        if space is None:
            message = (
                "Send the X-CueMem-User header (and optionally X-CueMem-Agent) to choose whose "
                "memories these tools use: this token has no single user of its own."
            )
            return JSONResponse({"detail": message}, 400)
        if not principal.allows(space):
            named = (("user", user_id), ("agent", agent_id))
            who = " with ".join(f"{kind} {value!r}" for kind, value in named if value)
            message = (
                f"This API key cannot access the memories of {who}: use a user in its scope "
                "(the X-CueMem-User and X-CueMem-Agent headers)."
            )
            return JSONResponse({"detail": message}, 403)
        return space


@contextmanager
def _provider_errors() -> Iterator[None]:
    """供应商出错（限流、余额不足、超时……）时给调用方的模型一句看得懂的话。别的异常 MCP 只回
    "Error executing tool <name>"：模型分不清是 CueMem 坏了，还是该等一会儿再试。"""
    try:
        yield
    except Exception as exc:
        failure = provider_failure(exc)
        if failure is None:
            raise
        raise ToolError(failure.for_model()) from exc


# MCP 的 recall 与各框架集成的 recall 工具用同一份文字格式（见 evidence.py）
format_recall = format_listing
