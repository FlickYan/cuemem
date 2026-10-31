"""访问令牌与作用域（多租户，N12 之后）：一个令牌能访问哪些用户的记忆。

令牌都以 Authorization: Bearer <令牌> 发送（MemoryClient 与 OpenAI SDK 的 api_key；界面登录时是
密码）。两种：

- 管理令牌：环境变量 CUEMEM_API_TOKEN（N11 起就有），不受作用域限制，包括 /api/v1/reset 删全部
  与不是由 user_id、agent_id 得来的空间（评测、手工起的名字）；
- 限定作用域的令牌：环境变量 CUEMEM_API_KEYS_FILE 指向的 YAML 文件，或环境变量 CUEMEM_API_KEYS
  里同样内容的 YAML / JSON 文本（只能设一个），启动时读一次：

    keys:
      - name: alice-app               # 名字：界面登录的用户名
        sha256: 2cf24dba5fb0a30e26…   # 令牌的 sha256（printf %s "$TOKEN" | sha256sum），文件里
                                      # 不存令牌本身；也可以直接写 token: <令牌>
        scopes:
          - user: alice               # alice 的全部记忆：不带 Agent 的，与在任何 Agent 下的
      - name: shop-backend
        sha256: …
        scopes:
          - {user: "*", agent: shop}  # 每个用户在 shop 这个 Agent 下的记忆（shop/<用户>）
          - {agent: shop}             # shop 自己的记忆（agent/shop：SDK 只给 agent_id 时）

  user、agent 是一个 ID 或 ID 的列表；"*" 是任何 ID，以 "*" 结尾的是前缀（"acme:*"）。只写 user
  时不限 Agent；只写 agent 时只是这个 Agent 自己的记忆，不含用户在它下面的记忆。

  作用域管的是能访问谁的记忆，不是能花多少：任何有效的令牌都能让服务器调用运营者的模型与向量
  供应商（记忆代理、/api 的对话、/api/v1/answer、/mcp、每次写入之后的抽取）。每个令牌可以另外写
  （第四轮体检，都可选，不写时与以前完全相同）：

        rate_per_minute: 60          # /api、/api/v1、/v1、/mcp 写的请求（POST、PUT、DELETE）
                                     # 每分钟最多这么多个（429）；GET 不算
        models: [qwen3.8-flash]      # 记忆代理只能用这些模型（目录里的别名或模型 ID，之外 403）
        max_output_tokens: 2000      # 记忆代理转发的 max_tokens 不超过它，n 只能是 1

  后两项只管记忆代理（/v1/chat/completions、/v1/responses）；别的接口用配置里的角色与模型。

作用域之外：请求里点名了空间（user_id、agent_id、空间 ID）时 403；按 ID 找的记忆与会话当作不存在
（404，history 是空列表），不让 ID 成为探测别人有没有某条记忆的途径。/api/spaces 与 /api/v1/spaces
只列出作用域里的空间，/api/v1/reset 只删作用域里的空间（管理令牌删全部）。记忆代理按 user 字段与
X-CueMem-User、X-CueMem-Agent 头判断；没有用户、原样转发的请求不碰记忆，照常转发。/mcp（MCP 的
Streamable HTTP，api/mcp.py）也按这两个头判断，不给时是令牌作用域里的第一个用户（Principal.home）。
这两个头里不是 ASCII 的 ID 由 header_id 解（UTF-8 的百分号编码 utf-8''%E5%BC%A0%E4%B8%89，或原样
的 UTF-8）。

判断按记忆空间（sdk.space_for 的结果）解回的用户与 Agent（sdk.space_owner），不按请求里的 ID：
"agent/x" 一律算 Agent x 自己的记忆——用户 x 在名叫 agent 的 Agent 下的记忆也落在这个空间，只能
访问用户 x 的令牌借 agent_id="agent" 读不到它。路径里的 %2F 在路由之前就解成了 "/"，同样按解出来
的空间判断；不是由 user_id、agent_id 得来的空间只有管理令牌能访问。

令牌只比较 sha256 摘要：secrets.compare_digest 与每个令牌都比一遍，不提前返回。没有配任何令牌时
不鉴权（只适合本机演示，见 api/app.py）。cuemem mcp（stdio）一个进程一个空间，不经过这里。

令牌的长度（第四轮体检）：读令牌时记下短于 MIN_TOKEN_CHARS 个字符的（Keys.weak，只记名字），
cuemem serve / proxy 监听非本机地址时拒绝启动（cli._require_token）。sha256: 量不出长度，不记。
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote_to_bytes

import yaml
from fastapi import Depends, HTTPException, Request, Response
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cuemem.errors import AccessKeysError
from cuemem.sdk import space_for, space_owner

TOKEN_ENV = "CUEMEM_API_TOKEN"
KEYS_FILE_ENV = "CUEMEM_API_KEYS_FILE"
KEYS_ENV = "CUEMEM_API_KEYS"
# 放在模块级：注解是字符串（from __future__ import annotations），FastAPI 解析不到函数里的局部变量
_BEARER = HTTPBearer(auto_error=False)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
# 监听非本机地址时令牌至少这么多个字符：仓库与文档里的占位符（your-token、fake-token、
# 换成你的令牌）都更短；secrets.token_urlsafe(16) 是 22 个，openssl rand -hex 32 是 64 个
MIN_TOKEN_CHARS = 16


def _matches(value: str, patterns: Sequence[str]) -> bool:
    return any(value.startswith(p[:-1]) if p.endswith("*") else value == p for p in patterns)


@dataclass(frozen=True)
class Scope:
    """作用域里的一项。users：用户 ID 的写法（None：不带用户，即 Agent 自己的记忆）；agents：Agent
    ID 的写法（None：users 给了时不限 Agent，包括不带 Agent）。"""

    users: tuple[str, ...] | None = None
    agents: tuple[str, ...] | None = None

    def covers(self, user_id: str | None, agent_id: str | None) -> bool:
        if user_id is None:  # Agent 自己的记忆：要有一项只写了 agent
            return (
                self.users is None
                and self.agents is not None
                and agent_id is not None
                and _matches(agent_id, self.agents)
            )
        if self.users is None or not _matches(user_id, self.users):
            return False
        return self.agents is None or (agent_id is not None and _matches(agent_id, self.agents))


@dataclass(frozen=True)
class Principal:
    """一个令牌是谁、能访问什么。admin：不受作用域限制（管理令牌；没配令牌时的本机演示）。

    rate_per_minute、models、max_output_tokens：限定作用域的令牌可选的用量限制（见模块说明），
    None 是不限。"""

    name: str
    admin: bool = False
    scopes: tuple[Scope, ...] = ()
    rate_per_minute: int | None = None
    models: tuple[str, ...] | None = None
    max_output_tokens: int | None = None

    def allows(self, space: str | None) -> bool:
        """能不能访问这个记忆空间：空间解回的（user_id，agent_id）要在作用域里。"""
        if self.admin:
            return True
        owner = space_owner(space) if space else None
        return owner is not None and any(scope.covers(*owner) for scope in self.scopes)

    def home(self) -> str | None:
        """作用域里第一个写死的空间（界面的默认记忆空间）；全是通配时 None。"""
        for scope in self.scopes:
            for user_id in scope.users or (None,):
                for agent_id in scope.agents or (None,):
                    if any(x is not None and x.endswith("*") for x in (user_id, agent_id)):
                        continue
                    space = space_for(user_id, agent_id)
                    if self.allows(space):
                        return space
        return None


ADMIN = Principal("admin", admin=True)


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _short(name: str, token: str | None) -> tuple[str, ...]:
    """直接给出的令牌短于 MIN_TOKEN_CHARS 个字符时 (name,)，否则（包括没有）空。"""
    return (name,) if token and len(token) < MIN_TOKEN_CHARS else ()


@dataclass(frozen=True)
class Keys:
    """配置的令牌：管理令牌与限定作用域的令牌，都只存 sha256 摘要。一个都没有时不鉴权。

    weak：直接给出、短于 MIN_TOKEN_CHARS 个字符的令牌的名字（管理令牌记作 CUEMEM_API_TOKEN），
    由 from_token、parse 记下；不带令牌本身，不参与相等比较。"""

    admin: bytes | None = None
    scoped: tuple[tuple[bytes, Principal], ...] = ()
    weak: tuple[str, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        digests = [key for key, _ in self.scoped]
        names = [p.name for _, p in self.scoped]
        if len(set(names)) != len(names):
            raise AccessKeysError("访问令牌的名字重复了：每个令牌的 name 要不同")
        if len(set(digests)) != len(digests) or (self.admin is not None and self.admin in digests):
            raise AccessKeysError(
                f"同一个令牌出现了两次（或者与管理令牌 {TOKEN_ENV} 相同）：每个令牌要不同"
            )

    @property
    def enabled(self) -> bool:
        return self.admin is not None or bool(self.scoped)

    @property
    def names(self) -> list[str]:
        return [p.name for _, p in self.scoped]

    @classmethod
    def from_token(cls, token: str | None) -> Keys:
        """只有管理令牌（以前的 CUEMEM_API_TOKEN）；None 或空字符串时不鉴权。"""
        return cls(_digest(token) if token else None, weak=_short(TOKEN_ENV, token))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Keys:
        """CUEMEM_API_TOKEN 与 CUEMEM_API_KEYS_FILE（或 CUEMEM_API_KEYS）；写错时 AccessKeysError。
        文件的相对路径相对于当前目录。"""
        env = os.environ if environ is None else environ
        token = env.get(TOKEN_ENV) or None
        path, inline = env.get(KEYS_FILE_ENV) or None, env.get(KEYS_ENV) or None
        if path and inline:
            raise AccessKeysError(f"{KEYS_FILE_ENV} 与 {KEYS_ENV} 只能设一个")
        if path:
            try:
                text = Path(path).expanduser().read_text(encoding="utf-8")
            except OSError as exc:
                raise AccessKeysError(
                    f"读不出 {KEYS_FILE_ENV}={path}：{exc.strerror or exc}"
                ) from exc
            return cls.parse(text, token, source=path)
        if inline:
            return cls.parse(inline, token, source=KEYS_ENV)
        return cls.from_token(token)

    @classmethod
    def parse(cls, text: str, admin_token: str | None = None, *, source: str = "keys") -> Keys:
        """限定作用域的令牌的 YAML / JSON 文本（格式见模块说明）。出错信息里不带令牌本身。"""
        try:
            raw = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            where = f"（第 {mark.line + 1} 行）" if mark is not None else ""
            raise AccessKeysError(f"{source} 不是合法的 YAML{where}") from None
        try:
            parsed = _KeysIn.model_validate(raw)
        except ValidationError as exc:  # 不用 str(exc)：它会带上写错的值，可能就是令牌
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '(top level)'}: {e['msg']}"
                for e in exc.errors()
            )
            raise AccessKeysError(f"{source} 的格式不对：{problems}") from None
        scoped = tuple(_key(entry, f"{source}: keys[{i}]") for i, entry in enumerate(parsed.keys))
        weak = _short(TOKEN_ENV, admin_token) + tuple(
            name for entry in parsed.keys for name in _short(entry.name, entry.token)
        )
        return cls(_digest(admin_token) if admin_token else None, scoped, weak)

    def resolve(self, token: str) -> Principal | None:
        """令牌 → Principal；不认识时 None。与每个令牌都比一遍（常数时间比较，不提前返回）。"""
        digest = _digest(token)
        found: Principal | None = None
        if self.admin is not None and secrets.compare_digest(digest, self.admin):
            found = ADMIN
        for key, principal in self.scoped:
            if secrets.compare_digest(digest, key):
                found = principal
        return found

    def login(self, username: str, password: str) -> bool:
        """界面的登录：密码是令牌。限定作用域的令牌要用它的名字做用户名，管理令牌用户名任意
        （用某个令牌的名字登录时，界面按那个令牌的作用域来）。"""
        principal = self.resolve(password)
        if principal is None:
            return False
        return principal.admin or username == principal.name

    def by_username(self, username: str | None) -> Principal | None:
        """界面里登录的用户 → Principal（见 login）。用户名不是哪个令牌的名字时，只可能是用管理
        令牌登录的。"""
        if not self.enabled:
            return ADMIN
        if username is None:
            return None
        for _, principal in self.scoped:
            if principal.name == username:
                return principal
        return ADMIN if self.admin is not None else None


class _ScopeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user: str | list[str] | None = None
    agent: str | list[str] | None = None


# 正整数，不收 "5" 与 true 这样的写法（YAML 里写错了引号不该悄悄变成别的限制）
_PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class _KeyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sha256: str | None = None
    token: str | None = None
    scopes: list[_ScopeIn]
    rate_per_minute: _PositiveInt | None = None
    models: list[str] | None = None
    max_output_tokens: _PositiveInt | None = None


class _KeysIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keys: list[_KeyIn]


def _key(entry: _KeyIn, where: str) -> tuple[bytes, Principal]:
    if not entry.name or entry.name != entry.name.strip():
        raise AccessKeysError(f"{where}: name 不能为空，前后不能有空格（它是界面登录的用户名）")
    where = f"{where}（{entry.name}）"
    if (entry.sha256 is None) == (entry.token is None):
        raise AccessKeysError(f"{where}: sha256 与 token 要写一个，只写一个")
    if entry.sha256 is not None and not _SHA256.fullmatch(entry.sha256):
        raise AccessKeysError(f"{where}: sha256 要是 64 位十六进制")
    if entry.token is not None and not entry.token:
        raise AccessKeysError(f"{where}: token 不能为空")
    if not entry.scopes:
        raise AccessKeysError(f"{where}: scopes 不能为空")
    if entry.models is not None and (not entry.models or not all(entry.models)):
        raise AccessKeysError(f"{where}: models 要列出至少一个模型，不能有空的名字")
    scopes = tuple(_scope(scope, f"{where}.scopes[{i}]") for i, scope in enumerate(entry.scopes))
    digest = bytes.fromhex(entry.sha256) if entry.sha256 is not None else _digest(entry.token or "")
    return digest, Principal(
        entry.name,
        scopes=scopes,
        rate_per_minute=entry.rate_per_minute,
        models=tuple(entry.models) if entry.models is not None else None,
        max_output_tokens=entry.max_output_tokens,
    )


def _scope(scope: _ScopeIn, where: str) -> Scope:
    if scope.user is None and scope.agent is None:
        raise AccessKeysError(f"{where}: 至少写 user 与 agent 中的一个")
    return Scope(_patterns(scope.user, f"{where}.user"), _patterns(scope.agent, f"{where}.agent"))


def _patterns(value: str | list[str] | None, where: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    patterns = [value] if isinstance(value, str) else value
    if not patterns:
        raise AccessKeysError(f"{where}: 不能是空列表")
    for pattern in patterns:
        if not pattern or "*" in pattern[:-1]:
            raise AccessKeysError(
                f'{where}: {pattern!r} 不行：写一个 ID，"*"（任何 ID），或以 "*" 结尾的前缀'
            )
    return tuple(patterns)


# ---------- FastAPI ----------


UNAUTHORIZED = "缺少或错误的访问令牌"


def bearer_principal(keys: Keys, authorization: str | None) -> Principal | None:
    """Authorization 头（与 HTTPBearer 一样解析）→ Principal；令牌不对时 None。没有配任何令牌时
    不鉴权，按管理令牌算。authenticate 与 /mcp（api/mcp.py）共用。"""
    if not keys.enabled:
        return ADMIN
    scheme, token = get_authorization_scheme_param(authorization)
    return keys.resolve(token) if scheme.lower() == "bearer" and token else None


# RFC 8187 的 ext-value（charset'language'百分号编码）当作整个请求头的值
_EXT_VALUE = re.compile(r"(?i)utf-8'[A-Za-z0-9-]*'((?:[A-Za-z0-9!#$&+.^_`|~-]|%[0-9A-Fa-f]{2})*)")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def header_id(value: str | None) -> str | None:
    """请求头 X-CueMem-User、X-CueMem-Agent 里的 ID（第四轮体检）；没有或是空的时 None。

    Starlette 按 Latin-1 解请求头：curl、Go、aiohttp 原样发的 UTF-8 '张三' 以前成了
    'å¼\\xa0ä¸\\x89'，悄悄落进另一个空间；Node 的 fetch（Claude Code 的 --header）与 httpx 根本
    发不出 Latin-1 之外的字符。依次：

    - utf-8'<语言>'<百分号编码>（RFC 8187，不分大小写）：按 UTF-8 解，Node 的客户端只能这样发；
    - ASCII：原样；
    - 别的按 UTF-8 重新解；不是 UTF-8 时原样（Node 的 fetch 与 requests 按 Latin-1 发的 'José'，
      以前就对，不能变成 400）。

    以 utf-8' 开头却写错、解出来不是 UTF-8 或带控制字符时 ValueError。"""
    if not value:
        return None
    if value[:6].lower() == "utf-8'":
        match = _EXT_VALUE.fullmatch(value)
        if match is None:
            raise ValueError("malformed utf-8'' value")
        try:
            decoded = unquote_to_bytes(match.group(1)).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("the percent-encoded bytes are not UTF-8") from None
    elif value.isascii():
        return value
    else:
        try:
            decoded = value.encode("latin-1").decode("utf-8")
        except UnicodeDecodeError:  # 真的是 Latin-1 的字节
            decoded = value
    if _CONTROL.search(decoded):
        raise ValueError("control character")
    return decoded or None


def identity_headers(headers: Mapping[str, str]) -> tuple[str | None, str | None]:
    """X-CueMem-User 与 X-CueMem-Agent 的 ID（header_id；headers 的键是小写，Starlette 的 Headers
    也行）。写错时 ValueError，英文的说明写着是哪个头、非 ASCII 的 ID 怎么发。"""
    ids: list[str | None] = []
    for name in ("X-CueMem-User", "X-CueMem-Agent"):
        try:
            ids.append(header_id(headers.get(name.lower())))
        except ValueError as exc:
            raise ValueError(
                f"The {name} header is not a valid ID ({exc}): send IDs that are not ASCII as "
                f"UTF-8 percent-encoded, e.g. {name}: utf-8''%E5%BC%A0%E4%B8%89 for 张三."
            ) from None
    return ids[0], ids[1]


def authenticate(keys: Keys, request: Request) -> Principal:
    """认出 Bearer 令牌（bearer_principal），记在 request.state.principal 上（principal_of
    读它）。令牌不对时 401。"""
    principal = bearer_principal(keys, request.headers.get("Authorization"))
    if principal is None:
        raise HTTPException(401, UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    request.state.principal = principal
    return principal


def guard(keys: Keys) -> list[Any]:
    """路由的依赖：authenticate。FastAPI 先读完、解析完请求体才执行依赖，所以带请求体的路由还要
    用 authenticated_route 做路由类（令牌不对时不读请求体）；这个依赖照样挂着，OpenAPI 的安全
    方案（/docs 的 Authorize）由它的 HTTPBearer 而来。"""

    def check(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
    ) -> None:
        authenticate(keys, request)  # 令牌由 authenticate 自己从请求头读

    return [Depends(check)]


def authenticated_route(keys: Keys) -> type[APIRoute]:
    """路由类：先 authenticate，再交给 FastAPI 读请求体、解析参数、执行依赖。以前令牌只由 guard
    检查，而 FastAPI 在执行依赖之前就把请求体读完、解析完：不带令牌的请求也能让服务器解析任意大的
    JSON（内存是请求体的二十多倍），表单里的文件先写进临时文件，写坏的 JSON 回 422 而不是 401。
    接口的签名与 OpenAPI 不变。"""

    class AuthenticatedRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handler = super().get_route_handler()

            async def run(request: Request) -> Response:
                authenticate(keys, request)
                return await handler(request)

            return run

    return AuthenticatedRoute


def principal_of(request: Request) -> Principal:
    """这次请求的令牌（guard 认出来的）。路由没挂 guard 时报错：不放行，不当成管理令牌。"""
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        raise RuntimeError("route mounted without cuemem.api.auth.guard")
    return principal


def ui_login(keys: Keys) -> Callable[[str, str], bool] | None:
    """界面（Gradio）的登录检查；没有配令牌时 None（不登录）。"""
    return keys.login if keys.enabled else None
