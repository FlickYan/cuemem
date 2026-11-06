"""访问令牌与作用域（多租户，N12 之后）：一个令牌能访问哪些用户的记忆。

令牌都以 Authorization: Bearer <令牌> 发送（MemoryClient 与 OpenAI SDK 的 api_key；界面登录时是
密码）。两种：

- 管理令牌：环境变量 CUEMEM_API_TOKEN（N11 起就有），不受作用域限制，包括 /api/v1/reset 删全部
  与不是由 user_id、agent_id 得来的空间（评测、手工起的名字）；
- 限定作用域的令牌：环境变量 CUEMEM_API_KEYS_FILE 指向的 YAML 文件，或环境变量 CUEMEM_API_KEYS
  里同样内容的 YAML / JSON 文本（只能设一个）。环境变量里的文本启动时读一次；文件在 cuemem
  serve / proxy 运行期间改了会重新读入（KeyRing，第六轮体检），不用重启。cuemem keys
  add|list|revoke 管理这个文件（生成令牌、只存摘要）：

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
        max_concurrent: 4            # /api、/api/v1、/v1、/mcp 同时在处理的请求最多这么多个
                                     # （429；不写时是 CUEMEM_MAX_CONCURRENT_PER_KEY，默认 8）

  后两项只管记忆代理（/v1/chat/completions、/v1/responses）；别的接口用配置里的角色与模型。
  models 里的别名只放行这个别名（连同目录里它的 extra_body）：写 deepseek-flash 的令牌不能用
  同一个模型 ID 打开思考的 deepseek-flash-thinking。只写 max_output_tokens 管不住调用方的开关
  （enable_thinking、thinking 原样转发），要固定它们得写 models。

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
import json
import logging
import os
import re
import secrets
import threading
import time
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

logger = logging.getLogger(__name__)

TOKEN_ENV = "CUEMEM_API_TOKEN"
KEYS_FILE_ENV = "CUEMEM_API_KEYS_FILE"
KEYS_ENV = "CUEMEM_API_KEYS"
# 放在模块级：注解是字符串（from __future__ import annotations），FastAPI 解析不到函数里的局部变量
_BEARER = HTTPBearer(auto_error=False)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
# Keys.parse 用的加载器：只认安全的标签；装了 libyaml 时用 C 写的那个
_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
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
    None 是不限。max_concurrent：同时在处理的请求数的上限（api/limits.py），None 时用
    CUEMEM_MAX_CONCURRENT_PER_KEY（默认 8）；管理令牌不限。"""

    name: str
    admin: bool = False
    scopes: tuple[Scope, ...] = ()
    rate_per_minute: int | None = None
    models: tuple[str, ...] | None = None
    max_output_tokens: int | None = None
    max_concurrent: int | None = None

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
            # 有 libyaml 时用 C 写的加载器（解出的对象一样，几百个令牌时快近十倍）：运行中的服务在
            # 事件循环上重新读入改过的文件
            raw = yaml.load(text, Loader=_SAFE_LOADER)
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


# KeyRing 最多隔这么多秒看一次文件有没有变（stat，单调时钟）
RELOAD_INTERVAL_S = 1.0


class KeyRing:
    """会重新读入的令牌（第六轮体检）：读的接口与 Keys 相同（enabled、admin、scoped、names、
    weak、resolve、login、by_username），交给当前的 Keys 快照。cuemem serve 与 proxy 用它：吊销、
    新增限定作用域的令牌改 CUEMEM_API_KEYS_FILE 的文件就行，不用重启（重启时进行中的 run 只能等
    下一个进程接手）。

    path 为 None 时是静态的（没设 CUEMEM_API_KEYS_FILE：只有管理令牌，或 CUEMEM_API_KEYS 里的
    文本），从不 stat，与直接用 Keys 完全相同。给了 path 时 current() 最多每 RELOAD_INTERVAL_S 秒
    stat 一次文件，(st_ino, st_mtime_ns, st_size) 变了就重新读、解析，用一次赋值换上新的快照
    （原子地改名换上的文件 inode 不同，修改时间与大小相同也看得出来）。管理令牌
    （CUEMEM_API_TOKEN）是启动时的那个，不重新读。

    下面这些时候不换，留着原来的快照，日志里记一条警告（只写令牌的名字），文件再变时再试：读不
    出（文件不在了）、解析出错、新的一个令牌都没有而原来有（不让清空的文件关掉鉴权），以及有短于
    MIN_TOKEN_CHARS 个字符的令牌而 allow_weak 为 False（cuemem serve / proxy 这时会拒绝启动，见
    cli._require_token）。限速的令牌桶按令牌的名字记（api/limits.py），重新读入后照旧。

    假定一个进程（uvicorn 一个 worker，cuemem serve 与 proxy 都这样启动）；不处理 SIGHUP。

    换上新的快照时，令牌换了（同一个名字的 sha256 变了）或去掉了的名字交给 on_rotate 登记的
    回调（第六轮复审）：挂着界面时 create_app 用它把这些名字在界面里的登录退出（Gradio 在内存里
    记着 cookie → 用户名，by_username 只按名字找令牌）。以前同名换了令牌、或 cuemem keys revoke
    之后又 keys add 回来，用泄露的旧令牌登录的界面会话照旧能用这个令牌的作用域。"""

    def __init__(
        self,
        keys: Keys,
        path: Path | None = None,
        *,
        admin_token: str | None = None,
        allow_weak: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._keys = keys
        self.path = path
        self._admin_token = admin_token
        self.allow_weak = allow_weak
        self._clock = clock
        self._lock = threading.Lock()
        self._retired: frozenset[str] = frozenset()  # 重新读入时去掉了的令牌的名字（by_username）
        self._on_rotate: list[Callable[[frozenset[str]], None]] = []
        self._checked = clock()
        self._signature = self._stat() if path is not None else None

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, allow_weak: bool = True
    ) -> KeyRing:
        """同 Keys.from_env；设了 CUEMEM_API_KEYS_FILE 时会重新读入那个文件。先 stat 再读：读的
        时候文件又变了，下一次 current() 看得出来。"""
        env = os.environ if environ is None else environ
        path = env.get(KEYS_FILE_ENV) or None
        if path is None:
            return cls(Keys.from_env(env), allow_weak=allow_weak)
        resolved = Path(path).expanduser()
        signature = _signature(resolved)
        ring = cls(
            Keys.from_env(env),
            resolved,
            admin_token=env.get(TOKEN_ENV) or None,
            allow_weak=allow_weak,
        )
        ring._signature = signature
        return ring

    def _stat(self) -> tuple[int, int, int] | None:
        return _signature(self.path) if self.path is not None else None

    def on_rotate(self, callback: Callable[[frozenset[str]], None]) -> None:
        """换上新的快照之后调用 callback(名字)：令牌换了或去掉了的限定作用域的令牌的名字（有
        这样的名字时才调用）。在重新读入的那个线程里调用，拿着重新读入的锁：不能再调用 current()
        等会重新读入的方法。"""
        self._on_rotate.append(callback)

    def current(self) -> Keys:
        """当前的令牌；到了时间就看一眼文件（见类的说明）。另一个线程正在看时直接用原来的。"""
        if self.path is None or self._clock() - self._checked < RELOAD_INTERVAL_S:
            return self._keys
        if not self._lock.acquire(blocking=False):
            return self._keys
        try:
            self._checked = self._clock()
            signature = self._stat()
            if signature != self._signature:
                self._signature = signature
                self._reload(signature is not None)
        finally:
            self._lock.release()
        return self._keys

    def _reload(self, exists: bool) -> None:
        old, where = self._keys, f"{KEYS_FILE_ENV}={self.path}"
        keep = "，仍用原来的令牌"
        if not exists:
            logger.warning("%s 不在了%s", where, keep)
            return
        assert self.path is not None
        try:
            new = Keys.parse(
                self.path.read_text(encoding="utf-8"), self._admin_token, source=str(self.path)
            )
        except OSError as exc:
            logger.warning("读不出 %s：%s%s", where, exc.strerror or type(exc).__name__, keep)
            return
        except AccessKeysError as exc:  # 出错信息里不带令牌本身（Keys.parse）
            logger.warning("%s 重新读入失败：%s%s", where, exc, keep)
            return
        if old.enabled and not new.enabled:
            logger.warning("%s 里一个令牌都没有了：不会因此关掉鉴权%s", where, keep)
            return
        if new.weak and not self.allow_weak:
            logger.warning(
                "%s 里这些令牌短于 %d 个字符：%s%s",
                where,
                MIN_TOKEN_CHARS,
                "、".join(new.weak),
                keep,
            )
            return
        gone = [n for n in old.names if n not in new.names]
        self._retired = (self._retired | set(gone)) - set(new.names)
        self._keys = new  # 一次赋值：别的线程拿到的不是旧的就是新的
        digests = {p.name: digest for digest, p in new.scoped}
        rotated = frozenset(p.name for digest, p in old.scoped if digests.get(p.name) != digest)
        if rotated:
            for callback in self._on_rotate:
                callback(rotated)
        added = [n for n in new.names if n not in old.names]
        logger.warning(
            "已重新读入 %s：%d 个限定作用域的令牌（新增：%s；去掉：%s）",
            where,
            len(new.names),
            "、".join(added) or "无",
            "、".join(gone) or "无",
        )

    @property
    def enabled(self) -> bool:
        return self.current().enabled

    @property
    def admin(self) -> bytes | None:
        return self.current().admin

    @property
    def scoped(self) -> tuple[tuple[bytes, Principal], ...]:
        return self.current().scoped

    @property
    def names(self) -> list[str]:
        return self.current().names

    @property
    def weak(self) -> tuple[str, ...]:
        return self.current().weak

    def resolve(self, token: str) -> Principal | None:
        return self.current().resolve(token)

    def login(self, username: str, password: str) -> bool:
        return self.current().login(username, password)

    def by_username(self, username: str | None) -> Principal | None:
        """同 Keys.by_username，只是去掉了的令牌的名字不再算任何令牌：界面里用它登录、还没退出的
        人不能接着用（Keys.by_username 会把不认识的用户名当成用管理令牌登录的）。管理令牌登录时
        也不能用这些名字做用户名。"""
        keys = self.current()
        if username is not None and username in self._retired and keys.enabled:
            return None
        return keys.by_username(username)


def _signature(path: Path) -> tuple[int, int, int] | None:
    """文件的 (st_ino, st_mtime_ns, st_size)；不在、读不出时 None。"""
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_ino, st.st_mtime_ns, st.st_size


# 认令牌的函数都收这两种（KeyRing 读的接口与 Keys 相同）
AnyKeys = Keys | KeyRing


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
    max_concurrent: _PositiveInt | None = None


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
        max_concurrent=entry.max_concurrent,
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


# ---------- cuemem keys：改令牌文件（第六轮体检） ----------
# 按文本改，不用 yaml.safe_dump 重写：文件里的注释与写法留着。改完由调用方用 Keys.parse 核对
# 结果（cli 的 keys add、keys revoke），对不上时不写


def _keys_node(text: str) -> yaml.SequenceNode | None:
    """顶层 keys: 的列表节点（带行列位置）；不是 YAML、没有这一项时 None。"""
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return None
    if not isinstance(root, yaml.MappingNode):
        return None
    for key, value in root.value:
        if isinstance(key, yaml.ScalarNode) and key.value == "keys":
            return value if isinstance(value, yaml.SequenceNode) else None
    return None


def _yaml_value(values: Sequence[str]) -> str:
    """一个值（或几个值的列表）的 YAML：JSON 的双引号写法，"*" 与带冒号的 ID 也不会写错。"""
    items = [json.dumps(v, ensure_ascii=False) for v in values]
    return items[0] if len(items) == 1 else f"[{', '.join(items)}]"


def key_entry(
    name: str,
    sha256: str,
    *,
    users: Sequence[str] = (),
    agents: Sequence[str] = (),
    rate_per_minute: int | None = None,
    models: Sequence[str] | None = None,
    max_output_tokens: int | None = None,
    max_concurrent: int | None = None,
    indent: str = "  ",
) -> str:
    """一个限定作用域的令牌的 YAML（块状写法，列表项缩进 indent，以换行结尾）：一项作用域，
    users、agents 至少给一个（格式见模块说明）。只写摘要，不写令牌本身。"""
    pad = indent + "  "
    scope = [f"user: {_yaml_value(users)}"] if users else []
    scope += [f"agent: {_yaml_value(agents)}"] if agents else []
    lines = [
        f"{indent}- name: {_yaml_value([name])}",
        f'{pad}sha256: "{sha256}"',  # 带引号：全是数字（或像 1e5…）的摘要不加引号会读成数
        f"{pad}scopes:",
        f"{pad}  - {scope[0]}",
        *(f"{pad}    {item}" for item in scope[1:]),
    ]
    if rate_per_minute is not None:
        lines.append(f"{pad}rate_per_minute: {rate_per_minute}")
    if models is not None:
        lines.append(f"{pad}models: [{', '.join(_yaml_value([m]) for m in models)}]")
    if max_output_tokens is not None:
        lines.append(f"{pad}max_output_tokens: {max_output_tokens}")
    if max_concurrent is not None:
        lines.append(f"{pad}max_concurrent: {max_concurrent}")
    return "\n".join(lines) + "\n"


def append_key(text: str, entry: Callable[[str], str]) -> str | None:
    """在块状写法的 keys: 列表末尾加上 entry(列表项的缩进) 的文本，返回整个文件的文本。空的
    keys: [] 改成块状写法、加上这一项。keys: 不是块状写法的列表（[{...}] 这样的流式写法）、
    没有一项时 None：由调用方打印出来让运营者自己贴。keys 是文件里唯一的顶层项（Keys.parse
    不收别的），所以文件末尾还在这个列表里。"""
    node = _keys_node(text)
    if node is not None and node.flow_style and not node.value:
        # keys: []（例如 cuemem keys revoke 吊销了最后一个）：去掉 []，改成块状写法的第一项
        start, end = node.start_mark.index, node.end_mark.index
        rest = text[:start].rstrip(" ") + text[end:]
        return (rest if rest.endswith("\n") else rest + "\n") + entry("  ")
    if node is None or node.flow_style or not node.value:
        return None
    return (text if text.endswith("\n") else text + "\n") + entry(" " * node.start_mark.column)


def _end(node: yaml.Node) -> tuple[int, int | None]:
    """节点的内容到哪一行（从 0 起）的哪一列为止（None：到行尾）。块状的映射与列表的 end_mark
    在下一个记号（下一个列表项的 "-"）那里，所以找最后一个值。"""
    while isinstance(node, yaml.MappingNode | yaml.SequenceNode) and not node.flow_style:
        if not node.value:
            break
        last = node.value[-1]
        node = last[1] if isinstance(node, yaml.MappingNode) else last
    mark = node.end_mark
    return (mark.line - 1, None) if mark.column == 0 else (mark.line, mark.column)


def _find(node: yaml.SequenceNode, name: str) -> yaml.MappingNode | None:
    for item in node.value:
        if isinstance(item, yaml.MappingNode) and any(
            isinstance(key, yaml.ScalarNode)
            and key.value == "name"
            and isinstance(value, yaml.ScalarNode)
            and value.value == name
            for key, value in item.value
        ):
            return item
    return None


def key_lines(text: str, name: str) -> tuple[int, int] | None:
    """名叫 name 的令牌在文件里的行（从 0 起，含两端，不含它后面的注释与空行）；找不到时 None。"""
    node = _keys_node(text)
    item = _find(node, name) if node is not None else None
    if item is None:
        return None
    first, lines = item.start_mark.line, text.splitlines()
    while first > 0 and lines[first - 1].strip() == "-":  # "-" 单独一行的写法：从 "-" 算起
        first -= 1
    return first, _end(item)[0]


def remove_key(text: str, name: str) -> str | None:
    """删掉名叫 name 的令牌那几行（key_lines），返回整个文件的文本；它不是自成几行的列表项（与
    别的内容挤在同一行、"-" 单独一行这样的写法）时 None，由调用方让运营者自己删。删的是最后一个
    时 keys: 改成 keys: []（keys: 后面什么都没有是 null，Keys.parse 不收）。"""
    node = _keys_node(text)
    item = _find(node, name) if node is not None else None
    if node is None or item is None or node.flow_style:
        return None
    lines = text.splitlines(keepends=True)
    first, (last, column) = item.start_mark.line, _end(item)
    tail = lines[last][column:] if column is not None else ""
    if (
        not lines[first].startswith(" " * node.start_mark.column + "- ")
        or lines[first][: item.start_mark.column].strip() != "-"
        or (tail.strip() and not tail.lstrip().startswith("#"))
    ):
        return None
    kept = lines[:first] + lines[last + 1 :]
    if len(node.value) == 1:  # 删的是最后一个
        heads = [i for i, line in enumerate(lines[:first]) if line.startswith("keys:")]
        if len(heads) != 1:
            return None
        kept[heads[0]] = "keys: []" + lines[heads[0]][len("keys:") :]
    return "".join(kept)


# ---------- FastAPI ----------


UNAUTHORIZED = "缺少或错误的访问令牌"


def bearer_principal(keys: AnyKeys, authorization: str | None) -> Principal | None:
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

    以 utf-8' 开头却写错、百分号编码的字节不是 UTF-8，或者解出来的值（utf-8'' 与不是 ASCII 的）
    带控制字符时 ValueError；ASCII 的值原样返回，不查控制字符。"""
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


def authenticate(keys: AnyKeys, request: Request) -> Principal:
    """认出 Bearer 令牌（bearer_principal），记在 request.state.principal 上（principal_of
    读它）。令牌不对时 401。"""
    principal = bearer_principal(keys, request.headers.get("Authorization"))
    if principal is None:
        raise HTTPException(401, UNAUTHORIZED, headers={"WWW-Authenticate": "Bearer"})
    request.state.principal = principal
    return principal


def guard(keys: AnyKeys) -> list[Any]:
    """路由的依赖：authenticate。FastAPI 先读完、解析完请求体才执行依赖，所以带请求体的路由还要
    用 authenticated_route 做路由类（令牌不对时不读请求体）；这个依赖照样挂着，OpenAPI 的安全
    方案（/docs 的 Authorize）由它的 HTTPBearer 而来。"""

    def check(
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
    ) -> None:
        authenticate(keys, request)  # 令牌由 authenticate 自己从请求头读

    return [Depends(check)]


def authenticated_route(keys: AnyKeys) -> type[APIRoute]:
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


def ui_login(keys: AnyKeys) -> Callable[[str, str], bool] | None:
    """界面（Gradio）的登录检查；没有配令牌时 None（不登录）。"""
    return keys.login if keys.enabled else None
