"""Mem0 形状的 HTTP 记忆接口（/api/v1，N12 之后）：Python SDK（cuemem.sdk.Memory）的每个方法一个
接口，cuemem.client.MemoryClient 照同样的方法签名调用它们，代码在本地与远程之间只换一行。

    POST   /api/v1/memories                  add（messages、user_id、agent_id、run_id、metadata、
                                             expiration_date、timestamp、infer、wait）
    POST   /api/v1/memories/search           search（query、filters、top_k、mode、min_level、
                                             show_expired…）
    GET    /api/v1/memories                  get_all（filters 是 JSON 字符串）
    GET    /api/v1/memories/{id}             get（事实、对话片段或图片的 ID；不存在时 404）
    PUT    /api/v1/memories/{id}             update（text 或 data、metadata）
    GET    /api/v1/memories/{id}/history     history
    DELETE /api/v1/memories/{id}             delete（dry_run）
    POST   /api/v1/memories/batch-delete     delete_many（memory_ids、dry_run；MemoryClient 的
                                             batch_delete 也发到这里）
    DELETE /api/v1/memories                  delete_all（user_id、agent_id、run_id、dry_run：
                                             预演只能与 run_id 一起）
    GET    /api/v1/media/{id}                media（图片的原文件）
    POST   /api/v1/reset                     reset
    GET    /api/v1/status                    status（抽取的状态：待处理、失败的会话与原因）
    POST   /api/v1/context                   context（放进提示词的 <memory> 证据块，与记忆代理
                                             注入的相同；不调用模型）
    以及 answer、profile、runs（open_run）、runs/end（end_run）、consolidate、wait、export、
    import（load）、spaces。

作用域与 SDK 相同：user_id、agent_id 合起来定一个记忆空间，不碰原始的空间 ID（"/" 不用转义）。
鉴权与 /api 的其他接口相同（CUEMEM_API_TOKEN）；限定作用域的令牌（api/auth.py）只能访问自己的
用户：点名了作用域之外的 user_id、agent_id（包括写在 filters 里的）时 403，按 ID 找的记忆在作用域
之外时与不存在的一样（404；history 是空列表），spaces 只列出作用域里的，reset 只删作用域里的空间
（管理令牌删全部）。

报错是英文的 {"error": {"code", "type", "message"}}：type 是 SDK 抛出的异常类型（KeyError、
ValueError、TypeError 或 cuemem.errors 里的类型），MemoryClient 据此抛出同样的异常。请求体不合
格式（422）、令牌不对（401）在 /api/v1 下也写成这个形状（install_error_format）。模型与向量供应商
出错时的状态码与 /api 相同（api/failures.py：限流 429 带 Retry-After、余额不足 402、超时 504……），
code 是出错的类别（rate_limited、no_balance、timeout 等；向量库连不上是 503
vector_store_unavailable，超时是 504 vector_store_timeout）；向量接口的出错另带 upstream_status
（上游原来的状态码），MemoryClient 抛出的 EmbeddingAPIError 与本地一样带着 status 与 retry_after，
没有对应的本地异常时抛出的 MemoryAPIError 带着 status、code、type 与 retry_after（Retry-After 头）。

OpenAPI（/openapi.json 与它的快照 docs/openapi-v1.json，第三轮体检）：每个接口写明返回值的形状
（下面的 FactItem、SearchResult……）与出错的状态码（都是 V1ErrorBody），别的语言的客户端可以按它
生成。

图片只收 data URL：服务器不读自己磁盘上的文件（与记忆代理相同），MemoryClient 把本地路径先读成
data URL 再发；网络图片照 SDK 的规矩不下载。
"""

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Annotated, Any, Literal, cast

import anyio
from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.routing import APIRoute
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, WithJsonSchema
from starlette.exceptions import HTTPException as StarletteHTTPException

from cuemem import __version__
from cuemem.api.auth import (
    Keys,
    Principal,
    authenticated_route,
    guard,
    principal_of,
    space_cap_message,
    spaces_over_cap,
)
from cuemem.api.failures import provider_failure
from cuemem.core.models import Cardinality, FactCategory, SupersedeReason
from cuemem.errors import (
    ConcurrentWriteError,
    CueMemError,
    EmbeddingAPIError,
    IdempotencyConflictError,
    SessionLimitError,
    VectorStoreFullError,
)
from cuemem.read.controller import Router
from cuemem.sdk import (
    AnswerMode,
    Lam,
    Memory,
    MinLevel,
    SearchMode,
    _moment,
    _scope,
    space_for,
)
from cuemem.store.media import ACCEPTED_FORMATS

PREFIX = "/api/v1"
WAIT_TIMEOUT_S = 300.0  # POST /wait 最多等这么久（MemoryClient 默认的超时），之后 504
WAIT_POLL_S = 0.25  # POST /wait 每隔这么久看一次抽取做完没有
_CODES = {
    400: "invalid_request",
    401: "unauthorized",
    402: "no_balance",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "invalid_request",
    429: "rate_limited",
    502: "upstream_error",
    503: "unavailable",
    504: "timeout",
    500: "internal_error",
}


class V1Error(HTTPException):
    """/api/v1 的出错：状态码、错误码与 SDK 的异常类型（MemoryClient 据此抛出同样的异常）。
    code 不给时按状态码取（_CODES）；headers：例如限流时上游的 Retry-After。upstream_status：
    向量接口的出错（EmbeddingAPIError）原来的状态码，写进出错的 upstream_status，MemoryClient
    据此还原异常的 status（回给调用方的状态码可能不同：上游的 500 是 502）。"""

    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: str | None = None,
        code: str | None = None,
        headers: Mapping[str, str] | None = None,
        upstream_status: int | None = None,
    ) -> None:
        super().__init__(status, message, headers=dict(headers) if headers else None)
        self.error_type = error_type
        self.code = code
        self.upstream_status = upstream_status


class AddRequest(BaseModel):
    messages: str | dict[str, Any] | list[dict[str, Any]]
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    metadata: dict[str, Any] | None = None
    expiration_date: str | None = Field(None, description="ISO date or datetime")
    timestamp: str | float | None = Field(None, description="ISO datetime or Unix time")
    infer: bool = True
    wait: bool = False


# 时刻（search、context、answer 的 at，get_all 的 known_at）：原样交给接口，与本地的 Memory 用同一个
# sdk._moment 解析（第七轮复审）。以前由 Pydantic 解析：'20300101' 读成 1970 年的秒数，本地读成
# 2030-01-01；毫秒与只有数字的文本本地报错、这里照收。不收 true、false
Moment = Annotated[
    str | StrictInt | StrictFloat | None,
    Field(description="ISO date or datetime, or Unix time (seconds; above 2e10: milliseconds)"),
]


class SearchRequest(BaseModel):
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    limit: int | None = None
    mode: SearchMode = "auto"
    at: Moment = None
    router: Router = "learned"
    lam: Lam = None
    filters: dict[str, Any] | None = None
    top_k: int | None = None
    min_level: MinLevel = "l1"
    show_expired: bool = False


class ContextRequest(BaseModel):
    query: str
    user_id: str | None = None
    agent_id: str | None = None
    at: Moment = None
    mode: SearchMode = "auto"
    router: Router = "learned"
    lam: Lam = None
    min_level: MinLevel = "l1"
    wrap: bool = Field(True, description="Wrap the block in <memory>...</memory>")


class AnswerRequest(BaseModel):
    question: str
    user_id: str | None = None
    agent_id: str | None = None
    mode: AnswerMode = "auto"
    at: Moment = None
    router: Router = "learned"
    lam: Lam = None


class UpdateRequest(BaseModel):
    # text 与 data 是同一个参数（Mem0 2.x 的新旧两个名字，见 Memory.update）；MemoryClient 发 data
    text: str | None = None
    data: str | None = None
    metadata: dict[str, Any] | None = None


class BatchDeleteRequest(BaseModel):
    memory_ids: list[str]
    dry_run: bool = False


class RunRequest(BaseModel):
    run_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None


class EndRunRequest(BaseModel):
    run_id: str
    user_id: str | None = None
    agent_id: str | None = None
    wait: bool = False


class ConsolidateRequest(BaseModel):
    user_id: str | None = None
    agent_id: str | None = None
    rebuild: bool = False


class WaitRequest(BaseModel):
    user_id: str | None = None
    agent_id: str | None = None
    timeout: float | None = Field(
        None, ge=0, allow_inf_nan=False, description=f"Seconds; at most {WAIT_TIMEOUT_S:g}"
    )


class ImportRequest(BaseModel):
    data: dict[str, Any] | list[Any] = Field(
        description="A CueMem export (format cuemem), or Mem0 get_all output: a list of memories "
        'or {"results": [...]} (format mem0)'
    )
    user_id: str | None = None
    agent_id: str | None = None
    wait: bool = False
    format: Literal["cuemem", "mem0"] = Field(
        "cuemem", description="cuemem: a CueMem export; mem0: Mem0 get_all output"
    )


# ---------- 返回值的形状（只用于 OpenAPI，第三轮体检）----------
#
# 以前 /openapi.json 里每个接口的 200 都是"任意对象"，422 写的是 FastAPI 的 HTTPValidationError
# （实际是下面的 V1ErrorBody），别的出错一个都没写：按它生成的客户端（openapi-typescript、Orval）
# 只拿到 Record<string, any> 与一个不会出现的出错类型。这些模型照 SDK 现在给出的键写（sdk.py 的
# _fact_item、_unit_item、_image_item、_recall_out……），extra="allow"：以后多了键，旧的客户端照常
# 能用。接口的返回值不经它们校验、不被改写（见 V1Json）；tests/test_memory_api.py 核对 SDK 给出的
# 每个键都写在这里。


class _Out(BaseModel):
    model_config = ConfigDict(extra="allow")


class ErrorDetail(_Out):
    code: str = Field(description="Error category: invalid_request, forbidden, rate_limited, ...")
    type: str | None = Field(
        description="The SDK's exception type (KeyError, ValueError, TypeError or one of "
        "cuemem.errors); MemoryClient raises the same. null for 401 and for scope errors."
    )
    message: str
    upstream_status: int | None = Field(
        None, description="The embedding provider's own status code (EmbeddingAPIError)."
    )


class V1ErrorBody(_Out):
    """Every /api/v1 error has this shape, including 401, 404 and 422."""

    error: ErrorDetail


class FactItem(_Out):
    """A fact. Times are ISO 8601 in the server's local time."""

    id: str
    memory: str
    type: Literal["fact"]
    subject: str
    predicate: str
    object: str
    negated: bool = Field(
        description='True for a negative fact ("no longer drinks coffee"): subject, predicate '
        "and object then describe what is NOT true."
    )
    cardinality: Cardinality = Field(
        description='"single": a newer value replaces the old one (where the user lives); '
        '"multi": values accumulate (what the user likes).'
    )
    category: FactCategory
    importance: int
    status: str = Field(description='"current", or since when it is no longer true or corrected.')
    valid_from: str | None
    valid_to: str | None
    created_at: str
    invalidated_at: str | None
    superseded_by: str | None
    session_id: str
    source_turn_ids: list[str]
    metadata: dict[str, Any]
    expiration_date: str | None
    score: float | None = Field(None, description="Search only: for ranking, not a similarity.")
    space_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None


class FactEvent(FactItem):
    """A fact that add or update created (Mem0's ADD / UPDATE events)."""

    event: Literal["ADD", "UPDATE"]
    previous_memory: list[str] | None = None
    reason: SupersedeReason | None = None


class HistoryItem(FactItem):
    event: Literal["ADD", "UPDATE"]


class ExcerptItem(_Out):
    """A conversation excerpt (one round)."""

    id: str
    memory: str
    type: Literal["excerpt"]
    score: float | None = None
    created_at: str
    session_id: str
    source_turn_ids: list[str]
    metadata: dict[str, Any]
    expiration_date: str | None
    space_id: str | None = None


class ImageItem(_Out):
    """An image; GET /api/v1/media/{id} returns the file (path is on the server)."""

    id: str
    memory: str = Field(description="The image's caption.")
    type: Literal["image"]
    path: str
    created_at: str
    attached: bool
    metadata: dict[str, Any]
    expiration_date: str | None
    space_id: str | None = None


SearchHit = Annotated[FactItem | ExcerptItem, Field(discriminator="type")]
MemoryItem = Annotated[FactItem | ExcerptItem | ImageItem, Field(discriminator="type")]


class ConsolidationReport(_Out):
    sessions: int
    facts_extracted: int
    added: int
    superseded: int
    merged: int
    noop: int
    entities_created: int
    entities_linked: int
    llm_calls: int
    rules: dict[str, int]
    failed: list[str]
    skipped: list[str]
    elapsed_ms: float


class AddResult(_Out):
    results: list[FactEvent]
    session_id: str | None
    turn_ids: list[str]
    status: Literal["empty", "logged", "queued", "consolidated", "duplicate", "forgotten"]
    error: str | None = None
    consolidation: ConsolidationReport | None = None
    replayed: bool | None = None  # Idempotency-Key 已经写过：duplicate 或 forgotten


class SearchResult(_Out):
    results: list[SearchHit]
    images: list[ImageItem]
    profile: str
    level: Literal["L0", "L1", "L2"]
    see: bool
    reasons: list[str]


class ContextResult(_Out):
    """POST /context: the evidence block the memory proxy injects, ready for a prompt."""

    context: str = Field(
        description="The <memory> block (empty when there is nothing to add); without the "
        "<memory> tags when wrap is false"
    )
    level: Literal["L0", "L1", "L2"]
    reasons: list[str]
    recalled: int = Field(description="Items in the block; the profile counts as one")
    see: bool
    refs: dict[str, str] = Field(description="Label in the block -> memory id (see GET)")


class GetAllResult(_Out):
    results: list[FactItem]


class UpdateResult(_Out):
    message: str
    id: str
    results: list[FactEvent]
    session_id: str
    turn_ids: list[str]
    consolidation: ConsolidationReport | None = None
    error: str | None = None


class ForgetResult(_Out):
    message: str
    turns_deleted: int
    sessions_affected: list[str]
    facts_before: int
    facts_after: int
    images_deleted: int
    restated: int
    warning: str | None = None


class DryRunTurn(_Out):
    id: str
    session_id: str
    role: str
    text: str


class DryRunFact(_Out):
    id: str
    memory: str


class DryRunResult(_Out):
    """delete with dry_run=true: what would be deleted; nothing is."""

    message: str
    turns: list[DryRunTurn]
    sessions_affected: list[str]
    cascaded: int
    restated: int
    facts: list[DryRunFact]
    images: int


class BatchForgetResult(ForgetResult):
    """delete_many: deleted lists the ids that were deleted; missing lists unknown ids and ids
    outside this key's scope."""

    deleted: list[str]
    missing: list[str]


class BatchDryRunResult(DryRunResult):
    """delete_many with dry_run=true: deleted lists the ids that would be deleted; nothing is."""

    deleted: list[str]
    missing: list[str]


class ResetResult(_Out):
    message: str
    spaces_deleted: list[str]
    turns_deleted: int
    images_deleted: int


class Citation(_Out):
    ref: str
    id: str
    memory: str


class AnswerResult(_Out):
    answer: str
    evidence_sufficient: bool
    citations: list[Citation]
    level: str | None
    reasons: list[str]


class ProfileResult(_Out):
    profile: str


class RunResult(_Out):
    run_id: str


class EndRunResult(_Out):
    session_id: str
    turns: int
    consolidation: ConsolidationReport | Literal["queued"] | None


class WaitResult(_Out):
    ok: bool


class FailedSession(_Out):
    session_id: str
    attempts: int
    error: str = Field(description="The first line of the error.")
    retry_at: str | None = Field(
        description="When the background worker retries (ISO 8601); null when it will not "
        "retry by itself (too many attempts, or no background worker): call consolidate."
    )


class StatusResult(_Out):
    """Extraction status of one memory space."""

    pending: int = Field(description="Sessions waiting for extraction.")
    failed: list[FailedSession]
    background: bool = Field(description="Whether a background worker extracts.")
    errors: list[str] = Field(description="The worker's last errors in this space.")


class ExportImage(_Out):
    sha256: str
    mime: str
    data: str = Field(description="Base64.")


class ExportTurn(_Out):
    idx: int
    role: str
    text: str
    ts: str
    speaker: str | None
    images: list[ExportImage]
    metadata: dict[str, Any] | None = None
    expires_at: str | None = None
    edit_of: str | None = None


class ExportSession(_Out):
    session_id: str
    started_at: str
    turns: list[ExportTurn]


class ExportDerived(_Out):
    facts: list[dict[str, Any]]
    profile: str


class ExportDocument(_Out):
    """A memory space as a portable file (cuemem/portable.py); POST /api/v1/import takes it."""

    format: Literal["cuemem.space"]
    version: int
    exported_at: str
    space_id: str
    sessions: list[ExportSession]
    derived: ExportDerived | None = None


class ImportReport(_Out):
    space_id: str
    sessions: int
    turns_added: int
    images_added: int
    consolidation: ConsolidationReport | None = None
    skipped: int | None = Field(
        None, description="format mem0 only: memories skipped because their text was empty"
    )


class SpaceRow(_Out):
    space_id: str
    created_at: str
    sessions: int
    turns: int
    units: int
    current_facts: int
    facts: int
    entities: int


# 接口照旧返回 SDK 算出的字典，FastAPI 照旧按 dict[str, Any] 序列化（NaN 写成 null、键的顺序
# 不变）；WithJsonSchema({}) 只让 OpenAPI 不写"任意对象"，200 的形状由 responses 里的模型给出。
# 不用 response_model=None：那样 FastAPI 改走 json.dumps(allow_nan=False)，分数是 NaN 时 500
V1Json = Annotated[dict[str, Any], WithJsonSchema({})]
V1JsonList = Annotated[list[dict[str, Any]], WithJsonSchema({})]
_BINARY = {"type": "string", "format": "binary"}  # GET /api/v1/media/{id}：图片文件本身
# 它的 Content-Type（_image_type）：媒体库收的格式（ACCEPTED_FORMATS）在 PIL 的 Image.MIME 里的
# 类型，加上手机拍的 MPO（JPEG 的解码器认出来的格式是 MPO）与认不出时的通用二进制。写成具体的
# 类型：以前写的是 image/*，openapi-python-client 不认、生成的客户端里没有这个响应。直接列出来，
# 不在导入时调用 Image.init()（加载 PIL 的全部插件）；测试核对它与 PIL 的一致
_MEDIA_TYPES = (
    "application/octet-stream",
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/mpo",
    "image/png",
    "image/tiff",
    "image/webp",
)

_ERRORS = {
    400: "Invalid request (the SDK raised ValueError or TypeError).",
    401: "Missing or wrong access token.",
    402: "The model provider's balance is exhausted.",
    403: "The memory space is outside this API key's scope.",
    404: "No such memory, run or space (KeyError), or outside this key's scope.",
    409: "A concurrent write to the same run (ConcurrentWriteError).",
    413: "Request body too large.",
    422: "Malformed request body or parameters (type ValueError).",
    429: "Rate-limited by the model provider; see Retry-After.",
    502: "The model or embedding provider failed.",
    503: "The model provider or the vector store is temporarily unavailable (see Retry-After), "
    "or the local vector store has no room for a new space (code capacity).",
    504: "The model provider or the vector store timed out, or POST /wait reached its timeout "
    "(type TimeoutError).",
    500: "Unexpected server error (a bug); the traceback is in the server log.",
}


def _ok(model: Any) -> dict[int | str, dict[str, Any]]:
    return {200: {"model": model}}


@contextmanager
def _sdk_errors(*, by_id: bool = False) -> Iterator[None]:
    """SDK 的异常 → /api/v1 的出错。KeyError 只在按 ID 找记忆的接口上算"没有这条"（404），
    别处的 KeyError 是程序错误，照常 500。模型与向量供应商的出错与 /api 相同（api/failures.py）：
    限流 429 与暂时不可用 503 带上游的 Retry-After，余额不足 402，超时 504，内容审核拒绝 422，
    别的 502；以前 openai 的异常是没有说明的 500，向量接口限流、内容审核拒绝都是 502。"""
    try:
        yield
    except KeyError as exc:
        if not by_id:
            raise
        raise V1Error(404, str(exc.args[0] if exc.args else exc), error_type="KeyError") from exc
    except (ValueError, TypeError) as exc:
        raise V1Error(400, str(exc), error_type=type(exc).__name__) from exc
    except (ConcurrentWriteError, IdempotencyConflictError) as exc:
        raise V1Error(409, str(exc), error_type=type(exc).__name__) from exc
    except SessionLimitError as exc:  # open_run 或 add 重新打开 run 到了上限
        raise V1Error(
            429, str(exc), error_type="SessionLimitError", headers={"Retry-After": "60"}
        ) from exc
    except VectorStoreFullError as exc:  # 本地向量库没有余量建新的空间（已有的照常可用）
        raise V1Error(
            503, exc.for_client(), error_type="VectorStoreFullError", code="capacity"
        ) from exc
    except Exception as exc:
        failure = provider_failure(exc)
        if failure is not None:
            raise V1Error(
                failure.status,
                failure.for_model(),
                error_type=type(exc).__name__,
                code=failure.kind,
                headers={"Retry-After": failure.retry_after} if failure.retry_after else None,
                upstream_status=exc.status if isinstance(exc, EmbeddingAPIError) else None,
            ) from exc
        if isinstance(exc, CueMemError):
            raise V1Error(502, str(exc), error_type=type(exc).__name__) from exc
        raise


def _no_server_files(messages: str | dict[str, Any] | list[dict[str, Any]]) -> None:
    """图片只收 data URL（网络图片交给 SDK 报它自己的错）：本地路径与 file:// 指向的是服务器上的
    文件，远程调用方不能借此把服务器上的文件写进记忆。内容块写错时报 ValueError，不让 SDK 在
    取 image_url 时抛 KeyError。"""
    items = [messages] if isinstance(messages, dict) else messages
    for message in items if isinstance(items, list) else []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                raise ValueError("Each content block must be an object.")
            if block.get("type") != "image_url":
                continue
            value = block.get("image_url")
            url = value.get("url") if isinstance(value, dict) else value
            if not isinstance(url, str) or not url.startswith(("data:", "http://", "https://")):
                raise ValueError(
                    "Send images as data URLs: the server does not read files from its own disk "
                    "(MemoryClient converts local paths for you)."
                )


def _image_type(data: bytes) -> str:
    """图片的 Content-Type：按 PIL 认出的格式（只读文件头，只试媒体库收的格式），认不出时是通用的
    二进制。"""
    try:
        with Image.open(io.BytesIO(data), formats=ACCEPTED_FORMATS) as probe:
            return Image.MIME.get(probe.format or "", "application/octet-stream")
    except Exception:  # 坏图时 PIL 抛的异常五花八门
        return "application/octet-stream"


def _filters(raw: str | None) -> dict[str, Any] | None:
    """GET 的 filters 是 JSON 字符串（嵌套的条件放不进普通的查询参数）。"""
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, dict):
        raise V1Error(400, "filters must be a JSON object.", error_type="ValueError")
    return value


def _allow(who: Principal, space: str | None) -> None:
    """请求点名的空间在令牌的作用域之外：403。"""
    if not who.allows(space):
        raise V1Error(403, f"This API key cannot access the memory space {space!r}.")


def memories_router(
    memory: Memory,
    *,
    dependencies: list[Any] | None = None,
    route_class: type[APIRoute] = APIRoute,
) -> APIRouter:
    """/api/v1 下的记忆接口，包住一个 Memory（服务里是 Memory.from_system，与界面、记忆代理共用
    同一个记忆系统）。同步的接口：FastAPI 在线程池里执行，Memory 本身线程安全。

    dependencies 要有 api.auth.guard（认出令牌）：每个接口按它的作用域检查，没有时报错（500），
    不当成管理令牌放行。route_class：api.auth.authenticated_route，令牌不对时不读请求体。

    OpenAPI：每个接口的 200 写明形状（上面的模型），出错的状态码都写成 V1ErrorBody（422 也是，
    不再是 FastAPI 的 HTTPValidationError）。docs/openapi-v1.json 是它的快照（openapi_v1）。"""
    router = APIRouter(
        prefix=PREFIX,
        dependencies=dependencies or [],
        tags=["memories (v1)"],
        route_class=route_class,
        responses={
            status: {"model": V1ErrorBody, "description": text} for status, text in _ERRORS.items()
        },
        # operationId 是 SDK 的方法名（add、get_all……）：按规范生成的客户端里方法名与 MemoryClient
        # 相同。以前是 FastAPI 的缺省（add_api_v1_memories_post）。别的路由照旧（带路径，不会重名）
        generate_unique_id_function=lambda route: route.name,
    )

    def allow(
        http: Request,
        user_id: str | None,
        agent_id: str | None,
        filters: dict[str, Any] | None = None,
        *,
        creates: bool = False,
    ) -> None:
        """user_id、agent_id 定下的空间（filters 里的 user_id、agent_id 优先）要在作用域里。与 SDK
        用同一个函数解析 filters：检查的空间就是 SDK 要读的空间。都没给时报 SDK 的 ValueError。
        creates：这个接口会建新的空间（add、runs、consolidate），另查令牌的 max_spaces。"""
        user_id, agent_id, _, _ = _scope(user_id, agent_id, None, filters)
        space = space_for(user_id, agent_id)
        _allow(principal_of(http), space)
        if creates:
            room(http, space)

    def room(http: Request, space: str | None) -> None:
        """令牌写了 max_spaces、空间已经这么多个、这次写入又要建新的空间时 429（rate_limited）。"""
        who = principal_of(http)
        held = spaces_over_cap(who, space, lambda: memory.system.db.space_ids())
        if held is not None:
            raise V1Error(429, space_cap_message(who, held))

    def visible(http: Request, memory_id: str) -> bool:
        """按 ID 找的记忆：不存在的，或者在作用域里的。作用域之外的与不存在的一样处理。事实、
        对话片段与图片三种 ID 都查（与 Memory.get、delete 认的相同）：只查事实的话，限定作用域
        的令牌能按算得出来的片段 ID 删掉别人的对话。"""
        located = memory.system.db.memory_kind(memory_id)
        return located is None or principal_of(http).allows(located[0])

    def missing(memory_id: str) -> V1Error:
        return V1Error(404, f"No memory with id {memory_id}", error_type="KeyError")

    @router.post("/memories", responses=_ok(AddResult))
    def add(
        http: Request,
        request: AddRequest,
        idempotency_key: Annotated[
            str | None,
            Header(
                alias="Idempotency-Key",
                description=(
                    "Optional key (1-255 printable ASCII characters, e.g. a UUID) for retrying "
                    "an add safely: within 24 hours the same key and request for the same user "
                    "is written once and answered with status 'duplicate' (or 'forgotten'); the "
                    "same key with a different request is a 409 IdempotencyConflictError."
                ),
            ),
        ] = None,
    ) -> V1Json:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id, creates=True)
            _no_server_files(request.messages)
            return memory.add(
                request.messages,
                user_id=request.user_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                metadata=request.metadata,
                expiration_date=request.expiration_date,
                timestamp=request.timestamp,
                infer=request.infer,
                wait=request.wait,
                idempotency_key=idempotency_key,
            )

    @router.post("/memories/search", responses=_ok(SearchResult))
    def search(http: Request, request: SearchRequest) -> V1Json:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id, request.filters)
            return memory.search(
                request.query,
                user_id=request.user_id,
                agent_id=request.agent_id,
                run_id=request.run_id,
                limit=request.limit,
                mode=request.mode,
                at=_moment("at", request.at),
                router=request.router,
                lam=request.lam,
                filters=request.filters,
                top_k=request.top_k,
                min_level=request.min_level,
                show_expired=request.show_expired,
            )

    @router.post("/context", responses=_ok(ContextResult))
    def context(http: Request, request: ContextRequest) -> V1Json:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id)
            return memory.context(
                request.query,
                user_id=request.user_id,
                agent_id=request.agent_id,
                at=_moment("at", request.at),
                mode=request.mode,
                router=request.router,
                lam=request.lam,
                min_level=request.min_level,
                wrap=request.wrap,
            )

    @router.get("/memories", responses=_ok(GetAllResult))
    def get_all(
        http: Request,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        include_superseded: bool = False,
        valid_at: str | None = None,
        known_at: str | None = None,  # 时刻，见 Moment
        limit: int | None = None,
        filters: str | None = None,
        top_k: int | None = None,
        show_expired: bool = False,
    ) -> V1Json:
        where = _filters(filters)
        with _sdk_errors():
            allow(http, user_id, agent_id, where)
            return memory.get_all(
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                include_superseded=include_superseded,
                valid_at=valid_at,
                known_at=_moment("known_at", known_at),
                limit=limit,
                filters=where,
                top_k=top_k,
                show_expired=show_expired,
            )

    @router.delete(
        "/memories",
        responses=_ok(ForgetResult | DryRunResult),
        description="Delete all memories of a user (or agent), or only one run with run_id. "
        "dry_run=true needs run_id and only lists what would be deleted. A run_id with no "
        "conversation turns deletes nothing and answers 'No such run: nothing deleted'.",
    )
    def delete_all(
        http: Request,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
    ) -> V1Json:
        with _sdk_errors():
            allow(http, user_id, agent_id)
            return memory.delete_all(
                user_id=user_id, agent_id=agent_id, run_id=run_id, dry_run=dry_run
            )

    @router.get("/memories/{memory_id}", responses=_ok(MemoryItem))
    def get(http: Request, memory_id: str) -> V1Json:
        if not visible(http, memory_id):
            raise missing(memory_id)
        with _sdk_errors(by_id=True):
            item = memory.get(memory_id)
        if item is None:
            raise missing(memory_id)
        return item

    @router.put("/memories/{memory_id}", responses=_ok(UpdateResult))
    def update(http: Request, memory_id: str, request: UpdateRequest) -> V1Json:
        if not visible(http, memory_id):
            raise missing(memory_id)
        with _sdk_errors(by_id=True):
            return memory.update(
                memory_id, request.text, metadata=request.metadata, data=request.data
            )

    @router.get("/memories/{memory_id}/history", responses=_ok(list[HistoryItem]))
    def history(http: Request, memory_id: str) -> V1JsonList:
        if not visible(http, memory_id):
            return []  # 与不存在的 ID 相同
        with _sdk_errors(by_id=True):
            return memory.history(memory_id)

    @router.delete("/memories/{memory_id}", responses=_ok(ForgetResult | DryRunResult))
    def delete(http: Request, memory_id: str, dry_run: bool = False) -> V1Json:
        if not visible(http, memory_id):
            raise missing(memory_id)
        with _sdk_errors(by_id=True):
            return memory.delete(memory_id, dry_run=dry_run)

    @router.post(
        "/memories/batch-delete",
        responses=_ok(BatchForgetResult | BatchDryRunResult),
        description="Delete several memories (fact, excerpt or image ids) with one rebuild per "
        "memory space; at most 100 ids. Unknown ids and ids outside this key's scope are listed "
        "in missing, not an error. dry_run=true only lists what would be deleted.",
    )
    def delete_many(http: Request, request: BatchDeleteRequest) -> V1Json:
        # 一次写入（rate_per_minute 算一个）。管理令牌照 SDK；限定作用域的令牌只删作用域里的，
        # 别的 ID 与不存在的一样进 missing（与 reset 同样的分法）
        who = principal_of(http)
        with _sdk_errors():
            if who.admin:
                return memory.delete_many(request.memory_ids, dry_run=request.dry_run)
            return memory._delete_many(request.memory_ids, dry_run=request.dry_run, keep=who.allows)

    @router.get(
        "/media/{media_id}",
        response_class=Response,
        responses={200: {"content": {mime: {"schema": _BINARY} for mime in _MEDIA_TYPES}}},
        description="The original file of an image memory (the path in image items is on the "
        "server). Out-of-scope, non-image and unknown ids are 404.",
    )
    def media(http: Request, media_id: str) -> Response:
        # 一张图片的原文件（Memory.media）：图片项里的 path 是服务器上的路径，远程调用方用这个接口
        # 取。作用域之外的、不是图片的 ID 与不存在的一样（404）。说明写在 description 里（英文，
        # 进 OpenAPI），不写 docstring：FastAPI 把 docstring 当成接口的说明
        if not visible(http, media_id):
            raise missing(media_id)
        with _sdk_errors(by_id=True):
            data = memory.media(media_id)
        if data is None:
            raise missing(media_id)
        return Response(data, media_type=_image_type(data))

    @router.post(
        "/reset",
        responses=_ok(ResetResult),
        description="The admin token deletes every memory; a scoped key deletes the memory "
        "spaces in its scope.",
    )
    def reset(http: Request) -> V1Json:
        # 管理令牌删全部；限定作用域的令牌只删它作用域里的空间（每个都能用 delete_all 删掉），
        # 还开着的 run 也只结束这些空间里的。以前限定作用域的令牌一律 403
        who = principal_of(http)
        with _sdk_errors():
            if who.admin:
                return memory.reset()
            out = memory._reset(who.allows)
        return {**out, "message": "All memories in this key's scope deleted"}

    @router.post("/answer", responses=_ok(AnswerResult))
    def answer(http: Request, request: AnswerRequest) -> V1Json:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id)
            return memory.answer(
                request.question,
                user_id=request.user_id,
                agent_id=request.agent_id,
                mode=request.mode,
                at=_moment("at", request.at),
                router=request.router,
                lam=request.lam,
            )

    @router.get("/profile", responses=_ok(ProfileResult))
    def profile(http: Request, user_id: str | None = None, agent_id: str | None = None) -> V1Json:
        with _sdk_errors():
            allow(http, user_id, agent_id)
            return {"profile": memory.profile(user_id=user_id, agent_id=agent_id)}

    @router.post("/runs", responses=_ok(RunResult))
    def open_run(http: Request, request: RunRequest) -> V1Json:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id, creates=True)
            run_id = memory.open_run(
                request.run_id, user_id=request.user_id, agent_id=request.agent_id
            )
        return {"run_id": run_id}

    @router.post("/runs/end", responses=_ok(EndRunResult))
    def end_run(http: Request, request: EndRunRequest) -> V1Json:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id)
            return memory.end_run(
                request.run_id,
                user_id=request.user_id,
                agent_id=request.agent_id,
                wait=request.wait,
            )

    @router.post("/consolidate", responses=_ok(ConsolidationReport))
    def consolidate(http: Request, request: ConsolidateRequest) -> V1Json:
        with _sdk_errors():
            allow(http, request.user_id, request.agent_id, creates=True)
            return memory.consolidate(
                user_id=request.user_id, agent_id=request.agent_id, rebuild=request.rebuild
            )

    @router.post("/wait", responses=_ok(WaitResult))
    async def wait(http: Request, request: WaitRequest | None = None) -> V1Json:
        """给了 user_id、agent_id 时只等这个空间（要在作用域里）；不给时管理令牌（没配令牌时的
        本机演示）等全部，限定作用域的令牌只等它能访问的空间。以前不看作用域、等全部用户的抽取：
        限定作用域的令牌也能占着一个线程，还能从等了多久看出别的用户排着多少抽取。最多等
        WAIT_TIMEOUT_S 秒，没做完时 504（timeout，type TimeoutError）。"""
        # 异步的接口（第六轮体检）：在事件循环里每 WAIT_POLL_S 秒看一次，不占线程池的线程。以前
        # 是同步的，每个等着的请求占着 AnyIO 40 个线程中的一个，45 个并发的 wait 就让别的同步接口
        # 都等到它们超时
        request = request or WaitRequest()
        timeout = min(
            request.timeout if request.timeout is not None else WAIT_TIMEOUT_S, WAIT_TIMEOUT_S
        )
        who = principal_of(http)
        user_id, agent_id = request.user_id, request.agent_id
        spaces: Callable[[str], bool] | None
        if user_id is not None or agent_id is not None:
            with _sdk_errors():
                allow(http, user_id, agent_id)
            spaces = space_for(user_id, agent_id).__eq__
        else:
            spaces = None if who.admin else who.allows
        worker = memory.system.worker
        if worker is None:
            return {"ok": True}
        deadline = time.monotonic() + timeout
        while not worker.wait(spaces, 0):  # 只看一眼，不阻塞事件循环
            if time.monotonic() >= deadline:
                raise V1Error(
                    504,
                    f"Extraction did not finish within {timeout:g}s; "
                    "it continues in the background.",
                    error_type="TimeoutError",
                )
            # 客户端断开了（第六轮复审）：不再等，放开这个令牌同时在处理的名额（BodyLimit 在接口
            # 返回时才放开）。以前一直等到 timeout，丢下的 wait 最多占着名额 300 秒，8 个就让这个
            # 令牌的每个请求都 429。回的内容没人收
            if await http.is_disconnected():
                return cast(V1Json, Response(status_code=499))
            await anyio.sleep(min(WAIT_POLL_S, max(0.0, deadline - time.monotonic())))
        return {"ok": True}

    @router.get("/status", responses=_ok(StatusResult))
    def status(http: Request, user_id: str | None = None, agent_id: str | None = None) -> V1Json:
        with _sdk_errors():
            allow(http, user_id, agent_id)
            return memory.status(user_id=user_id, agent_id=agent_id)

    @router.get("/export", responses=_ok(ExportDocument))
    def export(
        http: Request, user_id: str | None = None, agent_id: str | None = None, derived: bool = True
    ) -> V1Json:
        # 边读边写（第八轮体检 P7）：与 Memory.export 的 JSON 逐字节相同，同一时间只有一句发言的
        # 原图在内存里。以前整个导出（每张原图的 base64）先放进一个 dict 再渲染，峰值约是响应体的
        # 三倍。没有这个空间时在开始写之前 404；写出途中出错只能截断（MemoryClient 解析时出错）。
        # 文档里的形状照旧（ExportDocument，docs/openapi-v1.json 不变）
        from cuemem.portable import iter_export_json

        with _sdk_errors(by_id=True):  # 没有这个空间：KeyError（404），与本地相同
            allow(http, user_id, agent_id)
            plan = memory._export_plan(user_id=user_id, agent_id=agent_id, derived=derived)
        stream = StreamingResponse(iter_export_json(plan), media_type="application/json")
        return cast(V1Json, stream)

    @router.post("/import", responses=_ok(ImportReport))
    def load(http: Request, request: ImportRequest) -> V1Json:
        # 写进哪个空间与 Memory.load 的取法相同：给了 user_id、agent_id 时是它们的空间，否则是
        # 文件里的 space_id。Mem0 的导出先换算（from_mem0），按换算出的空间查作用域：混着几个
        # 用户、又没给 user_id 时 400，作用域之外的空间 403
        data = request.data
        if request.format == "mem0":
            from cuemem.portable import from_mem0

            with _sdk_errors():
                converted = from_mem0(data, user_id=request.user_id, agent_id=request.agent_id)
            target: str | None = str(converted["space_id"])
        elif request.user_id or request.agent_id:
            target = space_for(request.user_id, request.agent_id)
        else:
            target = (
                str(data["space_id"]) if isinstance(data, dict) and "space_id" in data else None
            )
        _allow(principal_of(http), target)
        room(http, target)
        with _sdk_errors():
            return memory.load(
                data,
                user_id=request.user_id,
                agent_id=request.agent_id,
                wait=request.wait,
                format=request.format,
            )

    @router.get("/spaces", responses=_ok(list[SpaceRow]))
    def spaces(http: Request) -> V1JsonList:
        who = principal_of(http)
        return [row for row in memory.spaces() if who.allows(row["space_id"])]

    return router


def install_error_format(app: FastAPI) -> None:
    """/api/v1 下的出错（包括 401、422、找不到接口、没有处理的异常的 500）都写成
    {"error": {"code", "type", "message"}}；别的路径照 FastAPI 的默认（{"detail": …}，500 是
    text/plain）。"""

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, StarletteHTTPException)
        if not _v1(request):
            return await http_exception_handler(request, exc)
        return _error(
            exc.status_code,
            str(exc.detail),
            getattr(exc, "error_type", None),
            exc.headers,
            getattr(exc, "code", None),
            getattr(exc, "upstream_status", None),
        )

    async def validation_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, RequestValidationError)
        if not _v1(request):
            return await request_validation_exception_handler(request, exc)
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ()))}: {e.get('msg', '')}"
            for e in exc.errors()
        )
        return _error(422, f"Invalid request: {problems}", "ValueError")

    async def server_error(request: Request, exc: Exception) -> Response:
        # 没有处理的异常（程序错误）：/api/v1 下也是 {"error": …}，以前是 text/plain 的
        # "Internal Server Error"。type 是 None：写异常的类名的话，MemoryClient 会把服务端的
        # KeyError、ValueError 还原成同名的异常，看着像 404、400。消息里不带异常的内容（可能有
        # 别人的记忆）。不在这里记日志：Starlette 的 ServerErrorMiddleware 送出这个响应之后照样
        # 把异常抛出去，uvicorn 记下完整的 traceback
        if not _v1(request):
            return PlainTextResponse("Internal Server Error", status_code=500)  # 别的路径不变
        return _error(500, "Internal server error", None)

    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(Exception, server_error)


def _v1(request: Request) -> bool:
    return request.url.path.startswith(PREFIX + "/") or request.url.path == PREFIX


def _error(
    status: int,
    message: str,
    error_type: str | None,
    headers: Mapping[str, str] | None = None,
    code: str | None = None,
    upstream_status: int | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "code": code or _CODES.get(status, "error"),
        "type": error_type,
        "message": message,
    }
    if upstream_status is not None:
        body["upstream_status"] = upstream_status
    return JSONResponse({"error": body}, status, dict(headers or {}))


def openapi_v1() -> dict[str, Any]:
    """/api/v1 的 OpenAPI：只有 /api/v1 的接口与它们引用的组件，与 cuemem serve 的 /openapi.json
    里的这一部分相同（测试核对）。docs/openapi-v1.json 是它的快照（scripts/openapi_v1.py 写出，
    测试核对没有过时），别的语言的客户端按它生成，例如
    npx openapi-typescript docs/openapi-v1.json -o cuemem.d.ts。不打开记忆系统。"""
    keys = Keys()
    app = FastAPI(
        title="CueMem memory API",
        version=__version__,
        description="The /api/v1 memory API of `cuemem serve`, one endpoint per method of the "
        "Python SDK (cuemem.Memory); cuemem.MemoryClient is its Python client.",
    )
    router = memories_router(
        cast(Memory, None),  # 生成 OpenAPI 时不调用接口
        dependencies=guard(keys),
        route_class=authenticated_route(keys),
    )
    app.include_router(router)
    install_error_format(app)
    return v1_part(app.openapi())


def v1_part(spec: Mapping[str, Any]) -> dict[str, Any]:
    """一份 OpenAPI 里 /api/v1 的部分：这些接口，与它们（直接或间接）引用的组件。"""
    paths = {p: ops for p, ops in spec["paths"].items() if p.startswith(PREFIX + "/")}
    components = spec.get("components", {})
    schemas = components.get("schemas", {})
    wanted: set[str] = set()
    pending: list[Any] = [paths]
    while pending:
        for name in _refs(pending.pop()):
            if name not in wanted:
                wanted.add(name)
                pending.append(schemas[name])
    out: dict[str, Any] = {"schemas": {name: schemas[name] for name in sorted(wanted)}}
    if "securitySchemes" in components:
        out["securitySchemes"] = components["securitySchemes"]
    return {"openapi": spec["openapi"], "info": spec["info"], "paths": paths, "components": out}


def _refs(node: Any) -> Iterator[str]:
    """node 里引用的组件名（"$ref": "#/components/schemas/<名字>"）。"""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            yield ref.rsplit("/", 1)[-1]
        for value in node.values():
            yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)
