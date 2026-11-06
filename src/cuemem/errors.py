"""CueMem 的异常类型。"""

import sqlite3


class CueMemError(RuntimeError):
    """所有 CueMem 异常的基类。"""


class MissingAPIKeyError(CueMemError):
    """缺少调用某个供应商所需的 API Key。"""


class CacheMissError(CueMemError):
    """只读缓存模式下，请求没有命中缓存。"""


class StructuredOutputError(CueMemError):
    """多次重试后，模型仍未给出符合 Schema 的 JSON。"""


class EmbeddingAPIError(CueMemError):
    """向量化接口返回错误或格式异常。

    status 是最后一次响应的 HTTP 状态码（重试用尽时；格式异常、网络错误时为 None），
    retry_after 是它的 Retry-After 头：评测网关据此把限流原样转告外部系统（N12）。"""

    def __init__(
        self, message: str, *, status: int | None = None, retry_after: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class DatasetChecksumError(CueMemError):
    """下载的评测数据集与固定的版本不同（sha256 对不上），没有保存。"""


class ConfigMismatchError(CueMemError):
    """数据目录与当前配置不兼容，例如更换了向量模型。"""


class ConsolidationError(CueMemError):
    """巩固（抽取事实等）失败。"""


class ContentFilteredError(CueMemError):
    """供应商的内容审核拒绝了这次请求（常见误伤：普通的生活对话也可能被判为不当内容）。"""


class ConcurrentWriteError(CueMemError):
    """要写入的发言没有全部写进去：序号与会话里已有的发言撞上了（同一会话的并发写入没有串行）。
    这次什么都没写（发言、idempotency_key 与暂存的向量一起撤销），可以重试。"""


class IdempotencyConflictError(CueMemError):
    """add 的 idempotency_key（HTTP 的 Idempotency-Key 头）在这个用户下已经用于另一次不同的写入
    （消息或参数不同）：没有写入。重试要原样重发同一个请求，新的写入要换一个键。"""


class StorageLockedError(CueMemError):
    """本地模式的向量库已被另一个进程打开：Qdrant 本地模式同一时间只允许一个进程访问。"""


class DatabaseOpenError(CueMemError, sqlite3.OperationalError):
    """数据库、账本或缓存打不开（目录只读、磁盘满了、路径被占）。

    同时是 sqlite3.OperationalError：以前打开时原样抛出它，按它捕获的调用方照样捕获得到。"""


class VectorStoreOpenError(CueMemError):
    """本地模式的向量库打不开或建不了集合：多半是打开的文件太多（每个集合一直开着一个文件）。"""


class AccessKeysError(CueMemError):
    """限定作用域的访问令牌（CUEMEM_API_KEYS_FILE 或 CUEMEM_API_KEYS）读不出来或写错了。"""


class MemoryAPIError(CueMemError):
    """远程记忆接口（cuemem.client.MemoryClient）的出错里，没有对应的本地异常的那些：令牌不对、
    连不上服务、找不到接口、模型供应商限流等。status 是 HTTP 状态码（没有拿到响应时为 None），
    code 是 /api/v1 的错误码，type 是服务端的异常类型（例如 openai 的 RateLimitError；没有时为
    None），retry_after 是响应的 Retry-After 头（限流 429、暂时不可用 503 时据此退避）。

    没有拿到响应时 code 是客户端自己的错误码，sent 说请求有没有送到服务端：unreachable
    （sent=False：连不上，写入一定没做，可以重发）；timeout 与 disconnected（sent=True：超时、
    连接中途断开，写入可能已经提交：add 用同一个 idempotency_key 原样重发，服务端只写一次；没给键时
    不要盲目重发，用 export 核对）；transport（sent=True：别的传输出错）。
    拿到了 HTTP 响应时 sent 是 None。"""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        retry_after: str | None = None,
        type: str | None = None,
        sent: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.type = type
        self.sent = sent
