"""CueMem 的异常类型。"""


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


class ConfigMismatchError(CueMemError):
    """数据目录与当前配置不兼容，例如更换了向量模型。"""


class ConsolidationError(CueMemError):
    """巩固（抽取事实等）失败。"""


class ContentFilteredError(CueMemError):
    """供应商的内容审核拒绝了这次请求（常见误伤：普通的生活对话也可能被判为不当内容）。"""


class StorageLockedError(CueMemError):
    """本地模式的向量库已被另一个进程打开：Qdrant 本地模式同一时间只允许一个进程访问。"""
