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
    """向量化接口返回错误或格式异常。"""


class ConfigMismatchError(CueMemError):
    """数据目录与当前配置不兼容，例如更换了向量模型。"""
