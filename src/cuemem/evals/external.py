"""外部基线的公共部分（N12）：Graphiti、LangMem、Cognee、A-MEM 等都从这里派生。

与 N11 的 Mem0 基线（evals/mem0_baseline.py）同样的原则：**按它自己的流程写入与检索，回答与判分
和其他方法完全相同**，只换"记忆从哪来"。

- **模型**：外部系统用自己带的 OpenAI 兼容客户端，base_url 指向评测网关（evals/gateway.py）：
  聊天一律交给 extractor 角色的模型，向量一律交给 CueMem 的同一个向量器；同一本账、同一个缓存。
- **写入**：每份对话历史（LoCoMo 一段对话；LongMemEval 一道题的历史）第一次用到时写入一次，
  写完留下 build.json（写入用量从网关的桶里读），之后的运行直接复用。写入在独立线程里做，不算进
  当前这道题的回答用量；半成品（有目录、没有 build.json）下次整个清掉重建；失败过的历史不再
  反复重建（每次都要花钱）。会话 ID 改写过的历史（LongMemEval / MemLens 的默认，见
  evals/datasets.py）写在 `<数据集>-anon_<历史>` 下，build.json 记下写法（session_ids），与原始
  ID 的写入不混用。
- **写入的身份**分两部分，都记在 build.json 里：
  - 协议（protocol，"抽取模型|向量模型:维度"，见 protocol_of）：外部系统内部调用的模型。评测空间
    的目录（paths.eval_spaces / <系统>）不分协议，所以协议不同的写入**不删也不复用**，报
    ConfigMismatchError：pds 协议（只用 DeepSeek）的写入在 data/eval-spaces，pq 协议（只用通义）
    的在 data/eval-spaces-qwen，换协议要用 CUEMEM_PATHS__EVAL_SPACES 换目录。没记协议的旧记录
    按当前协议处理；记协议之前的 Graphiti、LangMem 写入记了 extractor 与 embedder，照样核对。
  - 写入标记（build_tag，子类设，如外部系统的版本与写入格式）：变了就清掉旧写入重建（同一协议，
    换的只是写法）。有这个字段之前，A-MEM 与 Cognee 把它接在 protocol 后面，照样认得出来。
- **检索**：子类返回一组证据（Hit），交给同一个回答模型、同一套提示
  （与 B2 朴素检索、M0 的格式相同）。检索时经网关的请求（`<系统>.search.<空间>` 桶，如 LangMem
  与 Cognee 向量化问题）在网关的服务线程里执行，不在这道题的 trace 里：网关把这期间这个桶的
  用量事件转记进这道题的 trace（Gateway.forward），于是算进 results.jsonl 的 usage、report.md
  的回答成本与 --max-cost。同一份对话历史的检索在空间锁里串行，所以同一个空间的题并行回答时
  也是逐题准确的；每题检索时经网关的用量另记在 details 的 search_usage 里，桶里仍有总数。

外部系统都不是 CueMem 的依赖，评测时用 uv 临时装上，例如：

    uv run --with graphiti-core cuemem eval run --dataset locomo --split test200r --methods MG
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from cuemem.config import Settings
from cuemem.core.models import Hit, MemoryUnit, UnitKind
from cuemem.core.trace import current_trace
from cuemem.embed import build_embedder
from cuemem.embed.base import TextEmbedder
from cuemem.errors import ConfigMismatchError
from cuemem.evals.datasets import (
    EvalSession,
    Sample,
    SessionIds,
    build_matches_session_ids,
    storage_space_id,
)
from cuemem.evals.gateway import Gateway
from cuemem.evals.methods import MethodContext, MethodOutput
from cuemem.llm.client import LLMClient

# 每个外部系统写入的花费上限（美元，实际花费）。写入不在回答的 trace 里，评测的 --max-cost 管不到它
DEFAULT_BUILD_BUDGET = 10.0
# 每题检索时经网关的用量（details 的 search_usage）记哪几项
SEARCH_FIELDS = ("chat_calls", "embedding_calls", "embedding_tokens", "cost", "spent")


class ExternalBaseline(ABC):
    """子类填 name、label、system（数据目录与网关桶的前缀），实现 build 与 retrieve。

    写入方式变了（外部系统的版本、写入格式……）要作废旧写入时，子类设 build_tag（见模块说明）。"""

    name: str = ""
    label: str = ""
    system: str = ""
    needs_index, needs_consolidation = False, False
    build_budget: float = DEFAULT_BUILD_BUDGET
    build_tag: str = ""  # 写入标记：与 build.json 里记的不同就重建（协议不同则报错，不重建）

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._space_locks: dict[str, threading.Lock] = {}
        self._ready = False
        self._build_spent = 0.0
        self._failed: dict[str, str] = {}
        self._schemes: dict[str, SessionIds] = {}  # 写入目录 → 会话 ID 的写法
        self._model_ids: dict[str, str] = {}  # 模型别名 → model_id（认旧的写入记录用）

    # ---------- 子类实现 ----------

    @abstractmethod
    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        """把一份对话历史写进外部系统（数据放在 directory 下），返回要记进 build.json 的统计。

        LLM 与向量请求发到 self.gateway.url(self.bucket(key))，写入用量由基类从桶里读出。"""

    @abstractmethod
    def retrieve(
        self, key: str, directory: Path, sample: Sample, sessions: Sequence[EvalSession]
    ) -> tuple[list[Hit], dict[str, Any]]:
        """为一道题检索，返回（交给回答模型的证据，记进结果 details 的检索信息）。

        检索时的 LLM 与向量请求（如果有）用 self.gateway.url(self.bucket(key, "search"))。"""

    def setup(self) -> None:  # noqa: B027 - 可选的钩子
        """第一次回答前调用一次（导入外部系统、关掉遥测等）。网关已经启动。"""

    def close(self, key: str) -> None:  # noqa: B027 - 可选的钩子
        """释放一份对话历史占用的资源（打开的数据库等）。"""

    # ---------- 公共流程 ----------

    def answer(self, ctx: MethodContext) -> MethodOutput:
        self._setup(ctx.reader.llm)
        # 会话 ID 改写过的历史另存一处（longmemeval_s-anon_<题号>），不与原始 ID 的写入混用
        key = space_key(storage_space_id(ctx.space_id, ctx.session_ids))
        directory = self.root / key
        with self._lock:
            self._schemes[key] = ctx.session_ids
        with self._space_lock(key):
            built = self._ensure_built(key, directory, ctx.sessions)
            hits, details, searched = self._retrieve(key, directory, ctx)
        a = ctx.reader.answer(ctx.sample.question, hits, question_time=ctx.sample.question_time)
        details = {"memories": len(hits), **details, "search_usage": searched}
        if built is not None:  # 这道题触发了写入：写入的统计记在它名下
            details[f"{self.system}_build"] = built
        return MethodOutput(a.answer, a.evidence_sufficient, hits=hits, details=details)

    def _retrieve(
        self, key: str, directory: Path, ctx: MethodContext
    ) -> tuple[list[Hit], dict[str, Any], dict[str, Any]]:
        """检索一道题。检索时经网关的请求（search 桶）的用量事件转记进这道题的 trace（见模块
        说明）；调用方持有这个空间的锁，同一个桶这段时间里只有这道题在用，前后两次读桶的差就是
        它的用量（另外返回，记进 details）。"""
        bucket = self.bucket(key, "search")
        before = self.gateway.usage(bucket)
        with self.gateway.forward(bucket, current_trace()):
            hits, details = self.retrieve(key, directory, ctx.sample, ctx.sessions)
        searched = _diff(self.gateway.usage(bucket), before)
        return hits, details, {field: searched.get(field, 0) for field in SEARCH_FIELDS}

    def bucket(self, key: str, stage: str = "build") -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{self.system}.{stage}.{key}")

    def _setup(self, llm: LLMClient) -> None:
        with self._lock:
            if self._ready:
                return
            settings = llm.settings
            self.root = settings.paths.eval_spaces / self.system
            self.root.mkdir(parents=True, exist_ok=True)
            self.llm = llm
            self.embedder: TextEmbedder = build_embedder(settings, ledger=llm.ledger)
            self.gateway = Gateway(llm, self.embedder)
            self.gateway.start()
            # 这次运行的协议：外部系统内部调用的模型与向量模型；写入记录里存一份，对不上就报错
            self.protocol = protocol_of(settings, self.embedder)
            self._model_ids = model_ids(settings)
            self.setup()
            self._ready = True

    def recorded_tag(self, record: Mapping[str, Any]) -> str:
        """写入记录里的写入标记：build_tag 字段；有这个字段之前，A-MEM 与 Cognee 把它接在
        protocol 后面（split_protocol）。子类可以覆盖，认出自己更早的写法。"""
        if "build_tag" in record:
            return str(record["build_tag"] or "")
        return split_protocol(str(record.get("protocol") or ""))[1]

    def _ensure_built(
        self, key: str, directory: Path, sessions: Sequence[EvalSession]
    ) -> dict[str, Any] | None:
        record = directory / "build.json"
        with self._lock:
            scheme = self._schemes.get(key, "native")
        replaced: dict[str, str] = {}
        if record.exists():
            stored = json.loads(record.read_text("utf-8"))
            # 另一个协议（抽取模型或向量模型）的写入：报错，不删也不复用（目录不分协议）
            check_protocol(stored, self.protocol, directory, self.system, self._model_ids)
            # 写入标记不同（写法变了），或会话 ID 的写法不同（写入里存着会话 ID，见
            # datasets.build_matches_session_ids）：同一协议的旧写入作废，重建
            tag = self.recorded_tag(stored)
            if tag == self.build_tag and build_matches_session_ids(stored, scheme):
                return None
            if tag != self.build_tag:
                replaced["build_tag"] = tag
        with self._lock:
            if key in self._failed:
                raise RuntimeError(
                    f"{self.system} 写入这份对话历史时失败过，不再重试：{self._failed[key]}"
                )
            if self._build_spent > self.build_budget:
                spent, budget = self._build_spent, self.build_budget
                raise RuntimeError(f"{self.system} 写入已花 ${spent:.2f}，超过预算 ${budget}")
        self.close(key)
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
        before = self.gateway.usage(self.bucket(key))
        started = datetime.now()
        try:
            with ThreadPoolExecutor(1) as pool:  # 独立线程：不算进这道题的回答用量
                stats = pool.submit(self.build, key, directory, sessions).result()
        except Exception as exc:
            with self._lock:
                self._failed[key] = f"{type(exc).__name__}: {exc}"[:300]
            raise
        usage = _diff(self.gateway.usage(self.bucket(key)), before)
        stats = {
            **stats,
            "protocol": self.protocol,
            "build_tag": self.build_tag,
            "sessions": len(sessions),
            "seconds": round((datetime.now() - started).total_seconds(), 1),
            **usage,
        }
        if replaced:  # 换掉的是哪一版写入
            stats["replaced"] = replaced
        if scheme != "native":  # LoCoMo 的写入记录不变
            stats["session_ids"] = scheme
        (directory / "build.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), "utf-8"
        )
        with self._lock:
            self._build_spent += float(usage.get("spent", 0.0))
        return stats

    def _space_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._space_locks.setdefault(key, threading.Lock())


def space_key(space_id: str) -> str:
    """评测空间 ID → 目录名（如 locomo:conv-42 → locomo_conv-42）。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", space_id)


# ---------- 写入的协议（外部基线与 M0 共用） ----------


def protocol_of(settings: Settings, embedder: TextEmbedder) -> str:
    """这次运行的写入协议"抽取模型|向量模型:维度"：外部系统内部调用的模型（extractor 角色的
    model_id）与向量模型。与 report.run_protocol 同一写法。"""
    model = settings.models[settings.roles["extractor"].model].model_id
    return f"{model}|{embedder.name}:{embedder.dim}"


def model_ids(settings: Settings) -> dict[str, str]:
    """模型别名 → model_id：记协议之前的 Graphiti、LangMem 写入记的是 extractor 的别名。"""
    return {alias: spec.model_id for alias, spec in settings.models.items()}


def split_protocol(protocol: str) -> tuple[str, str]:
    """build.json 里的 protocol → （"抽取模型|向量模型:维度"，接在后面的写入标记）。有 build_tag
    字段之前，A-MEM 与 Cognee 把写入标记接在协议后面（如 …|amem@ceffb86|max_tokens=4096|v4）。"""
    parts = protocol.split("|")
    return "|".join(parts[:2]), "|".join(parts[2:])


def recorded_protocol(
    record: Mapping[str, Any], aliases: Mapping[str, str] | None = None
) -> str | None:
    """写入记录的协议（不含写入标记）；没记协议的旧记录返回 None（按当前协议处理）。

    记 protocol 之前的 Graphiti、LangMem 写入记了 extractor（模型别名，aliases 换成 model_id）与
    embedder（Graphiti 的带维度；LangMem 的不带，维度另记在 dim），照样认出协议（与
    report._protocol_match 相同）。"""
    if record.get("protocol"):
        return split_protocol(str(record["protocol"]))[0]
    extractor, embedder = record.get("extractor"), record.get("embedder")
    if not extractor or not embedder:
        return None
    model = (aliases or {}).get(str(extractor), str(extractor))
    dim = f":{record['dim']}" if record.get("dim") is not None else ""
    return f"{model}|{embedder}{dim}"


def check_protocol(
    record: Mapping[str, Any],
    protocol: str,
    directory: Path,
    system: str,
    aliases: Mapping[str, str] | None = None,
) -> None:
    """写好的写入是另一个协议的（抽取模型或向量模型不同）时报 ConfigMismatchError。

    不删、也不复用：评测空间的目录不分协议，删了要花钱重写（Graphiti 一个协议约 $10、几个
    小时），复用则是拿别的模型写的记忆（与不同维度的向量）来比。"""
    built = recorded_protocol(record, aliases)
    if built is None or built == protocol:
        return
    raise ConfigMismatchError(
        f"{directory} 是另一个协议（{built}）的 {system} 写入，这次运行的协议是 {protocol}："
        "不复用，也不覆盖。评测空间的目录不分协议，换协议要换目录：pds 协议（只用 DeepSeek）"
        "用默认的 data/eval-spaces，pq 协议（只用通义）设 "
        "CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen，别的协议指向另一个目录"
        "（见 scripts/n12_baselines.py）；确实不要这份写入了，先把这个目录挪走再运行。"
    )


def memory_hit(
    space_id: str,
    item_id: str,
    text: str,
    when: datetime,
    rank: int,
    *,
    score: float = 0.0,
    route: str = "external",
    session_id: str = "",
    kind: UnitKind = "fact",
) -> Hit:
    """外部系统的一条记忆 → 回答模型看到的一条证据（时间取它自己记下的时间，没有就取会话时间）。"""
    unit = MemoryUnit(
        id=item_id,
        space_id=space_id,
        kind=kind,
        session_id=session_id,
        text=text,
        ts=when.replace(tzinfo=None),
        source_turn_ids=[],
    )
    return Hit(unit=unit, score=score, route=route, rank=rank)


def _diff(after: dict[str, Any], before: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field, value in after.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[field] = round(value - before.get(field, 0), 6)
    return out
