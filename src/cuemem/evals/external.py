"""外部基线的公共部分（N12）：Graphiti、LangMem、Cognee、A-MEM 等都从这里派生。

与 N11 的 Mem0 基线（evals/mem0_baseline.py）同样的原则：**按它自己的流程写入与检索，回答与判分
和其他方法完全相同**，只换"记忆从哪来"。

- **模型**：外部系统用自己带的 OpenAI 兼容客户端，base_url 指向评测网关（evals/gateway.py）：
  聊天一律交给 extractor 角色的模型，向量一律交给 CueMem 的同一个向量器；同一本账、同一个缓存。
- **写入**：每份对话历史（LoCoMo 一段对话；LongMemEval 一道题的历史）第一次用到时写入一次，
  写完留下 build.json（写入用量从网关的桶里读），之后的运行直接复用。写入在独立线程里做，不算进
  当前这道题的回答用量；半成品下次整个重建；失败过的历史不再反复重建（每次都要花钱）。
- **检索**：子类返回一组证据（Hit），交给同一个回答模型、同一套提示
  （与 B2 朴素检索、M0 的格式相同）。

外部系统都不是 CueMem 的依赖，评测时用 uv 临时装上，例如：

    uv run --with graphiti-core cuemem eval run --dataset locomo --split test200r --methods MG
"""

from __future__ import annotations

import json
import re
import shutil
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from cuemem.core.models import Hit, MemoryUnit, UnitKind
from cuemem.embed import build_embedder
from cuemem.embed.base import TextEmbedder
from cuemem.evals.datasets import EvalSession, Sample
from cuemem.evals.gateway import Gateway
from cuemem.evals.methods import MethodContext, MethodOutput
from cuemem.llm.client import LLMClient

# 每个外部系统写入的花费上限（美元，实际花费）。写入不在回答的 trace 里，评测的 --max-cost 管不到它
DEFAULT_BUILD_BUDGET = 10.0


class ExternalBaseline(ABC):
    """子类填 name、label、system（数据目录与网关桶的前缀），实现 build 与 retrieve。"""

    name: str = ""
    label: str = ""
    system: str = ""
    needs_index, needs_consolidation = False, False
    build_budget: float = DEFAULT_BUILD_BUDGET

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._space_locks: dict[str, threading.Lock] = {}
        self._ready = False
        self._build_spent = 0.0
        self._failed: dict[str, str] = {}

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
        key = space_key(ctx.space_id)
        directory = self.root / key
        with self._space_lock(key):
            built = self._ensure_built(key, directory, ctx.sessions)
            hits, details = self.retrieve(key, directory, ctx.sample, ctx.sessions)
        a = ctx.reader.answer(ctx.sample.question, hits, question_time=ctx.sample.question_time)
        details = {"memories": len(hits), **details}
        if built is not None:  # 这道题触发了写入：写入的统计记在它名下
            details[f"{self.system}_build"] = built
        return MethodOutput(a.answer, a.evidence_sufficient, hits=hits, details=details)

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
            # 这次运行的协议：外部系统内部调用的模型与向量模型；写入记录里存一份，换了就重建
            model = settings.models[settings.roles["extractor"].model].model_id
            self.protocol = f"{model}|{self.embedder.name}:{self.embedder.dim}"
            self.setup()
            self._ready = True

    def _ensure_built(
        self, key: str, directory: Path, sessions: Sequence[EvalSession]
    ) -> dict[str, Any] | None:
        record = directory / "build.json"
        if record.exists():
            built = json.loads(record.read_text("utf-8")).get("protocol")
            # 换了协议（抽取模型或向量模型）的写入不能复用；旧记录没有这个字段，按同一协议处理
            if built is None or built == self.protocol:
                return None
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
            "sessions": len(sessions),
            "seconds": round((datetime.now() - started).total_seconds(), 1),
            **usage,
        }
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
