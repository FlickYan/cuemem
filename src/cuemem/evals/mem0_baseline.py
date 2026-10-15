"""外部基线 M0：开源的 Mem0（mem0ai 2.x），N11。

按 Mem0 自己的流程写入与检索，回答与判分和其他方法完全相同，只换"记忆从哪来"：

- **写入**：每个会话调用一次 `Memory.add`（整个会话的消息）。Mem0 内部做一次 LLM 抽取
  （2.x 起只增不改，提示里带上最相关的 10 条已有记忆）、向量化与实体链接。
  CueMem 也是每个会话抽取一次。两个人之间的对话（LoCoMo）照 Mem0 论文的评测代码，每位说话人
  一个 user_id、各写一遍（自己的话是 user，对方的话是 assistant），抽取调用是 CueMem 的两倍。
- **同样的模型**：Mem0 的 LLM 换成 CueMem 写入路径用的同一个模型（extractor 角色），
  向量换成同一个向量模型。两者都走 CueMem 的客户端：同一本账、同一个本地缓存、同样的高峰暂停。
- **日期**：开源版不支持 `timestamp`（平台版才有），抽取提示里的"观察日期"默认是今天，
  "昨天""上周"会被换算到运行当天。这里把它换成会话日期，相当于平台版传了 timestamp。
- **检索**：`Memory.search(top_k=30)`，两人对话每位说话人各取 30 条。Mem0 2.x 的混合检索
  （语义 + BM25 + 实体加权）需要 spaCy 与 fastembed，运行命令里一并装上；每条记忆标上来源会话
  的日期（两人对话再注明是谁的记忆），交给同一个回答模型、同一套提示（与 B2 朴素检索的格式相同）。

mem0ai 不是 CueMem 的依赖，评测时用 uv 临时装上（遥测在导入前关闭）：

    uv run --with mem0ai --with spacy --with fastembed --with "en_core_web_sm @ $SPACY_SM" \\
        cuemem eval run --dataset locomo --split test200r --methods M0 --name locomo-test200r-mem0

其中 SPACY_SM 是 spaCy 英文小模型的地址（与 spaCy 3.8 配套）：
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import APITimeoutError, BadRequestError

from cuemem.core.models import Hit, MemoryUnit
from cuemem.core.trace import start_trace
from cuemem.embed import build_embedder
from cuemem.embed.base import EmbedKind, TextEmbedder
from cuemem.errors import ContentFilteredError
from cuemem.evals.datasets import (
    EvalSession,
    SessionIds,
    build_matches_session_ids,
    storage_space_id,
)
from cuemem.evals.methods import MethodContext, MethodOutput
from cuemem.llm.client import LLMClient, Message

TOP_K = 30  # Mem0 默认 20；多给一些，免得说限制了它的召回
USER = "user"
MAX_OPEN = 8  # 同时打开的 Mem0 存储（每个对话历史一个本地 Qdrant）
# 写入的花费上限（美元，实际花费）：写入不在回答的 trace 里，评测的 --max-cost 管不到它
BUILD_BUDGET = 10.0
_observation = threading.local()  # 当前线程正在写入的会话日期（替换抽取提示里的"今天"）


class _LLM:
    """Mem0 的 LLM 接口（generate_response）→ CueMem 的 LLMClient。

    与 CueMem 自己的抽取一样：内容审核误拒、请求超时、JSON 生成被中止时，改用备用模型
    （另一家供应商）再试一次。
    """

    def __init__(self, llm: LLMClient, scope: str) -> None:
        self.llm = llm
        self.scope = scope
        self.fallback = llm.settings.consolidation.fallback_model

    def generate_response(
        self,
        messages: list[Message],
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        **_: Any,
    ) -> str:
        def call(model: str | None) -> str:
            result = self.llm.chat(
                "extractor",
                messages,
                model=model,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice if tools else None,
                cache_scope=self.scope,
            )
            return _wrap_list(result.content or "") if response_format else result.content or ""

        try:
            return call(None)
        except (ContentFilteredError, APITimeoutError, BadRequestError) as exc:
            # 通义在 JSON 模式下偶尔中止生成（"Model output became abnormal"），重发多半还会中止
            aborted = "Model output became abnormal" in str(exc)
            if self.fallback is None or (isinstance(exc, BadRequestError) and not aborted):
                raise
            return call(self.fallback)


def _wrap_list(content: str) -> str:
    """通义的 JSON 模式偶尔直接返回记忆列表（`[{"id": …, "text": …}]`），少了外层的
    `{"memory": …}`；Mem0 解析失败时把整个会话当成"没有可记的"。OpenAI 的 JSON 模式不会返回
    顶层数组，所以这是换模型带来的差异，在适配层补上外层。"""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    if not text.startswith("["):
        return content
    try:
        items = json.loads(text, strict=False)
    except json.JSONDecodeError:
        return content
    return json.dumps({"memory": items}, ensure_ascii=False) if isinstance(items, list) else content


class _Embedder:
    """Mem0 的向量接口（embed / embed_batch）→ CueMem 的向量器（带内容哈希缓存）。"""

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder

    def embed(self, text: str, memory_action: str | None = None) -> list[float]:
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(
        self, texts: Sequence[str], memory_action: str | None = "add"
    ) -> list[list[float]]:
        kind: EmbedKind = "query" if memory_action == "search" else "document"
        return [row.tolist() for row in self.embedder.embed(list(texts), kind).dense]


def _import_mem0(home: Path) -> Any:
    """导入前关掉遥测（PostHog）并把 Mem0 的配置目录放进数据目录；不拉取远程公告。"""
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ.setdefault("MEM0_DIR", str(home))
    home.mkdir(parents=True, exist_ok=True)
    try:
        import mem0  # type: ignore[import-not-found, unused-ignore]
        import mem0.memory.main as mem0_main  # type: ignore[import-not-found, unused-ignore]
        import mem0.memory.notices as mem0_notices  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "M0 需要 mem0ai：用 uv run --with mem0ai ... 运行（见模块说明）"
        ) from exc
    mem0_notices._fetch_remote_config = lambda: None
    original = mem0_main.generate_additive_extraction_prompt

    def dated(*args: Any, **kwargs: Any) -> Any:
        day = getattr(_observation, "day", None)
        if day is not None:
            kwargs.setdefault("timestamp", day)
            kwargs.setdefault("current_date", day)
        return original(*args, **kwargs)

    mem0_main.generate_additive_extraction_prompt = dated
    return mem0


class Mem0Baseline:
    name, label = "M0", "外部基线：Mem0（开源版 2.x）"
    needs_index, needs_consolidation = False, False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._space_locks: dict[str, threading.Lock] = {}
        self._open: OrderedDict[str, Any] = OrderedDict()
        self._ready = False
        self._build_spent = 0.0
        self._failed: dict[str, str] = {}  # 写入失败过的对话历史：不再反复重建（每次都要花钱）

    def _setup(self, llm: LLMClient) -> None:
        """第一次回答时初始化：沿用评测的 LLM 客户端（同一本账），向量器按同一份配置构造。"""
        with self._lock:
            if self._ready:
                return
            settings = llm.settings
            self.root = settings.paths.eval_spaces / "mem0"
            self.llm = llm
            self.embedder = build_embedder(settings, ledger=llm.ledger)
            self.mem0 = _import_mem0(self.root / "_home")
            self._ready = True

    def answer(self, ctx: MethodContext) -> MethodOutput:
        self._setup(ctx.reader.llm)
        # 如 locomo:conv-42 → locomo_conv-42；会话 ID 改写过的另存（longmemeval_s-anon_<题号>）
        stored = storage_space_id(ctx.space_id, ctx.session_ids)
        key = re.sub(r"[^A-Za-z0-9_.-]", "_", stored)
        found: list[tuple[str | None, dict[str, Any]]] = []
        with self._space_lock(key):
            built = self._ensure_built(key, ctx.sessions, ctx.session_ids)
            memory = self._memory(key)
            for me in _perspectives(ctx.sessions):
                results = memory.search(
                    ctx.sample.question, top_k=TOP_K, filters={"user_id": me or USER}
                )["results"]
                found += [(me, item) for item in results]
        hits = [
            _hit(ctx.space_id, item, rank, me) for rank, (me, item) in enumerate(found, start=1)
        ]
        a = ctx.reader.answer(ctx.sample.question, hits, question_time=ctx.sample.question_time)
        details: dict[str, Any] = {"memories": len(hits)}
        if built is not None:  # 这道题触发了写入：写入的统计记在它名下
            details["mem0_build"] = built
        return MethodOutput(a.answer, a.evidence_sufficient, hits=hits, details=details)

    # ---------- 写入 ----------

    def _ensure_built(
        self, key: str, sessions: Sequence[EvalSession], session_ids: SessionIds = "native"
    ) -> dict[str, Any] | None:
        """第一次用到这份对话历史时写入 Mem0；写完留下 build.json，之后的运行直接复用。

        写入在独立线程里做：不算进当前这道题的回答用量（评测 trace 绑定在调用线程上），
        单独统计后写进 build.json。中途失败的半成品下次整个重建。

        记忆的元数据里存着会话 ID（证据的抬头会显示它）：LongMemEval / MemLens 的写入记下
        会话 ID 的写法，对不上就重建（改写之前的写入没有这条记录，都是原始 ID）。
        """
        directory = self.root / key
        record = directory / "build.json"
        if record.exists() and (
            session_ids == "native" or build_matches_session_ids(_read_json(record), session_ids)
        ):
            return None
        with self._lock:
            if key in self._failed:
                raise RuntimeError(f"Mem0 写入这份对话历史时失败过，不再重试：{self._failed[key]}")
            if self._build_spent > BUILD_BUDGET:
                spent = self._build_spent
                raise RuntimeError(f"Mem0 写入已花 ${spent:.2f}，超过预算 ${BUILD_BUDGET}")
        self._close(key)
        shutil.rmtree(directory, ignore_errors=True)
        try:
            with ThreadPoolExecutor(1) as pool:
                stats = pool.submit(self._build, key, sessions).result()
        except Exception as exc:
            with self._lock:
                self._failed[key] = f"{type(exc).__name__}: {exc}"[:300]
            raise
        if session_ids != "native":  # LoCoMo 的写入记录不变
            stats["session_ids"] = session_ids
        record.write_text(json.dumps(stats, indent=2), "utf-8")
        with self._lock:
            self._build_spent += float(stats.get("spent", 0.0))
        return stats

    def _build(self, key: str, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        from cuemem.evals.runner import usage_from_trace

        memory = self._memory(key)
        perspectives = _perspectives(sessions)
        added = 0
        failed: list[str] = []
        with start_trace("mem0_ingest", space=key) as trace:
            for session in sorted(sessions, key=lambda s: s.started_at):
                for me in perspectives:
                    messages = _messages(session, me)
                    if not messages:
                        continue
                    _observation.day = session.started_at.date().isoformat()
                    try:
                        result = memory.add(
                            messages,
                            user_id=me or USER,
                            metadata={
                                "session_id": session.session_id,
                                "session_time": session.started_at.isoformat(),
                            },
                        )
                        added += len(result.get("results", []))
                    except Exception as exc:  # 换了备用模型也失败的会话：跳过并计数，别的照常
                        who = f"（{me}）" if me else ""
                        error = f"{type(exc).__name__}: {exc}"
                        failed.append(f"{session.session_id}{who}: {error}"[:200])
                    finally:
                        _observation.day = None
        return {
            "sessions": len(sessions),
            "perspectives": [me or USER for me in perspectives],
            "memories": added,
            "failed_sessions": failed,
            **usage_from_trace(trace),
        }

    # ---------- 存储 ----------

    def _space_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._space_locks.setdefault(key, threading.Lock())

    def _memory(self, key: str) -> Any:
        """打开（或复用）一份对话历史的 Mem0 存储；最多同时开 MAX_OPEN 个，多了关掉最久没用的。"""
        with self._lock:
            if key in self._open:
                self._open.move_to_end(key)
                return self._open[key]
        directory = self.root / key
        directory.mkdir(parents=True, exist_ok=True)
        dim = self.embedder.dim
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "path": str(directory / "qdrant"),
                    "embedding_model_dims": dim,
                    "on_disk": True,
                },
            },
            # 占位配置：构造后立即换成 CueMem 的客户端，不会用这里的 Key 发请求
            "llm": {"provider": "openai", "config": {"model": "unused", "api_key": "fake-unused"}},
            "embedder": {
                "provider": "openai",
                "config": {"model": "unused", "api_key": "fake-unused", "embedding_dims": dim},
            },
            "history_db_path": str(directory / "history.db"),
        }
        memory = self.mem0.Memory.from_config(config)
        memory.llm = _LLM(self.llm, f"mem0|{key}")
        memory.embedding_model = _Embedder(self.embedder)
        with self._lock:
            self._open[key] = memory
            while len(self._open) > MAX_OPEN:
                old_key, _ = next(iter(self._open.items()))
                if self._space_locks.get(old_key, threading.Lock()).locked():
                    break  # 正在用：留着，下次再关
                self._close_locked(old_key)
        return memory

    def _close(self, key: str) -> None:
        with self._lock:
            self._close_locked(key)

    def _close_locked(self, key: str) -> None:
        memory = self._open.pop(key, None)
        if memory is None:
            return
        memory.close()
        memory.vector_store.client.close()


def _read_json(path: Path) -> dict[str, Any]:
    """读写入记录；读不了时当作空记录（于是按原始 ID 处理，对不上就重建）。"""
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _perspectives(sessions: Sequence[EvalSession]) -> list[str | None]:
    """按谁的视角写入：两个人之间的对话（LoCoMo：两位说话人都是 user 角色、带名字）
    照 Mem0 论文的评测代码，每位说话人一个 user_id；其余（用户与助手）只有一个视角（None）。"""
    turns = [t for s in sessions for t in s.turns]
    speakers = list(dict.fromkeys(t.speaker for t in turns if t.speaker))
    if len(speakers) == 2 and all(t.role == "user" for t in turns):
        return list(speakers)
    return [None]


def _messages(session: EvalSession, me: str | None) -> list[dict[str, str]]:
    """一个会话交给 Mem0 的消息。两人对话里，"我"的话是 user、对方的话是 assistant，都带名字
    （Mem0 2.x 的抽取提示专门说明了这种格式：assistant 也可能是一个讲自己经历的真人）。"""
    messages = []
    for t in session.turns:
        if not t.text.strip():
            continue
        role = t.role if me is None else ("user" if t.speaker == me else "assistant")
        content = f"{t.speaker}: {t.text}" if t.speaker else t.text
        messages.append({"role": role, "content": content})
    return messages


def _hit(space_id: str, item: dict[str, Any], rank: int, me: str | None = None) -> Hit:
    """Mem0 的一条记忆 → 回答模型看到的一条证据：时间取来源会话的时间。

    两人对话里，记忆按说话人分开存，抽取出的 "User" 指这位说话人；与 Mem0 论文的回答提示
    （"Memories for user A / B"）一样，每条注明是谁的记忆。"""
    meta = item.get("metadata") or {}
    when = meta.get("session_time") or item.get("created_at")
    text = str(item["memory"])
    if me is not None:
        text = f'[{me}\'s memory; "User" = {me}] {text}'
    unit = MemoryUnit(
        id=str(item["id"]),
        space_id=space_id,
        kind="fact",
        session_id=str(meta.get("session_id", "")),
        text=text,
        ts=datetime.fromisoformat(str(when)).replace(tzinfo=None),
        source_turn_ids=[],
    )
    return Hit(unit=unit, score=float(item.get("score") or 0.0), route="mem0", rank=rank)
