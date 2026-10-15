"""外部基线 ML：LangMem（langchain-ai/langmem 0.0.30），N12。

按 LangMem 文档里"后台形成记忆"的做法写入与检索，回答与判分和其他方法完全相同（见
evals/external.py），只换"记忆从哪来"：

- **写入**：`create_memory_store_manager`（LangGraph 的 InMemoryStore，带向量索引）。会话按时间
  顺序、每个会话调用一次（文档里对话结束、空闲下来后整段交给管理器处理的用法）。管理器先用会话的
  最后一句检索最相关的 5 条已有记忆，再让模型用工具调用（trustcall）新增或修改记忆：每个会话一次
  抽取调用，与 CueMem 相同（trustcall 校验失败时会多一两次修补调用）。
- **模型**：LangChain 的 `ChatOpenAI` 与 `OpenAIEmbeddings` 指向评测网关：聊天交给 extractor 角色
  的模型，向量交给 CueMem 的向量器（请求里的模型名只是占位）。每个会话只有一轮，LangMem 要求
  模型"在一次回复里并行调用多个工具"（每条新记忆、每处修改各一个调用）；ChatOpenAI 不写
  parallel_tool_calls（OpenAI 的默认就是 true），网关按 OpenAI 的默认发出 true——通义不设为 true
  时一次只回一个工具调用，每个会话就只能记一条。还没有已有记忆时不带 tool_choice（模型自己决定
  调不调工具）；有已有记忆时 trustcall 要求必须调用（"any"，ChatOpenAI 发出去是 "required"），
  网关原样转给模型（通义的非思考模式不保证照做：没调用的回复计入 no_tool_replies）。
- **检索**：`store.search(("memories",), query=问题, limit=60)`，即 LangMem 检索工具的做法；
  按名次装进约 4000 tokens（与 L1 的预算与装法相同：放不下的跳过，后面更短的还能放）。
  问题按查询向量化（网关对检索阶段的桶用 kind="query"，与 L1 相同；LangMem 参考的 OpenAI 向量
  不分查询与文档，通义的分）。每条记忆标上写下它的会话日期，交给同一个回答模型、同一套提示。

与 LangMem 自身默认（及唯一公开的 LoCoMo 跑法）不同的地方，以及原因：

1. Mem0 论文（arXiv 2504.19413）的评测代码（mem0ai/mem0 evaluation/src/langmem.py）跑的是
   "热路径"：每位说话人一个 create_react_agent（带 manage_memory / search_memory 工具），
   逐句喂给只属于自己的那个 Agent，提问时把 Agent 的自然语言回复当作记忆。这里不照搬：
   同一线程的消息检查点逐句累积（调用量与上下文都随对话长度平方增长）；每个 Agent 只看到
   一方的话；Agent 的回复是改写过的散文，没法交给同一个回答模型、也带不上日期。换成文档推荐的
   后台记忆管理器，每个会话处理一次。
2. 两人对话（LoCoMo）不分说话人：一份对话一个命名空间，整段会话（每句带说话人名字）交给管理器。
   LangMem 的提示只有"用户"与"Agent 自己"两种身份，把另一位说话人放进 assistant 角色会被当成
   Agent 本身；两人都放 user 角色时分两份写入只是重复同样的输入、调用翻倍。
3. 日期：LangMem 没有时间参数，照 Mem0 评测代码的格式在每句前面写上会话时间
   （`2023-05-08 (Mon) 13:56 | Caroline: …`）；另外按管理器返回的写入记下每条记忆第一次写下与
   最后一次修改所在的会话，回答模型看到的时间取最后一次（与第一次不同时在文本后注明）。
4. 检索条数：LangMem 检索工具默认 limit=10，这里取 60（与 M0 两位说话人各 30 条相当），再按
   L1 的 4000 tokens 装入，两者先到为准。LangMem 会把同一主题的信息合并成长记忆（两次试跑的
   conv-42：90 与 117 条，平均约 250 与 170 tokens，最长约 3600 与 600），实际装进去的多是
   8–24 条。
5. 存储：langgraph 自带的向量存储只有 InMemoryStore（其余要 Postgres 或另装 sqlite-vec），
   写完把条目与向量一起存成 store.json，检索时原样装回，不重新向量化。
6. 记忆 ID：LangMem 用 uuid4 给新记忆与提示里的会话标签编号，每次重建提示都不同、响应缓存全部
   落空。这里只把 langmem.knowledge.extraction 里的 uuid4 换成按（对话, 会话）定种子的随机数，
   ID 依旧是不透明的随机串，重建时命中缓存、几乎不花钱。
7. 其余全用默认：记忆提示 _MEMORY_INSTRUCTIONS、Memory 结构、enable_inserts=True、
   enable_deletes=False、query_limit=5、max_steps=1、没有 phases 与 query_model、
   索引字段 "$"（整个 JSON 值做向量）、不设 max_tokens（用模型自己的输出上限）。
8. 失败：模型调用出错时 trustcall 自己会把错误写进提示重试（有已有记忆时），直到 LangGraph
   的步数上限（25）报 GraphRecursionError，这里不改。最终失败的会话里，被内容审核拒绝的
   （网关换了备用模型也拒绝，重试没用；看网关桶里的 refusals）跳过并计数，与 M0 相同；
   其他错误让整份写入失败、不留 build.json，下次运行重建（定种子的 ID 让前面的会话全部命中
   响应缓存）——跳过它们，后面的会话会建在缺了一块的存储上，出错时已写进存储的部分记忆也
   对不上会话日期。被拒绝的与写入丢了的会话（没调用工具、补丁改不了、调用校验不过而一条也
   没写）合计超过 max(2, 10%) 时整份写入算失败；带工具的回复满 10 次却没有一次并行调用多个
   工具时也算失败（并行工具调用没生效）。模型调用了 PatchDoc 却明说"不用改"（patches 为空）
   的会话是 LangMem 照常的结果、不算写入丢了，另记在 unchanged_sessions 里：有已有记忆时
   trustcall 要求必须调用工具，通义觉得没有新东西时就这样回。
9. 写入记录（build.json）另外记下：每个会话写入的条数、没写入的会话、网关统计的工具调用
   （tool_calls、multi_tool_replies、no_tool_replies）与被长度上限截断的回复
   （truncated_replies，截断的工具调用 LangChain 会静默丢掉）、trustcall 跳过的补丁
   （skipped_edits，路径不对等）与校验不过被丢掉的调用（dropped_calls）、各包的版本。
   第二次试跑（deepseek 预设）：33 次抽取调用里 31 次一次回多个工具调用，共 169 个；DeepSeek
   有 4 次给 Memory 传了空参数，trustcall 让它修补时它又附带的 9 条 Memory 调用被丢掉
   （dropped_calls=9，trustcall 自身的行为）。通义预设的试跑（conv-42，qwen3.8-flash）：29 次
   抽取调用全都调用了工具（no_tool_replies=0），8 次一次回多个、共 39 个；通义多用一个 PatchDoc
   把新信息并进少数几条长记忆（共 20 条，最长 7461 个字符，2 条超过 max_chars），有 2 个会话
   回了空补丁（unchanged_sessions），1 个会话的补丁把路径写成列表 ["content"]、trustcall 改不了
   （skipped_edits，记作写入丢了）。
10. store.json 与 build.json 记下写入用的抽取模型与向量器；评测空间的目录不分预设，换了模型
    时删掉旧的写入重建，不沿用别的模型写出的记忆。
11. 已知的限制：InMemoryStore 索引整个 JSON 值，CueMem 的向量器只取前 max_chars（6000）个
    字符，本机的 bge-small 还只看前 512 tokens（约 2000 个字符）；LangMem 合并出的长记忆
    （第一次试跑有两条一万多字符）尾部的内容因此检索不到，LangMem 参考的
    text-embedding-3-small 能看 8191 tokens。build.json 记下最长记忆的字符数与超过 max_chars
    的条数（第二次试跑最长 2415 个字符，没有超过的）。

langmem 不是 CueMem 的依赖，评测时用 uv 临时装上（--extra local 只在用本机向量时需要，
如 deepseek 预设）。这里读写 InMemoryStore 的内部字段（_data、_vectors，在
langgraph-checkpoint 里），按 trustcall 的日志数跳过的补丁，所以相关的包都钉在试过的版本：

    uv run --extra local --with langmem==0.0.30 --with langchain-openai==1.7.2 \\
        --with langchain-core==1.6.10 --with langgraph==1.2.15 \\
        --with langgraph-checkpoint==4.3.0 --with trustcall==0.0.39 \\
        cuemem eval run --dataset locomo --split test200r --methods ML \\
        --name locomo-test200r-langmem
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import random
import uuid
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from cuemem.core.models import Hit
from cuemem.evals.datasets import EvalSession, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit
from cuemem.read.l1 import estimate_tokens

NAMESPACE = ("memories",)  # LangMem 快速上手与 Mem0 评测代码用的命名空间；一份对话一个存储
TOP_K = 60  # LangMem 检索工具默认 10；与 M0（两位说话人各 30 条）相当
BUDGET_TOKENS = 4000  # 与 CueMem L1 的上下文预算相同
CHAT_MODEL = "gpt-4o-mini"  # 只是请求里的名字（Mem0 论文用的模型）：网关一律交给 extractor 角色
EMBED_MODEL = "text-embedding-3-small"  # 同上：网关一律交给 CueMem 的向量器
STORE_FILE = "store.json"
MAX_OPEN = 8  # 同时装在内存里的存储
MAX_LOST = 2  # 被拒绝或没写入的会话最多容许 max(MAX_LOST, 10%) 个
MIN_TOOL_REPLIES = 10  # 带工具的回复满这么多次还没有一次并行调用多个工具：并行没生效
PACKAGES = (
    "langmem",
    "langchain-openai",
    "langchain-core",
    "langgraph",
    "langgraph-checkpoint",
    "trustcall",
)

# 正在写入的会话（给 LangMem 的 uuid4 定种子）与正在写入的对话（收集 trustcall 报告的问题）。
# 用 ContextVar 而不是按线程记：LangChain 把同步的步骤放进线程池跑（复制上下文），
# trustcall 丢掉校验不过的调用时就是在那里记的日志
_rng: contextvars.ContextVar[random.Random | None] = contextvars.ContextVar(
    "langmem_rng", default=None
)
_problems: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "langmem_problems", default=None
)


class _SeededUuid:
    """langmem.knowledge.extraction 里 uuid 模块的替身：写入时 uuid4 由（对话, 会话）定种子。

    只替换这一个模块里的名字，别的库照常用真随机的 uuid4；不在写入中时也是真随机。"""

    def __getattr__(self, name: str) -> Any:
        return getattr(uuid, name)

    def uuid4(self) -> uuid.UUID:
        rng = _rng.get()
        if rng is None:
            return uuid.uuid4()
        return uuid.UUID(int=rng.getrandbits(128), version=4)


class _Problems(logging.Handler):
    """把 trustcall（logger "extraction"）报告的问题记到正在写入的那份对话名下：补丁没能应用、
    找不到要改的记忆、工具调用校验不过被丢掉等。trustcall 只记日志、跳过，不报错。"""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)

    def emit(self, record: logging.LogRecord) -> None:
        seen = _problems.get()
        if seen is not None:
            seen.append(record.getMessage()[:200])


_PROBLEMS = _Problems()


def _import_langmem() -> None:
    try:
        import langmem.knowledge.extraction as extraction  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError as exc:  # pragma: no cover - 取决于运行环境
        raise RuntimeError(
            "ML 需要 langmem 与 langchain-openai：用 uv run --with langmem==0.0.30 "
            "--with langchain-openai ... 运行（见模块说明）"
        ) from exc
    extraction.uuid = _SeededUuid()
    logger = logging.getLogger("extraction")  # trustcall 的 logger
    if _PROBLEMS not in logger.handlers:
        logger.addHandler(_PROBLEMS)


class LangMemBaseline(ExternalBaseline):
    name, label, system = "ML", "外部基线：LangMem（0.0.30，后台记忆管理器）", "langmem"

    def __init__(self) -> None:
        super().__init__()
        self._open: OrderedDict[str, tuple[Any, dict[str, dict[str, str]]]] = OrderedDict()

    def setup(self) -> None:
        _import_langmem()

    # ---------- 写入 ----------

    def _ensure_built(
        self, key: str, directory: Path, sessions: Sequence[EvalSession]
    ) -> dict[str, Any] | None:
        # 评测空间的目录不分预设：别的模型写出的记忆不能沿用。删掉 build.json，基类就整个目录重建
        record = directory / "build.json"
        if record.exists() and not same_identity(record, self._identity()):
            record.unlink()
        return super()._ensure_built(key, directory, sessions)

    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        # 整份写入放在一个事件循环里：异步路径会把写入错误抛出来（同步的 invoke 在线程池里
        # 写存储，向量请求失败时静默丢掉那条记忆）
        return asyncio.run(self._build(key, directory, sessions))

    async def _build(
        self, key: str, directory: Path, sessions: Sequence[EvalSession]
    ) -> dict[str, Any]:
        import httpx
        from langgraph.store.memory import InMemoryStore  # type: ignore[import-not-found, unused-ignore]
        from langmem import create_memory_store_manager  # type: ignore[import-not-found, import-untyped, unused-ignore]

        bucket = self.bucket(key)
        url = self.gateway.url(bucket)
        before = self.gateway.usage(bucket)
        written: dict[str, dict[str, str]] = {}
        updates = 0
        puts_per_session: list[int] = []
        refused: list[str] = []
        empty: list[str] = []
        unchanged: list[str] = []  # 模型调用了工具、却明说没有要改的（见模块说明第 8 条）
        problems: list[str] = []
        collecting = _problems.set(problems)
        # 每份写入一个 HTTP 客户端、用完关掉：LangChain 默认按 base_url 缓存异步客户端，每份
        # 对话的桶地址都不同，缓存里会留下一个个绑在已关闭事件循环上的客户端。网关在本机：
        # 不走环境变量里的代理（LangChain 的默认客户端也不走）
        http = httpx.AsyncClient(trust_env=False)
        try:
            embed = _embeddings(url, http)
            store = InMemoryStore(index={"dims": self.embedder.dim, "embed": embed})
            model = _chat_model(url, http)
            manager = create_memory_store_manager(model, namespace=NAMESPACE, store=store)
            ordered = sorted(sessions, key=lambda s: s.started_at)
            for number, session in enumerate(ordered):
                messages = session_messages(session)
                if not messages:
                    continue
                seeded = _rng.set(random.Random(f"{key}|{number}|{session.session_id}"))
                start = self.gateway.usage(bucket)
                seen = len(problems)
                try:
                    puts = await manager.ainvoke({"messages": messages})
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    # trustcall 把模型调用的错误改写成提示重试，最后报的是 GraphRecursionError：
                    # 是不是内容审核拒绝，看这个会话期间网关桶里的拒绝次数
                    if self.gateway.usage(bucket)["refusals"] == start["refusals"]:
                        raise RuntimeError(  # 整份作废：下次重建，前面的会话命中缓存
                            f"LangMem 写入会话 {session.session_id} 失败：{error}"[:300]
                        ) from exc
                    refused.append(f"{session.session_id}: {error}"[:200])
                    continue
                finally:
                    _rng.reset(seeded)
                puts_per_session.append(len(puts))
                if not puts:
                    end = self.gateway.usage(bucket)
                    lost = not chose_no_change(start, end, problems[seen:])
                    (empty if lost else unchanged).append(session.session_id)
                updates += record_writes(written, puts, session)
        finally:
            _problems.reset(collecting)
            await http.aclose()
        usage = self.gateway.usage(bucket)
        replies = {
            field: usage[field] - before.get(field, 0)
            for field in ("tool_requests", "multi_tool_replies")
        }
        check_build(len(puts_per_session) + len(refused), refused, empty, replies)
        save_store(directory / STORE_FILE, store, written, self._identity())
        sizes = [
            len(json.dumps(item.value, sort_keys=True, ensure_ascii=False))
            for items in store._data.values()
            for item in items.values()
        ]
        skipped = [p for p in problems if p.startswith("Could not")]
        dropped = [p for p in problems if not p.startswith("Could not")]
        return {
            "memories": len(sizes),
            "updates": updates,
            "puts_per_session": puts_per_session,
            "empty_sessions": empty,
            "unchanged_sessions": unchanged,
            "refused_sessions": refused,
            "skipped_edits": len(skipped),  # trustcall 没能应用而跳过的修改
            "dropped_calls": len(dropped),  # trustcall 校验不过而丢掉的工具调用等
            "problem_examples": [*skipped[:2], *dropped[:2]],
            "longest_memory_chars": max(sizes, default=0),
            "memories_over_embed_limit": sum(
                size > self.llm.settings.embedding.max_chars for size in sizes
            ),
            "versions": package_versions(),
            **self._identity(),
        }

    def _identity(self) -> dict[str, Any]:
        """写入用的模型：记进 store.json 与 build.json，复用前核对。评测空间的目录不分预设，
        换了模型（如从 deepseek 预设换到通义）不能悄悄沿用别的模型写出的记忆。"""
        return {
            "extractor": self.llm.settings.roles[self.gateway.role].model,
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
        }

    # ---------- 检索 ----------

    def retrieve(
        self, key: str, directory: Path, sample: Sample, sessions: Sequence[EvalSession]
    ) -> tuple[list[Hit], dict[str, Any]]:
        store, written = self._store(key, directory)
        found = store.search(NAMESPACE, query=sample.question, limit=TOP_K)
        latest = max((s.started_at for s in sessions), default=sample.question_time)
        hits = [
            to_hit(key, item.key, item.value, written.get(item.key), rank, item.score, latest)
            for rank, item in enumerate(found, start=1)
        ]
        kept, tokens = pack(hits, BUDGET_TOKENS)
        return kept, {"langmem_found": len(found), "evidence_tokens": tokens}

    def _store(self, key: str, directory: Path) -> tuple[Any, dict[str, dict[str, str]]]:
        """装回（或复用）一份对话历史的存储；检索时的向量请求记在 search 桶里（按查询向量化）。"""
        with self._lock:
            if key in self._open:
                self._open.move_to_end(key)
                return self._open[key]
        url = self.gateway.url(self.bucket(key, "search"))
        opened = load_store(directory / STORE_FILE, _embeddings(url), self._identity())
        with self._lock:
            self._open[key] = opened
            while len(self._open) > MAX_OPEN:
                self._open.popitem(last=False)  # 只是内存里的字典，正在用的线程手里还有引用
        return opened

    def close(self, key: str) -> None:
        with self._lock:
            self._open.pop(key, None)


def _chat_model(url: str, http: Any) -> Any:
    from langchain_openai import ChatOpenAI  # type: ignore[import-not-found, unused-ignore]

    # 温度不设：网关用 extractor 角色的默认值。网关已在内部重试与换备用模型，这里只重试一次
    return ChatOpenAI(
        model=CHAT_MODEL,
        base_url=url,
        api_key="unused",
        max_retries=1,
        use_responses_api=False,
        http_async_client=http,
    )


def _embeddings(url: str, http: Any = None) -> Any:
    from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found, unused-ignore]

    # 网关只收字符串：不按 tiktoken 切成 token id
    return OpenAIEmbeddings(  # type: ignore[call-arg, unused-ignore]
        model=EMBED_MODEL,
        base_url=url,
        api_key="unused",
        check_embedding_ctx_length=False,
        max_retries=1,
        http_async_client=http,
    )


# ---------- 纯函数（不依赖 langmem，单独测试） ----------


def session_messages(session: EvalSession) -> list[dict[str, str]]:
    """一个会话交给管理器的消息（OpenAI 格式，文档里的写法）。每句前面写上会话时间（Mem0 评测
    代码的格式），两人对话再写上说话人；角色照原样（LoCoMo 两人都是 user，LangMem 会把它们
    合成一段对话记录）。"""
    stamp = f"{session.started_at:%Y-%m-%d (%a) %H:%M}"
    messages = []
    for turn in session.turns:
        if not turn.text.strip():
            continue
        who = f"{turn.speaker}: " if turn.speaker else ""
        messages.append({"role": turn.role, "content": f"{stamp} | {who}{turn.text}"})
    return messages


def record_writes(
    written: dict[str, dict[str, str]], puts: Sequence[Mapping[str, Any]], session: EvalSession
) -> int:
    """按管理器返回的写入（新增与修改）记下每条记忆第一次与最后一次写入的会话，返回修改的条数。

    删除默认关闭；即使打开，删掉的记忆也不会出现在检索结果里，留下的记录用不到。"""
    when = session.started_at.isoformat()
    updates = 0
    for put in puts:
        entry = written.get(str(put["key"]))
        if entry is None:
            written[str(put["key"])] = {
                "first_session": session.session_id,
                "first_time": when,
                "session": session.session_id,
                "time": when,
            }
        else:
            entry.update(session=session.session_id, time=when)
            updates += 1
    return updates


def chose_no_change(
    start: Mapping[str, Any], end: Mapping[str, Any], problems: Sequence[str]
) -> bool:
    """一个会话一条也没写时，是不是模型自己说"不用改"：这个会话期间（网关桶在 start 与 end 时的
    统计）模型调用了工具，没有不调用工具或被截断的回复，trustcall 也没报告跳过或丢掉的修改
    （problems）。这样的会话只剩一种可能：模型调用 PatchDoc 却没给补丁（patches 为空），trustcall
    照规矩不写。其余（没调用工具、补丁改不了、调用校验不过……）是写入丢了。"""
    return (
        end["tool_calls"] > start["tool_calls"]
        and end["no_tool_replies"] == start["no_tool_replies"]
        and end["truncated_replies"] == start["truncated_replies"]
        and not problems
    )


def check_build(
    sessions: int, refused: Sequence[str], empty: Sequence[str], replies: Mapping[str, int]
) -> None:
    """整份写入能不能用；不能用时报错（不留 build.json，下次运行重建）。

    sessions：交给管理器的会话数；empty：写入丢了的会话（模型明说不用改的不算，见
    chose_no_change）；replies：这份写入在网关桶里的 tool_requests 与 multi_tool_replies。"""
    allowed = max(MAX_LOST, sessions // 10)
    if len(refused) + len(empty) > allowed:
        raise RuntimeError(
            f"LangMem 有 {len(refused)} 个会话被内容审核拒绝、{len(empty)} 个会话的写入丢了"
            f"（共 {sessions} 个，最多容许 {allowed} 个）：{[*refused, *empty][:3]}"
        )
    if replies.get("tool_requests", 0) >= MIN_TOOL_REPLIES and not replies.get(
        "multi_tool_replies", 0
    ):
        raise RuntimeError(
            f"LangMem 的 {replies['tool_requests']} 次抽取都只回了一个工具调用：并行工具调用"
            "没有生效，每个会话只能记一条（检查网关是否发出 parallel_tool_calls）"
        )


def same_identity(record: Path, identity: Mapping[str, Any]) -> bool:
    """build.json 里记下的抽取模型与向量器是否与当前相同（旧记录缺字段时算不同）。"""
    try:
        built = json.loads(record.read_text("utf-8"))
    except (OSError, ValueError):
        return False
    return all(built.get(field) == value for field, value in identity.items())


def package_versions() -> dict[str, str]:
    """写入时实际用的版本（命令里钉住了，这里再记一份）。"""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def memory_text(value: Mapping[str, Any]) -> str:
    """存储里的值 → 记忆文本。默认结构是 {"kind": "Memory", "content": {"content": "..."}}。"""
    content = value.get("content", value)
    if isinstance(content, Mapping) and isinstance(content.get("content"), str):
        return str(content["content"])
    return json.dumps(content, ensure_ascii=False)


def to_hit(
    space: str,
    key: str,
    value: Mapping[str, Any],
    entry: Mapping[str, str] | None,
    rank: int,
    score: float | None,
    fallback: datetime,
) -> Hit:
    """一条检索结果 → 回答模型看到的证据：时间取最后一次写入它的会话（与第一次不同时注明）。"""
    text = memory_text(value)
    if entry is None:  # 不应发生：每条记忆都是管理器返回过的写入
        return memory_hit(space, key, text, fallback, rank, score=score or 0.0, route="langmem")
    when = datetime.fromisoformat(entry["time"])
    first = datetime.fromisoformat(entry["first_time"])
    if first.date() != when.date():
        text = f"{text} (first recorded {first:%Y-%m-%d}, last updated {when:%Y-%m-%d})"
    return memory_hit(
        space,
        key,
        text,
        when,
        rank,
        score=score or 0.0,
        route="langmem",
        session_id=entry["session"],
    )


def pack(hits: Sequence[Hit], budget: int) -> tuple[list[Hit], int]:
    """按名次装进预算（至少一条），返回（装下的证据，估计的 tokens）。与 L1 装轮的规则相同：
    放不下的跳过，后面更短的可能还放得下（LangMem 合并出的记忆有的上千 tokens）。"""
    kept: list[Hit] = []
    used = 0
    for hit in hits:
        cost = estimate_tokens(hit.unit.text)
        if kept and used + cost > budget:
            continue
        kept.append(hit)
        used += cost
    return kept, used


def snapshot(
    data: Mapping[tuple[str, ...], Mapping[str, Any]],
    vectors: Mapping[tuple[str, ...], Mapping[str, Mapping[str, list[float]]]],
    written: Mapping[str, Mapping[str, str]],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """InMemoryStore 的内容（_data 与 _vectors）→ 可以存成 JSON 的字典（带上写入用的模型）。"""
    items = []
    for namespace, entries in data.items():
        for key, item in entries.items():
            items.append(
                {
                    "namespace": list(namespace),
                    "key": key,
                    "value": item.value,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat(),
                    "vectors": dict(vectors.get(namespace, {}).get(key, {})),
                    "written": dict(written.get(key, {})),
                }
            )
    return {"identity": dict(identity), "items": items}


def save_store(
    path: Path, store: Any, written: Mapping[str, Mapping[str, str]], identity: Mapping[str, Any]
) -> None:
    payload = snapshot(store._data, store._vectors, written, identity)
    path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")


def load_store(
    path: Path, embeddings: Any, identity: Mapping[str, Any]
) -> tuple[Any, dict[str, dict[str, str]]]:
    """store.json → InMemoryStore（向量原样装回）与每条记忆的写入会话。

    写入用的模型与当前不同时拒绝装回（先核对，再导入 langgraph）。正常走不到这里：
    _ensure_built 已按 build.json 核对过，不同就重建。"""
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("identity") != dict(identity):
        raise RuntimeError(
            f"{path.parent} 是用 {payload.get('identity')} 写入的，当前是 {dict(identity)}："
            "换了模型就删掉这个目录重建"
        )
    from langgraph.store.base import Item  # type: ignore[import-not-found, unused-ignore]
    from langgraph.store.memory import InMemoryStore  # type: ignore[import-not-found, unused-ignore]

    store = InMemoryStore(index={"dims": identity["dim"], "embed": embeddings})
    written: dict[str, dict[str, str]] = {}
    for row in payload["items"]:
        namespace = tuple(row["namespace"])
        store._data[namespace][row["key"]] = Item(
            value=row["value"],
            key=row["key"],
            namespace=namespace,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
        store._vectors[namespace][row["key"]] = dict(row["vectors"])
        if row["written"]:
            written[row["key"]] = dict(row["written"])
    return store, written
