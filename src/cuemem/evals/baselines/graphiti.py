"""外部基线 MG：Graphiti（getzep/graphiti 0.30.2，Zep 的开源核心），N12。

照 Zep 论文（arXiv 2501.13956）和它公开的评测代码（getzep/zep-papers 的
kg_architecture_agent_memory：locomo_eval/ 与 zep_longmem_eval.ipynb）写入与检索；回答与判分
和其他方法完全相同（同一个回答模型、同一套提示），只换"记忆从哪来"：

- **写入**：每条发言一个 episode（`add_episode`，source=message），正文是"说话人: 内容"
  （LoCoMo 用说话人的名字，LongMemEval 用 user / assistant），reference_time 取会话时间；每份
  对话历史一个 group_id、一个图。两人对话不分视角：Zep 的 LoCoMo 评测就是把整段对话写进同一个
  图（Mem0 则按说话人各写一遍）。Graphiti 要求 episode 一条一条按顺序写（每条都要看到前 10 条
  消息和已有的图），所以一份对话历史的写入是串行的；每条 episode 有实体抽取、实体消歧、事实
  （边）抽取、事实去重与失效判断、实体摘要等多次 LLM 调用。
- **检索**：与 Zep 的评测相同，取 20 条事实（边）和 20 个实体（节点）。事实用 Graphiti 的
  cross_encoder 配方（BM25 + 向量 + BFS 召回，RRF 融合后的前 40 条交给重排模型，取前 20），
  实体用 RRF（BM25 + 向量）。事实的日期取事件时间（valid_at；没有就取说这句话的时间），正文后
  附有效期（"Date range: 起 - 止"，Zep 的 LongMemEval 模板）；实体给"名字: 摘要"，日期取最近
  一次提到它的发言。试点里每题回答的输入约 3.4k tokens（摘要每个最多 1000 字符），与 CueMem
  L1（约 4k）、M0（每人 30 条）相当。
- **同样的模型**：Graphiti 的 LLM 客户端（OpenAIGenericClient）与向量客户端（OpenAIEmbedder）
  指向评测网关：聊天交给 extractor 角色的模型，向量交给 CueMem 的向量器（维度照它的来）；
  重排用 CueMem 自己的重排模型（默认配置是 qwen3-rerank，与 CueMem L2 相同）。

与 Graphiti 默认用法、与 Zep 评测的差异（及原因）：

1. 重排模型：Graphiti 自带的两个 cross_encoder 都用不了。OpenAIRerankerClient 靠 logprobs，
   还用 OpenAI 分词器的 token ID 设 logit_bias，评测网关、DeepSeek、通义都不支持；
   BGERerankerClient 要装 torch 并下载 2GB 的 bge-reranker-v2-m3。这里把 CueMem 的重排模型
   包成 Graphiti 的 CrossEncoderClient，对**原问题**打分（Graphiti 会把检索用的查询串传给
   重排，这里那是去掉标点的 BM25 查询，见第 9 条）。没有重排模型的配置（deepseek 预设没有）
   退回 Graphiti 基础 `search()` 的默认做法：事实与实体都是 BM25 + 向量的 RRF（Graphiti 的
   RRF 配方不带 BFS）。每题的 details 里 edge_reranker 记下用的是哪一种。
2. BFS 深度 1（Graphiti 默认 3）：Kuzu 把每条事实存成一个中间节点，BFS 查询按可变长路径
   枚举所有走法。试点那段 LoCoMo 对话（164 个实体、493 条事实）上，深度 3 占用超过 12GB 内存、
   进程被系统杀掉；深度 2 用光 1GB 缓冲池。深度 1 取初检结果里的实体直接相连的事实，约 0.1 秒。
3. 图数据库：Graphiti 默认用 Neo4j，Zep 的评测用 Zep Cloud。这里用 0.30.2 自带的嵌入式 Kuzu
   （graphiti-core[kuzu]，不用另起服务），每份对话历史一个库。Kuzu 后端在 0.30.2 已标为弃用
   （上游 Kuzu 不再维护），有几处要补：
   - `Graphiti.build_indices_and_constraints` 对 Kuzu 什么也不做，全文索引建不起来，BM25 检索
     直接报错。这里在新建库时调用驱动自己的 `graph_ops.build_indices_and_constraints` 建一次；
   - KuzuDriver 没有 `_database` 属性，带 group_id 写入时报 AttributeError。这里把库名设成这份
     对话历史的 group_id；
   - kuzu.Database 默认每个库预留 8TB 地址空间，同时开十几个就打不开新库；缓冲池默认可用
     80% 的内存，一个失控的查询能拖垮整个评测进程。这里每个库限定 64GB 地址空间、1GB 缓冲池。
4. 结构化输出保持 OpenAIGenericClient 默认的 json_schema，由网关改成 json_object、把 schema
   写进系统提示（DeepSeek 与通义都不保证支持 json_schema）。没有照它文档对 DeepSeek 的建议改用
   json_object 模式：那样 schema 以"Respond with a JSON object in the following format"附在
   用户消息末尾，试点里 deepseek-flash 常把 schema 本身原样回过来（LongMemEval 前 30 条发言里
   3 条因此整条写入失败）；网关的写法 30 条全部成功。
5. 解析失败时重新采样：回复解析不了时，Graphiti 原样重发（最多 4 次，温度 1，本意是重新采样），
   网关的响应缓存却只会还回同一个坏回复。这里重试时把 max_tokens 减 1，请求落到新的缓存键上，
   真正重新采样（回复都很短，上限差一个 token 不影响结果）；第一次请求不变。回复是 JSON 却
   不合 schema 时 Graphiti 不重试（`Model(**回复)` 报错，整条 episode 失败），而通义的 JSON 模式
   不保证 schema：通义预设试点的 conv-42 前 125 条发言里 4 条因此失败（回了裸列表 `[]`、
   `[{...}]`，或在整数列表里放 null），超过第 11 条的 1% 上限、整份作废。这里按 Graphiti 自己
   解析的规则校验（check_reply），不合就当作解析失败、同样重新采样；合 schema 的回复照旧。
6. 发言时间：同一会话里每条发言的时间都是会话时间，而 Graphiti 取"前 10 条消息"时只按时间
   排序，同一时刻的发言先后不定。这里给会话里第 i 条发言加 i 秒，并让整份历史的时间严格递增
   （LongMemEval 约五分之一的历史里有开始时间相同的会话，不然两段会话的发言会交错进上下文），
   日期不变。数据集里的时间当作 UTC（Zep 的评测也是这样）。
7. 单条发言截到 8000 字符（Zep 的 LongMemEval 评测也这样截）；问题原样作为查询，不截到
   255 字符（那是 Zep Cloud 接口的限制），也不在前面加日期。
8. 回答：Zep 用自己的 FACTS / ENTITIES 模板和回答提示（gpt-4o-mini）；这里每条事实、每个实体
   各是一条证据，交给同一个回答模型、同一套提示。实体证据不标会话：摘要是跨会话累积的，Zep 的
   模板也不给。图片只有描述文字（数据集已并进正文）。
9. 查询：BM25 用去掉标点的问题（Graphiti 对 Neo4j 会先清理查询，对 Kuzu 原样传给全文索引，
   而 Kuzu 不去掉问号，"live?" 匹配不到 "lives"，带问号的问题常常一条也查不到）；查询向量用
   CueMem 向量器的 query 模式（与 CueMem、M0 相同：通义的向量区分查询与文档，Graphiti 原本
   用的 OpenAI 向量不分）。
10. 请求网关的超时放宽到 5 小时：评测遇到分时价格的高峰时段会暂停到高峰结束
    （peak_policy=wait），Graphiti 的请求要能等下去，否则超时重试几次后整条 episode 失败。
11. 写入失败的处理：重试后仍失败的发言跳过并记下；超过 1% 的发言失败（或连续 10 条失败）时
    整份作废，不留写入记录，下次运行重建（成功过的调用都在缓存里，重建几乎不花钱）。

写入记录（build.json）记下写入协议、抽取模型与向量模型，任何一项变了就自动重建；Graphiti 自己
的告警（抽取的实体名对不上而丢掉的事实、消歧编号越界、JSON 解析失败……）按类型计数记在
warnings 里。每题的 details 记下各路召回的候选数（candidates：BM25 与向量各找到几条），哪一路
整个失效（比如向量相似度都低于 Graphiti 的阈值 0.6）一眼看得出。检索指标按"轮"计：每条事实
标上它来源发言所在的轮，排在前面的事实已覆盖的轮不重复计；实体证据不计入指标。

试点实测（deepseek 预设，全部 deepseek-flash、本机 bge-small 向量，错峰时段）：LoCoMo conv-42
（629 条发言）写入 36 分钟、2014 次调用、428 万输入 tokens，标价 $1.42、实际 $0.56，没有失败的
发言；每题证据约 3.4k tokens。按写入协议 2 重建（runs/eval/pilot-graphiti-2）时 2014 次调用里
2012 次命中缓存（Graphiti 的提示里没有随机 ID，同样的输入得到同样的图：164 个实体、493 条事实），
111 秒、实际 $0.0003；Graphiti 告警 7 次，都是抽取的实体名对不上（7 条事实因此丢掉）。deepseek
预设没有重排模型，事实检索走 RRF；bge-small 的向量路每题只有 7–36 条事实候选过 0.6 的阈值
（BM25 路满 40 条）。重排与 BFS 这条路径这次没有联网跑（没有通义的 Key），由离线测试覆盖；BFS
深度 1 在 conv-42 的图上离线实测每题约 0.1 秒。

LongMemEval_S 的发言长得多（平均约 1000 字符，LoCoMo 约 120），前 30 条每条约 10 次调用、
2.2 万输入 tokens（标价约 $0.008 / 条），test100r 约 4.9 万条发言，写入要几百美元、一天以上，
超出下面的写入预算，跑之前先调高 build_budget。另外，LongMemEval 的证据会话 ID 形如
answer_xxx，回答提示的证据头会印出会话 ID（所有方法共有的问题），跑 LongMemEval 之前先在
公共代码里把会话 ID 匿名化。

graphiti-core 不是 CueMem 的依赖，评测时用 uv 临时装上（遥测在导入前关闭）：

    uv run --extra local --with "graphiti-core[kuzu]==0.30.2" cuemem eval run \\
        --dataset locomo --split test200r --methods MG --name locomo-test200r-graphiti

（默认预设：抽取 qwen3.8-flash、通义向量、qwen3-rerank 重排，需要 DASHSCOPE_API_KEY；
--extra local 只有 deepseek 预设的本机向量要用，默认预设加上也无妨。）
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import os
import re
import threading
import warnings
from collections import Counter, OrderedDict
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from cuemem.core.models import Hit
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit
from cuemem.read.rerank import Reranker

TOP_FACTS = 20  # Zep 的 LoCoMo / LongMemEval 评测：事实（边）20 条
TOP_ENTITIES = 20  # 实体（节点）20 个
BFS_DEPTH = 1  # 见模块说明第 2 条
MAX_MESSAGE_CHARS = 8000  # Zep 的 LongMemEval 评测把每条消息截到 8000 字符
SOURCE_DESCRIPTION = "chat message"
GRAPH_FILE = "graph.kuzu"
LEDGER_FILE = "episodes.json"
PROGRESS_FILE = "progress.json"  # 写入一份对话历史要一两个小时：中途的进度写在这里
PROGRESS_EVERY = 25
MAX_OPEN = 8  # 同时打开的图（每份对话历史一个 Kuzu 库）
MAX_FAILURE_STREAK = 10  # 连续这么多条发言写入失败：多半是网关或供应商出了问题，整份放弃
MAX_FAILED_SHARE = 0.01  # 失败的发言超过这个比例：图缺得太多，整份作废（见模块说明第 11 条）
GATEWAY_TIMEOUT = 5 * 3600.0  # 秒：高峰时段网关会暂停到高峰结束（见模块说明第 10 条）
KUZU_MAX_DB_SIZE = 1 << 36  # 每个库的地址空间上限（64GB；默认 8TB，见模块说明第 3 条）
KUZU_BUFFER_POOL = 1 << 30  # 每个库的缓冲池（1GB；默认可用 80% 的内存）
BUILD_PROTOCOL = 2  # 写入方式变了就加一：旧的图自动重建（2：整份历史的时间严格递增、重试换样本）

T = TypeVar("T")

# 正在写入的这份对话历史的告警计数：Graphiti 的协程都在同一个事件循环里跑，各自的上下文分开计
_WARNINGS: contextvars.ContextVar[Counter[str] | None] = contextvars.ContextVar(
    "cuemem_graphiti_warnings", default=None
)
_KUZU_LOCK = threading.Lock()


class GraphitiBaseline(ExternalBaseline):
    name, label, system = "MG", "外部基线：Graphiti（Zep 的开源核心，0.30.2）", "graphiti"
    # 逐条发言写入，LLM 调用比 Mem0 多得多：LoCoMo test200r 的 7 段对话按试点估算标价约 $10
    # （错峰实际约 $4），留出余量；LongMemEval 要调高（见模块说明）
    build_budget = 40.0

    def __init__(self) -> None:
        super().__init__()
        self._graphs: OrderedDict[str, _Graph] = OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._replaced: dict[str, dict[str, Any]] = {}
        self.reranker: Reranker | None = None

    # ---------- 钩子 ----------

    def setup(self) -> None:
        """导入 Graphiti（导入前关掉遥测），启动一个后台事件循环：Graphiti 是异步的，基类的
        build / retrieve 是同步的，协程都交给这个循环跑（HTTP 客户端与库连接都绑在它上面）。"""
        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
        try:
            import graphiti_core  # type: ignore[import-not-found, unused-ignore]  # noqa: F401
            import kuzu  # type: ignore[import-not-found, unused-ignore]  # noqa: F401
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise RuntimeError(
                'MG 需要 graphiti-core[kuzu]：用 uv run --with "graphiti-core[kuzu]==0.30.2" ...'
                "（见模块说明）"
            ) from exc
        from cuemem.memory import build_reranker

        # 没有重排模型的 Key 时为 None，事实检索退回 RRF（见模块说明第 1 条）；
        # 与评测运行器一样，和 LLM 客户端共用传输层（测试里是模拟接口）
        self.reranker = build_reranker(
            self.llm.settings,
            llm=self.llm,
            ledger=self.llm.ledger,
            http_client=self.llm._http_client,
        )
        # Graphiti 的告警每条 episode 都可能有：不刷屏，按类型计数记进写入记录
        log = logging.getLogger("graphiti_core")
        log.setLevel(logging.WARNING)
        log.propagate = False
        if not any(isinstance(handler, WarningCounter) for handler in log.handlers):
            log.addHandler(WarningCounter())
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, name="cuemem-graphiti", daemon=True).start()
        self._loop = loop

    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        graph = self._graph(key, directory)
        builder, client = self._graphiti(graph.driver, self.bucket(key))
        try:
            stats = self._run(self._ingest(key, directory, builder, graph, sessions))
        finally:
            self._run(client.close())
        replaced = self._replaced.pop(key, None)
        return {**self._build_id(), **stats, **({"replaced": replaced} if replaced else {})}

    def retrieve(
        self, key: str, directory: Path, sample: Sample, sessions: Sequence[EvalSession]
    ) -> tuple[list[Hit], dict[str, Any]]:
        graph = self._graph(key, directory)
        # 查询向量在本线程算（query 模式，见模块说明第 9 条），用量记进这道题的 trace
        question = sample.question.replace("\n", " ")
        vector = self.embedder.embed([question], "query").dense[0].tolist()
        encoder = _cross_encoder_type()(self.reranker, question) if self.reranker else None
        bucket = self.bucket(key, "search")
        before = self.gateway.usage(bucket)
        found, candidates = self._run(self._search(graph, question, vector, group_id(key), encoder))
        after = self.gateway.usage(bucket)
        facts = [
            fact_hit(key, edge, score, rank, graph.ledger)
            for rank, (edge, score) in enumerate(
                zip(found.edges, _scores(found.edge_reranker_scores, found.edges), strict=True),
                start=1,
            )
        ]
        mark_rounds(facts, sessions)
        entities = [
            entity_hit(key, node, score, rank, graph.ledger)
            for rank, (node, score) in enumerate(
                zip(found.nodes, _scores(found.node_reranker_scores, found.nodes), strict=True),
                start=1,
            )
        ]
        details = {
            "facts": len(facts),
            "entities": len(entities),
            "edge_reranker": (
                f"cross_encoder:{self.reranker.cfg.model}" if self.reranker else "rrf"
            ),
            "candidates": candidates,
            # 检索时 Graphiti 不该再经网关调用模型：记下来核对（应当都是 0）
            "search_usage": {
                field: round(after[field] - before[field], 6)
                for field in ("embedding_calls", "chat_calls", "spent")
            },
        }
        return facts + entities, details

    def close(self, key: str) -> None:
        with self._lock:
            self._close_locked(key)

    def _ensure_built(
        self, key: str, directory: Path, sessions: Sequence[EvalSession]
    ) -> dict[str, Any] | None:
        """换了写入协议、抽取模型或向量模型的图不能复用：删掉写入记录，交给基类重建。"""
        record = directory / "build.json"
        if record.exists():
            built = json.loads(record.read_text("utf-8"))
            stale = {
                field: built.get(field)
                for field, value in self._build_id().items()
                if built.get(field) != value
            }
            if stale:
                self._replaced[key] = stale
                record.unlink()
        return super()._ensure_built(key, directory, sessions)

    def _build_id(self) -> dict[str, Any]:
        return {
            "graphiti_protocol": BUILD_PROTOCOL,
            "extractor": self.llm.settings.roles["extractor"].model,
            "embedder": f"{self.embedder.name}:{self.embedder.dim}",
        }

    # ---------- 写入 ----------

    async def _ingest(
        self,
        key: str,
        directory: Path,
        graphiti: Any,
        graph: _Graph,
        sessions: Sequence[EvalSession],
    ) -> dict[str, Any]:
        from graphiti_core.nodes import EpisodeType  # type: ignore[import-not-found, unused-ignore]

        counts: Counter[str] = Counter()
        _WARNINGS.set(counts)  # 这个协程（及它派生的任务）里的告警记到这份历史名下
        group = group_id(key)
        todo = [
            (session, index, body)
            for session in sorted(sessions, key=lambda s: s.started_at)
            for index, turn in enumerate(session.turns)
            if (body := episode_body(turn)) is not None
        ]
        failed: list[str] = []
        streak = 0
        previous: datetime | None = None
        for done, (session, index, body) in enumerate(todo, start=1):
            when = previous = reference_time(session.started_at, index, after=previous)
            name = f"{session.session_id}#{index}"
            try:
                result = await graphiti.add_episode(
                    name=name,
                    episode_body=body,
                    source_description=SOURCE_DESCRIPTION,
                    reference_time=when,
                    source=EpisodeType.message,
                    group_id=group,
                )
            except Exception as exc:  # 重试后仍失败的发言：跳过并记下，别的照常
                failed.append(f"{name}: {type(exc).__name__}: {exc}"[:200])
                streak += 1
                if streak >= MAX_FAILURE_STREAK:
                    raise RuntimeError(f"连续 {streak} 条发言写入失败：{failed[-1]}") from exc
                continue
            streak = 0
            graph.ledger.add(
                result.episode.uuid,
                session.session_id,
                index,
                when,
                [node.uuid for node in result.nodes],
            )
            if done % PROGRESS_EVERY == 0:
                self._progress(key, directory, done, len(todo), len(failed), counts)
        if len(failed) > MAX_FAILED_SHARE * len(todo):
            raise RuntimeError(
                f"{len(failed)} / {len(todo)} 条发言写入失败（上限 {MAX_FAILED_SHARE:.0%}）："
                f"{failed[-1]}"
            )
        graph.ledger.save()
        return {
            "episodes": len(graph.ledger.episodes),
            "failed_episodes": failed,
            "warnings": dict(counts.most_common()),
            **await graph_counts(graph.driver, group),
        }

    def _progress(
        self, key: str, directory: Path, done: int, total: int, failed: int, counts: Counter[str]
    ) -> None:
        usage = self.gateway.usage(self.bucket(key))
        progress = {
            "episodes": done,
            "of": total,
            "failed": failed,
            "warnings": sum(counts.values()),
            "chat_calls": usage["chat_calls"],
            "spent": usage["spent"],
            "at": datetime.now().isoformat(timespec="seconds"),
        }
        (directory / PROGRESS_FILE).write_text(json.dumps(progress), "utf-8")

    # ---------- 检索 ----------

    @staticmethod
    async def _search(
        graph: _Graph, question: str, vector: list[float], group: str, encoder: Any
    ) -> tuple[Any, dict[str, int]]:
        """事实与实体一次检索（见模块说明第 1、2、9 条）；另外数一数各路召回的候选。"""
        from graphiti_core.search.search import (  # type: ignore[import-not-found, unused-ignore]
            search,
        )
        from graphiti_core.search.search_config import (  # type: ignore[import-not-found, unused-ignore]
            SearchConfig,
        )
        from graphiti_core.search.search_config_recipes import (  # type: ignore[import-not-found, unused-ignore]
            EDGE_HYBRID_SEARCH_CROSS_ENCODER,
            EDGE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_RRF,
        )
        from graphiti_core.search.search_filters import (  # type: ignore[import-not-found, unused-ignore]
            SearchFilters,
        )

        clients = graph.searcher.clients
        if encoder is not None:
            edge_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.edge_config.model_copy(
                update={"bfs_max_depth": BFS_DEPTH}
            )
            clients = clients.model_copy(update={"cross_encoder": encoder})
        else:
            edge_config = EDGE_HYBRID_SEARCH_RRF.edge_config
        node_config = NODE_HYBRID_SEARCH_RRF.node_config
        config = SearchConfig(
            edge_config=edge_config, node_config=node_config, limit=max(TOP_FACTS, TOP_ENTITIES)
        )
        query = keyword_query(question)
        found = await search(clients, query, [group], config, SearchFilters(), query_vector=vector)
        found.edges, found.edge_reranker_scores = (
            found.edges[:TOP_FACTS],
            found.edge_reranker_scores[:TOP_FACTS],
        )
        found.nodes, found.node_reranker_scores = (
            found.nodes[:TOP_ENTITIES],
            found.node_reranker_scores[:TOP_ENTITIES],
        )
        candidates = await channel_counts(
            graph.driver, query, vector, group, edge_config, node_config, 2 * config.limit
        )
        return found, candidates

    # ---------- 图与客户端 ----------

    def _run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._loop is None:
            coroutine.close()
            raise RuntimeError("Graphiti 的事件循环还没有启动")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def _graphiti(self, driver: Any, bucket: str) -> tuple[Any, Any]:
        """一个 Graphiti 实例：LLM、向量都经网关的这个桶；返回（实例，要关闭的 HTTP 客户端）。"""
        from graphiti_core import Graphiti  # type: ignore[import-not-found, unused-ignore]
        from graphiti_core.embedder.openai import (  # type: ignore[import-not-found, unused-ignore]
            OpenAIEmbedder,
            OpenAIEmbedderConfig,
        )
        from graphiti_core.llm_client.config import (  # type: ignore[import-not-found, unused-ignore]
            LLMConfig,
        )
        from openai import AsyncOpenAI

        url = self.gateway.url(bucket)
        # 网关背后的 LLMClient 自己会重试上游错误；这里只为本机连接的偶发问题重试一次
        client = AsyncOpenAI(
            base_url=url, api_key="cuemem-gateway", timeout=GATEWAY_TIMEOUT, max_retries=1
        )
        model = self.llm.settings.roles["extractor"].model  # 只是记在网关的账上，不改变路由
        config = LLMConfig(api_key="cuemem-gateway", base_url=url, model=model, small_model=model)
        # 结构化输出保持默认的 json_schema（见模块说明第 4 条）；重试时重新采样（第 5 条）
        llm = _resampling_client_type()(config, client=client)
        embedder = OpenAIEmbedder(
            OpenAIEmbedderConfig(
                embedding_model=self.embedder.name,
                embedding_dim=self.embedder.dim,
                api_key="cuemem-gateway",
                base_url=url,
            ),
            client=client,
        )
        # 必须给一个重排器，否则 Graphiti 会去读 OPENAI_API_KEY；写入用不到它，检索时每题
        # 换成绑定了原问题的那个（见 _search）
        placeholder = _cross_encoder_type()(None, "")
        graphiti = Graphiti(
            graph_driver=driver, llm_client=llm, embedder=embedder, cross_encoder=placeholder
        )
        return graphiti, client

    def _graph(self, key: str, directory: Path) -> _Graph:
        """打开（或复用）一份对话历史的图；最多同时开 MAX_OPEN 个，多了关掉最久没用的。"""
        with self._lock:
            if key in self._graphs:
                self._graphs.move_to_end(key)
                return self._graphs[key]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / GRAPH_FILE
        fresh = not path.exists()
        driver = open_kuzu(path)
        client = None
        try:
            # 0.30.2 的 KuzuDriver 没有 _database，带 group_id 写入时直接报 AttributeError；
            # 一份对话历史一个库，库名就是它的 group_id
            driver._database = group_id(key)
            if fresh:  # 全文索引只能建一次，重复建会报错
                self._run(driver.graph_ops.build_indices_and_constraints(driver))
            searcher, client = self._graphiti(driver, self.bucket(key, "search"))
            graph = _Graph(driver, searcher, client, EpisodeLedger.load(directory / LEDGER_FILE))
        except BaseException:  # 打开到一半失败：别把库文件一直占着
            if client is not None:
                self._run(client.close())
            close_kuzu(driver)
            raise
        with self._lock:
            self._graphs[key] = graph
            for old in list(self._graphs)[: max(0, len(self._graphs) - MAX_OPEN)]:
                if not self._space_locks.get(old, threading.Lock()).locked():  # 正在用的留着
                    self._close_locked(old)
        return graph

    def _close_locked(self, key: str) -> None:
        graph = self._graphs.pop(key, None)
        if graph is None:
            return
        self._run(graph.client.close())
        close_kuzu(graph.driver)


@dataclass
class _Graph:
    driver: Any  # KuzuDriver：写入与检索共用（同一进程里一个库文件只能打开一次）
    searcher: Any  # 检索用的 Graphiti（网关的 search 桶）
    client: Any  # searcher 的 HTTP 客户端
    ledger: EpisodeLedger


class EpisodeLedger:
    """episode → （会话，发言下标，时间）；实体 → 最近一次提到它的 episode。

    Graphiti 的事实只记来源 episode 的 ID：靠它把证据标回会话与发言（证据的会话、日期，以及
    检索指标里的发言召回）。写入时记下，存成空间目录里的 JSON。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.episodes: dict[str, tuple[str, int, datetime]] = {}
        self.mentions: dict[str, str] = {}

    @classmethod
    def load(cls, path: Path) -> EpisodeLedger:
        ledger = cls(path)
        if path.exists():
            raw = json.loads(path.read_text("utf-8"))
            ledger.episodes = {
                uuid: (str(session), int(index), datetime.fromisoformat(when))
                for uuid, (session, index, when) in raw["episodes"].items()
            }
            ledger.mentions = dict(raw["mentions"])
        return ledger

    def add(
        self,
        episode: str,
        session_id: str,
        index: int,
        when: datetime,
        nodes: Sequence[str],
    ) -> None:
        self.episodes[episode] = (session_id, index, naive(when))
        for node in nodes:
            self.mentions[node] = episode

    def save(self) -> None:
        raw = {
            "episodes": {
                uuid: [session, index, when.isoformat()]
                for uuid, (session, index, when) in self.episodes.items()
            },
            "mentions": self.mentions,
        }
        self.path.write_text(json.dumps(raw, ensure_ascii=False), "utf-8")

    def source(self, episodes: Sequence[str]) -> tuple[str, list[int], datetime | None]:
        """一条事实的来源：最早那条 episode 的会话、该会话里提到它的发言下标、说这句话的时间。"""
        known = sorted((self.episodes[e] for e in episodes if e in self.episodes), key=_by_time)
        if not known:
            return "", [], None
        session, _, when = known[0]
        return session, sorted({i for s, i, _ in known if s == session}), when

    def last_mention(self, node: str) -> datetime | None:
        episode = self.episodes.get(self.mentions.get(node, ""))
        return episode[2] if episode else None


def _by_time(entry: tuple[str, int, datetime]) -> datetime:
    return entry[2]


class WarningCounter(logging.Handler):
    """Graphiti 自己的告警与错误按类型计数（记到当前正在写入的那份对话历史名下）。"""

    def emit(self, record: logging.LogRecord) -> None:
        counts = _WARNINGS.get()
        if counts is not None:
            counts[warning_key(record)] += 1


# ---------- Graphiti 类的子类（用到时才定义：graphiti-core 不是依赖） ----------


@functools.cache
def _cross_encoder_type() -> type[Any]:
    from graphiti_core.cross_encoder.client import (  # type: ignore[import-not-found, unused-ignore]
        CrossEncoderClient,
    )

    class CueMemCrossEncoder(CrossEncoderClient):  # type: ignore[misc, unused-ignore]
        """CueMem 的重排模型 → Graphiti 的 cross-encoder：对原问题打分（见模块说明第 1 条）。
        打分在线程池里跑，带着构造时（这道题的线程里）的上下文：重排的用量记进这道题的 trace。"""

        def __init__(self, reranker: Reranker | None, question: str) -> None:
            self.reranker, self.question = reranker, question
            self.context = contextvars.copy_context()

        async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
            if self.reranker is None:
                raise RuntimeError("没有重排模型：这次检索不该用到 cross_encoder")
            scores = await asyncio.get_running_loop().run_in_executor(
                None, self.context.run, self.reranker.scores, self.question, passages
            )
            return ranked(passages, scores)

    return CueMemCrossEncoder


@functools.cache
def _resampling_client_type() -> type[Any]:
    from graphiti_core.llm_client.config import (  # type: ignore[import-not-found, unused-ignore]
        DEFAULT_MAX_TOKENS,
    )
    from graphiti_core.llm_client.openai_generic_client import (  # type: ignore[import-not-found, unused-ignore]
        OpenAIGenericClient,
    )

    class ResamplingClient(OpenAIGenericClient):  # type: ignore[misc, unused-ignore]
        """重试时 max_tokens 减 1，绕开网关缓存、真正重新采样（见模块说明第 5 条）。

        Graphiti 的重试（tenacity）每次都把同一个 messages 列表传给 _generate_response：
        按列表的 id 数第几次尝试，整个请求结束时清掉。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.attempts: dict[int, int] = {}

        async def generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            messages = args[0] if args else kwargs.get("messages")
            try:
                return await super().generate_response(*args, **kwargs)
            finally:
                self.attempts.pop(id(messages), None)

        async def _generate_response(
            self,
            messages: list[Any],
            response_model: Any = None,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            *args: Any,
            **kwargs: Any,
        ) -> dict[str, Any]:
            attempt = self.attempts.get(id(messages), 0)
            self.attempts[id(messages)] = attempt + 1
            reply: dict[str, Any] = await super()._generate_response(
                messages, response_model, max_tokens - attempt, *args, **kwargs
            )
            if response_model is not None:  # 不合 schema 也重新采样（见模块说明第 5 条）
                check_reply(reply, response_model)
            return reply

    return ResamplingClient


def open_kuzu(path: Path) -> Any:
    """打开（或新建）一个 Kuzu 库，限定地址空间与缓冲池（见模块说明第 3 条）。KuzuDriver 用默认
    参数建 kuzu.Database，没有别的入口：构造时临时换掉 kuzu.Database（加锁，别的线程不受影响）。"""
    import kuzu  # type: ignore[import-not-found, unused-ignore]
    from graphiti_core.driver.kuzu_driver import (  # type: ignore[import-not-found, unused-ignore]
        KuzuDriver,
    )

    with _KUZU_LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # Kuzu 后端已标为弃用
        database = kuzu.Database
        kuzu.Database = functools.partial(
            database, max_db_size=KUZU_MAX_DB_SIZE, buffer_pool_size=KUZU_BUFFER_POOL
        )
        try:
            return KuzuDriver(db=str(path))
        finally:
            kuzu.Database = database


def close_kuzu(driver: Any) -> None:
    """KuzuDriver.close() 什么也不做；要释放库文件（和它的地址空间）得自己关。"""
    driver.client.close()
    driver.db.close()


# ---------- 纯函数（测试直接覆盖） ----------


def group_id(key: str) -> str:
    """Graphiti 的 group_id 只允许字母、数字、- 和 _。"""
    return "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in key)


def keyword_query(question: str) -> str:
    """BM25 用的查询：只留下词。Kuzu 的全文索引不去掉问号，"live?" 匹配不到 "lives"，查询里
    有这样的词时常常整个查不到；Graphiti 对 Neo4j 会先做 lucene_sanitize，对 Kuzu 原样传过去。"""
    return " ".join(re.findall(r"\w+", question)) or question


def episode_body(turn: EvalTurn) -> str | None:
    """一条发言 → episode 正文"说话人: 内容"（Graphiti 的消息格式：冒号前是说话人）。"""
    text = turn.text.strip()
    if not text:
        return None
    return f"{turn.speaker or turn.role}: {text[:MAX_MESSAGE_CHARS]}"


def reference_time(started_at: datetime, index: int, *, after: datetime | None = None) -> datetime:
    """会话时间（当作 UTC）加上发言下标那么多秒，并且晚于前一条发言 after（见模块说明第 6 条）。"""
    base = started_at if started_at.tzinfo else started_at.replace(tzinfo=UTC)
    when = base + timedelta(seconds=index)
    return max(when, after + timedelta(seconds=1)) if after is not None else when


def naive(when: datetime) -> datetime:
    """Graphiti 与 Kuzu 的时间（UTC）→ 不带时区的时间（数据集的时间本来就当作 UTC 写进去）。"""
    return when.astimezone(UTC).replace(tzinfo=None) if when.tzinfo else when


def fact_text(fact: str, valid_at: datetime | None, invalid_at: datetime | None) -> str:
    """Zep 的 LongMemEval 模板：FACT (Date range: from - to)。"""
    start = f"{naive(valid_at):%Y-%m-%d}" if valid_at else "date unknown"
    end = f"{naive(invalid_at):%Y-%m-%d}" if invalid_at else "present"
    return f"{fact} (Date range: {start} - {end})"


def fact_hit(key: str, edge: Any, score: float, rank: int, ledger: EpisodeLedger) -> Hit:
    """Graphiti 的一条事实（边）→ 一条证据：日期取事件时间，没有就取说这句话的时间。"""
    session, turns, said = ledger.source(list(edge.episodes or []))
    when = edge.valid_at or said or edge.reference_time or edge.created_at
    hit = memory_hit(
        key,
        f"fact:{edge.uuid}",
        fact_text(edge.fact, edge.valid_at, edge.invalid_at),
        naive(when),
        rank,
        score=score,
        route="graphiti_fact",
        session_id=session,
    )
    hit.unit.meta["turn_indices"] = turns  # 检索指标用（再经 mark_rounds 换成轮）
    return hit


def entity_hit(key: str, node: Any, score: float, rank: int, ledger: EpisodeLedger) -> Hit:
    """Graphiti 的一个实体 → 一条证据"名字: 摘要"：日期取最近一次提到它的发言（摘要是到那时
    为止的累积）。不标会话（见模块说明第 8 条），也就不算进检索指标。"""
    summary = node.summary.strip() or "(no summary)"
    return memory_hit(
        key,
        f"entity:{node.uuid}",
        f"About {node.name} (entity summary): {summary}",
        naive(ledger.last_mention(node.uuid) or node.created_at),
        rank,
        score=score,
        route="graphiti_entity",
        kind="entity",
    )


class SchemaMismatch(json.JSONDecodeError):
    """回复是合法的 JSON，却不是要的结构。继承 JSONDecodeError：Graphiti 的重试只认它这类错误。"""


def check_reply(reply: Any, model: type[Any]) -> None:
    """回复不合 response_model 时报 SchemaMismatch，让 Graphiti 重试（重新采样）。

    Graphiti 拿到回复后用 `Model(**reply)` 解析，不合就让整条 episode 失败；通义的 JSON 模式
    不保证 schema（试点里约 3% 的 episode 因此失败：回了裸列表 `[]`、`[{...}]`，或在整数列表
    里放 null）。校验与 Graphiti 自己的解析相同，合 schema 的回复照常返回。"""
    if not isinstance(reply, dict):
        name = getattr(model, "__name__", "the schema")
        logging.getLogger("graphiti_core").warning(f"Reply is not a JSON object for {name}")
        raise SchemaMismatch(f"Expected a JSON object for {name}", json.dumps(reply)[:200], 0)
    try:
        model.model_validate(reply)
    except ValueError as exc:  # pydantic 的 ValidationError 是 ValueError
        name = getattr(model, "__name__", "the schema")
        logging.getLogger("graphiti_core").warning(f"Reply does not match {name}")
        raise SchemaMismatch(f"Reply does not match {name}: {exc}"[:500], "", 0) from exc


def ranked(passages: Sequence[str], scores: Sequence[float]) -> list[tuple[str, float]]:
    """Graphiti 的 cross-encoder 约定：（文本，分数），分数从高到低（同分保持原顺序）。"""
    return sorted(zip(passages, scores, strict=True), key=lambda pair: -pair[1])


def round_starts(session: EvalSession) -> list[int]:
    """每条发言所在"轮"的第一条发言下标。与 CueMem 的 build_rounds 相同：用户发言开启新的一轮，
    其后的助手回复并入该轮（LoCoMo 两人都是 user，每条发言各成一轮）。"""
    starts: list[int] = []
    for index, turn in enumerate(session.turns):
        starts.append(index if turn.role == "user" or not starts else starts[-1])
    return starts


def mark_rounds(hits: Sequence[Hit], sessions: Sequence[EvalSession]) -> None:
    """检索指标按"轮"计，与 CueMem 的命中单位、指标里的 relevant_units 一致：每条事实的
    turn_indices 换成它来源发言所在的整轮，排在前面的事实已覆盖的轮不再计。几条事实常出自同一条
    发言，重复计会让发言召回虚高、NDCG 超过 1。"""
    starts = {session.session_id: round_starts(session) for session in sessions}
    covered: set[tuple[str, int]] = set()
    for hit in hits:
        session = hit.unit.session_id
        rounds = starts.get(session, [])
        fresh: set[int] = set()
        for index in hit.unit.meta.get("turn_indices", []):
            start = rounds[index] if 0 <= index < len(rounds) else index
            if (session, start) in covered:
                continue
            covered.add((session, start))
            fresh.update([i for i, s in enumerate(rounds) if s == start] or [index])
        hit.unit.meta["turn_indices"] = sorted(fresh)


def warning_key(record: logging.LogRecord) -> str:
    """告警的类型：%-格式的消息就是模板；拼好的消息（f-string）只取第一行、数字换成 #。"""
    text = re.sub(r"\d+", "#", str(record.msg).strip().split("\n", 1)[0])
    return f"{record.levelname}: {text[:100]}"


def _scores(scores: Sequence[float], items: Sequence[Any]) -> list[float]:
    """重排分数与结果一一对应；对不上时（不同版本的行为差异）记 0。"""
    return list(scores) if len(scores) == len(items) else [0.0] * len(items)


async def channel_counts(
    driver: Any,
    query: str,
    vector: list[float],
    group: str,
    edge_config: Any,
    node_config: Any,
    limit: int,
) -> dict[str, int]:
    """各路召回各自找到几条候选（与 search() 内部同样的取法、同样的相似度阈值）。"""
    from graphiti_core.search.search_filters import (  # type: ignore[import-not-found, unused-ignore]
        SearchFilters,
    )
    from graphiti_core.search.search_utils import (  # type: ignore[import-not-found, unused-ignore]
        edge_fulltext_search,
        edge_similarity_search,
        node_fulltext_search,
        node_similarity_search,
    )

    groups, filters = [group], SearchFilters()
    facts_bm25 = await edge_fulltext_search(driver, query, filters, groups, limit)
    facts_cosine = await edge_similarity_search(
        driver, vector, None, None, filters, groups, limit, edge_config.sim_min_score
    )
    entities_bm25 = await node_fulltext_search(driver, query, filters, groups, limit)
    entities_cosine = await node_similarity_search(
        driver, vector, filters, groups, limit, node_config.sim_min_score
    )
    return {
        "fact_bm25": len(facts_bm25),
        "fact_cosine": len(facts_cosine),
        "entity_bm25": len(entities_bm25),
        "entity_cosine": len(entities_cosine),
    }


async def graph_counts(driver: Any, group: str) -> dict[str, int]:
    """写入后的图有多大：实体、事实、被后来的信息判为失效的事实、episode。"""
    queries = {
        "entities": "MATCH (n:Entity {group_id: $gid}) RETURN count(n) AS n",
        "facts": "MATCH (e:RelatesToNode_ {group_id: $gid}) RETURN count(e) AS n",
        "invalidated_facts": (
            "MATCH (e:RelatesToNode_ {group_id: $gid}) "
            "WHERE e.invalid_at IS NOT NULL OR e.expired_at IS NOT NULL RETURN count(e) AS n"
        ),
        "graph_episodes": "MATCH (e:Episodic {group_id: $gid}) RETURN count(e) AS n",
    }
    out: dict[str, int] = {}
    for name, query in queries.items():
        try:
            records, _, _ = await driver.execute_query(query, gid=group)
        except Exception:  # 只是统计：查不了就不记，别让写好的图作废
            continue
        out[name] = int(records[0]["n"]) if records else 0
    return out
