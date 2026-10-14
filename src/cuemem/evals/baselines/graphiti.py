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
- **检索**：与 Zep 的评测相同，取 20 条事实（边）和 20 个实体（节点）。事实的日期取事件时间
  （valid_at；没有就取说这句话的时间），正文后附有效期（"Date range: 起 - 止"，Zep 的
  LongMemEval 模板）；实体给"名字: 摘要"，日期取最近一次提到它的发言。试点里每题回答的输入约
  3.4k tokens（摘要每个最多 1000 字符），与 CueMem L1（约 4k）、M0（每人 30 条）相当。
- **同样的模型**：Graphiti 的 LLM 客户端（OpenAIGenericClient）与向量客户端（OpenAIEmbedder）
  指向评测网关：聊天交给 extractor 角色的模型，向量交给 CueMem 的向量器（维度照它的来）。

与 Graphiti 默认用法、与 Zep 评测的差异（及原因）：

1. 重排：Zep 的评测对事实用 cross_encoder 重排、对实体用 RRF。Graphiti 的 cross_encoder 实现
   OpenAIRerankerClient 靠 logprobs 给每条候选打分，评测网关不返回 logprobs；另一个实现
   BGERerankerClient 要装 torch 并下载 2GB 的 bge-reranker-v2-m3。所以两边都用 Graphiti 自带的
   RRF 混合检索配方（BM25 + 向量：EDGE_HYBRID_SEARCH_RRF 与 NODE_HYBRID_SEARCH_RRF 的组合），
   也就是 Graphiti 基础 `search()` 的默认做法。事实一侧可能因此比 Zep 的结果偏低。
2. 图数据库：Graphiti 默认 Neo4j，Zep 的评测用 Zep Cloud。这里用 0.30.2 自带的嵌入式 Kuzu
   （graphiti-core[kuzu]，不用另起服务），每份对话历史一个库。Kuzu 后端在 0.30.2 已标为弃用
   （上游 Kuzu 不再维护），接线上有两处要补：`Graphiti.build_indices_and_constraints` 对 Kuzu
   是空操作，全文索引建不起来、BM25 检索直接报错，这里在新建库时调用驱动自己的
   `graph_ops.build_indices_and_constraints` 建一次；KuzuDriver 没有 `_database` 属性，带
   group_id 写入时报 AttributeError，这里把库名设成这份对话历史的 group_id。
3. 结构化输出保持 OpenAIGenericClient 默认的 json_schema，由网关改成 json_object、把 schema
   写进系统提示（DeepSeek 与通义都不保证支持 json_schema）。没有照它文档对 DeepSeek 的建议改用
   json_object 模式：那样 schema 以"Respond with a JSON object in the following format"附在
   用户消息末尾，试点里 deepseek-flash 常把 schema 本身原样回过来（LongMemEval 前 30 条发言里
   3 条因此整条写入失败）；网关的写法 30 条全部成功。
4. 发言时间：同一会话里每条发言的时间都是会话时间，而 Graphiti 取"前 10 条消息"时只按时间
   排序，同一时刻的发言先后不定；这里给会话里第 i 条发言加 i 秒，让上下文窗口按发言顺序取，
   日期不变。数据集里的时间当作 UTC（Zep 的评测也是这样）。
5. 单条发言截到 8000 字符（Zep 的 LongMemEval 评测也这样截）；问题原样作为查询，不截到
   255 字符（那是 Zep Cloud 接口的限制），也不在前面加日期。
6. 回答：Zep 用自己的 FACTS / ENTITIES 模板和回答提示（gpt-4o-mini）；这里每条事实、每个实体
   各是一条证据，交给同一个回答模型、同一套提示。图片只有描述文字（数据集已并进正文）。
7. 请求网关的超时放宽到 5 小时：评测遇到分时价格的高峰时段会暂停到高峰结束
   （peak_policy=wait），Graphiti 的请求要能等下去，否则超时重试几次后整条 episode 失败。
8. BM25 的查询去掉标点（向量仍用原问题）：Graphiti 对 Neo4j 会先清理查询，对 Kuzu 原样传给
   全文索引，而 Kuzu 不去掉问号（"live?" 匹配不到 "lives"），带问号的问题常常一条也查不到。

试点实测（deepseek 预设，全部 deepseek-flash、本机 bge-small 向量，错峰时段）：LoCoMo conv-42
（629 条发言）写入 36 分钟、2014 次调用、428 万输入 tokens，标价 $1.42、实际 $0.56，没有失败的
发言；每题证据约 3.4k tokens。LongMemEval_S 的发言长得多（平均约 1000 字符，LoCoMo 约 120），
前 30 条每条约 10 次调用、2.2 万输入 tokens（标价约 $0.008 / 条），test100r 约 4.9 万条发言，
写入要几百美元、一天以上，超出下面的写入预算，跑之前先调高 build_budget。

graphiti-core 不是 CueMem 的依赖，评测时用 uv 临时装上（遥测在导入前关闭）：

    uv run --extra local --with "graphiti-core[kuzu]==0.30.2" cuemem eval run \\
        --dataset locomo --split test200r --methods MG --name locomo-test200r-graphiti

（--extra local 只有 deepseek 预设的本机向量要用，默认预设加上也无妨。）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import warnings
from collections import OrderedDict
from collections.abc import Coroutine, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from cuemem.core.models import Hit
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit

TOP_FACTS = 20  # Zep 的 LoCoMo / LongMemEval 评测：事实（边）20 条
TOP_ENTITIES = 20  # 实体（节点）20 个
MAX_MESSAGE_CHARS = 8000  # Zep 的 LongMemEval 评测把每条消息截到 8000 字符
SOURCE_DESCRIPTION = "chat message"
GRAPH_FILE = "graph.kuzu"
LEDGER_FILE = "episodes.json"
PROGRESS_FILE = "progress.json"  # 写入一份对话历史要一两个小时：中途的进度写在这里
PROGRESS_EVERY = 25
MAX_OPEN = 8  # 同时打开的图（每份对话历史一个 Kuzu 库）
MAX_FAILURE_STREAK = 10  # 连续这么多条发言写入失败：多半是网关或供应商出了问题，整份放弃
GATEWAY_TIMEOUT = 5 * 3600.0  # 秒：高峰时段网关会暂停到高峰结束（见模块说明第 7 条）

T = TypeVar("T")


class GraphitiBaseline(ExternalBaseline):
    name, label, system = "MG", "外部基线：Graphiti（Zep 的开源核心，0.30.2）", "graphiti"
    # 逐条发言写入，LLM 调用比 Mem0 多得多：LoCoMo test200r 的 7 段对话按试点估算标价约 $10
    # （错峰实际约 $4），留出余量；LongMemEval 要调高（见模块说明）
    build_budget = 40.0

    def __init__(self) -> None:
        super().__init__()
        self._graphs: OrderedDict[str, _Graph] = OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None

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
        # 抽取结果对不上实体名之类的告警每条 episode 都可能有，只留错误
        logging.getLogger("graphiti_core").setLevel(logging.ERROR)
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, name="cuemem-graphiti", daemon=True).start()
        self._loop = loop

    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        graph = self._graph(key, directory)
        builder, client = self._graphiti(graph.driver, self.bucket(key))
        try:
            return self._run(self._ingest(key, directory, builder, graph, sessions))
        finally:
            self._run(client.close())

    def retrieve(
        self, key: str, directory: Path, sample: Sample, sessions: Sequence[EvalSession]
    ) -> tuple[list[Hit], dict[str, Any]]:
        built = json.loads((directory / "build.json").read_text("utf-8")).get("embedder")
        if built not in (None, self._embedder_id()):  # 图里的向量与查询向量得是同一个模型
            raise RuntimeError(
                f"{directory} 是用向量模型 {built} 写入的，现在是 {self._embedder_id()}："
                "删掉这个目录重建"
            )
        graph = self._graph(key, directory)
        bucket = self.bucket(key, "search")
        before = self.gateway.usage(bucket)
        found = self._run(self._search(graph.searcher, sample.question, group_id(key)))
        after = self.gateway.usage(bucket)
        facts = [
            fact_hit(key, edge, score, rank, graph.ledger)
            for rank, (edge, score) in enumerate(
                zip(found.edges, _scores(found.edge_reranker_scores, found.edges), strict=True),
                start=1,
            )
        ]
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
            # 检索时的向量请求走网关的 search 桶，不在这道题的 trace 里，单独记下
            "search_usage": {
                field: round(after[field] - before[field], 6)
                for field in ("embedding_calls", "chat_calls", "spent")
            },
        }
        return facts + entities, details

    def close(self, key: str) -> None:
        with self._lock:
            self._close_locked(key)

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

        group = group_id(key)
        todo = [
            (session, index, body)
            for session in sorted(sessions, key=lambda s: s.started_at)
            for index, turn in enumerate(session.turns)
            if (body := episode_body(turn)) is not None
        ]
        failed: list[str] = []
        streak = 0
        for done, (session, index, body) in enumerate(todo, start=1):
            when = reference_time(session.started_at, index)
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
            except Exception as exc:  # 换了备用模型也失败的发言：跳过并记下，别的照常
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
                self._progress(key, directory, done, len(todo), len(failed))
        if failed and not graph.ledger.episodes:
            raise RuntimeError(f"没有一条发言写入成功：{failed[-1]}")
        graph.ledger.save()
        return {
            "extractor": self.llm.settings.roles["extractor"].model,
            "embedder": self._embedder_id(),
            "episodes": len(graph.ledger.episodes),
            "failed_episodes": failed,
            **await graph_counts(graph.driver, group),
        }

    def _embedder_id(self) -> str:
        return f"{self.embedder.name}:{self.embedder.dim}"

    def _progress(self, key: str, directory: Path, done: int, total: int, failed: int) -> None:
        usage = self.gateway.usage(self.bucket(key))
        progress = {
            "episodes": done,
            "of": total,
            "failed": failed,
            "chat_calls": usage["chat_calls"],
            "spent": usage["spent"],
            "at": datetime.now().isoformat(timespec="seconds"),
        }
        (directory / PROGRESS_FILE).write_text(json.dumps(progress), "utf-8")

    # ---------- 检索 ----------

    @staticmethod
    async def _search(graphiti: Any, question: str, group: str) -> Any:
        """事实与实体一次检索：两边都是 RRF 混合检索（见模块说明第 1 条）。向量用原问题，
        BM25 用去掉标点的问题（见模块说明第 8 条）。"""
        from graphiti_core.search.search import (  # type: ignore[import-not-found, unused-ignore]
            search,
        )
        from graphiti_core.search.search_config import (  # type: ignore[import-not-found, unused-ignore]
            SearchConfig,
        )
        from graphiti_core.search.search_config_recipes import (  # type: ignore[import-not-found, unused-ignore]
            EDGE_HYBRID_SEARCH_RRF,
            NODE_HYBRID_SEARCH_RRF,
        )
        from graphiti_core.search.search_filters import (  # type: ignore[import-not-found, unused-ignore]
            SearchFilters,
        )

        config = SearchConfig(
            edge_config=EDGE_HYBRID_SEARCH_RRF.edge_config,
            node_config=NODE_HYBRID_SEARCH_RRF.node_config,
            limit=max(TOP_FACTS, TOP_ENTITIES),
        )
        # 与 Graphiti 自己的 search_ 一样算查询向量（换行换成空格）
        vector = await graphiti.embedder.create(input_data=[question.replace("\n", " ")])
        found = await search(
            graphiti.clients,
            keyword_query(question),
            [group],
            config,
            SearchFilters(),
            query_vector=vector,
        )
        found.edges, found.edge_reranker_scores = (
            found.edges[:TOP_FACTS],
            found.edge_reranker_scores[:TOP_FACTS],
        )
        found.nodes, found.node_reranker_scores = (
            found.nodes[:TOP_ENTITIES],
            found.node_reranker_scores[:TOP_ENTITIES],
        )
        return found

    # ---------- 图与客户端 ----------

    def _run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        if self._loop is None:
            coroutine.close()
            raise RuntimeError("Graphiti 的事件循环还没有启动")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def _graphiti(self, driver: Any, bucket: str) -> tuple[Any, Any]:
        """一个 Graphiti 实例：LLM、向量都经网关的这个桶；返回（实例，要关闭的 HTTP 客户端）。"""
        from graphiti_core import Graphiti  # type: ignore[import-not-found, unused-ignore]
        from graphiti_core.cross_encoder.openai_reranker_client import (  # type: ignore[import-not-found, unused-ignore]
            OpenAIRerankerClient,
        )
        from graphiti_core.embedder.openai import (  # type: ignore[import-not-found, unused-ignore]
            OpenAIEmbedder,
            OpenAIEmbedderConfig,
        )
        from graphiti_core.llm_client.config import (  # type: ignore[import-not-found, unused-ignore]
            LLMConfig,
        )
        from graphiti_core.llm_client.openai_generic_client import (  # type: ignore[import-not-found, unused-ignore]
            OpenAIGenericClient,
        )
        from openai import AsyncOpenAI

        url = self.gateway.url(bucket)
        # 网关背后的 LLMClient 自己会重试上游错误；这里只为本机连接的偶发问题重试一次
        client = AsyncOpenAI(
            base_url=url, api_key="cuemem-gateway", timeout=GATEWAY_TIMEOUT, max_retries=1
        )
        model = self.llm.settings.roles["extractor"].model  # 只是记在网关的账上，不改变路由
        config = LLMConfig(api_key="cuemem-gateway", base_url=url, model=model, small_model=model)
        # 结构化输出保持默认的 json_schema（见模块说明第 3 条）
        llm = OpenAIGenericClient(config, client=client)
        embedder = OpenAIEmbedder(
            OpenAIEmbedderConfig(
                embedding_model=self.embedder.name,
                embedding_dim=self.embedder.dim,
                api_key="cuemem-gateway",
                base_url=url,
            ),
            client=client,
        )
        # 必须给一个重排器，否则 Graphiti 会去读 OPENAI_API_KEY；检索配方都是 RRF，用不到它
        reranker = OpenAIRerankerClient(config, client=client)
        graphiti = Graphiti(
            graph_driver=driver, llm_client=llm, embedder=embedder, cross_encoder=reranker
        )
        return graphiti, client

    def _graph(self, key: str, directory: Path) -> _Graph:
        """打开（或复用）一份对话历史的图；最多同时开 MAX_OPEN 个，多了关掉最久没用的。"""
        with self._lock:
            if key in self._graphs:
                self._graphs.move_to_end(key)
                return self._graphs[key]
        from graphiti_core.driver.kuzu_driver import (  # type: ignore[import-not-found, unused-ignore]
            KuzuDriver,
        )

        directory.mkdir(parents=True, exist_ok=True)
        path = directory / GRAPH_FILE
        fresh = not path.exists()
        with warnings.catch_warnings():  # Kuzu 后端已标为弃用（见模块说明第 2 条）
            warnings.simplefilter("ignore", DeprecationWarning)
            driver = KuzuDriver(db=str(path))
        # 0.30.2 的 KuzuDriver 没有 _database，带 group_id 写入时直接报 AttributeError；
        # 一份对话历史一个库，库名就是它的 group_id
        driver._database = group_id(key)
        if fresh:  # 全文索引只能建一次，重复建会报错
            self._run(driver.graph_ops.build_indices_and_constraints(driver))
        searcher, client = self._graphiti(driver, self.bucket(key, "search"))
        graph = _Graph(driver, searcher, client, EpisodeLedger.load(directory / LEDGER_FILE))
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
        graph.driver.client.close()  # KuzuDriver.close() 什么也不做；要释放库文件得自己关
        graph.driver.db.close()


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

    def last_mention(self, node: str) -> tuple[str, datetime | None]:
        episode = self.episodes.get(self.mentions.get(node, ""))
        return (episode[0], episode[2]) if episode else ("", None)


def _by_time(entry: tuple[str, int, datetime]) -> datetime:
    return entry[2]


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


def reference_time(started_at: datetime, index: int) -> datetime:
    """会话时间（当作 UTC）加上发言下标那么多秒：让同一会话的发言有先后（见模块说明第 4 条）。"""
    base = started_at if started_at.tzinfo else started_at.replace(tzinfo=UTC)
    return base + timedelta(seconds=index)


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
    hit.unit.meta["turn_indices"] = turns  # 检索指标里的发言召回用
    return hit


def entity_hit(key: str, node: Any, score: float, rank: int, ledger: EpisodeLedger) -> Hit:
    """Graphiti 的一个实体 → 一条证据"名字: 摘要"：日期与会话取最近一次提到它的发言
    （摘要是到那时为止的累积）。"""
    session, when = ledger.last_mention(node.uuid)
    summary = node.summary.strip() or "(no summary)"
    return memory_hit(
        key,
        f"entity:{node.uuid}",
        f"About {node.name} (entity summary): {summary}",
        naive(when or node.created_at),
        rank,
        score=score,
        route="graphiti_entity",
        session_id=session,
        kind="entity",
    )


def _scores(scores: Sequence[float], items: Sequence[Any]) -> list[float]:
    """重排分数与结果一一对应；对不上时（不同版本的行为差异）记 0。"""
    return list(scores) if len(scores) == len(items) else [0.0] * len(items)


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
