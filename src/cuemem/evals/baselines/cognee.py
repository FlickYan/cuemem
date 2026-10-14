"""外部基线 MC：Cognee（topoteretes/cognee 1.6.4），N12。

照 Cognee 自己在长对话基准 BEAM 上的做法写入与检索（随包发布的 cognee/eval_framework/beam：
REPORT.md、local_ingest.py、report_artifacts/100k_fixed 的检索配置），回答与判分和其他方法完全
相同（见 evals/external.py），只换"记忆从哪来"：

- **写入**：一份对话历史一个 Cognee 数据集。会话按时间顺序逐个 `cognee.add` 一个 JSON 列表文件、
  再 `cognee.cognify(chunker=JsonListChunker, chunk_size=4096)`（BEAM 的 local_ingest：一个会话
  一个文档，一个"轮"一个块，不做重叠切块）。每个块前面写上位置与日期
  （`[Session 3, Turn 5, date 2023-05-08 (Mon) 13:56]`，BEAM 预处理的格式），每句写上说话人。
  cognify 对每个块做一次图谱抽取（实体与关系）和一次摘要，再把块、实体、关系向量化。
- **模型**：Cognee 的 LiteLLM 客户端（`openai/<extractor 模型>`）与 OpenAI 兼容的向量客户端都指向
  评测网关：聊天交给 extractor 角色的模型，向量交给 CueMem 的向量器。模型名 LiteLLM 不认识，
  结构化输出走 Cognee 为自定义端点准备的"JSON 对象 + 提示里写 schema + 校验失败让模型自己改"。
- **检索**：Cognee 1.6.4 的默认检索 HYBRID_COMPLETION（HybridRetriever：块、实体及其关系、
  关系事实三路），参数取 BEAM 100K 报告的固定配置 `hybrid_completion_20_20`
  （chunks_top_k=20、entities_top_k=20，其余默认）。只取它交给自己回答模型的那份上下文
  （get_retrieved_objects，与 eval_framework 的 AnswerGeneratorExecutor 相同），不让 Cognee 回答：
  每个块是一条证据（时间取它所在的会话）；每个实体一条（Cognee 自己的排版：名称、类型、描述、
  关系），时间取图里包含它的块所在的最后一个会话，并注明提到它的所有会话日期；每条关系事实
  一条（文字里带着块的位置前缀时取那个会话的时间）。按名次轮流从三路取，装进约 4000 tokens
  （与 CueMem L1 的预算相同）。

与 Cognee 自身默认及 BEAM 跑法不同的地方，以及原因：

1. 两人对话（LoCoMo）的"轮"：BEAM 的一轮是"用户一句 + 助手一句"。LoCoMo 两位说话人都是真人，
   这里把相邻的两句话合成一轮（各带名字）；用户与助手的对话（LongMemEval）照 BEAM 配对。
   超过 4096 tokens 的一轮按 BEAM 的做法拆成几部分，前缀注明 Part i/n。
2. 不做会话蒸馏（cognee.session.distill_session）与全局上下文索引（global context index）。
   全局索引在 BEAM 报告的固定配置里建了但不参与检索（include_global_context_index=false），
   建了只花钱；会话蒸馏是给"用户问、Agent 答"的会话准备的（逐轮分析用户对上一个回答的反馈、
   给回答打分），LoCoMo 是两个人聊天，套不上，而且每轮多一次 LLM 调用。报告也说没有做消融，
   这两个部件各自的作用不清楚。
3. 回答：BEAM 按题型选回答提示，这里与所有方法一样交给同一个回答模型、同一套提示，Cognee 只
   负责检索。实体与关系事实本来没有日期，这里按图里"块包含实体"的边补上提到它的会话日期
   （Mem0 的每条记忆也标了来源会话的日期）；关系事实的文字里有块的位置前缀时取那个会话，
   找不到来源的标对话最后一个会话的时间，文本里注明日期未知。图里的时间信息（抽取出的
   Timestamp 节点、"spoke_at" 之类的关系）原样保留在实体的关系里。
4. 证据量：BEAM 配置在 LoCoMo 上的上下文是 2.5k–3.7k tokens（试跑的 3 题），全部装得下；
   LongMemEval 的助手回复很长（平均一块约 500 tokens），20 个块就上万 tokens，所以统一装进
   4000 tokens（三路按名次轮流取，放不下的跳过）。
5. 进程隔离：Cognee 的配置（数据目录、模型端点）是进程级的，数据库引擎按路径缓存在进程里。
   这里每份对话历史用独立的子进程写入（写完退出），检索时为每份对话历史开一个常驻子进程
   （最多同时开 4 个），子进程只配置一次、只碰自己的目录。所以不同对话可以并行写入，
   不需要全局锁；代价是每个子进程启动时导入 Cognee 要十来秒。
6. 运行环境：关掉遥测（TELEMETRY_DISABLED）、日志文件、会话缓存（CACHING=false，不用会话记忆）
   与多用户访问控制（ENABLE_BACKEND_ACCESS_CONTROL=false：每份对话历史本来就有自己的系统目录）；
   LiteLLM 用本地的模型价格表，不联网下载；父进程的密钥不传给子进程（子进程只和评测网关说话；
   云上环境的 AWS_* 会让 Cognee 以为要把文件存到 S3）。其余全用默认：图库 Ladybug、
   向量库 LanceDB、关系库 SQLite（都放在评测空间的目录下）、默认的 KnowledgeGraph 抽取提示、
   每块用 LLM 写摘要、HybridRetriever 的其余参数（每个实体最多 10 条关系、5 条关系事实）。
7. 写入失败的会话（网关换了备用模型也失败）跳过并计数；超过一半失败时整份写入算失败。
   Cognee 的增量处理只跳过已经处理完的文档，失败的会话应会在下一个会话的 cognify 里再试一次
   （这条路径没有实测过：试跑里没有失败的会话）。

cognee 不是 CueMem 的依赖，评测时用 uv 临时装上（--extra local 只在用本机向量时需要，
如 deepseek 预设）：

    uv run --extra local --with cognee==1.6.4 \\
        cuemem eval run --dataset locomo --split test200r --methods MC \\
        --name locomo-test200r-cognee

试过的版本：cognee 1.6.4（Python 3.12）。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from cuemem.core.models import Hit
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit
from cuemem.read.l1 import estimate_tokens

CHUNKS_TOP_K = 20  # BEAM 100K 固定配置 hybrid_completion_20_20（Cognee 默认 5）
ENTITIES_TOP_K = 20  # 同上（Cognee 默认 5）
BUDGET_TOKENS = 4000  # 与 CueMem L1 的上下文预算相同
MAX_ITEM_TOKENS = 4096  # BEAM 预处理与 local_ingest 的块大小上限
MAX_MENTION_DATES = 6  # 实体后面最多列出几个提到它的会话日期
MAX_OPEN = 4  # 同时开着的检索子进程
STARTUP_SECONDS = 300.0  # 子进程导入 Cognee、打开数据库的时限
QUERY_SECONDS = 600.0  # 一次检索的时限（只有一次向量请求，正常不到一秒）
BUILD_SECONDS = 6 * 3600.0  # 一份对话历史写入的时限
MANIFEST = "manifest.json"  # 写入的会话文件与会话编号 → 会话 ID、时间
WORKER_LOG = "cognee_worker.log"  # 子进程的日志（Cognee 的输出都在这里）
SECRET_ENV = re.compile(r"^AWS_|KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL", re.IGNORECASE)
PREFIX = re.compile(r"\A\[Session (\d+), Turn (\d+)(?:, Part (\d+)/(\d+))?, date [^\]]*\]\n?")
MARKER = re.compile(r"\[Session (\d+), Turn \d+")  # 文本中间出现的位置前缀（关系事实里）


# ---------- 纯函数（不依赖 cognee，单独测试） ----------


def dataset_name(key: str) -> str:
    """评测空间 → Cognee 数据集名（只留小写字母、数字与下划线）。"""
    return re.sub(r"[^a-z0-9_]+", "_", key.lower()).strip("_") or "conversation"


def turn_pairs(turns: Sequence[EvalTurn]) -> list[list[EvalTurn]]:
    """一个会话拆成"轮"。用户与助手的对话照 BEAM：用户的一句开始新的一轮，紧跟的助手回复并入；
    两位说话人都是用户角色（LoCoMo 两个真人）时，相邻两句合成一轮。空话跳过。"""
    spoken = [t for t in turns if t.text.strip()]
    two_people = all(t.role == "user" for t in spoken)
    rounds: list[list[EvalTurn]] = []
    for turn in spoken:
        current = rounds[-1] if rounds else None
        if current is None:
            rounds.append([turn])
        elif two_people:
            if len(current) >= 2:
                rounds.append([turn])
            else:
                current.append(turn)
        elif turn.role == "user" or any(t.role == "assistant" for t in current):
            rounds.append([turn])
        else:
            current.append(turn)
    return rounds


def _line(turn: EvalTurn) -> str:
    label = turn.speaker or turn.role.capitalize()
    return f"{label}: {turn.text.strip()}"


def _split_body(body: str, budget: int) -> list[str]:
    """超长的一轮按词切成几段（每段不超过 budget tokens 的估计值）。"""
    if estimate_tokens(body) <= budget:
        return [body]
    parts: list[str] = []
    current: list[str] = []
    for word in re.split(r"(\s+)", body):
        if current and estimate_tokens("".join([*current, word])) > budget:
            parts.append("".join(current).strip())
            current = []
        current.append(word)
    if "".join(current).strip():
        parts.append("".join(current).strip())
    return parts


def session_items(
    number: int, session: EvalSession, max_tokens: int = MAX_ITEM_TOKENS
) -> list[str]:
    """一个会话交给 JsonListChunker 的列表：一轮一项，前面写上位置与日期（BEAM 的格式）。"""
    stamp = f"{session.started_at:%Y-%m-%d (%a) %H:%M}"
    items: list[str] = []
    for index, pair in enumerate(turn_pairs(session.turns), start=1):
        body = "\n".join(_line(t) for t in pair)
        head = f"[Session {number}, Turn {index}, Part 99/99, date {stamp}]"
        parts = _split_body(body, max(64, max_tokens - estimate_tokens(head) - 1))
        for part_no, part in enumerate(parts, start=1):
            part_mark = f", Part {part_no}/{len(parts)}" if len(parts) > 1 else ""
            items.append(f"[Session {number}, Turn {index}{part_mark}, date {stamp}]\n{part}")
    return items


def manifest(sessions: Sequence[EvalSession]) -> dict[str, Any]:
    """写入用的清单：会话按时间排序后编号（从 1 开始），每个会话一个 JSON 列表文件。"""
    ordered = sorted(sessions, key=lambda s: s.started_at)
    rows = []
    for number, session in enumerate(ordered, start=1):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session.session_id)
        rows.append(
            {
                "number": number,
                "session_id": session.session_id,
                "started_at": session.started_at.isoformat(),
                "file": f"sessions/s{number:04d}_{safe}.json",
                "items": session_items(number, session),
            }
        )
    return {"sessions": rows}


def split_prefix(text: str) -> tuple[int | None, str]:
    """块文本 → （会话编号，去掉位置前缀的正文）；没有前缀时编号为 None。"""
    match = PREFIX.match(text)
    if match is None:
        return None, text
    return int(match.group(1)), text[match.end() :]


def to_hits(
    space: str, reply: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> tuple[list[Hit], dict[str, Any]]:
    """检索子进程的结果 → 交给回答模型的证据（三路按名次轮流装进预算）与检索信息。

    sessions 是清单里的会话（编号、会话 ID、时间）。"""
    by_number = {int(s["number"]): s for s in sessions}
    latest = max(sessions, key=lambda s: str(s["started_at"])) if sessions else None
    lanes: list[list[Hit]] = [[], [], []]
    for rank, chunk in enumerate(reply.get("chunks", []), start=1):
        number, body = split_prefix(str(chunk["text"]))
        source = by_number.get(number) if number is not None else None
        source = source or latest
        if source is None:
            continue
        lanes[0].append(
            _hit(space, f"chunk:{chunk['id']}", body.strip(), source, rank, chunk.get("score"))
        )
    for rank, entity in enumerate(reply.get("entities", []), start=1):
        mentioned = sorted(
            (by_number[n] for n in set(entity.get("sessions", [])) if n in by_number),
            key=lambda s: str(s["started_at"]),
        )
        text = str(entity["block"]).strip()
        if mentioned:
            dates = [f"{datetime.fromisoformat(s['started_at']):%Y-%m-%d}" for s in mentioned]
            shown = list(dict.fromkeys(dates))
            more = len(shown) - MAX_MENTION_DATES
            listed = ", ".join(shown[-MAX_MENTION_DATES:]) + (
                f" and {more} earlier" if more > 0 else ""
            )
            text += f"\n(Mentioned in sessions on: {listed})"
            source = mentioned[-1]
        elif latest is not None:
            text += "\n(Source session unknown)"
            source = latest
        else:
            continue
        lanes[1].append(_hit(space, f"entity:{entity['id']}", text, source, rank, None, "entity"))
    for rank, fact in enumerate(reply.get("facts", []), start=1):
        # 关系事实是边的文字；摘要、块与文档之间的边会带上块的位置前缀，按它定日期
        marker = MARKER.search(str(fact))
        source = by_number.get(int(marker.group(1))) if marker else None
        text = str(fact) if source else f"{fact} (knowledge-graph fact; date unknown)"
        source = source or latest
        if source is None:
            break
        lanes[2].append(_hit(space, f"fact:{rank}", text, source, rank, None, "fact"))
    kept, used = interleave(lanes, BUDGET_TOKENS)
    details = {
        "cognee_chunks": len(lanes[0]),
        "cognee_entities": len(lanes[1]),
        "cognee_facts": len(lanes[2]),
        "cognee_context_tokens": int(reply.get("context_tokens", 0)),
        "evidence_tokens": used,
    }
    return kept, details


def _hit(
    space: str,
    item_id: str,
    text: str,
    source: Mapping[str, Any],
    rank: int,
    score: Any,
    kind: str = "round",
) -> Hit:
    when = datetime.fromisoformat(str(source["started_at"]))
    value = float(score) if isinstance(score, (int, float)) else 0.0
    return memory_hit(
        space,
        item_id,
        text,
        when,
        rank,
        score=value,
        route=f"cognee_{kind}",
        session_id=str(source["session_id"]),
        kind="entity" if kind == "entity" else ("fact" if kind == "fact" else "round"),
    )


def interleave(lanes: Sequence[Sequence[Hit]], budget: int) -> tuple[list[Hit], int]:
    """几路证据按名次轮流取（第一名的块、第一名的实体、第一条事实、第二名的块……），
    装进预算（至少一条）；放不下的跳过，后面更短的还能放。返回（装下的证据，估计的 tokens）。"""
    kept: list[Hit] = []
    used = 0
    depth = max((len(lane) for lane in lanes), default=0)
    for position in range(depth):
        for lane in lanes:
            if position >= len(lane):
                continue
            hit = lane[position]
            cost = estimate_tokens(hit.unit.text)
            if kept and used + cost > budget:
                continue
            kept.append(hit)
            used += cost
    return kept, used


def worker_env(
    directory: Path, url: str, chat_model: str, embed_model: str, dim: int
) -> dict[str, str]:
    """子进程的环境变量：Cognee 在导入时读配置，所以在启动前全部给好。

    父进程里的密钥一概不传：子进程只和评测网关说话；云上环境的 AWS_* 还会让 Cognee 以为
    要用 S3 存文件（需要 s3fs）。"""
    inherited = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    return {
        **inherited,
        # 遥测、日志文件、会话缓存、多用户访问控制都关掉；LiteLLM 不联网拉价格表
        "TELEMETRY_DISABLED": "1",
        "COGNEE_LOG_FILE": "false",
        "LOG_LEVEL": "ERROR",
        "CACHING": "false",
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "SYSTEM_ROOT_DIRECTORY": str(directory / "cognee_system"),
        "DATA_ROOT_DIRECTORY": str(directory / "cognee_data"),
        # 聊天：LiteLLM 的 openai/<名字> + 自定义 api_base；名字它不认识，结构化输出把
        # schema 写进提示（JSON 对象模式）
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": f"openai/{chat_model}",
        "LLM_ENDPOINT": url,
        "LLM_API_KEY": "unused",
        # 向量：Cognee 给 vLLM、TEI 一类服务准备的 OpenAI 兼容客户端（不经 LiteLLM）
        "EMBEDDING_PROVIDER": "openai_compatible",
        "EMBEDDING_MODEL": embed_model,
        "EMBEDDING_ENDPOINT": url,
        "EMBEDDING_API_KEY": "unused",
        "EMBEDDING_DIMENSIONS": str(dim),
    }


# ---------- 子进程（导入 cognee） ----------


def _configure(directory: Path) -> Any:
    import cognee  # type: ignore[import-not-found, unused-ignore]

    # 环境变量已经给了目录；再用 Cognee 自己的设置函数把关系库、图库、向量库的路径一并改过来
    cognee.config.system_root_directory(str(directory / "cognee_system"))
    cognee.config.data_root_directory(str(directory / "cognee_data"))
    return cognee


async def _build(directory: Path) -> dict[str, Any]:
    cognee = _configure(directory)
    from cognee.modules.chunking.JsonListChunker import (  # type: ignore[import-not-found, unused-ignore]
        JsonListChunker,
    )
    from cognee.modules.engine.operations.setup import (  # type: ignore[import-not-found, unused-ignore]
        setup,
    )

    plan = json.loads((directory / MANIFEST).read_text("utf-8"))
    dataset = str(plan["dataset"])
    await setup()
    failed: list[str] = []
    rows = plan["sessions"]
    for row in rows:
        path = directory / str(row["file"])
        try:
            await cognee.add(str(path), dataset_name=dataset)
            await cognee.cognify(
                datasets=[dataset], chunker=JsonListChunker, chunk_size=MAX_ITEM_TOKENS
            )
        except Exception as exc:  # 换了备用模型也失败的会话：跳过并计数，别的照常
            failed.append(f"{row['session_id']}: {type(exc).__name__}: {exc}"[:300])
    if len(failed) * 2 > len(rows):
        raise RuntimeError(f"Cognee 一半以上的会话写入失败：{failed[:3]}")
    return {"failed_sessions": failed, **await _graph_stats()}


async def _graph_stats() -> dict[str, Any]:
    from cognee.infrastructure.databases.graph import (  # type: ignore[import-not-found, unused-ignore]
        get_graph_engine,
    )

    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    types: dict[str, int] = {}
    for _, props in nodes:
        kind = str((props or {}).get("type", "?"))
        types[kind] = types.get(kind, 0) + 1
    return {"nodes": len(nodes), "edges": len(edges), "node_types": types}


async def _search(query: str) -> dict[str, Any]:
    from cognee.infrastructure.databases.graph import (  # type: ignore[import-not-found, unused-ignore]
        get_graph_engine,
    )
    from cognee.modules.retrieval.exceptions.exceptions import (  # type: ignore[import-not-found, unused-ignore]
        NoDataError,
    )
    from cognee.modules.retrieval.hybrid.entities import (  # type: ignore[import-not-found, unused-ignore]
        format_entities,
    )
    from cognee.modules.retrieval.hybrid.results import (  # type: ignore[import-not-found, unused-ignore]
        payload,
        result_id,
    )
    from cognee.modules.retrieval.hybrid_retriever import (  # type: ignore[import-not-found, unused-ignore]
        HybridRetriever,
    )

    retriever = HybridRetriever(chunks_top_k=CHUNKS_TOP_K, entities_top_k=ENTITIES_TOP_K)
    try:
        found = await retriever.get_retrieved_objects(query=query)
    except NoDataError:
        return {"chunks": [], "entities": [], "facts": [], "context_tokens": 0}
    context = await retriever.get_context_from_objects(query=query, retrieved_objects=found)
    chunks = [
        {
            "id": result_id(chunk),
            "text": str(payload(chunk).get("text") or ""),
            "score": getattr(chunk, "score", None),
        }
        for chunk in found.get("chunks", [])
    ]
    entities = []
    for entity in found.get("entities", []):
        block = format_entities([entity])
        if block:
            entities.append(
                {"id": entity.get("id"), "block": block.split("\n", 1)[1], "sessions": []}
            )
    if entities:
        mentions = await _mentions([str(e["id"]) for e in entities], await get_graph_engine())
        for entity in entities:
            entity["sessions"] = sorted(mentions.get(str(entity["id"]), set()))
    facts = [str(fact["text"]) for fact in found.get("facts", []) if fact.get("text")]
    return {
        "chunks": [c for c in chunks if c["text"]],
        "entities": entities,
        "facts": facts,
        "context_tokens": estimate_tokens(str(context or "")),
    }


async def _mentions(entity_ids: Sequence[str], graph: Any) -> dict[str, set[int]]:
    """实体 → 图里包含它的块所在的会话编号（块的位置前缀里写着）。"""
    nodes, edges = await graph.get_neighborhood(list(entity_ids), depth=1)
    texts: dict[str, str] = {}
    for node in nodes or []:
        if isinstance(node, (list, tuple)) and len(node) == 2 and isinstance(node[1], dict):
            props = node[1]
            if props.get("type") == "DocumentChunk" and isinstance(props.get("text"), str):
                texts[str(node[0])] = props["text"]
    wanted = set(entity_ids)
    found: dict[str, set[int]] = {}
    for edge in edges or []:
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        a, b = str(edge[0]), str(edge[1])
        for entity, other in ((a, b), (b, a)):
            if entity in wanted and other in texts:
                number, _ = split_prefix(texts[other])
                if number is not None:
                    found.setdefault(entity, set()).add(number)
    return found


def _worker_main(argv: Sequence[str]) -> None:
    """子进程入口：`build <目录>` 写入后退出；`serve <目录>` 逐行读查询、逐行回结果。

    协议只走原来的标准输出；Cognee 与 LiteLLM 打印的东西都改到标准错误（父进程写进日志）。"""
    mode, directory = argv[0], Path(argv[1])
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    loop = asyncio.new_event_loop()
    try:
        if mode == "build":
            stats = loop.run_until_complete(_build(directory))
            protocol.write(json.dumps({"ok": True, "stats": stats}, ensure_ascii=False) + "\n")
            return
        _configure(directory)
        protocol.write(json.dumps({"ready": True}) + "\n")
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            try:
                reply = {"ok": True, **loop.run_until_complete(_search(str(request["query"])))}
            except Exception as exc:  # 一次检索出错不影响后面的
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
            protocol.write(json.dumps(reply, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        protocol.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        raise
    finally:
        protocol.flush()
        loop.close()


# ---------- 父进程 ----------


class Worker:
    """一个子进程：按行发 JSON 请求、按行收 JSON 回复，带超时；日志写进目录下的文件。"""

    def __init__(self, argv: Sequence[str], env: Mapping[str, str], log: Path) -> None:
        log.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = log
        self._log: IO[bytes] = log.open("ab")
        self.process = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            env=dict(env),
        )
        self._lock = threading.Lock()

    def receive(self, timeout: float) -> dict[str, Any]:
        """读一行回复；子进程退出或超时都报错（附上日志的最后几行）。"""
        assert self.process.stdout is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                tail = self.tail()
                self.stop(kill=True)
                raise TimeoutError(f"Cognee 子进程 {timeout:.0f} 秒没有回应：{tail}")
            ready, _, _ = select.select([self.process.stdout], [], [], min(remaining, 5.0))
            if not ready:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Cognee 子进程退出了（{self.process.returncode}）：{self.tail()}"
                    )
                continue
            line = self.process.stdout.readline()
            if not line:
                self.process.wait(timeout=30)
                raise RuntimeError(
                    f"Cognee 子进程退出了（{self.process.returncode}）：{self.tail()}"
                )
            reply: dict[str, Any] = json.loads(line)
            return reply

    def ask(self, request: Mapping[str, Any], timeout: float = QUERY_SECONDS) -> dict[str, Any]:
        with self._lock:
            assert self.process.stdin is not None
            self.process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode())
            self.process.stdin.flush()
            return self.receive(timeout)

    def tail(self, lines: int = 8) -> str:
        """日志的最后几行（报错时附上）。"""
        if not self._log.closed:
            self._log.flush()
        try:
            text = self._log_path.read_text("utf-8", errors="replace")
        except OSError:
            return ""
        return " | ".join(text.strip().splitlines()[-lines:])[-1500:]

    def stop(self, *, kill: bool = False) -> None:
        """关掉子进程：先关标准输入让它自己退出（检索子进程读完就退），等不及再强杀。"""
        if self.process.poll() is None:
            if kill:
                self.process.kill()
            elif self.process.stdin is not None:
                with contextlib.suppress(OSError):  # 子进程已经关了管道
                    self.process.stdin.close()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
        self._log.close()


class CogneeBaseline(ExternalBaseline):
    name, label, system = "MC", "外部基线：Cognee（1.6.4，混合检索）", "cognee"

    def __init__(self) -> None:
        super().__init__()
        self._workers: OrderedDict[str, Worker] = OrderedDict()

    def setup(self) -> None:
        import importlib.util

        if importlib.util.find_spec("cognee") is None:
            raise RuntimeError(
                "MC 需要 cognee：用 uv run --with cognee==1.6.4 ... 运行（见模块说明）"
            )

    def _env(self, directory: Path, url: str) -> dict[str, str]:
        chat_model = self.llm.settings.roles[self.gateway.role].model
        return worker_env(directory, url, chat_model, self.embedder.name, self.embedder.dim)

    def _argv(self, mode: str, directory: Path) -> list[str]:
        return [sys.executable, "-m", __name__, mode, str(directory)]

    # ---------- 写入 ----------

    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        plan = {"dataset": dataset_name(key), **manifest(sessions)}
        for row in plan["sessions"]:
            path = directory / row["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row["items"], ensure_ascii=False, indent=1), "utf-8")
        (directory / MANIFEST).write_text(json.dumps(plan, ensure_ascii=False), "utf-8")
        env = self._env(directory, self.gateway.url(self.bucket(key)))
        worker = Worker(self._argv("build", directory), env, directory / WORKER_LOG)
        try:
            reply = worker.receive(BUILD_SECONDS)
        finally:
            worker.stop()
        if not reply.get("ok"):
            raise RuntimeError(f"Cognee 写入失败：{reply.get('error')}")
        items = sum(len(row["items"]) for row in plan["sessions"])
        return {"dataset": plan["dataset"], "chunks_written": items, **reply["stats"]}

    # ---------- 检索 ----------

    def retrieve(
        self, key: str, directory: Path, sample: Sample, sessions: Sequence[EvalSession]
    ) -> tuple[list[Hit], dict[str, Any]]:
        plan = json.loads((directory / MANIFEST).read_text("utf-8"))
        try:
            reply = self._worker(key, directory).ask({"query": sample.question})
        except (OSError, RuntimeError, TimeoutError):
            self.close(key)  # 子进程坏了：这道题记错误，下一道题重开一个
            raise
        if not reply.get("ok"):
            raise RuntimeError(f"Cognee 检索失败：{reply.get('error')}")
        return to_hits(key, reply, plan["sessions"])

    def _worker(self, key: str, directory: Path) -> Worker:
        """打开（或复用）一份对话历史的检索子进程；最多同时开 MAX_OPEN 个，多了关掉最久没用的。

        检索时的向量请求记在 search 桶里。"""
        with self._lock:
            if key in self._workers:
                self._workers.move_to_end(key)
                return self._workers[key]
        env = self._env(directory, self.gateway.url(self.bucket(key, "search")))
        worker = Worker(self._argv("serve", directory), env, directory / WORKER_LOG)
        try:
            ready = worker.receive(STARTUP_SECONDS)
        except Exception:
            worker.stop()
            raise
        if not ready.get("ready"):
            worker.stop()
            raise RuntimeError(f"Cognee 检索子进程没能启动：{ready.get('error')}")
        idle: list[Worker] = []
        with self._lock:
            self._workers[key] = worker
            while len(self._workers) > MAX_OPEN:
                old_key = next(iter(self._workers))
                if self._space_locks.get(old_key, threading.Lock()).locked():
                    break  # 正在用：留着，下次再关
                idle.append(self._workers.pop(old_key))
        for old in idle:
            old.stop()
        return worker

    def close(self, key: str) -> None:
        with self._lock:
            worker = self._workers.pop(key, None)
        if worker is not None:
            worker.stop()


if __name__ == "__main__":  # 子进程：python -m cuemem.evals.baselines.cognee build|serve <目录>
    _worker_main(sys.argv[1:])
