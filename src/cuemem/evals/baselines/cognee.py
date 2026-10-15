"""外部基线 MC：Cognee（topoteretes/cognee 1.6.4），N12。

照 Cognee 自己在长对话基准 BEAM 上的做法写入与检索（随包发布的 cognee/eval_framework/beam：
REPORT.md、preprocessing/preprocess.py、local_ingest.py、report_artifacts/100k_fixed 的检索配置），
回答与判分和其他方法完全相同（见 evals/external.py），只换"记忆从哪来"：

- **写入**：一份对话历史一个 Cognee 数据集。会话按时间顺序逐个 `cognee.add` 一个 JSON 列表文件、
  再 `cognee.cognify(chunker=JsonListChunker, chunk_size=4096)`（local_ingest：一个会话一个文档，
  一"轮"一个块，不做重叠切块；data_per_batch、chunks_per_batch 也取它的默认值）。每一项照 BEAM
  预处理的 format_turn_pair 排版：`Session: n` / `Turn: m` / `Time anchor: <会话时间>`，空一行，
  再是 `User:` 与 `Assistant:` 各自的话。cognify 对每个块做一次图谱抽取（实体与关系）和一次摘要，
  再把块、实体、关系向量化。
- **会话蒸馏**（用户与助手的对话，如 LongMemEval）：与 local_ingest 相同，写入一个会话的同时逐轮
  分析用户的话（harvest_session_candidates，并发 50），写完后 distill_session 把提炼出的经验
  （"Session learning"）作为文档 add + cognify 进同一个数据集，检索时与其他块一样能被取到。
  会话缓存用 local_ingest 的设置（CACHING=true、CACHE_BACKEND=fs、AUTO_FEEDBACK=true），
  放在评测空间的目录下。
- **模型**：Cognee 的 LiteLLM 客户端（`openai/<extractor 模型>`）与 OpenAI 兼容的向量客户端都指向
  评测网关：聊天交给 extractor 角色的模型，向量交给 CueMem 的向量器。模型名 LiteLLM 不认识，
  结构化输出走 Cognee 为自定义端点准备的"JSON 对象 + 提示里写 schema + 校验失败让模型自己改"。
- **检索**：Cognee 1.6.4 的默认检索 HYBRID_COMPLETION（HybridRetriever：块、实体及其关系、
  关系事实三路），参数取 BEAM 100K 报告的固定配置 `hybrid_completion_20_20`
  （chunks_top_k=20、entities_top_k=20，其余默认）。取它交给自己回答模型的那份上下文
  （get_retrieved_objects，与 eval_framework 的 AnswerGeneratorExecutor 相同），**全部**交给
  回答模型（不截断），不让 Cognee 回答：每个块、每个实体（Cognee 自己的排版：名称、类型、描述、
  关系）、每条关系事实各是一条证据。
- **日期**：块的日期就是它所在会话的时间。实体与关系事实本来没有日期，这里按 Cognee 自己的
  出处记录补上：图里的每条边在 provenance_edge_evidence 里记着是从哪些块抽出来的
  （get_edge_evidence_records，Cognee 给回答附出处时用的就是它），块又在哪个会话里。
  关系事实（EdgeType 向量行）先找到图里文字相同的边，再取这些边的出处（一端是块的边，如块 →
  实体的包含边、摘要 → 块的 made_from 边，直接取块的会话）；有几个出处时取最近的
  会话、并注明各次的日期；找不到出处的（很少）标对话第一个会话的时间并注明日期未知、会话
  记为 unknown，不会被当成最新的记忆。实体的时间取它的描述来自的那个块（"Document chunk
  mentions <名字>: <描述>" 的包含边），实体下的每条关系各自注明出处的日期。蒸馏出的经验取它
  所在的会话。评测指标也按出处算：块带上它覆盖的发言下标，实体与事实的会话是它们的出处
  （unknown 不计入会话召回）。

与 Cognee 自身默认及 BEAM 跑法不同的地方，以及原因：

1. 两人对话（LoCoMo）的"轮"：BEAM 的一轮是"用户一句 + 助手一句"。LoCoMo 两位说话人都是真人，
   这里把相邻的两句话合成一轮，标签用说话人的名字（`Melanie:` / `Caroline:`）；用户与助手的对话
   照 BEAM 配对，标签是 `User:` / `Assistant:`（local_ingest 的会话分析靠这两个标签找出每一轮）。
   超过 4096 tokens 的一轮拆成几部分，头部多一行 `Part: i/n`（BEAM 预处理是先压缩再拆，这里不
   压缩：压缩要额外调 LLM 改写原文）。Time anchor 写会话时间（含星期与时刻），不用 BEAM 数据集
   自己的日期格式。
2. 不建全局上下文索引（global context index）：BEAM 报告的固定配置里建了但不参与检索
   （include_global_context_index=false），建了只花钱。两人对话（LoCoMo）不做会话蒸馏：
   local_ingest 只从 `User:`/`Assistant:` 标签里找出用户的问题与助手的回答，两个真人聊天没有
   这样的轮，它的流程在这里取不到任何东西。
3. 回答：BEAM 按题型选回答提示，这里与所有方法一样交给同一个回答模型、同一套提示，Cognee 只
   负责检索。上下文用 1.6.4 自己的排版：BEAM 报告里那次 100K 运行的上下文在每个块前还有一段
   块摘要（"[Passage Summary]"），1.6.4 的 format_passages 已经不输出它，这里跟随 1.6.4。
   回答模型按时间排列证据（与其他方法相同），不是 Cognee 的"块、实体、事实"三段。实体与事实的
   日期见上：Cognee 自己的回答模型看不到这些日期，这里补上是因为所有方法的证据都带日期
   （Mem0 的每条记忆也标了来源会话的日期）。
4. 证据量不设上限：Cognee 的 20 块 + 20 个实体 + 5 条事实全部交给回答模型（BEAM 报告里也是原样
   交给回答模型）。LoCoMo 上约 2.5k–4k tokens，与 CueMem L1 的预算相近；LongMemEval 的助手回复长
   （一块平均约 500 tokens），上下文约 1 万多 tokens，比其他方法多，回答的输入 tokens 记在报告里。
5. 进程隔离：Cognee 的配置（数据目录、模型端点）是进程级的，数据库引擎按路径缓存在进程里。
   这里每份对话历史用独立的子进程写入（写完退出），检索时为每份对话历史开一个常驻子进程
   （最多同时开 4 个），子进程只配置一次、只碰自己的目录。所以不同对话可以并行写入，
   不需要全局锁；代价是每个子进程启动时导入 Cognee 要十来秒。
6. 运行环境：关掉遥测（TELEMETRY_DISABLED）、日志文件与多用户访问控制
   （ENABLE_BACKEND_ACCESS_CONTROL=false：每份对话历史本来就有自己的系统目录）；检索时不用会话
   缓存（CACHING=false：每道题是独立的提问，不是一段会话）。LiteLLM 用本地的模型价格表，不联网
   下载。父进程的密钥不传给子进程，子进程也不读项目的 .env（Cognee 默认从当前目录往上找 .env
   并覆盖环境变量，这里用 COGNEE_ENV_FILE 指向空间目录里的一个空文件；LiteLLM 导入时也会找
   .env 补上没设的变量，LITELLM_MODE=PRODUCTION 关掉它）：子进程只和评测网关说话；
   云上环境的 AWS_* 还会让 Cognee 以为要把文件存到 S3。其余全用默认：图库 Ladybug、向量库
   LanceDB、关系库 SQLite（都放在评测空间的目录下）、默认的 KnowledgeGraph 抽取提示、每块用 LLM
   写摘要、HybridRetriever 的其余参数（每个实体最多 10 条关系、5 条关系事实）。
7. 写入失败：Cognee 的 cognify 处理数据集里所有没处理完的文档，一个文档出错就把整次运行回滚
   （run_tasks + cognify_rollback_handler），失败的会话会在之后每个会话的 cognify 里重跑、连带
   把新会话一起回滚。BEAM 的 local_ingest 遇错直接停下；这里与 M0 一样跳过失败的会话：每个会话
   之后按 Cognee 自己记的处理状态（Data.pipeline_status）判断成败，没处理完的文档从数据集里删掉
   （cognee.datasets.delete_data），之后的会话不受牵连。删掉的会话马上重写一次（限流之类的暂时
   错误多半已经过去；网关有响应缓存，确定性的错误会原样再来一次，只多花时间），仍然失败的跳过，
   记在 build.json 的 failed_sessions 里（重写过的记在 retried_sessions）；超过一半失败时整份
   写入算失败。
8. 本机向量（deepseek 预设的 bge-small，输入上限 512 tokens）：Cognee 会把 chunk_size 4096 降到
   向量模型的上限（实际值记在 build.json 的 effective_chunk_size），JsonListChunker 仍然一轮一块，
   超过 512 tokens 的块向量化时被截断（CueMem 自己的方法用同一个向量器，同样截断）。
9. 检索时问题应按查询向量化（与 CueMem 的检索、M0 相同；通义的向量模型区分 query 与
   document）：检索子进程只向量化问题，请求都记在 `cognee.search.<空间>` 桶里，并带上
   input_type=query（EMBEDDING_INPUT_TYPE）。网关要按检索阶段的桶（或 input_type）选查询向量；
   网关没有这条规则时按文档向量化。deepseek 预设的 bge-small 没有检索指令，两者相同。这些请求
   在子进程与网关里，不在这道题的 trace 里：网关把用量转记进这道题（见 evals/external.py）。

写入记录（build.json）记下协议（抽取模型与向量模型，换了就报错、不删，见 evals/external.py）与
写入标记（build_tag：cognee 的版本与这里的写入格式，换了就重建；早先接在 protocol 后面，照样
认得）；装的 cognee 不是 1.6.4 时直接报错（这里用到它的内部模块与默认值）。

cognee 不是 CueMem 的依赖，评测时用 uv 临时装上（--extra local 只在用本机向量时需要，
如 deepseek 预设）：

    uv run --extra local --with cognee==1.6.4 \\
        cuemem eval run --dataset locomo --split test200r --methods MC \\
        --name locomo-test200r-cognee

LongMemEval 同样（--dataset longmemeval_s），会自动做会话蒸馏。

试过的版本：cognee 1.6.4（Python 3.12）。LoCoMo 一份对话（29 个会话）的写入与检索实测过；
会话蒸馏在 LongMemEval 一道题的 4 个会话上实测过（写入、蒸馏出经验、检索时取到经验）。
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any

from cuemem.core.models import Hit
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit
from cuemem.read.l1 import estimate_tokens

COGNEE_VERSION = "1.6.4"  # 用到它的内部模块与默认值：别的版本直接报错
INGEST_FORMAT = "beam-turn-pair"  # 写入格式；改了就换个名字，旧的写入记录不再复用
CHUNKS_TOP_K = 20  # BEAM 100K 固定配置 hybrid_completion_20_20（Cognee 默认 5）
ENTITIES_TOP_K = 20  # 同上（Cognee 默认 5）
MAX_ITEM_TOKENS = 4096  # BEAM 预处理与 local_ingest 的块大小上限
DATA_PER_BATCH, CHUNKS_PER_BATCH = 100, 1000  # local_ingest 的默认值
CANDIDATE_CONCURRENCY = 50  # local_ingest 的默认值：会话分析的并发
EVIDENCE_PER_EDGE = 100  # 每条边最多取几个出处（取全）
MAX_SOURCE_DATES = 3  # 一条关系后面最多列出几个出处日期
MAX_OPEN = 4  # 同时开着的检索子进程
STARTUP_SECONDS = 300.0  # 子进程导入 Cognee、打开数据库的时限
QUERY_SECONDS = 600.0  # 一次检索的时限（只有一次向量请求，正常不到一秒）
BUILD_SECONDS = 6 * 3600.0  # 一份对话历史写入的时限
MANIFEST = "manifest.json"  # 写入的会话文件与会话编号 → 会话 ID、时间、每一轮的发言下标
WORKER_LOG = "cognee_worker.log"  # 子进程的日志（Cognee 的输出都在这里）
ENV_FILE = "cognee.env"  # 子进程读的 .env：空文件（不读项目的 .env）
UNKNOWN_SESSION = "unknown"  # 找不到出处的证据记在这个会话名下
CONTAINS = "Document chunk mentions "  # Cognee 的块 → 实体包含边的文字前缀
SECRET_ENV = re.compile(r"^AWS_|KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL", re.IGNORECASE)
PREFIX = re.compile(r"\ASession: (\d+)\nTurn: (\d+)\n(?:Part: \d+/\d+\n)?Time anchor: [^\n]*\n+")
LESSON = re.compile(r"\A# Session learning \(session s(\d{4})_[^)\n]*\)\n*")
# 会话蒸馏的会话 ID 写成 s<编号>_session（distill_session_id）；写入记录里记下它，早先带着题号的
# 蒸馏写入不再复用（只涉及做了蒸馏的历史，LoCoMo 不做蒸馏、不受影响）
LESSON_TAG = "session"
SESSION_FILE = re.compile(r"\As(\d{4})_")  # 会话文件名（Cognee 的 Data.name）→ 会话编号


# ---------- 纯函数（不依赖 cognee，单独测试） ----------


def dataset_name(key: str) -> str:
    """评测空间 → Cognee 数据集名（只留小写字母、数字与下划线）。"""
    return re.sub(r"[^a-z0-9_]+", "_", key.lower()).strip("_") or "conversation"


def two_people(sessions: Sequence[EvalSession]) -> bool:
    """整份历史里没有助手的话 → 两个人的对话（LoCoMo 的两位说话人都是 user 角色）。"""
    return not any(t.role == "assistant" for s in sessions for t in s.turns)


def turn_pairs(turns: Sequence[EvalTurn], pairs_of_people: bool) -> list[list[EvalTurn]]:
    """一个会话拆成"轮"。用户与助手的对话照 BEAM：用户的一句开始新的一轮，紧跟的助手回复并入；
    两个人的对话里相邻两句合成一轮。空话跳过。"""
    rounds: list[list[EvalTurn]] = []
    for turn in (t for t in turns if t.text.strip()):
        current = rounds[-1] if rounds else None
        if current is None:
            rounds.append([turn])
        elif pairs_of_people:
            if len(current) >= 2:
                rounds.append([turn])
            else:
                current.append(turn)
        elif turn.role == "user" or any(t.role == "assistant" for t in current):
            rounds.append([turn])
        else:
            current.append(turn)
    return rounds


def _label(turn: EvalTurn, pairs_of_people: bool) -> str:
    if pairs_of_people:
        return turn.speaker or "User"
    return turn.role.capitalize()  # local_ingest 的会话分析按 User:/Assistant: 找出每一轮


def session_rounds(session: EvalSession, pairs_of_people: bool) -> list[tuple[list[int], str]]:
    """一个会话的每一轮：（这一轮的发言在会话里的下标，正文）。正文照 format_turn_pair 排版。"""
    position = {id(t): i for i, t in enumerate(session.turns)}
    return [
        (
            [position[id(t)] for t in pair],
            "\n\n".join(f"{_label(t, pairs_of_people)}:\n{t.text.strip()}" for t in pair),
        )
        for pair in turn_pairs(session.turns, pairs_of_people)
    ]


def _header(number: int, turn: int, anchor: str, part: tuple[int, int] | None = None) -> str:
    lines = [f"Session: {number}", f"Turn: {turn}"]
    if part is not None:
        lines.append(f"Part: {part[0]}/{part[1]}")
    lines.append(f"Time anchor: {anchor}")
    return "\n".join(lines)


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
    number: int,
    session: EvalSession,
    pairs_of_people: bool,
    max_tokens: int = MAX_ITEM_TOKENS,
) -> list[str]:
    """一个会话交给 JsonListChunker 的列表：一轮一项，照 BEAM 的 format_turn_pair 排版。"""
    anchor = f"{session.started_at:%Y-%m-%d (%a) %H:%M}"
    items: list[str] = []
    for turn, (_, body) in enumerate(session_rounds(session, pairs_of_people), start=1):
        head = _header(number, turn, anchor, (99, 99))
        parts = _split_body(body, max(64, max_tokens - estimate_tokens(head) - 2))
        for part_no, part in enumerate(parts, start=1):
            part_of = (part_no, len(parts)) if len(parts) > 1 else None
            items.append(f"{_header(number, turn, anchor, part_of)}\n\n{part}")
    return items


def manifest(sessions: Sequence[EvalSession]) -> dict[str, Any]:
    """写入用的清单：会话按时间排序后编号（从 1 开始），每个会话一个 JSON 列表文件；
    rounds 是每一轮（Turn m）的发言下标，检索时据此给块标上它覆盖的发言（只用于评测指标）。"""
    pairs_of_people = two_people(sessions)
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
                "items": session_items(number, session, pairs_of_people),
                "rounds": [turns for turns, _ in session_rounds(session, pairs_of_people)],
            }
        )
    return {"two_people": pairs_of_people, "sessions": rows}


def distill_session_id(number: int) -> str:
    """会话蒸馏用的会话 ID；蒸馏出的经验文档开头写着它，检索时据此找回是哪个会话。

    只用会话编号：经验文档要交给 cognify 的模型抽图谱，Cognee 还按它建节点集
    （session_learnings:<会话 ID>，实体的关系里看得到）。早先这里带着数据集名，也就是题号，
    LongMemEval 拒答题的题号以 _abs 结尾，等于把"这题该拒答"写进了记忆（N12 修复）。
    子进程的数据都在这份历史自己的目录里，编号已经不会重。"""
    return f"s{number:04d}_{LESSON_TAG}"


def split_prefix(text: str) -> tuple[int | None, int | None, str]:
    """块文本 → （会话编号，第几轮，去掉头部的正文）。

    会话蒸馏出的经验没有轮次，正文保留"# Session learning"的标题；认不出来源时编号为 None。"""
    match = PREFIX.match(text)
    if match is not None:
        return int(match.group(1)), int(match.group(2)), text[match.end() :]
    lesson = LESSON.match(text)
    if lesson is not None:
        return int(lesson.group(1)), None, f"# Session learning\n\n{text[lesson.end() :]}"
    return None, None, text


def to_hits(
    space: str, reply: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]
) -> tuple[list[Hit], dict[str, Any]]:
    """检索子进程的结果 → 交给回答模型的证据（Cognee 的上下文全部保留）与检索信息。

    sessions 是清单里的会话（编号、会话 ID、时间、每一轮的发言下标）。"""
    by_number = {int(s["number"]): s for s in sessions}
    if not by_number:
        return [], {"cognee_chunks": 0, "cognee_entities": 0, "cognee_facts": 0}
    first = min(by_number.values(), key=lambda s: str(s["started_at"]))
    hits: list[Hit] = []
    undated = 0

    def add(item_id: str, text: str, rows: list[Mapping[str, Any]], rank: int, **kw: Any) -> None:
        nonlocal undated
        source = rows[-1] if rows else None
        if source is None:
            undated += 1
            text += " (date unknown)"
        hits.append(_hit(space, item_id, text, source, first, rank, **kw))

    for rank, chunk in enumerate(reply.get("chunks", []), start=1):
        number, turn, body = split_prefix(str(chunk["text"]))
        rows = _rows(by_number, [number] if number is not None else [])
        rounds = rows[0].get("rounds", []) if rows else []
        turns = rounds[turn - 1] if turn is not None and 0 < turn <= len(rounds) else []
        kind = "lesson" if number is not None and turn is None else "round"  # 会话蒸馏出的经验
        item = f"chunk:{chunk['id']}"
        add(item, body.strip(), rows, rank, score=chunk.get("score"), turns=turns, kind=kind)
    for rank, entity in enumerate(reply.get("entities", []), start=1):
        lines = [str(entity["header"]).strip()]
        for edge in entity.get("edges", []):
            lines.append(f"- {edge['text']}{_dated(_rows(by_number, edge.get('sessions', [])))}")
        rows = _rows(by_number, entity.get("sessions", []))
        add(f"entity:{entity['id']}", "\n".join(lines), rows, rank, kind="entity")
    for rank, fact in enumerate(reply.get("facts", []), start=1):
        rows = _rows(by_number, fact.get("sessions", []))
        text = str(fact["text"]) + (_dated(rows) if len(rows) > 1 else "")
        add(f"fact:{fact['id']}", text, rows, rank, kind="fact")
    details = {
        "cognee_chunks": len(reply.get("chunks", [])),
        "cognee_entities": len(reply.get("entities", [])),
        "cognee_facts": len(reply.get("facts", [])),
        "cognee_undated": undated,
        "cognee_context_tokens": int(reply.get("context_tokens", 0)),
        "evidence_tokens": sum(estimate_tokens(h.unit.text) for h in hits),
    }
    return hits, details


def _rows(by_number: Mapping[int, Mapping[str, Any]], numbers: Any) -> list[Mapping[str, Any]]:
    """会话编号 → 清单里的会话（去重，按时间排）。"""
    found = {int(n): by_number[int(n)] for n in numbers or [] if int(n) in by_number}
    return sorted(found.values(), key=lambda s: str(s["started_at"]))


def _dated(rows: Sequence[Mapping[str, Any]]) -> str:
    """一条关系的出处日期（最近的几个），没有出处时为空。"""
    dates = list(
        dict.fromkeys(f"{datetime.fromisoformat(str(s['started_at'])):%Y-%m-%d}" for s in rows)
    )
    if not dates:
        return ""
    shown, more = dates[-MAX_SOURCE_DATES:], len(dates) - MAX_SOURCE_DATES
    listed = ", ".join(shown) + (f", and {more} earlier" if more > 0 else "")
    return f" (from the conversation{'s' if len(dates) > 1 else ''} on {listed})"


def _hit(
    space: str,
    item_id: str,
    text: str,
    source: Mapping[str, Any] | None,
    first: Mapping[str, Any],
    rank: int,
    *,
    score: Any = None,
    kind: str = "round",
    turns: Sequence[int] = (),
) -> Hit:
    """一条证据：时间与会话取它的出处；找不到出处时取第一个会话的时间、会话记为 unknown
    （不会被当成最新的记忆，也不算进会话召回）。"""
    when = datetime.fromisoformat(str((source or first)["started_at"]))
    hit = memory_hit(
        space,
        item_id,
        text,
        when,
        rank,
        score=float(score) if isinstance(score, (int, float)) else 0.0,
        route=f"cognee_{kind}",
        session_id=str(source["session_id"]) if source is not None else UNKNOWN_SESSION,
        kind="entity" if kind == "entity" else ("fact" if kind == "fact" else "round"),
    )
    if turns:
        hit.unit.meta["turn_indices"] = [int(i) for i in turns]  # 评测指标用：块覆盖了哪些发言
    return hit


@dataclass
class GraphIndex:
    """检索子进程里图的索引（图写完就不变，每个子进程建一次），用来给实体与事实找出处。

    chunk_session：块 → 会话编号；type_edges：EdgeType ID（关系事实的 ID）→ 图里文字相同的边
    （edge_object_id，两端）；mentions：实体 → 包含它的块（会话编号，包含边的文字）。"""

    chunk_session: dict[str, int] = field(default_factory=dict)
    type_edges: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)
    mentions: dict[str, list[tuple[int, str]]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        nodes: Sequence[Any],
        edges: Sequence[Any],
        edge_type_id: Callable[[str, Mapping[str, Any]], str | None],
    ) -> GraphIndex:
        """nodes 是 (ID, 属性)，edges 是 (起点, 终点, 关系名, 属性)，即 get_graph_data 的格式；
        edge_type_id 是 Cognee 的 connection_edge_type_id（边 → 它的 EdgeType 向量行 ID）。"""
        index = cls()
        for node in nodes:
            node_id, props = str(node[0]), node[1] or {}
            if props.get("type") == "DocumentChunk" and isinstance(props.get("text"), str):
                number, _, _ = split_prefix(props["text"])
                if number is not None:
                    index.chunk_session[node_id] = number
        for edge in edges:
            source, target, relation = str(edge[0]), str(edge[1]), str(edge[2])
            props = edge[3] if len(edge) > 3 and isinstance(edge[3], dict) else {}
            type_id = edge_type_id(relation, props)
            object_id = str(props.get("edge_object_id") or "")
            if type_id:
                index.type_edges.setdefault(type_id, []).append((object_id, source, target))
            if relation == "contains" and source in index.chunk_session:
                text = str(props.get("edge_text") or "")
                index.mentions.setdefault(target, []).append((index.chunk_session[source], text))
        return index

    def entity_sessions(self, entity_id: str, description: str) -> list[int]:
        """实体的描述来自哪些会话：Cognee 每次抽到它都会用这个块的描述覆盖旧的，包含边的文字是
        "Document chunk mentions <名字>: <描述>"。对不上时取提到它的所有会话。"""
        mentioned = self.mentions.get(entity_id, [])
        wanted = f": {description.strip()}"
        matched = {
            n
            for n, text in mentioned
            if description.strip() and text.startswith(CONTAINS) and text.endswith(wanted)
        }
        return sorted(matched or {n for n, _ in mentioned})

    def fact_edges(self, fact_id: str) -> tuple[list[str], set[int]]:
        """关系事实 → （图里文字相同的边的 edge_object_id，一端是块的那些边的会话）。

        一端是块的边：块 → 实体的包含边、摘要 → 块的 made_from 边等，会话就是块的会话；
        实体之间的关系交给出处记录。"""
        found = self.type_edges.get(fact_id, [])
        sessions = {
            self.chunk_session[end]
            for _, *ends in found
            for end in ends
            if end in self.chunk_session
        }
        return [o for o, _, _ in found if o], sessions


def worker_env(
    directory: Path,
    url: str,
    chat_model: str,
    embed_model: str,
    dim: int,
    *,
    mode: str = "serve",
    distill: bool = False,
) -> dict[str, str]:
    """子进程的环境变量：Cognee 在导入时读配置，所以在启动前全部给好。

    父进程里的密钥一概不传：子进程只和评测网关说话；云上环境的 AWS_* 还会让 Cognee 以为要用 S3
    存文件（需要 s3fs）。项目的 .env 也不能读（数据目录就是仓库根目录时那里有 CueMem 的 .env，
    里面有 Key、也可能有 LLM_* 之类的变量）：Cognee 导入时从当前目录往上找 .env 并覆盖环境变量，
    所以在空间目录里放一个空文件，用 COGNEE_ENV_FILE 指过去（会创建这个文件）；LiteLLM 导入时
    也会 load_dotenv（不覆盖已有的变量，但会补上 Key 与 AWS_* 之类），LITELLM_MODE=PRODUCTION
    时不读。"""
    directory.mkdir(parents=True, exist_ok=True)
    env_file = directory / ENV_FILE
    if not env_file.exists():
        env_file.write_text("", "utf-8")
    inherited = {k: v for k, v in os.environ.items() if not SECRET_ENV.search(k)}
    env = {
        **inherited,
        "COGNEE_ENV_FILE": str(env_file),
        "LITELLM_MODE": "PRODUCTION",  # 只关掉 LiteLLM 导入时的 load_dotenv
        # 遥测、日志文件、多用户访问控制都关掉；LiteLLM 不联网拉价格表
        "TELEMETRY_DISABLED": "1",
        "COGNEE_LOG_FILE": "false",
        "LOG_LEVEL": "ERROR",
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
    if mode == "build" and distill:
        # 会话蒸馏要会话缓存：取 local_ingest 的设置（文件缓存放在 DATA_ROOT_DIRECTORY 下）
        env.update({"CACHING": "true", "CACHE_BACKEND": "fs", "AUTO_FEEDBACK": "true"})
    else:
        env["CACHING"] = "false"
    if mode == "serve":
        env["EMBEDDING_INPUT_TYPE"] = "query"  # 检索时只向量化问题
    return env


# ---------- 子进程（导入 cognee） ----------


def _configure(directory: Path) -> Any:
    import cognee  # type: ignore[import-not-found, unused-ignore]

    # 环境变量已经给了目录；再用 Cognee 自己的设置函数把关系库、图库、向量库的路径一并改过来
    cognee.config.system_root_directory(str(directory / "cognee_system"))
    cognee.config.data_root_directory(str(directory / "cognee_data"))
    return cognee


async def _dataset_id(cognee: Any, name: str, user: Any = None) -> Any:
    for dataset in await cognee.datasets.list_datasets(user=user):
        if dataset.name == name:
            return dataset.id
    return None


async def _settle(cognee: Any, dataset_id: Any, user: Any) -> tuple[set[str], list[str]]:
    """按 Cognee 自己记的处理状态清点数据集：返回（处理完的文档名，删掉的没处理完的文档名）。

    没处理完的文档留在数据集里，之后每次 cognify 都会重跑它，它再出错就把整次运行（包括新会话）
    一起回滚；删掉之后互不牵连。"""
    from cognee.modules.pipelines.models.DataItemStatus import (  # type: ignore[import-not-found, unused-ignore]
        is_data_item_completed,
    )

    done: set[str] = set()
    dropped: list[str] = []
    if dataset_id is None:
        return done, dropped
    for data in await cognee.datasets.list_data(dataset_id, user=user):
        status = ((data.pipeline_status or {}).get("cognify_pipeline") or {}).get(str(dataset_id))
        if is_data_item_completed(status):
            done.add(str(data.name))
            continue
        await cognee.datasets.delete_data(dataset_id, data.id, user=user)
        dropped.append(str(data.name))
    return done, dropped


async def _build(directory: Path) -> dict[str, Any]:
    cognee = _configure(directory)
    from cognee.infrastructure.llm.utils import (  # type: ignore[import-not-found, unused-ignore]
        resolve_chunk_size,
    )
    from cognee.modules.chunking.JsonListChunker import (  # type: ignore[import-not-found, unused-ignore]
        JsonListChunker,
    )
    from cognee.modules.engine.operations.setup import (  # type: ignore[import-not-found, unused-ignore]
        setup,
    )
    from cognee.modules.users.methods import (  # type: ignore[import-not-found, unused-ignore]
        get_default_user,
    )

    plan = json.loads((directory / MANIFEST).read_text("utf-8"))
    dataset = str(plan["dataset"])
    distill = bool(plan.get("distill"))
    await setup()
    user = await get_default_user()
    ingest_beam: Any = None
    if distill:  # BEAM 的 local_ingest：导入时按它的设置配置日志与会话缓存
        import importlib

        ingest_beam = importlib.import_module("cognee.eval_framework.beam.local_ingest")
    failed: list[str] = []
    retried: list[str] = []  # 第一次没写成、删掉后重写的会话
    dropped: list[str] = []
    distilled: list[dict[str, Any]] = []
    dataset_id: Any = None
    rows = [row for row in plan["sessions"] if row["items"]]
    for row in rows:
        path = directory / str(row["file"])

        async def ingest(path: Path = path) -> None:
            await cognee.add(str(path), dataset_name=dataset, data_per_batch=DATA_PER_BATCH)
            await cognee.cognify(
                datasets=[dataset],
                chunker=JsonListChunker,
                chunk_size=MAX_ITEM_TOKENS,
                data_per_batch=DATA_PER_BATCH,
                chunks_per_batch=CHUNKS_PER_BATCH,
            )

        steps: list[Any] = [ingest()]
        session_id = distill_session_id(int(row["number"]))
        if ingest_beam is not None:  # 与 local_ingest 相同：写入的同时逐轮分析
            turns = ingest_beam.parse_turns(path)
            steps.append(
                ingest_beam.harvest_session_candidates(
                    turns=turns, user=user, session_id=session_id, concurrency=CANDIDATE_CONCURRENCY
                )
            )
        outcome = await asyncio.gather(*steps, return_exceptions=True)
        dataset_id = dataset_id or await _dataset_id(cognee, dataset, user)
        done, gone = await _settle(cognee, dataset_id, user)
        dropped += gone
        if path.stem not in done:  # 多半是限流之类的暂时错误：已经删掉了，重新写入一次
            retried.append(str(row["session_id"]))
            outcome[0] = (await asyncio.gather(ingest(), return_exceptions=True))[0]
            dataset_id = dataset_id or await _dataset_id(cognee, dataset, user)
            done, gone = await _settle(cognee, dataset_id, user)
            dropped += gone
        if path.stem not in done:  # 成败看 Cognee 自己记的状态，不看 cognify 有没有抛错
            error = outcome[0] if isinstance(outcome[0], BaseException) else "not cognified"
            failed.append(f"{row['session_id']}: {_brief(error)}"[:300])
            continue
        if ingest_beam is None:
            continue
        if isinstance(outcome[1], BaseException):
            distilled.append({"session": row["session_id"], "error": _brief(outcome[1])})
            continue
        try:
            report = await ingest_beam.distill_session(
                session_id=session_id, dataset_name=dataset, user=user
            )
        except Exception as exc:  # 蒸馏失败不影响会话本身
            report = {"status": "errored", "error": _brief(exc)}
        _, gone = await _settle(cognee, dataset_id, user)  # 经验文档 cognify 失败时同样删掉
        dropped += gone
        keep = ("status", "document_count", "error")  # document_count：发布的经验文档数
        distilled.append({"session": row["session_id"], **{k: report.get(k) for k in keep}})
    if len(failed) * 2 > len(rows):
        raise RuntimeError(f"Cognee 一半以上的会话写入失败：{failed[:3]}")
    stats = await _graph_stats()
    present = stats.pop("_sessions")
    missing = [  # 写入成功、图里却没有它的块的会话（应当为空）
        str(row["session_id"])
        for row in rows
        if int(row["number"]) not in present
        and not any(f.startswith(f"{row['session_id']}:") for f in failed)
    ]
    out = {
        "failed_sessions": failed,
        "retried_sessions": retried,
        "dropped_documents": dropped,
        "sessions_without_chunks": missing,
        "effective_chunk_size": await resolve_chunk_size(MAX_ITEM_TOKENS),
        **stats,
    }
    if distill:
        out["distillation"] = {
            "sessions": len(distilled),
            "lesson_documents": sum(int(d.get("document_count") or 0) for d in distilled),
            "errors": [d for d in distilled if d.get("error")][:10],
        }
    return out


def _brief(error: Any) -> str:
    return f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)


async def _graph_stats() -> dict[str, Any]:
    from cognee.infrastructure.databases.graph import (  # type: ignore[import-not-found, unused-ignore]
        get_graph_engine,
    )

    graph = await get_graph_engine()
    nodes, edges = await graph.get_graph_data()
    types: dict[str, int] = {}
    sessions: set[int] = set()
    for _, props in nodes:
        kind = str((props or {}).get("type", "?"))
        types[kind] = types.get(kind, 0) + 1
        if kind == "DocumentChunk":
            number, turn, _ = split_prefix(str((props or {}).get("text") or ""))
            if number is not None and turn is not None:
                sessions.add(number)
    return {"nodes": len(nodes), "edges": len(edges), "node_types": types, "_sessions": sessions}


class _Searcher:
    """检索子进程：HybridRetriever 取上下文，再用图的索引与出处记录给实体、事实找出处。"""

    def __init__(self, directory: Path) -> None:
        self.cognee = _configure(directory)
        plan = json.loads((directory / MANIFEST).read_text("utf-8"))
        self.dataset = str(plan["dataset"])
        self.index: GraphIndex | None = None
        self.dataset_id: Any = None

    async def _graph(self) -> GraphIndex:
        from cognee.infrastructure.databases.graph import (  # type: ignore[import-not-found, unused-ignore]
            get_graph_engine,
        )
        from cognee.modules.retrieval.hybrid.facts import (  # type: ignore[import-not-found, unused-ignore]
            connection_edge_type_id,
        )

        if self.index is None:
            nodes, edges = await (await get_graph_engine()).get_graph_data()

            def type_id(relation: str, props: Mapping[str, Any]) -> str | None:
                edge = {"relationship_name": relation, "properties": dict(props)}
                found: str | None = connection_edge_type_id(edge)
                return found

            self.index = GraphIndex.build(nodes, edges, type_id)
            self.dataset_id = await _dataset_id(self.cognee, self.dataset)
        return self.index

    async def _evidence(self, edge_ids: Sequence[str], index: GraphIndex) -> dict[str, set[int]]:
        """边（edge_object_id）→ 抽出它的块所在的会话（Cognee 的出处记录）。"""
        from uuid import UUID

        from cognee.modules.provenance.edge_evidence.lookup import (  # type: ignore[import-not-found, unused-ignore]
            get_edge_evidence_records,
        )

        ids = []
        for edge_id in dict.fromkeys(edge_ids):
            with contextlib.suppress(ValueError):
                ids.append(UUID(edge_id))
        if not ids or self.dataset_id is None:
            return {}
        records = await get_edge_evidence_records(
            ids,
            self.dataset_id,
            per_edge_limit=EVIDENCE_PER_EDGE,
            total_limit=EVIDENCE_PER_EDGE * len(ids),
        )
        found: dict[str, set[int]] = {}
        for record in records:
            number = index.chunk_session.get(str(record.chunk_id))
            if number is not None:
                found.setdefault(str(record.edge_id), set()).add(number)
        return found

    async def search(self, query: str) -> dict[str, Any]:
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
        index = await self._graph()
        chunks = [
            {
                "id": result_id(chunk),
                "text": str(payload(chunk).get("text") or ""),
                "score": getattr(chunk, "score", None),
            }
            for chunk in found.get("chunks", [])
        ]
        entities: list[dict[str, Any]] = []
        for entity in found.get("entities", []):
            block = format_entities([{**entity, "edges": []}])  # Cognee 的排版：名称、类型、描述
            if not block:
                continue
            edges = [e for e in entity.get("edges", []) if e.get("text")]
            entities.append(
                {
                    "id": str(entity.get("id")),
                    "header": block.split("\n", 1)[1],
                    "sessions": index.entity_sessions(
                        str(entity.get("id")), str(entity.get("description") or "")
                    ),
                    "edges": [
                        {
                            "text": str(e["text"]),
                            "edge_id": str(e.get("edge_object_id") or ""),
                            "source": str(e.get("source_id") or ""),
                        }
                        for e in edges
                    ],
                }
            )
        facts: list[dict[str, Any]] = []
        for fact in found.get("facts", []):
            if fact.get("text"):
                fact_id = str(fact.get("id"))  # EdgeType 向量行的 ID
                edge_ids, sessions = index.fact_edges(fact_id)
                facts.append(
                    {
                        "id": fact_id,
                        "text": str(fact["text"]),
                        "edge_ids": edge_ids,
                        "sessions": sessions,
                    }
                )
        wanted = [e["edge_id"] for x in entities for e in x["edges"]]
        evidence = await self._evidence(wanted + [i for f in facts for i in f["edge_ids"]], index)
        for entity in entities:
            for edge in entity["edges"]:
                sessions = set(evidence.get(edge.pop("edge_id"), set()))
                source = edge.pop("source")
                if source in index.chunk_session:  # 块 → 实体的包含边
                    sessions.add(index.chunk_session[source])
                edge["sessions"] = sorted(sessions)
        for fact in facts:
            sessions = set(fact.pop("sessions"))
            for edge_id in fact.pop("edge_ids"):
                sessions |= evidence.get(edge_id, set())
            fact["sessions"] = sorted(sessions)
        return {
            "chunks": [c for c in chunks if c["text"]],
            "entities": entities,
            "facts": facts,
            "context_tokens": estimate_tokens(str(context or "")),
        }


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
        searcher = _Searcher(directory)
        protocol.write(json.dumps({"ready": True}) + "\n")
        for line in sys.stdin:
            if not line.strip():
                continue
            request = json.loads(line)
            try:
                reply = {"ok": True, **loop.run_until_complete(searcher.search(request["query"]))}
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
    # 换了 cognee 版本或写入格式的写入不能复用（setup 保证装的就是 COGNEE_VERSION）
    build_tag = f"cognee=={COGNEE_VERSION}|{INGEST_FORMAT}"

    def __init__(self) -> None:
        super().__init__()
        self._workers: OrderedDict[str, Worker] = OrderedDict()

    def setup(self) -> None:
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("cognee")
        except PackageNotFoundError:
            raise RuntimeError(
                f"MC 需要 cognee：用 uv run --with cognee=={COGNEE_VERSION} ... 运行（见模块说明）"
            ) from None
        if installed != COGNEE_VERSION:
            raise RuntimeError(
                f"MC 按 cognee {COGNEE_VERSION} 写的（用到它的内部模块与默认值），装的是 "
                f"{installed}：用 uv run --with cognee=={COGNEE_VERSION} ... 运行"
            )

    def _env(self, directory: Path, url: str, mode: str, distill: bool = False) -> dict[str, str]:
        chat_model = self.llm.settings.roles[self.gateway.role].model
        return worker_env(
            directory,
            url,
            chat_model,
            self.embedder.name,
            self.embedder.dim,
            mode=mode,
            distill=distill,
        )

    def _argv(self, mode: str, directory: Path) -> list[str]:
        return [sys.executable, "-m", __name__, mode, str(directory)]

    # ---------- 写入 ----------

    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        plan = manifest(sessions)
        plan = {"dataset": dataset_name(key), "distill": not plan["two_people"], **plan}
        for row in plan["sessions"]:
            path = directory / row["file"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row["items"], ensure_ascii=False, indent=1), "utf-8")
        (directory / MANIFEST).write_text(json.dumps(plan, ensure_ascii=False), "utf-8")
        url = self.gateway.url(self.bucket(key))
        env = self._env(directory, url, "build", distill=bool(plan["distill"]))
        worker = Worker(self._argv("build", directory), env, directory / WORKER_LOG)
        try:
            reply = worker.receive(BUILD_SECONDS)
        finally:
            worker.stop()
        if not reply.get("ok"):
            raise RuntimeError(f"Cognee 写入失败：{reply.get('error')}")
        items = sum(len(row["items"]) for row in plan["sessions"])
        lessons = {"lesson_ids": LESSON_TAG} if plan["distill"] else {}  # LoCoMo 的记录不变
        return {
            "dataset": plan["dataset"],
            "distill": plan["distill"],
            "chunks_written": items,
            **lessons,
            **reply["stats"],
        }

    def recorded_tag(self, record: Mapping[str, Any]) -> str:
        """做了会话蒸馏、但蒸馏的会话 ID 还带着题号的写入（见 distill_session_id）不能复用：
        算作另一个写入标记，基类就重建（协议不同时基类先报错，不会走到重建）。"""
        tag = super().recorded_tag(record)
        if record.get("distill") and record.get("lesson_ids") != LESSON_TAG:
            return f"{tag}|lesson_ids={record.get('lesson_ids')}"
        return tag

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
        env = self._env(directory, self.gateway.url(self.bucket(key, "search")), "serve")
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
