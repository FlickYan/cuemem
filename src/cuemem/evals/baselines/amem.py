"""外部基线 MA：A-MEM（agiresearch/A-mem，Zettelkasten 式的 agentic memory），N12。

照 A-MEM 论文（arXiv 2502.12110）在 LoCoMo 上的评测代码（作者的复现仓库 WujiangXu/AgenticMemory
的 test_advanced.py 与 memory_layer.py）写入与检索，回答与判分和其他方法完全相同（见
evals/external.py），只换"记忆从哪来"：

- **写入**：每句话一条笔记（`Speaker <名字> says : <原话>`），时间是所在会话的时间（论文直接
  传 LoCoMo 的 date_time 字符串，这里照它的格式写，如 `1:56 pm on 8 May, 2023`），按时间顺序
  逐条 `add_note`。每条笔记先由 LLM 生成关键词、上下文与标签（`analyze_content`，论文的
  Note Construction），再做记忆演化（`process_memory`：取最相近的 5 条笔记，由 LLM 决定是否
  建立链接、改写自己的标签、改写这些邻居的上下文与标签）；每演化 100 次整理一次（重新向量化）。
  每条笔记两次 LLM 调用，只能逐条串行（后一条的演化依赖前面的结果）。
- **检索**：评测代码的做法：先让 LLM 把问题变成几个关键词（`generate_query_llm`，它的提示原样
  照搬），用关键词做向量检索取前 k 条笔记，每条后面带上它链接的笔记（`find_related_memories_raw`）。
  k 默认 10，见下文第 6 条。每条笔记是一条证据：时间取所在会话，正文是论文给回答模型的格式
  （内容、上下文、关键词、标签）；笔记是哪一句话也记下（turn_indices），发言召回等检索指标照算。
- **模型**：A-MEM 的 LLM 请求（照它的 OpenAIController 发）指向评测网关（聊天交给 extractor
  角色的模型）；向量不用 sentence-transformers 的 all-MiniLM-L6-v2，改由网关交给 CueMem 的
  向量器。

与 A-MEM 自身默认及论文评测不同的地方，以及原因：

1. 包：PyPI 上的 agentic-memory 0.1.2 不是 A-MEM（是另一个作者的 ContextOS），这里装 A-MEM
   官方仓库 agiresearch/A-mem，固定到提交 ceffb86。论文的 LoCoMo 数字出自另一个仓库
   （AgenticMemory，不是可安装的包），它的流程在这里用 A-MEM 包的部件照着走。
2. 向量库：A-MEM 的 ChromaRetriever 用进程内的临时 Chroma，同一进程里所有记忆系统共用一个名为
   memories 的集合，而且每构造一个记忆系统就把整个 Chroma 清空——多份对话并行写入会互相覆盖，
   写完也存不下来。这里换成每份对话一个持久化的 Chroma 集合（放在评测空间的目录下，接口与
   ChromaRetriever 相同），笔记本身存成 notes.json，之后的运行直接复用。
3. 笔记构造：这个版本的 `add_note` 不调用 `analyze_content`（关键词、上下文、标签只有调用方
   传进来才有），这里先调用它的 `analyze_content`，结果传给 `add_note`——与论文代码
   （MemoryNote 构造时分析）和作者后来的 A-mem-sys 相同。
4. 向量化的内容：这个版本只向量化笔记原文；论文（式 3）与评测代码向量化"原文 + 上下文 +
   关键词 + 标签"，这里照论文代码写入时的格式（`content:… context:… keywords: … tags: …`）；
   整理时的格式不同，见第 5 条。
5. 演化时邻居的编号：这个版本把 5 个邻居标成"memory index:0…4"（检索结果里的名次），随后
   update_neighbor 按这个名次去改整个记忆库的第 0…4 条（最早写入的几条，而不是这些邻居），
   strengthen 记下的链接也对不上任何笔记。这里照论文代码给邻居标上它们在记忆库里的真实序号，
   演化改的就是这些邻居，链接也能找到（A-mem-sys 用笔记 ID 修了同一个问题）。整理（每 100 次
   演化）在原版里重新加入已有 ID 会被 Chroma 忽略，什么也没更新；这里照论文代码按笔记当前的
   上下文、关键词与标签重新向量化，格式也照论文代码整理时的（`<原文> , <上下文> <关键词>
   <标签>`：不带字段名，关键词与标签以空格分隔），与写入时的不同。之后新写入的笔记仍用写入时
   的格式，所以与论文代码一样，最后一次整理前的笔记是整理的格式、之后的是写入的格式（pq 协议
   的 conv-42：628 次演化整理 6 次，629 条里前 601 条是整理的格式）。A-mem-sys 写入与整理用
   同一种格式，这里照论文代码。
6. k：默认 10，论文的主要设置（"we primarily employ k=10"）与评测代码的默认值。论文表 8 给
   GPT-4o-mini / GPT-4o 调到 40–50（按题型，但最小是 40，统一取 40 用不着题型），作者的 README
   也说要调 k 才能复现论文的数字；deepseek-flash、qwen3.8-flash 与它们同档，所以 k=40 才是作者
   对这一档模型的设置。主结果仍用 10，是为了证据量与其他方法可比：笔记演化后很长，每条又带出
   它链接的笔记。deepseek 预设（回复上限 2000）的试跑里上下文平均约 500 字符、二十个标签，
   conv-42 上 k=10 平均给回答模型约 9k tokens（CueMem L1 约 4k），k=40 约 27k；pq 协议
   （qwen3.8-flash 写的上下文短一半）的前 3 题 k=10 约 2k–8k（v3、v4 写入），k=40 约 12k–23k
   （v2 写入）。
   k=40 作为作者设置的敏感性运行：环境变量
   CUEMEM_AMEM_K=40，运行名另起（见下面的命令）；k 只影响检索，写入照样复用，只多花回答的
   tokens。评测的 --k 管不到 MA（外部基线构造时不带参数）。链接带出的笔记去重（论文代码会重复
   列出），每条最多带 k 条。
7. 回答：论文的评测按题型用不同的回答提示（对抗题甚至把标准答案列成选项之一），这里与所有
   方法一样交给同一个回答模型、同一套提示，A-MEM 只负责检索。检索用的关键词那次 LLM 调用不经
   HTTP，直接用评测的 LLM 客户端在回答线程里发同样的请求（extractor 角色的模型、A-MEM 控制器的
   系统提示与 temperature、网关同样的 JSON 约束与备用模型），这样它算进每道题的用量与
   --max-cost（经网关就只落在网关的桶里）。关键词按"查询"向量化，与网关对检索阶段的桶
   （`amem.search.<空间>`）的做法、CueMem 与 M0 的检索相同：通义的向量区分查询与文档（A-MEM
   自己的 all-MiniLM 不区分；deepseek 预设的 bge-small 也不区分）。
8. 说话人：两人对话（LoCoMo）照论文写 `Speaker <名字> says : …`（论文原文少一个空格，
   "Speaker Carolinesays"，这里补上）；用户与助手的对话（LongMemEval）用 user / assistant 当名字。
9. 结构化输出与回复上限：A-MEM 用 OpenAI 的 json_schema（strict）；网关把它改成 JSON 模式并把
   schema 写进系统提示（DeepSeek 与通义不保证支持 json_schema）。请求照 A-MEM 的
   OpenAIController（系统提示"You must respond with a JSON object."、temperature 0.7），回复
   上限是 4096 tokens：论文代码与这个版本的控制器是 1000，作者现在的 A-mem-sys 不设上限。演化
   的回复要把 5 个邻居的上下文和标签都重写一遍，deepseek-flash 还按行排版 JSON，回复越来越长；
   截断的 JSON 解析不了，A-MEM 就放弃这次演化（deepseek-flash 试跑 conv-42：上限 1000 时 1257 次
   调用有 129 次截断，2000 时 36 次）。4096 实际上就是不设上限，又是各家都接受的值。代价是演化
   成功得多，邻居的上下文被改写得更长：1000 → 2000 时上下文平均 305 → 516 字符，每条证据长约
   40%。
   JSON 模式不保证 schema（strict 的结构化输出保证），所以回复不合 schema 时重新采样，最多两次：
   回复解析不了（被截断）、不是 JSON 对象、或缺了 schema 里列的字段（见 fits）。同样的请求只会从
   网关的缓存里拿回同一个坏回复，所以第 i 次重采样把回复上限减 i（回复远短于上限，不影响结果；
   与 MG 的做法相同）；仍不合就照原样交给 A-MEM，它照旧放弃。pq 协议第一次试跑（不重采样）的
   conv-42 上：14 / 629 次分析回的是 schema 本身（`{"type": "object", "properties": …}`，其中
   6 次把内容包在 properties 里），分析因此为空；5 次演化在 4096 处截断，都是 qwen3.8-flash
   的 JSON 生成失控（actions 里无休止地列数字，D-050 那种，加大上限也没用）；1 次演化回了列表。
   这 6 次演化被放弃。重采样后见文末的试跑。
10. 失败：A-MEM 自己吞掉 LLM 的异常（分析失败就留空，演化失败就不演化），笔记会悄悄缺了分析或
   演化；论文代码里演化的调用出错会直接中断。这里：网关回 5xx、连不上之类的临时错误与空回复
   （DeepSeek 高峰时把请求挂起十几分钟后回空）等 30 秒、2 分钟、5 分钟各重试一次，仍然失败就
   中止整份写入（不写 build.json，下次运行重建）；供应商拒绝的请求（400，如内容审核，重试也
   一样）照 A-MEM 的做法跳过并计数，超过一半才算写入失败。被截断、不是 JSON、不合 schema 的
   回复（每次采样都数）、因此重采样的次数、重采样完仍不能用的调用与分析为空的笔记也计数，写进
   build.json。
11. 网关客户端不设超时，也不让 SDK 重发：网关背后是评测的 LLM 客户端，高峰时段（wait 策略）
   它暂停到高峰结束，A-MEM 与其他方法一样等着；客户端超时只会让 A-MEM 吞掉异常、丢掉分析或
   演化，而 SDK 重发的请求与原请求相同，LLM 客户端会把它合并到还在等的那次调用上，并不会重新
   请求上游。DeepSeek 有时把请求挂起十几分钟（只发空行保活，高峰时更常见），写入就等它；
   挂起后回的空内容按第 10 条重试。
12. 写入缓存：基类按抽取模型与向量模型区分协议，换了（例如从 pds 换到 pq 协议）不覆盖已有的
    写入，直接报错（评测空间的目录不分协议）；用通义的协议要用另一个目录：
    CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen（与 scripts/n12_baselines.py 相同）。写入
    标记（build_tag）是 A-MEM 的提交、回复上限与写入格式的版本，变了旧写入就作废重建；早先它接
    在 build.json 的 protocol 后面（`…|amem@ceffb86|max_tokens=4096|v4`），基类照样认得。

A-MEM 不是 CueMem 的依赖，评测时用 uv 临时装上（--extra local 只在用本机向量时需要，如
deepseek 预设）。它依赖 sentence-transformers，会顺带装上 torch（约 5 GB，这里用不到，但
`uv run --with` 没法只跳过某个依赖）。N12 对比脚本按 `<协议>-<数据集>-<系统>` 归组：

    AMEM=git+https://github.com/agiresearch/A-mem@ceffb860f0712bbae97b184d440df62bc910ca8d
    # pds 协议（只用 DeepSeek）
    CUEMEM_PRESET=deepseek uv run --extra local --with "agentic-memory @ $AMEM" \\
        cuemem eval run --dataset locomo --split test200r --methods MA --name pds-locomo-amem
    # 作者给 GPT-4o 级模型的 k（敏感性运行，复用上面的写入）。运行名不能以 pds-locomo- 开头：
    # 对比脚本会把它归进来，按名字排在 pds-locomo-amem 之后，覆盖 k=10 的结果
    CUEMEM_PRESET=deepseek CUEMEM_AMEM_K=40 uv run --extra local --with "agentic-memory @ $AMEM" \\
        cuemem eval run --dataset locomo --split test200r --methods MA --name k40-pds-locomo-amem
    # pq 协议（只用通义，要 DASHSCOPE_API_KEY）：写入放在另一个目录，不碰 pds 的写入
    CUEMEM_PRESET=qwen CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen \\
        uv run --with "agentic-memory @ $AMEM" cuemem eval run --dataset locomo \\
        --split test200r --methods MA --peak warn --name pq-locomo-amem

试过的版本：agiresearch/A-mem@ceffb86、chromadb 1.5.9（Python 3.12）。conv-42（29 个会话、
629 句）的写入：deepseek 预设、回复上限 2000（第一版）1257 次调用、77 分钟，名义 $1.44（错峰
实际 $0.69）；pq 协议、上限 4096、重采样、整理照论文代码的格式（v4）1257 次 A-MEM 调用，网关
收到 1294 次（37 次重采样：28 次不合 schema、9 次截断，重采样后全都合规），628 次演化全部成功、
没有分析为空，整理 6 次（前 601 条是整理的格式、后 28 条是写入的格式），名义 $0.44（通义没有
错峰折扣，实际同价），从头写约 2.7 小时（每句约 15 秒）。
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

from openai import APIConnectionError, BadRequestError, InternalServerError, RateLimitError

from cuemem.core.models import Hit
from cuemem.errors import ContentFilteredError
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit
from cuemem.evals.gateway import _with_schema
from cuemem.read.l1 import estimate_tokens

REPO = "git+https://github.com/agiresearch/A-mem"
COMMIT = "ceffb860f0712bbae97b184d440df62bc910ca8d"
DEFAULT_K = 10  # 论文的主要设置与评测代码的默认值（见模块说明第 6 条）
K_ENV = "CUEMEM_AMEM_K"  # 改 k（作者给 GPT-4o 级模型的是 40）
MAX_TOKENS = 4096  # A-MEM 的 OpenAIController 是 1000，A-mem-sys 不设上限（见模块说明第 9 条）
# 写入格式的版本：笔记、向量化的文本或演化的替换方法改了就加一，旧写入作废重建
# 2：笔记记下发言下标；回复上限 4096；调用失败时中止写入
# 3：回复不合 schema（解析不了、被截断、回显 schema、不是对象）时重新采样
# 4：整理时照论文代码整理的格式重新向量化（不带字段名，关键词与标签以空格分隔）
BUILD_VERSION = 4
JSON_SYSTEM = "You must respond with a JSON object."  # A-MEM 的 OpenAIController 的系统提示
NOTES = "notes.json"  # 写完后的全部笔记（含演化后的上下文、标签与链接），按写入顺序
CHROMA = "chroma"  # 持久化的 Chroma 目录
COLLECTION = "memories"  # 与 A-MEM 的集合同名
BUILD_LOG = "build.log"  # 写入进度（一份 LoCoMo 对话要写一个多小时）
PROGRESS_EVERY = 50
EMBED_BATCH = 256
MAX_OPEN = 8  # 同时打开的对话（每份一个 Chroma）
# 网关的临时错误（5xx、连不上、限流）重试前等多久（秒）。网关背后的 LLM 客户端已经按 SDK 的
# 退避重试过几次，到这里多半是上游短暂不可用，所以等得更久
RETRY_WAITS = (30.0, 120.0, 300.0)
# 回复不合 schema 时最多再采样几次（见模块说明第 9 条）：第 i 次重采样把回复上限减 i，请求落到
# 网关缓存的新键上（与 MG 的做法相同）
RESAMPLES = 2


class EmptyReplyError(Exception):
    """上游回了空内容：DeepSeek 高峰时会把请求挂起十几分钟（只发空行保活），最后回一个空回复。
    当临时错误重试（空回复不进缓存，重试会重新请求上游）；JSON 模式下模型不会真的回空。"""


TRANSIENT = (APIConnectionError, InternalServerError, RateLimitError, EmptyReplyError)

# 论文评测代码（test_advanced.py 的 generate_query_llm）里把问题变成关键词的提示
QUERY_PROMPT = (
    "Given the following question, generate several keywords, using 'cosmos' as the separator."
    "\n\nQuestion: {question}\n\n"
    'Format your response as a JSON object with a "keywords" field containing the selected '
    "text.\n\nExample response format:\n"
    '{{"keywords": "keyword1, keyword2, keyword3"}}'
)  # 照搬原文，只去掉了缩进
QUERY_SCHEMA: dict[str, Any] = {
    "name": "response",
    "schema": {
        "type": "object",
        "properties": {"keywords": {"type": "string"}},
        "required": ["keywords"],
        "additionalProperties": False,
    },
    "strict": True,
}
_INDEX = re.compile(r"^\s*(?:memory\s*index\s*[:#]?\s*)?(\d+)\s*$", re.IGNORECASE)


# ---------- 纯函数（不依赖 A-MEM，单独测试） ----------


@dataclass
class Note:
    """写完后的一条 A-MEM 笔记，外加它是哪个会话的第几句话（证据的日期、会话号与检索指标）。"""

    id: str
    content: str
    timestamp: str
    session_id: str
    session_time: datetime
    turn_index: int = -1  # 在会话里的发言下标（与数据集相同，空话也占位）；-1 是旧写入，不知道
    keywords: list[str] = field(default_factory=list)
    context: str = "General"
    tags: list[str] = field(default_factory=list)
    links: list[Any] = field(default_factory=list)  # 演化时 LLM 给的链接，原样保存，用时再解析
    category: str = "Uncategorized"

    def as_json(self) -> dict[str, Any]:
        return {**asdict(self), "session_time": self.session_time.isoformat()}

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Note:
        return cls(
            id=str(row["id"]),
            content=str(row["content"]),
            timestamp=str(row.get("timestamp", "")),
            session_id=str(row.get("session_id", "")),
            session_time=datetime.fromisoformat(str(row["session_time"])),
            turn_index=int(row.get("turn_index", -1)),
            keywords=strings(row.get("keywords")),
            context=str(row.get("context") or "General"),
            tags=strings(row.get("tags")),
            links=list(row.get("links") or []),
            category=str(row.get("category") or "Uncategorized"),
        )


def top_k() -> int:
    """检索的 k：默认 10，环境变量 CUEMEM_AMEM_K 可以改（见模块说明第 6 条）。"""
    raw = os.environ.get(K_ENV, "").strip()
    if not raw:
        return DEFAULT_K
    if not raw.isdigit() or int(raw) <= 0:
        raise ValueError(f"{K_ENV} 要是正整数，现在是 {raw!r}")
    return int(raw)


def strings(value: Any) -> list[str]:
    """LLM 给的关键词或标签 → 字符串列表（偶尔是一个逗号分隔的字符串）。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def when_text(moment: datetime) -> str:
    """会话时间 → LoCoMo 的 date_time 格式（论文把这个字符串原样当作笔记的时间）。"""
    hour = moment.hour % 12 or 12
    half = "am" if moment.hour < 12 else "pm"
    return f"{hour}:{moment.minute:02d} {half} on {moment.day} {moment:%B}, {moment.year}"


def note_content(turn: EvalTurn) -> str:
    """一句话 → 一条笔记的原文（论文评测代码的格式，补上它少的空格）。"""
    speaker = turn.speaker or turn.role
    return f"Speaker {speaker} says : {turn.text.strip()}"


def clean_analysis(raw: Any) -> dict[str, Any]:
    """analyze_content 的结果 → add_note 的参数（类型不对的字段丢掉，留给 A-MEM 用默认值）。"""
    data = raw if isinstance(raw, Mapping) else {}
    context = data.get("context")
    if isinstance(context, list):  # 论文代码也这样处理：列表拼成一句
        context = " ".join(str(c) for c in context)
    return {
        "keywords": strings(data.get("keywords")),
        "context": str(context).strip() if context else "General",
        "tags": strings(data.get("tags")),
    }


def note_document(content: str, context: Any, keywords: Any, tags: Any) -> str:
    """向量化的文本：论文 LoCoMo 代码里 add_note 交给检索器的格式（原文、上下文、关键词、标签）。"""
    return (
        f"content:{content} context:{context or 'General'} "
        f"keywords: {', '.join(strings(keywords))} tags: {', '.join(strings(tags))}"
    )


def consolidated_document(content: str, context: Any, keywords: Any, tags: Any) -> str:
    """整理时重新向量化的文本：论文代码 consolidate_memories 的格式（原文、" , "，再接上下文、
    关键词、标签，不带字段名，关键词与标签以空格分隔）。与 note_document 不同（见模块说明
    第 5 条）。"""
    return (
        f"{content} , {context or 'General'} "
        f"{' '.join(strings(keywords))} {' '.join(strings(tags))}"
    )


def link_targets(links: Sequence[Any], ids: Mapping[str, int], count: int, own: int) -> list[int]:
    """演化时记下的链接 → 笔记序号。LLM 给的可能是序号（12、"12"、"memory index:12"）或笔记 ID；
    超出范围、指向自己或重复的丢掉。"""
    found: list[int] = []
    for link in links:
        index: int | None = None
        if isinstance(link, bool):
            continue
        if isinstance(link, int):
            index = link
        elif isinstance(link, str):
            match = _INDEX.match(link)
            index = int(match.group(1)) if match else ids.get(link.strip())
        if index is not None and 0 <= index < count and index != own and index not in found:
            found.append(index)
    return found


def expand(
    ranked: Sequence[int], notes: Sequence[Note], ids: Mapping[str, int], k: int
) -> list[tuple[int, bool]]:
    """论文评测代码的 find_related_memories_raw：前 k 条笔记，每条后面跟着它链接的笔记
    （每条最多 k 条）。去重（原代码会重复列出）。返回（笔记序号，是否是链接带出的）。"""
    picked: list[tuple[int, bool]] = []
    seen: set[int] = set()
    for index in ranked[:k]:
        if index in seen:
            continue
        picked.append((index, False))
        seen.add(index)
        for neighbor in link_targets(notes[index].links, ids, len(notes), index)[:k]:
            if neighbor not in seen:
                picked.append((neighbor, True))
                seen.add(neighbor)
    return picked


def note_text(note: Note) -> str:
    """交给回答模型的一条证据：论文评测代码的格式（时间由证据的抬头给出）。"""
    return (
        f"memory content: {note.content}\n"
        f"memory context: {note.context}\n"
        f"memory keywords: {', '.join(note.keywords)}\n"
        f"memory tags: {', '.join(note.tags)}"
    )


def to_hits(
    space: str,
    notes: Sequence[Note],
    picked: Sequence[tuple[int, bool]],
    scores: Mapping[int, float],
) -> list[Hit]:
    hits: list[Hit] = []
    for rank, (index, linked) in enumerate(picked, start=1):
        note = notes[index]
        hit = memory_hit(
            space,
            note.id,
            note_text(note),
            note.session_time,
            rank,
            score=scores.get(index, 0.0),
            route="amem_link" if linked else "amem",
            session_id=note.session_id,
            kind="round",
        )
        if note.turn_index >= 0:  # 检索指标用：一条笔记就是一句话
            hit.unit.meta["turn_indices"] = [note.turn_index]
        hits.append(hit)
    return hits


def round_starts(session: EvalSession) -> list[int]:
    """每句话所在"轮"的第一句的下标。与 CueMem 的 build_rounds 相同：用户发言开启新的一轮，
    其后的助手回复并入该轮（LoCoMo 两人都是 user，每句各成一轮）。"""
    starts: list[int] = []
    for index, turn in enumerate(session.turns):
        starts.append(index if turn.role == "user" or not starts else starts[-1])
    return starts


def merge_rounds(hits: Sequence[Hit], sessions: Sequence[EvalSession]) -> None:
    """检索指标以 CueMem 的"轮"为单位，A-MEM 的笔记却是一句话：同一轮的几句（LongMemEval 的
    用户发言与助手回复）都被取到时，发言下标并到排得最前的那条上，后面的不再计——否则同一轮
    算两次，NDCG 会超过 1。只并真正取到的句子，没取到的半轮不算命中。LoCoMo 不受影响。"""
    starts = {session.session_id: round_starts(session) for session in sessions}
    first: dict[tuple[str, int], Hit] = {}
    for hit in hits:
        indices = hit.unit.meta.get("turn_indices") or []
        if not indices:
            continue
        index = int(indices[0])
        rounds = starts.get(hit.unit.session_id, [])
        key = (hit.unit.session_id, rounds[index] if 0 <= index < len(rounds) else index)
        if key in first:
            kept = first[key].unit.meta
            kept["turn_indices"] = sorted({*kept["turn_indices"], index})
            hit.unit.meta["turn_indices"] = []
        else:
            first[key] = hit


def parse_keywords(response: str | None, question: str) -> str:
    """关键词生成的回复 → 检索用的文本（评测代码：取 JSON 的 keywords，解析不了就用原文）。"""
    text = (response or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, Mapping) and "keywords" in data:
        keywords = data["keywords"]
        text = ", ".join(strings(keywords)) if isinstance(keywords, list) else str(keywords)
    return text.strip() or question


def json_schema_of(response_format: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """A-MEM 请求里的 response_format → 要的 JSON schema（只要求 JSON 对象时是空 schema）。"""
    if not response_format:
        return None
    if response_format.get("type") == "json_schema":
        wrapper = response_format.get("json_schema") or {}
        return dict(wrapper.get("schema") or wrapper)
    return {} if response_format.get("type") == "json_object" else None


def fits(content: str | None, schema: Mapping[str, Any]) -> bool:
    """回复能不能当作 OpenAI 结构化输出（strict）给的结果：是 JSON 对象，schema 里列的字段都在。

    网关把 json_schema 改成 JSON 模式，模型就不保证照 schema 回：被截断（qwen3.8-flash 偶尔生成
    失控，D-050）、把 schema 原样回显成 `{"type": "object", "properties": …}`、回一个列表。A-MEM
    对这些一律放弃（分析留空、不演化）。字段的值不细查：A-MEM 自己与这里的整理都容忍类型出入
    （如链接给成整数序号、关键词给成一个字符串）。"""
    try:
        data = json.loads(content or "")
    except json.JSONDecodeError:
        return False
    return shaped(data, schema)


def shaped(data: Any, schema: Mapping[str, Any]) -> bool:
    """解析好的回复是 JSON 对象，schema 里列的字段都在（见 fits）。"""
    return isinstance(data, dict) and all(key in data for key in schema.get("properties") or {})


def save_notes(path: Path, notes: Sequence[Note]) -> None:
    rows = [note.as_json() for note in notes]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")


def load_notes(path: Path) -> list[Note]:
    return [Note.from_json(row) for row in json.loads(path.read_text("utf-8"))]


# ---------- A-MEM 的检索器与 LLM 控制器换成走评测网关的版本 ----------


class _Controller:
    """代替 A-MEM 的 OpenAIController：同样的请求（系统提示、response_format、temperature），
    客户端指向评测网关，回复上限是 MAX_TOKENS。

    A-MEM 自己会吞掉异常，所以这里先处理再交给它：临时错误按 waits 重试；仍然失败的记进
    failed（写入循环见到就中止整份写入），供应商拒绝的（400）记进 rejected。回复不合 schema
    时重新采样，最多 resamples 次（见模块说明第 9 条），仍不合就照原样交给 A-MEM（它会放弃）。
    顺带数调用、被截断、不是 JSON 与不合 schema 的回复（每次采样都数）。"""

    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = MAX_TOKENS,
        *,
        waits: Sequence[float] = RETRY_WAITS,
        sleep: Callable[[float], None] = time.sleep,
        resamples: int = RESAMPLES,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.waits = tuple(waits)
        self.sleep = sleep
        self.resamples = resamples
        self.calls = 0
        self.failed = 0
        self.rejected = 0
        self.retries = 0
        self.empty_replies = 0
        self.truncated = 0
        self.invalid_json = 0
        self.schema_mismatch = 0  # 是 JSON，却不是对象或缺了 schema 里的字段（如回显 schema）
        self.resampled = 0  # 因此多发的请求
        self.unusable = 0  # 重采样完仍不合 schema、照原样交给 A-MEM 的调用
        self.errors: list[str] = []
        self._lock = threading.Lock()

    def get_completion(
        self, prompt: str, response_format: dict[str, Any] | None = None, temperature: float = 0.7
    ) -> str:
        with self._lock:
            self.calls += 1
        schema = json_schema_of(response_format)
        content = ""
        for attempt in range(self.resamples + 1):
            if attempt:
                with self._lock:
                    self.resampled += 1
            # 第 i 次重采样把上限减 i：同样的请求只会从网关的缓存里拿回同一个坏回复
            content = self._sample(prompt, response_format, temperature, self.max_tokens - attempt)
            if schema is None or fits(content, schema):
                return content
        with self._lock:
            self.unusable += 1
        return content

    def _sample(
        self,
        prompt: str,
        response_format: dict[str, Any] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        extra: dict[str, Any] = {"response_format": response_format} if response_format else {}

        def send() -> Any:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JSON_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            if not str(response.choices[0].message.content or "").strip():
                with self._lock:
                    self.empty_replies += 1
                raise EmptyReplyError(f"empty reply ({response.choices[0].finish_reason})")
            return response

        choice = self.attempt(send).choices[0]
        content = str(choice.message.content or "")
        with self._lock:
            self.truncated += choice.finish_reason == "length"
        schema = json_schema_of(response_format)
        if schema is not None:
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                with self._lock:
                    self.invalid_json += 1
            else:
                if not shaped(data, schema):
                    with self._lock:
                        self.schema_mismatch += 1
        return content

    def attempt(self, call: Callable[[], Any]) -> Any:
        """调一次网关（聊天或向量）。临时错误（含空回复）等一会儿重试；最后的错误记下来再抛出。"""
        for wait in (*self.waits, None):
            try:
                return call()
            except TRANSIENT as exc:
                if wait is None:
                    self._record(exc, rejected=False)
                    raise
                with self._lock:
                    self.retries += 1
                self.sleep(wait)
            except Exception as exc:
                self._record(exc, rejected=isinstance(exc, BadRequestError))
                raise
        raise AssertionError("unreachable")  # pragma: no cover

    def check(self) -> None:
        """写入循环每调用一次 A-MEM 就检查：有调用最终失败（A-MEM 已经吞掉了异常）就中止写入。"""
        with self._lock:
            failed, rejected, calls, errors = self.failed, self.rejected, self.calls, self.errors
        if failed:
            raise RuntimeError(
                f"A-MEM 的 {failed} 次调用失败（A-MEM 会吞掉异常，留下缺分析或缺演化的笔记），"
                f"整份写入作废：{errors}"
            )
        if calls >= 20 and rejected > calls / 2:
            raise RuntimeError(f"A-MEM 的 LLM 调用一多半被拒绝：{errors}")

    def _record(self, exc: Exception, *, rejected: bool) -> None:
        with self._lock:
            if rejected:
                self.rejected += 1
            else:
                self.failed += 1
            if len(self.errors) < 5:
                self.errors.append(f"{type(exc).__name__}: {exc}"[:200])


class _Store:
    """A-MEM 的 ChromaRetriever 的替身：每份对话一个持久化的 Chroma 集合，向量由调用方给出
    （写入时走评测网关；检索时由回答线程算好查询向量再交给 nearest）。方法与 ChromaRetriever
    同名，A-MEM 的写入流程照常调用。"""

    def __init__(
        self, directory: Path, embed: Callable[[list[str]], list[list[float]]] | None = None
    ) -> None:
        import chromadb  # type: ignore[import-not-found, unused-ignore]
        from chromadb.config import Settings  # type: ignore[import-not-found, unused-ignore]

        directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(directory), settings=Settings(anonymized_telemetry=False)
        )
        # 不给 Chroma 向量函数（否则它会用自带的 ONNX 模型）：向量都由 embed 算好再交给它
        self.collection = self.client.get_or_create_collection(
            COLLECTION, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )
        self._embed = embed
        self.consolidations = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed is None:
            raise RuntimeError("这个 Chroma 集合只读（检索时打开），不能向量化")
        return self._embed(texts)

    def add_document(self, document: str, metadata: Mapping[str, Any], doc_id: str) -> None:
        """A-MEM 的 add_note 交来笔记原文与元数据：照论文代码写入时的格式向量化。"""
        text = note_document(
            document, metadata.get("context"), metadata.get("keywords"), metadata.get("tags")
        )
        self.upsert([(doc_id, text, metadata)])

    def upsert(self, rows: Sequence[tuple[str, str, Mapping[str, Any]]]) -> None:
        """（ID，要向量化的文本，元数据）：文本已按写入或整理的格式拼好，元数据只存时间。"""
        for start in range(0, len(rows), EMBED_BATCH):
            batch = rows[start : start + EMBED_BATCH]
            texts = [text for _, text, _ in batch]
            self.collection.upsert(
                ids=[doc_id for doc_id, _, _ in batch],
                embeddings=self.embed(texts),
                documents=texts,
                metadatas=[{"timestamp": str(meta.get("timestamp", ""))} for _, _, meta in batch],
            )

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5) -> dict[str, Any]:
        """与 ChromaRetriever.search 相同的返回形状（ids / distances 是一层嵌套的列表）。"""
        if int(self.collection.count()) == 0 or k <= 0:
            return {"ids": [[]], "distances": [[]]}
        return self.nearest(self.embed([query])[0], k)

    def nearest(self, vector: list[float], k: int) -> dict[str, Any]:
        count = int(self.collection.count())
        if count == 0 or k <= 0:
            return {"ids": [[]], "distances": [[]]}
        result = self.collection.query(
            query_embeddings=[vector], n_results=min(k, count), include=["distances"]
        )
        return {"ids": result["ids"], "distances": result["distances"]}

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


class _Detached:
    """构造 AgenticMemorySystem 时代替 ChromaRetriever：原版会连上进程内共享的 Chroma 并把它
    清空，还会加载 sentence-transformers 模型。构造完马上换成 _Store。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.client = SimpleNamespace(reset=lambda: None)


def _find_related_memories(system: Any, query: str, k: int = 5) -> tuple[str, list[int]]:
    """替换 A-MEM 的 find_related_memories：邻居标上在记忆库里的真实序号（论文代码的做法），
    process_memory 按这个序号改邻居、记链接。格式照原版（memory index、talk start time …）。"""
    if not system.memories:
        return "", []
    position = {note_id: i for i, note_id in enumerate(system.memories)}
    notes = list(system.memories.values())
    text = ""
    indices: list[int] = []
    for note_id in system.retriever.search(query, k)["ids"][0]:
        index = position.get(note_id)
        if index is None:
            continue
        note = notes[index]
        text += (
            f"memory index:{index}\ttalk start time:{note.timestamp}\t"
            f"memory content: {note.content}\tmemory context: {note.context}\t"
            f"memory keywords: {note.keywords!s}\tmemory tags: {note.tags!s}\n"
        )
        indices.append(index)
    return text, indices


def _consolidate_memories(system: Any) -> None:
    """替换 A-MEM 的 consolidate_memories：全部笔记按当前的上下文、关键词与标签重新向量化，用
    论文代码整理时的格式（consolidated_document，见模块说明第 5 条）。"""
    rows = [
        (
            note.id,
            consolidated_document(note.content, note.context, note.keywords, note.tags),
            {"timestamp": note.timestamp},
        )
        for note in system.memories.values()
    ]
    system.retriever.upsert(rows)
    system.retriever.consolidations += 1


@dataclass
class _Space:
    """检索时打开的一份对话：笔记、ID → 序号、只读的 Chroma 集合。"""

    notes: list[Note]
    ids: dict[str, int]
    store: _Store


class AMemBaseline(ExternalBaseline):
    name, label, system = "MA", "外部基线：A-MEM（agiresearch/A-mem，agentic memory）", "amem"
    # A-MEM 写入的参数（见模块说明第 12 条）
    build_tag = f"amem@{COMMIT[:7]}|max_tokens={MAX_TOKENS}|v{BUILD_VERSION}"

    def __init__(self) -> None:
        super().__init__()
        self._spaces: OrderedDict[str, _Space] = OrderedDict()
        self.k = top_k()  # 构造时就读：值不对时 build_methods 直接报错
        if self.k != DEFAULT_K:
            self.label = f"{self.label}，k={self.k}"

    def setup(self) -> None:
        """导入 A-MEM 前关掉 Chroma 的遥测与 LiteLLM 联网取价格表；构造时不碰共享的 Chroma。"""
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        try:
            import agentic_memory.memory_system as amem  # type: ignore[import-not-found, unused-ignore]
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise RuntimeError(
                f'MA 需要 A-MEM：用 uv run --with "agentic-memory @ {REPO}@{COMMIT}" ... 运行'
                "（见模块说明）"
            ) from exc
        amem.ChromaRetriever = _Detached
        self.amem = amem
        self.model = self.llm.settings.roles[self.gateway.role].model

    # ---------- 网关客户端 ----------

    def _embedder(self, url: str, llm: _Controller) -> Callable[[list[str]], list[list[float]]]:
        from openai import OpenAI

        # 不设超时、不让 SDK 重发：见模块说明第 11 条；临时错误由 llm.attempt 重试并记账
        client = OpenAI(base_url=url, api_key="unused", timeout=None, max_retries=0)
        name = self.embedder.name

        def embed(texts: list[str]) -> list[list[float]]:
            reply = llm.attempt(lambda: client.embeddings.create(model=name, input=texts))
            return [row.embedding for row in sorted(reply.data, key=lambda row: row.index)]

        return embed

    def _controller(self, url: str) -> _Controller:
        from openai import OpenAI

        client = OpenAI(base_url=url, api_key="unused", timeout=None, max_retries=0)
        return _Controller(client, self.model)

    # ---------- 写入 ----------

    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        url = self.gateway.url(self.bucket(key))
        llm = self._controller(url)
        store = _Store(directory / CHROMA, self._embedder(url, llm))
        system = self.amem.AgenticMemorySystem(
            model_name=self.embedder.name, llm_backend="openai", llm_model=self.model, api_key="x"
        )
        system.retriever = store
        system.llm_controller.llm = llm
        system.find_related_memories = MethodType(_find_related_memories, system)
        system.consolidate_memories = MethodType(_consolidate_memories, system)
        ordered = sorted(sessions, key=lambda s: s.started_at)
        total = sum(1 for s in ordered for t in s.turns if t.text.strip())
        origin: list[tuple[str, EvalSession, int]] = []
        empty = 0  # 分析为空的笔记（在这里数：演化会改写邻居的上下文，写完再数就数不出来）
        started = datetime.now()
        log = directory / BUILD_LOG
        try:
            for session in ordered:
                when = when_text(session.started_at)
                for index, turn in enumerate(session.turns):
                    if not turn.text.strip():
                        continue
                    content = note_content(turn)
                    analysis = clean_analysis(system.analyze_content(content))
                    llm.check()
                    if not analysis["keywords"] and analysis["context"] == "General":
                        empty += 1
                    origin.append((system.add_note(content, time=when, **analysis), session, index))
                    llm.check()
                    if len(origin) % PROGRESS_EVERY == 0 or len(origin) == total:
                        elapsed = (datetime.now() - started).total_seconds()
                        with log.open("a", encoding="utf-8") as out:
                            out.write(
                                f"{datetime.now():%H:%M:%S} {len(origin)}/{total} notes, "
                                f"{system.evo_cnt} evolved, {llm.rejected} rejected calls, "
                                f"{llm.retries} retries, {llm.resampled} resampled, "
                                f"{llm.unusable} unusable, {elapsed:.0f}s\n"
                            )
            notes = [
                _note(system.memories[note_id], session, index)
                for note_id, session, index in origin
            ]
        finally:
            store.close()
        save_notes(directory / NOTES, notes)
        ids = {note.id: i for i, note in enumerate(notes)}
        links = [link_targets(n.links, ids, len(notes), i) for i, n in enumerate(notes)]
        return {
            "memories": len(notes),
            "evolved": int(system.evo_cnt),
            "linked_notes": sum(1 for found in links if found),
            "links": sum(len(found) for found in links),
            "unresolved_links": sum(len(n.links) for n in notes) - sum(map(len, links)),
            "empty_analyses": empty,
            "consolidations": store.consolidations,
            "amem_calls": llm.calls,
            "rejected_calls": llm.rejected,
            "retries": llm.retries,
            "empty_replies": llm.empty_replies,
            # 以下三项按每次采样数；resampled 是因此多发的请求，unusable_replies 是重采样完仍
            # 不合 schema、被 A-MEM 放弃的调用
            "truncated": llm.truncated,
            "invalid_json": llm.invalid_json,
            "schema_mismatch": llm.schema_mismatch,
            "resampled": llm.resampled,
            "unusable_replies": llm.unusable,
            "error_samples": llm.errors,  # 不能叫 errors：基类会用网关桶里的错误计数覆盖它
            "max_tokens": llm.max_tokens,
            "amem_commit": COMMIT,
        }

    # ---------- 检索 ----------

    def retrieve(
        self, key: str, directory: Path, sample: Sample, sessions: Sequence[EvalSession]
    ) -> tuple[list[Hit], dict[str, Any]]:
        space = self._open(key, directory)
        query, query_model = self._keywords(key, sample.question)
        # 查询向量也在本线程算：与网关对检索阶段的桶的做法相同（kind="query"），用量记进这道题
        vector = self.embedder.embed([query], "query").dense[0].tolist()
        found = space.store.nearest(vector, self.k)
        ranked: list[int] = []
        scores: dict[int, float] = {}
        for note_id, distance in zip(found["ids"][0], found["distances"][0], strict=True):
            index = space.ids.get(note_id)
            if index is not None:
                ranked.append(index)
                scores[index] = round(1.0 - float(distance), 6)  # 余弦距离 → 相似度
        picked = expand(ranked, space.notes, space.ids, self.k)
        hits = to_hits(key, space.notes, picked, scores)
        merge_rounds(hits, sessions)
        return hits, {
            "amem_k": self.k,
            "amem_query": query,
            "amem_query_model": query_model,
            "amem_retrieved": len(ranked),
            "amem_linked": sum(1 for _, linked in picked if linked),
            "amem_tokens": sum(estimate_tokens(h.unit.text) for h in hits),
        }

    def _keywords(self, key: str, question: str) -> tuple[str, str]:
        """论文评测代码的 generate_query_llm：请求与写入时相同（A-MEM 控制器的系统提示与
        temperature，JSON 约束照网关写进系统提示），但在回答线程里发，用量算进这道题（见模块说明
        第 7 条）。选模型也与网关相同：内容审核误拒、超时或 JSON 生成失控时换备用模型（通义预设是
        qwen3.7-plus，D-050）。回复不合 schema 时与写入一样重新采样（见模块说明第 9 条）。备用
        模型也被内容审核拒绝时照评测代码的兜底，用问题原文检索。
        返回（检索用的文本，回答它的模型别名）。"""
        messages = _with_schema(
            [
                {"role": "system", "content": JSON_SYSTEM},
                {"role": "user", "content": QUERY_PROMPT.format(question=question)},
            ],
            QUERY_SCHEMA,
        )
        primary = self.llm.settings.roles[self.gateway.role].model
        content, alias = "", ""
        for attempt in range(RESAMPLES + 1):
            kwargs: dict[str, Any] = {
                "temperature": 0.7,
                "max_tokens": MAX_TOKENS - attempt,  # 重采样落到缓存的新键上
                "response_format": {"type": "json_object"},
                "cache_scope": f"gateway|{self.bucket(key, 'search')}",
            }
            try:
                # 网关自己的选模型逻辑；不走 gateway.chat，因为它按"当前 trace"给桶记账，在
                # 回答线程里会把这道题此前的用量也记进检索的桶
                result, alias, _ = self.gateway._call(messages, kwargs, primary)
            except ContentFilteredError:
                return question, ""
            content = result.content or ""
            if fits(content, QUERY_SCHEMA["schema"]):
                break
        return parse_keywords(content, question), alias

    def _open(self, key: str, directory: Path) -> _Space:
        """打开（或复用）一份写好的对话；最多同时开 MAX_OPEN 个，多了关掉最久没用的。"""
        with self._lock:
            if key in self._spaces:
                self._spaces.move_to_end(key)
                return self._spaces[key]
        notes = load_notes(directory / NOTES)
        space = _Space(
            notes=notes,
            ids={note.id: i for i, note in enumerate(notes)},
            store=_Store(directory / CHROMA),
        )
        idle: list[_Space] = []
        with self._lock:
            self._spaces[key] = space
            while len(self._spaces) > MAX_OPEN:
                old_key = next(iter(self._spaces))
                if self._space_locks.get(old_key, threading.Lock()).locked():
                    break  # 正在用：留着，下次再关
                idle.append(self._spaces.pop(old_key))
        for old in idle:
            old.store.close()
        return space

    def close(self, key: str) -> None:
        with self._lock:
            space = self._spaces.pop(key, None)
        if space is not None:
            space.store.close()


def _note(note: Any, session: EvalSession, turn_index: int) -> Note:
    """A-MEM 的 MemoryNote（演化之后的状态）→ 要保存的笔记。"""
    return Note(
        id=str(note.id),
        content=str(note.content),
        timestamp=str(note.timestamp),
        session_id=session.session_id,
        session_time=session.started_at,
        turn_index=turn_index,
        keywords=strings(note.keywords),
        context=str(note.context or "General"),
        tags=strings(note.tags),
        links=list(note.links or []),
        category=str(note.category or "Uncategorized"),
    )
