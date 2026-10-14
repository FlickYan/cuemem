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
  k 取 10，见下文第 6 条。每条笔记是一条证据：时间取所在会话，正文是论文给回答模型的格式
  （内容、上下文、关键词、标签）。
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
   关键词 + 标签"，这里照论文代码的格式（`content:… context:… keywords: … tags: …`）。
5. 演化时邻居的编号：这个版本把 5 个邻居标成"memory index:0…4"（检索结果里的名次），随后
   update_neighbor 按这个名次去改整个记忆库的第 0…4 条（最早写入的几条，而不是这些邻居），
   strengthen 记下的链接也对不上任何笔记。这里照论文代码给邻居标上它们在记忆库里的真实序号，
   演化改的就是这些邻居，链接也能找到（A-mem-sys 用笔记 ID 修了同一个问题）。整理（每 100 次
   演化）在原版里重新加入已有 ID 会被 Chroma 忽略，什么也没更新；这里照论文代码按笔记当前的
   上下文与标签重新向量化。
6. k：取 10，论文的主要设置（"we primarily employ k=10"），也是评测代码的默认值，论文用过的
   最小值。论文表 8 给 GPT-4o-mini / GPT-4o 按题型调到 40 或 50，但任何方法都拿不到题型；而且
   笔记演化后很长（上下文平均约 500 字符、二十几个标签），每条又带出它链接的笔记（平均 3.4 条）：
   试跑的 conv-42 上 k=10 平均给回答模型 23 条笔记、约 9.4k tokens（4.7k–14.6k），对话原文
   只有这 23 句，其余是 A-MEM 自己生成的上下文、关键词与标签——已经是 CueMem L1（约 4k）的
   两倍多，k=40 会到约 27k。所以不再往上调，也不截短。链接带出的笔记去重（论文代码会重复
   列出），每条最多带 k 条。
7. 回答：论文的评测按题型用不同的回答提示（对抗题甚至把标准答案列成选项之一），这里与所有
   方法一样交给同一个回答模型、同一套提示，A-MEM 只负责检索。检索用的关键词生成那次调用记在
   网关的 search 桶里，用量写进每道题的 details（amem_search）。
8. 说话人：两人对话（LoCoMo）照论文写 `Speaker <名字> says : …`（论文原文少一个空格，
   "Speaker Carolinesays"，这里补上）；用户与助手的对话（LongMemEval）用 user / assistant 当名字。
9. 结构化输出：A-MEM 用 OpenAI 的 json_schema（strict）；网关把它改成 JSON 模式并把 schema
   写进系统提示（DeepSeek 与通义不保证支持 json_schema）。请求照 A-MEM 的 OpenAIController
   （系统提示"You must respond with a JSON object."、temperature 0.7），只有回复上限从 1000
   放宽到 2000 tokens：演化的回复要把 5 个邻居的上下文和标签都重写一遍，deepseek-flash 还按行
   排版 JSON，邻居的上下文越改越长，回复也越来越长。截断的 JSON 解析不了，A-MEM 就放弃这次
   演化。试跑 conv-42（1257 次调用）：上限 1000 时 129 次被截断，629 条笔记只有 502 条演化；
   上限 2000 时还有 36 次被截断（演化回复的中位数约 1100 tokens），592 条演化。再放宽，
   上下文会接着变长，写入更慢更贵，这里停在 2000。
10. 失败：A-MEM 自己吞掉 LLM 的异常（分析失败就留空，演化失败就不演化）。这里按对话统计失败的
   调用、被截断的回复与不是 JSON 的回复，写进 build.json；超过一半的调用失败时整份写入算失败，
   不留缓存。

A-MEM 不是 CueMem 的依赖，评测时用 uv 临时装上（--extra local 只在用本机向量时需要，如
deepseek 预设）。它依赖 sentence-transformers，会顺带装上 torch（约 5 GB，这里用不到，但
`uv run --with` 没法只跳过某个依赖）：

    AMEM=git+https://github.com/agiresearch/A-mem@ceffb860f0712bbae97b184d440df62bc910ca8d
    uv run --extra local --with "agentic-memory @ $AMEM" \\
        cuemem eval run --dataset locomo --split test200r --methods MA --name locomo-test200r-amem

（N12 对比脚本按 `<协议>-<数据集>-<系统>` 归组，协议运行可命名为 pds-locomo-amem。）
试过的版本：agiresearch/A-mem@ceffb86、chromadb 1.5.9（Python 3.12）。试跑（deepseek 预设）：
conv-42（29 个会话、629 句）写入 1257 次调用、77 分钟，实际 $0.69（名义 $1.44）。
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

from cuemem.core.models import Hit
from cuemem.evals.datasets import EvalSession, EvalTurn, Sample
from cuemem.evals.external import ExternalBaseline, memory_hit
from cuemem.read.l1 import estimate_tokens

REPO = "git+https://github.com/agiresearch/A-mem"
COMMIT = "ceffb860f0712bbae97b184d440df62bc910ca8d"
TOP_K = 10  # 论文的主要设置与评测代码的默认值（见模块说明第 6 条）
MAX_TOKENS = 2000  # A-MEM 的 OpenAIController 是 1000：演化的回复常被截断（见模块说明第 9 条）
JSON_SYSTEM = "You must respond with a JSON object."  # A-MEM 的 OpenAIController 的系统提示
NOTES = "notes.json"  # 写完后的全部笔记（含演化后的上下文、标签与链接），按写入顺序
CHROMA = "chroma"  # 持久化的 Chroma 目录
COLLECTION = "memories"  # 与 A-MEM 的集合同名
BUILD_LOG = "build.log"  # 写入进度（一份 LoCoMo 对话要写一个多小时）
PROGRESS_EVERY = 50
EMBED_BATCH = 256
MAX_OPEN = 8  # 同时打开的对话（每份一个 Chroma）
# 等网关回复的时限（秒）。DeepSeek 偶尔把一个非流式请求挂起很久，期间只发空行保活，LLM 客户端的
# 读超时等不到；网关在上游返回前一个字节也不发，这里超时后 OpenAI SDK 重发（最多两次）。
# 正常的调用几秒就回来（试跑里 conv-42 有一次挂了 10 分钟，整份写入都在等它）
GATEWAY_TIMEOUT = 180.0

# 论文评测代码（test_advanced.py 的 generate_query_llm）里把问题变成关键词的提示
QUERY_PROMPT = (
    "Given the following question, generate several keywords, using 'cosmos' as the separator."
    "\n\nQuestion: {question}\n\n"
    'Format your response as a JSON object with a "keywords" field containing the selected '
    "text.\n\nExample response format:\n"
    '{{"keywords": "keyword1, keyword2, keyword3"}}'
)  # 照搬原文，只去掉了缩进
QUERY_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {"keywords": {"type": "string"}},
            "required": ["keywords"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
_INDEX = re.compile(r"^\s*(?:memory\s*index\s*[:#]?\s*)?(\d+)\s*$", re.IGNORECASE)


# ---------- 纯函数（不依赖 A-MEM，单独测试） ----------


@dataclass
class Note:
    """写完后的一条 A-MEM 笔记，外加它来自哪个会话（证据的日期与会话号）。"""

    id: str
    content: str
    timestamp: str
    session_id: str
    session_time: datetime
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
            keywords=strings(row.get("keywords")),
            context=str(row.get("context") or "General"),
            tags=strings(row.get("tags")),
            links=list(row.get("links") or []),
            category=str(row.get("category") or "Uncategorized"),
        )


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
        hits.append(
            memory_hit(
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
        )
    return hits


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


def save_notes(path: Path, notes: Sequence[Note]) -> None:
    rows = [note.as_json() for note in notes]
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")


def load_notes(path: Path) -> list[Note]:
    return [Note.from_json(row) for row in json.loads(path.read_text("utf-8"))]


# ---------- A-MEM 的检索器与 LLM 控制器换成走评测网关的版本 ----------


class _Controller:
    """代替 A-MEM 的 OpenAIController：同样的请求（系统提示、response_format、temperature），
    客户端指向评测网关，回复上限是 MAX_TOKENS。顺带数调用、失败、被截断与不是 JSON 的回复
    （A-MEM 自己会吞掉异常，不数就看不出来）。"""

    def __init__(self, client: Any, model: str, max_tokens: int = MAX_TOKENS) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.calls = 0
        self.failed = 0
        self.truncated = 0
        self.invalid_json = 0
        self.errors: list[str] = []
        self._lock = threading.Lock()

    def get_completion(
        self, prompt: str, response_format: dict[str, Any] | None = None, temperature: float = 0.7
    ) -> str:
        with self._lock:
            self.calls += 1
        extra: dict[str, Any] = {"response_format": response_format} if response_format else {}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JSON_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=self.max_tokens,
                **extra,
            )
        except Exception as exc:
            with self._lock:
                self.failed += 1
                if len(self.errors) < 5:
                    self.errors.append(f"{type(exc).__name__}: {exc}"[:200])
            raise
        choice = response.choices[0]
        content = str(choice.message.content or "")
        with self._lock:
            self.truncated += choice.finish_reason == "length"
        if response_format is not None:
            try:
                json.loads(content)
            except json.JSONDecodeError:
                with self._lock:
                    self.invalid_json += 1
        return content


class _Store:
    """A-MEM 的 ChromaRetriever 的替身：每份对话一个持久化的 Chroma 集合，向量由调用方给出
    （走评测网关）。方法与 ChromaRetriever 同名，A-MEM 的写入流程照常调用。"""

    def __init__(self, directory: Path, embed: Callable[[list[str]], list[list[float]]]) -> None:
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
        self.embed = embed
        self.consolidations = 0

    def add_document(self, document: str, metadata: Mapping[str, Any], doc_id: str) -> None:
        self.upsert([(doc_id, document, metadata)])

    def upsert(self, rows: Sequence[tuple[str, str, Mapping[str, Any]]]) -> None:
        for start in range(0, len(rows), EMBED_BATCH):
            batch = rows[start : start + EMBED_BATCH]
            texts = [
                note_document(content, meta.get("context"), meta.get("keywords"), meta.get("tags"))
                for _, content, meta in batch
            ]
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
        count = int(self.collection.count())
        if count == 0 or k <= 0:
            return {"ids": [[]], "distances": [[]]}
        result = self.collection.query(
            query_embeddings=self.embed([query]), n_results=min(k, count), include=["distances"]
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
    """替换 A-MEM 的 consolidate_memories：按笔记当前的上下文与标签重新向量化（论文代码的做法）。"""
    rows = [(note.id, note.content, _metadata(note)) for note in system.memories.values()]
    system.retriever.upsert(rows)
    system.retriever.consolidations += 1


def _metadata(note: Any) -> dict[str, Any]:
    return {
        "context": note.context,
        "keywords": note.keywords,
        "tags": note.tags,
        "timestamp": note.timestamp,
    }


@dataclass
class _Space:
    """检索时打开的一份对话：笔记、ID → 序号、Chroma 集合、生成关键词用的 LLM 控制器。"""

    notes: list[Note]
    ids: dict[str, int]
    store: _Store
    controller: _Controller


class AMemBaseline(ExternalBaseline):
    name, label, system = "MA", "外部基线：A-MEM（agiresearch/A-mem，agentic memory）", "amem"

    def __init__(self) -> None:
        super().__init__()
        self._spaces: OrderedDict[str, _Space] = OrderedDict()

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

    def _embedder(self, url: str) -> Callable[[list[str]], list[list[float]]]:
        from openai import OpenAI

        client = OpenAI(base_url=url, api_key="unused", timeout=GATEWAY_TIMEOUT)
        name = self.embedder.name

        def embed(texts: list[str]) -> list[list[float]]:
            reply = client.embeddings.create(model=name, input=texts)
            return [row.embedding for row in sorted(reply.data, key=lambda row: row.index)]

        return embed

    def _controller(self, url: str) -> _Controller:
        from openai import OpenAI

        client = OpenAI(base_url=url, api_key="unused", timeout=GATEWAY_TIMEOUT)
        return _Controller(client, self.model)

    # ---------- 写入 ----------

    def build(self, key: str, directory: Path, sessions: Sequence[EvalSession]) -> dict[str, Any]:
        url = self.gateway.url(self.bucket(key))
        store = _Store(directory / CHROMA, self._embedder(url))
        llm = self._controller(url)
        system = self.amem.AgenticMemorySystem(
            model_name=self.embedder.name, llm_backend="openai", llm_model=self.model, api_key="x"
        )
        system.retriever = store
        system.llm_controller.llm = llm
        system.find_related_memories = MethodType(_find_related_memories, system)
        system.consolidate_memories = MethodType(_consolidate_memories, system)
        ordered = sorted(sessions, key=lambda s: s.started_at)
        total = sum(1 for s in ordered for t in s.turns if t.text.strip())
        origin: list[tuple[str, EvalSession]] = []
        started = datetime.now()
        log = directory / BUILD_LOG
        try:
            for session in ordered:
                when = when_text(session.started_at)
                for turn in session.turns:
                    if not turn.text.strip():
                        continue
                    content = note_content(turn)
                    analysis = clean_analysis(system.analyze_content(content))
                    origin.append((system.add_note(content, time=when, **analysis), session))
                    if len(origin) % PROGRESS_EVERY == 0 or len(origin) == total:
                        elapsed = (datetime.now() - started).total_seconds()
                        with log.open("a", encoding="utf-8") as out:
                            out.write(
                                f"{datetime.now():%H:%M:%S} {len(origin)}/{total} notes, "
                                f"{system.evo_cnt} evolved, {llm.failed} failed calls, "
                                f"{elapsed:.0f}s\n"
                            )
                if llm.calls >= 20 and llm.failed > llm.calls / 2:
                    raise RuntimeError(f"A-MEM 的 LLM 调用一多半失败：{llm.errors}")
            notes = [
                _note(system.memories[note_id], session.session_id, session.started_at)
                for note_id, session in origin
            ]
        finally:
            store.close()
        if llm.calls and llm.failed > llm.calls / 2:
            raise RuntimeError(f"A-MEM 的 LLM 调用一多半失败：{llm.errors}")
        save_notes(directory / NOTES, notes)
        ids = {note.id: i for i, note in enumerate(notes)}
        links = [link_targets(n.links, ids, len(notes), i) for i, n in enumerate(notes)]
        return {
            "memories": len(notes),
            "evolved": int(system.evo_cnt),
            "linked_notes": sum(1 for found in links if found),
            "links": sum(len(found) for found in links),
            "unresolved_links": sum(len(n.links) for n in notes) - sum(map(len, links)),
            "empty_analyses": sum(1 for n in notes if not n.keywords and n.context == "General"),
            "consolidations": store.consolidations,
            "amem_calls": llm.calls,
            "failed_calls": llm.failed,
            "truncated": llm.truncated,
            "invalid_json": llm.invalid_json,
            "error_samples": llm.errors,  # 不能叫 errors：基类会用网关桶里的错误计数覆盖它
            "max_tokens": llm.max_tokens,
            "top_k": TOP_K,
            "amem_commit": COMMIT,
        }

    # ---------- 检索 ----------

    def retrieve(
        self, key: str, directory: Path, sample: Sample, sessions: Sequence[EvalSession]
    ) -> tuple[list[Hit], dict[str, Any]]:
        space = self._open(key, directory)
        bucket = self.bucket(key, "search")
        before = self.gateway.usage(bucket)
        prompt = QUERY_PROMPT.format(question=sample.question)
        reply = space.controller.get_completion(prompt, QUERY_FORMAT)  # temperature 取 A-MEM 默认
        query = parse_keywords(reply, sample.question)
        found = space.store.search(query, TOP_K)
        ranked: list[int] = []
        scores: dict[int, float] = {}
        for note_id, distance in zip(found["ids"][0], found["distances"][0], strict=True):
            index = space.ids.get(note_id)
            if index is not None:
                ranked.append(index)
                scores[index] = round(1.0 - float(distance), 6)  # 余弦距离 → 相似度
        picked = expand(ranked, space.notes, space.ids, TOP_K)
        hits = to_hits(key, space.notes, picked, scores)
        after = self.gateway.usage(bucket)
        fields = ("chat_calls", "cache_hits", "input_tokens", "output_tokens", "cost", "spent")
        usage = {name: round(after[name] - before.get(name, 0), 6) for name in fields}
        return hits, {
            "amem_query": query,
            "amem_retrieved": len(ranked),
            "amem_linked": sum(1 for _, linked in picked if linked),
            "amem_tokens": sum(estimate_tokens(h.unit.text) for h in hits),
            "amem_search": usage,
        }

    def _open(self, key: str, directory: Path) -> _Space:
        """打开（或复用）一份写好的对话；最多同时开 MAX_OPEN 个，多了关掉最久没用的。
        检索时的 LLM 与向量请求记在 search 桶里。"""
        with self._lock:
            if key in self._spaces:
                self._spaces.move_to_end(key)
                return self._spaces[key]
        url = self.gateway.url(self.bucket(key, "search"))
        notes = load_notes(directory / NOTES)
        space = _Space(
            notes=notes,
            ids={note.id: i for i, note in enumerate(notes)},
            store=_Store(directory / CHROMA, self._embedder(url)),
            controller=self._controller(url),
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


def _note(note: Any, session_id: str, started_at: datetime) -> Note:
    """A-MEM 的 MemoryNote（演化之后的状态）→ 要保存的笔记。"""
    return Note(
        id=str(note.id),
        content=str(note.content),
        timestamp=str(note.timestamp),
        session_id=session_id,
        session_time=started_at,
        keywords=strings(note.keywords),
        context=str(note.context or "General"),
        tags=strings(note.tags),
        links=list(note.links or []),
        category=str(note.category or "Uncategorized"),
    )
