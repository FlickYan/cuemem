"""评测数据集：下载、加载，并统一成 Sample 结构。

- LongMemEval（ICLR 2025）：500 道题，每道题自带一份约 12 万 tokens 的对话历史；
- LoCoMo（ACL 2024）：10 段两人长对话，每段对话对应约 200 道题；
- MemLens（2026）：789 道图文多会话记忆题，同一批题按对话历史长度分 32K–256K 四档，
  图片附在发言上（N6 接入 32K 与 128K 两档）。

对话历史（haystack）单独存放、按 space_key 共享：LoCoMo 同一段对话的约 200 道题共用一份，
避免每道题复制一遍造成内存浪费。

会话 ID（N12 修复的泄漏）：LongMemEval_S 的证据会话 ID 一律以 answer_ 开头（948 个），其余会话是
sharegpt_、ultrachat_ 或一串十六进制；而回答模型看到的每条检索证据都标着会话 ID，等于告诉它哪些
是证据。加载时按会话在这份历史里的时间顺序改写成中性的 session_<i>（与 LoCoMo 自带的写法相同），
证据会话与证据发言随之改写，检索指标不变；原 ID 留在 EvalSession.source_id，只用于追溯，不进
任何提示。MemLens 同样改写。两种写法的记忆不能混用：持久化的写入（CueMem 的评测空间、Mem0 与
外部基线的写入）按 storage_name / storage_space_id 分开存，写入记录里另记一份 session_ids 核对。

消融用环境变量 CUEMEM_EVAL_RAW_SESSION_IDS（只影响 LongMemEval / MemLens）：
- =1：恢复原始 ID，与修复之前逐字节相同（旧的空间、写入与响应缓存照常命中）。但它与默认之间
  不只差证据抬头里的几个字：轮、事实等记录的 ID 是含会话 ID 的哈希，排序遇到并列时按 ID 定
  先后（同一会话的几轮时间相同，回答模型看到的先后就由哈希决定；检索分数并列时也是），换了
  会话 ID，这些先后跟着变。这部分差异与泄漏无关、没有偏向，但会算进两者的差里。
- =label：会话 ID 照常改写（存储、排序、检索与默认完全相同，共用改写后的空间与写入），只在
  回答模型看到的证据抬头里显示原始 ID（source_labels，经 read.reader.session_labels 生效）：
  与默认之间只差抬头里的会话名，是泄漏本身的干净消融；提示与旧运行不同，不命中旧的缓存。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import httpx2

from cuemem.core.models import Conversation, SessionIn, TurnIn
from cuemem.errors import DatasetChecksumError

# 会话 ID 的写法：native 是数据集自带、本来就中性的（LoCoMo 的 session_<n>）；anon 是加载时按时间
# 顺序改写的；raw 是 LongMemEval / MemLens 的原始 ID（会泄漏证据会话，只用于消融）
SessionIds = Literal["native", "anon", "raw"]
RAW_SESSION_IDS_ENV = "CUEMEM_EVAL_RAW_SESSION_IDS"
RAW_LABELS_VALUE = "label"  # CUEMEM_EVAL_RAW_SESSION_IDS=label：只有证据抬头显示原始 ID
ANON_SUFFIX = "-anon"  # 改写过会话 ID 的持久化写入：数据集名加这个后缀，与旧的写入分开
_ANONYMIZED_DATASETS = ("longmemeval", "memlens_")

# LongMemEval 题型 → 论文中的五种能力（拒答题单独算作 abstention）
LONGMEMEVAL_ABILITIES = {
    "single-session-user": "information_extraction",
    "single-session-assistant": "information_extraction",
    "single-session-preference": "information_extraction",
    "multi-session": "multi_session_reasoning",
    "temporal-reasoning": "temporal_reasoning",
    "knowledge-update": "knowledge_update",
}

# LoCoMo 数据中的类别编号 → 名称；与 Mem0 等工作的统计一致（841 / 282 / 321 / 96 / 446 题）
LOCOMO_CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


@dataclass(frozen=True)
class DatasetSpec:
    """一个评测数据集的下载方式。URL 都固定在上游的一个提交上（Hugging Face 的 resolve/<提交>/、
    GitHub 的 <提交>/data/…），不跟着 main 变；sha256 是那个提交里这个文件的哈希，下载后核对，
    对不上就不保存（download）。上游改了题目、标准答案或图片时，新机器上下载到的仍是发表时的那份。
    """

    name: str
    filename: str
    url: str
    sha256: str
    description: str
    # 一同下载的附加文件：（文件名，URL，sha256）；.tar.gz 核对之后解压到同一目录
    extras: tuple[tuple[str, str, str], ...] = ()
    optional: bool = False  # 不随"全部下载"一起下，点名才下载


# MemLens：上游最后一次提交（2026-06-14），N6、N9 下载时的 main 就是它
_MEMLENS = (
    "https://huggingface.co/datasets/xiyuRenBill/MEMLENS/resolve/"
    "afa101a1907cc37db40b50d649547964387b96b7/"
)
_MEMLENS_EXTRAS = (
    (
        "memlens/agent_subset_195.json",
        _MEMLENS + "agent_subset_195.json",
        "ae954eca5eb8478c9a1855844bee6fba7a6641f5ea14bd91c3ea49bf9f50b2aa",
    ),
    (
        "memlens/release_images.tar.gz",
        _MEMLENS + "release_images.tar.gz",
        "8d0be814ab3ffe99ed3ea84ed04eacd9f06423c4068da2c24e4749c7ca3940f6",
    ),
)


DATASETS = {
    "longmemeval_s": DatasetSpec(
        "longmemeval_s",
        "longmemeval_s_cleaned.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
        "98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_s_cleaned.json",
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "LongMemEval_S 清洗版：500 题，约 277 MB，MIT 许可",
    ),
    "locomo": DatasetSpec(
        "locomo",
        "locomo10.json",
        "https://raw.githubusercontent.com/snap-research/locomo/"
        "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json",
        "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        "LoCoMo：10 段对话、1986 题，约 3 MB",
    ),
    # 混合流（N7）里"不需要记忆"的通用请求，只用不带参考材料的四类（mixed.py）
    "dolly": DatasetSpec(
        "dolly",
        "databricks-dolly-15k.jsonl",
        "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
        "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl",
        "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
        "databricks-dolly-15k：15,011 条人工撰写的指令，约 13 MB，CC BY-SA 3.0",
    ),
    # N11：真实的聊天机器人用户消息（会话的第一句），检验门控在真实流量上的误召回；
    # 只取 WildChat-1M 的一个分片，固定到发布时的提交
    "wildchat": DatasetSpec(
        "wildchat",
        "wildchat/train-00000-of-00014.parquet",
        "https://huggingface.co/datasets/allenai/WildChat-1M/resolve/"
        "7d6490e462285cf85d91eabea0f9a954fbddcd1f/data/train-00000-of-00014.parquet",
        "abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c",
        "WildChat-1M 的第 1 个分片：约 7 万段真实的聊天机器人对话，约 231 MB，ODC-BY",
        optional=True,
    ),
    "memlens_32k": DatasetSpec(
        "memlens_32k",
        "memlens/dataset_32k.json",
        _MEMLENS + "dataset_32k.json",
        "4c75acc3e0a7e1cc71bc962bff3d2a0f1f35e268bc28254ca85f1d9867b03eb9",
        "MemLens 32K 档：789 题，约 98 MB；图片 4,695 张约 225 MB（CC-BY-4.0）",
        _MEMLENS_EXTRAS,
    ),
    "memlens_128k": DatasetSpec(
        "memlens_128k",
        "memlens/dataset_128k.json",
        _MEMLENS + "dataset_128k.json",
        "ad148ccc936d698db8e05a6b0df69f9342407468eb69a4ae70d8f540334bea14",
        "MemLens 128K 档：789 题，约 369 MB；图片与 32K 档共用",
        _MEMLENS_EXTRAS,
    ),
}

# MemLens 题型 → 名称（五种能力，与论文一致）；拒答题的标准答案是 "Insufficient information"
MEMLENS_TYPES = (
    "information_extraction",
    "multi_session_reasoning",
    "temporal_reasoning",
    "knowledge_update",
    "answer_refusal",
)


@dataclass(slots=True)
class EvalTurn:
    role: Literal["user", "assistant"]
    text: str
    speaker: str | None = None
    has_answer: bool = False  # 是否是答案证据所在的发言
    images: list[str] = field(default_factory=list)  # 附带的图片（本地路径，MemLens）


@dataclass(slots=True)
class EvalSession:
    session_id: str
    started_at: datetime
    turns: list[EvalTurn]
    # 数据集里的原始会话 ID（LongMemEval / MemLens）：只用于追溯，不交给任何模型
    source_id: str = ""


@dataclass(slots=True)
class Sample:
    dataset: str
    sample_id: str
    space_key: str  # 共用同一份对话历史的题目 space_key 相同
    question: str
    answer: str
    question_type: str
    ability: str
    is_abstention: bool
    question_time: datetime
    evidence_session_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSet:
    """一组题目及其对话历史。session_ids 是会话 ID 的写法；raw_labels 为 True 时（只在 anon 下）
    证据抬头显示原始 ID（label 消融，见模块说明）。"""

    dataset: str
    samples: list[Sample]
    haystacks: dict[str, list[EvalSession]]
    session_ids: SessionIds = "native"
    raw_labels: bool = False

    def sessions_of(self, sample: Sample) -> list[EvalSession]:
        return self.haystacks[sample.space_key]


# ---------- 会话 ID ----------


def _requested() -> str:
    return os.environ.get(RAW_SESSION_IDS_ENV, "").strip().lower()


def raw_session_ids_requested() -> bool:
    """环境变量 CUEMEM_EVAL_RAW_SESSION_IDS=1：LongMemEval / MemLens 保留原始会话 ID。"""
    return _requested() in {"1", "true", "yes", "on"}


def raw_session_labels_requested() -> bool:
    """环境变量 CUEMEM_EVAL_RAW_SESSION_IDS=label：会话 ID 照常改写，只有证据抬头显示原始 ID。"""
    return _requested() == RAW_LABELS_VALUE


def source_labels(sessions: Iterable[EvalSession]) -> dict[str, str]:
    """改写后的会话 ID → 数据集的原始 ID（label 消融里证据抬头显示的会话名）。"""
    return {s.session_id: s.source_id for s in sessions if s.source_id}


def run_session_ids(info: Mapping[str, Any]) -> SessionIds:
    """一次已有运行（run.json）的会话 ID 写法。分析已有运行的脚本要按它加载数据集、找写入，
    不能看当前的环境变量。改写之前的运行没有记下写法：LongMemEval / MemLens 的都是原始 ID。"""
    recorded = info.get("session_ids")
    if recorded in ("native", "anon", "raw"):
        return cast(SessionIds, recorded)
    dataset = str((info.get("config") or {}).get("dataset", ""))
    return session_id_scheme(dataset, raw=True)


def run_raw_labels(info: Mapping[str, Any]) -> bool:
    """一次已有运行是不是 label 消融（证据抬头显示原始 ID；run.json 的 session_labels）。"""
    return info.get("session_labels") == "source"


def session_id_scheme(dataset: str, raw: bool | None = None) -> SessionIds:
    """这个数据集加载后的会话 ID 写法。raw 为 None 时看环境变量。"""
    if not dataset.startswith(_ANONYMIZED_DATASETS):
        return "native"
    if raw is None:
        raw = raw_session_ids_requested()
    return "raw" if raw else "anon"


def storage_name(dataset: str, scheme: SessionIds | None = None) -> str:
    """持久化写入用的数据集名：改写过会话 ID 的加上 -anon。

    改写之前的写入都是原始 ID，所以原始 ID 沿用原来的位置（消融时直接复用旧的写入），改写过的
    另起一处；LoCoMo 不变。scheme 为 None 时按环境变量（与加载时一致）。"""
    if scheme is None:
        scheme = session_id_scheme(dataset)
    return f"{dataset}{ANON_SUFFIX}" if scheme == "anon" else dataset


def storage_space_id(space_id: str, scheme: SessionIds) -> str:
    """评测空间 ID（<数据集>:<对话历史>）→ 持久化写入用的 ID（数据集名换成 storage_name）。"""
    dataset, sep, rest = space_id.partition(":")
    return f"{storage_name(dataset, scheme)}{sep}{rest}" if sep else space_id


def build_matches_session_ids(build: Mapping[str, Any], scheme: SessionIds) -> bool:
    """外部系统（Mem0、Graphiti……）的写入记录（build.json）与这次的会话 ID 写法是否相同。

    这些写入在记忆里存着会话 ID（回答模型看到的证据抬头会显示它），不同写法不能复用。
    LoCoMo（native）不核对、也不记；改写之前的写入没有这条记录，都是原始 ID（raw）。"""
    if scheme == "native":
        return True
    return build.get("session_ids", "raw") == scheme


def _raw_labels(scheme: SessionIds, requested: bool | None) -> bool:
    """label 消融只在改写过的会话 ID 上有意义（原始 ID 时抬头本来就是原始 ID）。"""
    if scheme != "anon":
        return False
    return raw_session_labels_requested() if requested is None else requested


def anonymize_sessions(sessions: Sequence[EvalSession]) -> dict[str, str]:
    """把一份对话历史的会话 ID 改写成 session_<i>：i 是会话在这份历史里的时间顺序（从 1 开始，
    同一时间的保持原顺序）。原 ID 记进 source_id。返回 原 ID → 新 ID。

    同一份历史里出现两次的原 ID（LongMemEval_S 有 13 份历史各有一个，内容相同、时间不同）按
    最早的那次编号、共用一个新 ID：只换名字，哪些会话算同一个与原始 ID 时完全相同。写入的内容
    相同，但含会话 ID 的哈希 ID（轮、事实）变了，并列时的先后跟着变（见模块说明）。"""
    order = sorted(range(len(sessions)), key=lambda i: sessions[i].started_at)
    mapping: dict[str, str] = {}
    for i in order:
        mapping.setdefault(sessions[i].session_id, f"session_{len(mapping) + 1}")
    for session in sessions:
        session.source_id = session.source_id or session.session_id
        session.session_id = mapping[session.session_id]
    return mapping


def to_session_ins(sessions: Iterable[EvalSession]) -> list[SessionIn]:
    return [
        SessionIn(
            session_id=s.session_id,
            started_at=s.started_at,
            turns=[
                TurnIn(role=t.role, text=t.text, speaker=t.speaker, images=list(t.images))
                for t in s.turns
            ],
        )
        for s in sessions
    ]


def to_conversation(space_id: str, sessions: Iterable[EvalSession]) -> Conversation:
    return Conversation(space_id=space_id, sessions=to_session_ins(sessions))


def evidence_turns(sessions: Iterable[EvalSession]) -> set[tuple[str, int]]:
    """标为证据的发言：（会话 ID，发言下标）。"""
    return {(s.session_id, i) for s in sessions for i, t in enumerate(s.turns) if t.has_answer}


# ---------- 下载 ----------


def dataset_path(directory: Path, name: str) -> Path:
    return directory / DATASETS[name].filename


def download(name: str, directory: Path, *, timeout_s: float = 600.0) -> Path:
    """下载数据集到 directory；已存在则跳过（不重新核对）。先写临时文件，sha256 与固定的值一致
    才改名，避免留下半个或不对的文件；压缩包核对之后才解压。对不上时报 DatasetChecksumError。"""
    spec = DATASETS[name]
    for filename, url, sha256 in spec.extras:
        extra = directory / filename
        if filename.endswith(".tar.gz"):
            unpacked = extra.parent / filename.split("/")[-1].removesuffix(".tar.gz")
            if unpacked.exists():
                continue
            _fetch(url, extra, timeout_s, sha256)
            with tarfile.open(extra) as archive:
                archive.extractall(extra.parent, filter="data")
            extra.unlink()
        elif not extra.exists():
            _fetch(url, extra, timeout_s, sha256)
    target = directory / spec.filename
    if not (target.exists() and target.stat().st_size > 0):
        _fetch(spec.url, target, timeout_s, spec.sha256)
    return target


def _fetch(url: str, target: Path, timeout_s: float, sha256: str) -> None:
    """边下载边算 sha256；与 sha256 一致才把临时文件改名成 target，否则删掉它并报错。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()
    try:
        with (
            httpx2.stream("GET", url, follow_redirects=True, timeout=timeout_s) as response,
            partial.open("wb") as f,
        ):
            response.raise_for_status()
            for chunk in response.iter_bytes():
                f.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != sha256:
            raise DatasetChecksumError(
                f"下载的 {target.name} 与固定的版本不同：sha256 应为 {sha256}，实际是 "
                f"{digest.hexdigest()}（{url}）。没有保存。URL 固定在上游的一个提交上，内容不该变；"
                "多半是下载被代理或镜像截断、改写了，换个网络重试"
            )
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.rename(target)


# ---------- 加载 ----------


def load(
    name: str,
    path: Path,
    *,
    ids: Collection[str] | None = None,
    raw_ids: bool | None = None,
    raw_labels: bool | None = None,
) -> EvalSet:
    """raw_ids：LongMemEval / MemLens 是否保留原始会话 ID；raw_labels：改写后是否只在证据抬头
    显示原始 ID（label 消融）。都是 None 时看环境变量（见模块说明）。"""
    if name == "longmemeval_s":
        return load_longmemeval(path, ids=ids, dataset=name, raw_ids=raw_ids, raw_labels=raw_labels)
    if name == "locomo":
        return load_locomo(path, ids=ids)
    if name.startswith("memlens_"):
        return load_memlens(path, ids=ids, dataset=name, raw_ids=raw_ids, raw_labels=raw_labels)
    raise KeyError(f"未知数据集 {name}，可选：{sorted(DATASETS)}")


def protocol_samples(evalset: EvalSet, types: Collection[str] | None = None) -> list[Sample]:
    """评测协议选出的题（保持原来的顺序）：给了题型时只留这些题型（LoCoMo 的对抗题也可以点名）；
    没给时 LoCoMo 去掉对抗题（与文献一致：Mem0 等工作只报前四类），其余数据集全留。

    eval run、eval probe 与两个混合流脚本（n7_mixed、n9_mixed）都经这里选题，规则只写这一处：
    改它就改了发表的题目集合（LoCoMo dev3 的 497 题里评 385 题，test7 的 1489 题里评 1155 题）。"""
    if types:
        wanted = set(types)
        return [s for s in evalset.samples if s.question_type in wanted]
    if evalset.dataset == "locomo":
        return [s for s in evalset.samples if s.question_type != "adversarial"]
    return list(evalset.samples)


def _parse_longmemeval_date(text: str) -> datetime:
    # 形如 "2023/05/20 (Sat) 02:21"；去掉星期，避免依赖系统语言环境
    return datetime.strptime(re.sub(r"\s*\([A-Za-z]+\)", "", text).strip(), "%Y/%m/%d %H:%M")


def load_longmemeval(
    path: Path,
    *,
    ids: Collection[str] | None = None,
    dataset: str = "longmemeval_s",
    raw_ids: bool | None = None,
    raw_labels: bool | None = None,
) -> EvalSet:
    """加载 LongMemEval。ids 不为空时只转换这些题目（完整文件约 277 MB）。

    会话 ID 默认按时间顺序改写成 session_<i>，证据会话随之改写（见模块说明）；
    raw_ids=True（或环境变量 CUEMEM_EVAL_RAW_SESSION_IDS=1）保留原始 ID；raw_labels=True（或
    CUEMEM_EVAL_RAW_SESSION_IDS=label）照常改写，只在证据抬头显示原始 ID。"""
    scheme = session_id_scheme(dataset, raw_ids)
    wanted = set(ids) if ids is not None else None
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: list[Sample] = []
    haystacks: dict[str, list[EvalSession]] = {}
    for item in raw:
        qid = item["question_id"]
        if wanted is not None and qid not in wanted:
            continue
        sessions = [
            EvalSession(
                session_id=sid,
                started_at=_parse_longmemeval_date(date),
                turns=[
                    EvalTurn(
                        role=turn["role"],
                        text=turn["content"],
                        has_answer=bool(turn.get("has_answer", False)),
                    )
                    for turn in turns
                ],
                source_id=sid,
            )
            for sid, date, turns in zip(
                item["haystack_session_ids"],
                item["haystack_dates"],
                item["haystack_sessions"],
                strict=True,
            )
        ]
        evidence = list(item.get("answer_session_ids", []))
        if scheme == "anon":
            mapping = anonymize_sessions(sessions)
            # 不在历史里的证据会话（数据里没有）原样留着：只用于指标，召回不到，与改写前相同
            evidence = [mapping.get(sid, sid) for sid in evidence]
        haystacks[qid] = sessions
        qtype = item["question_type"]
        samples.append(
            Sample(
                dataset=dataset,
                sample_id=qid,
                space_key=qid,  # 每道题有自己的对话历史
                question=item["question"],
                answer=str(item["answer"]),
                question_type=qtype,
                ability=LONGMEMEVAL_ABILITIES[qtype],
                is_abstention=qid.endswith("_abs"),
                question_time=_parse_longmemeval_date(item["question_date"]),
                evidence_session_ids=evidence,
            )
        )
    return EvalSet(
        dataset=dataset,
        samples=samples,
        haystacks=haystacks,
        session_ids=scheme,
        raw_labels=_raw_labels(scheme, raw_labels),
    )


def _parse_locomo_date(text: str) -> datetime:
    # 形如 "1:56 pm on 8 May, 2023"
    return datetime.strptime(text.strip(), "%I:%M %p on %d %B, %Y")


def load_locomo(path: Path, *, ids: Collection[str] | None = None) -> EvalSet:
    """加载 LoCoMo。两位说话人都映射为 user 角色并保留名字；图片以 BLIP 描述的形式并入文本。"""
    wanted = set(ids) if ids is not None else None
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: list[Sample] = []
    haystacks: dict[str, list[EvalSession]] = {}
    for conv in raw:
        conv_id = conv["sample_id"]
        dialogue = conv["conversation"]
        numbers = sorted(
            int(m.group(1)) for key in dialogue if (m := re.fullmatch(r"session_(\d+)", key))
        )
        sessions: list[EvalSession] = []
        dia_index: dict[str, tuple[str, int]] = {}
        for n in numbers:
            session_id = f"session_{n}"
            turns: list[EvalTurn] = []
            for i, turn in enumerate(dialogue[session_id]):
                text = turn["text"]
                if turn.get("blip_caption"):
                    text = f"{text} [shares a photo: {turn['blip_caption']}]"
                turns.append(EvalTurn(role="user", text=text, speaker=turn["speaker"]))
                dia_index[turn["dia_id"]] = (session_id, i)
            started = _parse_locomo_date(dialogue[f"{session_id}_date_time"])
            sessions.append(EvalSession(session_id=session_id, started_at=started, turns=turns))
        question_time = max(s.started_at for s in sessions)  # 没有提问时间，取最后一个会话的时间

        conv_samples: list[Sample] = []
        evidence_marks: set[tuple[str, int]] = set()
        for index, qa in enumerate(conv["qa"]):
            sample_id = f"{conv_id}-q{index}"
            if wanted is not None and sample_id not in wanted:
                continue
            category = LOCOMO_CATEGORIES[qa["category"]]
            is_adversarial = category == "adversarial"
            refs = [dia_index[d] for d in _split_evidence(qa.get("evidence", [])) if d in dia_index]
            evidence_marks.update(refs)
            answer = qa.get("answer")
            if is_adversarial or answer is None:
                # 对抗题没有标准答案，只有一个误导性答案；正确行为是指出对话里没有这件事
                answer = (
                    "The conversation does not support this question; a misleading answer "
                    f"would be: {qa.get('adversarial_answer', '')}"
                )
            conv_samples.append(
                Sample(
                    dataset="locomo",
                    sample_id=sample_id,
                    space_key=conv_id,
                    question=qa["question"],
                    answer=str(answer),
                    question_type=category,
                    ability=category,
                    is_abstention=is_adversarial,
                    question_time=question_time,
                    evidence_session_ids=sorted({sid for sid, _ in refs}),
                    meta={"evidence_turns": [list(r) for r in refs]},
                )
            )
        if not conv_samples:
            continue
        # LoCoMo 的证据是逐题标注的：把所有题的证据发言都标上，按题计算召回时再用 meta 里的列表
        for session in sessions:
            for i, turn in enumerate(session.turns):
                turn.has_answer = (session.session_id, i) in evidence_marks
        haystacks[conv_id] = sessions
        samples.extend(conv_samples)
    return EvalSet(dataset="locomo", samples=samples, haystacks=haystacks)


def load_memlens(
    path: Path,
    *,
    ids: Collection[str] | None = None,
    dataset: str = "memlens_32k",
    raw_ids: bool | None = None,
    raw_labels: bool | None = None,
) -> EvalSet:
    """加载 MemLens。格式与 LongMemEval 相同，发言上多了图片（release_images 下的相对路径）。

    证据只标到会话（answer_session_ids）。每道题的关键图片放在 needle_images/ 下：把证据会话里
    带关键图片的发言标为证据发言，检索指标里的"发言召回"就是"关键图片所在轮的召回"。
    图片路径只用来读图（交给模型的是重新编码的图片本身，路径与文件名不进提示）。
    会话 ID 与 LongMemEval 一样默认改写成 session_<i>（MemLens 的原始 ID 是 sess_<哈希>，
    看不出证据，改写是为了两个数据集一致）。
    """
    scheme = session_id_scheme(dataset, raw_ids)
    wanted = set(ids) if ids is not None else None
    image_dir = path.parent / "release_images"
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples: list[Sample] = []
    haystacks: dict[str, list[EvalSession]] = {}
    for item in raw:
        qid = item["question_id"]
        if wanted is not None and qid not in wanted:
            continue
        evidence = set(item.get("answer_session_ids") or [])
        sessions = [
            EvalSession(
                session_id=sid,
                started_at=_parse_longmemeval_date(date),
                turns=[
                    EvalTurn(
                        role="assistant" if turn["role"] == "assistant" else "user",
                        text=turn["content"],
                        images=[str(image_dir / im["file"]) for im in turn.get("images") or []],
                        has_answer=sid in evidence
                        and any(
                            str(im["file"]).startswith("needle_images/")
                            for im in turn.get("images") or []
                        ),
                    )
                    for turn in turns
                ],
                source_id=sid,
            )
            for sid, date, turns in zip(
                item["haystack_session_ids"],
                item["haystack_dates"],
                item["haystack_sessions"],
                strict=True,
            )
        ]
        evidence_ids = list(item.get("answer_session_ids") or [])
        if scheme == "anon":
            mapping = anonymize_sessions(sessions)
            evidence_ids = [mapping.get(sid, sid) for sid in evidence_ids]
        haystacks[qid] = sessions
        qtype = item["question_type"]
        samples.append(
            Sample(
                dataset=dataset,
                sample_id=qid,
                space_key=qid,
                question=item["question"],
                answer=str(item["answer"]),
                question_type=qtype,
                ability=qtype,
                is_abstention=qtype == "answer_refusal",
                question_time=_parse_longmemeval_date(item["question_date"]),
                evidence_session_ids=evidence_ids,
                meta={"question_subtype": item.get("question_subtype", "")},
            )
        )
    return EvalSet(
        dataset=dataset,
        samples=samples,
        haystacks=haystacks,
        session_ids=scheme,
        raw_labels=_raw_labels(scheme, raw_labels),
    )


def _split_evidence(values: Iterable[str]) -> list[str]:
    """少数证据写成 "D8:6; D9:17" 这样的合并字符串，这里拆开。"""
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in re.split(r"[;,]", value) if part.strip())
    return out


def sample_evidence_turns(sample: Sample, sessions: Iterable[EvalSession]) -> set[tuple[str, int]]:
    """某道题自己的证据发言。LoCoMo 逐题标注（存在 meta 中）；LongMemEval 用发言上的 has_answer。"""
    if "evidence_turns" in sample.meta:
        return {(sid, int(i)) for sid, i in sample.meta["evidence_turns"]}
    return evidence_turns(sessions)
