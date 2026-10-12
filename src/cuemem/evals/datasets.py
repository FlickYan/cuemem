"""评测数据集：下载、加载，并统一成 Sample 结构。

- LongMemEval（ICLR 2025）：500 道题，每道题自带一份约 12 万 tokens 的对话历史；
- LoCoMo（ACL 2024）：10 段两人长对话，每段对话对应约 200 道题；
- MemLens（2026）：789 道图文多会话记忆题，同一批题按对话历史长度分 32K–256K 四档，
  图片附在发言上（N6 接入 32K 与 128K 两档）。

对话历史（haystack）单独存放、按 space_key 共享：LoCoMo 同一段对话的约 200 道题共用一份，
避免每道题复制一遍造成内存浪费。
"""

from __future__ import annotations

import json
import re
import tarfile
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx2

from cuemem.core.models import Conversation, SessionIn, TurnIn

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
    name: str
    filename: str
    url: str
    description: str
    # 一同下载的附加文件：（文件名，URL）；.tar.gz 下载后解压到同一目录
    extras: tuple[tuple[str, str], ...] = ()


_MEMLENS = "https://huggingface.co/datasets/xiyuRenBill/MEMLENS/resolve/main/"
_MEMLENS_EXTRAS = (
    ("memlens/agent_subset_195.json", _MEMLENS + "agent_subset_195.json"),
    ("memlens/release_images.tar.gz", _MEMLENS + "release_images.tar.gz"),
)


DATASETS = {
    "longmemeval_s": DatasetSpec(
        "longmemeval_s",
        "longmemeval_s_cleaned.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"
        "longmemeval_s_cleaned.json",
        "LongMemEval_S 清洗版：500 题，约 277 MB，MIT 许可",
    ),
    "locomo": DatasetSpec(
        "locomo",
        "locomo10.json",
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
        "LoCoMo：10 段对话、1986 题，约 3 MB",
    ),
    # 混合流（N7）里"不需要记忆"的通用请求，只用不带参考材料的四类（mixed.py）
    "dolly": DatasetSpec(
        "dolly",
        "databricks-dolly-15k.jsonl",
        "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/"
        "databricks-dolly-15k.jsonl",
        "databricks-dolly-15k：15,011 条人工撰写的指令，约 13 MB，CC BY-SA 3.0",
    ),
    "memlens_32k": DatasetSpec(
        "memlens_32k",
        "memlens/dataset_32k.json",
        _MEMLENS + "dataset_32k.json",
        "MemLens 32K 档：789 题，约 98 MB；图片 4,695 张约 225 MB（CC-BY-4.0）",
        _MEMLENS_EXTRAS,
    ),
    "memlens_128k": DatasetSpec(
        "memlens_128k",
        "memlens/dataset_128k.json",
        _MEMLENS + "dataset_128k.json",
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
    """一组题目及其对话历史。"""

    dataset: str
    samples: list[Sample]
    haystacks: dict[str, list[EvalSession]]

    def sessions_of(self, sample: Sample) -> list[EvalSession]:
        return self.haystacks[sample.space_key]


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
    """下载数据集到 directory；已存在则跳过。先写临时文件，完成后再改名，避免留下半个文件。"""
    spec = DATASETS[name]
    for filename, url in spec.extras:
        extra = directory / filename
        if filename.endswith(".tar.gz"):
            unpacked = extra.parent / filename.split("/")[-1].removesuffix(".tar.gz")
            if unpacked.exists():
                continue
            _fetch(url, extra, timeout_s)
            with tarfile.open(extra) as archive:
                archive.extractall(extra.parent, filter="data")
            extra.unlink()
        elif not extra.exists():
            _fetch(url, extra, timeout_s)
    target = directory / spec.filename
    if not (target.exists() and target.stat().st_size > 0):
        _fetch(spec.url, target, timeout_s)
    return target


def _fetch(url: str, target: Path, timeout_s: float) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with (
        httpx2.stream("GET", url, follow_redirects=True, timeout=timeout_s) as response,
        partial.open("wb") as f,
    ):
        response.raise_for_status()
        for chunk in response.iter_bytes():
            f.write(chunk)
    partial.rename(target)


# ---------- 加载 ----------


def load(name: str, path: Path, *, ids: Collection[str] | None = None) -> EvalSet:
    if name == "longmemeval_s":
        return load_longmemeval(path, ids=ids, dataset=name)
    if name == "locomo":
        return load_locomo(path, ids=ids)
    if name.startswith("memlens_"):
        return load_memlens(path, ids=ids, dataset=name)
    raise KeyError(f"未知数据集 {name}，可选：{sorted(DATASETS)}")


def _parse_longmemeval_date(text: str) -> datetime:
    # 形如 "2023/05/20 (Sat) 02:21"；去掉星期，避免依赖系统语言环境
    return datetime.strptime(re.sub(r"\s*\([A-Za-z]+\)", "", text).strip(), "%Y/%m/%d %H:%M")


def load_longmemeval(
    path: Path, *, ids: Collection[str] | None = None, dataset: str = "longmemeval_s"
) -> EvalSet:
    """加载 LongMemEval。ids 不为空时只转换这些题目（完整文件约 277 MB）。"""
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
            )
            for sid, date, turns in zip(
                item["haystack_session_ids"],
                item["haystack_dates"],
                item["haystack_sessions"],
                strict=True,
            )
        ]
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
                evidence_session_ids=list(item.get("answer_session_ids", [])),
            )
        )
    return EvalSet(dataset=dataset, samples=samples, haystacks=haystacks)


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
    path: Path, *, ids: Collection[str] | None = None, dataset: str = "memlens_32k"
) -> EvalSet:
    """加载 MemLens。格式与 LongMemEval 相同，发言上多了图片（release_images 下的相对路径）。

    证据只标到会话（answer_session_ids）。每道题的关键图片放在 needle_images/ 下：把证据会话里
    带关键图片的发言标为证据发言，检索指标里的"发言召回"就是"关键图片所在轮的召回"。
    """
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
            )
            for sid, date, turns in zip(
                item["haystack_session_ids"],
                item["haystack_dates"],
                item["haystack_sessions"],
                strict=True,
            )
        ]
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
                evidence_session_ids=list(item.get("answer_session_ids") or []),
                meta={"question_subtype": item.get("question_subtype", "")},
            )
        )
    return EvalSet(dataset=dataset, samples=samples, haystacks=haystacks)


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
