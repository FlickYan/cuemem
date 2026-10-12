"""记忆探针（设计文档 §6.2，N7）：每次提问都做、不调用 LLM 的一次检索，产出检索控制器的特征。

探针就是 L1 的打分（read/l1.py 的 candidates）：一次查询向量化，轮、事实、摘要、图片描述各走
dense 召回，再加一路以文搜图。控制器选了 L1 时直接拿这份候选打包，不重复检索；选 L2 时，L2 对
原问题的打分命中向量缓存。所以走 L1、L2 时探针几乎没有额外成本；最后选了 L0 时，多花的是一次
查询向量化（几十个 tokens）和几十毫秒。

特征分四组，名字带前缀：

- q_ 查询线索（只看问题文本）：第一 / 第二人称、指向过去的对话、问自己过去做过的事、聚合、时长、
  先后、比较、时间表达及其宽度、"最近 / 最早 / 以前"、疑问词、视觉细节词、求推荐、长度、专有名词；
- p_ 探针反馈（看检索结果）：各路 top-1 相似度、与第 2 名的差、相对候选池的 z 分数，前 10 轮的
  会话分散度与时间跨度，与 top-1 接近的事实条数（同类事情散落多处）、被取代的事实，
  问题里点名的实体数，图片命中；
- m_ 记忆规模：空间里的会话数、轮数、事实数、图片数；
- r_ 回答之后（只用于反应式升级）：L1 的回答是否自认证据不足、是否像"不知道"。

所有特征都是数值（布尔记 0 / 1），缺失记 0；控制器先标准化再用。
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from cuemem.core.models import Answer
from cuemem.read.l1 import L1Candidates, L1Config, L1Retriever
from cuemem.read.query import (
    COMPARE_WORDS,
    DURATION_WORDS,
    ORDER_WORDS,
    QueryInfo,
    analyze,
)
from cuemem.store.sqlite import MemoryDB

# 粗略识别"不知道"式回答（报告里的误拒答统计也用它）
DONT_KNOW = re.compile(
    r"\b(i don't know|i do not know|i don't have (any )?(information|record|details)"
    r"|i can't (tell|determine|give|say)|i cannot (tell|determine|give|say)"
    r"|no (information|record|mention) (of|about)|(isn't|is not|aren't|are not) mentioned"
    r"|not enough information)\b|不知道|没有(相关)?(的)?(信息|记录)",
    re.IGNORECASE,
)

_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself|we|our|us)\b|我", re.I)
_SECOND_PERSON = re.compile(r"\b(you|your|yours|you've|you'd|you're)\b|你|您", re.I)
_VISUAL = re.compile(
    r"\b(colou?rs?|photos?|pictures?|pics?|images?|screenshots?|selfies?|look(s|ed)? like"
    r"|appearance|wearing|outfit|logo|brand|label|sign|shape|pattern|design|written|printed"
    r"|shown|depicted|visible|background)\b"
    r"|照片|图片|图里|图中|颜色|长什么样|画面|上面写",
    re.I,
)
_RECOMMEND = re.compile(
    r"\b(recommend\w*|suggest\w*|advice|tips|ideas?|should i|any good|what would you)\b"
    r"|推荐|建议|有什么好",
    re.I,
)
# 指向过去的对话（"你上次说的""我之前提到过"）与问自己过去做过的事（"我买过……吗"）
_RECALL = re.compile(
    r"\b(remember|recall|remind me|mention(ed)?|told you|you told|you said|you recommended"
    r"|you suggested|you mentioned|we talked|we discussed|we chatted|last time|earlier"
    r"|previous(ly)? (chat|conversation))\b|记得|提到过|上次|之前说",
    re.I,
)
_PAST_SELF = re.compile(
    r"\b(did i|have i|had i|was i|were we|did we|have we|i've|we've"
    r"|i (had|did|was|went|bought|got|made|used|tried|visited|started|finished|said|told|asked"
    r"|planned|wanted|attended|read|watched|took|received|spent|paid|saw|met))\b"
    r"|我(买|去|做|用|看|说|吃|参加|收到|花)过|我有没有",
    re.I,
)
_YES_NO = re.compile(
    r"^\s*(is|are|was|were|do|does|did|have|has|had|can|could|will|would|should|am)\b", re.I
)
_WH = {
    "when": re.compile(
        r"\bwhen\b|\bwhat (date|day|time|year|month)\b|什么时候|哪天|几号|哪年", re.I
    ),
    "where": re.compile(r"\bwhere\b|哪里|在哪", re.I),
    "who": re.compile(r"\bwho(m|se)?\b|谁", re.I),
    "which": re.compile(r"\bwhich\b|哪个|哪一", re.I),
    "why": re.compile(r"\bwhy\b|为什么", re.I),
    "how_many": re.compile(r"\bhow (many|much)\b|多少|几个|几次", re.I),
}
_WORD = re.compile(r"[A-Za-z0-9']+|[一-鿿]")

TOP_ROUNDS = 10  # 看前 10 轮的会话分散度与时间跨度
NEAR_FACT = 0.05  # 与 top-1 事实的相似度差在这个范围内算"接近"
SCORE_KINDS = ("round", "fact", "summary")


def query_features(question: str, query: QueryInfo | None = None) -> dict[str, float]:
    """只看问题文本的线索（q_ 组），不检索。"""
    query = query or analyze(question, datetime.now())
    words = _WORD.findall(question)
    width = query.time_range.days if query.time_range is not None else 0.0
    features = {
        "q_words": math.log1p(len(words)),
        "q_first_person": float(bool(_FIRST_PERSON.search(question))),
        "q_second_person": float(bool(_SECOND_PERSON.search(question))),
        "q_aggregation": float(query.aggregation),
        "q_duration": float(bool(DURATION_WORDS.search(question))),
        "q_order": float(bool(ORDER_WORDS.search(question))),
        "q_compare": float(bool(COMPARE_WORDS.search(question))),
        "q_temporal": float(query.temporal),
        "q_time_range": float(query.time_range is not None),
        "q_time_width": math.log1p(width),
        "q_latest": float(query.wants_latest),
        "q_first": float(query.wants_first),
        "q_history": float(query.wants_history),
        "q_visual": float(bool(_VISUAL.search(question))),
        "q_recommend": float(bool(_RECOMMEND.search(question))),
        "q_recall": float(bool(_RECALL.search(question))),
        "q_past_self": float(bool(_PAST_SELF.search(question))),
        "q_yes_no": float(bool(_YES_NO.search(question))),
        "q_number": float(any(c.isdigit() for c in question)),
        "q_proper": math.log1p(_proper_nouns(question)),
    }
    for name, pattern in _WH.items():
        features[f"q_wh_{name}"] = float(bool(pattern.search(question)))
    return features


def _proper_nouns(question: str) -> int:
    """大写开头的词（不算句首与 "I"）：人名、地名、品牌——问的是具体的人和事。"""
    count = 0
    for sentence in re.split(r"[.?!]\s+", question):
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
        count += sum(1 for w in words[1:] if w[0].isupper() and w not in ("I", "I'm", "I've"))
    return count


def answer_features(answer: Answer) -> dict[str, float]:
    """回答之后的信号（r_ 组）：反应式升级用。"""
    return {
        "r_insufficient": float(answer.evidence_sufficient is False),
        "r_dont_know": float(bool(DONT_KNOW.search(answer.answer or ""))),
    }


@dataclass
class ProbeResult:
    features: dict[str, float]
    candidates: L1Candidates  # 控制器选了 L1 时直接打包，不重复检索
    ms: float


class Probe:
    def __init__(self, db: MemoryDB, l1: L1Retriever) -> None:
        self.db = db
        self.l1 = l1

    def run(
        self,
        space_id: str,
        question: str,
        question_time: datetime,
        cfg: L1Config | None = None,
    ) -> ProbeResult:
        started = time.perf_counter()
        cands = self.l1.candidates(space_id, question, question_time, cfg or L1Config())
        features = {
            **query_features(question, cands.query),
            **self.feedback(space_id, question, question_time, cands),
            **self.memory(space_id),
        }
        return ProbeResult(features, cands, (time.perf_counter() - started) * 1000)

    def memory(self, space_id: str) -> dict[str, float]:
        stats = self.db.space_stats(space_id)
        return {f"m_{name}": math.log1p(count) for name, count in stats.items()}

    def feedback(
        self, space_id: str, question: str, question_time: datetime, cands: L1Candidates
    ) -> dict[str, float]:
        """探针反馈（p_ 组）：检索分数的分布、命中的分散程度、版本冲突、实体与图片命中。"""
        features: dict[str, float] = {}
        for kind in SCORE_KINDS:
            features.update(_score_stats(f"p_{kind}", cands.routes.get((kind, "dense"), [])))
        captions = [
            hits[0][1]
            for (kind, modality), hits in cands.routes.items()
            if kind in ("image", "image_iso") and modality == "dense" and hits
        ]
        features["p_caption_top1"] = max(captions, default=0.0)
        visual = cands.routes.get(("media", "visual"), [])
        features["p_visual_top1"] = visual[0][1] if visual else 0.0
        features["p_visual_gap"] = visual[0][1] - visual[1][1] if len(visual) > 1 else 0.0

        # 前 10 轮（融合后的排名）：会话分散度、分数集中度、时间跨度、最强命中离现在多久
        top = cands.ranked[:TOP_ROUNDS]
        units = [cands.round_units[rid] for rid, _ in top]
        if units:
            mass: dict[str, float] = defaultdict(float)
            for (_, score), unit in zip(top, units, strict=True):
                mass[unit.session_id] += score
            stamps = [u.ts for u in units]
            features["p_sessions"] = len(mass) / TOP_ROUNDS
            features["p_session_share"] = max(mass.values()) / (sum(mass.values()) or 1.0)
            features["p_span_days"] = math.log1p(
                (max(stamps) - min(stamps)).total_seconds() / 86400
            )
            age = (question_time - units[0].ts).total_seconds() / 86400
            features["p_age_days"] = math.log1p(max(0.0, age))
        else:
            features.update(p_sessions=0.0, p_session_share=0.0, p_span_days=0.0, p_age_days=0.0)
        in_range = cands.debug.get("in_time_range", 0)
        features["p_in_range"] = math.log1p(in_range if isinstance(in_range, int) else 0)

        # 事实：与 top-1 接近的条数（同一类事情散落在多处，聚合题的信号）、被取代的旧版本
        fact_hits = cands.routes.get(("fact", "dense"), [])
        if fact_hits:
            best = fact_hits[0][1]
            features["p_fact_near"] = math.log1p(
                sum(1 for _, s in fact_hits if s >= best - NEAR_FACT)
            )
        else:
            features["p_fact_near"] = 0.0
        top_facts = [cands.facts[fid] for fid, _ in fact_hits[:TOP_ROUNDS] if fid in cands.facts]
        features["p_fact_superseded"] = math.log1p(sum(1 for f in top_facts if not f.is_current))
        fact_rounds = {
            rid for fid, _ in fact_hits[:TOP_ROUNDS] for rid in cands.fact_rounds.get(fid, [])
        }
        features["p_fact_support"] = (
            sum(1 for rid, _ in top if rid in fact_rounds) / len(top) if top else 0.0
        )
        features["p_images_top"] = math.log1p(sum(len(u.meta.get("media_ids", [])) for u in units))
        features["p_entities"] = math.log1p(self._entity_hits(space_id, question))
        return features

    def _entity_hits(self, space_id: str, question: str) -> int:
        """问题里点名了多少个记忆中的实体（名字或别名，按词边界匹配；不算用户本人）。"""
        lowered = question.lower()
        count = 0
        for entity in self.db.get_entities(space_id):
            names = [entity.name, *entity.aliases]
            for name in names:
                name = name.strip().lower()
                if len(name) < 3 or name in ("user", "the user", "用户"):
                    continue
                if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", lowered):
                    count += 1
                    break
        return count


def _score_stats(prefix: str, hits: list[tuple[str, float]]) -> dict[str, float]:
    """一路召回的分数分布：top-1、与第 2 名的差、相对候选池的 z 分数、前 10 名内的落差。"""
    scores = [s for _, s in hits]
    if not scores:
        return {f"{prefix}_{name}": 0.0 for name in ("top1", "gap", "z", "drop")}
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
    return {
        f"{prefix}_top1": scores[0],
        f"{prefix}_gap": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        f"{prefix}_z": (scores[0] - mean) / std if std > 1e-9 else 0.0,
        f"{prefix}_drop": scores[0] - scores[min(TOP_ROUNDS, len(scores)) - 1],
    }
