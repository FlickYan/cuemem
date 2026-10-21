"""记忆图与个性化 PageRank（L2 的图扩散，设计文档 §6.5，借鉴 HippoRAG）。

节点是实体、事实和轮，边有四种：

- 事实—实体：事实的主语，以及抽取时消歧过的、事实里提到的实体；
- 事实—轮：事实的来源（溯源关系）；
- 实体—轮：实体的名字或别名出现在这一轮的原话里。抽取只把少数关系记成了事实，原话里提到得更多；
- 事实—事实：语义相近的事实（向量余弦不低于阈值，每条最多连 k 个邻居）。它把散落在不同会话、
  说的是同一类事情的事实连起来（"买了耳环""奶奶送的项链"），相当于结构化的伪相关反馈。

只用抽取出的实体建图几乎没有边可走：dev100 的空间平均约 230 条事实、150 个实体，"用户"本人连着
约 95% 的事实，其余实体的中位度数是 1。所以补了后两种边，并把枢纽实体（连着的事实超过总数一定
比例的，例如"用户"、LoCoMo 里的两位说话人）排除在图外：经过它们的扩散等于把分数均匀撒开。

个性化 PageRank：r = (1 − d)·s + d·Pᵀr，s 是种子分布（问题里点名的实体、第一轮检索排在前面的
事实与轮），d 越小越集中在种子附近（d = 0.5 时一半的分数留在种子上，四分之一到一跳邻居）。
用 scipy.sparse 迭代求解，几百个节点只要几毫秒。
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cuemem.core.models import Entity, Fact
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore
from cuemem.write.entities import is_generic, normalize_name

if TYPE_CHECKING:  # 建图时才导入 scipy（图扩散默认关闭）：打开记忆系统时少约 0.2 秒
    from scipy import sparse

NodeKind = str  # "entity" / "fact" / "round"


@dataclass(frozen=True)
class GraphConfig:
    damping: float = 0.5  # 继续沿边走的概率；1 − damping 回到种子
    iterations: int = 50
    tolerance: float = 1e-8
    hub_ratio: float = 0.2  # 连着的事实超过总数这个比例（且至少 10 条）的实体视为枢纽，不进图
    mention_edges: bool = True  # 实体—轮：名字出现在原话里
    similar_edges: bool = True  # 事实—事实：语义相近
    similar_threshold: float = 0.8
    similar_k: int = 5
    min_alias_chars: int = 3  # 太短的名字（"Al""Jo"）在原话里误匹配太多，不用来连边


@dataclass
class MemoryGraph:
    ids: list[str]
    kinds: list[NodeKind]
    transition: sparse.csr_matrix  # 按行归一化的转移矩阵 P
    index: dict[str, int]
    alias_pattern: re.Pattern[str] | None
    alias_to_entity: dict[str, str]
    entity_degree: dict[str, int]
    hubs: set[str]
    stats: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls, db: MemoryDB, vectors: VectorStore, space_id: str, cfg: GraphConfig
    ) -> MemoryGraph:
        facts = db.get_facts(space_id)
        entities = {e.id: e for e in db.get_entities(space_id)}
        rounds = db.list_units(space_id, kinds=["round"])

        degree: dict[str, int] = defaultdict(int)
        for fact in facts:
            for eid in _fact_entities(fact):
                degree[eid] += 1
        hub_limit = max(10, cfg.hub_ratio * len(facts))
        hubs = {eid for eid, n in degree.items() if n > hub_limit}
        kept = [e for e in entities.values() if e.id not in hubs]

        ids = [e.id for e in kept] + [f.id for f in facts] + [r.id for r in rounds]
        kinds = ["entity"] * len(kept) + ["fact"] * len(facts) + ["round"] * len(rounds)
        index = {node: i for i, node in enumerate(ids)}
        edges: dict[tuple[int, int], float] = {}

        def connect(a: str, b: str, weight: float = 1.0) -> None:
            if a in index and b in index and a != b:
                i, j = sorted((index[a], index[b]))
                edges[(i, j)] = max(edges.get((i, j), 0.0), weight)

        turn_to_round = {t: r.id for r in rounds for t in r.source_turn_ids}
        for fact in facts:
            for eid in _fact_entities(fact):
                connect(fact.id, eid)
            for turn_id in fact.source_turn_ids:
                if turn_id in turn_to_round:
                    connect(fact.id, turn_to_round[turn_id])
        stats = {"fact_entity_or_round": len(edges)}

        alias_to_entity = _alias_map(kept, cfg.min_alias_chars)
        pattern = _alias_pattern(alias_to_entity)
        if cfg.mention_edges and pattern is not None:
            before = len(edges)
            for unit in rounds:
                for alias in {m.casefold() for m in pattern.findall(unit.text)}:
                    mentioned = alias_to_entity.get(alias)
                    if mentioned is not None:
                        connect(mentioned, unit.id)
            stats["entity_round"] = len(edges) - before

        if cfg.similar_edges and len(facts) > 1:
            before = len(edges)
            for a, b, weight in _similar_pairs(vectors, space_id, facts, cfg):
                connect(a, b, weight)
            stats["fact_fact"] = len(edges) - before

        from scipy import sparse

        n = len(ids)
        rows = [i for i, _ in edges] + [j for _, j in edges]
        cols = [j for _, j in edges] + [i for i, _ in edges]
        data = list(edges.values()) * 2
        adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
        out_weight = np.asarray(adjacency.sum(axis=1)).ravel()
        inverse = np.divide(1.0, out_weight, out=np.zeros_like(out_weight), where=out_weight > 0)
        transition = sparse.csr_matrix(sparse.diags(inverse) @ adjacency)
        stats.update(
            nodes=n, edges=len(edges), entities=len(kept), facts=len(facts), rounds=len(rounds)
        )
        return cls(
            ids=ids,
            kinds=kinds,
            transition=transition,
            index=index,
            alias_pattern=pattern,
            alias_to_entity=alias_to_entity,
            entity_degree=dict(degree),
            hubs=hubs,
            stats=stats,
        )

    def link_entities(self, texts: Sequence[str]) -> dict[str, float]:
        """在问题与规划出的实体名里找出图中的实体。越专指（连着的事实越少）的实体权重越高。"""
        if self.alias_pattern is None:
            return {}
        found: dict[str, float] = {}
        for text in texts:
            for alias in {m.casefold() for m in self.alias_pattern.findall(text)}:
                eid = self.alias_to_entity.get(alias)
                if eid is not None and eid in self.index:
                    found[eid] = 1.0 / math.log2(2 + self.entity_degree.get(eid, 0))
        return found

    def personalized_pagerank(self, seeds: dict[str, float], cfg: GraphConfig) -> dict[str, float]:
        """以 seeds（节点 ID → 权重）为重启分布的 PageRank，返回各节点的分数（总和为 1）。"""
        start = np.zeros(len(self.ids))
        for node, weight in seeds.items():
            if node in self.index and weight > 0:
                start[self.index[node]] += weight
        if start.sum() == 0:
            return {}
        start /= start.sum()
        transposed = self.transition.T.tocsr()
        dangling = np.asarray(self.transition.sum(axis=1)).ravel() == 0
        scores = start.copy()
        for _ in range(cfg.iterations):
            # 没有出边的节点（孤立的轮）把分数交回种子分布，保证总和不变
            spread = transposed @ scores + scores[dangling].sum() * start
            updated = (1 - cfg.damping) * start + cfg.damping * spread
            if np.abs(updated - scores).sum() < cfg.tolerance:
                scores = updated
                break
            scores = updated
        return {self.ids[i]: float(scores[i]) for i in np.flatnonzero(scores)}

    def ranked(self, scores: dict[str, float], kind: NodeKind) -> list[tuple[str, float]]:
        """某一类节点按 PageRank 分数从高到低排列。"""
        items = [
            (node, score)
            for node, score in scores.items()
            if score > 0 and self.kinds[self.index[node]] == kind
        ]
        return sorted(items, key=lambda item: (-item[1], item[0]))


def _fact_entities(fact: Fact) -> list[str]:
    return list(dict.fromkeys([fact.subject_id, *fact.entity_ids]))


def _alias_map(entities: Sequence[Entity], min_chars: int) -> dict[str, str]:
    """名字与别名（小写）→ 实体 ID。泛称（"my dog""那只金毛"）与太短的名字不用。

    同一个名字对应多个实体时（两个不同的 "Sarah"），按出现顺序保留第一个。
    """
    out: dict[str, str] = {}
    for entity in entities:
        for name in [entity.name, *entity.aliases]:
            key = name.strip().casefold()
            if len(key) < min_chars or is_generic(name) or not normalize_name(name):
                continue
            out.setdefault(key, entity.id)
    return out


def _alias_pattern(aliases: dict[str, str]) -> re.Pattern[str] | None:
    if not aliases:
        return None
    ordered = sorted(aliases, key=len, reverse=True)  # 长名字优先："Linda Woo" 先于 "Linda"
    body = "|".join(re.escape(a) for a in ordered)
    return re.compile(rf"(?<![\w])(?:{body})(?![\w])", re.IGNORECASE)


def _similar_pairs(
    vectors: VectorStore, space_id: str, facts: Sequence[Fact], cfg: GraphConfig
) -> list[tuple[str, str, float]]:
    stored = vectors.dense_vectors(space_id, [f.id for f in facts])
    ids = [f.id for f in facts if f.id in stored]
    if len(ids) < 2:
        return []
    matrix = np.vstack([stored[i] for i in ids]).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1.0, norms)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -1.0)
    pairs: list[tuple[str, str, float]] = []
    k = min(cfg.similar_k, len(ids) - 1)
    for i in range(len(ids)):
        neighbours = np.argpartition(-similarity[i], k - 1)[:k]
        for j in neighbours:
            if similarity[i, j] >= cfg.similar_threshold:
                pairs.append((ids[i], ids[j], float(similarity[i, j])))
    return pairs
