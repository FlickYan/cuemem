"""朴素检索（B2 基线）：把问题向量化，在"轮"上做 dense top-k。

N4 会在此基础上加入 sparse 路、RRF 融合、时间感知与重排。
"""

from __future__ import annotations

from cuemem.core.models import Hit
from cuemem.core.trace import span
from cuemem.embed.base import TextEmbedder
from cuemem.store.sqlite import MemoryDB
from cuemem.store.vector import VectorStore


class NaiveRetriever:
    def __init__(self, db: MemoryDB, vectors: VectorStore, embedder: TextEmbedder) -> None:
        self.db = db
        self.vectors = vectors
        self.embedder = embedder

    def search(self, space_id: str, query: str, k: int) -> list[Hit]:
        with span("retrieve", route="dense", k=k) as record:
            query_vec = self.embedder.embed([query], "query").dense[0]
            # 只检索"轮"：巩固之后空间里还有事实、摘要等单元，B2 保持原样以便对比
            vector_hits = self.vectors.search_dense(space_id, query_vec, k, kinds=["round"])
            units = {u.id: u for u in self.db.get_units(space_id, [h.unit_id for h in vector_hits])}
            hits = [
                Hit(unit=units[h.unit_id], score=h.score, route="dense", rank=rank)
                for rank, h in enumerate(vector_hits, start=1)
                if h.unit_id in units
            ]
            record["hits"] = [[hit.unit.id, round(hit.score, 4)] for hit in hits]
        return hits
