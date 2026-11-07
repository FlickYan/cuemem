"""实体的名字与别名在一段文字里出现了哪些（实体图连边、实体列表判断过期共用）。"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable

from cuemem.core.models import Entity


class Mentions:
    """实体图找陈述里出现了哪些实体的名字或别名（三个字及以上，不分大小写，前后不是单词字符，
    即正则的 (?<!\\w)别名(?!\\w)）。以前每条事实把每个实体的每个别名都试一遍，实体数 × 事实数：
    5000 个实体、1 万条事实要 19 秒，界面每次刷新都算。现在把别名放进字典，每条事实只在单词边界
    之间、不长于最长别名的片段里查，结果与原来的正则逐条相同（包括两头是标点的别名、中文名字在
    不分词的中文里照旧找不到）。"""

    def __init__(self, entities: Iterable[Entity]) -> None:
        self.index: dict[str, set[str]] = {}
        for e in entities:
            for name in [e.name, *e.aliases]:
                if len(name) >= 3:
                    self.index.setdefault(name.lower(), set()).add(e.id)
        self.longest = max(map(len, self.index), default=0)

    def find(self, text: str) -> set[str]:
        text = text.lower()
        if not self.index:
            return set()
        word = [c.isalnum() or c == "_" for c in text]  # 与 re 的 Unicode \w 相同
        n = len(text)
        ends = [q for q in range(1, n + 1) if q == n or not word[q]]
        found: set[str] = set()
        for p in range(n):
            if p and word[p - 1]:
                continue
            for q in ends[bisect_right(ends, p) :]:
                if q - p > self.longest:
                    break
                found |= self.index.get(text[p:q], set())
        return found
