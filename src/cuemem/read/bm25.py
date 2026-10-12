"""BM25 关键词检索：混合检索里 sparse 路的另一种实现，与 text-embedding-v4 的词项权重对比（N4）。

分词针对对话记忆做了几处处理：
- 去掉每轮开头的 "User:" "Assistant:" 角色标签（每一轮都有，只会稀释分数）；多人对话里的
  说话人名字保留（"Caroline 说了什么"时有用）；
- 数字整体保留（"8000"、"12.5"、"5k"），不拆成单个数字；
- 中文按单字 + 相邻二字组切分，不依赖分词词典。

每个记忆空间只有几百到几千条文本，索引在内存里现建，几毫秒。
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Sequence

_ROLE_LABEL = re.compile(r"^(?:User|Assistant):\s*", re.MULTILINE)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*|[\u4e00-\u9fff]")
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have he her his how
    i if in into is it its just me my of on or our she so than that the their them then there these
    they this to too us was we were what when where which who whom why will with would you your
    yours i'm i've i'd don't didn't can't it's that's there's what's""".split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    cleaned = _ROLE_LABEL.sub("", text).lower()
    tokens = [t for t in _TOKEN.findall(cleaned) if t not in STOPWORDS]
    bigrams = [
        a + b
        for a, b in itertools.pairwise(tokens)
        if len(a) == 1 and len(b) == 1 and _is_cjk(a) and _is_cjk(b)
    ]
    return tokens + bigrams


def _is_cjk(token: str) -> bool:
    return "\u4e00" <= token <= "\u9fff"


class BM25:
    def __init__(self, documents: Sequence[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        frequency: Counter[str] = Counter()
        for counts in self.docs:
            frequency.update(counts.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """返回 (文档下标, 分数)，按分数降序；没有任何词命中的文档不返回。"""
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self.idf]
        if not terms or not self.docs:
            return []
        scores: list[tuple[int, float]] = []
        for i, counts in enumerate(self.docs):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / (self.avg_length or 1))
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda pair: -pair[1])
        return scores[:k]
