"""谓词表：规范化、同义词归并与基数（单值 / 多值）。

基数决定冲突规则（设计文档 §5.3）：
- 单值谓词（住在哪、在哪工作、职位……）：同一主语只能有一个当前值，新值取代旧值；
- 多值谓词（喜欢什么、去过哪、养了什么宠物……）：累积，除非出现明确的否定。

这里登记的单值谓词可以直接按规则取代，不需要调用 LLM；没有登记的谓词由抽取模型标注基数，
有冲突迹象时交给 LLM 裁决（D-019）。
"""

from __future__ import annotations

import re
import unicodedata

from cuemem.core.models import Cardinality

# 规范谓词 → 同义写法。抽取模型被要求优先使用规范谓词，这张表兜住它没照做的情况
SYNONYMS: dict[str, tuple[str, ...]] = {
    "lives_in": (
        "resides_in",
        "residence",
        "location",
        "current_city",
        "city_of_residence",
        "lives_at",
        "living_in",
        "based_in",
        "home_city",
        "current_location",
    ),
    "works_at": (
        "employer",
        "works_for",
        "company",
        "workplace",
        "employed_at",
        "employed_by",
        "current_employer",
    ),
    "job_title": (
        "occupation",
        "job",
        "profession",
        "role",
        "position",
        "works_as",
        "title",
        "job_role",
        "current_job",
        "current_role",
    ),
    "relationship_status": ("marital_status", "dating_status"),
    "age": ("current_age",),
    "birthday": ("date_of_birth", "born_on", "birth_date", "birthdate"),
    "nationality": ("citizenship",),
    "salary": ("income", "annual_salary"),
    "weight": ("body_weight", "current_weight"),
    "height": (),
    "name": ("full_name", "is_named"),
    "likes": ("enjoys", "loves", "is_fond_of", "favorite_activity", "interested_in"),
    "dislikes": ("hates", "does_not_like"),
    "owns": ("possesses", "bought_item"),
    "has_pet": ("pet", "owns_pet"),
    "visited": ("traveled_to", "went_to", "been_to"),
    "attended": ("participated_in", "went_to_event"),
    "hobby": ("hobbies",),
}

# 登记过的单值谓词：同一主语、同一谓词、不同宾语 → 直接按规则取代
SINGLE_VALUED = frozenset(
    {
        "lives_in",
        "works_at",
        "job_title",
        "relationship_status",
        "age",
        "birthday",
        "nationality",
        "salary",
        "weight",
        "height",
        "name",
    }
)

# 登记过的多值谓词：即使抽取模型标成单值，也按多值处理
MULTI_VALUED = frozenset(
    {
        "likes",
        "dislikes",
        "owns",
        "has_pet",
        "visited",
        "attended",
        "hobby",
        "bought",
        "uses",
        "plans_to",
        "has_family_member",
        "has_friend",
        "education",
    }
)

# 写进抽取提示词的推荐谓词；容易混淆的几个附上释义
SUGGESTED = (
    "lives_in (home city or home; not a trip or temporary stay), "
    "works_at (employer or workplace, not the city), "
    "job_title (role or position), relationship_status, age, birthday, education, "
    "likes, dislikes, owns, has_pet, has_family_member, has_friend, visited, attended, "
    "bought, uses, hobby, goal, plans_to, health_condition, routine"
)

_CANONICAL = {alias: canon for canon, aliases in SYNONYMS.items() for alias in aliases}

# ASCII 与汉字以外的字符：一个都没有的文本（绝大多数）不用逐字检查
_BEYOND = re.compile(r"[^\x00-\x7f\u4e00-\u9fff]")
_LATIN_OR_CJK = ("LATIN", "FULLWIDTH LATIN", "FULLWIDTH DIGIT", "CJK")


def lossy(text: str) -> bool:
    """原来的规范化会丢掉这段文本里的字：含有西里尔、希腊、韩文、假名、阿拉伯、希伯来等
    文字（拉丁字母与汉字以外、又不是 '²' '½' 这类数字符号的字母或数字）。

    带重音的拉丁字母（Zoë、Wüsthof）不算：照旧按原来的规则规范化（'Zoë' → 'zo'），评测里
    已有的事实、实体与对账用到的键一个都不变。
    """
    if not _BEYOND.search(text):
        return False
    return any(
        ch.isalnum()
        and _BEYOND.match(ch)
        and unicodedata.category(ch) != "No"
        and not unicodedata.name(ch, "").startswith(_LATIN_OR_CJK)
        for ch in text
    )


def unicode_key(text: str) -> str:
    """不限文字的规范形式：NFKC、casefold，连续的非字母数字换成一个空格。"""
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def normalize_predicate(predicate: str) -> str:
    """转成 snake_case 并归并同义词：'Lives In' → 'lives_in'，'employer' → 'works_at'。"""
    text = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "_", predicate.strip().lower()).strip("_")
    if not text or lossy(predicate):  # 'живёт_в'、'사는_곳' 原来都成了 related_to
        folded = unicodedata.normalize("NFKC", predicate).casefold()
        text = re.sub(r"[\W_]+", "_", folded).strip("_")
    text = text or "related_to"
    return _CANONICAL.get(text, text)


def registered_cardinality(predicate: str) -> Cardinality | None:
    """登记表中的基数；未登记返回 None（以抽取模型的标注为准）。"""
    if predicate in SINGLE_VALUED:
        return "single"
    if predicate in MULTI_VALUED:
        return "multi"
    return None


def normalize_value(text: str) -> str:
    """比较宾语是否相同时用的规范形式：小写、去标点与多余空白、去掉常见冠词。

    含有拉丁字母与汉字以外文字的文本（'서울'、'Москва'）按 unicode_key 规范化：原来的规则
    只留 [0-9a-z] 与汉字，'서울' 与 '부산' 都成了 ''，被当成同一个值。
    """
    flat = unicode_key(text) if lossy(text) else _legacy_flat(text)
    return re.sub(r"^(the|a|an|my) ", "", flat)


def legacy_value_key(text: str) -> str:
    """原来的 normalize_value，只用来派生事实 ID：已有的事实 ID 一个都不变。"""
    return re.sub(r"^(the|a|an|my) ", "", _legacy_flat(text))


def _legacy_flat(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text.lower()).strip()
