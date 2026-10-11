"""谓词表：规范化、同义词归并与基数（单值 / 多值）。

基数决定冲突规则（设计文档 §5.3）：
- 单值谓词（住在哪、在哪工作、职位……）：同一主语只能有一个当前值，新值取代旧值；
- 多值谓词（喜欢什么、去过哪、养了什么宠物……）：累积，除非出现明确的否定。

这里登记的单值谓词可以直接按规则取代，不需要调用 LLM；没有登记的谓词由抽取模型标注基数，
有冲突迹象时交给 LLM 裁决（D-019）。
"""

from __future__ import annotations

import re

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


def normalize_predicate(predicate: str) -> str:
    """转成 snake_case 并归并同义词：'Lives In' → 'lives_in'，'employer' → 'works_at'。"""
    text = re.sub(r"[^0-9a-zA-Z一-鿿]+", "_", predicate.strip().lower()).strip("_")
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
    """比较宾语是否相同时用的规范形式：小写、去标点与多余空白、去掉常见冠词。"""
    flat = re.sub(r"[^0-9a-z一-鿿]+", " ", text.lower()).strip()
    return re.sub(r"^(the|a|an|my) ", "", flat)
