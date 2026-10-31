"""实体消歧：把每个会话里提到的名字对应到记忆空间中已有的实体，或新建实体。

三级策略，越往后越贵、越少用到：
1. 名字或别名精确匹配（规范化后）；
2. 描述向量相似度：高于合并阈值直接合并，低于灰区下限视为新实体；
3. 灰区（相似但不确定）才调用 LLM 确认，同一会话的灰区实体合并成一次调用。

"我的狗""那只金毛"这类泛称不进别名索引：它们在不同时间可能指不同的对象。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, field_validator

from cuemem.core.ids import stable_id
from cuemem.core.models import Entity, EntityType
from cuemem.core.trace import span
from cuemem.embed.base import FloatArray
from cuemem.errors import ContentFilteredError
from cuemem.llm.client import LLMClient
from cuemem.write.extract import ExtractedEntity
from cuemem.write.predicates import lossy, unicode_key

USER_NAMES = frozenset({"user", "the user", "i", "me", "myself", "用户", "我"})
ASSISTANT_NAMES = frozenset({"assistant", "the assistant", "ai", "ai assistant", "助手"})
_GENERIC_PREFIX = re.compile(
    r"^(my|the|a|an|our|his|her|their|this|that|your)\s|^(我的|我家|我们的|那个|这个|那只|这只)"
)

ENTITY_SYSTEM = """You link entity mentions to a memory of known entities.
For each new entity, decide whether it is the same real-world entity as one of the known candidates listed under it. Answer with that candidate's id, or null if it is a different entity. Similar names or descriptions are not enough: they must refer to the same individual person, pet, place, object or event."""  # noqa: E501


class EntityMatch(BaseModel):
    new: int
    same_as: str | None = None

    @field_validator("same_as", mode="before")
    @classmethod
    def _same_as(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip().strip("[]").lower()
        return text or None


class EntityMatches(BaseModel):
    matches: list[EntityMatch] = []


def normalize_name(name: str) -> str:
    """别名索引与实体 ID 用的名字键。拉丁字母与汉字以外的文字（'Анна'、'민준'）按 unicode_key
    规范化：原来的规则只留 [0-9a-z] 与汉字，这些名字都成了 ''。"""
    if lossy(name):
        return unicode_key(name)
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", name.casefold()).strip()


def is_generic(alias: str) -> bool:
    """泛称（"my dog""那只金毛"）不能当别名：同一个说法在不同时间可能指不同对象。"""
    return bool(_GENERIC_PREFIX.match(normalize_name(alias) + " ")) or len(alias.strip()) < 2


def user_entity_id(space_id: str) -> str:
    return stable_id(space_id, "entity", "user")


def is_user(name: str) -> bool:
    return normalize_name(name) in USER_NAMES


@dataclass
class EntityState:
    """一个空间的实体快照：别名索引 + 描述向量。巩固过程中随会话逐步更新。"""

    space_id: str
    entities: dict[str, Entity] = field(default_factory=dict)
    alias_index: dict[str, str] = field(default_factory=dict)  # 规范化名字 → 实体 ID
    vectors: dict[str, FloatArray] = field(default_factory=dict)

    def add(self, entity: Entity, vector: FloatArray | None) -> None:
        self.entities[entity.id] = entity
        for name in [entity.name, *entity.aliases]:
            key = normalize_name(name)
            if key and not is_generic(name):
                self.alias_index.setdefault(key, entity.id)
        if vector is not None:
            self.vectors[entity.id] = vector

    def lookup(self, name: str) -> str | None:
        key = normalize_name(name)
        if key in USER_NAMES:
            return (
                user_entity_id(self.space_id)
                if user_entity_id(self.space_id) in self.entities
                else None
            )
        return self.alias_index.get(key)


@dataclass
class ResolutionReport:
    created: int = 0
    linked_exact: int = 0
    linked_vector: int = 0
    linked_llm: int = 0
    llm_calls: int = 0
    changed: dict[str, Entity] = field(default_factory=dict)  # 本次新建或更新过的实体
    created_ids: set[str] = field(default_factory=set)
    touched: set[str] = field(default_factory=set)  # 本会话提到过的实体：mentions 每个会话只加一次


class EntityResolver:
    def __init__(
        self,
        llm: LLMClient,
        *,
        merge_threshold: float = 0.90,
        gray_threshold: float = 0.75,
        role: str = "extractor",
        fallback_model: str | None = None,
    ) -> None:
        self.llm = llm
        self.fallback_model = fallback_model
        self.merge_threshold = merge_threshold
        self.gray_threshold = gray_threshold
        self.role = role

    def ensure_people(
        self, state: EntityState, names: Sequence[str], ts: datetime, report: ResolutionReport
    ) -> None:
        """对话参与者（用户本人，或多人对话里的各位说话人）一定是实体。"""
        for name in names:
            if normalize_name(name) in ASSISTANT_NAMES:
                continue
            if normalize_name(name) in USER_NAMES:
                entity_id, display = user_entity_id(state.space_id), "User"
            else:
                key = normalize_name(name)  # 只有符号或 emoji 的名字键是 ''：按原样区分
                entity_id = stable_id(state.space_id, "entity", key or name.strip(), "person")
                display = name
            if entity_id in state.entities:
                continue
            entity = Entity(
                id=entity_id,
                space_id=state.space_id,
                name=display,
                type="person",
                description="the user"
                if display == "User"
                else "a participant in the conversations",
                first_seen=ts,
                last_seen=ts,
            )
            state.add(entity, None)
            if display != "User" and normalize_name(name):  # 空键会让任何空宾语都指向这个人
                state.alias_index[normalize_name(name)] = entity_id
            report.created += 1
            report.changed[entity_id] = entity

    def name_user(
        self, state: EntityState, name: str, ts: datetime, report: ResolutionReport
    ) -> None:
        """ "用户名叫林晓"：把名字登记为用户实体的别名，之后提到"林晓"就指向用户本人。"""
        key = normalize_name(name)
        if not key or key in USER_NAMES or is_generic(name):
            return
        self.ensure_people(state, ["User"], ts, report)
        uid = user_entity_id(state.space_id)
        if state.alias_index.get(key) not in (None, uid):
            return  # 已经是另一个实体的名字：不擅自合并
        self._link(state, uid, ExtractedEntity(name=name.strip(), type="person"), ts, report)
        state.alias_index[key] = uid

    def resolve(
        self,
        state: EntityState,
        mentions: Sequence[ExtractedEntity],
        vectors: dict[str, FloatArray],
        ts: datetime,
        report: ResolutionReport,
        *,
        cache_scope: str | None = None,
    ) -> None:
        """把本会话提到的实体并入 state。vectors：实体描述文本 → 向量（已提前批量算好）。"""
        gray: list[tuple[ExtractedEntity, list[str]]] = []
        for mention in _dedupe(mentions):
            key = normalize_name(mention.name)
            if not key or key in ASSISTANT_NAMES:
                continue
            existing = state.lookup(mention.name) or next(
                (state.lookup(a) for a in mention.aliases if not is_generic(a) and state.lookup(a)),
                None,
            )
            if existing is not None and _compatible(state.entities[existing].type, mention.type):
                self._link(state, existing, mention, ts, report)
                report.linked_exact += 1
                continue
            vector = vectors.get(descriptor(mention))
            candidates = self._similar(state, mention.type, vector)
            if candidates and candidates[0][1] >= self.merge_threshold:
                self._link(state, candidates[0][0], mention, ts, report)
                report.linked_vector += 1
            elif candidates and candidates[0][1] >= self.gray_threshold:
                gray.append(
                    (mention, [cid for cid, sim in candidates[:3] if sim >= self.gray_threshold])
                )
            else:
                self._create(state, mention, vector, ts, report)
        if gray:
            decisions = self._ask_llm(state, gray, cache_scope)
            report.llm_calls += 1
            for i, (mention, _) in enumerate(gray, start=1):
                target = decisions.get(i)
                if target is not None:
                    self._link(state, target, mention, ts, report)
                    report.linked_llm += 1
                else:
                    self._create(state, mention, vectors.get(descriptor(mention)), ts, report)

    def subject_id(
        self, state: EntityState, name: str, ts: datetime, report: ResolutionReport
    ) -> str | None:
        """事实主语对应的实体；没有登记过的主语新建一个"other"类型的实体。助手返回 None。"""
        key = normalize_name(name)
        if not key or key in ASSISTANT_NAMES:
            return None
        if key in USER_NAMES:
            self.ensure_people(state, ["User"], ts, report)
            uid = user_entity_id(state.space_id)
            if uid not in report.touched:
                self._link(state, uid, ExtractedEntity(name="User", type="person"), ts, report)
            return uid
        found = state.lookup(name)
        if found is not None:
            if found not in report.touched:
                self._link(state, found, ExtractedEntity(name=name.strip()), ts, report)
            return found
        mention = ExtractedEntity(name=name.strip(), type="other")
        return self._create(state, mention, None, ts, report)

    # ---------- 内部实现 ----------

    def _similar(
        self, state: EntityState, etype: EntityType, vector: FloatArray | None
    ) -> list[tuple[str, float]]:
        if vector is None:
            return []
        ids = [
            eid
            for eid, entity in state.entities.items()
            if eid in state.vectors and _compatible(entity.type, etype)
        ]
        if not ids:
            return []
        matrix = np.vstack([state.vectors[eid] for eid in ids])
        sims = matrix @ vector
        order = np.argsort(-sims)
        return [(ids[i], float(sims[i])) for i in order]

    def _link(
        self,
        state: EntityState,
        entity_id: str,
        mention: ExtractedEntity,
        ts: datetime,
        report: ResolutionReport,
    ) -> None:
        entity = state.entities[entity_id]
        aliases = list(entity.aliases)
        for alias in [mention.name, *mention.aliases]:
            if (
                alias
                and not is_generic(alias)
                and normalize_name(alias) != normalize_name(entity.name)
                and normalize_name(alias) not in {normalize_name(a) for a in aliases}
            ):
                aliases.append(alias.strip())
        updated = entity.model_copy(
            update={
                # 先以主语身份出现、类型未知的实体，后来知道了具体类型就补上
                "type": mention.type if entity.type == "other" else entity.type,
                "description": entity.description or mention.description,
                "aliases": aliases,
                "last_seen": max(entity.last_seen, ts),
                "mentions": entity.mentions + (0 if entity_id in report.touched else 1),
            }
        )
        report.touched.add(entity_id)
        state.add(updated, state.vectors.get(entity_id))
        report.changed[entity_id] = updated

    def _create(
        self,
        state: EntityState,
        mention: ExtractedEntity,
        vector: FloatArray | None,
        ts: datetime,
        report: ResolutionReport,
    ) -> str:
        key = normalize_name(mention.name)
        entity_id = stable_id(state.space_id, "entity", key or mention.name.strip(), mention.type)
        if entity_id in state.entities:  # 同名同类型：确定性 ID 相同，按同一实体处理
            self._link(state, entity_id, mention, ts, report)
            return entity_id
        entity = Entity(
            id=entity_id,
            space_id=state.space_id,
            name=mention.name.strip(),
            type=mention.type,
            aliases=[
                a for a in dict.fromkeys(mention.aliases) if not is_generic(a) and a != mention.name
            ],
            description=mention.description,
            first_seen=ts,
            last_seen=ts,
        )
        state.add(entity, vector)
        report.created += 1
        report.changed[entity_id] = entity
        report.created_ids.add(entity_id)
        report.touched.add(entity_id)
        return entity_id

    def _ask_llm(
        self,
        state: EntityState,
        gray: list[tuple[ExtractedEntity, list[str]]],
        cache_scope: str | None,
    ) -> dict[int, str]:
        labels: dict[str, str] = {}
        blocks: list[str] = []
        for i, (mention, candidate_ids) in enumerate(gray, start=1):
            lines = [f"New {i}: {descriptor(mention)}"]
            for eid in candidate_ids:
                label = f"c{len(labels) + 1}"
                labels[label] = eid
                lines.append(f"  [{label}] {state.entities[eid].descriptor}")
            blocks.append("\n".join(lines))
        messages = [
            {"role": "system", "content": ENTITY_SYSTEM},
            {"role": "user", "content": "\n\n".join(blocks)},
        ]
        try:
            with span("resolve_entities", items=len(gray)):
                output = self.llm.structured(
                    self.role,
                    messages,
                    EntityMatches,
                    cache_scope=cache_scope,
                    fallback_model=self.fallback_model,
                ).value
        except ContentFilteredError:  # 内容审核都拒绝：保守处理，全部视为新实体
            return {}
        allowed = {i: set(ids) for i, (_, ids) in enumerate(gray, start=1)}
        decisions: dict[int, str] = {}
        for match in output.matches:
            target = labels.get(match.same_as or "")
            if target is not None and target in allowed.get(match.new, set()):
                decisions[match.new] = target
        return decisions


def descriptor(mention: ExtractedEntity) -> str:
    return f"{mention.name} ({mention.type}): {mention.description}".strip()


def _compatible(a: str, b: str) -> bool:
    return a == b or "other" in (a, b)


def _dedupe(mentions: Sequence[ExtractedEntity]) -> list[ExtractedEntity]:
    """同一会话里重复列出的实体合并（别名取并集）。"""
    merged: dict[str, ExtractedEntity] = {}
    for mention in mentions:
        key = normalize_name(mention.name)
        if key in merged:
            first = merged[key]
            aliases = list(dict.fromkeys([*first.aliases, *mention.aliases]))
            merged[key] = first.model_copy(
                update={"aliases": aliases, "description": first.description or mention.description}
            )
        else:
            merged[key] = mention
    return list(merged.values())
