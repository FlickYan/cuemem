"""脚本化的模拟模型：按系统提示区分调用类型（抽取 / 对账 / 消歧 / 回答），返回预先写好的结果。

配合 FakeAPI 使用：fake_api.chat_replies = [ScriptedModel(...)]。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

Reply = dict[str, Any]


def _fact(
    statement: str,
    predicate: str,
    obj: str,
    *,
    subject: str = "User",
    turns: list[int],
    importance: int = 3,
    category: str = "attribute",
    cardinality: str = "single",
    date: str | None = None,
    negated: bool = False,
) -> Reply:
    return {
        "statement": statement,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "category": category,
        "cardinality": cardinality,
        "negated": negated,
        "date": date,
        "importance": importance,
        "turns": turns,
    }


# examples/toy_conversation.json 每个会话的抽取结果（按会话日期索引）
TOY_EXTRACTIONS: dict[str, Reply] = {
    "2025-03-01": {
        "summary": "用户介绍自己是北京一家互联网公司的产品经理，周末领养了金毛 Max。",
        "facts": [
            _fact("用户名叫林晓。", "name", "林晓", turns=[0], importance=5),
            _fact(
                "用户在北京一家互联网公司工作。",
                "works_at",
                "北京一家互联网公司",
                turns=[0],
                importance=5,
            ),
            _fact("用户是产品经理。", "job_title", "产品经理", turns=[0], importance=4),
            _fact(
                "用户在 2025 年 2 月底领养了一只名叫 Max 的金毛。",
                "has_pet",
                "Max",
                turns=[2],
                importance=4,
                category="relationship",
                cardinality="multi",
                date="2025-02",
            ),
            _fact("Max 是一只金毛寻回犬。", "breed", "金毛", subject="Max", turns=[2]),
        ],
        "entities": [
            {"name": "Max", "type": "pet", "aliases": ["我的狗"], "description": "用户领养的金毛"}
        ],
    },
    "2025-04-12": {
        "summary": "用户想买一台 1.5 公斤以下的笔记本，预算约 8000 元。",
        "facts": [
            _fact(
                "用户买笔记本的预算约 8000 元。",
                "laptop_budget",
                "8000 元",
                turns=[0],
                category="plan",
            ),
            _fact(
                "用户想要 1.5 公斤以下的笔记本。",
                "laptop_weight_preference",
                "1.5 公斤以下",
                turns=[2],
                category="preference",
            ),
        ],
        "entities": [],
    },
    "2025-06-20": {
        "summary": "Max 挑食；用户计划年底去巴厘岛旅行。",
        "facts": [
            _fact(
                "Max 只肯吃鸡肉味的狗粮。",
                "food_preference",
                "鸡肉味狗粮",
                subject="Max",
                turns=[0],
                importance=2,
                category="preference",
            ),
            _fact(
                "用户计划在 2025 年底去巴厘岛旅行。",
                "plans_to",
                "年底去巴厘岛旅行",
                turns=[2],
                category="plan",
                cardinality="multi",
                date="2025-12",
            ),
        ],
        "entities": [{"name": "巴厘岛", "type": "place", "description": "印尼的旅游目的地"}],
    },
    "2025-09-10": {
        "summary": "用户 8 月跳槽到新加坡一家电商公司做 AI 产品经理，笔记本预算降到 6000 元。",
        "facts": [
            _fact(
                "用户于 2025 年 8 月跳槽到新加坡一家电商公司。",
                "works_at",
                "新加坡一家电商公司",
                turns=[0],
                importance=5,
                date="2025-08",
            ),
            _fact(
                "用户现在是 AI 产品经理。",
                "job_title",
                "AI 产品经理",
                turns=[0],
                importance=4,
                date="2025-08",
            ),
            _fact(
                "用户搬到了新加坡。",
                "lives_in",
                "新加坡",
                turns=[0, 2],
                importance=5,
                date="2025-08",
            ),
            _fact(
                "用户把买笔记本的预算降到了 6000 元。",
                "laptop_budget",
                "6000 元",
                turns=[2],
                category="plan",
            ),
        ],
        "entities": [],
    },
    "2025-09-28": {
        "summary": "Max 跟用户来了新加坡，喜欢在东海岸公园散步。",
        "facts": [
            _fact("Max 跟用户一起搬到了新加坡。", "lives_in", "新加坡", subject="Max", turns=[0]),
            _fact(
                "Max 喜欢在东海岸公园散步。",
                "likes",
                "在东海岸公园散步",
                subject="Max",
                turns=[0],
                category="preference",
                cardinality="multi",
            ),
        ],
        "entities": [{"name": "东海岸公园", "type": "place", "description": "新加坡的海滨公园"}],
    },
}

EMPTY_EXTRACTION: Reply = {"summary": "", "facts": [], "entities": []}

# 示例对话里真正的"更新"：换公司、换职位、预算下调
TOY_UPDATE_KEYWORDS = ("预算", "公司", "产品经理")


def reconcile_by_keyword(*keywords: str, action: str = "update") -> Callable[[str], Reply]:
    """新事实与某条已有事实含有同一个关键词时给出 action，否则 add。"""

    def decide(prompt: str) -> Reply:
        decisions = []
        for block in prompt.split("\n\n"):
            head = re.match(r"New fact (\d+): (.*)", block)
            if not head:
                continue
            target = None
            for keyword in (k for k in keywords if k in head.group(2)):
                for label, text in re.findall(r"\[(e\d+)\] (.*)", block):
                    if keyword in text and target is None:
                        target = label
            decisions.append(
                {"new": int(head.group(1)), "action": action if target else "add", "target": target}
            )
        return {"decisions": decisions}

    return decide


def echo_extraction(prompt: str) -> Reply:
    """通用抽取替身：把每条用户发言原样记成一条事实（评测流程测试用）。"""
    facts = [
        _fact(text, "said", text[:40], turns=[int(n)], importance=4, cardinality="multi")
        for n, text in re.findall(r"\[t(\d+)\] User: (.*)", prompt)
    ]
    first = facts[0]["statement"] if facts else ""
    return {"summary": first, "facts": facts, "entities": []}


def growing_extraction(prompt: str) -> Reply:
    """抽取替身：会话越长说得越具体（真实的抽取模型常这样换说法）。"I like hiking" 在两句的会话
    里抽成 "The user likes hiking."，会话长了抽成 "The user enjoys hiking in the Alps."；
    其余用户发言记成 "The user said: ..."。"""
    hiking = (
        "The user enjoys hiking in the Alps."
        if len(re.findall(r"\[t\d+\]", prompt)) > 2
        else "The user likes hiking."
    )
    facts = []
    for n, text in re.findall(r"\[t(\d+)\] User: (.*)", prompt):
        statement = hiking if "hiking" in text else f"The user said: {text}"
        facts.append(_fact(statement, "said", statement[:40], turns=[int(n)], cardinality="multi"))
    return {"summary": "", "facts": facts, "entities": []}


class ScriptedModel:
    def __init__(
        self,
        extractions: dict[str, Reply] | Callable[[str], Reply] | None = None,
        *,
        reconcile: Callable[[str], Reply] | None = None,
        entity: Callable[[str], Reply] | None = None,
        reader: Callable[[str], Reply] | None = None,
        planner: Callable[[str], Reply] | None = None,
        checker: Callable[[str], Reply] | None = None,
        captioner: Callable[[str], Reply] | None = None,
        router: Callable[[str], Reply] | None = None,
        agent: Callable[[dict[str, Any]], str | Reply] | None = None,
        regrown: Callable[[str], Reply] | None = None,
    ) -> None:
        self.extractions = extractions if extractions is not None else {}
        # 对话 Agent（N8）：拿到整个请求体（消息与工具），返回文字回答或工具调用
        # （{"tool_calls": [...]}）；默认回答 "OK"
        self.agent = agent or (lambda body: "OK")
        self.agent_requests: list[dict[str, Any]] = []
        # 零样本 LLM 路由（N7）：默认按问题里的线索粗分——"一共"走深度检索，问常识的不用记忆
        self.router = router or (
            lambda prompt: {
                "level": "DEEP" if "一共" in prompt else "NONE" if "首都" in prompt else "FAST",
                "look_at_images": False,
            }
        )
        self.reconcile = reconcile or (lambda prompt: {"decisions": []})
        # 遗忘后的复述检查（N10）：默认没有复述
        self.regrown = regrown or (lambda prompt: {"same": []})
        self.entity = entity or (lambda prompt: {"matches": []})
        self.reader = reader or (
            lambda prompt: {"answer": "ok", "citations": [], "evidence_sufficient": True}
        )
        self.planner = planner or (lambda prompt: {"answer_type": "fact", "sub_queries": []})
        self.checker = checker or (lambda prompt: {"sufficient": True})
        self.captioner = captioner or (
            lambda prompt: {
                "caption": "a red sign" if "marked >>>" in prompt else "an isolated red sign",
                "text_in_image": "CUE 42",
                "tags": ["sign"],
            }
        )
        self.images: list[int] = []  # 每次调用附带的图片张数
        self.calls: Counter[str] = Counter()
        self.prompts: dict[str, list[str]] = defaultdict(list)

    def __call__(self, body: dict[str, Any]) -> str | Reply:
        messages = body["messages"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "How your memory works:" in system:
            self.calls["agent"] += 1
            self.agent_requests.append(body)
            return self.agent(body)
        content = messages[-1]["content"]
        if isinstance(content, list):  # 带图的请求：文本块拼起来，记下图片张数
            prompt = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            self.images.append(sum(1 for b in content if b.get("type") == "image_url"))
        else:
            prompt = content
        if system.startswith("You build the long-term memory"):
            kind = "extract"
            if callable(self.extractions):
                out = self.extractions(prompt)
            else:
                date = re.search(r"Conversation date: (\d{4}-\d{2}-\d{2})", prompt)
                out = self.extractions.get(date.group(1) if date else "", EMPTY_EXTRACTION)
        elif system.startswith("You maintain a long-term memory"):
            kind, out = "reconcile", self.reconcile(prompt)
        elif system.startswith("You link entity mentions"):
            kind, out = "entity", self.entity(prompt)
        elif system.startswith("You are a personal assistant"):
            kind, out = "read", self.reader(prompt)
        elif system.startswith("You plan memory retrieval"):
            kind, out = "plan", self.planner(prompt)
        elif system.startswith("You check whether memories"):
            kind, out = "check", self.checker(prompt)
        elif system.startswith("You describe an image"):
            kind, out = "caption", self.captioner(prompt)
        elif system.startswith("You decide how much memory retrieval"):
            kind, out = "route", self.router(prompt)
        elif system.startswith("You check whether information the user asked to forget"):
            kind, out = "regrown", self.regrown(prompt)
        else:
            self.calls["other"] += 1
            return "OK"
        self.calls[kind] += 1
        self.prompts[kind].append(prompt)
        return json.dumps(out, ensure_ascii=False)
