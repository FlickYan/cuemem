"""演示剧本的回放与核对（N8）。

剧本（demo/shop/script.json）是几段模拟日期的会话：每段里是用户的消息（可带图），以及可选的
"遗忘"动作。回放时每段会话结束后同步巩固，下一段就能用上抽取出的事实与画像。每条消息可以写：

- expect：几组关键词，每组里至少出现一个（不区分大小写）——回答对不对；
- expect_not：都不能出现——比如素食的用户不该被推荐皮鞋，遗忘之后不该再说出生日；
  以 "re:" 开头的是正则表达式（例如"带着价格出现"才算推荐，提一句"跳过了皮鞋"不算）；
- expect_level：希望控制器选的档（只报告，不算失败：控制器是按概率选的）。

回放用于两件事：演示前把前几段会话建好（--until），以及整段剧本端到端核对。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from cuemem.agent.runtime import Reply
from cuemem.api.service import ChatService
from cuemem.read.controller import Router


class ScriptMessage(BaseModel):
    text: str | None = None
    images: list[str] = Field(default_factory=list)  # 相对剧本文件的路径
    expect: list[list[str]] = Field(default_factory=list)
    expect_not: list[str] = Field(default_factory=list)
    expect_level: str | None = None
    forget: str | None = None  # 遗忘：当前事实里含这个词的，删掉它们出自的整轮对话
    note: str = ""


class ScriptSession(BaseModel):
    title: str = ""
    at: datetime
    live: bool = False  # 演示时现场操作的一段（回放 --until 不含它）
    messages: list[ScriptMessage]


class DemoScript(BaseModel):
    title: str = ""
    space: str
    note: str = ""
    sessions: list[ScriptSession]

    @classmethod
    def load(cls, path: Path) -> DemoScript:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass
class TurnResult:
    session: int  # 第几段会话，从 1 起
    index: int  # 段内第几条，从 1 起
    text: str
    reply: Reply | None = None
    failures: list[str] = field(default_factory=list)
    level: str = ""  # 实际的档位（再感知记作 +see）
    expected_level: str | None = None
    forgot: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "session": self.session,
            "index": self.index,
            "text": self.text,
            "passed": self.passed,
            "failures": self.failures,
            "level": self.level,
            "expected_level": self.expected_level,
        }
        if self.forgot is not None:
            out["forgot"] = self.forgot
        if self.reply is not None:
            out["reply"] = self.reply.model_dump(mode="json")
        return out


def check(text: str, message: ScriptMessage) -> list[str]:
    """回答是否满足剧本的期望；返回不满足的地方。以 "re:" 开头的是正则表达式。"""
    failures = [
        f"缺少 {' / '.join(group)}"
        for group in message.expect
        if not any(_found(word, text) for word in group)
    ]
    failures += [f"不该出现 {word}" for word in message.expect_not if _found(word, text)]
    return failures


def _found(pattern: str, text: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], text, re.IGNORECASE) is not None
    return pattern.lower() in text.lower()


def replay(
    service: ChatService,
    script: DemoScript,
    *,
    root: Path,
    space: str | None = None,
    until: int | None = None,
    router: Router | None = None,
    lam: float | None = None,
    on_turn: Callable[[TurnResult], None] | None = None,
) -> list[TurnResult]:
    """按剧本对话；每段会话结束后同步巩固。until：只回放前几段。

    每条消息的时间固定为会话开始后每条 2 分钟（不随真实耗时变化），同一份剧本重放时请求
    逐字相同，命中本地缓存，结果可复现。
    """
    space_id = space or script.space
    results: list[TurnResult] = []
    for number, scripted in enumerate(script.sessions, start=1):
        if until is not None and number > until:
            break
        # 会话 ID 固定（召回记忆的标题里会写出会话 ID，随机的 ID 会让重放的请求对不上缓存）
        session = service.start(
            space_id, at=scripted.at, session_id=f"s{number}-{scripted.at:%Y%m%d-%H%M}"
        )
        for index, message in enumerate(scripted.messages, start=1):
            if message.forget:
                result = TurnResult(number, index, f"[forget] {message.forget}")
                result.forgot = _forget(service, space_id, message.forget)
            else:
                text = message.text or ""
                reply = service.chat(
                    session.session_id,
                    text,
                    images=[str(root / p) for p in message.images],
                    at=scripted.at + timedelta(minutes=2 * (index - 1)),
                    router=router,
                    lam=lam,
                )
                level = reply.trace.level + ("+see" if reply.trace.see else "")
                result = TurnResult(
                    number,
                    index,
                    text,
                    reply=reply,
                    failures=check(reply.plain, message),
                    level=level,
                    expected_level=message.expect_level,
                )
            results.append(result)
            if on_turn is not None:
                on_turn(result)
        service.end(session.session_id, wait=True)
    return results


def _forget(service: ChatService, space_id: str, keyword: str) -> dict[str, Any]:
    facts = [f for f in service.facts(space_id, current_only=True) if keyword in f["statement"]]
    if not facts:
        return {"facts": [], "turns_deleted": 0}
    report = service.forget(space_id, fact_ids=[f["id"] for f in facts])
    return {
        "facts": [f["statement"] for f in facts],
        "turns_deleted": report.turns_deleted,
        "facts_before": report.facts_before,
        "facts_after": report.facts_after,
    }


def save_report(results: list[TurnResult], path: Path, **meta: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        **meta,
        "turns": [r.as_dict() for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
