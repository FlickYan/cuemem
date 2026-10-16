"""冻结方法的请求快照（test_eval_leak.py）：在迷你数据上单独跑一个方法，取它写进响应缓存的全部键。

缓存键是供应商、模型、消息、参数（温度、max_tokens、response_format、工具与 tool_choice）与
extra_body 的规范化哈希，正是决定已有缓存能否命中的东西。这里只用修复会话 ID 泄漏（N12）之前
就有的接口：同一份代码在基线提交上也能原样运行，用来核对快照（见 test_eval_leak.py）。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from fakes import FakeAPI
from scripted import ScriptedModel, echo_extraction

from cuemem.config import Settings
from cuemem.evals.datasets import EvalSet
from cuemem.evals.runner import EvalRunner, RunConfig

# LoCoMo 迷你数据里问这道题时 Agent 一直检索、从不交卷：步数用完后由标准 reader 兜底作答
FORCED = "Which year"
QUERIES = ["Max", "marathon", "Alice", "Bob", "2023"]


def agent_reply(body: dict[str, Any]) -> dict[str, Any]:
    """带工具的回答（B3、B6a）：先 memory_search 一次，问"什么时候"的题再算一次日期，然后交卷。"""
    messages = body["messages"]
    content = messages[1]["content"]
    if not isinstance(content, str):
        content = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    question = content.rsplit("Question: ", 1)[-1]
    step = sum(1 for m in messages if m["role"] == "tool")
    if step == 0 or FORCED in question:
        name, args = "memory_search", {"query": QUERIES[step % len(QUERIES)]}
    elif step == 1 and question.startswith("When"):
        name, args = "date_diff", {"start": "2023-05-08", "end": "2023-05-20"}
    else:
        name = "answer"
        args = {"answer": "golden retriever", "citations": ["m1"], "evidence_sufficient": True}
    call = {"id": f"call_{step + 1}", "name": name, "arguments": json.dumps(args)}
    return {"tool_calls": [call]}


def reply(model: ScriptedModel) -> Any:
    """reader 固定回答、judge 判对，带工具的请求交给 agent_reply，其余（抽取、对账、规划……）
    交给脚本化的模型。"""

    def respond(body: dict[str, Any]) -> str | dict[str, Any]:
        if body.get("tools"):
            return agent_reply(body)
        first = body["messages"][0]
        content = body["messages"][-1]["content"]
        if isinstance(content, str) and content.startswith("I will give you"):
            return "yes"
        if first["role"] == "system" and first["content"].startswith(
            "You are a personal assistant"
        ):
            return json.dumps(
                {"answer": "golden retriever", "citations": [], "evidence_sufficient": True}
            )
        return str(model(body))

    return respond


def cache_keys(settings: Settings) -> list[str]:
    conn = sqlite3.connect(settings.paths.llm_cache)
    try:
        return sorted(row[0] for row in conn.execute("SELECT key FROM llm_cache"))
    finally:
        conn.close()


def method_keys(
    settings: Settings, fake_api: FakeAPI, evalset: EvalSet, method: str
) -> tuple[int, str]:
    """单线程跑一个方法（响应缓存是空的），返回（请求数，全部缓存键排序后的 sha256）。"""
    fake_api.chat_replies = [reply(ScriptedModel(echo_extraction))]
    config = RunConfig(
        name=method, dataset=evalset.dataset, split=None, methods=[method], workers=1
    )
    summary = EvalRunner(settings, config, http_client=fake_api.client()).run(evalset)
    assert summary.errors == 0
    keys = cache_keys(settings)
    assert len(keys) == len(fake_api.chat_requests)  # 每个请求都进了缓存，都在快照里
    return len(keys), hashlib.sha256("\n".join(keys).encode()).hexdigest()
