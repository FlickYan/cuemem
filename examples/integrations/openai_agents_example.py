"""OpenAI Agents SDK + CueMem：两段对话，第二段记得第一段说过的事。

    uv sync --extra local --extra openai-agents
    export DEEPSEEK_API_KEY=sk-...
    CUEMEM_PRESET=deepseek uv run python examples/integrations/openai_agents_example.py

Agent 走 DeepSeek 的 OpenAI 兼容接口（Chat Completions，deepseek-flash，关掉思考模式）；CueMem 用
deepseek 预设（抽取也用 deepseek-flash，向量在本机算）。花费不到一美分。

数据写在 CUEMEM_HOME（不设时：源码仓库里是仓库根目录）；演示用户每次运行先清空。换成通义千问：
base_url 用 https://maas.qwencloudapi.com/compatible-mode/v1、Key 用 DASHSCOPE_API_KEY、模型
qwen3.8-flash、extra_body={"enable_thinking": False}，CueMem 用默认配置（不设 CUEMEM_PRESET）。
"""

from __future__ import annotations

import asyncio
import os

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from cuemem import Memory
from cuemem.integrations.openai_agents import (
    CueMemSession,
    MemoryInputFilter,
    memory_input_filter,
    memory_tools,
)

USER = "openai-agents-demo"
INSTRUCTIONS = (
    "You are a friendly personal assistant with long-term memory. Memories recalled for the "
    "user's latest message appear in a <memory> block at its start; use them naturally. Call "
    "recall to look deeper, remember when the user shares something worth keeping, and forget "
    "when they ask you to. Keep answers short."
)


async def say(
    agent: Agent[None],
    session: CueMemSession,
    config: RunConfig,
    recall: MemoryInputFilter,
    text: str,
) -> None:
    result = await Runner.run(agent, text, session=session, run_config=config)
    evidence = recall.last
    level = f"{evidence.level}, {evidence.recalled} recalled" if evidence else "no recall"
    print(f"\nUser: {text}\n[memory controller: {level}]\nAssistant: {result.final_output}")


async def main() -> None:
    set_tracing_disabled(True)  # trace 默认发往 OpenAI；这里没有 OpenAI 的 Key
    memory = Memory(preset=os.environ.get("CUEMEM_PRESET") or "deepseek")
    memory.delete_all(user_id=USER)  # 演示用户从空白开始

    client = AsyncOpenAI(
        base_url="https://api.deepseek.com", api_key=os.environ["DEEPSEEK_API_KEY"]
    )
    agent = Agent[None](
        name="Assistant",
        instructions=INSTRUCTIONS,
        model=OpenAIChatCompletionsModel("deepseek-flash", client),
        model_settings=ModelSettings(extra_body={"thinking": {"type": "disabled"}}),
        tools=memory_tools(memory, user_id=USER),
    )
    recall = memory_input_filter(memory, user_id=USER)  # 每次调用模型前：控制器先选档
    config = RunConfig(call_model_input_filter=recall)

    print("=== Conversation 1 ===")
    async with CueMemSession("monday", memory, user_id=USER) as monday:
        await say(agent, monday, config, recall, "Hi! I just moved to Lisbon and I'm vegetarian.")
        await say(agent, monday, config, recall, "What's the capital of Japan?")
        # 对话结束：交给巩固抽取事实。这里同步等它做完，好让下一段马上用上
        report = await monday.end(wait=True)
        print(f"\n[conversation 1 consolidated: {report}]")

    print("\n=== Conversation 2 (new session, empty short-term history) ===")
    async with CueMemSession("tuesday", memory, user_id=USER) as tuesday:
        await say(agent, tuesday, config, recall, "Can you suggest a dinner idea for tonight?")
        await tuesday.end(wait=True)

    print("\n=== What CueMem knows now ===")
    for fact in memory.get_all(user_id=USER)["results"]:
        print(f"- {fact['memory']}")
    memory.close()


if __name__ == "__main__":
    asyncio.run(main())
