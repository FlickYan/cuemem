"""OpenAI Agents SDK + CueMem：两段对话，第二段记得第一段说过的事。

    uv sync --extra local --extra openai-agents
    export DEEPSEEK_API_KEY=sk-...
    CUEMEM_PRESET=deepseek CUEMEM_HOME=$(mktemp -d) \\
        uv run python examples/integrations/openai_agents_example.py

Agent 用 CueMem 配置里 agent 角色的模型（同一个模型目录、同一个 Key，走 OpenAI 兼容的 Chat
Completions）：deepseek 预设是 deepseek-flash（关掉思考模式），抽取也用它，向量在本机算。
花费不到一美分。只有通义千问（QwenCloud）的 Key 时：设 DASHSCOPE_API_KEY、不设 CUEMEM_PRESET，
用默认配置（qwen3.8-flash），不需要 --extra local。

数据写在 CUEMEM_HOME（不设时：源码仓库里是仓库根目录）；演示用户每次运行先清空。数据目录会记住
向量模型：deepseek 预设（本机向量）别用默认配置的数据目录，上面的命令用一个临时目录。
"""

from __future__ import annotations

import asyncio

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
    "recall to look deeper, and forget when the user asks you to forget something. "
    "Keep answers short."
)


def agent_model(memory: Memory) -> tuple[OpenAIChatCompletionsModel, ModelSettings]:
    """CueMem 配置里 agent 角色的模型（与记忆代理转发时相同的模型目录与 Key）。"""
    settings = memory.settings
    spec = settings.models[settings.roles["agent"].model]
    provider = settings.providers[spec.provider]
    client = AsyncOpenAI(base_url=provider.base_url, api_key=provider.api_key())
    model = OpenAIChatCompletionsModel(spec.model_id, client)
    return model, ModelSettings(extra_body=spec.extra_body or None)  # 例如关掉思考模式


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
    set_tracing_disabled(True)  # trace 默认发往 OpenAI（包括注入的记忆）；这里没有 OpenAI 的 Key
    # CUEMEM_PRESET 选预设；不设时用默认配置：抽取、检索与 agent 角色用 QwenCloud（默认的回答模型是
    # DeepSeek，这里用不到）
    memory = Memory()
    memory.delete_all(user_id=USER)  # 演示用户从空白开始

    model, settings = agent_model(memory)
    agent = Agent[None](
        name="Assistant",
        instructions=INSTRUCTIONS,
        model=model,
        model_settings=settings,
        # 对话由 CueMemSession 写进记忆，不给 remember（否则模型会把同一件事再记一遍）
        tools=[t for t in memory_tools(memory, user_id=USER) if t.name != "remember"],
    )
    recall = memory_input_filter(memory, user_id=USER)  # 每次调用模型前：控制器先选档
    config = RunConfig(call_model_input_filter=recall)

    print("=== Conversation 1 ===")
    async with CueMemSession("monday", memory, user_id=USER) as monday:
        await say(agent, monday, config, recall, "Hi! I just moved to Lisbon and I'm vegetarian.")
        await say(agent, monday, config, recall, "What's the capital of Japan?")
        # 对话结束：交给巩固抽取事实。这里同步等它做完，好让下一段马上用上
        # （不调用的话，async with 结束时交给后台；连 async with 都没有时，空闲 30 分钟后交出）
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
