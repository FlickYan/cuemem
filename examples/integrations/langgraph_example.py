"""LangChain / LangGraph 接 CueMem（N12）：create_agent + 中间件，两个线程，后一个想起前一个。

模型走 DeepSeek 的 OpenAI 兼容接口（langchain-openai 不在 CueMem 的可选依赖里，运行时临时装上）；
记忆用 deepseek 预设：抽取也用 deepseek-flash，向量在本机算，只要 DEEPSEEK_API_KEY。

    uv run --extra local --extra langgraph --with langchain-openai \\
        python examples/integrations/langgraph_example.py

数据写在临时目录里，跑完即删（设了 CUEMEM_HOME 时写在那里，不删）。每一轮打印控制器选的档：
L0 表示这句话不需要回忆（只给画像），L1 / L2 是快速 / 深度检索。
"""

from __future__ import annotations

import os
import tempfile
from contextlib import ExitStack
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from cuemem import Memory
from cuemem.integrations.langgraph import CueMemMiddleware


def main() -> None:
    llm = ChatOpenAI(
        base_url="https://api.deepseek.com",
        model="deepseek-flash",
        api_key=os.environ["DEEPSEEK_API_KEY"],
        extra_body={"thinking": {"type": "disabled"}},  # V4 系列默认开启思考模式
    )
    with ExitStack() as stack:
        home = os.environ.get("CUEMEM_HOME") or stack.enter_context(tempfile.TemporaryDirectory())
        preset = os.environ.get("CUEMEM_PRESET", "deepseek")
        memory = stack.enter_context(Memory(preset=preset, home=home))
        # user_id 不固定：每次调用从 config["configurable"]["user_id"] 里取（LangGraph Server 上
        # 有登录用户时用登录用户）；tools=True 顺带给模型 recall / remember / forget 三个工具
        cuemem = CueMemMiddleware(memory, tools=True)
        agent = create_agent(
            llm,
            system_prompt="You are a friendly running coach. Keep answers under 80 words.",
            middleware=[cuemem],
            checkpointer=InMemorySaver(),
        )

        def chat(thread_id: str, text: str) -> dict[str, Any]:
            config: dict[str, Any] = {"configurable": {"thread_id": thread_id, "user_id": "alice"}}
            out = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
            recall = out.get("cuemem") or {}
            why = "; ".join(recall.get("reasons", []))
            level = f"{recall.get('level')} ({why})" if why else recall.get("level")
            print(f"[{thread_id}] > {text}\n[{level}] {out['messages'][-1].text}\n")
            return config

        # 周一的对话
        monday = chat("monday", "Hi! I'm training for the Berlin marathon in September.")
        chat("monday", "My left knee gets sore after about 15 km, so I'm wary of long runs.")
        # 这段对话结束：整段抽取成事实（wait=True 同步完成；默认交给后台）
        ended = cuemem.end_thread(monday, wait=True)
        print("consolidated:", ended["consolidation"], "\n")
        for fact in memory.get_all(user_id="alice")["results"]:
            print(" -", fact["memory"])
        print()

        # 周五，新的线程：模型的上下文里没有周一的对话，靠 CueMem 想起来
        chat("friday", "What long run would you suggest for me this weekend?")
        chat("friday", "What's the capital of Australia?")  # 与记忆无关：控制器多半选 L0
        # 应用退出前：进行中的线程（friday）都交给后台抽取；不调用时空闲 30 分钟后也会交出
        cuemem.close()
        memory.wait()


if __name__ == "__main__":
    main()
