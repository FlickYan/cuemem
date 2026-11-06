"""用 OpenAI SDK 接 CueMem 的记忆代理（N12）：只改 base_url，应用代码不动。

先在另一个终端启动代理（任选其一）：
    uv run cuemem proxy                      # 只有代理，端口 8787
    uv run cuemem serve                      # 界面 + 接口 + 代理，端口 7860（base_url 用 :7860/v1）

再运行：
    uv run python examples/openai_proxy.py

模型默认 qwen3.8-flash（默认配置）；用了预设的代理用 CUEMEM_MODEL 换成预设里的模型：
--preset openai 用 gpt-5.4-mini，deepseek 用 deepseek-flash，ollama 用 llama3.1-8b。
代理只转发它有 Key 的模型（GET /v1/models 列出这些），别的返回 503 model_not_available。

每轮的响应头里有控制器的决定：X-CueMem-Level（L0 表示这句话不需要回忆）、X-CueMem-Reasons。
会话空闲 30 分钟（cuemem proxy --idle-minutes）后整段抽取成事实；第二次运行时就能想起来。
"""

from __future__ import annotations

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("CUEMEM_PROXY_URL", "http://127.0.0.1:8787/v1"),
    api_key=os.environ.get("CUEMEM_API_TOKEN", "not-needed"),  # 代理设了访问令牌时填令牌
)

history = [{"role": "system", "content": "You are a friendly shopping assistant."}]
for text in ["Hi! I wear EU 40 running shoes.", "Any tips for my next pair?"]:
    history.append({"role": "user", "content": text})
    raw = client.chat.completions.with_raw_response.create(
        # 配置里的模型别名或模型 ID；不认识的用 agent 角色的模型
        model=os.environ.get("CUEMEM_MODEL", "qwen3.8-flash"),
        messages=history,
        user="alice",  # 记忆按用户隔离；也可以用请求头 X-CueMem-User
    )
    reply = raw.parse().choices[0].message.content or ""
    history.append({"role": "assistant", "content": reply})
    print(f"> {text}\n[{raw.headers.get('x-cuemem-level')}] {reply}\n")
