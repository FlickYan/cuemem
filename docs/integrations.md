# 框架集成（N12）

[English](en/integrations.md)

CueMem 接进现成 Agent 框架的五种方式，从改动最少到最深：

| 方式 | 适合 | 谁决定要不要想起 | 见 |
|---|---|---|---|
| OpenAI 兼容的记忆代理 | 任何用 OpenAI SDK 的应用，代码不动：Chat Completions、embeddings 与非流式的纯文字 Responses 调用 | 控制器（每轮） | `cuemem proxy`，README |
| MCP Server | Claude Code、Claude Desktop 等 MCP 客户端 | 模型调用 `recall`（控制器只决定深浅） | `cuemem mcp`（stdio），或 `cuemem serve` 的 `/mcp`（HTTP，令牌同上） |
| 框架集成（本页） | OpenAI Agents SDK、LangGraph / LangChain、Claude 的 memory tool | 控制器（每次调用模型之前），模型还可以用工具补查 | `cuemem.integrations.*` |
| Python SDK | 自己写的 Agent 循环 | 调用方 | `from cuemem import Memory` |
| HTTP 记忆接口 | 别的语言写的应用，或几个进程共用一个服务 | 调用方 | `cuemem serve` 的 `/api/v1/memories`；`from cuemem import MemoryClient` |

后两种的每个方法、接口与错误码见[接口参考](api.md)。`MemoryClient` 的方法与签名与 `Memory` 完全相同，代码在本地数据目录与共用的服务之间只换一行；下面的框架集成仍然要本地的 `Memory` 或 `AsyncMemory`，传给它们 `MemoryClient` 时在构造时就抛 TypeError（以前之后才 AttributeError；`claude.with_recall` 现在抛出，不再只是警告）。

三个集成共用 `cuemem/evidence.py`：控制器先选档（L0 不检索、只给画像），取回的事实与对话片段排成与记忆代理、回答模型相同的证据块，放在最新一句用户消息的开头；系统提示与之前的消息不动，供应商的前缀缓存照常命中。工具（recall / remember / forget）的名字、说明与语义与 MCP 相同。

- 证据块与 recall 工具的清单都声明里面是从过去的对话里引来的资料、不是指令；存下的 `<memory`、`</memory` 注入前会转义，关不上证据块。这只能减少、不能杜绝从记忆来的提示注入。
- OpenAI Agents 与 LangGraph 里，当前对话里还在请求中的轮不进证据，带照片的轮也一样（`evidence.stored_text` 与 `in_context_rounds`，与记忆代理共用）。
- 遗忘照实说：删除做完、用剩下的日志重建派生记忆却出错时（例如模型供应商余额不足），MCP、Agents 与 LangGraph 的 forget 工具回 "Deleted N conversation turn(s). Rebuilding the memory derived from the rest failed (…); it finishes on the next consolidation…"，不报成出错，模型不会告诉用户什么都没删。
- 写入要么全写、要么不留：向量化或向量库出错时什么都不写，重试不会写出重复的发言。发言提交之后不再抛出：写不进去的 trace 只在 `cuemem.core.trace` 日志里记一条警告。
- 遗忘等整次召回做完：自动召回与把 L0 改走 L1 的 recall 工具在一次读取里做完，同一空间的遗忘排在它们之前或之后，不会插在中间。

安装：`pip install "cuemem[integrations] @ git+https://github.com/FlickYan/cuemem"`（CueMem 还没有发布到 PyPI，PyPI 上名叫 cuemem 的包不是这个项目），源码仓库里 `uv sync --extra integrations`；只要一个框架时用 `openai-agents`、`langgraph` 或 `anthropic`。只有 DeepSeek 的 Key 时加上可选依赖 `local` 并用 `Memory(preset="deepseek")`。可运行的示例及各自要的 Key 与命令见 [examples/README.md](../examples/README.md)（英文）。

**高峰时段**：SDK，以及包在 `Memory` 外面的各个集成，缺省 `peak_policy="warn"`：DeepSeek 工作日的高峰时段（北京时间）里照常按高峰价调用模型，不会停下来等高峰结束。不用预设时也一样，默认配置的回答模型就是 DeepSeek。批量任务想等错峰价时用 `Memory(peak_policy="wait")`，开始等待时 SDK 在 `cuemem.sdk` 日志上记一条警告。评测与 `cuemem eval run --peak` 不受影响。

## OpenAI Agents SDK

`cuemem.integrations.openai_agents`（openai-agents 0.23）。三样东西可以分开用：

- `memory_tools(memory, user_id=...)`：recall、remember、forget 三个工具；
- `memory_input_filter(memory, user_id=...)`：交给 `RunConfig.call_model_input_filter`，SDK 每次调用模型之前先过控制器、注入证据（只改这次发给模型的副本）；
- `CueMemSession(session_id, memory, user_id=...)`：Session 协议的实现。内层会话（默认 SQLiteSession）原样保存短期历史，用户与助手的文字同时写进 CueMem（一问一答为一轮），对话结束后整段抽取成事实。

```python
from agents import Agent, RunConfig, Runner
from cuemem import Memory
from cuemem.integrations.openai_agents import (
    CueMemSession,
    close_sessions,
    memory_input_filter,
    memory_tools,
)

memory = Memory(preset="deepseek")


def user_of(context):  # 多个用户共用一个 Agent 时按运行上下文给身份
    return context["user_id"]


agent = Agent(name="Assistant", model=..., tools=memory_tools(memory, user_id=user_of))
config = RunConfig(
    call_model_input_filter=memory_input_filter(memory, user_id=user_of),
    trace_include_sensitive_data=False,  # 注入的记忆不上传到 SDK 的 trace 后台
)

async with CueMemSession("chat-1", memory, user_id="alice") as session:
    await Runner.run(
        agent, "I moved to Lisbon", session=session, run_config=config, context={"user_id": "alice"}
    )
close_sessions(memory)  # 退出前、memory.close() 之前
```

- 会话：写入时标记为进行中、不抽取；`async with` 结束、`end()`、空闲 30 分钟或 `close_sessions()` 时交给后台抽取。之后同一个 SDK 会话再有对话，写进新的一段（对话日期是新的）。
- 服务端保存对话状态（`conversation_id`、`previous_response_id`）时用 `memory_input_filter(..., placement="instructions")`：证据接在这次调用的 instructions 后面，不会随用户消息存到服务端。
- 遗忘：`pop_item`、`clear_session` 撤回的消息在长期记忆里一并遗忘；forget 工具删掉的发言，下次读历史时也从内层会话里去掉。
- 输入护栏拦下的、模型没有回应过的用户消息不写进长期记忆。
- 写不进去的留着重试：一批写不进去时（向量接口或 Qdrant 故障），`CueMemSession` 留着它，下一批或 `end()` 时先按原来的时间写它。重试没有用的（大部分 4xx、内容审核拒绝、输入不合法，例如像素超过 Pillow 上限的图片，这一批单独丢掉），或这个空间之后遗忘过发言的，丢掉（trace 事件 `agents_mirror_dropped`），不把遗忘之前的消息写回去。留着的只在进程内存里。
- 改走记忆代理：Agents SDK 缺省的模型类调用 Responses API，通常带工具或流式，记忆代理对这些回 501。用这里的集成，或者把 `OpenAIChatCompletionsModel` 指向记忆代理。

## LangGraph / LangChain

`cuemem.integrations.langgraph`（langgraph 1.2、langchain 1.4）。LangChain 1.x 的 `create_agent` 用中间件 `CueMemMiddleware`；仍在用 `langgraph.prebuilt.create_react_agent` 的，用 `recall_hook` / `save_hook`（pre_model_hook / post_model_hook）；也可以只用 `memory_tools`。

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from cuemem import Memory
from cuemem.integrations.langgraph import CueMemMiddleware

memory = Memory(preset="deepseek")  # 异步应用用 AsyncMemory
cuemem = CueMemMiddleware(memory, tools=True)  # 同时给模型 recall / remember / forget
agent = create_agent(llm, tools, middleware=[cuemem], checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "t1", "user_id": "alice"}}
out = agent.invoke({"messages": [{"role": "user", "content": "I moved to Munich"}]}, config)
out["cuemem"]  # 这一轮控制器的决定：level、reasons、recalled

cuemem.end_thread(config)  # 对话结束：整段抽取（空闲 30 分钟的线程也会交出）
cuemem.close()  # 应用退出时，memory.close() 之前；异步用 aend_thread / aclose
```

- 身份：`user_id` 不在构造时固定时，从 `config["configurable"]` 取；部署在 LangGraph Server 上时优先用登录的用户，不信任客户端传来的 ID。
- 一个 LangGraph 线程对应 CueMem 的一个会话。结构化输出、`return_direct` 工具结束的轮也会写入；LangChain 的 SummarizationMiddleware 生成的摘要不会被当成用户的话。`end_thread` 或空闲交出之后又接着聊的线程，结束时整段重新抽取，空间的派生记忆从原始日志重放，事实等于从日志重建的结果（不会留着上一次抽取的说法）。
- 遗忘：forget 删的是 CueMem 里的原始日志与派生记忆；当前线程的检查点里那几条消息还在（检查点不归 CueMem 管）。

## Claude 的 memory tool

`cuemem.integrations.claude`（anthropic 1.14，工具类型 `memory_20250818`）。Claude 的 memory tool 在客户端执行：模型对 `/memories` 目录做 view、create、str_replace、insert、delete、rename。`CueMemMemoryTool` 把这个目录接到 CueMem 上：

- 只读的生成文件：`/memories/profile.md`（画像）、`/memories/facts.md`（当前事实，带日期、ID 与有效状态），写入时报错说明；
- Claude 自己建的文件存在每个用户的小库里（数据目录下的 `claude_memory.sqlite`），每次改动同时作为一条笔记写进 CueMem 的原始日志，由巩固抽成事实；删掉文件就遗忘它写进去的内容；
- 路径严格校验（不能出 `/memories`），有大小上限；`with_recall` / `awith_recall` 在发给 Claude 之前先过控制器、注入证据。

```python
import os

import anthropic
from cuemem import Memory
from cuemem.integrations.claude import CueMemMemoryTool, delete_all, with_recall

# 默认配置：通义千问抽取与检索、DeepSeek 回答；或 Memory(preset="deepseek")
memory = Memory()
client = anthropic.Anthropic()
model = os.environ["ANTHROPIC_MODEL"]  # 支持 memory tool 的 Claude 模型

with CueMemMemoryTool(memory, user_id="alice") as tool:
    history = []  # 只追加：with_recall 只动最新一句用户消息
    for text in ["I moved to Lisbon in 2026-09.", "What should I cook tonight?"]:
        user = {"role": "user", "content": text}
        history, evidence = with_recall([*history, user], memory, user_id="alice")
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=16000,
            tools=[tool],
            messages=history,
        )
        for message in runner:
            history.append(message.to_param())
            results = runner.generate_tool_call_response()
            if results is not None:
                history.append(results)

delete_all(memory, user_id="alice")  # 记忆与 Claude 的文件一起删
```

- 没有在真实的 Claude 上跑过（开发环境没有 Anthropic 的 Key；DeepSeek 的 Anthropic 兼容接口不接受 `memory_20250818`）。离线测试直接驱动工具的各个命令；联网冒烟用 DeepSeek 驱动了同样的命令（以普通工具的形式）。
- 从文件里删掉一行，记作"这一条不再成立"，由抽取与对账决定后果；只有删掉整个文件才确定遗忘。
- 笔记记下之后供应商出错：改动已经作为笔记写进日志、抽取却出错时（例如余额不足），文件的改动照样成功，笔记只记一次，下次巩固时补上抽取。删除之后重建出错时，删除也照样成功，并说明重建要等下次。
- Claude 的文件不在 `Memory.export` 里（导出只有原始日志里的笔记）。

## 局限

- 进行中的会话按进程记在内存里：多个进程共用一个数据目录（Qdrant 服务端）时，一个进程可能提前交出另一个进程还在进行的会话。
- 模型回答期间的遗忘：注入的记忆在模型还在回答时被遗忘了，记忆代理照样返回回答，但不存下（用户的话照样存下）。三个集成同样是"召回 → 模型 → 写回"，没有这项检查，这样的回答在那里仍可能写回（没有复现过）。
- 三个集成都只注入文字证据；控制器想看原图（`see`）时，原图不会附进模型调用（与记忆代理相同）。
- 都是同步的 CueMem 在线程里跑：异步框架里用 `AsyncMemory` 或各集成的 async 版本。
- 供应商出错与过期：各集成自己的记忆工具不像 MCP 那样把供应商的出错（限流、余额不足、超时）换成说明，出来的是供应商客户端的异常。带 `expiration_date` 的记忆只在 SDK 的 `search`、`get_all` 与画像里排除，各集成在删除之前仍看得到。
- 每次自动召回与 recall 工具调用都在数据目录的 `runs/traces/*.jsonl` 里记一条 `retrieve` trace（提问、检索模式与取回的 ID）；遗忘时，列着被删记忆的 trace 连同提问的查询向量缓存一起删掉（提问里有 U+2028、U+2029、U+0085 时也一样），其中记着的子查询、追加查询与 Agent 自己检索的查询向量也删，`delete_all` 删掉这个空间的全部 trace。遗忘时重写不了的 trace 文件（磁盘满）整个删掉，记一条警告，遗忘照常成功。在那之前 trace 一直留着，除非设了 `CUEMEM_TRACES_RETENTION_DAYS`：`cuemem serve` 与 `cuemem proxy` 每天按它删一遍，只用 SDK 的应用可以调 `memory.system.prune_traces(days)`。
