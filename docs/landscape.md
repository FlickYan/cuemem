# 开源同类项目：对比与借鉴（N12）

[English](en/landscape.md)

> 两个用途：说清楚 CueMem 和 GitHub 上知名的同类项目（Mem0、Letta、Graphiti / Zep、Cognee、Supermemory……）比起来强在哪、缺什么；记录 N12 照着它们补了什么、刻意没照搬什么。论文层面的定位见[相关工作](related-work.md)，这里只看工程化的项目。
>
> 核对日期 2026-10-14。星数与许可证来自 GitHub API；接口细节来自各项目的 README 与源码。开发环境访问不到的官方文档站（docs.mem0.ai、docs.letta.com、help.getzep.com、supermemory.ai），内容来自搜索摘要，表里标了"（摘要）"；没能核实的写"未核实"。各家自己报告的分数用的回答模型、判分模型和提示词都不同，**不能**与本项目的数字直接比较（§4）。

## 1. 结论

- **研究部分在同类项目里少见**：把"要不要想起、想多深"做成可训练、可单独评测的控制器；双时间轴、失效而不删除；回答时取回原图再看一眼；测试集只用一次、方法事先冻结的评测，外部基线（Mem0）在同样的模型与判分下对比。这些在下面的项目里大多没有，或者没有公开的消融。
- **N12 之前缺的是"能拿来用"的那一层**。用过 Mem0 的人第一反应是 `pip install` 之后三行代码，CueMem 当时做不到：
  - 配置文件放在仓库里、数据写在仓库目录下，pip 安装后跑不起来；
  - 模型只有 DeepSeek 与通义（向量还是通义的原生接口），没有 OpenAI、没有本地模型；
  - 只能通过命令行、HTTP 接口和 MCP 用，没有 Python SDK，也没有"改个 base_url 就接上"的方式；
  - 不能导出、导入，遗忘没有预演，抽取记什么不能按场景调整，没有容器镜像。
- **N12 照着主流项目把这一层补上了**（§5），同时保住 CueMem 自己的语义：事实不就地改写、检索先过控制器、遗忘走完整的级联。
- 刻意没有照搬的：让 LLM 自己改记忆作为默认（Letta）、只增不改（Mem0 v3）、图数据库与二十几种向量库后端（§6）。

## 2. 同类项目一览

| 项目 | 星数 | 许可 | 记忆的形态 | 谁决定检索 | 知识更新 |
|---|---|---|---|---|---|
| [mem0ai/mem0](https://github.com/mem0ai/mem0) | 67.2k | Apache-2.0 | 抽取出的事实 + 实体链接（图只在付费平台上） | 应用调用 `search`；OpenAI 兼容的包装每轮自动检索 | v3（2026-04）只增不改，不再 UPDATE / DELETE |
| [letta-ai/letta](https://github.com/letta-ai/letta)（+ letta-code 3.6k） | 25.1k | Apache-2.0 | 常驻上下文的记忆块、归档记忆、对话记忆；Letta Code 用 git 管理记忆目录 | LLM 自己调工具 | LLM 改写记忆块；后台的 sleep-time 智能体整理 |
| [getzep/graphiti](https://github.com/getzep/graphiti)（zep 5.0k） | 31.7k | Apache-2.0 | 时序知识图谱：情景、实体、带有效期的事实边、社区 | 应用调用 `search`；MCP 工具 | 双时间轴，旧边失效而不删除 |
| [topoteretes/cognee](https://github.com/topoteretes/cognee) | 32.3k | Apache-2.0 | 知识图谱 + 向量，可配本体 | 应用调用 `recall`，规则路由选检索策略 | 矛盾检测开关（默认关） |
| [supermemoryai/supermemory](https://github.com/supermemoryai/supermemory) | 31.2k | MIT | 事实与文档，记忆之间有 updates / extends / derives 关系，静态与动态画像 | 应用、中间件、MCP，或改 base URL 的代理 | 新的取代旧的（`isLatest`），历史保留；临时事实自动过期 |
| [MemTensor/MemOS](https://github.com/MemTensor/MemOS) | 11.9k | Apache-2.0 | MemCube：明文、KV cache、LoRA、偏好记忆 | 应用调用 `search`（fast / fine 两档） | 自然语言反馈纠正 |
| [NevaMind-AI/memU](https://github.com/NevaMind-AI/memU) | 14.5k | 见仓库 | Markdown 技能"wiki" | 宿主 Agent 的指令文件 | 宿主 Agent 改写 |
| [Mirix-AI/MIRIX](https://github.com/Mirix-AI/MIRIX) | 3.5k | Apache-2.0 | 六类记忆，各由一个智能体管理（基于 Letta） | 应用调用 `retrieve_with_conversation` | 各记忆智能体改写 |
| [memodb-io/memobase](https://github.com/memodb-io/memobase) | 2.9k | Apache-2.0 | 结构化用户画像（主题 / 子主题）+ 事件时间线 | 总是注入（`context()`） | 画像槽位覆盖 |
| [langchain-ai/langmem](https://github.com/langchain-ai/langmem) | 1.7k | MIT | LangGraph store 里的自由文本或 Pydantic 记忆 | LLM 调工具，或后台管理器 | 管理器可以 insert / update / delete |
| [agiresearch/A-mem](https://github.com/agiresearch/A-mem) | 1.2k | MIT | Zettelkasten 笔记，新笔记会"演化"旧笔记 | 应用调用 `search` | LLM 改写相邻笔记 |
| [redis/agent-memory-server](https://github.com/redis/agent-memory-server) | 322 | 见仓库 | 会话工作记忆 → 后台提升为长期记忆 | LLM 调工具或注入提示 | 去重；`forget(policy, dry_run)` |
| **CueMem** | — | MIT | 不可变原始日志 + 派生的事实（双时间轴）、实体、摘要、画像、多粒度索引、原图 | **可训练的控制器**（L0 / L1 / L2，是否看原图），模型可再调工具补查 | 失效而不删除，分清"现实变了"与"说错了"（LLM 确认） |

2025–2026 年的几个变化值得注意：Mem0 v3 改成只增不改（GitHub [issue #4956](https://github.com/mem0ai/mem0/issues/4956) 反映矛盾的事实会堆积、检索不看新近），图记忆移到付费平台；Letta 的 Python 服务端移到 archive 分支，主力换成 TypeScript 的 Letta Code，Python 客户端标为弃用，Agent File 从 Letta Code 里去掉；Zep 社区版弃用，Graphiti 是开源内核；Cognee 新增 `remember / recall / improve / forget` 接口和可移植格式 COGX；Redis Agent Memory Server 的开源版移到 `V0/`，自称研究原型。

## 3. 能力对照

"未见"指在核对的源码与文档里没有找到，不代表一定没有。

| 能力 | CueMem（N12） | Mem0 | Letta | Graphiti / Zep | Cognee | Supermemory |
|---|---|---|---|---|---|---|
| Python SDK | `Memory` / `AsyncMemory`，方法名照 Mem0 | `Memory` / `AsyncMemory` | 弃用（主力是 TS） | 全异步的 `Graphiti` | `remember` / `recall` 等 | `supermemory` 客户端 |
| 零代码接入 | OpenAI 兼容的 HTTP 代理（`/v1/chat/completions`，流式） | 客户端包装（`Mem0().chat.completions.create`） | — | — | — | Memory Router（改 base URL；文档已重定向，现状未核实） |
| 每轮都检索吗 | 否：控制器先判断，L0 不检索 | 包装里每轮检索 | LLM 决定 | 应用决定 | 规则路由 | 每轮 |
| 知识更新 | 失效而不删除，版本链 `history()` | v3 只增不改；`history()` | LLM 改写 | 失效而不删除 | 矛盾检测（默认关） | `isLatest` 版本链 |
| 两条时间轴 | 事件时间与记录时间分开回放：`valid_at`（那时现实中成立的）、`known_at`（系统那时相信的） | 记录时间；过期时间 `expiration_date` | 未见 | 有：边上记 `valid_at` / `invalid_at` 与 `created_at` / `expired_at` | 未见 | 未见 |
| 图片 | 存原图、上下文感知描述、以图搜图、回答时再看原图 | 视觉模型转成文字（`enable_vision`） | 未见 | 不支持（文本与 JSON） | 视觉或转写模型（细节未核实） | OCR |
| 遗忘 | 删日志 + 重放，沿引用扩散，找没有引用的转述；**可预演** | `delete` / `delete_all`；过期时间 | 编辑记忆 | `remove_episode` | `forget` | `forget_matching(dry_run)` |
| 导出 / 导入 | `export` / `load`（原始日志 + 原图，派生记忆重建） | 只在付费平台 | Agent File（已从 Letta Code 去掉） | 未见 | COGX，可导入 Mem0 / Zep / Letta | 未见 |
| 自定义抽取 | `consolidation.instructions` | `custom_instructions` | 记忆块的标签与上限 | Pydantic 实体 / 边类型、`custom_extraction_instructions` | 自定义数据模型、本体 | 未见 |
| 本地模型、不要 Key | 预设 `ollama` | Ollama、LM Studio、vLLM | 未核实 | Ollama 等兼容接口 | Ollama、llama.cpp，本地抽取器 | 本地模式 |
| 部署 | Docker 镜像、compose（+ Qdrant 服务端）、MCP、HTTP | docker compose、托管平台、托管 MCP | `letta server` | REST、MCP 镜像 | Docker、云服务 | 本地模式、Docker、托管 MCP |
| 存储后端 | SQLite + Qdrant（本地或服务端） | 约 26 种向量库 | Postgres 等 | Neo4j、FalkorDB、Neptune | LanceDB、pgvector、Kuzu、Neo4j…… | 托管 / 本地 |
| 公开评测 | 测试集只用一次、方法事先冻结；三个基准、三种回答模型、消融、人工抽检、第二判分 | `memory-benchmarks` 仓库（平台版的模型未公开） | 博客 | 论文 | 自有评测 | 自有评测 |
| 其他语言 / 框架集成 | 只有 Python（与 MCP、HTTP、记忆代理）；OpenAI Agents SDK、LangGraph / LangChain、Claude 的 memory tool（[框架集成](integrations.md)） | TS、十几个框架 | TS、Vercel AI SDK | 十几个框架 | TS、Rust、Claude Code 插件 | TS、Vercel AI SDK 中间件、多个框架 |

## 4. 公开的分数（不能直接比较）

下面是各家**自己报告**的数字。回答模型从 gpt-4o-mini 到 GPT-5、Gemini-3 不等，判分模型有 gpt-4o-mini、GPT-4o、GPT-4.1，也有不公开的；Mem0 的 LoCoMo 判分提示要求"宽松给分"；LoCoMo 有的去掉第 5 类（对抗题）有的不去；检索预算从 1.6K 到 7K tokens 不等；LongMemEval 在 2025 年 9 月发布过清洗版。一两个点的差距在这些差异面前没有意义。

**LoCoMo**（LLM 判分的准确率，多数只算第 1–4 类）

| 系统 | 分数 | 回答 / 判分模型 | 来源 | 备注 |
|---|---|---|---|---|
| Mem0（论文） | 66.9；图变体 68.4 | gpt-4o-mini / gpt-4o-mini | arXiv 2504.19413 | Zep、LangMem、OpenAI memory 由 Mem0 代跑：66.0、58.1、52.9 |
| Zep（更正后） | 75.1 | 未写明 | getzep/zep-papers | 原先报告约 84%，第 5 类计算有误、已撤回；Mem0 重跑得 58.4，双方有争议 |
| Letta（文件系统智能体） | 74.0 | gpt-4o-mini | Letta 博客（摘要） | |
| Memobase v0.0.37 | 75.8 | Mem0 的评测代码 | memobase 仓库 | |
| MemOS | 75.8（2025-11） | gpt-4o-mini | MemOS 仓库 | 用自己的评测框架报 88.8 |
| MIRIX | 85.4 | gpt-4.1-mini / GPT-4.1 | arXiv 2507.07957（摘要） | |
| Hindsight | 89.6 / 85.7 / 83.2 | Gemini-3 / OSS-120B / OSS-20B | vectorize-io/hindsight-benchmarks | 说明里称有独立复现 |
| Mem0 平台 v3 | 92.5 | 未公开 | mem0ai/memory-benchmarks | 托管平台、检索 top-200，开源版没有这些优化 |
| **CueMem** L1 / L2 / 控制器 | 79.7 / 84.5 / 82.2 | deepseek-flash / qwen3.8-max | [技术报告](report.md) | 后 7 段对话 1,155 题，测试集只用一次 |

**LongMemEval_S**（500 题；CueMem 是事先划出的 400 题测试集）

| 系统 | 分数 | 回答模型 | 来源 | 备注 |
|---|---|---|---|---|
| 全上下文（原论文） | 60.6 | GPT-4o | arXiv 2410.10813（摘要） | 待对照 PDF |
| Zep | 71.2 / 63.8 | gpt-4o / gpt-4o-mini | arXiv 2501.13956 | 同一篇里全上下文 60.2 / 55.4 |
| Supermemory | 81.6 / 84.6 / 85.2 | gpt-4o / GPT-5 / Gemini-3 | supermemory.ai（摘要） | 另一篇"约 99%"的文章自称是戏仿，不计 |
| Mastra Observational Memory | 84.2 / 94.9 | gpt-4o / gpt-5-mini | mastra.ai（摘要） | |
| Hindsight | 91.4 / 89.0 / 83.6 | Gemini-3 / OSS-120B / OSS-20B | hindsight-benchmarks | |
| Mem0 开源版 | 91.0 | 抽取、回答、判分都用 GPT-5 | mem0ai/memory-benchmarks | 平台版 94.4，模型未公开 |
| MemOS | 77.8（2025-11） | gpt-4o-mini | MemOS 仓库 | 自有评测框架 89.2 |
| **CueMem** L1 / L2 / 控制器 | 90.5 / 93.2 / 90.5 | deepseek-flash | [技术报告](report.md) | 全上下文基线 73.0 |

**唯一可以直接比较的一组**是 N11 在同样的抽取模型、向量、回答与判分下跑的开源 Mem0 2.2（D-055）：LoCoMo 200 题 80.5% 对 CueMem L1 81.0%（持平），LongMemEval 100 题 83.0% 对 92.0%（p = 0.049），Mem0 的写入成本是 2.3–2.9 倍，丢分在时间推理、知识更新与助手说过的细节。这一组测的 mem0ai 2.2.1 就是只增不改的 v3 算法：PyPI 上的包版本号是 2.x，源码里的抽取提示写的是 "V3 Additive Extraction Prompt (ADD-only)"，`add` 不再让 LLM 决定 UPDATE / DELETE。

## 5. N12 借鉴了什么

| 补上的能力 | 照着谁 | CueMem 的做法与不同 | 位置 | 决策 |
|---|---|---|---|---|
| 配置随包发布、数据目录、`pip install` 后能用 | 各家的 Python 包 | 源码仓库里行为不变；安装后数据在 `~/.cuemem`（`CUEMEM_HOME`） | `config.py` | D-060 |
| 换模型供应商：OpenAI、Ollama（本地）、任意 OpenAI 兼容接口 | Mem0、Graphiti、Cognee 的"自带模型" | 预设叠加在默认配置上；向量走 OpenAI 兼容的 `/embeddings`；推理模型的参数差异在模型目录里声明；默认配置的请求逐字不变 | `configs/presets/`、`embed/openai_compat.py` | D-060 |
| Python SDK：`add / search / get_all / get / history / delete / delete_all`，`AsyncMemory`，`from_config` | Mem0 | 同样的方法名与 `{"results": [...]}`；兼容 v3 的 `filters=`、`top_k=`；`add(wait=True)` 返回 ADD / UPDATE 事件，旧版本留在版本链里；`search` 先过控制器 | `sdk.py` | D-061 |
| 改 base_url 就有记忆 | Mem0 的 proxy、Supermemory 的 Memory Router、Memobase 的 OpenAI 补丁 | 每轮先过控制器（L0 不注入检索结果）；证据块放在最后一句用户消息前面、系统提示不动（前缀缓存照常命中）；流式；决定写在响应头里 | `api/proxy.py` | D-062 |
| 导出 / 导入 | Letta 的 Agent File、Cognee 的 COGX、MemOS 的 dump / load | 只搬原始日志与原图，派生记忆在目标机器上重建；遗忘留下的序号空缺照原样保留 | `portable.py` | D-063 |
| 遗忘的预演 | Supermemory、Redis Agent Memory Server 的 `dry_run` | 预演同样沿引用扩散、找没有引用的转述，什么都不删 | `memory.py` | D-063 |
| 按场景调整抽取 | Mem0 的 `custom_instructions`、Graphiti 的 `custom_extraction_instructions` | 追加在抽取提示最后；改了会重建派生记忆（巩固指纹变了） | `write/extract.py` | D-064 |
| 容器镜像与 compose | Mem0、Graphiti、Cognee 的 Docker 部署 | 一个进程同时提供界面、HTTP 接口与记忆代理；compose 带 Qdrant 服务端 | `Dockerfile`、`docker-compose.yml` | D-060 |

## 6. 刻意没有照搬的

- **让 LLM 自己决定检索、自己改记忆作为默认**（Letta、LangMem 的工具式记忆）：这正是 CueMem 研究的对照组（B3 ReAct）。N9 的结论是它在文本上强、在图片上弱（MemLens 46.7%），成本也更高（LoCoMo 每千题 $3.15，L1 是 $1.44）。保留为 `mode="agent"` 与 Agent 的 `memory_search` 逃生通道，不作默认。
- **只增不改**（Mem0 v3）：省掉对账的调用，但矛盾的事实会堆积，"现在住哪"要回答模型临场判断。CueMem 的 H4 实验里有效期本身在知识更新题上没有显示出收益（各变体都是 96.5%，见技术报告），但版本链让 `history()`、按时间回放、遗忘后旧事实自动恢复成为可能，这些是产品能力，不只是分数。
- **图数据库、二十几种向量库**：单用户的记忆量级下，嵌入式 SQLite + Qdrant 足够（N11 的延迟基准：服务端模式 4.6 万个检索单元时读路径本机部分 91 ms）；多实例部署要换掉 SQLite 是已知局限（D-056），这比多接几种向量库更要紧。
- **托管平台的功能**（webhook、分类、平台版检索优化）：不在开源项目的范围里。

## 7. 还差什么

按"用户最先会碰到"的顺序：

1. **更多框架集成**：已经有 OpenAI Agents SDK、LangGraph / LangChain 与 Claude 的 memory tool（[框架集成](integrations.md)）；还缺 TypeScript 与 Vercel AI SDK 中间件、LangGraph 的 `BaseStore`、CrewAI、AutoGen 等。Claude 的 memory tool 没有在真实的 Claude 上跑过。
2. **多租户**：只有一个共享的访问令牌；记忆空间按 user_id 隔离，但没有按用户鉴权。多实例部署需要把 SQLite 换成 Postgres。
3. **更多外部基线**：在同一协议下跑 Graphiti、LangMem、Cognee、A-MEM、Letta（需要 API Key 与预算）。
4. **预设的验证**：OpenAI 与 Ollama 预设只做了离线测试，没有在真实模型上跑评测；巩固的相似度阈值是按通义向量调的，换向量模型后可能要重调。
5. **记忆的元数据与过期**：Mem0 的 `metadata` 过滤与 `expiration_date`、Supermemory 的临时事实自动过期。
6. **命令行的英文界面**：CLI 的提示目前是中文；SDK、HTTP、MCP、代理与界面是英文。
