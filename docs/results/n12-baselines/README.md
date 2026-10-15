# N12 结果：同一协议下的外部基线（LoCoMo；Mem0、Graphiti、LangMem、Cognee、A-MEM）

N11 只有一个外部基线：开源的 Mem0 2.2（D-055），用的还是 CueMem 的原协议（抽取用通义、回答用 DeepSeek）。各家自己报告的分数，回答模型、判分模型与题目都不一样，不能直接比（[landscape.md §4](../../landscape.md)）。N12 再接入四个知名的开源记忆系统——Graphiti（Zep 的开源核心）、LangMem、Cognee、A-MEM——连同 Mem0 一起，在**同一个协议**下与 CueMem 比（D-065）：同样的抽取模型、向量、回答模型与判分，各自用自己的写入与检索流程。比较只在 LoCoMo test200r（7 段对话、200 题）上做，每个配置跑一次（§7）。

协议有两个，每个都只用一家供应商的模型：

- **pds**（只用 DeepSeek）：所有角色都是 deepseek-flash，向量在本机算。
- **pq**（只用通义）：DeepSeek 的余额在工作中途用完（2026-10-15 只剩 CNY 0.08）时加的。之后充值，pds 也跑完了。

原始数字：[pds-locomo.json](pds-locomo.json)、[pq-locomo.json](pq-locomo.json)（`scripts/n12_baselines.py` 从各次运行汇总，不调用 API）。A-MEM（MA）的完整运行还没结束，表里记作 MA-PENDING。

## 结论

1. **pds（写入一侧没有一方针对这个模型调过）**：CueMem L2 87.0%，外部系统里最高的是 Mem0 81.0%。
   - L2 对已出结果的四个外部系统都显著更好：Mem0 +17 / −5（p = 0.0169）、LangMem +25 / −7（p = 0.0021）、Cognee +27 / −8（p = 0.0019）、Graphiti +62 / −6（p < 0.001）。
   - 但 L2 每题的输入是它们的两倍多到将近四倍（12,255 对 3,358–5,683 tokens，含规划与充分性检查的调用），回答成本每千题 $4.11 对 $1.09–1.79。
   - 输入量与它们相近的 L1（82.5%，4,333 tokens）与 Mem0（81.0%）、LangMem（78.0%）、Cognee（77.5%）打平（p = 0.6900、0.1996、0.0987），只显著高于 Graphiti（59.0%）。
2. **pq**：L2 85.0% 显著高于 Mem0（77.5%，p = 0.0135）与 Cognee（74.0%，p = 0.0005），每题输入同样多得多（14,022 对 3,611–6,689 tokens）；L1 81.5% 显著高于 Cognee（p = 0.0167），与 Mem0 打平（p = 0.2005）。
3. **Graphiti 与 LangMem 对模型很敏感**：换成 qwen3.8-flash 后，Graphiti 从 59.0% 降到 41.5%，LangMem 从 78.0% 降到 30.0%。问题主要出在写入（§4）：
   - Graphiti 用自己的提示让 qwen3.8-flash 抽实体，抽得少，对不上实体的事实被丢掉；
   - LangMem 把新信息并进少数几条长记忆，202 个会话里有 23 个一条也没写进去。

   读的一侧也有影响：pq 下 Graphiti 向量路过 0.6 阈值的候选少得多（阈值没有按通义向量重调），LangMem 的长记忆超过向量器的 6000 字符上限，尾部检索不到；这两项的影响没有单独量过。

   两者都没有为通义重新调提示或阈值。所以 pq 上这两个数字说明的是"默认配置换了模型会怎样"；拿 CueMem 与它们比，看 pds（pds 下 Graphiti 没有重排模型，也不是 Zep 评测的原配方，见 §7）。
4. **写入成本**（名义成本，7 段对话）：
   - CueMem pds $0.53、pq $0.24；
   - LangMem 与 CueMem 相近（$0.56、$0.23；pq 上更低，但有 23 个会话没写进任何东西）；
   - Mem0 $1.38、$0.71；Cognee $2.85、$1.47；
   - Graphiti $9.99、$4.51：每条发言一个 episode，pds 下 14,240 次 LLM 调用，CueMem 是 438 次；
   - A-MEM：MA-PENDING。

   回答成本上 L1 与 Mem0、LangMem 相近（pds 每千题 $1.38、$1.37、$1.48）。
5. **主场优势**：CueMem 的抽取提示与阈值是用 qwen3.8-flash 与通义向量开发的（N3–N7），pq 是它的主场；pds 上写入一侧没有一方调过，但 deepseek-flash 是 CueMem 主实验的回答模型（§6）。L2 在 pds 上同样显著领先；L1 与 Mem0 在两个协议上都打平。
6. **A-MEM**：MA-PENDING。

## 1. 比较了什么、怎么比

### 1.1 方法

| 代号 | 方法 | 版本 / 说明 |
|---|---|---|
| B1 | 全上下文 | 整段对话交给回答模型 |
| B2 | 朴素检索 | 按轮做向量检索，取前 10 轮 |
| B4-nobm25 | CueMem L1 | |
| B5 | CueMem L2 | |
| B6 | CueMem 控制器 | 仓库里默认的控制器模型，没有按 pds / pq 重新训练 |
| M0 | Mem0 | mem0ai 2.2.1 开源版：只增不改的 V3 算法（源码里的抽取提示标着 "V3 Additive Extraction Prompt (ADD-only with memory linking)"） |
| MG | Graphiti | graphiti-core 0.30.2，嵌入式图库 Kuzu |
| ML | LangMem | langmem 0.0.30 |
| MC | Cognee | cognee 1.6.4 |
| MA | A-MEM | agiresearch/A-mem@ceffb86 |

[landscape.md](../../landscape.md) 的计划里还有 Letta，这次没有接入：它的记忆由 Agent 自己调工具管理，更接近 B3（ReAct），不是"换一套写入与检索"（D-055、D-065）。

### 1.2 协议

| 角色 | pds | pq |
|---|---|---|
| 抽取（extractor；外部系统内部的 LLM 调用也都交给它） | deepseek-flash | qwen3.8-flash |
| 规划（planner，L2）、回答（reader）、vlm、captioner、agent | deepseek-flash | qwen3.8-flash |
| 判分（judge） | deepseek-flash | qwen3.8-max |
| 向量 | BAAI/bge-small-en-v1.5（本机 fastembed，384 维，不加检索指令） | qwen3.7-text-embedding（1024 维，查询与文档分开向量化） |
| 重排（CueMem L2、Graphiti） | 无 | qwen3-rerank |
| 备用模型（巩固与评测网关的调用失败时换用） | 无 | qwen3.7-plus |
| 预设 | `CUEMEM_PRESET=deepseek` | `CUEMEM_PRESET=qwen`（新增的 `src/cuemem/configs/presets/qwen.yaml`） |
| 写入目录 | `data/eval-spaces` | `data/eval-spaces-qwen`（`CUEMEM_PATHS__EVAL_SPACES`） |

- **pds 的判分就是回答模型**，可能偏向自己的回答；pq 的判分与回答是同一家族。N11 的第二判分实验里，deepseek-flash 与 qwen3.8-max 判分的一致率是 94.7–97.9%。
- 作为对照，N9 / N11 的原协议是：抽取 qwen3.8-flash、通义向量、回答 deepseek-flash、判分 qwen3.8-max、备用模型 deepseek-flash。这里没有用原协议跑外部系统。

### 1.3 公平规则

- **同一批题**：LoCoMo test200r，200 道非对抗题，按题型分层（单跳 111、多跳 36、时间 40、开放域 13），来自 7 段对话（conv-42、43、44、47、48、49、50；4,431 条发言、202 个会话）。只比较所有方法都有结果的题；两个协议都没有丢题。
- **同样的模型**：外部系统各自带着 OpenAI 兼容的客户端，`base_url` 指向评测网关：聊天一律交给 extractor 角色的模型，向量一律交给同一个向量器（§1.4）。
- **各用自己的流程**：写入与检索照各家作者公开的评测代码（有的话），只换"记忆从哪来"。偏离上游的地方逐条写在各模块的说明里（§2）。
- **同样的回答与判分**：检索到的证据交给同一个回答模型、同一套回答提示（与 B2、M0 的格式相同），每条证据都带会话日期；判分相同。
- **不为任何一方调参**：外部系统的提示与阈值用作者的默认值。检索条数照作者公开评测的设置（Zep 的 20 条事实 + 20 个实体、Cognee BEAM 报告的 20 / 20、A-MEM 论文的 k = 10）；Mem0 与 LangMem 放宽到与 L1 的证据量相当（Mem0 每位说话人 30 条，默认 20；LangMem 60 条，检索工具默认 10）。为了跑得起来改的其他参数（如 Graphiti 的 BFS 深度、A-MEM 的回复上限）都在 §2 逐条列出。CueMem 也没有按 DeepSeek 改提示与阈值（§5.2）。
- **统计**：每个配置跑一次；同一批题上做配对的精确符号检验；95% 区间是 Wilson 区间。
- **成本**：写入是一次性的，单独列；回答成本只算回答（检索时的模型调用也算在内），不含写入与判分。名义成本按高峰时段、不计任何缓存的标价（D-010）。两个协议的单价不同，成本只在同一协议内比。

### 1.4 评测网关

[`src/cuemem/evals/gateway.py`](../../../src/cuemem/evals/gateway.py)：把 CueMem 的 LLM 客户端与向量器包成 OpenAI 兼容的聊天与向量接口。每次调用都走 CueMem 的客户端（同一本账、同一个响应缓存、同样的高峰暂停），按"桶"分别统计每份对话历史的写入与检索用量。为了让外部系统得到它们假定的 OpenAI 语义：

- `response_format` 是 json_schema 时改成 json_object，并把 schema 写进系统提示（DeepSeek 与通义都不保证支持 json_schema）；
- 带工具的请求没写 `parallel_tool_calls` 时按 OpenAI 的默认发出 true（通义不设 true 时一次只回一个工具调用）；
- 检索阶段的桶（`<系统>.search.<空间>`）里的向量请求按查询向量化，与 CueMem 自己的检索相同；
- 上游限流时回 429 并带上 Retry-After，按桶计数；
- JSON 模式下解析不了的回复不从缓存重放：同样的请求再来时重新调用。

## 2. 各外部系统的接法

### Mem0（M0）

与 N11 相同（D-055），见 [`src/cuemem/evals/mem0_baseline.py`](../../../src/cuemem/evals/mem0_baseline.py)。

- **写入**：每个会话调用一次 `Memory.add`。两人对话照 Mem0 论文的评测代码，每位说话人一个 user_id、各写一遍，所以抽取调用是 404 次（202 个会话 × 2）。
- **检索**：`search(top_k=30)`，两位说话人各取 30 条；装上 spaCy 与 fastembed，让混合检索（语义 + BM25 + 实体）完整生效。
- **与上游的差异**：开源版不支持 `timestamp`，抽取提示里的"观察日期"换成会话日期；通义的 JSON 模式偶尔只回记忆列表，适配层补上外层（N11）。

### Graphiti（MG）

graphiti-core 0.30.2。照 Zep 论文（arXiv 2501.13956）与 getzep/zep-papers 的评测代码，见 [`src/cuemem/evals/baselines/graphiti.py`](../../../src/cuemem/evals/baselines/graphiti.py)。

- **写入**：每条发言一个 episode（"说话人: 内容"），整段对话写进同一个图（与 Zep 的 LoCoMo 评测相同）。episode 只能串行写入，每条都有实体抽取、消歧、事实抽取、去重与失效判断等多次 LLM 调用：pds 下每段对话 1,722–2,208 次。
- **检索**：与 Zep 的评测相同，取 20 条事实与 20 个实体。
  - 有重排模型时（pq），事实走 Graphiti 的 cross_encoder 配方：BM25 + 向量 + BFS 召回，RRF 融合后的前 40 条交给 qwen3-rerank，取前 20。
  - 没有重排模型时（pds），退回 Graphiti 基础 `search()` 的默认做法：BM25 + 向量的 RRF，不带 BFS。
  - 实体都用 RRF。事实附上有效期。
- **与上游的差异**：
  - 重排：Graphiti 自带的两个 cross_encoder 都用不了（一个靠 logprobs 与 OpenAI 分词器的 logit_bias，另一个要装 torch、下载 2GB 的模型），换成 CueMem 的重排模型，对原问题打分。
  - BFS 深度 1（Graphiti 默认 3）：深度 3 时 Kuzu 的变长路径查询超过 12GB 内存。只影响 pq（pds 不走 BFS）。
  - 图库用 0.30.2 自带的 Kuzu（Graphiti 默认 Neo4j，Zep 的评测用 Zep Cloud），补了几处 Kuzu 后端的缺口（全文索引、库名、内存上限）。
  - BM25 查询去掉标点（Kuzu 不去问号，带问号的问题常常一条也查不到）。
  - 同一会话里第 i 条发言的时间加 i 秒，让"前 10 条消息"的顺序确定。

### LangMem（ML）

langmem 0.0.30（相关的 langchain、langgraph、trustcall 版本都钉住）。照 LangMem 文档里"后台形成记忆"的做法，见 [`src/cuemem/evals/baselines/langmem.py`](../../../src/cuemem/evals/baselines/langmem.py)。

- **写入**：`create_memory_store_manager`，会话按时间顺序、每个会话处理一次（整段会话，每句前面写上说话人与时间）。管理器先检索最相关的 5 条已有记忆，再让模型用工具调用（trustcall）新增或修改记忆。一段对话一个命名空间，不分说话人。
- **检索**：`store.search(limit=60)`，再按 L1 的 4000 tokens 预算装入；每条记忆标上最后修改它的会话日期。
- **与上游的差异**：
  - 不照搬 Mem0 论文评测代码里的"热路径"（每位说话人一个 ReAct Agent，逐句喂）：调用量与上下文随对话长度平方增长，Agent 的回复是改写过的散文，没法交给同一个回答模型。
  - 检索条数 60（LangMem 检索工具默认 10），与 M0 两人各 30 条相当。
  - 记忆 ID 用按（对话，会话）定种子的随机数：原来的 uuid4 每次都不同，响应缓存全部落空。
  - 其余都用默认：记忆提示、Memory 结构、enable_deletes=False、query_limit=5。

### Cognee（MC）

cognee 1.6.4。照 Cognee 自己在长对话基准 BEAM 上的做法（随包发布的 `cognee/eval_framework/beam`），见 [`src/cuemem/evals/baselines/cognee.py`](../../../src/cuemem/evals/baselines/cognee.py)。

- **写入**：一段对话一个数据集；每个会话 `add` 一个 JSON 列表文件，再 `cognify`（一轮一块，对每块做图谱抽取与摘要，再向量化）。LoCoMo 两位说话人的相邻两句合成一"轮"。
- **检索**：默认的 HYBRID_COMPLETION，参数取 BEAM 100K 报告的固定配置（chunks_top_k=20、entities_top_k=20）。它交给自己回答模型的上下文——20 个块、20 个实体、5 条关系事实——**全部**交给回答模型，不截断（BEAM 报告里也是这样）。实体与事实的日期按 Cognee 自己的出处记录补上。
- **与上游的差异**：
  - 不建全局上下文索引（BEAM 的配置里建了，但不参与检索）。
  - 会话蒸馏只用于用户与助手的对话，LoCoMo 两个真人聊天不做。
  - 超过 4096 tokens 的一轮直接拆开，不先用 LLM 压缩。
  - pds 的 bge-small 只收 512 tokens，Cognee 自动把块的上限从 4096 降到 510。
  - 每段对话一个子进程（Cognee 的配置是进程级的）。

### A-MEM（MA）

agiresearch/A-mem@ceffb86（PyPI 上的 agentic-memory 0.1.2 不是 A-MEM）。照 A-MEM 论文（arXiv 2502.12110）在 LoCoMo 上的评测代码（作者的复现仓库 WujiangXu/AgenticMemory），见 [`src/cuemem/evals/baselines/amem.py`](../../../src/cuemem/evals/baselines/amem.py)。

- **写入**：每句话一条笔记。先由 LLM 生成关键词、上下文与标签，再做记忆演化（取最相近的 5 条笔记，由 LLM 决定是否建立链接、改写这些邻居）；每演化 100 次整理一次。每条笔记两次 LLM 调用、只能串行，pq 下约 15 秒一句。
- **检索**：LLM 先把问题变成几个关键词，向量检索前 k = 10 条笔记（论文的主要设置），每条带出它链接的笔记。
- **与上游的差异**：
  - 每段对话一个持久化的 Chroma 集合（原版所有记忆系统共用一个临时集合，每次构造都清空）。
  - 演化时按邻居在记忆库里的真实序号改写（原版按检索名次去改记忆库的第 0–4 条）。
  - 整理时照论文代码的格式重新向量化（写入协议 v4）。
  - 回复上限 4096 tokens（原版 1000，截断的 JSON 让演化被放弃）。
  - k = 40（作者给 GPT-4o-mini / GPT-4o 一档模型的设置）留作敏感性运行（`CUEMEM_AMEM_K=40`）。
- **结果**：MA-PENDING。

## 3. 结果（LoCoMo test200r）

### 3.1 准确率与回答成本

| 方法 | pds 准确率 | pds 95% 区间 | pds 回答 $ / 千题 | pq 准确率 | pq 95% 区间 | pq 回答 $ / 千题 | 每题输入 tokens（检索与回答的调用，pds / pq） |
|---|---|---|---|---|---|---|---|
| B1 全上下文 | 84.5% | 78.8–88.9% | $6.887 | 80.0% | 73.9–85.0% | $3.544 | 22,706 / 23,406 |
| B2 朴素检索（向量 top-10） | 59.5% | 52.6–66.1% | $0.371 | 64.0% | 57.1–70.3% | $0.215 | 1,006 / 1,231 |
| **CueMem L1** | 82.5% | 76.6–87.1% | $1.382 | 81.5% | 75.5–86.3% | $0.790 | 4,333 / 5,034 |
| **CueMem L2** | **87.0%** | 81.6–91.0% | $4.108 | **85.0%** | 79.4–89.3% | $2.293 | 12,255 / 14,022 |
| CueMem 控制器（B6） | 83.5% | 77.7–88.0% | $2.814 | 83.5% | 77.7–88.0% | $1.424 | 8,480 / 8,788 |
| M0 Mem0 2.2.1 | 81.0% | 75.0–85.8% | $1.371 | 77.5% | 71.2–82.7% | $0.771 | 4,289 / 4,874 |
| MG Graphiti 0.30.2 | 59.0% | 52.1–65.6% | $1.092 | 41.5% | 34.9–48.4% | $0.744 | 3,358 / 3,611 |
| ML LangMem 0.0.30 | 78.0% | 71.8–83.2% | $1.481 | 30.0% | 24.1–36.7% | $0.693 | 4,623 / 4,382 |
| MC Cognee 1.6.4 | 77.5% | 71.2–82.7% | $1.787 | 74.0% | 67.5–79.6% | $1.040 | 5,683 / 6,689 |
| MA A-MEM | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING |

- 两个协议的单价不同（deepseek-flash 的输入标价是 qwen3.8-flash 的两倍），回答成本只在同一列里比。
- 证据量相近：L1、Mem0、Graphiti、LangMem 每题的输入在 3,358–5,034 tokens 之间；Cognee 不截断，略多；L2（含规划与充分性检查的调用）与全上下文多得多。
- B2 只靠向量检索，pq 上更高（64.0% 对 59.5%）多半是向量模型的差别：pds 用的是本机的 384 维小模型，B2 的发言召回（任一）pds 67.0%、pq 84.5%。

### 3.2 写入

| 方法 | pds 名义成本 | pds 实际花费 | pds LLM 调用 | pds 耗时（秒） | pq 名义成本 | pq 实际花费 | pq LLM 调用 | pq 耗时（秒） |
|---|---|---|---|---|---|---|---|---|
| B2 朴素检索（只向量化） | $0.0000 | $0.0000 | 0 | 257 | $0.0063 | $0.0063 | 0 | 116 |
| CueMem（L1 / L2 / 控制器共用） | $0.5290 | $0.2222 | 438 | 903 | $0.2376 | $0.1511 | 393 | 1,274 |
| M0 Mem0 | $1.3815 | $0.2420 | 404 | 1,888 | $0.7122 | $0.3168 | 404 | 3,538 |
| MG Graphiti | $9.9923 | $1.5084 | 14,240 | 7,265 | $4.5147 | $3.9077 | 12,092 | 30,421 |
| ML LangMem | $0.5630 | $0.2386 | 211 | 23,276 | $0.2252 | $0.0635 | 207 | 1,006 |
| MC Cognee | $2.8450 | $0.7975 | 4,547 | 10,111 | $1.4659 | $1.1542 | 4,549 | 15,233 |
| MA A-MEM | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING |

7 段对话合计。**系统之间比名义成本**；实际花费与耗时受缓存与时段影响，不能横着比：

- **名义成本**：按高峰时段、不计任何缓存的标价，相当于从头写一遍的价格。
- **实际花费**：扣掉本地响应缓存的命中、供应商的前缀缓存折扣与 DeepSeek 的错峰折扣。写入在缓存里有现成结果时几乎为零：
  - pds 的 conv-42：Graphiti、LangMem 与 Cognee 的写入大多命中之前写入留下的缓存（Graphiti 2,014 次调用里 2,012 次）；
  - pds 的 Graphiti 运行中途续跑过一次，其余六段也各有 1,096–1,176 次调用命中缓存；
  - pq 的 LangMem 有 5 段对话的写入全部命中之前写入留下的缓存；pq 的 Graphiti conv-42 是 1,688 次里 1,453 次。
- **LLM 调用**：最终那次写入的聊天调用次数（含命中缓存的）；CueMem 是写入与巩固。
- **耗时**：每段对话最终那次写入的墙钟时间之和（各段并行写入），命中缓存的部分很快。Mem0 没有记墙钟，取各次调用的延迟之和。pds 的 LangMem 运行在北京时间 17:01 开始，正是 DeepSeek 的高峰时段，按 `--peak wait` 暂停到 18 点：其余六段对话（conv-42 命中缓存）各只有 25–32 次调用，耗时却在 3,827–3,940 秒。

### 3.3 逐题对比

CueMem L1、L2 对其他方法的赢 / 输题数（同一题上 CueMem 对、对方错算赢），精确符号检验：

| 对方 | pds：L1 | pds：L2 | pq：L1 | pq：L2 |
|---|---|---|---|---|
| B1 全上下文 | +9 / −13（p = 0.5235） | +12 / −7（p = 0.3593） | +17 / −14（p = 0.7201） | +19 / −9（p = 0.0872） |
| B2 朴素检索 | +49 / −3（p < 0.001） | +59 / −4（p < 0.001） | +40 / −5（p < 0.001） | +48 / −6（p < 0.001） |
| CueMem 控制器（B6） | +6 / −8（p = 0.7905） | +7 / −0（p = 0.0156） | +5 / −9（p = 0.4240） | +6 / −3（p = 0.5078） |
| M0 Mem0 | +14 / −11（p = 0.6900） | **+17 / −5（p = 0.0169）** | +19 / −11（p = 0.2005） | **+24 / −9（p = 0.0135）** |
| MG Graphiti | **+55 / −8（p < 0.001）** | **+62 / −6（p < 0.001）** | **+85 / −5（p < 0.001）** | **+89 / −2（p < 0.001）** |
| ML LangMem | +24 / −15（p = 0.1996） | **+25 / −7（p = 0.0021）** | **+107 / −4（p < 0.001）** | **+113 / −3（p < 0.001）** |
| MC Cognee | +20 / −10（p = 0.0987） | **+27 / −8（p = 0.0019）** | **+25 / −10（p = 0.0167）** | **+30 / −8（p = 0.0005）** |
| MA A-MEM | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING |

L2 对 L1：pds +15 / −6（p = 0.0784），pq +15 / −8（p = 0.2100）。

### 3.4 分题型

| 方法 | pds 单跳（111） | 多跳（36） | 时间（40） | 开放域（13） | pq 单跳（111） | 多跳（36） | 时间（40） | 开放域（13） |
|---|---|---|---|---|---|---|---|---|
| B1 全上下文 | 97.3% | 63.9% | 77.5% | 53.8% | 95.5% | 47.2% | 75.0% | 53.8% |
| B2 朴素检索 | 68.5% | 38.9% | 62.5% | 30.8% | 76.6% | 30.6% | 75.0% | 15.4% |
| CueMem L1 | 90.1% | 66.7% | 87.5% | 46.2% | 91.9% | 55.6% | 85.0% | 53.8% |
| CueMem L2 | 93.7% | 75.0% | 90.0% | 53.8% | 94.6% | 63.9% | 82.5% | 69.2% |
| CueMem 控制器 | 90.1% | 75.0% | 85.0% | 46.2% | 92.8% | 63.9% | 80.0% | 69.2% |
| M0 Mem0 | 91.0% | 63.9% | 82.5% | 38.5% | 84.7% | 58.3% | 82.5% | 53.8% |
| MG Graphiti | 64.0% | 58.3% | 57.5% | 23.1% | 45.0% | 27.8% | 50.0% | 23.1% |
| ML LangMem | 86.5% | 63.9% | 77.5% | 46.2% | 33.3% | 27.8% | 22.5% | 30.8% |
| MC Cognee | 83.8% | 66.7% | 80.0% | 46.2% | 86.5% | 47.2% | 77.5% | 30.8% |
| MA A-MEM | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING | MA-PENDING |

- pds 上 L2 在多跳（75.0%，外部系统最高 66.7%）与时间题（90.0%，外部系统最高 82.5%）上领先明显；单跳题上 L2 与 Mem0 接近（93.7% 对 91.0%），L1 与 Mem0 相当（90.1% 对 91.0%）。
- pds 上 Graphiti 在每个题型上都是外部系统里最低的，不是某一类题的问题。
- 开放域只有 13 题，一两道题就能改变排序，不要据此下结论。

## 4. 换模型的影响：Graphiti 与 LangMem 为什么在通义下掉这么多

同一个系统、同样的代码，换的是整套模型：抽取与回答 deepseek-flash → qwen3.8-flash，向量 bge-small → 通义，判分 deepseek-flash → qwen3.8-max；Graphiti 还因为 pq 有重排模型而换了事实检索的配方（§2）。N11 在 LoCoMo 上实测 deepseek-flash 判分比 qwen3.8-max 宽松约 3.5 个点（[N11 §5.1](../n11-improvements/README.md)），所以 Mem0、Cognee、CueMem 几个点的变化分不清来自模型还是判分；Graphiti 与 LangMem 的大幅下降在写入统计里有直接证据（§4.1、§4.2）。

| 系统 | pds | pq |
|---|---|---|
| CueMem L1 / L2 | 82.5% / 87.0% | 81.5% / 85.0% |
| Mem0 | 81.0%（记忆 4,059 条） | 77.5%（2,907 条） |
| Cognee | 77.5% | 74.0% |
| LangMem | 78.0% | **30.0%** |
| Graphiti | 59.0% | **41.5%** |
| A-MEM | MA-PENDING | MA-PENDING |

### 4.1 Graphiti：抽出的实体少，事实跟着丢

| 7 段对话合计 | pds | pq |
|---|---|---|
| 实体 | 1,253 | 462 |
| 事实（边） | 3,426 | 1,774 |
| 每段对话的 "Target entity not found" 告警 | 4–15 | 118–270 |
| 发言召回（任一） | 73.5% | 54.5% |
| 每题向量路找到的事实候选（中位数） | 40（每路上限） | 8 |
| 每题向量路找到的实体候选（中位数） | 10 | 1 |

- Graphiti 先抽实体、再抽事实；事实的两端要对上已抽出的实体，对不上就丢掉这条事实，记一条 "Target entity not found" 告警。用 Graphiti 自己的提示，qwen3.8-flash 抽出的实体少：pq 下每段对话有 118–270 条这样的告警，pds 下 4–15 条。
- 结果是检索不到证据：pq 下 Graphiti 的发言召回（检索结果里至少有一条包含证据发言）是 54.5%，同协议的 B2 是 84.5%、CueMem L1 是 95.0%。
- pq 下的事实还经过 qwen3-rerank 重排，pds 下只是 RRF，所以差距不是缺了重排造成的。
- 向量路在 pq 下并没有整个失效（每题仍有中位数 8 条事实候选过 Graphiti 的 0.6 余弦阈值），所以阈值保持默认，没有调低（§5.2）。但候选比 pds 少得多，实体的向量路中位数只剩 1 个：其中多少因为实体本来就少、多少因为阈值没按通义向量重调，没有单独量过。

### 4.2 LangMem：并进少数几条长记忆，很多会话没写进去

| 7 段对话合计 | pds | pq |
|---|---|---|
| 记忆条数 | 705 | 123 |
| 一条也没写进去的会话 | 2 / 202 | 23 / 202 |
| 模型调用了 PatchDoc 却明说"不用改"的会话 | 0 | 5 |
| 抽取调用里一次回多个工具调用的 | 205 / 211 | 35 / 207 |
| 工具调用总数 | 1,043 | 245 |
| 超过向量器 6000 字符上限的记忆 | 2 | 12 |
| 会话召回（任一） | 88.0% | 49.0% |

- qwen3.8-flash 多用一个 PatchDoc 把新信息并进已有的几条长记忆（最长 14,118 个字符），很少一次新增多条。
- 写进去的少：23 个会话没写进任何东西（没调用工具、补丁改不了，或调用校验不过）；另有 5 个会话模型明说不用改（不算写入丢了，另记）。
- 长记忆尾部的内容检索不到：CueMem 的向量器只取前 6000 个字符（超过的 pq 12 条、pds 2 条；pds 的 bge-small 还只看前 512 tokens）。
- 结果是检索结果里至少命中一个证据会话的题只有 49.0%（pds 88.0%）。
- 没有改 LangMem 的提示：合并成长记忆是它在这个模型上的行为。

### 4.3 其他系统

- **Cognee**：准确率 77.5% → 74.0%（含判分的差别，见上）。两个协议下都没有失败、重写或丢弃的会话。实体 pds 1,261 个、pq 1,544 个；pds 的块受 bge-small 的 512 tokens 所限（上限 510），pq 是 4096。
- **Mem0**：准确率 81.0% → 77.5%（含判分的差别，见上），单跳题降得最多（91.0% → 84.7%）。两个协议下都没有失败的会话；pq 下记忆 2,907 条（pds 4,059 条）。
- **CueMem**：L1 pds 82.5%、pq 81.5%，L2 87.0%、85.0%。
- **A-MEM**：MA-PENDING。

## 5. 适配层加了什么、刻意没调什么

### 5.1 为了跑得完加的（不改各系统的算法）

外部系统默认假设的是 OpenAI 的模型。换成 DeepSeek 与通义后，有些失败它们会吞掉或放弃，所以适配层加了以下几项，并把触发次数写进每段对话的 `build.json`：

- **网关**：见 §1.4（json_schema 改写、并行工具调用、查询向量、429、不重放坏 JSON）。
- **重新采样**：回复解析不了时，Graphiti 会原样重发，但响应缓存只会还回同一个坏回复；回复是 JSON 却不合 schema 时，Graphiti 不重试，整条 episode 失败。这里按 Graphiti 自己的解析规则校验，不合也当作解析失败；重试时 `max_tokens` 逐次减 1，避开响应缓存，真正重新采样。A-MEM 的回复不合 schema 时同样重新采样，最多两次。
- **内容审核拒绝**：被拒的调用跳过并计数，不算失败：Mem0 与 LangMem 的会话、Graphiti 的发言（超过 max(2, 5%) 条才整份作废）、A-MEM 的调用（超过一半才算失败）。
  - pq 的 Graphiti conv-42 有 5 条发言被拒；它是在加入备用模型 qwen3.7-plus 之前写入的，其余六段 0 条。pds 没有被拒的。
- **写入丢失的记账**：
  - LangMem：没调用工具、补丁改不了、调用校验不过而一条也没写的会话算写入丢了，读不出的 PatchDoc 补丁也算。超过 max(2, 10%) 的会话丢失时整份写入失败。
    - pq 下这条检查拒掉了 7 段对话里的 4 段，所以**两个协议的完整运行都放宽到 25%**（`CUEMEM_LANGMEM_MAX_LOST_SHARE=0.25`，默认仍是 10%），丢失的会话照常计入结果。
    - pds 每段对话最多丢 1 个会话，在默认的 10% 下也能通过。
  - Cognee：失败的会话从数据集里删掉、马上重写一次，仍失败的跳过并记下（两个协议都是 0）。
  - Graphiti：重试后仍失败的发言超过 1% 时整份作废、下次重建（两个协议最终的写入里都没有失败的发言）。

### 5.2 刻意没调的

- **Graphiti 的向量相似度阈值 0.6**：pq 下向量路不是空的（中位数 8 条候选），保持默认。
- **LangMem 的提示与 Memory 结构**：都用默认值。
- **A-MEM 的 k = 10**：论文的主要设置；作者给 GPT-4o-mini / GPT-4o 一档模型的 k = 40 只作敏感性运行（结果：MA-PENDING）。
- **Cognee 的检索配置**：BEAM 报告的固定配置，不截断。
- **Mem0**：沿用 N11 的设置（每人 30 条）。
- **CueMem**：没有为 deepseek-flash 或 bge-small 改提示与阈值；控制器没有按两个协议重新训练。

## 6. 主场优势

- CueMem 的抽取提示与阈值是在 N3–N7 用 qwen3.8-flash 与通义向量开发的，**pq 是 CueMem 的主场**。
- **pds 上写入一侧没有一方调过**：没有任何系统的抽取提示是针对 deepseek-flash 或 bge-small 写的；CueMem 的巩固阈值是按通义向量调的，换到 bge-small 可能偏松或偏紧（见 [deepseek 预设](../../../src/cuemem/configs/presets/deepseek.yaml)的说明）。
- **但 pds 对 CueMem 的读路径不算完全中立**：deepseek-flash 是 N3–N11 主实验的回答模型，共用的回答提示、L1 的证据装法与控制器的准确率表都是在它上面定的。
- 外部系统的公开评测多用 OpenAI 的模型（Zep、Mem0、A-MEM 的论文用 gpt-4o-mini 一类），两个协议都不是它们的主场。
- 所以 pq 上的差距里有主场优势；pds 更接近"谁都没调过"（写入一侧）。pds 上 L2 对已出结果的四个外部系统仍显著领先（每题输入是它们的 2–4 倍），L1 与 Mem0、LangMem、Cognee 打平。

## 7. 局限

- **外部系统只跑了 LoCoMo**：LongMemEval 的写入太贵、太慢（Graphiti 每条发言一个 episode，test100r 约 4.9 万条发言，写入要几百美元、一天以上；A-MEM 约 15 秒一句）。外部系统在 LongMemEval 上只有 Mem0 的数字：N11 的原协议，以及泄漏消融里的几组。LongMemEval 的会话 ID 泄漏（D-067，见 [N12 泄漏记录](../n12-leak/README.md)）不影响 LoCoMo。
- **只有 200 题、每个配置跑一次**：N11 实测单次运行有 ±1 个点的波动、3–4% 的题会翻转。相差几个点的比较看配对检验，不看准确率差了多少。
- **判分**：pds 的判分就是回答模型，pq 的判分与回答同一家族；这次没有用另一家的模型重判。
- **LangMem 的写入丢失配额放宽到 25%**：用默认的 10%，pq 下 7 段对话有 4 段写不出来、没有结果。放宽后这些丢失算进了它的准确率。
- **Graphiti**：pds 没有重排模型，事实检索退回 BM25 + 向量的 RRF、不走 BFS；pq 走 cross-encoder 配方，但 BFS 深度 1（默认 3）。两个协议都不是 Zep 评测的原配方。pq 的 conv-42 写入时还没有备用模型，5 条被拒的发言跳过了。
- **向量器的输入上限**：所有方法共用 CueMem 的向量器，它只取前 6000 个字符，pds 的 bge-small 还只看前 512 tokens。受影响最大的是 LangMem 合并出的长记忆（超过 6000 字符的 pq 12 条、pds 2 条，尾部检索不到；LangMem 参考的 text-embedding-3-small 能看 8191 tokens），以及 Cognee 在 pds 下的块上限（从 4096 降到 510 tokens）。
- **适配不等于上游的原样**：每个系统都有偏离上游的地方（§2，详见各模块的说明），原因都写明了，但它们仍可能影响结果。外部系统换成 OpenAI 的模型会怎样，没有测。
- **写入的实际花费与耗时不能在系统之间比**（§3.2），只比名义成本。
- **Letta 没有接入**；A-MEM 的结果还没出来（MA-PENDING）。

## 8. 花费

数字来自 `data/ledger.sqlite`（不含命中本地缓存的调用），包括 N12 的试跑、两个协议的完整运行与泄漏消融，不只是这里的两张表。截至 2026-10-15 14:00 UTC，A-MEM 的完整运行还在进行，含它的总数：MA-PENDING。

- **通义**（从拿到 Key 起）：$22.61（扣除供应商的缓存折扣；名义 $29.56）。
- **DeepSeek**（2026-10-14 19:00 UTC 起：pds 的试跑与完整运行、泄漏消融的回答）：$12.04（扣除缓存折扣与错峰半价；名义 $33.72）。所有 DeepSeek 的运行都用 `--peak wait`，只在非高峰时段调用。

## 9. 复现

外部系统都不是 CueMem 的依赖，评测时用 `uv run --with` 临时装上。重跑时如果 `data/` 下的写入与响应缓存还在，会直接复用、几乎不花钱。从头跑时，服务端的回答不完全确定，数字会有小的波动（N11 §5.2）。

```bash
uv run cuemem data download locomo     # 划分 evals/splits/locomo_test200r.json 已在仓库里
export SPACY_SM=https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
export AMEM=git+https://github.com/agiresearch/A-mem@ceffb860f0712bbae97b184d440df62bc910ca8d
export CUEMEM_LANGMEM_MAX_LOST_SHARE=0.25   # 两个协议的完整运行都用
```

pds（要 `DEEPSEEK_API_KEY`；`--extra local` 装上本机向量）：

```bash
export CUEMEM_PRESET=deepseek
uv run --extra local cuemem eval run --dataset locomo --split test200r --peak wait \
    --max-cost 15 --methods B1,B2,B4-nobm25,B5,B6 --name pds-locomo-cuemem
uv run --extra local --with mem0ai==2.2.1 --with spacy --with fastembed \
    --with "en_core_web_sm @ $SPACY_SM" cuemem eval run --dataset locomo --split test200r \
    --peak wait --max-cost 5 --methods M0 --name pds-locomo-mem0
uv run --extra local --with "graphiti-core[kuzu]==0.30.2" cuemem eval run \
    --dataset locomo --split test200r --peak wait --workers 7 --max-cost 15 \
    --methods MG --name pds-locomo-graphiti
uv run --extra local --with langmem==0.0.30 --with langchain-openai==1.7.2 \
    --with langchain-core==1.6.10 --with langgraph==1.2.15 \
    --with langgraph-checkpoint==4.3.0 --with trustcall==0.0.39 \
    cuemem eval run --dataset locomo --split test200r --peak wait --workers 7 --max-cost 4 \
    --methods ML --name pds-locomo-langmem
uv run --extra local --with cognee==1.6.4 cuemem eval run --dataset locomo --split test200r \
    --peak wait --workers 7 --max-cost 8 --methods MC --name pds-locomo-cognee
uv run --extra local --with "agentic-memory @ $AMEM" cuemem eval run --dataset locomo \
    --split test200r --peak wait --workers 7 --max-cost 15 --methods MA --name pds-locomo-amem
uv run python scripts/n12_baselines.py --protocol pds --dataset locomo
```

pq（要 `DASHSCOPE_API_KEY`；写入放在另一个目录，不碰 pds 的写入）：

```bash
export CUEMEM_PRESET=qwen CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen
uv run cuemem eval run --dataset locomo --split test200r --peak warn \
    --max-cost 15 --methods B1,B2,B4-nobm25,B5,B6 --name pq-locomo-cuemem
uv run --with mem0ai==2.2.1 --with spacy --with fastembed \
    --with "en_core_web_sm @ $SPACY_SM" cuemem eval run --dataset locomo --split test200r \
    --peak warn --max-cost 8 --methods M0 --name pq-locomo-mem0
uv run --with "graphiti-core[kuzu]==0.30.2" cuemem eval run \
    --dataset locomo --split test200r --peak warn --workers 7 --max-cost 12 \
    --methods MG --name pq-locomo-graphiti
uv run --with langmem==0.0.30 --with langchain-openai==1.7.2 \
    --with langchain-core==1.6.10 --with langgraph==1.2.15 \
    --with langgraph-checkpoint==4.3.0 --with trustcall==0.0.39 \
    cuemem eval run --dataset locomo --split test200r --peak warn --workers 7 --max-cost 4 \
    --methods ML --name pq-locomo-langmem
uv run --with cognee==1.6.4 cuemem eval run --dataset locomo --split test200r \
    --peak warn --workers 7 --max-cost 6 --methods MC --name pq-locomo-cognee
uv run --with "agentic-memory @ $AMEM" cuemem eval run --dataset locomo \
    --split test200r --peak warn --workers 7 --max-cost 6 --methods MA --name pq-locomo-amem
uv run python scripts/n12_baselines.py --protocol pq --dataset locomo
```

- `--max-cost`（实际花费的上限，超过后不再开始新题）与 `--workers`（并发数）都与各运行的 `run.json` 相同；已出结果的运行都答完了 200 题，没有碰到上限。
- 运行名必须是 `<协议>-locomo-<系统>`：`n12_baselines.py` 把前缀为 `<协议>-locomo-` 的运行都归到一起，同一方法的结果按运行名的顺序后来的覆盖先来的。外部系统的写入目录按协议选，不看当前的 `CUEMEM_PATHS__EVAL_SPACES`。
- A-MEM 的 k = 40 敏感性运行复用同一份写入：加 `CUEMEM_AMEM_K=40`，运行名不能以 `<协议>-locomo-` 开头（会被 `n12_baselines.py` 归进来、覆盖 k = 10 的 MA 结果），如 `k40-pds-locomo-amem`。
