# 配置参考

[English](en/configuration.md) · 中文

CueMem 的每一个配置项、它从哪里来，以及怎样为自己的 OpenAI 兼容网关（OpenRouter、vLLM、LiteLLM、公司内部的代理）写一个预设。下面的表格由 `tests/test_docs_configuration.py` 对照 `src/cuemem/config.py` 里的配置模型检查：加了字段而这里没有对应的行，测试会失败。

API Key 不属于配置：它们放在环境变量或数据目录的 `.env` 里（`cuemem configure` 负责写）。不是配置字段的运行参数（`CUEMEM_HOME`、`CUEMEM_API_TOKEN`、`CUEMEM_LOG_LEVEL` 等）列在[环境变量](api.md#环境变量)一节。

## 配置从哪里来

四层，后面的覆盖前面的；嵌套的段逐个键合并：

1. **配置文件**：随包发布的 `src/cuemem/configs/default.yaml`（DeepSeek + Qwen）。环境变量 `CUEMEM_CONFIG`（或 `load_settings(config_path)`）指定的文件是**代替**它，不是叠加在它上面，所以那个文件必须完整：自己的 `providers`、`models` 与 `roles`。只想改几处时，写预设。
2. **预设**：`cuemem --preset <名字或路径>`、环境变量 `CUEMEM_PRESET` 或 `Memory(preset=...)`。名字是随包发布的预设之一（`openai`、`ollama`、`deepseek`、`qwen`）；其他的值当作 YAML 文件的路径。它深度合并在配置文件之上，所以**预设没有设的东西仍是 DeepSeek + Qwen 的默认值**。
3. **环境变量**：`CUEMEM_<段>__<字段>`，嵌套用两个下划线，例如 `CUEMEM_ROLES__READER__MODEL=qwen3.8-flash`、`CUEMEM_LLM_CACHE_MODE=off`。也可以写在数据目录的 `.env` 里；shell 里已经设了的变量优先于 `.env`。
4. **`Memory(config={...})`**：Python 里传的嵌套字典（或 `Settings` 对象），例如 `Memory(config={"retrieval": {"naive_k": 5}})`。

各层对不认识的键态度不同。`Memory(config=...)` 的字典会抛 ValueError，写明每个不认识的键和最像的已知键。YAML 文件与环境变量是宽松的：写错的键被忽略，用的仍是默认值。所以 `cuemem doctor` 会检查你自己写的文件（`CUEMEM_CONFIG` 的文件与按路径给的预设），每个不认识的键提醒一行：

```
配置文件 /path/my-preset.yaml 里有不认识的键 providers.gateway.max_concurency（是不是 providers.gateway.max_concurrency？），已忽略
```

值的类型不对、引用了没有定义的东西（角色用了 `models` 里没有的模型）在每一层都是错误：命令打印 `错误：配置有误：…` 并以退出码 1 结束。

## 字段

默认值是不用预设时实际生效的值。`providers`、`models`、`roles` 里的条目由你自己命名，它们的默认值一栏是某个字段不写时的值。

### 顶层字段（`Settings`）

各个段（`paths`、`providers`……）在下面各有一张表。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `llm_cache_mode` | `read_write` | 本地的 LLM 响应缓存：`read_write`；`read_only`（未命中就报错，用于严格复现）；`off` |
| `peak_policy` | `wait` | 供应商的高峰时段（`providers.<name>.peak_hours`）里遇到需要真实调用的请求：`wait` 暂停到高峰结束，`warn` 照常调用并记一条提醒，`allow` 照常调用。这个值管批量任务（评测）；SDK 与交互式命令（`ask`、`serve`、`proxy`、`mcp`）不看它、用 `warn`，SDK 的 `peak_policy` 参数可以另行指定 |
| `traces_retention_days` | `null` | trace 文件保留几天（≥ 1 的整数）；`null` 一直留着。`cuemem serve`、`cuemem proxy` 与 `cuemem mcp` 启动时与之后每天删掉更早的文件 |
| `expired_retention_days` | `null` | 记忆过期（`expiration_date`）之后再留几天就真的删掉（≥ 0 的整数）；`null` 不删（只是读取时不给出）。`cuemem serve`、`cuemem proxy` 与 `cuemem mcp` 启动时与之后每天执行 |

### `paths`

数据放在哪里。相对路径以数据目录为基准：`CUEMEM_HOME`；在源码仓库里运行时是仓库根目录，否则是 `~/.cuemem`。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `db` | `data/cuemem.sqlite` | 主数据库（SQLite）：原始日志与由它派生的一切 |
| `qdrant` | `data/qdrant` | 本地向量库（`vector_store.url` 为空时使用） |
| `llm_cache` | `data/cache/llm.sqlite` | LLM 响应缓存 |
| `embed_cache` | `data/cache/embed.sqlite` | 向量缓存 |
| `ledger` | `data/ledger.sqlite` | 调用量与成本的账本（`cuemem usage`） |
| `traces` | `runs/traces` | 每天一个的 trace 文件 |
| `datasets` | `data/datasets` | 下载的评测数据集 |
| `eval_runs` | `runs/eval` | 评测运行，每次一个子目录 |
| `eval_spaces` | `data/eval-spaces` | 评测用的记忆空间，跨运行复用 |
| `media` | `data/media` | 原图，按内容哈希存放 |
| `models` | `data/models` | 本机模型（fastembed 下载的向量模型） |

### `providers`

`providers.<name>`：每家模型供应商一项。默认配置里有 `deepseek` 与 `qwen`；`openai`、`ollama` 预设各加一家。预设只会加供应商，不会去掉默认的：没有被引用的供应商在 `cuemem doctor` 里标为未使用，也不需要 Key。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `base_url` | 必填 | OpenAI 兼容接口的地址，通常以 `/v1` 结尾 |
| `api_key_env` | `null` | 存放 Key 的环境变量的**名字**（不是 Key 本身）；不需要 Key 的服务（Ollama、vLLM）填 `null` |
| `max_concurrency` | `8` | 同时发往这家供应商的请求数 |
| `timeout_s` | `120` | 单次请求的超时（秒） |
| `max_retries` | `3` | 限流、5xx、连不上时重试几次。超时只在调用方没有兜底模型时重试 |
| `peak_hours` | `null` | 分时价格，见下；没有分时价格的供应商为 `null` |

### `providers.<name>.peak_hours`

一家供应商的分时价格（默认配置里 `deepseek` 设了）。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `timezone` | `Asia/Shanghai` | 时段所用的时区 |
| `weekdays` | `[0, 1, 2, 3, 4]` | 哪几天有高峰时段（周一 = 0） |
| `windows` | `["09:00-12:00", "14:00-18:00"]` | 高峰时段，`HH:MM-HH:MM` |
| `off_peak_factor` | `0.5` | 标价按高峰计；高峰之外的实际花费 = 标价 × 这个系数 |

### `models`

`models.<别名>`：模型目录。角色、兜底模型与记忆代理都按别名引用模型。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `provider` | 必填 | `providers` 里的名字 |
| `model_id` | 必填 | 发给供应商的模型名 |
| `price` | 必填 | 每百万 token 的标价，见下；只用于成本账本 |
| `currency` | `USD` | `USD` 或 `CNY` |
| `context_window` | `null` | 上下文窗口（token 数，仅作说明） |
| `vision` | `false` | 能不能看图。回答模型不能看图时只拿到图片的文字描述，不附原图 |
| `extra_body` | `{}` | 供应商特有的请求参数，原样发送（例如关闭思考模式） |
| `fixed_temperature` | `false` | 只接受默认温度的推理模型设 `true`：请求里不带 `temperature` |
| `max_tokens_param` | `max_tokens` | 输出长度上限的参数名：`max_tokens` 或 `max_completion_tokens` |

### `models.<alias>.price`

每百万 token 的标价，币种是模型的 `currency`。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `input` | 必填 | 输入 token（高峰时段、未命中缓存） |
| `output` | 必填 | 输出 token |
| `cached_input` | `null` | 命中供应商侧缓存的输入 token；`null` 表示与 `input` 相同 |

### `roles`

`roles.<角色>`：哪个模型干哪件事。一共七个角色：`extractor`（写入时的抽取）、`planner`（`mode="l2"` 的查询规划与充分性检查）、`reader`（回答）、`vlm`（回答时再看原图）、`captioner`（写入时的图片描述）、`judge`（评测判分）、`agent`（对话 Agent；记忆代理遇到不认识的模型名时也用它）。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `model` | 必填 | `models` 里的别名 |
| `temperature` | `0` | 采样温度（`fixed_temperature` 的模型不带） |
| `max_tokens` | `null` | 输出长度上限；`null` 由各次调用自己决定 |

### `embedding`

文本向量。数据目录记着建库时的向量模型，换了会明确报错：一个向量模型用一个数据目录。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `backend` | `dashscope` | `dashscope`（通义原生接口）、`openai`（任何 OpenAI 兼容的 `/embeddings`，地址取供应商的 `base_url`）、`fastembed`（本机 ONNX 推理，不要 Key，需要可选依赖 `local`）或 `hash`（离线哈希向量，仅用于测试与演示） |
| `provider` | `qwen` | `providers` 里的名字（`fastembed` 与 `hash` 不用） |
| `model` | `qwen3.7-text-embedding` | 向量模型 |
| `dim` | `1024` | 向量维度；要与模型实际返回的一致 |
| `sparse` | `false` | 同时取稀疏向量（只有 `dashscope` 支持） |
| `batch_size` | `10` | 每次请求的文本条数 |
| `max_chars` | `6000` | 每段文本向量化之前截到这么多字符 |
| `query_instruct` | `null` | 只加在查询上的指令（用带指令训练的模型时） |
| `send_dimensions` | `false` | `openai` 后端：请求里带 `dimensions`（只有支持截断维度的模型接受） |
| `native_url` | `https://maas.qwencloudapi.com/api/v1/services/embeddings/text-embedding/text-embedding` | `dashscope` 后端的接口地址 |
| `price_per_m` | `0.07` | 每百万 token 的价格（记账用） |
| `currency` | `USD` | `USD` 或 `CNY` |
| `timeout_s` | `60` | 单次请求的超时（秒） |
| `max_concurrency` | `2` | 同时在途的向量请求数 |
| `max_attempts` | `6` | 429、5xx、网络错误时最多尝试几次（指数退避） |

### `multimodal_embedding`

图片向量：图片与文字在同一个向量空间，用于以文搜图、以图搜图。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `backend` | `dashscope` | `dashscope`、`hash`（离线）或 `none`：不建图片向量。`none` 时图片照样保存、生成描述、按描述检索 |
| `provider` | `qwen` | `providers` 里的名字 |
| `model` | `tongyi-embedding-vision-plus` | 图文向量模型 |
| `dim` | `1152` | 向量维度 |
| `url` | `https://maas.qwencloudapi.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding` | 接口地址 |
| `price_per_m` | `0.108` | 每百万 token 的价格（记账用） |
| `currency` | `USD` | `USD` 或 `CNY` |
| `batch_size` | `8` | 每次请求的条数 |
| `max_side` | `768` | 图片向量化之前缩到最长边这么多像素 |
| `timeout_s` | `60` | 单次请求的超时（秒） |
| `max_concurrency` | `2` | 同时在途的请求数 |
| `max_attempts` | `6` | 429、5xx、网络错误时最多尝试几次 |

### `rerank`

`mode="l2"` 用的重排模型：对候选轮做 cross-encoder 打分。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `provider` | `qwen` | `providers` 里的名字；`null` 关掉重排（`mode="l2"` 跳过这一步） |
| `model` | `qwen3-rerank` | 重排模型 |
| `url` | `https://maas.qwencloudapi.com/api/v1/services/rerank/text-rerank/text-rerank` | 接口地址（通义原生的重排接口格式） |
| `price_per_m` | `0.1` | 每百万 token 的价格（记账用） |
| `currency` | `USD` | `USD` 或 `CNY` |
| `max_chars` | `4000` | 每段文本最多送这么多字符 |
| `timeout_s` | `60` | 单次请求的超时（秒） |
| `max_concurrency` | `4` | 同时在途的请求数 |
| `max_attempts` | `4` | 429、5xx、网络错误时最多尝试几次 |

### `vector_store`

向量存在哪里。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `url` | `null` | `null`：Qdrant 本地模式，数据在 `paths.qdrant`。填地址（`http://localhost:6333`）改用 Qdrant 服务端 |
| `api_key` | `null` | 服务端的 API Key。不要写进文件：在环境变量或 `.env` 里设 `CUEMEM_VECTOR_STORE__API_KEY` |

### `retrieval`

检索的条数与控制器。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `naive_k` | `8` | 调用方没给 `k` 时一次检索返回几条轮次 |
| `fact_k` | `10` | 事实增强检索：放进上下文的事实条数 |
| `fact_pool` | `50` | 事实增强检索：一次向量检索取回的候选数（轮与事实合计） |
| `controller` | `controller_model.json` | 检索控制器的模型文件（相对路径相对 `src/cuemem/read/`） |

### `consolidation`

异步巩固：抽取事实与实体、对账、会话摘要、核心记忆。相似度阈值是文本向量的余弦相似度。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `assistant_chars` | `600` | 抽取时每条助手回复保留的字符数；`null` 不截断 |
| `max_session_chars` | `24000` | 会话文本超过它时按轮次切成多个窗口分别抽取 |
| `workers` | `8` | 同一空间内并行抽取的会话数 |
| `background_workers` | `4` | 后台巩固同时处理的空间数（≥ 1；同一空间仍只有一个在巩固） |
| `fallback_model` | `deepseek-flash` | `models` 里的别名：主模型超时或被内容审核误拒时改用它；`null` 不兜底。它的供应商也要有 Key，而 `cuemem configure` 不会问 |
| `duplicate_threshold` | `0.93` | 陈述相似度超过它视为同一条事实（合并） |
| `conflict_threshold` | `0.75` | 单值事实与已有事实相似度超过它，交给 LLM 判断是新增、取代还是更正 |
| `multi_conflict_threshold` | `0.88` | 多值事实的同一个阈值 |
| `entity_merge_threshold` | `0.9` | 实体描述相似度超过它直接合并 |
| `entity_gray_threshold` | `0.75` | 介于两个阈值之间的实体交给 LLM 确认 |
| `profile_max_chars` | `1600` | 核心记忆的长度上限（约 400 tokens） |
| `profile_min_importance` | `4` | 进入核心记忆的最低重要度（1–5） |
| `instructions` | `null` | 给抽取的补充说明：记什么、不记什么（`Memory(instructions=...)`）。改了会重建派生记忆 |
| `language` | `null` | 记忆用什么语言写：`null` 是内置规则（按汉字的比例写成中文或英文），`auto` 跟用户的语言，别的值当作语言名（`Japanese`）。改了会重建派生记忆 |
| `caption_images` | `true` | 巩固时为图片生成描述；`captioner` 的模型不能看图时关掉 |
| `caption_max_side` | `768` | 图片送去描述之前缩到最长边这么多像素 |

### `proxy`

记忆代理（`cuemem proxy` 与 `cuemem serve` 的 `/v1`）。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `unknown_model` | `remap` | 请求的 `model` 不是 `models` 里的别名或模型 ID 时：`remap` 改用 `agent` 角色的模型（响应头 `X-CueMem-Model` 写明），`reject` 返回 400 |

## 为自己的网关写一个预设

下面的预设用于一个 OpenAI 兼容的网关，上面有一个对话模型和一个向量模型。存成 `my-preset.yaml`：

```yaml
providers:
  gateway:
    base_url: https://llm.example.com/v1
    api_key_env: GATEWAY_API_KEY
    max_concurrency: 4
models:
  my-chat:
    provider: gateway
    model_id: vendor/chat-model
    context_window: 128000
    vision: false
    price: {input: 0.5, output: 1.5}
roles:
  extractor: {model: my-chat}
  planner: {model: my-chat}
  reader: {model: my-chat}
  vlm: {model: my-chat}
  captioner: {model: my-chat}
  judge: {model: my-chat}
  agent: {model: my-chat, max_tokens: 1200}
embedding:
  backend: openai
  provider: gateway
  model: vendor/embedding-model
  dim: 1024
  batch_size: 32
  price_per_m: 0.02
multimodal_embedding: {backend: none}
rerank: {provider: null}
consolidation:
  fallback_model: null
  caption_images: false
```

先检查，再填 Key，然后测连通性（用一个新的数据目录，与用别的向量模型建的记忆分开）：

```bash
export CUEMEM_HOME=$(mktemp -d)                   # 或者你选的一个固定目录
cuemem --preset ./my-preset.yaml doctor           # 离线：看配置成了什么
cuemem --preset ./my-preset.yaml configure        # 只会问 GATEWAY_API_KEY
cuemem --preset ./my-preset.yaml doctor --live    # 每个模型与向量接口各调用一次（很小）
```

在源码仓库里命令是 `uv run cuemem …`。Python 里同一个文件是 `Memory(preset="./my-preset.yaml")`；想让一个 shell 里的所有命令都用它，`export CUEMEM_PRESET=/abs/path/my-preset.yaml`。

这时 `cuemem doctor` 应该显示所有角色都在 `gateway` 上、默认的供应商标着（未使用），以及这三行：

```
图片向量：关
重排：关
兜底模型：无
```

### 容易踩的坑

- **七个角色都要设。** 预设里没写的角色仍用默认的模型（`vlm` 与 `reader` 是 DeepSeek，其余是 Qwen），第一次调用时因为缺 Key 失败。`cuemem doctor` 的角色表写着每个角色用的是哪家供应商。
- **网关没有的能力要关掉**：`multimodal_embedding.backend: none` 与 `rerank.provider: null`。不关的话这两处仍走通义的原生接口。它们不在角色表里；doctor 把它们写在 `图片向量：` 与 `重排：` 两行。
- **`consolidation.fallback_model` 要设**成 `null`，或者你网关上的一个模型。默认是 `deepseek-flash`：主模型超时一次或被内容审核误拒一次，就会换到它。`cuemem configure` 不会问这家供应商的 Key，供应商表里它也标着未使用，所以缺 Key 时 doctor 把 `兜底模型：` 那一行标黄（`未设置 DEEPSEEK_API_KEY；不用兜底时设 consolidation.fallback_model: null`）。
- **模型不能看图**时，除了 `vision: false` 还要设 `consolidation.caption_images: false`，像随包的 `ollama` 预设那样。图片照样保存，只是不生成描述，回答时也不附原图。
- **一个向量模型一个数据目录。** 数据目录记着建库时的向量模型与维度，换了会明确报错。每个预设用自己的 `CUEMEM_HOME`。
- **`embedding.dim` 必须是模型实际返回的维度。** 模型支持截断维度时，设 `send_dimensions: true` 让请求里带上 `dim`。
- **写错的键会被忽略。** 每次改完都跑一遍 `cuemem doctor`，看有没有黄色的提醒。
- **推理模型**不接受 `temperature` 或 `max_tokens` 时，设 `fixed_temperature: true` 与 `max_tokens_param: max_completion_tokens`，像随包的 `openai` 预设那样。

`src/cuemem/configs/presets/` 里随包发布的预设是同样结构的完整例子。
