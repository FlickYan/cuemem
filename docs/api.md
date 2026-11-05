# 接口参考

[English](en/api.md) · 中文

程序使用 CueMem 的各种方式：Python SDK（`Memory`、`AsyncMemory`）、远程客户端（`MemoryClient`、`AsyncMemoryClient`）、`cuemem serve` 的 HTTP 接口（`/api` 与 Mem0 形状的 `/api/v1`）、OpenAI 兼容的记忆代理（`/v1`）、MCP Server（stdio 与 HTTP）、环境变量与可选依赖。签名以 CueMem 0.1.0 为准（`src/cuemem/sdk.py`、`client.py`、`api/`）。背后的决定见[决策记录](decisions.md)：D-061（SDK）、D-062（记忆代理）、D-063（导出）、D-056（部署与令牌）、D-069（第二轮体检，`/api/v1` 的大部分、`MemoryClient`、限定作用域的令牌、metadata、`update` 与 `reset` 都是这一轮加的）、D-070（第三轮体检，加了 `status`、HTTP 上的 MCP、请求的大小上限、闲置结束、带类型的客户端与下面 Mem0 的写法）、D-071（仓库公开之后的第四轮体检：按令牌的上限、Qdrant 的 API Key、公网监听更严的检查、请求头里不是 ASCII 的 ID、跨过重启的 run）、D-072（第五轮体检：`wait` 的上限、传输出错的代码、`valid_at` 按时间段、`cuemem reindex`）、D-073（第六轮体检：`context`、幂等的 `add`、`cuemem keys` 与令牌文件的热加载、每个令牌同时处理的请求有上限）。

语言：SDK 的返回值与它自己的报错、`/api/v1`、记忆代理与 MCP 工具是英文；命令行、较早的 `/api` 接口、401 的说明，以及核心模块更深处抛出的报错（缺 Key、向量库被占用、配置写错）是中文。

## 安装

| 需要什么 | 安装 |
|---|---|
| SDK、命令行、HTTP 接口、记忆代理、MCP | `pip install "cuemem @ git+https://github.com/FlickYan/cuemem"` |
| + 本机向量，不要 Key（fastembed；`deepseek` 预设要用） | 可选依赖 `local` |
| + `cuemem serve` 的浏览器界面（Gradio） | 可选依赖 `ui` |
| + 评测的图，以及控制器与门控的训练脚本（matplotlib、scikit-learn） | 可选依赖 `eval` |
| + 框架集成 | 可选依赖 `openai-agents`、`langgraph`、`anthropic`，三个都要时 `integrations` |

可选依赖可以叠加：`pip install "cuemem[ui,local] @ git+https://github.com/FlickYan/cuemem"`。CueMem 还没有发布到 PyPI，PyPI 上名叫 cuemem 的包不是这个项目：一律从 GitHub 安装。需要 Python 3.12 或更新的版本。源码仓库里 `uv sync` 经开发依赖组装上 `ui` 与 `eval`；`uv sync --no-dev` 时要加 `--extra ui --extra eval`。

- 没装 `ui` 时，`cuemem serve` 只提供 `/api`、`/api/v1`、`/mcp`、`/docs` 与 `/v1`，并说明怎么装界面（`pip install "cuemem[ui] @ git+https://github.com/FlickYan/cuemem"`，源码仓库里是 `uv sync --extra ui`）。
- 没装 `eval` 时，`cuemem eval run`（`--build-only` 除外）与 `cuemem eval report` 给出同样的安装提示、退出码 1；`eval split`、`eval probe`、`eval audit` 与 `data download` 照常。
- Docker 镜像缺省装 `ui` 与 `local`；`docker build --build-arg EXTRAS= .` 构建只有接口的镜像。镜像里缺可选依赖时办法是重新构建，例如 `docker build --build-arg EXTRAS="--extra ui --extra local" -t cuemem .`（再加 `--extra eval` 等），镜像里的提示也这样说：容器里的 pip 装进别的环境，镜像里也没有 git。
- 用 pip 装的依赖里有 `cuemem[ui]` 时 Pillow 仍解到 13 以下：gradio 声明了 `pillow<13`（只装 `cuemem[local]` 时解到 fastembed 0.9.1，它允许 Pillow 13），而 Pillow 12 有只在 13 修好的安全公告。接收别人图片的服务请用 Docker 镜像，或在源码仓库里 `uv sync --locked`，它们装的是 Pillow 13。两者发给模型的图片字节相同。
- `local` 在 Apple Silicon 上要 macOS 14 或更新（onnxruntime 1.24 起只有 macOS 14 arm64 的 wheel）。Intel Mac 上锁文件给 `local` 装 onnxruntime 1.23.2（Python 3.12 与 3.13；那里不支持 Python 3.14），`uv sync` 要从源码构建 cryptography（需要 Rust 工具链）；用 Docker 镜像可以避开这两点。
- 包带着 `py.typed`，mypy 与 pyright 会检查对 CueMem 的调用（升级之后可能报出新的、正确的错误）。

## 记忆空间与身份

`user_id` 与 `agent_id` 合起来定一个记忆空间，各空间互不相通：

| 参数 | 空间 |
|---|---|
| `user_id="alice"` | `alice` |
| `agent_id="shop"` | `agent/shop`（这个 Agent 自己的记忆） |
| `user_id="alice", agent_id="shop"` | `shop/alice`（Agent shop 记得的关于 alice 的事） |

- ID 里的 "/" 与 "%" 先做百分号转义，所以 `user_id="agent/x"` 与 `agent_id="x"` 不会落进同一个空间。两个 ID 至少给一个（否则 ValueError）。`cuemem.sdk.space_for(user_id, agent_id)` 算出空间，`space_owner(space)` 是它的逆。
- `run_id` 是空间里的一个会话（一段对话）。不给时每次 `add` 自成一个会话。
- 记忆 ID（事实、对话片段或图片的 `id`）本身属于某一个空间，所以 `get`、`history`、`update`、`delete` 与 `media` 只要 ID。

## Python SDK：`Memory`

```python
from cuemem import Memory

# Memory() 用默认配置，Memory(preset="ollama") 用本机模型
m = Memory(preset="openai")
m.add("I moved to Munich last month.", user_id="alice", wait=True)
m.search("Where do I live now?", user_id="alice")
m.close()  # 或者：with Memory() as m: ...
```

`Memory` 线程安全。方法返回普通的字典与列表，形状照 Mem0。

### 构造

```python
Memory(
    config: Mapping[str, Any] | Settings | None = None,
    *,
    preset: str | None = None,
    home: str | Path | None = None,
    instructions: str | None = None,
    background: bool = True,
    peak_policy: Literal["wait", "warn", "allow"] | None = None,
    http_client: httpx2.Client | None = None,
)
Memory.from_config(config: Mapping[str, Any], **kwargs) -> Memory
Memory.from_system(system: MemorySystem) -> Memory
```

| 参数 | 含义 |
|---|---|
| `config` | 嵌套的字典，与默认配置及预设深度合并，例如 `{"roles": {"reader": {"model": "qwen3.8-flash"}}}`（字段见 `src/cuemem/configs/default.yaml`）；也可以是 `Settings` 对象 |
| `preset` | `openai`、`ollama`、`deepseek`、`qwen`，或预设 YAML 的路径（不填时看环境变量 `CUEMEM_PRESET`） |
| `home` | 数据目录（默认 `CUEMEM_HOME`；源码仓库里是仓库根目录，否则 `~/.cuemem`） |
| `instructions` | 给抽取的补充说明：记什么、不记什么（Mem0 的 `custom_instructions`） |
| `background` | 在后台线程里抽取事实（默认开；`consolidation.background_workers` 个线程，缺省 4，同一空间一次一个） |
| `peak_policy` | 调用落在 DeepSeek 工作日高峰时段时怎么办：`warn`（SDK 的缺省：按高峰价照常调用并记下来）、`wait`（暂停到高峰结束，在 `cuemem.sdk` 日志上记一条警告）、`allow`（照常调用、不提醒） |
| `http_client` | 调用模型与向量供应商用的 httpx2 客户端（测试用它换上假的网络层） |

- **`config` 里不认识的键报 ValueError**：在打开任何东西之前，列出每个带点的键名与最接近的写法。Mem0 格式的配置另有改写提示：`llm` 与 `embedder` 换成 `preset="openai"` 或 `models` / `roles`，`custom_instructions` 换成 `Memory(instructions=...)`，`vector_store` 的 provider 配置换成 `{"vector_store": {"url": "http://host:6333"}}`；`preset`、`home`、`config_path` 与 `env_file` 是关键字参数。以前不认识的键悄悄忽略，跑的是 DeepSeek + 通义的缺省配置。`Settings` 对象、`CUEMEM_*` 环境变量与 YAML 文件照旧宽松。
- `from_config(config)` 就是 `Memory(config)`：CueMem 的配置字典，不是 Mem0 的 `MemoryConfig`。
- `from_system` 包住进程里已经打开的记忆系统（HTTP 服务与记忆代理就这样用）。本地模式的 Qdrant 同一时间只允许一个进程打开数据目录，`cuemem serve` 正占着的目录再 `Memory()` 会报 `StorageLockedError`；这时用 `MemoryClient` 连服务。

### 方法

| 方法 | 返回 |
|---|---|
| `add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, expiration_date=None, timestamp=None, infer=True, wait=False, idempotency_key=None)` | `{"results": [事件], "session_id", "turn_ids", "status", ...}` |
| `search(query, *, user_id=None, agent_id=None, run_id=None, limit=None, mode="auto", at=None, router="learned", lam=None, filters=None, top_k=None, min_level="l1", show_expired=False)` | `{"results", "images", "profile", "level", "see", "reasons"}` |
| `context(query, *, user_id=None, agent_id=None, at=None, mode="auto", router="learned", lam=None, min_level="l1", wrap=True)` | `{"context", "level", "reasons", "recalled", "see", "refs"}` |
| `answer(question, *, user_id=None, agent_id=None, mode="auto", at=None, router="learned", lam=None)` | `{"answer", "evidence_sufficient", "citations", "level", "reasons"}` |
| `get_all(*, user_id=None, agent_id=None, run_id=None, include_superseded=False, valid_at=None, known_at=None, limit=None, filters=None, top_k=None, show_expired=False)` | `{"results": [事实]}` |
| `get(memory_id)` | 一条事实、对话片段或图片，或 `None` |
| `media(media_id)` | 图片原文件的字节，或 `None` |
| `history(memory_id)` | 一条事实的版本链，从最早到最新，每个版本带 `"event": "ADD"` 或 `"UPDATE"`；别的 ID 是 `[]` |
| `update(memory_id, text=None, *, metadata=None, data=None)` | `{"message", "id", "results", "session_id", "turn_ids", ...}` |
| `delete(memory_id, *, dry_run=False)` | 遗忘报告 |
| `delete_all(*, user_id=None, agent_id=None, run_id=None)` | 遗忘报告 |
| `reset()` | `{"message", "spaces_deleted", "turns_deleted", "images_deleted"}` |
| `profile(*, user_id=None, agent_id=None)` | 画像的文字 |
| `status(*, user_id=None, agent_id=None)` | `{"pending", "failed", "background", "errors"}` |
| `open_run(run_id=None, *, user_id=None, agent_id=None)` | run 的 ID |
| `end_run(run_id, *, user_id=None, agent_id=None, wait=False)` | `{"session_id", "turns", "consolidation"}` |
| `run(run_id=None, *, user_id=None, agent_id=None, wait=False)` | 上下文管理器，给出 run 的 ID |
| `close_idle_runs(idle_s, *, now=None)` | 结束的 run 数（只在本地；`cuemem serve` 每分钟调用一次） |
| `consolidate(*, user_id=None, agent_id=None, rebuild=False)` | 巩固报告 |
| `wait(*, user_id=None, agent_id=None, timeout=None)` | `None`：等后台线程把已登记的抽取做完（给了 ID 时只等这个用户或 Agent 的）；`timeout` 秒内没做完时 TimeoutError（抽取在后台照常继续） |
| `export(*, user_id=None, agent_id=None, derived=True)` | `cuemem.space` 格式的导出（能直接写成 JSON 的字典） |
| `load(data, *, user_id=None, agent_id=None, wait=False)` | 导入报告 |
| `spaces()` | 每个空间一个字典：`space_id`、`created_at`、`sessions`、`turns`、`units`、`current_facts`、`facts`、`entities` |
| `close()` | `None`：结束这个对象开着的 run（登记它们的抽取），关闭记忆系统 |

**`add`**。`messages` 是一句话、一条 OpenAI 格式的消息或消息的列表；只存用户与助手的消息。图片放在内容块里（`{"type": "image_url", "image_url": {"url": ...}}`），可以是 data URL 或本地路径；http(s) 的图片地址不下载（ValueError），像素超过 Pillow 的 `Image.MAX_IMAGE_PIXELS`（缺省 89,478,485）的图片以 ValueError "Image too large" 拒绝，什么都不存。发言在线写入，不调用 LLM，立即可检索；事实在后台抽取。

- `run_id`：写进这个会话。每轮 `add` 一次同一个 `run_id` 时，每次都把整段会话重新抽取一遍（还要重放派生记忆）；一段多轮对话用 `with m.run(...)` 包起来，结束时只抽取一次。
- `metadata`：JSON 对象，键是字符串，记在这次写入的每句发言上。它不给抽取模型看，请求不变。由这些发言得来的事实、对话片段与图片都带上它，`filters` 可以按它筛。不是字典时 TypeError；键不是字符串、值写不成 JSON 时 ValueError。
- `expiration_date`：日期、时间或 ISO 字符串；只给日期时到那天结束。过期的记忆不再出现在 `search`、`get_all` 与画像里，除非 `show_expired=True`；`get` 与 `history` 照样取得到，要 `delete` 才删。同一件事后来又说过一次、没给过期时间，它就不过期。过期的记忆也不出现在 `answer()`、记忆代理注入的 `<memory>`、MCP 与 HTTP 的 recall（`cuemem mcp`、`/mcp`、`/api/spaces/{space}/recall`）、各处的画像（`profile()`、MCP 的 profile 工具、`GET /api/spaces/{space}/profile`、界面的画像栏）、`cuemem serve` 的对话 Agent（它的画像、召回与 `memory_search` 工具），以及 Claude、OpenAI Agents 与 LangGraph 的集成里（召回、自动召回，Claude memory tool 的 `profile.md`、`facts.md` 与 `history/`）。过期按现在的时钟算，不按提问的时间 `at`。浏览原始数据的事实列表（`GET /api/spaces/{space}/facts`、界面的事实列表）、实体图（`GET /api/spaces/{space}/graph`、界面的实体图）、图库（`GET /api/spaces/{space}/media`、界面的图库）、`cuemem ask` 与 `cuemem inspect` 仍给出过期的记忆，`get`、`history` 与导出也是。
- `timestamp`：这些话是什么时候说的（时间、ISO 字符串或 Unix 时间戳；默认现在）。带时区的时间换算成本机时间。
- `infer=False`：只写日志（`status: "logged"`），之后用 `consolidate()` 抽取。空间里别处的遗忘不会抽取这样的会话，导入它们的导出文件也不会（见导出与导入）；`consolidate()`、`add(wait=True)` 与 `update()` 会，它们巩固的是整个空间。
- `wait=True`：同步抽取，返回 Mem0 风格的事件（`status: "consolidated"`，另有 `consolidation`）。run 还开着时不能用（ValueError：用 `end_run(..., wait=True)` 结束它）。发言存下之后抽取出错（例如模型供应商余额不足）时，`add` 不抛出：返回 `status: "queued"` 与 `error`；发言已经存下，下次巩固时补上抽取，不要再调用一遍。
- 写进去之前出错（向量接口不可用、图片不对）时照常抛出，什么都没存，可以原样重试（经 `MemoryClient` 时只在代码是 `unreachable`、请求根本没到服务端时可以，见下）。发言提交之后不会再抛出：写不进去的 trace（磁盘满、trace 路径不对）只在 `cuemem.core.trace` 日志里记一条警告，下一句的序号与发言写在同一个事务里。
- `idempotency_key`：调用方自己给的键（1–255 个可打印的 ASCII 字符，例如一个 UUID；否则 ValueError），让重发的 add 安全。24 小时内，同一个用户（空间）下同一个键、同一个请求只写一次：重发返回 `status: "duplicate"`、原来的 `session_id` 与 `turn_ids`、`results: []` 与 `replayed: true`，`wait=True` 也不抽取。那次写下的发言已经遗忘了时（`delete`、`delete_all(run_id=...)`、任何遗忘），重发返回 `status: "forgotten"`（`session_id` 为 None，`turn_ids` 为 []），什么都不写。同一个键用于不同的请求（`messages`、`user_id`、`agent_id`、`run_id`、`metadata`、`expiration_date`、`timestamp` 或 `infer` 不同；`wait` 不比）时抛 `IdempotencyConflictError`，什么都不写。库里只存键的 sha256。幂等键在 24 小时内保护重发，也跨过对写下的记忆的 forget() / DELETE（重发返回 status "forgotten"）。delete_all() 与 reset() 连同别的一起删掉这个用户的幂等记录，所以之后到的重发是一次新的写入。一个边角情况：带键的 add 提交之后抛 `ConcurrentWriteError`（同一个 `run_id` 里同时进行的写入拿走了同样的序号）时，记录里的发言 ID 并没有全部写下。不给键时 `add` 与以前完全相同。
- `status` 是 `empty`（没有可存的消息）、`logged`、`queued`、`consolidated` 之一，带 `idempotency_key` 时还可能是 `duplicate` 与 `forgotten`（这时 `replayed` 为真）。并发写入撞了发言序号时抛 `ConcurrentWriteError`。

事件是一条带 `"event": "ADD"` 或 `"UPDATE"` 的事实；UPDATE 另有 `previous_memory`（被取代的陈述）与 `reason`（`changed`：现实变了；`corrected`：之前说错了）。

**`search`**。检索控制器决定想多深；`level`（`L0`、`L1`、`L2`）、`see`（控制器是否决定看原图）与 `reasons`（英文）说明它选了什么、为什么。

- `min_level="l1"`（缺省）：调用 `search` 就是已经决定要查记忆，控制器选了 L0 时改走 L1，reasons 是 `["raised from L0: the caller asked to recall"]`，与 MCP、各框架集成相同。`min_level="l0"` 时照控制器的决定，L0 只返回画像。
- `mode`：`auto`（控制器），或固定的 `l0`、`l1`、`l2`。`router`：`learned`（缺省）、`hybrid`（门控拿不准时问 LLM）、`rule`、`llm`。`lam`：成本旋钮 λ（缺省取控制器模型里的值；越小越舍得花钱），≥ 0 的有限数。`mode`、`router`、`min_level` 与 `lam` 最先检查，类型与 `/api/v1` 相同，空的空间也查：写错时抛 ValueError，说明是 HTTP 的那句去掉 `Invalid request: body.` 前缀，例如 `mode: Input should be 'auto', 'l0', 'l1' or 'l2'`。
- `at`：提问的时间，相对日期按它换算。
- `limit` 或 `top_k`：截断 `results`，要是正整数（0、负数与布尔值报 ValueError）。`run_id`：只要这个会话的记忆。`show_expired=True` 时过期的记忆也给（画像里始终没有）。
- `filters`：Mem0 v3 的写法。`user_id`、`agent_id`、`run_id` 可以写在这里代替关键字参数（`user_id`、`agent_id` 只能是一个 ID）。其余的键都是 metadata 的条件；`created_at`、`updated_at` 与 `categories`（Mem0 保留的名字）在 metadata 没有同名的键时也对记忆自己的字段（结果里 `created_at` 的 ISO 字符串；事实的 category，作为一个元素的列表）：
  - `{"键": 值}` 是等于，`{"键": "*"}` 是有这个键，几个键并列时都要满足；
  - `{"键": {"eq" | "ne" | "gt" | "gte" | "lt" | "lte": v, "in" | "nin": [...], "contains" | "icontains": v}}`；没有这个键时只有 `ne` 与 `nin` 成立；
  - 值是列表时（`categories`，或 metadata 里的列表），`eq`、`ne`、`in` 与 `nin` 逐项比较：`in` 有一项在列表里就成立，`nin` 一项都不在才成立，`eq` / `ne` 对列表里的项（参数本身是列表时比整个列表）；`{"键": 值}` 同 `eq`；
  - ISO 字符串按字符串比较，所以 `{"created_at": {"gte": "2026-01-01"}}` 能用，只写日期的上界如 `{"lte": "2026-08-01"}` 不包括那天晚些时候的时间；
  - `{"AND": [条件, ...]}`、`{"OR": [...]}`、`{"NOT": [...]}`（NOT：一个都不满足）；
  - 条件里的 `user_id`、`agent_id`、`run_id` 指记忆所在的空间与会话，例如 `{"run_id": {"in": ["r1", "r2"]}}`；
  - 不认识的运算报 ValueError。
- 检索照常在整个空间里做，再按会话、metadata 与过期时间筛，所以结果可能少于 `limit`。画像是整个空间的，去掉过期的事实。
- `score` 只用来排序：L1 是排名融合的分数（第一名约 0.016），不是相似度，所以没有 `threshold`（按 Mem0 缺省的 0.1 截断会什么都不剩）。Mem0 的 `rerank` 对应 `mode="l2"`。

结果里每一项带 `type`：

| `type` | 字段 |
|---|---|
| `fact` | `id`、`memory`（陈述）、`subject`、`predicate`、`object`、`category`、`importance`、`status`、`valid_from`、`valid_to`、`created_at`、`invalidated_at`、`superseded_by`、`session_id`、`source_turn_ids`、`metadata`、`expiration_date`；有时还有 `score`、`user_id`、`agent_id` 或 `space_id` |
| `excerpt` | `id`、`memory`（那一轮对话）、`score`、`created_at`、`session_id`、`source_turn_ids`、`metadata`、`expiration_date`（`get` 给出的没有 `score`，另有 `space_id`） |
| `image`（在 `images` 里） | `id`、`memory`（图片描述）、`path`（服务器上的路径；原文件用 `media` 取）、`created_at`、`attached`、`metadata`、`expiration_date`（`get` 给出的 `attached` 是 false，另有 `space_id`） |

`metadata` 是出处各句 metadata 的合并（同一个键取后面那句的）；`expiration_date` 是出处里最晚的过期时间，有一句没给就是 `None`。

**`context`**。返回记忆代理注入的那段证据块，直接放进提示词：画像、带日期与有效状态（当前 / 不再成立 / 已更正）的事实、按时间排列的对话片段，记忆里的 `<memory` / `</memory` 已经转义，开头一段说明这些是资料、不是指令。不调用模型。`wrap=True`（缺省）时包在 `<memory>\n...\n</memory>` 里：就是记忆代理放在最后一句用户消息前面的那段，去掉末尾的空行；`wrap=False` 时只有块本身。没有要给的内容时 `context` 是空串（空的空间时 `reasons` 是 `["no memories yet"]`）。`recalled` 是条数（画像算一条）。`refs` 是块里每个编号（`[f1]` 记作 `f1`、`[m1]` 记作 `m1`……）对应的、`get()` 认得的记忆 ID。与 `search` 相同，缺省的 `min_level="l1"` 把控制器选的 L0 抬到 L1（reasons：`raised from L0: the caller asked to recall`）；`min_level="l0"` 照控制器的决定，与记忆代理逐字节相同。过期的记忆（`expiration_date`）不给，画像里也去掉。`mode`、`router`、`min_level` 或 `lam` 写错时与 `search` 一样抛 ValueError（HTTP 422）。拿 `search()` 的结果自己拼提示词会丢掉日期与有效状态，也不转义 `</memory>`：

```python
prompt = m.context("Where do I work?", user_id="alice")["context"] + "\n\nWhere do I work?"
```

**`answer`**。检索，再用配置的回答模型回答。`mode` 另外可以是 `agent`（回答模型自己决定什么时候调用 `memory_search` 工具）与 `hatch`（控制器先召回，回答模型还可以补查）。`citations` 是 `{"ref", "id", "memory"}`。`mode` 只能是 `auto`、`l0`、`l1`、`l2`、`agent` 与 `hatch`（否则 ValueError；评测的基线 `naive` 与 `facts` 只在 `MemorySystem.ask` 与命令行里），`router` 与 `lam` 的检查同 `search`。

**`get_all`**。缺省只给当前有效的事实。`include_superseded=True` 加上被取代的版本；`valid_at` 回放那时现实中成立的：`"2025"`（全年）、`"2025-06"`（整个月）或 `"2025-06-07"`（那一天），或 date、datetime；年或月给出那段时间里任何时候成立的事实，所以先后两个值都可能出现（`valid_at="2026"` 可能同时给出旧地址与新地址）。结束时间只知道年或月的值，算在那一年或那个月里；更正了的记忆从它更正的那个值开始的时候算起（它自己写明的开始时间优先），所以查更正之前的日期给出的是更正之后的值。带时间的字符串整个按 ISO 时间解析，带时差或 `Z` 的先换算成本机时间再取日期，与 datetime 相同。别的写法报 ValueError（`GET /api/v1/memories` 400）；`known_at=<时间>` 回放系统那一刻相信的。`run_id`、`filters`、`limit` / `top_k` 与 `show_expired` 同 `search`；缺省不给过期的事实。

**`get`、`media` 与 `history`**。`search` 给出的每个 ID 都能按 ID 用。`get` 返回一条事实、一段对话片段（一轮对话）或一张图片，过期的也给；ID 不存在时是 `None`。`media(media_id)` 返回图片原文件的字节（别的 ID、或原文件已经不在时是 `None`）。`history` 返回一条事实的版本链；对话片段与图片没有版本（`[]`）。

**`update`**。事实不就地改写：派生记忆是原始日志的函数。`update` 在一段新的 `edit-…` 会话里记一句用户说的 `"Correction: <text>"`（排在原来那条之后），再巩固；由它抽出的事实以 `corrected` 取代原来那条，不问 LLM。新事实继承原来那条的 metadata（`metadata` 里的键覆盖）与过期时间；`history()` 里看得到前后两个版本，遗忘这句更正时原来那条恢复。

- `text` 是 Mem0 2.x 的名字，`data` 是它旧的名字：给一个就行（两个都给且不同时 ValueError）。只给 `metadata` 时 ValueError：metadata 记在出处的发言上，要随一句更正写进去。
- `message` 是 `"Memory updated"`。
- 从这句话里抽不出事实时，原来那条不变，这句更正也从日志里遗忘，`turn_ids` 是空的，`message` 写明。
- 更正记下之后抽取出错时，`message` 写明、`error` 给出原因；不要再调用一遍。
- ID 不存在时 KeyError；这条已经被取代（请改当前的版本）、它是对话片段或图片（只有事实能改）、或者文字是空的时 ValueError。

**`delete` 与 `delete_all`**。遗忘对应"被遗忘权"。`delete` 删掉事实出自的那一轮对话，沿引用找到之后引用过它的回复，并找出没有引用的转述，再用剩下的日志重建派生记忆：被它取代过的旧事实重新成为当前的。它也认对话片段与图片的 ID，遗忘那一轮或那张图。`dry_run=True` 只列出会删什么，什么都不删。`delete_all` 删掉整个空间：日志、派生记忆、向量、缓存与原图，还有这个空间的 trace（连同其中提问的查询向量）与库里写着它的元信息；给了 `run_id` 时只删这个会话。发言数超过 SQLite 参数上限（32,766）的用户也删得掉：查询是分批的。

- 报告里有 `message`、`turns_deleted`、`sessions_affected`、`facts_before`、`facts_after`、`images_deleted`、`restated`；删除做完、用剩下的日志重建出错时另有 `warning`（下次巩固时补上）。
- 还没删任何东西就出错的遗忘（向量接口不可用、向量库或数据库出错）照常抛出、什么都不删，可以原样重试。
- 删除提交之后的清理从不抛出。某天的 trace 文件重写不了时（例如磁盘满了），CueMem 把那一天的文件整个删掉：同一天别的用户的调试 trace 也没了，`cuemem.core.trace` 记一条警告，这些 trace 的查询向量一起删掉。重写不了也删不掉的文件写在遗忘 trace 的 `traces_left` 属性里，要手工删。清理向量缓存出错时（例如 "database or disk is full"，或源码仓库里与评测共用的 `embed.sqlite` 报 "database is locked"），`cuemem.memory` 记一条警告，遗忘的 trace 记 `embed_cache_left: true`；删除照样成功，这次清理不再重试。`delete_all` 与 `reset` 之后同样。
- 除了提问的查询向量，遗忘还删掉 L2 的子查询与追加查询、Agent 自己的记忆检索与导购演示的商品检索的查询向量，用 DashScope 多模态向量器时还删以文搜图的查询向量，以及别的记忆不再用的被删图片的向量。
- 同一个记忆系统上开着的对话会话（界面、`/api`）在任何一次删除之后的下一轮去掉被删的内容，不管删除从哪里来（包括 `/api/v1` 的 delete、`delete_all` 与 `reset`）；删除重建期间开始的会话，下一轮取回重建好的画像。
- 一句发言都没删、事实却还在时（它出处的发言已经不在日志里，旧版本可能留下这种情况），`message` 是 `"Nothing deleted"`，`warning` 让调用方 `consolidate(rebuild=True)`（之后检索仍缺对话片段时再 `cuemem reindex`）。
- 删一条事实或一段对话片段遗忘的是整轮对话，同一轮的别的 ID 之后报 KeyError，与不存在的 ID 一样。照 Mem0 写的 `for r in results: m.delete(r["id"])` 要接住 KeyError。
- 不要在 `m.system.reading(space)` 里调用 `m.system` 的 `forget`、`delete_space` 或 `consolidate`：会抛 RuntimeError，而不是死锁。

**`reset`**。删掉数据目录里的每个空间，包括 HTTP 服务、记忆代理与演示写的空间，并结束开着的 run。不可恢复。经 `/api/v1` 用限定作用域的令牌调用时，只删这个令牌作用域里的空间。

**`status`**。一个用户（或 Agent）的记忆的抽取状态：`pending`（等着抽取的会话数）；`failed`，每个抽取失败的会话一项，带 `session_id`、`attempts`、`error`（出错的第一行）与 `retry_at`（后台线程什么时候重试，ISO 时间；失败到 3 次、流水线版本不同、或没有后台线程时是 `null`，这时要 `consolidate()`）；`background`（有没有后台线程）；`errors`，后台线程在这个空间最近的 3 条出错。后台抽取失败时还在 `cuemem.write.consolidate` 日志里记一条警告，同一空间、同一类出错只记一次。

**run**。`open_run` 开始一段对话：之后 `add(run_id=...)` 照常在线写入、立即可检索，但到 `end_run` 才抽取，那时整段交给抽取（`wait=True` 当场抽取并返回报告，否则 `consolidation` 是 `"queued"`）。`with m.run(user_id="alice") as run_id:` 两步都做。本地的 SDK 自己不结束 run；`cuemem serve` 下闲置 30 分钟的 `/api/v1` run 会被结束（`close_idle_runs`），同一个 `run_id` 的下一次 `add` 重新打开它：它结束时整段再抽取一次、从日志把这个用户的整个空间重放一遍（D-045），重放期间这个用户的事实在 `search` 与 `get_all` 里暂时没有，约为一次抽取的时间。

- **跨过重启**：服务停止或崩溃时还开着的 run 跨过重启，用同一个 `run_id` 的下一次 `add`（或 `open_run`）接着写。`cuemem serve` 与 `cuemem proxy` 下，启动后第一个闲置期里别的请求不接手它，没人接着写的在启动一个闲置期（`--idle-minutes`）之后交给抽取；本进程交接过、调用方又回来接着写的 run，像闲置结束的 run 一样重新打开，但 `open_run` 与 `end_run` 会去掉这条记录：`end_run` 之后同一个 `run_id` 的 `add` 是普通的会话（`add(wait=True)` 照常巩固）。这样留给调用方的 run，重建时也跳过（`consolidate(rebuild=True)`，或部署改了模型、巩固配置之后的自动重建）。本地的 SDK 与命令行第一次碰到这个空间时，就把调用方不接着写的 run 交给抽取。
- **接不上的**：之前的进程已经因闲置结束或交给抽取的 run，重启之后算普通会话，因为那个进程只把这两种记在自己的内存里；本进程交接过、超过 10,000 个之后不再记得的 run 也一样：之后每次 `add` 都整段再抽取、重放这个用户的整个空间（D-045），每次重放期间这个用户的事实在 `search` 与 `get_all` 里暂时没有。长时间停顿的客户端应该在自然的段落处 `end_run`、再开一个新的 run。

**导出与导入**。`export` 返回 `cuemem.space` 格式的版本 2：会话、发言（有 `metadata`、`expires_at`、`edit_of` 时带上）与 base64 的原图，另可附上事实与画像的快照（导入时不用）。`load` 把原始日志写回去，再重新抽取派生记忆（`wait=True` 时当场做）。不给 `user_id`、`agent_id` 时导入到文件里原来的空间，这时文件里没有 `space_id` 就报 ValueError。版本 1 的文件照样能导入；文件里有任何一张图或一处 metadata 不对时，什么都不写，格式坏了（缺键、类型不对）的文件报 `ValueError("Malformed export: …")`。只写了日志（`add(infer=False)`）、从没登记过抽取的会话带 `"infer": false`；导入时同样不登记，后台线程不去抽取它，直到显式的 `consolidate()`、`load(wait=True)` 或 `cuemem import --consolidate`，与源空间相同。进行中的 run 从不带这个键。格式仍是版本 2；旧版本的 CueMem 忽略这个键、照样抽取，`infer` 不是布尔值的是格式坏了的文件。

### 从 Mem0 迁过来

| Mem0 | CueMem |
|---|---|
| `Memory.from_config({"llm": ..., "embedder": ..., "vector_store": ...})` | ValueError 并给出改写提示：用 `preset="openai"` 或 `models` / `roles`、`{"vector_store": {"url": ...}}` 与 `Memory(instructions=...)` |
| `update(memory_id, text=..., metadata=...)` | 相同；`data=` 是别名，只改 metadata 时 ValueError |
| `update(..., expiration_date=None)` 去掉过期时间 | 还不支持 |
| `search(..., threshold=0.1)` | 没有 threshold：分数只用来排序（L1 是排名融合，第一名约 0.016） |
| `search(..., rerank=True)` | `mode="l2"` |
| `search` / `get_all` 列出过期的记忆 | `show_expired=True` |
| `delete(id)` 删一条记忆 | 删整轮对话，同一轮之后的 ID 报 KeyError |
| `MemoryClient(api_key=..., host=...)` | 相同（`host` 是 `base_url` 的别名）；`MemoryClient("m0-...")` 把 Key 当第一个参数时报 ValueError，第一个参数是地址 |
| `filters` 里的 `created_at`、`updated_at`、`categories` | 相同，对记忆自己的字段；事实的别的字段（`category`、`predicate`、`status`……）是 metadata 的条件，metadata 里没有这个键时什么都不匹配 |
| `add()` 返回抽取出的事件 | 只在 `wait=True` 时；不带时 `results` 是 `[]`、`status` 是 `"queued"` |
| `update` 返回的内容 | `id` 是原来的 ID，`results` 里是新的事实；原来的 ID 之后属于被取代的版本 |
| `history` 的行是 `old_memory` / `new_memory` | 完整的事实，带 `event` `ADD` 或 `UPDATE` |
| 不存在的 ID 上 `update` / `delete` 报 ValueError（`get` 返回 `None`） | KeyError（`get` 也返回 `None`） |
| `search` 与 `get_all` 缺省给 20 条（mem0ai 2.2.1 的 `top_k=20`） | 没有缺省：全部给，除非传 `limit` 或 `top_k` |

### `AsyncMemory`

`AsyncMemory(...)` 的参数相同，每个方法都是带类型的协程，签名与 `Memory` 上的相同（在线程池里执行；测试两个方向都核对）。`AsyncMemory.from_system(system)` 包住已经打开的记忆系统。`async with AsyncMemory() as m:` 退出时关闭，`async with m.run(user_id="alice") as run_id:` 是异步的 run。别的属性（`.system`、`.settings`）交给同步对象。

### 异常与日志

`cuemem.errors` 定义了 `CueMemError` 与它的子类：`MissingAPIKeyError`、`CacheMissError`、`StructuredOutputError`、`EmbeddingAPIError`（带 `status` 与 `retry_after`）、`DatasetChecksumError`、`ConfigMismatchError`（更新的 CueMem 写的库也是它：升级，或者用新版本导出再在这里导入）、`ConsolidationError`、`ContentFilteredError`、`ConcurrentWriteError`、`IdempotencyConflictError`（`idempotency_key` 用在了另一次不同的 add 上）、`StorageLockedError`、`VectorStoreOpenError`（本地模式的 Qdrant 打不开或建不了集合，多半是打开的文件太多，或者读不出它的 `meta.json`；说明里写着办法，后者是 `cuemem doctor --repair-vectors`）、`AccessKeysError` 与 `MemoryAPIError`（带 `status`、`code`、`type`、`retry_after` 与 `sent`）。SDK 还会按上面说的抛 KeyError、ValueError、TypeError 与 RuntimeError；OpenAI 客户端库的异常在 CueMem 自己的重试之后原样抛出。

日志：`cuemem.write.consolidate`（后台抽取失败，英文）、`cuemem.core.trace`（trace 写不进去，或遗忘时只好整个删掉某天的 trace 文件）、`cuemem.memory`（遗忘的发言的图片文件删不掉，或遗忘提交之后的清理出错）、`cuemem.sdk`（高峰时段的等待）与 `cuemem.api.proxy`。

## 远程客户端：`MemoryClient`

```python
from cuemem import MemoryClient

m = MemoryClient("http://127.0.0.1:7860", api_key="...")  # 代替 Memory()
m.add("I moved to Berlin", user_id="alice", metadata={"source": "crm"}, wait=True)
m.search("Where do I live?", user_id="alice", filters={"source": "crm"})
```

```python
MemoryClient(
    base_url: str | None = None,
    *,
    api_key: str | None = None,
    host: str | None = None,
    timeout: float = 300.0,
    http_client: httpx2.Client | None = None,
)
```

- 调用 `cuemem serve` 的 `/api/v1`。除 `from_config`、`from_system` 与 `close_idle_runs` 之外，`Memory` 的每个公开方法都有，签名逐一相同（有测试核对），返回的就是服务端的 `Memory` 算出来的结果，本地与远程之间只换一行。几个进程、或别的机器上的应用可以这样共用一个服务。
- `base_url` 缺省是 `http://127.0.0.1:7860`；`host` 是它的别名（Mem0 的叫法），两个都给时 ValueError。要以 `http://` 或 `https://` 开头（否则构造时就 ValueError，报错里不带这个值：它可能是令牌）；末尾带 `/api/v1`（下面 curl 示例的写法）也认。
- `api_key` 不填时取环境变量 `CUEMEM_API_TOKEN`；可以是管理令牌，也可以是限定作用域的令牌。`timeout` 是单次请求的秒数（`wait=True` 的 `add` 要等抽取做完）；自带 `http_client` 时照它自己的超时。
- 内容块里的本地图片路径（与 `file://`）在客户端读成 data URL 再发；服务器不读自己磁盘上的文件。`media(id)` 从服务端取图片原文件。
- `metadata` 发送之前照 SDK 的规则检查（JSON 会悄悄把非字符串的键变成字符串），`limit` 与 `top_k` 也是（JSON 的 `true` 会变成 1）。
- 时间参数（`timestamp`、`at`、`known_at`、`expiration_date`）按 ISO 格式发送。不带时区的时间按服务器的本机时间理解，客户端与服务器不在一个时区时请带上时区。
- 抛出与 `Memory` 相同的异常：KeyError、ValueError、TypeError 与 `cuemem.errors` 里的类型；`EmbeddingAPIError` 带着 `status` 与 `retry_after`。别的 HTTP 出错（令牌不对、连不上服务、超出令牌作用域的 403、其他供应商出错）是 `MemoryAPIError`，带 `status`（没有拿到响应时为 None）、`code`（错误码见下）、`type`（服务端的异常类，例如 `RateLimitError`）、`retry_after`（响应的 Retry-After，429 与 503 时有）与 `sent`（请求有没有到服务端：False、True，拿到了 HTTP 响应时是 None）。没有异常类型的 404 另外说明 `base_url` 要是 `cuemem serve` 的根地址。没有拿到响应时 `code` 是客户端一侧的：`unreachable`（`sent=False`：什么都没写，可以重发；连接被拒绝时问那里是不是在运行 `cuemem serve`）、`timeout`（`sent=True`：客户端的超时之内没有响应，请求可能已经处理了）、`disconnected`（`sent=True`：连接中途断开）、`transport`（`sent=True`：别的传输出错）。客户端不自动重试：写入只在 `unreachable` 之后、或者带 `retry_after` 的 429 与 503 之后重发；`timeout` 与 `disconnected` 之后可能已经存下了。带了 `idempotency_key` 的 `add` 可以用同一个键重发（只写一次，见上面的 `add`）；没带时先查 `export`（还在排队等抽取的 add 不在 `get_all` 里）。这两种出错的说明是 "the request may have been processed (repeat it with the same idempotency_key, or check export)"。别的租户的 ID 与不存在的一样，`get()` 返回 `None`。
- `wait()` 让服务端最多等这个客户端的读超时减 5 秒（`cuemem.client.WAIT_MARGIN_S`；读超时的一半更大时取一半），不给 `timeout` 时也是，所以等不完时抛的是 TimeoutError（服务端的 504），不是 `MemoryAPIError(code="timeout")`；缺省 300 秒的客户端下服务端最多等 295 秒。
- `close()`（或 `with MemoryClient(...) as m:`）结束这个客户端开的 run。远程开了、一直没结束的 run，`cuemem serve` 在闲置 30 分钟（`--idle-minutes`）后结束它，同一个 `run_id` 的下一次 `add` 重新打开；服务重启时还开着的 run，由带它的 `run_id` 的下一次 `add` 接着写（见上面的 run）。
- 框架集成要本地的 `Memory` 或 `AsyncMemory`：传 `MemoryClient` 时在构造时就抛 TypeError。
- 这个模块只导入 httpx2 与 `cuemem._async`，不加载记忆系统。

`AsyncMemoryClient(...)` 是异步版本，带类型的协程与 `AsyncMemory` 相同。

## HTTP 接口：`cuemem serve`

`cuemem serve`（默认 `http://127.0.0.1:7860`）在同一个进程、同一个记忆系统上提供：`/api`（与 Agent 的会话、对话、记忆浏览、遗忘）、`/api/v1`（Mem0 形状的记忆接口）、`/v1`（记忆代理）、`/mcp`（Streamable HTTP 的 MCP）、`/docs`（OpenAPI 文档），装了 `ui` 时根路径 `/` 是 Gradio 界面。

| 开关 | 缺省 | 含义 |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `7860` | 监听地址 |
| `--space` | `shop-demo` | 界面默认的记忆空间 |
| `--scenario` | 源码仓库里 `shop`，pip 安装后 `plain` | 导购演示还是通用助手 |
| `--router` / `--lam` | `learned` / 控制器模型里的值 | 对话用的检索控制器 |
| `--no-ui` | 关 | 只启动接口、记忆代理与 MCP |
| `--no-mcp` | 关 | 不在 `/mcp` 提供 MCP |
| `--idle-minutes` | `30` | 对话会话、`/api/v1` 的 run 与记忆代理的会话闲置这么久后结束、交给抽取（要大于 0）；也是启动之后过多久把之前的进程留下的会话交给抽取 |
| `--insecure-no-auth` | 关 | 没有任何令牌也在非本机地址上（或在本机地址上、`CUEMEM_ALLOWED_HOSTS` 却列了别的主机名时）启动。监听非本机地址时 Host 只能是 IP 地址、localhost、监听的主机名或 `CUEMEM_ALLOWED_HOSTS` 里的名字；监听本机地址时只能是 127.0.0.1、localhost、[::1]、监听地址或 `CUEMEM_ALLOWED_HOSTS` 里的名字，与设了令牌时相同。设了令牌时只是不查令牌是否够 16 个字符 |
| `--insecure-weak-token` | 关 | 监听非本机地址、或在反向代理后面监听本机地址（`CUEMEM_ALLOWED_HOSTS`）时也允许短于 16 个字符的令牌 |
| `--unknown-model` | `proxy.unknown_model`（`remap`） | 记忆代理收到不认识的模型时：`remap` 或 `reject` |

每个 `cuemem` 命令都先把进程打开文件数的软上限调到硬上限（最多 65,536）。Qdrant 本地模式每个集合（每个记忆空间一个，有图的两个）一直开着一个文件、启动时全部打开，用户多的服务器请用 Qdrant 服务端（`cuemem migrate-vectors --to URL`，再设 `CUEMEM_VECTOR_STORE__URL`）。`cuemem doctor` 列出集合数与上限。

### 鉴权与限定作用域的令牌

- 没有配任何令牌时不鉴权，只能监听没有对外的本机地址：监听非本机地址又没有令牌时拒绝启动，除非加 `--insecure-no-auth`；监听本机地址、`CUEMEM_ALLOWED_HOSTS` 却列了不只指本机的主机名或 `*` 时（经反向代理对外；只指本机的项是 127.x、::1、localhost 与 *.localhost，可带端口或方括号）也一样。加了 `--insecure-no-auth` 又监听非本机地址时，Host 只能是 IP 地址、localhost、监听的主机名或 `CUEMEM_ALLOWED_HOSTS` 里的名字；监听本机地址时，不管有没有令牌，只能是 127.0.0.1、localhost、[::1]、监听地址或 `CUEMEM_ALLOWED_HOSTS` 里的名字（否则 400，防 DNS 重绑定；列表里有 `*` 时不查 Host）。
- 监听非本机地址、以及像上面那样在反向代理后面监听本机地址时，每个令牌至少 16 个字符（`cuemem.api.auth.MIN_TOKEN_CHARS`）：管理令牌与直接写出的 `token:` 都算，每个 Docker 容器也是。更短的让 `cuemem serve` 与 `cuemem proxy` 在打开记忆系统之前以退出码 2 退出，说明里写着是哪一项、从不打印令牌本身，除非加 `--insecure-weak-token`（compose 里加进 `command:`）。`--insecure-no-auth` 也跳过这项检查：设了令牌时服务带着这个短令牌启动，请求仍要带它；监听非本机地址时与那里任何设了令牌的监听一样不查 Host，监听本机地址时照上面查 Host。用 `openssl rand -hex 32` 生成令牌。`sha256:` 的条目量不出长度，不查。
- 服务说的是明文 HTTP。设了令牌又监听非本机地址（或 `CUEMEM_ALLOWED_HOSTS` 让本机地址对外）时，启动时提示令牌（也是界面的密码）与记忆都不加密地经过网络；可信网络之外要在前面放 TLS 反向代理，见 README 的[公网部署](../README.zh-CN.md#公网部署)。CueMem 不限制登录与令牌出错的次数，在反向代理里做。
- `CUEMEM_API_TOKEN` 是管理令牌，请求带 `Authorization: Bearer <令牌>`。`GET /api/health` 不要令牌。`/api` 与 `/api/v1` 在读请求体之前就认令牌，没认上的请求不论请求体是什么都回 401。
- 限定作用域的令牌（`CUEMEM_API_KEYS_FILE` 是 YAML 文件的路径，`CUEMEM_API_KEYS` 是同样内容的 YAML 或 JSON 文本，只能设一个）给每个令牌列出它能访问的用户与 Agent：

  ```yaml
  keys:
    - name: alice-app                 # 也是这个令牌在界面里登录的用户名
      sha256: <64 位十六进制>          # printf %s "$TOKEN" | sha256sum（或者 token: <令牌本身>）
      scopes:
        - {user: alice}               # alice 的全部记忆：不带 Agent 的，与在任何 Agent 下的
    - name: shop-backend
      sha256: <64 位十六进制>
      scopes:
        - {user: "*", agent: shop}    # 每个用户在 Agent shop 下的记忆（shop/<用户>）
        - {agent: shop}               # shop 自己的记忆（agent/shop）
    - name: alice-app-limited         # 可选的按令牌上限（四项都可以不写）
      sha256: <64 位十六进制>
      scopes: [{user: alice}]
      rate_per_minute: 60             # 每分钟 POST、PUT、DELETE 的请求数
      models: [qwen3.8-flash]         # 这个令牌经记忆代理能用的模型
      max_output_tokens: 2000         # 这个令牌经记忆代理的输出上限
      max_concurrent: 4               # 同时在处理的请求数（缺省 8）
  ```

  值是一个 ID 或 ID 的列表；`*` 是任何 ID，`acme:*` 是前缀。`{agent: A}` 只是 Agent A 自己的记忆。`agent/x` 一律算 Agent x 自己的记忆，所以限定作用域的令牌用不了名字就叫 `agent` 的 Agent。不是由 user_id、agent_id 得来的空间（评测空间、名字里有两个以上的 `/`）只有管理令牌能访问；`{user: "*"}` 包括 `shop-demo` 这类普通名字的空间。未知字段报错；启动时文件写错，`cuemem serve` 与 `cuemem proxy` 打印 `错误：…`、退出 1。令牌只按 sha256 摘要做常数时间比较，与每个令牌都比一遍。
- **改令牌不用重启**。`cuemem serve` 与 `cuemem proxy` 运行期间重新读入 `CUEMEM_API_KEYS_FILE` 的文件（最多每秒看一次；改动约一秒内生效，不用重启）。写在 `CUEMEM_API_KEYS` 里的与 `CUEMEM_API_TOKEN` 仍只在启动时读一次。下面这些时候不换，原来的令牌照旧有效，日志里记一条只写令牌名字的警告：文件不在了、读不出或写错了；没设 `CUEMEM_API_TOKEN` 时文件里一个令牌都不剩（重新读入从不关掉鉴权）；文件里有短于 16 个字符的明文令牌、而服务器在这种部署下会拒绝用它们启动。按令牌的限速按名字记，重新读入不清零。用吊销了的令牌登录的界面会话什么都不能再用；换了令牌（同一个名字换了 sha256，包括 `cuemem keys revoke NAME` 之后又 `cuemem keys add NAME`）时，用这个名字登录的界面会话全部退出；只改作用域或限额不让人退出。假定服务器只有一个进程（serve 与 proxy 都这样启动 uvicorn）；不处理 SIGHUP。
- **`cuemem keys`** 修改这个文件，不用手工算摘要：

  ```bash
  TOKEN=$(cuemem keys add alice-app --user alice)      # 新令牌只打印这一次，只打到标准输出
  cuemem keys add shop-backend --user '*' --agent shop  # 一个作用域：shop 这个 Agent 下的每个用户
  cuemem keys add shop-self --agent shop                # shop 自己的记忆
  cuemem keys add capped --user bob --rate-per-minute 60 --models qwen3.8-flash --max-output-tokens 2000 --max-concurrent 4
  cuemem keys list                                      # 名字、作用域与限额；从不显示令牌与摘要
  cuemem keys revoke alice-app                          # 运行中的 serve/proxy 约一秒内不再认它
  ```

  这几个命令改 `--file` 或 `CUEMEM_API_KEYS_FILE`（数据目录的 `.env` 里设的也算）指的文件，保留注释，只存 sha256（加引号），原子地重写、权限 0600；`keys add` 在文件不在时新建。每次 `keys add` 写一个作用域：`--user` 的值，给了 `--agent` 时连同它的值；只给 `--agent` 是这个 Agent 自己的记忆；`--user`、`--agent` 与 `--models` 都可以重复。令牌写在 `CUEMEM_API_KEYS` 里时它们拒绝（`keys list` 照样列出）。`keys:` 写成流式（`keys: [...]`）的文件不动，`keys add` 改为打印要贴进去的那一项。`keys revoke` 只删这一项的几行；这一项与别的内容写在同一行、或者它的 `-` 单独一行时拒绝，并写明要手工删哪几行。吊销最后一个令牌时写成 `keys: []`，并提醒没有 `CUEMEM_API_TOKEN` 时运行中的服务在停掉之前仍认原来的令牌。手工的做法照旧可用：`printf %s "$TOKEN" | sha256sum`，写一项 `sha256:`。
- **作用域管的是能访问谁的记忆，不是能花多少。** 任何有效的令牌都能让服务器调用运营者的模型供应商：`/v1/chat/completions` 与 `/v1/responses`（带不带 user 都行；`model`、`n`、`max_tokens` 与 `enable_thinking` 这类开关原样转发）、`/v1/embeddings`（最多 2,048 条）、`/api` 的对话、`POST /api/v1/answer`、`/mcp`，以及每一次写入（之后的抽取每写入一个字节约向上游发 2 个字符）。只把令牌给你愿意让它花这笔钱的人，在供应商那里设花费上限，在反向代理里按 `Authorization` 头限流（例如 nginx 的 `limit_req_zone $http_authorization ...`），调小 `CUEMEM_MAX_BODY_BYTES` 限住一次写入的花费。
- **可选的按令牌上限**（正整数或非空的列表；不写时什么都不变，转发的请求体逐字节相同）：
  - `rate_per_minute`：在 `/api`、`/api/v1`、`/v1` 与 `/mcp` 上每分钟最多这么多个除 GET、HEAD、OPTIONS 之外的请求（POST、PUT、DELETE）；经 POST 的检索与回答、`/v1/embeddings` 与每次 MCP 调用（Streamable HTTP 用 POST）也算，GET 请求不算。做法是容量为这个数、每秒补 `rate_per_minute / 60` 个的令牌桶。超过时 429，带 `Retry-After`（等到下一个名额的秒数）；`/api/v1` 下 code 是 `rate_limited`，`/api` 下是中文的 `{"detail"}`，`/v1` 与 `/mcp` 下是 `{"detail": "Too many requests for this access key: ..."}`。
  - `models`：记忆代理（`/v1/chat/completions`，含流式，与 `/v1/responses`）在检索、开会话与转发之前拒绝别的模型，回 OpenAI 格式的 403（type `permission_error`，param `model`，code `forbidden`）。列出的别名只放行这个别名与它在目录里的设置：`models: [deepseek-flash]` 不放行 `deepseek-flash-thinking`；列出的模型 ID 本身不是别名时，放行这个 ID 的每个别名。这种令牌的请求里目录的 `extra_body` 盖过调用方给的，便宜的模型不能借开关变成思考模式。`/v1/embeddings` 不受 `models` 管。
  - `max_output_tokens`：记忆代理把 `max_tokens` 与 `max_completion_tokens` 截到这个数，并且总按模型目录里这个模型的写法（`max_tokens_param`；除了 OpenAI 预设的模型都是 `max_tokens`）写上一份：两个都没给时是这个数，给 DeepSeek、通义的模型只发 `max_completion_tokens` 时也转发截过的 `max_tokens`；`n` 不是 1 时 400（param `n`）。只写它管不住调用方的开关（`enable_thinking`、`thinking` 原样转发），要固定它们得写 `models`。
  - `max_concurrent`：在 `/api`、`/api/v1`、`/v1` 与 `/mcp` 上同时在处理的请求（任何方法）最多这么多个。不写时是 `CUEMEM_MAX_CONCURRENT_PER_KEY`（缺省 8），见下。
  - 管不到的：界面的会话（agent 角色的模型，`max_tokens` 1200）、一次写入的大小（只由 `CUEMEM_MAX_BODY_BYTES` 管）、调用方加的供应商字段（`enable_search` 等）。按日的花费预算没有做。
- **每个令牌同时处理的请求**。配了令牌时，每个限定作用域的令牌在 `/api/`、`/api/v1`、`/v1` 与 `/mcp` 上同时在处理的请求（任何方法）最多 8 个（`CUEMEM_MAX_CONCURRENT_PER_KEY`，正整数，或者令牌自己的 `max_concurrent`；这个变量写错时 `cuemem serve` 与 `cuemem proxy` 以 1 退出）。超过时回 429，带 `Retry-After: 1`：`/api/v1` 下 code 是 `rate_limited`，`/api` 下是中文的 `{"detail": "这个访问令牌同时在处理的请求太多：最多 N 个（环境变量 CUEMEM_MAX_CONCURRENT_PER_KEY，或令牌的 max_concurrent），请稍后重试"}`，`/api/v1`、`/v1` 与 `/mcp` 下的说明是 "Too many concurrent requests for this access key: at most N at once (set with CUEMEM_MAX_CONCURRENT_PER_KEY or the key's max_concurrent); retry shortly."。管理令牌与不鉴权的演示不限。服务器用 40 个线程跑同步的接口，几个限定作用域的令牌加起来仍能占满（每个 8 个时五个就够）；很多令牌共用一台服务器时可以调低。
- 令牌在作用域之外时：请求点名了用户、Agent 或空间（包括 `filters` 里的用户、`POST /api/sessions` 的缺省空间与导入的目标空间）回 **403**；按 ID 找的记忆与会话回 **404**（history 是 `[]`），与不存在的一样。`/api/spaces` 与 `/api/v1/spaces` 只列出这个令牌的空间。限定作用域的令牌 `POST /api/v1/reset` 只删它作用域里的空间（通配的租户令牌会清空整个租户）。
- 界面先要求登录：密码是令牌。限定作用域的令牌用它的名字做用户名；管理令牌的用户名任意。经界面发送的文件必须是经界面上传的：指向服务器上别的文件的路径一律拒绝。
- `/api/`、`/v1/`、`/gradio_api/`、`/mcp`、`/login` 与 `/config` 不接受别的网站发来的浏览器请求（403，`Origin: null` 也是；同源与不带 Origin 的请求照常），带着这样的 `Origin` 请求界面首页 `/` 的也是（`/config` 与 `/` 带着登录的令牌名字；在地址栏打开、点链接跳过来时不带 `Origin`，照常）。`/logout` 没法按 `Origin` 查（`<img>` 与页面跳转不带它）：带 `Sec-Fetch-Site: cross-site` 的 `/logout` 回 403，同源的登出与不发这个头的客户端（旧浏览器、curl）照常；`CUEMEM_ALLOWED_HOSTS` 里有 `*` 时不查。监听本机地址时只接受 Host 为 127.0.0.1、localhost、[::1] 与监听地址本身的请求；加了 `--insecure-no-auth` 又监听非本机地址时，接受任何 IP 地址、localhost 与监听的主机名。别的回 400；别的主机名（反向代理的、`nas.local`、别的容器用的 compose 服务名）写进 `CUEMEM_ALLOWED_HOSTS`。设了令牌又监听非本机地址时不查 Host。
- `/mcp` 上的 MCP 用同样的令牌与作用域（见下）；stdio 的 `cuemem mcp` 不经过 HTTP，没有令牌。

### 请求的大小上限

`cuemem serve` 与 `cuemem proxy` 在读请求体之前查上限：

| 路径 | 上限 | 用什么设 |
|---|---|---|
| `/api/spaces/import`、`/api/v1/import` | 1 GiB（管理令牌或不鉴权时）；限定作用域的令牌按 `CUEMEM_MAX_BODY_BYTES` | `CUEMEM_MAX_IMPORT_BYTES` |
| `/api/…`、`/v1/…`、`/gradio_api/upload` | 64 MiB | `CUEMEM_MAX_BODY_BYTES` |
| 别的路径（界面的登录与事件接口、`/mcp`） | 8 MiB | 固定 |

- 超过时回 413（`/api/v1` 下 code 是 `payload_too_large`）；限定作用域的令牌导入时，说明里写着更大的导入要用管理令牌。两个变量都是正整数的字节数（`67108864`，不是 `64MB`）；写成别的时 serve 与 proxy 打印 `错误：…`、退出 1。
- 解析出来的 JSON 约是请求体的 24 倍，所以一个限定作用域的令牌同时在处理的请求体加起来也不超过 `CUEMEM_MAX_BODY_BYTES`：超过时 429、带 `Retry-After: 1`（`/api/v1` 下 code 是 `rate_limited`；`MemoryClient` 抛 `retry_after="1"` 的 `MemoryAPIError`，OpenAI SDK 会自己重试）。一个令牌因此最多占约 25 倍的上限（缺省约 1.6 GB）。好几个令牌一起来、界面的事件接口不按令牌算：按这个给服务器留内存，或者调小 `CUEMEM_MAX_BODY_BYTES`。管理令牌与不鉴权的部署没有这个总量。
- 像素超过 Pillow 的 `Image.MAX_IMAGE_PIXELS`（缺省 89,478,485）的图片拒绝：`/api/v1` 400，`/api` 409（对话在召回、调用模型之前先查每张上传的图），记忆代理带着这样的图片的写回只丢这一条。1,600 万像素以上的解码最多同时 2 个。
- 图片只收 JPEG、PNG、WEBP、GIF、BMP 与 TIFF（手机拍的 MPO 算 JPEG）。别的格式（AVIF、XBM、ICO……）与不是图片的文件在任何解码器运行之前就拒绝：`/api/v1` 400，message 是 `Not a valid image (accepted formats: JPEG, PNG, WEBP, GIF, BMP, TIFF).`（以前是 500），`/api` 409，SDK 报 ValueError（以前是 `PIL.UnidentifiedImageError`），带着这样附件的导入整个失败。记忆代理与 MCP 带着这样的图片的写回只丢这一条。已经存下的别的格式的图照常可用。

### `/api/v1`：记忆接口

SDK 的每个方法一个接口。请求体与查询参数的名字、缺省值与 SDK 的关键字参数相同，响应就是 `Memory` 的返回值。

| 接口 | SDK 方法 |
|---|---|
| `POST /api/v1/memories`，`{messages, user_id, agent_id, run_id, metadata, expiration_date, timestamp, infer, wait}`，可选的请求头 `Idempotency-Key` | `add`（这个头就是 `idempotency_key`：24 小时内重发回 200，`status` 是 `duplicate` 或 `forgotten`，`"replayed": true`；同一个键用于不同的请求回 409 `conflict`，type `IdempotencyConflictError`；键写错回 400） |
| `POST /api/v1/memories/search`，`{query, user_id, agent_id, run_id, limit, mode, at, router, lam, filters, top_k, min_level, show_expired}` | `search` |
| `GET /api/v1/memories?user_id&agent_id&run_id&include_superseded&valid_at&known_at&limit&top_k&show_expired&filters=<JSON>` | `get_all` |
| `GET /api/v1/memories/{id}` | `get`（不存在时 404） |
| `PUT /api/v1/memories/{id}`，`{text, metadata}`（或用 `data` 代替 `text`） | `update` |
| `GET /api/v1/memories/{id}/history` | `history` |
| `DELETE /api/v1/memories/{id}?dry_run=` | `delete` |
| `DELETE /api/v1/memories?user_id&agent_id&run_id` | `delete_all` |
| `GET /api/v1/media/{id}` | `media`：图片原文件，带它的 Content-Type（别的 ID、或超出令牌作用域时 404） |
| `POST /api/v1/reset` | `reset`（限定作用域的令牌只删它的空间，message 是 "All memories in this key's scope deleted"） |
| `POST /api/v1/context`，`{query, user_id, agent_id, at, mode, router, lam, min_level, wrap}` | `context`，返回 `ContextResult`；作用域之外 403，没有 `user_id` 也没有 `agent_id` 时 400，写错的值 422；不调用模型 |
| `POST /api/v1/answer`，`{question, user_id, agent_id, mode, at, router, lam}` | `answer` |
| `GET /api/v1/profile?user_id&agent_id` | `profile`，返回 `{"profile": ...}` |
| `GET /api/v1/status?user_id&agent_id` | `status` |
| `POST /api/v1/runs`，`{run_id, user_id, agent_id}` | `open_run`，返回 `{"run_id": ...}` |
| `POST /api/v1/runs/end`，`{run_id, user_id, agent_id, wait}` | `end_run` |
| `POST /api/v1/consolidate`，`{user_id, agent_id, rebuild}` | `consolidate` |
| `POST /api/v1/wait`，可选 `{user_id, agent_id, timeout}` | `wait`，返回 `{"ok": true}`。给了 ID 时令牌要管得着（否则 403）；不给时管理令牌等全部，限定作用域的令牌只等它能访问的空间。服务端最多等 300 秒，之后 504 `timeout`（type `TimeoutError`，`MemoryClient` 抛 TimeoutError）。等的时候不占服务器的线程：每 0.25 秒看一次，所以最多在抽取做完之后 0.25 秒返回。客户端断开时，服务端 0.25 秒内不再等，放开这个令牌的并发名额（`max_concurrent`） |
| `GET /api/v1/export?user_id&agent_id&derived=` | `export`（没有这个空间时 404） |
| `POST /api/v1/import`，`{data, user_id, agent_id, wait}` | `load` |
| `GET /api/v1/spaces` | `spaces` |

`messages` 里的图片只收 data URL：本地路径与 `file://` 回 400，服务器不读自己磁盘上的文件。

`/api/v1` 的出错都是 `{"error": {"code", "type", "message"}}` 这个形状（`V1ErrorBody`）。`type` 是 SDK 抛出的异常（`KeyError`、`ValueError`、`TypeError`、`cuemem.errors` 里的类型，或供应商异常的类名），向量接口的出错另带 `upstream_status`。`/api/v1` 下的 401、413、422、500、找不到接口、别的网站的浏览器请求与 Host 不对也是这个形状（401 的 message 是 `/api` 的中文说明）；别的路径的 500 仍是纯文本。

| 状态码 | `code` | 什么时候 |
|---|---|---|
| 400 | `invalid_request` | ValueError 或 TypeError：没给 user_id 与 agent_id、filters 写错、`limit` 小于 1、图片给了路径、太大、格式不收或不是图片，导出文件不对……；不接受的 Host（说明里写着 `CUEMEM_ALLOWED_HOSTS`） |
| 401 | `unauthorized` | 没带令牌或令牌不对 |
| 402 | `no_balance` | 模型供应商的账户余额不足 |
| 403 | `forbidden` | 超出令牌的作用域；别的网站发来的浏览器请求 |
| 404 | `not_found` | 记忆 ID 不存在、导出的空间不存在、别的租户的 ID、找不到接口 |
| 409 | `conflict` | `ConcurrentWriteError`；`IdempotencyConflictError`（`Idempotency-Key` 用在了另一次不同的 add 上） |
| 413 | `payload_too_large` | 请求体超过上限 |
| 422 | `invalid_request` 或 `content_filtered` | 请求体不合格式（type 是 `ValueError`）；供应商的内容审核拒绝了这次请求 |
| 429 | `rate_limited` | 重试之后仍被限流、限定作用域的令牌同时在处理的请求体超过上限、用完了令牌的 `rate_per_minute`，或者令牌已经有 `max_concurrent` 个请求在处理；带 `Retry-After` |
| 502 | `upstream_error` | 其余供应商出错（5xx、连不上、鉴权失败）与 CueMem 的其他出错 |
| 503 | `unavailable` | 供应商暂时不可用，带 `Retry-After` |
| 504 | `timeout` | 供应商超时；`POST /api/v1/wait` 等过了上限（type `TimeoutError`） |
| 500 | `internal_error` | 服务端出了意料之外的错（是 bug），调用栈在服务端的日志里。`type` 是 null，message 从不带异常的细节 |
| （没有） | `unreachable`、`timeout`、`disconnected`、`transport` | 没有拿到响应时 `MemoryClient` 在客户端一侧给的代码（`status` 是 None；见上面的 `sent`）。客户端的 `timeout` 与 504 的代码同名，用 `status` 区分 |

`/openapi.json`（与 `/docs`）给 `/api/v1` 的每个响应一个有名字的结构（按 `type` 区分的 `FactItem`、`ExcerptItem`、`ImageItem`，`SearchResult`，`AddResult`，`ForgetResult` 或 `DryRunResult`，`StatusResult`，`ContextResult`，`ExportDocument`……），每个出错状态都写成 `V1ErrorBody`，包括 500。21 个 operationId 就是 SDK 的方法名（`add`、`get_all`、`search`、`context`……）：按 D-072 之前的快照生成的客户端要重新生成，方法名会变。`GET /api/v1/media/{id}` 写明 `image/jpeg`、`image/png`、`image/webp`、`image/gif`、`image/bmp`、`image/tiff`、`image/mpo` 与 `application/octet-stream`。`/api/v1` 那部分的快照提交在 [`docs/openapi-v1.json`](openapi-v1.json)；用 `uv run python scripts/openapi_v1.py` 重新生成（`--check` 在它过期时退出码 1，测试会跑它）。没有手写的 TypeScript 客户端；生成类型、用它调用接口：

```bash
npx openapi-typescript docs/openapi-v1.json -o cuemem.d.ts   # 再用 openapi-fetch，带 Authorization: Bearer <令牌>
```

示例（令牌是 `CUEMEM_API_TOKEN` 或限定作用域的令牌）：

```bash
export CUEMEM=http://127.0.0.1:7860/api/v1 TOKEN=...
curl -s -X POST $CUEMEM/memories -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I moved to Berlin"}],"user_id":"alice","metadata":{"source":"crm"},"wait":true}'
curl -s -X POST $CUEMEM/memories/search -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"query":"Where do I live?","user_id":"alice","filters":{"source":"crm"},"top_k":5}'
curl -s -G $CUEMEM/memories -H "Authorization: Bearer $TOKEN" \
  --data-urlencode user_id=alice --data-urlencode 'filters={"source":"crm"}'
curl -s $CUEMEM/memories/<id> -H "Authorization: Bearer $TOKEN"
curl -s -X PUT $CUEMEM/memories/<id> -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"text":"I live in Munich"}'
curl -s $CUEMEM/memories/<id>/history -H "Authorization: Bearer $TOKEN"
curl -s "$CUEMEM/status?user_id=alice" -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE "$CUEMEM/memories/<id>?dry_run=true" -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE "$CUEMEM/memories/<id>" -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE "$CUEMEM/memories?user_id=alice" -H "Authorization: Bearer $TOKEN"
curl -s -X POST $CUEMEM/memories -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Idempotency-Key: 6f1c2b9e-3d4a-4e8b-9a51-2f7d0c8e1b34" -d '{"messages":"I moved to Berlin","user_id":"alice"}'
curl -s -H "Authorization: Bearer $CUEMEM_API_TOKEN" -X POST http://127.0.0.1:7860/api/v1/context \
  -d '{"query":"Where do I work?","user_id":"alice"}' -H 'content-type: application/json'
```

### `/api`：会话、对话与记忆浏览

这些接口用原始的空间 ID（`alice`、`shop/alice`；"/" 也可以写成 `%2F`），出错时回 `{"detail": 中文说明}`。

| 接口 | 做什么 |
|---|---|
| `GET /api/health` | `{"ok", "version", "open_sessions"}`，不要令牌 |
| `GET /api/spaces` | 记忆空间（按令牌的作用域筛） |
| `POST /api/sessions`，`{space, at}` | 与 Agent 开始一段会话（缺省空间是 `--space`） |
| `POST /api/sessions/{session_id}/messages`，表单字段 `text`、`images`（文件）、`mode`、`router`、`lam` | Agent 的一轮：召回、带引用回答、写回；返回回答与它的轨迹 |
| `POST /api/sessions/{session_id}/end?wait=` | 结束会话并交给抽取 |
| `POST /api/spaces/{space}/recall`，`{query, mode, router, lam, at}` | 只召回、不回答；控制器选档（L0 只给画像） |
| `GET /api/spaces/{space}/profile`、`/facts?current_only=`、`/entities`、`/graph`、`/media`、`/media/{media_id}`、`/sessions`、`/status` | 浏览这个空间（`/status` 只列出这个空间的后台出错） |
| `POST /api/spaces/{space}/forget`，`{session_ids, turn_ids, fact_ids, media_ids, dry_run}` | 遗忘（`dry_run` 时只预演）；删除之后重建出错时带 `warning` |
| `GET /api/spaces/{space}/export?derived=` | 导出 |
| `POST /api/spaces/import?space=`，请求体是导出的文件 | 导入（文件不对或格式坏了时 409，包括没有 `space_id`、也没给 `?space` 的） |
| `POST /api/spaces/{space}/consolidate` | 当场执行待处理的抽取 |

闲置 `--idle-minutes`（30）分钟的对话会话会结束、交给抽取；之后再给它发消息回 404 并说明原因，要开新的会话（界面自己会开，并说明原因）。

状态码：会话不存在或已结束、图片不存在、导出不存在的空间时 404；输入不对时 409（包括图片太大）；请求体超过上限时 413；上传的不是图片时 415；`mode`、`router` 写错时 422；401、403 同上；供应商出错时 429（带 `Retry-After`）、402、503（带 `Retry-After`）、504、422（内容审核）与 502（其余）。

## 记忆代理：`/v1`

`cuemem proxy` 单独运行记忆代理（默认 `http://127.0.0.1:8787/v1`，探活在 `/health`）；`cuemem serve` 把同样的接口挂在 `/v1`。任何 OpenAI SDK 指过来即可：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key="<CUEMEM_API_TOKEN、限定作用域的令牌；不鉴权时任意值>",
)
client.chat.completions.create(model="gpt-5.4-mini", messages=[...], user="alice")
```

| `cuemem proxy` 的开关 | 缺省 | 含义 |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8787` | 监听地址 |
| `--user` | 无 | 请求没有点名用户时用的用户（不填时这类请求原样转发、不碰记忆） |
| `--mode` | `auto` | 检索档位：`auto`（控制器）、`l0`、`l1`、`l2` |
| `--idle-minutes` | `30` | 会话空闲多久后结束并抽取；也是启动之后过多久把之前的进程留下的会话交给抽取 |
| `--insecure-no-auth`、`--insecure-weak-token`、`--unknown-model` | | 同 `cuemem serve` |

| 接口 | 做什么 |
|---|---|
| `POST /v1/chat/completions` | 召回，把证据放在最后一句用户消息前面，转发给供应商，写回；流式与非流式 |
| `POST /v1/responses` | Responses API 的一个子集，走同样的路：非流式、纯文字输入（字符串，或 `input_text` / `output_text` 的消息；`developer` 当作 `system`），以及 `instructions`、`max_output_tokens`、`temperature`、`top_p` 与 `user`。`metadata`、`store`、`parallel_tool_calls`、`service_tier`、`prompt_cache_key` 与 `safety_identifier` 接受但不用。流式、工具、`previous_response_id`、结构化的 text 格式、图片与 function call 的输入回 501（code `not_implemented`），带 `x-should-retry: false`；这些请用 `/v1/chat/completions` |
| `POST /v1/embeddings` | 配置里的向量模型（文档向量）：`encoding_format` 是 `float` 或 `base64`，`dimensions` 只能等于模型本来的维度，最多 2,048 条非空的文字输入（token 数组回 400）。调用方的文字不碰记忆、不进向量缓存。供应商出错时的状态码与 HTTP 接口相同（429 带 `Retry-After`、402、503、504、502） |
| `GET /v1/models` | 配置里服务端设了供应商 Key（或不需要 Key）的模型别名，与向量模型 |

- **身份**：请求头 `X-CueMem-User` 或请求体的 `user` 字段（请求头优先），可加 `X-CueMem-Agent`；都没有时用 `--user`。限定作用域的令牌要覆盖这个用户与 Agent，否则在检索、开会话与转发之前回 OpenAI 格式的 403（type `permission_error`，code `forbidden`）。没有点名用户的请求原样转发、不碰记忆。`/v1/models` 与 `/v1/embeddings` 任何有效的令牌都能用。
- **不是 ASCII 的用户或 Agent ID** 在这两个请求头里写成 RFC 8187 的 UTF-8 百分号编码，例如张三是 `X-CueMem-User: utf-8''%E5%BC%A0%E4%B8%89`；Node 的客户端（包括 Claude Code）只能这样发，curl 原样发的 UTF-8 也认，记忆代理也可以把 ID 放进请求体的 `user` 字段。`utf-8''` 写错、百分号编码的字节不是 UTF-8，或者 `utf-8''` 与非 ASCII 的值里带控制字符时 400（记忆代理是 OpenAI 格式的 `invalid_request_error`，param `user`）；原样发来、不是 UTF-8 的字节按 Latin-1 读，ASCII 的值原样。
- **花费**：作用域管的是能访问谁的记忆，不是能花多少；`model`、`n`、`max_tokens` 与 `enable_thinking` 这类开关原样转发，除非令牌写了 `models` 或 `max_output_tokens`（见[鉴权与限定作用域的令牌](#鉴权与限定作用域的令牌)）。不在令牌 `models` 里的模型回 OpenAI 格式的 403（type `permission_error`，param `model`），写了 `max_output_tokens` 时 `n` 不是 1 回 400（param `n`）。
- **会话**：`X-CueMem-Session` 指定会话。不给时，请求里没有之前的消息、或空闲超过 `--idle-minutes` 之后，开一段新的会话；并行的几段对话要各带自己的会话头。闲置的会话另由每分钟一次的清理结束，`cuemem proxy` 与 `cuemem serve` 挂着的代理都是。
- **响应头**：`X-CueMem-Level`（`L0`、`L1`、`L2`；没有用记忆时是 `none`，检索出错、照常不带记忆转发时是 `error`）、`X-CueMem-Recalled`、`X-CueMem-Reasons`、`X-CueMem-Session`，以及每个响应都有的 `X-CueMem-Model`：实际转发给上游的模型。响应体是供应商的原样。
- **模型**：`model` 是配置里的别名或模型 ID；不带 `model` 的请求用 agent 角色的模型。不认识的模型也改用 agent 角色的模型，每个模型名在日志里警告一次（`proxy.unknown_model: remap`，缺省）；设成 `reject`（配置项、`CUEMEM_PROXY__UNKNOWN_MODEL` 或 `--unknown-model reject`）时在检索之前回 400，code `model_not_found`。服务端没设供应商 Key 的目录模型（例如 `--preset openai` 下没有 `DASHSCOPE_API_KEY` 时的 `qwen3.8-flash`）在检索、开会话之前回 503，错误码 `model_not_available`（type `server_error`，param `model`，响应头 `x-should-retry: false`），说明里写着缺的环境变量与能用的模型；不认识的名字改用的 agent 模型没有 Key 时也是这样。`GET /v1/models` 只列服务端有 Key 的模型（加上向量模型），`unknown_model: reject` 的 400 列的也是这些。限定作用域的令牌请求作用域之外的用户时仍先回 403。
- **出错**：上游的出错原样返回，带上 `retry-after`、`retry-after-ms` 与 `x-should-retry`，什么都不写。连不上上游时回 OpenAI 格式的 502，超时 504。写回失败（例如向量接口暂时不可用）时留在进程内存里，这个会话下一次写回之前先写它，会话结束之前再试一次；进程退出就丢了。带着超过像素上限或格式不收的图片的写回只丢这一条。等上游回答期间（例如长的流式回答时别的调用方删除），注入的记忆被遗忘了的话，回答照样返回给调用方，但不存下；用户的话照样存下。
- 请求体必须是 JSON（`Content-Type: application/json`，否则 415），上限是 `CUEMEM_MAX_BODY_BYTES`（413）。

```bash
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash","input":"Where should I buy boots?","user":"alice"}' http://127.0.0.1:8787/v1/responses
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.7-text-embedding","input":["hello"]}' http://127.0.0.1:8787/v1/embeddings
```

## MCP Server

同样的四个工具有两种提供方式：

- **HTTP：`cuemem serve` 的 `/mcp`**（Streamable HTTP，无状态，JSON 响应），与服务共用记忆系统。令牌同 `/api`（不对时 401，带 `WWW-Authenticate: Bearer`）。`X-CueMem-User` 与 `X-CueMem-Agent` 请求头选空间，与记忆代理相同；不带时限定作用域的令牌用它自己的第一个用户，管理令牌、不鉴权与只有通配的令牌要带 `X-CueMem-User`（否则 400）。作用域之外的用户或 Agent 回 403、什么都不写；别的网站的 Origin 回 403。不是 ASCII 的 ID 在这两个请求头里写成 `utf-8''<UTF-8 的百分号编码>`（RFC 8187），例如张三是 `X-CueMem-User: utf-8''%E5%BC%A0%E4%B8%89`，Node 的客户端（包括 Claude Code）只能这样发；原样的 UTF-8 也认（别的原样字节按 Latin-1 读，ASCII 的值原样）；`utf-8''` 写错、百分号编码的字节不是 UTF-8，或者 `utf-8''` 与非 ASCII 的值里带控制字符时 400（`{"detail"}`）。请求体按固定 8 MiB 那一档，MCP SDK 自己另有 4 MiB 的上限。应用的 lifespan 还没启动时（uvicorn 总会启动它）`/mcp` 回 503。`cuemem serve --no-mcp` 关掉它。注册的例子：

  ```bash
  claude mcp add --transport http cuemem http://127.0.0.1:7860/mcp \
    --header "Authorization: Bearer $KEY" --header "X-CueMem-User: alice"
  ```

- **stdio：`cuemem mcp --space <空间> [--router learned]`**，一个进程一个空间（缺省空间 `default`）。它不经过 HTTP，限定作用域的令牌管不到它。它自己打开数据目录，所以本地模式的 Qdrant 下，`cuemem serve` 占着同一个目录时它起不来：这时用 serve 的 `/mcp`。注册：`claude mcp add cuemem -- cuemem mcp --space me`，源码仓库里 `claude mcp add cuemem -- uv --directory /path/to/cuemem run cuemem mcp --space me`。启动时出错（缺 Key、向量库打不开）写到 stderr、退出码 1。

| 工具 | 做什么 |
|---|---|
| `remember(text)` | 记一条笔记，立即可检索，后台抽取 |
| `recall(query, deep=False)` | 至少一次 L1 召回（控制器可以决定查得更深；`deep=True` 强制 L2）；返回画像、事实与对话片段，各带 ID 与日期；不调用回答模型。存下的文字（对话片段、画像）逐行以 `\| ` 开头，多行的事实用 ` / ` 连成一行，清单以 `End of recalled memories.` 结尾 |
| `forget(ids)` | 按 `recall` 给出的 ID 遗忘并重建；删除之后重建出错时照实说明 |
| `profile()` | 画像 |

模型供应商出错时回英文的 ToolError（限流、余额不足、暂时不可用、超时、内容审核），供应商给了等待时间时带 "in about N s"。

## 环境变量

任何配置项都可以写成 `CUEMEM_<段>__<字段>`（嵌套用双下划线），例如 `CUEMEM_ROLES__READER__MODEL=qwen3.8-flash`。变量也可以写在数据目录的 `.env` 里，只有 `CUEMEM_HOME`、`CUEMEM_ENV_FILE` 与 `CUEMEM_CONFIG` 例外：它们决定读哪个 `.env`、哪份配置，要在 shell 里设。

| 变量 | 含义 |
|---|---|
| `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、`OPENAI_API_KEY` | 供应商的 Key（`cuemem configure` 写进 `.env`） |
| `CUEMEM_HOME` | 数据目录 |
| `CUEMEM_ENV_FILE` | 另一个 `.env` 文件 |
| `CUEMEM_CONFIG` | 代替随包发布的默认配置的配置文件 |
| `CUEMEM_PRESET` | 预设：`openai`、`ollama`、`deepseek`、`qwen`，或 YAML 的路径 |
| `CUEMEM_PEAK_POLICY` | 评测等批量任务的高峰策略：`wait`、`warn` 或 `allow`；SDK 与交互式的命令（`ask`、`serve`、`proxy`、`mcp`）一律用 `warn`，SDK 的 `peak_policy` 参数优先于它 |
| `CUEMEM_API_TOKEN` | `cuemem serve` 与 `cuemem proxy` 的管理令牌；`MemoryClient` 缺省的 `api_key` |
| `CUEMEM_API_KEYS_FILE` / `CUEMEM_API_KEYS` | 限定作用域的令牌：YAML 文件的路径（运行中的 serve 与 proxy 约一秒内重新读入；`cuemem keys` 修改它），或同样内容的文本（启动时读一次；只能设一个） |
| `CUEMEM_MAX_CONCURRENT_PER_KEY` | 一个限定作用域的令牌在 `/api`、`/api/v1`、`/v1` 与 `/mcp` 上同时在处理的请求数上限（正整数，缺省 8；令牌的 `max_concurrent` 优先；超过时 429） |
| `CUEMEM_ALLOWED_HOSTS` | 额外接受的 Host 与 Origin 主机名（逗号分隔；`*` 表示两项都不检查）；不鉴权又监听非本机地址时用主机名访问也要写在这里。监听本机地址时列了不只指本机的名字（或 `*`），说明经反向代理对外，这时必须有至少 16 个字符的令牌 |
| `CUEMEM_MAX_BODY_BYTES` | `/api`、`/v1` 与界面上传的请求体上限（字节），也是一个限定作用域的令牌同时在处理的请求体的总量（缺省 67108864，即 64 MiB） |
| `CUEMEM_MAX_IMPORT_BYTES` | 管理令牌或不鉴权时导入的请求体上限（字节；缺省 1073741824，即 1 GiB） |
| `CUEMEM_TRACES_RETENTION_DAYS` | 按天的 trace 文件留几天（≥ 1 的整数；不设就一直留着）。`cuemem serve` 与 `cuemem proxy` 启动时与之后每天删一遍更早的文件，连同其中提问的查询向量；只用 SDK 的应用可以调 `memory.system.prune_traces(days)` |
| `CUEMEM_CONSOLIDATION__BACKGROUND_WORKERS` | 后台同时巩固几个空间（缺省 4；同一空间仍一次一个） |
| `CUEMEM_VECTOR_STORE__URL` | 用 Qdrant 服务端代替本地模式（带 `http://` 或 `https://`） |
| `CUEMEM_VECTOR_STORE__API_KEY` | Qdrant 服务端的 API Key（作为 `api-key` 请求头发出；服务端的 `QDRANT__SERVICE__API_KEY` 或 Qdrant Cloud 的 Key），`cuemem migrate-vectors` 也带上。空或不设表示没有 Key。只从环境变量或 `.env` 读，没有命令行参数。URL 是 `http://` 时 qdrant-client 每个客户端警告一次 "Api key is used with an insecure connection."：本机与 compose 的内部网络上无妨，远处的服务端与 Cloud 用 `https://`。`Memory(config={"vector_store": {"url": ..., "api_key": ...}})` 也认 |
| `CUEMEM_EMBEDDING__BACKEND` | 例如 `hash`：离线的哈希向量（只用于测试与演示） |
| `CUEMEM_PROXY__UNKNOWN_MODEL` | `remap`（缺省）或 `reject` |
| `CUEMEM_RETRIEVAL__CONTROLLER` | 换一个控制器模型文件 |
| `GRADIO_ANALYTICS_ENABLED` | 设成 `True` 才打开 Gradio 的使用统计与版本检查（缺省关；界面不联系第三方，字体 IBM Plex Sans 是 Gradio 自带的） |
| `GRADIO_TEMP_DIR` | 界面放上传与图库副本的目录；缺省时 `cuemem serve` 每次启动建一个只给本人的目录（0700），停下时删掉 |
| `CUEMEM_EVAL_RAW_SESSION_IDS`、`CUEMEM_LANGMEM_MAX_LOST_SHARE`、`CUEMEM_AMEM_K` | 只用于评测（见 D-067 与 D-065） |
| `CUEMEM_TEST_QDRANT_URL`、`CUEMEM_TEST_QDRANT_API_KEY` | 只用于测试：对这个 Qdrant 服务端跑 `tests/test_qdrant_server.py`，服务端要 Key 时带上这个 Key |

## 命令行

`cuemem --help` 列出全部命令：`configure`、`doctor`、`ingest`、`export`、`import`、`consolidate`、`reindex`、`forget`、`ask`、`search-image`、`spaces`、`usage`、`serve`、`proxy`、`chat`、`demo`、`migrate-vectors`、`mcp`、`keys`、`inspect`、`data` 与 `eval`。`cuemem <命令> --help` 说明各自的开关；帮助与提示都是中文。`cuemem --preset <名字> <命令>` 对任何命令换预设（`openai`、`ollama`、`deepseek`、`qwen`，或预设 YAML 的路径）；换预设要用新的数据目录（`CUEMEM_HOME`），因为数据目录记着它的向量模型。每个命令都先把进程打开文件数的软上限调到硬上限（最多 65,536；从不调低；macOS 上依次退到 10,240 与 4,096）；库与 SDK 不改它，本地空间很多的应用要自己调（`ulimit -n`）。`cuemem doctor` 列出本地向量库的集合数与这个上限，超过一半时标黄并写明办法。

- 预设名写错或 `CUEMEM_*` 的值不合法时，任何命令都只打印一行"错误：配置有误：…"（字段、原因与收到的值，密钥类的值不回显）并以 1 退出，不再打印调用栈；`cuemem mcp` 写到 stderr。SDK 照旧抛 ValueError 或 pydantic 的 ValidationError。
- 文件权限：命令行把 umask 收紧到至少 077（从不放宽），写的文件都是 0600、目录 0700；SDK 与 `MemorySystem.from_settings` 新建缺的数据目录时也是 0700。已经存在的目录一律不改。POSIX 上 `cuemem doctor` 还查数据目录、`.env` 与 `CUEMEM_API_KEYS_FILE` 的权限，别人能读时写出要运行的 `chmod 700` / `chmod 600`；`serve` 与 `proxy` 启动时提醒这两个密钥文件。`cuemem configure` 经一个 0600 的临时文件替换 `.env`（符号链接的 `.env` 链接留着；它所在的目录要可写）。
- 用 Qdrant 服务端时 `cuemem doctor` 写 `向量库：<url>（API Key 已设/未设）`；`cuemem migrate-vectors` 给目标带上 `CUEMEM_VECTOR_STORE__API_KEY`。
- `cuemem doctor` 还打印数据目录的剩余空间（少于 200 MiB 或没有写权限时标黄）。打不开 SQLite 数据库时（目录只读、磁盘满），自己打开它的命令（`ingest`、`import`、`export`、`consolidate`、`reindex`、`forget`、`ask`、`search-image`、`spaces`、`inspect` 与 `doctor --vectors`）只打印一行、写明数据目录，以 1 退出；`serve`、`proxy`、`chat`、`demo` 与 `mcp` 仍打印调用栈。
- `cuemem doctor --repair-vectors`：本地向量库的 `data/qdrant/meta.json` 读不出、不在了而集合还在、或者某个集合的配置不合法时，按 `collection/<名字>/storage.sqlite` 重写它。先停掉 `cuemem serve` 或 `cuemem proxy`：别的进程拿着向量库时拒绝。稠密向量的维度从存着的点读（空的集合用 `embedding.dim` 或 `multimodal_embedding.dim`），原来的文件改名为 `meta.json.broken-<时间>`。不要手动删掉 `meta.json`：Qdrant 会写一个空的，已有的集合就再也找不到了。
- `cuemem doctor --vectors`（要显式加）：逐个空间比较数据库里各种单元的个数与向量库里的点数，开了图片向量时再比较图片的行与图片的点。对不上的空间标黄并给出 `cuemem reindex --space S`，最后一行是 `cuemem reindex --all`；都对得上时打印 `向量库：N 个空间的点数与数据库一致`。本地向量库正被别的进程（serve）拿着时跳过这项并说明，doctor 的其余检查照常。
- `cuemem reindex [--space S | --all] [--full]`：按数据库补齐向量库（不调用 LLM；向量缓存是热的时候也不发请求）：没有向量的单元与图片重新向量化，删掉数据库里已经没有的点；`--full` 把每个向量都重写一遍。`--space` 与 `--all` 必须给且只给一个（否则退出码 2），空间不存在时退出码 1。每个空间打印一行：补上了哪几种、各几个，删掉的多余的点，原图不在了的图片，事实表里找不到的事实单元。本地模式下要打开向量库，先停掉 serve。在 Python 里：`MemorySystem.reindex(space_id, *, only_missing=True)`（例如 `m.system.reindex(space)`）返回一个字典：每种补上的单元一个计数（`round`、`fact`、`summary`、`entity`、`image`、`image_iso`），`images`、`orphans`、`image_orphans`、`missing_files`，有时还有 `facts_missing`；在这个空间的 `reading()` 里调用时与遗忘一样抛 RuntimeError。
- `cuemem export` 与 `cuemem spaces` 只打开 SQLite 数据库：`cuemem serve` 开着（它拿着本地向量库）、或者向量库打不开时照样能用。`cuemem export -o FILE` 先写同一目录下的临时文件、落盘后换名替换 FILE：导出失败（例如磁盘满）时旧的备份不动，只打印一行原因（退出码 1）；FILE 是符号链接时替换它指向的文件，`/dev/stdout`、FIFO 直接写。`cuemem import` 遇到截断的或不是 UTF-8 的文件时打印一行（`不是有效的 CueMem 导出文件`）并以 1 退出。`cuemem inspect facts --valid-at` 接受上面 `valid_at` 的写法，别的以 2 退出。
- `cuemem keys add|list|revoke [--file PATH]`：管理 `CUEMEM_API_KEYS_FILE` 里限定作用域的令牌（见[鉴权与限定作用域的令牌](#鉴权与限定作用域的令牌)）。`keys add NAME --user U… [--agent A…] [--rate-per-minute N] [--models M…] [--max-output-tokens N] [--max-concurrent N]` 把新令牌只打印一次、单独一行打到标准输出；`keys list` 从不显示令牌与摘要。
- 数据目录自动升级；不支持降级。D-073 起数据库里还可能有表 `add_keys`（幂等的记录），自动建，不改数据格式的版本；旧的 v7 CueMem 不管它。更新的 CueMem 写的库以 ConfigMismatchError 拒绝打开、原样不动：要换回旧版本，用新版本导出（`cuemem export`）再导入。

### 备份与恢复

- **要留的**（在数据目录里，源码仓库里是 `data/`）：`cuemem.sqlite` 连同它的 `-wal` 与 `-shm`（原始日志与派生记忆）、`ledger.sqlite`、`media/`（原图）与 `qdrant/`（或 Qdrant 服务端的存储）；`claude_memory.sqlite` 连同它的 `-wal` 与 `-shm`（只在用了 Claude memory tool 的集成时有：Claude 写的 `/memories` 文件；它不在 `cuemem export` 里，也不能从日志重建）；`cache/` 可留可不留（能重建，但要花请求）。恢复用不着 `runs/`。
- **怎么备份**：停掉 `cuemem serve` 再复制整个数据目录，或者每个 SQLite 文件（`cuemem.sqlite`、`ledger.sqlite`，有的话还有 `claude_memory.sqlite`）用 `sqlite3 FILE ".backup DEST"`（或 Python 的 `sqlite3.Connection.backup`）。从不 `cp` 一个正在用、不带 `-wal` 的 `.sqlite`：用 WAL 时最近提交的还在 `-wal` 文件里。
- **恢复**：停掉 `cuemem serve`，先放回 SQLite 文件，再放回 `media/`，`qdrant/` 可选；然后跑 `cuemem doctor --vectors` 与 `cuemem reindex --all`。放回 `.backup` 的副本之前，先删掉数据目录里留着的 `cuemem.sqlite-wal` / `-shm`、`ledger.sqlite-wal` / `-shm` 与 `claude_memory.sqlite-wal` / `-shm`：serve 崩溃时会留下，SQLite 会把那个不相干的 WAL 重放到放回的文件上。停机时复制的整个目录连同它自己的 `-wal` 与 `-shm` 一起放回。SQLite 与 Qdrant 不是同一时刻的备份、或者向量库丢了（检索什么都没有、`get_all` 仍列得出）时，都这样补。按空间的 `cuemem export` 是可以带走的另一种做法，它带着原始日志与原图，不带向量，也不带 Claude memory tool 的文件。
