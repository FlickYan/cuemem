# 接口参考

[English](en/api.md) · 中文

程序使用 CueMem 的各种方式：Python SDK（`Memory`、`AsyncMemory`）、远程客户端（`MemoryClient`、`AsyncMemoryClient`）、`cuemem serve` 的 HTTP 接口（`/api` 与 Mem0 形状的 `/api/v1`）、OpenAI 兼容的记忆代理（`/v1`）、MCP Server、环境变量与可选依赖。签名以 CueMem 0.1.0 为准（`src/cuemem/sdk.py`、`client.py`、`api/`）。背后的决定见[决策记录](decisions.md)：D-061（SDK）、D-062（记忆代理）、D-063（导出）、D-056（部署与令牌）、D-069（第二轮体检，`/api/v1` 的大部分、`MemoryClient`、限定作用域的令牌、metadata、`update` 与 `reset` 都是这一轮加的）。

语言：SDK 的返回值与它自己的报错、`/api/v1`、记忆代理与 MCP 工具是英文；命令行、较早的 `/api` 接口、401 的说明，以及核心模块更深处抛出的报错（缺 Key、向量库被占用、配置写错）是中文。

## 安装

| 需要什么 | 安装 |
|---|---|
| SDK、命令行、HTTP 接口、记忆代理、MCP | `pip install "cuemem @ git+https://github.com/FlickYan/cuemem"` |
| + 本机向量，不要 Key（fastembed；`deepseek` 预设要用） | 可选依赖 `local` |
| + `cuemem serve` 的浏览器界面（Gradio） | 可选依赖 `ui` |
| + 评测的图，以及控制器与门控的训练脚本（matplotlib、scikit-learn） | 可选依赖 `eval` |
| + 框架集成 | 可选依赖 `openai-agents`、`langgraph`、`anthropic`，三个都要时 `integrations` |

可选依赖可以叠加：`pip install "cuemem[ui,local] @ git+https://github.com/FlickYan/cuemem"`。需要 Python 3.12 或更新的版本。源码仓库里 `uv sync` 经开发依赖组装上 `ui` 与 `eval`；`uv sync --no-dev` 时要加 `--extra ui --extra eval`。

- 没装 `ui` 时，`cuemem serve` 只提供 `/api`、`/api/v1`、`/docs` 与 `/v1`，并说明怎么装界面（`pip install "cuemem[ui]"`，源码仓库里是 `uv sync --extra ui`）。
- 没装 `eval` 时，`cuemem eval run`（`--build-only` 除外）与 `cuemem eval report` 给出同样的安装提示、退出码 1；`eval split`、`eval probe`、`eval audit` 与 `data download` 照常。
- Docker 镜像缺省装 `ui` 与 `local`；`docker build --build-arg EXTRAS= .` 构建只有接口的镜像。

## 记忆空间与身份

`user_id` 与 `agent_id` 合起来定一个记忆空间，各空间互不相通：

| 参数 | 空间 |
|---|---|
| `user_id="alice"` | `alice` |
| `agent_id="shop"` | `agent/shop`（这个 Agent 自己的记忆） |
| `user_id="alice", agent_id="shop"` | `shop/alice`（Agent shop 记得的关于 alice 的事） |

- ID 里的 "/" 与 "%" 先做百分号转义，所以 `user_id="agent/x"` 与 `agent_id="x"` 不会落进同一个空间。两个 ID 至少给一个（否则 ValueError）。`cuemem.sdk.space_for(user_id, agent_id)` 算出空间，`space_owner(space)` 是它的逆。
- `run_id` 是空间里的一个会话（一段对话）。不给时每次 `add` 自成一个会话。
- 记忆 ID（事实的 `id`）本身属于某一个空间，所以 `get`、`history`、`update`、`delete` 只要 ID。

## Python SDK：`Memory`

```python
from cuemem import Memory

m = Memory(preset="openai")          # Memory() 用默认配置，Memory(preset="ollama") 用本机模型
m.add("I moved to Munich last month.", user_id="alice", wait=True)
m.search("Where do I live now?", user_id="alice")
m.close()                            # 或者：with Memory() as m: ...
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
| `background` | 在后台线程里抽取事实（默认开） |
| `peak_policy` | 调用落在 DeepSeek 工作日高峰时段时怎么办：`warn`（SDK 的缺省：按高峰价照常调用并记下来）、`wait`（暂停到高峰结束，在 `cuemem.sdk` 日志上记一条警告）、`allow`（照常调用、不提醒） |
| `http_client` | 调用模型与向量供应商用的 httpx2 客户端（测试用它换上假的网络层） |

`from_system` 包住进程里已经打开的记忆系统（HTTP 服务与记忆代理就这样用）。本地模式的 Qdrant 同一时间只允许一个进程打开数据目录，`cuemem serve` 正占着的目录再 `Memory()` 会报 `StorageLockedError`；这时用 `MemoryClient` 连服务。

### 方法

| 方法 | 返回 |
|---|---|
| `add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, expiration_date=None, timestamp=None, infer=True, wait=False)` | `{"results": [事件], "session_id", "turn_ids", "status", ...}` |
| `search(query, *, user_id=None, agent_id=None, run_id=None, limit=None, mode="auto", at=None, router="learned", lam=None, filters=None, top_k=None, min_level="l1")` | `{"results", "images", "profile", "level", "see", "reasons"}` |
| `answer(question, *, user_id=None, agent_id=None, mode="auto", at=None, router="learned", lam=None)` | `{"answer", "evidence_sufficient", "citations", "level", "reasons"}` |
| `get_all(*, user_id=None, agent_id=None, run_id=None, include_superseded=False, valid_at=None, known_at=None, limit=None, filters=None, top_k=None)` | `{"results": [事实]}` |
| `get(memory_id)` | 一条事实，或 `None` |
| `history(memory_id)` | 版本链，从最早到最新，每个版本带 `"event": "ADD"` 或 `"UPDATE"` |
| `update(memory_id, data, *, metadata=None)` | `{"message", "id", "results", "session_id", "turn_ids", ...}` |
| `delete(memory_id, *, dry_run=False)` | 遗忘报告 |
| `delete_all(*, user_id=None, agent_id=None, run_id=None)` | 遗忘报告 |
| `reset()` | `{"message", "spaces_deleted", "turns_deleted", "images_deleted"}` |
| `profile(*, user_id=None, agent_id=None)` | 画像的文字 |
| `open_run(run_id=None, *, user_id=None, agent_id=None)` | run 的 ID |
| `end_run(run_id, *, user_id=None, agent_id=None, wait=False)` | `{"session_id", "turns", "consolidation"}` |
| `run(run_id=None, *, user_id=None, agent_id=None, wait=False)` | 上下文管理器，给出 run 的 ID |
| `consolidate(*, user_id=None, agent_id=None, rebuild=False)` | 巩固报告 |
| `wait()` | `None`：等后台线程把已登记的抽取做完 |
| `export(*, user_id=None, agent_id=None, derived=True)` | `cuemem.space` 格式的导出（能直接写成 JSON 的字典） |
| `load(data, *, user_id=None, agent_id=None, wait=False)` | 导入报告 |
| `spaces()` | 每个空间一个字典：`space_id`、`created_at`、`sessions`、`turns`、`units`、`current_facts`、`facts`、`entities` |
| `close()` | `None`：结束这个对象开着的 run（登记它们的抽取），关闭记忆系统 |

**`add`**。`messages` 是一句话、一条 OpenAI 格式的消息或消息的列表；只存用户与助手的消息。图片放在内容块里（`{"type": "image_url", "image_url": {"url": ...}}`），可以是 data URL 或本地路径；http(s) 的图片地址不下载（ValueError）。发言在线写入，不调用 LLM，立即可检索；事实在后台抽取。

- `run_id`：写进这个会话。每轮 `add` 一次同一个 `run_id` 时，每次都把整段会话重新抽取一遍（还要重放派生记忆）；一段多轮对话用 `with m.run(...)` 包起来，结束时只抽取一次。
- `metadata`：JSON 对象，键是字符串，记在这次写入的每句发言上。它不给抽取模型看，请求不变。由这些发言得来的事实、对话片段与图片都带上它，`filters` 可以按它筛。不是字典时 TypeError；键不是字符串、值写不成 JSON 时 ValueError。
- `expiration_date`：日期、时间或 ISO 字符串；只给日期时到那天结束。过期的记忆不再出现在 `search`、`get_all` 与画像里；`get` 与 `history` 照样取得到，要 `delete` 才删。同一件事后来又说过一次、没给过期时间，它就不过期。过期只在 SDK 读取时生效（`/api/v1` 与 `MemoryClient` 也一样）；`answer()`、记忆代理、MCP、`/api` 与各框架集成在删除之前仍看得到过期的记忆。
- `timestamp`：这些话是什么时候说的（时间、ISO 字符串或 Unix 时间戳；默认现在）。带时区的时间换算成本机时间。
- `infer=False`：只写日志（`status: "logged"`），之后用 `consolidate()` 抽取。
- `wait=True`：同步抽取，返回 Mem0 风格的事件（`status: "consolidated"`，另有 `consolidation`）。run 还开着时不能用（ValueError：用 `end_run(..., wait=True)` 结束它）。发言存下之后抽取出错（例如模型供应商余额不足）时，`add` 不抛出：返回 `status: "queued"` 与 `error`；发言已经存下，下次巩固时补上抽取，不要再调用一遍。
- 写进去之前出错（向量接口不可用、图片不对）时照常抛出，什么都没存，可以原样重试。
- `status` 是 `empty`（没有可存的消息）、`logged`、`queued`、`consolidated` 之一。并发写入撞了发言序号时抛 `ConcurrentWriteError`。

事件是一条带 `"event": "ADD"` 或 `"UPDATE"` 的事实；UPDATE 另有 `previous_memory`（被取代的陈述）与 `reason`（`changed`：现实变了；`corrected`：之前说错了）。

**`search`**。检索控制器决定想多深；`level`（`L0`、`L1`、`L2`）、`see`（控制器是否决定看原图）与 `reasons`（英文）说明它选了什么、为什么。

- `min_level="l1"`（缺省）：调用 `search` 就是已经决定要查记忆，控制器选了 L0 时改走 L1，reasons 是 `["raised from L0: the caller asked to recall"]`，与 MCP、各框架集成相同。`min_level="l0"` 时照控制器的决定，L0 只返回画像。
- `mode`：`auto`（控制器），或固定的 `l0`、`l1`、`l2`。`router`：`learned`（缺省）、`hybrid`（门控拿不准时问 LLM）、`rule`、`llm`。`lam`：成本旋钮 λ（缺省取控制器模型里的值；越小越舍得花钱）。
- `at`：提问的时间，相对日期按它换算。
- `limit` 或 `top_k`：截断 `results`。`run_id`：只要这个会话的记忆。
- `filters`：Mem0 v3 的写法。`user_id`、`agent_id`、`run_id` 可以写在这里代替关键字参数（`user_id`、`agent_id` 只能是一个 ID）。其余的键都是 metadata 的条件：
  - `{"键": 值}` 是等于，`{"键": "*"}` 是有这个键，几个键并列时都要满足；
  - `{"键": {"eq" | "ne" | "gt" | "gte" | "lt" | "lte": v, "in" | "nin": [...], "contains" | "icontains": v}}`；没有这个键时只有 `ne` 与 `nin` 成立；
  - `{"AND": [条件, ...]}`、`{"OR": [...]}`、`{"NOT": [...]}`（NOT：一个都不满足）；
  - 条件里的 `user_id`、`agent_id`、`run_id` 指记忆所在的空间与会话，例如 `{"run_id": {"in": ["r1", "r2"]}}`；
  - 不认识的运算报 ValueError。
- 检索照常在整个空间里做，再按会话、metadata 与过期时间筛，所以结果可能少于 `limit`。画像是整个空间的，去掉过期的事实。

结果里每一项带 `type`：

| `type` | 字段 |
|---|---|
| `fact` | `id`、`memory`（陈述）、`subject`、`predicate`、`object`、`category`、`importance`、`status`、`valid_from`、`valid_to`、`created_at`、`invalidated_at`、`superseded_by`、`session_id`、`source_turn_ids`、`metadata`、`expiration_date`；有时还有 `score`、`user_id`、`agent_id` 或 `space_id` |
| `excerpt` | `id`、`memory`（那一轮对话）、`score`、`created_at`、`session_id`、`source_turn_ids`、`metadata`、`expiration_date` |
| `image`（在 `images` 里） | `id`、`memory`（图片描述）、`path`、`created_at`、`attached`、`metadata`、`expiration_date` |

`metadata` 是出处各句 metadata 的合并（同一个键取后面那句的）；`expiration_date` 是出处里最晚的过期时间，有一句没给就是 `None`。

**`answer`**。检索，再用配置的回答模型回答。`mode` 另外可以是 `agent`（回答模型自己决定什么时候调用 `memory_search` 工具）与 `hatch`（控制器先召回，回答模型还可以补查）。`citations` 是 `{"ref", "id", "memory"}`。

**`get_all`**。缺省只给当前有效的事实。`include_superseded=True` 加上被取代的版本；`valid_at="2025-06"` 回放那时现实中成立的；`known_at=<时间>` 回放系统那一刻相信的。`run_id` 与 `filters` 同 `search`；过期的事实不给。

**`update`**。事实不就地改写：派生记忆是原始日志的函数。`update` 在一段新的 `edit-…` 会话里记一句用户说的 `"Correction: <data>"`（排在原来那条之后），再巩固；由它抽出的事实以 `corrected` 取代原来那条，不问 LLM。新事实继承原来那条的 metadata（`metadata` 里的键覆盖）与过期时间；`history()` 里看得到前后两个版本，遗忘这句更正时原来那条恢复。

- `message` 是 `"Memory updated"`。
- 从 `data` 里抽不出事实时，原来那条不变，这句更正也从日志里遗忘，`turn_ids` 是空的，`message` 写明。
- 更正记下之后抽取出错时，`message` 写明、`error` 给出原因；不要再调用一遍。
- ID 不存在时 KeyError；这条已经被取代（请改当前的版本）或 `data` 是空的时 ValueError。

**`delete` 与 `delete_all`**。遗忘对应"被遗忘权"。`delete` 删掉事实出自的那一轮对话，沿引用找到之后引用过它的回复，并找出没有引用的转述，再用剩下的日志重建派生记忆：被它取代过的旧事实重新成为当前的。`dry_run=True` 只列出会删什么，什么都不删。`delete_all` 删掉整个空间（日志、派生记忆、向量、缓存与原图）；给了 `run_id` 时只删这个会话。报告里有 `message`、`turns_deleted`、`sessions_affected`、`facts_before`、`facts_after`、`images_deleted`、`restated`；删除做完、用剩下的日志重建出错时另有 `warning`（下次巩固时补上）。ID 不存在时 KeyError。

**`reset`**。删掉数据目录里的每个空间，包括 HTTP 服务、记忆代理与演示写的空间，并结束开着的 run。不可恢复。

**run**。`open_run` 开始一段对话：之后 `add(run_id=...)` 照常在线写入、立即可检索，但到 `end_run` 才抽取，那时整段交给抽取（`wait=True` 当场抽取并返回报告，否则 `consolidation` 是 `"queued"`）。`with m.run(user_id="alice") as run_id:` 两步都做。崩溃的进程留下的 run，由下一个进程第一次碰到这个空间时交给抽取。

**导出与导入**。`export` 返回 `cuemem.space` 格式的版本 2：会话、发言（有 `metadata`、`expires_at`、`edit_of` 时带上）与 base64 的原图，另可附上事实与画像的快照（导入时不用）。`load` 把原始日志写回去，再重新抽取派生记忆（`wait=True` 时当场做）。不给 `user_id`、`agent_id` 时导入到文件里原来的空间，这时文件里没有 `space_id` 就报 ValueError。版本 1 的文件照样能导入；文件里有任何一张图或一处 metadata 不对时，什么都不写。

### `AsyncMemory`

`AsyncMemory(...)` 的参数相同，每个方法都可以 await（在线程池里执行）。`async with AsyncMemory() as m:` 退出时关闭，`async with m.run(user_id="alice") as run_id:` 是异步的 run。

### 异常

`cuemem.errors` 定义了 `CueMemError` 与它的子类：`MissingAPIKeyError`、`CacheMissError`、`StructuredOutputError`、`EmbeddingAPIError`（带 `status` 与 `retry_after`）、`DatasetChecksumError`、`ConfigMismatchError`、`ConsolidationError`、`ContentFilteredError`、`ConcurrentWriteError`、`StorageLockedError`、`AccessKeysError` 与 `MemoryAPIError`（带 `status` 与 `code`）。SDK 还会按上面说的抛 KeyError、ValueError 与 TypeError；OpenAI 客户端库的异常在 CueMem 自己的重试之后原样抛出。

## 远程客户端：`MemoryClient`

```python
from cuemem import MemoryClient

m = MemoryClient("http://127.0.0.1:7860", api_key="...")   # 代替 Memory()
m.add("I moved to Berlin", user_id="alice", metadata={"source": "crm"}, wait=True)
m.search("Where do I live?", user_id="alice", filters={"source": "crm"})
```

```python
MemoryClient(
    base_url: str = "http://127.0.0.1:7860",
    *,
    api_key: str | None = None,
    timeout: float = 300.0,
    http_client: httpx2.Client | None = None,
)
```

- 调用 `cuemem serve` 的 `/api/v1`。除 `from_config`、`from_system` 之外，`Memory` 的每个公开方法都有，签名逐一相同（有测试核对），返回的就是服务端的 `Memory` 算出来的结果，本地与远程之间只换一行。几个进程、或别的机器上的应用可以这样共用一个服务。
- `api_key` 不填时取环境变量 `CUEMEM_API_TOKEN`；可以是管理令牌，也可以是限定作用域的令牌。`timeout` 是单次请求的秒数（`wait=True` 的 `add` 要等抽取做完）；自带 `http_client` 时照它自己的超时。
- 内容块里的本地图片路径（与 `file://`）在客户端读成 data URL 再发；服务器不读自己磁盘上的文件。
- `metadata` 发送之前照 SDK 的规则检查（JSON 会悄悄把非字符串的键变成字符串）。
- 时间参数（`timestamp`、`at`、`known_at`、`expiration_date`）按 ISO 格式发送。不带时区的时间按服务器的本机时间理解，客户端与服务器不在一个时区时请带上时区。
- 抛出与 `Memory` 相同的异常：KeyError、ValueError、TypeError 与 `cuemem.errors` 里的类型；`EmbeddingAPIError` 带着 `status` 与 `retry_after`。别的 HTTP 出错（令牌不对、连不上服务、超出令牌作用域的 403、其他供应商出错）是 `MemoryAPIError`，带 `status`（连不上时为 None）与 `code`（错误码见下）。别的租户的 ID 与不存在的一样，`get()` 返回 `None`。
- `close()`（或 `with MemoryClient(...) as m:`）结束这个客户端开的 run。远程开了、一直没结束的 run，在服务端开着，直到服务重启。
- 这个模块只导入 httpx2，不加载记忆系统。

`AsyncMemoryClient(...)` 是异步版本，与 `AsyncMemory` 对应。

## HTTP 接口：`cuemem serve`

`cuemem serve`（默认 `http://127.0.0.1:7860`）在同一个进程、同一个记忆系统上提供：`/api`（与 Agent 的会话、对话、记忆浏览、遗忘）、`/api/v1`（Mem0 形状的记忆接口）、`/v1`（记忆代理）、`/docs`（OpenAPI 文档），装了 `ui` 时根路径 `/` 是 Gradio 界面。

| 开关 | 缺省 | 含义 |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `7860` | 监听地址 |
| `--space` | `shop-demo` | 界面默认的记忆空间 |
| `--scenario` | 源码仓库里 `shop`，pip 安装后 `plain` | 导购演示还是通用助手 |
| `--router` / `--lam` | `learned` / 控制器模型里的值 | 对话用的检索控制器 |
| `--no-ui` | 关 | 只启动接口与记忆代理 |
| `--insecure-no-auth` | 关 | 没有任何令牌也在非本机地址上启动 |
| `--unknown-model` | `proxy.unknown_model`（`remap`） | 记忆代理收到不认识的模型时：`remap` 或 `reject` |

### 鉴权与限定作用域的令牌

- 没有配任何令牌时不鉴权，只能监听本机地址：监听非本机地址又没有令牌时拒绝启动，除非加 `--insecure-no-auth`。
- `CUEMEM_API_TOKEN` 是管理令牌，请求带 `Authorization: Bearer <令牌>`。`GET /api/health` 不要令牌。
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
  ```

  值是一个 ID 或 ID 的列表；`*` 是任何 ID，`acme:*` 是前缀。`{agent: A}` 只是 Agent A 自己的记忆。`agent/x` 一律算 Agent x 自己的记忆，所以限定作用域的令牌用不了名字就叫 `agent` 的 Agent。不是由 user_id、agent_id 得来的空间（评测空间、名字里有两个以上的 `/`）只有管理令牌能访问；`{user: "*"}` 包括 `shop-demo` 这类普通名字的空间。未知字段报错。令牌在启动时读一次；文件写错时 `cuemem serve` 与 `cuemem proxy` 打印 `错误：…`、退出 1。令牌只按 sha256 摘要做常数时间比较，与每个令牌都比一遍。
- 令牌在作用域之外时：请求点名了用户、Agent 或空间（包括 `filters` 里的用户、`POST /api/sessions` 的缺省空间与导入的目标空间）回 **403**；按 ID 找的记忆与会话回 **404**（history 是 `[]`），与不存在的一样。`/api/spaces` 与 `/api/v1/spaces` 只列出这个令牌的空间，`POST /api/v1/reset` 只有管理令牌能用。
- 界面先要求登录：密码是令牌。限定作用域的令牌用它的名字做用户名；管理令牌的用户名任意。
- `/api`、`/v1` 与 `/gradio_api` 不接受别的网站发来的浏览器请求（403）。监听本机地址时只接受 Host 为 127.0.0.1、localhost、[::1] 与监听地址本身的请求；经反向代理或别的主机名访问时，把主机名写进 `CUEMEM_ALLOWED_HOSTS`。
- MCP 不在这些令牌后面（见下）。

### `/api/v1`：记忆接口

SDK 的每个方法一个接口。请求体与查询参数的名字、缺省值与 SDK 的关键字参数相同，响应就是 `Memory` 的返回值。

| 接口 | SDK 方法 |
|---|---|
| `POST /api/v1/memories`，`{messages, user_id, agent_id, run_id, metadata, expiration_date, timestamp, infer, wait}` | `add` |
| `POST /api/v1/memories/search`，`{query, user_id, agent_id, run_id, limit, mode, at, router, lam, filters, top_k, min_level}` | `search` |
| `GET /api/v1/memories?user_id&agent_id&run_id&include_superseded&valid_at&known_at&limit&top_k&filters=<JSON>` | `get_all` |
| `GET /api/v1/memories/{id}` | `get`（不存在时 404） |
| `PUT /api/v1/memories/{id}`，`{data, metadata}` | `update` |
| `GET /api/v1/memories/{id}/history` | `history` |
| `DELETE /api/v1/memories/{id}?dry_run=` | `delete` |
| `DELETE /api/v1/memories?user_id&agent_id&run_id` | `delete_all` |
| `POST /api/v1/reset` | `reset`（只有管理令牌） |
| `POST /api/v1/answer`，`{question, user_id, agent_id, mode, at, router, lam}` | `answer` |
| `GET /api/v1/profile?user_id&agent_id` | `profile`，返回 `{"profile": ...}` |
| `POST /api/v1/runs`，`{run_id, user_id, agent_id}` | `open_run`，返回 `{"run_id": ...}` |
| `POST /api/v1/runs/end`，`{run_id, user_id, agent_id, wait}` | `end_run` |
| `POST /api/v1/consolidate`，`{user_id, agent_id, rebuild}` | `consolidate` |
| `POST /api/v1/wait` | `wait`，返回 `{"ok": true}` |
| `GET /api/v1/export?user_id&agent_id&derived=` | `export`（没有这个空间时 404） |
| `POST /api/v1/import`，`{data, user_id, agent_id, wait}` | `load` |
| `GET /api/v1/spaces` | `spaces` |

`messages` 里的图片只收 data URL：本地路径与 `file://` 回 400，服务器不读自己磁盘上的文件。

出错的形状是 `{"error": {"code", "type", "message"}}`。`type` 是 SDK 抛出的异常（`KeyError`、`ValueError`、`TypeError`、`cuemem.errors` 里的类型，或供应商异常的类名），向量接口的出错另带 `upstream_status`。`/api/v1` 下的 401、422 与找不到接口也是这个形状（401 的 message 是 `/api` 的中文说明）。

| 状态码 | `code` | 什么时候 |
|---|---|---|
| 400 | `invalid_request` | ValueError 或 TypeError：没给 user_id 与 agent_id、filters 写错、图片给了路径、导出文件不对…… |
| 401 | `unauthorized` | 没带令牌或令牌不对 |
| 402 | `no_balance` | 模型供应商的账户余额不足 |
| 403 | `forbidden` | 超出令牌的作用域；不是管理令牌却调用 `reset` |
| 404 | `not_found` | 记忆 ID 不存在、导出的空间不存在、别的租户的 ID、找不到接口 |
| 409 | `conflict` | `ConcurrentWriteError` |
| 422 | `invalid_request` 或 `content_filtered` | 请求体不合格式（type 是 `ValueError`）；供应商的内容审核拒绝了这次请求 |
| 429 | `rate_limited` | 重试之后仍被限流，带 `Retry-After` |
| 502 | `upstream_error` | 其余供应商出错（5xx、连不上、鉴权失败）与 CueMem 的其他出错 |
| 503 | `unavailable` | 供应商暂时不可用，带 `Retry-After` |
| 504 | `timeout` | 供应商超时 |

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
  -d '{"data":"I live in Munich"}'
curl -s $CUEMEM/memories/<id>/history -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE "$CUEMEM/memories/<id>?dry_run=true" -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE "$CUEMEM/memories/<id>" -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE "$CUEMEM/memories?user_id=alice" -H "Authorization: Bearer $TOKEN"
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
| `GET /api/spaces/{space}/profile`、`/facts?current_only=`、`/entities`、`/graph`、`/media`、`/media/{media_id}`、`/sessions`、`/status` | 浏览这个空间 |
| `POST /api/spaces/{space}/forget`，`{session_ids, turn_ids, fact_ids, media_ids, dry_run}` | 遗忘（`dry_run` 时只预演）；删除之后重建出错时带 `warning` |
| `GET /api/spaces/{space}/export?derived=` | 导出 |
| `POST /api/spaces/import?space=`，请求体是导出的文件 | 导入（文件不对时 409，包括没有 `space_id`、也没给 `?space` 的） |
| `POST /api/spaces/{space}/consolidate` | 当场执行待处理的抽取 |

状态码：会话或图片不存在、导出不存在的空间时 404；输入不对时 409；上传的不是图片时 415；`mode`、`router` 写错时 422；401、403 同上；供应商出错时 429（带 `Retry-After`）、402、503（带 `Retry-After`）、504、422（内容审核）与 502（其余）。

## 记忆代理：`/v1`

`cuemem proxy` 单独运行记忆代理（默认 `http://127.0.0.1:8787/v1`，探活在 `/health`）；`cuemem serve` 把同样的接口挂在 `/v1`。任何 OpenAI SDK 指过来即可：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="<CUEMEM_API_TOKEN、限定作用域的令牌；不鉴权时任意值>")
client.chat.completions.create(model="gpt-5.4-mini", messages=[...], user="alice")
```

| `cuemem proxy` 的开关 | 缺省 | 含义 |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8787` | 监听地址 |
| `--user` | 无 | 请求没有点名用户时用的用户（不填时这类请求原样转发、不碰记忆） |
| `--mode` | `auto` | 检索档位：`auto`（控制器）、`l0`、`l1`、`l2` |
| `--idle-minutes` | `30` | 会话空闲多久后结束并抽取 |
| `--insecure-no-auth`、`--unknown-model` | | 同 `cuemem serve` |

| 接口 | 做什么 |
|---|---|
| `POST /v1/chat/completions` | 召回，把证据放在最后一句用户消息前面，转发给供应商，写回；流式与非流式 |
| `POST /v1/responses` | Responses API 的一个子集，走同样的路：非流式、纯文字输入（字符串，或 `input_text` / `output_text` 的消息；`developer` 当作 `system`），以及 `instructions`、`max_output_tokens`、`temperature`、`top_p` 与 `user`。`metadata`、`store`、`parallel_tool_calls`、`service_tier`、`prompt_cache_key` 与 `safety_identifier` 接受但不用。流式、工具、`previous_response_id`、结构化的 text 格式、图片与 function call 的输入回 501（code `not_implemented`），带 `x-should-retry: false`；这些请用 `/v1/chat/completions` |
| `POST /v1/embeddings` | 配置里的向量模型（文档向量）：`encoding_format` 是 `float` 或 `base64`，`dimensions` 只能等于模型本来的维度，最多 2,048 条非空的文字输入（token 数组回 400）。调用方的文字不碰记忆、不进向量缓存。供应商出错时的状态码与 HTTP 接口相同（429 带 `Retry-After`、402、503、504、502） |
| `GET /v1/models` | 配置里的模型别名与向量模型 |

- **身份**：请求头 `X-CueMem-User` 或请求体的 `user` 字段（请求头优先），可加 `X-CueMem-Agent`；都没有时用 `--user`。限定作用域的令牌要覆盖这个用户与 Agent，否则在检索、开会话与转发之前回 OpenAI 格式的 403（type `permission_error`，code `forbidden`）。没有点名用户的请求原样转发、不碰记忆。`/v1/models` 与 `/v1/embeddings` 任何有效的令牌都能用。
- **会话**：`X-CueMem-Session` 指定会话。不给时，请求里没有之前的消息、或空闲超过 `--idle-minutes` 之后，开一段新的会话；并行的几段对话要各带自己的会话头。
- **响应头**：`X-CueMem-Level`（`L0`、`L1`、`L2`；没有用记忆时是 `none`，检索出错、照常不带记忆转发时是 `error`）、`X-CueMem-Recalled`、`X-CueMem-Reasons`、`X-CueMem-Session`，以及每个响应都有的 `X-CueMem-Model`：实际转发给上游的模型。响应体是供应商的原样。
- **模型**：`model` 是配置里的别名或模型 ID；不带 `model` 的请求用 agent 角色的模型。不认识的模型也改用 agent 角色的模型，每个模型名在日志里警告一次（`proxy.unknown_model: remap`，缺省）；设成 `reject`（配置项、`CUEMEM_PROXY__UNKNOWN_MODEL` 或 `--unknown-model reject`）时在检索之前回 400，code `model_not_found`。
- **出错**：上游的出错原样返回，带上 `retry-after`、`retry-after-ms` 与 `x-should-retry`，什么都不写。连不上上游时回 OpenAI 格式的 502，超时 504。写回失败（例如向量接口暂时不可用）时留在进程内存里，这个会话下一次写回之前先写它，会话结束之前再试一次；进程退出就丢了。
- 请求体必须是 JSON（`Content-Type: application/json`，否则 415）。

```bash
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash","input":"Where should I buy boots?","user":"alice"}' http://127.0.0.1:8787/v1/responses
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.7-text-embedding","input":["hello"]}' http://127.0.0.1:8787/v1/embeddings
```

## MCP Server

`cuemem mcp --space <空间> [--router learned]` 经 stdio 提供记忆工具，一个进程一个空间（缺省空间 `default`）。它不经过 HTTP，限定作用域的令牌管不到它。它自己打开数据目录，所以本地模式的 Qdrant 下，`cuemem serve` 占着同一个目录时它起不来。注册的例子：`claude mcp add cuemem -- cuemem mcp --space me`，源码仓库里 `claude mcp add cuemem -- uv --directory /path/to/cuemem run cuemem mcp --space me`。

| 工具 | 做什么 |
|---|---|
| `remember(text)` | 记一条笔记，立即可检索，后台抽取 |
| `recall(query, deep=False)` | 至少一次 L1 召回（控制器可以决定查得更深；`deep=True` 强制 L2）；返回画像、事实与对话片段，各带 ID 与日期；不调用回答模型 |
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
| `CUEMEM_API_KEYS_FILE` / `CUEMEM_API_KEYS` | 限定作用域的令牌：YAML 文件的路径，或同样内容的文本（只能设一个） |
| `CUEMEM_ALLOWED_HOSTS` | 额外接受的 Host 与 Origin 主机名（逗号分隔；`*` 表示两项都不检查） |
| `CUEMEM_VECTOR_STORE__URL` | 用 Qdrant 服务端代替本地模式 |
| `CUEMEM_EMBEDDING__BACKEND` | 例如 `hash`：离线的哈希向量（只用于测试与演示） |
| `CUEMEM_PROXY__UNKNOWN_MODEL` | `remap`（缺省）或 `reject` |
| `CUEMEM_RETRIEVAL__CONTROLLER` | 换一个控制器模型文件 |
| `CUEMEM_EVAL_RAW_SESSION_IDS`、`CUEMEM_LANGMEM_MAX_LOST_SHARE`、`CUEMEM_AMEM_K` | 只用于评测（见 D-067 与 D-065） |

## 命令行

`cuemem --help` 列出全部命令：`configure`、`doctor`、`ingest`、`export`、`import`、`consolidate`、`forget`、`ask`、`search-image`、`spaces`、`usage`、`serve`、`proxy`、`chat`、`demo`、`migrate-vectors`、`mcp`、`inspect`、`data` 与 `eval`。`cuemem <命令> --help` 说明各自的开关；帮助与提示都是中文。`cuemem --preset <名字> <命令>` 对任何命令换预设。
