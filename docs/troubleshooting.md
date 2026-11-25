# 排查问题

[English](en/troubleshooting.md) · 中文

一条出错信息是什么意思、该怎么办。命令行的提示与核心抛出的错误是中文，这里照打印出来的样子引用，会变的部分写成 `<…>`；HTTP 上的出错是英文。每一行都出自代码里的一条信息，或者[接口参考](api.md)、[部署](deployment.md)、[配置参考](configuration.md)、[框架集成](integrations.md)里写明的行为，细节在那几页。

先跑 `cuemem doctor`（源码仓库里是 `uv run cuemem doctor`）：它写出版本、生效的配置、设了哪些 Key、文件的权限、每个空间抽取失败与排着队的任务，以及缓存与 trace 的大小。`cuemem doctor --live` 还检查配置的各个模型连不连得上。

## 第一次运行

| 看到的 | 原因 | 怎么办 |
|---|---|---|
| `缺少环境变量 <KEY>（<provider>），请运行 cuemem configure`，或者 `向量化需要环境变量 <KEY>，…`（`MissingAPIKeyError`） | 某个模型角色或向量模型用的供应商没有设 Key。默认配置要 `DEEPSEEK_API_KEY` 与 `DASHSCOPE_API_KEY` | 在真实终端里运行 `cuemem configure`，再 `cuemem doctor --live`。只有 OpenAI 的 Key 时用 `--preset openai`、`Memory(preset="openai")` 或 `CUEMEM_PRESET=openai`。一个 Key 都没有、只想试写入与检索时，在单独的 `CUEMEM_HOME` 里设 `CUEMEM_EMBEDDING__BACKEND=hash`，或者运行 `examples/keyless.py`；抽取与回答仍然要 LLM 的 Key |
| `数据库 <path> 建立时的 <key> 是 <A>，当前配置是 <B>。混用会导致向量不可比…`（`ConfigMismatchError`） | 数据目录会记住建它时用的向量模型（命令失败了也一样），换一个就拒绝：两个模型的向量不能比 | 每个预设、每个向量模型用自己的数据目录（`CUEMEM_HOME`，SDK 里是 `Memory(home=...)`），或者用 `CUEMEM_PATHS__DB` 与 `CUEMEM_PATHS__QDRANT` 指到新的位置。确定不要那里的记忆时，删掉那个数据库与向量库再重新导入 |
| 任何命令都只打印一行 `错误：配置有误：…`，退出码 1 | 预设名写错了，或者某个 `CUEMEM_*` 的值不合法。用 `docker run --env-file` 时，写在值后面同一行的注释会算进值里 | 改掉这一行点名的值；`.env` 里的注释单独一行。配置项与缺省值见[配置参考](configuration.md) |
| `cuemem doctor` 用黄色打印 `配置文件 <file> 里有不认识的键 <key>（是不是 <near>？），已忽略` | 自己写的配置文件（`CUEMEM_CONFIG` 的文件，或按路径给的预设）里有写错的键。读 YAML 时不严格，这个键被忽略，用的是缺省值 | 改对这个键，再跑一次 `cuemem doctor` |
| `cuemem doctor` 的 `兜底模型：` 一行是黄色的，写着 `未设置 <ENV>；不用兜底时设 consolidation.fallback_model: null` | 抽取的兜底模型用的供应商没有设 Key（缺省是 `deepseek-flash`）；自己写的预设常常忘了改它 | 把 `consolidation.fallback_model` 设成 `null`，或者你的网关上的一个模型；见[自己写预设时容易踩的坑](configuration.md#容易踩的坑) |
| `cuemem ingest` 打印 `错误：<file> 不是有效的对话文件（<N> 处）` 或 `错误：<file> 不是 UTF-8 文本`，以 1 退出 | 文件不是对话的格式，或者不是 UTF-8 文本 | 格式照 `examples/toy_conversation.en.json`。Mem0 的导出用 `cuemem import --from mem0`，CueMem 的导出用 `cuemem import` |
| `cuemem import` 打印 `<file> 是 zip：ChatGPT、Claude 的数据导出加上 --from chatgpt 或 --from claude（再给 --user U）` | 给了 ChatGPT 或 Claude 的数据导出，却没有写 `--from` | 加上 `--from chatgpt` 或 `--from claude`，再给 `--user U`；`--dry-run` 先看会导入什么 |
| 脚本退出时打印一段 Qdrant 的 "Exception ignored" 调用栈 | `Memory` 没有关。这段调用栈无害 | 调用 `m.close()`，或者用 `with Memory(...) as m:` 打开 |
| `pip install -U "cuemem @ git+https://github.com/FlickYan/cuemem"` 什么都没装 | 版本号没变时 pip 跳过 git 地址 | `pip install -U --force-reinstall "cuemem[...] @ git+https://github.com/FlickYan/cuemem@vX.Y.Z"`；用 `cuemem --version` 核对（见[升级](deployment.md#升级)） |

## 一个数据库一个进程

| 看到的 | 原因 | 怎么办 |
|---|---|---|
| `数据库 <path> 正被另一个进程使用（pid <N>，<command>）…`，或者 `向量库 <path> 正被另一个进程使用…`（`StorageLockedError`） | 一个数据库同一时间只能由一个进程打开，用 Qdrant 服务端时也一样。`cuemem serve` 或别的进程拿着它时，第二个进程（SDK 的 `Memory()`、stdio 的 `cuemem mcp`、`cuemem ingest`、`import`、`reindex`……）被拒绝 | 停掉信息里点名的那个进程，或者经它来访问：`MemoryClient`、`/api/v1`、`/mcp`，导入用 `cuemem import --url http://127.0.0.1:7860`。只读的命令（`export`、`spaces`、`inspect`、`doctor`、`keys`、`usage`）在 serve 运行时照样能用。`CUEMEM_ALLOW_SHARED_DB=1` 关掉这把锁，风险自负 |
| Claude Code 的钩子（`cuemem hook …`）什么都没做 | 钩子出错时一律放行：任何失败都是退出码 0、不输出，免得挡住 Claude Code。常见的原因：没设 `CUEMEM_HOOK_USER`；`CUEMEM_URL`（缺省 `http://127.0.0.1:7860`）上没有 `cuemem serve` 在回应；令牌不对；设了 `HTTP_PROXY` 却没有给 127.0.0.1 设 `NO_PROXY` | 给钩子的命令加 `--debug`：它往 stderr 写一行原因（见 [Claude Code 的钩子](integrations.md#claude-code-的钩子)） |
| 本来要导入本地的库，`cuemem import` 却打印 `经 <url> 导入。` | 环境里设了 `CUEMEM_URL`（比如给钩子用的），这时 `cuemem import` 经那个服务导入 | 运行这条命令时去掉 `CUEMEM_URL` |
| `MemoryClient` 连不上本机的服务（`MemoryAPIError`，`status` 是 None；连接被拒绝时 `code` 是 `unreachable`） | `base_url` 上没有服务在监听；或者设了 `HTTP_PROXY` 却没有给 127.0.0.1 设 `NO_PROXY`，请求发给了代理 | 启动 `cuemem serve`，或者把 127.0.0.1 加进 `NO_PROXY`。404 的信息里说 `base_url` 要是 `cuemem serve` 的根地址时，是地址里带了 `/api/v1` 这样的路径 |

## 启动服务

| 看到的 | 原因 | 怎么办 |
|---|---|---|
| `…没有设置 CUEMEM_API_TOKEN，拒绝启动…`，退出码 2 | `cuemem serve` 与 `cuemem proxy` 监听非本机地址又没有令牌时拒绝启动；监听本机地址、`CUEMEM_ALLOWED_HOSTS` 却列了不只指本机的主机名时也一样 | 在 `.env` 里设 `CUEMEM_API_TOKEN`（例如用 `openssl rand -hex 32` 生成），或者用 `CUEMEM_API_KEYS_FILE` 给出限定作用域的令牌。只在可信的网络里才加 `--insecure-no-auth` |
| `…这些访问令牌短于 16 个字符，拒绝启动：<names>`，退出码 2 | 同样的部署下令牌至少 16 个字符；信息里只列令牌的名字 | 用 `openssl rand -hex 32` 生成的令牌。确实要用短令牌时加 `--insecure-weak-token`（compose 里加进 `command:`） |
| `docker compose up -d` 返回了，http://127.0.0.1:7860/ 却没有响应 | 这条命令在容器启动之后就返回；cuemem 容器启动出错、在反复重启，多半是缺 Key，或令牌短于 16 个字符 | `docker compose ps` 显示它在 Restarting，`docker compose logs cuemem` 里有一行原因。改好 `.env` 再 `docker compose up -d`（见[部署](deployment.md#部署)） |
| `<file> 打不开（…）：检查目录权限与磁盘空间`，退出码 1 | SQLite 打不开这个文件：目录只读，或者磁盘满了 | 腾出空间或改好权限；`cuemem doctor` 写着数据目录的剩余空间 |
| `<file> 打不开（…）：文件损坏或不是 SQLite 数据库，从备份恢复…` | `cuemem.sqlite`、`ledger.sqlite`、`cache/llm.sqlite` 或 `cache/embed.sqlite` 损坏，或者不是 SQLite 数据库 | `cuemem.sqlite` 从备份恢复。LLM 缓存、向量缓存与账本可以直接删掉、会重建（账本删掉就没有以前的用量记录了）；见[备份与恢复](deployment.md#备份与恢复) |
| `<path> 是更新的 CueMem 写的（数据格式 v<N>，这个版本只支持到 v<M>）…` | 这个库是更新的 CueMem 写的；不支持降级 | 升级（见[升级](deployment.md#升级)），或者用新版本导出（`cuemem export`）再在这里导入 |

## 浏览器里与 HTTP 上

| 看到的 | 原因 | 怎么办 |
|---|---|---|
| `http://127.0.0.1:7860/` 显示的是带 `"ui": false` 的 JSON，不是界面 | 浏览器界面是可选依赖 `ui`，或者服务是加 `--no-ui` 启动的。0.10.0 之前这里是没有说明的 404 | `ui_hint` 字段写着是哪一种，以及这种安装方式下的安装命令。这段 JSON 还列出接口文档、健康检查与记忆接口在哪里 |
| `/docs` 是空白页 | 它从 cdn.jsdelivr.net 加载 Swagger UI 5.34.1，带完整性校验，浏览器要连得上这个域名 | 用不需要联网的 `/openapi.json`，或者仓库里的 [docs/openapi-v1.json](openapi-v1.json) |
| `/redoc` 回 404 | 0.9.0 起不再提供 | 用 `/docs` 或 `/openapi.json` |
| 400 `Invalid host header: this server accepts only the host names it listens on and those in CUEMEM_ALLOWED_HOSTS.` | 请求的 Host 不是服务接受的，多半是经反向代理或别的主机名访问 | 把这些主机名写进 `CUEMEM_ALLOWED_HOSTS`（逗号分隔，可以写在 `.env` 里）；见[公网部署](deployment.md#公网部署) |
| 点名某个用户、Agent 或空间时 403 | 限定作用域的令牌管不到它。别的网站发来的浏览器请求也是 403 | 换一个作用域包括这个用户与 Agent 的令牌（`cuemem keys list` 列出作用域） |
| 400，说某个 ID `must be at most 256 characters` | 0.10.0 起用户与 Agent 的 ID 在各处都最多 256 个字符，`add` 与 `open_run` 的 `run_id` 也是 | 把很长的 ID（比如原样的 JWT）换算成短的 |
| 413 | 请求体超过上限，读之前就拒绝：`/api`、`/v1` 与界面上传 64 MiB，导入 1 GiB，别处 8 MiB | 用 `CUEMEM_MAX_BODY_BYTES` 与 `CUEMEM_MAX_IMPORT_BYTES` 改上限；`cuemem import --url --batch N` 把很大的导入分批发 |
| 429 | 限定作用域的令牌超过了它的 `rate_per_minute`、同时处理的请求数（缺省 8 个）或 `max_spaces`；一个用户有 64 个进行中的会话或 run（`CUEMEM_MAX_OPEN_SESSIONS`）；或者记忆代理上 code 是 `budget_exceeded`：这个令牌的 `max_daily_tokens` 用完了，要到本地时间的午夜 | 照 `Retry-After` 等；在自然的段落处结束 run；调高令牌的上限。`cuemem usage --by-key` 列出有预算的令牌今天的 token 数 |
| 503，code 是 `capacity` | Qdrant 本地模式每个用户空间一直开着一个文件，进程快到打开文件数的上限了，建不了新的空间（已有的空间的第一个图片集合也一样）；已有的空间照常可用 | `cuemem doctor` 列出集合数与上限。删掉不用的空间、调高 `ulimit -n`，或者换 Qdrant 服务端：`cuemem migrate-vectors --to URL`，再设 `CUEMEM_VECTOR_STORE__URL` |
| 503 `vector_store_unavailable`、504 `vector_store_timeout`，或者 `/api/ready` 回 503、写着 `vector_store` 或 `worker` | Qdrant 服务端没有回应或太慢；`worker` 是后台抽取的线程退出了 | 让 Qdrant 服务端恢复。这期间记忆代理照样回答，只是不带召回的记忆；Docker 的健康检查探的是 `/api/ready`，所以恢复之前容器显示 unhealthy |

## 记忆代理

| 看到的 | 原因 | 怎么办 |
|---|---|---|
| 响应头里是 `X-CueMem-Level: error` | 召回失败了，请求不带记忆照常转发。`cuemem.api.proxy` 这个 logger 按用户空间与出错的类型各警告一次：`CueMem could not recall memories for a proxy request (…); forwarding without them.` | 看这条警告里出错的类型与状态码；完整的出错记在 DEBUG 级别（`CUEMEM_LOG_LEVEL=DEBUG`） |
| 503 `model_not_available` | 请求点名的目录模型，服务端没有设它的供应商的 Key | 设上这个 Key，或者用 `GET /v1/models` 列出的模型 |
| 400 `model_not_found` | 服务是加 `--unknown-model reject` 启动的，模型名不在它的目录里。不加这个参数时，不认识的名字改用 agent 角色的模型，`X-CueMem-Model` 写着实际用的模型 | 用 `GET /v1/models` 里的名字 |
| `/v1/responses` 回 501 `not_implemented` | 请求用了记忆代理没有覆盖的那部分 Responses API：工具、`previous_response_id`、`background`、文件输入等 | 这些请用 `/v1/chat/completions` |

## 抽取与检索

| 看到的 | 原因 | 怎么办 |
|---|---|---|
| `<N> 个会话抽取失败（<reason>）：<session ids>`，`cuemem ingest --consolidate`、`cuemem consolidate` 或 `cuemem import --consolidate` 以 1 退出；或者 `add(wait=True)` 返回 `status: "queued"` 与 `error` | 发言存下之后抽取出错了（比如 Key 过期、余额不足）。发言都在，后台重试 3 次之后就停了 | 按这一行给的原因改好，再运行 `cuemem consolidate --space <space>`。`cuemem doctor` 按空间列出失败的会话与原因；`m.status(user_id=...)` 与 `GET /api/v1/status` 给出一个用户的 |
| 抽取失败的出错以 `OutputTruncatedError` 开头 | 单独一句发言抽出的内容就超过了供应商的输出上限 | 把发言写短些，或者换输出上限更大的抽取模型 |
| 刚写入的事实 `search` 里还没有，只有对话片段 | 每次 `add` 立即可以按对话片段检索；事实要等抽取做完（在后台，或者 run 结束时） | 传 `wait=True`，或者用 `m.status(...)` 看还有什么排着队 |
| `search` 什么都没有，`get_all` 却列得出记忆 | 向量库丢了，或者与数据库对不上 | `cuemem doctor --vectors` 列出哪些空间对不上；`cuemem reindex --space S`（或 `--all`）补齐，不调用 LLM |
| 进程被杀之后 `data/qdrant/meta.json` 是空的，本地向量库里的集合找不到了 | 建集合时进程被杀掉了 | 停掉 serve，运行 `cuemem doctor --repair-vectors`。不要手动删掉 `meta.json`（见[备份与恢复](deployment.md#备份与恢复)） |
| `cuemem doctor` 打印 `媒体库里有 <N> 个被中断的写入或遗忘…` | 写图片或遗忘时进程被杀掉了 | `cuemem serve`、`cuemem proxy` 与 `cuemem mcp` 在启动时与之后每小时清理；`cuemem reindex --all` 现在就清理 |

这里没有的信息，在[接口参考](api.md)里搜它的错误码。
