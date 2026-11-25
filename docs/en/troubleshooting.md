# Troubleshooting

English · [中文](../troubleshooting.md)

What a failure message means and what to do about it. Command-line messages and errors raised in the core are in Chinese, so they are quoted here as printed, with `<…>` for the parts that vary; HTTP errors are in English. Every row comes from a message in the code or from a behaviour described in the [API reference](api.md), the [deployment guide](deployment.md), the [configuration reference](configuration.md) or the [integrations guide](integrations.md); those pages have the details.

Start with `cuemem doctor` (`uv run cuemem doctor` in a checkout): it prints the version, the configuration in effect, the keys that are set, file permissions, failed and pending extractions per space, and the size of the caches and traces. `cuemem doctor --live` also checks that the configured models can be reached.

## First run

| What you see | Why | What to do |
|---|---|---|
| `缺少环境变量 <KEY>（<provider>），请运行 cuemem configure`, or `向量化需要环境变量 <KEY>，…` (`MissingAPIKeyError`) | The key of a provider that a model role or the embedder uses is not set. The default configuration needs `DEEPSEEK_API_KEY` and `DASHSCOPE_API_KEY` | Run `cuemem configure` in a real terminal, then `cuemem doctor --live`. With only an OpenAI key, use `--preset openai`, `Memory(preset="openai")` or `CUEMEM_PRESET=openai`. To try writing and retrieval without any key, set `CUEMEM_EMBEDDING__BACKEND=hash` in a separate `CUEMEM_HOME`, or run `examples/keyless.py`; extraction and answering still need an LLM key |
| `数据库 <path> 建立时的 <key> 是 <A>，当前配置是 <B>。混用会导致向量不可比…` (`ConfigMismatchError`) | A data directory remembers the embedding model it was built with, even after a command that failed, and refuses another one: vectors of two models cannot be compared | Give each preset or embedding model its own data directory (`CUEMEM_HOME`, or `Memory(home=...)` in the SDK), or point `CUEMEM_PATHS__DB` and `CUEMEM_PATHS__QDRANT` at a new place. If the memories there are not needed, delete that database and vector store and import again |
| `错误：配置有误：…`, exit code 1, from any command | An unknown preset name or an invalid `CUEMEM_*` value. With `docker run --env-file`, a comment after a value on the same line becomes part of the value | Fix the value the line names; keep comments in `.env` on their own lines. Fields and defaults are in the [configuration reference](configuration.md) |
| `cuemem doctor` prints, in yellow, `配置文件 <file> 里有不认识的键 <key>（是不是 <near>？），已忽略` | A misspelt key in a config file you wrote (the `CUEMEM_CONFIG` file, or a preset given by path). YAML loading is lenient, so the key was ignored and the default applies | Correct the key and run `cuemem doctor` again |
| `cuemem doctor` prints the `兜底模型：` line in yellow, with `未设置 <ENV>；不用兜底时设 consolidation.fallback_model: null` | The extraction fallback model is on a provider whose key is not set (the default is `deepseek-flash`); a custom preset often leaves it there | Set `consolidation.fallback_model` to `null` or to a model on your gateway; see the [pitfalls of a custom preset](configuration.md#pitfalls) |
| `cuemem ingest` prints `错误：<file> 不是有效的对话文件（<N> 处）` or `错误：<file> 不是 UTF-8 文本` and exits 1 | The file is not in the conversation format, or is not UTF-8 text | The format is that of `examples/toy_conversation.en.json`. A Mem0 export goes through `cuemem import --from mem0`, a CueMem export through `cuemem import` |
| `cuemem import` prints `<file> 是 zip：ChatGPT、Claude 的数据导出加上 --from chatgpt 或 --from claude（再给 --user U）` | A ChatGPT or Claude data export was passed without `--from` | Add `--from chatgpt` or `--from claude`, with `--user U`; `--dry-run` shows what would be imported |
| A script prints a Qdrant "Exception ignored" traceback when it exits | The `Memory` was not closed. The traceback is harmless | Call `m.close()`, or open it as `with Memory(...) as m:` |
| `pip install -U "cuemem @ git+https://github.com/FlickYan/cuemem"` installs nothing new | pip skips a git URL whose version did not change | `pip install -U --force-reinstall "cuemem[...] @ git+https://github.com/FlickYan/cuemem@vX.Y.Z"`; check with `cuemem --version` ([Upgrading](deployment.md#upgrading)) |

## One process per database

| What you see | Why | What to do |
|---|---|---|
| `数据库 <path> 正被另一个进程使用（pid <N>，<command>）…`, or `向量库 <path> 正被另一个进程使用…` (`StorageLockedError`) | A database is opened by one process at a time, also with a Qdrant server. A second process (an SDK `Memory()`, a stdio `cuemem mcp`, `cuemem ingest`, `import`, `reindex`, …) is refused while `cuemem serve` or another holder runs | Stop the process the message names, or go through it: `MemoryClient`, `/api/v1`, `/mcp`, and `cuemem import --url http://127.0.0.1:7860` for imports. Commands that only read (`export`, `spaces`, `inspect`, `doctor`, `keys`, `usage`) work next to a running serve. `CUEMEM_ALLOW_SHARED_DB=1` turns the lock off at your own risk |
| The Claude Code hooks (`cuemem hook …`) do nothing | The hooks fail open: every failure is exit code 0 with empty output, so that Claude Code is never blocked. Usual causes: `CUEMEM_HOOK_USER` is not set, no `cuemem serve` answers at `CUEMEM_URL` (default `http://127.0.0.1:7860`), a wrong token, or `HTTP_PROXY` set without `NO_PROXY` for 127.0.0.1 | Run the hook command with `--debug`: it writes one line with the reason to stderr ([Claude Code hooks](integrations.md#claude-code-hooks)) |
| `cuemem import` prints `经 <url> 导入。` when a local import was meant | `CUEMEM_URL` is set in the environment (for the hooks, say), and `cuemem import` then imports through that server | Unset `CUEMEM_URL` for that command |
| `MemoryClient` cannot reach a local server (`MemoryAPIError` with `status` None; `code` is `unreachable` for a refused connection) | Nothing listens at `base_url`, or `HTTP_PROXY` is set without `NO_PROXY` for 127.0.0.1, so the request goes to the proxy | Start `cuemem serve`, or add 127.0.0.1 to `NO_PROXY`. A 404 whose message says that `base_url` must be the root of `cuemem serve` means the URL carries a path such as `/api/v1` |

## Starting a server

| What you see | Why | What to do |
|---|---|---|
| `…没有设置 CUEMEM_API_TOKEN，拒绝启动…`, exit code 2 | `cuemem serve` and `cuemem proxy` refuse to listen beyond loopback without a token; the same holds on loopback when `CUEMEM_ALLOWED_HOSTS` names a host that is not loopback-like | Set `CUEMEM_API_TOKEN` in `.env` (for example from `openssl rand -hex 32`) or give scoped keys through `CUEMEM_API_KEYS_FILE`. On a trusted network only, `--insecure-no-auth` |
| `…这些访问令牌短于 16 个字符，拒绝启动：<names>`, exit code 2 | In the same set-ups, tokens must be at least 16 characters; the message lists key names only | Use tokens from `openssl rand -hex 32`. `--insecure-weak-token` accepts short ones (in compose, add it to `command:`) |
| `docker compose up -d` returns, but http://127.0.0.1:7860/ does not answer | The command returns once the containers start; the cuemem container is restarting on a start-up error, usually a missing key or a token shorter than 16 characters | `docker compose ps` shows it Restarting and `docker compose logs cuemem` shows a one-line error. Fix `.env` and run `docker compose up -d` again ([Deployment](deployment.md#deployment)) |
| `<file> 打不开（…）：检查目录权限与磁盘空间`, exit code 1 | SQLite cannot open the file: a read-only directory or a full disk | Free space or fix the permissions; `cuemem doctor` prints the free space of the data directory |
| `<file> 打不开（…）：文件损坏或不是 SQLite 数据库，从备份恢复…` | `cuemem.sqlite`, `ledger.sqlite`, `cache/llm.sqlite` or `cache/embed.sqlite` is corrupt or is not a SQLite database | Restore `cuemem.sqlite` from a backup. The LLM cache, the embedding cache and the ledger can be deleted and are rebuilt (the ledger loses its usage history); see [Backup and recovery](deployment.md#backup-and-recovery) |
| `<path> 是更新的 CueMem 写的（数据格式 v<N>，这个版本只支持到 v<M>）…` | The database was written by a newer CueMem; downgrades are not supported | Upgrade ([Upgrading](deployment.md#upgrading)), or export with the newer version (`cuemem export`) and import here |

## In the browser and over HTTP

| What you see | Why | What to do |
|---|---|---|
| `http://127.0.0.1:7860/` shows JSON with `"ui": false` instead of the UI | The browser UI is the optional `ui` extra, or the server was started with `--no-ui`. Before 0.10.0 this was a bare 404 | The `ui_hint` field says which, with the install command for this kind of install. The JSON also lists where the API docs, the health routes and the memory API are |
| `/docs` is a blank page | It loads Swagger UI 5.34.1 from cdn.jsdelivr.net with integrity checks, so the browser needs to reach that host | Use `/openapi.json`, which works offline, or [docs/openapi-v1.json](../openapi-v1.json) in the repository |
| `/redoc` returns 404 | It is no longer served since 0.9.0 | Use `/docs` or `/openapi.json` |
| 400 `Invalid host header: this server accepts only the host names it listens on and those in CUEMEM_ALLOWED_HOSTS.` | The request's Host is not one the server accepts, typically behind a reverse proxy or under another hostname | List those hostnames in `CUEMEM_ALLOWED_HOSTS` (comma-separated, can go in `.env`); see [Public deployment](deployment.md#public-deployment) |
| 403 for a user, agent or space | The scoped key does not cover it. Browser requests from other origins also get 403 | Use a key whose scope includes that user and agent (`cuemem keys list` shows the scopes) |
| 400 saying an id `must be at most 256 characters` | Since 0.10.0 user and agent ids are at most 256 characters on every surface, and a `run_id` on `add` and `open_run` | Map long ids (a raw JWT, say) to shorter ones |
| 413 | The request body is over the limit, refused before it is read: 64 MiB for `/api`, `/v1` and UI uploads, 1 GiB for imports, 8 MiB elsewhere | `CUEMEM_MAX_BODY_BYTES` and `CUEMEM_MAX_IMPORT_BYTES` change the limits; `cuemem import --url --batch N` sends a large import in pieces |
| 429 | A scoped key went over its `rate_per_minute`, its concurrent requests (8 by default) or its `max_spaces`; a user has 64 open sessions or runs (`CUEMEM_MAX_OPEN_SESSIONS`); or, on the proxy with code `budget_exceeded`, the key's `max_daily_tokens` is used up until local midnight | Honour `Retry-After`; end runs at natural breaks; raise the limit on the key. `cuemem usage --by-key` shows today's tokens per budgeted key |
| 503 with code `capacity` | Local-mode Qdrant keeps one open file per user space, and the process is near its open-file limit, so no new space (or an existing space's first image collection) can be created; existing spaces keep working | `cuemem doctor` shows the collection count against the limit. Delete unused spaces, raise `ulimit -n`, or move to a Qdrant server: `cuemem migrate-vectors --to URL`, then `CUEMEM_VECTOR_STORE__URL` |
| 503 `vector_store_unavailable`, 504 `vector_store_timeout`, or `/api/ready` answering 503 with `vector_store` or `worker` | The Qdrant server does not answer or is slow; `worker` means the background extraction threads have died | Bring the Qdrant server back. The memory proxy keeps answering meanwhile, without recalled memory, and the Docker health check probes `/api/ready`, so the container shows unhealthy until then |

## The memory proxy

| What you see | Why | What to do |
|---|---|---|
| `X-CueMem-Level: error` on a response | Recall failed and the request was forwarded without memory. The `cuemem.api.proxy` logger warns once per user space and error type: `CueMem could not recall memories for a proxy request (…); forwarding without them.` | Read the warning for the error type and status; the full error is logged at DEBUG (`CUEMEM_LOG_LEVEL=DEBUG`) |
| 503 `model_not_available` | The request names a catalogue model whose provider key is not set on the server | Set that key, or use a model that `GET /v1/models` lists |
| 400 `model_not_found` | The server runs with `--unknown-model reject` and the model name is not in its catalogue. Without that flag an unknown name goes to the agent role's model, and `X-CueMem-Model` names the model used | Use a name from `GET /v1/models` |
| 501 `not_implemented` on `/v1/responses` | The request uses a part of the Responses API the proxy does not cover: tools, `previous_response_id`, `background`, file inputs and others | Use `/v1/chat/completions` for those |

## Extraction and search

| What you see | Why | What to do |
|---|---|---|
| `<N> 个会话抽取失败（<reason>）：<session ids>`, and `cuemem ingest --consolidate`, `cuemem consolidate` or `cuemem import --consolidate` exits 1; or `add(wait=True)` returns `status: "queued"` with an `error` | Extraction failed after the turns were stored (an expired key or an empty balance, for example). The turns are safe, and the background worker stops retrying after 3 attempts | Fix the cause the line gives, then run `cuemem consolidate --space <space>`. `cuemem doctor` lists the failed sessions and reasons per space; `m.status(user_id=...)` and `GET /api/v1/status` do the same for one user |
| A failed extraction whose error starts with `OutputTruncatedError` | One single message produced more extraction output than the provider's output limit | Shorter messages, or an extraction model with a larger output limit |
| A fact just written is not in `search` yet, only the conversation excerpt | Each `add` is searchable at once as an excerpt; facts appear when extraction has run (in the background, or when the run ends) | Pass `wait=True`, or call `m.status(...)` to see what is pending |
| `search` returns nothing while `get_all` still lists memories | The vector store is lost or out of step with the database | `cuemem doctor --vectors` shows which spaces differ; `cuemem reindex --space S` (or `--all`) restores them without LLM calls |
| `data/qdrant/meta.json` is empty after a kill, and the collections of the local vector store are not found | The process was killed while a collection was being created | Stop serve and run `cuemem doctor --repair-vectors`. Do not delete `meta.json` by hand ([Backup and recovery](deployment.md#backup-and-recovery)) |
| `cuemem doctor` prints `媒体库里有 <N> 个被中断的写入或遗忘…` | The process was killed during an image write or a forget | `cuemem serve`, `cuemem proxy` and `cuemem mcp` clean up at start-up and hourly; `cuemem reindex --all` does it now |

If the message is not here, search the [API reference](api.md) for its error code.
