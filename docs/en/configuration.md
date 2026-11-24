# Configuration reference

English · [中文](../configuration.md)

Every setting of CueMem, where it comes from, and how to write a preset for your own OpenAI-compatible gateway (OpenRouter, vLLM, LiteLLM, a company proxy). The tables below are checked against the configuration models in `src/cuemem/config.py` by `tests/test_docs_configuration.py`, so a field cannot be added without a row here.

API keys are not configuration: they live in environment variables or in the `.env` file of the data directory (`cuemem configure` writes it). The operational variables that are not fields of the configuration (`CUEMEM_HOME`, `CUEMEM_API_TOKEN`, `CUEMEM_LOG_LEVEL` and the others) are listed under [Environment variables](api.md#environment-variables).

## Where settings come from

Four layers, later ones winning; nested sections are merged key by key:

1. **The config file**: the packaged `src/cuemem/configs/default.yaml` (DeepSeek + Qwen). The `CUEMEM_CONFIG` variable (or `load_settings(config_path)`) names a file that is read **instead of** it, not on top of it, so that file has to be complete: its own `providers`, `models` and `roles`. To change a few things, write a preset.
2. **A preset**: `cuemem --preset <name or path>`, the `CUEMEM_PRESET` variable or `Memory(preset=...)`. A name is one of the packaged presets (`openai`, `ollama`, `deepseek`, `qwen`); anything else is the path of a YAML file. It is deep-merged over the config file, so **whatever a preset does not set keeps the DeepSeek + Qwen default**.
3. **Environment variables**: `CUEMEM_<SECTION>__<FIELD>`, nested with double underscores, for example `CUEMEM_ROLES__READER__MODEL=qwen3.8-flash` or `CUEMEM_LLM_CACHE_MODE=off`. They can also go in the `.env` file of the data directory; a variable already set in the shell wins over `.env`.
4. **`Memory(config={...})`**: a nested dict (or a `Settings` object) from Python, for example `Memory(config={"retrieval": {"naive_k": 5}})`.

Unknown keys are treated differently by layer. A `Memory(config=...)` dict raises ValueError, naming each unknown key and the nearest known one. YAML files and environment variables are lenient: a misspelt key is ignored and the default stays in force. `cuemem doctor` therefore checks the files you wrote (the `CUEMEM_CONFIG` file and a preset given by path) and prints one line per unknown key:

```
配置文件 /path/my-preset.yaml 里有不认识的键 providers.gateway.max_concurency（是不是 providers.gateway.max_concurrency？），已忽略
```

(configuration file …: unknown key …, did you mean …? ignored). A value of the wrong type or an undefined reference (a role naming a model that is not in `models`) is an error in every layer: the command prints `错误：配置有误：…` and exits 1.

## Fields

Defaults are the values in force with no preset. For `providers`, `models` and `roles`, whose entries you name yourself, the default column is the value of a field you leave out.

### Top-level fields (`Settings`)

The sections (`paths`, `providers`, …) each have a table below.

| Field | Default | Meaning |
|---|---|---|
| `llm_cache_mode` | `read_write` | The local LLM response cache: `read_write`; `read_only` (a miss is an error, for strict reproduction); `off` |
| `peak_policy` | `wait` | What a call that needs a real request does inside a provider's peak hours (`providers.<name>.peak_hours`): `wait` pauses until they end, `warn` calls and records a reminder, `allow` calls. This value applies to batch jobs (evaluations); the SDK and the interactive commands (`ask`, `serve`, `proxy`, `mcp`) use `warn` regardless, and the SDK's `peak_policy` argument overrides it |
| `traces_retention_days` | `null` | Keep daily trace files this many days (an integer ≥ 1); `null` keeps them forever. `cuemem serve` and `cuemem proxy` delete older files at start-up and then daily |
| `expired_retention_days` | `null` | Delete memories this many days after their `expiration_date` (an integer ≥ 0); `null` never deletes them (they are only left out when reading). Applied by `cuemem serve` and `cuemem proxy` at start-up and then daily |

### `paths`

Where data lives. Relative paths are relative to the data directory: `CUEMEM_HOME`, else the repository root in a source checkout, else `~/.cuemem`.

| Field | Default | Meaning |
|---|---|---|
| `db` | `data/cuemem.sqlite` | The main SQLite database (the raw log and everything derived from it) |
| `qdrant` | `data/qdrant` | The local vector store (used when `vector_store.url` is empty) |
| `llm_cache` | `data/cache/llm.sqlite` | The LLM response cache |
| `embed_cache` | `data/cache/embed.sqlite` | The embedding cache |
| `ledger` | `data/ledger.sqlite` | The usage and cost ledger (`cuemem usage`) |
| `traces` | `runs/traces` | Daily trace files |
| `datasets` | `data/datasets` | Downloaded evaluation datasets |
| `eval_runs` | `runs/eval` | One sub-directory per evaluation run |
| `eval_spaces` | `data/eval-spaces` | Memory spaces built for evaluations, reused across runs |
| `media` | `data/media` | Original images, stored by content hash |
| `models` | `data/models` | Local models (the embedding models fastembed downloads) |

### `providers`

`providers.<name>`: one entry per model provider. The default file defines `deepseek` and `qwen`; the `openai` and `ollama` presets add their own. A preset adds providers, it does not remove the default ones: a provider nothing refers to is shown as unused by `cuemem doctor` and needs no key.

| Field | Default | Meaning |
|---|---|---|
| `base_url` | required | The OpenAI-compatible endpoint, usually ending in `/v1` |
| `api_key_env` | `null` | The NAME of the environment variable that holds the key (never the key itself); `null` for a service that needs no key (Ollama, vLLM) |
| `max_concurrency` | `8` | Requests in flight to this provider at once |
| `timeout_s` | `120` | Timeout of one request, in seconds |
| `max_retries` | `3` | Retries on rate limits, 5xx and connection errors. A timeout is retried only when the caller has no fallback model |
| `peak_hours` | `null` | Time-of-day pricing, see below; `null` for a provider without it |

### `providers.<name>.peak_hours`

Time-of-day pricing of one provider (the default file sets it for `deepseek`).

| Field | Default | Meaning |
|---|---|---|
| `timezone` | `Asia/Shanghai` | The time zone of the windows |
| `weekdays` | `[0, 1, 2, 3, 4]` | The days the windows apply to (Monday = 0) |
| `windows` | `["09:00-12:00", "14:00-18:00"]` | The peak windows, `HH:MM-HH:MM` |
| `off_peak_factor` | `0.5` | List prices are peak prices; the cost outside the windows is the list price times this factor |

### `models`

`models.<alias>`: the model catalogue. Roles, the fallback model and the memory proxy refer to models by alias.

| Field | Default | Meaning |
|---|---|---|
| `provider` | required | A key of `providers` |
| `model_id` | required | The model name sent to the provider |
| `price` | required | List price per million tokens, see below; used for the cost ledger only |
| `currency` | `USD` | `USD` or `CNY` |
| `context_window` | `null` | The context window in tokens (informational) |
| `vision` | `false` | Whether the model reads images. An answering model without it gets image descriptions instead of the images |
| `extra_body` | `{}` | Provider-specific request parameters, sent as they are (for example to switch a thinking mode off) |
| `fixed_temperature` | `false` | `true` for reasoning models that accept only the default temperature: requests carry no `temperature` |
| `max_tokens_param` | `max_tokens` | The name of the output-limit parameter: `max_tokens` or `max_completion_tokens` |

### `models.<alias>.price`

List prices per million tokens, in the model's `currency`.

| Field | Default | Meaning |
|---|---|---|
| `input` | required | Input tokens (peak hours, no cache hit) |
| `output` | required | Output tokens |
| `cached_input` | `null` | Input tokens served from the provider's prompt cache; `null` means the same as `input` |

### `roles`

`roles.<role>`: which model does which job. There are seven roles: `extractor` (extraction when writing), `planner` (query planning and the sufficiency check of `mode="l2"`), `reader` (answers), `vlm` (answers that look at the original images), `captioner` (image descriptions when writing), `judge` (scoring in evaluations) and `agent` (the chat agent; also the model the memory proxy uses for an unknown model name).

| Field | Default | Meaning |
|---|---|---|
| `model` | required | A key of `models` |
| `temperature` | `0` | The sampling temperature (not sent to a `fixed_temperature` model) |
| `max_tokens` | `null` | The output limit; `null` leaves it to each call |

### `embedding`

The text embedder. A data directory remembers the embedding model it was built with and refuses another one, so use one data directory per embedding model.

| Field | Default | Meaning |
|---|---|---|
| `backend` | `dashscope` | `dashscope` (Qwen's native endpoint), `openai` (any OpenAI-compatible `/embeddings`, at the provider's `base_url`), `fastembed` (local ONNX inference, no key, needs the `local` extra) or `hash` (offline hash vectors, tests and demos only) |
| `provider` | `qwen` | A key of `providers` (not used by `fastembed` and `hash`) |
| `model` | `qwen3.7-text-embedding` | The embedding model |
| `dim` | `1024` | The vector size; it must match what the model returns |
| `sparse` | `false` | Also request sparse vectors (`dashscope` only) |
| `batch_size` | `10` | Texts per request |
| `max_chars` | `6000` | Each text is cut to this many characters before embedding |
| `query_instruct` | `null` | An instruction added to queries only, for models trained with one |
| `send_dimensions` | `false` | `openai` backend: send `dimensions` (only models that support shortened vectors accept it) |
| `native_url` | `https://maas.qwencloudapi.com/api/v1/services/embeddings/text-embedding/text-embedding` | The endpoint of the `dashscope` backend |
| `price_per_m` | `0.07` | Price per million tokens, for the ledger |
| `currency` | `USD` | `USD` or `CNY` |
| `timeout_s` | `60` | Timeout of one request, in seconds |
| `max_concurrency` | `2` | Embedding requests in flight at once |
| `max_attempts` | `6` | Attempts on 429, 5xx and network errors (exponential back-off) |

### `multimodal_embedding`

Image vectors: images and text in one vector space, for text-to-image and image-to-image search.

| Field | Default | Meaning |
|---|---|---|
| `backend` | `dashscope` | `dashscope`, `hash` (offline) or `none`: no image vectors. With `none`, images are still stored, described and found through their descriptions |
| `provider` | `qwen` | A key of `providers` |
| `model` | `tongyi-embedding-vision-plus` | The image-text embedding model |
| `dim` | `1152` | The vector size |
| `url` | `https://maas.qwencloudapi.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding` | The endpoint |
| `price_per_m` | `0.108` | Price per million tokens, for the ledger |
| `currency` | `USD` | `USD` or `CNY` |
| `batch_size` | `8` | Items per request |
| `max_side` | `768` | Images are scaled down to this longest side (pixels) before embedding |
| `timeout_s` | `60` | Timeout of one request, in seconds |
| `max_concurrency` | `2` | Requests in flight at once |
| `max_attempts` | `6` | Attempts on 429, 5xx and network errors |

### `rerank`

The reranker of `mode="l2"`: a cross-encoder scores the candidate turns.

| Field | Default | Meaning |
|---|---|---|
| `provider` | `qwen` | A key of `providers`; `null` switches reranking off (`mode="l2"` then skips that step) |
| `model` | `qwen3-rerank` | The rerank model |
| `url` | `https://maas.qwencloudapi.com/api/v1/services/rerank/text-rerank/text-rerank` | The endpoint (Qwen's native rerank format) |
| `price_per_m` | `0.1` | Price per million tokens, for the ledger |
| `currency` | `USD` | `USD` or `CNY` |
| `max_chars` | `4000` | Each passage is cut to this many characters |
| `timeout_s` | `60` | Timeout of one request, in seconds |
| `max_concurrency` | `4` | Requests in flight at once |
| `max_attempts` | `4` | Attempts on 429, 5xx and network errors |

### `vector_store`

Where vectors are stored.

| Field | Default | Meaning |
|---|---|---|
| `url` | `null` | `null`: Qdrant local mode in `paths.qdrant`. A URL (`http://localhost:6333`) uses a Qdrant server |
| `api_key` | `null` | The server's API key. Keep it out of files: set `CUEMEM_VECTOR_STORE__API_KEY` in the environment or `.env` |

### `retrieval`

Retrieval sizes and the controller.

| Field | Default | Meaning |
|---|---|---|
| `naive_k` | `8` | How many turns a search returns when the caller gives no `k` |
| `fact_k` | `10` | Fact-augmented retrieval: how many facts go into the context |
| `fact_pool` | `50` | Fact-augmented retrieval: candidates fetched by one vector search (turns and facts together) |
| `controller` | `controller_model.json` | The retrieval controller's model file (a relative path is relative to `src/cuemem/read/`) |

### `consolidation`

Asynchronous consolidation: extraction of facts and entities, reconciliation, session summaries and the profile. The similarity thresholds are cosine similarities of the text embedder.

| Field | Default | Meaning |
|---|---|---|
| `assistant_chars` | `600` | Characters kept of each assistant reply during extraction; `null` keeps all |
| `max_session_chars` | `24000` | A longer session is split by turns into several extraction windows |
| `workers` | `8` | Sessions of one space extracted in parallel |
| `background_workers` | `4` | Spaces the background worker consolidates at once (≥ 1; one space is still one at a time) |
| `fallback_model` | `deepseek-flash` | A key of `models`, used when the primary model times out or its content filter rejects a request; `null` for none. Its provider needs a key even though `cuemem configure` does not ask for it |
| `duplicate_threshold` | `0.93` | Statements more similar than this are the same fact (merged) |
| `conflict_threshold` | `0.75` | A single-valued fact this similar to an existing one goes to the LLM to decide between add, supersede and correct |
| `multi_conflict_threshold` | `0.88` | The same for multi-valued facts |
| `entity_merge_threshold` | `0.9` | Entities whose descriptions are more similar than this are merged |
| `entity_gray_threshold` | `0.75` | Entities between the two thresholds go to the LLM to confirm |
| `profile_max_chars` | `1600` | The size limit of the profile (about 400 tokens) |
| `profile_min_importance` | `4` | The lowest importance (1 to 5) that enters the profile |
| `instructions` | `null` | Extra extraction instructions: what to remember and what not (`Memory(instructions=...)`). Changing it rebuilds the derived memory |
| `language` | `null` | The language memories are written in: `null` is the built-in rule (Chinese or English by the share of Han characters), `auto` follows the user's language, any other value is taken as a language name (`Japanese`). Changing it rebuilds the derived memory |
| `caption_images` | `true` | Describe images during consolidation; switch it off when the `captioner` model cannot read images |
| `caption_max_side` | `768` | Images are scaled down to this longest side (pixels) before captioning |

### `proxy`

The memory proxy (`cuemem proxy` and `/v1` of `cuemem serve`).

| Field | Default | Meaning |
|---|---|---|
| `unknown_model` | `remap` | A request whose `model` is not an alias or model id of `models`: `remap` uses the `agent` role's model (named in the `X-CueMem-Model` response header), `reject` answers 400 |

## Writing a preset for your own gateway

A preset for an OpenAI-compatible gateway that serves one chat model and one embedding model. Save it as `my-preset.yaml`:

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

Check it, fill in the key, then test the connection (a fresh data directory keeps it apart from any memory built with another embedding model):

```bash
export CUEMEM_HOME=$(mktemp -d)                   # or a permanent directory of your choice
cuemem --preset ./my-preset.yaml doctor           # offline: what is configured
cuemem --preset ./my-preset.yaml configure        # asks for GATEWAY_API_KEY only
cuemem --preset ./my-preset.yaml doctor --live    # one tiny call per model and embedder
```

In a source checkout the commands are `uv run cuemem …`. From Python the same file is `Memory(preset="./my-preset.yaml")`; for every command of a shell, `export CUEMEM_PRESET=/abs/path/my-preset.yaml`.

`cuemem doctor` should then show every role on `gateway`, the default providers as unused (`（未使用）`), and these three lines:

```
图片向量：关
重排：关
兜底模型：无
```

(image vectors: off; reranker: off; fallback model: none).

### Pitfalls

- **Set all seven roles.** A role the preset leaves out keeps its default model (`vlm` and `reader` on DeepSeek, the others on Qwen), and its first call fails on the missing key. The roles table of `cuemem doctor` shows which provider each role uses.
- **Switch off what your gateway does not offer**: `multimodal_embedding.backend: none` and `rerank.provider: null`. Both otherwise stay on Qwen's native endpoints. They are not in the roles table; doctor prints them as `图片向量：` and `重排：`.
- **Set `consolidation.fallback_model`** to `null`, or to a model on your gateway. The default is `deepseek-flash`: one timeout or one content-filter rejection of the primary model switches to it. `cuemem configure` does not ask for that provider's key and the providers table shows it as unused, so doctor prints the `兜底模型：` line in yellow when the key is missing (`未设置 DEEPSEEK_API_KEY；不用兜底时设 consolidation.fallback_model: null`).
- **A model that cannot read images** needs `consolidation.caption_images: false` as well as `vision: false`, as in the packaged `ollama` preset. Images are still stored; they get no description, and answers do not attach them.
- **One data directory per embedding model.** A data directory remembers the embedding model and dimension it was built with and refuses another one with a clear error. Give each preset its own `CUEMEM_HOME`.
- **`embedding.dim` must be the size the model returns.** If the model supports shortened vectors, set `send_dimensions: true` to ask for `dim`.
- **Misspelt keys are ignored.** Run `cuemem doctor` after every edit and read the yellow lines.
- **Reasoning models** that reject `temperature` or `max_tokens` need `fixed_temperature: true` and `max_tokens_param: max_completion_tokens`, as in the packaged `openai` preset.

The packaged presets in `src/cuemem/configs/presets/` are complete examples in the same shape.
