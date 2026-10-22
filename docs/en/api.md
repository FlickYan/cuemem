# API reference

English · [中文](../api.md)

Every way a program talks to CueMem: the Python SDK (`Memory`, `AsyncMemory`), the remote client (`MemoryClient`, `AsyncMemoryClient`), the HTTP API of `cuemem serve` (`/api` and the Mem0-shaped `/api/v1`), the OpenAI-compatible memory proxy (`/v1`), the MCP server, the environment variables and the install extras. Signatures are those of CueMem 0.1.0 (`src/cuemem/sdk.py`, `client.py`, `api/`). The decisions behind them are in the [decision records](decisions.md): D-061 (SDK), D-062 (proxy), D-063 (export), D-056 (deployment and tokens), D-069 (the second audit round, which added most of `/api/v1`, `MemoryClient`, scoped keys, metadata, `update` and `reset`).

Language: the SDK's return values and its own errors, `/api/v1`, the proxy and the MCP tools are in English. The command line, the older `/api` routes, the 401 message and errors raised deeper in the core (a missing key, a locked vector store, a configuration mistake) are in Chinese.

## Install

| What you need | Install |
|---|---|
| SDK, CLI, HTTP API, memory proxy, MCP | `pip install "cuemem @ git+https://github.com/FlickYan/cuemem"` |
| + local embeddings with no key (fastembed; the `deepseek` preset needs it) | extra `local` |
| + the browser UI of `cuemem serve` (Gradio) | extra `ui` |
| + evaluation charts and the controller and gate training scripts (matplotlib, scikit-learn) | extra `eval` |
| + framework integrations | extras `openai-agents`, `langgraph`, `anthropic`, or all three: `integrations` |

Extras combine: `pip install "cuemem[ui,local] @ git+https://github.com/FlickYan/cuemem"`. Python 3.12 or later. In a source checkout, `uv sync` installs `ui` and `eval` through the dev group; `uv sync --no-dev` needs `--extra ui --extra eval`.

- Without `ui`, `cuemem serve` serves `/api`, `/api/v1`, `/docs` and `/v1` only and prints how to install the UI (`pip install "cuemem[ui]"`, or `uv sync --extra ui` in a checkout).
- Without `eval`, `cuemem eval run` (except `--build-only`) and `cuemem eval report` exit with code 1 and the same kind of hint; `eval split`, `eval probe`, `eval audit` and `data download` work.
- The Docker image installs `ui` and `local` by default; `docker build --build-arg EXTRAS= .` builds an API-only image.

## Memory spaces and identity

`user_id` and `agent_id` together name one memory space, and spaces share nothing:

| Arguments | Space |
|---|---|
| `user_id="alice"` | `alice` |
| `agent_id="shop"` | `agent/shop` (the agent's own memory) |
| `user_id="alice", agent_id="shop"` | `shop/alice` (what agent shop remembers about alice) |

- A "/" or "%" inside an id is percent-escaped, so `user_id="agent/x"` and `agent_id="x"` never meet. At least one of the two ids is required (ValueError otherwise). `cuemem.sdk.space_for(user_id, agent_id)` computes the space; `space_owner(space)` is its inverse.
- `run_id` is one session (one conversation) inside a space. Without it, each `add` is a session of its own.
- A memory id (the `id` of a fact) already belongs to one space, so `get`, `history`, `update` and `delete` take only the id.

## Python SDK: `Memory`

```python
from cuemem import Memory

# Memory() for the default config, Memory(preset="ollama") for local models
m = Memory(preset="openai")
m.add("I moved to Munich last month.", user_id="alice", wait=True)
m.search("Where do I live now?", user_id="alice")
m.close()  # or: with Memory() as m: ...
```

`Memory` is thread-safe. Its methods return plain dicts and lists, shaped like Mem0's.

### Constructor

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

| Parameter | Meaning |
|---|---|
| `config` | A nested dict deep-merged over the default config and the preset, for example `{"roles": {"reader": {"model": "qwen3.8-flash"}}}` (fields in `src/cuemem/configs/default.yaml`), or a `Settings` object |
| `preset` | `openai`, `ollama`, `deepseek`, `qwen`, or the path of a preset YAML (default: the `CUEMEM_PRESET` variable) |
| `home` | The data directory (default `CUEMEM_HOME`; the repository root in a source checkout, `~/.cuemem` otherwise) |
| `instructions` | Extra extraction instructions: what to remember and what not (Mem0's `custom_instructions`) |
| `background` | Extract facts in a background thread (default on) |
| `peak_policy` | What to do when a call falls in DeepSeek's weekday peak hours: `warn` (the SDK's default: call at peak price and record it), `wait` (pause until off-peak, with a warning on the `cuemem.sdk` logger), `allow` (call without a note) |
| `http_client` | An httpx2 client for the model and embedding providers (tests use it to inject a fake network) |

`from_system` wraps a memory system that is already open in the process (the HTTP service and the proxy use it). Local-mode Qdrant lets only one process open a data directory, so `Memory()` on a directory that `cuemem serve` holds raises `StorageLockedError`; use `MemoryClient` against the server instead.

### Methods

| Method | Returns |
|---|---|
| `add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, expiration_date=None, timestamp=None, infer=True, wait=False)` | `{"results": [events], "session_id", "turn_ids", "status", ...}` |
| `search(query, *, user_id=None, agent_id=None, run_id=None, limit=None, mode="auto", at=None, router="learned", lam=None, filters=None, top_k=None, min_level="l1")` | `{"results", "images", "profile", "level", "see", "reasons"}` |
| `answer(question, *, user_id=None, agent_id=None, mode="auto", at=None, router="learned", lam=None)` | `{"answer", "evidence_sufficient", "citations", "level", "reasons"}` |
| `get_all(*, user_id=None, agent_id=None, run_id=None, include_superseded=False, valid_at=None, known_at=None, limit=None, filters=None, top_k=None)` | `{"results": [facts]}` |
| `get(memory_id)` | a fact, or `None` |
| `history(memory_id)` | the version chain, oldest first, each version with `"event": "ADD"` or `"UPDATE"` |
| `update(memory_id, data, *, metadata=None)` | `{"message", "id", "results", "session_id", "turn_ids", ...}` |
| `delete(memory_id, *, dry_run=False)` | a forget report |
| `delete_all(*, user_id=None, agent_id=None, run_id=None)` | a forget report |
| `reset()` | `{"message", "spaces_deleted", "turns_deleted", "images_deleted"}` |
| `profile(*, user_id=None, agent_id=None)` | the profile text |
| `open_run(run_id=None, *, user_id=None, agent_id=None)` | the run id |
| `end_run(run_id, *, user_id=None, agent_id=None, wait=False)` | `{"session_id", "turns", "consolidation"}` |
| `run(run_id=None, *, user_id=None, agent_id=None, wait=False)` | a context manager that yields the run id |
| `consolidate(*, user_id=None, agent_id=None, rebuild=False)` | a consolidation report |
| `wait()` | `None`, once the background thread has done the queued extraction |
| `export(*, user_id=None, agent_id=None, derived=True)` | a `cuemem.space` export (a JSON-serialisable dict) |
| `load(data, *, user_id=None, agent_id=None, wait=False)` | an import report |
| `spaces()` | one dict per space: `space_id`, `created_at`, `sessions`, `turns`, `units`, `current_facts`, `facts`, `entities` |
| `close()` | `None`; ends this object's open runs (their extraction is registered) and closes the memory system |

**`add`.** `messages` is a string, one OpenAI-format message, or a list of them; only user and assistant messages are stored. Images go in content blocks (`{"type": "image_url", "image_url": {"url": ...}}`) as data URLs or local paths; http(s) image URLs are not fetched (ValueError). The turns are written online, with no LLM call, and are searchable at once; facts are extracted in the background.

- `run_id`: write into that session. Calling `add` once per turn with the same `run_id` re-extracts the whole session (and replays derived memory) every time; wrap a multi-turn conversation in `with m.run(...)` instead, which extracts it once when it ends.
- `metadata`: a JSON object with string keys, stored with each turn of this call. It is not shown to the extraction model, so requests do not change. Facts, excerpts and images derived from those turns carry it, and `filters` can select on it. TypeError for a non-dict; ValueError for non-string keys or values that JSON cannot hold.
- `expiration_date`: a date, a datetime or an ISO string; a bare date lasts to the end of that day. Expired memories leave `search`, `get_all` and the profile; `get` and `history` still return them, and only `delete` removes them. A fact said again later without an expiry does not expire. Expiry is applied when the SDK reads (and so in `/api/v1` and `MemoryClient`); `answer()`, the proxy, MCP, the `/api` routes and the framework integrations still see expired memories until they are deleted.
- `timestamp`: when the messages were said (a datetime, an ISO string or Unix time; default now). Times with a zone are converted to local time.
- `infer=False`: log only (`status: "logged"`); extract later with `consolidate()`.
- `wait=True`: extract now and return Mem0-style events (`status: "consolidated"`, plus `consolidation`). Not allowed while the run is open (ValueError: end the run with `end_run(..., wait=True)`). If extraction fails after the turns were stored (say, the provider is out of balance), `add` does not raise: it returns `status: "queued"` and an `error`. The turns are stored and their extraction is retried on the next consolidation, so the call must not be repeated.
- An error before anything is written (an embedding outage, a bad image) raises, stores nothing, and the same call can be retried.
- `status` is one of `empty` (no message to store), `logged`, `queued` and `consolidated`. A concurrent write that lost its turn numbers raises `ConcurrentWriteError`.

An event is a fact item with `"event": "ADD"` or `"UPDATE"`; an UPDATE also has `previous_memory` (the statements it superseded) and `reason` (`changed`: the world changed; `corrected`: the earlier statement was a mistake).

**`search`.** The retrieval controller decides how deeply to recall. `level` (`L0`, `L1`, `L2`), `see` (whether the controller chose to look at original photos) and `reasons` (in English) say what it chose and why.

- `min_level="l1"` (the default): calling `search` is already a decision to look at memory, so an L0 from the controller is raised to L1, with reasons `["raised from L0: the caller asked to recall"]`, as in MCP and the framework integrations. `min_level="l0"` keeps the controller's decision, and L0 returns only the profile.
- `mode`: `auto` (the controller), or a fixed `l0`, `l1` or `l2`. `router`: `learned` (default), `hybrid` (asks an LLM when the gate is unsure), `rule` or `llm`. `lam`: the cost knob λ (default: the controller model's value; smaller spends more).
- `at`: the question time; relative dates are resolved against it.
- `limit` or `top_k`: truncate `results`. `run_id`: keep only that session's memories.
- `filters`: Mem0 v3's form. `user_id`, `agent_id` and `run_id` may go here instead of the keywords (`user_id` and `agent_id` must be single ids). Every other key is a metadata condition:
  - `{"key": value}` is equality, `{"key": "*"}` means the key is present, and several keys must all hold;
  - `{"key": {"eq" | "ne" | "gt" | "gte" | "lt" | "lte": v, "in" | "nin": [...], "contains" | "icontains": v}}`; when the key is missing, only `ne` and `nin` hold;
  - `{"AND": [cond, ...]}`, `{"OR": [...]}` and `{"NOT": [...]}` (NOT: none of them holds);
  - inside conditions, `user_id`, `agent_id` and `run_id` refer to where the memory lives, for example `{"run_id": {"in": ["r1", "r2"]}}`;
  - an unknown operator raises ValueError.
- Retrieval runs over the whole space, and the session, metadata and expiry filters are applied afterwards, so there can be fewer results than `limit`. The profile is the whole space's, without expired facts.

Result items carry a `type`:

| `type` | Fields |
|---|---|
| `fact` | `id`, `memory` (the statement), `subject`, `predicate`, `object`, `category`, `importance`, `status`, `valid_from`, `valid_to`, `created_at`, `invalidated_at`, `superseded_by`, `session_id`, `source_turn_ids`, `metadata`, `expiration_date`; `score`, `user_id`, `agent_id` or `space_id` where they apply |
| `excerpt` | `id`, `memory` (the conversation round), `score`, `created_at`, `session_id`, `source_turn_ids`, `metadata`, `expiration_date` |
| `image` (in `images`) | `id`, `memory` (the caption), `path`, `created_at`, `attached`, `metadata`, `expiration_date` |

`metadata` merges the source turns' metadata (a later turn wins on the same key); `expiration_date` is the latest expiry among the sources, or `None` if any source has none.

**`answer`.** Retrieves and answers with the configured reader. `mode` also accepts `agent` (the reader decides by itself when to call a `memory_search` tool) and `hatch` (the controller recalls first, then the reader may search more). `citations` are `{"ref", "id", "memory"}`.

**`get_all`.** Current facts by default. `include_superseded=True` adds the replaced versions; `valid_at="2025-06"` replays what was true in the world then; `known_at=<datetime>` replays what the system believed then. `run_id` and `filters` work as in `search`; expired facts are left out.

**`update`.** Facts are never rewritten in place: derived memory is a function of the raw log. `update` logs the user turn `"Correction: <data>"` in a new `edit-…` session placed after the original, and consolidates; the fact extracted from it supersedes the original as `corrected`, without asking an LLM. The new fact inherits the original's metadata (keys in `metadata` override) and its expiry; `history()` shows both versions, and deleting the correction restores the original.

- `message` is `"Memory updated"`.
- If nothing can be extracted from `data`, the original stays as it was, the correction is forgotten again, `turn_ids` is empty and `message` says so.
- If extraction fails after the correction was logged, `message` says so and `error` gives the reason; do not repeat the call.
- KeyError for an unknown id; ValueError when the memory was already replaced (update the current version) or `data` is empty.

**`delete` and `delete_all`.** Forgetting implements the right to be forgotten. `delete` removes the conversation round the fact came from, follows citations into later replies and finds uncited restatements, then rebuilds derived memory from the rest of the log, so a fact it had replaced becomes current again. `dry_run=True` lists what would be deleted and deletes nothing. `delete_all` deletes the whole space (log, derived memory, vectors, caches and photos); with `run_id`, only that session. The report has `message`, `turns_deleted`, `sessions_affected`, `facts_before`, `facts_after`, `images_deleted` and `restated`, plus `warning` when the deletion happened but rebuilding from the rest failed (it finishes on the next consolidation). KeyError for an unknown id.

**`reset`.** Deletes every space in the data directory, including those written by the HTTP service, the proxy and the demo, and ends the open runs. It cannot be undone.

**Runs.** `open_run` starts a conversation: `add(run_id=...)` writes online and is searchable at once, but nothing is extracted until `end_run`, which hands the whole run to extraction (`wait=True` extracts now and returns the report; otherwise `consolidation` is `"queued"`). `with m.run(user_id="alice") as run_id:` does both. A run left open by a crashed process is handed to extraction the first time the next process touches that space.

**Export and import.** `export` returns format `cuemem.space` version 2: sessions, turns (with `metadata`, `expires_at` and `edit_of` when set) and the original images in base64, plus an optional snapshot of facts and the profile that import ignores. `load` writes the raw log back and extracts derived memory again (`wait=True` does it now). Without `user_id` or `agent_id` it imports into the file's own space, and a file without `space_id` then raises ValueError. Version-1 files still import; if any image or metadata in the file is invalid, nothing is written.

### `AsyncMemory`

`AsyncMemory(...)` takes the same arguments, and every method is awaitable (it runs in a thread pool). `async with AsyncMemory() as m:` closes it, and `async with m.run(user_id="alice") as run_id:` is the async run.

### Exceptions

`cuemem.errors` defines `CueMemError` and its subclasses `MissingAPIKeyError`, `CacheMissError`, `StructuredOutputError`, `EmbeddingAPIError` (with `status` and `retry_after`), `DatasetChecksumError`, `ConfigMismatchError`, `ConsolidationError`, `ContentFilteredError`, `ConcurrentWriteError`, `StorageLockedError`, `AccessKeysError` and `MemoryAPIError` (with `status` and `code`). The SDK also raises KeyError, ValueError and TypeError as described above, and errors of the OpenAI client library pass through after CueMem's own retries.

## Remote client: `MemoryClient`

```python
from cuemem import MemoryClient

m = MemoryClient("http://127.0.0.1:7860", api_key="...")  # instead of Memory()
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

- It calls `/api/v1` of `cuemem serve`. Every public method of `Memory` except `from_config` and `from_system` exists with an identical signature (a test compares them) and returns what `Memory` returns on the server, so switching between local and remote is a one-line change. Several processes, or apps on other machines, can share one server this way.
- `api_key` falls back to the `CUEMEM_API_TOKEN` environment variable; it can be the admin token or a scoped key. `timeout` is per request in seconds (an `add` with `wait=True` waits for extraction); with your own `http_client`, that client's timeout applies.
- Local image paths (and `file://`) in content blocks are read on the client and sent as data URLs; the server never reads its own disk.
- `metadata` is checked as in the SDK before it is sent (JSON would silently turn non-string keys into strings).
- Times (`timestamp`, `at`, `known_at`, `expiration_date`) are sent in ISO format. Naive times are read in the server's local time, so include a time zone when client and server differ.
- It raises the same exceptions as `Memory`: KeyError, ValueError, TypeError and the `cuemem.errors` types; an `EmbeddingAPIError` keeps `status` and `retry_after`. Other HTTP failures (a wrong token, an unreachable server, 403 outside a key's scope, other provider errors) raise `MemoryAPIError` with `status` (None when the server is unreachable) and `code` (see the error codes below). `get()` of another tenant's id returns `None`, as for a missing id.
- `close()` (or `with MemoryClient(...) as m:`) ends the runs this client opened. A run opened remotely and never ended stays open on the server until it restarts.
- The module imports only httpx2, not the memory stack.

`AsyncMemoryClient(...)` is the async variant, like `AsyncMemory`.

## HTTP API: `cuemem serve`

`cuemem serve` (default `http://127.0.0.1:7860`) serves, in one process and over one memory system: `/api` (sessions with the agent, chat, browsing, forgetting), `/api/v1` (the Mem0-shaped memory API), `/v1` (the memory proxy), `/docs` (OpenAPI) and, with the `ui` extra, the Gradio UI at `/`.

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `7860` | Bind address |
| `--space` | `shop-demo` | The UI's default space |
| `--scenario` | `shop` in a source checkout, `plain` when installed | The demo shopping assistant or a plain assistant |
| `--router` / `--lam` | `learned` / the controller model's | The controller used by chat |
| `--no-ui` | off | API and proxy only |
| `--insecure-no-auth` | off | Start on a non-loopback address without any token |
| `--unknown-model` | `proxy.unknown_model` (`remap`) | What the proxy does with an unknown model: `remap` or `reject` |

### Authentication and scoped keys

- With no token configured there is no authentication. That is allowed only on a loopback address: a non-loopback bind without a token refuses to start unless `--insecure-no-auth` is given.
- `CUEMEM_API_TOKEN` is the admin token. Send `Authorization: Bearer <token>`. `GET /api/health` needs no token.
- Scoped keys (`CUEMEM_API_KEYS_FILE`, the path of a YAML file, or `CUEMEM_API_KEYS`, the same content inline as YAML or JSON; set only one) give each key a list of users and agents:

  ```yaml
  keys:
    - name: alice-app                 # also the UI username for this key
      sha256: <64 hex digits>         # printf %s "$TOKEN" | sha256sum   (or token: <the token itself>)
      scopes:
        - {user: alice}               # all of alice's memory, with or without an agent
    - name: shop-backend
      sha256: <64 hex digits>
      scopes:
        - {user: "*", agent: shop}    # every user's memory under agent shop (shop/<user>)
        - {agent: shop}               # shop's own memory (agent/shop)
  ```

  A value is an id or a list of ids; `*` matches any id and `acme:*` is a prefix. `{agent: A}` covers only agent A's own memory. `agent/x` always counts as agent x's own memory, so scoped keys cannot use an agent literally named `agent`. Spaces not made from user_id and agent_id (evaluation spaces, names with two or more `/`) are admin-only; `{user: "*"}` covers plain-named spaces such as `shop-demo`. Unknown fields are rejected. The keys are read once at startup; a broken file makes `cuemem serve` and `cuemem proxy` print `错误：…` and exit 1. Tokens are compared as sha256 digests in constant time against every key.
- Outside its scope a key gets **403** when the request names the user, agent or space (users inside `filters`, the default space of `POST /api/sessions` and the target of an import included), and **404** (or `[]` for history) for memory ids and session ids, exactly as if they did not exist. `/api/spaces` and `/api/v1/spaces` list only the key's spaces, and `POST /api/v1/reset` is admin-only.
- The UI asks to sign in: the password is the token. A scoped key signs in with its name as the username; the admin token accepts any username.
- Browser requests from other origins get 403 on `/api`, `/v1` and `/gradio_api`. On a loopback bind, only the Hosts 127.0.0.1, localhost, [::1] and the bind address itself are accepted; list a reverse proxy's hostnames in `CUEMEM_ALLOWED_HOSTS`.
- MCP is not behind these keys (see below).

### `/api/v1`: the memory API

One route per SDK method. Bodies and query parameters carry the same names and defaults as the SDK keywords, and the response is what `Memory` returns.

| Route | SDK method |
|---|---|
| `POST /api/v1/memories` with `{messages, user_id, agent_id, run_id, metadata, expiration_date, timestamp, infer, wait}` | `add` |
| `POST /api/v1/memories/search` with `{query, user_id, agent_id, run_id, limit, mode, at, router, lam, filters, top_k, min_level}` | `search` |
| `GET /api/v1/memories?user_id&agent_id&run_id&include_superseded&valid_at&known_at&limit&top_k&filters=<JSON>` | `get_all` |
| `GET /api/v1/memories/{id}` | `get` (404 when missing) |
| `PUT /api/v1/memories/{id}` with `{data, metadata}` | `update` |
| `GET /api/v1/memories/{id}/history` | `history` |
| `DELETE /api/v1/memories/{id}?dry_run=` | `delete` |
| `DELETE /api/v1/memories?user_id&agent_id&run_id` | `delete_all` |
| `POST /api/v1/reset` | `reset` (admin token only) |
| `POST /api/v1/answer` with `{question, user_id, agent_id, mode, at, router, lam}` | `answer` |
| `GET /api/v1/profile?user_id&agent_id` | `profile`, as `{"profile": ...}` |
| `POST /api/v1/runs` with `{run_id, user_id, agent_id}` | `open_run`, as `{"run_id": ...}` |
| `POST /api/v1/runs/end` with `{run_id, user_id, agent_id, wait}` | `end_run` |
| `POST /api/v1/consolidate` with `{user_id, agent_id, rebuild}` | `consolidate` |
| `POST /api/v1/wait` | `wait`, as `{"ok": true}` |
| `GET /api/v1/export?user_id&agent_id&derived=` | `export` (404 for an unknown space) |
| `POST /api/v1/import` with `{data, user_id, agent_id, wait}` | `load` |
| `GET /api/v1/spaces` | `spaces` |

Images in `messages` must be data URLs: local paths and `file://` get 400, because the server does not read files from its own disk.

Errors have the shape `{"error": {"code", "type", "message"}}`. `type` is the exception the SDK raised (`KeyError`, `ValueError`, `TypeError`, a `cuemem.errors` class, or the provider exception's class), and embedder errors add `upstream_status`. 401, 422 and unknown routes under `/api/v1` use the same shape (the 401 message is the Chinese text of `/api`).

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | ValueError or TypeError: no user_id or agent_id, a bad filter, an image path, a malformed export… |
| 401 | `unauthorized` | Missing or wrong token |
| 402 | `no_balance` | The model provider's account is out of balance |
| 403 | `forbidden` | Outside the key's scope; `reset` without the admin token |
| 404 | `not_found` | Unknown memory id, unknown space on export, another tenant's id, unknown route |
| 409 | `conflict` | `ConcurrentWriteError` |
| 422 | `invalid_request` or `content_filtered` | A malformed request body (type `ValueError`); the provider's content filter rejected the request |
| 429 | `rate_limited` | Still rate-limited after retries, with `Retry-After` |
| 502 | `upstream_error` | Other provider errors (5xx, connection, authentication) and other CueMem errors |
| 503 | `unavailable` | The provider is unavailable, with `Retry-After` |
| 504 | `timeout` | The provider timed out |

Examples (the token is `CUEMEM_API_TOKEN` or a scoped key):

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

### `/api`: sessions, chat and the memory browser

These routes take raw space ids (`alice`, `shop/alice`; a "/" may also be written `%2F`) and answer errors as `{"detail": <Chinese text>}`.

| Route | What it does |
|---|---|
| `GET /api/health` | `{"ok", "version", "open_sessions"}`; no token needed |
| `GET /api/spaces` | The spaces (filtered by the key's scope) |
| `POST /api/sessions` with `{space, at}` | Start a chat session with the agent (default space: `--space`) |
| `POST /api/sessions/{session_id}/messages`, form fields `text`, `images` (files), `mode`, `router`, `lam` | One agent turn: recall, answer with citations, write back; returns the reply and its trace |
| `POST /api/sessions/{session_id}/end?wait=` | End the session and hand it to extraction |
| `POST /api/spaces/{space}/recall` with `{query, mode, router, lam, at}` | Recall without answering; the controller picks the level (L0 gives only the profile) |
| `GET /api/spaces/{space}/profile`, `/facts?current_only=`, `/entities`, `/graph`, `/media`, `/media/{media_id}`, `/sessions`, `/status` | Browse the space |
| `POST /api/spaces/{space}/forget` with `{session_ids, turn_ids, fact_ids, media_ids, dry_run}` | Forget, or preview with `dry_run`; `warning` when the rebuild after the deletion failed |
| `GET /api/spaces/{space}/export?derived=` | Export |
| `POST /api/spaces/import?space=`, with the export file as the body | Import (409 for an invalid file, including one without `space_id` when `?space` is not given) |
| `POST /api/spaces/{space}/consolidate` | Run pending extraction now |

Status codes: 404 for an unknown session or image, or an export of an unknown space; 409 for invalid input; 415 for an upload that is not an image; 422 for a bad `mode` or `router`; 401 and 403 as above; and, for provider failures, 429 (with `Retry-After`), 402, 503 (with `Retry-After`), 504, 422 (content filter) and 502 (other provider errors).

## Memory proxy: `/v1`

`cuemem proxy` runs the proxy on its own (default `http://127.0.0.1:8787/v1`, health check at `/health`); `cuemem serve` mounts the same routes at `/v1`. Point any OpenAI SDK at it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8787/v1",
    api_key="<CUEMEM_API_TOKEN, a scoped key, or anything without auth>",
)
client.chat.completions.create(model="gpt-5.4-mini", messages=[...], user="alice")
```

| Flag of `cuemem proxy` | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `8787` | Bind address |
| `--user` | none | The user for requests that name none (otherwise they pass through without memory) |
| `--mode` | `auto` | Recall level: `auto` (the controller), `l0`, `l1` or `l2` |
| `--idle-minutes` | `30` | A session idle this long ends and is extracted |
| `--insecure-no-auth`, `--unknown-model` | | As for `cuemem serve` |

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | Recall, inject the evidence before the last user message, forward to the provider, write back; streaming and non-streaming |
| `POST /v1/responses` | A subset of the Responses API over the same path: non-streaming text input (a string, or message items with `input_text` / `output_text` parts; `developer` is treated as `system`), `instructions`, `max_output_tokens`, `temperature`, `top_p` and `user`. `metadata`, `store`, `parallel_tool_calls`, `service_tier`, `prompt_cache_key` and `safety_identifier` are accepted and ignored. Streaming, tools, `previous_response_id`, structured text formats, images and function-call items get 501 (code `not_implemented`) with `x-should-retry: false`; use `/v1/chat/completions` for them |
| `POST /v1/embeddings` | The configured embedding model (document vectors): `encoding_format` `float` or `base64`, `dimensions` only equal to the model's size, up to 2,048 non-empty string inputs (token arrays get 400). Callers' texts do not touch memory and do not enter the embedding cache. Provider errors map as on the HTTP API (429 with `Retry-After`, 402, 503, 504, 502) |
| `GET /v1/models` | The configured model aliases and the embedding model |

- **Identity:** the `X-CueMem-User` header or the request's `user` field (the header wins), optionally `X-CueMem-Agent`; `--user` is the fallback. A scoped key must cover that user and agent, otherwise an OpenAI-format 403 (type `permission_error`, code `forbidden`) comes before any recall, session or forwarding. Requests that name no user pass through without memory. `/v1/models` and `/v1/embeddings` accept any valid key.
- **Sessions:** `X-CueMem-Session` names one. Without it, a request with no earlier messages, or one after `--idle-minutes` of silence, starts a new session; parallel conversations need their own session headers.
- **Response headers:** `X-CueMem-Level` (`L0`, `L1`, `L2`; `none` when memory was not used, `error` when recall failed and the request went ahead without memory), `X-CueMem-Recalled`, `X-CueMem-Reasons`, `X-CueMem-Session`, and on every response `X-CueMem-Model`, the upstream model actually used. The body is the provider's, unchanged.
- **Models:** `model` is a configured alias or model id; a request without `model` uses the agent role's model. An unknown model also goes to the agent role's model, with a warning logged once per name (`proxy.unknown_model: remap`, the default). With `reject` (the config key, `CUEMEM_PROXY__UNKNOWN_MODEL` or `--unknown-model reject`) it gets 400 with code `model_not_found`, before any recall.
- **Failures:** upstream errors come back as they are, with `retry-after`, `retry-after-ms` and `x-should-retry`, and nothing is written. An unreachable upstream gives an OpenAI-format 502 and a timeout 504. A write-back that fails (say, during an embedding outage) is kept in process memory and written before that session's next write-back, or tried once more before the session ends; it is lost if the process exits.
- Requests must be JSON (`Content-Type: application/json`; 415 otherwise).

```bash
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash","input":"Where should I buy boots?","user":"alice"}' http://127.0.0.1:8787/v1/responses
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.7-text-embedding","input":["hello"]}' http://127.0.0.1:8787/v1/embeddings
```

## MCP server

`cuemem mcp --space <space> [--router learned]` serves memory tools over stdio, one space per process (default space `default`). It does not go through HTTP, and scoped keys do not apply to it. It opens the data directory itself, so with local-mode Qdrant it cannot run while `cuemem serve` holds the same directory. Register it with, for example, `claude mcp add cuemem -- cuemem mcp --space me`, or from a checkout `claude mcp add cuemem -- uv --directory /path/to/cuemem run cuemem mcp --space me`.

| Tool | What it does |
|---|---|
| `remember(text)` | Saves a note; searchable at once, extracted in the background |
| `recall(query, deep=False)` | At least an L1 recall (the controller may go deeper; `deep=True` forces L2); returns the profile, facts and excerpts with ids and dates; no answer model is called |
| `forget(ids)` | Forgets by the ids that `recall` returned and rebuilds; says so when the rebuild after the deletion failed |
| `profile()` | The profile |

Provider failures come back as an English ToolError (rate limit, insufficient balance, unavailable, timeout, content filter), with "in about N s" when the provider said so.

## Environment variables

Any configuration field can be set as `CUEMEM_<SECTION>__<FIELD>` (nested with double underscores), for example `CUEMEM_ROLES__READER__MODEL=qwen3.8-flash`. Variables can also go in the `.env` file of the data directory, except `CUEMEM_HOME`, `CUEMEM_ENV_FILE` and `CUEMEM_CONFIG`, which locate it and must be set in the shell.

| Variable | Meaning |
|---|---|
| `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, `OPENAI_API_KEY` | Provider keys (`cuemem configure` writes them to `.env`) |
| `CUEMEM_HOME` | The data directory |
| `CUEMEM_ENV_FILE` | Another `.env` file |
| `CUEMEM_CONFIG` | A config file to use instead of the packaged default |
| `CUEMEM_PRESET` | A preset: `openai`, `ollama`, `deepseek`, `qwen`, or a YAML path |
| `CUEMEM_PEAK_POLICY` | `wait`, `warn` or `allow` for batch jobs such as evaluations; the SDK and the interactive commands (`ask`, `serve`, `proxy`, `mcp`) use `warn` regardless, and the SDK's `peak_policy` argument overrides it |
| `CUEMEM_API_TOKEN` | The admin token of `cuemem serve` and `cuemem proxy`; `MemoryClient`'s default `api_key` |
| `CUEMEM_API_KEYS_FILE` / `CUEMEM_API_KEYS` | Scoped keys: the path of a YAML file, or the same content inline (set only one) |
| `CUEMEM_ALLOWED_HOSTS` | Extra hostnames accepted as Host and Origin (comma-separated; `*` turns both checks off) |
| `CUEMEM_VECTOR_STORE__URL` | A Qdrant server instead of local mode |
| `CUEMEM_EMBEDDING__BACKEND` | For example `hash`: offline hash embeddings (tests and demos only) |
| `CUEMEM_PROXY__UNKNOWN_MODEL` | `remap` (default) or `reject` |
| `CUEMEM_RETRIEVAL__CONTROLLER` | Another controller model file |
| `CUEMEM_EVAL_RAW_SESSION_IDS`, `CUEMEM_LANGMEM_MAX_LOST_SHARE`, `CUEMEM_AMEM_K` | Evaluation only (see D-067 and D-065) |

## Command line

`cuemem --help` lists the commands: `configure`, `doctor`, `ingest`, `export`, `import`, `consolidate`, `forget`, `ask`, `search-image`, `spaces`, `usage`, `serve`, `proxy`, `chat`, `demo`, `migrate-vectors`, `mcp`, `inspect`, `data` and `eval`. `cuemem <command> --help` describes the options; help and messages are in Chinese. `cuemem --preset <name> <command>` applies a preset to any command.
