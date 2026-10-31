# API reference

English · [中文](../api.md)

Every way a program talks to CueMem: the Python SDK (`Memory`, `AsyncMemory`), the remote client (`MemoryClient`, `AsyncMemoryClient`), the HTTP API of `cuemem serve` (`/api` and the Mem0-shaped `/api/v1`), the OpenAI-compatible memory proxy (`/v1`), the MCP server (stdio and HTTP), the environment variables and the install extras. Signatures are those of CueMem 0.1.0 (`src/cuemem/sdk.py`, `client.py`, `api/`). The decisions behind them are in the [decision records](decisions.md): D-061 (SDK), D-062 (proxy), D-063 (export), D-056 (deployment and tokens), D-069 (the second audit round, which added most of `/api/v1`, `MemoryClient`, scoped keys, metadata, `update` and `reset`), D-070 (the third, which added `status`, MCP over HTTP, request size limits, idle timeouts, typed clients and the Mem0 call shapes below) and D-071 (the fourth, after the repository went public: per-key limits, the Qdrant API key, stricter checks for public binds, non-ASCII identity headers and runs that survive a restart).

Language: the SDK's return values and its own errors, `/api/v1`, the proxy and the MCP tools are in English. The command line, the older `/api` routes, the 401 message and errors raised deeper in the core (a missing key, a locked vector store, a configuration mistake) are in Chinese.

## Install

| What you need | Install |
|---|---|
| SDK, CLI, HTTP API, memory proxy, MCP | `pip install "cuemem @ git+https://github.com/FlickYan/cuemem"` |
| + local embeddings with no key (fastembed; the `deepseek` preset needs it) | extra `local` |
| + the browser UI of `cuemem serve` (Gradio) | extra `ui` |
| + evaluation charts and the controller and gate training scripts (matplotlib, scikit-learn) | extra `eval` |
| + framework integrations | extras `openai-agents`, `langgraph`, `anthropic`, or all three: `integrations` |

Extras combine: `pip install "cuemem[ui,local] @ git+https://github.com/FlickYan/cuemem"`. CueMem is not on PyPI yet, and a package named `cuemem` on PyPI is not this project: always install from GitHub. Python 3.12 or later. In a source checkout, `uv sync` installs `ui` and `eval` through the dev group; `uv sync --no-dev` needs `--extra ui --extra eval`.

- Without `ui`, `cuemem serve` serves `/api`, `/api/v1`, `/mcp`, `/docs` and `/v1` only and prints how to install the UI (`pip install "cuemem[ui] @ git+https://github.com/FlickYan/cuemem"`, or `uv sync --extra ui` in a checkout).
- Without `eval`, `cuemem eval run` (except `--build-only`) and `cuemem eval report` exit with code 1 and the same kind of hint; `eval split`, `eval probe`, `eval audit` and `data download` work.
- The Docker image installs `ui` and `local` by default; `docker build --build-arg EXTRAS= .` builds an API-only image. In the Docker image the remedy for a missing extra is to rebuild, e.g. `docker build --build-arg EXTRAS="--extra ui --extra local" -t cuemem .` (with `--extra eval` and so on added), and that is what the hint says there: pip inside the container installs elsewhere and the image has no git.
- pip installs that include `cuemem[ui]` still resolve Pillow < 13, because gradio declares `pillow<13` (`cuemem[local]` alone resolves fastembed 0.9.1, which allows Pillow 13); Pillow 12 has known security advisories that are fixed only in 13. Servers that accept images from other people should use the Docker image or `uv sync --locked` from a checkout, which install Pillow 13. Image bytes sent to models are the same under both.
- The package ships `py.typed`, so mypy and pyright check your calls into CueMem (and may report new, correct errors after an upgrade).

## Memory spaces and identity

`user_id` and `agent_id` together name one memory space, and spaces share nothing:

| Arguments | Space |
|---|---|
| `user_id="alice"` | `alice` |
| `agent_id="shop"` | `agent/shop` (the agent's own memory) |
| `user_id="alice", agent_id="shop"` | `shop/alice` (what agent shop remembers about alice) |

- A "/" or "%" inside an id is percent-escaped, so `user_id="agent/x"` and `agent_id="x"` never meet. At least one of the two ids is required (ValueError otherwise). `cuemem.sdk.space_for(user_id, agent_id)` computes the space; `space_owner(space)` is its inverse.
- `run_id` is one session (one conversation) inside a space. Without it, each `add` is a session of its own.
- A memory id (the `id` of a fact, excerpt or image) already belongs to one space, so `get`, `history`, `update`, `delete` and `media` take only the id.

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
| `background` | Extract facts in background threads (default on; `consolidation.background_workers` threads, 4 by default, one space at a time each) |
| `peak_policy` | What to do when a call falls in DeepSeek's weekday peak hours: `warn` (the SDK's default: call at peak price and record it), `wait` (pause until off-peak, with a warning on the `cuemem.sdk` logger), `allow` (call without a note) |
| `http_client` | An httpx2 client for the model and embedding providers (tests use it to inject a fake network) |

- **Unknown keys in `config` raise ValueError**, naming each dotted key and the nearest known one, before anything is opened. A Mem0-format config gets a migration hint: `llm` and `embedder` become `preset="openai"` or `models` / `roles`, `custom_instructions` becomes `Memory(instructions=...)`, and a `vector_store` provider config becomes `{"vector_store": {"url": "http://host:6333"}}`. `preset`, `home`, `config_path` and `env_file` belong in keyword arguments. Before, unknown keys were ignored and the DeepSeek + Qwen defaults ran. A `Settings` object, `CUEMEM_*` variables and YAML files stay lenient.
- `from_config(config)` is `Memory(config)`: CueMem's config dict, not Mem0's `MemoryConfig`.
- `from_system` wraps a memory system that is already open in the process (the HTTP service and the proxy use it). Local-mode Qdrant lets only one process open a data directory, so `Memory()` on a directory that `cuemem serve` holds raises `StorageLockedError`; use `MemoryClient` against the server instead.

### Methods

| Method | Returns |
|---|---|
| `add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, expiration_date=None, timestamp=None, infer=True, wait=False)` | `{"results": [events], "session_id", "turn_ids", "status", ...}` |
| `search(query, *, user_id=None, agent_id=None, run_id=None, limit=None, mode="auto", at=None, router="learned", lam=None, filters=None, top_k=None, min_level="l1", show_expired=False)` | `{"results", "images", "profile", "level", "see", "reasons"}` |
| `answer(question, *, user_id=None, agent_id=None, mode="auto", at=None, router="learned", lam=None)` | `{"answer", "evidence_sufficient", "citations", "level", "reasons"}` |
| `get_all(*, user_id=None, agent_id=None, run_id=None, include_superseded=False, valid_at=None, known_at=None, limit=None, filters=None, top_k=None, show_expired=False)` | `{"results": [facts]}` |
| `get(memory_id)` | a fact, excerpt or image item, or `None` |
| `media(media_id)` | the original image file as bytes, or `None` |
| `history(memory_id)` | the version chain of a fact, oldest first, each version with `"event": "ADD"` or `"UPDATE"`; `[]` for other ids |
| `update(memory_id, text=None, *, metadata=None, data=None)` | `{"message", "id", "results", "session_id", "turn_ids", ...}` |
| `delete(memory_id, *, dry_run=False)` | a forget report |
| `delete_all(*, user_id=None, agent_id=None, run_id=None)` | a forget report |
| `reset()` | `{"message", "spaces_deleted", "turns_deleted", "images_deleted"}` |
| `profile(*, user_id=None, agent_id=None)` | the profile text |
| `status(*, user_id=None, agent_id=None)` | `{"pending", "failed", "background", "errors"}` |
| `open_run(run_id=None, *, user_id=None, agent_id=None)` | the run id |
| `end_run(run_id, *, user_id=None, agent_id=None, wait=False)` | `{"session_id", "turns", "consolidation"}` |
| `run(run_id=None, *, user_id=None, agent_id=None, wait=False)` | a context manager that yields the run id |
| `close_idle_runs(idle_s, *, now=None)` | the number of runs ended (local only; `cuemem serve` calls it every minute) |
| `consolidate(*, user_id=None, agent_id=None, rebuild=False)` | a consolidation report |
| `wait()` | `None`, once the background threads have done the queued extraction |
| `export(*, user_id=None, agent_id=None, derived=True)` | a `cuemem.space` export (a JSON-serialisable dict) |
| `load(data, *, user_id=None, agent_id=None, wait=False)` | an import report |
| `spaces()` | one dict per space: `space_id`, `created_at`, `sessions`, `turns`, `units`, `current_facts`, `facts`, `entities` |
| `close()` | `None`; ends this object's open runs (their extraction is registered) and closes the memory system |

**`add`.** `messages` is a string, one OpenAI-format message, or a list of them; only user and assistant messages are stored. Images go in content blocks (`{"type": "image_url", "image_url": {"url": ...}}`) as data URLs or local paths; http(s) image URLs are not fetched (ValueError), and an image with more pixels than Pillow's `Image.MAX_IMAGE_PIXELS` (89,478,485 by default) is refused with ValueError "Image too large", with nothing stored. The turns are written online, with no LLM call, and are searchable at once; facts are extracted in the background.

- `run_id`: write into that session. Calling `add` once per turn with the same `run_id` re-extracts the whole session (and replays derived memory) every time; wrap a multi-turn conversation in `with m.run(...)` instead, which extracts it once when it ends.
- `metadata`: a JSON object with string keys, stored with each turn of this call. It is not shown to the extraction model, so requests do not change. Facts, excerpts and images derived from those turns carry it, and `filters` can select on it. TypeError for a non-dict; ValueError for non-string keys or values that JSON cannot hold.
- `expiration_date`: a date, a datetime or an ISO string; a bare date lasts to the end of that day. Expired memories leave `search`, `get_all` and the profile unless `show_expired=True`; `get` and `history` still return them, and only `delete` removes them. A fact said again later without an expiry does not expire. Expiry is applied when the SDK reads (and so in `/api/v1` and `MemoryClient`); `answer()`, the proxy, MCP, the `/api` routes and the framework integrations still see expired memories until they are deleted.
- `timestamp`: when the messages were said (a datetime, an ISO string or Unix time; default now). Times with a zone are converted to local time.
- `infer=False`: log only (`status: "logged"`); extract later with `consolidate()`. A forget elsewhere in the space does not extract such sessions; `consolidate()`, `add(wait=True)` and `update()` do, because they consolidate the whole space.
- `wait=True`: extract now and return Mem0-style events (`status: "consolidated"`, plus `consolidation`). Not allowed while the run is open (ValueError: end the run with `end_run(..., wait=True)`). If extraction fails after the turns were stored (say, the provider is out of balance), `add` does not raise: it returns `status: "queued"` and an `error`. The turns are stored and their extraction is retried on the next consolidation, so the call must not be repeated.
- An error before anything is written (an embedding outage, a bad image) raises, stores nothing, and the same call can be retried. Once the turns are committed, nothing else raises: a trace that cannot be written (a full disk, a bad traces path) is logged as a warning on the `cuemem.core.trace` logger, and the next turn number is written in the same transaction as the turns.
- `status` is one of `empty` (no message to store), `logged`, `queued` and `consolidated`. A concurrent write that lost its turn numbers raises `ConcurrentWriteError`.

An event is a fact item with `"event": "ADD"` or `"UPDATE"`; an UPDATE also has `previous_memory` (the statements it superseded) and `reason` (`changed`: the world changed; `corrected`: the earlier statement was a mistake).

**`search`.** The retrieval controller decides how deeply to recall. `level` (`L0`, `L1`, `L2`), `see` (whether the controller chose to look at original photos) and `reasons` (in English) say what it chose and why.

- `min_level="l1"` (the default): calling `search` is already a decision to look at memory, so an L0 from the controller is raised to L1, with reasons `["raised from L0: the caller asked to recall"]`, as in MCP and the framework integrations. `min_level="l0"` keeps the controller's decision, and L0 returns only the profile.
- `mode`: `auto` (the controller), or a fixed `l0`, `l1` or `l2`. `router`: `learned` (default), `hybrid` (asks an LLM when the gate is unsure), `rule` or `llm`. `lam`: the cost knob λ (default: the controller model's value; smaller spends more).
- `at`: the question time; relative dates are resolved against it.
- `limit` or `top_k`: truncate `results`; a positive integer (ValueError for 0, negative numbers and booleans). `run_id`: keep only that session's memories. `show_expired=True` also returns expired memories (the profile never shows them).
- `filters`: Mem0 v3's form. `user_id`, `agent_id` and `run_id` may go here instead of the keywords (`user_id` and `agent_id` must be single ids). Every other key is a metadata condition:
  - `{"key": value}` is equality, `{"key": "*"}` means the key is present, and several keys must all hold;
  - `{"key": {"eq" | "ne" | "gt" | "gte" | "lt" | "lte": v, "in" | "nin": [...], "contains" | "icontains": v}}`; when the key is missing, only `ne` and `nin` hold;
  - `{"AND": [cond, ...]}`, `{"OR": [...]}` and `{"NOT": [...]}` (NOT: none of them holds);
  - inside conditions, `user_id`, `agent_id` and `run_id` refer to where the memory lives, for example `{"run_id": {"in": ["r1", "r2"]}}`;
  - an unknown operator raises ValueError.
- Retrieval runs over the whole space, and the session, metadata and expiry filters are applied afterwards, so there can be fewer results than `limit`. The profile is the whole space's, without expired facts.
- `score` only ranks results: at L1 it is a rank-fusion score (about 0.016 for the top hit), not a similarity, so there is no `threshold` (Mem0's default cutoff of 0.1 would remove everything). Mem0's `rerank` corresponds to `mode="l2"`.

Result items carry a `type`:

| `type` | Fields |
|---|---|
| `fact` | `id`, `memory` (the statement), `subject`, `predicate`, `object`, `category`, `importance`, `status`, `valid_from`, `valid_to`, `created_at`, `invalidated_at`, `superseded_by`, `session_id`, `source_turn_ids`, `metadata`, `expiration_date`; `score`, `user_id`, `agent_id` or `space_id` where they apply |
| `excerpt` | `id`, `memory` (the conversation round), `score`, `created_at`, `session_id`, `source_turn_ids`, `metadata`, `expiration_date` (from `get`: no `score`, plus `space_id`) |
| `image` (in `images`) | `id`, `memory` (the caption), `path` (a path on the server; fetch the bytes with `media`), `created_at`, `attached`, `metadata`, `expiration_date` (from `get`: `attached` is false, plus `space_id`) |

`metadata` merges the source turns' metadata (a later turn wins on the same key); `expiration_date` is the latest expiry among the sources, or `None` if any source has none.

**`answer`.** Retrieves and answers with the configured reader. `mode` also accepts `agent` (the reader decides by itself when to call a `memory_search` tool) and `hatch` (the controller recalls first, then the reader may search more). `citations` are `{"ref", "id", "memory"}`.

**`get_all`.** Current facts by default. `include_superseded=True` adds the replaced versions; `valid_at="2025-06"` replays what was true in the world then; `known_at=<datetime>` replays what the system believed then. `run_id`, `filters`, `limit` / `top_k` and `show_expired` work as in `search`; expired facts are left out by default.

**`get`, `media` and `history`.** Every id that `search` returns works by id. `get` returns a fact, an excerpt (a conversation round) or an image item, expired ones included, or `None` for an unknown id. `media(media_id)` returns the original file of an image as bytes (`None` for other ids, or when the file is gone). `history` returns a fact's version chain; excerpts and images have no versions (`[]`).

**`update`.** Facts are never rewritten in place: derived memory is a function of the raw log. `update` logs the user turn `"Correction: <text>"` in a new `edit-…` session placed after the original, and consolidates; the fact extracted from it supersedes the original as `corrected`, without asking an LLM. The new fact inherits the original's metadata (keys in `metadata` override) and its expiry; `history()` shows both versions, and deleting the correction restores the original.

- `text` is Mem0 2.x's name and `data` its older alias: pass one of them (both, if different, raise ValueError). `metadata` alone raises ValueError, because metadata lives on the source turns and goes in with a correction.
- `message` is `"Memory updated"`.
- If nothing can be extracted from the text, the original stays as it was, the correction is forgotten again, `turn_ids` is empty and `message` says so.
- If extraction fails after the correction was logged, `message` says so and `error` gives the reason; do not repeat the call.
- KeyError for an unknown id; ValueError when the memory was already replaced (update the current version), when it is an excerpt or image (only facts can be updated), or when the text is empty.

**`delete` and `delete_all`.** Forgetting implements the right to be forgotten. `delete` removes the conversation round the fact came from, follows citations into later replies and finds uncited restatements, then rebuilds derived memory from the rest of the log, so a fact it had replaced becomes current again. It also takes excerpt and image ids, forgetting that round or image. `dry_run=True` lists what would be deleted and deletes nothing. `delete_all` deletes the whole space: log, derived memory, vectors, caches and photos, and also the space's traces (with the query vectors of their questions) and the metadata rows that name it; with `run_id`, only that session.

- The report has `message`, `turns_deleted`, `sessions_affected`, `facts_before`, `facts_after`, `images_deleted` and `restated`, plus `warning` when the deletion happened but rebuilding from the rest failed (it finishes on the next consolidation).
- A forget that fails before anything is deleted (the embedding provider is down, a vector-store or database error) raises and deletes nothing, so it can be retried as is.
- After the deletion commits, cleanup never raises. If a daily trace file cannot be rewritten (for example the disk is full), CueMem deletes that whole day's file instead: other users' debug traces of that day are lost, a warning is logged on `cuemem.core.trace`, and the query vectors of those traces are purged. Files that can be neither rewritten nor deleted are named in the forget trace's `traces_left` attribute and must be removed by hand. If purging the embedding cache fails (for example "database or disk is full", or "database is locked" on the `embed.sqlite` a source checkout shares with evaluations), CueMem logs one warning on `cuemem.memory` and records `embed_cache_left: true` on the forget trace; the delete still succeeds and the purge is not retried. The same holds after `delete_all` and `reset`.
- Besides the questions' query vectors, forgetting purges the query vectors of L2's sub-queries and follow-up queries, of the agent's own memory searches and of the shop demo's catalogue searches, and (with the DashScope multimodal embedder) the text-to-image query vectors and the vectors of deleted images that no other memory uses.
- Open chat sessions on the same memory system (the UI, `/api`) drop deleted content on their next turn after any deletion, whichever entry point made it (`/api/v1` delete, `delete_all` and `reset` included); a session started while the deletion was rebuilding takes the rebuilt profile on its next turn.
- `message` is `"Nothing deleted"`, with a `warning` to run `consolidate(rebuild=True)`, when no turn was deleted but the fact still exists (its source turns are no longer in the log, as an older version could leave behind).
- Deleting a fact or an excerpt forgets the whole round, so other ids from that round then raise KeyError, like any unknown id. A Mem0-style loop `for r in results: m.delete(r["id"])` should catch KeyError.
- Do not call `forget`, `delete_space` or `consolidate` on `m.system` from inside `m.system.reading(space)`: it raises RuntimeError instead of deadlocking.

**`reset`.** Deletes every space in the data directory, including those written by the HTTP service, the proxy and the demo, and ends the open runs. It cannot be undone. Through `/api/v1` with a scoped key, it deletes only the spaces in that key's scope.

**`status`.** The extraction state of one user's (or agent's) memory: `pending` (sessions waiting for extraction); `failed`, one entry per failed session with `session_id`, `attempts`, `error` (its first line) and `retry_at` (when the background worker retries it, as an ISO time; `null` once attempts reach 3, when the pipeline version differs, or without a background worker: run `consolidate()` then); `background` (whether a worker runs); and `errors`, the worker's last 3 errors for this space. Background extraction failures are also logged as a warning on the `cuemem.write.consolidate` logger, once per space and kind of error.

**Runs.** `open_run` starts a conversation: `add(run_id=...)` writes online and is searchable at once, but nothing is extracted until `end_run`, which hands the whole run to extraction (`wait=True` extracts now and returns the report; otherwise `consolidation` is `"queued"`). `with m.run(user_id="alice") as run_id:` does both. The local SDK never ends a run by itself; under `cuemem serve`, `/api/v1` runs idle for 30 minutes are ended (`close_idle_runs`), and the next `add` with the same `run_id` reopens the run: when it ends, the run is extracted again in full and the user's whole space is replayed from the log (D-045), and while that replay runs the user's facts are missing from `search` and `get_all`, for about the extraction latency.

- **Across a restart.** A run left open when the server stops or crashes survives the restart: the next `add` (or `open_run`) with the same `run_id` continues it. Under `cuemem serve` and `cuemem proxy`, nothing else adopts it during the first idle period after start-up, and any that nobody continues are handed to extraction one idle period (`--idle-minutes`) after start-up. A run this process handed over and the client then continues is reopened like an idle-closed run. In the local SDK and the CLI, the first touch of the space hands over runs the caller is not continuing.
- **What is not resumed.** A run that an earlier process had already closed for idleness or handed to extraction is treated as a plain session after the restart, because that process kept both records only in its own memory; so is a run this process handed over and no longer remembers (beyond 10,000 runs): each later `add` re-extracts it and replays the user's whole space (D-045), and the user's facts are missing from `search` and `get_all` during each replay. Clients that pause for long should call `end_run` at natural breaks and open a new run.

**Export and import.** `export` returns format `cuemem.space` version 2: sessions, turns (with `metadata`, `expires_at` and `edit_of` when set) and the original images in base64, plus an optional snapshot of facts and the profile that import ignores. `load` writes the raw log back and extracts derived memory again (`wait=True` does it now). Without `user_id` or `agent_id` it imports into the file's own space, and a file without `space_id` then raises ValueError. Version-1 files still import; if any image or metadata in the file is invalid, nothing is written, and a malformed file (missing keys, wrong types) raises `ValueError("Malformed export: …")`.

### Coming from Mem0

| Mem0 | CueMem |
|---|---|
| `Memory.from_config({"llm": ..., "embedder": ..., "vector_store": ...})` | ValueError with a migration hint: use `preset="openai"` or `models` / `roles`, `{"vector_store": {"url": ...}}` and `Memory(instructions=...)` |
| `update(memory_id, text=..., metadata=...)` | The same; `data=` is an alias, and a metadata-only update raises ValueError |
| `update(..., expiration_date=None)` to clear an expiry | Not supported yet |
| `search(..., threshold=0.1)` | No threshold: scores only rank (rank fusion at L1, about 0.016 for the top hit) |
| `search(..., rerank=True)` | `mode="l2"` |
| `search` / `get_all` showing expired memories | `show_expired=True` |
| `delete(id)` removes one memory | Removes the whole conversation round, so later ids from it raise KeyError |
| `MemoryClient(api_key=..., host=...)` | The same (`host` is an alias of `base_url`); `MemoryClient("m0-...")` with the key as the first argument raises ValueError, because the first argument is the URL |

### `AsyncMemory`

`AsyncMemory(...)` takes the same arguments, and every method is a typed coroutine with the same signature as on `Memory` (it runs in a thread pool; a test compares the two classes both ways). `AsyncMemory.from_system(system)` wraps an open memory system. `async with AsyncMemory() as m:` closes it, and `async with m.run(user_id="alice") as run_id:` is the async run. Other attributes (`.system`, `.settings`) pass through to the sync object.

### Exceptions and logs

`cuemem.errors` defines `CueMemError` and its subclasses `MissingAPIKeyError`, `CacheMissError`, `StructuredOutputError`, `EmbeddingAPIError` (with `status` and `retry_after`), `DatasetChecksumError`, `ConfigMismatchError`, `ConsolidationError`, `ContentFilteredError`, `ConcurrentWriteError`, `StorageLockedError`, `VectorStoreOpenError` (local-mode Qdrant cannot open or create a collection, usually because of too many open files; the message names the collection count, the limits and what to do), `AccessKeysError` and `MemoryAPIError` (with `status`, `code`, `type` and `retry_after`). The SDK also raises KeyError, ValueError, TypeError and RuntimeError as described above, and errors of the OpenAI client library pass through after CueMem's own retries.

Loggers: `cuemem.write.consolidate` (background extraction failures, in English), `cuemem.core.trace` (a trace that cannot be written, or a daily trace file that a forget had to delete whole), `cuemem.memory` (an image file of a forgotten turn that cannot be deleted, or cleanup after a forget's commit that failed), `cuemem.sdk` (peak-hour waits) and `cuemem.api.proxy`.

## Remote client: `MemoryClient`

```python
from cuemem import MemoryClient

m = MemoryClient("http://127.0.0.1:7860", api_key="...")  # instead of Memory()
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

- It calls `/api/v1` of `cuemem serve`. Every public method of `Memory` except `from_config`, `from_system` and `close_idle_runs` exists with an identical signature (a test compares them) and returns what `Memory` returns on the server, so switching between local and remote is a one-line change. Several processes, or apps on other machines, can share one server this way.
- `base_url` defaults to `http://127.0.0.1:7860`; `host` is an alias (Mem0's name), and giving both raises ValueError. It must start with `http://` or `https://` (ValueError at construction, without echoing the value, which might be a token); a trailing `/api/v1`, as in the curl examples below, is accepted.
- `api_key` falls back to the `CUEMEM_API_TOKEN` environment variable; it can be the admin token or a scoped key. `timeout` is per request in seconds (an `add` with `wait=True` waits for extraction); with your own `http_client`, that client's timeout applies.
- Local image paths (and `file://`) in content blocks are read on the client and sent as data URLs; the server never reads its own disk. `media(id)` fetches an image's bytes from the server.
- `metadata` is checked as in the SDK before it is sent (JSON would silently turn non-string keys into strings), and so are `limit` and `top_k` (JSON `true` would become 1).
- Times (`timestamp`, `at`, `known_at`, `expiration_date`) are sent in ISO format. Naive times are read in the server's local time, so include a time zone when client and server differ.
- It raises the same exceptions as `Memory`: KeyError, ValueError, TypeError and the `cuemem.errors` types; an `EmbeddingAPIError` keeps `status` and `retry_after`. Other HTTP failures (a wrong token, an unreachable server, 403 outside a key's scope, other provider errors) raise `MemoryAPIError` with `status` (None when the server is unreachable), `code` (see the error codes below), `type` (the server's exception class, such as `RateLimitError`) and `retry_after` (the Retry-After header, for 429 and 503). A 404 without an error type adds that `base_url` must be the root of `cuemem serve`; a refused connection asks whether `cuemem serve` runs there. There is no client-side retry: retrying a POST can write twice. `get()` of another tenant's id returns `None`, as for a missing id.
- `close()` (or `with MemoryClient(...) as m:`) ends the runs this client opened. A run opened remotely and never ended is ended by `cuemem serve` after 30 idle minutes (`--idle-minutes`) and reopened by the next `add` with the same `run_id`; a run still open when the server restarts is continued by the next `add` with its `run_id` (see Runs above).
- The framework integrations need a local `Memory` or `AsyncMemory`: passing a `MemoryClient` raises TypeError at construction.
- The module imports only httpx2 and `cuemem._async`, not the memory stack.

`AsyncMemoryClient(...)` is the async variant, with the same typed coroutines as `AsyncMemory`.

## HTTP API: `cuemem serve`

`cuemem serve` (default `http://127.0.0.1:7860`) serves, in one process and over one memory system: `/api` (sessions with the agent, chat, browsing, forgetting), `/api/v1` (the Mem0-shaped memory API), `/v1` (the memory proxy), `/mcp` (MCP over Streamable HTTP), `/docs` (OpenAPI) and, with the `ui` extra, the Gradio UI at `/`.

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `127.0.0.1` / `7860` | Bind address |
| `--space` | `shop-demo` | The UI's default space |
| `--scenario` | `shop` in a source checkout, `plain` when installed | The demo shopping assistant or a plain assistant |
| `--router` / `--lam` | `learned` / the controller model's | The controller used by chat |
| `--no-ui` | off | API, proxy and MCP only |
| `--no-mcp` | off | Do not serve MCP at `/mcp` |
| `--idle-minutes` | `30` | Chat sessions, `/api/v1` runs and proxy sessions idle this long end and go to extraction (must be > 0); also the delay after start-up before sessions left open by an earlier process are handed to extraction |
| `--insecure-no-auth` | off | Start on a non-loopback address without any token; the Host must then be an IP address, localhost, the bind hostname or a name in `CUEMEM_ALLOWED_HOSTS`. With a token set, it only skips the 16-character check |
| `--insecure-weak-token` | off | Allow tokens shorter than 16 characters on a non-loopback address |
| `--unknown-model` | `proxy.unknown_model` (`remap`) | What the proxy does with an unknown model: `remap` or `reject` |

Every `cuemem` command raises its process's soft limit on open files to the hard limit (at most 65,536). Local-mode Qdrant keeps one file open per collection (one per memory space, two with photos) and opens all of them at start-up, so on a server with many users use a Qdrant server (`cuemem migrate-vectors --to URL`, then `CUEMEM_VECTOR_STORE__URL`). `cuemem doctor` prints the collection count against the limit.

### Authentication and scoped keys

- With no token configured there is no authentication. That is allowed only on a loopback address: a non-loopback bind without a token refuses to start unless `--insecure-no-auth` is given; the Host must then be an IP address, localhost, the bind hostname or a name in `CUEMEM_ALLOWED_HOSTS` (400 otherwise, against DNS rebinding).
- On a non-loopback address every token must be at least 16 characters (`cuemem.api.auth.MIN_TOKEN_CHARS`): the admin token and plaintext `token:` entries alike, every Docker container included. A shorter one makes `cuemem serve` and `cuemem proxy` exit with code 2 before the memory system opens, naming the entry but never printing the token, unless `--insecure-weak-token` is given (in compose, add it to `command:`). `--insecure-no-auth` skips this check too: with a token configured, the server then starts with the short token, which is still required, and, as on any bind with a token, does not check the Host. Generate tokens with `openssl rand -hex 32`. `sha256:` entries cannot be measured and are not checked.
- The server speaks plain HTTP. On a non-loopback bind with tokens, start-up prints a notice that the token (which is also the UI password) and the memories cross the network unencrypted; outside a trusted network put a TLS reverse proxy in front, as described in the README's [Public deployment](../../README.md#public-deployment). CueMem does not rate-limit sign-ins or failed tokens; do that in the reverse proxy.
- `CUEMEM_API_TOKEN` is the admin token. Send `Authorization: Bearer <token>`. `GET /api/health` needs no token. `/api` and `/api/v1` check the token before reading the request body, so an unauthenticated request gets 401 whatever its body.
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
    - name: alice-app-limited         # optional per-key limits (all three are optional)
      sha256: <64 hex digits>
      scopes: [{user: alice}]
      rate_per_minute: 60             # POST, PUT and DELETE requests per minute
      models: [qwen3.8-flash]         # the models the proxy may use for this key
      max_output_tokens: 2000         # the proxy's output cap for this key
  ```

  A value is an id or a list of ids; `*` matches any id and `acme:*` is a prefix. `{agent: A}` covers only agent A's own memory. `agent/x` always counts as agent x's own memory, so scoped keys cannot use an agent literally named `agent`. Spaces not made from user_id and agent_id (evaluation spaces, names with two or more `/`) are admin-only; `{user: "*"}` covers plain-named spaces such as `shop-demo`. Unknown fields are rejected. The keys are read once at startup; a broken file makes `cuemem serve` and `cuemem proxy` print `错误：…` and exit 1. Tokens are compared as sha256 digests in constant time against every key.
- **A scoped key limits whose memory a caller reaches, not what it spends.** Any valid key makes the server call the operator's providers: `/v1/chat/completions` and `/v1/responses` (with or without a user; `model`, `n`, `max_tokens` and switches such as `enable_thinking` are forwarded as sent), `/v1/embeddings` (up to 2,048 inputs), chat on `/api`, `POST /api/v1/answer`, `/mcp`, and every write (extraction later sends about 2 characters upstream per byte written). Give keys only to parties you would trust with that budget, set spend limits at the provider, rate-limit by the `Authorization` header in a reverse proxy (for example nginx `limit_req_zone $http_authorization ...`), and lower `CUEMEM_MAX_BODY_BYTES` to bound what one write can cost.
- **Optional per-key limits** (all positive integers or a non-empty list; without them nothing changes and forwarded bodies are byte-identical):
  - `rate_per_minute`: at most this many requests other than GET, HEAD and OPTIONS (POST, PUT, DELETE) per minute on `/api`, `/api/v1`, `/v1` and `/mcp`; searches and answers over POST, `/v1/embeddings` and every MCP call (Streamable HTTP is POST) count too, and GET requests are not counted. It is a token bucket of that capacity, refilled at `rate_per_minute / 60` per second. Beyond it: 429 with `Retry-After` (the seconds until the next request is allowed), code `rate_limited` under `/api/v1`, a Chinese `{"detail"}` under `/api`, and `{"detail": "Too many requests for this access key: ..."}` under `/v1` and `/mcp`.
  - `models`: the memory proxy (`/v1/chat/completions`, streaming included, and `/v1/responses`) refuses other models with an OpenAI-format 403 (type `permission_error`, param `model`, code `forbidden`) before any recall, session or forwarding. A listed alias admits only that alias, with its catalogue settings: `models: [deepseek-flash]` does not admit `deepseek-flash-thinking`. A listed model id that is not itself an alias admits every alias of that id. For such keys the catalogue's `extra_body` replaces the caller's values, so a cheap model cannot be switched to thinking. `/v1/embeddings` is not restricted by `models`.
  - `max_output_tokens`: the proxy caps `max_tokens` and `max_completion_tokens` at this value (and sets it when the request gives neither), and an `n` other than 1 gets 400 (param `n`). On its own it does not pin caller switches such as `enable_thinking` or `thinking`, which are forwarded as sent; only `models` replaces them with the catalogue's values.
  - Not covered: UI sessions (the agent role's model, `max_tokens` 1200), the size of one write (bounded only by `CUEMEM_MAX_BODY_BYTES`) and provider fields a caller adds (`enable_search` and the like). A daily spending budget per key is not implemented.
- Outside its scope a key gets **403** when the request names the user, agent or space (users inside `filters`, the default space of `POST /api/sessions` and the target of an import included), and **404** (or `[]` for history) for memory ids and session ids, exactly as if they did not exist. `/api/spaces` and `/api/v1/spaces` list only the key's spaces. `POST /api/v1/reset` with a scoped key deletes only the spaces in its scope (a wildcard tenant key wipes the whole tenant).
- The UI asks to sign in: the password is the token. A scoped key signs in with its name as the username; the admin token accepts any username. Files sent through the UI must have been uploaded through it: paths to other files on the server are refused.
- Browser requests from other origins get 403 on `/api`, `/v1`, `/mcp`, `/gradio_api` and the UI's sign-in `/login` (including `Origin: null`; same-origin and Origin-less sign-ins work as before). On a loopback bind, only the Hosts 127.0.0.1, localhost, [::1] and the bind address itself are accepted. With `--insecure-no-auth` on a non-loopback bind, any IP-literal Host, localhost and the bind hostname are accepted. Otherwise 400; list other hostnames (a reverse proxy's, `nas.local`, a compose service name used by another container) in `CUEMEM_ALLOWED_HOSTS`. Authenticated non-loopback binds do not check the Host.
- MCP at `/mcp` uses the same tokens and scopes (see below); the stdio `cuemem mcp` does not go through HTTP and has no tokens.

### Request size limits

Request bodies are checked before they are read, on `cuemem serve` and on `cuemem proxy`:

| Path | Limit | Set with |
|---|---|---|
| `/api/spaces/import`, `/api/v1/import` | 1 GiB (the admin token or no authentication); `CUEMEM_MAX_BODY_BYTES` for scoped keys | `CUEMEM_MAX_IMPORT_BYTES` |
| `/api/…`, `/v1/…`, `/gradio_api/upload` | 64 MiB | `CUEMEM_MAX_BODY_BYTES` |
| everything else (the UI's sign-in and event routes, `/mcp`) | 8 MiB | fixed |

- Above the limit the answer is 413 (code `payload_too_large` under `/api/v1`); for a scoped key's import the message says that larger imports need the admin token. Both variables take a positive number of bytes (`67108864`, not `64MB`); any other value makes serve and proxy print `错误：…` and exit 1.
- A parsed JSON body takes about 24 times its size in memory, so the request bodies one scoped key has in flight together are capped at `CUEMEM_MAX_BODY_BYTES` too: above that, 429 with `Retry-After: 1` (code `rate_limited` under `/api/v1`; `MemoryClient` raises `MemoryAPIError` with `retry_after="1"`, and the OpenAI SDK retries by itself). One key can then hold about 25 × the body limit (about 1.6 GB at the default). Many keys at once and the UI's event routes are not counted per key: size the server's memory for them, or lower `CUEMEM_MAX_BODY_BYTES`. Admin and no-auth deployments are not budgeted.
- Images with more pixels than Pillow's `Image.MAX_IMAGE_PIXELS` (89,478,485 by default) are refused: 400 on `/api/v1`, 409 on `/api` (chat checks every upload before recall or any model call), and a proxy write-back that carries one is dropped alone. Decodes above 16 million pixels run two at a time.
- Images must be JPEG, PNG, WEBP, GIF, BMP or TIFF (phone MPO photos count as JPEG). Other formats (AVIF, XBM, ICO, …) and files that are not images are refused before any decoder runs: 400 on `/api/v1` with the message `Not a valid image (accepted formats: JPEG, PNG, WEBP, GIF, BMP, TIFF).` (it used to be 500), 409 on `/api`, a ValueError from the SDK (it used to be `PIL.UnidentifiedImageError`), and an import with such an attachment fails as a whole. A proxy or MCP write-back carrying such an image is dropped alone. Images already stored in another format keep working.

### `/api/v1`: the memory API

One route per SDK method. Bodies and query parameters carry the same names and defaults as the SDK keywords, and the response is what `Memory` returns.

| Route | SDK method |
|---|---|
| `POST /api/v1/memories` with `{messages, user_id, agent_id, run_id, metadata, expiration_date, timestamp, infer, wait}` | `add` |
| `POST /api/v1/memories/search` with `{query, user_id, agent_id, run_id, limit, mode, at, router, lam, filters, top_k, min_level, show_expired}` | `search` |
| `GET /api/v1/memories?user_id&agent_id&run_id&include_superseded&valid_at&known_at&limit&top_k&show_expired&filters=<JSON>` | `get_all` |
| `GET /api/v1/memories/{id}` | `get` (404 when missing) |
| `PUT /api/v1/memories/{id}` with `{text, metadata}` (or `data` instead of `text`) | `update` |
| `GET /api/v1/memories/{id}/history` | `history` |
| `DELETE /api/v1/memories/{id}?dry_run=` | `delete` |
| `DELETE /api/v1/memories?user_id&agent_id&run_id` | `delete_all` |
| `GET /api/v1/media/{id}` | `media`: the image bytes with their Content-Type (404 for other ids or outside the key's scope) |
| `POST /api/v1/reset` | `reset` (a scoped key: only its spaces, message "All memories in this key's scope deleted") |
| `POST /api/v1/answer` with `{question, user_id, agent_id, mode, at, router, lam}` | `answer` |
| `GET /api/v1/profile?user_id&agent_id` | `profile`, as `{"profile": ...}` |
| `GET /api/v1/status?user_id&agent_id` | `status` |
| `POST /api/v1/runs` with `{run_id, user_id, agent_id}` | `open_run`, as `{"run_id": ...}` |
| `POST /api/v1/runs/end` with `{run_id, user_id, agent_id, wait}` | `end_run` |
| `POST /api/v1/consolidate` with `{user_id, agent_id, rebuild}` | `consolidate` |
| `POST /api/v1/wait` | `wait`, as `{"ok": true}` |
| `GET /api/v1/export?user_id&agent_id&derived=` | `export` (404 for an unknown space) |
| `POST /api/v1/import` with `{data, user_id, agent_id, wait}` | `load` |
| `GET /api/v1/spaces` | `spaces` |

Images in `messages` must be data URLs: local paths and `file://` get 400, because the server does not read files from its own disk.

Every `/api/v1` error has the shape `{"error": {"code", "type", "message"}}` (`V1ErrorBody`). `type` is the exception the SDK raised (`KeyError`, `ValueError`, `TypeError`, a `cuemem.errors` class, or the provider exception's class), and embedder errors add `upstream_status`. 401, 413, 422, unknown routes, cross-site browser requests and a wrong Host header under `/api/v1` use the same shape (the 401 message is the Chinese text of `/api`).

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | ValueError or TypeError: no user_id or agent_id, a bad filter, `limit` below 1, an image path, an image too large, in an unsupported format or not an image, a malformed export…; a Host header that is not accepted (the message names `CUEMEM_ALLOWED_HOSTS`) |
| 401 | `unauthorized` | Missing or wrong token |
| 402 | `no_balance` | The model provider's account is out of balance |
| 403 | `forbidden` | Outside the key's scope; a cross-site browser request |
| 404 | `not_found` | Unknown memory id, unknown space on export, another tenant's id, unknown route |
| 409 | `conflict` | `ConcurrentWriteError` |
| 413 | `payload_too_large` | The request body is above the size limit |
| 422 | `invalid_request` or `content_filtered` | A malformed request body (type `ValueError`); the provider's content filter rejected the request |
| 429 | `rate_limited` | Still rate-limited after retries, a scoped key's bodies in flight above the limit, or a scoped key's `rate_per_minute` used up; with `Retry-After` |
| 502 | `upstream_error` | Other provider errors (5xx, connection, authentication) and other CueMem errors |
| 503 | `unavailable` | The provider is unavailable, with `Retry-After` |
| 504 | `timeout` | The provider timed out |

`/openapi.json` (and `/docs`) gives every `/api/v1` response a named schema (`FactItem`, `ExcerptItem` and `ImageItem` discriminated on `type`, `SearchResult`, `AddResult`, `ForgetResult` or `DryRunResult`, `StatusResult`, `ExportDocument`, …) and documents each error status as `V1ErrorBody`. A snapshot of the `/api/v1` part is committed as [`docs/openapi-v1.json`](../openapi-v1.json); regenerate it with `uv run python scripts/openapi_v1.py` (`--check` exits 1 when it is stale, and the tests run it). There is no hand-written TypeScript client; generate types and call the API with them:

```bash
npx openapi-typescript docs/openapi-v1.json -o cuemem.d.ts   # then openapi-fetch with Authorization: Bearer <token>
```

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
  -d '{"text":"I live in Munich"}'
curl -s $CUEMEM/memories/<id>/history -H "Authorization: Bearer $TOKEN"
curl -s "$CUEMEM/status?user_id=alice" -H "Authorization: Bearer $TOKEN"
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
| `GET /api/spaces/{space}/profile`, `/facts?current_only=`, `/entities`, `/graph`, `/media`, `/media/{media_id}`, `/sessions`, `/status` | Browse the space (`/status` lists only this space's background errors) |
| `POST /api/spaces/{space}/forget` with `{session_ids, turn_ids, fact_ids, media_ids, dry_run}` | Forget, or preview with `dry_run`; `warning` when the rebuild after the deletion failed |
| `GET /api/spaces/{space}/export?derived=` | Export |
| `POST /api/spaces/import?space=`, with the export file as the body | Import (409 for an invalid or malformed file, including one without `space_id` when `?space` is not given) |
| `POST /api/spaces/{space}/consolidate` | Run pending extraction now |

A chat session idle for `--idle-minutes` (30) is ended and extracted; a later message to it gets 404 saying so, so start a new session (the UI does this by itself and says why).

Status codes: 404 for an unknown or ended session or image, or an export of an unknown space; 409 for invalid input (an image too large included); 413 for a body above the size limit; 415 for an upload that is not an image; 422 for a bad `mode` or `router`; 401 and 403 as above; and, for provider failures, 429 (with `Retry-After`), 402, 503 (with `Retry-After`), 504, 422 (content filter) and 502 (other provider errors).

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
| `--idle-minutes` | `30` | A session idle this long ends and is extracted; also the delay after start-up before sessions left open by an earlier process are handed to extraction |
| `--insecure-no-auth`, `--insecure-weak-token`, `--unknown-model` | | As for `cuemem serve` |

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | Recall, inject the evidence before the last user message, forward to the provider, write back; streaming and non-streaming |
| `POST /v1/responses` | A subset of the Responses API over the same path: non-streaming text input (a string, or message items with `input_text` / `output_text` parts; `developer` is treated as `system`), `instructions`, `max_output_tokens`, `temperature`, `top_p` and `user`. `metadata`, `store`, `parallel_tool_calls`, `service_tier`, `prompt_cache_key` and `safety_identifier` are accepted and ignored. Streaming, tools, `previous_response_id`, structured text formats, images and function-call items get 501 (code `not_implemented`) with `x-should-retry: false`; use `/v1/chat/completions` for them |
| `POST /v1/embeddings` | The configured embedding model (document vectors): `encoding_format` `float` or `base64`, `dimensions` only equal to the model's size, up to 2,048 non-empty string inputs (token arrays get 400). Callers' texts do not touch memory and do not enter the embedding cache. Provider errors map as on the HTTP API (429 with `Retry-After`, 402, 503, 504, 502) |
| `GET /v1/models` | The configured model aliases and the embedding model |

- **Identity:** the `X-CueMem-User` header or the request's `user` field (the header wins), optionally `X-CueMem-Agent`; `--user` is the fallback. A scoped key must cover that user and agent, otherwise an OpenAI-format 403 (type `permission_error`, code `forbidden`) comes before any recall, session or forwarding. Requests that name no user pass through without memory. `/v1/models` and `/v1/embeddings` accept any valid key.
- **Non-ASCII user or agent IDs** go in these headers percent-encoded as UTF-8 in the RFC 8187 form, e.g. `X-CueMem-User: utf-8''%E5%BC%A0%E4%B8%89` for 张三. This is the only form Node clients such as Claude Code can send; raw UTF-8 bytes (curl) are accepted too, and the proxy can also take the ID in the body `user` field. A malformed `utf-8''` value, percent-encoded bytes that are not UTF-8, or a control character in a `utf-8''` or non-ASCII value gets 400 (on the proxy an OpenAI-format `invalid_request_error` with param `user`); other raw bytes that are not UTF-8 are read as Latin-1, and ASCII values pass unchanged.
- **Spending:** a scoped key limits whose memory a caller reaches, not what it spends; `model`, `n`, `max_tokens` and switches such as `enable_thinking` are forwarded as sent unless the key sets `models` or `max_output_tokens` (see [Authentication and scoped keys](#authentication-and-scoped-keys)). A model outside a key's `models` gets an OpenAI-format 403 (type `permission_error`, param `model`), and an `n` other than 1 under `max_output_tokens` gets 400 (param `n`).
- **Sessions:** `X-CueMem-Session` names one. Without it, a request with no earlier messages, or one after `--idle-minutes` of silence, starts a new session; parallel conversations need their own session headers. Idle sessions are also ended by a sweep every minute, in `cuemem proxy` and in the proxy mounted by `cuemem serve`.
- **Response headers:** `X-CueMem-Level` (`L0`, `L1`, `L2`; `none` when memory was not used, `error` when recall failed and the request went ahead without memory), `X-CueMem-Recalled`, `X-CueMem-Reasons`, `X-CueMem-Session`, and on every response `X-CueMem-Model`, the upstream model actually used. The body is the provider's, unchanged.
- **Models:** `model` is a configured alias or model id; a request without `model` uses the agent role's model. An unknown model also goes to the agent role's model, with a warning logged once per name (`proxy.unknown_model: remap`, the default). With `reject` (the config key, `CUEMEM_PROXY__UNKNOWN_MODEL` or `--unknown-model reject`) it gets 400 with code `model_not_found`, before any recall.
- **Failures:** upstream errors come back as they are, with `retry-after`, `retry-after-ms` and `x-should-retry`, and nothing is written. An unreachable upstream gives an OpenAI-format 502 and a timeout 504. A write-back that fails (say, during an embedding outage) is kept in process memory and written before that session's next write-back, or tried once more before the session ends; it is lost if the process exits. A write-back that carries an image above the pixel limit, or in an unsupported format, is dropped alone. If a memory the proxy injected is forgotten while the upstream model is still answering (for example another client's DELETE during a long stream), the answer is returned to the caller but not stored; the user's message is stored.
- Requests must be JSON (`Content-Type: application/json`; 415 otherwise), and bodies are capped at `CUEMEM_MAX_BODY_BYTES` (413).

```bash
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash","input":"Where should I buy boots?","user":"alice"}' http://127.0.0.1:8787/v1/responses
curl -s -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.7-text-embedding","input":["hello"]}' http://127.0.0.1:8787/v1/embeddings
```

## MCP server

The same four tools are served two ways:

- **Over HTTP, by `cuemem serve` at `/mcp`** (Streamable HTTP, stateless, JSON responses), sharing the server's memory system. It takes the `/api` token (401 with `WWW-Authenticate: Bearer` otherwise). The `X-CueMem-User` and `X-CueMem-Agent` headers choose the space, as in the proxy; without them a scoped key uses its own first user, while the admin token, no-auth mode and wildcard-only keys must send `X-CueMem-User` (400). A user or agent outside the key's scope gets 403 and nothing is written; a cross-site Origin gets 403. Non-ASCII IDs go in these headers as `utf-8''<percent-encoded UTF-8>` (RFC 8187), e.g. `X-CueMem-User: utf-8''%E5%BC%A0%E4%B8%89` for 张三, the only form Node clients such as Claude Code can send; raw UTF-8 bytes are accepted too (other raw bytes are read as Latin-1, and ASCII values pass unchanged), and a malformed `utf-8''` value, percent-encoded bytes that are not UTF-8, or a control character in a `utf-8''` or non-ASCII value get 400 (`{"detail"}`). The body limit is the fixed 8 MiB tier, and the MCP SDK has its own 4 MiB limit. If the app's lifespan has not started (uvicorn always starts it), `/mcp` answers 503. `cuemem serve --no-mcp` turns it off. Register it with, for example:

  ```bash
  claude mcp add --transport http cuemem http://127.0.0.1:7860/mcp \
    --header "Authorization: Bearer $KEY" --header "X-CueMem-User: alice"
  ```

- **Over stdio, by `cuemem mcp --space <space> [--router learned]`**, one space per process (default space `default`). It does not go through HTTP, and scoped keys do not apply to it. It opens the data directory itself, so with local-mode Qdrant it cannot run while `cuemem serve` holds the same directory: use serve's `/mcp` then. Register it with `claude mcp add cuemem -- cuemem mcp --space me`, or from a checkout `claude mcp add cuemem -- uv --directory /path/to/cuemem run cuemem mcp --space me`. Start-up errors (a missing key, a vector store that cannot open) go to stderr with exit code 1.

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
| `CUEMEM_ALLOWED_HOSTS` | Extra hostnames accepted as Host and Origin (comma-separated; `*` turns both checks off); also needed for DNS names of a token-less non-loopback server |
| `CUEMEM_MAX_BODY_BYTES` | Body limit in bytes for `/api`, `/v1` and UI uploads, and a scoped key's bodies in flight (default 67108864, 64 MiB) |
| `CUEMEM_MAX_IMPORT_BYTES` | Body limit in bytes for imports with the admin token or without authentication (default 1073741824, 1 GiB) |
| `CUEMEM_TRACES_RETENTION_DAYS` | Keep daily trace files this many days (an integer ≥ 1; unset keeps them forever). `cuemem serve` and `cuemem proxy` delete older files at start-up and then daily, with the query vectors of their questions; SDK-only apps can call `memory.system.prune_traces(days)` |
| `CUEMEM_CONSOLIDATION__BACKGROUND_WORKERS` | How many spaces the background worker consolidates at once (default 4; one space is still one at a time) |
| `CUEMEM_VECTOR_STORE__URL` | A Qdrant server instead of local mode (with `http://` or `https://`) |
| `CUEMEM_VECTOR_STORE__API_KEY` | API key of the Qdrant server (sent as the `api-key` header; the server's `QDRANT__SERVICE__API_KEY` or a Qdrant Cloud key), also by `cuemem migrate-vectors`. Empty or unset = no key. Read from the environment or `.env` only; there is no command-line flag. With an `http://` URL, qdrant-client warns once per client "Api key is used with an insecure connection.": harmless on loopback and on the compose network; use `https://` for a remote or Cloud server. `Memory(config={"vector_store": {"url": ..., "api_key": ...}})` takes it too |
| `CUEMEM_EMBEDDING__BACKEND` | For example `hash`: offline hash embeddings (tests and demos only) |
| `CUEMEM_PROXY__UNKNOWN_MODEL` | `remap` (default) or `reject` |
| `CUEMEM_RETRIEVAL__CONTROLLER` | Another controller model file |
| `GRADIO_ANALYTICS_ENABLED` | `True` opts in to Gradio's usage analytics and version check (off by default; the UI contacts no third party, and its font, IBM Plex Sans, ships with Gradio) |
| `GRADIO_TEMP_DIR` | Where the UI keeps uploads and gallery copies; by default `cuemem serve` makes a private directory for each start (0700) and deletes it on exit |
| `CUEMEM_EVAL_RAW_SESSION_IDS`, `CUEMEM_LANGMEM_MAX_LOST_SHARE`, `CUEMEM_AMEM_K` | Evaluation only (see D-067 and D-065) |
| `CUEMEM_TEST_QDRANT_URL`, `CUEMEM_TEST_QDRANT_API_KEY` | Tests only: run `tests/test_qdrant_server.py` against that Qdrant server, with that API key when the server requires one |

## Command line

`cuemem --help` lists the commands: `configure`, `doctor`, `ingest`, `export`, `import`, `consolidate`, `forget`, `ask`, `search-image`, `spaces`, `usage`, `serve`, `proxy`, `chat`, `demo`, `migrate-vectors`, `mcp`, `inspect`, `data` and `eval`. `cuemem <command> --help` describes the options; help and messages are in Chinese. `cuemem --preset <name> <command>` applies a preset to any command. Every command first raises the process's soft open-file limit to the hard limit (at most 65,536; never lowered; on macOS it falls back to 10,240 and 4,096); the library and the SDK leave the limit alone, so an app with many local spaces raises it itself (`ulimit -n`). `cuemem doctor` prints the local vector store's collection count against that limit, in yellow with the remedy when it passes half of it.

- A configuration mistake (an unknown preset, or an invalid `CUEMEM_*` value) makes any command print one line, `错误：配置有误：…` (field, reason and the value received, except for secrets), and exit 1, instead of a traceback; `cuemem mcp` prints it on stderr. The SDK still raises ValueError or pydantic's ValidationError.
- File modes: the command line tightens its umask to at least 077 (never loosening it), so every file it writes is 0600 and every directory 0700; the SDK and `MemorySystem.from_settings` create missing data directories at 0700. Existing directories are never changed. On POSIX, `cuemem doctor` also checks the permissions of the data directories, `.env` and `CUEMEM_API_KEYS_FILE` and prints the `chmod 700` / `chmod 600` commands for anything others can read; `serve` and `proxy` warn about the two secret files at start-up. `cuemem configure` replaces `.env` through a 0600 temporary file (a symlinked `.env` keeps its link; the directory holding it must be writable).
- `cuemem doctor` shows `向量库：<url>（API Key 已设/未设）` with a Qdrant server; `cuemem migrate-vectors` sends `CUEMEM_VECTOR_STORE__API_KEY` to the target.
