# API reference

English · [中文](../api.md)

Every way a program talks to CueMem: the Python SDK (`Memory`, `AsyncMemory`), the remote client (`MemoryClient`, `AsyncMemoryClient`), the HTTP API of `cuemem serve` (`/api` and the Mem0-shaped `/api/v1`), the OpenAI-compatible memory proxy (`/v1`), the MCP server (stdio and HTTP), the environment variables and the install extras. Signatures are those of CueMem 0.8.0 (`src/cuemem/sdk.py`, `client.py`, `api/`). The decisions behind them are in the [decision records](decisions.md): D-061 (SDK), D-062 (proxy), D-063 (export), D-056 (deployment and tokens), D-069 (the second audit round, which added most of `/api/v1`, `MemoryClient`, scoped keys, metadata, `update` and `reset`), D-070 (the third, which added `status`, MCP over HTTP, request size limits, idle timeouts, typed clients and the Mem0 call shapes below) and D-071 (the fourth, after the repository went public: per-key limits, the Qdrant API key, stricter checks for public binds, non-ASCII identity headers and runs that survive a restart), D-072 (the fifth: `wait` limits, transport error codes, `valid_at` periods, `cuemem reindex`), D-073 (the sixth: `context`, idempotent `add`, `cuemem keys` with hot reload of the keys file, and a per-key cap on concurrent requests), D-074 (the seventh: `delete_many` and dry runs of `delete_all`, the same `at` forms everywhere, `show_expired` on the browsing routes, optional deletion of expired memories, one process per database, caps on open sessions and retry backlogs) and D-075 (the eighth: parked proxy sessions, room before the open-file limit and `max_spaces`, streamed exports, readiness routes and vector-store error codes, one-line logs, `cuemem --version` and upgrades, `negated` and `cardinality` on fact items, and the Mem0 importer).

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
- `local` on Apple Silicon needs macOS 14 or later (onnxruntime 1.24 and later ship only macOS 14 arm64 wheels). On Intel Macs the lock installs onnxruntime 1.23.2 for `local` (Python 3.12 and 3.13; Python 3.14 is not supported there), and `uv sync` builds cryptography from source, which needs a Rust toolchain; the Docker image avoids both.
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
- `from_system` wraps a memory system that is already open in the process (the HTTP service and the proxy use it). A database can be opened by one process at a time (a lock on `<database>.lock`, in local mode and with a Qdrant server alike), so `Memory()` on a data directory that `cuemem serve` holds raises `StorageLockedError`, naming the holder's pid and command; use `MemoryClient` against the server instead (see [One process per database](#one-process-per-database)). A second `Memory()` on the same data directory in the same process is refused too.

### Methods

| Method | Returns |
|---|---|
| `add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, expiration_date=None, timestamp=None, infer=True, wait=False, idempotency_key=None)` | `{"results": [events], "session_id", "turn_ids", "status", ...}` |
| `search(query, *, user_id=None, agent_id=None, run_id=None, limit=None, mode="auto", at=None, router="learned", lam=None, filters=None, top_k=None, min_level="l1", show_expired=False)` | `{"results", "images", "profile", "level", "see", "reasons"}` |
| `context(query, *, user_id=None, agent_id=None, at=None, mode="auto", router="learned", lam=None, min_level="l1", wrap=True)` | `{"context", "level", "reasons", "recalled", "see", "refs"}` |
| `answer(question, *, user_id=None, agent_id=None, mode="auto", at=None, router="learned", lam=None)` | `{"answer", "evidence_sufficient", "citations", "level", "reasons"}` |
| `get_all(*, user_id=None, agent_id=None, run_id=None, include_superseded=False, valid_at=None, known_at=None, limit=None, filters=None, top_k=None, show_expired=False)` | `{"results": [facts]}` |
| `get(memory_id)` | a fact, excerpt or image item, or `None` |
| `media(media_id)` | the original image file as bytes, or `None` |
| `history(memory_id)` | the version chain of a fact, oldest first, each version with `"event": "ADD"` or `"UPDATE"`; `[]` for other ids |
| `update(memory_id, text=None, *, metadata=None, data=None)` | `{"message", "id", "results", "session_id", "turn_ids", ...}` |
| `delete(memory_id, *, dry_run=False)` | a forget report |
| `delete_all(*, user_id=None, agent_id=None, run_id=None, dry_run=False)` | a forget report (with `dry_run=True` and `run_id`: a preview) |
| `delete_many(memory_ids, *, dry_run=False)` | a forget report summed over the ids, with `deleted` and `missing` |
| `batch_delete(memories)` | the same as `delete_many`, for `memories=[{"memory_id": ...}]` (Mem0's shape) |
| `reset()` | `{"message", "spaces_deleted", "turns_deleted", "images_deleted"}` |
| `profile(*, user_id=None, agent_id=None)` | the profile text |
| `status(*, user_id=None, agent_id=None)` | `{"pending", "failed", "background", "errors"}` |
| `open_run(run_id=None, *, user_id=None, agent_id=None)` | the run id |
| `end_run(run_id, *, user_id=None, agent_id=None, wait=False)` | `{"session_id", "turns", "consolidation"}` |
| `run(run_id=None, *, user_id=None, agent_id=None, wait=False)` | a context manager that yields the run id |
| `close_idle_runs(idle_s, *, now=None)` | the number of runs ended (local only; `cuemem serve` calls it every minute) |
| `consolidate(*, user_id=None, agent_id=None, rebuild=False)` | a consolidation report |
| `wait(*, user_id=None, agent_id=None, timeout=None)` | `None`, once the background threads have done the queued extraction (with ids, only that user's or agent's); TimeoutError after `timeout` seconds (extraction continues in the background); ValueError for a negative `timeout` |
| `export(*, user_id=None, agent_id=None, derived=True)` | a `cuemem.space` export (a JSON-serialisable dict) |
| `load(data, *, user_id=None, agent_id=None, wait=False, format="cuemem")` | an import report (with `format="mem0"`, also `skipped`) |
| `spaces()` | one dict per space: `space_id`, `created_at`, `sessions`, `turns`, `units`, `current_facts`, `facts`, `entities` |
| `close()` | `None`; ends this object's open runs (their extraction is registered) and closes the memory system |

**`add`.** `messages` is a string, one OpenAI-format message, or a list of them; only user and assistant messages are stored. Images go in content blocks (`{"type": "image_url", "image_url": {"url": ...}}`) as data URLs or local paths; http(s) image URLs are not fetched (ValueError), and an image with more pixels than Pillow's `Image.MAX_IMAGE_PIXELS` (89,478,485 by default) is refused with ValueError "Image too large", with nothing stored. The turns are written online, with no LLM call, and are searchable at once; facts are extracted in the background.

- `run_id`: write into that session. Calling `add` once per turn with the same `run_id` re-extracts the whole session (and replays derived memory) every time; wrap a multi-turn conversation in `with m.run(...)` instead, which extracts it once when it ends.
- `metadata`: a JSON object with string keys, stored with each turn of this call. It is not shown to the extraction model, so requests do not change. Facts, excerpts and images derived from those turns carry it, and `filters` can select on it. TypeError for a non-dict; ValueError for non-string keys or values that JSON cannot hold.
- `expiration_date`: a date, a datetime or an ISO string; a bare date lasts to the end of that day. Expired memories leave `search`, `get_all` and the profile unless `show_expired=True`; `get` and `history` still return them, and only `delete` removes them, unless `CUEMEM_EXPIRED_RETENTION_DAYS` is set: then `cuemem serve` and `cuemem proxy` really delete turns N days after they expired (0: every turn that has expired by the time of the run), at start-up and then on the first sweep of each new day, so a turn can outlive its expiry by up to about a day plus N. That is a forget: the turn and the facts, summaries, vectors, images, caches and traces derived from it are removed, and each affected space is replayed once (consolidation calls, mostly served from the LLM cache; the space is held exclusively while it runs). Only the expired turn itself is deleted: later replies that cite it and have no expiry are kept, and a fact said again later without an expiry comes back from that turn. Exports already written are not touched. SDK-only deployments run `cuemem forget --expired [--space S] [--older-than-days N]`. A fact said again later without an expiry does not expire. Expired memories are also left out of `answer()`, the memory proxy's injected `<memory>` block, MCP and HTTP recall (`cuemem mcp`, `/mcp`, `/api/spaces/{space}/recall`), the profile everywhere (`profile()`, the MCP profile tool, `GET /api/spaces/{space}/profile`, the UI's profile pane), the chat agent of `cuemem serve` (its profile, recall and `memory_search` tool) and the Claude, OpenAI Agents and LangGraph integrations (recall, auto-recall, and the Claude memory tool's `profile.md`, `facts.md` and `history/`). On these surfaces expired memories are left out inside each search route, so they do not take candidate slots from valid memories (SDK `search` filters them after retrieval instead, see below). Expiry is judged against the current wall-clock time, not the question time `at`. The raw fact browser (`GET /api/spaces/{space}/facts`, the UI's fact list), the entity graph (`/graph`, the UI's graph pane) and the image gallery (`/media`, the UI's gallery) hide expired memories too unless `show_expired=true` (UI: the "Show expired" checkbox); their rows carry `expired`. The UI's forget picker still lists expired items so they can be deleted. `cuemem ask` leaves them out unless `--show-expired`. `cuemem inspect facts` still lists them, marked `已过期` (`--hide-expired` hides them). `get`, `history` and export still return them. `GET /api/spaces/{space}/entities` hides entities that only expired facts mention (as subject, as a linked entity, or by name or alias in the statement or object; superseded facts count); `?show_expired=true` lists them, and every row has `expired`. Entities that no fact mentions stay listed, since there is nothing to judge them by, and a merged description can still carry wording from an expired session, because descriptions have no per-session provenance. `cuemem inspect entities` marks such entities `已过期` (`--hide-expired` hides them).
- `timestamp`: when the messages were said (a datetime, an ISO string or Unix time; default now). Times with a zone are converted to local time.
- `infer=False`: log only (`status: "logged"`); extract later with `consolidate()`. A forget elsewhere in the space does not extract such sessions, and neither does importing an export of them (see Export and import); `consolidate()`, `add(wait=True)` and `update()` do, because they consolidate the whole space.
- `wait=True`: extract now and return Mem0-style events (`status: "consolidated"`, plus `consolidation`). Not allowed while the run is open (ValueError: end the run with `end_run(..., wait=True)`). If extraction fails after the turns were stored (say, the provider is out of balance), `add` does not raise: it returns `status: "queued"` and an `error`. The turns are stored and their extraction is retried on the next consolidation, so the call must not be repeated.
- An error before anything is written (an embedding outage, a bad image) raises, stores nothing, and the same call can be retried (through `MemoryClient`, only after code `unreachable`, which means the request never reached the server; see below). Once the turns are committed, nothing else raises: a trace that cannot be written (a full disk, a bad traces path) is logged as a warning on the `cuemem.core.trace` logger, and the next turn number is written in the same transaction as the turns.
- `idempotency_key`: a key of your choice (1–255 printable ASCII characters, for example a UUID; ValueError otherwise) that makes a retried add safe. Within 24 hours, the same key with the same request for the same user (space) is written once: the repeat returns `status: "duplicate"` with the original `session_id` and `turn_ids`, `results: []` and `replayed: true`; with `wait=True` (and `infer=True`) it first waits for this user's extraction to finish, as `add(wait=True)` does, and adds `error` if that session's extraction failed (a run that is still open is not extracted). If those turns have been forgotten (`delete`, `delete_all(run_id=...)`, any forget), the repeat returns `status: "forgotten"` (`session_id` None, `turn_ids` []) and writes nothing. The same key with a different request (`messages`, `user_id`, `agent_id`, `run_id`, `metadata`, `expiration_date`, `timestamp` or `infer` differ; `wait` is not compared) raises `IdempotencyConflictError` and writes nothing. Only the key's sha256 is stored. An idempotency key protects a retry for 24 hours, and through forget()/DELETE of the written memories (the retry returns status "forgotten"). delete_all() and reset() remove the user's idempotency records along with everything else, so a retry that arrives after them is a new write. Without a key, `add` behaves exactly as before.
- `status` is one of `empty` (no message to store), `logged`, `queued` and `consolidated`, or, with `idempotency_key`, `duplicate` and `forgotten` (then `replayed` is true). A concurrent write in the same `run_id` that lost its turn numbers raises `ConcurrentWriteError`: nothing is written (no turns, no idempotency record), and the add can be retried.

An event is a fact item with `"event": "ADD"` or `"UPDATE"`; an UPDATE also has `previous_memory` (the statements it superseded) and `reason` (`changed`: the world changed; `corrected`: the earlier statement was a mistake).

**`search`.** The retrieval controller decides how deeply to recall. `level` (`L0`, `L1`, `L2`), `see` (whether the controller chose to look at original photos) and `reasons` (in English) say what it chose and why.

- `min_level="l1"` (the default): calling `search` is already a decision to look at memory, so an L0 from the controller is raised to L1, with reasons `["raised from L0: the caller asked to recall"]`, as in MCP and the framework integrations. `min_level="l0"` keeps the controller's decision, and L0 returns only the profile.
- `mode`: `auto` (the controller), or a fixed `l0`, `l1` or `l2`. `router`: `learned` (default), `hybrid` (asks an LLM when the gate is unsure), `rule` or `llm`. `lam`: the cost knob λ (default: the controller model's value; smaller spends more), a finite number ≥ 0. `mode`, `router`, `min_level` and `lam` are checked first, with the same types as `/api/v1`, even on an empty space: an invalid value raises ValueError with the HTTP message without its `Invalid request: body.` prefix, for example `mode: Input should be 'auto', 'l0', 'l1' or 'l2'`.
- `at`: the question time; relative dates are resolved against it. It takes a datetime, a date (midnight), an ISO 8601 string (an offset or `Z` is converted to local time; compact `20300101` and week forms such as `2030-W01-2` work too) or a Unix timestamp (a number, or a digit-only string that is not ISO; seconds, read as milliseconds above 2e10 in absolute value), the same on all four classes and over HTTP. Anything else raises ValueError starting with `at:`, even for a user with no memories.
- `limit` or `top_k`: truncate `results`; a positive integer (ValueError for 0, negative numbers and booleans). `run_id`: keep only that session's memories. `show_expired=True` also returns expired memories (the profile never shows them).
- `filters`: Mem0 v3's form. `user_id`, `agent_id` and `run_id` may go here instead of the keywords (`user_id` and `agent_id` must be single ids). Every other key is a metadata condition; `created_at`, `updated_at` and `categories` (Mem0's reserved names) also match the memory's own fields (the item's `created_at` ISO string; a fact's category as a one-item list) unless metadata sets the same key:
  - `{"key": value}` is equality, `{"key": "*"}` means the key is present, and several keys must all hold;
  - `{"key": {"eq" | "ne" | "gt" | "gte" | "lt" | "lte": v, "in" | "nin": [...], "contains" | "icontains": v}}`; when the key is missing, only `ne` and `nin` hold;
  - when the value is a list (`categories`, or a list in metadata), `eq`, `ne`, `in` and `nin` compare element by element: `in` holds if any element is listed, `nin` if none is, `eq` / `ne` compare against the elements (an argument that is itself a list compares the whole list); `{"key": value}` behaves like `eq`;
  - ISO strings compare as strings, so `{"created_at": {"gte": "2026-01-01"}}` works, while a date-only upper bound such as `{"lte": "2026-08-01"}` excludes datetimes later that day;
  - `{"AND": [cond, ...]}`, `{"OR": [...]}` and `{"NOT": [...]}` (NOT: none of them holds);
  - inside conditions, `user_id`, `agent_id` and `run_id` refer to where the memory lives, for example `{"run_id": {"in": ["r1", "r2"]}}`;
  - an unknown operator raises ValueError.
- Retrieval runs over the whole space, and the session, metadata and expiry filters are applied afterwards, so there can be fewer results than `limit`. The profile is the whole space's, without expired facts.
- `score` only ranks results: at L1 it is a rank-fusion score (about 0.016 for the top hit), not a similarity, so there is no `threshold` (Mem0's default cutoff of 0.1 would remove everything). Mem0's `rerank` corresponds to `mode="l2"`.

Result items carry a `type`:

| `type` | Fields |
|---|---|
| `fact` | `id`, `memory` (the statement), `subject`, `predicate`, `object`, `negated`, `cardinality`, `category`, `importance`, `status`, `valid_from`, `valid_to`, `created_at`, `invalidated_at`, `superseded_by`, `session_id`, `source_turn_ids`, `metadata`, `expiration_date`; `score`, `user_id`, `agent_id` or `space_id` where they apply |
| `excerpt` | `id`, `memory` (the conversation round), `score`, `created_at`, `session_id`, `source_turn_ids`, `metadata`, `expiration_date` (from `get`: no `score`, plus `space_id`) |
| `image` (in `images`) | `id`, `memory` (the caption), `path` (a path on the server; fetch the bytes with `media`), `created_at`, `attached`, `metadata`, `expiration_date` (from `get`: `attached` is false, plus `space_id`) |

`metadata` merges the source turns' metadata (a later turn wins on the same key); `expiration_date` is the latest expiry among the sources, or `None` if any source has none. `negated` is true for a negative fact ("no longer drinks coffee"): `subject`, `predicate` and `object` then state what is *not* true. `cardinality` is `"single"` (a newer value replaces the old one) or `"multi"` (values accumulate). Servers older than 0.8.0 leave both keys out, so read them with `item.get("negated", False)`; there is no `negated` filter, since other filter keys are metadata conditions.

**`context`.** Returns the same evidence block the memory proxy injects, ready to put in a prompt: the profile, dated facts with their status (current / no longer true / corrected) and time-ordered conversation excerpts, with `<memory` / `</memory` inside memories escaped and the note that says they are information, not instructions, at the top. It makes no model call. With `wrap=True` (the default) the block is wrapped as `<memory>\n...\n</memory>`: the text the proxy puts in front of the last user message, minus the trailing blank line; with `wrap=False` you get the bare block. `context` is an empty string when there is nothing to add (on an empty space, with `reasons` `["no memories yet"]`). `recalled` counts items (the profile counts as one). `refs` maps each label in the block (`f1` for `[f1]`, `m1` for `[m1]`, ...) to a memory id that `get()` accepts. As in `search`, the default `min_level="l1"` raises an L0 choice of the controller to L1 (reasons: `raised from L0: the caller asked to recall`); `min_level="l0"` follows the controller and matches the proxy byte for byte. Expired memories (`expiration_date`) are left out, including from the profile. An invalid `mode`, `router`, `min_level` or `lam` raises ValueError, as in `search` (HTTP 422). Building a prompt from `search()` results instead loses the dates and status and does not escape `</memory>`:

```python
prompt = m.context("Where do I work?", user_id="alice")["context"] + "\n\nWhere do I work?"
```

**`answer`.** Retrieves and answers with the configured reader. `mode` also accepts `agent` (the reader decides by itself when to call a `memory_search` tool) and `hatch` (the controller recalls first, then the reader may search more). `citations` are `{"ref", "id", "memory"}`. `mode` is one of `auto`, `l0`, `l1`, `l2`, `agent` and `hatch` (ValueError otherwise; the evaluation baselines `naive` and `facts` exist only on `MemorySystem.ask` and the CLI), and `router` and `lam` are checked as in `search`.

**`get_all`.** Current facts by default. `include_superseded=True` adds the replaced versions; `valid_at` replays what was true in the world then: `"2025"` (the whole year), `"2025-06"` (the whole month) or `"2025-06-07"` (that day), or a date or datetime, and a year or a month returns the facts that held at any time in it, so successive values can both appear (`valid_at="2026"` can return the old and the new address). A year or month includes a value that ended in that year or month when only the year or month of its end is known; a corrected memory counts from when the value it corrected started (an explicit start date of its own wins), so a date before the correction returns the corrected value. A string with a time is parsed as a full ISO datetime, and one with an offset or `Z` is converted to local time before its date is taken, as a datetime is. Any other form raises ValueError (400 on `GET /api/v1/memories`); `known_at` replays what the system believed then; it takes the same forms as `search`'s `at` (ValueError starting with `known_at:` otherwise). `run_id`, `filters`, `limit` / `top_k` and `show_expired` work as in `search`; expired facts are left out by default.

**`get`, `media` and `history`.** Every id that `search` returns works by id. `get` returns a fact, an excerpt (a conversation round) or an image item, expired ones included, or `None` for an unknown id. `media(media_id)` returns the original file of an image as bytes (`None` for other ids, or when the file is gone). `history` returns a fact's version chain; excerpts and images have no versions (`[]`).

**`update`.** Facts are never rewritten in place: derived memory is a function of the raw log. `update` logs the user turn `"Correction: <text>"` in a new `edit-…` session placed after the original, and consolidates; the fact extracted from it supersedes the original as `corrected`, without asking an LLM. The new fact inherits the original's metadata (keys in `metadata` override) and its expiry; `history()` shows both versions, and deleting the correction restores the original.

- `text` is Mem0 2.x's name and `data` its older alias: pass one of them (both, if different, raise ValueError). `metadata` alone raises ValueError, because metadata lives on the source turns and goes in with a correction.
- `message` is `"Memory updated"`.
- If nothing can be extracted from the text, the original stays as it was, the correction is forgotten again (without replaying the space, since it left nothing derived), `turn_ids` is empty and `message` says so.
- If extraction fails after the correction was logged, `message` says so and `error` gives the reason; do not repeat the call.
- KeyError for an unknown id; ValueError when the memory was already replaced (update the current version), when it is an excerpt or image (only facts can be updated), or when the text is empty.

**`delete` and `delete_all`.** Forgetting implements the right to be forgotten. `delete` removes the conversation round the fact came from, follows citations into later replies and finds uncited restatements, then rebuilds derived memory from the rest of the log, so a fact it had replaced becomes current again. It also takes excerpt and image ids, forgetting that round or image. `dry_run=True` lists what would be deleted and deletes nothing. `delete_all` deletes the whole space: log, derived memory, vectors, caches and photos, and also the space's traces (with the query vectors of their questions), the metadata rows that name it and the Claude memory tool's files in `claude_memory.sqlite`; with `run_id`, only that session. A `run_id` with no turns deletes nothing and returns `message` `"No such run: nothing deleted"` with zero counts (earlier versions answered "Memories of this run deleted"). `delete_all(run_id=..., dry_run=True)` previews it as `delete`'s dry run does; `dry_run` without `run_id` raises ValueError (HTTP 400), because deleting a whole user has no preview. Users with more turns than SQLite takes parameters (32,766) are deleted too: the queries are batched.

- The report has `message`, `turns_deleted`, `sessions_affected`, `facts_before`, `facts_after`, `images_deleted` and `restated`, plus `warning` when the deletion happened but rebuilding from the rest failed (it finishes on the next consolidation).
- A forget that fails before anything is deleted (the embedding provider is down, a vector-store or database error) raises and deletes nothing, so it can be retried as is.
- After the deletion commits, cleanup never raises. If a daily trace file cannot be rewritten (for example the disk is full), CueMem deletes that whole day's file instead: other users' debug traces of that day are lost, a warning is logged on `cuemem.core.trace`, and the query vectors of those traces are purged. Files that can be neither rewritten nor deleted are named in the forget trace's `traces_left` attribute and must be removed by hand. If purging the embedding cache fails (for example "database or disk is full", or "database is locked" on the `embed.sqlite` a source checkout shares with evaluations), CueMem logs one warning on `cuemem.memory` and records `embed_cache_left: true` on the forget trace; the delete still succeeds and the purge is not retried. The same holds after `delete_all` and `reset`.
- Besides the questions' query vectors, forgetting purges the query vectors of L2's sub-queries and follow-up queries, of the agent's own memory searches and of the shop demo's catalogue searches, and (with the DashScope multimodal embedder) the text-to-image query vectors and the vectors of deleted images that no other memory uses.
- Open chat sessions on the same memory system (the UI, `/api`) drop deleted content on their next turn after any deletion, whichever entry point made it (`/api/v1` delete, `delete_all` and `reset` included); a session started while the deletion was rebuilding takes the rebuilt profile on its next turn.
- `message` is `"Nothing deleted"`, with a `warning` to run `consolidate(rebuild=True)` (and, if search still misses conversation excerpts afterwards, `cuemem reindex`), when no turn was deleted but the fact still exists (its source turns are no longer in the log, as an older version could leave behind).
- Deleting a fact or an excerpt forgets the whole round, so other ids from that round then raise KeyError, like any unknown id. Each `delete` also replays the log, and a replay can give surviving facts new ids, so a loop over ids from an earlier `get_all` can hit KeyError on facts it never deleted. Use `delete_many` for several ids; a Mem0-style loop `for r in results: m.delete(r["id"])` should catch KeyError.
- **`delete_many(memory_ids, *, dry_run=False)`** forgets several memories (fact, excerpt or image ids) with one forget and one rebuild per memory space: every id is resolved first, then each space is forgotten once. At most 100 ids after removing duplicates (`cuemem.sdk.DELETE_MANY_MAX`; ValueError beyond, since each deleted fact still costs one restatement check with a cheap model); a single string raises TypeError. Unknown ids are listed in `missing`, not raised (over HTTP, ids outside the key's scope too); `deleted` lists the ids deleted (in a dry run, those that would be). It returns `message`, `deleted`, `missing`, `turns_deleted`, `sessions_affected`, `facts_before`, `facts_after`, `images_deleted` and `restated`, summed over the spaces, and at most one `warning`; with `dry_run=True`, `message`, `deleted`, `missing`, `turns`, `sessions_affected`, `cascaded`, `restated`, `facts` and `images`. Across several spaces it deletes space by space, so after a failure the earlier spaces are already deleted. `batch_delete(memories=[{"memory_id": ...}, ...])` is Mem0's form of the same call (without a dry run).
- Forgetting also deletes the Claude memory tool's files (`claude_memory.sqlite` in the data directory) whose notes came from the deleted turns, and deleting a space deletes all of its files; nothing is created when the file does not exist. A store at a custom `store=` path is purged by `cuemem.integrations.claude.delete_all` or the next time the tool opens that space.
- A forget whose sessions are deleted whole and left nothing derived (no facts, no created or linked entities, no images, no citations from other turns, no pending, failed or stale jobs in the space) skips the replay; the result is the same as with one. Since D-075 it also replays when another session of the space grew through `add(..., infer=False)` after it was extracted (its content no longer matches its job), and a session written only with `infer=False` (no job at all) takes the fast path when it is not open and has no open mark.
- Do not call `forget`, `delete_space` or `consolidate` on `m.system` from inside `m.system.reading(space)`: it raises RuntimeError instead of deadlocking.

**`reset`.** Deletes every space in the data directory, including those written by the HTTP service, the proxy and the demo (and the Claude memory tool's files in `claude_memory.sqlite`), and ends the open runs. It cannot be undone. Through `/api/v1` with a scoped key, it deletes only the spaces in that key's scope.

**`status`.** The extraction state of one user's (or agent's) memory: `pending` (sessions waiting for extraction); `failed`, one entry per failed session with `session_id`, `attempts`, `error` (its first line) and `retry_at` (when the background worker retries it, as an ISO time; `null` once attempts reach 3, when the pipeline version differs, or without a background worker: run `consolidate()` then); `background` (whether a worker runs); and `errors`, the worker's last 3 errors for this space. Background extraction failures are also logged as a warning on the `cuemem.write.consolidate` logger, once per space and kind of error.

**Runs.** `open_run` starts a conversation: `add(run_id=...)` writes online and is searchable at once, but nothing is extracted until `end_run`, which hands the whole run to extraction (`wait=True` extracts now and returns the report; otherwise `consolidation` is `"queued"`). `with m.run(user_id="alice") as run_id:` does both. The local SDK never ends a run by itself; under `cuemem serve`, `/api/v1` runs idle for 30 minutes are ended (`close_idle_runs`), and the next `add` with the same `run_id` reopens the run: when it ends, the run is extracted again in full and the user's whole space is replayed from the log (D-045), and while that replay runs the user's facts are missing from `search` and `get_all`, for about the extraction latency. Under `cuemem serve` reopening a run this way (or resuming one left open by an earlier process) counts as opening a run: when the space already has `CUEMEM_MAX_OPEN_SESSIONS` open runs, the `add` gets 429 `rate_limited` (`Retry-After: 60`; `SessionLimitError` from `MemoryClient`) and writes nothing, and the run stays reopenable, so a retry after a slot frees continues it without re-extraction. An `add` whose `run_id` would not reopen anything (a plain session) is never refused.

- **Across a restart.** A run left open when the server stops or crashes survives the restart: the next `add` (or `open_run`) with the same `run_id` continues it. Under `cuemem serve` and `cuemem proxy`, nothing else adopts it during the first idle period after start-up, and any that nobody continues are handed to extraction one idle period (`--idle-minutes`) after start-up. A run this process handed over and the client then continues is reopened like an idle-closed run, but `open_run` and `end_run` drop that record: after `end_run`, an `add` with the same `run_id` is a plain session (`add(wait=True)` consolidates). Runs held for the client this way are also skipped by a rebuild (`consolidate(rebuild=True)`, or the automatic rebuild after a deploy changed the model or the consolidation config). In the local SDK and the CLI, the first touch of the space hands over runs the caller is not continuing.
- **What is not resumed.** A run that an earlier process had already closed for idleness or handed to extraction is treated as a plain session after the restart, because that process kept both records only in its own memory; so is a run this process handed over and no longer remembers (beyond 10,000 runs): each later `add` re-extracts it and replays the user's whole space (D-045), and the user's facts are missing from `search` and `get_all` during each replay. Clients that pause for long should call `end_run` at natural breaks and open a new run.

**Export and import.** `export` returns format `cuemem.space` version 2: sessions, turns (with `metadata`, `expires_at` and `edit_of` when set) and the original images in base64, plus an optional snapshot of facts and the profile that import ignores. `load` writes the raw log back and extracts derived memory again (`wait=True` does it now). Without `user_id` or `agent_id` it imports into the file's own space, and a file without `space_id` then raises ValueError. Version-1 files still import; if any image or metadata in the file is invalid, nothing is written, and a malformed file (missing keys, wrong types) raises `ValueError("Malformed export: …")`. A session that was only logged (`add(infer=False)`) and never queued for extraction carries `"infer": false`; importing the file does not queue it either, so the background worker leaves it alone until an explicit `consolidate()`, `load(wait=True)` or `cuemem import --consolidate`, as on the source. Open runs never carry the key. The format stays version 2; older CueMem versions ignore the key and extract such sessions, and a non-boolean `infer` is a malformed export. A document that is not a JSON object raises `ValueError("Malformed export: a CueMem export is a JSON object")`.

**Importing from Mem0.** `load(data, format="mem0", user_id=None, agent_id=None, wait=False)` takes Mem0's `get_all` output (a list of memories, `{"results": [...]}`, or the paginated `{"count", "next", "results"}`) and imports it through the same transactional import (`cuemem.portable.from_mem0` converts it; any other `format` raises ValueError). Each Mem0 memory becomes one session, `mem0-<sha256 of the id and updated_at (or created_at)>[:16]`, with one turn: the time is `updated_at` (else `created_at`), Mem0's timezone-aware times converted to local time; the role is `assistant` when Mem0's `role` or `attributed_to` says so, else `user`; the speaker is `actor_id`; the metadata is Mem0's plus `mem0_id`, `mem0_run_id` and `mem0_updated_at`; `expiration_date` becomes the expiry. Re-importing the same export adds nothing and calls no model; a memory edited in Mem0 since then arrives as a new session, and reconciliation supersedes the old fact, which stays as history. Without `user_id` or `agent_id` every memory must belong to the same Mem0 user and agent (ValueError otherwise, listing the pairs); with them, everything goes into that space. The whole input is checked first: a bad id, text, metadata, timestamp or `expiration_date` raises `ValueError("Malformed Mem0 export: …")` and nothing is written. Memories with empty text are skipped and counted in the report's `skipped`. Export from Mem0 with `m.get_all(filters={"user_id": U}, top_k=<large>, show_expired=True)`: with its defaults (`top_k=20`, `show_expired=False`) Mem0 silently truncates the list and leaves expired memories out. Each memory costs one extraction call when consolidated; Mem0's history (earlier versions) is not imported; facts come from CueMem's extractor, so their wording differs from Mem0's, and Mem0's third-person notes are read as user turns.

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
| `batch_delete(memories=[{"memory_id": ...}])` | The same on all four classes, plus `delete_many(ids, dry_run=False)`; at most 100 ids, one rebuild per space, unknown ids listed in `missing` |
| Moving existing memories over | `m.load(mem0.get_all(filters={"user_id": U}, top_k=<large>, show_expired=True), format="mem0", user_id=U)`, or `cuemem import FILE --from mem0`; re-imports add nothing (see Importing from Mem0) |
| `MemoryClient(api_key=..., host=...)` | The same (`host` is an alias of `base_url`); `MemoryClient("m0-...")` with the key as the first argument raises ValueError, because the first argument is the URL |
| `filters` on `created_at`, `updated_at`, `categories` | The same, against the memory's own fields; filters on other fact fields (`category`, `predicate`, `status`, …) are metadata conditions and match nothing unless the metadata has that key |
| `add()` returns the extracted events | Only with `wait=True`; without it, `results` is `[]` and `status` is `"queued"` |
| What `update` returns | `id` is the old id and `results` holds the new fact; the old id then belongs to a superseded version |
| `history` rows with `old_memory` / `new_memory` | Full fact items with `event` `ADD` or `UPDATE` |
| `update` / `delete` of an unknown id raise ValueError (`get` returns `None`) | KeyError (`get` returns `None` too) |
| `search` and `get_all` return 20 results by default (`top_k=20` in mem0ai 2.2.1) | No default: everything, unless you pass `limit` or `top_k` |

### `AsyncMemory`

`AsyncMemory(...)` takes the same arguments, and every method is a typed coroutine with the same signature as on `Memory` (it runs in a thread pool; a test compares the two classes both ways). `AsyncMemory.from_system(system)` wraps an open memory system. `async with AsyncMemory() as m:` closes it, and `async with m.run(user_id="alice") as run_id:` is the async run. Other attributes (`.system`, `.settings`) pass through to the sync object.

Cancelling a coroutine (`task.cancel()`, or a timeout of `asyncio.wait_for` or `asyncio.timeout`) does not stop the call: it finishes in a worker thread of the event loop's default executor, so a cancelled `add`, `update`, `delete` or `forget` may still be applied. Pass `idempotency_key` and repeat with the same key, and bound waiting with `wait(timeout=...)` (and the client's `timeout=`) rather than `wait_for`.

### Exceptions and logs

`cuemem.errors` defines `CueMemError` and its subclasses `MissingAPIKeyError`, `CacheMissError`, `StructuredOutputError`, `EmbeddingAPIError` (with `status` and `retry_after`), `DatasetChecksumError`, `ConfigMismatchError` (also for a database written by a newer CueMem: upgrade, or export with the newer version and import here), `ConsolidationError`, `ContentFilteredError`, `ConcurrentWriteError`, `IdempotencyConflictError` (an `idempotency_key` reused for a different add), `StorageLockedError` (the database, in any mode, or the local Qdrant store is in use by another process; the message names its pid and command), `DatabaseOpenError` (the database, the ledger or a cache cannot be opened: a read-only directory, a full disk, a data path blocked by a file, or a file that is corrupt or not a SQLite database, then with a restore-from-backup hint; also a `sqlite3.OperationalError`), `SessionLimitError` (too many open runs or chat sessions in one memory space; HTTP 429), `VectorStoreOpenError` (local-mode Qdrant cannot open or create a collection, usually because of too many open files, or its `meta.json` cannot be read; the message says what to do, for the latter `cuemem doctor --repair-vectors`), `VectorStoreFullError` (local mode refuses to create a new space's collection near the open-file limit, with `collections` and `limit`; HTTP 503, code `capacity`), `AccessKeysError` and `MemoryAPIError` (with `status`, `code`, `type`, `retry_after` and `sent`). The SDK also raises KeyError, ValueError, TypeError and RuntimeError as described above, and errors of the OpenAI client library pass through after CueMem's own retries.

Loggers: `cuemem.write.consolidate` (background extraction failures, in English), `cuemem.core.trace` (a trace that cannot be written, or a daily trace file that a forget had to delete whole), `cuemem.memory` (an image file of a forgotten turn that cannot be deleted, or cleanup after a forget's commit that failed), `cuemem.sdk` (peak-hour waits), `cuemem.store.files` (a write to the embedding cache, LLM cache or usage ledger skipped on a full or read-only disk; the database lock turned off) and `cuemem.api.proxy`. The library never configures logging; `cuemem serve`, `cuemem proxy` and `cuemem mcp` do (see [Logs](#http-api-cuemem-serve)).

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
- Times (`timestamp`, `at`, `known_at`, `expiration_date`) are sent in ISO format when they are datetimes or dates, and strings and Unix timestamps as they are; the server reads `at` and `known_at` with the same parser as `Memory`. Naive times are read in the server's local time, so include a time zone when client and server differ.
- It raises the same exceptions as `Memory`: KeyError, ValueError, TypeError and the `cuemem.errors` types; an `EmbeddingAPIError` keeps `status` and `retry_after`. Other HTTP failures (a wrong token, an unreachable server, 403 outside a key's scope, other provider errors) raise `MemoryAPIError` with `status` (None when no response arrived), `code` (see the error codes below), `type` (the server's exception class, such as `RateLimitError`), `retry_after` (the Retry-After header, for 429 and 503) and `sent` (whether the request reached the server: False, True, or None when an HTTP response came back). A 404 without an error type adds that `base_url` must be the root of `cuemem serve`. Without a response, `code` is a client-side code: `unreachable` (`sent=False`: nothing was written and the call can be repeated; a refused connection asks whether `cuemem serve` runs there), `timeout` (`sent=True`: no response within the client timeout, and the request may have been processed), `disconnected` (`sent=True`: the connection broke off mid-request) or `transport` (`sent=True`: any other transport error). There is no client-side retry: repeat a write only after `unreachable`, or after a 429 or 503 with `retry_after`; after `timeout` or `disconnected` it may already be stored. An `add` that carried an `idempotency_key` can be repeated with the same key (it is written once, see `add` above); otherwise check `export` first (an add still queued for extraction is not in `get_all` yet). The message of these two errors says "the request may have been processed (repeat it with the same idempotency_key, or check export)". `get()` of another tenant's id returns `None`, as for a missing id.
- `wait()` asks the server to wait at most the client's read timeout minus 5 s (`cuemem.client.WAIT_MARGIN_S`; half the read timeout if that is larger), also when `timeout` is not given, so a backlog that outlasts it raises TimeoutError (the server's 504) rather than `MemoryAPIError(code="timeout")`; with the default 300 s client the server waits at most 295 s.
- `close()` (or `with MemoryClient(...) as m:`) ends the runs this client opened. A run opened remotely and never ended is ended by `cuemem serve` after 30 idle minutes (`--idle-minutes`) and reopened by the next `add` with the same `run_id`; a run still open when the server restarts is continued by the next `add` with its `run_id` (see Runs above).
- The framework integrations need a local `Memory` or `AsyncMemory`: passing a `MemoryClient` raises TypeError at construction.
- The module imports only httpx2 and `cuemem._async`, not the memory stack.

`AsyncMemoryClient(...)` is the async variant, with the same typed coroutines as `AsyncMemory`. Cancellation behaves as for `AsyncMemory`.

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
| `--insecure-no-auth` | off | Start on a non-loopback address (or on loopback with `CUEMEM_ALLOWED_HOSTS` naming a non-local host) without any token. On a non-loopback bind the Host must then be an IP address, localhost, the bind hostname or a name in `CUEMEM_ALLOWED_HOSTS`; on loopback it must be 127.0.0.1, localhost, [::1], the bind address or a name in `CUEMEM_ALLOWED_HOSTS`, as with a token. With a token set, it only skips the 16-character check |
| `--insecure-weak-token` | off | Allow tokens shorter than 16 characters on a non-loopback address, or on loopback behind a reverse proxy (`CUEMEM_ALLOWED_HOSTS`) |
| `--unknown-model` | `proxy.unknown_model` (`remap`) | What the proxy does with an unknown model: `remap` or `reject` |
| `--access-log/--no-access-log` | off | One log line per request (uvicorn's access log); with it on, uvicorn's own INFO start-up lines appear too |

Every `cuemem` command raises its process's soft limit on open files to the hard limit (at most 65,536). Local-mode Qdrant keeps one file open per collection (one per memory space, two with photos) and opens all of them at start-up, so on a server with many users use a Qdrant server (`cuemem migrate-vectors --to URL`, then `CUEMEM_VECTOR_STORE__URL`). `cuemem doctor` prints the collection count against the limit. Local mode keeps room for the database, images and connections: a write that would create a new collection is refused once `collections + 1` would exceed the soft limit minus a reserve of a tenth of it, at least 256 but at most half (`cuemem.store.vector.reserved_files`; 128 at a soft limit of 256, 256 from 512 up to 2,560). The request gets 503 with code `capacity` (`VectorStoreFullError`; on the proxy an OpenAI-format `server_error` with `x-should-retry: false`, before anything is forwarded), existing spaces keep working, and the server can still restart and delete spaces; `cuemem doctor` turns red at the same line. Delete unused spaces, raise `ulimit -n`, or move to a Qdrant server. A key with `user: X` and no `agent:` admits any `agent_id`, and every new one is a new space with its own collection, so give less-trusted clients `--agent` or `--max-spaces` (see below).

- **Open sessions per space.** `CUEMEM_MAX_OPEN_SESSIONS` (a positive integer, default 64; an invalid value makes `cuemem serve` and `cuemem proxy` exit 1) caps, per memory space and each counted separately, the open `/api` chat sessions (`POST /api/sessions`, the web UI, `cuemem chat`), the memory proxy's sessions and the `/api/v1` runs. At the cap, a new chat session first ends the least recently active session that has no messages yet; if every open session has messages, `POST /api/sessions` answers 429 with `Retry-After: 60` (the UI says so). End a session (`POST /api/sessions/{id}/end`) or wait for the idle sweep. A new proxy session first parks the least recently seen one that holds no write-backs waiting for retry: it stays open (not extracted) and continues under the same id when its caller returns, and it is extracted once after `--idle-minutes` without a request or at shutdown (at most 4,096 parked per process, `cuemem.api.proxy.PARKED_KEPT`; beyond that the oldest parked one is extracted). If every in-memory session holds write-backs, the proxy answers 429. A new `run_id` beyond the cap gets 429 (code `rate_limited`, type `SessionLimitError`, which `MemoryClient` raises), and so does an `add` that would reopen an idle-ended or left-open run (see Runs above); reopening an open run is fine, and other runs are never ended for the caller. The local SDK `Memory` has no cap (`max_open_runs = None`).
- **Stopping.** On SIGTERM, serve and proxy wait at most 20 s for requests in flight, then retry the proxy's buffered write-backs, end chat sessions and parked proxy sessions, wait up to 30 s for background extraction, close the database (checkpointing its WAL) and remove the UI's private temporary directory, then exit with code 143. This holds under systemd, `kill`, `docker run --init`, compose `init: true` and other supervisors as well as in the shipped image (which used to exit 0 after `docker stop`); a second SIGTERM during cleanup ends the process at once, except in the shipped image without `--init`, where cuemem is PID 1, the kernel ignores it and only `docker stop`'s SIGKILL ends the cleanup early. docker-compose gives this 90 s (`stop_grace_period: 90s`); with a single `docker run`, add `--stop-timeout 90`, since Docker's default 10 s kills the server mid-shutdown and buffered write-backs are lost.
- **Health and readiness.** `GET /api/health` (serve) and `GET /health` (proxy) say only that the process is up; they are async, so they answer even when the 40 worker threads are busy. `GET /api/ready` (serve) and `GET /ready` (proxy), both without a token, answer 200 `{"ok": true}` or 503 `{"ok": false, "failing": [...]}` with `Retry-After: 5`, naming only the failing components: `vector_store` (with a Qdrant server, `get_collections` through a separate client with a 2 s timeout; always ready in local mode) and `worker` (a background consolidation thread has died). They check neither the database nor the disk (`cuemem doctor` does). Point load balancers and container health checks at the readiness route; the Docker image's `HEALTHCHECK` does, and ignores `HTTP_PROXY`.
- **Logs.** serve, proxy and `cuemem mcp` write one line per record to stderr, `YYYY-MM-DD HH:MM:SS,mmm LEVEL logger.name: message`, followed by a plain traceback when there is one, whether or not `/mcp` is mounted (so `docker compose logs | grep APIConnectionError` works). `CUEMEM_LOG_LEVEL` sets the level (DEBUG, INFO, WARNING, ERROR or CRITICAL, case-insensitive; default WARNING; any other value exits with code 2); it is read from the process environment, so Docker's `env_file` and compose `environment` pass it but a local `.env` file does not. serve logs no requests unless `--access-log`; proxy logs every request unless `--no-access-log`.
- **Full disks.** On a full or read-only disk the server keeps answering: a failed write to the embedding cache, the LLM response cache or the usage ledger logs one warning on `cuemem.store.files` and is skipped until writes succeed again, and a paid call is recorded in the ledger before its reply is cached. Evaluations stay strict and fail. The caches keep every new question's vector and reply (about 6–7 KB per distinct question with 1,024-dimension vectors); only `CUEMEM_TRACES_RETENTION_DAYS` prunes query vectors.

### Authentication and scoped keys

- With no token configured there is no authentication. That is allowed only on a loopback address that is not exposed: a non-loopback bind without a token refuses to start unless `--insecure-no-auth` is given, and so does a loopback bind whose `CUEMEM_ALLOWED_HOSTS` names a host that is not loopback-like, or `*` (a reverse-proxy setup; loopback-like entries are 127.x, ::1, localhost and *.localhost, with or without port or brackets). With `--insecure-no-auth` on a non-loopback bind the Host must be an IP address, localhost, the bind hostname or a name in `CUEMEM_ALLOWED_HOSTS`; on a loopback bind, with or without a token, it must be 127.0.0.1, localhost, [::1], the bind address or a name in `CUEMEM_ALLOWED_HOSTS` (400 otherwise, against DNS rebinding; with `*` in the list the Host is not checked).
- On a non-loopback address, and on loopback behind a reverse proxy as above, every token must be at least 16 characters (`cuemem.api.auth.MIN_TOKEN_CHARS`): the admin token and plaintext `token:` entries alike, every Docker container included. A shorter one makes `cuemem serve` and `cuemem proxy` exit with code 2 before the memory system opens, naming the entry but never printing the token, unless `--insecure-weak-token` is given (in compose, add it to `command:`). `--insecure-no-auth` skips this check too: with a token configured, the server then starts with the short token, which is still required; on a non-loopback bind it then does not check the Host, as with any token there, while a loopback bind checks the Host as above. Generate tokens with `openssl rand -hex 32`. `sha256:` entries cannot be measured and are not checked.
- The server speaks plain HTTP. On a non-loopback bind with tokens (or a loopback one that `CUEMEM_ALLOWED_HOSTS` exposes), start-up prints a notice that the token (which is also the UI password) and the memories cross the network unencrypted; outside a trusted network put a TLS reverse proxy in front, as described in the README's [Public deployment](../../README.md#public-deployment). CueMem does not rate-limit sign-ins or failed tokens; do that in the reverse proxy.
- `CUEMEM_API_TOKEN` is the admin token. Send `Authorization: Bearer <token>`. `GET /api/health` and `GET /api/ready` need no token. `/api` and `/api/v1` check the token before reading the request body, so an unauthenticated request gets 401 whatever its body.
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
    - name: alice-app-limited         # optional per-key limits (all five are optional)
      sha256: <64 hex digits>
      scopes: [{user: alice}]
      rate_per_minute: 60             # POST, PUT and DELETE requests per minute
      models: [qwen3.8-flash]         # the models the proxy may use for this key
      max_output_tokens: 2000         # the proxy's output cap for this key
      max_concurrent: 4               # requests in progress at once (default 8)
      max_spaces: 3                   # memory spaces this key may hold (unlimited by default)
  ```

  A value is an id or a list of ids; `*` matches any id and `acme:*` is a prefix. `{agent: A}` covers only agent A's own memory. `agent/x` always counts as agent x's own memory, so scoped keys cannot use an agent literally named `agent`. Spaces not made from user_id and agent_id (evaluation spaces, names with two or more `/`) are admin-only; `{user: "*"}` covers plain-named spaces such as `shop-demo`. `{user: alice}` also admits any agent id, and each new `agent_id` is a new space with its own vector collection (an open file in local mode); give keys for less-trusted clients an `agent` or `max_spaces`. Unknown fields are rejected; a broken file makes `cuemem serve` and `cuemem proxy` print `错误：…` and exit 1 at start-up. Tokens are compared as sha256 digests in constant time against every key.
- **Changing keys without a restart.** `cuemem serve` and `cuemem proxy` re-read the `CUEMEM_API_KEYS_FILE` file while running (checked at most once a second; changes apply within about a second, no restart). Inline `CUEMEM_API_KEYS` and `CUEMEM_API_TOKEN` are still read once at start-up. A reload is refused, and the previous keys stay active with one log warning that names only key names, if the file is missing, unreadable or invalid, if it would leave no keys while `CUEMEM_API_TOKEN` is unset (a reload never turns authentication off), or if it contains plaintext tokens shorter than 16 characters in a setup where the server would refuse to start with them. Per-key rate limits are tracked by key name and are not reset by a reload. A UI session signed in with a revoked key loses access, and rotating a key (a new sha256 under the same name, including `cuemem keys revoke NAME` followed by `cuemem keys add NAME`) signs out every UI session that used that name; changing only a key's scopes or limits signs no one out. This assumes one server process, which is how serve and proxy start uvicorn; there is no SIGHUP handler.
- **`cuemem keys`** edits that file, so you never hash a token by hand:

  ```bash
  TOKEN=$(cuemem keys add alice-app --user alice)      # prints the new token once, on stdout only
  cuemem keys add shop-backend --user '*' --agent shop  # one scope: every user under agent shop
  cuemem keys add shop-self --agent shop                # agent shop's own memories
  cuemem keys add capped --user bob --rate-per-minute 60 --models qwen3.8-flash --max-output-tokens 2000 --max-concurrent 4
  cuemem keys add bob-app --user bob --max-spaces 5           # at most 5 spaces (bob's, under any agent)
  cuemem keys list                                      # names, scopes, limits; never tokens or digests
  cuemem keys revoke alice-app                          # running serve/proxy stop accepting it within ~1 s
  ```

  The commands edit the file named by `--file` or `CUEMEM_API_KEYS_FILE` (the data directory's `.env` counts), keep its comments, store only the sha256 (quoted), and rewrite the file atomically with mode 0600, and print the file's absolute path. `keys add` creates the file if it does not exist (its directory must exist; otherwise it exits 1 and prints no token). Running servers read a newly created file only when they were started with `CUEMEM_API_KEYS_FILE` pointing at it (in Docker, the container's path), so start or restart them with that setting; edits to a file a server already watches take effect within about a second. Each `keys add` writes one scope: the `--user` values, plus the `--agent` values when given; `--agent` alone means that agent's own memories; `--user`, `--agent` and `--models` can be repeated. `keys add` with `--user` but neither `--agent` nor `--max-spaces` prints a reminder on stderr that the key admits any agent id. They refuse when the keys are given inline in `CUEMEM_API_KEYS` (`keys list` still lists those). A file whose `keys:` is written flow-style (`keys: [...]`, other than an empty `keys: []`) is left untouched: `keys add` prints the entry to paste and the token and exits 1, and the token is accepted only once the entry is pasted into `keys:`. `keys revoke` removes only that entry's lines; when the entry shares a line with other content, or its `-` sits on a line of its own, it refuses and names the line range to delete by hand. Revoking the last key writes `keys: []` and warns that, without `CUEMEM_API_TOKEN`, running servers keep the old keys until they are stopped. A later `keys add` turns `keys: []` back into a block list and adds to it. The manual way still works: `printf %s "$TOKEN" | sha256sum` and a `sha256:` entry.
- **A scoped key limits whose memory a caller reaches, not what it spends.** Any valid key makes the server call the operator's providers: `/v1/chat/completions` and `/v1/responses` (with or without a user; `model`, `n`, `max_tokens` and switches such as `enable_thinking` are forwarded as sent), `/v1/embeddings` (up to 2,048 inputs), chat on `/api`, `POST /api/v1/answer`, `/mcp`, and every write (extraction later sends about 2 characters upstream per byte written). Give keys only to parties you would trust with that budget, set spend limits at the provider, rate-limit by the `Authorization` header in a reverse proxy (for example nginx `limit_req_zone $http_authorization ...`), and lower `CUEMEM_MAX_BODY_BYTES` to bound what one write can cost.
- **Optional per-key limits** (all positive integers or a non-empty list; without them nothing changes and forwarded bodies are byte-identical):
  - `rate_per_minute`: at most this many requests other than GET, HEAD and OPTIONS (POST, PUT, DELETE) per minute on `/api`, `/api/v1`, `/v1` and `/mcp`; searches and answers over POST, `/v1/embeddings` and every MCP call (Streamable HTTP is POST) count too, and GET requests are not counted. It is a token bucket of that capacity, refilled at `rate_per_minute / 60` per second. Beyond it: 429 with `Retry-After` (the seconds until the next request is allowed), code `rate_limited` under `/api/v1`, a Chinese `{"detail"}` under `/api`, and `{"detail": "Too many requests for this access key: ..."}` under `/v1` and `/mcp`.
  - `models`: the memory proxy (`/v1/chat/completions`, streaming included, and `/v1/responses`) refuses other models with an OpenAI-format 403 (type `permission_error`, param `model`, code `forbidden`) before any recall, session or forwarding. A listed alias admits only that alias, with its catalogue settings: `models: [deepseek-flash]` does not admit `deepseek-flash-thinking`. A listed model id that is not itself an alias admits every alias of that id. For such keys the catalogue's `extra_body` replaces the caller's values, so a cheap model cannot be switched to thinking. `/v1/embeddings` is not restricted by `models`.
  - `max_output_tokens`: the proxy caps `max_tokens` and `max_completion_tokens` at this value, and always writes the cap under the parameter name the model's catalogue entry uses (`max_tokens_param`; `max_tokens` except for the OpenAI preset's models): with neither given it is the cap, and sending only `max_completion_tokens` to a DeepSeek or Qwen model also forwards `max_tokens` with the clamped value; and an `n` other than 1 gets 400 (param `n`). On its own it does not pin caller switches such as `enable_thinking` or `thinking`, which are forwarded as sent; only `models` replaces them with the catalogue's values.
  - `max_concurrent`: at most this many requests in progress at once on `/api`, `/api/v1`, `/v1` and `/mcp` (any method). Without it the limit is `CUEMEM_MAX_CONCURRENT_PER_KEY` (default 8); see below.
  - `max_spaces`: at most this many memory spaces in the key's scope. Once it holds that many, a write that would create a new space gets 429 `rate_limited` (the message names the cap; writes to its existing spaces still work): `/api/v1` add, runs, consolidate and import; `/api` session start, import and consolidate; the proxy (an OpenAI-format `rate_limit_error` with `x-should-retry: false`, before any recall or forwarding); MCP `remember` (a tool error). The admin token and no-auth mode are not limited. Concurrent requests that each create a space can overshoot the cap by one or two. Set it with `cuemem keys add --max-spaces N`; `keys list` shows it.
  - Not covered: UI sessions (the agent role's model, `max_tokens` 1200), the size of one write (bounded only by `CUEMEM_MAX_BODY_BYTES`) and provider fields a caller adds (`enable_search` and the like). A daily spending budget per key is not implemented.
- **Concurrent requests per key.** When keys are configured, each scoped key may have at most 8 requests in progress at once on `/api/`, `/api/v1`, `/v1` and `/mcp` (any method; `CUEMEM_MAX_CONCURRENT_PER_KEY`, a positive integer, or the key's own `max_concurrent`; `cuemem serve` and `cuemem proxy` exit 1 when the variable is invalid). Beyond that the answer is 429 with `Retry-After: 1`: code `rate_limited` under `/api/v1`, the Chinese `{"detail": "这个访问令牌同时在处理的请求太多：最多 N 个（环境变量 CUEMEM_MAX_CONCURRENT_PER_KEY，或令牌的 max_concurrent），请稍后重试"}` under `/api`, and "Too many concurrent requests for this access key: at most N at once (set with CUEMEM_MAX_CONCURRENT_PER_KEY or the key's max_concurrent); retry shortly." under `/api/v1`, `/v1` and `/mcp`. The admin token and the no-auth demo are not limited. The server runs synchronous routes on 40 threads, so several scoped keys together can still take all of them (with 8 per key, five keys can); operators sharing a server among many keys can lower the cap.
- Outside its scope a key gets **403** when the request names the user, agent or space (users inside `filters`, the default space of `POST /api/sessions` and the target of an import included), and **404** (or `[]` for history) for memory ids and session ids, exactly as if they did not exist. `/api/spaces` and `/api/v1/spaces` list only the key's spaces. `POST /api/v1/reset` with a scoped key deletes only the spaces in its scope (a wildcard tenant key wipes the whole tenant).
- The UI asks to sign in: the password is the token. A scoped key signs in with its name as the username; the admin token accepts any username. Files sent through the UI must have been uploaded through it: paths to other files on the server are refused.
- Browser requests from other origins get 403 on `/api/`, `/v1/`, `/gradio_api/`, `/mcp`, `/login` and `/config` (including `Origin: null`; same-origin and Origin-less requests work as before), and so does a request to the UI's index page `/` that carries such an `Origin` (`/config` and `/` hold the signed-in key's name; opening the UI in the address bar or from a link sends no `Origin` and is unaffected). `/logout` cannot be checked by `Origin` (`<img>` and navigations send none): a `/logout` request with `Sec-Fetch-Site: cross-site` gets 403, while same-origin sign-outs and clients that do not send that header (older browsers, curl) work as before; with `*` in `CUEMEM_ALLOWED_HOSTS` this check is off. On a loopback bind, only the Hosts 127.0.0.1, localhost, [::1] and the bind address itself are accepted. With `--insecure-no-auth` on a non-loopback bind, any IP-literal Host, localhost and the bind hostname are accepted. Otherwise 400; list other hostnames (a reverse proxy's, `nas.local`, a compose service name used by another container) in `CUEMEM_ALLOWED_HOSTS`. Authenticated non-loopback binds do not check the Host.
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
| `POST /api/v1/memories` with `{messages, user_id, agent_id, run_id, metadata, expiration_date, timestamp, infer, wait}`, optional header `Idempotency-Key` | `add` (the header is `idempotency_key`: a repeat within 24 hours returns 200 with `status` `duplicate` or `forgotten` and `"replayed": true`; the same key with a different request is 409 `conflict`, type `IdempotencyConflictError`; a malformed key is 400). 429 `rate_limited` when the add would reopen an idle-ended or left-open run while the space has `CUEMEM_MAX_OPEN_SESSIONS` open runs (type `SessionLimitError`, `Retry-After: 60`; nothing is written; retry once a run ends), or create a space beyond the key's `max_spaces` |
| `POST /api/v1/memories/search` with `{query, user_id, agent_id, run_id, limit, mode, at, router, lam, filters, top_k, min_level, show_expired}` | `search` |
| `GET /api/v1/memories?user_id&agent_id&run_id&include_superseded&valid_at&known_at&limit&top_k&show_expired&filters=<JSON>` | `get_all` |
| `GET /api/v1/memories/{id}` | `get` (404 when missing) |
| `PUT /api/v1/memories/{id}` with `{text, metadata}` (or `data` instead of `text`) | `update` |
| `GET /api/v1/memories/{id}/history` | `history` |
| `DELETE /api/v1/memories/{id}?dry_run=` | `delete` |
| `DELETE /api/v1/memories?user_id&agent_id&run_id&dry_run` | `delete_all` (`dry_run=true` needs `run_id`, 400 otherwise) |
| `POST /api/v1/memories/batch-delete` with `{memory_ids, dry_run}` | `delete_many`, as `BatchForgetResult` or `BatchDryRunResult` (operationId `delete_many`; 400 for more than 100 ids; ids outside a scoped key's scope are listed in `missing`, like unknown ones; one write for `rate_per_minute`, dry runs too). `MemoryClient.batch_delete` uses it as well |
| `GET /api/v1/media/{id}` | `media`: the image bytes with their Content-Type (404 for other ids or outside the key's scope) |
| `POST /api/v1/reset` | `reset` (a scoped key: only its spaces, message "All memories in this key's scope deleted") |
| `POST /api/v1/context` with `{query, user_id, agent_id, at, mode, router, lam, min_level, wrap}` | `context`, as `ContextResult`; 403 outside the key's scope, 400 without `user_id` or `agent_id`, 422 for a malformed value; no model call |
| `POST /api/v1/answer` with `{question, user_id, agent_id, mode, at, router, lam}` | `answer` |
| `GET /api/v1/profile?user_id&agent_id` | `profile`, as `{"profile": ...}` |
| `GET /api/v1/status?user_id&agent_id` | `status` |
| `POST /api/v1/runs` with `{run_id, user_id, agent_id}` | `open_run`, as `{"run_id": ...}`; 429 (type `SessionLimitError`, `Retry-After: 60`) when the space already has `CUEMEM_MAX_OPEN_SESSIONS` open runs |
| `POST /api/v1/runs/end` with `{run_id, user_id, agent_id, wait}` | `end_run` |
| `POST /api/v1/consolidate` with `{user_id, agent_id, rebuild}` | `consolidate` |
| `POST /api/v1/wait` with optional `{user_id, agent_id, timeout}` | `wait`, as `{"ok": true}`. With ids the key must cover them (403); without ids the admin token waits for everything and a scoped key only for the spaces it can access. The server waits at most 300 s, then 504 `timeout` (type `TimeoutError`, which `MemoryClient` raises as TimeoutError). The wait holds no server thread: it checks every 0.25 s, so it can return up to 0.25 s after extraction finishes. If the client disconnects, the server stops waiting within 0.25 s and releases the key's concurrency slot (`max_concurrent`) |
| `GET /api/v1/export?user_id&agent_id&derived=` | `export` (404 for an unknown space). The body is streamed (chunked, no `Content-Length`) with the same bytes as before; the server holds about one image's raw bytes at a time. An error after the first byte truncates the body (`MemoryClient` then raises a JSON parse error), and an image forgotten mid-stream is left out |
| `POST /api/v1/import` with `{data, user_id, agent_id, wait, format}` | `load`; `format` is `cuemem` (default) or `mem0`, and with `mem0` `data` may be a JSON array. A Mem0 export is converted before the scope check: a key scoped to another user gets 403, a file that mixes users without `user_id` 400; the report has `skipped` |
| `GET /api/v1/spaces` | `spaces` |

Images in `messages` must be data URLs: local paths and `file://` get 400, because the server does not read files from its own disk. `at` (search, context, answer) and `known_at` (`GET /memories`) are read with the SDK's parser (see `search` above): a body takes a string or a number, a query parameter a string; an invalid value is 400 (it used to be Pydantic's 422), and `"20300101"` now means 2030-01-01 (it used to be read as Unix seconds).

Every `/api/v1` error has the shape `{"error": {"code", "type", "message"}}` (`V1ErrorBody`). `type` is the exception the SDK raised (`KeyError`, `ValueError`, `TypeError`, a `cuemem.errors` class, or the provider exception's class), and embedder errors add `upstream_status`. 401, 413, 422, 500, unknown routes, cross-site browser requests and a wrong Host header under `/api/v1` use the same shape (the 401 message is the Chinese text of `/api`); other paths keep their plain-text 500.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | ValueError or TypeError: no user_id or agent_id, a bad filter, `limit` below 1, an image path, an image too large, in an unsupported format or not an image, a malformed export…; a Host header that is not accepted (the message names `CUEMEM_ALLOWED_HOSTS`) |
| 401 | `unauthorized` | Missing or wrong token |
| 402 | `no_balance` | The model provider's account is out of balance |
| 403 | `forbidden` | Outside the key's scope; a cross-site browser request |
| 404 | `not_found` | Unknown memory id, unknown space on export, another tenant's id, unknown route |
| 409 | `conflict` | `ConcurrentWriteError` (nothing was written; retry); `IdempotencyConflictError` (an `Idempotency-Key` reused for a different add) |
| 413 | `payload_too_large` | The request body is above the size limit |
| 422 | `invalid_request` or `content_filtered` | A malformed request body (type `ValueError`); the provider's content filter rejected the request |
| 429 | `rate_limited` | Still rate-limited after retries, a scoped key's bodies in flight above the limit, a scoped key's `rate_per_minute` used up, or a scoped key with `max_concurrent` requests already in progress; with `Retry-After`. Also a new run, or an `add` that would reopen a run, beyond `CUEMEM_MAX_OPEN_SESSIONS` (type `SessionLimitError`, `Retry-After: 60`), and a write that would create a space beyond the key's `max_spaces` |
| 502 | `upstream_error` | Other provider errors (5xx, connection, authentication) and other CueMem errors |
| 503 | `unavailable` | The provider is unavailable, with `Retry-After` |
| 503 | `vector_store_unavailable` | The Qdrant server is unreachable or answered 502, 503 or 504; `Retry-After: 5` (only with a Qdrant server) |
| 503 | `capacity` | Local mode has no room for a new space's collection near the open-file limit (type `VectorStoreFullError`, which `MemoryClient` raises); existing spaces still work |
| 504 | `timeout` | The provider timed out; `POST /api/v1/wait` past its limit (type `TimeoutError`) |
| 504 | `vector_store_timeout` | The Qdrant server did not answer in time |
| 500 | `internal_error` | Unexpected server error (a bug); the traceback is in the server log. `type` is null and the message never contains exception details |
| (none) | `unreachable`, `timeout`, `disconnected`, `transport` | Client-side codes of `MemoryClient` when no response arrived (`status` None; see `sent` above). The client-side `timeout` shares its name with the 504 code; `status` tells them apart |

`/openapi.json` (and `/docs`) gives every `/api/v1` response a named schema (`FactItem`, `ExcerptItem` and `ImageItem` discriminated on `type`, `SearchResult`, `AddResult`, `ForgetResult` or `DryRunResult`, `BatchForgetResult` or `BatchDryRunResult`, `StatusResult`, `ContextResult`, `ExportDocument`, …) and documents each error status as `V1ErrorBody`, 500 included. The 22 operationIds are the SDK method names (`add`, `get_all`, `search`, `context`, `delete_many`, …): clients generated from a snapshot older than D-072 must be regenerated, and their method names change. `GET /api/v1/media/{id}` declares `image/jpeg`, `image/png`, `image/webp`, `image/gif`, `image/bmp`, `image/tiff`, `image/mpo` and `application/octet-stream`. A snapshot of the `/api/v1` part is committed as [`docs/openapi-v1.json`](../openapi-v1.json); regenerate it with `uv run python scripts/openapi_v1.py` (`--check` exits 1 when it is stale, and the tests run it). There is no hand-written TypeScript client; generate types and call the API with them:

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
curl -s -X POST $CUEMEM/memories/batch-delete -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"memory_ids":["<id>","<id>"],"dry_run":true}'
curl -s -X POST $CUEMEM/memories -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -H "Idempotency-Key: 6f1c2b9e-3d4a-4e8b-9a51-2f7d0c8e1b34" -d '{"messages":"I moved to Berlin","user_id":"alice"}'
curl -s -H "Authorization: Bearer $CUEMEM_API_TOKEN" -X POST http://127.0.0.1:7860/api/v1/context \
  -d '{"query":"Where do I work?","user_id":"alice"}' -H 'content-type: application/json'
```

### `/api`: sessions, chat and the memory browser

These routes take raw space ids (`alice`, `shop/alice`; a "/" may also be written `%2F`) and answer errors as `{"detail": <Chinese text>}`.

| Route | What it does |
|---|---|
| `GET /api/health` | `{"ok", "version", "open_sessions"}`; no token needed; liveness only |
| `GET /api/ready` | Readiness: 200 `{"ok": true}`, or 503 `{"ok": false, "failing": ["vector_store" and/or "worker"]}` with `Retry-After: 5`; no token needed (see Health and readiness) |
| `GET /api/spaces` | The spaces (filtered by the key's scope) |
| `POST /api/sessions` with `{space, at}` | Start a chat session with the agent (default space: `--space`); 429 with `Retry-After: 60` when the space has `CUEMEM_MAX_OPEN_SESSIONS` open sessions that all have messages |
| `POST /api/sessions/{session_id}/messages`, form fields `text`, `images` (files), `mode`, `router`, `lam` | One agent turn: recall, answer with citations, write back; returns the reply and its trace |
| `POST /api/sessions/{session_id}/end?wait=` | End the session and hand it to extraction |
| `POST /api/spaces/{space}/recall` with `{query, mode, router, lam, at}` | Recall without answering; the controller picks the level (L0 gives only the profile) |
| `GET /api/spaces/{space}/profile`, `/facts?current_only=&show_expired=`, `/entities?show_expired=`, `/graph?show_expired=`, `/media?show_expired=`, `/media/{media_id}`, `/sessions`, `/status` | Browse the space (`/status` lists only this space's background errors). `/facts`, `/entities`, `/graph` and `/media` leave out expired memories unless `show_expired=true` (`/entities`: entities that only expired facts mention); fact, entity and image rows carry `expired` |
| `POST /api/spaces/{space}/forget` with `{session_ids, turn_ids, fact_ids, media_ids, dry_run}` | Forget, or preview with `dry_run`; `warning` when the rebuild after the deletion failed |
| `GET /api/spaces/{space}/export?derived=` | Export, streamed as on `/api/v1` |
| `POST /api/spaces/import?space=`, with the export file as the body | Import (409 for an invalid or malformed file, including one without `space_id` when `?space` is not given) |
| `POST /api/spaces/{space}/consolidate` | Run pending extraction now |

A chat session idle for `--idle-minutes` (30) is ended and extracted; a later message to it gets 404 saying so, so start a new session (the UI does this by itself and says why).

Status codes: 404 for an unknown or ended session or image, or an export of an unknown space; 409 for invalid input (an image too large included); 413 for a body above the size limit; 415 for an upload that is not an image; 422 for a bad `mode` or `router`; 429 (with `Retry-After: 60`) at the open-session cap, and 429 for a key over its `max_spaces`; 401 and 403 as above; for provider failures, 429 (with `Retry-After`), 402, 503 (with `Retry-After`), 504, 422 (content filter) and 502 (other provider errors); 503 (`Retry-After: 5`) or 504 when the Qdrant server is unreachable or slow; and 503 with a Chinese detail when local mode has no room for a new space.

## Memory proxy: `/v1`

`cuemem proxy` runs the proxy on its own (default `http://127.0.0.1:8787/v1`, liveness at `/health`, readiness at `/ready`, as `/api/health` and `/api/ready` above; `/health`'s `open_runs` counts in-memory sessions, not parked ones); `cuemem serve` mounts the same routes at `/v1`. Point any OpenAI SDK at it:

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
| `--access-log/--no-access-log` | on | One log line per request |

| Route | What it does |
|---|---|
| `POST /v1/chat/completions` | Recall, inject the evidence before the last user message, forward to the provider, write back; streaming and non-streaming |
| `POST /v1/responses` | A subset of the Responses API over the same path: non-streaming text input (a string, or message items with `input_text` / `output_text` parts; `developer` is treated as `system`), `instructions`, `max_output_tokens`, `temperature`, `top_p` and `user`. `metadata`, `store`, `parallel_tool_calls`, `service_tier`, `prompt_cache_key` and `safety_identifier` are accepted and ignored. Streaming, tools, `previous_response_id`, structured text formats, images and function-call items get 501 (code `not_implemented`) with `x-should-retry: false`; use `/v1/chat/completions` for them |
| `POST /v1/embeddings` | The configured embedding model (document vectors): `encoding_format` `float` or `base64`, `dimensions` only equal to the model's size, up to 2,048 non-empty string inputs (token arrays get 400). Callers' texts do not touch memory and do not enter the embedding cache. Provider errors map as on the HTTP API (429 with `Retry-After`, 402, 503, 504, 502) |
| `GET /v1/models` | The configured model aliases whose provider key is set on the server (or that need none), and the embedding model |

- **Identity:** the `X-CueMem-User` header or the request's `user` field (the header wins), optionally `X-CueMem-Agent`; `--user` is the fallback. A scoped key must cover that user and agent, otherwise an OpenAI-format 403 (type `permission_error`, code `forbidden`) comes before any recall, session or forwarding. Requests that name no user pass through without memory. `/v1/models` and `/v1/embeddings` accept any valid key.
- **Non-ASCII user or agent IDs** go in these headers percent-encoded as UTF-8 in the RFC 8187 form, e.g. `X-CueMem-User: utf-8''%E5%BC%A0%E4%B8%89` for 张三. This is the only form Node clients such as Claude Code can send; raw UTF-8 bytes (curl) are accepted too, and the proxy can also take the ID in the body `user` field. A malformed `utf-8''` value, percent-encoded bytes that are not UTF-8, or a control character in a `utf-8''` or non-ASCII value gets 400 (on the proxy an OpenAI-format `invalid_request_error` with param `user`); other raw bytes that are not UTF-8 are read as Latin-1, and ASCII values pass unchanged.
- **Spending:** a scoped key limits whose memory a caller reaches, not what it spends; `model`, `n`, `max_tokens` and switches such as `enable_thinking` are forwarded as sent unless the key sets `models` or `max_output_tokens` (see [Authentication and scoped keys](#authentication-and-scoped-keys)). A model outside a key's `models` gets an OpenAI-format 403 (type `permission_error`, param `model`), and an `n` other than 1 under `max_output_tokens` gets 400 (param `n`).
- **Sessions:** `X-CueMem-Session` names one. Without it, a request with no earlier messages, or one after `--idle-minutes` of silence, starts a new session; parallel conversations need their own session headers. Idle sessions are also ended by a sweep every minute, in `cuemem proxy` and in the proxy mounted by `cuemem serve`. Open sessions per user (memory space) are capped by `CUEMEM_MAX_OPEN_SESSIONS` (64): a new session first parks the least recently seen one that has no write-backs waiting for retry. A parked session stays open: it is not extracted, and a request naming it (or continuing the same implicit conversation) picks it up under the same id; it is extracted once, after `--idle-minutes` without a request or when the proxy stops. At most 4,096 sessions are parked per process (`PARKED_KEPT`); beyond that the oldest parked one is extracted. If every in-memory session has write-backs waiting (during an outage), the request gets an OpenAI-format 429 (type `rate_limit_error`, code `rate_limited`, `Retry-After: 60`) before any recall or forwarding, and a parked session it named stays parked.
- **Response headers:** `X-CueMem-Level` (`L0`, `L1`, `L2`; `none` when memory was not used, `error` when recall failed and the request went ahead without memory), `X-CueMem-Recalled`, `X-CueMem-Reasons`, `X-CueMem-Session`, and on every response `X-CueMem-Model`, the upstream model actually used. The body is the provider's, unchanged.
- **Models:** `model` is a configured alias or model id; a request without `model` uses the agent role's model. An unknown model also goes to the agent role's model, with a warning logged once per name (`proxy.unknown_model: remap`, the default). With `reject` (the config key, `CUEMEM_PROXY__UNKNOWN_MODEL` or `--unknown-model reject`) it gets 400 with code `model_not_found`, before any recall. A catalogue model whose provider key is not set on the server (for example `qwen3.8-flash` under `--preset openai` without `DASHSCOPE_API_KEY`) gets 503 with error code `model_not_available` (type `server_error`, param `model`, header `x-should-retry: false`), before any recall or session; the message names the missing environment variable and the models you can use. This also applies when an unknown name is remapped to an agent model that has no key. `GET /v1/models` lists only the models the server has keys for (plus the embedding model), and the `unknown_model: reject` 400 lists the same set. A scoped key asking for a user outside its scope still gets 403 first.
- **Failures:** upstream errors come back as they are, with `retry-after`, `retry-after-ms` and `x-should-retry`, and nothing is written. An unreachable upstream gives an OpenAI-format 502 and a timeout 504. A write-back that fails (say, during an embedding outage) is kept in process memory and written before that session's next write-back, or tried once more before the session ends; it is lost if the process exits. During a long outage at most 100 write-backs or 32 MiB per session are kept (counted as text plus image data-URL characters); older ones are dropped with a warning and a `proxy_write_dropped` trace event (`reason=backlog`). A write-back that carries an image above the pixel limit, or in an unsupported format, is dropped alone. If a memory the proxy injected is forgotten while the upstream model is still answering (for example another client's DELETE during a long stream), the answer is returned to the caller but not stored; the user's message is stored.
- **Capacity:** a request for a user whose space does not exist yet gets, before any recall, session or forwarding, an OpenAI-format 503 (type `server_error`, code `capacity`, `x-should-retry: false`) when local mode has no room for its collection, and a 429 (type `rate_limit_error`, code `rate_limited`, `x-should-retry: false`) when the key already holds its `max_spaces`. A Qdrant outage does not fail the chat: recall fails open (`X-CueMem-Level: error`) and the request is forwarded without memory.
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

- **Over stdio, by `cuemem mcp --space <space> [--router learned]`**, one space per process (default space `default`). It does not go through HTTP, and scoped keys do not apply to it. It opens the data directory itself, so it cannot run while `cuemem serve` holds the same database, whether vectors are local or on a Qdrant server (`StorageLockedError`): use serve's `/mcp` then. Register it with `claude mcp add cuemem -- cuemem mcp --space me`, or from a checkout `claude mcp add cuemem -- uv --directory /path/to/cuemem run cuemem mcp --space me`. Start-up errors (a missing key, a vector store that cannot open) go to stderr with exit code 1.

| Tool | What it does |
|---|---|
| `remember(text)` | Saves a note; searchable at once, extracted in the background |
| `recall(query, deep=False)` | At least an L1 recall (the controller may go deeper; `deep=True` forces L2); returns the profile, facts and excerpts with ids and dates; no answer model is called. Stored text (excerpts, the profile) is quoted line by line with `\| `, multi-line facts are joined with ` / `, and the listing ends with `End of recalled memories.` |
| `forget(ids)` | Forgets by the ids that `recall` returned and rebuilds; says so when the rebuild after the deletion failed |
| `profile()` | The profile, quoted as `recall` quotes stored text: the note that it is information, not instructions, then `Profile:`, every line prefixed with `\| `, and `End of recalled memories.`; `Nothing is known about the user yet.` when it is empty |

Provider failures come back as an English ToolError (rate limit, insufficient balance, unavailable, timeout, content filter), with "in about N s" when the provider said so; so do a Qdrant server that is unreachable ("The vector store (Qdrant) is unreachable or temporarily unavailable. Try again later.") or slow, a new space that local mode has no room for, and a `remember` into a new space beyond the key's `max_spaces`.

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
| `CUEMEM_API_KEYS_FILE` / `CUEMEM_API_KEYS` | Scoped keys: the path of a YAML file (re-read by running serve and proxy within about a second; `cuemem keys` edits it), or the same content inline (read once at start-up; set only one) |
| `CUEMEM_MAX_CONCURRENT_PER_KEY` | Requests one scoped key may have in progress at once on `/api`, `/api/v1`, `/v1` and `/mcp` (a positive integer, default 8; a key's `max_concurrent` overrides it; 429 beyond) |
| `CUEMEM_ALLOWED_HOSTS` | Extra hostnames accepted as Host and Origin (comma-separated; `*` turns both checks off); also needed for DNS names of a token-less non-loopback server. A name that is not loopback-like (or `*`) on a loopback bind means a reverse proxy exposes the server, so a token of at least 16 characters is then required |
| `CUEMEM_MAX_BODY_BYTES` | Body limit in bytes for `/api`, `/v1` and UI uploads, and a scoped key's bodies in flight (default 67108864, 64 MiB) |
| `CUEMEM_MAX_IMPORT_BYTES` | Body limit in bytes for imports with the admin token or without authentication (default 1073741824, 1 GiB) |
| `CUEMEM_TRACES_RETENTION_DAYS` | Keep daily trace files this many days (an integer ≥ 1; unset keeps them forever). `cuemem serve` and `cuemem proxy` delete older files at start-up and then daily, with the query vectors of their questions; SDK-only apps can call `memory.system.prune_traces(days)` |
| `CUEMEM_EXPIRED_RETENTION_DAYS` | Delete memories this many days after their `expiration_date` (an integer ≥ 0; 0 deletes every memory that has expired by the time of the daily run; unset, the default, never deletes them). `cuemem serve` and `cuemem proxy` apply it at start-up and then daily, with one forget and one replay per affected space; SDK-only apps run `cuemem forget --expired` |
| `CUEMEM_MAX_OPEN_SESSIONS` | Open chat sessions, memory-proxy sessions and `/api/v1` runs per memory space, each counted separately (a positive integer, default 64; see [Open sessions per space](#http-api-cuemem-serve)) |
| `CUEMEM_ALLOW_SHARED_DB` | `1` (or `true`, `yes`) lets several processes open one database, without the lock and at your own risk (a warning is logged); see [One process per database](#one-process-per-database) |
| `CUEMEM_LOG_LEVEL` | Log level of `cuemem serve`, `cuemem proxy` and `cuemem mcp`: DEBUG, INFO, WARNING (default), ERROR or CRITICAL; any other value exits with code 2. Read from the process environment only, not from `.env` (Docker's `env_file` and compose `environment` pass it) |
| `CUEMEM_BUILD` | The build id shown by `cuemem --version` and `cuemem doctor`; the Docker image sets it from the build argument of the same name. Empty or `unknown` means: look it up (pip's record of a git install, then `git rev-parse` in a checkout) |
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

`cuemem --version` prints `cuemem 0.8.0 (build <12-character commit>)`, or `(build unknown)` when no build id is found (from `CUEMEM_BUILD`, then the commit pip recorded for a git install, then `git rev-parse` in a source checkout). `/api/health` deliberately does not show the build, since it needs no token; in Docker, `docker compose exec cuemem cuemem --version` or `docker inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' <container>` show it. The version goes up once per merged audit round (0.8.0 for the eighth, D-075), and each such state is tagged `vX.Y.Z` on main. `cuemem --help` lists the commands: `configure`, `doctor`, `ingest`, `export`, `import`, `consolidate`, `reindex`, `forget`, `ask`, `search-image`, `spaces`, `usage`, `serve`, `proxy`, `chat`, `demo`, `migrate-vectors`, `mcp`, `keys`, `inspect`, `data` and `eval`. `cuemem <command> --help` describes the options; help and messages are in Chinese. `cuemem --preset <name> <command>` applies a preset (`openai`, `ollama`, `deepseek`, `qwen`, or the path of a preset YAML) to any command; another preset needs its own data directory (`CUEMEM_HOME`), because a data directory remembers its embedding model. Every command first raises the process's soft open-file limit to the hard limit (at most 65,536; never lowered; on macOS it falls back to 10,240 and 4,096); the library and the SDK leave the limit alone, so an app with many local spaces raises it itself (`ulimit -n`). `cuemem doctor` prints the local vector store's collection count against that limit, in yellow with the remedy when it passes half of it.

- A configuration mistake (an unknown preset, or an invalid `CUEMEM_*` value) makes any command print one line, `错误：配置有误：…` (field, reason and the value received, except for secrets), and exit 1, instead of a traceback; `cuemem mcp` prints it on stderr. The SDK still raises ValueError or pydantic's ValidationError.
- File modes: the command line tightens its umask to at least 077 (never loosening it), so every file it writes is 0600 and every directory 0700; the SDK and `MemorySystem.from_settings` create missing data directories at 0700. Existing directories are never changed. On POSIX, `cuemem doctor` also checks the permissions of the data directories, `.env` and `CUEMEM_API_KEYS_FILE` and prints the `chmod 700` / `chmod 600` commands for anything others can read; `serve` and `proxy` warn about the two secret files at start-up. `cuemem configure` replaces `.env` through a 0600 temporary file (a symlinked `.env` keeps its link; the directory holding it must be writable).
- `cuemem doctor` shows `向量库：<url>（API Key 已设/未设）` with a Qdrant server; `cuemem migrate-vectors` sends `CUEMEM_VECTOR_STORE__API_KEY` to the target.
- `cuemem doctor` also prints the data directory's free space (yellow below 200 MiB or when the directory is not writable). When the SQLite database cannot be opened (a read-only directory, a full disk), the commands that open it themselves (`ingest`, `import`, `export`, `consolidate`, `reindex`, `forget`, `ask`, `search-image`, `spaces`, `inspect` and `doctor --vectors`) print one line naming the data directory and exit 1, and so do `serve`, `proxy`, `chat`, `demo` and `mcp` (`mcp` on stderr): `<file> 打不开（<error>）：检查目录权限与磁盘空间`. When the database, the ledger, `llm.sqlite` or `embed.sqlite` is corrupt or not a SQLite database, every command that opens that file (those above, `usage` and `doctor --live` included) prints `<file> 打不开（file is not a database）：文件损坏或不是 SQLite 数据库，从备份恢复（见 README 的备份与恢复）；缓存与账本可以删掉重建` and exits 1: restore `cuemem.sqlite` from a backup; the caches and the ledger can simply be deleted and are rebuilt. The SDK raises `cuemem.errors.DatabaseOpenError` with the same messages (still an `sqlite3.OperationalError`), after releasing the database lock and the connection, so a long-running process can catch it and retry.
- `cuemem doctor` starts with the `cuemem --version` line and, read-only and next to a running serve, also reports the extraction jobs of the main database: per space, failed jobs with the count, the first line of one error and the fix `cuemem consolidate --space S` (the background worker stops retrying after 3 attempts, for example with an expired key or an empty balance, and writes still return 200 meanwhile); spaces extracted under an older configuration or pipeline version, which the worker does not rebuild, with the same command (a rebuild served from the LLM cache where it can); an unfinished rebuild (`rebuild-pending`), which the worker finishes on its next run; and the number of pending jobs (`排队等待巩固的任务：N 个`). The listing stops at 20 lines (`……还有 N 条`). One more line gives the sizes of the LLM and embedding caches (each with `-wal`) and of `runs/traces`, in yellow with a pointer to `CUEMEM_TRACES_RETENTION_DAYS` when that is unset and the total is above 1 GiB. A fresh data directory gets no database from this, and evaluation spaces are not covered.
- `cuemem doctor --repair-vectors`: rebuilds the local store's `data/qdrant/meta.json` from `collection/<name>/storage.sqlite` when it does not parse, is missing beside existing collections, or holds an invalid collection entry. Stop `cuemem serve` or `cuemem proxy` first: it refuses while another process holds the store. Each dense size is read from a stored point (empty collections use `embedding.dim` or `multimodal_embedding.dim`), and the old file is kept as `meta.json.broken-<timestamp>`. Do not delete `meta.json` by hand: Qdrant writes an empty one and the existing collections are orphaned.
- `cuemem doctor --vectors` (opt-in): per space, compares the database's unit counts by kind with the vector store's point counts, and media rows with image points when image vectors are on. Mismatching spaces are printed in yellow with `cuemem reindex --space S`, followed by a `cuemem reindex --all` line; a consistent store prints `向量库：N 个空间的点数与数据库一致`. When the local store is held by another process (serve), the check is skipped with a note and the rest of doctor still runs.
- `cuemem reindex [--space S | --all] [--full]`: restores the vector store from the database (no LLM calls; with a warm embedding cache, no requests either): re-embeds the units and images that have no vector and deletes points the database no longer has; `--full` rewrites every vector. Exactly one of `--space` and `--all` is required (exit code 2 otherwise), and an unknown space exits 1. It prints one line per space: the kinds filled with their counts, the orphan points deleted, images whose original file is missing, and fact units missing from the facts table. It needs the database to itself, so stop serve first (it is refused with `StorageLockedError` otherwise). It also deletes originals left in `media/` by image writes or forgets that a kill interrupted, and prints `媒体库：删掉 N 张没有提交的写入或遗忘留下的原图` when there were any. In Python: `MemorySystem.reindex(space_id, *, only_missing=True)` (for example `m.system.reindex(space)`) returns a dict with one count per kind filled (`round`, `fact`, `summary`, `entity`, `image`, `image_iso`), `images`, `orphans`, `image_orphans`, `missing_files`, and `facts_missing` when there are any; it raises RuntimeError inside `reading()` of that space, as forget does.
- `cuemem export` and `cuemem spaces` open only the SQLite database, so they work while `cuemem serve` runs (it holds the local vector store) and when the vector store cannot be opened. `cuemem export -o FILE` writes a temporary file in the same directory and renames it over FILE after fsync, so a failed export (for example a full disk) leaves the previous backup untouched and prints one line (exit 1); a symlinked FILE keeps its link, and `/dev/stdout` and FIFOs are written directly. `cuemem import` of a truncated or non-UTF-8 file prints one line (`不是有效的 CueMem 导出文件`) and exits 1. `cuemem import FILE --from mem0 [--user U] [--agent A] [--consolidate]` imports a Mem0 `get_all` export (see Importing from Mem0): without `--user` and `--agent` it groups the memories by their `user_id` and `agent_id`, converts and checks every group first, then imports each group into its own space as its own transaction and prints one line per space (re-running after a failure adds only what is missing); `--space` cannot be combined with `--from mem0`, and `--user` / `--agent` also work with the default `--from cuemem`. A Mem0 file imported without `--from mem0` gets a one-line hint to add it. `cuemem inspect entities` marks entities that only expired facts mention `· 已过期` (`--hide-expired` leaves them out). `cuemem inspect facts --valid-at` takes the `valid_at` forms above and exits 2 for any other. `cuemem inspect facts` marks expired facts `已过期` (`--hide-expired` leaves them out), and `cuemem ask` leaves expired memories out unless `--show-expired`.
- `cuemem forget --expired [--space S] [--older-than-days N] [--yes]` deletes turns whose `expiration_date` passed more than N days ago (default 0), with what was derived from them, one forget and one replay per space, and prints one report line per space. Without `--space` it covers every space; with `--expired`, `--session`, `--turn`, `--fact` and `--dry-run` are refused. Without `--expired`, `--space` is still required.
- `cuemem doctor` reports image writes or forgets that a kill interrupted (`媒体库里有 N 个被中断的写入或遗忘…`) and points to `cuemem reindex --all`; `cuemem serve` and `cuemem proxy` clean them up at start-up and hourly. Their markers live in `media/.journal/`, which is empty when nothing is in flight.
- <a id="one-process-per-database"></a>**One process per database.** The memory system takes a lock on `<database>.lock` next to the database (for example `data/cuemem.sqlite.lock`, mode 0600; it uses `flock`, which NFS may not honour). A second process that opens the same database gets `StorageLockedError` naming the holder's pid and command, also when vectors live on a Qdrant server: `serve`, `proxy`, stdio `mcp`, `chat`, `demo`, `ingest`, `import`, `consolidate`, `reindex`, `forget` (`--dry-run` included), `ask`, `search-image` and SDK `Memory()`. Commands that only read the database work next to a running serve: `export`, `spaces`, `inspect facts|entities|sources|profile`, `doctor` (with a Qdrant server also `--vectors`; in local vector mode `--vectors` prints a skip line instead of its check while serve holds the store), `keys` and `usage` (for example `docker compose exec cuemem cuemem inspect facts --space default`); for questions, use the HTTP API (`/api/v1/answer`) or MCP over `/mcp`. Talk to the running process through `MemoryClient` or HTTP, or set `CUEMEM_ALLOW_SHARED_DB=1` to skip the lock at your own risk (one process adopting another's open runs, `reindex` deleting the other process's in-flight vectors, colliding turn numbers). Two data directories pointed at the same Qdrant server and space still share collections; the lock does not cover that.
- `cuemem keys add|list|revoke [--file PATH]`: manage scoped keys in `CUEMEM_API_KEYS_FILE` (see [Authentication and scoped keys](#authentication-and-scoped-keys)). `keys add NAME --user U… [--agent A…] [--rate-per-minute N] [--models M…] [--max-output-tokens N] [--max-concurrent N] [--max-spaces N]` prints the new token once, alone on stdout; `keys list` never shows tokens or digests.
- A data directory upgrades automatically; downgrades are not supported. Since D-073 the database may also hold the table `add_keys` (idempotency records), created automatically without a schema-version change; an older v7 CueMem ignores it. Since D-074 it gets the indexes `idx_facts_session` and `idx_add_keys_created`, created on open, also without a schema-version change. A database written by a newer CueMem is refused with ConfigMismatchError and left untouched (the message points to the README's Upgrading section): to move to an older version, export with the newer one and import.

### Upgrading

Each recipe ends with `cuemem doctor`, whose first line shows the version and build now running.

- **Docker Compose:** `git pull && CUEMEM_BUILD=$(git rev-parse --short=12 HEAD) docker compose up -d --build`. docker-compose.yml sets `pull_policy: build`, so a plain `docker compose up -d` rebuilds the image too (quickly with a warm cache; before 0.8.0 it restarted the old image); `docker compose build --pull` also refreshes the base image. Then `docker compose exec cuemem cuemem doctor`.
- **A single container:** `docker build --build-arg CUEMEM_BUILD=$(git rev-parse --short=12 HEAD) -t cuemem .`, then remove and recreate the container with the same `-v cuemem-data:/data` (and `--stop-timeout 90`).
- **pip:** `pip install -U --force-reinstall "cuemem[ui,local] @ git+https://github.com/FlickYan/cuemem@vX.Y.Z"` (pick your extras), or pin a tag such as `@v0.8.0` once it exists. pip does not reinstall a git URL whose version has not changed, so a plain `-U` between two commits of the same version does nothing; `--force-reinstall` does. Adding `--no-deps` skips dependency changes, so leave it out when the dependencies changed.
- **A source checkout:** `git pull && uv sync --locked`.

### Backup and restore

- **What to keep** (in the data directory, `data/` in a checkout): `cuemem.sqlite` with its `-wal` and `-shm` files (the raw log and derived memory), `ledger.sqlite`, `media/` (the original images) and `qdrant/` (or the Qdrant server's storage); `claude_memory.sqlite` with its `-wal` and `-shm` (only if you use the Claude memory tool integration: the `/memories` files Claude wrote; it is not in `cuemem export` and cannot be rebuilt from the log); `cache/` is optional (it can be rebuilt, but that costs requests). `runs/`, `media/.journal/` and `cuemem.sqlite.lock` are not needed for recovery.
- **How:** stop `cuemem serve` and copy the whole data directory, or copy each SQLite file (`cuemem.sqlite`, `ledger.sqlite` and, if present, `claude_memory.sqlite`) with `sqlite3 FILE ".backup DEST"` (or Python's `sqlite3.Connection.backup`). Never `cp` a live `.sqlite` file without its `-wal`: with WAL, recent commits are still in the `-wal` file. Copy a Qdrant server's storage only with the Qdrant server stopped; otherwise leave it out and rebuild the vectors with `cuemem reindex --all` after a restore (free with a warm embedding cache).
- **With Docker / compose:** in the image the data directory is `/data/data` (`CUEMEM_HOME=/data`); it holds `cuemem.sqlite`, `ledger.sqlite`, `claude_memory.sqlite`, `media/` and `cache/`, and `/data/runs` is not needed. The image has Python but no `sqlite3` command.
  - Simplest, with cuemem stopped: `docker compose stop cuemem && docker compose cp cuemem:/data/data ./cuemem-backup && docker compose start cuemem` (`docker compose cp` works on a stopped container). This keeps each database's own `-wal` and `-shm`, and `cache/`, so a later reindex sends no embedding requests.
  - Live, with cuemem running (it skips files that do not exist, since connecting would create an empty one, includes `cache/embed.sqlite` so a later reindex sends no embedding requests, and copies `media/` after the databases):

    ```bash
    docker compose exec -T cuemem python - <<'EOF'
    import pathlib, shutil, sqlite3
    data, out = pathlib.Path("/data/data"), pathlib.Path("/data/backup")
    for name in ("cuemem.sqlite", "ledger.sqlite", "claude_memory.sqlite", "cache/embed.sqlite"):
        if (data / name).exists():  # sqlite3.connect would create an empty file
            (out / name).parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(out / name) as copy:
                sqlite3.connect(data / name).backup(copy)
    if (data / "media").exists():  # the originals, after the databases
        shutil.copytree(data / "media", out / "media", dirs_exist_ok=True)
    EOF
    docker compose cp cuemem:/data/backup ./cuemem-backup
    ```

  - Vectors: do not copy the `qdrant-storage` volume while the qdrant service runs; stop it too, or skip the vectors and rebuild them after a restore.
  - Restore: `docker compose stop cuemem`; `docker compose run --rm --user 0 --entrypoint sh cuemem -c 'rm -f /data/data/*.sqlite-wal /data/data/*.sqlite-shm /data/data/cache/*.sqlite-wal /data/data/cache/*.sqlite-shm'` (stale WAL files would be replayed onto the restored copies); `docker compose cp ./cuemem-backup/. cuemem:/data/data/`; `docker compose run --rm --user 0 --entrypoint chown cuemem -R cuemem /data/data` (files copied in belong to root, and the image runs as uid 1000); then `docker compose run --rm cuemem doctor --vectors` and `docker compose run --rm cuemem reindex --all`, and `docker compose start cuemem`. These commands were written from the image and compose file and not run against a live compose stack; check them once on yours.
- **Restore:** with `cuemem serve` stopped, put back the SQLite files, then `media/`, then optionally `qdrant/`; then run `cuemem doctor --vectors` and `cuemem reindex --all`. Before putting back a `.backup` copy, delete any `cuemem.sqlite-wal` / `-shm`, `ledger.sqlite-wal` / `-shm` and `claude_memory.sqlite-wal` / `-shm` left in the data directory: a crashed serve leaves them, and SQLite would replay that unrelated WAL onto the restored file. A copy of the stopped directory comes back together with its own `-wal` and `-shm`. SQLite and Qdrant snapshots taken at different moments are repaired the same way, and so is a lost vector store (search returns nothing while `get_all` still lists memories). `cuemem export` per space is the portable alternative; it carries the raw log and the images, not the vectors or the Claude memory tool's files.
