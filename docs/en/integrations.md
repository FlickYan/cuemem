# Framework integrations (N12)

English edition of [docs/integrations.md](../integrations.md).

There are six ways to plug CueMem into an existing agent stack, from least to most code:

| Way in | Good for | Who decides whether to recall | Where |
|---|---|---|---|
| OpenAI-compatible memory proxy | Any app on the OpenAI SDK, with no code changes: Chat Completions, embeddings, and a subset of the Responses API (text and image input, text output, streamed or not; no tools) | The controller, every turn | `cuemem proxy`, README |
| MCP server | Claude Code, Claude Desktop and other MCP clients | The model calls `recall`; the controller only picks the depth | `cuemem mcp` (stdio), or `/mcp` of `cuemem serve` (HTTP, behind the same tokens; POST only, with tool titles and read-only / destructive hints since D-076); [client configuration files](#mcp-client-configuration-files) |
| Claude Code hooks | Claude Code, without relying on the model to call a tool | The controller, on every prompt (`hook recall`); `hook profile` alone only shows the profile | `cuemem hook profile \| recall \| capture`, against a running `cuemem serve`; see [Claude Code hooks](#claude-code-hooks) |
| Framework integrations (this page) | OpenAI Agents SDK, LangGraph / LangChain, Claude's memory tool | The controller, before every model call; the model can also search with tools | `cuemem.integrations.*` |
| Python SDK | Your own agent loop | You | `from cuemem import Memory` |
| Memory API over HTTP | Apps in other languages, or several processes sharing one server | You | `/api/v1/memories` of `cuemem serve`; `from cuemem import MemoryClient` |

Every method, route and error code of the last two is in the [API reference](api.md). `MemoryClient` has exactly the methods and signatures of `Memory`, so code moves between a local data directory and a shared server by changing one line; the integrations below still need a local `Memory` or `AsyncMemory`, and passing them a `MemoryClient` raises TypeError at construction (before, an AttributeError came later; `claude.with_recall` now raises instead of warning).

All three integrations share `cuemem/evidence.py`.
- **Controller first:** the controller picks a level (L0 recalls nothing and gives only the profile).
- **One evidence format:** the recalled facts and conversation excerpts are laid out exactly as the proxy and the reader lay them out, and placed at the start of the latest user message.
- **Prefix caching is kept:** the system prompt and earlier messages are untouched, so provider prefix caching keeps working.
- **Same tools as MCP:** recall / remember / forget have the same names, descriptions and semantics as the MCP tools.
- **Memories are framed as data:** the evidence block and the recall tools' listing say that their content is quoted from past conversations, information and not instructions, and a stored `<memory` or `</memory` is escaped so it cannot close the block early. The recall listing quotes stored text line by line with '| ' and ends with 'End of recalled memories.' (multi-line facts are joined with ' / '), so a stored message cannot pass itself off as part of the listing, such as a `Facts:` block with real ids next to the forget tool. The MCP `profile` tool quotes the profile the same way (`Profile:`, every line prefixed with '| ', then 'End of recalled memories.'). This reduces prompt injection from stored memories but cannot prevent it.
- **Nothing twice:** in OpenAI Agents and LangGraph, rounds of the current conversation that are still in the request are left out of the evidence, photo rounds included (`evidence.stored_text` and `in_context_rounds`, shared with the proxy).
- **Forgetting says what happened:** when the deletion is done but rebuilding the derived memory from the rest of the log fails (say, the provider is out of balance), the forget tools of MCP, Agents and LangGraph answer "Deleted N conversation turn(s). Rebuilding the memory derived from the rest failed (…); it finishes on the next consolidation…" instead of an error, so the model does not tell the user nothing was deleted.
- **Writes are all or nothing:** when embedding or the vector store fails, nothing is written, so a write can be retried without duplicating turns. Once the turns are committed nothing raises: a trace that cannot be written is only a warning on the `cuemem.core.trace` logger.
- **A forget waits for a whole recall:** automatic recall and the recall tools that raise L0 to L1 read under one hold, so a forget in the same space runs before or after them, never in between.

Install with `pip install "cuemem[integrations] @ git+https://github.com/FlickYan/cuemem"` (CueMem is not on PyPI yet; a package named `cuemem` there is not this project) or, in a checkout, `uv sync --extra integrations`; one framework alone is `openai-agents`, `langgraph` or `anthropic`. With only a DeepSeek key, add the `local` extra and use `Memory(preset="deepseek")`. Runnable examples, with the keys and commands each needs, are listed in [examples/README.md](../../examples/README.md).

**Peak hours.** The SDK, and so every integration built on a `Memory`, defaults to `peak_policy="warn"`: during DeepSeek's weekday peak hours (Beijing time) model calls go out at peak price instead of pausing until the peak ends. This applies even without a preset, because the default configuration's reader is DeepSeek. For batch work that should wait for the off-peak price, use `Memory(peak_policy="wait")`; the SDK then logs a warning on the `cuemem.sdk` logger when it starts waiting. Evaluations and `cuemem eval run --peak` are not affected.

## OpenAI Agents SDK

`cuemem.integrations.openai_agents` (openai-agents 0.24). It has three parts, usable separately:

- `memory_tools(memory, user_id=...)`: the recall, remember and forget tools.
- `memory_input_filter(memory, user_id=...)`: pass it as `RunConfig.call_model_input_filter`. Before every model call, it runs the controller and injects the evidence into the copy sent to the model.
- `CueMemSession(session_id, memory, user_id=...)`: an implementation of the Session protocol.
  - An inner session (SQLiteSession by default) keeps the exact short-term history.
  - User and assistant text is also written into CueMem, one question and answer per round.
  - When the conversation ends, the whole session is extracted into facts.

```python
from agents import Agent, ModelSettings, RunConfig, Runner
from cuemem import Memory
from cuemem.integrations.openai_agents import (
    CueMemSession,
    close_sessions,
    memory_input_filter,
    memory_tools,
)

memory = Memory(preset="deepseek")


def user_of(context):  # one Agent shared by many users: identity from the run context
    return context["user_id"]


agent = Agent(name="Assistant", model=..., tools=memory_tools(memory, user_id=user_of))
config = RunConfig(
    call_model_input_filter=memory_input_filter(memory, user_id=user_of),
    trace_include_sensitive_data=False,  # keep recalled memories out of SDK traces
    model_settings=ModelSettings(store=False),  # the provider keeps no copy of responses
)

async with CueMemSession("chat-1", memory, user_id="alice") as session:
    await Runner.run(
        agent, "I moved to Lisbon", session=session, run_config=config, context={"user_id": "alice"}
    )
close_sessions(memory)  # at shutdown, before memory.close()
```

- **Sessions:**
  - Sessions are written as open and not extracted.
  - They are handed to background extraction when `async with` ends, on `end()`, after 30 idle minutes, or on `close_sessions()`.
  - A later conversation in the same SDK session becomes a new segment, with its own conversation date.
- **Server-managed state:** with `conversation_id` or `previous_response_id`, use `memory_input_filter(..., placement="instructions")`. The evidence then goes into that call's instructions, which keeps it out of the conversation items that later calls replay; each stored response still keeps its own instructions (see below).
- **Forgetting:**
  - Messages removed with `pop_item` or `clear_session` are also forgotten in long-term memory.
  - Turns deleted with the forget tool are pruned from the inner session the next time history is read.
  - **The provider's copy:** the Agents SDK leaves `ModelSettings.store` unset, so the Responses API stores every response (its instructions and input, evidence included, in either placement) for at least 30 days unless you pass `ModelSettings(store=False)` or the organization has zero data retention. That copy is outside `forget` and `delete_all`. For client-held history (`CueMemSession`, `result.to_input_list()`) use `RunConfig(model_settings=ModelSettings(store=False), call_model_input_filter=..., trace_include_sensitive_data=False)`, as above. Server-managed state (`previous_response_id`, `conversation_id`) needs stored responses, so there the provider's copy cannot be avoided and forget cannot reach it. (Checked against openai-agents 0.24 and openai 3.40; see OpenAI's documentation for its retention terms.)
- **Unanswered messages:** user messages the model never answered (for example, blocked by an input guardrail) are not stored.
- **Write-backs that fail are retried:** when a batch cannot be written (an embedding or Qdrant outage), `CueMemSession` keeps it and writes it, with its original time, before the next batch or in `end()`. A batch is dropped (trace event `agents_mirror_dropped`) when retrying cannot help (most 4xx errors, a content-filter refusal, invalid input such as an image above Pillow's pixel limit, which is dropped alone, or a local vector store with no room for a new collection, `VectorStoreFullError`, since D-076) or when the space has forgotten turns in the meantime, so messages from before a forget are never written back. Kept batches live only in process memory: at most 100 batches or 32 MiB per session (text plus image data-URL characters); during a long outage older ones are dropped with a warning and an `agents_mirror_dropped` trace event (`reason=backlog`). Since D-077 these events are really written: each is a one-event trace line named after the event, with the attribute `space` and an event that carries `reason` (`forgotten` or `backlog`) or `error` (the exception type name) and `status`, and never a message text or a memory id. Such a line is skipped while a forget of that space is running, and for a space that was just deleted and is still empty.
- **Through the memory proxy instead:** the Agents SDK's default model class calls the Responses API, usually with tools, and the proxy answers Responses calls that carry tools with 501 (Responses calls without tools work: text and image input, and since D-076 streamed as well as non-streamed). Use this integration, or point `OpenAIChatCompletionsModel` at the proxy: a tool loop through the proxy then carries the turn's memory on every model call (before 0.9.0 only on the first call of a turn; the follow-up requests went out without it), as this integration and LangGraph do.

## LangGraph / LangChain

`cuemem.integrations.langgraph` (langgraph 1.2, langchain 1.4).
- With LangChain 1.x's `create_agent`, use the `CueMemMiddleware` middleware.
- With the older `langgraph.prebuilt.create_react_agent`, use `recall_hook` and `save_hook` (pre_model_hook and post_model_hook).
- To give the model tools only, use `memory_tools`.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from cuemem import Memory
from cuemem.integrations.langgraph import CueMemMiddleware

memory = Memory(preset="deepseek")  # AsyncMemory for async apps
cuemem = CueMemMiddleware(memory, tools=True)  # also gives the model recall / remember / forget
agent = create_agent(llm, tools, middleware=[cuemem], checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "t1", "user_id": "alice"}}
out = agent.invoke({"messages": [{"role": "user", "content": "I moved to Munich"}]}, config)
out["cuemem"]  # the controller's decision this turn: level, reasons, recalled

# Conversation over: extract the whole thread (idle threads are handed over after 30 min)
cuemem.end_thread(config)
cuemem.close()  # at shutdown, before memory.close(); async: aend_thread / aclose
```

- **Identity:** when `user_id` is not fixed at construction, it is taken from `config["configurable"]`. On LangGraph Server, the authenticated user takes precedence over any id the client sends.
- **Threads:**
  - One LangGraph thread is one CueMem session.
  - Turns that end in structured output or a `return_direct` tool are saved.
  - Summaries written by LangChain's SummarizationMiddleware are not mistaken for the user's words.
  - A thread that continues after `end_thread` or the idle hand-over is extracted again in full when it ends, and the space's derived memory is replayed from the raw log, so its facts equal a rebuild from the log (no stale wording from the earlier extraction).
- **Forgetting:** forget removes the raw log and derived memory in CueMem. The messages stay in the current thread's checkpoints, which CueMem does not manage.

## Claude's memory tool

`cuemem.integrations.claude` (anthropic 1.14, tool type `memory_20250818`). Claude's memory tool runs client-side: the model calls view, create, str_replace, insert, delete and rename on a `/memories` directory. `CueMemMemoryTool` backs that directory with CueMem:

- **Read-only generated files:** `/memories/profile.md` (the profile) and `/memories/facts.md` (current facts with dates, ids and validity). Writes to them are rejected with an explanation.
- **Claude's own files:**
  - They are stored per user in a small side store, `claude_memory.sqlite` in the data directory (a SQLite database in WAL mode). Back it up with the data directory, including its `-wal` and `-shm` files; see [Backup and recovery](../../README.md#backup-and-recovery).
  - Every change is also written as a note into CueMem's raw log, which consolidation turns into facts.
  - Deleting a file forgets what it contributed.
  - `Memory.delete_all`, `reset` and `delete` (and the `/api/v1` delete routes) delete the matching files in this store at once: all of a deleted user's files, or the files whose notes a forget removed. The module's own `delete_all` is needed only for a store at a custom `store=` path; it also returns `memory_files_deleted`.
- **Safety:** strict path validation keeps everything inside `/memories`, and file sizes are limited.
- **Recall:** `with_recall` / `awith_recall` run the controller and inject evidence before the request goes to Claude.

Set `export ANTHROPIC_MODEL=<model id>` first (a Claude model that supports the memory tool); `examples/integrations/claude_example.py` exits with a message naming the variable when it is unset.

```python
import os

import anthropic
from cuemem import Memory
from cuemem.integrations.claude import CueMemMemoryTool, delete_all, with_recall

# default config: Qwen extracts and retrieves, DeepSeek answers; or Memory(preset="deepseek")
memory = Memory()
client = anthropic.Anthropic()
model = os.environ["ANTHROPIC_MODEL"]  # a Claude model that supports the memory tool

with CueMemMemoryTool(memory, user_id="alice") as tool:
    history = []  # append-only: with_recall only touches the newest user message
    for text in ["I moved to Lisbon in 2026-09.", "What should I cook tonight?"]:
        user = {"role": "user", "content": text}
        history, evidence = with_recall([*history, user], memory, user_id="alice")
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=16000,
            tools=[tool],
            messages=history,
            thread=False,  # send the whole history each step (anthropic>=2.3), so forget covers it
        )
        for message in runner:
            history.append(message.to_param())
            results = runner.generate_tool_call_response()
            if results is not None:
                history.append(results)

delete_all(memory, user_id="alice")  # memories and Claude's files; memory.delete_all does both too
```

- **Not run against the real Claude API:** there was no Anthropic key, and DeepSeek's Anthropic-compatible endpoint rejects `memory_20250818`. Offline tests drive every tool command directly. The live smoke test drove the same commands through DeepSeek as a regular tool.
- **Removing a line from a file** records "this no longer holds"; extraction and reconciliation decide what follows. Only deleting the whole file is a guaranteed forget.
- **Provider failures after the note is logged:** if extraction fails once a change has been written as a note (say, the provider is out of balance), the file change still succeeds and the note is logged once; extraction finishes on the next consolidation. A delete whose rebuild fails after the deletion also succeeds, and says that the rebuild finishes later.
- **Export:** Claude's files are not in `Memory.export`, which only has the notes in the raw log.
- **anthropic 2.3 and later:** `tool_runner` keeps the conversation in a thread the server holds by default, so the requests within one turn after the first send only the new messages (the earlier ones, evidence block included, stay as sent). The pattern above still starts each turn from the full `history`. With a thread, the provider stores the conversation server-side, evidence blocks included; that copy is outside CueMem's forget and `delete_all`. The snippet above and `examples/integrations/claude_example.py` pass `thread=False` (it needs anthropic 2.3 or later; on 1.x, which CueMem's `anthropic>=1.14` floor still accepts, remove the argument, since 1.x already sends the whole history). Each step then re-sends the whole history, so everything sent stays within CueMem's forget and `delete_all`. Removing it trades forget coverage for smaller requests, because the default server-held thread keeps the evidence blocks on the provider's side. For how long the provider keeps threads or requests, see Anthropic's documentation. The example needs `ANTHROPIC_MODEL` set to a Claude model that supports the memory tool.

## Claude Code hooks

`cuemem hook profile | recall | capture` (D-077) plug CueMem into Claude Code without going through the model: with MCP the model has to decide to call `recall`, and nobody writes the conversation into memory. The hooks call a running `cuemem serve` through `MemoryClient` (one process per data directory, so a hook cannot open the database itself). [examples/claude-code-hooks.json](../../examples/claude-code-hooks.json) is a ready settings fragment:

| Hook event | Command | What it does |
|---|---|---|
| `SessionStart` | `cuemem hook profile` | Puts the profile into the context (a `<memory>` block of at most 4,000 characters). Read-only: it opens no run |
| `UserPromptSubmit` | `cuemem hook recall --capture` | Recalls once for the prompt and prints the `<memory>` evidence block (`context()` with `max_chars`, default 8,000, and the current session excluded); with `--capture` it then stores the prompt. Empty prompts and prompts starting with `/` are skipped |
| `Stop` | `cuemem hook capture` | Stores the assistant's last message (`--no-assistant` turns that off) |
| `SessionEnd` | `cuemem hook capture --end` | Ends the session's run, which hands the whole conversation to extraction |

- **Settings:** each entry is `"type": "command"` with `"timeout": 10` (5 for `SessionEnd`, which also raises Claude Code's 1.5 s budget for that event). The hook's own budget is `--timeout 8` seconds for the whole hook, not per request, so it ends before Claude Code cancels it.
- **Identity and address:** set `CUEMEM_HOOK_USER` (required: without it the hooks do nothing), optionally `CUEMEM_HOOK_AGENT` (for example one per project), `CUEMEM_URL` (default `http://127.0.0.1:7860`) and `CUEMEM_API_TOKEN`, in the shell that starts Claude Code and not in a settings file that is committed. Prefer a scoped key (`cuemem keys add <name> --user <id>`) over the admin token. The same values can be given as `--user`, `--agent` and `--url`. If `HTTP_PROXY` is set without `NO_PROXY` for 127.0.0.1, the hook, like any `MemoryClient`, cannot reach a local serve.
- **Fail open:** every failure (the server is down, a wrong token, a timeout, input that is not JSON, even a mistyped option) is exit code 0 with empty stdout, because exit code 2 would block the prompt on `UserPromptSubmit` and keep the assistant from stopping on `Stop`. `--debug` writes one line with the reason to stderr (never the token), which Claude Code shows in its debug log. For the same reason `cuemem hook` without a subcommand prints nothing; use `--help`.
- **Output:** plain text only, the quoted `<memory>` block that the proxy and `context()` produce, never a control field. Claude Code caps hook output at 10,000 characters, so `--max-chars` is clamped to 1,000–9,999. Claude Code keeps injected hook context in the transcript, so a block is added on every prompt and accumulates over a long session: use a smaller `--max-chars` (for example 3,000) for long sessions.
- **One session is one run** (`cc-` plus the Claude Code session id): the prompts and final answers are searchable at once, but facts are extracted only after the session ends or has been idle for 30 minutes on the server. The current session's own excerpts are excluded from recall (the model already has them), so after a compaction they are not recalled until the session ends.
- **What is stored:** the prompts and, on `Stop`, the assistant's last message, each cut at `--max-turn-chars` (8,000). Answers can contain code and secrets from the repository; `--no-assistant` leaves them out. Unpaired surrogates in the prompt or the answer (text cut in the middle of an emoji) are replaced with U+FFFD before recall and capture. When recall fails and `--capture` is set, the prompt is still stored.
- **Two set-ups:** the recall block already contains the profile whenever one exists, so with `hook recall` configured the `SessionStart` entry is largely redundant; `hook profile` alone is the cheap set-up with no request per prompt. `recall` takes `--mode` (default `auto`) and `--min-level` (default `l0`: a prompt the controller sees no need to recall for gets only the profile).
- **Limits:** `UserPromptSubmit` also fires for scheduled tasks, reports from background subagents and messages from other sessions, so those are recalled for and, with `--capture`, stored as user turns. Async hooks are not used. The quoting reduces prompt injection through stored text but cannot prevent it. The stdin fields and the output rules were checked against Claude Code's hooks page (code.claude.com/docs/en/hooks) in 2026-11; the end-to-end check ran against a real `cuemem serve` with offline hash embeddings, not inside Claude Code.

## MCP client configuration files

The MCP server (`cuemem mcp` over stdio, or `/mcp` of `cuemem serve` over HTTP; tools, headers and errors are in the [API reference](api.md#mcp-server)) can be registered with a command such as `claude mcp add`, or through the client's configuration file. The files below were checked against each client's own page in 2026-11; the formats are the clients' and change with them, so follow the link when an entry is not accepted. `tests/test_docs_mcp_snippets.py` keeps the `cuemem` arguments in these blocks valid.

### stdio: Claude Code, Cursor, Claude Desktop

The client starts `cuemem mcp` itself, one memory space per entry. The same entry works in three files:

- Claude Code: `.mcp.json` at the project root ([docs](https://code.claude.com/docs/en/mcp)).
- Cursor: `.cursor/mcp.json` in the project, or `~/.cursor/mcp.json` ([docs](https://cursor.com/docs/context/mcp)).
- Claude Desktop: `claude_desktop_config.json` (Settings → Developer → Edit Config; on macOS `~/Library/Application Support/Claude/`, on Windows `%APPDATA%\Claude\`; [docs](https://modelcontextprotocol.io/docs/develop/connect-local-servers)). This file takes local stdio servers only; a remote server is added in the app as a connector.

From a source checkout:

```json
{
  "mcpServers": {
    "cuemem": {
      "command": "uv",
      "args": ["--directory", "/abs/path/to/cuemem", "run", "cuemem", "mcp", "--space", "me"]
    }
  }
}
```

After a pip install (`cuemem` on the PATH):

```json
{
  "mcpServers": {
    "cuemem": {
      "command": "cuemem",
      "args": ["mcp", "--space", "me"]
    }
  }
}
```

### HTTP: Claude Code and Cursor

Against a running `cuemem serve`. Claude Code needs `"type": "http"` (an entry with a `url` and no `type` is read as a stdio server) and expands `${VAR}`:

```json
{
  "mcpServers": {
    "cuemem": {
      "type": "http",
      "url": "http://127.0.0.1:7860/mcp",
      "headers": {
        "Authorization": "Bearer ${CUEMEM_MCP_KEY}",
        "X-CueMem-User": "alice"
      }
    }
  }
}
```

Cursor takes the entry without `type` and writes variables as `${env:NAME}`:

```json
{
  "mcpServers": {
    "cuemem": {
      "url": "http://127.0.0.1:7860/mcp",
      "headers": {
        "Authorization": "Bearer ${env:CUEMEM_MCP_KEY}",
        "X-CueMem-User": "alice"
      }
    }
  }
}
```

### VS Code

`.vscode/mcp.json` ([docs](https://code.visualstudio.com/docs/copilot/reference/mcp-configuration)) has a top-level `servers` key and a `type` on every entry:

```json
{
  "servers": {
    "cuemem": {
      "type": "stdio",
      "command": "uv",
      "args": ["--directory", "/abs/path/to/cuemem", "run", "cuemem", "mcp", "--space", "me"]
    }
  }
}
```

Over HTTP, with the token asked for once and kept by VS Code instead of written in the file:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "cuemem-key",
      "description": "CueMem key",
      "password": true
    }
  ],
  "servers": {
    "cuemem": {
      "type": "http",
      "url": "http://127.0.0.1:7860/mcp",
      "headers": {
        "Authorization": "Bearer ${input:cuemem-key}",
        "X-CueMem-User": "alice"
      }
    }
  }
}
```

### Notes for every client

- **Provider keys (stdio).** Under the default configuration `cuemem mcp` needs the provider keys. It reads them from the `.env` file of the data directory (`cuemem configure` writes it): the checkout root with `uv --directory`, `~/.cuemem/.env` after a pip install. An `env` block in the entry works too, and is also where `CUEMEM_HOME` or `CUEMEM_PRESET` go. Without the keys the server exits with code 1 and the error on stderr, which the client shows in its MCP log.
- **Absolute paths.** GUI clients start servers with a minimal PATH. If `uv` or `cuemem` is not found, give its absolute path as `command` (`which uv`, `which cuemem`).
- **Tokens.** `.mcp.json` is usually committed, so keep the token in an environment variable (`${CUEMEM_MCP_KEY}` above; set it in the shell that starts the client). Prefer a scoped key for one user, made with `cuemem keys add <name> --user alice`, over the admin token.
- **`X-CueMem-User`** is required with the admin token, without authentication and with wildcard-only keys (400 otherwise); a scoped key that names a user defaults to its first user. `X-CueMem-Agent` is optional. A non-ASCII id is written `utf-8''<percent-encoded UTF-8>`, for example `utf-8''%E5%BC%A0%E4%B8%89` for 张三.
- **One process per database.** The stdio server opens the data directory itself and cannot share it with a running `cuemem serve` (`StorageLockedError`): use the HTTP entry then.

## Limitations

- **One process per database:** a database can be opened by one process at a time. A second process (stdio `cuemem mcp`, `cuemem proxy`, `cuemem reindex`, an SDK `Memory()`) opening the same database gets `StorageLockedError` naming the holder's pid and command, also when vectors live on a Qdrant server. Talk to the running process through `MemoryClient` or HTTP (`/api/v1`, `/mcp`), or set `CUEMEM_ALLOW_SHARED_DB=1` to skip the lock at your own risk (adoption of each other's open runs, `reindex` deleting the other process's in-flight vectors, colliding turn numbers). The lock is a file `<database>.lock` next to the database (for example `data/cuemem.sqlite.lock`); it uses `flock`, which NFS may not honour. Two data directories pointing at the same Qdrant server and space still share collections; the lock does not cover that.
- **A forget during the model call:** if a memory that was injected is forgotten while the model is still answering, the memory proxy returns the answer but does not store it (the user's message is stored); since D-076 this also holds for the answers of tool-call follow-ups. The integrations have the same recall → model → write shape without that check, so such an answer may still be written back there (not reproduced).
- **Images:** the integrations inject text evidence only. When the controller wants to look at the original photo (`see`), the photo is not attached to the model call (the same as the proxy).
- **Async:** CueMem itself is synchronous and runs in threads. In async frameworks use `AsyncMemory` or each integration's async variants.
- **Provider errors and expiry:** the integrations' own memory tools do not turn provider errors (rate limits, an empty balance, timeouts) into explanations the way MCP does; the provider client's error comes through. Memories whose `expiration_date` has passed are left out of the integrations' recall, auto-recall and profile, and of the Claude memory tool's generated `profile.md`, `facts.md` and `history/` (D-072; before, only the SDK's `search`, `get_all` and profile left them out). Expiry is judged against the current time; a memory restated later without an expiry stays.
- **Traces:** every automatic recall and recall tool call writes exactly one `retrieve` trace (the question, the retrieval mode and the IDs it retrieved; also when the controller's L0 is raised to L1: before D-077 a raised search wrote two lines, and the single line now carries a `raised` event) to `runs/traces/*.jsonl` in the data directory. Retrieval also records a `profile` event with the ids of the facts behind the profile it showed, so forgetting a profile fact deletes the traces of answers that only paraphrased it (in practice, most traces of that space). Forgetting deletes the traces that list a deleted memory, together with the cached query vectors of their questions (also when the question contains U+2028, U+2029 or U+0085) and of the sub-queries, follow-up queries and the agent's own searches recorded in them, and `delete_all` deletes every trace of the space. A trace file that cannot be rewritten during a forget (a full disk) is deleted whole, with a warning; the forget still succeeds. Traces are kept until then unless `CUEMEM_TRACES_RETENTION_DAYS` is set: `cuemem serve`, `cuemem proxy` and `cuemem mcp` apply it at start-up and then daily, and an app that only uses the SDK can call `memory.system.prune_traces(days)`. Failures and drops in the integrations write one-event trace lines since D-077 (`claude_recall_failed`, `claude_memory_purged`, `claude_memory_pruned`, `claude_memory_rollback_failed`, `agents_recall_failed`, `agents_mirror_failed`, `agents_mirror_dropped`, `agents_forget_failed`, `agents_prune_failed`, `agents_close_failed`, `agents_unanswered_dropped`, `langgraph_recall_failed`, `langgraph_write_failed`, `langgraph_close_failed`, `langgraph_tool_failed`), holding only the space, counts, and the error's type name and status; `delete_all` removes them. The success events `claude_recall`, `agents_recall` and `langgraph_recall` no longer exist (they were never written; the `retrieve` line is the record of a recall). A failed LangGraph memory tool also logs a warning on `cuemem.integrations.langgraph`.
