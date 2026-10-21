# Framework integrations (N12)

English edition of [docs/integrations.md](../integrations.md).

There are four ways to plug CueMem into an existing agent stack, from least to most code:

| Way in | Good for | Who decides whether to recall | Where |
|---|---|---|---|
| OpenAI-compatible memory proxy | Any app on the OpenAI SDK, with no code changes | The controller, every turn | `cuemem proxy`, README |
| MCP server | Claude Code, Claude Desktop and other MCP clients | The model calls `recall`; the controller only picks the depth | `cuemem mcp` |
| Framework integrations (this page) | OpenAI Agents SDK, LangGraph / LangChain, Claude's memory tool | The controller, before every model call; the model can also search with tools | `cuemem.integrations.*` |
| Python SDK | Your own agent loop | You | `from cuemem import Memory` |

All three integrations share `cuemem/evidence.py`.
- **Controller first:** the controller picks a level (L0 recalls nothing and gives only the profile).
- **One evidence format:** the recalled facts and conversation excerpts are laid out exactly as the proxy and the reader lay them out, and placed at the start of the latest user message.
- **Prefix caching is kept:** the system prompt and earlier messages are untouched, so provider prefix caching keeps working.
- **Same tools as MCP:** recall / remember / forget have the same names, descriptions and semantics as the MCP tools.
- **Memories are framed as data:** the evidence block and the recall tools' listing say that their content is quoted from past conversations, information and not instructions, and a stored `<memory` or `</memory` is escaped so it cannot close the block early. This reduces prompt injection from stored memories but cannot prevent it.
- **Nothing twice:** in OpenAI Agents and LangGraph, rounds of the current conversation that are still in the request are left out of the evidence, photo rounds included (`evidence.stored_text` and `in_context_rounds`, shared with the proxy).

Install with `uv sync --extra integrations`, or separately with `--extra openai-agents`, `--extra langgraph` or `--extra anthropic`. With only a DeepSeek key, add `--extra local` and use `Memory(preset="deepseek")`.

**Peak hours.** The SDK, and so every integration built on a `Memory`, defaults to `peak_policy="warn"`: during DeepSeek's weekday peak hours (Beijing time) model calls go out at peak price instead of pausing until the peak ends. This applies even without a preset, because the default configuration's reader is DeepSeek. For batch work that should wait for the off-peak price, use `Memory(peak_policy="wait")`; the SDK then logs a warning on the `cuemem.sdk` logger when it starts waiting. Evaluations and `cuemem eval run --peak` are not affected.

## OpenAI Agents SDK

`cuemem.integrations.openai_agents` (openai-agents 0.23). It has three parts, usable separately:

- `memory_tools(memory, user_id=...)`: the recall, remember and forget tools.
- `memory_input_filter(memory, user_id=...)`: pass it as `RunConfig.call_model_input_filter`. Before every model call, it runs the controller and injects the evidence into the copy sent to the model.
- `CueMemSession(session_id, memory, user_id=...)`: an implementation of the Session protocol.
  - An inner session (SQLiteSession by default) keeps the exact short-term history.
  - User and assistant text is also written into CueMem, one question and answer per round.
  - When the conversation ends, the whole session is extracted into facts.

```python
from agents import Agent, RunConfig, Runner
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
- **Server-managed state:** with `conversation_id` or `previous_response_id`, use `memory_input_filter(..., placement="instructions")`. The evidence then goes into that call's instructions and is never stored server-side with the user message.
- **Forgetting:**
  - Messages removed with `pop_item` or `clear_session` are also forgotten in long-term memory.
  - Turns deleted with the forget tool are pruned from the inner session the next time history is read.
- **Unanswered messages:** user messages the model never answered (for example, blocked by an input guardrail) are not stored.

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
  - They are stored per user in a small side store, `claude_memory.sqlite` in the data directory.
  - Every change is also written as a note into CueMem's raw log, which consolidation turns into facts.
  - Deleting a file forgets what it contributed.
- **Safety:** strict path validation keeps everything inside `/memories`, and file sizes are limited.
- **Recall:** `with_recall` / `awith_recall` run the controller and inject evidence before the request goes to Claude.

```python
import anthropic
from cuemem import Memory
from cuemem.integrations.claude import CueMemMemoryTool, delete_all, with_recall

# default config: Qwen extracts and retrieves, DeepSeek answers; or Memory(preset="deepseek")
memory = Memory()
client = anthropic.Anthropic()

with CueMemMemoryTool(memory, user_id="alice") as tool:
    history = []  # append-only: with_recall only touches the newest user message
    for text in ["I moved to Lisbon in 2026-09.", "What should I cook tonight?"]:
        user = {"role": "user", "content": text}
        history, evidence = with_recall([*history, user], memory, user_id="alice")
        runner = client.beta.messages.tool_runner(
            model="claude-opus-5-5",
            max_tokens=16000,
            tools=[tool],
            messages=history,
        )
        for message in runner:
            history.append(message.to_param())
            results = runner.generate_tool_call_response()
            if results is not None:
                history.append(results)

delete_all(memory, user_id="alice")  # memories and Claude's files, together
```

- **Not run against the real Claude API:** there was no Anthropic key, and DeepSeek's Anthropic-compatible endpoint rejects `memory_20250818`. Offline tests drive every tool command directly. The live smoke test drove the same commands through DeepSeek as a regular tool.
- **Removing a line from a file** records "this no longer holds"; extraction and reconciliation decide what follows. Only deleting the whole file is a guaranteed forget.
- **Export:** Claude's files are not in `Memory.export`, which only has the notes in the raw log.

## Limitations

- **Open sessions:** they are tracked in memory per process. When several processes share a data directory (with a Qdrant server), one process can hand over a session that another is still using.
- **Images:** the integrations inject text evidence only. When the controller wants to look at the original photo (`see`), the photo is not attached to the model call (the same as the proxy).
- **Async:** CueMem itself is synchronous and runs in threads. In async frameworks use `AsyncMemory` or each integration's async variants.
- **Traces:** every automatic recall and recall tool call writes a `retrieve` trace (the question, the retrieval mode and the IDs it retrieved) to `runs/traces/*.jsonl` in the data directory. Forgetting deletes the traces that list a deleted memory, together with the cached query vectors of their questions.
