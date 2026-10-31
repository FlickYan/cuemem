# Examples

Run these from a checkout of the repository (`uv sync` first). Each script's own docstring (in Chinese) has more detail. Which API keys a CueMem configuration needs:

| Configuration | Keys |
|---|---|
| Default | `DEEPSEEK_API_KEY` and `DASHSCOPE_API_KEY` (QwenCloud) |
| `CUEMEM_PRESET=openai` (or `--preset openai`) | `OPENAI_API_KEY` |
| `CUEMEM_PRESET=deepseek` | `DEEPSEEK_API_KEY`; embeddings run locally and need the `local` extra |
| `CUEMEM_PRESET=ollama` | none (a local Ollama) |
| Hash embeddings (`embedding.backend: hash`) | none for writing and retrieval; answering and extraction still need an LLM key |

A data directory remembers the embedding model it was first used with, even by a command that failed, and refuses a different one later. Below, the scripts that keep data run with `CUEMEM_HOME=$(mktemp -d)`, so trying another preset never ties your real data directory (the repository root in a checkout, `~/.cuemem` after a pip install, `/data` in Docker) to a different embedding model.

| File | What it shows | Keys | Command |
|---|---|---|---|
| [`keyless.py`](keyless.py) | Write one message and retrieve it with offline hash embeddings: no model is called | none | `uv run python examples/keyless.py` |
| [`quickstart.py`](quickstart.py) | The Python SDK: `add` with extraction, `search` (the controller picks the depth), `answer` with citations, a fact's version history, and forgetting with a dry run first. Data goes to a temporary directory | default, `openai` or `ollama` | `uv run python examples/quickstart.py` (add `--preset openai` or `--preset ollama`) |
| [`openai_proxy.py`](openai_proxy.py) | The OpenAI SDK through CueMem's memory proxy: only `base_url` changes. The response headers carry the controller's decision (`X-CueMem-Level`, `X-CueMem-Reasons`) | those of the proxy's configuration | Start `uv run cuemem proxy` (port 8787) in another terminal, then `uv run python examples/openai_proxy.py`. With `uv run cuemem serve`, set `CUEMEM_PROXY_URL=http://127.0.0.1:7860/v1`; with an access token, set `CUEMEM_API_TOKEN`. The proxy keeps memory in its own data directory |
| [`toy_conversation.en.json`](toy_conversation.en.json) ([Chinese](toy_conversation.json)) | Five dated sessions in which facts change (a new job, a move to Singapore, a lower budget), for the CLI walkthrough in the README | default (or any preset) | `uv run cuemem ingest examples/toy_conversation.en.json --consolidate`, then `uv run cuemem ask "Where do I work now?" --space demo-en --at 2025-10-10` |
| [`integrations/claude_example.py`](integrations/claude_example.py) | Claude's memory tool backed by CueMem: Claude reads and writes `/memories`, and CueMem turns each change into versioned facts | `ANTHROPIC_API_KEY` and `DASHSCOPE_API_KEY` | `uv sync --extra anthropic`, then `CUEMEM_HOME=$(mktemp -d) uv run python examples/integrations/claude_example.py`. With only a DeepSeek key: `uv sync --extra anthropic --extra local` and `CUEMEM_PRESET=deepseek DEEPSEEK_API_KEY=... CUEMEM_HOME=$(mktemp -d) uv run python examples/integrations/claude_example.py` |
| [`integrations/langgraph_example.py`](integrations/langgraph_example.py) | LangChain / LangGraph: `create_agent` with CueMem's middleware; a second thread remembers the first | `DEEPSEEK_API_KEY` (the `deepseek` preset) | `uv run --extra local --extra langgraph --with langchain-openai python examples/integrations/langgraph_example.py` (data goes to a temporary directory unless `CUEMEM_HOME` is set) |
| [`integrations/openai_agents_example.py`](integrations/openai_agents_example.py) | The OpenAI Agents SDK with a CueMem session, an input filter and memory tools; the second conversation remembers the first | `DEEPSEEK_API_KEY` (the `deepseek` preset), or `DASHSCOPE_API_KEY` with the default configuration | `uv sync --extra local --extra openai-agents`, then `CUEMEM_PRESET=deepseek CUEMEM_HOME=$(mktemp -d) uv run python examples/integrations/openai_agents_example.py` |

Without a checkout, install CueMem from GitHub (it is not on PyPI; a package named `cuemem` there is not this project): `pip install "cuemem[integrations,local] @ git+https://github.com/FlickYan/cuemem"`, then run the scripts with `python` instead of `uv run python`. The integrations are described in [docs/en/integrations.md](../docs/en/integrations.md).
