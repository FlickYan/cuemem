# Open-source peers: comparison and what CueMem borrowed (N12)

English edition of [docs/landscape.md](../landscape.md), which remains the source of truth.

> This document answers two questions: how CueMem compares with well-known open-source memory projects on GitHub (Mem0, Letta, Graphiti / Zep, Cognee, Supermemory, …), and what N12 borrowed from them, and what it deliberately did not. For the research-paper view, see the Chinese [related work](../related-work.md); this page covers engineering projects only.
>
> Checked on 2026-10-14. Stars and licences come from the GitHub API; API details come from each project's README and source. Vendor docs sites that the development environment could not reach (docs.mem0.ai, docs.letta.com, help.getzep.com, supermemory.ai) were read through search snippets and are marked "(snippet)"; anything not confirmed is marked "unverified". Every score that a project reports about itself uses its own reader, judge and prompts, and **cannot** be compared directly with CueMem's numbers (§4).

## 1. Summary

- **The research side is rare among these projects.** CueMem makes "should we recall, and how deeply" a trainable controller that is evaluated on its own. Facts carry two timelines and are invalidated, not deleted. The original photo is looked at again at answer time. The evaluation uses test sets once, with methods frozen in advance, and runs an external baseline (Mem0) under the same models and judge. Most of the projects below have none of these, or publish no ablations for them.
- **Before N12, CueMem lacked the layer that makes a library usable.** Anyone who has used Mem0 expects three lines of code after `pip install`. CueMem could not do that:
  - its config lived in the repository and its data was written next to it, so a pip-installed copy did not run;
  - the only models were DeepSeek and Qwen, with embeddings through Qwen's native API: no OpenAI, no local models;
  - it was usable only from the CLI, the HTTP API and MCP: no Python SDK and no "change the base_url" integration;
  - there was no export or import, no preview before forgetting, no way to tune what extraction keeps for a given app, and no container image.
- **N12 adds that layer, following the mainstream projects (§5),** while keeping CueMem's own semantics: facts are never rewritten in place, recall goes through the controller first, and forgetting runs the full cascade.
- **Deliberately not copied:** LLM-edited memory as the default (Letta), add-only memory (Mem0 v3), graph databases and twenty-odd vector-store backends (§6).

## 2. The projects

| Project | Stars | Licence | Memory model | Who triggers retrieval | Updates |
|---|---|---|---|---|---|
| [mem0ai/mem0](https://github.com/mem0ai/mem0) | 67.2k | Apache-2.0 | Extracted facts + entity links (graph memory only on the paid platform) | The app calls `search`; the OpenAI-compatible wrapper searches every turn | v3 (2026-04) is add-only: no more UPDATE / DELETE |
| [letta-ai/letta](https://github.com/letta-ai/letta) (+ letta-code 3.6k) | 25.1k | Apache-2.0 | In-context memory blocks, archival and recall memory; Letta Code keeps memory in git | The LLM, through tools | The LLM rewrites blocks; sleep-time agents consolidate in the background |
| [getzep/graphiti](https://github.com/getzep/graphiti) (zep 5.0k) | 31.7k | Apache-2.0 | Temporal knowledge graph: episodes, entities, fact edges with validity, communities | The app calls `search`; MCP tools | Two timelines; old edges are invalidated, not deleted |
| [topoteretes/cognee](https://github.com/topoteretes/cognee) | 32.3k | Apache-2.0 | Knowledge graph + vectors, optional ontologies | The app calls `recall`; a rule-based router picks the search strategy | A contradiction-detection flag (off by default) |
| [supermemoryai/supermemory](https://github.com/supermemoryai/supermemory) | 31.2k | MIT | Facts and documents linked by updates / extends / derives; static and dynamic profiles | The app, middleware, MCP, or a base-URL proxy | Newest wins (`isLatest`) with history kept; temporary facts expire |
| [MemTensor/MemOS](https://github.com/MemTensor/MemOS) | 11.9k | Apache-2.0 | MemCubes: text, KV cache, LoRA and preference memory | The app calls `search` (fast / fine) | Correction by natural-language feedback |
| [NevaMind-AI/memU](https://github.com/NevaMind-AI/memU) | 14.5k | see repo | A "wiki" of Markdown skills | The host agent's instruction file | The host agent rewrites |
| [Mirix-AI/MIRIX](https://github.com/Mirix-AI/MIRIX) | 3.5k | Apache-2.0 | Six memory types, each run by its own agent (built on Letta) | The app calls `retrieve_with_conversation` | Each memory agent rewrites |
| [memodb-io/memobase](https://github.com/memodb-io/memobase) | 2.9k | Apache-2.0 | Structured user profile (topic / sub-topic) + event timeline | Always injected (`context()`) | Profile slots are overwritten |
| [langchain-ai/langmem](https://github.com/langchain-ai/langmem) | 1.7k | MIT | Free-text or Pydantic memories in a LangGraph store | LLM tools, or a background manager | The manager can insert / update / delete |
| [agiresearch/A-mem](https://github.com/agiresearch/A-mem) | 1.2k | MIT | Zettelkasten notes; new notes "evolve" old ones | The app calls `search` | The LLM rewrites neighbouring notes |
| [redis/agent-memory-server](https://github.com/redis/agent-memory-server) | 322 | see repo | Session working memory promoted to long-term memory | LLM tools or prompt injection | Deduplication; `forget(policy, dry_run)` |
| **CueMem** | — | MIT | Immutable raw log + derived facts (two timelines), entities, summaries, profile, multi-granularity index, original photos | **A trainable controller** (L0 / L1 / L2, photos or not); the model can search more with a tool | Invalidate, don't delete; "the world changed" vs "it was a mistake", confirmed by an LLM |

A few changes in 2025–2026 matter here:
- **Mem0** v3 became add-only. GitHub [issue #4956](https://github.com/mem0ai/mem0/issues/4956) reports that contradictory facts pile up and that retrieval ignores recency. Graph memory moved to the paid platform.
- **Letta** moved its Python server to an archive branch. Active work is the TypeScript Letta Code, the Python client is deprecated, and Agent File was removed from Letta Code.
- **Zep** deprecated its Community Edition; Graphiti is the open-source core.
- **Cognee** added a `remember / recall / improve / forget` API and a portable format, COGX.
- **Redis Agent Memory Server** moved its open-source server to `V0/` and calls it a research artifact.

## 3. Capabilities side by side

"Not seen" means it was not found in the sources checked, not that it certainly does not exist.

| Capability | CueMem (N12) | Mem0 | Letta | Graphiti / Zep | Cognee | Supermemory |
|---|---|---|---|---|---|---|
| Python SDK | `Memory` / `AsyncMemory`, Mem0's method names | `Memory` / `AsyncMemory` | Deprecated (TypeScript first) | Fully async `Graphiti` | `remember` / `recall` and others | `supermemory` client |
| Zero-code integration | OpenAI-compatible HTTP proxy (`/v1/chat/completions`, streaming) | Client-side wrapper (`Mem0().chat.completions.create`) | — | — | — | Memory Router (base URL; its docs now redirect, current status unverified) |
| Retrieves on every turn? | No: the controller decides first; L0 retrieves nothing | Every turn in the wrapper | The LLM decides | The app decides | Rule-based routing | Every turn |
| Updates | Invalidate, don't delete; version chain via `history()` | v3 add-only; `history()` | The LLM rewrites | Invalidate, don't delete | Contradiction detection (off by default) | `isLatest` version chain |
| Two timelines | Replayed separately: `valid_at` (what was true then) and `known_at` (what the system believed then) | Recorded time; `expiration_date` | Not seen | Yes: edges carry `valid_at` / `invalid_at` and `created_at` / `expired_at` | Not seen | Not seen |
| Images | Original kept, context-aware captions, image-to-image search, the photo looked at again at answer time | A vision model turns it into text (`enable_vision`) | Not seen | No (text and JSON) | Vision or transcription models (details unverified) | OCR |
| Forgetting | Delete from the log and replay; follows citations; finds uncited restatements; **dry run** | `delete` / `delete_all`; expiry | Edit memory | `remove_episode` | `forget` | `forget_matching(dry_run)` |
| Export / import | `export` / `load` (raw log + photos; derived memory is rebuilt) | Paid platform only | Agent File (removed from Letta Code) | Not seen | COGX, with importers from Mem0 / Zep / Letta | Not seen |
| Custom extraction | `consolidation.instructions` | `custom_instructions` | Block labels and limits | Pydantic entity / edge types, `custom_extraction_instructions` | Custom data models, ontologies | Not seen |
| Local models, no key | `ollama` preset | Ollama, LM Studio, vLLM | Unverified | Ollama and other compatible endpoints | Ollama, llama.cpp, a local extractor | Local mode |
| Deployment | Docker image, compose (+ Qdrant server), MCP, HTTP | docker compose, hosted platform, hosted MCP | `letta server` | REST, MCP image | Docker, cloud | Local mode, Docker, hosted MCP |
| Storage | SQLite + Qdrant (local or server) | ~26 vector stores | Postgres and others | Neo4j, FalkorDB, Neptune | LanceDB, pgvector, Kuzu, Neo4j, … | Hosted / local |
| Published evaluation | Test sets used once, methods frozen; three benchmarks, three readers, ablations, hand audit, a second judge | `memory-benchmarks` repo (platform models undisclosed) | Blog | Paper | Own harness | Own harness |
| Other languages / framework integrations | Python only (plus MCP, HTTP, the proxy) | TypeScript, a dozen frameworks | TypeScript, Vercel AI SDK | A dozen frameworks | TypeScript, Rust, Claude Code plugin | TypeScript, Vercel AI SDK middleware, several frameworks |

## 4. Published scores (not comparable)

The numbers below are **self-reported** by each project. They use different setups:
- **Readers** range from gpt-4o-mini to GPT-5 and Gemini-3.
- **Judges** include gpt-4o-mini, GPT-4o and GPT-4.1, and some are undisclosed. Mem0's LoCoMo judge prompt asks the judge to be generous.
- **LoCoMo's adversarial category** is dropped by some and kept by others.
- **Retrieval budgets** range from 1.6K to 7K tokens.
- **LongMemEval** released a cleaned dataset in September 2025.

Against differences like these, a gap of one or two points means nothing.

**LoCoMo** (LLM-judged accuracy, mostly categories 1–4)

| System | Score | Reader / judge | Source | Notes |
|---|---|---|---|---|
| Mem0 (paper) | 66.9; graph variant 68.4 | gpt-4o-mini / gpt-4o-mini | arXiv 2504.19413 | Zep, LangMem and OpenAI memory, run by Mem0: 66.0, 58.1, 52.9 |
| Zep (corrected) | 75.1 | Not stated | getzep/zep-papers | The original ~84% miscounted category 5 and was withdrawn; Mem0's re-run gave 58.4; still disputed |
| Letta (filesystem agent) | 74.0 | gpt-4o-mini | Letta blog (snippet) | |
| Memobase v0.0.37 | 75.8 | Mem0's harness | memobase repo | |
| MemOS | 75.8 (2025-11) | gpt-4o-mini | MemOS repo | 88.8 with its own harness |
| MIRIX | 85.4 | gpt-4.1-mini / GPT-4.1 | arXiv 2507.07957 (snippet) | |
| Hindsight | 89.6 / 85.7 / 83.2 | Gemini-3 / OSS-120B / OSS-20B | vectorize-io/hindsight-benchmarks | Claims independent reproduction |
| Mem0 platform v3 | 92.5 | Undisclosed | mem0ai/memory-benchmarks | Managed platform, top-200 retrieval; the open-source SDK lacks these optimisations |
| **CueMem** L1 / L2 / controller | 79.7 / 84.5 / 82.2 | deepseek-flash / qwen3.8-max | [technical report](report.md) | 1,155 questions from the last 7 conversations; test set used once |

**LongMemEval_S** (500 questions; CueMem uses a 400-question test split drawn in advance)

| System | Score | Reader | Source | Notes |
|---|---|---|---|---|
| Full context (original paper) | 60.6 | GPT-4o | arXiv 2410.10813 (snippet) | To be checked against the PDF |
| Zep | 71.2 / 63.8 | gpt-4o / gpt-4o-mini | arXiv 2501.13956 | Full context in the same paper: 60.2 / 55.4 |
| Supermemory | 81.6 / 84.6 / 85.2 | gpt-4o / GPT-5 / Gemini-3 | supermemory.ai (snippet) | A later "~99%" post calls itself a parody; excluded |
| Mastra Observational Memory | 84.2 / 94.9 | gpt-4o / gpt-5-mini | mastra.ai (snippet) | |
| Hindsight | 91.4 / 89.0 / 83.6 | Gemini-3 / OSS-120B / OSS-20B | hindsight-benchmarks | |
| Mem0 open source | 91.0 | GPT-5 for extraction, answers and judging | mem0ai/memory-benchmarks | Platform 94.4, models undisclosed |
| MemOS | 77.8 (2025-11) | gpt-4o-mini | MemOS repo | 89.2 with its own harness |
| **CueMem** L1 / L2 / controller | 90.5 / 93.2 / 90.5 | deepseek-flash | [technical report](report.md) | Full-context baseline 73.0 |

**The only directly comparable pair** comes from N11: open-source Mem0 2.2, run with the same extraction model, embeddings, reader and judge (D-055).
- LoCoMo, 200 questions: Mem0 80.5% vs CueMem L1 81.0% (a tie).
- LongMemEval, 100 questions: 83.0% vs 92.0% (p = 0.049).
- Mem0's writes cost 2.3–2.9 times as much. It loses on temporal reasoning, knowledge updates and details the assistant gave.

That was Mem0 2.x; the add-only v3 has not been tested.

## 5. What N12 borrowed

| New capability | Following | CueMem's version and how it differs | Where | Decision |
|---|---|---|---|---|
| Config shipped in the package, a data directory, works after `pip install` | Every Python package | Unchanged in a source checkout; installed copies keep data in `~/.cuemem` (`CUEMEM_HOME`) | `config.py` | D-060 |
| Switching model providers: OpenAI, Ollama (local), any OpenAI-compatible endpoint | "Bring your own model" in Mem0, Graphiti, Cognee | Presets overlay the default config. Embeddings go through the OpenAI-compatible `/embeddings`. Reasoning-model parameter quirks are declared in the model catalogue. Requests under the default config are byte-for-byte unchanged | `configs/presets/`, `embed/openai_compat.py` | D-060 |
| Python SDK: `add / search / get_all / get / history / delete / delete_all`, `AsyncMemory`, `from_config` | Mem0 | The same method names and `{"results": [...]}`. Accepts v3's `filters=` and `top_k=`. `add(wait=True)` returns ADD / UPDATE events, and old versions stay in the chain. `search` goes through the controller | `sdk.py` | D-061 |
| Memory by changing the base_url | Mem0's proxy, Supermemory's Memory Router, Memobase's OpenAI patch | The controller decides first: at L0, no retrieved memories are injected. The evidence block goes before the last user message and the system prompt is untouched, so prefix caching keeps working. Streaming. The decision is reported in response headers | `api/proxy.py` | D-062 |
| Export / import | Letta's Agent File, Cognee's COGX, MemOS's dump / load | Only the raw log and photos are moved; derived memory is rebuilt on the target. Gaps left by forgetting are preserved | `portable.py` | D-063 |
| A dry run before forgetting | `dry_run` in Supermemory and Redis Agent Memory Server | The preview follows citations and finds uncited restatements in the same way, and deletes nothing | `memory.py` | D-063 |
| Tuning extraction for an app | Mem0's `custom_instructions`, Graphiti's `custom_extraction_instructions` | Appended to the extraction prompt; changing it rebuilds derived memory (the consolidation fingerprint changes) | `write/extract.py` | D-064 |
| Container image and compose | Docker deployments of Mem0, Graphiti, Cognee | One process serves the UI, the HTTP API and the memory proxy; compose adds a Qdrant server | `Dockerfile`, `docker-compose.yml` | D-060 |

## 6. Deliberately not copied

- **The LLM deciding retrieval and editing memory by default** (tool-style memory in Letta and LangMem). This is the comparison arm of CueMem's research (B3, ReAct). N9 found it strong on text and weak on images (46.7% on MemLens), and more expensive ($3.15 per 1,000 LoCoMo questions against $1.44 for L1). It is kept as `mode="agent"` and as the agent's `memory_search` escape hatch, not as the default.
- **Add-only memory** (Mem0 v3). It saves the reconciliation calls, but contradictory facts pile up, and "where do I live now" is left to the reader to judge on the spot. In CueMem's H4 experiment, validity periods by themselves showed no gain on knowledge-update questions (every variant scored 96.5%; see the technical report). But the version chain is what makes `history()`, replaying either timeline, and restoring an older fact after forgetting possible. These are product capabilities, not just scores.
- **Graph databases and twenty-odd vector stores.** At the scale of one user's memory, embedded SQLite + Qdrant is enough: in N11's latency benchmark, the local part of the read path takes 91 ms with a Qdrant server at 46k retrieval units. Replacing SQLite for multi-instance deployments is the known limitation (D-056), and it matters more than more vector stores.
- **Hosted-platform features** (webhooks, categories, platform-only retrieval tuning): out of scope for an open-source project.

## 7. What is still missing

In the order a user would run into them:

1. **Framework integrations**: a LangGraph `BaseStore`, the OpenAI Agents SDK, Vercel AI SDK middleware, a backend for Claude's memory tool (Supermemory has one). Today the ways in are the SDK, HTTP, MCP and the proxy.
2. **Multi-tenancy**: there is one shared access token. Memory spaces are isolated by user_id, but there is no per-user authentication. Multi-instance deployments need Postgres instead of SQLite.
3. **More external baselines** under the same protocol: Graphiti, LangMem, Letta, and the add-only Mem0 v3. These need API keys and budget.
4. **Validating the presets**: the OpenAI and Ollama presets are covered by offline tests only and have not been evaluated on real models. The consolidation similarity thresholds were tuned for Qwen embeddings and may need re-tuning for another embedding model.
5. **Metadata and expiry**: Mem0's `metadata` filters and `expiration_date`; Supermemory's automatically expiring temporary facts.
6. **An English CLI**: CLI messages are in Chinese. The SDK, HTTP API, MCP, proxy and UI are in English.
