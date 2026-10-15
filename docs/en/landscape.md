# Open-source peers: comparison and what CueMem borrowed (N12)

English edition of [docs/landscape.md](../landscape.md), which remains the source of truth.

> This document answers two questions: how CueMem compares with well-known open-source memory projects on GitHub (Mem0, Letta, Graphiti / Zep, Cognee, Supermemory, …), and what N12 borrowed from them, and what it deliberately did not. For the research-paper view, see the Chinese [related work](../related-work.md); this page covers engineering projects only.
>
> Checked on 2026-10-14. Stars and licences come from the GitHub API; API details come from each project's README and source. Vendor docs sites that the development environment could not reach (docs.mem0.ai, docs.letta.com, help.getzep.com, supermemory.ai) were read through search snippets and are marked "(snippet)"; anything not confirmed is marked "unverified". Every score that a project reports about itself uses its own reader, judge and prompts, and **cannot** be compared directly with CueMem's numbers (§4).

## 1. Summary

- **The research side is rare among these projects.** CueMem makes "should we recall, and how deeply" a trainable controller that is evaluated on its own. Facts carry two timelines and are invalidated, not deleted. The original photo is looked at again at answer time. The evaluation uses test sets once, with methods frozen in advance, and runs external baselines (Mem0; from N12 also Graphiti, LangMem, Cognee and A-MEM) under the same models and judge. Most of the projects below have none of these, or publish no ablations for them.
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
| Other languages / framework integrations | Python only (plus MCP, HTTP, the proxy); OpenAI Agents SDK, LangGraph / LangChain, Claude's memory tool ([integrations](integrations.md)) | TypeScript, a dozen frameworks | TypeScript, Vercel AI SDK | A dozen frameworks | TypeScript, Rust, Claude Code plugin | TypeScript, Vercel AI SDK middleware, several frameworks |

## 4. Scores: published (not comparable) and same-protocol

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
| **CueMem** L1 / L2 / controller | 90.5 / 93.2 / 90.5 | deepseek-flash | [technical report](report.md) | Full-context baseline 73.0; see the session-ID leak note below |

### Same-protocol comparison (N12)

The scores above cannot be compared. What can be compared directly is the external systems we ran ourselves under the same protocol as CueMem (D-065). The full tables, per-type results, costs and paired tests are in the Chinese [N12 external baselines](../results/n12-baselines/README.md).

- **Method**: each system uses its own write and read pipeline, following its authors' published evaluation code where it exists. The extraction model and embeddings (through the eval gateway), the reader prompt, the reader model and the judge are the same as CueMem's.
- **Data**: LoCoMo test200r (200 questions, stratified, no adversarial questions) over 7 conversations (conv-42, 43, 44, 47, 48, 49, 50): 4,431 turns, 202 sessions. Graphiti, LangMem, Cognee and A-MEM were run on LoCoMo only (Mem0 also has LongMemEval test100r runs, from N11 and the leak ablation).
- **Two protocols**:
  - pds (DeepSeek only): every role is deepseek-flash, including the judge (the same model as the reader, so there is a self-preference risk); embeddings are the local BAAI/bge-small-en-v1.5; no reranker.
  - pq (Qwen only): extraction, answering and the external systems' internal calls use qwen3.8-flash; the judge is qwen3.8-max (the same family as the reader); embeddings are qwen3.7-text-embedding, reranking is qwen3-rerank.
- One run per configuration; pairwise comparisons are exact sign tests on the same questions.

| Method | pds | pq | Nominal write cost, pds / pq |
|---|---|---|---|
| Full context | 84.5% | 80.0% | — |
| Naive RAG (dense top-10) | 59.5% | 64.0% | $0 / $0.006 |
| **CueMem L1** | 82.5% | 81.5% | $0.529 / $0.238 |
| **CueMem L2** | **87.0%** | **85.0%** | Same (L1, L2 and the controller share one write) |
| CueMem controller | 83.5% | 83.5% | Same |
| Mem0 2.2.1 (open source) | 81.0% | 77.5% | $1.381 / $0.712 |
| LangMem 0.0.30 | 78.0% | 30.0% | $0.563 / $0.225 |
| Cognee 1.6.4 | 77.5% | 74.0% | $2.845 / $1.466 |
| Graphiti (graphiti-core 0.30.2, Kuzu) | 59.0% | 41.5% | $9.992 / $4.515 |
| A-MEM (agiresearch/A-mem@ceffb86, k = 10) | 58.0% | 59.5% | $17.567 / $3.136 |
| A-MEM, k = 40 (sensitivity, same build) | 75.0% | 65.0% | Same |

Write costs are totals for the 7 conversations at list price, without caching (D-010); these are the figures to compare. Actual spend (after cache hits and discounts; it depends on timing and on what earlier pilot runs had cached, so it is for budgeting, not for comparing methods), pds / pq: CueMem $0.222 / $0.151, Mem0 $0.242 / $0.317, LangMem $0.239 / $0.064, Cognee $0.797 / $1.154, Graphiti $1.508 / $3.908, A-MEM $7.877 / $2.878.

- **CueMem L2 is ahead of all five external systems under both protocols, and every difference is significant**:
  - pds: against Mem0 +17 / −5 (p = 0.017), LangMem +25 / −7 (p = 0.002), Cognee +27 / −8 (p = 0.002), Graphiti +62 / −6 (p < 0.001), A-MEM +64 / −6 (p < 0.001);
  - pq: against Mem0 +24 / −9 (p = 0.014), Cognee +30 / −8 (p < 0.001), LangMem, Graphiti and A-MEM all p < 0.001;
  - also against A-MEM with k = 40 (pds +32 / −8, p < 0.001);
  - it is above full context, but not significantly (pds +12 / −7, p = 0.36; pq +19 / −9, p = 0.087).
- **L1, whose recall makes no LLM call, is not significantly different from Mem0, LangMem and Cognee under pds**: against Mem0 +14 / −11 (p = 0.69), LangMem +24 / −15 (p = 0.20), Cognee +20 / −10 (p = 0.10); against Graphiti +55 / −8 and A-MEM +56 / −7 (p < 0.001), and against A-MEM with k = 40 +27 / −12 (p = 0.024). Under pq it is significantly ahead of Cognee, +25 / −10 (p = 0.017), and A-MEM (p < 0.001), and not significantly different from Mem0, +19 / −11 (p = 0.20).
- **A-MEM is sensitive to how many notes it retrieves**: with the paper's main setting, k = 10, it is level with naive RAG; with k = 40, the authors' setting for GPT-4o-class models, it gains 17 points under pds (vs k = 10, +40 / −6, p < 0.001), at 29,550 input tokens per question, 2.4 times L2's.
- **Model sensitivity**: CueMem, Mem0, Cognee and A-MEM (k = 10) move by 0–3.5 points from pds to pq; Graphiti and LangMem drop sharply.
  - Graphiti: with its own prompts, qwen3.8-flash extracts far fewer entities (1,253 → 462 over the 7 conversations; facts 3,426 → 1,774). Under pq each conversation logged 118–270 "Target entity not found" warnings, and those facts were dropped. Turn recall under pq is 0.545 (naive RAG 0.845, CueMem L1 0.95).
  - LangMem: qwen3.8-flash uses PatchDoc to merge new information into a few long memories (705 → 123), and the sessions that lost their writes go from 2 / 202 to 23 / 202.
  - Neither system's prompts or thresholds were re-tuned for Qwen. Conversely, CueMem's extraction prompts and thresholds were developed with qwen3.8-flash and Qwen embeddings (N3–N7), so pq is its home ground. pds is the protocol where no system's prompts were tuned for the model, and L2's lead holds there too.

In N11, under the original protocol (Qwen for extraction and embeddings, deepseek-flash as reader, qwen3.8-max as judge), open-source Mem0 2.2 was compared with CueMem (D-055): on 200 LoCoMo questions, 80.5% against L1's 81.0% (a tie); on 100 LongMemEval questions, 83.0% against 92.0% (p = 0.049). Mem0's writes cost 2.3–2.9 times as much, and it lost on temporal reasoning, knowledge updates and details the assistant gave. Every retrieval-based LongMemEval number published in N2–N11 (this pair and CueMem's L1 / L2 / controller in the LongMemEval table above included; full context is unaffected) carried a session-ID leak, now fixed and measured (D-067, Chinese [leak ablation](../results/n12-leak/README.md)): in the test100r ablation, L1, L2 and the controller show no measurable inflation (a point or two cannot be ruled out); after the fix the same setup gives L1 90% against Mem0 79%, so Mem0's published 83% may be a few points high.

The mem0ai 2.2.1 tested here is the add-only v3 algorithm: the PyPI package is numbered 2.x, but its source labels the extraction prompt "V3 Additive Extraction Prompt (ADD-only with memory linking)", and `add` no longer asks the LLM to UPDATE or DELETE.

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

1. **More framework integrations**: the OpenAI Agents SDK, LangGraph / LangChain and Claude's memory tool are done ([integrations](integrations.md)). Still missing: TypeScript and Vercel AI SDK middleware, a LangGraph `BaseStore`, CrewAI, AutoGen. The Claude memory tool backend has not been run against the real Claude API.
2. **Multi-tenancy**: there is one shared access token. Memory spaces are isolated by user_id, but there is no per-user authentication. Multi-instance deployments need Postgres instead of SQLite.
3. **More external baselines**: Graphiti, LangMem, Cognee and A-MEM have been run under the same protocol (LoCoMo, DeepSeek-only and Qwen-only protocols; see §4). Not yet run: Letta, MemOS, Supermemory and others. Graphiti, LangMem, Cognee and A-MEM were run on LoCoMo only (Mem0 also has LongMemEval test100r runs, from N11 and the leak ablation): their LongMemEval writes would cost too much and take too long (Graphiti writes one episode per message; A-MEM takes about 15 s per turn under Qwen). Each configuration was run once.
4. **Validating the presets**: the OpenAI and Ollama presets are covered by offline tests only and have not been evaluated on real models. The consolidation similarity thresholds were tuned for Qwen embeddings and may need re-tuning for another embedding model.
5. **Metadata and expiry**: Mem0's `metadata` filters and `expiration_date`; Supermemory's automatically expiring temporary facts.
6. **An English CLI**: CLI messages are in Chinese. The SDK, HTTP API, MCP, proxy and UI are in English.
