# CueMem

**Cue-driven, self-regulating, multimodal long-term memory for LLM agents**

English · [简体中文](README.zh-CN.md) · [Technical report](docs/en/report.md) · [Design](docs/en/design.md) · [API reference](docs/en/api.md) · [vs. Mem0, Letta, Zep…](docs/en/landscape.md) · [Demo guide](demo/README.en.md) · [60-second video](https://youtu.be/HbGS0_Yo4kI)

CueMem gives an LLM agent a memory that lasts across months of conversations:

- it **writes** what users say into facts that carry validity periods;
- it **recalls** them at the right depth;
- it **looks at the original photo again** when a question needs visual detail;
- it **forgets on request**, including replies that only restated the forgotten information.

The question the project studies is **who should decide whether to recall, and how deeply**. One option is the LLM itself, calling a search tool. The other is a cheap, trainable controller that runs before the LLM is called and never calls an LLM itself. CueMem builds the controller, evaluates both approaches on three public benchmarks under a frozen protocol, and reports where each one wins.

<a href="https://youtu.be/HbGS0_Yo4kI"><img alt="CueMem in 60 seconds: watch the intro on YouTube" src="docs/assets/video-thumbnail.png" width="720"></a>

*In the video, costs are answer-time costs (writing a history into memory costs extra, once; see [Results at a glance](#results-at-a-glance)), and its LongMemEval numbers predate the session-ID leak fix (D-067).*

![A shopping assistant that remembers: cross-session recall, a changed shoe size, finding the pair from a photo shared in August, a budget that changed, and forgetting a fact](docs/assets/demo.gif)

*The demo: Cue, a shopping assistant, on its third chat with a user. It recalls her across sessions, knows her shoe size changed, finds look-alikes of a photo she shared two months earlier, explains how her budget changed (with citations), and forgets a fact on request. The right-hand panel shows what the controller chose and why, what went into the context, and which memories the answer cites.*

## Use it

As a library, with Mem0's method names and call shapes (the differences are listed below):

```python
from cuemem import Memory

# OPENAI_API_KEY; "ollama" runs locally with no key; leave it out for DeepSeek + Qwen
m = Memory(preset="openai")

# Written online, extracted into facts; returns ADD / UPDATE events
events = m.add("I moved to Munich last month.", user_id="alice", wait=True)["results"]

m.search("Where do I live now?", user_id="alice")  # the controller decides how deeply to recall
m.answer("Where do I live now?", user_id="alice")  # an answer that cites the facts it used
m.history(events[0]["id"])  # every version of the fact: when and why it changed
m.delete(events[0]["id"], dry_run=True)  # what forgetting would delete, before deleting anything

# A multi-turn chat: each add is searchable at once, and the chat is extracted once, when it ends
with m.run(user_id="alice") as run_id:
    m.add("My budget for running shoes is $180.", user_id="alice", run_id=run_id)
    m.add("Actually, make that $150.", user_id="alice", run_id=run_id)
```

Calling `add` once per turn with the same `run_id` also works, but it re-extracts the whole run every time and replays derived memory; wrap the chat in `with m.run(...)` instead. The SDK defaults to `peak_policy="warn"`: during DeepSeek's weekday peak hours calls go out at peak price instead of pausing until the peak ends; batch jobs that want the off-peak price pass `Memory(peak_policy="wait")`. Close the memory when you are done, with `m.close()` or by opening it as `with Memory(...) as m:`; a script that exits without closing it prints a harmless Qdrant "Exception ignored" traceback.

The rest of Mem0's API is there too, with CueMem's semantics:

```python
m.add(
    "I prefer aisle seats.",
    user_id="alice",
    metadata={"source": "crm"},
    expiration_date="2026-12-31",
)
m.search("Seat preference?", user_id="alice", filters={"source": {"in": ["crm", "app"]}})
# A logged correction (Mem0's text=, or positionally); history() keeps both versions
m.update(events[0]["id"], text="I moved to Berlin, not Munich.")
m.delete_all(user_id="alice", run_id=run_id)  # forget one conversation; reset() wipes every space
```

- **metadata and expiry:** metadata is stored with the raw turns (never shown to the extraction model), carried by every fact and excerpt derived from them, and usable in `filters` with Mem0's operators (`eq`, `ne`, `gt`, `in`, `contains`, `AND` / `OR` / `NOT`, …). Expired memories leave `search`, `get_all` and the profile; `get` and `history` still show them.
- **`update`** logs "Correction: …" in the raw log; the fact extracted from it supersedes the old one as *corrected* (no reconciliation call needed), and forgetting the correction brings the old fact back.
- **`search` never comes back empty because the controller chose L0:** calling it is already a decision to look, so L0 is raised to L1 (`min_level="l0"` keeps the controller's choice). `answer()` still lets the controller skip memory.
- **Writes are safe to retry.** If embedding or the vector store fails, nothing is stored and the same call can be repeated. If extraction fails after the turns were stored, `add(wait=True)` returns `status: "queued"` with an `error` instead of raising, and extraction finishes on the next consolidation. `m.status(user_id="alice")` lists pending and failed extractions with their errors and the time of the automatic retry, and background failures are logged as warnings.
- **Where it differs from Mem0:** a config dict is CueMem's, not Mem0's (unknown keys raise ValueError, with a migration hint for Mem0-format configs); scores only rank results (rank fusion at L1, so there is no `threshold`; Mem0's `rerank` is `mode="l2"`); deleting a fact forgets the whole conversation round it came from, so other ids from that round then raise KeyError; `limit` and `top_k` must be positive; `show_expired=True` lists expired memories. Every id that `search` returns works with `get` and `delete`, and `media(id)` returns an image's bytes. Details in the [API reference](docs/en/api.md#coming-from-mem0).

Against a running `cuemem serve`, `MemoryClient` has the same methods and signatures, so switching between local and remote is one line, and several processes can share one server (local-mode Qdrant allows one process per data directory). The framework integrations below still need a local `Memory` (a `MemoryClient` raises TypeError there). Underneath is a Mem0-shaped HTTP API, `/api/v1/memories`, usable from any language; its OpenAPI snapshot ([docs/openapi-v1.json](docs/openapi-v1.json)) generates TypeScript types (`npx openapi-typescript docs/openapi-v1.json`), and the Python package ships type hints (`py.typed`):

```python
from cuemem import MemoryClient

m = MemoryClient("http://127.0.0.1:7860", api_key="...")  # instead of Memory(); same calls as above
```

```bash
curl -s -X POST http://127.0.0.1:7860/api/v1/memories -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"messages": "I moved to Berlin", "user_id": "alice", "wait": true}'
curl -s -X POST http://127.0.0.1:7860/api/v1/memories/search -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"query": "Where do I live?", "user_id": "alice"}'
```

Every method, route, error code and environment variable is in the [API reference](docs/en/api.md).

Or as a drop-in proxy for any app that uses an OpenAI-compatible API, with no code changes beyond the base URL (for Chat Completions, embeddings and plain-text Responses calls):

```bash
uv run cuemem --preset openai proxy          # http://127.0.0.1:8787/v1
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="unused")
messages = [{"role": "user", "content": "Any tips for my next running shoes?"}]
client.chat.completions.create(model="gpt-5.4-mini", messages=messages, user="alice")
# Response headers say what the controller did: X-CueMem-Level: L0 (no recall) / L1 / L2
```

- **Models:** every response names the upstream model actually used in `X-CueMem-Model`. A model name the proxy does not know is sent to the agent role's model with a logged warning; `--unknown-model reject` (or `proxy.unknown_model: reject`) answers 400 `model_not_found` instead.
- **Other endpoints:** `/v1/embeddings` goes to the configured embedding model and touches no memory. `/v1/responses` supports non-streaming text input with the same recall and write-back; streaming, tools, `previous_response_id` and images get 501, so use Chat Completions for those (the OpenAI Agents SDK has its own [integration](docs/en/integrations.md)).
- **Failures:** provider errors come back with their `Retry-After`, and an unreachable or timed-out upstream gives 502 or 504. A write-back that fails during an outage is kept in memory and written before the session's next one.
- The proxy accepts JSON requests only (`curl -d` needs `-H 'Content-Type: application/json'`) and refuses browser requests from other origins. Recalled memories go into a `<memory>` block that tells the model they are quoted from past conversations, information and not instructions, and a stored `</memory>` cannot close the block early. This reduces prompt injection from stored memories but cannot rule it out.

Or inside an agent framework: the OpenAI Agents SDK, LangGraph / LangChain, and Claude's memory tool, with the controller deciding before every model call ([integrations](docs/en/integrations.md)).

Also: an HTTP API with a memory browser (`cuemem serve`), an MCP server (stdio, or HTTP at `/mcp` of `cuemem serve`), export and import of a user's memory (`cuemem export / import`), access tokens scoped to users and agents, and a Docker image (`docker compose up`, with an access token). See [Quick start](#quick-start), [Deployment](#deployment) and the [examples](examples/) (an index of the examples with the keys and commands each needs).

## Results at a glance

Everything below comes from one frozen pass over the test sets (MemLens's 195 questions were also reported in N6, with methods fixed in advance), with every method, prompt and the controller model frozen beforehand. The reader is deepseek-flash, the judge is qwen3.8-max, and costs are nominal (peak list price, no cache) and cover answering only; the one-off write path (embedding + consolidation) is excluded.

| Method | LongMemEval (400 q) | LoCoMo (1,155 q) | MemLens 32K (195 q) | MemLens 128K (195 q) |
|---|---|---|---|---|
| Full context (long-context model) | 73.0% · $31.88 / 1k q | 82.0% · $6.90 | 51.8% · $8.25 | 46.2% · $40.60 |
| Naive RAG (dense top-10) | 82.0% · $1.68 | 68.2% · $0.42 | — | — |
| **L1 fast recall** (no LLM call) | 90.5% · $1.60 | 79.7% · $1.44 | 63.1% · $1.96 | 55.4% · $1.94 |
| **L2 deep recall** | **93.2%** · $3.41 | 84.5% · $3.25 | **65.6%** · $4.41 | **65.1%** · $4.43 |
| **Learned controller** (picks the level) | 90.5% · $2.42 | 82.2% · $2.21 | 60.0% · $3.14 | 60.5% · $3.61 |
| Agentic retrieval (ReAct) | 92.0% · $3.11 | **85.5%** · $3.15 | 46.7% · $4.21 | — |
| Controller first, LLM may search more | 90.0% · $5.35 | 84.4% · $4.21 | 61.5% · $8.10 | — |

On MemLens, L1 and L2 attach the top-3 original photos at answer time ("re-perception").

Writing a history into memory costs extra, once: about $43 per 1,000 histories on LongMemEval, $15 on MemLens 32K and $54 on MemLens 128K (embedding plus consolidation; from N9's build of 100 of the LongMemEval test histories and N6's build of the same 195 MemLens histories). These three datasets ask one question per history, so with the write counted, L1 costs about 1.4×, 2.1× and 1.4× as much as full context per question there; it pays back from about the second question asked of a history (the third on MemLens 32K). LoCoMo asks about 165 questions per conversation, so its writes add only about $0.21 per 1,000 questions and L1 stays about 4× cheaper than full context. Nominal cost also ignores provider prefix caching, which full context could use when one history is asked repeatedly.

In the LongMemEval column, every row except full context was measured before N12 fixed a session-ID leak (D-067): the evidence headers showed session IDs that gave away the evidence sessions. The [leak ablation](docs/en/report.md#35-the-longmemeval-session-id-leak-n12-note) ([Chinese record](docs/results/n12-leak/README.md)) finds no measurable inflation for L1, L2 or the controller (agentic retrieval and "controller first, LLM may search more" were not ablated). It used 100 of these questions, and the 95% intervals still allow up to about 8 points of inflation for a single method (L1 +2.0, 95% CI −3.7 to +8.1), or about 3–4 points averaged over L1, L2 and the controller; the 100 questions include only 5 abstention questions, so the `_abs` part of the leak is barely measured. The gap to full context is unchanged, and the numbers are left as published.

1. **Memory beats a long context window.** On LongMemEval, L1 is 17.5 points more accurate than the full-context baseline at about 1/20 of the answer-time cost (writing each history once costs about as much as 1.4 full-context answers there; see the note under the table). On MemLens, full context loses 5.6 points going from 32K to 128K tokens of history, while L2 with photos stays flat.
2. **The write path is the biggest single win.** The same naive retriever goes from 82.0% to 91.0% on LongMemEval (+53 / −17 questions, p < 0.001) once it can use the write path's output: facts with validity periods, fact-augmented index keys, and a profile.
3. **The controller pays off by deciding *whether* to recall, not *how deeply*.**
   - In a stream where 20% of queries need memory, it costs 41% less than always retrieving, with no loss in accuracy.
   - On memory questions alone, it sits between always-L1 and always-L2: +60 / −37 questions against L1, +15 / −64 against L2.
   - The frozen model reproduced its development-set estimate on the test sets: 81.2% → 81.6%.
4. **Letting the LLM search for itself is strong on text and weak on images.** ReAct is the best method on LoCoMo but scores 46.7% on MemLens, because its search tool returns only text. "Controller first, LLM may search more" holds up on all three datasets, at the highest cost.
5. **Most errors are not retrieval misses.** Of the errors read by hand, only 1 / 38 on LongMemEval and 5 / 40 on LoCoMo were evidence that never reached the context. The common causes were picking the wrong level, miscounting, and, on LoCoMo, a wrong gold answer (10 of 40 sampled errors).

**After the final evaluation (N11).** The table above is from that frozen pass. N11 and N12 reran stratified test subsets as labelled post-hoc analyses (Mem0, the rerun, the external baselines, the leak ablation, 3–4-question adapter pilots), and no frozen method changed. Controller changes were made on the development set and only replayed on the test results:

- **Controller v2.** In spaces with photos it looks up an image-space accuracy table instead of adding and clipping a photo gain, and questions about visual detail no longer go to L0. Post-hoc replay: 81.6% → 81.9% on memory questions (+10 / −5), all of it on MemLens. For a new reader model, recalibrating the accuracy table made things worse; lowering λ from 30 to 15 recovered most of the gap to always-L2 at 16–24% lower cost. Later, v3 attaches photos only when a question is about one (D-058). v2 attached photos to almost every question in such spaces, and that photo is what made the demo agent answer the previous question again. Development accuracy is unchanged, and photos go to a third of the questions instead of almost all of them.
- **The gate on real traffic.** With the first messages of real WildChat conversations as the queries that need no memory, the gate sends 10.5% of them to retrieval (17.3% on the synthetic stream) and the stream costs 46% less than always retrieving (41% before).
- **External baseline: Mem0.** Open-source Mem0 2.2 with the same extraction model, embeddings, reader and judge: 80.5% on 200 LoCoMo questions and 83.0% on 100 LongMemEval questions. CueMem's L1 ties on LoCoMo (81.0%) and is 9 points higher on LongMemEval (92.0%; nominal p = 0.049, 0.147 after a Holm correction over the three comparisons against Mem0 in that run), and Mem0's writes cost 2.3–2.9 times as much. Mem0 finds the right sessions; it loses on temporal reasoning, knowledge updates and details the assistant gave, which its add-only extraction does not keep. These LongMemEval numbers predate the session-ID leak fix (D-067, [leak ablation](docs/en/report.md#35-the-longmemeval-session-id-leak-n12-note); [Chinese record](docs/results/n12-leak/README.md)): rerun after the fix with the same configuration, L1 scores 90.0% and Mem0 79.0% on the same 100 questions, so Mem0's published 83.0% may be a few points high.
- **Judging and variance.** A second judge from another provider agrees on 94.7–97.9% of 12,835 judgements (κ 0.82–0.93) and keeps the method ranking (Kendall τ 0.87–1.0). A rerun with the cache off moves accuracy by at most 0.9 points, with 3–4% of questions flipping. One N9 comparison does not survive: L1 being 2.3 points below full context on LoCoMo (not significant under the second judge, 1.5 points in the rerun). The rerun keeps the same memory spaces, so it leaves out write-path randomness. A hand audit of 50 judgements, stratified by dataset, agrees with the judge on all 48 that were labelled; 2 LoCoMo items were left unlabelled, and both judges marked them wrong.
- **Multiple comparisons.** The p-values in these documents are per comparison. After a Holm correction within each result's families of tests (`scripts/multiplicity.py`, the `*-holm.json` files next to the results, D-069), the five numbered N9 findings above still hold, but several comparisons at p 0.01–0.05 are only nominal: L1 against Mem0 on LongMemEval (above), L2 against Mem0 under DeepSeek only (below), L2's 2.5-point lead over full context on LoCoMo, the open-domain and temporal types of "L2 beats L1 on LoCoMo", and the controller against the rule-based router on LoCoMo, among others. The report and the result records label each one and give the adjusted p.
- **Write path.** Reconciliation now tells "a different scope", "the world changed" and "it was a mistake" apart, and sees the message each new fact came from: "Mia is 31" alone cannot tell a correction from a birthday. On a small invented conversation run 5 times, all three cases are right every time (N9's prompt: 3 / 5, 5 / 5, 0 / 5).
- **Scale.** The local work on the read path grows linearly with local Qdrant (about 1.08 s at 46k retrieval units) and is 91 ms with a Qdrant server; one process serves 55 reads per second with the server (8.7 in local mode).

Details in the [technical report, §12](docs/en/report.md#12-n11-follow-up-work-after-the-test-sets).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/results/n9-final/pareto-dark.png">
  <img alt="Accuracy against nominal answer cost per 1,000 questions on the four test sets (the one-off write is excluded; full context is off the scale on LongMemEval and MemLens 128K and is drawn at the right edge with its real numbers); the curve is the learned controller at different settings of its cost knob" src="docs/results/n9-final/pareto-light.png" width="900">
</picture>

Hypotheses, stated before the experiments and resolved on the test sets:

| | Hypothesis | Outcome |
|---|---|---|
| H1 | Probe feedback + query cues beat query text alone, and beat uncertainty-triggered escalation | **Half supported.** True for "recall or not" (gate AUC 0.990 on the test stream); "L1 or L2" is decided by how the question is phrased |
| H2 | The controller matches always-deep-recall accuracy at much lower cost | **Half supported.** Holds in a mixed stream (−41% cost); not on memory questions alone (−2.8 points) |
| H3 | A controller is more robust than agentic retrieval across reader models | **Refuted** with three readers. ReAct is not less stable. The controller has to be tuned per reader: recalibrating its accuracy table did not help, setting its cost knob λ per reader did (N11) |
| H4 | Validity periods ("invalidate, don't delete") help knowledge-update questions | **Not supported.** All variants score 96.5% on those questions |
| H5 | Re-perceiving original photos helps visual-detail questions | **Supported.** +12.3 points (32K) and +6.7 points (128K), concentrated in information extraction |
| H6 | Context-aware image captions beat isolated captions | **Not supported** |

The full analysis is in the [technical report](docs/en/report.md): controller behaviour on the test sets, three reader models, ablations, error analysis, and limitations.

## How it works

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/architecture-dark.svg">
  <img alt="Architecture: the read path runs a probe, a controller that picks L0, L1 or L2 and whether to attach photos, and a reader; the write path appends turns online and consolidates them in the background into facts, entities, captions and a profile" src="docs/assets/architecture-light.svg" width="900">
</picture>

- **Write path.**
  - Turns are appended to an immutable log online, without an LLM call (about 100 ms).
  - Background consolidation turns each session into atomic facts with bi-temporal validity, entities, summaries, photo captions and a core profile. Supersession is confirmed by an LLM, never by rule alone.
  - All derived memory can be rebuilt from the log, so **forgetting** means deleting turns and replaying the rest. It follows citations into later replies, and finds uncited restatements in the log.
- **Read path.**
  - A **memory probe** (one L1 scoring pass, no LLM, 54 features) feeds a **two-stage controller**. A gate estimates whether the message needs memory at all, and a reasoning-type model estimates what kind of memory question it is.
  - The controller then picks the action with the best expected accuracy minus λ × cost: **L0** (profile only), **L1** (multi-granularity dense recall packed to a 4k-token budget, no LLM call) or **L2** (query planning, a sufficiency check, and taking notes before answering), each with or without the original photos.
  - λ is a business knob: how much one extra correct answer is worth.
- **Agent runtime.** A hand-written agent loop sits on top:
  - the controller recalls first, and the model may call `memory_search` when something is missing;
  - the context is assembled from partitioned budgets with a stable prefix, so provider prefix caching keeps working;
  - the trace UI shows every decision;
  - there is also an HTTP API and an MCP server.

## How CueMem compares

Checked against twelve open-source memory projects on 2026-10-14; the full comparison, with sources and every project's self-reported scores, is in [docs/en/landscape.md](docs/en/landscape.md).

| | CueMem | Mem0 | Letta | Graphiti / Zep | Supermemory |
|---|---|---|---|---|---|
| Who decides to recall | **A trainable controller**, before the LLM is called (L0 / L1 / L2, photos or not) | The app (`search`); the proxy recalls every turn | The LLM, through tools | The app (`search`) | The app, or every turn through its proxy / middleware |
| When a fact changes | Invalidated, not deleted; two timelines; `history()` | v3 is add-only | The LLM rewrites a block | Invalidated, not deleted; two timelines | Newest wins, history kept |
| Images | Originals kept and looked at again at answer time | Turned into text | Not seen | Text and JSON only | OCR |
| Forgetting | Deletes the source conversation, follows citations, finds uncited restatements; dry run | `delete` | Edit memory | `remove_episode` | `forget_matching`, dry run |
| Evaluation | Frozen methods, one frozen pass over the test sets, external baselines under the same models and judge | Own benchmark repo | Blog | Paper | Own |

**Same protocol, same models (N12).** We ran five open-source memory systems ourselves on 200 LoCoMo questions. Each uses its own write and read pipeline; the extraction model, embeddings, reader and judge are the same as CueMem's (under DeepSeek only, the judge is the reader itself). There are two protocols, DeepSeek only and Qwen only, with one run per configuration.

| LoCoMo test200r | DeepSeek only | Qwen only |
|---|---|---|
| CueMem L1 / L2 | 82.5% / **87.0%** | 81.5% / **85.0%** |
| Mem0 2.2.1 | 81.0% | 77.5% |
| LangMem 0.0.30 | 78.0% | 30.0% |
| Cognee 1.6.4 | 77.5% | 74.0% |
| Graphiti (graphiti-core 0.30.2) | 59.0% | 41.5% |
| A-MEM (k = 10 / k = 40) | 58.0% / 75.0% | 59.5% / 65.0% |
| Full context | 84.5% | 80.0% |

L2 is ahead of all five systems under both protocols (paired sign tests, nominal p ≤ 0.017), A-MEM with k = 40 included. After a Holm correction over the ten tests of L1 and L2 against the five systems in each protocol, every one of these leads stays significant except L2 against Mem0 under DeepSeek only (+17 / −5, adjusted p = 0.068). L1, whose recall makes no LLM call, is not significantly different from Mem0, LangMem and Cognee under DeepSeek, and is significantly ahead of Graphiti and A-MEM. A-MEM is shown at the paper's main setting (k = 10) and at the authors' setting for GPT-4o-class models (k = 40), which uses 2.4 times L2's input per question. Graphiti and LangMem drop sharply under Qwen: most of the drop is on the write side (with their own prompts qwen3.8-flash extracts 462 entities instead of 1,253, and LangMem loses the writes of 23 of 202 sessions); read-side effects of the shared harness (Graphiti's 0.6 threshold not retuned for Qwen embeddings, LangMem's long memories cut at the embedder's 6,000-character input) were not measured separately. CueMem's extraction prompts and thresholds were developed with qwen3.8-flash and Qwen embeddings (N3–N7), so the Qwen-only protocol is its home ground. Under DeepSeek only, no system's extraction prompts were tuned for deepseek-flash, but deepseek-flash was CueMem's reader in N2–N9 (the shared reader prompt and the controller's accuracy table were developed with it), so neither protocol is fully neutral; L2 leads under DeepSeek only too (nominally only against Mem0). The tests are conditional on one build per system: write-path randomness was not measured. Each adapter's departures from upstream (retrieval budgets, Graphiti without a reranker under DeepSeek, A-MEM's bug fixes) are listed in the [technical report, §13.4](docs/en/report.md#134-limitations). Details in the [technical report, §13](docs/en/report.md#13-n12-external-systems-under-the-same-protocol) (Chinese record: [N12 external baselines](docs/results/n12-baselines/README.md)) and the [comparison](docs/en/landscape.md#same-protocol-comparison-n12).

CueMem's gaps are on the product side: a Python SDK only (other languages go through the HTTP API; TypeScript types can be generated from its OpenAPI snapshot, but there is no hand-written TypeScript client), SQLite + Qdrant only (no multi-instance deployment; local-mode Qdrant keeps one open file per user space, so many users need a Qdrant server), three framework integrations so far ([what is missing](docs/en/landscape.md#7-what-is-still-missing)).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The default models are DeepSeek and Qwen (via QwenCloud) behind OpenAI-compatible APIs; `--preset openai` switches everything to OpenAI, and `--preset ollama` runs everything locally without keys (run `ollama pull llama3.1:8b` and `ollama pull nomic-embed-text` first).

```bash
uv sync                                   # install
uv run cuemem configure                   # enter API keys into .env (run in a real terminal)
uv run cuemem doctor --live               # check config and model connectivity

# Write, consolidate, inspect, ask, forget
uv run cuemem ingest examples/toy_conversation.en.json --consolidate
uv run cuemem inspect facts --space demo-en                  # fact timeline: what was superseded, when, by what
uv run cuemem inspect facts --space demo-en --valid-at 2025-06
uv run cuemem ask "Where do I work now?" --space demo-en --at 2025-10-10 --show-context
uv run cuemem ask "How many times have I changed jobs?" --space demo-en --mode auto --show-context
uv run cuemem forget --space demo-en --session s4             # derived facts, vectors and caches go too
uv run cuemem usage --today                                   # calls and cost
uv run cuemem export --space demo-en -o demo-en.json          # raw log + photos; import elsewhere with cuemem import
uv run cuemem --preset ollama doctor                          # the same commands work with any preset
```

Without a checkout, install from GitHub: `pip install "cuemem @ git+https://github.com/FlickYan/cuemem"` installs the core: the SDK, `MemoryClient`, the CLI, the HTTP API, the memory proxy and MCP. Optional extras add the rest and can be combined, for example `pip install "cuemem[ui,local] @ git+https://github.com/FlickYan/cuemem"`. CueMem is not on PyPI yet; a package named `cuemem` on PyPI is not this project, so always install from GitHub as shown.

| Extra | Adds |
|---|---|
| `local` | Local embeddings with no key (fastembed; the `deepseek` preset needs it) |
| `ui` | The browser UI of `cuemem serve` (Gradio). Without it, `cuemem serve` serves only `/api`, `/api/v1`, `/docs` and the `/v1` proxy, and prints how to install the UI |
| `eval` | Evaluation charts and the controller and gate training scripts (matplotlib, scikit-learn). Without it, `cuemem eval run` and `eval report` say which extra to install and exit 1 |
| `integrations` | The OpenAI Agents SDK, LangGraph / LangChain and Anthropic integrations (or one of `openai-agents`, `langgraph`, `anthropic`) |

In a checkout, `uv sync` installs `ui` and `eval` through the dev group; add `--extra local` or `--extra integrations` as needed. A pip install that includes `cuemem[ui]` still resolves Pillow below 13, because gradio declares `pillow<13` (`cuemem[local]` alone gets fastembed 0.9.1 and Pillow 13), and Pillow 12 has known security advisories that are fixed only in 13; a server that accepts images from other people should use the Docker image or `uv sync --locked` in a checkout, which install Pillow 13. With pip, the commands are `cuemem …` instead of `uv run cuemem …`, data goes to `~/.cuemem` (`CUEMEM_HOME` to change it), and `cuemem serve` starts the plain assistant, because the shop demo's product catalogue exists only in the repository (`cuemem demo` reads the `catalog.json` next to its script). `cuemem --help` and `import cuemem.sdk` take about 0.3–0.4 s on the development machine. CI installs the wheel outside the repository and runs the CLI, the SDK and the server on every push.

Command-line messages are in Chinese. The agent, the web UI, the demo, `/api/v1`, the proxy and the MCP tools are in English, as are the SDK's return values and its own errors; errors raised deeper in the core (a missing key, a locked vector store, a configuration mistake) are in Chinese. The [API reference](docs/en/api.md) is in English. A configuration mistake (an unknown preset, an invalid `CUEMEM_*` value) makes any command print one line, `错误：配置有误：…`, and exit 1.

To try writing and retrieval without any API key, run `uv run python examples/keyless.py`. It uses hash embeddings in a temporary data directory:

```python
import tempfile
from cuemem import Memory

with (
    tempfile.TemporaryDirectory() as home,
    Memory(home=home, config={"embedding": {"backend": "hash"}}, background=False) as m,
):
    m.add(
        [{"role": "user", "content": "I just started a new job at Acme in Berlin."}],
        user_id="alice",
        infer=False,
    )
    print(m.search("Where do I work?", user_id="alice")["results"][0]["memory"])
```

Over HTTP, start `CUEMEM_HOME=$(mktemp -d) CUEMEM_EMBEDDING__BACKEND=hash uv run cuemem serve`, then:

```bash
curl -X POST http://127.0.0.1:7860/api/v1/memories -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "I just started a new job at Acme in Berlin."}], "user_id": "alice", "infer": false}'
curl -X POST http://127.0.0.1:7860/api/v1/memories/search -H 'Content-Type: application/json' \
  -d '{"query": "Where do I work?", "user_id": "alice"}'
```

Extraction, answering (`cuemem ask`, `Memory.answer`) and `--consolidate` need an LLM key. A data directory remembers its embedding model, even after a command that failed, so keep the trial out of `./data`, `~/.cuemem` and the Docker volume.

The agent demo uses Qwen models only:

```bash
uv run cuemem demo demo/shop/script.json --fresh --until 2   # build the demo memory (August and September chats)
uv run cuemem serve                       # UI at http://127.0.0.1:7860/ , API docs at /docs
uv run cuemem chat --space me             # chat in the terminal; "/image path text" attaches a photo
uv run cuemem demo demo/shop/script.json --fresh   # replay the whole script with 14 checks
claude mcp add cuemem -- uv --directory "$PWD" run cuemem mcp --space me   # memory as MCP tools
uv run cuemem proxy                       # OpenAI-compatible memory proxy at http://127.0.0.1:8787/v1
uv run python examples/quickstart.py --preset openai   # the SDK in one script
```

Evaluation and reproduction:

```bash
uv run cuemem data download               # LongMemEval_S, LoCoMo, MemLens 32K / 128K
uv run cuemem eval run --split pilot10    # 10-question smoke run of the baselines (~$0.4)
uv run python scripts/reproduce.py --dry-run      # every step behind the N2–N11 numbers
uv run python scripts/reproduce.py --only N9      # the final evaluation (near-free with the original cache)
uv run python scripts/reproduce.py --only N9 --cache-only   # LLM requests from the local cache only
```

One command regenerates the numbers only on the machine that holds the original response cache: `data/` (the cache, the evaluation spaces) and `runs/` are not in the repository, so on a fresh clone the script makes paid API calls, and with `--cache-only` it stops at the first miss. Request bytes and cache keys are pinned by offline tests (D-068, D-070), but the full N2–N11 replay has not been re-run since the audits, whose machine holds only the N12-era cache. The script never overwrites the frozen controller files (the training steps only check them against the committed files) and stops instead of copying an errored or budget-stopped report into `docs/results`. An analysis step whose input runs have no results is skipped, and the analysis scripts themselves (`n11_variance.py`, `n11_mem0.py`, `n11_second_judge.py report`, `n12_leak.py`) exit 1 without writing when a run is missing; the N11 latency benchmark and the N8 demo summary, which depend on the machine, are written under `runs/` instead of over the published files. With `--cache-only` a cache miss is an error rather than an API call, and a request the fallback model answered at build time replays from the fallback's cached reply; check the outcome with `git diff docs/results src/cuemem/read`. The N12 numbers are not in the script: the docstring of `scripts/n12_baselines.py` lists their runs.

Development: `uv run pytest` (offline, no keys), `uv run pytest -m live` (needs keys), `uv run ruff check . && uv run mypy`. The offline suite runs behind a network guard (`tests/netguard.py`): it strips proxy variables and provider keys, sets `GRADIO_ANALYTICS_ENABLED=False`, and fails any test that tries to reach a non-loopback host or an environment proxy (`-m live` is exempt). The 23 tests it skips locally run in CI: the external baselines' adapter tests in the weekly "外部基线" workflow (pinned versions, offline, with the commands in each test file's docstring; Mem0's also needs the spaCy model wheel and fastembed's `Qdrant/bm25` model downloaded first, `uv run --with ... python -c "from fastembed import SparseTextEmbedding; SparseTextEmbedding('Qdrant/bm25')"`), and the Qdrant server tests in the `qdrant-server` job (locally: `docker run -d -p 127.0.0.1:6333:6333 qdrant/qdrant:v1.19.1`, then `CUEMEM_TEST_QDRANT_URL=http://localhost:6333 uv run pytest tests/test_qdrant_server.py`; against a server with an API key, also set `CUEMEM_TEST_QDRANT_API_KEY`, as CI does). Contributors should use uv 0.12.24, the version CI and the Docker image pin.

### Deployment

By default CueMem uses embedded SQLite and Qdrant in local mode, which allows one process per data directory: apps that should share a memory talk to one `cuemem serve` through `MemoryClient` or `/api/v1`. Installed with pip, it keeps its data in `~/.cuemem` (`CUEMEM_HOME` to change it). Local mode keeps one open file per user space (two with photos) and loads all vectors into RAM, so a shared server with many users should use a Qdrant server (`cuemem migrate-vectors --to URL`, then `CUEMEM_VECTOR_STORE__URL`, below). Every `cuemem` command raises its open-file soft limit to the hard limit, and `cuemem doctor` shows the collection count against it.

- **Access tokens.** `cuemem serve` and `cuemem proxy` refuse to listen beyond loopback without a token (add `--insecure-no-auth` to run without authentication on a trusted network; without a token, only IP addresses, localhost, the bind hostname and the names in `CUEMEM_ALLOWED_HOSTS` are accepted as Host). Off loopback, tokens must be at least 16 characters (generate them with `openssl rand -hex 32`); shorter ones refuse to start (exit code 2) unless `--insecure-weak-token` is given (in compose, add it to `command:`); `sha256:` entries cannot be measured and are not checked. `CUEMEM_API_TOKEN` is the admin token. Scoped keys, in a YAML file named by `CUEMEM_API_KEYS_FILE` (or inline in `CUEMEM_API_KEYS`), limit each key to listed users and agents, for example `{name: alice-app, sha256: <printf %s "$TOKEN" | sha256sum>, scopes: [{user: alice}]}`. A key gets 403 for a user, agent or space outside its scope, on `/api`, `/api/v1`, `/mcp`, the UI and the proxy alike, other tenants' memory ids look like missing ones (404), and `POST /api/v1/reset` deletes only the key's own spaces. Format and rules are in the [API reference](docs/en/api.md#authentication-and-scoped-keys).
- **What a key can spend.** A scoped key limits whose memory a caller reaches, not what it spends. Any valid key makes the server call the operator's providers: `/v1/chat/completions` and `/v1/responses` (with or without a user; model, `n`, `max_tokens` and switches such as `enable_thinking` are forwarded as sent), `/v1/embeddings` (up to 2,048 inputs), chat on `/api`, `POST /api/v1/answer`, `/mcp`, and every write (extraction later sends about 2 characters upstream per byte written). Give keys only to parties you would trust with that budget, set spend limits at the provider, rate-limit by the `Authorization` header in a reverse proxy (for example nginx `limit_req_zone $http_authorization ...`), and lower `CUEMEM_MAX_BODY_BYTES` to bound what one write can cost. A key can also carry optional limits, for example `{name: alice-app, sha256: ..., scopes: [{user: alice}], rate_per_minute: 60, models: [qwen3.8-flash], max_output_tokens: 2000}`: `rate_per_minute` counts its write requests (POST, PUT, DELETE; reads are not counted) on `/api`, `/api/v1`, `/v1` and `/mcp` and answers 429 with `Retry-After` beyond it; `models` limits the proxy to the listed aliases or model ids (403 otherwise) and applies the catalogue's settings, such as `enable_thinking`, over the caller's; `max_output_tokens` caps or sets the proxy's `max_tokens` and refuses an `n` other than 1. Without these fields nothing changes. They do not cover UI sessions.
- **MCP over HTTP.** `cuemem serve` also serves the MCP tools at `/mcp` (Streamable HTTP), behind the same tokens; `X-CueMem-User` picks the user, and `--no-mcp` turns it off. Register it with `claude mcp add --transport http cuemem http://127.0.0.1:7860/mcp --header "Authorization: Bearer $KEY" --header "X-CueMem-User: alice"`. Non-ASCII IDs go in the header percent-encoded as UTF-8: `--header "X-CueMem-User: utf-8''%E5%BC%A0%E4%B8%89"` (raw UTF-8 bytes from curl work too). The stdio `cuemem mcp` cannot share a local data directory with a running serve.
- **Request limits.** Bodies are refused with 413 before they are read: 64 MiB for `/api`, `/v1` and UI uploads (`CUEMEM_MAX_BODY_BYTES`), 1 GiB for imports (`CUEMEM_MAX_IMPORT_BYTES`; raise it for very large exports), 8 MiB elsewhere. A scoped key imports at the body limit, and its request bodies in flight together are capped at the same limit (429 with `Retry-After: 1`), since parsed JSON takes about 24 times its size in memory; size a shared server's memory for the keys it serves, or lower the limit. Images must be JPEG, PNG, WEBP, GIF, BMP or TIFF, and images above Pillow's pixel limit (about 89 million pixels) are refused; both checks run before any decoder or model call.
- **Long-running servers.** Chat sessions, `/api/v1` runs and proxy sessions idle for 30 minutes end and go to extraction (`--idle-minutes`). The next add with the same `run_id` reopens an idle-closed run; when it ends, the run is extracted again in full and the user's whole space is replayed from the log (D-045), and while that replay runs the user's facts are missing from search and `get_all` for about the extraction latency. A run left open when `cuemem serve` stops or crashes survives the restart: the next add with the same `run_id` continues it, nothing else adopts it during the first idle period, and serve and proxy hand any that nobody continues to extraction one idle period after start-up. A run that the previous process had already closed for idleness (or one handed over that is no longer remembered, beyond 10,000 runs) is a plain session after the restart: each later add re-extracts it and replays the user's whole space. Clients that pause for long should call `end_run` at natural breaks and open a new run. Traces (`runs/traces/*.jsonl`, which hold the users' questions) are kept until forgotten unless `CUEMEM_TRACES_RETENTION_DAYS` is set. Up to 4 users' extractions run at once (`CUEMEM_CONSOLIDATION__BACKGROUND_WORKERS`).
- **The UI contacts no third party by default:** it uses the IBM Plex Sans font bundled with Gradio, and Gradio's usage analytics and version check (api.gradio.app, huggingface.co) are off unless you set `GRADIO_ANALYTICS_ENABLED=True`.
- **Provider failures** have their own status codes on the HTTP API instead of a bare 500: 429 (with `Retry-After`) for rate limits, 402 for an empty balance, 503 (with `Retry-After`) when the provider is unavailable, 504 for timeouts, 422 when its content filter refuses, and 502 for other provider errors.
- **Browsers.** Browser requests from other origins get 403 on `/api`, `/v1`, `/mcp`, the UI's `/gradio_api` and its sign-in `/login`. On a loopback address only 127.0.0.1, localhost and [::1] are accepted as the Host. With `--insecure-no-auth` on another address, only IP addresses, localhost and the bind hostname are. Behind a reverse proxy, or under another hostname, list those hostnames in `CUEMEM_ALLOWED_HOSTS` (comma-separated, can go in `.env`).
- **Files on disk.** CueMem creates its data files readable only by you (files 0600, new directories 0700); it never changes directories that already exist. In an existing checkout run `cuemem doctor`, which prints the `chmod 700 <dir>` / `chmod 600 <file>` commands for anything others can read (for example `chmod 700 data runs/traces`); it also checks `.env` and `CUEMEM_API_KEYS_FILE`. The UI keeps uploads and gallery copies in a private temporary directory that is removed on exit; set `GRADIO_TEMP_DIR` to choose your own.

For Docker, `install -m 600 .env.example .env` (in a checkout, the `.env` that `uv run cuemem configure` wrote is the same file compose reads). Then set `CUEMEM_API_TOKEN` (for example from `openssl rand -hex 32`) and the keys of your configuration:

- `DEEPSEEK_API_KEY` and `DASHSCOPE_API_KEY` for the default configuration;
- or `CUEMEM_PRESET=openai` with `OPENAI_API_KEY`;
- or `CUEMEM_PRESET=deepseek` with `DEEPSEEK_API_KEY` (the image includes local embeddings);
- or `CUEMEM_PRESET=ollama` with `CUEMEM_PROVIDERS__OLLAMA__BASE_URL=http://host.docker.internal:11434/v1` (Ollama on the host must listen on an address the container can reach, not only 127.0.0.1).

Keep comments in `.env` on their own lines: `docker run --env-file` takes everything after `=` as the value. `docker compose up -d` then runs the UI, the HTTP API and the memory proxy on 127.0.0.1:7860 with a Qdrant server, and stops with an error if the token is missing. It returns once the containers start: if http://127.0.0.1:7860/ does not answer, `docker compose ps` shows the cuemem container Restarting and `docker compose logs cuemem` shows a one-line error (usually a missing key, or a token shorter than 16 characters); fix `.env` and run `docker compose up -d` again. Setting `CUEMEM_VECTOR_STORE__API_KEY` in `.env` makes compose's Qdrant require that key and cuemem send it; left unset, the internal Qdrant (which publishes no port) runs without one, as before. A single container is `docker run -p 127.0.0.1:7860:7860 -v cuemem-data:/data --env-file .env cuemem` after `docker build -t cuemem .`. The image includes the UI and local embeddings (the `ui` and `local` extras); `--build-arg EXTRAS=` builds an API-only one, and a missing extra is added by rebuilding (for example `docker build --build-arg EXTRAS="--extra ui --extra local --extra eval" -t cuemem .`), not with pip inside the container. For scoped keys in a container, put the keys file in the data volume (`/data`) and point `CUEMEM_API_KEYS_FILE` at it, or put the keys inline in `CUEMEM_API_KEYS`; compose still requires `CUEMEM_API_TOKEN`. Without Docker, run a Qdrant server with an API key and protect the API with a token:

```bash
export CUEMEM_VECTOR_STORE__API_KEY=$(openssl rand -hex 32)   # also put it in .env
docker run -d --name cuemem-qdrant -p 127.0.0.1:6333:6333 -v cuemem-qdrant:/qdrant/storage \
  -e QDRANT__SERVICE__API_KEY="$CUEMEM_VECTOR_STORE__API_KEY" qdrant/qdrant:v1.19.1
uv run cuemem migrate-vectors --to http://localhost:6333   # copy existing local vectors first
export CUEMEM_VECTOR_STORE__URL=http://localhost:6333     # also put it in .env
echo "CUEMEM_API_TOKEN=$(openssl rand -hex 32)" >> .env   # the token never appears in shell history
uv run cuemem serve --host 0.0.0.0   # Bearer token for /api, /api/v1, /v1 and /mcp, sign-in for the UI
uv run python scripts/n11_latency.py --out runs/n11-latency   # offline latency benchmark, no API calls
```

Without `127.0.0.1`, Docker publishes the Qdrant port on every interface and bypasses ufw; without `-v`, `docker rm` (for example to upgrade the image) deletes every vector, and CueMem has no reindex command. CueMem sends `CUEMEM_VECTOR_STORE__API_KEY` (from the environment or `.env` only; there is no command-line flag) as Qdrant's `api-key` header, also from `cuemem migrate-vectors`, and `cuemem doctor` shows whether it is set. With an `http://` URL, qdrant-client warns once per client that the key is used over an insecure connection: harmless on loopback and on compose's internal network; use `https://` for a remote server or Qdrant Cloud. Serving on `0.0.0.0` as above suits a trusted network; to reach the server from the internet, keep it on 127.0.0.1 behind a TLS reverse proxy, as below.

#### Public deployment

`cuemem serve` speaks plain HTTP. The Bearer token and the UI password are the same secret, and without TLS they cross the network in cleartext together with every memory. To reach it from outside a trusted network:

- Without Docker, keep `cuemem serve` on 127.0.0.1; with compose, keep the `127.0.0.1:7860:7860` mapping.
- Put a TLS reverse proxy in front. Caddy forwards the Host and `X-Forwarded-Proto` by default: `mem.example.com { reverse_proxy 127.0.0.1:7860 }`. With nginx, inside the `location` that does `proxy_pass http://127.0.0.1:7860;`, set `proxy_set_header Host $host; proxy_set_header X-Forwarded-Proto $scheme; proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; proxy_buffering off;` (the last items keep the UI's streaming working).
- Set `CUEMEM_ALLOWED_HOSTS=mem.example.com` in `.env`, so the Host and Origin checks accept that name.
- Keep `.env` private (`chmod 600 .env`; `cuemem configure` and `install -m 600` create it that way) and use tokens from `openssl rand -hex 32`.
- CueMem does not rate-limit sign-ins or failed tokens; do that in the proxy, for example nginx `limit_req` on `/login` by client address and on the API by the `Authorization` header.

## Repository layout

| Path | What is there |
|---|---|
| `src/cuemem/sdk.py`, `client.py` | The Python SDK (`Memory`, `AsyncMemory`), shaped like Mem0's, and its remote twin (`MemoryClient`, `AsyncMemoryClient`) |
| `src/cuemem/configs/` | Default config and presets (`openai`, `ollama`, `deepseek`, `qwen`), shipped in the package |
| `src/cuemem/write/` | Online writer; consolidation: extraction, entity linking, fact reconciliation, captions, profile |
| `src/cuemem/read/` | Naive, L1 and L2 retrieval; probe; controller; reader; agentic reader |
| `src/cuemem/store/` | SQLite (log and derived memory), Qdrant local (vectors), content-addressed media |
| `src/cuemem/llm/`, `embed/` | Model clients with response cache, cost ledger, off-peak scheduling; embeddings |
| `src/cuemem/agent/`, `api/` | Agent loop and context assembly; HTTP API (`/api`, the memory API `/api/v1`), access tokens and scopes, Gradio UI, MCP server, OpenAI-compatible memory proxy, demo replay |
| `src/cuemem/portable.py` | Export and import of a memory space |
| `src/cuemem/evals/` | Datasets, splits, runner, judge, metrics, routing analysis |
| `scripts/` | Experiment scripts for every node, `reproduce.py`, figure generators |
| `docs/` | Report, design, decisions, roadmap, per-node results |

## Documentation

| Document | Language |
|---|---|
| Technical report: system, final results, controller, readers, ablations, errors, lessons | [English](docs/en/report.md) · [中文](docs/report.md) |
| Design: motivation, architecture, controller, multimodal memory, evaluation plan | [English](docs/en/design.md) · [中文](docs/design.md) |
| Demo guide (3 minutes) | [English](demo/README.en.md) · [中文](demo/README.md) |
| Comparison with open-source peers: capabilities, published scores, what N12 borrowed, what is missing | [English](docs/en/landscape.md) · [中文](docs/landscape.md) |
| API reference: SDK, `MemoryClient`, HTTP API (`/api`, `/api/v1`), memory proxy, MCP, environment variables, extras | [English](docs/en/api.md) · [中文](docs/api.md) |
| Framework integrations: OpenAI Agents SDK, LangGraph / LangChain, Claude's memory tool | [English](docs/en/integrations.md) · [中文](docs/integrations.md) |
| Decision records (71 ADRs) | [English](docs/en/decisions.md) · [中文](docs/decisions.md) |
| N12 results: same-protocol external baselines on LoCoMo; the LongMemEval session-ID leak, its fix and ablation | 中文: [external baselines](docs/results/n12-baselines/README.md), [session-ID leak](docs/results/n12-leak/README.md) |
| Roadmap and progress log, related work, per-node results | 中文: [roadmap](docs/roadmap.md), [related work](docs/related-work.md), [results](docs/results/) |

The project was built node by node, from N0 (design) to N10 (packaging), followed by N11 (follow-up work after the test sets) and N12 (comparison with open-source peers, and the SDK, proxy, presets, export and Docker that came out of it). After N12, a full audit fixed 24 verified problems in forgetting, concurrency, security, SDK writes, evaluation hygiene, packaging and read-path performance, without changing any evaluation request (D-068 in the decision records). A second round then did 14 of the 17 items that audit had deferred (D-069): all-or-nothing online writes, clear provider errors, cheaper reconciliation, consolidation and forgetting, faster start-up and optional `ui` / `eval` extras, pinned datasets and Holm-corrected significance, the rest of Mem0's API with `MemoryClient` and `/api/v1`, scoped access tokens, and wider proxy coverage, again with evaluation requests unchanged. A third round audited everything again from scratch and fixed 22 verified problems (D-070): forgetting and its locks, writes that could still raise after committing, residue after `delete_all`, invisible background failures (now `status()`), idle sessions, per-request work and open files at scale, the UI's file paths and input sizes, Mem0's call shapes, MCP over HTTP, typed clients, the reproduction script and several overstated claims, with evaluation requests unchanged once more. After the repository went public, a fourth round audited it through six lenses, verified 22 problems and fixed the 21 that needed a change (D-071): dependencies with security advisories and the CI supply chain, a security policy, public deployment (DNS rebinding without a token, weak tokens, per-key spending limits, a Qdrant API key, private file modes, no third-party calls from the UI), deletions reaching open sessions and leaving no residue, runs that survive a restart, installing from GitHub and a clearer first run, licence notices, and the claims against external baselines, with evaluation requests, image requests and published numbers unchanged. Every node's experiments and numbers are in the Chinese records above. The English documents cover the system, the final results, the N11 follow-up and every design decision.

## Security

Report vulnerabilities privately through GitHub; see [SECURITY.md](SECURITY.md).

## Licence

CueMem is released under the [MIT License](LICENSE).

The LongMemEval and MemLens judge prompts and A-MEM's keyword prompt included in the repository are MIT-licensed (MemLens's possibly also CC BY 4.0; see the notices); their notices are reproduced in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The evaluation datasets are downloaded by `cuemem data download`, are not redistributed, and keep their own licences:

- LongMemEval_S (cleaned): MIT;
- LoCoMo: CC BY-NC 4.0;
- MemLens: CC BY 4.0, with images under their source sites' licences;
- databricks-dolly-15k: CC BY-SA 3.0;
- WildChat-1M (one shard, downloaded only on request, N11): ODC-BY.

The repository stores only split IDs, Dolly line numbers, WildChat conversation hashes and generated same-topic questions derived from the benchmarks; a few of those generated questions repeat a benchmark question verbatim, and the ones derived from LoCoMo are CC BY-NC 4.0, for non-commercial use only (`evals/mixed/` is not under the MIT License); see the notices for how they are covered. The demo store, its products and its photos are fictional; the images were generated with qwen-image-3.0.
