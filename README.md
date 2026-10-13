# CueMem

**Cue-driven, self-regulating, multimodal long-term memory for LLM agents**

English · [简体中文](README.zh-CN.md) · [Technical report](docs/en/report.md) · [Design](docs/en/design.md) · [Demo guide](demo/README.en.md)

CueMem gives an LLM agent a memory that lasts across months of conversations:

- it **writes** what users say into facts that carry validity periods;
- it **recalls** them at the right depth;
- it **looks at the original photo again** when a question needs visual detail;
- it **forgets on request**, including replies that only restated the forgotten information.

The question the project studies is **who should decide whether to recall, and how deeply**. One option is the LLM itself, calling a search tool. The other is a cheap, trainable controller that runs before the LLM is called and never calls an LLM itself. CueMem builds the controller, evaluates both approaches on three public benchmarks under a frozen protocol, and reports where each one wins.

![A shopping assistant that remembers: cross-session recall, a changed shoe size, finding the pair from a photo shared in August, a budget that changed, and forgetting a fact](docs/assets/demo.gif)

*The demo: Cue, a shopping assistant, on its third chat with a user. It recalls her across sessions, knows her shoe size changed, finds look-alikes of a photo she shared two months earlier, explains how her budget changed (with citations), and forgets a fact on request. The right-hand panel shows what the controller chose and why, what went into the context, and which memories the answer cites.*

## Results at a glance

Everything below comes from test sets used once, with every method, prompt and the controller model frozen beforehand. The reader is deepseek-flash, the judge is qwen3.8-max, and costs are nominal (peak list price, no cache).

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

1. **Memory beats a long context window.** On LongMemEval, L1 is 17.5 points more accurate than the full-context baseline at about 1/20 of the cost. On MemLens, full context loses 5.6 points going from 32K to 128K tokens of history, while L2 with photos stays flat.
2. **The write path is the biggest single win.** The same naive retriever goes from 82.0% to 91.0% on LongMemEval (+53 / −17 questions, p < 0.001) once it can use the write path's output: facts with validity periods, fact-augmented index keys, and a profile.
3. **The controller pays off by deciding *whether* to recall, not *how deeply*.**
   - In a stream where 20% of queries need memory, it costs 41% less than always retrieving, with no loss in accuracy.
   - On memory questions alone, it sits between always-L1 and always-L2: +60 / −37 questions against L1, +15 / −64 against L2.
   - The frozen model reproduced its development-set estimate on the test sets: 81.2% → 81.6%.
4. **Letting the LLM search for itself is strong on text and weak on images.** ReAct is the best method on LoCoMo but scores 46.7% on MemLens, because its search tool returns only text. "Controller first, LLM may search more" holds up on all three datasets, at the highest cost.
5. **Most errors are not retrieval misses.** Of the errors read by hand, only 1 / 38 on LongMemEval and 5 / 40 on LoCoMo were evidence that never reached the context. The common causes were picking the wrong level, miscounting, and, on LoCoMo, a wrong gold answer (10 of 40 sampled errors).

**After the final evaluation (N11).** The test sets were used once; the follow-up work changes nothing in the table above. Controller changes were made on the development set and only replayed on the test results:

- **Controller v2.** In spaces with photos it looks up an image-space accuracy table instead of adding and clipping a photo gain, and questions about visual detail no longer go to L0. Post-hoc replay: 81.6% → 81.9% on memory questions (+10 / −5), all of it on MemLens. For a new reader model, recalibrating the accuracy table made things worse; lowering λ from 30 to 15 recovered most of the gap to always-L2 at 16–24% lower cost.
- **The gate on real traffic.** With the first messages of real WildChat conversations as the queries that need no memory, the gate sends 10.5% of them to retrieval (17.3% on the synthetic stream) and the stream costs 46% less than always retrieving (41% before).
- **External baseline: Mem0.** Open-source Mem0 2.2 with the same extraction model, embeddings, reader and judge: 80.5% on 200 LoCoMo questions and 83.0% on 100 LongMemEval questions. CueMem's L1 ties on LoCoMo (81.0%) and is 9 points higher on LongMemEval (92.0%, p = 0.049), and Mem0's writes cost 2.3–2.9 times as much. Mem0 finds the right sessions; it loses on temporal reasoning, knowledge updates and details the assistant gave, which its add-only extraction does not keep.
- **Judging and variance.** A second judge from another provider agrees on 94.7–97.9% of 12,835 judgements (κ 0.82–0.93) and keeps the method ranking (Kendall τ 0.87–1.0). A rerun with the cache off moves accuracy by at most 0.9 points, with 3–4% of questions flipping. One N9 comparison does not survive: L1 being 2.3 points below full context on LoCoMo (not significant under the second judge, 1.5 points in the rerun). A hand audit of 50 judgements is prepared and awaits labelling.
- **Write path.** Reconciliation now tells "a different scope", "the world changed" and "it was a mistake" apart, and sees the message each new fact came from: "Mia is 31" alone cannot tell a correction from a birthday. On a small invented conversation run 5 times, all three cases are right every time (N9's prompt: 3 / 5, 5 / 5, 0 / 5).
- **Scale.** The local work on the read path grows linearly with local Qdrant (about 1.08 s at 46k retrieval units) and is 91 ms with a Qdrant server; one process serves 55 reads per second with the server (8.7 in local mode).

Details in the [technical report, §12](docs/en/report.md#12-n11-follow-up-work-after-the-test-sets).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/results/n9-final/pareto-dark.png">
  <img alt="Accuracy against nominal cost per 1,000 questions on the four test sets; the curve is the learned controller at different settings of its cost knob" src="docs/results/n9-final/pareto-light.png" width="900">
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

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The models are DeepSeek and Qwen (via QwenCloud) behind OpenAI-compatible APIs.

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
```

Command-line messages are in Chinese. The agent, the web UI, the demo and the MCP tools are in English. Without API keys you can still try writing and retrieval offline with hash embeddings (`CUEMEM_EMBEDDING__BACKEND=hash`); answering still needs an LLM.

The agent demo uses Qwen models only:

```bash
uv run cuemem demo demo/shop/script.json --fresh --until 2   # build the demo memory (August and September chats)
uv run cuemem serve                       # UI at http://127.0.0.1:7860/ , API docs at /docs
uv run cuemem chat --space me             # chat in the terminal; "/image path text" attaches a photo
uv run cuemem demo demo/shop/script.json --fresh   # replay the whole script with 14 checks
claude mcp add cuemem -- uv --directory "$PWD" run cuemem mcp --space me   # memory as MCP tools
```

Evaluation and reproduction:

```bash
uv run cuemem data download               # LongMemEval_S, LoCoMo, MemLens 32K / 128K
uv run cuemem eval run --split pilot10    # 10-question smoke run of the baselines (~$0.4)
uv run python scripts/reproduce.py --dry-run      # every step behind every number in the docs
uv run python scripts/reproduce.py --only N9      # the final evaluation (near-free with the local cache)
```

Development: `uv run pytest` (offline, no keys), `uv run pytest -m live` (needs keys), `uv run ruff check . && uv run mypy`.

Deployment (N11). By default CueMem uses embedded SQLite and Qdrant in local mode, which allows one process per data directory. For more, run a Qdrant server and protect the API with a token:

```bash
docker run -d -p 6333:6333 qdrant/qdrant:v1.19.1
uv run cuemem migrate-vectors --to http://localhost:6333   # copy existing local vectors first
export CUEMEM_VECTOR_STORE__URL=http://localhost:6333     # or put it in .env
CUEMEM_API_TOKEN=your-token uv run cuemem serve --host 0.0.0.0   # Bearer token for /api, sign-in for the UI
uv run python scripts/n11_latency.py                       # offline latency benchmark, no API calls
```

## Repository layout

| Path | What is there |
|---|---|
| `src/cuemem/write/` | Online writer; consolidation: extraction, entity linking, fact reconciliation, captions, profile |
| `src/cuemem/read/` | Naive, L1 and L2 retrieval; probe; controller; reader; agentic reader |
| `src/cuemem/store/` | SQLite (log and derived memory), Qdrant local (vectors), content-addressed media |
| `src/cuemem/llm/`, `embed/` | Model clients with response cache, cost ledger, off-peak scheduling; embeddings |
| `src/cuemem/agent/`, `api/` | Agent loop and context assembly; HTTP API, Gradio UI, MCP server, demo replay |
| `src/cuemem/evals/` | Datasets, splits, runner, judge, metrics, routing analysis |
| `scripts/` | Experiment scripts for every node, `reproduce.py`, figure generators |
| `docs/` | Report, design, decisions, roadmap, per-node results |

## Documentation

| Document | Language |
|---|---|
| Technical report: system, final results, controller, readers, ablations, errors, lessons | [English](docs/en/report.md) · [中文](docs/report.md) |
| Design: motivation, architecture, controller, multimodal memory, evaluation plan | [English](docs/en/design.md) · [中文](docs/design.md) |
| Demo guide (3 minutes) | [English](demo/README.en.md) · [中文](demo/README.md) |
| Decision records (57 ADRs) | [English](docs/en/decisions.md) · [中文](docs/decisions.md) |
| Roadmap and progress log, related work, per-node results | 中文: [roadmap](docs/roadmap.md), [related work](docs/related-work.md), [results](docs/results/) |

The project was built node by node, from N0 (design) to N10 (packaging), followed by N11 (follow-up work after the test sets). Every node's experiments and numbers are in the Chinese records above. The English documents cover the system, the final results, the N11 follow-up and every design decision.

## Licence

CueMem is released under the [MIT License](LICENSE).

The LongMemEval and MemLens judge prompts included in the repository are also MIT-licensed; their notices are reproduced in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The evaluation datasets are downloaded by `cuemem data download`, are not redistributed, and keep their own licences:

- LongMemEval_S (cleaned): MIT;
- LoCoMo: CC BY-NC 4.0;
- MemLens: CC BY 4.0, with images under their source sites' licences;
- databricks-dolly-15k: CC BY-SA 3.0;
- WildChat-1M (one shard, downloaded only on request, N11): ODC-BY.

The repository stores only split IDs, Dolly line numbers, WildChat conversation hashes and generated same-topic questions derived from the benchmarks; see the notices for how they are covered. The demo store, its products and its photos are fictional; the images were generated with qwen-image-3.0.
