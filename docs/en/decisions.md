# Design decision records (ADR)

English edition of [docs/decisions.md](../decisions.md), which remains the source of truth. Each record gives the context, the decision, the alternatives and trade-offs, the consequences and how it was checked. Decisions can be reversed, but a reversal is recorded with its reason. Some measurements are condensed here; the Chinese records and the per-node results in [docs/results/](../results/) have the full tables.

| ID | Decision | Status |
|---|---|---|
| D-001 | When and how deeply to retrieve is decided by a separate controller | Adopted |
| D-002 | An immutable raw log; facts carry bi-temporal validity | Adopted |
| D-003 | Multimodal memory: compress at indexing time, re-perceive at answer time | Adopted |
| D-004 | All models through APIs: DeepSeek + Qwen, by role | Adopted (revised in N1) |
| D-005 | Embedded storage; Qdrant can switch to a server | Adopted (checked in N1) |
| D-006 | A hand-written agent runtime | Adopted |
| D-007 | Evaluation first | Adopted |
| D-008 | The judge and the main reader come from different model families | Adopted |
| D-009 | A synchronous core with thread pools | Adopted |
| D-010 | Costs: nominal cost for comparisons, actual spend tracked separately | Adopted |
| D-011 | Structured output: JSON mode + schema in the prompt + validated retries | Adopted |
| D-012 | No transactions across the two stores: deterministic IDs + write order | Adopted |
| D-013 | Qwen through QwenCloud; keys only in .env; costs in USD | Adopted |
| D-014 | Text embeddings: text-embedding-v4 in N1, qwen3.7-text-embedding (dense only) from N4 | Revised (N4) |
| D-015 | LoCoMo judged with LongMemEval's official templates, token F1 reported too; adversarial questions excluded by default | Adopted |
| D-016 | The evaluation runner: per-question stores, shared caches, trace metering, resumable runs, a budget guard | Adopted |
| D-017 | Reader prompt revised (v2) after a pilot run, on development data only | Adopted |
| D-018 | Extraction with qwen3.8-flash (confirmed against deepseek-flash on a sample) | Adopted |
| D-019 | Fact reconciliation: rules only merge, negate and accumulate; every supersession is confirmed by an LLM; "the world changed" vs "we were wrong" | Adopted (revised after measurement) |
| D-020 | Consolidation pipeline: stateless parallel extraction, stateful serial reconciliation; recorded time = conversation time | Adopted |
| D-021 | Forgetting = delete from the raw log + replay derived memory, and clear caches by ownership | Adopted (revised after N12: no more VACUUM, D-068) |
| D-022 | Core memory is a rule-built view over the fact table, not rewritten by an LLM | Adopted |
| D-023 | B2F isolates the write path's contribution: same retrieval, different write structure | Adopted |
| D-024 | DeepSeek off-peak scheduling: batch jobs pause at peak, spend counts the off-peak discount | Adopted (after N12: the SDK defaults to warn, D-068) |
| D-025 | L1 fusion: max within a modality, RRF across modalities | Adopted |
| D-026 | The sparse route uses BM25, not text-embedding-v4's term weights | Revised (N5: no sparse route with qwen3.7) |
| D-027 | Temporal awareness: width-discounted range boost + relative-time packing; summary prior and recency boost off by default | Adopted |
| D-028 | Faster evaluation: persisted memory spaces, retrieval-only tuning, single-flight requests | Adopted (revised after N12: a consolidation-config mismatch is an error, not a rebuild, D-068) |
| D-029 | L1 packs context to a token budget, not a fixed number of turns | Adopted |
| D-030 | L2 deep recall: query planning + best-rank multi-query fusion + sufficiency check and follow-up + note-taking | Adopted |
| D-031 | Graph diffusion (personalised PageRank): implemented, off by default | Implemented, off |
| D-032 | Re-ranking (cross-encoder): not adopted | Implemented, not adopted |
| D-033 | Two baselines for N7: a rule-based controller (B6r) and agentic retrieval (B3) | Adopted |
| D-034 | MemLens: development questions from outside the subset; the official 195 kept as the test set; official judge prompts | Adopted |
| D-035 | Writing images: originals stored by content hash; captions (context-aware + isolated) generated at consolidation and fed into fact extraction | Adopted (H6 not supported) |
| D-036 | Image retrieval: captions as keys, plus text-to-image search (on by default) | Adopted (better retrieval, same accuracy) |
| D-037 | Re-perception at answer time: attach the 3 most relevant original photos | Adopted (H5 supported) |
| D-038 | The controller: two stages (gate + reasoning type → table lookup), labelled by reasoning type | Adopted (N7) |
| D-039 | The mixed stream: general requests + same-topic hard negatives, to evaluate "whether to recall" | Adopted (N7) |
| D-040 | Default λ: "no worse than always retrieving"; auto mode uses the learned controller and reuses the probe's candidates | Adopted (N7) |
| D-041 | H3 refuted: with another reader, agentic retrieval is best; kept as a controller level for N9 | Recorded (N7) |
| D-042 | Agent: the controller recalls first, the model may search more (memory_search as an escape hatch) | Adopted (N8) |
| D-043 | Context: partitioned budgets + stable prefix; one profile snapshot per session; working memory evicted in blocks and excluded from retrieval | Adopted (N8) |
| D-044 | The agent uses qwen3.8-flash | Adopted (N8) |
| D-045 | Writing chats: append every turn, turn numbers only grow; open sessions are not consolidated; a new job supersedes a pending one | Adopted (N8) |
| D-046 | Forgetting removes whole rounds and follows citations; the profile can be cited; traces are cleaned too | Adopted (N8) |
| D-047 | Demo: a fictional store with generated images; English UI; a replayable script with checks; MCP recalls but does not answer | Adopted (N8) |
| D-048 | Final evaluation: each test set used once, methods and configuration frozen in advance; LoCoMo's last 7 conversations as the test set | Adopted (N9) |
| D-049 | Reader models: small, medium and large, on stratified subsets | Adopted (N9) |
| D-050 | Structured output also falls back to another model on request timeouts | Adopted (N9) |
| D-051 | Forgetting a fact also deletes messages that restated it without citing it | Adopted (N10; revised after N11) |
| D-052 | Controller v2: image-space accuracy table in spaces with photos; visual-detail questions never go to L0; test sets only replayed post hoc | Adopted (N11) |
| D-053 | Reconciliation separates "different scope", "the world changed" and "it was a mistake", and sees the source message; the pipeline version is not bumped | Adopted (N11) |
| D-054 | The agent's reply cache is scoped to the session | Adopted (N11) |
| D-055 | External baseline Mem0: same models, same reader and judge, its own write and read pipeline | Adopted (N11) |
| D-056 | Deployment: a clear error when the local store is locked; payload indexes on the server; a migration command; an optional access token | Adopted (N11; revised after N12: a token is required off loopback, D-068) |
| D-057 | The gate on real traffic: first messages of real chatbot conversations (WildChat) | Adopted (N11) |
| D-058 | Controller v3: in a space with photos, look at them only when the question is about one | Adopted (after N11) |
| D-059 | Gaps left by forgetting are marked on re-extraction; a pronoun next to a gap is not pinned on someone else | Adopted (after N11) |
| D-060 | Switching model providers: config shipped in the package, a data directory, presets (openai, ollama), an OpenAI-compatible embedding API; Docker | Adopted (N12) |
| D-061 | A Python SDK in Mem0's shape, with CueMem's semantics (version chain, controller, forgetting) | Adopted (N12) |
| D-062 | An OpenAI-compatible memory proxy: the controller decides first, evidence goes before the last user message | Adopted (N12) |
| D-063 | Export and import move only the raw log; forgetting can be previewed | Adopted (N12) |
| D-064 | Extra extraction instructions (custom instructions) | Adopted (N12) |
| D-065 | Same-protocol comparison with more external baselines: Graphiti, LangMem, Cognee, A-MEM; a DeepSeek-only (pds) and a Qwen-only (pq) protocol | Adopted (N12) |
| D-066 | Framework integrations: OpenAI Agents SDK, LangGraph / LangChain, Claude's memory tool; one shared implementation of recalled evidence | Adopted (N12) |
| D-067 | LongMemEval session-ID leak: fix, measure, disclose | Adopted (N12) |
| D-068 | A full audit: seven lenses, offline reproduction, adversarial verification, 24 fixes; evaluation requests and published numbers unchanged | Adopted (after N12) |

## D-001 A separate controller decides when and how deeply to retrieve, not the LLM

- **Status:** adopted (N0).
- **Context:** letting the LLM decide through tool calls makes quality depend on the model (small models are unreliable), makes behaviour hard to predict and evaluate, and lets model capability mask the project's own contribution.
- **Decision:** every turn first runs a memory probe that calls no LLM; a light, trainable controller then picks L0 / L1 / L2 (and whether to look at the original photos) with an explicit cost term, escalating when the evidence is insufficient. LLM-driven retrieval stays as a baseline (B3) and as an escape hatch.
- **Alternatives:** always retrieve (wastes tokens, adds distractors); fixed rules (brittle); uncertainty-triggered retrieval in the style of FLARE / DRAGIN (unreliable for private memory, and API models don't expose internal signals).
- **Consequences:** measurable, trainable, independent of the reader, with a cost knob; training labels have to be built, and the controller can overfit.
- **Checked by:** N7's Pareto curves, the comparison with B3, cross-dataset and cross-reader experiments (H1–H3).

## D-002 The raw log is immutable and derived memory can be rebuilt; facts carry bi-temporal validity

- **Status:** adopted (N0).
- **Context:** user information changes. Overwriting or deleting old facts loses history ("what did I use to…"), and derived data (summaries, facts) may be extracted wrongly and needs to be repairable.
- **Decision:** raw turns are the only source of truth. Facts, summaries, the profile and image captions are derived, carry provenance and can be rebuilt in full. On conflict, the old fact is invalidated (`valid_to`, `superseded_by`), not deleted.
- **Alternatives:** Mem0-style UPDATE / DELETE (simple, loses history); store raw text only, no extraction (conflicts never handled explicitly).
- **Consequences:** history questions can be answered, everything is auditable, extraction can be re-run after an upgrade; storage and code are more complex.

## D-003 Multimodal memory: compress at indexing time, re-perceive at answer time

- **Status:** adopted (N0).
- **Context:** MemLens found that long-context vision-language models degrade with length, while memory agents lose visual detail by compressing images when storing them.
- **Decision:** images are stored as image embeddings, context-aware captions with OCR, and entity bindings; at answer time the controller decides whether to fetch the original images for the model.
- **Alternatives:** captions only (cheap, loses detail); all images always in context (expensive, degrades with length).
- **Consequences:** stable with length and faithful to visual detail, at the cost of image tokens at answer time, which the controller manages. Checked in N6 on MemLens (H5, H6).

## D-004 All models through APIs: DeepSeek + Qwen, by role

- **Status:** proposed in N0, revised in N1.
- **Context:** the development machine is an Apple M2 with 8 GB of memory and about 29 GB of free disk. The available APIs are DeepSeek (chat only; deepseek-flash reads images) and Qwen (chat, vision, text and multimodal embeddings, re-ranking).
- **Decision:** LLMs and VLMs are reached through OpenAI-compatible APIs by role (design §9.3). Embeddings also go through the Qwen API, behind a content-hash cache; a local embedding backend has an interface but is not implemented.
- **Why the revision:** N0 planned a local BGE-M3. Running it on 8 GB competes with the browser and IDE; Qwen embeddings cost $0.07 per million tokens and the cache makes evaluation embedding cost negligible; the native API also returns sparse vectors; and nobody needs to download torch and model weights.
- **Cost:** one more network round trip per query embedding, and a dependency on the network.

## D-005 Embedded storage: SQLite + Qdrant local mode (switchable to a server) + in-process graph computation

- **Status:** adopted after an N1 spike.
- **Context:** one command should run it on a laptop, without Docker or external services, with little memory.
- **Decision:** SQLite for structured data and facts; Qdrant local mode for vectors (named dense / sparse / image vectors, payload filters on time and validity); scipy.sparse for PageRank. Every store sits behind an interface; setting `vector_store.url` switches to a Qdrant server without code changes.
- **Measured (N1, M2 / 8 GB, 1024-d dense + ~60-term sparse):** dense search p50 1.2 ms at 1,000 points, 4.7 ms at 5,000 and 16.9 ms at 20,000; sparse 12 / 61 / 245 ms; filtered dense 5 / 25 / 101 ms; peak process memory 223 MB / 661 MB / 1.7 GB.
- **Conclusion:** evaluation spaces hold a few thousand points, which local mode handles. Local mode is brute force in Python, so larger spaces should use a Qdrant server (HNSW, inverted indexes). N11 measured this end to end (D-056).
- **Alternatives:** Neo4j (strong graphs, heavy); Postgres + pgvector (needs a server); LanceDB (embedded, disk-friendly).

## D-006 A hand-written agent runtime, not LangChain / LlamaIndex at the core

- **Status:** adopted (N0).
- **Context:** the mechanisms this project most wants to show are the agent loop and harness, context assembly and memory reads and writes. A hand-written runtime keeps them explainable and testable on their own; framework abstractions would hide them.
- **Decision:** the loop, context assembly, tool calls and memory reads and writes are all hand-written; only base libraries (LLM SDK, store clients) are used.
- **Consequences:** every mechanism is explainable, controllable and testable; retries, concurrency and structured output have to be handled by hand (D-009, D-011). In N8 the loop, context assembly and tools in `src/cuemem/agent/` are about 1,000 lines; FastAPI, Gradio and the MCP SDK are only used for the service shell.

## D-007 Evaluation first

- **Status:** adopted (N0).
- **Context:** reported numbers on text memory benchmarks are close to saturation, and papers use different readers, judges and prompts, so their numbers don't compare.
- **Decision:** build the unified evaluation framework and baselines before any advanced method; compare every method in the same framework with the same reader and judge; the main result is the accuracy–cost Pareto curve.
- **Consequences:** every change comes with evidence; early progress looks slow.

## D-008 The judge and the main reader come from different model families

- **Status:** adopted (N1).
- **Context:** LLM judges prefer answers from their own family. LongMemEval's official judge is GPT-4o, which this project has no API for.
- **Decision:** the main reader is DeepSeek (deepseek-flash), the judge is Qwen (qwen3.8-max), fixed for every experiment. When a Qwen model is the reader, check for family bias by hand.
- **Alternatives:** the reader judges itself (strongly biased); a panel of judges (twice the cost; left for a robustness check). N11 re-judged the test-set answers with a second judge from the other family (deepseek-flash).

## D-009 A synchronous core with thread pools, not asyncio throughout

- **Status:** adopted (N1).
- **Decision:** all core code is synchronous. Where concurrency is needed (the evaluation runner, L2's parallel retrieval), thread pools; one semaphore per provider; locks around SQLite and Qdrant local mode.
- **Why:** the bottleneck is network I/O, which threads handle; synchronous code is easier to read, debug and explain; async is contagious, and SQLite and Qdrant local mode are synchronous libraries; FastAPI runs synchronous handlers in a thread pool.
- **Cost:** very high-concurrency services would do better with asyncio; the LLM client could switch to the async SDK then.

## D-010 Costs: nominal cost for comparisons, actual spend tracked separately

- **Status:** adopted (N1).
- **Context:** DeepSeek is half price off-peak, both providers discount cache hits, Qwen has free quotas for new users, and local cache hits cost nothing. Comparing methods by actual spend would make the same experiment cost different amounts at different times.
- **Decision:** the ledger records two costs per call. **Nominal cost** uses the peak, cache-miss list price, for comparisons and Pareto analysis. **Estimated spend** subtracts local cache hits, provider cache discounts and (from D-024) the off-peak discount, for budgets. Token counts and call counts are reported too.
- **Consequences:** conclusions don't depend on when an experiment ran; prices live in `src/cuemem/configs/default.yaml`.

## D-011 Structured output: JSON mode + schema in the prompt + validated retries

- **Status:** adopted (N1).
- **Context:** extraction, planning and judging need reliable structured output. Both providers support JSON mode, but strict JSON-schema support differs, and DeepSeek's JSON mode sometimes returns an empty reply.
- **Decision:** always use `response_format={"type": "json_object"}`, put the pydantic JSON schema into the system prompt, and validate with pydantic. On a validation error, feed the specific errors back and retry (3 attempts by default); empty replies are never cached.
- **Consequences:** provider-independent and observable (attempts are recorded); occasionally one more call.

## D-012 No transactions across the two stores: deterministic IDs + write order

- **Status:** adopted (N1).
- **Context:** a memory is written to both SQLite (content) and Qdrant (vectors), which cannot share a transaction.
- **Decision:** every ID is derived deterministically from content, so rewriting overwrites rather than duplicates. Vectors are written first and the database second, so anything in the database has a vector; after a failure, re-running fills the gap (embeddings hit the cache). The embedding model's fingerprint is stored in the database, and switching models raises an error instead of mixing vectors.
- **Consequences:** idempotent, recoverable writes with no transaction coordinator. (This is also why switching to a Qdrant server needs `cuemem migrate-vectors`, D-056.)

## D-013 Qwen through QwenCloud; keys only in .env; costs in USD

- **Status:** adopted (N1).
- **Context:** Alibaba merged DashScope, Bailian and Model Studio into two brands: QwenCloud outside mainland China and the Qianwen AI platform inside. Endpoints, keys and model lists differ (QwenCloud has no qwen3-vl-embedding; multimodal embeddings use the tongyi-embedding-vision series). This project's account is on QwenCloud.
- **Decision:** the default configuration points at QwenCloud (chat at `https://maas.qwencloudapi.com/compatible-mode/v1`, embeddings through the native API on the same domain, which alone returns sparse vectors). Keys live only in `.env`, entered with `cuemem configure` (no echo, masked, file mode 600); a missing key is reported before any call. Nominal costs use USD list prices, since both providers publish them.
- **Consequences:** no account information in the repository, and costs add up directly.

## D-014 Text embeddings: text-embedding-v4 in N1; qwen3.7-text-embedding (dense only) from N4

- **Status:** adopted in N1, revised in N4.
- **N1:** measured with real keys, qwen3.7-text-embedding took 164 ms per query (p50) and 1.5 s per batch of 10, and its "sparse" vectors are semantically expanded (about 523 terms; "Max is a golden retriever" expands to dog, puppy, dog food, even German "Hunde"). text-embedding-v4 took 52 ms and 113 ms, with true term weights (about 20 terms). On a small sample their semantic separation looked similar, so v4 became the default: 13× faster writes and real lexical weights for hybrid search.
- **N4 revision:** both reasons stopped holding. The sparse route moved to local BM25 (D-026), and dense-only qwen3.7 is only about 6× slower, on a one-off write. Semantic recall, which N1 never measured, differed a lot (30 stratified dev100 questions, turn-level dense top-10): turn recall (all) 74.1% → 89.3%, NDCG 54.3% → 72.7%, at the same price ($0.07 per million tokens). End to end, B2 went from 65% to 76% on dev100 and from 71.2% to 75.6% on LoCoMo's 385 questions; B2F barely moved (80% → 80%, 77.7% → 78.7%), because fact-augmented keys had already made up most of the weaker model's recall.
- **Lesson:** N1 compared speed and sparse-vector properties but never quantified semantic recall. Choose components by the target metric, and re-evaluate when the premises change.

## D-015 LoCoMo judged with LongMemEval's official templates; token F1 reported too; adversarial questions excluded by default

- **Status:** adopted (N2).
- **Context:** the LoCoMo paper scores with token F1 and no LLM judge; later work (e.g. Mem0) designed its own judge prompts.
- **Decision:** map question types to LongMemEval's official templates (the temporal template that allows an off-by-one day, the abstention template for adversarial questions), so both benchmarks use the same ruler; also report token F1. As in most papers, adversarial questions are excluded by default (`--types` includes them).
- **Consequences:** consistent across benchmarks, but LoCoMo numbers do not compare directly with papers that judge differently; the report says so.

## D-016 The evaluation runner

- **Status:** adopted (N2).
- **Decision and reasons:**
  - **Per-question stores, shared caches.** Each LongMemEval question has its own history, so each gets its own database and vector store, while the embedding and LLM caches are shared and persistent, so re-runs are free. Memory use stays flat, and runs can be concurrent on 8 GB.
  - **Metering through traces.** Each (question, method) runs in its own trace, which sums tokens, calls and nominal cost; judging runs in a separate trace, outside the method's cost. Traces nest, so inner calls (e.g. `MemorySystem.ask`) are counted in the outer one.
  - **Resumable runs.** Results are appended line by line to `results.jsonl` and finished combinations are skipped; failures go to `errors.jsonl` and are retried next time.
  - **A budget guard.** Actual spend (excluding local cache hits) accumulates, and no new combination starts beyond `--max-cost`.
  - **Traceability.** `run.json` records the configuration, the model IDs and prices per role, the embedding model and the code version (git commit, `-dirty` with uncommitted changes).
  - **Pilot before full runs** on a stratified 10-question sample.
- **Consequences:** one command reproduces a results table, and the cost and duration of a 100-question run can be estimated in advance.
- **After N12 (D-068):** when a resumed run still has questions to answer, it must also match, beyond D-028's checks, the models of every stored role (judge, extractor, planner and so on), `--k`, `--limit`, the question set, the consolidation and retrieval settings, and the controller's version and file content; otherwise it stops and asks for a new run name. Complete runs are not checked. `run.json` also records a hash of the question IDs, the controller's version and sha256, the command line and the CUEMEM_* environment (secrets removed), plus one entry per resume. `cuemem eval run --fail-on-error` makes an incomplete run exit non-zero (3 for errors, 4 for a budget stop); the default exit code is unchanged.

## D-017 Reader prompt revised (v2) after a pilot run, on development data only

- **Status:** adopted (N2).
- **Context:** after the pilot10 run, every wrong answer was read by hand. The judge was right every time, but two reader problems unrelated to any method showed up: over-refusal (advice questions such as "tips for getting around Tokyo" should use the user's memories, but the model said there were none; once it found the answer and still said it had no record) and language drift (an English question answered in Chinese because of a Chinese example in the system prompt).
- **Decision:** v2 removes the Chinese example, separates "say you don't know only when a factual question can't be answered" from "advice questions must use personal information", and asks for a direct answer when there is one. It applies to every method equally.
- **Result:** on pilot10, B1 went from 70% to 90% and the drift disappeared. Tuned only on development data (pilot10 ⊂ dev100); `test400` was not looked at before the final evaluation.

## D-018 Extraction with qwen3.8-flash

- **Status:** adopted (N3).
- **Context:** extraction is the highest-volume step of the write path (about 50 sessions per LongMemEval question, about 4,800 calls for dev100); the candidates were each provider's cheapest model.
- **Measured** on 42 evidence sessions and 30 distractor sessions from 30 stratified dev100 questions, same prompt: evidence-turn coverage 100% vs 100% and answer content-word recall 97.8% vs 97.8% (qwen3.8-flash vs deepseek-flash); $0.00086 vs $0.00192 per session; p50 latency 20.8 s vs 5.5 s.
- **Decision:** qwen3.8-flash by default. Quality was equal on these measures, extraction runs in the background, and it costs less than half. deepseek-flash is about 4× faster, for when bulk imports must be quick.
- **Cost and limits:** the first consolidation of dev100 took about 35 minutes; the response cache makes re-runs take minutes. Only "was the information extracted" was compared, not the accuracy of every detail.

## D-019 Fact reconciliation: rules only merge, negate and accumulate; every supersession is confirmed by an LLM

- **Status:** adopted in N3, revised after measurement on dev100.
- **Context:** each new fact has to be compared with existing memory: add, supersede, merge or ignore. Mem0 asks an LLM to choose ADD / UPDATE / DELETE / NONE for every new fact, which is simple but expensive (hundreds of facts per LongMemEval question) and hard to audit.
- **First design:** rules first. A registered single-valued predicate (residence, employer, job title…) with a new value supersedes the old one directly; only unregistered predicates go to the LLM.
- **Measurement overturned it.** After the first dev100 run, a hand check of a sample of replayed decisions found that **rule-based supersessions were right only 3 times out of 31 (about 10%)**. The extractor's predicate labels were unstable ("has a small wardrobe" and "walls affect the Wi-Fi at home" were both labelled `lives_in`), and LongMemEval's distractor sessions come from different people, so "the user is 32" and "the user is 20" contradict each other. LLM-confirmed supersessions were right about 18 times out of 28 (about 64%). Wrong "no longer true" labels misled the reader: two knowledge-update questions were answered wrongly because the new value had been marked invalid, while the ablation without validity labels got them right.
- **Revised decision:**
  1. Rules only do exact things: same subject, predicate and object → merge provenance; nearly identical statements with the same object → merge; negating an existing positive fact → supersede; a registered multi-valued predicate → add.
  2. Every supersession is confirmed by an LLM: a single-valued fact meeting a same-predicate or similar fact, or a multi-valued fact meeting a very similar one, goes to the LLM to choose update / correct / duplicate / add, with "if both can be true, or you are unsure, choose add". One batched call per session, at most 3 candidates per fact.
  3. Two kinds of supersession. **The world changed** (update: moving, a new job): the old fact's `valid_to` is set to the new fact's start. **We were wrong** (correct: the user corrects or refines it): only `invalidated_at` is set and `valid_to` stays empty, because it never "ended" in reality. A single time axis cannot tell "I moved" from "I misspoke".
  4. Similarity alone never merges: embeddings are insensitive to numbers and names ("budget 8,000" vs "budget 6,000" can be very similar), so a similarity merge also requires the same object.
  5. Event-time comparisons happen only when both facts carry an explicit event date; otherwise the order in which they were said decides.
- **Consequences:** more reconciliation calls, each short and batched; every decision records the rule that made it. Lesson: verify a rule's premise (here, reliable predicate labels) first; wrong structure misleads the model more than no structure. (N11 refined the prompt's definitions, D-053.)
- **After N12 (D-068):** "same subject" and "same object" in these rules compare normalised keys. Text in scripts other than Latin and Chinese (Cyrillic, Greek, Hangul, kana, Arabic, Hebrew and others) used to normalise to an empty string: speakers were merged into one entity, facts were dropped and different values were treated as duplicates. These scripts now get an NFKC + casefold key. Keys for Latin (accented letters included) and Chinese are unchanged, fact IDs are still derived from the original normal form, and evaluation keys and requests are unchanged.

## D-020 Consolidation pipeline: stateless parallel extraction, stateful serial reconciliation; recorded time = conversation time

- **Status:** adopted (N3).
- **Context:** a LongMemEval space has about 50 sessions. Extracting and reconciling session by session is too slow, but reconciliation depends on all earlier sessions.
- **Decision:**
  - **Two stages.** Extraction only sees the session (and its date): stateless, parallel, cached (a session shared by several questions is extracted once). Reconciliation and entity resolution depend on existing memory and run serially in session order. In between, all new statements, entity descriptors and summaries are embedded in one batch.
  - **Job key** = space + session + session content fingerprint + pipeline version: each job runs once, and appended turns or a new pipeline version change the key. A version change clears derived memory and replays.
  - **Atomic writes:** each session's results commit in one database transaction (vectors first, D-012).
  - **Recorded time is the conversation's time**, not the wall clock, so replays are deterministic.
  - **Late old messages don't override new state:** if a new fact's event time is earlier than the current fact's, it is written as a historical version. A test checks that importing new-then-old gives the same current facts as importing in order.
  - **Only the first 600 characters of each assistant reply are extracted from:** assistant text is 87% of LongMemEval's dialogue, and memory is mostly about the user; full replies stay in the log and the turn index.
  - **A background worker** runs consolidation per space without blocking chats.
- **Consequences:** a space's consolidation takes roughly "the slowest few extractions + serial reconciliation", and rebuilding derived memory is nearly free (all cache hits). Trace context has to be passed into worker threads (`contextvars.copy_context`) for costs to be counted.
- **After N12 (D-068):**
  - **Recorded time:** when the latest source message of a fact was said on a different day from the session start, the fact is recorded at that message's time, and the extraction prompt marks the date before messages on a new day; within one day the session start is kept (every evaluation session is like this, so requests are unchanged). Writes within the same second are reconciled in write order (generated session IDs carry an increasing counter, and default times are no longer truncated to the second); times with a time zone are converted to local time.
  - **Background worker:** failed extractions are retried with backoff (1 minute, then 4 minutes, at most 3 attempts), and so is a whole consolidation run that raises; the first start resumes unfinished jobs; shutdown waits at most 30 seconds and leaves the rest for the next start.
  - **A session that grows:** the job key still changes with the content, but when a consolidated session gains messages, derived memory is cleared and replayed from the raw log instead of extracting again on top of the old derived memory (D-045).

## D-021 Forgetting = delete from the raw log + replay derived memory, and clear caches by ownership

- **Status:** adopted (N3).
- **Context:** when a user says "forget this" or deletes something in the UI, it must be gone (the right to be forgotten under Singapore's PDPA and the EU's GDPR). The hard part is derived data: a fact may have superseded others, which should become current again once it is gone; entities, summaries and the profile may all be affected. Tracing these relations one by one is easy to get wrong.
- **Decision:**
  1. Derived memory is a function of the raw log. Forgetting deletes the turns from the log and replays derived memory from what remains; replays hit the cache, so unaffected sessions are not re-extracted.
  2. Caches are cleared too. Every LLM cache entry carries an ownership label (extraction per session; reconciliation and answers per space), and forgetting deletes the affected sessions' extraction entries and the space's reconciliation and answer entries. Embedding-cache entries for texts that existed before but not after are deleted as well, since vectors can leak content.
  3. Physical deletion: SQLite's `secure_delete` overwrites deleted content, followed by `VACUUM`. A forensic test reads the database file's bytes to check the deleted words are gone.
- **Alternatives:** cascading deletes along provenance (complex; restoring superseded facts is error-prone); soft deletes (not compliant).
- **Consequences:** correctness is easy to argue and test ("the state after forgetting equals the state if it had never been said"); the whole space is replayed, which is fast with the cache. (N8 extended this to citations and traces, D-046; N10 to uncited restatements, D-051; N11 scoped the reply cache per session, D-054.)
- **After N12 (D-068):**
  - **No more `VACUUM`** (revises point 3): the memory database keeps `secure_delete` on and, after deleting, commits and truncates its WAL (`MemoryDB.scrub`); the LLM and embedding caches truncate their WAL as soon as they delete entries, and close with the system. Nothing deleted is left in the files or the WALs, which the forensic test still scans byte by byte, also checking that no `VACUUM` ran. The cost is that the database file does not shrink after forgetting; `VACUUM` rewrites the whole file, which took over 4 seconds on a 146 MB database and stalled every read and write on it.
  - **Embedding cache** (extends point 2): the query vectors of the deleted messages, of the questions in deleted traces and of the deleted fact statements are removed too, and for messages with images also those of the text without the image marker.
  - **Concurrency:** forgetting, rebuilding and deleting a space hold that space's consolidation lock throughout and wait for answers, retrievals and agent turns in flight (one reader-writer gate per space), so neither background consolidation nor an answer can write deleted content back.
  - **Clean up on failure:** when the replay fails halfway, traces and caches are still cleaned and the WAL truncated before the exception reaches the caller; unfinished consolidation completes next time.
  - Photos uploaded through the HTTP API and the UI no longer leave a copy in `<data>/uploads`.

## D-022 Core memory is a rule-built view over the fact table, not rewritten by an LLM

- **Status:** adopted (N3).
- **Context:** the profile sits in the context every turn and must be accurate. Letting an LLM rewrite it can invent things or drift from the fact table (a superseded fact lingering in the profile).
- **Decision:** profile = the participants' current facts with importance ≥ 4, sorted by importance and recency, truncated to about 400 tokens, regenerated whenever facts change, with no LLM call.
- **Consequences:** it never contradicts the fact table, every line traces to a source, and it is free; it reads as a list rather than prose, which makes no difference to the reader.

## D-023 B2F isolates the write path's contribution: same retrieval, different write structure

- **Status:** adopted (N3).
- **Decision:** a method, B2F, keeps B2's dense top-10 retrieval and only adds the write path's output: fact-augmented keys (facts are retrieved with turns and pass their score to their source turns, the key expansion of the LongMemEval paper), facts with validity in the context (with superseded and superseding versions brought in together), and the core memory. Four ablations: keys only (no facts shown), current facts only (equivalent to overwrite updates), facts without validity labels, and no core memory.
- **Results:** dev100 B2 65% → B2F 80% (+18 / −3, p = 0.001), not significantly different from ORACLE's 83%; none of the ablations differed significantly from the full version. LoCoMo's first 3 conversations (385 questions, not tuned on): 71.2% → 77.7% (+42 / −17, p = 0.002), multi-hop 42% → 61%.
- **Method note:** comparing methods on the same questions with a paired sign test (only questions where exactly one is right) is much more sensitive than comparing two independent confidence intervals.

## D-024 DeepSeek off-peak scheduling

- **Status:** adopted (before N4).
- **Context:** DeepSeek charges double on weekdays 09:00–12:00 and 14:00–18:00 (Beijing time). The reader (deepseek-flash) dominates evaluation spend.
- **Decision:** providers declare peak hours in the configuration (time zone, weekdays, windows, off-peak factor 0.5). Before a real call, the LLM client checks the policy: `wait` pauses until the peak ends (cache hits are unaffected), `warn` calls and logs, `allow` just calls. Evaluations default to `wait`; interactive commands use `warn`. Nominal cost keeps the peak list price; estimated spend counts the discount. Heavy DeepSeek evaluations are scheduled for early mornings, 12–14, evenings and weekends.
- **Consequences:** up to half the spend saved, at the price of evaluations sometimes pausing for hours; tested with an injected clock.
- **After N12 (D-068):** the Python SDK (`Memory`, `AsyncMemory`) and the integrations built on it used to inherit the configured `wait`, so during DeepSeek's peak hours `answer()` silently waited for the peak to end (up to 4 hours). The SDK is an interactive entry point and now defaults to `warn`, like the CLI; `Memory(peak_policy="wait")` opts into off-peak waiting and logs a warning on `cuemem.sdk` when the wait starts. A Settings object keeps its own value. Evaluations and `cuemem eval run --peak` are unchanged.

## D-025 L1 fusion: max within a modality, RRF across modalities

- **Status:** adopted (N4).
- **Context:** L1 merges three granularities (turns, facts, session summaries) and two modalities (dense, sparse) into one ranking of turns. The first implementation ranked each unit type and modality separately and fused everything with RRF.
- **Measured** (dev100, retrieval only): per-type RRF gave a turn NDCG of 65.8%, 14 points below B2F's 80.1%, and "what did the assistant say" questions fell from 79% to 39%. RRF keeps only ranks and throws away score magnitudes: an assistant turn has no matching fact, so a strong direct hit was pushed down by several medium-ranked fact keys.
- **Decision:** first take the maximum within a modality (a turn's own score vs. its source facts' scores; dense cosines are comparable across unit types; sparse scores are normalised per list), then RRF only across modalities (sparse weight 0.5, by grid search). RRF is for fusing lists on different scales, not for splitting scores that were already comparable.
- **Consequences:** the dense part equals B2F exactly (a correctness check). Re-measured with qwen3.7: per-type RRF 75.3% NDCG vs 79.5%; "assistant said" questions 51.5% vs 95.5%.

## D-026 The sparse route uses BM25, not text-embedding-v4's term weights

- **Status:** adopted in N4; **revised in N5**: with qwen3.7, L1 and L2 use no sparse route by default (BM25 kept as an option; N4's B4 still evaluated with it).
- **N4:** BM25 beat v4's term weights on retrieval (turn recall (any) 100% vs 96.8%, NDCG 79.3% vs 77.1%; end to end 80% vs 82%, not significant), with controllable tokenisation, no API dependency and explainable matches.
- **After switching to dense-only qwen3.7:** the question became whether to keep a sparse route at all. Retrieval-only on dev100, BM25 made multi-session and temporal rankings worse (turn recall (all) 92.9% vs 96.8%, NDCG 79.5% vs 83.0%); end to end the two datasets went in opposite directions, neither significant (dev100 83% vs 81%, LoCoMo 84.9% vs 86.2%).
- **N5 re-check** (94 non-abstention dev100 questions, token-budget packing): BM25 never improved recall at any budget, with or without re-ranking, with one query or many; re-ranking didn't rescue it either.
- **Revised decision:** no sparse route by default (`L1Config.sparse = "none"`); BM25 stays as an option, useful with weaker embeddings. When end-to-end results don't differ significantly, choose the side with better retrieval metrics, simpler code and lower latency.

## D-027 Temporal awareness

- **Status:** adopted (N4).
- **Decision and evidence** (dev100):
  - **Parsing time expressions** in English and Chinese into ranges relative to the question time ("two weeks ago", "last Saturday", "in March", "last year", "Valentine's Day"). "today / tonight" in a question usually means now and is not a retrieval cue.
  - **Range boost:** turns in the range (or whose source facts' event time is in it) get a boost discounted by the range's width; wide ranges cover most of the history and would only scramble the ranking. Temporal NDCG 73% → 77%, but only 12 dev100 questions and 31 LoCoMo questions have a parsable expression, and end to end there was no measurable difference. Kept on: nearly free and explainable.
  - **Relative-time packing:** each piece of evidence is labelled "N days before the question". Removing it: dev100 83% → 79% (+5 / −1, p = 0.22), multi-session 54% → 42%; LoCoMo unchanged. Not significant but consistent, so kept.
  - **Session-summary prior and "latest / earliest" recency boost** lowered NDCG and are off by default.

## D-028 Faster evaluation

- **Status:** adopted (N4).
- **Decision:**
  - **Persisted memory spaces** under `paths.eval_spaces`, reused across runs (writes and consolidation are idempotent); `build.json` keeps the first build's write and consolidation statistics so reports still show write costs; spaces rebuild automatically when the embedding model or consolidation configuration changes.
  - **Retrieval-only tuning** (`scripts/tune_l1.py`) on persisted spaces: recall and NDCG only, no reader, about 30 seconds per dev100 pass and zero LLM cost, so it can run at peak hours.
  - **Single-flight requests:** identical concurrent requests call the API once.
  - **Resume checks:** resuming a run name checks the dataset, split, embedding model and reader first (after nearly mixing new results into an old run in N4).
- **Consequences:** a retrieval iteration went from minutes and dollars to half a minute and almost nothing.
- **After N12 (D-068):**
  - **No automatic rebuild** (revises "persisted memory spaces"): when a reused evaluation space's consolidation fingerprint or pipeline version does not match, the methods that need fact memory record a ConfigMismatchError that names each protocol's directory; B0, B1 and the external baselines still answer, and build.json is left alone; `cuemem eval run --rebuild-spaces` rebuilds. Before, one wrong `CUEMEM_PATHS__EVAL_SPACES` was enough to rewrite the published spaces of another protocol with this run's configuration. A different embedding model already lands in another directory.
  - **Resume checks** (extends the item of that name): while questions remain, the models of every role, `--k`, `--limit`, the question set, the consolidation and retrieval settings and the controller are checked too (D-016).

## D-029 L1 packs context to a token budget, not a fixed number of turns

- **Status:** adopted (N4).
- **Context:** a "turn" is a user message plus a long assistant reply in LongMemEval (about 490 tokens) but one line of chit-chat in LoCoMo (about 40 tokens); a fixed k = 10 gives the two datasets context that differs by more than 10×, and on LoCoMo 10 turns often miss the several pieces a multi-hop question needs.
- **Decision:** add turns in L1 score order until a 4k-token budget is used (at least 1 turn, at most 40); facts (10, plus version chains) and the core memory are extra.
- **Data note:** this was not decided on dev100 alone. A LoCoMo check with text-embedding-v4 (fixed-10 B4 76.1% vs B2F 77.7%) led to it, and the budget size looked at both datasets (2k clearly worse on dev100, similar to 4k on LoCoMo). LoCoMo is therefore not an independent test of D-029; N9's test sets are.
- **Measured** (qwen3.7): fixed 10 turns 82% on dev100 and 77.1% on LoCoMo; a 4k budget 83% (8.2 turns, 24% less input) and 84.9% (40 turns). On LoCoMo, 4k vs fixed 10 was +41 / −11 (p < 0.001), multi-hop 53% → 81%.
- **Consequences:** one setting works for both kinds of data. On LoCoMo the effective limit is the 40-turn cap, which was not tuned on LoCoMo.

## D-030 L2 deep recall

- **Status:** adopted (N5).
- **Context:** after N4, B4 got 17 dev100 questions wrong, and multi-session questions were at 54%. Errors were either aggregations whose evidence is scattered across chats (L1 found only part of it) or reading and reasoning errors that even ORACLE made (missed items, double counting, date arithmetic).
- **Decision** (defaults set by retrieval-only experiments on dev100, then checked end to end):
  1. **Query planning:** one qwen3.8-flash call returns the answer type, whether exhaustive recall is needed, 1–4 sub-queries, a time range and target entities; a rule-based plan if it fails.
  2. **Best-rank multi-query fusion:** each candidate takes its best rank score across queries, plus 0.1 × the sum to break ties. Sub-queries each find one piece; summing ranks (RRF) rewards turns that are mediocre for every query.
  3. **6k tokens, up to 60 turns;** 25 facts for counting, sums and lists (10 for L1).
  4. **Sufficiency check:** up to 2 checks (one qwen3.8-flash call each); if insufficient, retrieve with the queries it proposes and add 1k tokens each time.
  5. **Note-taking:** a reading prompt at the end of the context; the output writes notes before the answer.
  6. No graph diffusion, re-ranking or BM25 by default (D-031, D-032, D-026).
- **End to end** (dev100): B4 83%, L1 without BM25 81%, L1 with notes 86%, **L2 89%** (2.1× L1's cost, 3.25 LLM calls per question); without notes 81%, without the sufficiency check 85%, without planning 88%, with graph diffusion 88%. LoCoMo 385 (no tuning): L1 86.2%, L1 with notes 88.6%, **L2 89.1%** (vs B4 +27 / −11, p = 0.014).
- **Reading:** the biggest lever is how evidence is read, not retrieval: the same L2 evidence without notes scores 81%, L1's evidence with notes 86%. In retrieval, the follow-up step is what matters; planning's sub-queries help retrieval metrics but barely help end to end, because the sufficiency check proposes similar queries. Wrong "I don't know" answers fell from 6 to 1.
- **Acceptance:** the roadmap asked for significant gains on multi-session and temporal reasoning. Both datasets point the same way, but per-type differences did not reach p < 0.05 (N9's test sets later did on LoCoMo). Notes also lengthen answers, which lowers LoCoMo token F1 (0.236 → 0.195); the LLM judge is the main metric.

## D-031 Graph diffusion (personalised PageRank): implemented, off by default

- **Status:** implemented, off (N5).
- **Context:** the design's multi-hop tool for L2 was HippoRAG-style entity-graph diffusion.
- **The graph itself:** with only extracted relations, there is almost nowhere to walk. A dev100 space averages about 230 facts and 150 entities; "the user" is attached to about 95% of facts and the median degree of other entities is 1; on LoCoMo the two speakers hold 39–56% of facts each. The implemented graph therefore uses three node types (entities, facts, turns), four edge types (fact–entity, fact–source turn, entity mentions in raw text, similar facts with cosine ≥ 0.8), drops hub entities attached to more than 20% of facts, and seeds from named entities plus the top facts and turns of the first fusion pass.
- **Measured** (dev100, retrieval only): every variant was within ±0.5 points of recall; end to end B5-ppr 88% vs B5 89%. On LoCoMo (denser person relations) B5-ppr was 90.1% vs 89.1% (+8 / −4, p = 0.39). Off by default (`L2Config.ppr`).
- **Why it doesn't help here:** LongMemEval's multi-session questions are about finding all instances of one kind of thing scattered across chats, not hopping along entity relations; sub-queries and follow-up retrieval do that more directly. Graphs extracted from personal chats are stars centred on one person; PPR suits dense entity graphs that need bridging entities.

## D-032 Re-ranking (cross-encoder): not adopted

- **Status:** implemented, not adopted (N5).
- **Implementation:** QwenCloud's qwen3-rerank on the top 30 fused candidates; final rank = RRF of fused rank and re-rank rank; cached and metered.
- **Measured:** recall essentially unchanged, NDCG +1–1.5 points; about 15k tokens per question (≈ $0.0015, 10× the planning call, as much as answering with L1) plus a few hundred milliseconds.
- **Why not:** evidence is shown in time order, so ranking only decides which turns fit in the budget; what matters is recall within the budget, which re-ranking didn't improve. NDCG within the budget doesn't affect the reader. Re-ranking also didn't make BM25 useful.

## D-033 Two baselines for N7: a rule-based controller (B6r) and agentic retrieval (B3)

- **Status:** adopted (N5).
- **Decision:**
  - **B6r:** looks only at question cues, no LLM, no retrieval results: counting or aggregation, duration, order (excluding time expressions such as "last week") and comparison go to L2, everything else to L1. Written before looking at L2's per-question results.
  - **B3:** the same reader with `memory_search` (L1 retrieval, 2k tokens per call, deduplicated, numbered across calls) and `date_diff`, plus an `answer` tool; at most 5 steps, the last one must answer. Restricting tools was not enough: DeepSeek kept calling `memory_search` because it appeared in the history, so the first version answered "I don't know" after many steps; naming the tool in `tool_choice` fixed it.
- **dev100:** L1 81% ($0.0016), B6r 88% ($0.0028, 62% routed to L2), L2 89% ($0.0034), B3 86% ($0.0033, 2.8 searches on average), a hindsight-perfect router 92% ($0.0019, not deployable).
- **LoCoMo:** B6r sent only 8% to L2 and equalled L1 (86.0%): LoCoMo's multi-hop questions rarely contain the cue words. Rules on question wording don't transfer across datasets, which motivated the probe features and the cross-dataset evaluation in N7.
- **N7 correction:** "a lot of room up to the perfect router" was wrong. Most of the perfect router's gain comes from answers flipping at random (L2 rescued 45 dev questions and broke 17). Routing by the true question type is the realistic ceiling (D-038).

## D-034 MemLens: development questions from outside the subset; the official 195 kept as the test set; official judge prompts

- **Status:** adopted (N6).
- **Data:** MemLens (CC BY 4.0) has 789 questions in five abilities (information extraction, multi-session reasoning, temporal reasoning, knowledge update, refusal), each at 32K, 64K, 128K and 256K tokens of history, with images attached to turns.
- **Splits:** 100 development questions sampled by type from the 594 outside the official 195-question memory-agent subset (`memlens_dev100`); the 195 (`memlens_agent195`) kept whole as the test set, comparable with the paper's seven memory agents.
- **Judging:** the official `llm_judge.py` prompts, copied verbatim (MIT); answers over 500 words score 0, as in the official code. The official judge (Qwen3-VL-235B) is not on QwenCloud, so qwen3.8-max is used, as for the other datasets; comparisons with the paper note the difference.
- **Building and answering separately:** `--build-only` writes and consolidates with Qwen models only, so it can run at DeepSeek peak hours; answering runs off-peak. QwenCloud's JSON mode occasionally aborts generation ("Model output became abnormal"); structured output now retries with a hint, and one failed caption no longer fails a whole history.

## D-035 Writing images

- **Status:** adopted (N6; H6 not supported).
- **Decision:**
  - **Online write (no LLM):** originals stored by SHA-256 (one copy per image); one record per appearance (the same image means different things in different turns); image embeddings with tongyi-embedding-vision-plus (1,152-d, same space as text) in a per-space image collection.
  - **Consolidation:** two captions per image, **context-aware** (the image plus one turn before and after) and **isolated** (the image alone), plus text in the image and keywords, with qwen3.8-flash (reads images, cheapest, off the read path). Both are retrieval units that point back to their turn; the isolated caption is only for comparison.
  - **Entity binding:** fact extraction sees the context-aware caption in place of the image, so information in images reaches facts and entities. Sessions without images render exactly as before, so existing extraction caches still hit.
  - **Forgetting** deletes image records, image vectors and, if nothing else refers to it, the original file.
- **Cost:** MemLens 32K dev100 (1,279 images) cost $1.28 to build; two captions per image ≈ $0.00033.
- **Captions help** (dev100): text-only memory 38% → 48% with context-aware captions (+13 / −3, p = 0.021).
- **Context-aware vs isolated (H6): not supported.** Context-aware captions retrieve slightly better (65.1% vs 60.7% recall of the key image's turn), but isolated captions answered 3 more questions end to end (51% vs 48%, n.s.): context-aware captions spend words on what the image means in the conversation, isolated ones on visual detail, and MemLens mostly asks about visual detail.

## D-036 Image retrieval: captions as keys, plus text-to-image search (on by default)

- **Status:** adopted (N6; better retrieval, same accuracy).
- **Measured** (89 dev100 questions, recall of the key image's turn): text only 51.2%, + context-aware captions 64.2%, + facts 56.5% (fact keys take budget from other turns), **+ text-to-image search ×0.5: 85.3%** (multi-session 26% → 80%). Visual cues in the question ("that red bus", "the amount on the receipt") sit close to the image itself in the joint embedding space, while a caption is a lossy paraphrase.
- **Decision:** text-to-image search on by default, as a third modality in the RRF (weight 0.5); skipped automatically in spaces without images.
- **End to end, retrieval improved and accuracy didn't:** e.g. test 32K, key-image recall 61.7% → 87.2%, accuracy 62.6% → 63.1% (+8 / −7). Likely reason: the images it finds pull their turns into the budget and push out other evidence, while captions and facts already recover the dialogue around the key image. The default was fixed on development retrieval metrics and not changed after seeing test results.

## D-037 Re-perception at answer time: attach the 3 most relevant original photos

- **Status:** adopted (N6; H5 supported).
- **Decision:** images in the packed turns are ranked by caption-key and text-to-image ranks; the top 3, resized to 768 px on the long side, go after the text evidence and before the question, each marked "see picture k" in the text. The reader stays deepseek-flash, so methods differ only in whether originals are attached.
- **Measured:** L1 with photos vs without: dev100 +8 / −1 (51% → 58%), test 32K +26 / −2 (50.8% → 63.1%), test 128K +19 / −5 (48.7% → 55.9%), concentrated in information extraction (41% → 72% at 32K). About 670 extra input tokens per question (+11% nominal cost), +0.2 s at p50; full context with images costs 27k (32K) to 135k (128K) tokens per question.
- **Length stability** (the same 195 questions at 32K and 128K): full context 52.3% → 46.2% (p = 0.012) at 5× the cost; L1 with photos 63.1% → 55.9% (session recall 93% → 80%); L2 with photos 65.6% → 64.6% (n.s.). Memory systems are not automatically stable with length: L1's single retrieval loses evidence in long histories too, and L2's sub-queries and follow-up keep it stable.
- **Against ORACLE (evidence sessions only):** L2 with photos is 10 points higher at 32K (+30 / −11, p = 0.004). Giving the right things beats giving everything relevant.

## D-038 The controller: two stages (gate + reasoning type → table lookup), labelled by reasoning type

- **Status:** adopted (N7; revises the "advanced" version in design §6.3).
- **Context:** the plan was full-information offline labelling (run every action on every question and judge it), then learn a calibrated success probability per action and choose by P̂ − λ · ĉ. After labelling 585 development memory questions, the outcome labels turned out to be very noisy: L2 rescued 45 questions from L1 and broke 17; on single-fact questions the split was +6 / −5, essentially random. The hindsight-perfect router (cheapest correct action) reached 88.5% at $1.41 per 1k questions, better and cheaper than always-L2 by riding those flips; routing by the true question type reached only 84.4% at $2.44. A model fitted to per-action correctness learns "how hard is this question", not "which level is better".
- **Decision:** two stages.
  1. **Gate and reasoning type:** from the probe's 54 features, a binary logistic regression estimates P(no memory needed), and a multinomial logistic regression, trained on memory questions only, estimates the reasoning type (single fact, multi-session / multi-hop, temporal, knowledge update, preference, open inference, abstention).
  2. **Table lookup:** each level's accuracy per type comes from training data, shrunk towards the overall memory-question mean (prior strength 5); the photo gain is estimated only from data with images (MemLens). P̂(a | x) = Σ_t P(t | x) · acc(a | t); choose by P̂ − λ · ĉ(a), ties to the cheaper action.
  The gate is separate because non-memory queries outnumber memory questions 2:1 in the mixed stream and would crowd the type classifier. Standardised features are clipped to ±5, so a new embedding model or language cannot push the logits to extremes.
- **Alternatives** (cross-fitted, three development sets): per-action success probabilities (area under the memory-question curve 0.814 vs 0.820); deep / shallow only (similar, but seven types say more); Adaptive-RAG-style "cheapest correct action" labels (same noise, no λ); gradient-boosted trees (no better, harder to export).
- **Runtime:** two matrix products and a JSON file (`src/cuemem/read/controller_model.json`); no scikit-learn at runtime; tests check that training (`evals/routing.py`) and runtime (`read/controller.py`) agree exactly.
- **Results:** on memory questions B6 (λ = 30) reached 81.2% at $2.14 per 1k, +19 / −15 against always-L1, and did not beat the rule-based B6r (82.6%, $2.01). Multi-session / multi-hop is the hardest type to recognise (44% taken for single facts), and a bag-of-words classifier recognises types better than the probe features (78% vs 74%): the type is decided by how the question is worded; retrieval feedback doesn't help.
- **Consequences:** routing between L1 and L2 earns far less than the perfect router suggested. A better reading strategy (notes before answering, +$0.0003 for L1) beats every routing policy. The controller's value lies in whether to use memory at all (D-039).

## D-039 The mixed stream: general requests + same-topic hard negatives

- **Status:** adopted (N7).
- **Context:** every benchmark question needs memory, so L0 decisions cannot be evaluated; in real assistant traffic most turns don't need long-term memory.
- **Decision:**
  - **General requests:** databricks-dolly-15k (CC BY-SA 3.0), the four categories without reference text, 585 lines sampled by category; only line numbers are committed.
  - **Same-topic hard negatives:** for each memory question, qwen3.8-flash writes a question on the same topic that is not about the user ("When did Gina open her online store?" → "What steps does opening an online clothing store usually take?"); 585, committed in `evals/mixed/topic_negatives.jsonl`.
  - **Paired spaces:** a hard negative uses its memory question's space and time, so the probe finds strong hits on the same topic; general requests borrow a random space. Cross-validation keeps pairs in the same fold.
  - **Outcome assumption:** non-memory queries are answered well at any level, and the reader never says "I don't know" to them (the most favourable assumption for uncertainty-based escalation). Report memory-question accuracy and average cost at 20% and 50% memory questions, plus false recall and missed recall.
- **Measured** (20% memory questions): always-L1 $1.57 per 1k; B6 $0.87 (−45%) at 81.2%, false recall 19%, 2.4% of memory questions sent to L0; the zero-shot LLM router $0.72 at 80.7%, false recall 0.4%, but 5.3% of memory questions judged "no memory" and one extra call per query. AUC for "needs memory": question cues 0.79, probe feedback 0.96, both 0.98.
- **Limitations:** the generated negatives are stylistically uniform ("What are common…") and easier to spot than real traffic; MemLens questions carry formatting instructions. Cross-validated numbers are optimistic; the cross-dataset results (D-040) are closer to a new setting. N11 checked the gate on real chatbot messages (D-057).

## D-040 Default λ: "no worse than always retrieving"; auto mode uses the learned controller and reuses the probe's candidates

- **Status:** adopted (N7).
- **Default λ:** λ is a runtime knob, and sweeping it draws the Pareto curve. The default follows a product rule fixed in advance: the largest λ whose accuracy on the combined development memory questions is no lower than always-L1 ("the cheapest that is no worse than always retrieving"). That gives λ = 30 per dollar: one extra correct answer is worth at most 1/λ ≈ 3.3 cents. The first rule ("within 1 point of always-L2") only allowed λ = 0, which never saves anything in a mixed stream; the change was made, and recorded, before the test sets were opened.
- **Runtime:** `MemorySystem.ask(mode="auto")` and `cuemem ask --mode auto --router …`.
  - `learned` (default): run the probe, let the controller choose; if it chooses L1, pack the probe's candidates directly; attach 3 originals if it chooses re-perception. Without a model file, fall back to the rules.
  - `hybrid`: if P(no memory) is between 0.1 and 0.9, ask a zero-shot LLM router once and replace the gate's estimate with a fixed 0.95 / 0.05 (fixed in advance).
  - `llm` and `rule`: the zero-shot router and B6r.
  - The choice, each action's predicted success and utility, and the probe time are recorded and shown by `--show-context`.
- **Check:** the final model's live decisions on the development sets matched offline replay on all 585 questions.
- **Across datasets:** trained on the other two datasets, the gate treats LoCoMo's third-person questions as non-memory (55% of memory questions sent to L0, 86.5% → 43.9%). Twenty labelled target examples bring it back to 85.7%; MemLens needs about 100. In production, start from rules or the zero-shot router and calibrate after collecting a few dozen labelled logs.

## D-041 H3 refuted: with another reader, agentic retrieval is best; kept as a controller level for N9

- **Status:** recorded (N7; adds to D-001).
- **Context:** H3 expected the controller to be more stable than agentic retrieval (B3) across reader models, with a bigger advantage on small models. N7 re-ran L0 / L1 / L2 and B3 with qwen3.8-flash as a second reader on LongMemEval dev100 and LoCoMo's 385 questions (the controller's choices don't depend on the reader).
- **Measured** (deepseek-flash → qwen3.8-flash): always-L2 89.0% → 82.0% and 89.1% → 86.2%; B6 84.0% → 82.0% and 86.5% → 81.8%; **B3 86.0% → 91.0% and 88.8% → 90.1%**. With qwen3.8-flash, B3 beat always-L2 on LoCoMo (+28 / −13, p = 0.028) at $1.41 vs $2.28 per 1k, with fewer searches.
- **Reading:** qwen3.8-flash reads a large static block of retrieved text worse than deepseek-flash, but is better in a "search, read a few, decide" loop. Whether "let the LLM decide" is good depends on whether that model is good with tools. The controller's choices don't change with the reader; its results do. Two caveats: qwen3.8-flash is not a smaller model, and it shares a family with the judge (D-008), so absolute accuracies across readers don't compare.
- **Decision:** keep the controller's design (its value is in whether to use memory, and a cost ceiling); in N9, evaluate agentic retrieval as one of the controller's options across 2–3 readers, including a genuinely smaller one.

## D-042 Agent: the controller recalls first, the model may search more

- **Status:** adopted (N8; implements D-001's escape hatch and answers D-041).
- **Decision:** each message is first routed and recalled by the controller (with the LLM-free probe); the model can still call `memory_search` for anything missing, with numbering continuing and nothing repeated. At most 4 tool steps; the last step may not call tools.
- **Alternatives:** controller only (no answer when the gate is wrong); model only (at least one extra round trip per message, and the model may not search).
- **Measured** (the 13-turn demo script): the gate misjudged "How has my shoe budget changed since we first talked?" as 96% non-memory and chose L0; the model called `memory_search`, found the superseded $120 and the current $180, and answered with citations. 3 of 13 turns used the escape hatch; the others made no extra calls.
- **Consequences:** the controller decides the minimum to give, the model decides what is missing; both are in the trace and can be evaluated separately (B6a in N9).

## D-043 Context: partitioned budgets + a stable prefix

- **Status:** adopted (N8; design §6.6).
- **Decision:**
  1. Partitions and budgets: system prompt, tool descriptions, profile (600 tokens), working memory (3,000), recall (4,000; 6,000 for L2), current message.
  2. Stable prefix: system prompt, profile and working memory come first, so each request in a session starts with the previous request; the date and recalled memories go only into the last message and are not kept for the next turn. Citation numbers are stripped from working-memory replies.
  3. Profile snapshot: taken once per session, not refreshed by background consolidation mid-session (refreshed after forgetting).
  4. Block eviction: when working memory exceeds its budget, evict down to half in one go, so the prefix changes only on that turn; evicted turns are in long-term memory and retrievable.
  5. Retrieval excludes working memory: the probe, L1, L2 and `memory_search` skip turns and images already in context.
- **Measured** (`scripts/n8_context.py`, local cache off): in a 16-turn session, 70% of input came from the provider's prefix cache with this layout vs 52% with a conventional one; in the short demo script both were similar (54% vs 56%).

## D-044 The agent uses qwen3.8-flash

- **Status:** adopted (N8).
- **Why:** it was strongest in the search–read–decide loop (D-041), it reads images (finding look-alikes of a photo), it costs about half of deepseek-flash, and Qwen has no peak pricing. The controller's accuracy table was estimated with deepseek-flash, so its choices are somewhat off for this reader (see N11's λ finding, D-052).

## D-045 Writing chats

- **Status:** adopted (N8).
- **Decision:**
  1. After each turn, the new exchange is written online (log + index + image vectors, no LLM). Turn numbers are kept by the session and only grow: numbers freed by forgetting are never reused.
  2. Open sessions are not consolidated (`open_session` / `close_session`), including by consolidations triggered elsewhere and replays after forgetting; when a session ends it is extracted once, in the background.
  3. When a session's content changes, a newly enqueued job marks pending or failed jobs for it as superseded, so a session is never extracted twice.
- **Consequences:** what was just said is retrievable at once (after it leaves working memory); facts and the profile update when the session ends.
- **After N12 (D-068):**
  - **A session that grows after consolidation** (extends point 3; per-turn SDK `add` with the same run_id, a LangGraph thread that continues after it ended, the proxy reusing a named session): it used to be extracted again in full on top of the old derived memory, so wordings from the earlier extraction stayed. Now, when the LLM cache is writable, derived memory is cleared and replayed from the raw log (cache hits; only the grown session is extracted again). Only sessions that already had a job are enqueued, and open sessions are skipped. The cost is recomputing the whole space locally each time (about 12 s for 200 sessions), which is why the SDK gained `run()`: a conversation is extracted once, when it ends.
  - **Open-session markers** (extends point 2): `open_session` leaves a marker in the metadata and `close_session` removes it. Sessions still open when a process crashed are handed to extraction by the next process the first time it touches the space (`MemorySystem.adopt_open`; the SDK, the HTTP service and the proxy all do this), except a session the caller is about to continue. This assumes one process writes a database at a time.

## D-046 Forgetting removes whole rounds and follows citations; the profile can be cited; traces are cleaned too

- **Status:** adopted (N8; extends D-021).
- **Context:** in the first live replay, after forgetting "Ana's birthday", the assistant still said "Oct 3": an open session's greeting had cited the profile line, and the source round's assistant reply repeated the information.
- **Decision:**
  1. Deleting a fact or an image deletes the whole round it came from (user message + assistant reply).
  2. Every reply's citations (facts, excerpts, images, profile lines numbered [p1]…) are recorded in a provenance table that survives rebuilds. Forgetting also deletes rounds whose replies cited deleted memories, and keeps following until nothing new is found.
  3. Open sessions drop deleted rounds from working memory, invalidate deleted image labels and re-take the profile snapshot.
  4. Trace entries whose attributes contain deleted text are deleted.
- **Known limit** (fixed by D-051): replies that restate without citing are not found.
- **After N12 (D-068):** point 4 is wider: traces whose retrieval records (spans, events) list a deleted round, fact or image ID are deleted too, and the query vectors of their questions are purged. Matching by ID errs towards deleting: in a small space, forgetting one session can remove most answer traces (traces are JSONL for debugging, and no product code reads them). Retrievals made outside another trace (SDK search, the proxy, the integrations, MCP and HTTP recall) now record a retrieve trace, so forgetting finds those too.

## D-047 Demo: a fictional store with generated images; English UI; a replayable script; MCP recalls but does not answer

- **Status:** adopted (N8).
- **Decision:** a shopping-assistant scenario; the store CueMart, its 24 products and a "friend's shoes" photo are fictional, with images generated by qwen-image-3.0 (about $0.75 for 25, prompts committed). `demo/shop/script.json` holds three chats on simulated dates, each message with must-include / must-not-include checks and an expected level; replays are cache hits. The UI and script are in English. The MCP server offers `remember`, `recall` (returns numbered, dated facts and excerpts without calling a reader, so the client's own model reasons), `forget` and `profile`, over stdio.

## D-048 Final evaluation: each test set used once, methods and configuration frozen in advance

- **Status:** adopted (N9).
- **Decision:**
  1. Test sets: LongMemEval test400; LoCoMo's last 7 conversations (1,155 non-adversarial questions; the "first 385 questions" used in N3–N7 are the first 3 conversations, registered as `locomo_dev3`); MemLens's official 195 questions at 32K and 128K (used once before, in N6, with methods also fixed in advance).
  2. Frozen: the controller model (exported in N7, λ = 30), L1 / L2 configurations, all prompts and judge templates. The only new method, B6a (D-042), was defined before the test sets were opened. Nothing is tuned on the test sets, and results are reported as they come.
  3. The method list is fixed in `scripts/reproduce.py`. Controller methods send exactly the same requests as the fixed levels they choose, so they hit the cache: evaluating the controller equals "frozen model + each level's real results", checked against live runs question by question.
  4. A test version of the mixed stream: 1,750 Dolly requests disjoint from N7, plus a same-topic question generated for each test question.
- **Consequences:** test numbers can be read as untuned estimates; problems found on the test sets are recorded, not fixed (N11 then worked on them without touching the test sets for decisions, D-052).
- **After N12 (D-068):** what is frozen now has offline guards. `tests/test_eval_leak.py` pins the requests of B1, B2, B2F, B4, B5, B6, B3 and B6a by cache key, method by method (LoCoMo mini and the LongMemEval fixture with raw IDs), and `tests/test_frozen_config.py` pins the L1 / L2 configurations, the frozen methods' parameters and the sha256 of the three controller files. `scripts/reproduce.py` only checks the controller files and never overwrites them (the training scripts run with `--no-export --check`), and it never copies an errored or budget-stopped report into docs/results.

## D-049 Reader models: small, medium and large, on stratified subsets

- **Status:** adopted (N9).
- **Decision:** small: qwen3.6-35b-a3b (MoE, about 3B active parameters; priced from OpenRouter's listing); medium: deepseek-flash (main) and qwen3.8-flash; large: deepseek-v4-pro (4.4× deepseek-flash's price). Stratified subsets: 100 questions from test400 (`test100r`) and 200 from the LoCoMo test set (`test200r`). Each reader runs L0, L1, L2, B3 and B6a; B6 and B6r are computed offline from the levels' results.
- **Note:** the large tier has not been run.

## D-050 Structured output also falls back to another model on request timeouts

- **Status:** adopted (N9).
- **Context:** one test400 session (12 turns, about 19k characters of ordinary Q&A) made qwen3.8-flash's JSON-mode generation run away: streamed requests were aborted by the server, non-streamed ones never returned before the 120 s client timeout. Five retries over three hours all timed out. The same input sometimes returned in 6 seconds and sometimes not within 240: the problem was this model on this input, not the network.
- **Decision:** `LLMClient.structured` already switched to `fallback_model` (deepseek-flash during consolidation) on content-moderation refusals; it now also switches on `APITimeoutError`, after the SDK's own retries.
- **Why:** a runaway generation and a false moderation refusal are the same class of problem ("this model can't produce a result for this input"), and another provider's model gets through. Fallback results are cached like any other.
- **Consequences:** a timing-out session waits for the SDK's retries (4 × 120 s) before falling back, but no longer fails the whole question. After the change the rebuild succeeded on qwen's first try, so the fallback was never triggered; all test-set memory was extracted by qwen3.8-flash.

## D-051 Forgetting a fact also deletes messages that restated it without citing it

- **Status:** adopted (N10; fixes D-046's known limit).
- **Context:** while recording the README's demo GIF, forgetting "Ana turns 30 on October 3" in the UI (following the demo guide) and then asking "When is my sister's birthday?" still produced October 3. In the September chat the assistant had restated "her birthday is October 3rd" in a later reply, before the birthday was a citable fact; after deleting the source and replaying, the extractor pulled the fact out of that restatement again. An open session's greeting had also mentioned it without citing. The scripted replay passed only because it deleted every fact about Ana, which happened to delete the restating round too.
- **Decision:** when forgetting by fact (UI, HTTP, MCP and replays all go through `ChatService.forget`):
  1. Search the raw log for restatements: for each fact, take the 8 most similar rounds by vector (including open sessions) and let a cheap model (qwen3.8-flash) confirm which still reveal the information (full, paraphrased or partial; merely mentioning the same person doesn't count). Delete those rounds with the source.
  2. After the rebuild, check whether a semantically equivalent fact grew back (vector candidates + model confirmation); if so, delete its source and repeat, up to 3 times.
  3. The forget report counts restated rounds (`restated`) and regrown facts (`regrown`) separately.
- **Why:** forgetting means "this can't be recalled any more", not just "delete the sentence it came from". Citations are lineage; uncited restatements can only be found by content. Checking regrown facts alone is not enough, because open sessions are never consolidated yet keep the text in the model's window. Vectors propose candidates and the model decides: true restatements score 0.6–0.9 in cosine similarity and mere mentions of the same person 0.5–0.57, too close for a threshold.
- **Consequences:** one or two cheap calls per forgotten fact, plus a rebuild per regrowth round (15–35 s in the demo space). Limits: only the top 8 rounds are checked; deleting images doesn't search for restatements; the model may delete loosely related content (erring towards deletion, as privacy requires).
- **Revision (after N11, 2026-10-13): delete only the message that reveals it; restatements found by content don't start a citation cascade.**
  - **Found:** re-recording the demo GIF with the N11 code, forgetting "Ana turns 30 on October 3" deleted 14 messages (8 "later replies that cited it", 4 restatements) and took the facts from 14 to 8: the day's whole chat was gone, and so was "I've switched to EU 40 for running shoes". Two mechanisms compounded. Restatements were deleted as whole rounds, so the September round where the user switched to EU 40 went, user message included, because the assistant's reply mentioned Ana's gift in passing. The citation cascade then started from that round too, deleting every fact extracted from it and every later reply that cited the EU 40 fact. In the N10 recording those replies happened not to mention the birthday.
  - **Change:**
    1. The restatement check gives the model the candidate rounds message by message and deletes only the messages it confirms; whatever else the user said in that round stays.
    2. Restatements found by content (and the sources of regrown facts) are deleted without starting a citation cascade; the cascade starts only from the fact's own source.
    3. In an open session, a round leaves the working memory if either its user message or its reply was deleted.
  - **Result:** the re-recorded GIF deletes 4 messages (the source round, plus 2 replies that mentioned the birthday in passing: one in September, one in today's greeting), and facts go from 16 to 14. The last question gets "the only person I have on file is Ana, who you bought trekking poles for, but you never said she was your sister or mentioned a birthday". A regression test covers the case: the user's EU 40 message in the restating round, and a later answer citing it, both stay.
  - **New limit:** once the source round is deleted, a later message that refers to it by pronoun loses its referent. In one recording, "The trekking poles are perfect, I'll get those for her" was re-extracted as "plans to buy trekking poles for Lena" (the only woman left in memory). A real fix would change the extraction prompt (don't guess an unresolved referent), which invalidates every cached extraction in the evaluation spaces, so it was not done; the demo script now says "for Ana". Resolved later by D-059: gaps left by forgetting are marked on re-extraction, the sentence becomes "for someone else", and the pipeline version stays.
- **After N12 (D-068):** the deleted fact's statement used to stay in the embedding cache (the restatement search embedded it as a query, and the regrowth check embedded it again as a document); both are now purged. Traces of answers that restated it are deleted by the IDs in their retrieval records (D-046).

## D-052 Controller v2: image-space accuracy table, visual-detail questions never go to L0; test sets only replayed post hoc

- **Status:** adopted (N11; addresses the three controller issues N9 recorded on the test sets).
- **Context:** N9 found that adding a photo gain and clipping at 1.0 made L1 + photos and L2 + photos tie; that the gate misses questions referring to a shared photo; and that the per-level accuracy is calibrated on the main reader. The test sets had already been used, so they could not drive any decision.
- **Decision:**
  1. In a space with photos, look up an image-space accuracy table (reasoning type × {L0, L1, L2, L1 + photos, L2 + photos}), estimated only from training questions with photos and shrunk towards the overall mean of those questions (prior strength 5).
  2. In a space with photos, a question with a visual cue (`q_visual`) or a position cue (new `q_scene`: "on the left", "in the background", "in the photo"…) is lifted from L0 to L1 + photos. `q_scene` is used only by this rule, not by the classifiers.
  3. The gate, the type model and the λ rule are unchanged (the rule again gives 30).
  4. The frozen v1 is kept as `controller_model_v1.json` and used to reproduce N7–N9; v2 is the default; `retrieval.controller` chooses the file.
  5. The test sets are only used for a post-hoc replay of the frozen v2, reported, never tuned on.
  6. For a new reader model, set λ per reader rather than recalibrating the table.
- **Alternatives:** adding the photo gain in log-odds (still assumes the gain is the same for both levels); recalibrating the table per reader (tried; see below).
- **Consequences:**
  - Development (cross-fitted): 81.2% → 81.7% (+5 / −2, n.s.); memory questions sent to L0 2.4% → 1.7%; +1% cost.
  - Test (post-hoc): 81.6% → 81.9% (+10 / −5, n.s.), all on MemLens; text datasets unchanged question by question; still 2.5 points below always-L2.
  - **A correction to N9 §5.3:** the image table alone recovers the 6 questions attributed to saturation but changes about 70 other choices, netting +7 / −7 at 32K. Each cell is estimated from 100 development MemLens questions. The net gain comes from the floor rule (3 of the 5 memory questions v1 sent to L0).
  - Recalibrating the accuracy table on qwen3.8-flash's development results made things worse (LongMemEval 93% → 89%, LoCoMo 76.5% → 74.0%). Lowering λ to 15 recovered most of the gap to always-L2 for all three readers at 16–24% lower cost (a post-hoc curve on the test subsets; the default λ is unchanged).
- **After N12 (D-068):** `scripts/n11_controller_v2.py` has `--check`. The reproduction script runs it with `--no-export --check`, so it only compares the result with the committed `controller_model_v2.json` and stops without writing anything on a mismatch.

## D-053 Reconciliation separates "different scope", "the world changed" and "it was a mistake", and sees the source message; the pipeline version is not bumped

- **Status:** adopted (N11; fixes two reconciliation errors from N9's error analysis).
- **Context:** "up at 6:00 on Fridays" superseded "up at 6:30 on weekdays", so a question about how much earlier the user gets up on Fridays could not be answered. In the English example, "moved to Singapore" was labelled `corrected`, so the validity had no end and the timeline read as if the user had always lived there.
- **Decision:**
  1. `correct` only when the user says the earlier fact was a mistake ("sorry, I meant…", "actually it was…") or the new fact states the same thing more precisely; moving or changing jobs is `update`.
  2. `add` explicitly covers facts with a different or narrower scope: weekdays vs Fridays, at work vs at home, one trip vs another, even when the new fact is a recent change ("I work from home on Mondays now" does not replace "I work at the office"); `update`'s example changed from "a changed schedule" to "a schedule changed for the same days".
  3. **Each new fact in the gray zone comes with the message it was taken from** (the user's words, up to 400 characters). The extracted fact "Mia is 31" cannot tell a correction ("I got that wrong before") from a birthday; with the prompt change alone, the live test still labelled the correction as a change.
  4. The prompt's opening stays the same (the scripted test model dispatches on it).
  5. `PIPELINE_VERSION` is not bumped: the consolidation fingerprint holds the pipeline version, the extraction model and the consolidation parameters, not prompt text, so existing evaluation spaces are not marked stale and rebuilt.
- **Why:** bumping the version would rebuild every evaluation space (re-extraction and reconciliation), which is expensive and would stop N9's numbers being reproducible from the existing spaces. The revision affects memories written from now on.
- **Consequences:** a new live test (`tests/live/test_reconcile_live.py`) on an invented conversation checks that the weekday and Friday facts both stay current, that Shanghai → Singapore is `changed` and that a corrected age is `corrected`. Run 5 times, all three cases are right 5 / 5 (N9's prompt: 3 / 5, 5 / 5, 0 / 5); the prompt was revised twice against this conversation, so this only shows that the two errors are fixed. Rebuilding N3–N9 from scratch now sends different reconciliation prompts, so those memories would differ slightly; for an exact reproduction, check out the commit before N11 (a8b8f07).
- **After N12 (D-068):** an evaluation space whose fingerprint does not match is no longer rebuilt but reported as an error (D-028). A future `PIPELINE_VERSION` bump will therefore make existing evaluation spaces raise ConfigMismatchError until `--rebuild-spaces` is passed or the directory changes, instead of rebuilding them silently.

## D-054 The agent's reply cache is scoped to the session

- **Status:** adopted (N11).
- **Context:** the reply cache used to be labelled with the space (`space|ask`), so forgetting cleared every session's cached replies. In demo replays, every question after a forget then called the model again, which sometimes answered differently and failed a check.
- **Decision:** label reply-cache entries with the session (`space|session:<id>`). Forgetting clears the affected sessions' reply caches plus consolidation and the old `ask` label.
- **Why:** cache ownership should follow data lineage: a session's replies depend on what that session saw, and forgetting already finds the sessions that contained the forgotten content (D-046, D-051).
- **Consequences:** a test forgets a fact in session B and checks that session A's cached replies survive while B's are gone.

## D-055 External baseline Mem0: same models, same reader and judge, its own write and read pipeline

- **Status:** adopted (N11).
- **Context:** until now the project could only cite other systems' numbers from different setups (different readers, judges and prompts, see D-008). At least one external system had to run inside the same evaluation.
- **Decision:**
  1. Open-source Mem0 (mem0ai 2.2.1): the most common point of comparison, and it runs locally. It is evaluation method `M0` (`evals/mem0_baseline.py`); mem0ai is not a dependency and is installed for the run with `uv run --with`.
  2. Its own pipeline: one `add` per session (Mem0 2.x does one LLM extraction per call, add-only, with the 10 most related existing memories in the prompt), retrieval with `search(top_k=30)` (default 20), and spaCy and fastembed installed so its hybrid search (semantic + BM25 + entity boosts) is complete. LoCoMo is a conversation between two people, so it follows the evaluation code of the Mem0 paper: one user_id per speaker, each written once (that speaker's turns as user, the other's as assistant, both with names), and retrieval takes 30 from each side, each memory labelled with whose it is (the extracted "User" means that speaker). The first version wrote both speakers as user under one user_id, which left "User" ambiguous; this was changed before the API runs.
  3. The same models: Mem0's LLM is replaced by qwen3.8-flash (CueMem's extractor) and its embeddings by qwen3.7-text-embedding, both through CueMem's client (same ledger, same cache, same fallback to deepseek-flash on moderation refusals, timeouts and aborted JSON). The reader, reader prompt and judge are the same as for every method; each memory is labelled with its source session's time.
  4. Dates: the open-source version does not accept `timestamp`, so the "observation date" in its extraction prompt defaults to the day of the run and "yesterday" lands in 2026. The adapter sets it to the session date, as the hosted version's timestamp would.
  5. Data: LoCoMo `test200r` and LongMemEval `test100r` (N9's stratified subsets); CueMem's methods use their N9 results on the same questions.
  6. Telemetry off (`MEM0_TELEMETRY=False`), Mem0's config directory under `data/`, no remote notice fetch.
- **Alternatives:** Zep / Graphiti (needs a graph database server), Letta (an agent framework whose agent manages memory itself, closer to B3); left for later. Mem0 with its default OpenAI models would mix model differences into the comparison.
- **Consequences:** Qwen's JSON mode sometimes returns a bare list of memories without the `{"memory": …}` wrapper, and Mem0 then treats the whole session as having nothing to remember; OpenAI's JSON mode never returns a top-level array, so the adapter adds the wrapper (8 of about 5,400 extractions) and the 7 affected spaces were rebuilt. Results: 80.5% on 200 LoCoMo questions and 83.0% on 100 LongMemEval questions, against CueMem L1's 81.0% and 92.0% (the two LongMemEval numbers were measured with the session-ID leak; after the fix the same configuration gives L1 90.0% and Mem0 79.0%, see D-067 and the Chinese [leak ablation](../results/n12-leak/README.md)), with CueMem's nominal write cost at 34–44% of Mem0's. Mem0's write cost is measured separately (each space's `build.json`, not counted in answer usage) and capped at $10; a failing session is skipped and counted, and a history that fails to build is not retried for every question.
- **After N12 (D-068):** `scripts/reproduce.py` also pins mem0ai 2.2.1 (it used to install Mem0 without a version).

## D-056 Deployment: a clear error when the local store is locked; payload indexes on the server; a migration command; an optional access token

- **Status:** adopted (N11).
- **Context:** Qdrant local mode allows one process per directory; recording the demo with `cuemem serve` running and a CLI command at the same time gave only the library's internal `RuntimeError`. The HTTP API had no authentication. No load test had been done.
- **Decision:**
  1. A locked local directory raises `StorageLockedError`, which says which process to stop or how to switch to a server (`CUEMEM_VECTOR_STORE__URL`).
  2. In server mode, new collections get payload indexes on `kind` (keyword) and `ts` (float), the fields retrieval filters on.
  3. `cuemem migrate-vectors --to URL` copies every collection (vectors and payloads) from local mode to a server. Writes are idempotent, so memories already in the database are never written to the vector store again: switching without migrating gives an empty vector store.
  4. With `CUEMEM_API_TOKEN` set, everything under `/api` except the health check requires a Bearer token (constant-time comparison), and the UI asks to sign in (any username, the token as password). Without it there is no authentication, and `cuemem serve` warns when listening beyond localhost.
  5. An offline latency benchmark (`scripts/n11_latency.py`): synthetic spaces from 500 to 20,000 rounds, local mode and server, plus a single-process concurrency test.
- **Consequences:**
  - The local work on the read path grows linearly in local mode (about 1.08 s at 46k retrieval units) and is 91 ms with the server; throughput rises from 8.7 to 55 reads per second.
  - The load test showed that the synchronous Qdrant client cannot be shared by threads ("Bad file descriptor" in server mode), so both modes keep the global lock.
  - Limits: one shared token and no per-user isolation of memory spaces; several instances would need a server database instead of embedded SQLite.
- **After N12 (D-068):** reverses point 4's "warns when there is no token".
  - `cuemem serve` and `cuemem proxy` refuse to start (exit code 2) on a non-loopback address without `CUEMEM_API_TOKEN`; running without authentication on a trusted network needs an explicit `--insecure-no-auth`. The token is checked before the memory system is opened, so a refused start does not resume leftover extraction either.
  - Docker: compose stops with an error when the token is missing or empty, and publishes the port on the host's 127.0.0.1 only. The image listens on 0.0.0.0, so it needs a token, or a CMD override with `--insecure-no-auth`. Before, following the README's `docker compose up -d` left export, forget, import and the `/v1` proxy open to the whole network without authentication, and the empty token compose injected also hid a token written in /data/.env. The image includes the `local` extra (local embeddings) by default; CI builds it and checks its health endpoint.
  - Cross-origin browser requests (`/api/`, `/v1/`, `/gradio_api/`) get 403; on a loopback address the Host header is checked (against DNS rebinding). Behind a reverse proxy or under another hostname, set `CUEMEM_ALLOWED_HOSTS`.
  - The limit stays: there is still one shared token.

## D-057 The gate on real traffic: first messages of real chatbot conversations

- **Status:** adopted (N11; answers "the mixed stream behind the controller's result is synthetic").
- **Context:** the controller's "whether to recall" result (41% cheaper than always retrieving in a mixed stream) rests on synthetic non-memory queries: Dolly instructions and generated same-topic questions. Real messages are more colloquial and varied and often use the first person ("help me fix my résumé"), which may fool the gate.
- **Decision:**
  1. Use WildChat-1M (real conversations between users and a chatbot, ODC-BY), first shard: the **first message** of English, non-toxic, non-redacted conversations. The start of a conversation cannot refer to an earlier one, so none of them needs long-term memory. Same filter as Dolly (10–300 characters), deduplicated by text: 14,319 candidates, 1,750 sampled (as many as the test memory questions).
  2. Pair each with a test memory space and run the probe, as for N9's mixed stream; apply the frozen v1 and v2 controllers offline: false recall, gate AUC, and the cost with these messages replacing the synthetic non-memory queries. The zero-shot LLM router (B6z: qwen3.8-flash looking at the message alone) also judges the same messages, and with the hybrid (B6h) serves as a comparison.
  3. Only conversation hashes are committed (`evals/mixed/wildchat_test.json`); `cuemem data download wildchat` downloads the shard only when named (datasets gained an `optional` flag), from a URL pinned to the release commit.
- **Limits:** messages after the first are the ones more likely to need memory, and those are unlabelled; WildChat's users (of a free GPT web app) are not any particular product's users. The check only answers whether real messages that don't need long-term memory are wrongly sent to retrieval.

## D-058 Controller v3: in a space with photos, look at them only when the question is about one

- **Status:** adopted (after N11; finds and fixes "the agent answers the previous question again").
- **Context:** the agent now and then answered the previous question again: once in N11's script replay (S3·4) and once while re-recording the README GIF. It was written off as a random reader error.
  - **Reproduction:** in a sandbox copy with the cache off, the first 4 messages of the demo's third chat were run 6 times. The error came back twice: once the reply answered only the previous question, once it answered both.
  - **What the request showed:** the user asked "How has my shoe budget changed since we first talked?", but the controller chose L1 + photos and attached the photo of a friend's shoes just before the user's message. The previous message happened to be "find shoes that look like the pair I showed you", so the model ran the image search again.
  - **Root cause, controller v2:** in a space with photos it reads an accuracy table built from MemLens. Nearly every MemLens question is about an image, so looking always scores better. v2 therefore attached photos to 96–98% of questions in such spaces; in the demo, even "Hi Cue, I'm back!" got one.
- **Decision:**
  1. A threshold for re-perception. In a space with photos, looking at photos is considered only if the question has a visual cue (`q_visual`, `q_scene`) or the best image-caption match `p_caption_top1` is at least τ. Otherwise the controller chooses among the actions without photos, and the level is still chosen by accuracy and cost. The gate, the type model, the accuracy tables and λ are unchanged.
  2. τ is chosen on the development set only, under D-052's protocol: the largest τ whose accuracy on MemLens 32K dev100 is no lower than v2's. That is 0.64. The test sets are only replayed post hoc.
  3. The frozen v2 is kept as `controller_model_v2.json` and is pinned for reproducing N11; the default becomes v3 (`scripts/n11_see_gate.py` evaluates and exports it).
- **Alternatives:**
  - Visual cues only, ignoring caption similarity: only about a quarter of MemLens questions have visual words, and development accuracy drops 3 points (60.0%).
  - A prompt fix in the agent ("earlier messages have already been answered"): treats the symptom. The photo would still pull unrelated questions off course, and looking costs extra.
- **Consequences:**
  - **Development set:** for every τ from 0.30 to 0.64, accuracy stays at 62–63%, within one question; v2 scores 63.0%. The share of questions with photos falls from 98% to 31%. A hundred questions cannot separate differences this small: the threshold does not hurt accuracy, but it is not shown to help.
  - **Post-hoc test replay:**
    - MemLens 32K: 62.6% → 61.5% (+9 / −11, n.s.).
    - MemLens 128K: 61.0% → 62.1% (+9 / −7, n.s.).
    - Questions with photos: 96–97% → 31–33%, with 16% lower cost per 1,000 questions.
    - Text datasets: unchanged, question by question.
  - **Demo:**
    - Re-running the same 4 messages 8 times, the fourth message never got a photo and was answered correctly every time.
    - A full replay of the script passes 14 / 14.
    - The levels changed: the greeting and the size question get L1, the look-alike search gets L2 + photos, and the budget question gets L1. Expected levels in the script are reported, not checked.
  - **Limits:**
    - τ is an absolute similarity and must be re-chosen for a new embedding model.
    - A question about a photo that has no visual words and does not match the caption misses the photo. Its caption still comes back as text with the recalled memories.
- **After N12 (D-068):** `scripts/n11_see_gate.py` has `--check` and checks and exports before it writes see_gate.json. The reproduction script runs it with `--no-export --check`, so it only compares the result with the committed `controller_model.json` and never overwrites it.

## D-059 Gaps left by forgetting are marked when a session is re-extracted; a pronoun next to a gap is not pinned on someone else

- **Status:** adopted (after N11; resolves the limit recorded in D-051's revision).
- **Context:** after forgetting deletes the source round, the session is re-extracted, and "The trekking poles are perfect, I'll get those for her" has lost its referent.
  - **Why the model guesses:** the extraction prompt asks for names instead of pronouns, so the model picks someone from what is left. Earlier in the same chat the assistant had mentioned Lena ("a match to Lena's look"), so the fact became "buys trekking poles for Lena".
  - **Reproduction:** the demo's second chat was rebuilt as it looks after forgetting and extracted 12 times with the cache off. All 12 named Lena.
- **Decision:**
  1. Message numbers are never reused (D-021), so a gap left by forgetting is where the numbering jumps. When a session is rendered for extraction, each gap is marked `[deleted]`, and a note tells the model:
     - do not reconstruct what was deleted;
     - a pronoun next to a gap usually pointed into the deleted message, so do not resolve it to someone or something named elsewhere: write "someone", or leave the fact out;
     - this overrides the rule to use names instead of pronouns.
  2. Only sessions with gaps are rendered this way. Every other request is byte-for-byte unchanged, so the pipeline version is not bumped and all cached extractions stay valid.
     - Checked: none of the 42,071 sessions in the evaluation spaces has a gap.
     - A long session split into windows is judged as a whole: later windows don't start at 0, and that is not a gap.
- **Alternatives:**
  - Change the system prompt for every session ("don't guess an unclear referent"): this would invalidate about 4.8k cached extractions in the evaluation spaces, the rebuild would cost money, and it might move the evaluation numbers.
  - Also delete the messages around a gap: this deletes more, including things the user did not ask to forget.
- **Consequences:**
  - On the same rebuilt chat, the first wording of the note gave "someone" 7 times out of 12. Naming "a pronoun next to a gap" and stating that the note overrides the names rule gave "someone" 8 out of 8. The promotion, the $180 budget and the switch to EU 40 were still extracted.
  - In a sandbox built from the original script ("for her"), forgetting only the birthday fact, as in the UI, deleted 3 messages and took facts from 15 to 12. The poles fact was re-extracted as "Maya purchased trekking poles for someone else."
  - The demo script keeps "for Ana", since the README GIF was recorded with it.

## D-060 Switching model providers: config shipped in the package, a data directory, presets and an OpenAI-compatible embedding API

- **Status:** adopted (N12).
- **Context:** compared with Mem0, Graphiti and Cognee ([comparison](landscape.md)), CueMem was hard to pick up and use.
  - The default config lived in the repository's `configs/`, and relative paths resolved against the repository root. A pip-installed copy could not find its config and would write data next to site-packages.
  - The only models were DeepSeek and Qwen, and embeddings only went through Qwen's native API. There was no OpenAI and no local model that runs without a key.
  - Every request carried `temperature` and `max_tokens`, which OpenAI's reasoning models (the GPT-5 family) reject.
- **Decision:**
  1. The default config moves to `src/cuemem/configs/` and ships with the package. The data directory is `CUEMEM_HOME` if set; in a source checkout it is still the repository root (data/ and runs/ as before); an installed copy uses `~/.cuemem`. `.env` lives in the data directory too.
  2. Presets (`configs/presets/*.yaml`) are deep-merged over the default config:
     - `openai`: gpt-5.4-mini + text-embedding-3-small;
     - `ollama`: llama3.1:8b + nomic-embed-text, the same local setup as Mem0's docs, with no key.
     They are chosen with `CUEMEM_PRESET`, `cuemem --preset` or `Memory(preset=...)`. The default providers stay in the config; `configure` asks only for the keys that are used, and `doctor` marks the unused providers.
  3. `embedding.backend: openai` calls an OpenAI-compatible `/embeddings` at the provider's base_url. It returns dense vectors only; keyword retrieval was already done by the local BM25 (D-026). A dimension that does not match the config is a clear error.
  4. Optional components:
     - the model catalogue can declare parameter quirks: `fixed_temperature` (send no temperature) and `max_tokens_param` (`max_completion_tokens`);
     - a provider's `api_key_env` may be empty (local servers);
     - reranking and image embeddings can be turned off (`rerank.provider: null`, `multimodal_embedding.backend: none`);
     - a reader that cannot see images gets the text descriptions only, without the original photos.
  5. A Docker image (data volume `/data`, non-root) and a compose file that adds a Qdrant server.
- **Alternatives:**
  - LiteLLM, for hundreds of models at once: another dependency and another set of parameter conventions. CueMem's model calls already go through one OpenAI-compatible client; what was missing was the embedding API and a few parameter quirks.
  - Presets as Python dicts: YAML matches the default config, so users can write their own preset the same way (`--preset path/to/my.yaml`).
- **Consequences:**
  - Requests under the default config are byte-for-byte unchanged (including parameter order), so existing LLM and embedding caches still hit; `tests/test_providers.py` checks this.
  - One behaviour change: with an explicit config file (`CUEMEM_CONFIG`), relative paths used to resolve against the config file's grandparent directory; they now resolve against the data directory. No script in the repository relied on the old behaviour.
  - A built wheel was checked outside the repository: data goes to `~/.cuemem` and presets apply. The Docker image was built and run locally: the SDK inside the container, HTTP import and recall, and the Qdrant server under compose all worked.
  - **Limits:**
    - The OpenAI and Ollama presets are covered by offline tests only and have not been evaluated on real models.
    - OpenAI prices are taken from OpenRouter's listing, because the development environment could not reach OpenAI's site.
    - The consolidation similarity thresholds were tuned on Qwen embeddings (N3) and may be too loose or too strict for another embedding model. The effect is limited: a single-valued fact that meets the same predicate always goes to the LLM.
    - A data directory cannot mix embedding models: the database records the model it was built with and raises ConfigMismatchError otherwise.
- **After N12 (D-068):** the audit found that the pip-installed path did not actually work: `cuemem serve` defaulted to the shopping scenario, whose catalogue exists only in the repository, and crashed outside it.
  - Without `--scenario`, a source checkout still gets the shopping assistant and an installed copy gets the plain assistant; `cuemem demo` uses the catalog.json next to its script.
  - serve, chat, demo and proxy print a one-line error ("错误：…") and exit with code 1 on configuration errors such as a missing key, instead of a traceback. The commands in error messages match the installation (`cuemem …` or `uv run cuemem …`); with only OPENAI_API_KEY set they suggest the openai preset; the embedding-model mismatch error points to a fresh data directory instead of telling the user to delete data/.
  - The wheel is now checked outside the repository by CI's package job (`cuemem --help`, `cuemem doctor`, `scripts/smoke_wheel.py`, a serve health check). The Docker image includes the `local` extra by default (`--build-arg EXTRAS=` builds a slim one), and CI's docker job builds it and checks its health endpoint.

## D-061 A Python SDK in Mem0's shape, with CueMem's semantics

- **Status:** adopted (N12).
- **Context:** Mem0's `Memory().add(...)` / `.search(...)` has become the common interface for this kind of library. CueMem only had `MemorySystem`, an internal facade: you had to build a `Conversation` and `SessionIn` and understand spaces, consolidation and levels first.
- **Decision:** `from cuemem import Memory` (plus `AsyncMemory` and `Memory.from_config`).
  1. Method names and result shape follow Mem0: `add / search / get_all / get / history / delete / delete_all`, returning `{"results": [...]}` with a `memory` text on each item. Mem0 v3's `filters={"user_id": ...}` and `top_k=` are accepted too.
  2. **Scope:** `user_id` and `agent_id` together name a memory space (`alice`, `agent/shop`, `shop/alice`); `run_id` is one session within it. `add` takes OpenAI-format messages; images in content blocks may be data URLs or local paths.
  3. **The semantics stay CueMem's:**
     - `add` writes online first (no LLM call, searchable right away) and extracts in the background. `wait=True` extracts synchronously and returns Mem0-style ADD / UPDATE events; an UPDATE carries the superseded versions and the reason (the world changed / it was a mistake). `infer=False` only logs.
     - Facts are never rewritten in place. `history()` returns the whole version chain; `get_all` returns current facts by default, and `valid_at` / `known_at` replay either timeline.
     - `search` goes through the controller and returns the level and reasons in English. L0 means the message needs no recall, so only the profile is returned.
     - `delete` is the right to be forgotten: it deletes the conversation a fact came from and the uncited restatements, then rebuilds. `delete_all` removes the whole space.
  4. The orchestration of forgetting moves from the service layer to `MemorySystem.forget_memories`, shared by the SDK, the HTTP API and MCP.
- **Alternative:** copying Mem0's `update(memory_id, text)`. Rewriting a fact in place bypasses the raw log, and derived memory could no longer be rebuilt from it (D-021). To change a fact, say a correction as a user would, and reconciliation creates the new version.
- **Consequences:** `tests/test_sdk.py` covers add events, the version chain, replaying either timeline, scope isolation, image messages, logging first and consolidating later, an older version restored after deletion, background extraction and the async variant. Results and errors are in English, like the HTTP API, MCP and UI; the CLI stays in Chinese.
- **After N12 (D-068):**
  - **Thread safety:** concurrent `add` calls with the same run_id used to drop messages silently. Turn numbers are now allocated under a per-session lock, and a message that was not written raises `cuemem.errors.ConcurrentWriteError`.
  - **Extract a conversation once:** `open_run / end_run` and `with m.run(...)` (`async with` on AsyncMemory). Calling `add` once per turn with the same run_id still works, but re-extracts the run and replays derived memory every time (D-045). A run left open by a crashed process is handed to extraction the first time the next process touches that user.
  - **run_id scope:** a run_id in `filters` used to be ignored silently. It, and a new `run_id=` keyword, now narrow `search` and `get_all` to that session (a post-filter, so results can be fewer than the limit; the profile stays space-wide), and `delete_all(run_id=)` forgets only that run.
  - **Peak hours:** the default is `peak_policy="warn"` (D-024).
  - **Times:** times with a time zone are stored as local time; generated session IDs have a new format (same prefixes), and created_at / said_on carry microseconds.
  - **A correction to the consequences:** "Results and errors are in English, like the HTTP API, MCP and UI" is only partly true. The SDK's own errors are English, but most exceptions raised by the core and the HTTP API's error messages are Chinese (left for later; see D-068).

## D-062 An OpenAI-compatible memory proxy: the controller decides first, evidence goes before the last user message

- **Status:** adopted (N12).
- **Context:** Mem0's proxy, Supermemory's Memory Router and Memobase's OpenAI patch give an app memory "by changing the base_url", and they retrieve on every turn. CueMem's central finding is that every turn does not need retrieval: in a mixed stream the controller costs 41% less than always retrieving (46% on real traffic, D-057). A proxy is the most direct way to show it.
- **Decision:** `cuemem proxy` runs on its own (port 8787 by default), and `cuemem serve` mounts the same `/v1/chat/completions` and `/v1/models`.
  1. **Identity:** the request's `user` field or the `X-CueMem-User` header (optionally `X-CueMem-Agent`). Without either and without a default user, the request is forwarded as is and memory is not touched.
  2. **Recall:** the controller runs on the last user message. Rounds of the current session that are already in memory, and still verbatim in the request, are excluded from retrieval.
  3. **Injection:** the same evidence block the reader uses (profile, facts with validity, excerpts in time order), wrapped in `<memory>` and put at the start of the last user message, where Mem0's proxy puts it. The system prompt and earlier messages are untouched, so provider prefix caching keeps working. L0 adds the profile only; if nothing was found, nothing is added.
  4. **Forwarding:**
     - `model` is a configured alias or model ID; otherwise the agent role's model is used.
     - Provider-specific parameters (turning thinking off, …) and reasoning-model quirks come from the model catalogue.
     - Streaming and non-streaming are both supported, and responses are returned as is.
     - The decision goes in the `X-CueMem-Level / -Reasons / -Recalled / -Session` response headers, not the body, so strict clients are unaffected.
     - Upstream errors are returned as is and nothing is written.
  5. **Write-back:** the user's message (with data-URL and local images) and the answer are written online. Nothing is extracted while a session is open; the whole session is extracted when it ends, as in the agent (D-045). Session boundaries come from `X-CueMem-Session`. Without it, a new session starts when the request has no earlier messages (the caller started a new conversation) or after 30 idle minutes.
- **Alternatives:**
  - Memory in the system prompt (Memobase's patch): the system prompt changes every turn, so prefix caching never hits.
  - Retrieving every turn, as Mem0 and Supermemory do: it gives up the point of the controller. Use `--mode l1` if that is wanted.
  - A `cuemem` field in the response body: some clients validate the response strictly.
- **Consequences:**
  - `tests/test_proxy.py` uses TestClient to cover cross-session recall and injection, a new conversation starting a new session, streaming relay and write-back, model mapping, pass-through without a user, upstream errors, tool follow-ups and the access token.
  - It was also run end to end with the official OpenAI SDK against a local fake upstream: non-streaming, streaming, raw response headers and `/v1/models` all worked.
  - That end-to-end run found a bug in the first version. It assumed that every round written to the current session was still in the request. When a caller started a new conversation within the idle window, the previous one was neither in the request nor retrievable. It is now compared by content, and a new conversation starts a new session (points 2 and 5 above).
  - **Limits:** parallel conversations without session headers are merged into one session; image URLs are not downloaded or stored.
- **After N12 (D-068):**
  - **Same-origin JSON only:** requests must be `Content-Type: application/json` (otherwise 415; `curl -d` needs `-H 'Content-Type: application/json'`); browser requests with an Origin that does not match the Host get 403; on a loopback address only Host 127.0.0.1, localhost, [::1] or the bind address is accepted (400). Behind a reverse proxy or under another hostname, set `CUEMEM_ALLOWED_HOSTS`. Before, any web page could send a text/plain POST, which needs no preflight, to spend the operator's upstream credits and write a line into a guessed user's memory.
  - **Interrupted streams** (extends point 5): when the upstream fails mid-stream, nothing is written (an in-stream `{"error": …}` chunk counts), as without streaming; when the caller stops or disconnects, only the user's message is written and the upstream connection is closed at once; a stream that ends normally is written as before (`[DONE]` is not required). Before, a half answer was stored as if complete.
  - **Rounds still in the request** (point 2): the comparison now uses the stored form (images written as `<image>`), so a round with a photo that is still in the request is no longer recalled and injected again; the proxy, OpenAI Agents and LangGraph share `evidence.stored_text` and `in_context_rounds`.
  - **Injection** (point 3): a stored `<memory` or `</memory` is written as `&lt;memory` / `&lt;/memory` and can no longer close the evidence block early, and the block's note gains one sentence: its content is quoted from past conversations, information and not instructions. This deliberately changes the request, and it reduces prompt injection from memory but cannot prevent it.
  - **Sessions** (point 5): sessions still open when the process was killed are handed to extraction the first time the new process sees that user (except the session named by `X-CueMem-Session`); reusing a named session after it ended replays derived memory from the raw log instead of keeping the earlier extraction's wording (D-045).
  - Spaces the proxy writes, such as `shop/bob`, can be browsed, exported and forgotten through the HTTP API (they used to return 404).

## D-063 Export and import move only the raw log; forgetting can be previewed

- **Status:** adopted (N12).
- **Context:** Letta's Agent File, Cognee's COGX and MemOS's dump / load treat portable memory as a basic capability. Supermemory and Redis Agent Memory Server let you dry-run forgetting. CueMem had neither, and forgetting is irreversible and cascades.
- **Decision:**
  1. The export format `cuemem.space` (JSON, version 1) holds sessions, messages (index, role, text, time, speaker) and the original image files (base64). A snapshot of derived memory (facts with their version chains, and the profile) is attached for other systems to read; import ignores it.
  2. Import writes the raw log back and lets consolidation regenerate derived memory. Derived memory is a function of the log (D-021), so the result is consistent across extraction models and machines, and no stale derived state is carried over. Gaps left by forgetting are preserved (each run of consecutive indices is written separately) and are marked again on re-extraction (D-059). Import is idempotent.
  3. Entry points: `cuemem export / import`; the SDK's `export()` / `load()`; HTTP `GET /api/spaces/{space}/export` and `POST /api/spaces/import`.
  4. Forgetting dry run: `cuemem forget --dry-run` (which now also takes `--fact`), the SDK's `delete(id, dry_run=True)`, and `dry_run` in the HTTP API. The preview shares the target computation with real forgetting (following citations, finding restatements by content), lists the messages that would be deleted and the facts that would go with them, and deletes nothing.
- **Alternative:** exporting derived memory and writing it straight in on import. Faster, but fact IDs, entity merges and the profile depend on the extraction model and the reconciliation history, and would no longer match the target's log.
- **Consequences:**
  - Tests cover the round trip (index gaps, image hashes and bytes, times, re-extraction, idempotence), format validation and the CLI.
  - **Not exported:** the provenance table of which memories each reply cited, since fact IDs are regenerated on import. On an imported space, forgetting therefore cannot follow citations; finding restatements by content (D-051) still works.
  - The preview lacks the "regrew after rebuilding" step (D-051's regrown facts, which needs a real rebuild); the output says so.
- **After N12 (D-068):**
  - **Import validation:** image paths used to be built from the file's own sha256 and mime fields, which could write outside the temporary directory and create .py or .sqlite files. Images are now strictly base64-decoded and verified with PIL, named by the recomputed hash, and a declared hash that does not match is an error; if any image fails, the whole import fails and nothing is written (HTTP 409).
  - **Spaces with an agent:** `GET /api/spaces/{space}/export` and the other space routes returned 404 for spaces such as `bot/alice` and `shop/bob`; every `/api/spaces/{space}/…` route now accepts them, with "/" raw or written as %2F.
  - Times with a time zone in an import file are converted to local time.

## D-064 Extra extraction instructions (custom instructions)

- **Status:** adopted (N12).
- **Context:** different apps need different things remembered: a shopping assistant needs sizes, budgets and brands; a companion app needs moods and relationships. Mem0 has `custom_instructions`; Graphiti has `custom_extraction_instructions` and custom entity types. CueMem's extraction prompt was fixed.
- **Decision:** `consolidation.instructions` (`Memory(instructions=...)` in the SDK) is appended to the end of the extraction system prompt as "additional instructions from the application, which take precedence when they conflict".
  - **When empty:** the prompt is byte-for-byte unchanged and so is the consolidation fingerprint (the field is left out of it), so cached extractions and consolidated spaces are unaffected.
  - **When set:** the fingerprint changes, and consolidated spaces rebuild their derived memory on the next consolidation.
- **Alternative:** custom entity and fact types (Graphiti). That changes the output schema, the reconciliation rules and the predicate registry; plain-text instructions cover the common need first.
- **Consequences:** tests cover the prompt, the SDK parameter and the fingerprint change. The effect of instructions on extraction quality has not been evaluated.

## D-065 Same-protocol comparison with more external baselines

- **Status:** adopted (N12).
- **Context:**
  - Before N12, Mem0 was the only external system run inside the same evaluation (D-055). For other comparable projects there were only the scores each project reports itself, with different readers, judges, prompts and question sets; they cannot be compared directly with this project's numbers ([comparison](landscape.md) §4).
  - The question: with the same models and the same reader and judge, changing only where the memory comes from, is CueMem's write and read path better or worse than comparable projects, and what does writing cost?
  - Four open-source projects that run locally were added: Graphiti (the open core of Zep, a temporal knowledge graph), LangMem (LangChain's memory library), Cognee (knowledge graph + vectors) and A-MEM (Zettelkasten-style notes; the paper comes with LoCoMo evaluation code). Letta is still left out: its agent manages memory itself, which is closer to B3 (D-055).
- **Decision:**
  1. **Eval gateway** (`evals/gateway.py`): CueMem's own LLM client and embedder behind an OpenAI-compatible chat and embeddings API. Each external system keeps its own client, with `base_url` pointing at the gateway.
     - Every chat request goes to the extractor role's model, and every embedding request to the same embedder; one ledger, one cache.
     - A bucket name in the URL keeps usage separate per system and per conversation history, and the usage is recorded in build.json.
     - External systems get the OpenAI semantics they assume: a request with tools and no `parallel_tool_calls` is sent with true (otherwise Qwen returns one tool call at a time); buckets of the retrieval stage (`<system>.search.<space>`) embed in query mode; an upstream rate limit comes back as 429 with Retry-After and is counted; an unparsable JSON-mode reply is not replayed from the cache when the same request comes again.
  2. **Adapters follow the authors' evaluation code** (`evals/baselines/`), or the documented usage where no evaluation code is published. Answering and judging stay outside the systems: the evidence each one retrieves is labelled with its source session's time and goes to the same reader with the same prompt.
     - **MG** Graphiti (graphiti-core 0.30.2, embedded Kuzu): follows the Zep paper's evaluation code; one episode per message; retrieves 20 facts and 20 entities.
     - **ML** LangMem 0.0.30: the background memory manager from its docs, run once per session. The "hot path" the Mem0 paper used for LangMem is not copied: feeding an agent message by message makes calls grow with the square of the conversation length, and the memories are the agent's rewritten prose, with no dates.
     - **MC** Cognee 1.6.4: follows Cognee's own setup on the long-conversation benchmark BEAM; one chunk per turn, HYBRID_COMPLETION retrieval; all 20 chunks, 20 entities and 5 facts go to the reader.
     - **MA** A-MEM (agiresearch/A-mem@ceffb86): follows the paper's LoCoMo evaluation code; one note per message, with memory evolution; retrieval with k = 10.
     - **M0** Mem0 2.2.1 keeps N11's adapter (D-055). This version is the add-only V3 algorithm.

     Every departure from a system's default usage, and the reason for it, is written in that module's docstring.
  3. **Two protocols.** Within a protocol every system (CueMem, and the calls inside the external systems) uses the same models:
     - **pds** (DeepSeek only, `CUEMEM_PRESET=deepseek`): every role is deepseek-flash, the judge too; embeddings are local BAAI/bge-small-en-v1.5 (fastembed, 384-d, no query instruction); no reranker and no fallback model. Builds in data/eval-spaces.
     - **pq** (Qwen only, `CUEMEM_PRESET=qwen`, a preset added in N12): the extractor, planner, reader and other roles use qwen3.8-flash, the judge qwen3.8-max; embeddings qwen3.7-text-embedding (1024-d, separate query and document modes); reranking qwen3-rerank; consolidation and gateway fallback qwen3.7-plus. Builds in data/eval-spaces-qwen (`CUEMEM_PATHS__EVAL_SPACES`).
     - Why pq exists: the DeepSeek balance ran out halfway through (CNY 0.08 left on 2026-10-15), so a Qwen-only protocol was added to finish the comparison. The account was later topped up and all the other systems were also run under pds. Both protocols are reported: how much the same systems change under another provider's models is part of the result.
     - Neither protocol meets D-008: in pds the reader judges itself, and in pq the judge is from the reader's model family. In N11's second-judge experiment, deepseek-flash and qwen3.8-max agreed on 94.7–97.9% of judgements.
  4. **Rules:**
     - within a protocol: the same extraction model and embeddings (through the gateway), the same reader and reader prompt, and the same judge;
     - each system uses its own write and read pipeline, and its algorithm is not changed (for the changes needed just to run, see point 7);
     - one run per configuration; paired exact sign tests on the same questions;
     - write usage of the external systems (MG, ML, MC, MA) is read from the gateway buckets, and that of M0 and CueMem from the trace recorded while writing; both are kept apart from answer usage, and comparisons use nominal cost (D-010).
  5. **The four new external systems run LoCoMo test200r only** (Mem0 ran LongMemEval test100r in N11 and again in N12's leak ablation; see D-067): 200 stratified non-adversarial questions over the last 7 conversations (conv-42, 43, 44, 47, 48, 49, 50; 4,431 turns, 202 sessions). Seven conversations need seven builds. In LongMemEval each question has its own history, so test100r needs 100 builds:
     - N11's Mem0 build cost $9.76 nominal on LongMemEval test100r and $0.70 on LoCoMo;
     - Graphiti writes one episode per message; its module docstring estimates about 49,000 messages for test100r, several hundred dollars and more than a day of writing;
     - A-MEM makes two LLM calls per message and must make them one message at a time: about 15 s per message under Qwen.

     LangMem's and Cognee's LongMemEval builds were not estimated separately; they also run LoCoMo only, so the four new systems are compared on the same questions. CueMem's own methods are still evaluated on LongMemEval; the numbers re-run on test100r after the session-ID fix are in D-067 (test400 was not re-run).
  6. **Additions in the adapters so the systems run on these two providers** (every occurrence is counted in build.json):
     - A reply that is JSON but does not match the schema is resampled (MG, MA). OpenAI's strict structured output guarantees the schema; the JSON modes of DeepSeek and Qwen do not. A resample lowers the reply limit slightly so that the request lands on a new cache key; otherwise the cache would return the same bad reply.
     - Wrong refusals by content moderation (the refused messages are ordinary) are handled differently per system:
       - MG skips and counts refused messages; only above max(2, 5%) messages is the build rejected;
       - ML skips and counts refused sessions, which count towards the same limit as sessions that lost their writes (next item);
       - MA skips and counts refused calls, as A-MEM itself does (the note is kept, without analysis or evolution); only above one half is the build rejected;
       - M0 (N11's adapter) skips and counts every session that still fails after the fallback model, with no limit.
     - LangMem's lost writes are counted as they are (an unreadable PatchDoc patch also counts as lost). The build check used to reject a build when refused and lost sessions together exceeded max(2, 10%) of sessions; under pq it rejected 4 of 7 conversations. Full runs in both protocols use 25% (`CUEMEM_LANGMEM_MAX_LOST_SHARE=0.25`; the default is still 10%), and the losses are still recorded.
     - A failed Cognee session is deleted and written once more; if it fails again it is skipped and counted, and only above one half is the build rejected.
  7. **Departures from default usage needed just to run** (these two change Graphiti's retrieval):
     - Graphiti's BFS depth is 1 (default 3: Kuzu's variable-length path query used more than 12 GB at depth 3);
     - Graphiti's built-in rerankers cannot be used: under pq, qwen3-rerank is wrapped as its cross-encoder; pds has no reranker, so fact retrieval falls back to the default of Graphiti's basic search, an RRF of BM25 and vectors.
  8. **Deliberately not tuned:**
     - Graphiti's 0.6 cosine threshold: under pq the vector channel is not empty, with a median of 8 fact candidates per question;
     - LangMem's prompts and defaults;
     - A-MEM's k: the main result uses the paper's main setting of 10. The authors use 40 for GPT-4o-class models (the same tier as the models here), and their README says k must be tuned to reproduce the paper's numbers. 10 was chosen so that the amount of evidence stays comparable with the other methods (in a deepseek-preset pilot on conv-42, k = 10 gave about 9k tokens on average and k = 40 about 27k, against about 4k for CueMem L1). This may put A-MEM at a disadvantage; the sensitivity run with k = 40 (`CUEMEM_AMEM_K=40`) is under consequences;
     - Cognee's evidence for the reader is not capped, as in its BEAM report;
     - every system's extraction prompts, CueMem's included.
- **Alternatives:**
  - Quoting each project's own scores: different readers, judges and prompts make a difference of a point or two meaningless (landscape §4).
  - Tuning each system (thresholds, prompts, k): it could only be done on these test questions, which means using the test set to make choices (D-048), and tuning every system to its best multiplies the work and the cost. The price of not tuning is CueMem's home advantage, recorded under consequences.
  - Running the external systems on LongMemEval too: see point 5 for cost and time.
- **Consequences:**
  - Results (LoCoMo test200r, 200 questions, accuracy %):

  | Method | pds | pq |
  |---|---|---|
  | B1 full context | 84.5 | 80.0 |
  | B2 naive RAG (dense top-10) | 59.5 | 64.0 |
  | CueMem L1 (B4-nobm25) | 82.5 | 81.5 |
  | CueMem L2 (B5) | 87.0 | 85.0 |
  | CueMem controller (B6) | 83.5 | 83.5 |
  | M0 Mem0 2.2.1 | 81.0 | 77.5 |
  | MG Graphiti 0.30.2 | 59.0 | 41.5 |
  | ML LangMem 0.0.30 | 78.0 | 30.0 |
  | MC Cognee 1.6.4 | 77.5 | 74.0 |
  | MA A-MEM (k = 10) | 58.0 | 59.5 |
  | MA A-MEM, k = 40 (sensitivity) | 75.0 | 65.0 |

  - Paired tests (questions CueMem wins / loses):
    - pds: L2 beats M0 +17 / −5 (p = 0.017), ML +25 / −7 (p = 0.002), MC +27 / −8 (p = 0.002) and MG +62 / −6 (p < 0.001). L1 cannot be told apart from M0 (+14 / −11, p = 0.69), ML (+24 / −15, p = 0.20) or MC (+20 / −10, p = 0.10), and beats MG +55 / −8 (p < 0.001).
    - pq: L2 beats M0 +24 / −9 (p = 0.014) and MC +30 / −8 (p = 0.0005); L1 beats MC +25 / −10 (p = 0.017) and cannot be told apart from M0 (+19 / −11, p = 0.20). MG and ML fall far behind under pq (L1 vs MG +85 / −5, vs ML +107 / −4, p < 0.001).
    - MA (k = 10): L1 and L2 are significantly better in both protocols (pds L1 +56 / −7, L2 +64 / −6; pq L1 +52 / −8, L2 +58 / −7; all p < 0.001).
    - A-MEM sensitivity run with k = 40 (the authors' setting for GPT-4o-class models, reusing the same build): pds 75.0% (vs k = 10, +40 / −6, p < 0.001), pq 65.0% (+26 / −15, p = 0.12), with 29,550 / 16,323 input tokens per question. L1 and L2 are still significantly better: pds L1 +27 / −12 (p = 0.024), L2 +32 / −8 (p < 0.001); pq both p < 0.001. k = 10 does put A-MEM at a disadvantage, but with k = 40 it still does not catch up with L1.
    - In both protocols L2 is significantly above all five external systems, but it uses from just over two to nearly four times as many input tokens per question as M0, MG, ML and MC (for L2 this includes its planning and sufficiency-check calls). Mean input per question: under pds, L2 12,255 tokens, those four external systems 3,358–5,683 (answer cost $4.108 vs $1.092–1.787 per 1k questions) and A-MEM (k = 10) 11,247 ($3.475); under pq, 14,022 vs 3,611–6,689 (A-MEM 5,555). L1, with an input size similar to M0's and the others' (4,333 tokens under pds), cannot be told apart from M0, ML or MC in pds and is significantly above MG and MA. L2 and full context cannot be told apart (pds +12 / −7, p = 0.36; pq +19 / −9, p = 0.087).
  - **Model sensitivity:** the external systems' prompts and pipelines were mostly written against OpenAI models, and another provider's models can break them.
    - Graphiti 59.0 → 41.5. Over the 7 conversations, entities 1,253 → 462 and facts 3,426 → 1,774: with Graphiti's own prompts, qwen3.8-flash extracts few entities, and under pq each conversation logged 118–270 "Target entity not found" warnings (4–15 per conversation under pds; a fact whose endpoint matches no entity is dropped). Turn recall (any) under pq: Graphiti 54.5%, B2 84.5%, CueMem L1 95.0% (the methods pass different numbers of items to the reader, so only the rough ordering is meaningful). Under pq, 5 messages of conv-42 were refused by moderation and skipped (built before the qwen3.7-plus fallback was added; the other six conversations had 0). Under pds Graphiti is level with naive RAG (59.0 vs 59.5; under pds Graphiti has no reranker, see point 7).
    - LangMem 78.0 → 30.0. Qwen uses PatchDoc to merge new information into a few long memories: 705 → 123 memories; sessions that lost their writes 2 / 202 → 23 / 202, plus 5 sessions where Qwen explicitly returned an empty PatchDoc ("no change").
    - Cognee 77.5 → 74.0, with no failed, retried or dropped sessions in either protocol; Mem0 81.0 → 77.5; CueMem L1 82.5 → 81.5, L2 87.0 → 85.0.
    - A-MEM (k = 10) 58.0 → 59.5, about the same in both protocols (under pds 124 evolutions were abandoned because the reply ran past 4,096 tokens, under pq 4); with k = 40, 75.0 → 65.0.
  - **Home advantage:** CueMem's extraction prompts and thresholds were developed with qwen3.8-flash and Qwen embeddings (N3–N7), which is exactly the pq setup, so pq favours CueMem. In pds no system's prompts were tuned for deepseek-flash (CueMem's consolidation similarity thresholds were also tuned on Qwen embeddings), so the comparison is fairer. In pds, L2 is still significantly above the five external systems and L1 is level with M0, ML and MC; in pq, L1 is significantly above MC and MA and still level with M0.
  - **Write cost** (7 conversations; nominal cost / estimated spend, and LLM calls):

  | System | pds: nominal / spend | pds: calls | pq: nominal / spend | pq: calls |
  |---|---|---|---|---|
  | CueMem (shared by L1, L2, controller) | $0.529 / $0.2222 | 438 | $0.2376 / $0.1511 | 393 |
  | Mem0 | $1.3815 / $0.242 | 404 | $0.7122 / $0.3168 | 404 |
  | LangMem | $0.563 / $0.2386 | 211 | $0.2252 / $0.0635 | 207 |
  | Cognee | $2.845 / $0.7975 | 4,547 | $1.4659 / $1.1542 | 4,549 |
  | Graphiti | $9.9923 / $1.5084 | 14,240 | $4.5147 / $3.9077 | 12,092 |
  | A-MEM | $17.5673 / $7.8772 | 9,161 | $3.1362 / $2.8779 | 9,113 |

  Estimated spend is net of local cache hits (a conversation used in a pilot costs almost nothing), provider cache discounts and DeepSeek's off-peak half price (all pds runs were off-peak; Qwen has no off-peak price), so it depends on the order of runs and is not on the same basis in the two protocols; compare nominal cost (D-010). CueMem's nominal write cost is below Mem0's, Cognee's, Graphiti's and A-MEM's in both protocols; the most expensive build is A-MEM's under pds (deepseek-flash writes long evolution replies: 10.56M output tokens) and Graphiti's under pq. LangMem's is close to CueMem's, but under pq LangMem lost the writes of more than a tenth of the sessions.
  - Ledger (data/ledger.sqlite, excluding calls served from the local cache; as of 2026-10-15 16:30 UTC, after all runs had finished): Qwen since the key became available, $25.33 actual spend (after provider cache discounts; $32.32 nominal without discounts). DeepSeek since 2026-10-14 19:00 UTC (pds pilots and full runs, A-MEM's k = 40 run, the reader in the leak ablation), $19.01 actual spend (after cache discounts and the off-peak half price; $48.46 nominal without discounts); every DeepSeek run used `--peak wait` and ran off-peak only.
  - Full numbers (accuracy by question type, confidence intervals, answer cost, write calls and time, all paired tests) are in [pds-locomo.json](../results/n12-baselines/pds-locomo.json) and [pq-locomo.json](../results/n12-baselines/pq-locomo.json) (regenerate with `scripts/n12_baselines.py --protocol pds|pq --dataset locomo`).
  - **Limits:** one run per configuration; neither protocol's judge meets D-008; the four new external systems were evaluated only on LoCoMo's 7 conversations and 200 questions; the adapters are this project's code following the authors' evaluation code, not runs by the authors; under pds Graphiti has no reranker and fact retrieval uses RRF, which is not the recipe of Zep's evaluation; neither pds nor pq is the N9 / N11 protocol, so their numbers are not compared directly with N9 / N11; for CueMem's home advantage, see above.

## D-066 Framework integrations: OpenAI Agents SDK, LangGraph / LangChain, Claude's memory tool; one shared implementation of recalled evidence

- **Status:** adopted (N12).
- **Context:** Mem0, Graphiti, Supermemory and others integrate with a dozen frameworks each, and "how do I plug this into the agent framework I already use" is the first question a user asks. CueMem had the SDK, HTTP, MCP and the memory proxy. A framework integration has to settle two things: who decides whether to recall, and when a conversation is over and goes to extraction.
- **Decision:** three modules under `cuemem.integrations`, each an optional dependency (`--extra openai-agents / langgraph / anthropic`, together `integrations`); usage is in [integrations](integrations.md).
  1. **The controller decides first.** All three run the controller before every model call (L0 gives the profile only). The evidence block is the one the proxy and the reader use, placed at the start of the latest user message; the system prompt and earlier messages are untouched. The model can also search with recall / remember / forget tools, which have the same names and meaning as the MCP tools.
  2. **OpenAI Agents SDK:** `memory_input_filter` plugs into `call_model_input_filter`. `CueMemSession` implements the Session protocol: an inner session keeps the exact short-term history, and each question and answer becomes one CueMem round. With server-managed conversation state, `placement="instructions"` keeps the evidence out of what the server stores.
  3. **LangGraph / LangChain:** the LangChain 1.x middleware `CueMemMiddleware` recalls in wrap_model_call, changing only the request sent to the model, and writes back in after_agent, however the turn ends. The older `create_react_agent` uses `recall_hook` / `save_hook`. One thread is one session. On LangGraph Server the authenticated user is the identity, not an id the client sends.
  4. **Claude's memory tool** (`memory_20250818`): the `/memories` directory is backed by CueMem. `profile.md` and `facts.md` are read-only generated files. Files Claude creates live in a small side store; every change is also written to the raw log as a note, which consolidation turns into facts; deleting a file forgets what it contributed.
  5. **When a conversation ends:** on `end()` / `end_thread()`, leaving the context manager, 30 idle minutes, or `close_sessions()` / `close()`, the session goes to background extraction. Sessions a process did not hand over are handed over by the next process.
  6. **One implementation of recalled evidence.** The review found "recall → lay out evidence → inject" written once in the proxy and once in each integration, and the integrations imported private functions. It now lives in `cuemem/evidence.py`: `recall_evidence` / `inject_evidence` (automatic recall) and `recall_listing` / `format_listing` / `resolve_ids` (a listing for another agent, and forgetting by id). The proxy, the service layer, MCP and the three integrations all use it, with unchanged behaviour.
- **Alternatives:**
  - A LangGraph `BaseStore` backend: a Store is a generic key-value store whose reads and writes the app schedules, so there is no place for the controller. It could be added later as a read-only view of the facts.
  - Memory in the system prompt: it changes every turn, so prefix caching never hits (as in D-062).
  - Tools only, letting the model decide: that is Letta's approach and the opposite of CueMem's claim that the controller decides before the LLM is called. The tools stay as a way to search further.
- **Consequences:**
  - Each integration was implemented, then reviewed from two angles (faithfulness to the framework, correctness), and revised before merging. Real problems the reviews found: CI did not install the optional dependencies, so the integration tests were always skipped; a question and its answer were split into two rounds; cached evidence was reused across turns; LangChain's conversation summary was written back as the user's words; identity came from client-controlled configurable; the proxy and integrations read `file://` and local image paths. All are fixed, with regression tests.
  - 2,592 lines of offline tests; all three integrations had a live smoke test on the deepseek preset. Claude's memory tool has not been run against the real Claude API: there was no Anthropic key, and DeepSeek's Anthropic-compatible endpoint rejects `memory_20250818`, so the smoke test declared the same backend as a regular tool.
  - **Limits:** open sessions are tracked in memory per process; only text evidence is injected, so when the controller wants to look at a photo, the photo is not attached to the model call; no TypeScript or Vercel AI SDK yet.
- **After N12 (D-068):**
  - **Framing the evidence:** a stored `<memory` / `</memory` is escaped before injection, and both the evidence block and the recall tools' listing say that their content is information, not instructions. The integrations' requests change accordingly, so smoke runs cached by request miss the cache once.
  - **Rounds still in context:** OpenAI Agents no longer recalls and injects again a round with a photo that is still in its input; the three comparisons are now `evidence.stored_text` and `in_context_rounds`.
  - **A LangGraph thread that continues after it ended:** it is still extracted in full when it ends, but the space's derived memory is replayed from the raw log, so its facts equal a rebuild from the log (D-045).
  - **Peak hours:** integrations built on the SDK default to `peak_policy="warn"` (D-024).
  - Recall now records a retrieve trace, which forgetting finds and deletes (D-046).

## D-067 LongMemEval session-ID leak: fix, measure, disclose

- **Status:** adopted (N12).
- **Context:**
  - In LongMemEval_S (`longmemeval_s_cleaned.json`), the IDs of all 948 evidence sessions start with `answer_` (the evidence IDs of all 30 abstention questions also contain `_abs`); other sessions are `sharegpt_`, `ultrachat_` or a hex string.
  - `format_memories` in `read/reader.py` printed `· session <ID>` in the header of each retrieved item, so the reader could see which excerpts came from evidence sessions.
  - **Affected:** every retrieval method: B2, the tool outputs of B3 / B6a, CueMem L1 / L2 / controller, M0, and MG / ML / MC / MA (their code path shows the same headers, but they were never run on LongMemEval; see D-065).
  - **Not affected:** B0, L0, B1 (full context numbers sessions 1..n) and ORACLE. LoCoMo is not affected (`session_1..N` are chronological ordinals); MemLens raw IDs are `sess_<hash>` and do not reveal evidence.
  - Of the LongMemEval numbers published from N2 to N11, those of the retrieval methods were all produced with this leak.
  - The audit found a second leak: the session ID in Cognee's distillation contained the LongMemEval question ID (with `_abs` for abstention questions) and reached its lesson documents. It is fixed too; LoCoMo is not affected, and no published number used it.
- **Decision:**
  1. **Rewrite at load time:** LongMemEval and MemLens sessions become `session_<i>` in date order within each history (the same form LoCoMo uses). Evidence sessions and evidence turns are rewritten with them, so retrieval metrics are unchanged. The raw ID is kept in `EvalSession.source_id` for tracing only and never reaches a model's prompt.
  2. **Switch** `CUEMEM_EVAL_RAW_SESSION_IDS` (LongMemEval / MemLens only):
     - `=1` restores raw IDs, byte-for-byte as before the fix (old spaces, builds and caches still hit);
     - `=label` keeps the rewritten IDs in the spaces and puts raw IDs back only in the evidence headers the reader sees. The only difference from the default is the session names in the headers, which isolates the leak itself.
  3. **The two schemes never mix:** rewritten builds live under `<dataset>-anon` directories; the scheme is recorded in the space database, build.json and run.json; a build with the wrong scheme is not reused, and the comparison scripts stop with an error when runs with different schemes are put together. LoCoMo requests are byte-for-byte unchanged, with a test.
  4. **Reproduction:** `scripts/reproduce.py` keeps raw IDs for the historical steps, so the prompts are word for word the same as then. With the response cache from that time, the published numbers reproduce; without it, a re-run varies with sampling (N12's raw arm is one example; see consequences).
  5. **Measure:** three arms on LongMemEval test100r, with the same reader and judge (`scripts/n12_leak.py`), once under the original protocol (orig: extraction qwen3.8-flash, embeddings qwen3.7-text-embedding, reader deepseek-flash, judge qwen3.8-max) and once under pq. orig builds live in data/eval-spaces-default (the consolidation fingerprint includes the fallback model, so they cannot be shared with pq).
     - **label vs anon:** only the evidence headers differ; this is the leak itself.
     - **raw vs label:** same headers, only the IDs inside the spaces differ. Rounds within a session are ordered by a hash that contains the session ID, so renaming reorders them: this is renaming noise, present before the fix and on LoCoMo too.
     - **raw vs anon:** both together, i.e. before vs after the fix.
  6. **Disclose:** published numbers stay as they are; this record and the N12 leak ablation state the extent of the leak, the fix and the measured effect.
- **Alternatives:**
  - Only removing the session ID from the reader's headers: the session name is useful, since the reader uses it to tell which excerpts belong to the same session (LoCoMo shows it too). The ID also reaches prompts by other paths, Cognee's lesson documents being one, so changing the reader alone does not close the leak.
  - Renumbering in the reader at display time: again only covers the reader. Rewriting once at load time keeps the raw ID away from everything downstream (CueMem's builds, the external systems, the retrieval metrics).
  - Not disclosing and fixing it quietly: the numbers after the fix would no longer match the published ones without explanation, and the leak may have helped some method; without measuring, it is unknown which and by how much.
- **Consequences:**
  - Ablation (LongMemEval test100r, 100 questions, accuracy %):

  | Method | orig raw | orig anon | orig label | pq raw | pq anon | pq label |
  |---|---|---|---|---|---|---|
  | B1 full context | 81.0 | 81.0 | — | 77.0 | 77.0 | — |
  | B2 naive RAG | 82.0 | 86.0 | 87.0 | 82.0 | 83.0 | 84.0 |
  | CueMem L1 | 92.0 | 90.0 | 92.0 | 90.0 | 91.0 | 89.0 |
  | CueMem L2 | 93.0 | 96.0 | 95.0 | 94.0 | 93.0 | 93.0 |
  | CueMem controller | 88.0 | 89.0 | 90.0 | 89.0 | 90.0 | 87.0 |
  | M0 Mem0 2.2.1 | — | 79.0 | 85.0 | 83.0 | — | — |

  B1 is not affected by the leak and has no label arm. Mem0's raw arm was not re-run under orig (N11's published 83.0% is that condition); under pq only Mem0's raw arm was run, and its anon and label arms were not. `ARMS` in `scripts/n12_leak.py` lists these three run names, and the runs that do not exist are skipped.
  - Pooled label vs anon (summed over retrieval methods; same questions, so not independent): orig +21 / −12 (+1.8 points per method on average, p = 0.16, Mem0 included); pq +10 / −14 (Mem0 excluded: its anon and label arms were not run; −1.0 points, p = 0.54). No single method is significant; the largest is Mem0 under orig, +9 / −3 (p = 0.15).
  - Published N11 test100r numbers (orig, raw IDs; CueMem's methods come from the N9 test runs, D-055): B1 80.0%, B2 86.0%, L1 92.0%, L2 96.0%, controller 92.0%, Mem0 83.0%. After the fix (orig, anon): L1 90.0%, Mem0 79.0%. The raw arm was re-run in N12; no reader call of B1, B2, L1 or L2 came from the response cache, so they were sampled afresh. Against the published numbers: B1 81.0 vs 80.0, B2 82.0 vs 86.0, L2 93.0 vs 96.0, controller 88.0 vs 92.0; only L1 matches. Arms are therefore compared only within the N12 runs.
  - **Resolution:** 100 questions and one run per arm. Renaming noise alone (orig raw vs label, pooled +5 / −14, −2.25 points, p = 0.064) is larger than the leak effect measured (+1.8 points), and the N12 raw re-run differs from N11's published numbers by up to 4 points; B1, which does not depend on IDs, still moved by 1 point. Effects of a few points cannot be resolved here.
  - **Conclusion:** on test100r, no inflation of L1, L2 or the controller by the leak was measured; Mem0's label vs anon is +9 / −3 (p = 0.15, not significant), so its published 83.0% may be a few points high; the gaps between retrieval methods and full context are unchanged.
  - **Limits:**
    - only B2, L1, L2, the controller and Mem0 were re-measured, on test100r. B3, B6a (whose model sees the same headers in tool outputs), ablations such as B2F and "L1 + take notes first", and N9's full test400 were not re-run, so the effect on their published numbers is unknown; the same holds for the LongMemEval part of D-041 (H3);
    - the learned controller's accuracy table and λ were fit on outcomes from the leak era and have not been refit;
    - the order of rounds within a session still depends on the IDs;
    - one run per arm.
  - LongMemEval and MemLens evaluations use rewritten IDs by default from now on.
  - Raw data: [summary.json](../results/n12-leak/summary.json) (regenerate with `scripts/n12_leak.py`).
- **After N12 (D-068):** keeping the two protocols' builds in separate directories used to be a convention only. A reused evaluation space whose consolidation fingerprint does not match is now reported as an error instead of being rebuilt, so one wrong `CUEMEM_PATHS__EVAL_SPACES` can no longer rewrite the other protocol's spaces. The reproduction script gained `--cache-only` (LLM requests read the local cache only, and a miss is an error), which checks point 4's claim that the published numbers reproduce with the response cache from that time.

## D-068 A full audit: seven lenses, offline reproduction, adversarial verification, 24 fixes; evaluation requests and published numbers unchanged

- **Status:** adopted (after N12).
- **Context:**
  - After N12, CueMem is no longer just evaluation code: it has a Python SDK, a memory proxy, three framework integrations, an HTTP API and UI, and a Docker image, which other people will install and deploy. These entry points were added within a few days in N12, each with its own tests, but nobody had gone through them end to end asking "how does this break?". The evaluation side (the reproduction script, resuming, the guards on the frozen methods) had only been patched when needed.
  - So the whole project was audited through seven lenses: core correctness (writes, reads, temporal logic, forgetting and replay, the storage layer), robustness (concurrency, resources and failures), security and privacy, performance, evaluation hygiene and reproducibility, product gaps, and test quality. Each lens only read code: no changes to the repository, no API calls, no evaluations. Every finding had to be reproduced offline (with the repository's FakeAPI and ScriptedModel stand-ins, or read-only copies of the databases and caches under data/), with a measurement of how often it occurs in the published data and whether the fix would change evaluation requests.
  - Every candidate then went through adversarial verification: another reviewer re-ran the reproduction, looked for counter-examples, checked the claims, narrowed or rewrote the fix, and judged whether it would break the invariant below. 24 findings were kept. Another 17 were confirmed but left for later, because they touch a lot of code for little gain or need a decision of their own (see consequences).
  - The 24 fixes were implemented in two lanes, merged, and the whole result reviewed again. That review found 12 more problems: mostly corner cases under concurrency, fixes wired into only some of the entry points, or tests that never reached the fix itself. All are fixed, each with a test that fails before its fix. The groups below describe the state after the merge and the review.
- **Decision:**
  1. **Invariant and rules:**
     - LLM and embedding requests under the default configuration and in evaluations stay byte-for-byte the same, and so do cache keys: the N2–N12 numbers must replay from data/cache unchanged. The frozen methods (B1, B2, B4-nobm25 = L1, B5 = L2, B6, and the three controller files) keep their behaviour;
     - every behaviour fix has a regression test, checked to fail on the code before the fix;
     - every deliberate change of a default is listed in its group below.
  2. **Forgetting and privacy** (revises D-021, D-046, D-051):
     - **Forgetting vs background consolidation:** forgetting, rebuilding and deleting a space hold that space's consolidation lock throughout. Before, forgetting while the background worker was halfway through a consolidation let the worker write the deleted session's facts, vectors and extraction cache back ("The user's PIN is SECRET-1234." was still a current fact after forgetting).
     - **Forgetting vs reads in flight:** one reader-writer gate per space. Answers, retrievals and agent turns hold it shared, forgetting and deleting a space hold it exclusive: cleanup starts once the reads in flight have finished, and new reads in that space wait behind the forget. Before, an answer that had already seen the deleted memory wrote it into the LLM cache and a trace after forgetting had cleaned them.
     - **Clean caches:** the LLM and embedding caches truncate their WAL as soon as they delete entries (before, the deleted text stayed in llm.sqlite-wal until the process exited). The embedding cache also drops the query vectors of the deleted messages, of the questions in deleted traces and of the deleted fact statements. SDK search, the proxy, the integrations, and MCP and HTTP recall now record a retrieve trace each, through which forgetting finds and purges the query vector of the question; messages with images are purged once more as text without the image marker (the proxy's query text). Closing the system closes the caches it opened.
     - **Traces:** besides traces whose attributes contain deleted text, traces whose retrieval records (spans, events) list a deleted round, fact or image ID are deleted. This errs towards deleting: in a small space, forgetting one session can remove most answer traces.
     - **Clean up on failure:** when the replay fails halfway (say, insufficient balance during reconciliation), traces and caches are still cleaned and the WAL truncated before the exception reaches the caller; unfinished consolidation completes next time.
     - **No `VACUUM` on every forget:** `secure_delete` plus a truncated WAL leaves no deleted content in the file (overflow pages included, with a forensic test). `VACUUM` rewrites the whole file: over 4 seconds for a 146 MB database, with every read and write on it stalled meanwhile.
     - **Uploaded photos:** photos sent through the HTTP API and the UI used to leave a copy in `<data>/uploads` that forgetting could not reach, while the report said images_deleted=1. Sessions now read the copy in the media store, the upload is deleted when the request ends, and copies left by earlier versions are removed at startup; Gradio's temporary copies in the UI are deleted every hour once they are an hour old.
     - **Changed defaults:** the database file no longer shrinks after forgetting; forgetting waits for answers, retrievals and consolidation in flight in that space; a retrieval made outside another trace (SDK search, the proxy, the three integrations, MCP and HTTP recall) adds one line to runs/traces/*.jsonl.
     - **Evaluation:** evaluations never forget or start the worker, and each question is answered inside its own trace (no extra retrieve trace), so requests, cache keys and usage are unchanged.
  3. **Concurrency and durability** (revises D-020, D-045):
     - **Failed extractions retry on their own:** background jobs that failed (rate limit, insufficient balance, timeout) are retried with backoff, after 1 minute and 4 minutes, at most 3 attempts; when a whole consolidation run raises (reconciliation or embedding errors), the space is retried with the same backoff, up to 3 times in a row. Before, a failed job was never extracted again unless someone ran an explicit consolidate.
     - **Resume after a restart:** the first start of the worker submits every space with pending jobs or with failed jobs under 3 attempts. Open sessions leave a marker in the metadata: the first time the SDK, the HTTP service or the proxy touches a space, sessions that were open when a process crashed, and are not open in this process, go to extraction (`MemorySystem.adopt_open`), except a session the caller is about to continue. Shutdown waits at most 30 seconds, then logs a warning and leaves the job for the next start.
     - **A session that grows:** when a consolidated session gains messages (per-turn SDK `add` with the same run_id, a LangGraph thread that continues after it ended, the proxy reusing a named session), it used to be extracted again in full and only added to, so the earlier extraction's wording stayed ("likes hiking" next to "likes hiking in the Alps"). Now derived memory is cleared and replayed from the raw log (cache hits; only the grown session is extracted again), so derived memory again equals a rebuild from the log. The replay only enqueues sessions that already had a job: `infer=False` log-only sessions are not extracted, sessions skipped by content moderation are not sent again, and failed jobs keep their attempt count and backoff. Without a writable LLM cache there is no replay, and a warning is logged. The cost is recomputing the whole space locally each time (about 12 s for 200 sessions, no LLM call when cached), and reconciliation and entity requests for later sessions may be new: write a multi-part conversation inside an SDK run (group 5).
     - **Race corners:** an explicit consolidate reads the open sessions only once it holds the lock; `cuemem demo --fresh` wipes the space inside the lock.
     - **Changed defaults:** `cuemem serve`, `cuemem proxy` and `Memory(background=True)` may send leftover extraction requests right after startup; background failures go to worker.errors and the status API (the last 100 only).
     - **Evaluation:** evaluations never start the worker, and every session is ingested once with exactly one job (the audit scanned 19,472 sessions in 414 evaluation databases), so the replay never fires; request bodies for whole-conversation ingest, strict consolidation and every kind of question are identical before and after.
  4. **Security of the HTTP API, the proxy and Docker** (revises D-056, D-062, D-063, D-066):
     - **A token is required:** `cuemem serve` and `cuemem proxy` refuse to start (exit code 2) on a non-loopback address without `CUEMEM_API_TOKEN`; running without authentication needs an explicit `--insecure-no-auth`. The token is checked before the memory system is opened, so a refused start does nothing. `docker compose up` stops with an error without a token, and publishes the port on the host's 127.0.0.1 only. Before, following the README's `docker compose up -d` left export, forget, import and the `/v1` proxy open to the whole network without authentication, and the empty token compose injected also hid a token written in /data/.env.
     - **Requests from other sites:** when `/api/`, `/v1/` or the UI's own `/gradio_api/` receives a request with an Origin header, the Origin must match the Host (or be listed in `CUEMEM_ALLOWED_HOSTS`), otherwise 403; `Origin: null` is refused too. Requests without an Origin (SDK, curl, gradio_client) are unaffected. On a loopback address only Host 127.0.0.1, localhost, [::1] or the bind address is accepted (400, against DNS rebinding). A new environment variable, `CUEMEM_ALLOWED_HOSTS` (comma-separated, can go in .env), lists a reverse proxy's or another hostname: it joins the Host allow-list on loopback addresses and is accepted as a same-origin Origin on any address; `*` turns both checks off. The proxy accepts JSON only (415). Before, any web page could send a text/plain POST, which needs no preflight, through the proxy to spend the operator's upstream credits and write a line into a guessed user's memory; with `--host ::1`, the UI's API accepted any origin.
     - **Import:** image paths are no longer built from the file's own sha256 and mime fields (which could write outside the temporary directory and create .py or .sqlite files). Images are strictly decoded and verified with PIL and named by the recomputed hash; one bad image fails the whole import, with nothing written (HTTP 409).
     - **Interrupted streams:** when the upstream fails, nothing is written (as without streaming); when the caller stops or disconnects, only the user's message is written and the upstream connection is released at once. Before, a half answer was stored as if complete and later extracted into facts.
     - **Injected evidence:** a stored `<memory` or `</memory` (any case, spaces allowed) is written as `&lt;memory` / `&lt;/memory` and can no longer close the evidence block early; the evidence block and the recall listing both say that their content is quoted from past conversations, information and not instructions. This reduces prompt injection from memory but cannot prevent it.
     - **Spaces with an agent:** spaces such as `bot/alice`, or `shop/bob` written by the proxy, can be browsed, exported and forgotten on every `/api/spaces/{space}/…` route (they returned 404), with "/" raw or written as %2F.
     - **Photo rounds still in the request:** the proxy and OpenAI Agents no longer recall and inject again a round with a photo that is still in the request; the three "already in context" comparisons became `evidence.stored_text` and `in_context_rounds`.
     - **Changed defaults:** no token off loopback means no start (reversing D-056's warning), so the Docker image needs a token; compose publishes on the host's loopback only; proxy requests must be JSON; cross-origin browser requests get 403; on loopback addresses the Host is checked, so access through a reverse proxy or another hostname needs `CUEMEM_ALLOWED_HOSTS` (or a reverse proxy that forwards the Host). The evidence block that the proxy, MCP and the three integrations send to the model has one more sentence, and the recall listing one more line: this deliberately changes those requests, so integration smoke runs cached by request miss the cache once.
     - **Evaluation:** the evaluation gateway does not go through these two apps; accepted requests are handled exactly as before; evaluations, the gateway, the frozen methods and the demo agent do not use `evidence.py`, so their requests are byte-for-byte unchanged.
  5. **SDK writes and temporal correctness** (revises D-019, D-020, D-024, D-061):
     - **Concurrent `add`:** concurrent `add` calls with the same run_id used to drop messages silently (16 threads stored 2 messages, and the returned IDs pointed at other callers' messages, so forgetting by them deleted the wrong thing). Turn numbers are now allocated under a per-session lock, and a message that was not written raises `ConcurrentWriteError`.
     - **Times with a time zone** are converted to local time and stored without a zone (on a UTC+8 machine '10:00Z' is stored as 18:00), and existing '+00:00' values are converted when read. Before, a single 'Z' timestamp blocked consolidation of the space for good, and time-scoped search raised TypeError.
     - **Writes within the same second:** generated session IDs carry a strictly increasing nanosecond counter and default times are no longer truncated to the second, so writes within one second are reconciled in order. Before, the order was random, and the old address became current half the time.
     - **Sessions spanning several days:** when the latest source message of a fact was said on a different day from the session start, the fact is recorded at that message's time, and the extraction prompt marks the date before messages on a new day; within one day the session start is kept.
     - **Other scripts:** names and objects in Cyrillic, Greek, Hangul, kana, Arabic, Hebrew and similar scripts used to normalise to an empty string, so speakers were merged into one entity, facts were dropped and different cities were treated as duplicates. These scripts now get an NFKC + casefold key; keys for Latin (accented letters included) and Chinese are unchanged, and fact IDs are still derived from the original normal form.
     - **Peak hours:** the SDK and the integrations used to inherit the batch jobs' `peak_policy=wait`, so during DeepSeek's peak hours `answer()` silently waited for the peak to end (up to 4 hours). `Memory` now defaults to warn, like the CLI's interactive commands; batch writers who want the off-peak price pass `peak_policy="wait"`.
     - **Extract a conversation once:** the new `Memory.open_run / end_run` and `with m.run(...) as run_id` (`async with` on AsyncMemory): inside a run, `add` still writes online and is searchable at once, and the whole run is extracted once when it ends. Calling `add` once per turn with the same run_id still works, but re-extracts and replays every turn.
     - **run_id scope:** a run_id in `filters` used to be ignored silently. It, and a new `run_id=` keyword, now narrow `search` and `get_all` to that session (a post-filter, so results can be fewer than the limit), and `delete_all(run_id=)` forgets only that run.
     - **Changed defaults:** the SDK's peak policy is warn; generated session IDs have a new format (same prefixes); created_at and said_on returned by the API and the SDK carry microseconds; input with a time zone is stored as local time.
     - **Evaluation:** evaluations never construct the SDK's Memory, and the peak policy is not part of any cache key; evaluation times carry no zone, all 204k evaluation turns sit at their session's start, so there are no date markers; the consolidation sort key is unchanged. The normalisation was checked on 129,715 values from 5,169 extraction outputs in data/cache: fact IDs and subject, object and predicate keys are all unchanged; only one Hebrew alias in one LongMemEval space enters the alias index, where it matches no other name or object, so requests are unchanged.
  6. **Evaluation hygiene and reproducibility** (revises D-016, D-028, D-048, D-052, D-053, D-055, D-058, D-067):
     - **A space built under another configuration:** when a reused evaluation space's consolidation fingerprint or pipeline version does not match, it is no longer cleared and rebuilt silently (before, one wrong `CUEMEM_PATHS__EVAL_SPACES` rewrote another protocol's published spaces with this run's configuration). Methods that need fact memory record a ConfigMismatchError that names each protocol's directory; B0, B1 and the external baselines still answer; `cuemem eval run --rebuild-spaces` rebuilds.
     - **Resuming:** while questions remain, the models of the judge, extractor, planner and other roles, `--k`, `--limit`, the question set, the consolidation and retrieval settings, and the controller's version and file content must match too. run.json also records a hash of the question IDs, the controller's version and sha256, the command line and the CUEMEM_* environment (secrets removed), plus one entry per resume. Complete runs are not checked, so the reproduction script still regenerates their reports.
     - **Guarding the frozen requests:** the only guard hashed the messages of B1, B2 and B4 requests on LoCoMo, and the audit's drift plugins (one more sentence in a prompt, one more space in a tool description, the L1 budget from 4000 to 3900, a different cache-key canonicalisation) all passed it. Requests are now pinned by cache key, method by method: B1, B2, B2F, B4, B5, B6, B3 and B6a on LoCoMo mini, and B1, B2, B4, B5, B6, B3 and B6a on the LongMemEval fixture with raw IDs (`FROZEN_REQUESTS` in `tests/test_eval_leak.py`); `tests/test_frozen_config.py` pins the L1, L2 and N4 default configurations, the parameters of the eight frozen methods and the sha256 of the three controller files; `tests/test_llm_client.py` and `tests/test_embedding.py` pin how request_key and the embedding cache keys are written. Every value was recomputed on commit e7064ef, from before the audit, and matched.
     - **The reproduction script:** `scripts/reproduce.py` no longer rewrites the frozen controllers. The three training scripts only check (`--no-export --check`) and stop without writing anything when the retrained result differs from the committed file. Evaluation runs and probes run with `--fail-on-error` (exit code 3 on errors, 4 at the budget limit), and errored or budget-stopped reports are no longer copied into docs/results; steps that copy no report tolerate only the very errors they had before (such as the N9 question that content moderation refuses every time). The new `--cache-only` makes LLM requests read the local cache only, with a miss as an error; afterwards, check the results with `git diff docs/results src/cuemem/read`. Mem0 is pinned to 2.2.1.
     - **Changed default behaviour:** an evaluation space with a mismatched fingerprint is no longer rebuilt automatically; a future pipeline-version bump or a new consolidation field will likewise make existing evaluation spaces report this error until `--rebuild-spaces` is passed or the directory changes. Resuming checks the items above. The new switches `--rebuild-spaces`, `--fail-on-error` (`eval run` and `eval probe`), `--check`, `--no-export` and `--cache-only` are all off by default, and the default exit codes of both evaluation commands are unchanged; run directly, the training scripts still export.
     - **Evaluation:** these are guards and records and change no request; `scripts/reproduce.py` still regenerates the reports of complete runs. The audit checked read-only that all 414 published CueMem evaluation spaces on this machine are accepted under their own configuration, and that the 10 run.json files in docs/results pass the new resume checks.
  7. **Packaging and first run** (revises D-056, D-060):
     - After pip install, `cuemem serve` no longer crashes looking for the shopping demo's catalogue: without `--scenario`, a source checkout still gets the shopping assistant and an installed copy the plain assistant; `cuemem demo` uses the catalog.json next to its script.
     - On errors such as a missing key, serve, chat, demo and proxy print a one-line error ("错误：…") and exit with code 1 instead of a traceback (which made Docker restart the container in a loop). Commands in error messages match the installation (`cuemem …` or `uv run cuemem …`); with only OPENAI_API_KEY set they suggest the openai preset; the embedding-model mismatch error points to a fresh data directory instead of telling the user to delete data/.
     - The Docker image includes the `local` extra (local embeddings, needed by the deepseek preset) by default, and `--build-arg EXTRAS=` builds a slim image. CI has two new jobs: package installs the wheel outside the repository and runs `cuemem --help`, `cuemem doctor`, an SDK smoke test (`scripts/smoke_wheel.py`) and a serve health check; docker builds the image and checks its health endpoint.
     - **Changed defaults:** outside a source checkout, `cuemem serve` defaults to the plain assistant; the Docker image ships fastembed and onnxruntime and is larger.
     - **Evaluation:** evaluations do not use these entry points; the changes are error messages and the default scenario, and no model request changes.
  8. **Read-path performance:**
     - The probe's count of entities named in the question filters by substring first (before, every read compiled one regular expression per name, and past 512 names Python's regex cache stopped helping: about 0.8 s at 12k entities). When text-to-image search has hits, only the messages holding the hit images are loaded by ID (before, the whole raw log). The fact table gains two indexes (superseded by, invalidated at), which existing databases add when opened, without a schema-version bump.
     - `scripts/n11_latency.py` gains opt-in `--entities-per-round` and `--photos` (default 0, with synthetic data byte-identical to before; runs that use them write latency-extended.json).
     - **Changed defaults:** none.
     - **Evaluation:** probe features, controller decisions, retrieval results and requests are unchanged: on the 414 evaluation databases, the old and new code counted entities for 33,119 questions with identical results. The published N11 latency table was not re-run.
- **Alternatives:**
  - **Fixing the 17 deferred findings in the same batch:** most of them change the hot write path or the failure paths, or need a decision of their own (the timeout fallback, for instance, would change requests on the failure path; indexed reconciliation first needs a read-only cache replay of every evaluation space to prove the gray-zone requests unchanged). In the same batch, "evaluation requests are byte-for-byte unchanged" would have been hard to verify.
  - **A generation counter between forgetting and reads** (an answer checks whether its space was forgotten before writing the cache): it protects the cache but not agent replies written to the log. The reader-writer gate covers both, and without a forget it costs one condition-variable round trip.
  - **Cross-process guards** (conditional job-state updates, rolling back the vectors a run wrote): they cannot stop a reconciliation computed from stale state, and could delete legitimate vectors with the same IDs. The premise stays: one process writes a database at a time.
  - **Recording every fact at its latest source message's time** (the audit's original proposal): with the demo agent's turns 2 minutes apart, same-session facts swapped places in the profile, more than 5 of 20 agent requests changed, and the N8 script replay no longer hit the cache. With the day-level rule, every request stays byte-for-byte the same.
  - **Switching all normalisation to Unicode `\w`:** 400 accented values in the evaluation extractions would have changed the keys of 126 statements. Only scripts other than Latin and Chinese get the new key.
  - **Stopping the reproduction script on any error:** N9 has a question that content moderation refuses every time, so the script would stop there on every run; hence the two exit codes 3 and 4, and steps that copy no report tolerate only the errors they had before.
  - **Changing the proxy's query text** so that forgetting can find it: that would change the proxy's existing retrieval requests and cache keys; instead, forgetting also purges the text without the image marker.
- **Consequences:**
  - Code and tests: 75 files changed (not counting these docs), 103 new test functions (677 offline tests in total, parametrised cases included); ruff and mypy pass. Every behaviour fix was checked to fail on the code before it.
  - **Evaluation requests and published numbers are unchanged:** after the merge of the two lanes and the review fixes, `FROZEN_REQUESTS` in `tests/test_eval_leak.py` (request counts and cache-key hashes for 15 method × dataset pairs), `tests/test_frozen_config.py` (the L1, L2 and N4 default configurations, the parameters of the eight frozen methods, the sha256 of the three controller files) and the golden cache-key values in `tests/test_llm_client.py` and `tests/test_embedding.py` all pass, and none had to be re-pinned; the N2–N12 numbers still replay from data/cache.
  - **Requests that change on purpose** are only on paths evaluations never take: the evidence block's note and the recall listing of the proxy, MCP and the three integrations, and the repeated excerpts that the proxy and OpenAI Agents no longer inject; date markers in the extraction prompts of sessions spanning several days; and, after a session grows, possibly new reconciliation and entity requests for later sessions during the replay.
  - **Left for later** (confirmed; why not in this batch in parentheses):
    1. The online write is not atomic: if embedding or Qdrant fails, the messages are already in the log but have no index and no consolidation job, and a retry by the caller writes them twice (reorders the hot write path);
    2. Provider failures: the timeout fallback reaches the fallback model only after the SDK's retries, about 4 × timeout_s (D-050), and the HTTP API turns provider errors into a bare 500 (changes requests on the failure path; needs its own decision);
    3. Reconciliation compares every new fact with every fact, so the replays of forgetting and rebuilding are quadratic in CPU (an index first needs proof that the gray-zone requests stay byte-for-byte the same);
    4. Every forget pays again for the whole space's reconciliation and entity requests, serially, up to 4 times (revises D-021; depends on every cache entry being tagged with its session);
    5. Redundant full scans in local Qdrant: caption search in text-only spaces, summary search with zero weight, and L2 and at_least_l1 not reusing the probe's retrieval;
    6. Every consolidation reloads all facts, entities and their vectors, and `add(wait=True)` also re-hashes every session;
    7. Slow start-up: `cuemem --help` takes 3.5 s and `import cuemem.sdk` 2.4–2.9 s (eager imports in the package `__init__` files and the CLI);
    8. Heavy dependencies: gradio, matplotlib and scikit-learn, used only by the UI and evaluations, are core dependencies (36 extra packages, about 250 MB);
    9. Significance claims have no multiple-comparison correction: under Holm's correction several claims at p ≈ 0.013–0.04 are no longer significant, and N12's tests assume one build per system (docs and statistics only; accuracies are unchanged);
    10. The nominal embedding cost depends on the state of the embedding cache, against D-010 (B2's write cost under pq is understated by about 57%, about $0.008 in absolute terms; fixing it changes published cost figures);
    11. Development/test twins: LongMemEval's question/abstention twins straddle dev100 and test400, and the development and test mixed streams share 16 hard negatives (the test sets are already spent and the effect is small; group future splits by twin);
    12. A Mem0-shaped HTTP memory API (`/api/v1/memories`), a remote MemoryClient and a TypeScript client;
    13. The remaining Mem0 compatibility gaps: `metadata`, `update()`, `reset()`, expiry, and SDK search returning an empty list at L0;
    14. Multi-tenancy: one shared token, so any caller can read or delete any user's memory; MCP is stdio only, one space per process;
    15. The proxy has Chat Completions only: `/v1/responses` and `/v1/embeddings` return 404, and unknown models are silently sent to the agent role's model;
    16. Most developer-facing text is Chinese: exceptions (the HTTP API's errors included), SDK docstrings, the CLI and the examples; there is no English SDK, HTTP or proxy reference (every string that reaches a model must be left alone);
    17. `cuemem eval run` has no tests, and the rule that drops LoCoMo's adversarial questions is copied in four places.
  - **Remaining limits:**
    - forgetting still does not purge the query vectors of L2's planned sub-queries or of the agent's own searches, nor the image-embedding cache; Gradio's temporary copies in the UI can outlive a forget by up to about 2 hours;
    - matching traces by ID over-deletes;
    - adopting sessions left open by a crash and allocating turn numbers under a per-session lock both assume one process writes a database at a time;
    - the proxy's write-back after the upstream answer is not under the read gate (a stream can run long);
    - conversation language is still detected only as Chinese or English, and the BM25 tokenizer and hash embedder still ignore other scripts;
    - B6a's photo path still has no offline request snapshot;
    - the pipeline version was not bumped for existing multi-day sessions or sessions stored with time zones; they are corrected at their next re-consolidation or rebuild;
    - the Dockerfile change was checked only by dependency resolution; the image was not built on the development machine (no Docker) and is built by CI's docker job.
