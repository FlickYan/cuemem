# CueMem design

> v1.1 · 2026-10-13 · English edition of [docs/design.md](../design.md). This document explains what the system does and why. The test-set results are in the [technical report](report.md). Design decisions are recorded in [decisions.md](decisions.md) ([中文](../decisions.md)), cited here as D-xxx. The design was drafted in N0, and each section was filled in with the implementation as its node was completed.

## 0. Summary

- **Problem.** How can an LLM agent correctly remember, recall and update what a user told it across months and hundreds of sessions, including photos the user shared?
- **Three convictions.**
  1. **Long context is not memory.** However large the window, deciding what to keep, when to recall it and which version is still valid remains an information-management problem.
  2. **Quality should come from the memory system, not from the model's ability.** Whether to retrieve, and how deeply, should not be left entirely to the LLM.
  3. **Multimodal memory cannot store text descriptions only.** Compressing for the index is fine, but answering must be able to look at the original again.
- **Approach.** Three pillars, with evaluation first:
  - **Write.** An immutable log, consolidated asynchronously into facts with bi-temporal validity, an entity graph, multi-granularity indexes and a user profile.
  - **Read.** A memory probe that calls no LLM (one retrieval scoring pass; measured p50 0.2–0.5 s, mostly the query embedding) feeds a **retrieval controller**. The controller makes a cost-aware choice between L0 (no retrieval), L1 (fast recall) and L2 (deep recall).
  - **Multimodal.** Context-aware captions and visual embeddings serve as the index, and the original photos are **re-perceived** at answer time when needed.
- **Deliverables.**
  - Accuracy–cost Pareto curves, ablations and error analysis on LongMemEval and LoCoMo (text) and MemLens (images and text), all under one evaluation framework.
  - An explainable demo that shows why a memory was recalled.

## 1. Problem and motivation

### 1.1 Why long context is not memory

| # | Problem | Explanation |
|---|---|---|
| 1 | Cost grows linearly with history | Re-reading the whole history every turn costs O(N) per turn and O(N²) over a conversation. Prompt caching lowers the price, not the amount of information the model must process. |
| 2 | Effective context is far shorter than nominal | Lost in the Middle, RULER, NoLiMa and Chroma's Context Rot report all show that longer inputs make information harder to use, especially when non-literal reasoning is needed. LongMemEval reports clear drops for long-context LLMs and commercial assistants on long-term interactive memory. |
| 3 | Knowledge goes stale | The history contains both "I live in Beijing" and "I moved to Singapore". Long context leaves the conflict for the model to resolve on the spot; a memory system can resolve it at write time and keep the old version. |
| 4 | Persistence and sharing | Context disappears when a session ends; memory has to persist across sessions and devices, and even be shared between agents. |
| 5 | Governance | Users should be able to see, change and delete what an agent remembers (Singapore's PDPA, the EU's GDPR). Information stuffed into a context window cannot be governed. |
| 6 | Cost of modalities | Image and video tokens are expensive. MemLens found that long-context vision-language models (LVLMs) degrade clearly as conversations grow. |

> **A context window decides how much fits. A memory system decides what should be in it.**

### 1.2 The core question: who decides whether to recall

| Approach | Who decides | Strengths | Weaknesses |
|---|---|---|---|
| Always retrieve (naive RAG) | A fixed pipeline | Simple, stable | Wastes tokens and adds noise on easy turns; not deep enough on hard ones |
| Model decides (agentic RAG, MemGPT-style) | The LLM, through tool calls | Flexible | Depends on the model, and small models do it badly; hard to predict and to evaluate; an extra LLM round trip; **the system's value is masked by the model's ability** |
| Rules | Hand-written rules | Controllable, explainable | Brittle, incomplete coverage |
| **Decoupled retrieval controller (this project)** | A light trainable policy plus signals from memory | Measurable, trainable, independent of the reader model, cost is tunable | Needs training labels |

**Key insight.** N7 tested this; the first half holds (§6.3, §8.6).

- In general-knowledge RAG, "should I retrieve?" is roughly "does the model know?". That is why FLARE and DRAGIN trigger retrieval on model uncertainty.
- For **private memory**, the model's parameters never contained the user's information. Asked "what is my dog's name?", a model may confidently invent a name or simply refuse, so its uncertainty is not a reliable signal.
- What matters is **whether this turn depends on the user's past**, and two kinds of cheap signals can tell:
  - **Cues in the query:** first person and possessives ("my"), references ("that one", "last time"), temporal expressions ("last week", "in March"), aggregation and order words ("how many times", "the first"), visual-detail words ("in the photo", "what color").
  - **Feedback from a memory probe:** one hybrid retrieval with the current query, with no LLM call. Is there a strong hit? Are hits concentrated in one session or spread over many? Are entities hit? Are superseded facts among them?

This matches where leading systems went in 2026. Agent Zero Memory explicitly aims for *memory-driven rather than model-driven* quality, with an intent gate on its read path, and its accuracy varies by only 3.4 points across 8 LLM backbones. This project differs in treating the controller as a **first-class, separately evaluated, learnable component**, with:

- explicit label construction;
- a cost knob λ;
- Pareto analysis;
- a head-to-head comparison with agentic retrieval under the same framework.

The project does not reject agentic retrieval. It is a baseline (B3), and it is kept in the agent as an **escape hatch**: the controller recalls first, and the model can call `memory_search` itself if something is missing (N8, D-042).

### 1.3 What makes multimodal memory hard

MemLens (2026; 789 image-and-text multi-session questions, 80.4% with image evidence) points to the gap:

- long-context LVLMs look at images directly and do well on short contexts, but degrade as the conversation grows;
- memory-augmented agents are stable with length, but **compressing at storage time** (e.g. turning an image into a text description) loses visual detail;
- without the evidence images, frontier LVLMs fall below 2%, so the image itself cannot be replaced;
- multi-session reasoning keeps most systems below 30%; neither kind of method solves it alone.

A problem that is often overlooked: **an image's meaning comes from the conversation.** A photo of a golden retriever is visually just "a golden retriever". Once the user says "this is Max, I just adopted him", it should be bound to the entity Max, so that "what breed is Max?" can be answered later.

The response here: **at index time**, compress with context-aware descriptions and visual embeddings (stable with length); **at answer time**, the controller decides whether to fetch the originals and re-perceive them (visual fidelity restored).

## 2. Goals and non-goals

**Goals**

- **G1** Cross-session long-term memory that handles knowledge updates and temporal reasoning correctly.
- **G2** Adaptive retrieval: tunable cost, measurable, independent of the LLM backbone.
- **G3** Multimodal: images are first-class (video is an extension).
- **G4** Explainable and controllable: every memory that enters the context can be traced to its source; users can view, change and delete memories.
- **G5** Rigorous evaluation: public benchmarks, one protocol, ablations, cost accounting.
- **G6** Engineering quality: typed, testable, reproducible, and runnable on an 8 GB laptop.

**Non-goals**

- No training or fine-tuning of large models (the controller is a small model).
- Not chasing leaderboard SOTA. Reported numbers on text memory benchmarks reached 93–96% in 2026 (e.g. Agent Zero Memory: LongMemEval 95.60%, LoCoMo 93.60%), relying on strong backbones and their own evaluation settings. This project focuses on relative comparisons and Pareto efficiency within one framework.
- No distributed production deployment, though §12 sketches how to scale.
- Video is outside the MVP (listed as an extension).

## 3. Design principles

1. **Memory-driven, not model-driven.** Quality comes from the memory's structure and retrieval policy. The LLM is a replaceable component, validated with several backbones.
2. **Take the cheap path first and escalate when needed:** L0 → L1 → L2.
3. **Intelligence at write time as much as at read time.** Organise, deduplicate and resolve conflicts when writing, rather than leaving everything to retrieval.
4. **The raw log is the only source of truth.** Facts, summaries, the profile and image captions are derived data: rebuildable, traceable, and deletable by source.
5. **Measure everything.** Every component has metrics; every turn's decisions have a trace.
6. **Laptop first.** All models (LLM, VLM, embeddings) are APIs; all storage is embedded; an 8 GB laptop runs it.

## 4. System overview

### 4.1 Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/architecture-dark.svg">
  <img alt="Architecture: read path with probe, controller, L0/L1/L2 and reader; write path with online write, memory store and asynchronous consolidation" src="../assets/architecture-light.svg" width="900">
</picture>

The N0 plan also had a reranker in L1, graph diffusion (PPR) in L2, and reactive escalation from L1 to L2 when evidence is insufficient. All three were implemented. None helped enough in testing to be on by default (D-031, D-032; §6.3).

### 4.2 Memory types

Following CoALA's taxonomy of memory for language agents:

| Memory type | Implementation here | Storage | When it enters the context |
|---|---|---|---|
| Working memory | Recent turns and the current task state | In memory | Every turn |
| Core memory (user profile) | Important, still-valid facts: identity, long-term preferences, current goals (capped at about 400 tokens; 600 in the agent) | SQLite | Every turn (L0) |
| Episodic memory | Raw turns and session summaries (multi-granularity) | SQLite + vector index | L1 / L2 retrieval |
| Semantic memory | Atomic facts with validity periods, plus an entity graph | SQLite + vector index + graph | L1 / L2 retrieval |
| Multimodal memory | Originals, context-aware captions, visual embeddings, entity binding | Files + vector index + graph | L1 / L2 retrieval, re-perception on demand |
| Procedural memory (optional) | How the user likes to interact ("keep answers short") | Folded into core memory | L0 |

### 4.3 Data model (simplified)

Full definitions are in `src/cuemem/core/models.py`.

```python
class Turn:  # raw turn: the only source of truth, immutable
    id: str
    space_id: str
    session_id: str
    role: Literal["user", "assistant"]
    speaker: str | None  # speaker name in multi-party conversations (e.g. LoCoMo)
    text: str
    media_ids: list[str]
    ts: datetime


class Episode:  # session segment, the "summary" granularity (one per session for now)
    id: str
    session_id: str
    turn_ids: list[str]
    start_ts: datetime
    end_ts: datetime
    summary: str


class Entity:  # people, pets, places, organisations, objects, events...
    id: str
    name: str
    type: str
    aliases: list[str]  # proper names only ("Maxie"), not descriptions ("my dog")
    description: str
    first_seen: datetime
    last_seen: datetime
    mentions: int  # number of sessions that mention it


class Fact:  # atomic fact (semantic memory), bi-temporal
    id: str
    subject_id: str  # subject entity
    predicate: str  # snake_case, e.g. lives_in, job_title, 5k_personal_best
    object: str
    statement: str  # a self-contained sentence: pronouns resolved, dates made absolute
    category: str  # attribute / preference / possession / relationship / event / plan / other
    cardinality: Literal["single", "multi"]  # single: new replaces old; multi: accumulate
    negated: bool  # "no longer likes ..."
    valid_from: str | None  # event time: when it became true (ISO date, may be month or year only)
    valid_to: str | None  # event time: when it stopped being true (None = still true)
    recorded_at: datetime  # record time: when the system learned it (conversation time)
    invalidated_at: datetime | None  # record time: when the system stopped believing this version
    superseded_by: str | None  # which fact replaced it
    supersede_reason: Literal["changed", "corrected"] | None  # reality changed / knowledge changed
    importance: int  # 1–5
    entity_ids: list[str]
    source_turn_ids: list[str]  # provenance: which raw turns support it


class CoreMemory:  # core memory: a materialised view of important, current facts (D-022)
    text: str
    fact_ids: list[str]


class MediaItem:  # an image (video and audio as extensions)
    id: str
    sha256: str
    path: str
    caption: str
    ocr_text: str
    tags: list[str]  # context-aware description
    entity_ids: list[str]
    source_turn_id: str
    ts: datetime
```

Notes:

- **Event times are partial dates.** Conversations often say only "May" or "last year", and forcing a day would create false precision. ISO strings in one format compare lexicographically; dates of different precision compare at their common precision ("2025-05" vs "2025-05-07" compares months only).
- **No confidence field.** LLM self-reported confidence is poorly calibrated. Trust comes from provenance instead: how many turns support a fact.
- **One view for retrieval units.** Turns, facts, session summaries and entities are all `MemoryUnit`s, distinguished by `kind` in the vector index, with time, session and validity in the payload for filtering.

Every record belongs to a **memory space** (`space_id`): one user in production, one sample in evaluation, strictly isolated from each other.

## 5. Write path

### 5.1 Online write (synchronous, no LLM)

1. Append the raw turn to the log (SQLite), with session and timestamp.
2. Embed the text and write it to the index.
3. Images: store the original by content hash, compute the visual embedding, queue a captioning job.
4. Update working memory.

**All or nothing** (second audit round after N12, D-069). The steps that can fail (storing the photo, embedding, writing vectors) run first, and one transaction at the end writes the space, session, turns, image records, rounds and consolidation job. If anything fails, nothing is committed: vectors written without a record and photos nothing refers to are removed on a best-effort basis, so the caller can simply retry without writing twice.

Target: writing never blocks the conversation; under 100 ms per turn (excluding model cold start).

### 5.2 Asynchronous consolidation

**Trigger.**

- Online writing only registers a consolidation job per session (milliseconds, no LLM).
- Background threads run jobs when idle (`ConsolidationWorker`). Jobs can also run synchronously: the `cuemem consolidate` command and evaluation do this.
- Up to `consolidation.background_workers` spaces (default 4) consolidate at once; one space is still one run at a time, in session order. Repeated submits of a space are merged, and a submit while it runs triggers exactly one more run. Provider concurrency is still capped per provider, and peak memory is up to 4 loaded run states (third audit round, D-070). Failed extractions are retried with backoff, logged once per space and kind of error, and listed by `Memory.status()`.
- The idea follows Sleep-time Compute: move computation to idle time to cut online latency.

**Pipeline** (D-020):

```text
pending jobs ──parallel──▶ A extract: one LLM call per session → facts, entities, summary (stateless, cacheable)
             ──batched───▶ B embed: all new statements, entity descriptions and summaries at once
             ──serial────▶ C in session-time order: entity linking → fact reconciliation
                                → vectors first, then the database (one transaction)
             ────────────▶ D regenerate the core memory
```

| Step | What it does | Cost control |
|---|---|---|
| Extraction | One structured LLM call. Facts: subject–predicate–object, a self-contained statement, category, cardinality, event date, importance, source turns. Also entities and a session summary | qwen3.8-flash (D-018); only the first 600 characters of each assistant reply; response cache, so a shared session is extracted once |
| Entity linking | Merge new entities into existing ones ("Max", "Maxie"); "the user's name is Lin Xiao" makes "Lin Xiao" point to the user | Exact name or alias match → description embedding similarity → an LLM only in the grey zone |
| Fact reconciliation | Compare with existing facts: ADD / SUPERSEDE / MERGE / NOOP (§5.3) | Merge, negation and multi-value accumulation by rule; supersession confirmed by an LLM, with a session's grey-zone facts batched into one call, each with the message it came from (to tell a correction from a change, N11) |
| Multi-granularity index | Facts, summaries and entities go into the vector index; facts serve as "augmented keys" pointing back to raw turns | Batched embedding plus a content-hash cache |
| Profile update | Pick important, currently valid facts and generate a length-capped core memory | Rules, no LLM (D-022) |
| Image captions | A VLM writes a caption, OCR text, tags and entity binding, seeing the image together with the turns around it (§7) | Each image processed once |

Engineering constraints:

- **Idempotent**: job key = memory space + session + session content fingerprint + pipeline version.
- **Replayable**: derived memory can be cleared and rebuilt from the raw log; a pipeline version change triggers a rebuild automatically.
- **Atomic**: each session's results commit in one transaction; a failed session is retried without affecting others.
- **Cached**: LLM responses and embeddings are all cached, so rebuilding is nearly free.

"Fact-augmented keys" come from a LongMemEval finding: extending retrieval keys with extracted facts, while still returning raw turns as evidence (keys separate from values), improves recall.

### 5.3 Knowledge updates: bi-temporal validity

Each fact has two timelines (after Zep / Graphiti):

- **event time** `valid_from` / `valid_to`: when it held in reality;
- **record time** `recorded_at` / `invalidated_at`: when the system learned it and when it stopped believing this version.

An example from `examples/toy_conversation.en.json`:

| Time | The user says | What the system does |
|---|---|---|
| 2025-03-01 | "I'm a product manager at an internet company in Beijing." | ADD works_at = an internet company in Beijing; ADD job_title = product manager |
| 2025-09-10 | "I changed jobs last month. I'm now an AI product manager at an e-commerce company in Singapore." | The new fact works_at = e-commerce company in Singapore has valid_from = 2025-08. It conflicts with the old job, and an LLM confirms an update → SUPERSEDE the old job: valid_to = 2025-08, invalidated_at = 2025-09-10 |
| Question | "Where do I work now?" | Use the current fact |
| Question | "What did I do before I changed jobs?" | Use the superseded fact (history is not deleted) |
| Question (event time) | "Where did I work in June 2025?" | `valid_at = 2025-06` → Beijing |
| Question (record time) | "On September 1, where did the system think the user worked?" | `known_at = 2025-09-01` → Beijing (the job change was not known yet) |

**There are two kinds of supersession** (D-019), which is exactly why two timelines are needed:

- **Reality changed** (moving, a new job, a new personal best): the old fact's `valid_to` = the new fact's start.
- **Knowledge changed** ("sorry, I meant ...", added detail): the old fact only gets `invalidated_at`. Its `valid_to` stays empty, because it never "ended" in reality; the system simply stopped believing it.

A predicate's **cardinality** decides the conflict rules:

- **Single-valued facts** (registered predicates such as where one lives or works, and job title, or facts the extractor marks single):
  - An existing fact with the same predicate, or a similar one, goes to an LLM, which chooses update / correct / duplicate / add.
  - The first design superseded registered predicates by rule. On dev100 the sampled precision was only about 10%, because the extractor's predicate labels are unstable, so it was changed to LLM confirmation (D-019).
- **Registered multi-valued predicates** (likes, owns, has visited ...) accumulate. An explicit negation of an existing fact ("no longer likes ...") supersedes it by rule.
- **Unregistered multi-valued predicates** go to the LLM when highly similar to an existing fact; otherwise they are added.
- **Merging** happens only for the same predicate and object, or a nearly identical statement with the same object, and only appends provenance. Embeddings are insensitive to numbers, so similarity alone cannot justify a merge.
- **Temporal order** uses event times when both facts have explicit ones, and otherwise the order in which they were said.

**A late-arriving old message does not overwrite the current state.** If a new fact's event time is earlier than the current fact's (e.g. an older conversation imported later), it is written as a historical version. Tests check that importing the newer session first and the older one second gives the same result as chronological import.

**Corrections and expiry in the SDK** (second audit round after N12, D-069). `update(memory_id, data)` does not rewrite in place. It logs a user message "Correction: …", and the raw layer records which fact it corrects. Reconciliation takes one of the facts extracted from that message and supersedes the old fact as a change of knowledge, without asking an LLM (reconciliation alone would just add to a registered multi-valued predicate). `expiration_date` and `metadata` are stored in the raw layer and kept out of the extraction prompt; expired memories are excluded only by the SDK's search, get_all and profile, and `valid_to` is not rewritten.

### 5.4 Forgetting and privacy

- **No automatic deletion.** Inspired by MemoryBank's Ebbinghaus curve, "memory strength" decays with time and grows with access, but it affects **ranking** only and never deletes data (N4).
- **Explicit forgetting** (D-021). Derived memory is a function of the raw log, so forgetting means deleting from the log and replaying the derived memory from what remains.
  - Old facts that the deleted session had superseded become current again automatically.
  - Caches are purged too: the deleted sessions' extraction cache, the space's answer cache, and embeddings of text that no longer appears. Of the space's reconciliation and entity-linking cache, only the entries the replay hits exactly are kept (a hit means that request was computed from the remaining log alone), and the rest are deleted (since the second audit round after N12, D-069; before, the whole cache was cleared and paid for again on replay).
  - SQLite runs with `secure_delete`; after forgetting, the memory database and both caches commit and truncate their WAL (no `VACUUM` since the audit after N12, D-068). Tests read the bytes of the database files and their WALs to confirm the deleted content is gone. If the replay fails halfway, the cleanup still runs before the exception reaches the caller.
  - **Three phases** (third audit round, D-070): first the network steps, with the database untouched (embed and upsert the rounds that lose a turn; delete the vectors of replaced rounds, derived units and doomed images); then the LLM cache; then one SQLite transaction that swaps the rounds, deletes the turns, clears derived memory and re-registers the space's jobs (`Consolidator.requeue`, shared with rebuild: only sessions that had a job, with their outcomes kept). Files, the replay and cleanup come after the commit, and unfinished replay work goes to the background worker. A failure before the commit deletes nothing. Rebuild likewise deletes derived vectors before its clear transaction, so a vector-store error leaves the space intact.
  - **Locks:** a space's consolidation lock first, then its read gate. Forgetting waits for reads in flight and new reads queue behind it, but reads go on while a forget waits for a background consolidation; forgetting inside a held read raises RuntimeError. A raised search (L0 → L1) holds one read across both retrievals.
  - **Deleting a space** (`delete_all`, `reset`) also deletes its traces with their query vectors, its turn-number rows, its consolidation fingerprint and the open-session marks a crash left.
- **Whole rounds, following citations** (N8, D-046).
  - Deleting a fact or an image deletes the round it came from: the user's words and the assistant's reply.
  - The agent records which memories every reply cited (each profile line is citable too). A later reply that cited deleted memory is deleted as well, and the chase continues from there.
  - Open sessions drop the deleted content, and trace entries containing the deleted text are removed (trace files are split into lines on `\n` only, so a question containing U+2028 is found too).
- **Uncited restatements** (N10, D-051). A reply that restated the information without citing it cannot be reached through citations: for example, an assistant saying in passing "her birthday is October 3, order soon", or a greeting in an open session. Such replies are now found in the raw log:
  - for each deleted fact, the nearest rounds are retrieved, including open sessions;
  - a cheap model goes through their messages one by one and confirms which still reveal the information. Mentioning the same person or topic does not count.
  - Only those messages are deleted, and no citation cascade starts from them (revised after N11): what else the user said in the same round stays. After the rebuild, any fact that grows back from elsewhere has its source deleted as well, for up to 3 rounds.
- **Gaps are marked on re-extraction** (after N11, D-059). Once the source is deleted, a remaining message may point into it with a pronoun ("I'll get those for her"), and the model would pin it on someone else.
  - **How gaps are found:** message numbers are never reused, so the gaps left by forgetting are where the numbering jumps.
  - **What the extractor sees:** re-extraction marks each gap `[deleted]` and says that a pronoun next to a gap must not be resolved to someone else.
  - **Effect on the cache:** sessions that were never edited send byte-for-byte the same requests, so all cached extractions stay valid.
- **Isolation.** Memory spaces share nothing.

## 6. Read path

### 6.1 L0: resident context

Every turn includes the system instructions, the core memory (user profile) and working memory (recent turns). The first two form a **stable prefix** that hits the provider's prompt cache. Many turns need only L0: greetings, general questions, follow-ups to the previous turn.

**Implementation** (`ask(mode="l0")`, evaluation method L0): the core memory only, no retrieval.

- The benchmarks ask single questions, so there is no working memory.
- On the development sets, L0 answers 13% of LongMemEval, 7.8% of LoCoMo and 23% of MemLens questions: unanswerable questions, and those whose answer is in the core memory. L1 answers all of those too.
- So on memory questions L0 only saves money; its real use is for queries that need no memory (§6.3, D-039).

### 6.2 The memory probe

The probe runs every turn and **calls no LLM**. It does one query embedding (needed anyway), one hybrid retrieval (top-K), and rule-based cue extraction, and outputs the controller's features. The design target was tens of milliseconds; the measured times below are dominated by the embedding API (about 110 ms per query) and local Qdrant search in large spaces.

**Implementation** (`read/probe.py`): the probe is one L1 scoring pass, and every route's raw scores travel with the candidates. When the controller picks L1, those candidates are packed directly, with no second retrieval. There are 54 features:

- **26 question cues**: first and second person; references to past conversations; asking about one's own actions; aggregation, duration, order and comparison; temporal expressions and their width; question words; visual-detail words; requests for recommendations.
- **24 probe-feedback features**:
  - top-1 similarity per route (turns, facts, summaries); the gap to the second hit; z-score against the candidate pool; the drop over the top 10;
  - session spread, score concentration and time span of the top 10 turns, and how old the strongest hit is;
  - how many facts are close to the top-1, superseded facts among them, entities named in the question;
  - top-1 for photo captions and for text-to-image search.
- **4 memory-size features**: sessions, turns, current facts, images.

Two post-answer features (claims insufficient evidence, sounds like "I don't know") are used only for reactive escalation. Since N11 there is also a position cue, `q_scene`, used only by controller v2's floor rule and not by the classifiers.

Probe time at p50 is 56 ms on LoCoMo, 169 ms on MemLens and 271 ms on LongMemEval: larger spaces are slower with local Qdrant. The planned "conversation state" features were not implemented, because all three benchmarks ask single questions and have no conversation state to evaluate.

### 6.3 The retrieval controller ★

**Action space**: A = {L0, L1, L2} × {re-perception on / off}. Multimodal memory (N6) was built before the controller (N7), so the controller decides from the start whether to look at the originals again.

**Decision rule** (utility maximisation):

$$
\pi_\lambda(x) = \arg\max_{a \in A} \Big[ \hat{P}(\text{correct} \mid x, a) - \lambda \cdot \hat{c}(a \mid x) \Big]
$$

- $x$ is the probe's features, $\hat{P}$ the calibrated probability that action $a$ answers correctly, and $\hat{c}$ its expected cost.
- **λ is a run-time knob.** Settings differ in their tolerance for cost: customer service wants low latency, a deep personal assistant can spend more. Adjust λ without retraining; sweeping λ traces the Pareto curve.

**Where labels come from** (offline, full information): offline, every action can be run. Each training query is answered with L0, L1 and L2, judged, and its correctness and real cost recorded.

**What N7 built** (`read/controller.py`, `evals/routing.py`; D-038 to D-040; results in [the N7 results](../results/n7-controller/README.md), in Chinese):

- **Two stages instead of fitting each level's success directly.**
  - The outcome labels turned out to be very noisy. The evidence for L1 and L2 on the same question differs only slightly, yet the answer can flip: on the 585 development questions, L2 rescued 45 and broke 17.
  - So the probe features first estimate the probability that the message needs no memory (the gate, binary), and its reasoning type (single fact, multi-session / multi-hop, temporal, knowledge update, preference, open inference, unanswerable; multinomial logistic regression).
  - Then each level's accuracy on that type is looked up: P̂(a | x) = Σ_t P(t | x) · acc(a | t). The re-perception gain per type comes from MemLens.
  - At run time only numpy is needed, and the parameters are one JSON file.
- **Default λ = 30**: over the development memory questions combined, the largest λ whose accuracy is no lower than always-L1 ("the cheapest that is no worse than always retrieving").
- **Three ways to run it**:
  - the learned controller (default, no LLM call);
  - a hybrid that asks a zero-shot LLM router when the gate is unsure;
  - rules and the zero-shot LLM router as baselines.
- **Findings** (development sets, confirmed on the test sets in N9):
  - The controller's value is in **whether to recall**. In a stream with 20% memory questions it is 41–45% cheaper than always retrieving, with no loss of memory accuracy.
  - **Between L1 and L2 there is little to gain.** Even routing by the true question type saves only 29% against always-L2 and loses 0.9 points. The learned depth decision did not beat the rules.
  - **Escalating on model uncertainty is worst** (77.6%).
  - **The gate does not transfer zero-shot to a new conversation style** (LoCoMo across datasets: 86.5% → 43.9%). Twenty labelled target examples bring it back to 85.7%.
  - **Reactive escalation on top of prediction** adds 0.5 points for +10% cost and is off by default.
  - **With a different reader (qwen3.8-flash), agentic retrieval (B3) became the best method**, so H3 failed. The agent therefore keeps it as an escape hatch (D-041, D-042).
  - **Known issues found on the test sets (N9), recorded and not fixed at the time** (D-048):
    - the photo gain is added and clipped at 1.0, so L1 + photos ties with L2 + photos and the cheaper one always wins;
    - the per-level accuracy is calibrated on the main reader.

**N11 revision: controller v2** (D-052; the N9 issues fixed on the development set, with the test sets only replayed post hoc):

- In a space with photos, the controller no longer adds a photo gain to the no-photo success rate and clips at 1.0 (which clipped both levels to 1.0 and lost their order). It looks up an image-space accuracy table instead: reasoning type × {L0, L1, L2, L1 + photos, L2 + photos}, estimated only from training questions with photos and shrunk towards the overall mean in the same way.
- In a space with photos, a question with a visual cue or a position cue (a new feature, `q_scene`, used only by this rule and not by the classifiers) is lifted from L0 to L1 + photos.
- The gate, the reasoning-type model and λ are unchanged. The frozen v1 is kept as `controller_model_v1.json` for reproducing N7 and N9; the `retrieval.controller` setting chooses the model file.
- Development set 81.2% → 81.7%; post-hoc test replay 81.6% → 81.9%, all of it on MemLens. For a new reader, recalibrating the accuracy table does not help, while setting λ per reader does (λ = 15 recovers most of the gap to always-L2 at 16–24% lower cost).

**After N11: controller v3** (D-058):

- **The problem:** the image-space accuracy table comes from MemLens, where nearly every question is about an image. As a result, v2 attached photos to 96–98% of questions in spaces with photos. In the demo, the budget question also got the photo of a friend's shoes, and the agent answered the previous question (find look-alike shoes) again.
- **The change:** v3 adds a threshold. Photos are considered only if the question has a visual cue or the best caption match is at least 0.64, a value chosen on the development set. Otherwise the controller chooses among the actions without photos.
- **The result:** development accuracy is unchanged, and the share of questions with photos falls from 98% to 31%. In the post-hoc test replay, cost per 1,000 questions is 16% lower and accuracy moves by about 1 point either way (n.s.).
- **Reproducibility:** the frozen v2 is kept as `controller_model_v2.json`.

**Online learning (extension).** In production only the chosen action's outcome is observed (partial feedback), so the problem becomes a contextual bandit (LinUCB / Thompson sampling):

- the reward is correctness − λ · cost;
- user corrections ("you got that wrong") are negative feedback;
- validate first by simulation on offline logs.

### 6.4 L1 fast recall (no LLM call)

The implementation is in `read/l1.py` (evaluation method B4; since N5 the default is B4-nobm25).

1. **One multi-granularity index.** Turns, facts, session summaries, entities and (from N6) photo captions are all memory units in one vector collection, distinguished by `kind`.
2. **Multi-route recall.** Turns, facts and summaries each have a dense route (qwen3.7-text-embedding, D-014), top 30 each. BM25 was implemented as a sparse route but, with qwen3.7, did not improve recall at any budget; it is off by default (D-026, revised).
3. **Fusion** (D-025).
   - Within a modality, take the max: a turn's score is the max of its own score and its source facts' scores. That is the fact-augmented key: the key is the fact, the value is still the raw conversation.
   - Across modalities, use reciprocal rank fusion.
   - Rank-only fusion by unit type pushes strong direct hits down: NDCG was 4 points lower on dev100, and "what the assistant said" questions fell from 96% to 52%.
4. **Temporal awareness** (D-027).
   - Rules parse temporal expressions and convert them to an interval relative to **the time of the question**, not the system clock.
   - Turns in the interval, or turns whose source facts' event time falls in it, get a boost discounted by the interval's width.
5. **Time-aware packing.**
   - Evidence is shown in chronological order.
   - Each item carries its absolute date, **how many days before the question** it was said, and its validity, so the reader does not have to do date arithmetic.

   For a question asked on 2025-10-10:

```text
Notes from memory: short facts summarized from earlier conversations ...
[f1] (said on 2025-03-01, 223 days before the question; no longer true from 2025-08, see [f2]) The user works at an internet company in Beijing.
[f2] (said on 2025-09-10, 30 days before the question; since 2025-08; current) The user changed jobs in August 2025 to an e-commerce company in Singapore.

Retrieved conversation excerpts:
[m1] 2025-09-10 (Wed) 19:00 · 30 days (about 4 weeks) before the question · session s4
User: Some news: I changed jobs last month. I'm now an AI product manager at an e-commerce company in Singapore.
```

6. **Budget** (D-029).
   - Turns are added from the highest score down until the token budget is used: 4k by default, at least 1 turn, at most 40.
   - On top of that come 10 facts (completed with their version chains) and the core memory.
   - A "turn" differs a lot across datasets: about 490 tokens on LongMemEval (8 turns fit), but a single sentence of about 40 tokens on LoCoMo (40 fit). A fixed count would give the two datasets context that differs more than tenfold.

**What we learned.**

- On LoCoMo's development conversations, L1 was clearly better than naive retrieval with facts (84.9% vs 78.7%, p = 0.001), and multi-hop went from 62% to 81%.
- The largest lever was packing to a token budget: taking a fixed 10 turns scores only 77.1% on LoCoMo.
- Temporal boosts and BM25 made no significant difference on either dataset.
- Retrieval p95 is 132 ms, of which about 110 ms is the query embedding.

### 6.5 L2 deep recall (with LLM calls)

The implementation is in `read/l2.py` (evaluation method B5). Each query still uses L1's scoring. The extra cost is 2–4 calls to a cheap model (qwen3.8-flash), plus the reader writing notes first.

1. **Query planning** (`read/plan.py`, one call, structured output):
   - the answer type: count, sum, list, date, duration, order, comparison, yes/no, single fact, recommendation;
   - whether the answer must be **exhaustive**: count, sum and list questions need every item scattered across sessions;
   - 1–4 sub-queries, each looking for one piece and phrased the way the user would have said it ("I viewed a house", not "number of house viewings");
   - a time interval: the rule-based one when the rules can parse it, otherwise the model's absolute dates, checked to be before the question and of plausible width;
   - target entities, used as seeds for graph diffusion.

   If planning fails (invalid output, content filter), it falls back to the rule-based plan and retrieval proceeds.
2. **Multi-query recall with "best rank" fusion.**
   - The original question and each sub-query get an L1 scoring pass, with all queries embedded in one batch.
   - Each candidate takes its best rank score across queries, plus a little of its rank-score sum to break ties.
   - Each sub-query looks for a different piece ("how old now", "which year did they move to the US"), and each piece's best hit should make it into the context.
   - Summing rank scores (RRF) rewards turns that rank middling for several queries; on dev100, multi-query recall with RRF gained nothing (D-030).
3. **Packing**: a 6k-token budget, at most 60 turns (L1: 4k, 40). Count, sum and list questions get more facts (25, about 30 tokens each). Evidence is still shown chronologically, so ranking only decides which turns fit.
4. **Sufficiency check and follow-up retrieval** (one call each, at most 2; after CRAG's "evaluate, then correct").
   - A checker reads the question and compressed evidence (the facts, and the first 700 characters of each turn) and judges whether it suffices.
   - If not, it says what is missing and proposes new queries; turns found by follow-up retrieval get another 1k tokens.
   - The prompt is explicit that needing a calculation (a sum, a date difference, an age) is not insufficient, and neither is a conflict between old and new values (the latest wins).
5. **Note-taking before answering** (Chain-of-Note style).
   - The reader's system prompt is identical to the other methods'. A reading instruction generated from the plan is appended to the context:
     - first take numbered notes with dates and relevant details;
     - for counts, list each item, count the same thing once, and include things mentioned in passing;
     - for time questions, compute differences explicitly from the "N days before the question" labels;
     - mark evidence inside the asked-about period with ★.
   - The notes field comes before the answer in the output format, so the model writes notes first.
6. **Graph diffusion and reranking** (implemented, off by default).
   - The memory graph is tripartite: entities, facts and turns. It adds mention edges (an entity name appears in a turn) and edges between semantically close facts. Hub entities connected to too many facts are left out (the user, LoCoMo's two speakers).
   - Seeds are the entities named in the question and the top facts and turns of the first pass. Personalised PageRank runs on scipy.sparse, after HippoRAG.
   - Reranking uses QwenCloud's qwen3-rerank on the top 30 fused candidates.
   - Neither improved recall within the budget on dev100 (D-031, D-032), nor on the test sets (report §7).
7. **Abstention**: the reader's rule (say "I don't know" if the evidence lacks the answer), with no separate threshold. False refusals (answering "I don't know" to an answerable question) are counted.

**Tool interface** (`read/agent.py`). It powers baseline B3 (agentic retrieval), the agent's escape hatch, and the MCP server.

- `memory_search(query)` returns numbered, dated facts and conversation excerpts: L1 retrieval, 2k tokens per call, deduplicated across calls.
- `date_diff(start, end)` returns the difference between two dates.

**Rule-based controller** (`read/controller.py`, B6r, N7's baseline). It looks only at question cues, with no LLM and no retrieval results:

- questions about counts or aggregation, durations, order (first, last, before, after ...) and comparisons go to L2;
- everything else goes to L1;
- the rules were written before seeing L2's per-question results.

**What we learned.**

- The largest single lever is note-taking: with the same L2 evidence and no notes, dev100 loses 8 questions.
- Of the retrieval steps, only follow-up retrieval mattered on the development set (+4 / −0). Planning, graph diffusion and reranking were not significant end to end. On the test sets, follow-up retrieval did not reproduce either (report §7).

### 6.6 Context assembly and the reader

**The reader in evaluation** answers one question at a time.

- **Input**: system prompt + core memory + evidence with IDs, dates and validity + the question.
- **Structured output**: `answer`, `citations` (IDs of the memories used) and `evidence_sufficient`; L2 has a notes-first format (D-030).
- **Why citations**: they make every answer traceable, and they make "did retrieval hit the evidence" measurable.

**The conversational agent** (N8, D-043) assembles partitions, each with a token budget:

```text
[System]          role, memory rules, citation format, domain notes + tool specs          ┐
[Profile]         profile snapshot (≤ 600 tokens), numbered [p1]..., taken at session start │ stable prefix: append-only,
[Working memory]  raw turns of this session (≤ 3,000 tokens; evicts half at once when full) ┘ provider prefix cache keeps hitting
[Latest message]  date + this turn's recalled memories (≤ 4,000 tokens, L2 6,000; re-perceived photos) + the user's words and photos
```

- **Recalled memories** go only into the last message and are recomputed next turn, never entering the history. Assistant replies in the history drop their citation markers.
- **Working memory** is excluded from the probe and from retrieval, so recall slots go to content not already in context. Turns evicted from working memory are in long-term memory and can be retrieved.
- **Measured**: in a 16-turn session, 70% of input tokens hit the provider's prefix cache with the stable layout, against 52% for a layout that puts recalled memories in the system prompt. With heavy recall and a short history the two are about the same.

### 6.7 Trace

Every turn writes a structured trace:

- the probe's features;
- each action's utility estimate and the final decision;
- the memories hit by each route, with scores;
- tokens per partition;
- per-stage latency;
- LLM call counts.

It feeds the demo's memory-trace panel, error analysis and the controller's training data.

### 6.8 Agent runtime and services

For each user message (`src/cuemem/agent/runtime.py`):

1. **Recall.** The controller picks a level from the probe (L0 / L1 / L2, and whether to look at photos again when there are photos), and recalls while excluding working memory.
2. **Assemble** the request from the partitions of §6.6.
3. **Answer.** The model can call tools, for at most 4 steps; the last step may not call a tool.
   - `memory_search` is the escape hatch for what the controller did not fetch (D-042);
   - `date_diff` computes dates;
   - domain tools come from the scenario: the shopping demo can search products by text and find visually similar products.
4. **Write back.**
   - The exchange is appended to the raw log and indexed online, photos and image embeddings included, along with which memories the reply cited (provenance).
   - An open session is not consolidated; when it ends, the whole session goes to background extraction (D-045).

Each turn returns a trace:

- the controller's decision and reasons, and each level's expected accuracy and utility;
- tokens per partition;
- recalled and cited memories;
- tool calls;
- usage (including provider prefix-cache hits) and per-stage timing.

Services (`src/cuemem/api/`):

- **FastAPI**: sessions and chat (with photos), memory browsing (profile, fact timeline, entity graph, photos, sessions), forgetting, consolidation, and a retrieve-only `recall`. API docs at `/docs`. After N12 came a Mem0-shaped `/api/v1` that mirrors the SDK (what `MemoryClient` calls) and scoped access tokens (second audit round, D-069; see the [API reference](api.md)).
- **Gradio UI**, mounted on the same app (an optional `ui` extra since the second audit round; without it the server offers only the APIs and the memory proxy): chat, a memory-trace panel, and a memory browser where a fact, a photo or a session can be forgotten.
- **MCP server** (stdio, and since the third audit round Streamable HTTP at `/mcp` of `cuemem serve`, behind the same tokens and scopes as `/api`): `remember` / `recall` / `forget` / `profile`, so other agents can use the memory directly. `recall` lets the controller pick a level and returns the numbered, dated memories as they are, without calling an answering model (D-047, D-070).
- **Guards in front of the app** (D-068, D-070), in this order: the Host allow-list, same-origin browser requests, then request body limits (`api/limits.py`: imports 1 GiB, `/api`, `/v1` and uploads 64 MiB, everything else 8 MiB; scoped keys import at the body limit and have an in-flight budget, 429). `/api` and `/api/v1` check the token before reading the body. Under `/api/v1` every refusal has the `V1ErrorBody` shape. Images above Pillow's pixel limit are refused before they are stored or sent to a model, and decodes above 16M pixels run two at a time. The mounted UI turns Gradio's own file-path checks on (`demo.has_launched`).
- **Idle timeouts:** under `cuemem serve`, chat sessions, `/api/v1` runs and the mounted proxy's sessions idle for `--idle-minutes` (30) end and go to extraction; a sweep thread checks every minute and, with `traces_retention_days`, also deletes old trace files with their query vectors.
- **Demo script**: `demo/shop/script.json`, replayed and checked with `cuemem demo`.

## 7. Multimodal memory

The implementation is spread across the write, retrieval and answer code, and MemLens is wired into evaluation (D-034):

- **writing**: `write/online.py`, `write/caption.py`, `store/media.py`, `embed/multimodal.py`;
- **retrieval**: `read/l1.py`;
- **answering**: `read/reader.py`.

### 7.1 Three representations of an image

| Representation | Purpose | How it is produced |
|---|---|---|
| Original | Re-perception at answer time | Stored by content hash in the media store (one copy per image), with one record per occurrence |
| Visual embedding | Text-to-image and image-to-image search | QwenCloud's tongyi-embedding-vision-plus (1,152 dimensions, images and text in one space), a separate image collection per space; computed during online writing |
| Context-aware caption + text in the image + keywords | Text-retrieval key; readable evidence in L1 / L2; input to fact extraction | Generated during consolidation by qwen3.8-flash from the image **and the turn before and after it**; an isolated, image-only caption is also generated for comparison (H6) |

**Entity binding.** During fact extraction, the caption is filled back into the turn where the image appeared. Information in the image ("a red number 390 double-decker bus", "$120 on the receipt") then reaches facts and the entity graph too (D-035).

QwenCloud does not offer qwen3-vl-embedding, which would embed an image together with its conversation (it returns 403). So "context-aware" can only be achieved through captions.

### 7.2 Retrieval

- **Captions as keys.** A context-aware caption is a retrieval unit. When it is hit, its score goes to the turn that contains the image, the same way fact-augmented keys work (max within a modality).
- **Text-to-image search.**
  - The question is encoded with the same image–text embedding model and searched against the image collection.
  - The turn containing the image joins RRF as a third modality (weight 0.5; on by default, skipped when the space has no images).
  - On the MemLens development set it raised recall of the turns holding key images from 56% to 85%, the biggest lever in multimodal retrieval (D-036).
- **Image-to-image search** has an interface (`VectorStore.search_images`) but is not evaluated, since MemLens questions are text.
- Results are packed together with text memory under the token budget.

### 7.3 Re-perception at answer time

- **In place.** Images in the turns that enter the context are written as `[Image i3: caption Text in the image: …]`.
- **With re-perception on**:
  - the top 3 originals by relevance (caption-key and text-to-image ranks) are resized to 768 pixels on the long side and attached after the text evidence, before the question;
  - the matching positions are marked "see picture k";
  - the reader is still deepseek-flash, which can read images, so methods differ only in whether they show the originals.
- **Cost.** Image tokens count as the answer's input tokens and enter the Pareto analysis. When attaching photos is worth it is the controller's decision.

**What we learned** (MemLens, the official 195-question subset; D-034 to D-037):

- **Re-perception** adds 12 points at 32K and 7 points at 128K over captions alone (H5 supported).
- **L2 with re-perception** scores 65.6% / 64.6% at 32K / 128K, against 52.3% / 46.2% for a long-context model looking at the images directly, at 1/2 and 1/9 of its cost.
- **Stability with length**: from 32K to 128K, full context and L1 degrade (p = 0.012 and 0.016, uncorrected; they stay significant after a Holm correction over these three pairs, but not once the two other methods compared across lengths are counted; see the note on D-037); only L2 holds.
- **Two intuitions were not supported**:
  - context-aware captions did not beat isolated captions (H6);
  - text-to-image search greatly improved retrieval but not accuracy.

### 7.4 Extension: video and speech (☆)

Split video into shots, take keyframes plus speech transcripts, and form clip-level episodic memories that reuse the same indexes, controller and re-perception. This maps directly onto short-video platforms. M3-Agent (ByteDance Seed) has done systematic work on long-video memory and is a useful reference.

## 8. Evaluation plan

### 8.1 Datasets

| Dataset | Modality | Size | Role |
|---|---|---|---|
| LongMemEval_S (cleaned, MIT) | Text | 500 questions, five abilities (information extraction, multi-session reasoning, temporal reasoning, knowledge update, abstention), 30 of them unanswerable; each question's history has about 48 sessions, 243 user turns, about 100k tokens | Main text benchmark |
| LoCoMo (CC BY-NC 4.0) | Text (910 turns carry images, folded in as BLIP captions) | 10 long two-person conversations of 370–690 turns; 1,986 questions: single-hop 841, multi-hop 282, temporal 321, open-domain 96, adversarial 446 | Second text benchmark; cross-dataset generalisation of the controller. Adversarial questions excluded, as in the literature |
| MemLens (CC BY 4.0) | Interleaved images and text | 789 questions, five abilities, four lengths from 32K to 256K; an official 195-question memory-agent subset | Main multimodal benchmark |
| Mixed stream (built in N7) | Text / images | Memory questions plus queries that need no memory: general requests from databricks-dolly-15k (CC BY-SA 3.0, line numbers only) and a same-topic general question generated for each memory question (hard negatives); unrelated queries borrow the memory questions' spaces | Evaluates "should we recall at all" (the L0 decision, D-039) |

**Splits**:

- **LongMemEval_S.**
  - 100 questions stratified by (type, unanswerable) form the development set `dev100`, used for tuning and for training the controller; it has 6 unanswerable questions.
  - The other 400 form `test400`, used only in the final evaluation. The split files are committed in `evals/splits/`.
  - Ten questions from dev100 form `pilot10`, for dry runs and cost estimates.
- **MemLens.** The official 195-question memory-agent subset is the test set, so results compare with the paper's memory agents. The development set is 100 questions stratified by type from the 594 outside the subset (D-034).
- **LoCoMo.**
  - The first 3 conversations (385 non-adversarial questions) were the development set in N3–N7.
  - The last 7 (1,155 non-adversarial questions) are the test set; the split files are `locomo_dev3` and `locomo_test7`.
- **Swapping the reader (N9).**
  - 100 questions stratified from test400 by (type, unanswerable): `test100r`.
  - 200 questions stratified by type from the LoCoMo test set: `test200r`.

### 8.2 Metrics

- **Quality.** Accuracy, overall and by question type.
  - The LLM judge copies the official LongMemEval template verbatim, with regression tests against the official code. The calling convention is the same too: one user message, temperature 0, max_tokens 10, correct if the reply contains "yes".
  - The judge is fixed to qwen3.8-max, from a different family than the main reader (D-008).
  - LoCoMo has no official LLM judge; it uses the same template, plus token F1 (D-015).
  - MemLens uses its official judge prompts (D-034).
- **Retrieval** (retrieval methods only, excluding unanswerable questions): session recall (any / all), turn recall (any), and turn NDCG. The unit is a turn; a turn is relevant if it contains an evidence utterance.
- **Cost**: input and output tokens per question (image tokens included), LLM calls, money; write cost per thousand turns.
- **Latency**: p50 / p95 per stage (probe, L1, L2, reader).
- **Controller**: routing accuracy, regret, escalation rate.
- **Stability**: variation in accuracy across reader models.

### 8.3 Baselines

All methods use **the same reader and the same judge**:

| ID | Method | Notes |
|---|---|---|
| B0 | No memory | Lower bound; unanswerable questions come out right by default |
| B1 | Full context | The whole history in a long-context model (the most expensive; run once on development and once in the final evaluation) |
| B2 | Naive RAG | Turn-level dense top-k (k = 10), retrieval every turn |
| B2F | Naive RAG + fact memory (N3) | Same retrieval as B2, plus the write path's output: fact-augmented keys, facts with validity, core memory (D-023); four ablations |
| ORACLE | Evidence sessions only | A reference, not deployable: how much the reader gets right with perfect retrieval, separating retrieval errors from reading errors |
| B3 | Agentic retrieval (N5) | The reader has `memory_search` and `date_diff` and decides whether, what and how often to retrieve (ReAct), at most 5 steps; the last step must answer |
| B4 | This system, always L1 (N4) | No controller: L1 + facts + core memory, packed to a token budget; six ablations. Since N5 the default L1 has no BM25 (B4-nobm25) |
| B5 | This system, always L2 (N5) | No controller: planning + multi-query recall + sufficiency check and follow-up + note-taking; ablations and variants |
| B6r | This system + rule-based controller (N5) | Chooses L1 or L2 from question cues; the baseline for N7's learned controller |
| B4-text / -cap / -iso / -mm / -see | The multimodal ladder (N6, MemLens) | Text only (images as placeholders) → context-aware captions (-iso: isolated) → facts and text-to-image search (-mm) → re-perceive 3 originals (-see); B5-see is L2 with re-perception |
| L0 | Resident context only (N7) | Core memory (profile) only, no retrieval; the controller's cheapest level. B0 gives nothing at all |
| B6 | This system + learned controller (N7) | Probe + two-stage controller choosing among L0 / L1 / L2 and whether to re-perceive, by λ; sweeping λ draws the curve; default λ = 30 |
| B6h / B6z | Hybrid / zero-shot LLM router (N7) | B6h asks an LLM once when the gate is unsure; B6z lets qwen3.8-flash pick a level from the question text alone |
| B6a | Controller first, LLM may search more (N9) | B6 recalls, the recalled memories come with the question, and the reader may call `memory_search` for anything missing: the N8 agent's approach (D-042) |
| M0 | External baseline: Mem0 (N11) | Open-source mem0ai 2.2.1 writing with its own pipeline (one `add` per session; on LoCoMo one user_id per speaker, as in the Mem0 paper) and reading with its own `search` (top-30, full hybrid search); its LLM and embeddings replaced by the models of this system's write path; same reader and judge (D-055) |

### 8.4 Ablations

- **Planned in N0**: no fact extraction; no validity (overwrite on conflict); no PPR graph diffusion; no time-aware packing; no reranking; captions only (no re-perception); isolated image captions (no conversational context); controller feature subsets.
- **Write path** (N3, all based on B2F with the retrieval unchanged):
  - fact-augmented keys only (no facts or profile shown, as in the LongMemEval paper);
  - current facts only (equivalent to overwriting);
  - facts without validity labels;
  - no core memory.
- **L1** (N4, based on B4): a fixed 10 turns; a fixed 5 turns; a 2k-token budget; no temporal boost; no relative-time labels; no BM25.
- **L2** (N5, based on B5): no query planning (a rule-based plan, the original question as the only query); no sufficiency check or follow-up; no notes (the same reading as L1); with graph diffusion. Two comparisons were added:
  - L1 with L2's budget (B4-b6k);
  - L1 with notes first (B4-note).

### 8.5 Safeguards

- Temperature 0 and a response cache make results reproducible.
- Judge templates are copied verbatim, with regression tests against the official code.
- A hand audit of 50 judgements, half right and half wrong (`cuemem eval audit` exports them and computes agreement). N11 samples them from the N9 test-set judgements, stratified by dataset (`scripts/n11_audit.py`); all 48 labelled items agree with the judge, and 2 were left unlabelled.
- A second judge (N11): a model from another family (deepseek-flash) re-judges every answer of the main methods on the test sets, to check agreement and whether the ranking of methods changes.
- Repeated runs (N11): the main methods are rerun with the cache off, to measure how much the same configuration varies between two runs, as a yardstick for differences between methods.
- `pilot10` dry runs come first: read the errors one by one and check whether a problem lies in the method, the reader or the judge before running at full size.
- 95% bootstrap confidence intervals.
- Separate development and test sets.
- Every result carries its cost.
- Error attribution: errors of retrieval methods are split automatically into "retrieval miss", "hit the evidence but wrong while ORACLE is right" and "ORACLE also wrong". Finer categories are labelled by hand.
- When citing other reports, note the differences in setup: reader, judge and prompt differ, so numbers do not compare directly.

**Main figure**: the accuracy–cost Pareto curve. Each baseline is a point; B6 traces a curve as λ varies.

### 8.6 Research hypotheses

| ID | Hypothesis | Outcome (details in report §9) |
|---|---|---|
| H1 | For deciding how deeply to retrieve, probe feedback + query cues beat query text alone, and beat triggering on model uncertainty | **Half supported** (N7 / N9). True for "recall or not": AUC with question cues alone 0.79, probe feedback alone 0.96, both 0.98 (development); the frozen gate 0.990 on the test stream; escalating on model uncertainty was worst. Not true for "L1 or L2": the reasoning type depends mostly on phrasing |
| H2 | The controller reaches close to always-L2 accuracy at much lower average cost | **False on memory questions alone, true in the mixed stream** (N7 / N9). On the test sets B6 is 2.8 points below always-L2 at 31% lower cost; in the mixed stream it is 41% cheaper than always retrieving with no accuracy loss |
| H3 | Compared with agentic retrieval, the controller is more stable across reader models and has a larger advantage on small models | **Refuted** (N7 / N9). With three readers (qwen3.6-35b-a3b, qwen3.8-flash, deepseek-flash; deepseek-v4-pro not run for lack of balance), B3 fluctuates no more than the controller and loses only a little on the smallest model. The controller's per-level accuracy is calibrated on the main reader |
| H4 | Validity periods ("invalidate, don't delete") help knowledge-update questions without hurting questions about past states | **Not supported** (N9). On test400's knowledge-update questions, the full version, "current facts only" and "no validity labels" all score 96.5%. The write path gains 9 points overall, but not from validity labels |
| H5 | Re-perceiving originals significantly helps questions needing visual detail, and the controller can control the image-token cost | **Supported** (N6 / N9). +12 points at 32K, +7 at 128K, concentrated in information extraction; the photo-gain clipping makes the controller choose wrongly between L1 + photos and L2 + photos |
| H6 | Context-aware captions beat isolated captions, especially for entity binding | **Not supported** (N6). Slightly better retrieval; end to end the isolated captions got 3 more questions right (n.s.) |

Hypotheses can be refuted. A refutation is just as valuable a result, and it is reported as it is.

### 8.7 Cost control

- **Write cost** grows linearly with history tokens (LongMemEval_S is about 500 × 115k tokens). So development uses a 100-question subset, a **session-level content-hash cache** extracts each session once, and extraction uses a cheap model.
- **The full-context baseline B1** is the most expensive and is run only when necessary. In N9 it was the largest single item, about a third of DeepSeek's spend: each LongMemEval question carries its own ~106k-token history, so the provider cache cannot help.
- **Nominal cost.** All costs are reported at nominal list price (peak, no cache) so that runs at different times compare. Actual spend is tracked separately, after cache hits and DeepSeek's off-peak discount (D-010).
- **Scheduling.** All DeepSeek calls ran off-peak (D-024).
- **Actual spend per node**: N2 about $4.8, N3 about $6, N4 about $2.9, N5 about $4.6, N6 about $23.7, N7 about $3.6, N8 about $0.95, N9 about $48.3. A rerun with a warm cache is nearly free.

## 9. Technology and constraints

### 9.1 Development environment

The development machine is an Apple M2 with **8 GB of RAM** and about **29 GB** of free disk (checked 2026-10-12). Therefore:

- **No local models.** LLMs, VLMs and embeddings are DeepSeek and Qwen APIs (D-004).
- **Caching.** Embeddings are cached by content hash, so each text is paid for once; LLM responses are cached too, so repeated experiments cost nothing.
- **No large downloads** (LongMemEval_M, MemLens 256K) unless needed.
- **Embedded storage, no Docker.** A memory space beyond roughly 20k units can switch to a Qdrant server (D-005).

### 9.2 Choices

| Layer | Default | Why | Alternatives |
|---|---|---|---|
| Engineering | Python 3.12 + uv + ruff + pytest + pydantic v2 | Modern Python tooling; locked, reproducible dependencies | conda |
| LLM / VLM | DeepSeek + Qwen behind OpenAI-compatible APIs, configured per role (§9.3) | Both offer OpenAI-compatible APIs; two families allow cross-backbone experiments and reduce judge bias | — |
| Text embeddings | qwen3.7-text-embedding (API, 1,024 dims, dense only) | Much better semantic recall than text-embedding-v4 (naive retrieval on dev100: 65% → 76% with the same code); no local memory; local BM25 as an optional keyword route (off by default since N5) | text-embedding-v4; local BGE-M3 |
| Image embeddings | QwenCloud tongyi-embedding-vision-plus | Text and images in one space, so text-to-image and image-to-image search work; no local memory | Local SigLIP 2 |
| Reranking | Rule-based scoring; qwen3-rerank as an L2 option | Rule scoring is free and explainable; a model reranker did not improve recall within the budget and is off (D-032) | Local bge-reranker-v2-m3 |
| Vector store | Qdrant (local mode, configurable to a server) | Named vectors (dense / sparse / image) + payload filters; about 5 ms dense search under 5k units (D-005) | LanceDB |
| Structured storage | SQLite | Zero ops, a single portable file | Postgres |
| Graph computation | PPR on scipy.sparse | Small, fast, no extra service | NetworkX; Neo4j |
| Controller | numpy at run time; scikit-learn for training (in the optional `eval` extra) | Explainable, fast to train | A small encoder |
| Service and UI | FastAPI + Gradio (one app; Gradio is the optional `ui` extra) | An interactive, explainable demo, fast | Streamlit |
| Agent tool protocol | Official MCP Python SDK (stdio, and Streamable HTTP at `/mcp`) | Other agents can use the memory directly | Custom HTTP tools |
| Evaluation | Own harness + pandas + matplotlib (matplotlib is in the optional `eval` extra) | One protocol, under control | — |

### 9.3 Model roles

| Role | Default model | Why |
|---|---|---|
| extractor (write-time extraction) | qwen3.8-flash | Highest call volume; the cheapest per token ($0.15 / $0.47 per million) |
| planner (L2 planning and sufficiency check) | qwen3.8-flash | On the read path; must be fast and cheap |
| reader (main experiments) | deepseek-flash (thinking off) | 1M context, so the full-context baseline runs directly; accepts images, so text and image experiments share one reader |
| captioner (write-time image captions) | qwen3.8-flash | Reads images; high volume and off the read path, so the cheapest |
| vlm (re-perception at answer time) | deepseek-flash | Same as the main reader, so methods differ only in showing the originals |
| judge (evaluation) | qwen3.8-max | A strong model from a different family than the main reader, to reduce self-preference (D-008) |
| agent (conversational agent, N8) | qwen3.8-flash | Strongest in the "search once, read, decide" loop (D-041); reads images; no time-of-day pricing (D-044) |
| Reader experiments (N9) | qwen3.6-35b-a3b, qwen3.8-flash, deepseek-flash (deepseek-v4-pro planned, not run) | A spread of capability tiers to test H3 (D-049) |

Model IDs and prices are in `src/cuemem/configs/default.yaml`; `cuemem doctor --live` checks connectivity. DeepSeek V4 models have thinking on by default; requests turn it off explicitly for extraction and answering.

### 9.4 Code layout

```text
rag/
├── src/cuemem/
│   ├── core/          # data models, trace
│   ├── llm/           # OpenAI-compatible client, structured output, cache, cost ledger, off-peak scheduling
│   ├── embed/         # text / image embeddings (cached)
│   ├── store/         # SQLite (log, derived memory, jobs, provenance), vector index, media store
│   ├── write/         # online write, consolidation (extraction / linking / reconciliation / summaries / profile / captions)
│   ├── read/          # probe, controller, L1, L2, reranking, reader, memory tools
│   ├── agent/         # hand-written agent loop: context assembly, tools, sessions; the shopping scenario
│   ├── api/           # service layer, FastAPI, Gradio UI, MCP server, demo replay; OpenAI-compatible memory proxy (N12); /api/v1 and scoped tokens (after N12)
│   ├── configs/       # YAML configuration: defaults and presets (openai, ollama, deepseek, qwen), shipped in the package (N12)
│   ├── memory.py      # facade: write, consolidate, retrieve, answer, forget
│   ├── sdk.py         # Python SDK: Memory / AsyncMemory, shaped like Mem0's (N12)
│   ├── client.py      # remote SDK: MemoryClient / AsyncMemoryClient, calling /api/v1 (after N12)
│   ├── portable.py    # export and import of a memory space (N12)
│   └── evals/         # evaluation: datasets, splits, methods, judge, runner, metrics, reports, routing analysis
├── evals/             # development / test splits and the mixed stream (committed)
├── demo/shop/         # demo: product catalogue, generated images, script
├── examples/          # toy conversations (English and Chinese); SDK and memory-proxy examples (N12)
├── tests/
├── scripts/           # experiment and report scripts for every node, reproduce.py
├── Dockerfile, docker-compose.yml   # container image; CueMem + a Qdrant server (N12)
└── docs/
```

There is no separate multimodal directory: captions live in `write/caption.py`, image–text embeddings in `embed/multimodal.py`, originals in `store/media.py`, and re-perception in `read/reader.py`.

## 10. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| API cost and rate limits | Evaluations too expensive or slow | Response cache; session-level dedup; development subsets; cheap extraction model; a ledger for every run |
| 8 GB of RAM | Local models run out of memory or swap | No local models; embeddings via API |
| About 29 GB of disk | Datasets or models do not fit | No oversized downloads; one cache directory; regular clean-up |
| LLM judge bias | Untrustworthy conclusions | Fixed judge and prompts; manual audit of agreement; confidence intervals |
| Saturated text benchmarks | Absolute gains not significant | Centre the work on Pareto efficiency, controller analysis and multimodal memory; report as is |
| Controller overfitting | Fails on a new dataset | Cross-dataset evaluation; simple explainable models; feature ablations |
| Scope creep | Never finishing | MVP path first; extensions marked clearly |
| Data licences | Compliance | Research use only; follow each dataset's licence (MemLens data CC BY 4.0, images under their source sites' licences) |

## 11. Resolved questions

| Question | Decision |
|---|---|
| Image embeddings: tongyi-embedding-vision-plus (API) or local SigLIP 2? | tongyi-embedding-vision-plus (1,152 dims, one space for images and text; text-to-image search is the biggest multimodal lever). A fused image-and-conversation embedding (qwen3-vl-embedding) is not available on QwenCloud and was not compared (D-036) |
| Sparse route: text-embedding-v4's term weights or BM25? | BM25 with role labels removed and numbers kept whole (D-026); after switching to qwen3.7 the sparse route was revisited and is off by default |
| Text embeddings: text-embedding-v4 or qwen3.7-text-embedding? | qwen3.7-text-embedding, dense only: much better semantic recall; slower writing is a one-time cost (D-014, revised) |
| The controller: one calibrated classifier per action, or a single multi-class model? | Neither: two stages, a gate plus a reasoning-type classifier, then a per-type accuracy table. Per-question outcome labels are too noisy (D-038) |
| Graph: entity–fact–episode tripartite, or the entity graph alone? | The extracted entity graph has almost no edges to walk ("the user" connects to about 95% of facts; other entities have median degree 1). A tripartite graph with mention edges and similar-fact edges, without hub entities, still did not improve recall; off by default (D-031) |
| Evidence format: structured (JSON) or natural language? Chain-of-Note reading? | Natural language with dates, relative-time and validity labels; L2 takes notes before answering (D-030) |
| Consolidation trigger: session end, every N turns, or idle time? | Jobs are registered per session at write time and run by a background thread when idle (§5.2) |
| Extraction model: qwen3.8-flash or deepseek-flash? | qwen3.8-flash (D-018) |

## 12. Scaling to a large deployment (sketch)

The project runs on a laptop, but the architecture was designed with millions of users in mind:

- **Shard by memory space.** Each user's memory is naturally independent. Retrieval services are stateless, and the vector store is partitioned by tenant (Qdrant server supports multi-tenancy by payload partitioning).
- **Asynchronous write path.**
  - Online writes only append to the raw log and enqueue a message.
  - Consolidation workers consume asynchronously, processing one user's jobs serially for consistency.
  - Idempotent job keys and "vectors first, then the database" (D-012) carry over to a distributed setting.
- **Hot and cold tiers.** Profiles and recent memory sit in a cache, the long tail in cheaper storage; retrieval checks the hot tier first.
- **Cost control.**
  - The controller keeps most turns at L0 / L1, with expensive L2 for the few hard questions.
  - λ is configured per scenario: customer service wants low latency, a deep personal assistant can spend more.
  - The stable-prefix layout keeps provider prefix caching effective.
- **Compliance and privacy.** Follow Singapore's PDPA and the EU's GDPR; cascade deletion along provenance (the right to be forgotten), including uncited restatements; audit logs.
- **Continual learning.** Implicit feedback in production (e.g. the user correcting "you got that wrong") becomes a reward signal for a contextual bandit that updates the controller online. The first step is setting λ per reader model; N11 found that recalibrating the accuracy table per reader did not help.

**Already done (N11, D-056):**

- **Server mode.** Setting `CUEMEM_VECTOR_STORE__URL` (e.g. `http://localhost:6333`, with `docker run -p 6333:6333 qdrant/qdrant:v1.19.1`) switches to a Qdrant server; new collections get payload indexes on `kind` and `ts`. Local mode allows one process per data directory, and a second one now gets a `StorageLockedError` that says so. Copy existing local data first with `cuemem migrate-vectors --to URL`: memories already in the database are not written to the vector store again, so switching without migrating gives an empty vector store.
- **Scale** (`scripts/n11_latency.py`, excluding the ~110 ms query embedding call in production): the local work on the read path grows linearly in local mode, about 1.08 s at 46k retrieval units, against 91 ms with the server. Concurrency in one process: local mode saturates at about 9 reads per second; the server reaches 55 per second with 16 threads. Local mode also keeps one open SQLite file per collection (one per memory space, two with photos), opens all of them at start-up and keeps every vector in RAM (about 20 KB per unit at 1,024 dimensions, summed over all spaces), so a server for many users needs a Qdrant server. The `cuemem` CLI raises its open-file soft limit to the hard limit, `cuemem doctor` shows the collection count against it, and a store that cannot open raises `VectorStoreOpenError` with the remedy (D-070). Per-request work no longer grows with the number of sessions in a space (D-070).
- **Access tokens.** With `CUEMEM_API_TOKEN` set, the HTTP API requires a Bearer token and the UI asks to sign in. In N11 that was the only, shared token. The second audit round after N12 (D-069) added scoped tokens (`CUEMEM_API_KEYS_FILE` or `CUEMEM_API_KEYS`): each token reaches only the users and agents it lists, and `/api`, `/api/v1`, the UI and the memory proxy all check this. `CUEMEM_API_TOKEN` remains the admin token. MCP over HTTP (`/mcp`) uses the same tokens and scopes, and a scoped key's `/api/v1/reset` deletes only its own spaces (D-070). Running several instances would also need a server database instead of embedded SQLite.
- **Traces and retention.** Every turn and retrieval writes a line to `runs/traces/<day>.jsonl`, which holds the users' questions; forgetting and deleting a space remove the matching lines with the query vectors of their questions. `traces_retention_days` (`CUEMEM_TRACES_RETENTION_DAYS`, unset by default) makes `cuemem serve` and `cuemem proxy` delete older daily files at start-up and then daily; the LLM cache is untouched, and SDK-only apps call `memory.system.prune_traces(days)`. A trace that cannot be written is a warning, never a failed request.
- **Tests.** `uv run pytest` runs offline behind a network guard (`tests/netguard.py`: only loopback, no environment proxy, provider keys removed, `GRADIO_ANALYTICS_ENABLED=False`); `uv run pytest -m live` needs keys and is exempt. The external baselines' adapter tests run in CI's "外部基线" workflow with pinned versions, and the Qdrant server tests in the `qdrant-server` job (`CUEMEM_TEST_QDRANT_URL` locally).

## Appendix: glossary

| Term | Meaning |
|---|---|
| Memory space | All memory of one user (or one evaluation sample), isolated from others |
| Memory probe | A cheap retrieval and cue extraction that runs every turn without an LLM |
| Retrieval controller (gate) | The light policy that chooses L0 / L1 / L2 |
| L0 / L1 / L2 | No retrieval / fast recall (no LLM) / deep recall (an LLM plans and checks) |
| Consolidation | Asynchronously turning the raw log into facts, entities, summaries and the profile |
| Bi-temporal | Event time (when it held in reality) plus record time (when the system learned it) |
| Re-perception | Giving the original photo to the VLM at answer time instead of only its description |
| λ | The cost knob: higher is cheaper, lower is more accurate |
| RRF | Reciprocal rank fusion: merging retrieval routes by reciprocal rank |
| PPR | Personalised PageRank: graph diffusion from seed nodes |
