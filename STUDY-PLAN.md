# Study Plan

Milestone-based, no dates. Move on when the exit criterion passes, not when a week ends.
Current position lives in [`PROGRESS.md`](PROGRESS.md).

**How to work a lesson**

1. Read the lesson `README.md` — the idea, when to use it, how it fails, what it costs.
2. Read the `ragkit/` module it introduces. The code is the real material; the lesson is
   the commentary.
3. Run `run.py` and look at the output, especially the retrieved chunks.
4. Do `exercise.md`, then run the eval and compare against the predicted delta.
5. Append the result to `benchmarks/results.md` — including when it got worse.
6. Update `PROGRESS.md` and commit.

The habit that matters is step 5. Predicting the delta *before* you measure it is how
you find out whether you actually understand the technique. When the prediction is wrong,
that surprise is the lesson.

---

## M0 — Foundation ✅

### Lesson 00 — Setup
Environment, the three provider protocols, the corpus. Why a RAG codebase should never
import a vendor SDK outside its provider layer, and why "local by default" is a design
constraint rather than a convenience.

**Exit:** `pytest` green on a clean clone with no API keys.

---

## M1 — Naive RAG, end to end

### Lesson 01 — Naive RAG
The whole pipeline at its simplest: load, chunk, embed, store, search by cosine
similarity, stuff the top k into a prompt, generate. Build this first so you have
something concrete to criticise — every later lesson is a response to a specific way
this version fails.

**Concepts:** ingest pipeline, vector store, top-k similarity, prompt assembly, the
"answer only from the context" instruction and why it is necessary but not sufficient.

### Lesson 02 — Chunking
Fixed, recursive, and token-aware splitting; overlap; the precision/context tension.
Chunking is the highest-leverage decision in the pipeline because nothing downstream can
recover what chunking destroyed.

**Exercise:** sweep chunk size across 256/512/1024 and predict which wins before running.

**Exit:** `ragkit.cli ask` returns a grounded answer offline.

---

## M2 — Evaluation ⭐

### Lesson 03 — The evaluation harness
The pivot point of the whole course. A golden set, retrieval metrics (recall@k,
precision@k, MRR, nDCG@k), generation metrics (groundedness, answer relevance), and an
LLM judge with a deterministic fallback.

Why both halves are measured separately: high retrieval with low groundedness is a
prompting problem; low retrieval with high groundedness means the generator is doing
well with bad material. One end-to-end number cannot tell you which, and so cannot tell
you where to work.

**Concepts:** golden sets and how to build one without fooling yourself, judge bias,
temperature zero for reproducibility, why you never compare scores across judges.

**Exit:** baseline numbers for M1 committed to `benchmarks/results.md` and reproducible
across two runs.

---

## M3 — The production baseline

### Lesson 04 — Hybrid search and RRF
BM25 alongside dense retrieval, fused with Reciprocal Rank Fusion. Why the two fail
differently, why their scores cannot be compared directly, and why RRF's rank-only
approach sidesteps the problem without tuning.

**Exercise:** find a query the dense retriever fails and BM25 nails. Identifiers are the
easy case — try to find a harder one.

### Lesson 05 — Reranking
Cross-encoders versus bi-encoders, the over-fetch pattern, and the ceiling rule:
**recall at shortlist depth bounds final quality.** A reranker cannot find what
retrieval missed.

**Exercise:** sweep shortlist depth 10/25/50/100. Find where quality stops improving and
you are just paying for it.

**Exit:** measured lift over the M2 baseline, with the deltas discussed in each README.

---

## M4 — Query transformation and structure

### Lesson 06 — Query transformation
Rewriting, HyDE, multi-query fan-out, step-back prompting. The user's question is
frequently not a good search query; these techniques fix the query rather than the index.
Each costs at least one extra model call, so each must earn it.

### Lesson 07 — Contextual retrieval
Anthropic's chunk-prefixing technique: a cheap model writes a sentence situating each
chunk in its document before embedding. This is also where prompt caching and the Batch
API earn their place — the naive implementation re-sends the document once per chunk.

### Lesson 08 — Parent-document and metadata
Small-to-big retrieval: search over small precise chunks, return their larger parents.
Metadata filtering and why a pre-filter beats a post-filter.

### Lesson 09 — Routing
Query routing across multiple indexes, and when one index is the better answer.

**Exit:** every technique has a measured row, **including the ones that did not help.**

---

## M5 — Agentic and graph

### Lesson 10 — Agentic RAG
CRAG-style retrieval grading with fallback, self-RAG reflection, and multi-hop retrieval
through tool use. The pipeline stops being a straight line and starts making decisions,
which means it can now fail in loops — so budgets and stopping conditions are part of
the design, not an afterthought.

### Lesson 11 — GraphRAG
Entity and relation extraction, graph construction, traversal retrieval. Strong for
"how do X and Y relate" questions that vector search structurally cannot answer;
expensive to build and to keep current.

**Exit:** multi-hop questions that the M3 baseline fails are answered correctly, with
cost and latency recorded next to quality.

---

## M6 — Production

### Lesson 12 — Hardening and the inspector
Prompt caching, the Citations API, cost and token accounting, structured logging and
tracing, retries and timeouts, index versioning and re-ingest. Then the React inspector:
retrieved chunks, per-stage scores, the assembled prompt, tokens, cost, and highlighted
citations.

The inspector is the best teaching device in the repo. Seeing *why* a chunk was retrieved
teaches more than any paragraph about it.

**Exit:** a test asserts `cache_read_input_tokens > 0`; the inspector renders a full
pipeline trace for a live query.

---

## Reading alongside

Primary sources beat blog posts. Anthropic's contextual retrieval write-up for lesson 07,
the original BM25 and RRF papers for lesson 04, the cross-encoder literature for lesson
05, and the CRAG and Self-RAG papers for lesson 10. Read them after building the naive
version, not before — the papers make far more sense once you have felt the problem.
