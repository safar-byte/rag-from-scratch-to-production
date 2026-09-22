# Progress

The authoritative "where am I" file. Pull this repo on any machine, read this file, and
you have the full context. Update it and commit before ending a session.

**Current position:** M2 complete. The harness runs and has already produced a
negative result worth acting on. Next up: M3 — hybrid search and reranking (04-05).

---

## Milestone status

| Milestone | Scope | Lessons | Status |
|---|---|---|---|
| M0 | Foundation: env, provider protocols, corpus, CI | 00 | ✅ done |
| M1 | Naive RAG end to end, offline | 01–02 | ✅ done (generation unverified) |
| M2 | Evaluation harness ⭐ | 03 | ✅ done |
| M3 | Hybrid search + reranking | 04–05 | ⬜ next |
| M4 | Query transformation + structural retrieval | 06–09 | ⬜ |
| M5 | Agentic RAG + GraphRAG | 10–11 | ⬜ |
| M6 | Production hardening + inspector UI | 12 | ⬜ |

## ⚠️ The measurement problem, now measured

M2 started by attacking this rather than by writing metric code, and the corpus grew
from 7 documents / 41 chunks to **12 documents / 80 chunks**. It was not enough, and now
there are numbers saying so:

| kind | n | recall@1 | recall@3 | recall@5 | max R@1 |
|---|---|---|---|---|---|
| conceptual | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| lookup | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 5 | 0.500 | **0.900** | 0.900 | 0.500 |
| unanswerable | 3 | 1.000 | 1.000 | 1.000 | 1.000 |

**17 of 25 questions are saturated.** Lookup and conceptual are perfect at every depth,
so no technique in lessons 04-11 can show a gain on them.

**multi_hop recall@1 = 0.500 is a perfect score, not a failure.** A question with two
relevant documents caps at 0.5 at k=1 by arithmetic. The harness now prints a
`max R@1` column and colours a score green when it is at its cap, because that row gets
misread otherwise. This was nearly recorded as a retrieval problem.

**`multi_hop recall@3 = 0.900` is the only unsaturated number in the table** - one
question of five misses one of its two documents at depth 3. That is the entire
measurable headroom available to M3.

The harness now detects this itself: it warns on saturation, and `benchmarks/results.md`
stamps `⚠️ ceiling` on rows it cannot trust.

**What M3 must do about it.** Hybrid search and reranking cannot be evaluated on 25
questions where 17 are saturated. Before drawing any conclusion in lesson 04-05, grow the
corpus to roughly 40+ documents and add golden questions that target vocabulary mismatch
(ask in words the document does not use). Lesson 03's exercise 6 asks the reader to do
exactly this, and it is not busywork - it is the precondition for M3 meaning anything.

## What is done

**M0 — Foundation**

- Python 3.12 conda env; `pyproject.toml` with `local` / `cloud` / `graph` / `dev`
  extras so a clean clone tests green without torch.
- `ragkit/config.py` — settings, `RAG_PROFILE` selection, all model IDs in one place.
- `ragkit/types.py` — `Document`, `Chunk`, `Scored`, `Citation`, `Usage`, `Answer`.
- `ragkit/providers/` — `Embedder` / `Reranker` / `Generator` protocols plus both
  backends (sentence-transformers + Ollama; Voyage + Claude).
- `data/` — seven original documents, ~2,600 words.
- CI: ruff + pytest on the local profile only.

**M1 — Naive RAG**

- `ragkit/ingest/` — loader and two chunkers (fixed, recursive).
- `ragkit/store/` — `VectorStore` protocol and an embedded Chroma adapter. Chroma
  returns cosine *distance*; the adapter flips it so higher is better everywhere else.
- `ragkit/retrieve/dense.py`, `ragkit/generate/` — retrieval, prompt assembly, marker
  based citation extraction.
- `ragkit/pipeline.py` — the one object lessons 01–12 modify.
- `ragkit/cli.py` — `ingest`, `search`, `ask`, `status`.
- Lessons 01 and 02 with runnable scripts and exercises.
- 31 tests, all offline.

**Verified end to end with real models:** `ingest` builds a 41-chunk index with
BGE-small (384-dim) in ~46s; `search` and `status` work; retrieval quality is good
(cosine 0.82 for "reciprocal rank fusion", correct chunk at rank 1).

**Two real chunking bugs found and fixed**, both silent — no error, just lost recall.
Regression tests added for each:

- *Orphaned headings.* The greedy buffer stranded a heading at the end of the previous
  chunk, so the section body lost its strongest keyword and the previous chunk got a
  label for content it did not contain. Fixed by `_merge_headings`.
- *Mid-word overlap.* Slicing the last N characters produced overlaps starting
  `"rly always retrieval depth"`. Fixed by `_tail_words`.

**M2 - Evaluation**

- Corpus grown to 12 documents / 80 chunks (added vector indexes, query understanding,
  filtering and tenancy, observability, cost and latency).
- `ragkit/eval/golden.yaml` - 25 questions across four kinds, relevance judged at
  **document** level so the set survives a change to `chunk_size`.
- `ragkit/eval/metrics.py` - recall@k, precision@k, MRR, nDCG@k, plus a
  `rank_of_first_relevant` diagnostic that separates "ranked 12" from "absent".
  Tested against hand-computed arithmetic.
- `ragkit/eval/judge.py` - an LLM judge and a deterministic one. Both, on purpose: the
  only way to know whether to trust a judge is to compare it against something that
  cannot have opinions.
- `ragkit/eval/run.py` - the runner, with saturation detection and the `max R@1` cap
  column.
- `ragkit/eval/report.py` - appends to `benchmarks/results.md` and writes per-question
  JSON to `benchmarks/runs/`.
- Lesson 03 with README and exercise.
- 57 tests total, all offline.

**Baseline recorded:** `R@1 0.900 | R@3 0.980 | R@5 0.980 | MRR 1.000 | nDCG@5 0.985`.

## Unverified

**`ragkit ask` has not been run end to end.** Ollama is not installed on this machine,
so generation is untested on the local profile. Everything up to and including prompt
assembly is verified; the final model call is not.

To close this:

```bash
# install Ollama, then:
ollama serve
ollama pull qwen2.5:7b
python -m ragkit.cli ask "What is reciprocal rank fusion?"
```

## What is next

**M3 - hybrid search and reranking (lessons 04-05).** BM25 via `bm25s`, reciprocal rank
fusion, then cross-encoder reranking over an over-fetched shortlist.

*Do first:* grow the corpus and add vocabulary-mismatch questions, per the section above.
Measuring hybrid search against a saturated benchmark produces a confident "no
improvement" that is a fact about the corpus, not about hybrid search.

*Hypothesis to test:* the one failing multi-hop question should be the one that hybrid
search fixes, since multi-hop questions span documents that may share no vocabulary with
the query.

*Exit criteria:* a measured lift over the M2 baseline on the unsaturated questions, with
the deltas discussed in each lesson README.

## Open questions

- Voyage model IDs in `config.py` (`voyage-3-large`, `rerank-2.5`) are defaults written
  from memory and **not verified** against Voyage's current model list. Check before the
  first cloud run. Overridable via `.env`, so nothing local is blocked.
- Local judge reliability is unknown. A 7B model grading groundedness may be too noisy;
  compare it and the lexical fallback against ~20 hand labels before trusting either.
- Chroma holds its HNSW segment file open for the process lifetime, so temp dirs cannot
  be cleaned on Windows during a test run. `tests/conftest.py` uses
  `ignore_cleanup_errors=True`. Revisit if the store ever needs an explicit `close()`.

## Decision log

**2026-09-22 — Local by default, cloud opt-in.** Three protocols and a profile switch.
Iterating on retrieval is most of the work and is free offline.

**2026-09-22 — Evaluation at lesson 03, before any optimisation.** Without it every
later technique is a matter of taste.

**2026-09-22 — Lessons build one shared library.** Thirteen standalone scripts would
duplicate everything and teach nothing about structure.

**2026-09-22 — Chroma, not Qdrant.** No Docker on this machine. Qdrant stays as an
adapter behind the same interface.

**2026-09-22 — Original corpus rather than a public-domain dump.** Clear licence, small
enough to commit, specific enough facts for verifiable golden answers. *Revisit in M2:*
the same smallness is now the main measurement problem.

**2026-09-22 — Python 3.12, not the machine's 3.14.** No torch wheels for 3.14.

**2026-09-22 — `ingest` resets the index by default.** Partially re-indexing a corpus
whose chunk boundaries moved leaves orphans, and a stale index is far harder to debug
than a slow rebuild. Incremental ingest arrives in lesson 12 with index versioning.
