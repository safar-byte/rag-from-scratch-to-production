# Progress

The authoritative "where am I" file. Pull this repo on any machine, read this file, and
you have the full context. Update it and commit before ending a session.

**Current position:** M1 complete except for one unverified step (see below).
Next up: M2 — the evaluation harness, lesson 03.

---

## Milestone status

| Milestone | Scope | Lessons | Status |
|---|---|---|---|
| M0 | Foundation: env, provider protocols, corpus, CI | 00 | ✅ done |
| M1 | Naive RAG end to end, offline | 01–02 | ✅ done (generation unverified) |
| M2 | Evaluation harness ⭐ | 03 | ⬜ next |
| M3 | Hybrid search + reranking | 04–05 | ⬜ |
| M4 | Query transformation + structural retrieval | 06–09 | ⬜ |
| M5 | Agentic RAG + GraphRAG | 10–11 | ⬜ |
| M6 | Production hardening + inspector UI | 12 | ⬜ |

## ⚠️ The thing to deal with first in M2

**The corpus is too small to measure anything.** It is 7 documents and 41 chunks.
Retrieving the top 5 means retrieving 12% of the entire corpus, so every method looks
excellent and nothing discriminates between them.

Measured during lesson 01: the chunk holding the answer ranks **#1** for every
identifier query tried, including ones deliberately designed to defeat dense retrieval:

| Query | Rank of the answering chunk |
|---|---|
| What is error code TX-4491? | 1 |
| What does exit status 75 mean? | 1 |
| What is `FUSION_CONSTANT` set to? | 1 |
| Which port does the query service use? | 1 |

`data/07-operations.md` was added specifically to provide identifiers used in passing,
in prose that is not about identifiers — the case dense retrieval genuinely struggles
with. It still ranks 1, because of the corpus size, not because the problem is not real.

This is a **ceiling effect**, and it threatens the premise of the whole repo: if
recall@5 is already at 1.0, lessons 04 and 05 cannot show a lift, and "measure every
technique" becomes theatre.

**So M2 has to start by fixing it, not by writing metric code.** Options, in rough order
of preference:

1. Grow the corpus to a few hundred chunks — enough that top-5 is a genuinely selective
   ask. Needs roughly 30–50 documents rather than 7. Could be written, or assembled from
   a permissively licensed technical corpus.
2. Report metrics at k=1 and k=3 as well as k=5, which is more discriminating at any
   corpus size, and worth doing regardless.
3. Add deliberately adversarial distractor documents — near-duplicates and documents
   that discuss the same terms in a different sense.

Do (2) unconditionally. (1) is the real fix. Until the corpus grows, **treat every
number in `benchmarks/results.md` as provisional** and say so in the table.

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

**M2 — the evaluation harness (lesson 03).** In order: fix the ceiling problem above,
then build the golden set, then retrieval metrics (recall@k, precision@k, MRR, nDCG@k),
then generation metrics (groundedness, answer relevance) with an LLM judge and a
deterministic fallback, then the report writer that appends to `benchmarks/results.md`.

*Exit criteria:* baseline numbers for M1 committed and identical across two runs.

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
