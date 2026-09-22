# Progress

The authoritative "where am I" file. Pull this repo on any machine, read this file, and
you have the full context. Update it and commit before ending a session.

**Current position:** all 13 lessons built, all 7 milestones done. The course is
complete and every technique in it has been measured.

---

## Milestone status

| Milestone | Scope | Lessons | Status |
|---|---|---|---|
| M0 | Foundation: env, provider protocols, corpus, CI | 00 | ✅ |
| M1 | Naive RAG end to end, offline | 01–02 | ✅ |
| M2 | Evaluation harness ⭐ | 03 | ✅ |
| M3 | Hybrid search + reranking | 04–05 | ✅ |
| M4 | Query transformation + structural retrieval | 06–09 | ✅ |
| M5 | Agentic RAG + GraphRAG | 10–11 | ✅ |
| M6 | Production hardening + inspector UI | 12 | ✅ |

101 tests, all offline. CI green. Verified end to end with real models and a browser.

---

## What the measurements actually said

The course produced more negative results than positive ones. That is the outcome an
eval-first repo should produce, and they are the most valuable thing in it.

### Retrieval

| Strategy | R@1 | R@3 | R@5 | nDCG@5 | vocab R@5 | ms |
|---|---|---|---|---|---|---|
| dense | 0.824 | 0.932 | 0.932 | 0.925 | 0.750 | 1078 |
| bm25 | 0.689 | 0.919 | 0.919 | 0.859 | 0.625 | 1014 |
| hybrid | 0.824 | 0.905 | 0.946 | 0.925 | 0.750 | 1048 |
| **rerank** | 0.811 | **0.960** | **0.987** | **0.946** | **1.000** | 1613 |

**Hybrid search lost to plain dense retrieval.** RRF weights retrievers equally, and
BM25 is confidently wrong on vocabulary mismatch (`vocab R@1` 0.250), so equal-weight
fusion propagated that. Reranking rescued it, and the decisive test explained why:
dense→rerank gives `vocab R@5` 0.875 against hybrid→rerank's 1.000. **Hybrid is worse as
a final ranker and better as a candidate pool feeding a cross-encoder.**

**The shortlist sweep overturned its own default.** Depth 10 matches depth 25 at a
quarter the latency, and depth 50 is *worse* (R@5 0.987 → 0.960): a small cross-encoder
given more distractors makes more mistakes. `rerank_candidates` is now 10.

**Routing diverts 4 of 44 questions and zero vocabulary-mismatch ones**, which is lesson
04's failure fixed at its root rather than dampened by a tuned constant.

**GraphRAG does not fit this corpus.** 34 entities, 4 edges, mean degree 1.6. Sixteen
independent explanatory essays have almost no entity co-occurrence. Not a tuning problem.

### The refusal problem, and what fixed it

The worst result in the repo: the best pipeline scored **0.000 on correct refusal**,
inventing "the default value is 1" for `SHARD_REPLICATION_FACTOR`, a setting that does
not exist. The system prompt already told it to decline.

Two mechanisms, at different price points:

| Mechanism | Catches | False refusals | Cost |
|---|---|---|---|
| Score floor 0.60 | 5/10 | **0/34** | one float comparison |
| CRAG grading | the rest | measured per corpus | one model call |

The floor is calibrated, not guessed — 0.60 is the highest threshold refusing no valid
question here, and past 0.63 it rejects real questions faster than it catches fakes.

**The five it cannot catch score higher than real questions:** `MAX_CHUNK_BYTES` 0.689,
`SHARD_REPLICATION_FACTOR` 0.678, against a genuine question at 0.632. That is not a
tuning failure, it is the method's ceiling — a similarity score measures topical
closeness, not answerability. Closing the rest requires reading the passages.

One finding worth carrying: **the gate reads the dense score even when reranking**,
because cross-encoder scores separate answerable from unanswerable far worse (lowest
answerable −9.80 vs highest unanswerable −4.16). The stronger ranker is the weaker
calibration signal.

---

## Bugs found, all silent

None of these raised an error. Every one cost quality quietly, which is what makes them
worth recording.

- **Orphaned headings** — the greedy chunk buffer stranded each heading at the end of
  the *previous* chunk, so the section body lost its strongest keyword.
- **Mid-word overlap** — slicing the last N characters produced overlaps starting
  `"rly always retrieval depth"`.
- **A lying sweep** — the shortlist sweep showed a perfectly flat curve because
  `run_eval` always requested depth 20 for its rank diagnostic, silently widening any
  smaller shortlist. Fixed with `deep_k`. **The measurement harness is also code.**
- **Empty answers scored as refusals** — Ollama returns a reasoning model's chain of
  thought in a separate `thinking` field; budget exhaustion left `response` empty, which
  the harness would have graded as correct refusal. A truncation bug masquerading as
  good behaviour.
- **Entity extraction junk** — a `\s` in the Title Case pattern matched across newlines,
  producing `"Vector Search An"`; singleton entities were dropped entirely, making
  `FUSION_CONSTANT` unfindable.

---

## Environment

- **Python 3.12 conda env `rag`.** System Python is 3.14, no torch wheels.
- **No Docker** → embedded Chroma. **No GPU** → CPU-only inference, which set two
  defaults: `qwen2.5:1.5b` (a 4B *reasoning* model took over 10 minutes for one query,
  spending ~1000 tokens thinking) and `ms-marco-MiniLM-L-6-v2` (~90MB vs ~2.3GB).
- **Ollama** at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`, not on PATH. Needs
  `ollama serve`.
- **The CLI pays model load per invocation** — a one-shot `ragkit ask` reports ~40s of
  "retrieval" that is almost all loading BGE. The eval harness measures ~1s per query.

---

## Known limitations

**The corpus is too small, and it bounds everything.** 16 documents, 113 chunks. 17 of
44 golden questions are saturated at 1.000, so most techniques in lessons 06–11 cannot
show a gain here — and a "no improvement" row is a fact about this corpus, not about the
technique. `vocab_mismatch` is the only category with real headroom, which is why it
appears in nearly every conclusion above.

Adjacent rows in `benchmarks/results.md` differ by one or two questions. **Port the
method, not the numbers.**

**Not deployable as-is**: CORS allows localhost unconditionally, no auth, no rate
limiting, no index versioning. Lesson 12's exercise 6 walks through this.

**Voyage model IDs** in `config.py` are written from memory and unverified against
Voyage's current list. Overridable in `.env`, so nothing local is blocked.

---

## If you pick this up again

The highest-value work, in order:

1. **Grow the corpus** to a few hundred documents. Everything else is bounded by this.
   Lesson 03's exercise 5 walks through it.
2. **Measure CRAG's refusal rate properly**, including false refusals on answerable
   questions. A grader that refuses everything scores perfectly and is useless.
3. **Run the cloud profile** — prompt caching, the Citations API and real cost numbers
   are all unexercised. Needs a `VOYAGE_API_KEY`.
4. **Self-RAG reflection**, if lesson 10's exercise 6 finds a failure mode CRAG misses.

---

## Decision log

**2026-09-22 — Local by default, cloud opt-in.** Iterating on retrieval is most of the
work and is free offline.

**2026-09-22 — Evaluation at lesson 03, before any optimisation.** Without it every
later technique is a matter of taste. Vindicated repeatedly.

**2026-09-22 — Lessons build one shared library.** Thirteen standalone scripts would
duplicate everything and teach nothing about structure.

**2026-09-22 — Chroma, not Qdrant.** No Docker. Qdrant stays an adapter behind the same
interface.

**2026-09-22 — Original corpus rather than a public-domain dump.** Clear licence,
verifiable golden answers. Its smallness is now the main limitation.

**2026-09-22 — `ingest` resets the index by default.** A stale index is harder to debug
than a slow rebuild.

**2026-09-22 — Document-level relevance in the golden set.** Chunk ids change whenever
`chunk_size` changes, so a chunk-level set invalidates itself during the experiment you
most want to run.

**2026-09-22 — `rerank_candidates` 25 → 10.** Measured, not conventional.

**2026-09-22 — No Self-RAG reflection.** CRAG catches the measured problem earlier and
more cheaply. Fix a failure at the earliest stage that can see it.
