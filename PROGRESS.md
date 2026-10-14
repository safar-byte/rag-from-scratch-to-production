# Progress

The authoritative "where am I" file. Pull this repo on any machine, read this file, and
you have the full context. Update it and commit before ending a session.

**Current position:** all 13 lessons built, all 7 milestones done. Every lesson now also
has a detailed Google Colab notebook (`lessons/NN/colab.ipynb`, index in
`colab/README.md`), each executed end to end with Ollama before commit. Building them
measured three things that had never been run and overturned one earlier claim — see
"What the Colab build measured" below.

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
| — | Colab notebooks for every lesson | 00–12 | ✅ |

120 tests, all offline. CI green. Verified end to end with real models and a browser.

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

The worst result in the repo: the pipeline invents answers to questions the corpus does
not cover, including "the default value is 1" for `SHARD_REPLICATION_FACTOR`, a setting
that exists nowhere. The system prompt already told it to decline.

**This number was wrong twice before it was right,** which is worth more than the number:

1. First measured over a **3-question** unanswerable set. A rate over three questions is
   not a rate. Grown to 10.
2. Then `looks_like_refusal` matched `"not provided"` but not `"does not provide"`, so
   genuine refusals — *"The passage does not provide any information about the population
   of Tokyo"* — were scored as confabulations. Reported 0.000; true value 0.300.

Corrected, the three levers:

| Lever | Correct refusal | False refusals on answerable | Cost |
|---|---|---|---|
| default prompt | 0.300 | 0/5 | — |
| **strict prompt** | **0.900** | **1/5 (20%)** | free |
| score floor 0.60 | 0.500 | 0/34 | one float comparison |

**Prompting is the largest lever and it is free**, which contradicts what an earlier
draft of this repo asserted. It pays by over-refusing — one valid question in five,
including "What port does the query service listen on by default?", which the corpus
plainly answers. The score floor is weaker and refuses nothing valid.

The floor is calibrated, not guessed: 0.60 is the highest threshold refusing no valid
question here, and past 0.63 it rejects real questions faster than it catches fakes.
**The five it cannot catch score higher than real questions** — `MAX_CHUNK_BYTES` 0.689,
`SHARD_REPLICATION_FACTOR` 0.678, against a genuine question at 0.632. That is the
method's ceiling, not a tuning failure: a similarity score measures topical closeness,
not answerability.

One finding worth carrying: **the gate reads the dense score even when reranking**,
because cross-encoder scores separate answerable from unanswerable far worse (lowest
answerable −9.80 vs highest unanswerable −4.16). The stronger ranker is the weaker
calibration signal.

### What the Colab build measured

Writing each notebook meant running every experiment again, on the current 44-question
golden set, on a different CPU machine. Quality columns are comparable to the rows above;
`ms` is not.

- **Contextual retrieval, first measurement** (rows L07 in `benchmarks/results.md`).
  Helps dense: R@3 0.943 → 0.977, vocab R@5 0.750 → 1.000. **Hurts rerank** when the
  cross-encoder reads the prefix: R@5 0.989 → 0.966, vocab R@5 1.000 → 0.875. Reranking
  the same shortlist on source `text` only recovers it exactly (0.989 / 1.000). The 1.5B
  prefix writer ignores the prompt's "no 'this excerpt'" rule and sometimes adds facts the
  document lacks; a bi-encoder averages that away, a cross-encoder reads every word.
  **Use the prefix to find candidates, not to judge them.** `CrossEncoderReranker` still
  scores `embed_text` — not changed, because a better prefix writer (Haiku) may flip this.
- **CRAG with the local grader is strictly worse than the free strict prompt.** Correct
  refusal 1.000 but 8/10 false refusals on lookup questions (strict prompt: 0.900 and
  1/10). The 1.5B grader marks nearly everything `incorrect`, including "exit status 75",
  whose answer is in the top passage. Answers open item #2 for the local profile; the cloud
  grader is still unmeasured.
- **Multi-hop: no gain.** 7 multi-hop questions, R@5 0.929 single-shot and 0.929 with up
  to 2 hops, at up to 2 extra model calls each.
- **Parent-document's recall gain did not reproduce.** Rerank and parent now tie exactly
  (R@1 0.841 both). The earlier +0.030 was one question of 37 against an older rerank
  configuration, and the diversity mechanism credited for it cannot move document-level
  recall much: `dedupe_to_documents` already collapses same-document chunks before
  scoring. The diversity is real for the generator (4.9 vs 3.2 distinct documents in the
  top 5) — it acts on the prompt, not the metric.
- **The BGE query prefix measured as no help here.** Over the 34 answerable questions,
  paragraph-level ranking with the prefix: answer first 0.853, without: 0.882 (one
  question). The model card calls it optional for v1.5. Kept as the default; lesson 00
  now teaches the measurement instead of the claim.
- **Rewrite → rerank reproduced as a loss** (R@5 0.989 → 0.966, vocab 1.000 → 0.875), the
  **shortlist sweep reproduced** (depth 10 best R@5, depth 50 worse, shallow best R@1),
  and the **0.60 score floor reproduced** (5/10 caught, 0/34 wrongly refused).

---

## Bugs found, all silent

None of these raised an error. Every one cost quality quietly, which is what makes them
worth recording.

- **Orphaned headings** — the greedy chunk buffer stranded each heading at the end of
  the *previous* chunk, so the section body lost its strongest keyword.
- **Mid-word overlap** — slicing the last N characters produced overlaps starting
  `"rly always retrieval depth"`.
- **A refusal detector with a gap** — `"not provided"` matched, `"does not provide"`
  did not, so correct refusals were scored as confabulations and the reported rate
  was 0.000 instead of 0.300. Regression-tested against the model's real answers.
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

- **Chunk offsets are approximate when overlap is on** (found, not fixed).
  `RecursiveChunker` joins the overlap tail with a single space and shifts `start_char`
  by `len(tail) + 1`, but the source separator is often `"\n\n"`, so
  `document.text[start_char:end_char]` is a character off. Nothing raises; a citation
  highlighter would be misaligned. Lesson 02's notebook shows it and sets the fix as an
  exercise.
- **Multi-hop trace misreports a crashed planner** (found, not fixed).
  `MultiHopRetriever._next_query` swallows exceptions and returns `None`, the same signal
  as `ENOUGH`, so a planner that is down is logged as "model said enough". The loop is
  safe; the trace lies. Lesson 10's notebook demonstrates it with a raising fake.
- **`lessons/08-.../run.py --filter` crashed** (fixed). It filtered through the hybrid
  pipeline, and `Bm25Retriever` correctly raises on a `where` clause. Now filters through
  dense → rerank.

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
2. **Fix the two silent bugs the Colab build found** (chunk offsets with overlap; the
   multi-hop trace on planner failure) — both small, both need a regression test.
3. **CRAG with a stronger grader.** Measured locally with the 1.5B model: it loses to the
   strict prompt (8/10 false refusals). The open question is now whether a stronger
   grader (qwen2.5:7b on a GPU, or Haiku on the cloud profile) beats 0.900 refusal with
   fewer than 1-in-10 false refusals.
4. **Run the cloud profile** — prompt caching, the Citations API and real cost numbers
   are all unexercised. Needs a `VOYAGE_API_KEY`.
5. **Self-RAG reflection**, if lesson 10's exercise 6 finds a failure mode CRAG misses.

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

**2026-10-14 — Colab notebooks per lesson, executed before commit.** Each lesson gets a
`colab.ipynb` that teaches from first principles, re-implements the technique by hand,
and runs the experiments live. Dependencies are installed from `pyproject` rather than
`pip install -e .`, so Colab's Python version never trips the 3.12 pin. Notebooks quote
the repo's numbers and the numbers measured while writing them, and say so when they
differ, rather than presenting either as the truth.
