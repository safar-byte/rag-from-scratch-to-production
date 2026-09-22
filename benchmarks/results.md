# Results

The spine of this repo. Every lesson that changes the pipeline appends a row here, via
`python -m ragkit.eval.run --write --label "..."` rather than by hand.

**Rules for this table**

1. The local profile is the headline, so results are reproducible by anyone without API
   keys. Cloud runs get their own section.
2. A row is added **whether or not the technique helped.** A negative result is the most
   useful thing in the table — it is the difference between knowing and assuming.
3. Cost and latency sit next to quality. "Better" that costs 40× more is a tradeoff.
4. Predict the delta before running, and record the surprise when you are wrong.

## 🔴 Refusal: the number that matters, and the correction to it

The best pipeline scores **R@5 = 0.987** on retrieval. Asked about things the corpus does
not cover, it invents answers — including a default value for `SHARD_REPLICATION_FACTOR`,
a setting that exists nowhere:

| Question (nothing in the corpus answers it) | What the system said |
|---|---|
| What is the capital of France? | "Paris is the capital of France." |
| Default value of `SHARD_REPLICATION_FACTOR`? | **"The default value ... is 1."** |
| Default value of `MAX_CHUNK_BYTES`? | **"is 1024 bytes."** |
| Which cloud provider? | **"AWS (Amazon Web Services)"** |

### A correction, because the first number here was wrong

This table originally reported **0.000** correct refusal. That figure came from a
3-question sample, and when the set grew to 10 it was also wrong for a second reason:
`looks_like_refusal` matched `"not provided"` but not `"does not provide"`, so genuine
refusals like *"The passage does not provide any information about the population of
Tokyo"* were scored as confabulations.

**A gap in the detector is indistinguishable from a failure in the model** once it
reaches this table. Fixed, with a regression test naming the exact phrase.

### The measured levers, after the fix

| Lever | Correct refusal | False refusals on answerable | Cost |
|---|---|---|---|
| default prompt | 0.300 | 0/5 | — |
| **strict prompt** | **0.900** | **1/5 (20%)** | free |
| score floor 0.60 | 0.500 | 0/34 | one float comparison |

Two things worth carrying:

**Prompting is far more effective than this repo first claimed.** An earlier draft said a
prompt "reduces this; it does not eliminate it". Measured, the strict prompt takes
refusal from 0.300 to 0.900 — the single largest lever available, and it is free.

**It buys that by over-refusing.** One valid question in five was declined — including
"What port does the query service listen on by default?", which the corpus plainly
answers. The score floor is weaker but refuses nothing valid. Which tradeoff is right
depends entirely on whether a wrong answer or a missing answer costs you more, and the
only reason that choice is visible at all is that both directions were measured.

## ⚠️ Read this before trusting any number here

Rows marked **⚠️ ceiling** were produced on a corpus too small to discriminate. At 113
chunks, retrieving the top 5 means retrieving 4% of everything, and recall sits near 1.0
for almost any method. 17 of the 44 golden questions are saturated at 1.000 outright. A technique showing no gain on such a row has **not been shown to
be useless** — it has been shown to be untestable at this scale.

This is the most common way RAG benchmarks mislead, including this one. `recall@1` is
the most discriminating column while the ceiling holds; prefer it over `recall@5`.
Growing the corpus is tracked in `PROGRESS.md`.

## Local profile

Corpus: `data/` (16 documents, 113 chunks). Golden set: `ragkit/eval/golden.yaml`
(44 questions: 10 lookup, 9 conceptual, 8 vocabulary-mismatch, 7 multi-hop,
10 unanswerable). Embeddings BGE-small, generation
`qwen2.5:1.5b` at temperature 0, reranking `ms-marco-MiniLM-L-6-v2`.

| Run | Retriever | Chunks | R@1 | R@3 | R@5 | MRR | nDCG@5 | Grounded | Relevance | Refusal | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
<!-- local-results -->
| L06: rewrite -> rerank | rewrite->rerank(hybrid(dense+bm25,rrf60),n=10) | 113 | 0.795 | 0.943 | 0.966 | 0.914 | 0.920 | - | - | - | 3507 |
| L11: graph traversal only | graph(h=1) | 113 | 0.409 | 0.409 | 0.409 | 0.432 | 0.414 | - | - | - | 0 |
| L09: router (both branches reranked) | router(lex=rerank(bm25,n=10),sem=rerank(hybrid(dense+bm25,rrf60),n=10)) | 113 | 0.841 | 0.966 | 0.989 | 0.949 | 0.955 | - | - | - | 2556 |
| L09: router | router(lex=bm25,sem=rerank(hybrid(dense+bm25,rrf60),n=10)) | 113 | 0.818 | 0.966 | 0.989 | 0.938 | 0.946 | - | - | - | 1749 |
| L08: parent | parent(rerank(hybrid(dense+bm25,rrf60),n=10),w=600) | 113 | 0.841 | 0.966 | 0.989 | 0.949 | 0.953 | - | - | - | 4750 |
| L05 + generation (qwen2.5:1.5b, deterministic judge) | rerank(hybrid(dense+bm25,rrf60),n=10) | 113 | 0.811 | 0.960 | 0.987 | 0.939 | 0.946 | 0.672 | 0.649 | 0.000 | 35513 |
| L05: rerank, shortlist 10 (swept) | rerank(hybrid(dense+bm25,rrf60),n=10) | 113 | 0.811 | 0.960 | 0.987 | 0.939 | 0.946 | - | - | - | 1613 |
| L04-05: rerank | rerank(hybrid(dense+bm25,rrf60),n=25) | 113 | 0.811 | 0.960 | 0.987 | 0.939 | 0.946 | - | - | - | 3386 |
| L04-05: hybrid | hybrid(dense+bm25,rrf60) | 113 | 0.824 | 0.905 | 0.946 | 0.933 | 0.925 | - | - | - | 1048 |
| L04-05: bm25 | bm25 | 113 | 0.689 | 0.919 | 0.919 | 0.857 | 0.859 | - | - | - | 1014 |
| M3 baseline: dense only, 113 chunks, 37 questions | dense | 113 | 0.824 | 0.932 | 0.932 | 0.940 | 0.925 | - | - | - | 1078 |
| M2 baseline: dense only, recursive 512/64 | dense | 80 | 0.900 | 0.980 | 0.980 | 1.000 | 0.985 | - | - | - | 2431 |

## Cloud profile

Claude for generation, Voyage for embeddings and reranking.

| Run | Retriever | Chunks | R@1 | R@3 | R@5 | MRR | nDCG@5 | Grounded | Relevance | Refusal | ms | $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
<!-- cloud-results -->
| _awaiting a cloud run_ | - | - | - | - | - | - | - | - | - | - | - | - |

## What is still unmeasured

Honest inventory, because this table's whole claim is that every technique earns a row.

| Technique | Status |
|---|---|
| hyde, multiquery, stepback | not run — one model call per question, ~20 min each locally |
| contextual retrieval | implemented; the ingest pass is ~45 min on CPU and has not been run |
| CRAG grading | implemented; needs a generation run to measure refusal and false refusals |
| multi-hop | implemented; needs a generation run |
| cloud profile | never run — no `VOYAGE_API_KEY` |

Every one has a command in its lesson. The prior for the three remaining transforms is
that they lose, since `rewrite` did and for structural reasons that apply to all of them
— but a prior is not a measurement, and they are listed here rather than quietly assumed.

## Reading this table

**R@1 first.** While the ceiling holds it is the only column with room to move.

**Recall bounds everything downstream.** A document never retrieved cannot be reranked,
cited, or reasoned over. If recall is flat and groundedness moves, the change affected
generation, not retrieval — and vice versa. That separation is the entire reason both
halves are measured rather than one end-to-end score.

**Refusal is the column people forget.** It scores only the unanswerable questions: did
the system decline, or confabulate? A pipeline that tops every other column and scores
near zero here is not deployable — and it will not look broken until you measure it.

**The `Refusal 0.000` in the row below is stale.** It was produced against the
3-question unanswerable set with the buggy detector. The corrected figures are in the
section at the top of this file; the row is kept rather than edited, because quietly
rewriting a recorded measurement is exactly the habit this table exists to prevent.

Per-question detail for every run is in `benchmarks/runs/*.json`. The aggregate tells
you whether something improved; the JSON tells you which questions moved, which is what
you need when a change helps on average and breaks one question badly.
