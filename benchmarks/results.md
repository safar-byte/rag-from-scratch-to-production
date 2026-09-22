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

## ⚠️ Read this before trusting any number here

Rows marked **⚠️ ceiling** were produced on a corpus too small to discriminate. At 80
chunks, retrieving the top 5 means retrieving 6% of everything, and recall sits near 1.0
for almost any method. A technique showing no gain on such a row has **not been shown to
be useless** — it has been shown to be untestable at this scale.

This is the most common way RAG benchmarks mislead, including this one. `recall@1` is
the most discriminating column while the ceiling holds; prefer it over `recall@5`.
Growing the corpus is tracked in `PROGRESS.md`.

## Local profile

Corpus: `data/`. Golden set: `ragkit/eval/golden.yaml` (25 questions: 8 lookup,
9 conceptual, 5 multi-hop, 3 unanswerable). Embeddings BGE-small, generation
`qwen2.5:1.5b` at temperature 0, reranking `ms-marco-MiniLM-L-6-v2`.

| Run | Retriever | Chunks | R@1 | R@3 | R@5 | MRR | nDCG@5 | Grounded | Relevance | Refusal | ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
<!-- local-results -->
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

## Reading this table

**R@1 first.** While the ceiling holds it is the only column with room to move.

**Recall bounds everything downstream.** A document never retrieved cannot be reranked,
cited, or reasoned over. If recall is flat and groundedness moves, the change affected
generation, not retrieval — and vice versa. That separation is the entire reason both
halves are measured rather than one end-to-end score.

**Refusal is the column people forget.** It scores only the three unanswerable
questions: did the system decline, or confabulate? A pipeline that tops every other
column and scores 0 here is not deployable.

Per-question detail for every run is in `benchmarks/runs/*.json`. The aggregate tells
you whether something improved; the JSON tells you which questions moved, which is what
you need when a change helps on average and breaks one question badly.
