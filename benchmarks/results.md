# Results

The spine of this repo. Every lesson that changes the pipeline appends a row here.

**Rules for this table**

1. Only the local profile appears in the headline table, so results are reproducible by
   anyone without API keys. Cloud runs go in a separate section.
2. A row is added **whether or not the technique helped.** A negative result is the most
   useful thing in the table — it is the difference between knowing and assuming.
3. Cost and latency sit next to quality. "Better" that costs 40× more is a tradeoff.
4. Predict the delta before running. Record the surprise when you are wrong.

## Local profile

Corpus: `data/` (6 documents, ~2,100 words). Golden set: `ragkit/eval/golden.yaml`.
Embeddings BGE-small, generation `qwen2.5:7b` at temperature 0.

| Lesson | Pipeline | Recall@5 | MRR | nDCG@5 | Grounded | Relevance | ms/query | Notes |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | _awaiting M2; the harness lands in lesson 03_ |

## Cloud profile

Claude for generation, Voyage for embeddings and reranking. Cost is per query, averaged
over the golden set.

| Lesson | Pipeline | Recall@5 | MRR | nDCG@5 | Grounded | Relevance | $/query | Notes |
|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | _awaiting M2_ |

## Reading this table

Recall@5 is the number to watch first: it bounds everything downstream, because a chunk
that was never retrieved cannot be reranked, cited, or reasoned over. If recall is flat
and groundedness moves, the change affected generation, not retrieval — and vice versa.
That separation is the whole reason both halves are measured.
