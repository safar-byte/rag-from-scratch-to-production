# Lesson 05 — Reranking

```bash
python -m ragkit.eval.run --retrieval-only --strategy rerank
python lessons/05-reranking/run.py --sweep          # shortlist depth sweep
```

## Bi-encoders and cross-encoders

An embedding model is a **bi-encoder**: it encodes the query and each document
separately. Document vectors are precomputed at ingest and search is milliseconds. The
query and the document never meet until a dot product, which limits how well relevance
can be judged.

A **cross-encoder** feeds query and document through the model together and outputs one
relevance score. Attention runs across both texts, so it can weigh how the query relates
to each part of the passage. Much more accurate — and nothing can be precomputed, so it
is orders of magnitude slower. Scoring a million chunks per query is not a system, it is
a stalled request.

Hence two stages: cheap retrieval produces a shortlist, the cross-encoder reorders it.

## The rule that matters most

> **Recall at the shortlist depth is a hard ceiling on final quality.**

A reranker reorders; it cannot conjure. If the answer is not in the top 25, no reranker
puts it in the top 5. When reranking disappoints, the problem is nearly always shortlist
depth or upstream recall — not the reranker.

This is also why the last lesson's disappointing result was not the end of the story.

## What actually happened

| Strategy | R@1 | R@3 | R@5 | nDCG@5 | vocab R@3 | vocab R@5 | ms |
|---|---|---|---|---|---|---|---|
| dense | 0.824 | 0.932 | 0.932 | 0.925 | 0.750 | 0.750 | 1078 |
| hybrid | 0.824 | 0.905 | 0.946 | 0.925 | 0.625 | 0.750 | 1048 |
| **rerank** (hybrid→CE) | 0.811 | **0.960** | **0.987** | **0.946** | **0.875** | **1.000** | 3386 |

Reranking recovers everything hybrid lost and then some. On the vocabulary-mismatch
questions — the only unsaturated, uncapped category in this benchmark — recall@5 goes
from 0.750 to **1.000**. Every one of those questions now surfaces its document.

## The experiment that explains it

Lesson 04 ended with a hypothesis: hybrid produces a better *pool*, not a better
*ranking*. Same reranker, different shortlist source:

| Shortlist source | ALL R@3 | ALL R@5 | nDCG@5 | vocab R@5 |
|---|---|---|---|---|
| dense → rerank | 0.960 | 0.960 | 0.941 | 0.875 |
| **hybrid → rerank** | 0.960 | **0.987** | 0.946 | **1.000** |

Confirmed. **Hybrid search is worse than dense as a final ranker and better as a
candidate pool feeding a cross-encoder.** BM25 pulls documents into the top 25 that dense
alone misses; RRF then orders them badly; the cross-encoder fixes the ordering.

That is the real argument for hybrid retrieval, and it is not the one usually given. It
is not "fusion produces a better ranking" — here it demonstrably does not. It is "two
retrievers with different failure modes assemble a better candidate set, and a reranker
turns a better candidate set into a better answer."

Neither technique is worth much alone on this corpus. Together they take vocab_mismatch
recall@5 from 0.750 to 1.000.

## What it cost

Latency went from ~1050ms to ~3390ms per query — **3.2×**. On this corpus and this
machine, all of that is the cross-encoder scoring 25 candidates on CPU.

Whether that trade is worth it depends entirely on context. Inside a 3-second generation
it is invisible. In an autocomplete box it is unacceptable. The table records both
columns so the question can be answered rather than assumed.

Note also **R@1 dropped slightly, 0.824 → 0.811.** Reranking is not free at the very top;
the cross-encoder occasionally demotes a correct rank-1 result. It buys a large gain at
k=3 and k=5 for a small loss at k=1. If your application shows exactly one passage, that
trade is bad. Measure the k you actually use.

## Over-fetch depth

`rerank_candidates` in `config.py` (default 25) is the shortlist depth, and it is the
dial that matters:

- Too shallow and the ceiling binds — the reranker cannot find what retrieval missed.
- Too deep and cost grows linearly while quality flattens. A hosted reranker bills per
  document scored, so doubling the shortlist doubles that bill.

Do not take 25 on faith. `run.py --sweep` measures it.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 06 — Query transformation](../06-query-transformation/). Retrieval is now decent;
the next question is whether the *query* is the weak link.
