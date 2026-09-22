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

## Over-fetch depth — the sweep that changed the default

`rerank_candidates` is the shortlist depth. The received wisdom is 25, and "deeper is at
least as good, just slower". Both turn out to be wrong here.

```
| depth |   R@1 |   R@3 |   R@5 | nDCG@5 | vocab R@5 | ms/query |
|     3 | 0.838 | 0.960 | 0.973 |  0.946 |     0.875 |      258 |
|     5 | 0.838 | 0.960 | 0.973 |  0.946 |     0.875 |      263 |
|    10 | 0.811 | 0.960 | 0.987 |  0.946 |     1.000 |      466 |
|    25 | 0.811 | 0.960 | 0.987 |  0.944 |     1.000 |     1828 |
|    50 | 0.811 | 0.960 | 0.960 |  0.932 |     0.875 |     2831 |
```

Three things, none of them obvious:

**Depth 10 is the whole win.** It matches depth 25 exactly — R@5 0.987, vocab R@5 1.000 —
at roughly a quarter of the latency. The default in `config.py` is now 10, on this
evidence rather than on convention.

**Depth 50 is worse.** Not slower-but-equal: *worse*, R@5 0.987 → 0.960 and vocab
1.000 → 0.875. A small cross-encoder handed more distractors makes more mistakes, and
some of those mistakes push a correct passage out of the top 5. The intuition that a
deeper shortlist is monotonically safe is simply false, and this is why the rule is
"recall at shortlist depth is a *ceiling*", not "a deeper shortlist is better".

**Shallow depths have the best R@1** (0.838 vs 0.811). With fewer candidates there are
fewer chances to demote the correct rank-1 result. If your application shows exactly one
passage, the right depth here is 3, not 10.

So there is no single best depth — there is a best depth *for the k you actually serve*.

### A caution about this table

These numbers come from 37 questions on a 113-chunk corpus, and the differences between
adjacent rows are one or two questions changing. Do not port the number 10 to your own
corpus; port the sweep. The first version of this sweep was also silently broken — see
below.

### The sweep that lied

The first run of this sweep showed quality perfectly flat from depth 5 to 100. That was
not a finding, it was a bug in the harness: `run_eval` always requested depth 20 for its
rank diagnostic, so any shortlist configured below 20 was silently widened to 20 and the
small depths all measured the same pipeline.

`run_eval` now takes `deep_k` to switch the diagnostic off. Worth remembering as a class
of bug rather than an incident — **the measurement harness is also code, and it can lie
to you with a perfectly plausible table.** The tell was that the result was too clean: a
genuinely flat curve across a 20× range should have prompted suspicion, not a conclusion.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 06 — Query transformation](../06-query-transformation/). Retrieval is now decent;
the next question is whether the *query* is the weak link.
