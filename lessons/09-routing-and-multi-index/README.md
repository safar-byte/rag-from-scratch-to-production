# Lesson 09 — Routing

```bash
python -m ragkit.eval.run --retrieval-only --strategy router
python lessons/09-routing-and-multi-index/run.py
```

## Unfinished business from lesson 04

Hybrid search lost to plain dense retrieval. The reason was specific and measurable: RRF
weights every retriever equally, and on vocabulary-mismatch questions BM25 scored 0.250
at rank 1 — not merely unhelpful but *confidently wrong*. Equal-weight fusion propagated
that confident wrongness.

Two candidate fixes were named there:

**Weighted RRF** — give dense more votes. It would work, and it is tuning a global
constant against 44 questions. That is overfitting with extra steps, and it applies the
same compromise to every query regardless of what the query needs.

**Routing** — decide per query. This lesson.

## The insight

The two retrievers do not fail randomly. Which one wins is predictable from the *shape*
of the query:

| Query | Wins | Why |
|---|---|---|
| "What does exit status 75 mean?" | BM25 | the literal token is the query |
| "What is `FUSION_CONSTANT` set to?" | BM25 | exact key name, no semantic content |
| "Why do I get charged twice?" | dense | shares no vocabulary with the answer |
| "My scanned paperwork comes back blank" | dense | needs meaning, not matching |

Identifiers, config keys, error codes and version numbers are *typographically marked*.
A regex recognises them, and a regex costs nothing and can be read.

## The asymmetry that sets the default

`HeuristicRouter` is deliberately conservative about routing to lexical, because **the
two mistakes do not cost the same**:

- Routing a prose question to BM25 is bad. BM25 cannot bridge vocabulary mismatch at
  all, so the correct chunk may not appear at any depth.
- Routing an identifier question to dense is survivable. Dense retrieval on identifiers
  is weak but not blind, and the surrounding prose often carries it.

So when the router is unsure, it chooses semantic. `LlmRouter` falls back the same way on
any error or unexpected reply. **Asymmetric costs should decide your defaults**, and
that reasoning generalises well beyond routing.

## What it actually routes

Run over the golden set, the heuristic sends **4 of 44** questions to lexical:

| kind | lexical | semantic |
|---|---|---|
| conceptual | 0 | 9 |
| lookup | **2** | 8 |
| multi_hop | 0 | 7 |
| unanswerable | **2** | 8 |
| vocab_mismatch | **0** | 8 |

Two things to read off that table.

**Zero vocabulary-mismatch questions were routed to lexical.** Those are precisely the
questions BM25 damages, and the router protects all eight of them. That is the lesson 04
failure fixed at its root, rather than dampened by a tuned constant.

**Only 4 questions are diverted at all**, so the aggregate effect on this benchmark will
be small by construction. That is not a weak result, it is the correct one: routing is a
targeted intervention, and a router that reclassified half the corpus would be doing
something else. Judge it on whether the questions it moved were the right ones.

## Heuristic or model?

`LlmRouter` is more flexible and costs a full generation before retrieval starts. On this
corpus the regex is as good and free — which is the usual outcome, because the thing
being classified is a surface property of the text.

Reach for the model when the routing decision needs world knowledge a regex cannot
express: "is this a question about our product or about the industry", "does this need
current data or historical". Not for "does this contain an identifier".

There is also a debugging argument for the heuristic. When a query is routed badly you
can see exactly which pattern fired. A model returns a label and no reason.

## The general pattern

Routing is worth understanding as more than a BM25/dense switch. The pattern is: **when
one component is systematically better for an identifiable subset of traffic, classify
the traffic instead of compromising globally.** The same shape applies to:

- cheap model for simple questions, expensive for hard ones
- skip reranking on queries where retrieval is already confident
- skip query transformation on queries that are already keyword-shaped (lesson 06's
  exercise 5)
- send time-sensitive questions to a live source rather than the index

Every one of those is the same decision: a global compromise, or a per-query choice.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 10 — Agentic RAG](../10-agentic-rag/), where the pipeline starts making
decisions that can fail in loops.
