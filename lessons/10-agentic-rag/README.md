# Lesson 10 — Agentic RAG

```bash
python lessons/10-agentic-rag/run.py "What is the default value of SHARD_REPLICATION_FACTOR?"
python lessons/10-agentic-rag/run.py --multihop "Dense retrieval fails on error codes. Which method fixes that and why?"
```

The pipeline stops being a straight line here and starts making decisions.

## Why this lesson exists: the other half of the refusal problem

Lesson 03 found the worst result in this repo — the pipeline scored **0.000 on correct
refusal** and invented a default value for a setting that does not exist. The score floor
in lesson 12's abstention gate catches half of those for free. It structurally cannot
catch the other half, and the reason is worth sitting with:

```
"What is the default value of MAX_CHUNK_BYTES?"      cosine 0.689   (invented)
"What is the default value of RETRIEVAL_FANOUT?"     cosine 0.685   (real)
```

Both are questions about configuration defaults over a corpus full of configuration
defaults. They retrieve the same passages with the same confidence. **No threshold
separates them**, because a similarity score measures topical closeness, not
answerability.

Telling them apart requires reading the passages and asking whether they contain the
answer. That is a model call. That is CRAG.

## Corrective RAG

```
retrieve -> grade
    correct    -> answer
    ambiguous  -> widen the search once, then answer
    incorrect  -> refuse
```

The grader's entire job is one distinction the prompt hammers on: *being about the same
topic is not the same as containing the answer.* Most grading mistakes are made by
scoring topical similarity — which is exactly what the cheap threshold already does, so
a grader that makes that mistake adds cost and nothing else.

The `incorrect` branch never calls the generator, so the model is never given the
opportunity to invent something plausible from passages that do not support it.

## Self-RAG and reflection

CRAG grades the *retrieval*. The natural next step is grading the *answer* — generate,
check it against the context, regenerate if it is ungrounded.

This repo implements the first and not the second, deliberately. Reflection doubles or
triples generation cost, and on this corpus the measured groundedness problem is
concentrated in the unanswerable questions, which CRAG already catches before generation
happens at all. Adding reflection would be paying for a fix to a problem that has already
been fixed upstream, which is the most common way agentic pipelines get expensive.

The general rule: **fix a failure at the earliest stage that can see it.** Refusing
before generation is cheaper and more reliable than generating and then catching it.

## Multi-hop retrieval

Some questions cannot be answered by any single retrieval, because the terms needed for
the second search appear only in the results of the first:

> Dense retrieval fails on error codes. Which retrieval method fixes that, and why does
> it work where embeddings do not?

One search finds the passage about dense retrieval's weakness. Nothing in the question
says "BM25" — that passage is reachable only after reading the first result.

`MultiHopRetriever` lets the model write the follow-up query. That is useful and
hazardous, so three properties are enforced:

**A hard cap.** `max_hops` bounds extra searches. A loop that decides its own inputs
needs a bound that it does not control.

**Monotonic accumulation.** Results only ever grow, deduplicated by chunk id. A bad
follow-up query wastes time; it cannot destroy a good result set.

**Convergence detection.** A follow-up repeating an earlier query means the loop has
converged, and continuing burns the whole budget for nothing. Without this check, a model
that keeps rephrasing the same thing costs you the full hop limit on every query.

## The thing to be careful about

An agentic pipeline can fail in ways a straight-line one cannot: loops, runaway cost,
and cascading model errors where a bad grading decision routes into a bad correction.

Every loop in this module takes its bound as a constructor argument rather than a
constant buried in a branch — because the bound is a deployment decision, and a limit
you cannot see is a limit you will not tune.

The grader also **fails open**: if it errors, the query proceeds ungraded. An unavailable
grader should degrade the system to its previous behaviour, not take it down. The cost of
that choice is that a silently broken grader looks like a working one, so `Grading.reason`
records what happened.

## Cost

Only 7 of 44 golden questions are multi-hop. Running hops on all of them multiplies cost
for nothing, which is an argument for routing into multi-hop (lesson 09) rather than
defaulting to it.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 11 — GraphRAG](../11-graphrag/), and a corpus that turns out to be the wrong
shape for it.
