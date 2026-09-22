# Exercise — Lesson 10

## 1. Watch CRAG catch what the threshold cannot

```bash
python lessons/10-agentic-rag/run.py "What is the default value of SHARD_REPLICATION_FACTOR?"
python lessons/10-agentic-rag/run.py "What is the default value of RETRIEVAL_FANOUT?"
```

The first is invented; the second is real. Their cosine scores are 0.689 and 0.685 — no
threshold separates them.

Does the grader? Read the verdict and the reason for both. If it grades the invented one
`correct`, read the passages it was shown and work out whether *you* could have told from
them alone.

## 2. Measure the refusal rate

```bash
python -m ragkit.eval.run --strategy rerank --no-llm-judge --score-floor 0.60
```

Compare `correct_refusal` against the ungated baseline of 0.000. Then work out what a
CRAG-gated run would score by grading the 10 unanswerable questions by hand first, and
predicting.

The interesting number is not just how many it caught but **how many real questions it
wrongly refused.** A grader that refuses everything scores perfectly on refusal and is
useless. Always check both directions.

## 3. Find the multi-hop question that needs a second search

```bash
python lessons/10-agentic-rag/run.py --multihop "Dense retrieval fails on error codes. Which retrieval method fixes that, and why does it work where embeddings do not?"
```

Read the follow-up query the model wrote. Was it a genuinely new search, or a rephrasing
of the original? Did the second hop add documents the first missed?

Then try it on a lookup:

```bash
python lessons/10-agentic-rag/run.py --multihop "What port does the query service use?"
```

It should stop after hop 0. If it does not, you are paying two extra model calls for a
question one search answered — which is the argument for routing into multi-hop rather
than defaulting to it.

## 4. Break the loop guards, deliberately

Three protections, each worth removing once to see what it was doing:

**Remove the convergence check** (the "follow-up repeated an earlier query" branch). Find
a question where the model keeps rephrasing. How many hops does it burn?

**Make accumulation destructive** — replace the results each hop instead of merging. Find
a case where a bad follow-up query loses a good first-hop result.

**Raise `max_hops` to 10.** Time it. Now imagine that in a request handler with a 30-second
timeout.

Put all three back.

## 5. Make the grader fail

The grader fails open: if it errors, the query proceeds ungraded. Force it:

```python
class BrokenGrader(ContextGrader):
    def grade(self, question, results):
        raise RuntimeError("grader is down")
```

Confirm queries still work. Then consider: how would you notice this in production? A
silently broken grader looks exactly like a working one, and the refusal rate quietly
returns to 0.000.

What would you alert on?

## 6. Argue the other side

This repo does not implement Self-RAG reflection, on the grounds that CRAG already
catches the measured problem earlier and more cheaply.

Find a failure mode reflection would catch that CRAG cannot. (Hint: CRAG grades the
retrieval. What if retrieval is fine and the *answer* misreads it?) Is that failure
present in this corpus? How would you measure it before building anything?

## Record

In `PROGRESS.md`: the refusal rate with CRAG and how many real questions it wrongly
refused; and your answer to step 6.
