# Lesson 03 — Evaluation

The pivot point of the whole course.

```bash
python -m ragkit.eval.run --retrieval-only            # no generation, no judge
python -m ragkit.eval.run                             # full, needs Ollama
python -m ragkit.eval.run --write --label "my change" # append to benchmarks/results.md
```

## Why this comes before any optimisation

Everything in lessons 04 through 11 is a claim: hybrid search helps, reranking helps,
contextual retrieval helps. Every one of those claims is true *on some corpus, for some
questions*, and false elsewhere. Without measurement you are picking techniques by
reputation, and reputation is how people end up with a slow expensive pipeline that is
no better than the naive one they started with.

So the harness comes first, and every later lesson has to produce a number.

## The two halves

Retrieval and generation fail independently, so they are measured separately. This is
not tidiness; it is the entire diagnostic value of the harness:

| Retrieval | Groundedness | Diagnosis |
|---|---|---|
| high | high | working |
| high | low | prompting or model problem — retrieval found it, generation fumbled it |
| low | high | generation is doing well with bad material; **fix retrieval** |
| low | low | start with retrieval; generation cannot be assessed until it has material |

One end-to-end score collapses all four rows into "bad" and tells you nothing about
where to work.

## Retrieval metrics

**recall@k** — fraction of relevant documents in the top k. The most important single
number, because a document never retrieved cannot be reranked, cited, or reasoned over.
Recall is the ceiling on everything downstream.

**precision@k** — fraction of the top k that is relevant.

**MRR** — mean of 1/rank of the first relevant result. Rewards a correct answer at the
very top, ignores everything after it.

**nDCG@k** — credits relevant results by 1/log2(rank+1), normalised against the best
possible ordering. Unlike MRR it keeps caring after the first hit, which is what makes
it the right metric for the multi-hop questions where two documents must both be found.

All at **document** level, not chunk level. Chunk ids change every time `chunk_size`
changes, so a chunk-level golden set silently invalidates itself the moment you tune
chunking — which is precisely the experiment you most want to run.

## Generation metrics

**Groundedness** — is every claim supported by the retrieved context? Low means
hallucination.

**Relevance** — does the answer address the question? An answer can be perfectly
grounded and completely unresponsive.

**Correct refusal** — scored only on the three unanswerable questions. Did the system
decline, or confabulate? This is the most under-tested behaviour in RAG and the fastest
way to tell a demo from a deployable system.

### What it caught

The best pipeline in this repo — R@5 = 0.987, top of every retrieval metric — scores
**0.000** on correct refusal. It answered all three:

| Question (nothing in the corpus answers it) | What it said |
|---|---|
| What is the capital of France? | "Paris is the capital of France." |
| Default value of `SHARD_REPLICATION_FACTOR`? | **"The default value ... is 1."** |
| Which cloud provider in production? | **"AWS (Amazon Web Services)"** |

The middle one is the one to sit with. `SHARD_REPLICATION_FACTOR` does not exist
anywhere — it was invented for the golden set precisely because it *sounds* like it
should. The system produced a plausible default and stated it as confidently as the
answers it got right.

And the system prompt already says to decline when the context is insufficient. It
declines anyway roughly never. **A prompt instruction reduces confabulation; it does not
eliminate it.** That gap is the entire reason this is a measured column and not an
assumption, and it is why a golden set without unanswerable questions is not finished.

Note what would have happened without this column: every retrieval number improved
across lessons 04 and 05, the table looked like steady progress, and the system was
confabulating on 100% of the questions it should have refused the whole time.

## Two judges, on purpose

`DeterministicJudge` uses lexical overlap and keyword checks. Crude, free, perfectly
reproducible, and runs with no model at all — which is what extends the repo's offline
promise to evaluation rather than stopping it at retrieval.

`LlmJudge` grades against a rubric at temperature 0 on the cheap model.

Keeping both is methodology, not redundancy. **An LLM judge is a model with opinions,
and the only way to know whether to trust it is to compare it against something that
cannot have opinions.** Hand-label twenty answers, score them with both, and see how
they compare. If they disagree wildly, the judge is miscalibrated and its numbers are
not facts.

Two limits to state plainly, because they are easy to forget once the numbers start
looking authoritative: judges favour fluent, longer answers, and **scores from different
judges are not comparable** — changing the judge model invalidates every historical row
in the results table.

## What the first run actually showed

This is the interesting part, and it is a negative result.

| kind | n | recall@1 | recall@3 | recall@5 | max R@1 |
|---|---|---|---|---|---|
| conceptual | 9 | 1.000 | 1.000 | 1.000 | 1.000 |
| lookup | 8 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 5 | 0.500 | **0.900** | 0.900 | 0.500 |
| unanswerable | 3 | 1.000 | 1.000 | 1.000 | 1.000 |

**Lookup and conceptual are saturated.** Perfect at every depth. No technique in lessons
04 through 11 can show a gain on those seventeen questions, because there is no gain
available. If you ran hybrid search tomorrow and reported "no improvement", you would be
reporting a property of the corpus, not of hybrid search.

**The `max R@1` column exists because of a trap.** A multi-hop question has two relevant
documents, so recall@1 cannot mathematically exceed 0.5. Reading that row as "retrieval
fails half the time on multi-hop" is wrong — it is a *perfect* score at k=1. The harness
prints the arithmetic cap next to the score and colours the number green when it is at
the cap, because this misreading is otherwise guaranteed.

**`multi_hop recall@3 = 0.900` is the only genuinely unsaturated number in the table.**
One question out of five misses one of its two documents even at depth 3. That single
number is the entire measurable headroom in this harness right now.

The honest conclusion: **this corpus is too small.** 80 chunks, top-5 is 6% of
everything. The harness now says so itself — it prints a saturation warning and stamps
`⚠️ ceiling` on any row it cannot trust.

That is what a working eval does. It is supposed to tell you uncomfortable things, and
the first uncomfortable thing it said was that the benchmark it was built on cannot yet
support the conclusions the rest of the course wants to draw.

## Designing a golden set without fooling yourself

The failure mode is writing questions from the documents. Do that and you produce
questions phrased in the document's own vocabulary, which every retriever answers
perfectly, and you have measured nothing. The `lookup` and `conceptual` rows above are
partly that mistake.

Better: write the questions first from what a user would actually ask, then find which
documents answer them, and keep the ones where the answer exists but the phrasing does
not match. Four kinds of question are worth including deliberately:

- **lookup** — a specific value stated in one place
- **conceptual** — requires explanation, not extraction
- **multi-hop** — needs two or more documents
- **unanswerable** — the corpus genuinely does not cover it

Reporting a single mean over mixed kinds hides exactly the failures worth finding. The
harness breaks the score down by kind for that reason.

## Reproducibility

Temperature 0 everywhere. Two runs of the same configuration must produce the same
numbers, or you cannot attribute a delta to your change rather than to sampling noise.
The exception is the LLM judge, which can vary slightly even at temperature 0 — one more
reason to check it against the deterministic grader.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 04 — Hybrid search](../04-hybrid-search-rrf/). Now with a number to beat — and
with the knowledge that only the multi-hop questions can move.
