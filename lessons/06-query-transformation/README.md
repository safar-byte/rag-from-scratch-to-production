# Lesson 06 — Query transformation

```bash
python lessons/06-query-transformation/run.py "why do I get charged twice"
python -m ragkit.eval.run --retrieval-only --strategy rerank --transform rewrite
```

## The idea

Everything so far has tried to improve the *index*. This lesson improves the *query*.

A user's question is an input to a retrieval system, not a search query. It carries
words that are useless for matching — "can you tell me roughly what the default timeout
is" contains one term that matters — and it frequently omits the words that would find
the answer. Someone asking "why is it slow after a while?" shares no vocabulary at all
with a document about unbounded cache growth. That is the **vocabulary mismatch**
problem, and it is the oldest problem in information retrieval.

## The four techniques

**Rewrite** strips the question to a search query: drop the framing, keep the content
terms. Cheap and easy to inspect. Helps most on conversational input, does nothing on
input that was already keyword-shaped, and can hurt by discarding a term that mattered —
which is why `Rewrite` keeps the original alongside the rewrite rather than replacing it.

**HyDE** inverts the problem. Instead of improving the query, write a fake answer and
retrieve with *that*, on the premise that an answer sits closer in embedding space to
the real answer than a question does. The failure mode deserves stating plainly: when
the model lacks the domain knowledge it hallucinates a passage about the wrong subject
and retrieves confidently against it. **HyDE does not degrade gracefully — it fails by
being precisely wrong.**

**Multi-query** fans out into several phrasings and fuses the results. Genuinely
improves recall, because different phrasings surface different chunks. Costs one
generation plus N retrievals, and the gains flatten fast — past about four variants the
extra queries mostly return what the earlier ones already found.

**Step-back** asks a broader question alongside the specific one. For "why did the
indexer exit with status 75", the step-back question is "how does the indexer handle
locking". Good for why-questions, noise for lookups.

## The cost, which is the actual subject of this lesson

Every one of these inserts **a full model call before retrieval starts**. On this
CPU-only machine that is ~25s added to a query whose retrieval took 1s. Multi-query then
multiplies the retrieval passes, and with a reranker underneath it multiplies the
reranking passes too — which is how a technique that looks cheap on paper becomes the
slowest thing in the pipeline.

The framing that matters is therefore not "which of these should I use" but **"does any
of these buy more than it costs on my corpus"**. Run the eval with `--transform` and
find out before adopting one.

## What to expect here

Two structural reasons to be sceptical of a big win on *this* corpus:

1. **17 of 44 golden questions are saturated.** Lookup and conceptual sit at 1.000 at
   every depth. A transformation cannot improve what is already perfect.
2. **The vocabulary-mismatch questions are already handled** by hybrid retrieval plus
   reranking, which took `vocab R@5` to 1.000 in lesson 05. There is very little left
   for a query rewrite to fix.

This is what "measure before adopting" means in practice. Query transformation is a real
technique that solves a real problem — and on a corpus where that problem has already
been solved another way, paying a model call per query for it is waste. The same
technique on a corpus with genuinely mismatched vocabulary and no reranker would look
entirely different.

## Implementation notes

Multiple variants produce multiple rankings, which is the problem hybrid search already
solved, so this reuses **RRF** rather than inventing a second merge strategy. "Rankings
that must be merged" does not care whether the axis was different retrievers or
different phrasings.

`_clean_lines` is deliberately forgiving. Small models ignore "one per line, no
numbering" often enough that parsing has to cope, and a malformed variant is worse than
no variant — it still costs a full retrieval pass.

`Identity` skips fusion entirely rather than running RRF over one list, so the control
condition is measured against the untouched pipeline.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 07 — Contextual retrieval](../07-contextual-retrieval/), which fixes the index
instead — and is the technique most likely to actually pay here.
