# Lesson 02 — Chunking

```bash
python lessons/02-chunking/run.py
```

No models needed for the statistics — it reads the corpus and splits it.

## Why this is the highest-leverage lesson

Nothing downstream can recover information that chunking destroyed. A fact split across
two chunks so that neither one states it is gone. The best reranker in the world cannot
reorder its way to a passage that does not exist, and the best model cannot ground an
answer in text it was never shown.

Chunking is also the part everyone copies from a tutorial without measuring.

## The core tension

**Small chunks** retrieve precisely — the matching passage is not diluted by surrounding
text — but fragment context, so a two-sentence fact may land half in each of two chunks.

**Large chunks** preserve context but dilute the embedding. A 2000-token chunk spanning
twelve topics has an embedding that is a blurry average of all twelve, and is strongly
retrieved by nothing.

512 tokens with 64 of overlap is a common starting point. It is a starting point, not a
recommendation.

## What the run prints

On this corpus:

| strategy | chunks | mean | split mid-sentence |
|---|---|---|---|
| fixed(256/64) | 65 | 247 | 86% |
| fixed(512/64) | 31 | 450 | 80% |
| recursive(256/64) | 67 | 238 | 34% |
| **recursive(512/64)** | **34** | **411** | **5%** |
| recursive(1024/64) | 17 | 765 | 5% |

Fixed-size splitting cuts through a sentence roughly four times out of five. Recursive
splitting at the same size gets that to one in twenty, because it tries paragraph
breaks, then line breaks, then sentence boundaries before resorting to a hard cut.

**This table does not tell you which is best.** It measures damage, not answer quality,
and the relationship between the two is exactly what lesson 03 exists to establish. Notice
how tempting it is to conclude "recursive(512/64) wins" from a column that never touches
retrieval.

## Two bugs worth knowing about

Both were found while building this lesson, and both are the kind that never raise an
error. `tests/test_chunker.py` now guards each one.

**Orphaned headings.** A greedy buffer fills up to the size limit and flushes, which
means a heading that happens to land at the boundary ends the *previous* chunk. The
section body then loses its most discriminative keyword — "Dimensionality" is exactly
what someone would search for — and the previous chunk gains a label for content it does
not contain, inviting the wrong retrieval. `_merge_headings` glues each heading to the
section it introduces before the buffer ever sees it.

**Mid-word overlap.** Taking the last N *characters* of the previous piece produces
overlaps that begin `"rly always retrieval depth"` instead of `"nearly always"`. The
fragment is noise to the embedder and a junk token to BM25. `_tail_words` snaps the tail
forward to a word boundary.

Neither bug crashes anything. Both quietly cost recall. That is what most RAG bugs look
like, and it is why lesson 03 is where it is.

## Overlap

Overlap repeats the tail of one chunk at the head of the next so a fact on the boundary
survives somewhere. Typically 10–20% of chunk size. It costs storage and creates
near-duplicate results, which is a reason to deduplicate before building the prompt.

## Strategies not implemented here

**Semantic chunking** (lesson 07 territory) embeds each sentence and breaks where
consecutive similarity drops, so boundaries land where the topic actually turns. Costs
one embedding call per sentence at ingest.

**Structural chunking** uses the document's own markup. Where real structure exists it
usually beats anything inferred — `_merge_headings` is a small step in this direction.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 03 — Evaluation](../03-evaluation-harness/). Everything so far has been
unmeasured. That stops next.
