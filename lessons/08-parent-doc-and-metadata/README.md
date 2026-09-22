# Lesson 08 — Parent-document retrieval and metadata filtering

```bash
python -m ragkit.eval.run --retrieval-only --strategy parent
python lessons/08-parent-doc-and-metadata/run.py "how does chunk overlap work"
```

## Small-to-big

Lesson 02 left a tension unresolved. Small chunks retrieve precisely, because the
matching passage is not diluted; small chunks answer poorly, because the surrounding
context is gone. Large chunks do the reverse.

Small-to-big refuses the trade. **Search over small chunks, hand the generator the
larger region each hit came from.** Precision where precision matters — matching —
and context where context matters — answering.

## Where lesson 02's bookkeeping pays off

Every chunk carries `start_char` and `end_char` into its source document. That felt like
fussy overhead when it was introduced. It is what makes this lesson a slice:

```python
parent_text = document.text[start - window : end + window]
```

No second index. No parent/child table. No duplicated storage. Had those offsets not
been recorded at chunking time, small-to-big would need an entirely separate ingest path
— which is the usual reason it gets skipped.

Two details in the implementation:

**Windows snap to boundaries.** A window cut at an arbitrary character starts mid-word
and ends mid-clause, which is exactly the damage this technique exists to undo.

**Overlapping windows are merged, not both returned.** Two chunks 300 characters apart
expand into two nearly identical windows. Returning both wastes context budget and
pushes a genuinely different passage out of the top k — the opposite of the intent.
Deduplication matters more here than anywhere else in the pipeline.

## What to expect

Parent expansion changes what the *generator* sees, not what retrieval *finds*. So
document-level recall should be roughly unchanged — the same documents are retrieved —
while answer quality may improve because each passage arrives with its surroundings.

That makes this one of the few lessons where **the retrieval metrics are the wrong place
to look.** If recall moves much either way, something is wrong: expansion should be
adding context to hits, not changing which documents are hit. The number to watch is
groundedness, and the cost is context budget — a 600-character window on five hits is
3,000 characters of prompt, before anything else.

## Metadata filtering

The other half of this lesson, and mostly a warning.

**Post-filtering** retrieves the top k by similarity and then discards anything failing
the predicate. Trivial to add, and usually wrong. If a tenant owns 1% of the corpus,
retrieving the global top 10 and filtering leaves roughly zero results — and the failure
is silent, a thin result set with no error.

**Pre-filtering** restricts the candidate set before the search runs, so the top k is
computed over eligible documents only. Correct by construction, harder to implement
efficiently, and what `ChromaStore.search(where=...)` does.

The rule is about **selectivity** — what fraction of the corpus the predicate keeps:

| Selectivity | Approach |
|---|---|
| keeps most of the corpus | post-filter is fine |
| keeps a modest fraction | pre-filter |
| keeps under ~1% | a separate index per partition; the approximate index is useless |

At high selectivity an approximate index walks a graph overwhelmingly composed of
ineligible nodes, and an exact scan over the eligible subset is both simpler and faster.

**Whatever fields you filter on must live on the chunk.** Resolving a permission at query
time puts a network round trip inside the retrieval path and makes that system's outage
your search outage. Denormalise at ingest.

> `Bm25Retriever` raises `NotImplementedError` when handed a `where` clause rather than
> ignoring it. A filter that is silently dropped is a permission leak waiting to happen,
> and an honest failure is worth more than a convenient one.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 09 — Routing](../09-routing-and-multi-index/), which finally addresses the
unfinished business from lesson 04.
