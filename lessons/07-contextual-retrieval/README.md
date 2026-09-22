# Lesson 07 — Contextual retrieval

```bash
python lessons/07-contextual-retrieval/run.py --preview     # see prefixes, no indexing
python lessons/07-contextual-retrieval/run.py --build       # contextualise and re-index
python -m ragkit.eval.run --retrieval-only --strategy rerank
```

## The problem, concretely

A chunk that reads, in full:

> Revenue grew 3% over the previous quarter.

Which company? Which quarter? The chunk does not say, because the surrounding document
said it. Embedded as-is, it will never be retrieved by "How did Acme do in Q2 2024?" —
those words appear nowhere in it.

This is not an edge case. Pronouns, relative dates, and unqualified references ("the
system", "this approach", "the above") all produce chunks that are individually
unretrievable. The information is in the corpus and unreachable.

## The fix

Before embedding, ask a cheap model to write one or two sentences situating the chunk in
its document, and prepend that:

> This chunk is from Acme Corp's Q2 2024 earnings report, in the section on the cloud
> infrastructure segment.
>
> Revenue grew 3% over the previous quarter.

Now "Acme" and "Q2 2024" are in the embedded text, and the chunk is findable.

## Two details that are easy to get wrong

**The prefix is stored separately from the text.** `Chunk.context` and `Chunk.text` stay
distinct; `embed_text` joins them. If you concatenate at ingest, every subsequent
citation points at *generated* text rather than source text, and the system can no
longer prove anything it says. This is why `Chunk` had two fields from lesson 00 — the
exercise there asked you to guess why.

**The lexical index must get the prefix too.** The prefix adds exactly the keywords a
user is likely to type. `ragkit/retrieve/bm25.py` indexes `embed_text`, not `text`, for
this reason. Indexing the prefix for dense search only leaves half the value unclaimed,
and nothing will tell you that you did.

## Making it affordable

The naive implementation sends the whole document once per chunk — a 200-chunk document
transmitted 200 times. Three mechanisms fix it, in descending order of effect:

**1. Prompt caching.** The document is a stable prefix across every chunk derived from
it, so it is processed once and read from cache thereafter at a fraction of the input
cost. `ClaudeGenerator.generate(cache_system=True)` puts it in the cached system block.
Verify it worked by asserting `usage.cache_read_input_tokens > 0` — a zero means
something in the prefix varies between requests and you are paying full price silently.

**2. The Batch API**, roughly half price. This is a one-time ingest pass with no latency
requirement, which is precisely what batch is for.

**3. A cheap model.** Situating a chunk is an easy task. `get_generator(cheap=True)`.

Together these turn contextualising a corpus from prohibitive into a modest one-off.

## The local reality

None of the above applies offline, and the pass is simply slow: ~25s per chunk on CPU,
so ~45 minutes for this corpus's 113 chunks. Hence `ContextualiseCache`, keyed on the
document's content hash so editing a document invalidates the prefixes derived from it —
a stale prefix describing an older version of a document is worse than no prefix,
because it is confidently wrong.

Run it once, and every later experiment reads from the cache.

## When this is worth it

Contextual retrieval pays in proportion to how context-dependent your chunks are. A
corpus of self-contained reference entries gains little. A corpus of narrative documents,
meeting notes, threaded discussions, or anything heavy with pronouns and relative
references gains a great deal.

The diagnostic: read twenty random chunks with the document hidden. How many are
comprehensible on their own? That fraction predicts your gain better than any benchmark.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 08 — Parent-document retrieval](../08-parent-doc-and-metadata/), which attacks
the same problem from the other end: instead of adding context to a small chunk, return
the bigger region it came from.
