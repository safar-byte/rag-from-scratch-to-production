# Exercise — Lesson 11

## 1. Look at the graph that is not there

```bash
python lessons/11-graphrag/run.py
```

34 entities, 4 edges, mean degree 1.6. Read the arrow column: most entities connect to
nothing.

**Before tuning anything, explain why.** What is it about sixteen independent
explanatory essays that produces no co-occurrence? Describe the corpus shape that would
produce a dense graph instead — be concrete.

## 2. Try to force a denser graph

```bash
python - <<'EOF'
from ragkit.graph import build_graph
from ragkit.ingest import load_documents, chunk_documents, RecursiveChunker
chunks = chunk_documents(load_documents(), RecursiveChunker(512, 64))
for size in (256, 512, 1024):
    cs = chunk_documents(load_documents(), RecursiveChunker(size, 64))
    for mf in (1, 2, 3):
        g = build_graph(cs, min_frequency=mf)
        print(f"chunk={size} min_freq={mf} -> {g.stats()}")
EOF
```

Larger chunks contain more entities, so they co-occur more. Does that make the graph
*useful*, or just bigger? Look at what the new edges actually connect — are they
meaningful relationships or two terms that happened to land in the same 1024 characters?

This is the central trap of graph building: density is easy to manufacture and easy to
mistake for signal.

## 3. Find what the graph can answer that vectors cannot

```bash
python lessons/11-graphrag/run.py "How does BM25 relate to HNSW?"
python -m ragkit.cli search "How does BM25 relate to HNSW?" --top-k 3
```

Compare. Does the graph surface anything vector search missed? On this corpus, probably
not — and being able to say that with evidence is the point.

## 4. Break the extractor and watch the junk return

In `ragkit/graph/extract.py`, change `[ \t]` back to `\s` in the Title Case pattern:

```bash
python lessons/11-graphrag/run.py | head -25
```

Entities like `"Vector Search An"` reappear. Note that nothing errors, the graph still
builds, and the output still looks plausible at a glance. That is what makes extraction
bugs expensive.

Now also remove the trailing-stopword check in `_is_plausible` and look again. Put both
back.

## 5. Decide honestly

Write down, for a corpus you actually work with:

1. What fraction of your failing questions need information *connected across*
   documents, rather than found in one?
2. Would entities in your corpus recur across documents in varying combinations?
3. What would it cost to rebuild the graph on every corpus change?

If (1) is small or (2) is no, you have your answer, and you got it for the price of
reading the mean degree.

## 6. Build the version that would work

Design (do not implement) a typed-relation extractor: instead of co-occurrence, extract
`(subject, relation, object)` triples with a model.

- What would it cost on 113 chunks? On 100,000?
- How would you keep it current as documents change?
- How would you evaluate whether the extracted relations are *correct*?

That last question is the hard one, and it is why typed-relation GraphRAG is far less
common in production than in write-ups.

## Record

In `PROGRESS.md`: the mean degree at your best configuration from step 2, whether step 3
found anything, and your honest answer to step 5.
