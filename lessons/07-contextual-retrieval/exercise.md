# Exercise — Lesson 07

## 1. Diagnose your corpus before spending anything

Read twenty random chunks with the document hidden:

```bash
python - <<'EOF'
import random
from ragkit.ingest import load_documents, chunk_documents, RecursiveChunker
chunks = chunk_documents(load_documents(), RecursiveChunker(512, 64))
for c in random.sample(chunks, 20):
    print("=" * 70)
    print(c.text[:300])
EOF
```

For each: **could you tell what this is about without being told?** Count them.

That fraction predicts your gain from contextual retrieval better than any benchmark
will. A corpus of self-contained reference entries gains little; a corpus full of
pronouns and relative references gains a lot. Write the number down before running
anything.

## 2. Look at the prefixes

```bash
python lessons/07-contextual-retrieval/run.py --preview
```

Are the generated sentences accurate? Do they resolve "the system" and "this approach"
into what they actually refer to, or do they just restate the chunk?

A prefix that paraphrases the chunk adds nothing — it contributes no new keywords and no
new context. If that is what you are seeing, the prompt needs work before the pass is
worth running.

## 3. Run it and measure

```bash
python lessons/07-contextual-retrieval/run.py --build     # slow, once
python -m ragkit.eval.run --retrieval-only --strategy rerank --write --label "contextual"
```

Compare against the non-contextual `rerank` row. **Predict the delta first**, using your
count from step 1.

## 4. Prove the lexical half matters

The claim is that the prefix helps BM25 as well as dense retrieval. Test it:

```bash
python -m ragkit.eval.run --retrieval-only --strategy bm25
```

Compare against the pre-contextual BM25 row in `benchmarks/results.md`. If BM25 did not
move, check that `ragkit/retrieve/bm25.py` indexes `embed_text` and not `text` — that
one-word difference is half the value of the technique, and nothing errors if you get it
wrong.

## 5. Break the citation guarantee

In `ragkit/types.py`, change `embed_text` to also overwrite `text`. Re-index, ask a
question, and look at the cited passage in the inspector.

The citation now points at generated text. Nothing errors. Put it back, and note that
this is exactly why the two fields exist separately — the lesson 00 exercise asked you
to guess this.

## 6. Price it properly

Work out what contextualising 100,000 chunks would cost on the cloud profile:

- naive: the document re-sent per chunk
- with prompt caching on the document prefix
- with caching plus the Batch API

You do not need exact figures; you need the ratio. That ratio is the difference between
"we cannot afford this" and "we do this on every ingest".

## Record

In `PROGRESS.md`: your count from step 1, the measured delta from step 3, and whether
BM25 moved in step 4.
