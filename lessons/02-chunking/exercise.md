# Exercise — Lesson 02

## 1. Predict, then sweep

Before running anything, write down your answer: **which chunk size will give the best
answers on this corpus, and why?**

```bash
python lessons/02-chunking/run.py
```

Now sweep for real and look at what comes back:

```bash
for size in 256 512 1024; do
  echo "=== chunk_size=$size ==="
  python -m ragkit.cli ingest --chunk-size $size
  python -m ragkit.cli search "How does chunk overlap work?" --top-k 3
done
```

Bigger chunks mean each result carries more context and fewer results fit in the prompt.
Which felt better? Write it down — and note that "felt better" is not evidence. In
lesson 03 you will run this same sweep with numbers attached, and you should come back
and check whether your intuition here was right.

## 2. Find the damage

`run.py` counts chunks that do not end on terminal punctuation as a proxy for
mid-sentence cuts. Find an actual example:

```bash
python -m ragkit.cli ingest --chunk-size 256
python -m ragkit.cli search "cross-encoder" --top-k 5
```

Look for a chunk that starts or ends mid-thought. Is the fragment still useful? Would a
model be able to answer from it, or does it need the part that got cut off?

## 3. Break the heading fix deliberately

In `ragkit/ingest/chunker.py`, make `_merge_headings` return `parts` unchanged, then:

```bash
python -m pytest tests/test_chunker.py -q
```

Two tests should fail. Read them, then read the chunks the broken version produces:

```bash
python -m ragkit.cli ingest && python -m ragkit.cli search "dimensionality" --top-k 3
```

Does the chunk about dimensionality still contain the word "Dimensionality"? Put the fix
back when you are done.

## 4. Break the overlap fix too

Change `_tail_words` to `return text[-max_chars:]` — the naive version. Run the tests,
then look at what the overlaps start with. Put it back.

## 5. Think ahead

`Chunk` has separate `text` and `context` fields, and `embed_text` joins them. Nothing
sets `context` yet.

Given what you now know about chunks losing meaning when they are separated from their
document, guess what lesson 07 puts in that field — and why `text` is kept pristine
rather than just concatenating the two at ingest time.

## Record

Add to `PROGRESS.md`: your prediction from step 1, and whether step 2 changed your mind
about how much mid-sentence cutting actually matters.
