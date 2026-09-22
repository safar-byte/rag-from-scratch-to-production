# Exercise — Lesson 00

## 1. Check the environment

```bash
python lessons/00-setup/run.py
```

Only `python`, `core` and `corpus` must pass to read the code and run the tests. If
`python` is not 3.12, fix that before anything else — a newer interpreter fails the
torch install with a confusing "no matching distribution" message that looks like a
network problem.

## 2. Find the abstraction

Read `ragkit/providers/base.py`. Three protocols, and nothing outside
`ragkit/providers/` imports a vendor SDK.

```bash
grep -rn "import anthropic\|import voyageai\|import torch\|sentence_transformers" ragkit/ | grep -v providers/
```

That should return nothing. Why does it matter? Three reasons are in the lesson README —
find the one you think is most important and say why.

## 3. Why is "default is offline" worth a test?

```bash
python -m pytest tests/test_config.py -v
```

Read `test_default_profile_is_local`. It asserts a default. Tests usually assert
behaviour, so why is this one worth writing?

(The answer is not "defensive coding". Think about what a new person's first experience
of this repo would be if that default ever flipped.)

## 4. Guess ahead: why two text fields?

`ragkit/types.py` gives `Chunk` both `text` and `context`, joined by `embed_text`.
Nothing sets `context` yet.

Guess what lesson 07 puts there, and — the harder half — why `text` is kept pristine
rather than the two being concatenated at ingest.

Hint: what happens to a citation if you cite generated text?

## 5. Find the asymmetry

In `ragkit/providers/local.py`, `QUERY_INSTRUCTION` is applied in `embed_query` but not
`embed_documents`.

What breaks if you apply it to both? What breaks if you apply it to neither? Crucially:
**how would you notice either mistake?** Nothing raises.

## 6. Read the corpus critically

```bash
head -40 data/01-embeddings.md
cat data/README.md
```

The corpus was written for this repo rather than scraped. The README says it is
"deliberately awkward in places". Find one of those places.

Then a harder question: if you wrote both the corpus and the questions that test
retrieval over it, what kind of result would you expect — and should you trust it?
(Lesson 03 finds out the answer is "no", the hard way.)

## Record

Start a "Lesson findings" section in `PROGRESS.md`. Your answers to steps 4 and 6 are
the ones worth checking back on later.
