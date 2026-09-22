# Exercise — Lesson 01

## 1. Try to make dense retrieval fail (and notice that you cannot)

The standard demonstration of dense retrieval's weakness is an identifier: embeddings
capture meaning, and `TX-4491` has almost none. So this should fail:

```bash
python lessons/01-naive-rag/run.py "What is error code TX-4491?" --retrieval-only
```

**Predict the rank of the chunk containing `TX-4491`, then run it.**

It comes back at rank 1. So does every identifier in `data/07-operations.md`:

| Query | Rank of the chunk holding the answer |
|---|---|
| What is error code TX-4491? | 1 |
| What does exit status 75 mean? | 1 |
| What is `FUSION_CONSTANT` set to? | 1 |
| Which port does the query service use? | 1 |

Two things are going on, and both are worth more than a successful demo would have been.

**The corpus is too small.** It is 41 chunks. Asking for the top 5 means asking for 12%
of everything, and at that ratio almost any retrieval method looks excellent. This is
the **ceiling effect**, and it is the single most common way RAG benchmarks lie. A
technique that shows no improvement on a corpus this size has not been shown to be
useless — it has been shown to be untestable here.

**The prose around the identifier gives it away.** `TX-4491` sits inside a paragraph
explaining that embeddings fail on error codes. The words "error code" are right there,
so semantic matching works fine. A rigged demonstration.

`data/07-operations.md` was added to give the harder case: identifiers used in passing,
in prose that is not about identifiers. It still ranks 1 — because of the ceiling, not
because the problem is not real.

**Takeaway:** be suspicious of any RAG demonstration on a small corpus, including this
one. Lesson 03 builds the measurement, and the first thing it will show is how little
room there is to improve at this scale.

## 2. Probe the refusal

```bash
python lessons/01-naive-rag/run.py "What is the capital of France?"
```

The model knows the answer; the context does not contain it; the system prompt says to
refuse. Run it several times, on both profiles if you have keys.

Record how often it refuses cleanly, answers anyway, or hedges. That ratio is your
informal groundedness baseline — lesson 03 replaces it with a measured one.

## 3. Sweep top-k

```bash
for k in 1 3 5 10; do
  python lessons/01-naive-rag/run.py "How does chunk size affect retrieval?" --top-k $k
done
```

Somewhere the answer stops improving and extra passages just bury the relevant one.
Where, on this corpus?

**You cannot answer that rigorously yet** — you are eyeballing it. Sit with how
unsatisfying that is. It is the entire argument for lesson 03.

## 4. Look at the prompt

```bash
python lessons/01-naive-rag/run.py "What is BM25?" --top-k 10 --show-prompt
```

Of those ten passages, how many are actually about BM25? The rest is what "context
precision" measures and what reranking (lesson 05) exists to fix.

## Record

Put your answers in `PROGRESS.md` under "Lesson 01 findings". The refusal ratio from
step 2 is the one to write down — it is the before-measurement for lesson 03.
