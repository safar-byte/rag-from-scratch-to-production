# Lesson 01 — Naive RAG

The whole pipeline, at its simplest.

```bash
python -m ragkit.cli ingest
python lessons/01-naive-rag/run.py "What is reciprocal rank fusion?" --show-prompt
```

No Ollama yet? Add `--retrieval-only`. Retrieval is the half worth staring at anyway.

## The five steps

1. **Load** — read documents off disk (`ragkit/ingest/loader.py`).
2. **Chunk** — split them into retrievable units (`ragkit/ingest/chunker.py`).
3. **Embed** — turn each chunk into a vector (`ragkit/providers/`).
4. **Store** — index the vectors for nearest-neighbour search (`ragkit/store/chroma.py`).
5. **Retrieve and generate** — embed the question, take the top k, put them in a prompt
   (`ragkit/retrieve/dense.py`, `ragkit/generate/`).

That is genuinely all RAG is. Everything in lessons 04 through 11 is a modification to
step 5, and everything in lesson 12 is about running it reliably.

## Why build the weak version first

Because every later lesson is a response to a specific way *this* version fails, and the
techniques only make sense once you have seen the failure they fix. Read about hybrid
search before watching dense retrieval whiff on an error code and it is an abstract
recommendation. Watch it whiff first and it is obvious.

So: run it, and try to break it. Suggested attacks in the exercise below.

## Details that are not incidental

**Query and document embeddings go through different methods.** `embed_query` prepends
an instruction prefix; `embed_documents` does not. BGE is trained that way. Use the
wrong one and nothing errors — you just quietly lose recall. This is the archetypal RAG
bug: the system keeps working, slightly worse, forever.

**Chroma returns a distance; the store converts it to a similarity.** Higher is better
everywhere in this codebase. Mixing the two conventions produces a retriever that
confidently returns the *least* relevant chunks, and the code reads fine.

**Cosine, not L2.** Set in the collection metadata. Document length should not
influence relevance.

**The system prompt tells the model to say when the context is insufficient.** This is
necessary and not sufficient. Without it the model answers from parametric knowledge —
the exact failure RAG exists to prevent. With it, the model still does sometimes. That
gap is why lesson 03 measures groundedness rather than trusting the instruction.

## What is wrong with this pipeline

Quite a lot, and naming it now sets up the rest of the course:

| Problem | Fixed in |
|---|---|
| Fails on identifiers, error codes, rare terms | 04 — hybrid search |
| Top-k by cosine is a weak ranking | 05 — reranking |
| The user's question is often a poor search query | 06 — query transformation |
| Chunks that are meaningless out of context | 07 — contextual retrieval |
| Precise retrieval and full context are in tension | 08 — parent-document |
| No way to tell whether retrieval even worked | 09-10 — grading and correction |
| Cannot answer questions spanning two documents | 10-11 — multi-hop, GraphRAG |
| No caching, no citations, no cost visibility | 12 — production |
| **No idea whether any change makes it better** | **03 — evaluation** |

The last one is the one that matters. Everything above it is a guess until you can
measure. That is the next lesson but one, and it is deliberately early.

## Exercise

See [exercise.md](exercise.md). The short version: try to make dense retrieval fail on
an identifier, discover that you cannot on a corpus this small, and understand why that
is a more useful result than success would have been.

## Next

[Lesson 02 - Chunking](../02-chunking/), the decision that constrains everything above.
