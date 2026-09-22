# Lesson 12 — Production hardening and the inspector

```bash
uvicorn ragkit.api.main:app --port 8000     # terminal 1
cd ui && npm install && npm run dev         # terminal 2, then open localhost:5173
```

Everything in lessons 01–11 improved what the system *finds*. This one is about running
it: refusing safely, caching, citing, costing, and being able to see inside it.

## Refusing safely

The thread that started with lesson 03's refusal failure ends here. Three levers, all
measured, at different price points:

| Mechanism | Correct refusal | False refusals | Cost | Where |
|---|---|---|---|---|
| default prompt | 0.300 | 0/5 | — | baseline |
| **strict prompt** | **0.900** | **1/5** | free | `generate/prompt.py` |
| score floor 0.60 | 0.500 | 0/34 | one comparison | `generate/abstain.py` |
| CRAG grading | reads the passages | measure it | one model call | `agent/grade.py` |

The strict prompt is the largest lever and it is free — and it over-refuses, declining
one valid question in five. The score floor is weaker and refuses nothing valid. Pick by
whether a wrong answer or a missing answer costs you more; both were measured precisely
so that choice is visible.

The score floor is calibrated, not guessed: **0.60 is the highest threshold that refuses
no valid question on this corpus**, and past 0.63 it starts rejecting real questions
faster than it catches fakes. Recalibrate it for any other embedding model or corpus —
cosine scores from a different model live on a different scale entirely.

One finding worth carrying: the gate reads the **dense** score even when reranking,
because cross-encoder scores separate answerable from unanswerable far *worse* (lowest
answerable −9.80 against highest unanswerable −4.16). The stronger ranker is the weaker
calibration signal. Ranking quality and calibration are different properties, and a
reranker is trained only for the first.

## Prompt caching

The largest cost lever on the cloud path, and the easiest to break silently.

Caching is a **prefix match**, rendered `tools` → `system` → `messages`. Any byte change
anywhere in the prefix invalidates everything after it. So: stable content first, the
user's question last — which is exactly why `build_prompt` puts context before the
question, a decision made back in lesson 01 for this reason.

**Verify it.** Assert `usage.cache_read_input_tokens > 0` on a repeated request. A zero
means something in the prefix varies between calls — a timestamp, a request id, an
unsorted JSON object, a tool list built from a set — and you are paying full price with
nothing reporting it.

The biggest win is lesson 07's contextual-chunk pass, where the document is a stable
prefix across every chunk derived from it.

## Citations

Two mechanisms, and they are not equal.

**Marker parsing** (`generate/answer.py`) asks the model to emit `[1]` and resolves those
back to chunks. It is the fallback and it is unreliable by construction: the model can
cite a passage it did not use, or use one it did not cite, and nothing here can detect
either.

**The Citations API** returns spans the model actually grounded in, rather than numbers
it typed. Set `citations: {"enabled": true}` on each `document` block. Note it is
**incompatible with `output_config.format`** — a 400 — so the citation path and the
structured-output path stay separate, and you choose per route.

This is also why `Chunk.text` and `Chunk.context` stayed separate through lesson 07: cite
generated text and the system can no longer prove anything it says.

## Cost accounting

`Usage` carries input, output, cache-read and cache-write tokens plus a dollar figure, and
the eval reports total cost per run. Two habits:

**Judge cost per completed task, not per request.** A cheaper model needing two retries
and a clarifying turn is not cheaper.

**Record cost next to quality, always.** "Better" that costs 40× more is a tradeoff, not
a win, and a results table without a cost column cannot tell you which you have.

## Observability

Almost every bad RAG answer is a bad retrieval, and that is invisible from the final text.
Every stage's scores stay attached to the chunk all the way through the pipeline —
`dense_rank`, `bm25_rank`, the RRF contributions, `rank_before_rerank`, the cross-encoder
score — which is what lets the inspector answer *why* a passage was in front of the model.

The three diagnostic patterns from `data/11-observability.md`, now mechanised:

- **Correct chunk absent at any depth** → retrieval failed; reranking is irrelevant.
- **Correct chunk at rank 40, shortlist 25** → deepen the shortlist. The eval's
  `first_relevant_rank` distinguishes exactly this case from the one above.
- **Correct chunk reached the prompt, answer still wrong** → retrieval is fine; look at
  the prompt or the model.

## The inspector

The best teaching device in this repo, and the reason the API returns far more than the
answer.

A single result renders as its whole history:

```
dense rank 1 · dense cosine 0.5774 · bm25 rank 1 · bm25 3.1803
rrf from dense 0.0164 · rrf from bm25 0.0164 · rrf fused 0.0328
rank before rerank 1 · cross-encoder 2.9710
```

The reranker badge shows movement up or down, which turns lesson 05's abstract claim
("reranking is not free at the very top; R@1 dipped 0.824 → 0.811") into something you
watch happen on a specific query.

`POST /retrieve` skips the model entirely. Once the embedder is warm it returns in about
a second, which turns the debugging cycle from half a minute into something you can
actually iterate in. It is the endpoint you will use most.

## Index versioning

`ingest` resets the index by default, decided in lesson 01: partially re-indexing a
corpus whose chunk boundaries have moved leaves orphans, and a stale index is much harder
to debug than a slow rebuild.

For a real system, `data/07-operations.md` describes the arrangement: an
`INDEX_GENERATION` counter, queries pinned to the generation current when they started,
and the old generation retained until in-flight queries drain. That is what lets a
rebuild happen without downtime — and note that the operational corpus this repo searches
has been describing the technique the repo needs all along.

## Exercise

See [exercise.md](exercise.md).

## The end of the course

Go back to [`benchmarks/results.md`](../../benchmarks/results.md) and read it as one
table. That is what the course was actually for: not thirteen techniques, but the habit
of asking what each one was worth and being willing to write down the answer when it was
"nothing".
