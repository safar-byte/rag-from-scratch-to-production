# Goals

Why this repo exists, and how I will know it worked.

## Outcome goals

**1. Ship a public, production-grade RAG service.** Not a notebook. A service with an
API, an eval harness, cost accounting, and a UI, that someone else can clone and run.

**2. Be able to reason about retrieval quality from measurements.** The test is whether
I can look at a failing query and say which stage failed and why — retrieval depth,
chunking, the reranker, or the prompt — instead of guessing and changing things.

**3. Be able to justify every technique's cost.** For each technique in this repo I
should be able to state what it bought on this corpus, what it cost in latency and
dollars, and when I would not use it. "It's best practice" is not an answer.

**4. Have something worth showing.** A public repo that demonstrates judgement, not just
that I can call an embedding API.

## The real test

Given an unfamiliar corpus and a quality bar, I can build a RAG system that meets it,
justify each decision with a measurement, and say what I would do next if it were not
good enough.

## Skills checklist

Tick when you can explain it *and* have implemented it.

**Retrieval fundamentals**
- [ ] Explain what an embedding is and why cosine similarity is the usual metric
- [ ] Explain why dense retrieval fails on identifiers and error codes
- [ ] Explain the asymmetry of query vs document embeddings
- [ ] Choose a chunking strategy for a given corpus and defend it
- [ ] Explain the precision/context tension in chunk sizing

**Hybrid and reranking**
- [ ] Explain BM25's scoring and its `k1` / `b` parameters
- [ ] Explain why RRF uses ranks rather than scores, and what `k=60` does
- [ ] Explain the difference between a bi-encoder and a cross-encoder
- [ ] Explain why recall at shortlist depth caps final quality
- [ ] Pick a shortlist depth from measurement rather than convention

**Evaluation**
- [ ] Build a golden set without fooling myself
- [ ] Compute and interpret recall@k, MRR, nDCG@k
- [ ] Explain what groundedness measures and how it differs from relevance
- [ ] Diagnose from the two halves which stage is at fault
- [ ] State the limitations of LLM-as-judge

**Advanced**
- [ ] Implement contextual retrieval and make it affordable
- [ ] Explain when query transformation pays for its extra call
- [ ] Implement small-to-big retrieval and say when it beats plain chunks
- [ ] Implement a CRAG-style grading loop with a stopping condition
- [ ] Say when GraphRAG is worth its build cost — and when it is not

**Production**
- [ ] Design a prompt prefix that actually caches, and verify it
- [ ] Produce span-level grounded citations
- [ ] Account for cost per query and per completed task
- [ ] Version an index and re-ingest without downtime
- [ ] Trace a query through every stage of the pipeline

**Claude API**
- [ ] Use adaptive thinking and effort correctly (and never `budget_tokens` on Opus 5)
- [ ] Place cache breakpoints correctly and confirm the hit
- [ ] Use the Citations API, and know why it conflicts with structured outputs
- [ ] Use the Batch API for a bulk ingest-time pass

## Progress

| Milestone | Exit criterion | Done |
|---|---|---|
| M0 | `pytest` green on a clean clone, no API keys | ✅ |
| M1 | `ragkit.cli ask` returns a grounded answer offline | ⬜ |
| M2 | Reproducible baseline numbers in `benchmarks/results.md` | ⬜ |
| M3 | Measured lift from hybrid + reranking over baseline | ⬜ |
| M4 | Every M4 technique has a measured row, negatives included | ⬜ |
| M5 | Multi-hop questions the M3 baseline fails now pass | ⬜ |
| M6 | Cache-hit test passes; inspector renders a full trace | ⬜ |

## Anti-goals

Things I am deliberately **not** doing, so scope stays honest:

- Not training or fine-tuning an embedding model.
- Not building a general-purpose framework. This is one opinionated pipeline.
- Not chasing benchmark numbers on a public leaderboard corpus. The corpus here is small
  and specific on purpose, and the deltas matter more than the absolutes.
- Not supporting every vector database. Two adapters prove the interface; more is busywork.
