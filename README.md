# RAG: From Scratch to Production

A hands-on course and a working RAG service in the same repository. Thirteen lessons
take you from "embed, search, stuff into a prompt" to a system with hybrid retrieval,
reranking, contextual chunking, agentic correction loops, grounded citations, prompt
caching, and cost accounting — and every step of that journey is justified by a number
in [`benchmarks/results.md`](benchmarks/results.md), not by an opinion.

**Runs fully offline with no API keys.** Local models are the default. Cloud providers
(Claude + Voyage) are one config flag away when you want production-quality answers.

---

## Why this repo is shaped the way it is

Most RAG tutorials stop at the naive pipeline, which demos beautifully and then falls
apart on a real corpus. The three things that actually separate a toy from a production
system are not techniques at all:

1. **You can measure it.** The evaluation harness is Lesson 03 — before *any*
   optimisation. Every technique after it has to earn its place with a measured delta,
   and the ones that don't help are recorded as not helping. A RAG system you cannot
   measure is a RAG system you cannot improve.
2. **You can see inside it.** The retrieval inspector shows which stage surfaced each
   chunk and what it scored. Almost every RAG bug is a retrieval bug, and you cannot
   debug what the pipeline won't show you.
3. **You know what it costs.** Latency and dollars are recorded next to quality, because
   "better" that costs 40× more per query is a tradeoff, not a win.

The lessons stay thin. The real code accumulates in [`ragkit/`](ragkit/), so by Lesson 12
the lessons *are* the production service rather than thirteen disconnected scripts.

## Quickstart

```bash
git clone https://github.com/safar-byte/rag-from-scratch-to-production.git
cd rag-from-scratch-to-production

conda env create -f environment.yml
conda activate rag
pip install -e ".[dev,local]"

# generation on the local profile runs through Ollama
ollama serve &
ollama pull qwen2.5:1.5b

python -m ragkit.cli ingest
python -m ragkit.cli ask "What does the corpus say about retrieval?"
```

No `.env` needed — the default profile is local and offline. To switch to the cloud
path, `cp .env.example .env`, add your keys, and set `RAG_PROFILE=cloud`.

> **On CPU, model choice matters more than you would expect.** The defaults here are
> a 1.5B non-reasoning generator and a 90MB reranker, chosen by measurement on a
> GPU-less machine: a 4B *reasoning* model took over 10 minutes for one query,
> because it spends ~1000 tokens thinking before writing an answer. If you have a
> GPU, raise both — and re-measure rather than assume.

## The two profiles

| | `local` (default) | `cloud` (opt-in) |
|---|---|---|
| Embeddings | BGE-small via sentence-transformers | Voyage AI |
| Reranking | BGE cross-encoder | Voyage rerank |
| Generation | Ollama (`qwen2.5:1.5b`) | Claude |
| Vector store | Chroma, embedded | Chroma or Qdrant Cloud |
| Lexical | BM25 (`bm25s`) | same |
| API keys | none | `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` |
| Cost | free | metered, tracked per query |

Nothing outside [`ragkit/providers/`](ragkit/providers/) knows which profile is active.
Every lesson is written against three protocols — `Embedder`, `Reranker`, `Generator` —
so the same code runs both ways.

> Anthropic does not serve an embeddings endpoint, so the cloud profile really is two
> vendors. That is the honest shape of a production RAG stack, and it is why the
> abstraction has three protocols rather than one "LLM" object.

## Lessons

| # | Lesson | What you build | Milestone |
|---|---|---|---|
| 00 | [Setup](lessons/00-setup/) | Environment, provider protocols, the corpus | M0 |
| 01 | [Naive RAG](lessons/01-naive-rag/) | Ingest → embed → search → answer, end to end | M1 |
| 02 | [Chunking](lessons/02-chunking/) | Fixed, recursive and token-aware splitting | M1 |
| 03 | [**Evaluation**](lessons/03-evaluation-harness/) | Golden set, recall@k, MRR, nDCG, groundedness | M2 |
| 04 | [Hybrid search](lessons/04-hybrid-search-rrf/) | BM25 + dense, fused with RRF | M3 |
| 05 | [Reranking](lessons/05-reranking/) | Cross-encoder reranking over an over-fetched shortlist | M3 |
| 06 | [Query transformation](lessons/06-query-transformation/) | Rewriting, HyDE, multi-query, step-back | M4 |
| 07 | [Contextual retrieval](lessons/07-contextual-retrieval/) | LLM chunk prefixes, prompt caching, Batch API | M4 |
| 08 | [Parent-document & metadata](lessons/08-parent-doc-and-metadata/) | Small-to-big retrieval, metadata filters | M4 |
| 09 | [Routing](lessons/09-routing-and-multi-index/) | Query routing across multiple indexes | M4 |
| 10 | [Agentic RAG](lessons/10-agentic-rag/) | CRAG grading, self-RAG reflection, multi-hop tools | M5 |
| 11 | [GraphRAG](lessons/11-graphrag/) | Entity graph construction and traversal retrieval | M5 |
| 12 | [Production hardening](lessons/12-production-hardening/) | Caching, citations, cost, tracing, the inspector UI | M6 |

Each lesson folder holds exactly three files: `README.md` (the idea, when to use it, how
it fails, what it costs), `run.py` (thin — calls into `ragkit`), and `exercise.md` (a
task plus the eval delta you should expect).

## Results

`benchmarks/results.md` is the spine of this repo. Every lesson appends a row, so the
whole curriculum reads as one table showing what each technique was actually worth on
this corpus.

The first thing the harness reported was a problem with itself: 17 of the 25 golden
questions are **saturated** - perfect recall at every depth - so most techniques in
later lessons cannot show a measurable gain on this corpus yet. That finding is in the
table rather than hidden, and rows the harness cannot trust are stamped `⚠️ ceiling`.
An eval is supposed to tell you uncomfortable things.

## Project state

[`PROGRESS.md`](PROGRESS.md) is the authoritative "where am I" file, and
[`STUDY-PLAN.md`](STUDY-PLAN.md) / [`GOALS.md`](GOALS.md) carry the curriculum and the
learning goals. They are committed, so cloning this repo on another machine restores
the full context — including for Claude Code, which reads [`CLAUDE.md`](CLAUDE.md).

## License

MIT — see [LICENSE](LICENSE).
