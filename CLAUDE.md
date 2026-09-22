# CLAUDE.md

Context for Claude Code working in this repository, on any machine.

> **Read [`PROGRESS.md`](PROGRESS.md) first.** It is the authoritative record of where
> the work stands. **Update it and commit before ending a session** — it is the only
> thing that carries state between machines.

## What this is

A course and a production RAG service in one repo. Thirteen lessons in `lessons/` build
up a real library in `ragkit/`. By lesson 12 the lessons *are* the service.

The organising principle: **measure first, optimise second.** The eval harness is
lesson 03, before any optimisation, and every technique after it must justify itself
with a row in `benchmarks/results.md`. Techniques that don't help are recorded as not
helping — a negative result is a finding, not a failure, and deleting it makes the repo
dishonest.

## Setup

```bash
conda env create -f environment.yml     # python 3.12 — see below
conda activate rag
pip install -e ".[dev]"                 # add ",local" for the offline ML models
pytest
```

**Python must be 3.12.** The author's machine has 3.14 as the system Python; torch and
sentence-transformers do not publish wheels for it. If a dependency install fails with
"no matching distribution", check `python --version` before anything else.

There is **no Docker** on the author's machine, so every dependency must run embedded or
hosted. That is why the vector store is Chroma rather than a containerised Qdrant.

**There is also no GPU.** Ollama runs on CPU, and that shapes two defaults:

- The generator is `qwen2.5:1.5b`, not something larger and not a reasoning model. A
  reasoning model (`qwen3:4b`) took **over 10 minutes** for a single RAG query here,
  because it spends ~1000 tokens thinking before it writes anything. The 1.5B
  non-reasoning model answers in ~30s. Retrieval quality is what this repo teaches; the
  generator only has to read the context it is handed.
- The reranker is `ms-marco-MiniLM-L-6-v2` (~90MB) rather than a BGE reranker (~2.3GB).

If you are on a GPU box, both of these are worth revisiting — and worth re-measuring
rather than assuming.

**The CLI pays model load on every invocation.** A single `ragkit ask` reports ~40s of
"retrieval" that is almost entirely loading BGE into memory; the eval harness, which
loads once and runs 37 questions, measures ~1s per query. Do not read per-query cost off
a one-shot CLI run.

## Architecture

```
ragkit/
  config.py      Settings + profile selection. The only place model IDs live.
  types.py       Document, Chunk, Scored, Citation, Usage, Answer.
  providers/     base.py defines Embedder / Reranker / Generator protocols.
                 local.py = sentence-transformers + Ollama. cloud.py = Claude + Voyage.
  ingest/ store/ retrieve/ query/ graph/ agent/ generate/ eval/ api/
```

### Rules that keep this coherent

1. **Never import a vendor SDK outside `ragkit/providers/`.** Everything else is written
   against the three protocols. This is what lets the same code run offline and in the
   cloud, and it is the first thing to check if a change feels awkward.
2. **Lessons stay thin.** `lessons/NN/run.py` wires together `ragkit` and prints. If real
   logic is accumulating in a lesson file, it belongs in `ragkit/`.
3. **The local profile must always work with no API keys and no network.** This is the
   repo's central promise; `tests/test_config.py` guards it.
4. **Every retrieval technique gets a row in `benchmarks/results.md`.**
5. **Heavy imports go inside functions or constructors**, so importing the provider
   factory never pulls in torch.

## Claude API specifics (cloud profile)

Applied in `ragkit/providers/cloud.py`. These are all common stale-prior mistakes:

- Model IDs are **bare**: `claude-opus-5`, `claude-haiku-4-5`. **Never append a date
  suffix.** A test asserts this.
- Use `thinking={"type": "adaptive"}`. **`budget_tokens` returns a 400 on Opus 5** —
  tune depth with `output_config={"effort": "low"|"medium"|"high"|"xhigh"|"max"}`.
- Answer generation runs at `effort="low"`: answering from retrieved context is reading
  comprehension, not reasoning. Raise it for the agentic pipelines in lesson 10.
- Always check `stop_reason == "refusal"` before reading `content` — a refusal is an
  HTTP 200 with no answer text.
- With adaptive thinking the response contains thinking blocks; take only `type == "text"`.
- **Prompt caching** is a prefix match, rendered `tools` → `system` → `messages`. Stable
  content first, the question last. Verify with `usage.cache_read_input_tokens > 0`; a
  zero means something in the prefix varies between requests.
- **Citations**: `citations: {"enabled": true}` on `document` blocks gives span-level
  grounding natively. It is **incompatible with `output_config.format`** (400), so the
  citation path and the structured-output path stay separate.
- Anthropic serves **no embeddings endpoint** — embeddings are Voyage or local.
- Bulk passes (contextual chunking, LLM-judge) use `get_generator(cheap=True)`.

## Conventions

- `ruff` for lint and format, line length 100. Run `ruff check . && ruff format .`.
- Type hints everywhere; `from __future__ import annotations` at the top.
- Comments explain *why*, not *what*. The lesson READMEs carry the teaching; the code
  carries the reasoning behind non-obvious choices.
- Commit per milestone, and update `PROGRESS.md` in the same commit.

## Git

This repo belongs to the **personal** GitHub account (`safar-byte`), not the work
account that is the machine's global git identity.

- Identity is set **locally**: `Muhammed Safar` / `safarmuhmmed@gmail.com`.
  **Never run `git config --global`** here.
- The remote uses the `github-personal` SSH alias, not plain `github.com`:
  `git@github-personal:safar-byte/rag-from-scratch-to-production.git`
- Never commit `.env`, an index (`.chroma/`), or model weights.
