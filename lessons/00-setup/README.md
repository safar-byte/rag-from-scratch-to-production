# Lesson 00 — Setup

The environment, the corpus, and the one architectural decision everything else rests on.

## Set up

```bash
conda env create -f environment.yml
conda activate rag
pip install -e ".[dev,local]"
pytest
```

Python is pinned to **3.12** deliberately. Torch and sentence-transformers publish no
wheels for 3.14, so a newer interpreter fails at install with a confusing
"no matching distribution" error. If anything goes wrong during install, check
`python --version` before anything else.

The `[local]` extra pulls torch — a few GB. Skip it (`pip install -e ".[dev]"`) if you
only want to read code and run tests; the suite deliberately needs neither models nor
network.

For generation on the local profile you also need Ollama running:

```bash
ollama serve
ollama pull qwen2.5:7b
```

## The one decision that matters

Look at `ragkit/providers/base.py`. Three protocols — `Embedder`, `Reranker`,
`Generator` — and nothing else in the codebase imports a vendor SDK.

That sounds like ordinary hygiene. It is actually the decision that makes this repo
work, for three reasons:

**You can iterate for free.** Most of the time you spend on a RAG system goes into
retrieval logic: chunk boundaries, fusion weights, shortlist depth. None of that needs a
frontier model. Running those experiments locally at zero cost means you run more of
them, and running more of them is the whole game.

**You can test without mocking the world.** `tests/fakes.py` has a hash-based embedder
and an overlap-based reranker. They are semantically meaningless and completely
sufficient to prove the plumbing works, because quality is measured by the eval harness
(lesson 03), not by unit tests. A test suite that needs a GPU is a test suite nobody runs.

**Anyone can clone and run this.** A repo that demands two API keys before it prints
anything is a repo most people close.

The cost of the abstraction is real — one more indirection between you and the model —
and it is worth paying here. The signal that you got the boundary wrong is a lesson file
importing `anthropic` or `torch`.

## The two profiles

`RAG_PROFILE` selects the backend. Everything else is identical.

| | `local` (default) | `cloud` |
|---|---|---|
| Embeddings | BGE-small, 384-dim | Voyage |
| Reranking | BGE cross-encoder | Voyage rerank |
| Generation | Ollama `qwen2.5:7b` | Claude |
| Keys | none | `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` |

Note the shape of the cloud profile: it is **two vendors**, because Anthropic serves no
embeddings endpoint. That is not an inconvenience to abstract away — it is the honest
shape of a production RAG stack, and it is why there are three protocols rather than one
`LLM` object.

## The corpus

`data/` holds six short original documents about retrieval systems. They are written for
this repo rather than scraped, which keeps the licence unambiguous and — more usefully —
means the facts are specific enough to write verifiable golden answers against.

The corpus is also deliberately a little awkward: some facts are split across documents,
and a few terms are defined in one file and used in another. A corpus that is too clean
makes every retrieval strategy look equally good, which would make lesson 03 pointless.

## Exercise

See [exercise.md](exercise.md), and run the environment check first:

```bash
python lessons/00-setup/run.py
```

## Next

[Lesson 01 — Naive RAG](../01-naive-rag/) builds the whole pipeline at its simplest, so
there is something concrete to criticise.
