# Progress

The authoritative "where am I" file. Pull this repo on any machine, read this file, and
you have the full context. Update it and commit before ending a session.

**Current position:** Milestone M0 complete. Next up: M1, lessons 01–02.

---

## Milestone status

| Milestone | Scope | Lessons | Status |
|---|---|---|---|
| M0 | Foundation: env, provider protocols, corpus, CI | 00 | ✅ done |
| M1 | Naive RAG end to end, offline | 01–02 | ⬜ next |
| M2 | Evaluation harness ⭐ | 03 | ⬜ |
| M3 | Hybrid search + reranking | 04–05 | ⬜ |
| M4 | Query transformation + structural retrieval | 06–09 | ⬜ |
| M5 | Agentic RAG + GraphRAG | 10–11 | ⬜ |
| M6 | Production hardening + inspector UI | 12 | ⬜ |

## What is done

**M0 — Foundation**

- Python 3.12 conda env (`environment.yml`); `pyproject.toml` with `local`, `cloud`,
  `graph`, `dev` extras so a clean clone can test without downloading torch.
- `ragkit/config.py` — settings, `RAG_PROFILE` selection, all model IDs in one place.
- `ragkit/types.py` — `Document`, `Chunk`, `Scored`, `Citation`, `Usage`, `Answer`.
- `ragkit/providers/` — `Embedder` / `Reranker` / `Generator` protocols, plus both
  backends: local (sentence-transformers + Ollama) and cloud (Voyage + Claude).
- `data/` — six original documents on retrieval, ~2,100 words, written so the golden
  Q&A set has verifiable answers and some questions need two documents.
- `tests/` — 9 tests, no network, no models. Guards the default-is-offline promise and
  that the fakes still satisfy the protocols.
- CI on push and PR: ruff + pytest on the local profile only.

## What is next

**M1 — Naive RAG, end to end.** Loaders, fixed and recursive chunking, the Chroma store,
dense retrieval, prompt assembly, answer generation, and a `ragkit` CLI. Must work fully
offline.

*Exit criteria:* `python -m ragkit.cli ingest && python -m ragkit.cli ask "..."` returns
a grounded answer against `data/` with no API keys set.

## Open questions

- Voyage model IDs in `config.py` (`voyage-3-large`, `rerank-2.5`) are defaults written
  from memory and **not yet verified against Voyage's current model list**. Check before
  the first cloud run. They are overridable via `.env`, so this blocks nothing on local.
- Ollama is not installed on the author's machine yet. Needed before M1 can be run
  end to end, though ingest and retrieval can be built and tested without it.
- Local judge reliability for M2 is unknown. A 7B model grading groundedness may be too
  noisy; the deterministic lexical-overlap fallback exists for that reason, and the
  decision of which to trust should be made by comparing both against ~20 hand labels.

## Decision log

**2026-09-22 — Local by default, cloud opt-in.** Three protocols and a profile switch
rather than one backend. Learning happens by iterating on retrieval, and iterating is
free offline. Also makes the repo runnable by anyone who clones it.

**2026-09-22 — Evaluation at lesson 03, before any optimisation.** The whole curriculum
hangs off this. Without it, each later technique is a matter of taste.

**2026-09-22 — Lessons build one shared library rather than standing alone.** Thirteen
self-contained scripts would duplicate everything and teach nothing about structure.

**2026-09-22 — Chroma, not Qdrant, as the default store.** No Docker on the author's
machine. Qdrant stays as an adapter behind the same interface and is documented as the
production target via Qdrant Cloud.

**2026-09-22 — Original corpus rather than a public-domain dump.** Unambiguous licence,
small enough to commit, and specific enough facts to write verifiable golden answers.

**2026-09-22 — Python 3.12, not the machine's 3.14.** Torch and sentence-transformers
publish no 3.14 wheels.
