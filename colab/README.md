# Learn it in Google Colab

Every lesson has a `colab.ipynb` next to its `README.md`: a detailed, runnable version of
the lesson that needs nothing installed on your machine. Each notebook

- **teaches the topic from first principles** — the idea, why it exists, how it fails;
- **builds the technique by hand** in a few lines of numpy (cosine search, BM25, RRF,
  recall/nDCG, chunkers…) and then checks it against `ragkit`;
- **runs the lesson's experiments live** against the real corpus and golden set, with
  🔮 *predict-before-you-run* prompts;
- ends with **check-your-understanding questions** (answers hidden behind toggles) and
  **exercises**.

| # | Lesson | Open | Generator needed? | Suggested runtime |
|---|---|---|---|---|
| 00 | Setup, architecture and the corpus | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/00-setup/colab.ipynb) | one cell | CPU |
| 01 | Naive RAG, end to end | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/01-naive-rag/colab.ipynb) | yes | CPU (T4 faster) |
| 02 | Chunking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/02-chunking/colab.ipynb) | no | CPU |
| 03 | The evaluation harness ⭐ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/03-evaluation-harness/colab.ipynb) | yes | T4 recommended |
| 04 | Hybrid search and RRF | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/04-hybrid-search-rrf/colab.ipynb) | no | CPU |
| 05 | Reranking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/05-reranking/colab.ipynb) | no | CPU |
| 06 | Query transformation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/06-query-transformation/colab.ipynb) | yes | **T4** |
| 07 | Contextual retrieval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/07-contextual-retrieval/colab.ipynb) | yes | **T4** |
| 08 | Parent-document and metadata | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/08-parent-doc-and-metadata/colab.ipynb) | no | CPU |
| 09 | Routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/09-routing-and-multi-index/colab.ipynb) | one section | CPU |
| 10 | Agentic RAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/10-agentic-rag/colab.ipynb) | yes | T4 recommended |
| 11 | GraphRAG | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/11-graphrag/colab.ipynb) | no | CPU |
| 12 | Production hardening and the inspector | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/safar-byte/rag-from-scratch-to-production/blob/main/lessons/12-production-hardening/colab.ipynb) | yes | CPU |

## How the setup works

The first cell of every notebook clones this repository into the runtime and calls
`colab_setup.setup()` from this folder, which:

1. installs the dependencies listed in `pyproject.toml` directly (Colab already ships
   torch, so this takes about a minute) and imports `ragkit` from the clone — so Colab's
   Python version never trips the 3.12 pin, which exists for a torch-wheel problem on the
   author's machine, not for Colab;
2. if `USE_LLM` is ticked, installs [Ollama](https://ollama.com), starts it in the
   background and pulls `qwen2.5:1.5b` (~1GB, once per runtime);
3. is idempotent — re-running a setup cell, or opening a second notebook in the same
   runtime, skips whatever is already done.

Everything runs offline-style: no API keys. Lesson 12 has one optional cell that uses
`ANTHROPIC_API_KEY` / `VOYAGE_API_KEY` from Colab's **Secrets** panel if you add them.

**GPU:** *Runtime → Change runtime type → T4 GPU* for the lessons marked above. Ollama and
sentence-transformers use it automatically; nothing in `ragkit` changes. On CPU those
lessons still run, just slower (lesson 07 contextualises 113 chunks — 10–20 minutes on
CPU, a few on a T4; results are cached, so a re-run is instant).

**Your numbers may differ slightly** from the ones quoted in the notebooks: different
hardware can change a small model's output even at temperature 0, and adjacent results on
this corpus often differ by one question. Compare the *pattern*, not the third decimal.

**Nothing you change in Colab is saved to the repo.** To keep work, download the notebook
(*File → Download*) or save a copy to Drive.

The notebooks open from the `main` branch. To run a different branch, change `BRANCH` in
the setup cell.

## Running the notebooks locally

They also run in Jupyter on your own machine: the setup cell finds the repo you are
already in instead of cloning. It never installs Ollama outside Colab — if Ollama is
installed it is started and the model pulled; if not, generation cells skip politely.

```bash
pip install -e ".[dev,local,graph]" jupyter pandas matplotlib
jupyter lab lessons/
```
