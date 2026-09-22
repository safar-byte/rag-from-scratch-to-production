# Exercise — Lesson 05

## 1. Watch chunks move

```bash
python lessons/05-reranking/run.py "Why do I get charged twice?"
python lessons/05-reranking/run.py "Can somebody work out the original wording from the stored numbers?"
```

Each result shows its rank before and after reranking. The second query is one the
baseline ranked **8th** — watch where the cross-encoder puts it.

Find a case where the reranker moves something *down*. Read that chunk. Do you agree it
was overranked?

## 2. Sweep the shortlist depth

```bash
python lessons/05-reranking/run.py --sweep
```

**Predict first.** Where does quality stop improving — 10, 25, 50, 100?

The sweep takes several minutes; a CPU cross-encoder scoring 100 candidates for 37
questions is exactly the cost this lesson is about. Feeling that wait is part of the
point.

Two things to read off the table: where the quality curve flattens, and how the
millisecond column grows. It should be close to linear in depth. A hosted reranker bills
per document scored, so that column is also the bill.

Then set `rerank_candidates` in `config.py` to whatever the data supports rather than the
25 it currently says.

## 3. Prove the ceiling rule

> Recall at shortlist depth is a hard ceiling on final quality.

Test it. Set the shortlist to 3 and rerank to top-5:

```bash
python - <<'EOF'
from ragkit.config import get_settings
from ragkit.eval.run import run_eval
from ragkit.pipeline import RagPipeline
from ragkit.providers import get_reranker
from ragkit.retrieve import DenseRetriever, Bm25Retriever, HybridRetriever
from ragkit.retrieve.rerank import RerankingRetriever
s = get_settings(); p = RagPipeline(s)
base = HybridRetriever([DenseRetriever(p._store, p.embedder), Bm25Retriever(p._bm25)], candidates=25)
rr = get_reranker(s)
for depth in (3, 25):
    p._retriever = RerankingRetriever(base, rr, candidates=depth)
    r = run_eval(s, pipeline=p, retrieval_only=True, quiet=True)["retrieval"]
    print(f"shortlist={depth:3d}  R@5={r['recall@5']:.3f}")
EOF
```

At depth 3 the reranker cannot reach anything ranked 4th or worse, no matter how good it
is. Confirm that the score is capped by what retrieval supplied.

## 4. Was hybrid worth it? Re-run the decisive test

Lesson 04 concluded hybrid was not worth using. Lesson 05 concluded it was — as a pool,
not a ranker. Verify that yourself:

```bash
python - <<'EOF'
from ragkit.config import get_settings
from ragkit.eval.run import run_eval
from ragkit.pipeline import RagPipeline
from ragkit.providers import get_reranker
from ragkit.retrieve import DenseRetriever, Bm25Retriever, HybridRetriever
from ragkit.retrieve.rerank import RerankingRetriever
s = get_settings(); p = RagPipeline(s)
d = DenseRetriever(p._store, p.embedder)
h = HybridRetriever([d, Bm25Retriever(p._bm25)], candidates=25)
rr = get_reranker(s)
for name, base in (("dense->rerank", d), ("hybrid->rerank", h)):
    p._retriever = RerankingRetriever(base, rr, candidates=25)
    su = run_eval(s, pipeline=p, retrieval_only=True, quiet=True)
    print(name, su["retrieval"]["recall@5"], su["by_kind"]["vocab_mismatch"]["recall@5"])
EOF
```

State the conclusion in one sentence without using the word "better". What specifically
does hybrid contribute, and at which stage?

## 5. Count the cost honestly

Reranking took latency from ~1050ms to ~3390ms for a gain of 0.055 in R@5.

Write down, for each, whether you would deploy it:

- a support chatbot where the answer takes 4 seconds to generate anyway
- a search-as-you-type box
- a nightly batch job classifying 2 million documents
- a legal research tool where a missed document is a serious problem

The answer differs for each, and none of them is "reranking is good".

## 6. Swap the model

The default cross-encoder is `ms-marco-MiniLM-L-6-v2` (~90MB), chosen so the repo stays
runnable. Try the stronger one:

```bash
LOCAL_RERANK_MODEL=BAAI/bge-reranker-v2-m3 python -m ragkit.eval.run --retrieval-only --strategy rerank
```

That is a ~2.3GB download. Is the quality difference worth 25× the size on this corpus?
Record the answer either way — a "no" is as useful as a "yes".

## Record

In `PROGRESS.md`: the shortlist depth the sweep supports, your one-sentence statement
from step 4, and your deploy/do-not-deploy calls from step 5.
