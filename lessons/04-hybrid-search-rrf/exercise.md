# Exercise — Lesson 04

## 1. Find a query where each retriever wins

```bash
python lessons/04-hybrid-search-rrf/run.py "What does exit status 75 mean?"
python lessons/04-hybrid-search-rrf/run.py "Why do I get charged twice?"
python lessons/04-hybrid-search-rrf/run.py "FUSION_CONSTANT"
```

For each, note the overlap line at the bottom. Find:

- a query where BM25 clearly beats dense (identifiers are the easy case)
- a query where dense clearly beats BM25 (ask in words the document does not use)
- a query where the two agree completely

In the third case, what did fusion contribute? Nothing — and that is worth internalising.
Fusion only does work when the retrievers disagree.

## 2. Reproduce the negative result

```bash
python -m ragkit.eval.run --retrieval-only --strategy dense
python -m ragkit.eval.run --retrieval-only --strategy hybrid
```

Hybrid should score *worse* at R@3 than dense, and worse on `vocab_mismatch`.

Before reading on: why would adding a second retriever make results worse?

Then look at the BM25 row's `vocab_mismatch R@1 = 0.250`. RRF gives that ranking the
same number of votes as the dense one. On those questions BM25 is not neutral, it is
confidently wrong, and equal-weight fusion propagates confident wrongness.

## 3. Tune k, then think about what you did

`DEFAULT_K = 60` in `ragkit/retrieve/hybrid.py`. Try 10 and 200:

```bash
python - <<'EOF'
from ragkit.config import get_settings
from ragkit.eval.run import run_eval
from ragkit.pipeline import RagPipeline
from ragkit.retrieve import DenseRetriever, Bm25Retriever, HybridRetriever
s = get_settings(); p = RagPipeline(s)
d = DenseRetriever(p._store, p.embedder); b = Bm25Retriever(p._bm25)
for k in (10, 60, 200):
    p._retriever = HybridRetriever([d, b], k=k, candidates=25)
    r = run_eval(s, pipeline=p, retrieval_only=True, quiet=True)["retrieval"]
    print(f"k={k:3d}  R@1={r['recall@1']:.3f} R@3={r['recall@3']:.3f} nDCG@5={r['ndcg@5']:.3f}")
EOF
```

Low `k` makes the top of each list dominate; high `k` flattens toward counting
appearances.

Now the real question: if you find a `k` that beats 60, have you learned something about
RRF, or have you fitted a constant to 37 questions? How would you tell the difference?
(Hold out half the golden set. Tune on one half, report on the other.)

## 4. Try weighted fusion

Modify `reciprocal_rank_fusion` to accept per-retriever weights and multiply each
contribution. Give dense 2.0 and BM25 1.0.

Does hybrid now beat dense? Almost certainly yes — you have told it to mostly ignore the
retriever that was hurting. Is that a technique, or is it a knob fitted to this corpus?

Compare against lesson 09's routing, which decides *per query* rather than globally.

## 5. Where does BM25 actually win?

BM25 beat dense on `multi_hop R@3` (1.000 vs 0.929). Find out which question and why:

```bash
python -m ragkit.eval.run --retrieval-only --strategy bm25 --write --label "scratch-bm25"
python -m ragkit.eval.run --retrieval-only --strategy dense --write --label "scratch-dense"
```

Diff the multi-hop entries in the two JSON files in `benchmarks/runs/`. Multi-hop
questions span documents that may share no vocabulary with each other — does that explain
it? Remove the scratch rows from `results.md` when done.

## Record

In `PROGRESS.md`: which query types each retriever won, and your answer to step 3 about
tuning versus overfitting.
