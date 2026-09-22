# Lesson 04 — Hybrid search and RRF

```bash
python -m ragkit.cli ingest                                      # builds both indexes
python -m ragkit.eval.run --retrieval-only --strategy dense
python -m ragkit.eval.run --retrieval-only --strategy bm25
python -m ragkit.eval.run --retrieval-only --strategy hybrid
python lessons/04-hybrid-search-rrf/run.py "exit status 75"      # side by side
```

## The idea

Dense and lexical retrieval fail in opposite directions.

Dense retrieval matches *meaning*, so it bridges vocabulary mismatch — a user asking
about "being charged twice" can reach a document about "duplicate transactions" — and it
fails on tokens that carry no meaning: identifiers, version numbers, error codes.

BM25 matches *terms*, so it nails `TX-4491` and cannot connect "car" to "automobile".

Run both, merge the results, get the strengths of each. That is the theory. This lesson
is mostly about what happened when it was measured.

## Reciprocal Rank Fusion

You cannot merge by score. A cosine similarity of 0.82 and a BM25 score of 14.3 are not
comparable, and normalising is fragile because BM25 has no upper bound — the maximum
depends on the corpus, the query and the term statistics.

RRF discards the scores and uses only the ranks:

```
score(d) = Σ over retrievers r of  1 / (k + rank_r(d))
```

with `k = 60` conventionally. The constant damps the top ranks, so the gap between rank 1
and rank 2 is small. A document ranked moderately well by *both* retrievers can outrank
one ranked first by only one — because agreement between two independent methods is
strong evidence. RRF counts votes rather than averaging opinions.

No training, no normalisation, no per-corpus tuning.

## What actually happened

| Strategy | R@1 | R@3 | R@5 | nDCG@5 | vocab_mismatch R@3 | ms |
|---|---|---|---|---|---|---|
| dense | **0.824** | **0.932** | 0.932 | **0.925** | **0.750** | 1078 |
| bm25 | 0.689 | 0.919 | 0.919 | 0.859 | 0.625 | 1014 |
| hybrid | **0.824** | 0.905 | **0.946** | **0.925** | 0.625 | 1048 |

**Hybrid did not beat dense.** It tied at R@1, was *worse* at R@3 (0.905 vs 0.932), and
was worse on the vocabulary-mismatch questions — the very category hybrid search is
supposed to help with — dropping from 0.750 to 0.625.

If this repo were selling hybrid search, that row would not be here. It is here because
a negative result is a finding.

## Why it went wrong

**RRF weights every retriever equally.** That is its great virtue when both retrievers
are roughly as good, and its failure mode when one is systematically bad for a query
type.

On vocabulary-mismatch questions BM25 is not merely unhelpful, it is actively
misleading — it confidently ranks documents that share surface words with the query and
nothing else. Equal-weight fusion lets that bad ranking drag good dense results down. The
`vocab_mismatch R@1 = 0.250` in the BM25 row is the mechanism, visible directly.

Two fixes, and only one of them is honest:

- **Weighted RRF** — give the dense list more votes. Tempting, and it is tuning a
  constant against 37 questions, which is how you overfit a benchmark.
- **Routing** — send identifier-shaped queries to BM25 and prose questions to dense.
  Principled, and it is lesson 09.

## The result that changes the conclusion

BM25 *did* win somewhere: `multi_hop R@3` went 0.929 → 1.000. And the hybrid row's
`R@5 = 0.946` beats dense's 0.932 even though its `R@3` is worse.

That pattern — worse near the top, better deeper — is the clue. Hybrid is not producing a
better *ranking*; it is producing a better *pool*. Lesson 05 tests that directly, and it
is what rescues hybrid search here.

## Implementation notes

**Both indexes are built from one chunk list** (`ragkit/pipeline.py:ingest`). If they
ever disagreed about chunk boundaries, fusion would be merging rankings over two
different corpora, and the symptom would be an unexplained quality regression rather
than an error.

**BM25 indexes `embed_text`, not `text`.** When contextual retrieval (lesson 07) adds a
situating prefix, the lexical index benefits from those keywords too. That is half the
value of the technique and it is easy to leave on the table.

**Stemming is on.** Without it, "chunking" and "chunks" are unrelated terms and the
lexical half quietly stops matching obvious things.

**Per-retriever ranks survive fusion.** Every fused result carries `dense_rank`,
`bm25_rank` and the RRF contributions, because "which retriever put this here" is the
question you ask when debugging and it is unrecoverable once the lists merge.

## Exercise

See [exercise.md](exercise.md).

## Next

[Lesson 05 — Reranking](../05-reranking/), where hybrid search turns out to have been
worth building after all — for a different reason than advertised.
