"""Retrieval metrics.

Pure functions over ranked lists of document ids. No models, no I/O — which means they
can be tested against hand-computed values, and they are, in `tests/test_metrics.py`.
Getting a metric subtly wrong is a nasty failure: every later decision in the repo is
made by comparing these numbers, so a bug here does not produce an error, it produces
confident nonsense for weeks.

Relevance is binary and judged at document level. See the note at the top of
`golden.yaml` for why document level rather than chunk level.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k.

    The most important single retrieval number: a document that was never retrieved
    cannot be reranked, cited, or reasoned over, so recall is the ceiling on everything
    downstream.
    """
    if not relevant:
        return 1.0  # nothing to find; see `unanswerable` questions in the golden set
    found = set(retrieved[:k]) & set(relevant)
    return len(found) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    if not relevant:
        # No relevant documents exist, so any result is a false positive.
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / rank of the first relevant document, or 0 if none is retrieved.

    Rewards putting a correct answer at the very top and ignores everything after it.
    Averaged over a question set this is MRR.
    """
    if not relevant:
        return 1.0
    relevant_set = set(relevant)
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised discounted cumulative gain at k, with binary relevance.

    Credits a relevant document by 1/log2(rank + 1), so rank 1 is worth 1.0, rank 2
    about 0.63, rank 3 about 0.5. Normalised against the best achievable ordering, so
    a question with more relevant documents than k can still score 1.0.

    Unlike MRR this keeps caring after the first hit, which matters for the multi-hop
    questions where two documents must both be retrieved.
    """
    if not relevant:
        return 1.0
    relevant_set = set(relevant)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant_set
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant_set), k) + 1))
    return dcg / ideal if ideal else 0.0


def hit_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1.0 if any relevant document is in the top k. The most forgiving metric."""
    if not relevant:
        return 1.0
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def rank_of_first_relevant(retrieved: Sequence[str], relevant: Sequence[str]) -> int | None:
    """Where the first correct document actually landed.

    Not a metric — a diagnostic. When recall@5 is 0, the useful question is whether the
    answer was at rank 6 (deepen the shortlist) or absent entirely (fix retrieval). An
    aggregate score cannot distinguish those, and they need opposite responses.
    """
    relevant_set = set(relevant)
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return index
    return None
