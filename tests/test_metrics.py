"""Metrics tested against hand-computed values.

A subtly wrong metric does not raise; it produces confident nonsense that every later
decision in the repo is then based on. So these assert exact arithmetic, not properties.
"""

from __future__ import annotations

import math

import pytest

from ragkit.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    rank_of_first_relevant,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = ["a", "b", "c", "d", "e"]


def test_recall_counts_relevant_documents_found() -> None:
    assert recall_at_k(RETRIEVED, ["a", "c"], 5) == 1.0
    assert recall_at_k(RETRIEVED, ["a", "z"], 5) == 0.5
    assert recall_at_k(RETRIEVED, ["z"], 5) == 0.0


def test_recall_respects_the_cutoff() -> None:
    assert recall_at_k(RETRIEVED, ["d"], 3) == 0.0
    assert recall_at_k(RETRIEVED, ["d"], 4) == 1.0


def test_precision_is_the_fraction_of_the_top_k_that_is_relevant() -> None:
    assert precision_at_k(RETRIEVED, ["a", "b"], 2) == 1.0
    assert precision_at_k(RETRIEVED, ["a"], 2) == 0.5
    assert precision_at_k(RETRIEVED, ["a"], 5) == 0.2


def test_reciprocal_rank_uses_the_first_hit_only() -> None:
    assert reciprocal_rank(RETRIEVED, ["a"]) == 1.0
    assert reciprocal_rank(RETRIEVED, ["b"]) == 0.5
    assert reciprocal_rank(RETRIEVED, ["c"]) == pytest.approx(1 / 3)
    # A second relevant document further down changes nothing — that is the point.
    assert reciprocal_rank(RETRIEVED, ["b", "e"]) == 0.5
    assert reciprocal_rank(RETRIEVED, ["z"]) == 0.0


def test_ndcg_matches_hand_computed_values() -> None:
    assert ndcg_at_k(RETRIEVED, ["a"], 5) == 1.0
    # One relevant document at rank 2: DCG = 1/log2(3), ideal = 1/log2(2) = 1.
    assert ndcg_at_k(RETRIEVED, ["b"], 5) == pytest.approx(1 / math.log2(3))
    # Two relevant at ranks 1 and 2 is the ideal ordering for two documents.
    assert ndcg_at_k(RETRIEVED, ["a", "b"], 5) == pytest.approx(1.0)


def test_ndcg_keeps_caring_after_the_first_hit_unlike_mrr() -> None:
    # Both orderings put a relevant document first, so MRR cannot tell them apart.
    assert reciprocal_rank(["a", "b", "z"], ["a", "b"]) == reciprocal_rank(
        ["a", "z", "b"], ["a", "b"]
    )
    # nDCG can: the second ordering buries the other relevant document.
    assert ndcg_at_k(["a", "b", "z"], ["a", "b"], 3) > ndcg_at_k(["a", "z", "b"], ["a", "b"], 3)


def test_unanswerable_questions_score_perfectly_when_nothing_is_relevant() -> None:
    # A question with no relevant document cannot have its retrieval graded; scoring it
    # 0 would drag the mean down for behaviour that is not a retrieval failure. The
    # generation-side judge is what actually grades these.
    assert recall_at_k(RETRIEVED, [], 5) == 1.0
    assert reciprocal_rank(RETRIEVED, []) == 1.0
    assert ndcg_at_k(RETRIEVED, [], 5) == 1.0
    # Precision is the exception: every result is a false positive by definition.
    assert precision_at_k(RETRIEVED, [], 5) == 0.0


def test_hit_is_the_most_forgiving_metric() -> None:
    assert hit_at_k(RETRIEVED, ["e"], 5) == 1.0
    assert hit_at_k(RETRIEVED, ["e"], 4) == 0.0


def test_rank_diagnostic_distinguishes_near_miss_from_total_miss() -> None:
    # recall@3 is 0.0 in both cases, but these need opposite responses: deepen the
    # shortlist versus fix retrieval entirely.
    assert rank_of_first_relevant(RETRIEVED, ["d"]) == 4
    assert rank_of_first_relevant(RETRIEVED, ["z"]) is None


def test_empty_retrieval_scores_zero_rather_than_crashing() -> None:
    assert recall_at_k([], ["a"], 5) == 0.0
    assert precision_at_k([], ["a"], 5) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0
    assert ndcg_at_k([], ["a"], 5) == 0.0
