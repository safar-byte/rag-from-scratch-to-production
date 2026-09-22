"""The abstention gate.

This is the mechanism answering the worst result in the repo — a pipeline that scored
0.000 on correct refusal and invented a default value for a setting that does not exist.
"""

from __future__ import annotations

from ragkit.generate.abstain import (
    DEFAULT_SCORE_FLOOR,
    best_retrieval_score,
    decide,
    refusal_answer,
)
from ragkit.types import Chunk, Scored


def _scored(score: float, dense: float | None = None) -> Scored:
    scores = {"dense": dense} if dense is not None else {}
    return Scored(
        chunk=Chunk(chunk_id="c", doc_id="d.md", text="some text"), score=score, scores=scores
    )


def test_abstains_when_nothing_was_retrieved() -> None:
    decision = decide([])
    assert decision.abstain
    assert "nothing was retrieved" in decision.reason


def test_abstains_below_the_floor() -> None:
    assert decide([_scored(0.4, dense=0.4)], floor=0.60).abstain


def test_answers_above_the_floor() -> None:
    assert not decide([_scored(0.8, dense=0.8)], floor=0.60).abstain


def test_the_floor_boundary_is_inclusive_upward() -> None:
    # Exactly at the floor answers; the threshold is "below the floor refuses".
    assert not decide([_scored(0.60, dense=0.60)], floor=0.60).abstain
    assert decide([_scored(0.599, dense=0.599)], floor=0.60).abstain


def test_uses_the_dense_score_not_the_reranker_score() -> None:
    """Measured: cross-encoder scores separate answerable from unanswerable far worse.

    A reranked result carries a cross-encoder logit as `score` on a different scale
    entirely. Gating on it would reject far more real questions than it caught fakes,
    so the gate reads the dense component that fusion preserved.
    """
    reranked = _scored(-8.5, dense=0.82)  # strong cosine, negative CE logit
    assert best_retrieval_score([reranked]) == 0.82
    assert not decide([reranked], floor=0.60).abstain


def test_falls_back_to_the_headline_score_when_there_is_no_dense_component() -> None:
    assert best_retrieval_score([_scored(0.75)]) == 0.75


def test_takes_the_best_score_across_results() -> None:
    results = [_scored(0.3, dense=0.3), _scored(0.9, dense=0.9)]
    assert best_retrieval_score(results) == 0.9
    assert not decide(results, floor=0.60).abstain


def test_refusal_answer_spends_nothing_and_keeps_the_evidence() -> None:
    results = [_scored(0.4, dense=0.4)]
    answer = refusal_answer(decide(results, floor=0.60), results)

    assert answer.usage.input_tokens == 0
    assert answer.usage.cost_usd == 0.0
    assert answer.citations == []
    assert answer.trace["abstained"] is True
    # Contexts are kept even though nothing was generated: a wrong refusal has to be
    # debuggable, which means seeing what was rejected and how close it came.
    assert answer.contexts == results
    assert answer.trace["best_score"] == 0.4


def test_the_default_floor_is_the_measured_one() -> None:
    # 0.60 was the highest floor that refused no valid question on this corpus.
    assert DEFAULT_SCORE_FLOOR == 0.60
