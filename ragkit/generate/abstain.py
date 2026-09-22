"""Refusing to answer when retrieval did not find anything good enough.

The measured problem this exists for: the pipeline scored **0.000** on correct refusal.
Asked for the default value of a configuration setting that does not exist, it replied
"the default value is 1". The system prompt already instructed it to decline. Asking
nicely is not a control.

A retrieval-score floor is the cheap half of the fix. If the best chunk is not very
similar to the question, there is probably nothing in the corpus that answers it, and
refusing costs one comparison instead of a model call.

## What the floor can and cannot do

Measured on this corpus, 44 golden questions, BGE-small cosine:

    threshold | unanswerable caught | answerable wrongly refused
      0.55    |   2/10  (20%)       |   0/34  (0%)
      0.60    |   5/10  (50%)       |   0/34  (0%)      <- the default
      0.65    |   5/10  (50%)       |   4/34  (12%)
      0.70    |   7/10  (70%)       |   8/34  (24%)

0.60 is free: half the confabulations stopped, no valid question refused. Past 0.63 the
curve turns and it starts rejecting real questions faster than it catches fakes.

**The five it cannot catch are the dangerous ones.** They score *higher* than many
genuine questions:

    "default value of MAX_CHUNK_BYTES?"              0.689
    "default value of SHARD_REPLICATION_FACTOR?"     0.674
    "which embedding model in production?"           0.754
    "p99 latency target in the SLO?"                 0.736

against a genuine question scoring 0.632. That is not a tuning failure, it is the
method's ceiling: **a similarity score measures topical closeness, not answerability.**
A question about a configuration setting that does not exist looks exactly like a
question about one that does, because the corpus is full of configuration settings.

Closing the rest requires actually reading the retrieved context and judging whether it
addresses the question — which is a model call, and which is CRAG in lesson 10.

## Recalibrate this

The threshold is a property of the embedding model *and* the corpus, not a constant.
Cosine scores from a different model live on a different scale entirely. Re-run
`lessons/03-evaluation-harness/` style separation analysis before trusting the default
anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragkit.types import Answer, Scored, Usage

# Calibrated on BGE-small over `data/`: the highest floor that refuses nothing valid.
DEFAULT_SCORE_FLOOR = 0.60

REFUSAL_TEXT = (
    "The retrieved passages are not similar enough to this question for me to answer "
    "from them. The corpus does not appear to cover it."
)


@dataclass(slots=True)
class AbstentionDecision:
    abstain: bool
    best_score: float
    floor: float
    reason: str = ""


def best_retrieval_score(results: list[Scored]) -> float:
    """The best *dense* similarity among the results.

    Deliberately the dense score rather than `Scored.score`, which after reranking is a
    cross-encoder logit. Measured on this corpus, cross-encoder scores separate
    answerable from unanswerable questions considerably *worse* than plain cosine does:
    the lowest answerable scored -9.80 against a highest unanswerable of -4.16, so any
    threshold would reject far more real questions than it caught fakes.

    That is worth knowing on its own — the stronger model at ranking is the weaker one
    at signalling "there is nothing here". Ranking quality and calibration are different
    properties, and a reranker is trained only for the first.
    """
    scores = [s.scores.get("dense") for s in results if s.scores.get("dense") is not None]
    if scores:
        return max(scores)
    # No dense component (pure BM25, or a fake in tests): fall back to the headline
    # score. Unbounded scales make the floor meaningless, so callers on BM25-only
    # pipelines should keep the gate off rather than trust this.
    return max((s.score for s in results), default=0.0)


def decide(results: list[Scored], floor: float = DEFAULT_SCORE_FLOOR) -> AbstentionDecision:
    if not results:
        return AbstentionDecision(True, 0.0, floor, "nothing was retrieved")

    best = best_retrieval_score(results)
    if best < floor:
        return AbstentionDecision(
            True, best, floor, f"best similarity {best:.3f} is below the floor {floor:.2f}"
        )
    return AbstentionDecision(False, best, floor)


def refusal_answer(decision: AbstentionDecision, results: list[Scored]) -> Answer:
    """A refusal, shaped like any other answer.

    Contexts are still attached even though nothing was generated from them: the
    retrieval inspector should be able to show *what* was rejected and how close it
    came, otherwise a wrong refusal is undebuggable.
    """
    return Answer(
        text=REFUSAL_TEXT,
        citations=[],
        contexts=results,
        usage=Usage(),  # no model call was made, so nothing was spent
        trace={
            "abstained": True,
            "abstain_reason": decision.reason,
            "best_score": round(decision.best_score, 4),
            "score_floor": decision.floor,
        },
    )
