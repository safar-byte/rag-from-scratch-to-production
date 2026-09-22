"""Evaluation: the pivot point of this repo.

Everything before lesson 03 is unmeasured. Everything after it has to justify itself
with a row in `benchmarks/results.md`.
"""

from __future__ import annotations

from ragkit.eval.golden import GoldenQuestion, load_golden, validate_golden
from ragkit.eval.judge import DeterministicJudge, Grade, LlmJudge
from ragkit.eval.metrics import (
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    rank_of_first_relevant,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "DeterministicJudge",
    "GoldenQuestion",
    "Grade",
    "LlmJudge",
    "hit_at_k",
    "load_golden",
    "ndcg_at_k",
    "precision_at_k",
    "rank_of_first_relevant",
    "recall_at_k",
    "reciprocal_rank",
    "validate_golden",
]
