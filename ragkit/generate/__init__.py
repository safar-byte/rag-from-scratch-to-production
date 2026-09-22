"""Prompt assembly and answer generation."""

from __future__ import annotations

from ragkit.generate.abstain import (
    DEFAULT_SCORE_FLOOR,
    AbstentionDecision,
    best_retrieval_score,
    decide,
    refusal_answer,
)
from ragkit.generate.answer import extract_citations, generate_answer
from ragkit.generate.prompt import (
    STRICT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_prompt,
    format_context,
)

__all__ = [
    "DEFAULT_SCORE_FLOOR",
    "STRICT_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "AbstentionDecision",
    "best_retrieval_score",
    "build_prompt",
    "decide",
    "extract_citations",
    "format_context",
    "generate_answer",
    "refusal_answer",
]
