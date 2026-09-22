"""Prompt assembly and answer generation."""

from __future__ import annotations

from ragkit.generate.answer import extract_citations, generate_answer
from ragkit.generate.prompt import SYSTEM_PROMPT, build_prompt, format_context

__all__ = [
    "SYSTEM_PROMPT",
    "build_prompt",
    "extract_citations",
    "format_context",
    "generate_answer",
]
