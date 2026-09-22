"""Grading retrieved context before answering from it.

This is the other half of the refusal fix. The score floor in
`ragkit/generate/abstain.py` catches the unanswerable questions whose retrieval scores
are low — half of them, for free. It cannot catch the rest, and the reason is structural:

    "What is the default value of MAX_CHUNK_BYTES?"     cosine 0.689
    "What is the default value of RETRIEVAL_FANOUT?"    cosine 0.685  (real)

A similarity score measures topical closeness. Both questions are about configuration
defaults, and the corpus is full of configuration defaults, so both retrieve the same
passages with the same confidence. Only one of those settings exists.

Nothing about the *score* can tell them apart. Distinguishing them requires reading the
retrieved passages and asking whether they actually contain the answer — which is a
model call, and which is what CRAG (Corrective RAG) does.

## The trade

A grading call per query, before generation. On the cloud path that is a cheap model and
a short prompt; locally it is another ~25s. In exchange you catch the confabulations the
threshold structurally cannot, and you get a principled signal for when to fall back
(ask the user, widen the search, say you do not know) rather than answering anyway.

Whether that is worth it depends entirely on the cost of a confident wrong answer in
your application. For a support tool inventing configuration defaults, it plainly is.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from ragkit.providers.base import Generator
from ragkit.types import Scored


class Verdict(StrEnum):
    """CRAG's three outcomes."""

    CORRECT = "correct"  # the context answers the question -> generate
    AMBIGUOUS = "ambiguous"  # partially relevant -> generate, but hedge or widen
    INCORRECT = "incorrect"  # nothing here answers it -> refuse or fall back


@dataclass(slots=True)
class Grading:
    verdict: Verdict
    confidence: float
    reason: str = ""

    @property
    def should_answer(self) -> bool:
        return self.verdict is not Verdict.INCORRECT


GRADER_SYSTEM = """You decide whether retrieved passages actually answer a question.

Being on the same topic is NOT the same as containing the answer. Passages about
configuration settings do not answer a question about a setting they never mention.
This distinction is the whole task — most mistakes here are made by grading topical
similarity instead of answerability.

Reply with JSON only:
{"verdict": "correct" | "ambiguous" | "incorrect", "confidence": 0.0-1.0, "reason": "<8 words>"}

  correct    - the passages contain the information needed to answer
  ambiguous  - partially relevant; some of the answer is here, some is missing
  incorrect  - the passages do not answer the question, however related they look"""


class ContextGrader:
    """CRAG-style relevance grading of retrieved context.

    Fails open. A grader that errors or returns unparseable output must not block a
    query that retrieval handled fine — an unavailable grader should degrade the system
    to its ungraded behaviour, not take it down. That does mean a silently broken
    grader looks like a working one, so `Grading.reason` records what happened and the
    eval reports the verdict distribution.
    """

    def __init__(self, generator: Generator, max_context_chars: int = 3000) -> None:
        self._generator = generator
        self._max_context_chars = max_context_chars

    @property
    def name(self) -> str:
        return "crag_grader"

    def grade(self, question: str, results: list[Scored]) -> Grading:
        if not results:
            return Grading(Verdict.INCORRECT, 1.0, "nothing retrieved")

        blocks, used = [], 0
        for index, scored in enumerate(results, start=1):
            block = f"[{index}] {scored.chunk.text}"
            if used + len(block) > self._max_context_chars:
                break
            blocks.append(block)
            used += len(block)

        prompt = (
            f"Question: {question}\n\nRetrieved passages:\n{chr(10).join(blocks)}\n\nJSON verdict:"
        )
        try:
            raw, _usage = self._generator.generate(
                system=GRADER_SYSTEM, prompt=prompt, max_tokens=150
            )
        except Exception as exc:  # noqa: BLE001 - fail open, see the class docstring
            return Grading(Verdict.CORRECT, 0.0, f"grader unavailable: {type(exc).__name__}")

        parsed = _parse_json(raw)
        if parsed is None:
            return Grading(Verdict.CORRECT, 0.0, "grader returned unparseable output")

        try:
            verdict = Verdict(str(parsed.get("verdict", "correct")).strip().lower())
        except ValueError:
            verdict = Verdict.CORRECT
        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        return Grading(verdict, confidence, str(parsed.get("reason", ""))[:120])


def _parse_json(raw: str) -> dict | None:
    for candidate in (raw, *re.findall(r"\{.*?\}", raw, re.DOTALL)):
        try:
            parsed = json.loads(candidate.strip().strip("`"))
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
