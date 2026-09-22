"""Corrective RAG: grade the context, then decide what to do about it.

The pipeline stops being a straight line here and starts making decisions. That is the
point of an agentic pipeline and also its main hazard: a system that can decide to retry
can decide to retry forever. Every loop in this module has an explicit bound, and the
bound is a constructor argument rather than a magic number buried in a branch.

The flow:

    retrieve -> grade
        correct    -> answer
        ambiguous  -> widen the search once, then answer with what that returns
        incorrect  -> refuse

The `ambiguous` branch is the only one that costs an extra retrieval, which keeps the
common cases at one grading call. Refusing on `incorrect` is the behaviour the whole
refusal thread has been building toward: it catches the confabulations a score threshold
structurally cannot, because it is the only step that actually reads the passages.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ragkit.agent.grade import ContextGrader, Grading, Verdict
from ragkit.generate import generate_answer
from ragkit.providers.base import Generator
from ragkit.types import Answer, Scored, Usage

REFUSAL_TEXT = (
    "The retrieved passages do not contain the information needed to answer this "
    "question. The corpus does not appear to cover it."
)


@dataclass(slots=True)
class CragTrace:
    grading: Grading | None = None
    widened: bool = False
    steps: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "crag_verdict": self.grading.verdict.value if self.grading else None,
            "crag_confidence": round(self.grading.confidence, 3) if self.grading else None,
            "crag_reason": self.grading.reason if self.grading else "",
            "crag_widened": self.widened,
            "crag_steps": self.steps,
        }


class CorrectiveRag:
    """Retrieve, grade, correct, answer.

    `widen_factor` controls how much deeper the retry goes on an ambiguous verdict.
    Widening is the cheap correction available offline; a production system would also
    reach for query rewriting (lesson 06) or an external source here.
    """

    def __init__(
        self,
        retriever: object,
        generator: Generator,
        grader: ContextGrader,
        *,
        widen_factor: int = 3,
        strict_prompt: bool = True,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._grader = grader
        self._widen_factor = widen_factor
        self._strict_prompt = strict_prompt

    @property
    def name(self) -> str:
        return f"crag({getattr(self._retriever, 'name', '?')})"

    def ask(self, question: str, top_k: int = 5) -> Answer:
        started = time.perf_counter()
        trace = CragTrace()

        results: list[Scored] = self._retriever.retrieve(question, top_k=top_k)  # type: ignore[attr-defined]
        trace.steps.append(f"retrieved {len(results)}")

        grading = self._grader.grade(question, results)
        trace.grading = grading
        trace.steps.append(f"graded {grading.verdict.value}")

        if grading.verdict is Verdict.INCORRECT:
            # The branch this whole module exists for. No generation call is made, so
            # the model is never given the chance to invent a plausible answer from
            # passages that do not contain one.
            trace.steps.append("refused")
            return Answer(
                text=REFUSAL_TEXT,
                contexts=results,
                usage=Usage(),
                trace={
                    "strategy": "crag",
                    "abstained": True,
                    "total_ms": round((time.perf_counter() - started) * 1000, 1),
                    **trace.as_dict(),
                },
            )

        if grading.verdict is Verdict.AMBIGUOUS:
            # Widen once, not until satisfied. An unbounded correction loop is how an
            # agentic pipeline turns one slow query into an outage.
            widened = self._retriever.retrieve(question, top_k=top_k * self._widen_factor)  # type: ignore[attr-defined]
            if len(widened) > len(results):
                results = widened
                trace.widened = True
                trace.steps.append(f"widened to {len(results)}")

        answer = generate_answer(question, results, self._generator, strict=self._strict_prompt)
        answer.trace.update(
            {
                "strategy": "crag",
                "abstained": False,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
                **trace.as_dict(),
            }
        )
        return answer
