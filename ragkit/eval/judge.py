"""Grading generated answers.

Two graders, deliberately. The LLM judge is the real one; the deterministic grader is
the control.

Keeping both is not belt-and-braces, it is methodology. An LLM judge is a model with
opinions, and the only way to know whether to trust it is to compare it against
something that cannot have opinions. If the two disagree wildly on a sample you hand
label, the judge is miscalibrated and its scores should not be reported as fact.

The deterministic grader also keeps the harness runnable with no model at all, which is
what makes the repo's offline promise extend to evaluation rather than stopping at
retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ragkit.eval.golden import GoldenQuestion
from ragkit.types import Scored

# Phrasings that mean "I decline to answer from this context".
#
# This list is measurement code, and it was wrong once in a way worth recording. The
# first version had "not provided" but not "does not provide", so answers like
#
#     "The passage does not provide any information about the population of Tokyo."
#
# were scored as confabulations. That is a *correct refusal* counted as a failure, and
# it made the reported refusal rate 0.000 when the true rate was higher. A detector
# gap in the harness is indistinguishable from a model failure in the results table.
#
# Matching is substring-based on the lowercased answer, so keep entries short and
# distinctive; a long phrase fails on trivial wording changes.
REFUSAL_MARKERS = (
    # "does/do not <verb>" forms - the most common shape, and the one that was missed
    "does not provide",
    "do not provide",
    "does not contain",
    "do not contain",
    "does not mention",
    "do not mention",
    "does not specify",
    "do not specify",
    "does not say",
    "does not include",
    "does not appear",
    "doesn't provide",
    "doesn't contain",
    "doesn't mention",
    # absence statements
    "no information",
    "not mentioned",
    "not specified",
    "not provided",
    "not contain enough",
    "not enough information",
    "no relevant",
    "nothing in the",
    "is not present",
    "not found in",
    # inability statements
    "cannot answer",
    "can't answer",
    "cannot determine",
    "unable to answer",
    "insufficient",
)


@dataclass(slots=True)
class Grade:
    groundedness: float
    relevance: float
    correct_refusal: float | None = None
    notes: str = ""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9\-_.]*", text.lower()))


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


class DeterministicJudge:
    """No model. Lexical overlap and keyword checks.

    Crude, and honest about it: it cannot tell a well-grounded paraphrase from a lucky
    word match. Its value is that it is perfectly reproducible and free, so it can run
    on every commit and catch a regression that a sampled LLM judge would miss.
    """

    name = "deterministic"

    def grade(self, question: GoldenQuestion, answer: str, contexts: list[Scored]) -> Grade:
        if question.is_unanswerable:
            refused = looks_like_refusal(answer)
            return Grade(
                groundedness=1.0 if refused else 0.0,
                relevance=1.0 if refused else 0.0,
                correct_refusal=1.0 if refused else 0.0,
                notes="refused" if refused else "answered an unanswerable question",
            )

        # Groundedness proxy: how much of the answer's vocabulary came from the context.
        # Stopwords inflate this, so it is a floor rather than a measurement.
        context_tokens = set()
        for scored in contexts:
            context_tokens |= _tokens(scored.chunk.text)
        answer_tokens = _tokens(answer)
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0

        # Relevance proxy: did the expected strings show up at all.
        expected = question.answer_contains
        hits = sum(1 for needle in expected if needle.lower() in answer.lower())
        relevance = hits / len(expected) if expected else (1.0 if answer.strip() else 0.0)

        return Grade(
            groundedness=round(overlap, 3),
            relevance=round(relevance, 3),
            notes=f"{hits}/{len(expected)} expected strings present" if expected else "",
        )


JUDGE_SYSTEM = """You grade answers produced by a retrieval system. You are strict and
you return only JSON.

Score two things from 0.0 to 1.0:

groundedness: is every claim in the answer supported by the supplied context? An answer
that is correct in the world but not supported by the context scores 0. An answer that
correctly states the context does not cover the question scores 1.

relevance: does the answer address the question that was asked? An answer can be
perfectly grounded and completely unresponsive.

Return exactly: {"groundedness": <float>, "relevance": <float>, "notes": "<short reason>"}
"""


class LlmJudge:
    """Grades with a language model against the rubric above.

    Runs at temperature 0 and on the cheap model: judging is an easier task than
    answering, and a large model here buys little at several times the cost.

    Two limits worth stating plainly, because they are easy to forget once the numbers
    start looking authoritative. Judges are biased — they favour fluent and longer
    answers. And scores from different judges are not comparable, so changing the judge
    model invalidates every historical number in the results table.
    """

    name = "llm"

    def __init__(self, generator: object) -> None:
        self._generator = generator

    def grade(self, question: GoldenQuestion, answer: str, contexts: list[Scored]) -> Grade:
        context_text = (
            "\n\n".join(f"[{i}] {s.chunk.text}" for i, s in enumerate(contexts, start=1))
            or "(no context was retrieved)"
        )

        prompt = (
            f"Question:\n{question.question}\n\n"
            f"Context given to the answering model:\n{context_text}\n\n"
            f"Answer to grade:\n{answer}\n\n"
            "Return only the JSON object."
        )

        raw, _usage = self._generator.generate(system=JUDGE_SYSTEM, prompt=prompt, max_tokens=300)
        parsed = _parse_json(raw)
        if parsed is None:
            # A judge that failed to produce JSON has not graded anything. Recording a
            # 0 would be indistinguishable from a genuinely bad answer, so the run must
            # surface this instead of silently averaging it in.
            return Grade(
                groundedness=float("nan"),
                relevance=float("nan"),
                notes=f"judge returned unparseable output: {raw[:80]!r}",
            )

        grade = Grade(
            groundedness=_clamp(parsed.get("groundedness")),
            relevance=_clamp(parsed.get("relevance")),
            notes=str(parsed.get("notes", ""))[:200],
        )
        if question.is_unanswerable:
            grade.correct_refusal = 1.0 if looks_like_refusal(answer) else 0.0
        return grade


def _clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def _parse_json(raw: str) -> dict | None:
    """Small models wrap JSON in prose and fences often enough to be worth handling."""
    for candidate in (raw, *re.findall(r"\{.*?\}", raw, re.DOTALL)):
        try:
            parsed = json.loads(candidate.strip().strip("`"))
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
