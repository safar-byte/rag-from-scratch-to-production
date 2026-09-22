"""Loading and validating the golden set.

The validation matters more than it looks. A golden set that references a document which
no longer exists produces a question that can never be answered correctly, and it shows
up as a permanent, unexplained few points off recall. Nothing errors. You just have a
system that looks slightly worse than it is, forever, and you tune against the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ragkit.config import DATA_DIR

GOLDEN_PATH = Path(__file__).parent / "golden.yaml"


@dataclass(slots=True)
class GoldenQuestion:
    id: str
    question: str
    relevant: list[str] = field(default_factory=list)
    answer_contains: list[str] = field(default_factory=list)
    kind: str = "conceptual"

    @property
    def is_unanswerable(self) -> bool:
        return not self.relevant


def load_golden(path: Path | None = None) -> list[GoldenQuestion]:
    raw = yaml.safe_load((path or GOLDEN_PATH).read_text(encoding="utf-8"))
    questions = [
        GoldenQuestion(
            id=item["id"],
            question=" ".join(item["question"].split()),
            relevant=list(item.get("relevant") or []),
            answer_contains=list(item.get("answer_contains") or []),
            kind=item.get("kind", "conceptual"),
        )
        for item in raw["questions"]
    ]
    if not questions:
        raise ValueError("The golden set is empty.")
    return questions


def validate_golden(questions: list[GoldenQuestion], corpus_dir: Path | None = None) -> list[str]:
    """Return a list of problems. Empty means the golden set is coherent."""
    corpus = corpus_dir or DATA_DIR
    on_disk = {
        p.relative_to(corpus).as_posix()
        for p in corpus.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".rst"}
    }

    problems: list[str] = []
    seen: set[str] = set()
    for question in questions:
        if question.id in seen:
            problems.append(f"{question.id}: duplicate question id")
        seen.add(question.id)

        for doc_id in question.relevant:
            if doc_id not in on_disk:
                problems.append(f"{question.id}: relevant document {doc_id!r} is not in the corpus")

        if question.kind == "unanswerable" and question.relevant:
            problems.append(f"{question.id}: marked unanswerable but names relevant documents")
        if question.kind != "unanswerable" and not question.relevant:
            problems.append(
                f"{question.id}: has no relevant documents but is not marked unanswerable"
            )

    return problems
