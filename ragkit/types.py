"""Core data types shared by every stage of the pipeline.

These are deliberately plain: a chunk knows its text, where it came from, and any
scores it has picked up along the way. Keeping provenance on the chunk itself is what
makes the retrieval inspector (lesson 12) and grounded citations possible at all — if
you drop the source on the floor during retrieval, you cannot cite it later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """A source document before chunking."""

    doc_id: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A retrievable unit of text.

    `context` holds the LLM-generated situating prefix added by contextual retrieval
    (lesson 07). It is stored separately from `text` so you can embed `context + text`
    while still showing and citing the original span.
    """

    chunk_id: str
    doc_id: str
    text: str
    title: str = ""
    context: str = ""
    start_char: int = 0
    end_char: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """What actually goes to the embedding model."""
        return f"{self.context}\n\n{self.text}".strip() if self.context else self.text


@dataclass(slots=True)
class Scored:
    """A chunk with the scores it accumulated through the pipeline.

    `scores` is a per-stage record (``{"dense": 0.81, "bm25": 4.2, "rrf": 0.031,
    "rerank": 0.97}``) rather than a single float, because the interesting question in
    a hybrid pipeline is never "what was the score" but "which stage put this here".
    """

    chunk: Chunk
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    rank: int = 0


@dataclass(slots=True)
class Citation:
    """A span of a source document that a claim was grounded in."""

    doc_id: str
    title: str
    cited_text: str
    start_char: int | None = None
    end_char: int | None = None


@dataclass(slots=True)
class Usage:
    """Token and cost accounting for one generation call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class Answer:
    """The end of the pipeline: text, what it was grounded in, and what it cost."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[Scored] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    trace: dict[str, Any] = field(default_factory=dict)
