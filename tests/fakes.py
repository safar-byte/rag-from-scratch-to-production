"""Deterministic fake providers.

The whole point of the Protocol-based design is that the test suite never needs a
model, a GPU, or a network. These fakes are boring on purpose: a hash-based embedder
is not semantically meaningful, but it is stable, fast, and sufficient to prove the
plumbing works. Retrieval *quality* is measured by the eval harness (lesson 03), not
by unit tests.
"""

from __future__ import annotations

import hashlib
import math

from ragkit.types import Chunk, Usage


class FakeEmbedder:
    """Hashes text into a fixed-dimension unit vector."""

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return "fake-embedder"

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self._dimension)]
        norm = math.sqrt(sum(v * v for v in raw)) or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeReranker:
    """Scores by word overlap — crude, but monotonic and explainable."""

    @property
    def name(self) -> str:
        return "fake-reranker"

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
        terms = set(query.lower().split())
        scored = [
            (chunk, len(terms & set(chunk.embed_text.lower().split())) / (len(terms) or 1))
            for chunk in chunks
        ]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:top_k]


class FakeGenerator:
    """Echoes back a marker plus the prompt length, so tests can assert on wiring."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "fake-generator"

    def generate(self, *, system: str, prompt: str, max_tokens: int = 1024) -> tuple[str, Usage]:
        self.calls.append((system, prompt))
        return (
            f"[fake answer over {len(prompt)} prompt chars]",
            Usage(input_tokens=len(prompt) // 4, output_tokens=8),
        )
