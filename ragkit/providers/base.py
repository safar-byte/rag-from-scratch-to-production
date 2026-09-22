"""The three protocols every backend implements.

Structural typing (`Protocol`) rather than inheritance: a backend just has to have the
right methods. That keeps the vendor SDKs out of the core and makes the whole library
trivially testable with a fake — see `tests/fakes.py`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragkit.types import Chunk, Usage


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    Queries and documents are embedded through *separate* methods on purpose. Several
    strong embedding models are asymmetric and expect an instruction prefix on the
    query side only; collapsing both into one call silently costs you recall.
    """

    @property
    def dimension(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidates against the query.

    A cross-encoder reads the query and the chunk *together*, which is why it beats
    embedding similarity — and also why it is far too slow to run over the whole
    corpus. It reorders a shortlist; it cannot rescue a chunk retrieval never found.
    """

    @property
    def name(self) -> str: ...

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]: ...


@runtime_checkable
class Generator(Protocol):
    """Produces the answer."""

    @property
    def name(self) -> str: ...

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        max_tokens: int = 1024,
    ) -> tuple[str, Usage]: ...
