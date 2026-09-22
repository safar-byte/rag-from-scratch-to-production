"""The vector store interface.

Two implementations sit behind this: Chroma (embedded, the default — there is no Docker
on the development machine) and, later, Qdrant. Keeping the interface this small is
deliberate; a store that exposes its query DSL leaks into every caller and stops being
swappable.
"""

from __future__ import annotations

from typing import Any, Protocol

from ragkit.types import Chunk, Scored


class VectorStore(Protocol):
    @property
    def name(self) -> str: ...

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[Scored]: ...

    def get(self, chunk_ids: list[str]) -> list[Chunk]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...
