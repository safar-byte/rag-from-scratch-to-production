"""Dense retrieval: embed the query, find the nearest chunks.

This is the whole of "vector search", and on its own it is also the weakest retriever
in this repo. It matches meaning, which means it reliably fails on the things that
carry no meaning to an embedding model: identifiers, error codes, version numbers,
proper nouns. Lesson 04 adds BM25 for exactly that reason.
"""

from __future__ import annotations

from typing import Any

from ragkit.providers.base import Embedder
from ragkit.store.base import VectorStore
from ragkit.types import Scored


class DenseRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(
        self, query: str, top_k: int = 5, where: dict[str, Any] | None = None
    ) -> list[Scored]:
        if not query.strip():
            return []
        # embed_query, not embed_documents: asymmetric models want the instruction
        # prefix on this side only, and using the wrong one costs recall silently.
        vector = self._embedder.embed_query(query)
        return self._store.search(vector, top_k=top_k, where=where)
