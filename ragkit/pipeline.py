"""The pipeline: the one object that wires retrieval and generation together.

Every lesson from 01 to 12 changes this file or something it calls, and nothing else.
That is the point of the structure — you can read the diff for a lesson and see exactly
what the technique changed.

Right now it is the naive pipeline: embed the query, take the top k by cosine
similarity, stuff them into a prompt. Lessons 04 onward replace the retriever; the rest
of the shape stays.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ragkit.config import Settings, get_settings
from ragkit.generate import generate_answer
from ragkit.ingest import RecursiveChunker, chunk_documents, load_documents
from ragkit.ingest.chunker import Chunker
from ragkit.providers import get_embedder, get_generator
from ragkit.retrieve import DenseRetriever
from ragkit.store import get_store
from ragkit.types import Answer, Scored


class RagPipeline:
    """Ingest once, then query.

    Providers are constructed lazily. A local embedder loads a few hundred MB of model
    weights, so building a pipeline just to call `count()` should not pay that cost.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._store = get_store(self.settings)
        self._embedder: Any = None
        self._generator: Any = None
        self._retriever: Any = None

    # ---- lazy providers -----------------------------------------------------

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = get_embedder(self.settings)
        return self._embedder

    @property
    def generator(self) -> Any:
        if self._generator is None:
            self._generator = get_generator(self.settings)
        return self._generator

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = DenseRetriever(self._store, self.embedder)
        return self._retriever

    # ---- ingest -------------------------------------------------------------

    def ingest(
        self,
        source: Path | None = None,
        chunker: Chunker | None = None,
        *,
        reset: bool = True,
    ) -> dict[str, Any]:
        """Load, chunk, embed and index the corpus.

        `reset=True` by default: partially re-indexing a corpus whose chunk boundaries
        have moved leaves orphaned chunks behind, and a stale index is much harder to
        debug than a slow rebuild. Incremental ingest arrives in lesson 12 along with
        index versioning, where it can be done correctly.
        """
        chunker = chunker or RecursiveChunker(
            chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap
        )

        started = time.perf_counter()
        documents = load_documents(source)
        chunks = chunk_documents(documents, chunker)
        if not chunks:
            raise RuntimeError("Chunking produced nothing — check the corpus.")

        if reset:
            self._store.reset()

        vectors = self.embedder.embed_documents([c.embed_text for c in chunks])
        self._store.add(chunks, vectors)

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "chunker": chunker.name,
            "embedder": self.embedder.name,
            "dimension": self.embedder.dimension,
            "store": self._store.name,
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- query --------------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[Scored]:
        return self.retriever.retrieve(question, top_k=top_k or self.settings.top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        started = time.perf_counter()
        results = self.retrieve(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - started) * 1000

        answer = generate_answer(question, results, self.generator)
        answer.trace["retrieval_ms"] = round(retrieval_ms, 1)
        answer.trace["retriever"] = self.retriever.name
        answer.trace["profile"] = self.settings.profile.value
        answer.trace["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return answer

    def count(self) -> int:
        return self._store.count()
