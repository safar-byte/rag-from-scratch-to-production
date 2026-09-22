"""The pipeline: the one object that wires retrieval and generation together.

Every lesson from 01 to 12 changes this file or something it calls, and nothing else.
That is the point of the structure — you can read the diff for a lesson and see exactly
what the technique changed.

`RetrievalStrategy` is how the lessons stack up rather than replacing each other:

    dense    lesson 01   embed the query, take the nearest chunks
    bm25     lesson 04   lexical only, for comparison
    hybrid   lesson 04   dense + bm25 fused with RRF
    rerank   lesson 05   hybrid shortlist, reordered by a cross-encoder
    parent   lesson 08   rerank, then expand each hit to its surrounding window
    router   lesson 09   per-query choice between lexical and semantic

`transform` is orthogonal and wraps whichever strategy is selected (lesson 06).

Comparing them is a flag rather than a branch, which is what makes
`benchmarks/results.md` a table of measurements rather than a table of anecdotes.
"""

from __future__ import annotations

import time
from enum import StrEnum
from pathlib import Path
from typing import Any

from ragkit.config import Settings, get_settings
from ragkit.generate import abstain, generate_answer
from ragkit.ingest import RecursiveChunker, chunk_documents, load_documents
from ragkit.ingest.chunker import Chunker
from ragkit.providers import get_embedder, get_generator, get_reranker
from ragkit.query import TRANSFORMERS, TransformingRetriever
from ragkit.retrieve import (
    Bm25Index,
    Bm25Retriever,
    DenseRetriever,
    HeuristicRouter,
    HybridRetriever,
    ParentDocumentRetriever,
)
from ragkit.retrieve.rerank import RerankingRetriever
from ragkit.store import get_store
from ragkit.types import Answer, Scored


class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    RERANK = "rerank"
    PARENT = "parent"
    ROUTER = "router"


class RagPipeline:
    """Ingest once, then query.

    Providers are constructed lazily. A local embedder loads a few hundred MB of model
    weights and a cross-encoder loads more, so building a pipeline to call `count()`
    should not pay for either.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        strategy: RetrievalStrategy | str = RetrievalStrategy.DENSE,
        *,
        score_floor: float | None = None,
        strict_prompt: bool = False,
        transform: str = "identity",
        parent_window: int = 600,
    ) -> None:
        self.settings = settings or get_settings()
        self.strategy = RetrievalStrategy(strategy)
        # Both refusal levers are off by default so lesson 03 measures them rather than
        # the repo assuming they work. `score_floor=None` disables the gate entirely.
        self.score_floor = score_floor
        self.strict_prompt = strict_prompt
        self.transform = transform
        self.parent_window = parent_window
        self._documents: list[Any] | None = None
        self._store = get_store(self.settings)
        self._bm25 = Bm25Index(self.settings.chroma_path.parent / ".bm25")
        self._embedder: Any = None
        self._generator: Any = None
        self._reranker: Any = None
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
    def reranker(self) -> Any:
        if self._reranker is None:
            self._reranker = get_reranker(self.settings)
        return self._reranker

    @property
    def retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = self._build_retriever()
        return self._retriever

    def _build_retriever(self) -> Any:
        base = self._build_base_retriever()
        if self.transform == "identity":
            return base

        transformer_cls = TRANSFORMERS.get(self.transform)
        if transformer_cls is None:
            raise ValueError(
                f"Unknown transform {self.transform!r}. Options: {sorted(TRANSFORMERS)}"
            )
        transformer = (
            transformer_cls()
            if transformer_cls is TRANSFORMERS["identity"]
            else transformer_cls(self.generator)
        )
        return TransformingRetriever(base, transformer, candidates=self.settings.rerank_candidates)

    def _build_base_retriever(self) -> Any:
        dense = DenseRetriever(self._store, self.embedder)
        if self.strategy is RetrievalStrategy.DENSE:
            return dense

        lexical = Bm25Retriever(self._bm25)
        if self.strategy is RetrievalStrategy.BM25:
            return lexical

        if self.strategy is RetrievalStrategy.ROUTER:
            # Lexical queries go to BM25 alone; everything else takes the full
            # reranked path. Motivated directly by lesson 04's measurement, where
            # equal-weight fusion let BM25 damage vocabulary-mismatch questions.
            semantic = RerankingRetriever(
                HybridRetriever([dense, lexical], candidates=self.settings.rerank_candidates),
                self.reranker,
                candidates=self.settings.rerank_candidates,
            )
            return HeuristicRouter(lexical, semantic)

        hybrid = HybridRetriever([dense, lexical], candidates=self.settings.rerank_candidates)
        if self.strategy is RetrievalStrategy.HYBRID:
            return hybrid

        reranked = RerankingRetriever(
            hybrid, self.reranker, candidates=self.settings.rerank_candidates
        )
        if self.strategy is RetrievalStrategy.RERANK:
            return reranked

        return ParentDocumentRetriever(reranked, self.documents, window=self.parent_window)

    @property
    def documents(self) -> list[Any]:
        """Source documents, loaded once. Parent expansion slices them directly."""
        if self._documents is None:
            self._documents = load_documents()
        return self._documents

    def set_strategy(self, strategy: RetrievalStrategy | str, transform: str | None = None) -> None:
        """Switch strategy, reusing the loaded models. Used by the lesson sweeps."""
        self.strategy = RetrievalStrategy(strategy)
        if transform is not None:
            self.transform = transform
        self._retriever = None

    # ---- ingest -------------------------------------------------------------

    def ingest(
        self,
        source: Path | None = None,
        chunker: Chunker | None = None,
        *,
        reset: bool = True,
    ) -> dict[str, Any]:
        """Load, chunk, embed and index the corpus, into both indexes.

        The dense and lexical indexes are written from **one** chunk list rather than
        each chunking independently. If they ever disagreed about chunk boundaries,
        fusion would be combining rankings over two different corpora, and the bug
        would look like a mysterious quality regression rather than an error.

        `reset=True` by default: partially re-indexing a corpus whose chunk boundaries
        moved leaves orphaned chunks behind, and a stale index is far harder to debug
        than a slow rebuild. Incremental ingest arrives in lesson 12 with versioning.
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
        self._bm25.build(chunks)

        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "chunker": chunker.name,
            "embedder": self.embedder.name,
            "dimension": self.embedder.dimension,
            "store": self._store.name,
            "lexical": f"bm25({self._bm25.count()})",
            "seconds": round(time.perf_counter() - started, 2),
        }

    # ---- query --------------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[Scored]:
        return self.retriever.retrieve(question, top_k=top_k or self.settings.top_k)

    def ask(self, question: str, top_k: int | None = None) -> Answer:
        started = time.perf_counter()
        results = self.retrieve(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - started) * 1000

        # The abstention gate runs before generation, not after. Refusing here costs one
        # float comparison; letting the model see context it cannot answer from is how
        # you get "the default value is 1" for a setting that does not exist.
        if self.score_floor is not None:
            decision = abstain.decide(results, floor=self.score_floor)
            if decision.abstain:
                answer = abstain.refusal_answer(decision, results)
                answer.trace["retrieval_ms"] = round(retrieval_ms, 1)
                answer.trace["retriever"] = self.retriever.name
                answer.trace["strategy"] = self.strategy.value
                answer.trace["profile"] = self.settings.profile.value
                answer.trace["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
                return answer

        answer = generate_answer(question, results, self.generator, strict=self.strict_prompt)
        answer.trace["retrieval_ms"] = round(retrieval_ms, 1)
        answer.trace["retriever"] = self.retriever.name
        answer.trace["strategy"] = self.strategy.value
        answer.trace["profile"] = self.settings.profile.value
        answer.trace["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return answer

    def count(self) -> int:
        return self._store.count()
