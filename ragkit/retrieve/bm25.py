"""Lexical retrieval with BM25.

BM25 scores a document by how often the query's terms appear in it, discounted by how
common those terms are across the corpus and normalised by document length. It has no
idea what anything means, which is exactly why it is worth having: it succeeds on
identifiers, version numbers, error codes and rare proper nouns — the cases where an
embedding model's whole strength becomes its weakness.

The index lives on disk next to the vector store and is rebuilt by the same ingest, so
the two can never drift apart in content. They can still drift in *chunking*, which is
why both are written from one chunk list rather than each re-chunking independently.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from ragkit.types import Chunk, Scored


class Bm25Index:
    """A persisted BM25 index over the same chunks as the vector store.

    `bm25s` handles tokenising, stemming and stopword removal. Stemming matters more
    than it sounds: without it "chunking" and "chunks" are unrelated terms, and the
    lexical half of a hybrid retriever quietly stops matching the obvious things.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._retriever: Any = None
        self._chunks: list[Chunk] = []
        self._stemmer: Any = None

    @property
    def name(self) -> str:
        return "bm25"

    # ---- building -----------------------------------------------------------

    def build(self, chunks: list[Chunk]) -> None:
        import bm25s
        import Stemmer

        if not chunks:
            raise ValueError("Cannot build a BM25 index over no chunks.")

        self._chunks = chunks
        self._stemmer = Stemmer.Stemmer("english")
        # Index `embed_text`, not `text`: when contextual retrieval (lesson 07) adds a
        # situating prefix, the lexical index should benefit from those keywords too.
        # That is half the value of the technique and it is easy to leave on the table.
        tokens = bm25s.tokenize(
            [c.embed_text for c in chunks],
            stopwords="en",
            stemmer=self._stemmer,
            show_progress=False,
        )
        self._retriever = bm25s.BM25()
        self._retriever.index(tokens, show_progress=False)
        self.save()

    def save(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(self._path), corpus=None)
        with (self._path / "chunks.pkl").open("wb") as handle:
            pickle.dump(self._chunks, handle)
        (self._path / "meta.json").write_text(
            json.dumps({"n_chunks": len(self._chunks)}), encoding="utf-8"
        )

    def load(self) -> bool:
        """Load a previously built index. False if there is nothing to load."""
        import bm25s
        import Stemmer

        chunks_path = self._path / "chunks.pkl"
        if not chunks_path.exists():
            return False
        try:
            self._retriever = bm25s.BM25.load(str(self._path), load_corpus=False)
            with chunks_path.open("rb") as handle:
                self._chunks = pickle.load(handle)  # noqa: S301 - our own artefact
        except Exception:  # noqa: BLE001 - a corrupt index should rebuild, not crash
            return False
        self._stemmer = Stemmer.Stemmer("english")
        return True

    # ---- querying -----------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[Scored]:
        import bm25s

        if self._retriever is None and not self.load():
            raise RuntimeError("BM25 index has not been built. Run ingest first.")
        if not query.strip() or not self._chunks:
            return []

        tokens = bm25s.tokenize(query, stopwords="en", stemmer=self._stemmer, show_progress=False)
        k = min(top_k, len(self._chunks))
        indices, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)

        out: list[Scored] = []
        for rank, (index, score) in enumerate(zip(indices[0], scores[0], strict=True), start=1):
            # BM25 scores are unbounded and corpus-dependent, so they are kept as a
            # diagnostic only. Fusion uses ranks — see rrf.py for why.
            out.append(
                Scored(
                    chunk=self._chunks[int(index)],
                    score=float(score),
                    scores={"bm25": float(score)},
                    rank=rank,
                )
            )
        return out

    def count(self) -> int:
        return len(self._chunks)


class Bm25Retriever:
    """Retriever-shaped wrapper, so it drops into the pipeline like the dense one."""

    def __init__(self, index: Bm25Index) -> None:
        self._index = index

    @property
    def name(self) -> str:
        return "bm25"

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        if where:
            # Honest limitation rather than a silently ignored argument: metadata
            # filtering over the lexical index arrives with lesson 08.
            raise NotImplementedError("BM25 metadata filtering is not implemented yet.")
        return self._index.search(query, top_k=top_k)
