"""Chroma-backed vector store.

Chroma is the default because it runs embedded — no Docker, no server, no ops. It is
perfectly adequate up to the low millions of chunks, which is well past what this
corpus needs. The Qdrant adapter behind the same interface is the production path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragkit.types import Chunk, Scored

COLLECTION = "ragkit"


class ChromaStore:
    def __init__(self, path: Path, collection: str = COLLECTION) -> None:
        import chromadb

        path.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._collection_name = collection
        self._client = chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection,
            # Cosine, not the L2 default: document length should not affect relevance.
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def name(self) -> str:
        return f"chroma({self._collection_name})"

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            # Chroma metadata values must be scalars, so everything is flattened here
            # and rebuilt in `_to_chunk`.
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "context": c.context,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    **{
                        k: v
                        for k, v in c.metadata.items()
                        if isinstance(v, str | int | float | bool)
                    },
                }
                for c in chunks
            ],
        )

    @staticmethod
    def _to_chunk(chunk_id: str, text: str, meta: dict[str, Any]) -> Chunk:
        meta = dict(meta or {})
        return Chunk(
            chunk_id=chunk_id,
            doc_id=str(meta.pop("doc_id", "")),
            text=text,
            title=str(meta.pop("title", "")),
            context=str(meta.pop("context", "")),
            start_char=int(meta.pop("start_char", 0)),
            end_char=int(meta.pop("end_char", 0)),
            metadata=meta,
        )

    def search(
        self,
        vector: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[Scored]:
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, max(self.count(), 1)),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        if not ids:
            return []

        out: list[Scored] = []
        for rank, (chunk_id, text, meta, distance) in enumerate(
            zip(
                ids,
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            ),
            start=1,
        ):
            # Chroma returns cosine *distance*; flip it so higher is better everywhere
            # else in the codebase. Mixing the two conventions is a classic silent bug.
            similarity = 1.0 - float(distance)
            out.append(
                Scored(
                    chunk=self._to_chunk(chunk_id, text, meta),
                    score=similarity,
                    scores={"dense": similarity},
                    rank=rank,
                )
            )
        return out

    def get(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        return [
            self._to_chunk(cid, text, meta)
            for cid, text, meta in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
