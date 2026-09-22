"""Qdrant-backed vector store.

Chroma is the default here because the development machine has no Docker and an
embedded store keeps the repo runnable by anyone. Qdrant is the production path, and
this adapter exists to prove the `VectorStore` protocol is actually an abstraction
rather than a Chroma-shaped hole.

Two ways to reach it without Docker: Qdrant Cloud's free tier (set `QDRANT_URL` and
`QDRANT_API_KEY`), or the in-process local mode (`:memory:` or a path), which needs no
server at all.

## Why you would move off Chroma

Chroma is fine into the low millions of chunks. Past that, or when you need any of the
following, Qdrant is the better tool:

* **Real pre-filtering.** Qdrant applies payload filters *during* the HNSW walk rather
  than before or after it, which is what makes selective filters (lesson 08) work
  without either degrading recall or scanning everything.
* **Named vectors** — several embeddings per point, so you can run two models over one
  corpus and compare without a second index.
* **Quantisation** — scalar or product, with rescoring against the full vectors.
* **Horizontal scaling**, snapshots, and replica sets.

The interface below is identical to `ChromaStore`, which is the point: nothing outside
this file changes when you switch.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ragkit.types import Chunk, Scored

COLLECTION = "ragkit"


def _point_id(chunk_id: str) -> str:
    """Qdrant wants a UUID or an unsigned int, not an arbitrary string.

    A deterministic UUID5 keeps ingest idempotent — re-indexing the same chunk
    overwrites its point instead of adding a duplicate — while `chunk_id` is preserved
    in the payload so results still carry a readable identifier.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ragkit/{chunk_id}"))


class QdrantStore:
    def __init__(
        self,
        dimension: int,
        *,
        url: str | None = None,
        api_key: str | None = None,
        path: Path | None = None,
        collection: str = COLLECTION,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._collection = collection
        self._dimension = dimension
        if url:
            self._client = QdrantClient(url=url, api_key=api_key)
        else:
            # Local mode: in-process, no server, no Docker.
            self._client = QdrantClient(path=str(path) if path else ":memory:")

        if not self._client.collection_exists(collection):
            self._client.create_collection(
                collection_name=collection,
                # Cosine, matching ChromaStore. Document length should not affect
                # relevance, and switching this silently changes every score.
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    @property
    def name(self) -> str:
        return f"qdrant({self._collection})"

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from qdrant_client.models import PointStruct

        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "context": chunk.context,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    **{
                        k: v
                        for k, v in chunk.metadata.items()
                        if isinstance(v, str | int | float | bool)
                    },
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        # Batched: a single upsert of a large corpus can exceed the request size limit.
        for start in range(0, len(points), 256):
            self._client.upsert(self._collection, points=points[start : start + 256])

    @staticmethod
    def _to_chunk(payload: dict[str, Any]) -> Chunk:
        payload = dict(payload or {})
        return Chunk(
            chunk_id=str(payload.pop("chunk_id", "")),
            doc_id=str(payload.pop("doc_id", "")),
            text=str(payload.pop("text", "")),
            title=str(payload.pop("title", "")),
            context=str(payload.pop("context", "")),
            start_char=int(payload.pop("start_char", 0)),
            end_char=int(payload.pop("end_char", 0)),
            metadata=payload,
        )

    def _filter(self, where: dict[str, Any] | None) -> Any:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not where:
            return None
        return Filter(
            must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
        )

    def search(
        self, vector: list[float], top_k: int, where: dict[str, Any] | None = None
    ) -> list[Scored]:
        hits = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            # A genuine pre-filter: applied during the HNSW walk, not after it. This is
            # the main reason to prefer Qdrant once filters get selective.
            query_filter=self._filter(where),
            with_payload=True,
        ).points

        out: list[Scored] = []
        for rank, hit in enumerate(hits, start=1):
            # Qdrant returns cosine *similarity* directly, unlike Chroma's distance —
            # no flip needed here, and getting that backwards would silently invert
            # the ranking.
            score = float(hit.score)
            out.append(
                Scored(
                    chunk=self._to_chunk(hit.payload or {}),
                    score=score,
                    scores={"dense": score},
                    rank=rank,
                )
            )
        return out

    def get(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        records = self._client.retrieve(
            self._collection, ids=[_point_id(c) for c in chunk_ids], with_payload=True
        )
        return [self._to_chunk(r.payload or {}) for r in records]

    def count(self) -> int:
        return int(self._client.count(self._collection, exact=True).count)

    def reset(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        self._client.delete_collection(self._collection)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
        )
