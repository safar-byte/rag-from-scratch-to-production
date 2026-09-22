"""Vector storage."""

from __future__ import annotations

from ragkit.config import Settings, get_settings
from ragkit.store.base import VectorStore
from ragkit.store.chroma import ChromaStore


def get_store(settings: Settings | None = None, dimension: int | None = None) -> VectorStore:
    """Build the configured store.

    `dimension` is only needed by Qdrant, which fixes the vector size at collection
    creation. Chroma infers it from the first insert, which is why the rest of the
    codebase never had to pass it.
    """
    settings = settings or get_settings()

    if settings.vector_store == "chroma":
        return ChromaStore(settings.chroma_path)

    if settings.vector_store == "qdrant":
        from ragkit.store.qdrant import QdrantStore

        if dimension is None:
            raise ValueError(
                "Qdrant needs the embedding dimension up front - it is fixed when the "
                "collection is created. Pass dimension=embedder.dimension."
            )
        return QdrantStore(
            dimension,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            path=settings.chroma_path.parent / ".qdrant" if not settings.qdrant_url else None,
        )

    raise ValueError(
        f"Unknown VECTOR_STORE={settings.vector_store!r}. Options: 'chroma', 'qdrant'."
    )


__all__ = ["ChromaStore", "VectorStore", "get_store"]
