"""Vector storage."""

from __future__ import annotations

from ragkit.config import Settings, get_settings
from ragkit.store.base import VectorStore
from ragkit.store.chroma import ChromaStore


def get_store(settings: Settings | None = None) -> VectorStore:
    settings = settings or get_settings()
    if settings.vector_store != "chroma":
        raise ValueError(
            f"Unknown VECTOR_STORE={settings.vector_store!r}. Only 'chroma' is "
            "implemented; the Qdrant adapter lands with the production lessons."
        )
    return ChromaStore(settings.chroma_path)


__all__ = ["ChromaStore", "VectorStore", "get_store"]
