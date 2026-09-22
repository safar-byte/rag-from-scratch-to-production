"""Provider backends.

`get_embedder()`, `get_reranker()` and `get_generator()` are the only functions the
rest of the codebase calls. Everything downstream is written against the protocols in
`base.py`, so switching `RAG_PROFILE` between `local` and `cloud` changes nothing else.
"""

from __future__ import annotations

from ragkit.config import Profile, Settings, get_settings
from ragkit.providers.base import Embedder, Generator, Reranker


def get_embedder(settings: Settings | None = None) -> Embedder:
    settings = settings or get_settings()
    if settings.profile is Profile.CLOUD:
        from ragkit.providers.cloud import VoyageEmbedder

        return VoyageEmbedder(settings)
    from ragkit.providers.local import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(settings)


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()
    if settings.profile is Profile.CLOUD:
        from ragkit.providers.cloud import VoyageReranker

        return VoyageReranker(settings)
    from ragkit.providers.local import CrossEncoderReranker

    return CrossEncoderReranker(settings)


def get_generator(settings: Settings | None = None, *, cheap: bool = False) -> Generator:
    """Return a generator.

    `cheap=True` asks for the small/fast model. Bulk passes — contextual-chunk
    generation over a whole corpus, LLM-as-judge over an eval set — run hundreds of
    calls where the large model buys nothing, so they should always ask for it.
    """
    settings = settings or get_settings()
    if settings.profile is Profile.CLOUD:
        from ragkit.providers.cloud import ClaudeGenerator

        return ClaudeGenerator(settings, cheap=cheap)
    from ragkit.providers.local import OllamaGenerator

    return OllamaGenerator(settings)


__all__ = [
    "Embedder",
    "Generator",
    "Reranker",
    "get_embedder",
    "get_generator",
    "get_reranker",
]
