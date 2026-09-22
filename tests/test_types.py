from __future__ import annotations

from ragkit.types import Chunk


def test_embed_text_is_plain_text_without_context() -> None:
    chunk = Chunk(chunk_id="c1", doc_id="d1", text="The capital of France is Paris.")
    assert chunk.embed_text == "The capital of France is Paris."


def test_contextual_prefix_is_prepended_for_embedding_only() -> None:
    # Contextual retrieval (lesson 07) embeds context + text, but `text` stays pristine
    # so the citation still points at the original span.
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        text="Revenue grew 3%.",
        context="This chunk is from Acme's Q2 2024 earnings report.",
    )
    assert chunk.embed_text.startswith("This chunk is from Acme's Q2 2024")
    assert chunk.embed_text.endswith("Revenue grew 3%.")
    assert chunk.text == "Revenue grew 3%."
