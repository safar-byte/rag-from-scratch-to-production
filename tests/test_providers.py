"""The fakes must satisfy the same protocols the real backends do.

If this fails, the abstraction has drifted and a lesson written against the protocol
would break on one backend but not the other.
"""

from __future__ import annotations

from ragkit.providers.base import Embedder, Generator, Reranker
from ragkit.types import Chunk
from tests.fakes import FakeEmbedder, FakeGenerator, FakeReranker


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeReranker(), Reranker)
    assert isinstance(FakeGenerator(), Generator)


def test_embedder_is_deterministic_and_normalised() -> None:
    embedder = FakeEmbedder()
    first = embedder.embed_query("what is retrieval augmented generation")
    second = embedder.embed_query("what is retrieval augmented generation")
    assert first == second
    assert len(first) == embedder.dimension
    assert abs(sum(v * v for v in first) ** 0.5 - 1.0) < 1e-9


def test_reranker_orders_by_relevance_and_respects_top_k() -> None:
    chunks = [
        Chunk(chunk_id="a", doc_id="d", text="unrelated text about gardening"),
        Chunk(chunk_id="b", doc_id="d", text="retrieval augmented generation explained"),
    ]
    ranked = FakeReranker().rerank("retrieval augmented generation", chunks, top_k=1)
    assert len(ranked) == 1
    assert ranked[0][0].chunk_id == "b"
