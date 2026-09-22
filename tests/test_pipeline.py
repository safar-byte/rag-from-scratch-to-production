"""End-to-end pipeline tests against the real loader, chunker and Chroma store.

Only the embedder and generator are faked. The storage and retrieval path is the real
one, because that is where the interesting bugs live.
"""

from __future__ import annotations

from ragkit.pipeline import RagPipeline


def test_ingest_indexes_the_whole_corpus(pipeline: RagPipeline) -> None:
    stats = pipeline.ingest()
    assert stats["documents"] >= 6
    assert stats["chunks"] > stats["documents"]
    assert pipeline.count() == stats["chunks"]


def test_ingest_is_idempotent(pipeline: RagPipeline) -> None:
    first = pipeline.ingest()["chunks"]
    second = pipeline.ingest()["chunks"]
    assert first == second
    # reset=True means a re-ingest must not double the index.
    assert pipeline.count() == first


def test_retrieve_returns_ranked_scored_chunks(pipeline: RagPipeline) -> None:
    pipeline.ingest()
    results = pipeline.retrieve("reciprocal rank fusion", top_k=4)

    assert len(results) == 4
    assert [r.rank for r in results] == [1, 2, 3, 4]
    # Scores must be descending, and expressed as similarity (higher is better) —
    # Chroma hands back a distance, and the store is responsible for flipping it.
    assert results == sorted(results, key=lambda r: r.score, reverse=True)
    assert all("dense" in r.scores for r in results)


def test_retrieved_chunks_carry_their_provenance(pipeline: RagPipeline) -> None:
    pipeline.ingest()
    for result in pipeline.retrieve("chunking", top_k=3):
        chunk = result.chunk
        assert chunk.doc_id and chunk.chunk_id.startswith(chunk.doc_id)
        assert chunk.title
        assert chunk.end_char > chunk.start_char


def test_ask_produces_an_answer_with_context_and_a_trace(pipeline: RagPipeline) -> None:
    pipeline.ingest()
    answer = pipeline.ask("What is BM25?", top_k=3)

    assert answer.text
    assert len(answer.contexts) == 3
    assert answer.trace["retriever"] == "dense"
    assert answer.trace["profile"] == "local"
    assert answer.trace["total_ms"] >= 0


def test_the_prompt_actually_contains_the_retrieved_passages(pipeline: RagPipeline) -> None:
    pipeline.ingest()
    answer = pipeline.ask("What is BM25?", top_k=3)

    _system, prompt = pipeline._generator.calls[0]
    for index, result in enumerate(answer.contexts, start=1):
        assert f"[{index}]" in prompt
        # A distinctive slice of each chunk must survive into the prompt.
        assert result.chunk.text.strip()[:40] in prompt


def test_empty_query_retrieves_nothing(pipeline: RagPipeline) -> None:
    pipeline.ingest()
    assert pipeline.retrieve("   ") == []
