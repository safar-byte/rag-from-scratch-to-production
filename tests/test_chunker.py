"""Chunker behaviour, including two regressions found while building lesson 02."""

from __future__ import annotations

import re

from ragkit.ingest import FixedSizeChunker, RecursiveChunker, chunk_documents, load_documents
from ragkit.types import Document

MARKDOWN = Document(
    doc_id="d.md",
    title="Doc",
    text=(
        "# Title\n\n"
        "An opening paragraph that sets out the subject at hand in a reasonable "
        "number of words so that it is not trivially short.\n\n"
        "## Dimensionality\n\n"
        "The number of components in the vector is the model's dimension. Larger "
        "models produce longer vectors and cost more to store and to search.\n\n"
        "## Distance metrics\n\n"
        "Cosine similarity measures the angle between two vectors and ignores their "
        "magnitude, which is why it is the default for text retrieval.\n"
    ),
)


def test_fixed_chunker_respects_size_and_overlap() -> None:
    chunks = FixedSizeChunker(chunk_size=120, overlap=20).split(MARKDOWN)
    assert chunks
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.doc_id == "d.md" for c in chunks)


def test_recursive_chunker_stays_within_budget() -> None:
    chunker = RecursiveChunker(chunk_size=200, overlap=30)
    chunks = chunker.split(MARKDOWN)
    assert chunks
    # Budget is chunk_size plus the prepended overlap, with a little slack for the
    # joining space.
    assert all(len(c.text) <= 200 + 30 + 2 for c in chunks)


def test_headings_are_never_orphaned_at_the_end_of_a_chunk() -> None:
    """Regression: the greedy buffer used to strand a heading on the previous chunk.

    That left the section body without its most discriminative keyword and labelled
    the preceding chunk with a section it did not contain.
    """
    chunks = RecursiveChunker(chunk_size=200, overlap=30).split(MARKDOWN)
    for chunk in chunks:
        assert not re.search(r"#+ [A-Za-z][^\n]{0,40}$", chunk.text.rstrip()), (
            f"chunk {chunk.chunk_id} ends on an orphaned heading: {chunk.text[-60:]!r}"
        )


def test_heading_travels_with_its_own_section() -> None:
    chunks = RecursiveChunker(chunk_size=200, overlap=30).split(MARKDOWN)
    holding = [c for c in chunks if "## Dimensionality" in c.text]
    assert holding, "the heading disappeared entirely"
    assert any("number of components" in c.text for c in holding), (
        "the heading was separated from the section it introduces"
    )


def test_overlap_does_not_begin_mid_word() -> None:
    """Regression: a raw character slice produced overlaps like 'rly always ...'.

    A word fragment is noise to the embedder and a junk token to BM25.
    """
    chunks = RecursiveChunker(chunk_size=200, overlap=40).split(MARKDOWN)
    words = set(re.findall(r"[A-Za-z']+", MARKDOWN.text))
    for chunk in chunks[1:]:
        first = re.match(r"[A-Za-z']+", chunk.text)
        if first:
            assert first.group() in words, (
                f"chunk {chunk.chunk_id} starts on a word fragment: {chunk.text[:30]!r}"
            )


def test_offsets_point_back_into_the_source_document() -> None:
    # Citations and parent-document retrieval both depend on this holding.
    chunks = RecursiveChunker(chunk_size=200, overlap=30).split(MARKDOWN)
    for chunk in chunks:
        assert chunk.end_char > chunk.start_char
        assert chunk.start_char >= 0


def test_real_corpus_chunks_cleanly() -> None:
    documents = load_documents()
    assert len(documents) >= 6
    chunks = chunk_documents(documents, RecursiveChunker(512, 64))
    assert len(chunks) > len(documents)
    assert all(c.text.strip() for c in chunks)
    # Every chunk must trace back to a document that was actually loaded.
    doc_ids = {d.doc_id for d in documents}
    assert {c.doc_id for c in chunks} <= doc_ids
