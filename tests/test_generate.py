"""Prompt assembly and citation extraction."""

from __future__ import annotations

from ragkit.generate import SYSTEM_PROMPT, build_prompt, extract_citations, format_context
from ragkit.types import Chunk, Scored


def _scored(n: int) -> list[Scored]:
    return [
        Scored(
            chunk=Chunk(
                chunk_id=f"doc{i}.md::0000",
                doc_id=f"doc{i}.md",
                text=f"Passage number {i} says something specific and checkable.",
                title=f"Doc {i}",
            ),
            score=1.0 - i / 10,
            rank=i,
        )
        for i in range(1, n + 1)
    ]


def test_context_is_numbered_and_attributed() -> None:
    rendered = format_context(_scored(3))
    for i in (1, 2, 3):
        assert f"[{i}]" in rendered
        assert f"doc{i}.md" in rendered


def test_context_budget_drops_whole_passages_not_partial_ones() -> None:
    results = _scored(5)
    rendered = format_context(results, max_chars=200)
    # Whatever survives must be intact: a half passage can strand a claim away from
    # the qualifier that makes it true.
    for result in results:
        text = result.chunk.text
        assert text in rendered or text.split(".")[0] not in rendered


def test_prompt_puts_context_before_the_question() -> None:
    prompt = build_prompt("What does passage 2 say?", _scored(3))
    assert prompt.index("Context passages:") < prompt.index("Question:")


def test_system_prompt_forbids_answering_beyond_the_context() -> None:
    lowered = SYSTEM_PROMPT.lower()
    assert "only" in lowered
    assert "do not guess" in lowered


def test_empty_retrieval_produces_an_honest_prompt() -> None:
    prompt = build_prompt("Anything?", [])
    assert "No context passages were retrieved" in prompt


def test_citations_resolve_markers_back_to_chunks() -> None:
    results = _scored(3)
    citations = extract_citations("This is true [1] and also this [3].", results)
    assert [c.doc_id for c in citations] == ["doc1.md", "doc3.md"]


def test_citations_handle_grouped_and_repeated_markers() -> None:
    results = _scored(4)
    citations = extract_citations("Claim [2][4], restated [2], and [1, 3].", results)
    assert [c.doc_id for c in citations] == ["doc2.md", "doc4.md", "doc1.md", "doc3.md"]


def test_out_of_range_markers_are_ignored_rather_than_crashing() -> None:
    # A model citing [9] when three passages were supplied is a real occurrence, and
    # the only mechanism keeping marker-parsing honest is that it must not invent a
    # source. Lesson 12 replaces this with the Citations API for exactly this reason.
    citations = extract_citations("As shown [9] and [0] and [2].", _scored(3))
    assert [c.doc_id for c in citations] == ["doc2.md"]
