"""Query transformation, parent-document expansion and routing."""

from __future__ import annotations

from ragkit.query.retriever import TransformingRetriever
from ragkit.query.transform import Hyde, Identity, MultiQuery, Rewrite, StepBack, _clean_lines
from ragkit.retrieve.parent import ParentDocumentRetriever
from ragkit.retrieve.router import HeuristicRouter, looks_lexical
from ragkit.types import Chunk, Document, Scored, Usage


class _StubGenerator:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.calls = 0

    @property
    def name(self) -> str:
        return "stub"

    def generate(self, *, system: str, prompt: str, max_tokens: int = 1024) -> tuple[str, Usage]:
        self.calls += 1
        return self._reply, Usage()


class _StubRetriever:
    def __init__(self, name: str = "stub") -> None:
        self._name = name
        self.queries: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        self.queries.append(query)
        return [
            Scored(
                chunk=Chunk(chunk_id=f"{query[:6]}-{i}", doc_id="d.md", text=f"{query} {i}"),
                score=1.0 - i / 10,
                rank=i + 1,
            )
            for i in range(top_k)
        ]


# ---- parsing model output --------------------------------------------------


def test_clean_lines_strips_bullets_numbering_and_preamble() -> None:
    raw = "Here are the queries:\n1. first useful query\n- second useful query\n* third one here"
    assert _clean_lines(raw, 3) == [
        "first useful query",
        "second useful query",
        "third one here",
    ]


def test_clean_lines_drops_fragments_too_short_to_be_queries() -> None:
    # A junk variant still costs a full retrieval pass, so it is worse than nothing.
    assert _clean_lines("ok\nno\na genuinely usable query", 5) == ["a genuinely usable query"]


# ---- transformers ----------------------------------------------------------


def test_identity_is_the_control_condition() -> None:
    assert Identity().transform("what is BM25") == ["what is BM25"]


def test_rewrite_keeps_the_original_alongside_the_rewrite() -> None:
    # A rewrite that drops the one rare term in the question is a common failure, and
    # keeping both costs one retrieval rather than one model call.
    assert Rewrite(_StubGenerator("BM25 scoring")).transform("what is BM25?") == [
        "what is BM25?",
        "BM25 scoring",
    ]


def test_rewrite_does_not_duplicate_an_unchanged_query() -> None:
    assert Rewrite(_StubGenerator("what is BM25?")).transform("what is BM25?") == ["what is BM25?"]


def test_rewrite_survives_an_empty_reply() -> None:
    assert Rewrite(_StubGenerator("")).transform("what is BM25?") == ["what is BM25?"]


def test_hyde_retrieves_with_both_question_and_hypothetical() -> None:
    variants = Hyde(_StubGenerator("BM25 is a lexical ranking function.")).transform("what is BM25")
    assert len(variants) == 2
    assert variants[1].startswith("BM25 is a lexical")


def test_multiquery_is_capped() -> None:
    generator = _StubGenerator("first variant here\nsecond variant here\nthird variant here")
    variants = MultiQuery(generator, variants=2).transform("original question")
    assert variants[0] == "original question"
    assert len(variants) == 3  # original plus two


def test_stepback_produces_a_broader_question() -> None:
    variants = StepBack(_StubGenerator("How does locking work?")).transform("why exit status 75")
    assert variants == ["why exit status 75", "How does locking work?"]


# ---- the transforming retriever --------------------------------------------


def test_a_single_variant_skips_fusion_entirely() -> None:
    """The control must be measured against the untouched pipeline, not RRF-of-one."""
    retriever = _StubRetriever()
    results = TransformingRetriever(retriever, Identity()).retrieve("q", top_k=3)
    assert retriever.queries == ["q"]
    assert "rrf" not in results[0].scores


def test_multiple_variants_are_each_retrieved_and_fused() -> None:
    retriever = _StubRetriever()
    transformer = Rewrite(_StubGenerator("rewritten query"))
    results = TransformingRetriever(retriever, transformer).retrieve("original query", top_k=3)

    assert retriever.queries == ["original query", "rewritten query"]
    assert all("rrf" in r.scores for r in results)
    assert results[0].scores["n_variants"] == 2.0


# ---- routing ---------------------------------------------------------------


def test_identifier_shapes_are_detected_as_lexical() -> None:
    for query in (
        "What is FUSION_CONSTANT set to?",
        "What does error TX-4491 mean?",
        "What does exit status 75 mean?",
        "Which version, v2.14.3?",
    ):
        assert looks_lexical(query), query


def test_prose_questions_are_not_routed_to_lexical() -> None:
    # A false lexical routing is the damaging direction: BM25 degrades catastrophically
    # on vocabulary mismatch, while dense degrades gracefully on identifiers.
    for query in (
        "Why do I get charged twice?",
        "How does chunk overlap work?",
        "My scanned paperwork comes back blank",
        "Can somebody work out the original wording from the stored numbers?",
    ):
        assert not looks_lexical(query), query


def test_router_sends_each_query_to_the_right_retriever() -> None:
    lexical, semantic = _StubRetriever("bm25"), _StubRetriever("dense")
    router = HeuristicRouter(lexical, semantic)

    router.retrieve("What is FUSION_CONSTANT set to?", top_k=2)
    router.retrieve("Why do I get charged twice?", top_k=2)

    assert lexical.queries == ["What is FUSION_CONSTANT set to?"]
    assert semantic.queries == ["Why do I get charged twice?"]
    assert [d[1] for d in router.decisions] == ["lexical", "semantic"]


# ---- parent-document expansion ---------------------------------------------

DOC = Document(
    doc_id="d.md",
    title="Doc",
    text=(
        "Opening paragraph that establishes the subject.\n\n"
        "The middle paragraph contains the specific fact being searched for.\n\n"
        "A closing paragraph with further detail and consequences.\n"
    ),
)


class _ChunkRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    @property
    def name(self) -> str:
        return "chunks"

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        return [
            Scored(chunk=c, score=1.0 - i / 10, rank=i + 1)
            for i, c in enumerate(self._chunks[:top_k])
        ]


def _child(start: int, end: int, suffix: str = "") -> Chunk:
    return Chunk(
        chunk_id=f"c{start}{suffix}",
        doc_id="d.md",
        text=DOC.text[start:end],
        start_char=start,
        end_char=end,
    )


def test_parent_expansion_returns_more_text_than_the_child() -> None:
    start = DOC.text.index("The middle paragraph")
    child = _child(start, start + 20)
    results = ParentDocumentRetriever(_ChunkRetriever([child]), [DOC], window=400).retrieve("q", 3)

    assert len(results) == 1
    assert len(results[0].chunk.text) > len(child.text)
    assert results[0].scores["parent_chars"] > results[0].scores["child_chars"]


def test_overlapping_windows_are_merged_not_duplicated() -> None:
    """Two nearby hits expand into near-identical windows.

    Returning both wastes context budget and pushes a genuinely different passage out
    of the top k, which is the opposite of what small-to-big is for.
    """
    start = DOC.text.index("The middle paragraph")
    children = [_child(start, start + 20), _child(start + 25, start + 45, "b")]
    results = ParentDocumentRetriever(_ChunkRetriever(children), [DOC], window=400).retrieve("q", 5)

    assert len(results) == 1
    assert results[0].scores["merged_windows"] == 2.0


def test_expansion_stays_inside_the_document() -> None:
    child = _child(0, 20)
    results = ParentDocumentRetriever(_ChunkRetriever([child]), [DOC], window=9999).retrieve("q", 3)
    assert results[0].chunk.start_char == 0
    assert results[0].chunk.end_char <= len(DOC.text)
