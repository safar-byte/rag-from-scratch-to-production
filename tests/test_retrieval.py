"""Hybrid fusion and reranking.

RRF is tested against hand-computed arithmetic for the same reason the metrics are: it
silently decides the ordering everything downstream sees, and a wrong constant or an
off-by-one in the rank produces plausible output rather than an error.
"""

from __future__ import annotations

import pytest

from ragkit.retrieve.hybrid import HybridRetriever, reciprocal_rank_fusion
from ragkit.retrieve.rerank import RerankingRetriever
from ragkit.types import Chunk, Scored
from tests.fakes import FakeReranker


def _ranked(doc_ids: list[str]) -> list[Scored]:
    return [
        Scored(
            chunk=Chunk(chunk_id=doc_id, doc_id=doc_id, text=f"text of {doc_id}"),
            score=1.0 - i / 100,
            rank=i + 1,
        )
        for i, doc_id in enumerate(doc_ids)
    ]


class _StubRetriever:
    def __init__(self, name: str, doc_ids: list[str]) -> None:
        self._name = name
        self._doc_ids = doc_ids
        self.last_top_k: int | None = None

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        self.last_top_k = top_k
        return _ranked(self._doc_ids[:top_k])


# ---- RRF -------------------------------------------------------------------


def test_rrf_matches_hand_computed_scores() -> None:
    fused = reciprocal_rank_fusion({"a": _ranked(["x"]), "b": _ranked(["x"])}, k=60, top_k=5)
    # Rank 1 in both lists: 1/61 + 1/61.
    assert fused[0].score == pytest.approx(2 / 61)


def test_agreement_beats_a_single_strong_vote() -> None:
    # This is the whole design of RRF: 'b' is second in both lists and should outrank
    # 'a', which is first in one and absent from the other.
    fused = reciprocal_rank_fusion(
        {"dense": _ranked(["a", "b"]), "bm25": _ranked(["c", "b"])}, k=60, top_k=3
    )
    assert fused[0].chunk.doc_id == "b"


def test_k_controls_how_much_the_top_rank_dominates() -> None:
    """Small k favours a single strong vote; large k favours agreement.

    The rank gap has to be wide for this to show. Against two rank-2 votes, a single
    rank-1 vote loses at *every* positive k: 1/(k+1) < 2/(k+2) always. So 'b' sits at
    rank 5 in both lists while 'a' is rank 1 in one.

      k=1:   a = 1/2 = 0.500   b = 2/6  = 0.333  -> a
      k=60:  a = 1/61 = 0.016  b = 2/65 = 0.031  -> b
    """
    rankings = {
        "dense": _ranked(["a", "w", "x", "y", "b"]),
        "bm25": _ranked(["p", "q", "r", "s", "b"]),
    }
    assert reciprocal_rank_fusion(rankings, k=1, top_k=3)[0].chunk.doc_id == "a"
    assert reciprocal_rank_fusion(rankings, k=60, top_k=3)[0].chunk.doc_id == "b"


def test_a_single_top_vote_never_beats_two_second_place_votes() -> None:
    # 1/(k+1) < 2/(k+2) for every positive k, so no choice of k rescues 'a' here.
    # Worth pinning: it is the clearest statement of what RRF actually rewards.
    rankings = {"dense": _ranked(["a", "b"]), "bm25": _ranked(["c", "b"])}
    for k in (1, 10, 60, 200):
        assert reciprocal_rank_fusion(rankings, k=k, top_k=3)[0].chunk.doc_id == "b"


def test_fusion_preserves_each_retriever_rank_for_debugging() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": _ranked(["a", "b"]), "bm25": _ranked(["b", "a"])}, top_k=2
    )
    entry = next(s for s in fused if s.chunk.doc_id == "a")
    assert entry.scores["dense_rank"] == 1.0
    assert entry.scores["bm25_rank"] == 2.0
    assert "rrf" in entry.scores


def test_documents_in_one_list_only_still_appear() -> None:
    fused = reciprocal_rank_fusion({"dense": _ranked(["a"]), "bm25": _ranked(["b"])}, top_k=5)
    assert {s.chunk.doc_id for s in fused} == {"a", "b"}


def test_fused_ranks_are_renumbered_consecutively() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": _ranked(["a", "b", "c"]), "bm25": _ranked(["c", "b", "a"])}, top_k=3
    )
    assert [s.rank for s in fused] == [1, 2, 3]


# ---- hybrid wiring ---------------------------------------------------------


def test_hybrid_over_fetches_beyond_the_requested_top_k() -> None:
    # The candidate depth is the real recall ceiling: RRF can only rank what it was
    # given, so fetching only top_k from each retriever wastes the fusion.
    dense = _StubRetriever("dense", [f"d{i}" for i in range(50)])
    lexical = _StubRetriever("bm25", [f"b{i}" for i in range(50)])
    HybridRetriever([dense, lexical], candidates=25).retrieve("q", top_k=5)
    assert dense.last_top_k == 25
    assert lexical.last_top_k == 25


def test_hybrid_needs_at_least_one_retriever() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])


# ---- reranking -------------------------------------------------------------


def test_reranker_reorders_the_shortlist() -> None:
    base = _StubRetriever("base", ["irrelevant", "the target chunk"])
    reranked = RerankingRetriever(base, FakeReranker(), candidates=10).retrieve(
        "target chunk", top_k=2
    )
    assert reranked[0].chunk.doc_id == "the target chunk"


def test_reranker_records_the_rank_it_moved_from() -> None:
    # The delta between the two ranks is the only honest measure of what the reranker
    # contributed, and it is lost if the upstream rank is overwritten.
    base = _StubRetriever("base", ["irrelevant", "the target chunk"])
    reranked = RerankingRetriever(base, FakeReranker(), candidates=10).retrieve(
        "target chunk", top_k=2
    )
    assert reranked[0].scores["rank_before_rerank"] == 2.0
    assert reranked[0].rank == 1


def test_shortlist_depth_caps_what_reranking_can_reach() -> None:
    """The ceiling rule, as an executable assertion.

    The target sits at position 5. With a shortlist of 3 the reranker never sees it,
    so no amount of reranking quality can surface it.
    """
    docs = ["junk1", "junk2", "junk3", "junk4", "the target chunk"]
    base = _StubRetriever("base", docs)

    shallow = RerankingRetriever(base, FakeReranker(), candidates=3).retrieve("target chunk", 3)
    assert "the target chunk" not in {s.chunk.doc_id for s in shallow}

    deep = RerankingRetriever(base, FakeReranker(), candidates=10).retrieve("target chunk", 3)
    assert deep[0].chunk.doc_id == "the target chunk"


def test_reranking_an_empty_shortlist_returns_nothing() -> None:
    base = _StubRetriever("base", [])
    assert RerankingRetriever(base, FakeReranker(), candidates=10).retrieve("q", 5) == []
