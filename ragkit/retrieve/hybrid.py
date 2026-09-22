"""Hybrid retrieval: dense and lexical, fused by Reciprocal Rank Fusion.

The problem fusion has to solve is that the two retrievers' scores are not comparable.
A cosine similarity of 0.82 and a BM25 score of 14.3 live on different scales, and
normalising them is fragile because BM25 has no fixed upper bound — the maximum depends
on the corpus, the query, and the term statistics.

RRF sidesteps this by throwing the scores away and using only the ranks:

    score(d) = sum over retrievers r of  1 / (k + rank_r(d))

`k` damps the influence of the very top ranks. At k=60 the gap between rank 1 and rank 2
is small, so a document ranked moderately well by *both* retrievers can outrank one
ranked first by only one. That is the entire point: agreement between two independent
methods is strong evidence, and RRF is a way of counting votes rather than averaging
opinions.

No training, no normalisation, no per-corpus tuning. It is hard to beat for how simple
it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ragkit.types import Scored

DEFAULT_K = 60


class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]: ...


def reciprocal_rank_fusion(
    rankings: dict[str, list[Scored]],
    k: int = DEFAULT_K,
    top_k: int = 5,
) -> list[Scored]:
    """Fuse several ranked lists into one.

    `rankings` maps a retriever name to its results. Per-retriever ranks and raw scores
    are preserved on the fused result, because "which stage put this here" is the
    question you actually ask when debugging, and it is unrecoverable once the lists
    are merged.
    """
    fused: dict[str, Scored] = {}

    for retriever_name, results in rankings.items():
        for position, scored in enumerate(results, start=1):
            chunk_id = scored.chunk.chunk_id
            contribution = 1.0 / (k + position)

            if chunk_id not in fused:
                fused[chunk_id] = Scored(chunk=scored.chunk, score=0.0, scores={})
            entry = fused[chunk_id]
            entry.score += contribution
            entry.scores[f"{retriever_name}_rank"] = float(position)
            entry.scores[f"rrf_{retriever_name}"] = round(contribution, 6)
            # Keep the raw score too — it is meaningless for fusion and useful for
            # working out *why* a retriever ranked something where it did.
            entry.scores.update(scored.scores)

    ordered = sorted(fused.values(), key=lambda s: s.score, reverse=True)
    for rank, scored in enumerate(ordered, start=1):
        scored.rank = rank
        scored.scores["rrf"] = round(scored.score, 6)
    return ordered[:top_k]


class HybridRetriever:
    """Runs several retrievers and fuses their rankings.

    `candidates` is how deep each retriever goes before fusion, and it matters more than
    it looks. RRF can only rank documents that appeared in at least one input list, so
    this value is the real recall ceiling of the whole pipeline — including anything a
    reranker does downstream. Fetching 5 from each and fusing to 5 mostly wastes the
    fusion.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        k: int = DEFAULT_K,
        candidates: int = 25,
    ) -> None:
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever.")
        self._retrievers = list(retrievers)
        self._k = k
        self._candidates = candidates

    @property
    def name(self) -> str:
        return "hybrid(" + "+".join(r.name for r in self._retrievers) + f",rrf{self._k})"

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        depth = max(self._candidates, top_k)
        rankings = {
            retriever.name: retriever.retrieve(query, top_k=depth, where=where)
            for retriever in self._retrievers
        }
        return reciprocal_rank_fusion(rankings, k=self._k, top_k=top_k)
