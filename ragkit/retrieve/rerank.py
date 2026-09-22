"""Reranking: reorder a shortlist with a model that reads query and chunk together.

A bi-encoder — an ordinary embedding model — encodes the query and the document
separately, so document vectors are precomputed and search is fast. The two texts never
meet until the similarity calculation, which limits how well relevance can be judged.

A cross-encoder feeds both through the model at once and attends across them. Far more
accurate, and far too slow to run over a corpus, because nothing can be precomputed.

Hence the over-fetch pattern: cheap retrieval produces a shortlist, the cross-encoder
reorders it. The consequence that matters more than any other in this file:

    **Recall at the shortlist depth is a hard ceiling on final quality.**

A reranker reorders; it cannot conjure. If the answer is not in the shortlist, no
reranker puts it in the top 5. When reranking disappoints, the problem is nearly always
shortlist depth or upstream recall — not the reranker.
"""

from __future__ import annotations

from ragkit.providers.base import Reranker
from ragkit.types import Scored


class RerankingRetriever:
    """Wraps another retriever and reorders its shortlist.

    `candidates` is the over-fetch depth. Raising it improves the ceiling and costs
    linearly — a hosted reranker bills per document scored, and a local cross-encoder
    spends roughly proportional time. Lesson 05 sweeps it rather than guessing.
    """

    def __init__(self, retriever: object, reranker: Reranker, candidates: int = 25) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._candidates = candidates

    @property
    def name(self) -> str:
        return f"rerank({getattr(self._retriever, 'name', '?')},n={self._candidates})"

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        shortlist = self._retriever.retrieve(  # type: ignore[attr-defined]
            query, top_k=max(self._candidates, top_k), where=where
        )
        if not shortlist:
            return []

        by_id = {s.chunk.chunk_id: s for s in shortlist}
        reranked = self._reranker.rerank(query, [s.chunk for s in shortlist], top_k=top_k)

        out: list[Scored] = []
        for rank, (chunk, score) in enumerate(reranked, start=1):
            previous = by_id.get(chunk.chunk_id)
            # Carry the upstream scores forward and record where the chunk sat before
            # reranking. The delta between the two ranks is the only honest measure of
            # what the reranker contributed, and it is lost if you overwrite.
            scores = dict(previous.scores) if previous else {}
            if previous is not None:
                scores["rank_before_rerank"] = float(previous.rank)
            scores["rerank"] = round(float(score), 6)
            out.append(Scored(chunk=chunk, score=float(score), scores=scores, rank=rank))
        return out
