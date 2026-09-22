"""A retriever that transforms the query first, then fuses the results.

Several query variants produce several rankings, which is the same problem hybrid
search already solved — so this reuses RRF rather than inventing a second merge
strategy. "Rankings that must be merged" does not care whether the axis was different
retrievers or different phrasings of one question.
"""

from __future__ import annotations

from ragkit.query.transform import QueryTransformer
from ragkit.retrieve.hybrid import DEFAULT_K, reciprocal_rank_fusion
from ragkit.types import Scored


class TransformingRetriever:
    """Transform the query, retrieve for each variant, fuse.

    Wraps any retriever, so it composes with dense, hybrid or reranking underneath.
    Note the cost shape: one model call for the transform, then *N* retrieval passes.
    With a reranker underneath, N reranking passes too — which is how a technique that
    looks cheap on paper turns into the slowest thing in the pipeline.
    """

    def __init__(
        self,
        retriever: object,
        transformer: QueryTransformer,
        k: int = DEFAULT_K,
        candidates: int = 10,
    ) -> None:
        self._retriever = retriever
        self._transformer = transformer
        self._k = k
        self._candidates = candidates

    @property
    def name(self) -> str:
        inner = getattr(self._retriever, "name", "?")
        return f"{self._transformer.name}->{inner}"

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        variants = [v for v in self._transformer.transform(query) if v and v.strip()]
        if not variants:
            variants = [query]

        # One variant is the common case (Identity, or a transformer that declined to
        # change anything). Skip fusion entirely so the control condition is measured
        # against the untouched pipeline rather than against RRF-of-one.
        if len(variants) == 1:
            return self._retriever.retrieve(variants[0], top_k=top_k, where=where)  # type: ignore[attr-defined]

        depth = max(self._candidates, top_k)
        rankings: dict[str, list[Scored]] = {}
        for index, variant in enumerate(variants):
            results = self._retriever.retrieve(variant, top_k=depth, where=where)  # type: ignore[attr-defined]
            # The original query is variant 0 and is named so it stands out in a trace.
            rankings[f"q{index}" if index else "original"] = results

        fused = reciprocal_rank_fusion(rankings, k=self._k, top_k=top_k)
        for scored in fused:
            scored.scores["n_variants"] = float(len(variants))
        return fused
