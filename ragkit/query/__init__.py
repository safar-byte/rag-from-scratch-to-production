"""Query transformation: fixing the query rather than the index."""

from __future__ import annotations

from ragkit.query.retriever import TransformingRetriever
from ragkit.query.transform import (
    Hyde,
    Identity,
    MultiQuery,
    QueryTransformer,
    Rewrite,
    StepBack,
)

TRANSFORMERS = {
    "identity": Identity,
    "rewrite": Rewrite,
    "hyde": Hyde,
    "multiquery": MultiQuery,
    "stepback": StepBack,
}

__all__ = [
    "TRANSFORMERS",
    "Hyde",
    "Identity",
    "MultiQuery",
    "QueryTransformer",
    "Rewrite",
    "StepBack",
    "TransformingRetriever",
]
