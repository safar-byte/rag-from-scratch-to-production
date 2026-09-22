"""Agentic RAG: pipelines that make decisions rather than running straight through.

The hazard that comes with the capability: a pipeline that can decide to retry can
decide to retry forever. Every loop here has an explicit, injectable bound.
"""

from __future__ import annotations

from ragkit.agent.crag import CorrectiveRag
from ragkit.agent.grade import ContextGrader, Grading, Verdict
from ragkit.agent.multihop import MultiHopRetriever

__all__ = [
    "ContextGrader",
    "CorrectiveRag",
    "Grading",
    "MultiHopRetriever",
    "Verdict",
]
