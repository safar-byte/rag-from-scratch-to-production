"""Multi-hop retrieval: search, read, search again with what you learned.

Some questions cannot be answered by any single retrieval, because the terms needed for
the second search only appear in the result of the first. From this repo's own golden
set:

    "Dense retrieval fails on error codes. Which retrieval method fixes that,
     and why does it work where embeddings do not?"

One search finds the passage about dense retrieval's weakness. Nothing in the question
says "BM25", so the passage explaining *why* BM25 works is reached only after reading
the first result. A single query cannot span that gap however good the retriever is.

Multi-hop closes it by letting the model write the follow-up query. That is genuinely
useful and genuinely dangerous: the loop now decides its own inputs, so it needs a hard
iteration cap, a stopping condition, and accumulated results that never shrink.

## When not to use this

Most questions are not multi-hop, and running hops on all of them multiplies cost for
nothing. On this corpus only 7 of 44 golden questions span two documents. The
`sufficient` check below exists to stop early, and the honest deployment is to route
into multi-hop rather than default to it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ragkit.providers.base import Generator
from ragkit.types import Scored

FOLLOWUP_SYSTEM = """You decide whether retrieved passages are enough to answer a
question, and if not, what to search for next.

Reply with one of exactly these two forms and nothing else:

  ENOUGH
  SEARCH: <a short search query for the missing piece>

Ask for a follow-up only when something specific is genuinely missing. If the passages
already contain the answer, reply ENOUGH. A follow-up that just rephrases the original
question is worse than stopping."""


@dataclass(slots=True)
class HopTrace:
    hops: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    stopped_because: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "hops": len(self.queries),
            "hop_queries": self.queries,
            "hop_steps": self.hops,
            "stopped_because": self.stopped_because,
        }


class MultiHopRetriever:
    """Iterative retrieve-read-retrieve, with a hard cap.

    `max_hops` counts *extra* searches beyond the first, so the default of 2 means at
    most three retrievals and two model calls. Results accumulate and are deduplicated
    by chunk id; a later hop can never remove what an earlier one found, which keeps
    the loop monotonic and means a bad follow-up query wastes time rather than
    destroying a good result set.
    """

    def __init__(self, retriever: object, generator: Generator, *, max_hops: int = 2) -> None:
        self._retriever = retriever
        self._generator = generator
        self._max_hops = max_hops
        self.last_trace = HopTrace()

    @property
    def name(self) -> str:
        return f"multihop({getattr(self._retriever, 'name', '?')},h={self._max_hops})"

    def _next_query(self, question: str, results: list[Scored]) -> str | None:
        context = "\n".join(f"- {s.chunk.text[:220]}" for s in results[:5])
        try:
            text, _usage = self._generator.generate(
                system=FOLLOWUP_SYSTEM,
                prompt=f"Question: {question}\n\nSo far:\n{context}\n\nReply:",
                max_tokens=64,
            )
        except Exception:  # noqa: BLE001 - a planner failure stops the loop, not the query
            return None

        reply = text.strip()
        if "SEARCH:" not in reply.upper():
            return None
        query = reply[reply.upper().index("SEARCH:") + 7 :].strip().splitlines()[0].strip()
        return query.strip('"') or None

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        trace = HopTrace(queries=[query])
        seen: dict[str, Scored] = {}

        def absorb(results: list[Scored]) -> None:
            for scored in results:
                # Keep the best score seen for a chunk across hops.
                existing = seen.get(scored.chunk.chunk_id)
                if existing is None or scored.score > existing.score:
                    seen[scored.chunk.chunk_id] = scored

        absorb(self._retriever.retrieve(query, top_k=top_k, where=where))  # type: ignore[attr-defined]
        trace.hops.append(f"hop 0: {len(seen)} chunks")

        for hop in range(1, self._max_hops + 1):
            ordered = sorted(seen.values(), key=lambda s: s.score, reverse=True)
            follow_up = self._next_query(query, ordered)

            if follow_up is None:
                trace.stopped_because = "model said enough"
                break
            if follow_up.lower() in {q.lower() for q in trace.queries}:
                # A repeated query returns what is already held, so the loop has
                # converged. Without this check a model that keeps rephrasing the same
                # thing burns the full hop budget every single time.
                trace.stopped_because = "follow-up repeated an earlier query"
                break

            trace.queries.append(follow_up)
            absorb(self._retriever.retrieve(follow_up, top_k=top_k, where=where))  # type: ignore[attr-defined]
            trace.hops.append(f"hop {hop}: {follow_up!r} -> {len(seen)} chunks")
        else:
            trace.stopped_because = "hop limit reached"

        self.last_trace = trace
        started = time.perf_counter()
        out = sorted(seen.values(), key=lambda s: s.score, reverse=True)[:top_k]
        for rank, scored in enumerate(out, start=1):
            scored.rank = rank
            scored.scores["n_hops"] = float(len(trace.queries))
        _ = started
        return out
