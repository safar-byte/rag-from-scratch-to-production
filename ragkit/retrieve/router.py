"""Query routing: pick a retriever per query instead of fusing everything every time.

This exists because of a measurement, not a preference. Lesson 04 found that hybrid
search *lost* to plain dense retrieval, and the reason was specific: RRF weights every
retriever equally, so on vocabulary-mismatch questions — where BM25 scored 0.250 at
rank 1, actively misleading rather than merely unhelpful — the bad ranking dragged good
dense results down.

There were two candidate fixes. Weighted RRF tunes a global constant against 44
questions, which is overfitting with extra steps. Routing decides *per query*, using the
one thing that reliably predicts which retriever will win: whether the query hinges on a
literal token or on meaning.

    "What does exit status 75 mean?"          -> lexical wins, the token is the query
    "Why do I get charged twice?"             -> dense wins, no shared vocabulary

The heuristic router below needs no model and no training. The LLM router is more
flexible and costs a generation per query, which on most corpora is not worth it —
measure before reaching for it.
"""

from __future__ import annotations

import re
from typing import Protocol

from ragkit.types import Scored

# Tokens that carry little meaning to an embedding model but are exact, rare, and
# decisive for BM25. Deliberately conservative: a false "lexical" routing on a prose
# question is more damaging than missing one identifier, because dense retrieval
# degrades gracefully on identifiers while BM25 degrades catastrophically on prose.
IDENTIFIER_PATTERNS = (
    re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b"),  # SCREAMING_SNAKE config keys
    re.compile(r"\b[A-Z]{2,}-\d+\b"),  # TX-4491, MBV2-2846
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b"),  # 1.2, v2.14.3
    re.compile(r"\b(?:status|code|port|error)\s+\d+\b", re.I),  # "exit status 75"
    re.compile(r"\b[a-z_]+\([a-z_,\s]*\)"),  # function_name(args)
)


class Route(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]: ...


def looks_lexical(query: str) -> bool:
    """Does this query hinge on an exact token?"""
    return any(pattern.search(query) for pattern in IDENTIFIER_PATTERNS)


class HeuristicRouter:
    """Route on the shape of the query. No model, no latency, fully inspectable.

    Being able to read the routing rule is worth a lot here. When a query is routed
    badly you can see exactly which pattern fired, which is not true of a model that
    returns a label and no reason.
    """

    def __init__(self, lexical: Route, semantic: Route) -> None:
        self._lexical = lexical
        self._semantic = semantic
        self.decisions: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return f"router(lex={self._lexical.name},sem={self._semantic.name})"

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        lexical = looks_lexical(query)
        chosen = self._lexical if lexical else self._semantic
        self.decisions.append((query, "lexical" if lexical else "semantic"))

        results = chosen.retrieve(query, top_k=top_k, where=where)
        for scored in results:
            scored.scores["route"] = 1.0 if lexical else 0.0
        return results


ROUTER_SYSTEM = """Classify a search query as LEXICAL or SEMANTIC.

LEXICAL: the answer hinges on an exact token — an identifier, a configuration key name,
an error code, a version number, a function name. Matching the literal string matters.

SEMANTIC: the answer depends on meaning. The user may have used entirely different
words from the document that answers them.

Reply with exactly one word: LEXICAL or SEMANTIC."""


class LlmRouter:
    """Route with a model.

    More flexible than the heuristic and it costs a full generation before retrieval
    even starts. On this corpus the heuristic is as good and free, which is the usual
    outcome — reach for the model only when the routing decision genuinely needs
    world knowledge that a regex cannot express.

    Falls back to semantic on anything unexpected, because dense retrieval degrades
    gracefully on identifiers while BM25 degrades catastrophically on prose. When the
    router is unsure, the asymmetric cost of the two mistakes decides.
    """

    def __init__(self, generator: object, lexical: Route, semantic: Route) -> None:
        self._generator = generator
        self._lexical = lexical
        self._semantic = semantic
        self.decisions: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return f"llm_router(lex={self._lexical.name},sem={self._semantic.name})"

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        try:
            text, _usage = self._generator.generate(  # type: ignore[attr-defined]
                system=ROUTER_SYSTEM, prompt=f"Query: {query}\n\nClassification:", max_tokens=8
            )
            lexical = "LEXICAL" in text.strip().upper()
        except Exception:  # noqa: BLE001 - a router failure must not fail the query
            lexical = False

        chosen = self._lexical if lexical else self._semantic
        self.decisions.append((query, "lexical" if lexical else "semantic"))
        return chosen.retrieve(query, top_k=top_k, where=where)
