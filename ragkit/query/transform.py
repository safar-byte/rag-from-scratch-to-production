"""Transforming the query before retrieval.

The user's question is an input to a retrieval system, not a search query. Everything
here tries to close that gap, and every one of them costs at least one full model call
*before* retrieval starts — which on a local CPU profile can double end-to-end latency
for a few points of recall.

So the honest framing is not "which of these should I use" but "does any of these buy
more than it costs on my corpus". Lesson 06 measures that. On this corpus the answer is
mostly no, and knowing why is more useful than a list of techniques.

Each transformer returns one or more query strings. A transformer that returns several
is fused with RRF downstream, exactly like a multi-retriever setup, because "several
rankings that must be merged" is the same problem whichever axis produced them.
"""

from __future__ import annotations

import re
from typing import Protocol

from ragkit.providers.base import Generator

MAX_VARIANTS = 4


class QueryTransformer(Protocol):
    @property
    def name(self) -> str: ...

    def transform(self, query: str) -> list[str]: ...


def _clean_lines(raw: str, limit: int) -> list[str]:
    """Pull a list of queries out of whatever shape the model replied in.

    Small models ignore "one per line, no numbering" often enough that parsing has to
    be forgiving. Returning junk here is worse than returning nothing: a malformed
    variant still costs a full retrieval pass.
    """
    lines: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", text)  # bullets and numbering
        text = text.strip().strip('"').strip()
        if len(text) < 8 or text.endswith(":"):  # headers and preamble
            continue
        lines.append(text)
        if len(lines) >= limit:
            break
    return lines


class Identity:
    """No transformation. The control condition, and the one to beat."""

    @property
    def name(self) -> str:
        return "identity"

    def transform(self, query: str) -> list[str]:
        return [query]


REWRITE_SYSTEM = """You turn a user's question into a short search query.

Keep the words that carry meaning. Drop politeness, framing and hedging. Expand obvious
abbreviations. Do not answer the question. Reply with the query only, nothing else."""


class Rewrite:
    """Strip a question down to a search query.

    Cheap and easy to inspect. Helps most on conversational input and does essentially
    nothing on input that was already keyword-shaped. It can also hurt, by discarding a
    term that happened to matter — which is why the original query is kept alongside it
    rather than replaced.
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    @property
    def name(self) -> str:
        return "rewrite"

    def transform(self, query: str) -> list[str]:
        text, _usage = self._generator.generate(
            system=REWRITE_SYSTEM, prompt=f"Question: {query}\n\nSearch query:", max_tokens=64
        )
        rewritten = text.strip().strip('"').splitlines()[0].strip() if text.strip() else ""
        # Keep the original. A rewrite that drops the one rare term in the question is
        # a common failure, and keeping both costs one extra retrieval, not a model call.
        return [query, rewritten] if rewritten and rewritten.lower() != query.lower() else [query]


HYDE_SYSTEM = """Write a short passage that would answer the user's question, as though
it came from a technical document. Two or three sentences. Be specific and use the
vocabulary such a document would use. Do not hedge, and do not say you are unsure —
this text is used only for retrieval, never shown to anyone."""


class Hyde:
    """Hypothetical Document Embeddings.

    Instead of improving the query, write a fake answer and retrieve with *that*. The
    premise is that an answer sits closer in embedding space to the real answer than a
    question does, because answers resemble answers.

    The failure mode is worth stating plainly: when the model lacks the domain
    knowledge, it hallucinates a passage about the wrong subject and retrieves
    confidently against it. HyDE does not degrade gracefully — it fails by being
    precisely wrong, and it costs a full generation before retrieval even begins.
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    @property
    def name(self) -> str:
        return "hyde"

    def transform(self, query: str) -> list[str]:
        text, _usage = self._generator.generate(
            system=HYDE_SYSTEM, prompt=f"Question: {query}\n\nPassage:", max_tokens=200
        )
        passage = text.strip()
        # Retrieve with both. Pure HyDE discards the question entirely, which is a large
        # bet on the hypothetical passage being about the right thing.
        return [query, passage] if passage else [query]


MULTI_QUERY_SYSTEM = """Write three different search queries for the user's question.

Vary the vocabulary: use synonyms and alternative phrasings someone might have written
in a document. Keep each one short. One per line, no numbering, no other text."""


class MultiQuery:
    """Fan out into several phrasings and fuse the results.

    Improves recall because different phrasings surface different chunks, and it is a
    natural fit for rank fusion. Costs one generation plus N retrievals, and the gains
    flatten fast — past about four variants the extra queries mostly return what the
    earlier ones already found.
    """

    def __init__(self, generator: Generator, variants: int = 3) -> None:
        self._generator = generator
        self._variants = min(variants, MAX_VARIANTS)

    @property
    def name(self) -> str:
        return f"multiquery({self._variants})"

    def transform(self, query: str) -> list[str]:
        text, _usage = self._generator.generate(
            system=MULTI_QUERY_SYSTEM, prompt=f"Question: {query}\n\nQueries:", max_tokens=200
        )
        return [query, *_clean_lines(text, self._variants)]


STEP_BACK_SYSTEM = """Write one broader, more general question that would provide useful
background for answering the user's specific question. Reply with the question only."""


class StepBack:
    """Ask a more general question alongside the specific one.

    For "why did the indexer exit with status 75", the step-back question is "how does
    the indexer handle locking". The general question retrieves the background that
    makes the specific answer interpretable.

    Works well on why-questions and poorly on lookups, where the generalisation just
    adds noise — which is an argument for routing it rather than applying it always.
    """

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    @property
    def name(self) -> str:
        return "stepback"

    def transform(self, query: str) -> list[str]:
        text, _usage = self._generator.generate(
            system=STEP_BACK_SYSTEM,
            prompt=f"Specific question: {query}\n\nGeneral question:",
            max_tokens=64,
        )
        broader = text.strip().strip('"').splitlines()[0].strip() if text.strip() else ""
        return [query, broader] if broader else [query]
