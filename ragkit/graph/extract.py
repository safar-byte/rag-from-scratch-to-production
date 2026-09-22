"""Pulling entities out of chunks, to build a graph from.

Two backends, and the cheap one is the default on purpose.

**Heuristic** — regexes for the things that behave like entities in technical prose:
configuration keys, identifiers, capitalised multi-word terms, backticked code. Free,
instant, and perfectly reproducible. On a technical corpus it captures most of what
matters, because technical entities are overwhelmingly *typographically* marked.

**LLM** — a model reads each chunk and names the entities. Catches concepts the regexes
cannot (an entity mentioned only in lowercase prose), and costs a call per chunk: on
this corpus, ~113 calls, about 45 minutes locally.

The honest default is heuristic. Reach for the LLM when the corpus is prose rather than
technical documentation, and measure whether the extra entities change any answer — the
usual outcome is a much larger graph that retrieves the same things.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from ragkit.providers.base import Generator
from ragkit.types import Chunk

# Ordered roughly by precision. Each captures a class of thing that is reliably an
# entity in technical writing and reliably *not* an entity elsewhere.
ENTITY_PATTERNS = (
    re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b"),  # CONFIG_KEY
    re.compile(r"\b[A-Z]{2,}-\d+\b"),  # TX-4491
    re.compile(r"`([^`]{2,40})`"),  # `backticked`
    re.compile(r"\b(?:BM25|RRF|HNSW|IVF|nDCG|MRR|OCR|HyDE|CRAG)\b"),  # known acronyms
    # Title Case Terms. A literal space/tab class, not \s: with \s the pattern
    # matches across a line break and swallows the start of the next sentence,
    # producing entities like "Vector Search An".
    re.compile(r"\b[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+){1,3}\b"),
)

# Sentence-initial capitals produce enormous numbers of junk "entities". These are the
# openers that actually appear in this corpus; a general system would use a POS tagger.
STOP_ENTITIES = {
    "the",
    "this",
    "that",
    "a",
    "an",
    "it",
    "they",
    "there",
    "some",
    "most",
    "each",
    "both",
    "two",
    "three",
    "one",
    "when",
    "where",
    "which",
    "what",
    "why",
    "how",
    "if",
    "so",
    "and",
    "but",
    "for",
    "with",
    "without",
    "before",
    "after",
    "note",
    "hence",
    "then",
    "also",
    "only",
    "every",
    "any",
    "no",
    "not",
    "use",
    "used",
    "using",
    "do",
    "does",
    "doing",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
}


def _normalise(raw: str) -> str:
    return " ".join(raw.strip().strip("`.,;:()[]").split())


def _is_plausible(entity: str) -> bool:
    if len(entity) < 3 or len(entity) > 60:
        return False
    words = entity.split()
    # Reject on either end. A leading stopword means a sentence-initial capital was
    # mistaken for a name; a trailing one means the match ran past the real term, as in
    # "Dimensionality The".
    if words[0].lower() in STOP_ENTITIES or words[-1].lower() in STOP_ENTITIES:
        return False
    # A doubled word is a heading immediately followed by its own body: "Chunking
    # Chunking splits documents..." -> "Chunking Chunking".
    return not (len(words) == 2 and words[0].lower() == words[1].lower())


def extract_heuristic(chunk: Chunk, limit: int = 12) -> list[str]:
    """Typographically marked entities. No model, no cost."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(chunk.text):
            entity = _normalise(match.group(1) if match.groups() else match.group(0))
            key = entity.lower()
            if entity and key not in seen and _is_plausible(entity):
                seen.add(key)
                found.append(entity)
            if len(found) >= limit:
                return found
    return found


EXTRACT_SYSTEM = """List the key entities in the passage: specific technical concepts,
named components, configuration settings, algorithms, metrics.

Only things the passage actually discusses. No generic words. At most eight.

Reply with a JSON array of strings and nothing else."""


def extract_llm(chunk: Chunk, generator: Generator, limit: int = 8) -> list[str]:
    try:
        raw, _usage = generator.generate(
            system=EXTRACT_SYSTEM, prompt=f"Passage:\n{chunk.text}\n\nJSON array:", max_tokens=150
        )
    except Exception:  # noqa: BLE001 - fall back rather than fail the whole build
        return extract_heuristic(chunk, limit)

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return extract_heuristic(chunk, limit)
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return extract_heuristic(chunk, limit)

    out: list[str] = []
    for item in items if isinstance(items, list) else []:
        entity = _normalise(str(item))
        if entity and _is_plausible(entity):
            out.append(entity)
        if len(out) >= limit:
            break
    return out


def entity_frequencies(chunks: list[Chunk]) -> Counter[str]:
    """How often each entity appears across the corpus.

    Useful for pruning. An entity in almost every chunk is a stopword with extra steps:
    it connects everything to everything and makes graph traversal meaningless.
    """
    counter: Counter[str] = Counter()
    for chunk in chunks:
        for entity in extract_heuristic(chunk):
            counter[entity.lower()] += 1
    return counter
