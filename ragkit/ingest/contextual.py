"""Contextual retrieval: situating each chunk in its document before embedding.

The problem, concretely. A chunk that reads, in full:

    Revenue grew 3% over the previous quarter.

Which company? Which quarter? The chunk does not say, because the surrounding document
said it. Embedded as-is it will never be retrieved by "How did Acme do in Q2 2024?" —
those words appear nowhere in it. Pronouns, relative dates and unqualified references
("the system", "this approach", "the above") all produce chunks that are individually
unretrievable.

The fix is to ask a cheap model to write a sentence or two situating the chunk in its
document, and prepend that before embedding:

    This chunk is from Acme Corp's Q2 2024 earnings report, in the section on the
    cloud infrastructure segment.

    Revenue grew 3% over the previous quarter.

Two details that are easy to get wrong:

**The prefix is stored separately from the text.** `Chunk.context` and `Chunk.text` stay
distinct, and `embed_text` joins them. If you concatenate at ingest, every later
citation points at generated text rather than at the source, and you have quietly made
the system unable to prove anything.

**The lexical index must get the prefix too.** The prefix adds exactly the keywords a
user is likely to type. Indexing it for dense search and not for BM25 leaves half the
value of the technique on the table — `ragkit/retrieve/bm25.py` indexes `embed_text`
for this reason.

## Cost

The naive implementation sends the whole document once per chunk: a 200-chunk document
gets transmitted 200 times. Three things fix that, in descending order of importance:

1. **Prompt caching.** The document is a stable prefix across every chunk derived from
   it, so it is processed once and read from cache thereafter at a fraction of the input
   cost. `ClaudeGenerator.generate(cache_system=True)` puts the document in the cached
   system block. This is the single largest lever.
2. **The Batch API**, at roughly half price. This is a one-time ingest-time pass with no
   latency requirement, which is exactly what batch is for.
3. **A cheap model.** Situating a chunk is an easy task; `claude-haiku-4-5` does it as
   well as anything larger. `get_generator(cheap=True)`.

On the local profile none of that applies and the pass is simply slow — roughly 25s per
chunk on CPU. Hence the on-disk cache below: contextualising this corpus takes about 45
minutes once, and zero thereafter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ragkit.providers.base import Generator
from ragkit.types import Chunk, Document

CONTEXT_SYSTEM = """You situate an excerpt within the document it came from.

Write one or two sentences saying what the excerpt is about and where it sits in the
document. Name the specific subject, so the sentences make sense on their own — resolve
pronouns and vague references like "the system" or "this approach" into what they
actually refer to.

Do not summarise the excerpt's argument, do not add information the document does not
contain, and do not use phrases like "this excerpt" or "this chunk". Write the situating
sentences only."""


def _cache_key(chunk: Chunk, document: Document, model: str) -> str:
    # Keyed on the document content as well as the chunk, so editing a document
    # invalidates the prefixes derived from it. A stale prefix describing an older
    # version of the document is worse than no prefix: it is confidently wrong.
    payload = f"{model}|{document.metadata.get('content_sha', '')}|{chunk.chunk_id}|{chunk.text}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


class ContextualiseCache:
    """A tiny on-disk cache of generated prefixes.

    Contextualising is expensive and perfectly deterministic at temperature 0, so
    repeating it is pure waste. This turns a 45-minute local pass into a one-off.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}  # a corrupt cache should be rebuilt, not fatal

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data, indent=0), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._data)


def contextualise(
    chunks: list[Chunk],
    documents: list[Document],
    generator: Generator,
    *,
    cache: ContextualiseCache | None = None,
    max_document_chars: int = 12_000,
    progress: bool = True,
) -> list[Chunk]:
    """Fill `Chunk.context` for every chunk. Mutates and returns the same list."""
    by_doc = {d.doc_id: d for d in documents}
    model = getattr(generator, "name", "unknown")
    done = 0

    for index, chunk in enumerate(chunks, start=1):
        document = by_doc.get(chunk.doc_id)
        if document is None:
            continue

        key = _cache_key(chunk, document, model)
        if cache is not None and (hit := cache.get(key)) is not None:
            chunk.context = hit
            continue

        # Truncating the document weakens the prefix but keeps a long document from
        # blowing the context window. Real systems should section the document and pass
        # the enclosing section instead of the whole thing.
        body = document.text[:max_document_chars]
        prompt = (
            f"<document title={document.title!r}>\n{body}\n</document>\n\n"
            f"<excerpt>\n{chunk.text}\n</excerpt>\n\n"
            "Situating sentences:"
        )

        try:
            # cache_system=True on the cloud path makes the document a cached prefix,
            # so it is billed in full once per document rather than once per chunk.
            text, _usage = generator.generate(  # type: ignore[call-arg]
                system=CONTEXT_SYSTEM, prompt=prompt, max_tokens=150, cache_system=True
            )
        except TypeError:
            # The local generator has no cache_control concept.
            text, _usage = generator.generate(system=CONTEXT_SYSTEM, prompt=prompt, max_tokens=150)

        chunk.context = " ".join(text.split())
        done += 1
        if cache is not None:
            cache.set(key, chunk.context)
            if done % 10 == 0:
                cache.save()  # checkpoint, so a long pass survives an interruption
        if progress and index % 10 == 0:
            print(f"  contextualised {index}/{len(chunks)} ({done} generated)")

    if cache is not None:
        cache.save()
    return chunks
