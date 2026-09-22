"""Small-to-big retrieval: search precise chunks, return generous context.

The tension from lesson 02, restated: small chunks retrieve precisely because the
matching passage is not diluted, and answer poorly because the surrounding context is
gone. Large chunks do the reverse.

Small-to-big refuses the trade. Search over small chunks, then hand the generator the
larger region each hit came from. Precision where precision matters (matching), context
where context matters (answering).

This is where the `start_char` / `end_char` bookkeeping from lesson 02 earns its keep.
Because every chunk knows exactly where it sits in its source document, the parent
window is a slice — no second index, no parent/child table, no duplicated storage. Had
those offsets not been recorded at chunking time, this would need a whole extra ingest
path, which is why they were worth the fuss then.
"""

from __future__ import annotations

from ragkit.types import Chunk, Document, Scored


def _snap_to_boundary(text: str, position: int, *, forward: bool) -> int:
    """Move a slice point to the nearest paragraph or sentence break.

    A window cut at an arbitrary character starts mid-word and ends mid-clause, which
    is exactly the damage small-to-big is supposed to undo. Snapping costs nothing and
    keeps the expanded context readable.
    """
    if position <= 0:
        return 0
    if position >= len(text):
        return len(text)

    window = 200
    if forward:
        segment = text[position : position + window]
        for marker in ("\n\n", ". ", "\n"):
            found = segment.find(marker)
            if found != -1:
                return position + found + len(marker)
        return position
    segment = text[max(0, position - window) : position]
    for marker in ("\n\n", ". ", "\n"):
        found = segment.rfind(marker)
        if found != -1:
            return max(0, position - window) + found + len(marker)
    return position


class ParentDocumentRetriever:
    """Wraps a retriever and expands each hit into a surrounding window.

    `window` is how many characters of context to add on each side. Setting it larger
    than the document returns the whole document, which is the degenerate and sometimes
    correct case for a corpus of short documents.

    Deduplication matters more here than anywhere else in the pipeline. Two chunks 300
    characters apart expand into two nearly identical windows, and handing the generator
    the same text twice wastes context budget and pushes a genuinely different passage
    out of the top k. Overlapping windows are merged rather than both returned.
    """

    def __init__(self, retriever: object, documents: list[Document], window: int = 600) -> None:
        self._retriever = retriever
        self._documents = {d.doc_id: d for d in documents}
        self._window = window

    @property
    def name(self) -> str:
        return f"parent({getattr(self._retriever, 'name', '?')},w={self._window})"

    def _expand(self, chunk: Chunk) -> tuple[int, int, str] | None:
        document = self._documents.get(chunk.doc_id)
        if document is None:
            return None
        start = _snap_to_boundary(
            document.text, max(0, chunk.start_char - self._window), forward=False
        )
        end = _snap_to_boundary(
            document.text, min(len(document.text), chunk.end_char + self._window), forward=True
        )
        return start, end, document.text[start:end]

    def retrieve(self, query: str, top_k: int = 5, where: dict | None = None) -> list[Scored]:
        # Over-fetch, because merging overlapping windows collapses several hits into
        # one result and would otherwise leave fewer than top_k.
        hits = self._retriever.retrieve(query, top_k=top_k * 3, where=where)  # type: ignore[attr-defined]
        if not hits:
            return []

        merged: list[tuple[str, int, int, Scored]] = []
        for hit in hits:
            expanded = self._expand(hit.chunk)
            if expanded is None:
                continue
            start, end, text = expanded

            overlap = next(
                (
                    entry
                    for entry in merged
                    if entry[0] == hit.chunk.doc_id and start < entry[2] and end > entry[1]
                ),
                None,
            )
            if overlap is not None:
                # Widen the existing window instead of adding a near-duplicate. The
                # better-scoring hit already sits earlier in the list, so its score and
                # provenance are the ones kept.
                index = merged.index(overlap)
                doc_id, old_start, old_end, keep = overlap
                new_start, new_end = min(old_start, start), max(old_end, end)
                document = self._documents[doc_id]
                keep.chunk.text = document.text[new_start:new_end]
                keep.chunk.start_char, keep.chunk.end_char = new_start, new_end
                keep.scores["merged_windows"] = keep.scores.get("merged_windows", 1.0) + 1.0
                merged[index] = (doc_id, new_start, new_end, keep)
                continue

            parent = Chunk(
                chunk_id=f"{hit.chunk.chunk_id}+parent",
                doc_id=hit.chunk.doc_id,
                text=text,
                title=hit.chunk.title,
                context=hit.chunk.context,
                start_char=start,
                end_char=end,
                metadata=dict(hit.chunk.metadata),
            )
            scores = dict(hit.scores)
            scores["child_chars"] = float(len(hit.chunk.text))
            scores["parent_chars"] = float(len(text))
            merged.append(
                (hit.chunk.doc_id, start, end, Scored(chunk=parent, score=hit.score, scores=scores))
            )

        out = [entry[3] for entry in merged[:top_k]]
        for rank, scored in enumerate(out, start=1):
            scored.rank = rank
        return out
