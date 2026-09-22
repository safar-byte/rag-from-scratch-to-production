"""Splitting documents into retrievable chunks.

Chunking is the least glamorous part of a RAG system and routinely the
highest-leverage one: nothing downstream can recover information that chunking
destroyed. A fact split across two chunks so that neither one states it is simply gone,
no matter how good the reranker is.

Every chunker here records `start_char` / `end_char` into the source document. That
bookkeeping is what makes citations and parent-document retrieval (lesson 08) possible
later, and it is very annoying to retrofit.
"""

from __future__ import annotations

from typing import Protocol

from ragkit.types import Chunk, Document

# Tried in order: split on the most semantically meaningful boundary that works, and
# only fall through to cruder ones when a piece is still too large. The empty string is
# the final fallback — a hard character cut when nothing else splits the text.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


class Chunker(Protocol):
    @property
    def name(self) -> str: ...

    def split(self, document: Document) -> list[Chunk]: ...


def _is_heading(part: str) -> bool:
    """A Markdown ATX heading, or a short unpunctuated line acting as one."""
    stripped = part.strip()
    if not stripped or "\n" in stripped:
        return False
    return stripped.startswith("#")


def _merge_headings(parts: list[str], separator: str) -> list[str]:
    """Glue each heading onto the section it introduces.

    Without this, the greedy buffer below strands headings at the *end* of the previous
    chunk: a chunk finishes with "## Dimensionality" and the next one opens with that
    section's body. Both chunks are then worse. The body has lost the single strongest
    keyword it had — "Dimensionality" is exactly what someone would search for — and the
    preceding chunk has gained a label for content it does not contain, which is an
    invitation to retrieve the wrong passage.

    Headings are cheap to carry and highly discriminative, so they belong with their
    section. This is the least glamorous line of code in the repo and one of the more
    valuable ones.
    """
    if separator not in ("\n\n", "\n"):
        return parts

    merged: list[str] = []
    pending: list[str] = []
    for part in parts:
        if _is_heading(part):
            pending.append(part)
            continue
        merged.append(separator.join([*pending, part]) if pending else part)
        pending = []
    if pending:
        # Trailing headings with no body — keep them rather than silently drop them.
        merged.append(separator.join(pending))
    return merged


def _tail_words(text: str, max_chars: int) -> str:
    """The last `max_chars` of `text`, snapped forward to a word boundary.

    Slicing the tail by raw character count leaves overlaps starting mid-word — an
    overlap that begins "rly always retrieval depth" instead of "nearly always". That
    is a genuine retrieval cost, not just an aesthetic one: the fragment embeds as
    noise and BM25 tokenises it into a term that appears nowhere in any query.
    """
    if max_chars <= 0 or not text:
        return ""
    tail = text[-max_chars:]
    if len(text) > max_chars:
        # Drop a leading partial word, unless that would consume the whole tail.
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1 :]
    return tail.strip()


def _make_chunk(document: Document, text: str, start: int, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id}::{index:04d}",
        doc_id=document.doc_id,
        text=text,
        title=document.title,
        start_char=start,
        end_char=start + len(text),
        metadata=dict(document.metadata),
    )


class FixedSizeChunker:
    """Cuts every `chunk_size` characters, with `overlap` characters repeated.

    The simplest thing that works, and the baseline the others are measured against.
    It splits words and sentences in half, which is exactly the point: lesson 02 asks
    you to measure how much that actually costs before assuming it matters.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def name(self) -> str:
        return f"fixed({self.chunk_size}/{self.overlap})"

    def split(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.overlap
        for index, start in enumerate(range(0, len(text), step)):
            piece = text[start : start + self.chunk_size].strip()
            if piece:
                chunks.append(_make_chunk(document, piece, start, index))
            if start + self.chunk_size >= len(text):
                break
        return chunks


class RecursiveChunker:
    """Splits on the most meaningful separator that keeps pieces under the limit.

    The sensible default. It walks `separators` in order, splitting on the first one
    that lets it build pieces within `chunk_size`, and recurses into anything still too
    large. Paragraph boundaries survive where they can, and sentences are only broken
    when a single sentence genuinely exceeds the limit.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: list[str] | None = None,
    ) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or DEFAULT_SEPARATORS

    @property
    def name(self) -> str:
        return f"recursive({self.chunk_size}/{self.overlap})"

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = separators[0] if separators else ""
        remaining = separators[1:] if len(separators) > 1 else []

        # Last resort: no separator left, so cut on character count.
        if separator == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
                if text[i : i + self.chunk_size].strip()
            ]

        parts = _merge_headings(text.split(separator), separator)
        pieces: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{separator}{part}" if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue
            if buffer:
                pieces.append(buffer)
            # A single part over the limit has to be broken by a finer separator.
            if len(part) > self.chunk_size:
                pieces.extend(self._split_text(part, remaining))
                buffer = ""
            else:
                buffer = part
        if buffer:
            pieces.append(buffer)
        return [p for p in pieces if p.strip()]

    def split(self, document: Document) -> list[Chunk]:
        pieces = self._split_text(document.text, self.separators)
        if not pieces:
            return []

        chunks: list[Chunk] = []
        cursor = 0
        for index, piece in enumerate(pieces):
            stripped = piece.strip()
            # Locate the piece in the original so offsets stay true after stripping.
            found = document.text.find(stripped, cursor)
            start = found if found != -1 else cursor

            text = stripped
            if self.overlap and index > 0:
                # Prepend the tail of the previous piece so a fact on the boundary
                # survives in at least one chunk.
                tail = _tail_words(pieces[index - 1].strip(), self.overlap)
                if tail:
                    text = f"{tail} {stripped}".strip()
                    start = max(0, start - len(tail) - 1)

            chunks.append(_make_chunk(document, text, start, index))
            cursor = max(cursor, found + len(stripped)) if found != -1 else cursor
        return chunks


def chunk_documents(documents: list[Document], chunker: Chunker) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.split(document))
    return chunks
