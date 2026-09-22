"""Loading source documents off disk."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

from ragkit.config import DATA_DIR
from ragkit.types import Document

TEXT_SUFFIXES = {".md", ".txt", ".rst"}


def _title_from(path: Path, text: str) -> str:
    """Prefer the document's own H1 over its filename."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return path.stem.replace("-", " ").replace("_", " ")


def load_documents(source: Path | None = None) -> list[Document]:
    """Load every text document under `source` (defaults to `data/`).

    `doc_id` is the path relative to the source root, not a hash or a UUID. That makes
    an index diffable and a citation readable — "04-reranking.md" tells you something,
    "a3f9c1" does not.
    """
    root = source or DATA_DIR
    if not root.exists():
        raise FileNotFoundError(f"No corpus at {root}")

    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        # The corpus README describes the corpus; it is not part of it.
        if path.name.lower() == "readme.md":
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            Document(
                doc_id=path.relative_to(root).as_posix(),
                text=text,
                title=_title_from(path, text),
                metadata={
                    "source": str(path),
                    "suffix": path.suffix.lower(),
                    # Content hash, so a re-ingest can tell what actually changed.
                    "content_sha": hashlib.sha256(text.encode()).hexdigest()[:16],
                },
            )
        )

    if not documents:
        raise FileNotFoundError(f"No loadable documents under {root}")
    return documents


def iter_documents(source: Path | None = None) -> Iterable[Document]:
    yield from load_documents(source)
