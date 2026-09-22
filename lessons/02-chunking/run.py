"""Lesson 02 — compare chunking strategies on the real corpus.

    python lessons/02-chunking/run.py

Prints the chunk-size distribution for several configurations and shows what each one
retrieves for the same query. No models needed for the statistics; retrieval uses
whatever profile is configured.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from ragkit.ingest import FixedSizeChunker, RecursiveChunker, chunk_documents, load_documents

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    documents = load_documents()
    console.print(
        f"Corpus: {len(documents)} documents, {sum(len(d.text) for d in documents)} chars\n"
    )

    chunkers = [
        FixedSizeChunker(256, args.overlap),
        FixedSizeChunker(512, args.overlap),
        RecursiveChunker(256, args.overlap),
        RecursiveChunker(512, args.overlap),
        RecursiveChunker(1024, args.overlap),
    ]

    table = Table(title="Chunking strategies")
    for column in ("strategy", "chunks", "min", "mean", "max", "split mid-sentence"):
        table.add_column(column)

    for chunker in chunkers:
        chunks = chunk_documents(documents, chunker)
        sizes = [len(c.text) for c in chunks]
        # A crude proxy for damage: chunks that do not end on terminal punctuation
        # were almost certainly cut through the middle of a sentence.
        broken = sum(1 for c in chunks if not c.text.rstrip().endswith((".", "!", "?", ":")))
        table.add_row(
            chunker.name,
            str(len(chunks)),
            str(min(sizes)),
            str(sum(sizes) // len(sizes)),
            str(max(sizes)),
            f"{broken} ({broken * 100 // len(chunks)}%)",
        )

    console.print(table)
    console.print(
        "\n[dim]More chunks means more precise retrieval and more fragmented context.\n"
        "Which column actually predicts answer quality? You cannot tell from this\n"
        "table at all - that is what lesson 03 is for.[/dim]"
    )


if __name__ == "__main__":
    main()
