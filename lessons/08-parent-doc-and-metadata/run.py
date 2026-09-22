"""Lesson 08 - see the child chunk and the parent window side by side.

python lessons/08-parent-doc-and-metadata/run.py "how does chunk overlap work"
python lessons/08-parent-doc-and-metadata/run.py --filter 07-operations.md
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from ragkit.config import get_settings
from ragkit.pipeline import RagPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="How does chunk overlap work?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--window", type=int, default=600)
    parser.add_argument("--filter", default=None, help="Restrict retrieval to one doc_id.")
    args = parser.parse_args()

    settings = get_settings()
    console.rule(f"[bold]{args.question}")

    plain = RagPipeline(settings, strategy="rerank")
    if args.filter:
        # A pre-filter: the predicate is applied before the similarity search, so the
        # top k is computed over eligible documents only. Post-filtering the global top
        # k would return almost nothing for a selective predicate, silently.
        results = plain.retriever.retrieve(
            args.question, top_k=args.top_k, where={"doc_id": args.filter}
        )
        console.print(f"[dim]pre-filtered to doc_id={args.filter}[/dim]")
        for item in results:
            console.print(f"  {item.rank}. {item.chunk.doc_id}  {item.chunk.text[:90].strip()}...")
        return

    expanded = RagPipeline(settings, strategy="parent", parent_window=args.window)
    # Reuse the loaded models rather than paying the ~40s load twice.
    expanded._embedder = plain.embedder
    expanded._reranker = plain.reranker

    for child, parent in zip(
        plain.retrieve(args.question, top_k=args.top_k),
        expanded.retrieve(args.question, top_k=args.top_k),
        strict=False,
    ):
        console.print(
            Panel(
                child.chunk.text.strip(),
                title=f"child - {len(child.chunk.text)} chars",
                border_style="blue",
            )
        )
        merged = parent.scores.get("merged_windows")
        console.print(
            Panel(
                parent.chunk.text.strip(),
                title=(
                    f"parent - {len(parent.chunk.text)} chars"
                    + (f" (merged {int(merged)} windows)" if merged else "")
                ),
                border_style="green",
            )
        )
        console.print()

    console.print(
        "[dim]The parent is what reaches the generator. Retrieval still matched on the\n"
        "small chunk, so precision is unchanged - only the context around it grew.\n"
        "Watch groundedness, not recall: expansion should not change WHICH documents\n"
        "are found, only how much of each one the model gets to read.[/dim]"
    )


if __name__ == "__main__":
    main()
