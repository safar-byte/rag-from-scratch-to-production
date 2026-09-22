"""Lesson 04 - dense, BM25 and hybrid on the same query, side by side.

    python lessons/04-hybrid-search-rrf/run.py "exit status 75"
    python lessons/04-hybrid-search-rrf/run.py "why do I get charged twice"

Seeing the three rankings together is the point. The interesting queries are the ones
where the two retrievers disagree sharply, because that is where fusion either earns its
place or does damage.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from ragkit.pipeline import RagPipeline
from ragkit.retrieve import Bm25Retriever, DenseRetriever, HybridRetriever

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What does exit status 75 mean?")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = RagPipeline()
    if pipeline.count() == 0:
        console.print("[yellow]Index is empty - ingesting first.[/yellow]")
        pipeline.ingest()

    dense = DenseRetriever(pipeline._store, pipeline.embedder)
    lexical = Bm25Retriever(pipeline._bm25)
    hybrid = HybridRetriever([dense, lexical], candidates=25)

    console.rule(f"[bold]{args.question}")

    results = {
        "dense": dense.retrieve(args.question, top_k=args.top_k),
        "bm25": lexical.retrieve(args.question, top_k=args.top_k),
        "hybrid": hybrid.retrieve(args.question, top_k=args.top_k),
    }

    table = Table(title="Top results by retriever")
    table.add_column("#", justify="right")
    for name in results:
        table.add_column(name)

    for row in range(args.top_k):
        cells = []
        for name, scored in results.items():
            if row >= len(scored):
                cells.append("")
                continue
            item = scored[row]
            key = "dense" if name == "dense" else ("bm25" if name == "bm25" else "rrf")
            cells.append(f"{item.chunk.doc_id}\n[dim]{key}={item.scores.get(key, 0):.4f}[/dim]")
        table.add_row(str(row + 1), *cells)

    console.print(table)

    # Where the two retrievers disagree is where fusion does something. Agreement means
    # fusion had nothing to decide.
    dense_ids = {s.chunk.chunk_id for s in results["dense"]}
    bm25_ids = {s.chunk.chunk_id for s in results["bm25"]}
    overlap = len(dense_ids & bm25_ids)
    console.print(
        f"\n[dim]Overlap between dense and bm25 top-{args.top_k}: {overlap}/{args.top_k}. "
        "Low overlap means fusion is doing real work; high overlap means it has little "
        "to add.[/dim]"
    )

    console.print("\n[bold]Fused, with per-retriever ranks[/bold]")
    for item in results["hybrid"]:
        parts = [
            f"{key}={value:.0f}" for key, value in item.scores.items() if key.endswith("_rank")
        ]
        console.print(
            f"  {item.rank}. {item.chunk.doc_id:34s} rrf={item.score:.5f}  "
            f"[dim]{' '.join(parts)}[/dim]"
        )


if __name__ == "__main__":
    main()
