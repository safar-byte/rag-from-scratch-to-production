"""Lesson 11 - build the entity graph and look at how little there is.

    python lessons/11-graphrag/run.py                          # stats and top entities
    python lessons/11-graphrag/run.py "How does BM25 relate to HNSW?"

No model needed: entity extraction defaults to the heuristic backend.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from ragkit.config import get_settings
from ragkit.graph import GraphRetriever, build_graph, entity_frequencies
from ragkit.ingest import RecursiveChunker, chunk_documents, load_documents

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--hops", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    chunks = chunk_documents(
        load_documents(), RecursiveChunker(settings.chunk_size, settings.chunk_overlap)
    )
    graph = build_graph(chunks)
    stats = graph.stats()

    table = Table(title="Entity graph", show_header=False, box=None)
    for key, value in stats.items():
        table.add_row(f"[dim]{key}[/dim]", str(value))
    console.print(table)

    if stats["mean_degree"] < 3:
        console.print(
            "\n[yellow]Mean degree below 3: there is not enough structure here to\n"
            "traverse.[/yellow] [dim]This corpus is a set of independent explanatory\n"
            "essays, so entities barely co-occur across documents. That is a property of\n"
            "the corpus, not a parameter to tune - GraphRAG needs entities that recur\n"
            "across documents in varying combinations.[/dim]\n"
        )

    frequencies = entity_frequencies(chunks)
    console.print("[bold]Most frequent entities[/bold]")
    for entity, count in frequencies.most_common(12):
        neighbours = graph.neighbours(entity, limit=3)
        linked = (
            ", ".join(f"{graph.display.get(n, n)}({w})" for n, w in neighbours) or "[dim]none[/dim]"
        )
        console.print(f"  {graph.display.get(entity, entity):32s} x{count:<3d} -> {linked}")

    if not args.question:
        console.print(
            "\n[dim]Look at the arrow column. Most entities connect to nothing, which is\n"
            "the finding: there is no graph to walk. Pass a question to try anyway.[/dim]"
        )
        return

    retriever = GraphRetriever(graph, hops=args.hops)
    seeds = retriever._seed_entities(args.question)
    console.print(f"\n[bold]{args.question}[/bold]")
    console.print(f"[dim]seed entities: {seeds or 'none found - nothing to walk from'}[/dim]")
    for item in retriever.retrieve(args.question, top_k=args.top_k):
        console.print(f"  {item.rank}. {item.chunk.doc_id:30s} graph={item.score:.2f}")
        console.print(f"     [dim]{item.chunk.text[:88].strip()}...[/dim]")


if __name__ == "__main__":
    main()
