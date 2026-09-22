"""Lesson 05 - reranking, and a sweep of the shortlist depth.

    python lessons/05-reranking/run.py "why do I get charged twice"
    python lessons/05-reranking/run.py --sweep

The sweep is the point. `rerank_candidates` is the dial that sets the ceiling on final
quality and the linear driver of reranking cost, and 25 is a convention, not a finding.
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from ragkit.config import get_settings
from ragkit.eval.run import run_eval
from ragkit.pipeline import RagPipeline
from ragkit.providers import get_reranker
from ragkit.retrieve import Bm25Retriever, DenseRetriever, HybridRetriever
from ragkit.retrieve.rerank import RerankingRetriever

console = Console()


def sweep(depths: list[int]) -> None:
    settings = get_settings()
    pipeline = RagPipeline(settings)
    base = HybridRetriever(
        [DenseRetriever(pipeline._store, pipeline.embedder), Bm25Retriever(pipeline._bm25)],
        candidates=max(depths),
    )
    reranker = get_reranker(settings)

    table = Table(title="Shortlist depth sweep (hybrid -> cross-encoder)")
    for column in ("depth", "R@1", "R@3", "R@5", "nDCG@5", "vocab R@5", "ms/query"):
        table.add_column(column, justify="right")

    for depth in depths:
        pipeline._retriever = RerankingRetriever(base, reranker, candidates=depth)
        started = time.perf_counter()
        # deep_k=5 disables the eval's rank diagnostic, which would otherwise force
        # a shortlist of at least 20 and make every depth below that identical.
        summary = run_eval(
            settings, pipeline=pipeline, retrieval_only=True, quiet=True, top_k=5, deep_k=5
        )
        elapsed = (time.perf_counter() - started) * 1000 / len(summary["results"])

        overall = summary["retrieval"]
        vocab = summary["by_kind"].get("vocab_mismatch", {})
        table.add_row(
            str(depth),
            f"{overall['recall@1']:.3f}",
            f"{overall['recall@3']:.3f}",
            f"{overall['recall@5']:.3f}",
            f"{overall['ndcg@5']:.3f}",
            f"{vocab.get('recall@5', float('nan')):.3f}",
            f"{elapsed:.0f}",
        )

    console.print(table)
    console.print(
        "\n[dim]Find where quality stops climbing. Past that point you are paying\n"
        "linearly for nothing - and a hosted reranker bills per document scored.[/dim]"
    )


def single(question: str, top_k: int, depth: int) -> None:
    settings = get_settings()
    pipeline = RagPipeline(settings, strategy="rerank")
    if pipeline.count() == 0:
        console.print("[yellow]Index is empty - ingesting first.[/yellow]")
        pipeline.ingest()

    console.rule(f"[bold]{question}")
    for item in pipeline.retrieve(question, top_k=top_k):
        before = item.scores.get("rank_before_rerank")
        # The rank delta is the only honest measure of what the reranker contributed.
        if before is None:
            movement = ""
        elif before > item.rank:
            movement = f"[green]up {int(before)} -> {item.rank}[/green]"
        elif before < item.rank:
            movement = f"[red]down {int(before)} -> {item.rank}[/red]"
        else:
            movement = "[dim]unmoved[/dim]"
        console.print(f"  {item.rank}. {item.chunk.doc_id:34s} ce={item.score:+.3f}  {movement}")
        console.print(f"     [dim]{item.chunk.text[:100].strip()}...[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="Why do I get charged twice?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=25)
    parser.add_argument("--sweep", action="store_true", help="Sweep the shortlist depth.")
    args = parser.parse_args()

    if args.sweep:
        sweep([3, 5, 10, 25, 50])
    else:
        single(args.question, args.top_k, args.candidates)


if __name__ == "__main__":
    main()
