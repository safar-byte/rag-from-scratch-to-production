"""Lesson 10 - CRAG grading and multi-hop retrieval, with the decisions shown.

    python lessons/10-agentic-rag/run.py "What is the default value of SHARD_REPLICATION_FACTOR?"
    python lessons/10-agentic-rag/run.py --multihop "Which method fixes error-code lookup?"

The point is to watch the decisions, not just the answer: which verdict the grader
returned, whether it widened, and what follow-up queries the multi-hop loop wrote.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from ragkit.agent import ContextGrader, CorrectiveRag, MultiHopRetriever
from ragkit.config import get_settings
from ragkit.pipeline import RagPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "question", nargs="?", default="What is the default value of SHARD_REPLICATION_FACTOR?"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--multihop", action="store_true", help="Run the multi-hop retriever.")
    parser.add_argument("--hops", type=int, default=2)
    args = parser.parse_args()

    settings = get_settings()
    pipeline = RagPipeline(settings, strategy="rerank")
    console.rule(f"[bold]{args.question}")

    if args.multihop:
        retriever = MultiHopRetriever(pipeline.retriever, pipeline.generator, max_hops=args.hops)
        results = retriever.retrieve(args.question, top_k=args.top_k)
        trace = retriever.last_trace

        console.print("[bold]Hops[/bold]")
        for step in trace.hops:
            console.print(f"  [dim]{step}[/dim]")
        console.print(f"  [dim]stopped: {trace.stopped_because}[/dim]\n")
        console.print("[bold]Queries written by the model[/bold]")
        for index, query in enumerate(trace.queries):
            console.print(f"  {index}. {query}")
        console.print("\n[bold]Accumulated results[/bold]")
        for item in results:
            console.print(f"  {item.rank}. {item.chunk.doc_id:30s} {item.score:+.3f}")
        return

    # CRAG. The grader runs before generation, so an "incorrect" verdict means the
    # model is never shown context it cannot answer from.
    crag = CorrectiveRag(
        pipeline.retriever,
        pipeline.generator,
        ContextGrader(pipeline.generator),
    )
    answer = crag.ask(args.question, top_k=args.top_k)

    verdict = answer.trace.get("crag_verdict")
    colour = {"correct": "green", "ambiguous": "yellow", "incorrect": "red"}.get(
        str(verdict), "white"
    )
    console.print(
        Panel(
            answer.text,
            title=f"[{colour}]verdict: {verdict}[/{colour}]",
            subtitle=f"[dim]{answer.trace.get('crag_reason', '')}[/dim]",
            border_style=colour,
        )
    )
    console.print(f"[dim]steps: {' -> '.join(answer.trace.get('crag_steps', []))}[/dim]")
    console.print(
        f"[dim]widened: {answer.trace.get('crag_widened')} · {answer.trace.get('total_ms')}ms[/dim]"
    )

    console.print(
        "\n[dim]Try this against a question the corpus cannot answer. A score threshold\n"
        "cannot catch those - they are topically identical to real questions - but the\n"
        "grader reads the passages, which is the only thing that can.[/dim]"
    )


if __name__ == "__main__":
    main()
