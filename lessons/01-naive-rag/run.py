"""Lesson 01 — the naive RAG pipeline, end to end.

    python lessons/01-naive-rag/run.py "What is reciprocal rank fusion?"

Generation needs Ollama running on the local profile. Pass --retrieval-only to watch
the retrieval half without it — which is the half worth watching anyway.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from ragkit.generate import build_prompt
from ragkit.pipeline import RagPipeline

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="What is reciprocal rank fusion?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip generation.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the assembled prompt.")
    args = parser.parse_args()

    pipeline = RagPipeline()
    if pipeline.count() == 0:
        console.print("[yellow]Index is empty — ingesting first.[/yellow]")
        console.print(pipeline.ingest())

    console.rule(f"[bold]{args.question}")

    # Step 1: retrieval. Look at this before anything else — nearly every bad RAG
    # answer is a bad retrieval wearing a fluent voice.
    results = pipeline.retrieve(args.question, top_k=args.top_k)
    for result in results:
        console.print(
            Panel(
                result.chunk.text.strip(),
                title=f"[{result.rank}] {result.chunk.title}",
                subtitle=f"cosine {result.score:.4f} · {result.chunk.chunk_id}",
                border_style="blue",
            )
        )

    if args.show_prompt:
        console.print(
            Panel(build_prompt(args.question, results), title="Prompt", border_style="magenta")
        )

    if args.retrieval_only:
        return

    # Step 2: generation over exactly those passages.
    answer = pipeline.ask(args.question, top_k=args.top_k)
    console.print(Panel(answer.text, title="Answer", border_style="green"))
    if answer.citations:
        console.print("Cited: " + ", ".join(c.doc_id for c in answer.citations))
    console.print(f"[dim]{answer.trace}[/dim]")


if __name__ == "__main__":
    main()
