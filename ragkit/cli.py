"""Command line interface.

    ragkit ingest              # build the index
    ragkit ask "question"      # retrieve and answer
    ragkit search "question"   # retrieve only — no model call
    ragkit status              # what is indexed, and under which profile

`search` exists because it is the debugging tool you actually reach for. Nearly every
bad RAG answer is a retrieval failure, and looking at the chunks without waiting for
generation is the fastest way to confirm it.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ragkit.config import get_settings
from ragkit.pipeline import RagPipeline

app = typer.Typer(add_completion=False, help="A production RAG toolkit, built lesson by lesson.")
console = Console()


@app.command()
def ingest(
    chunk_size: int = typer.Option(None, help="Override the configured chunk size."),
    overlap: int = typer.Option(None, help="Override the configured chunk overlap."),
) -> None:
    """Load, chunk, embed and index the corpus."""
    settings = get_settings()
    if chunk_size is not None:
        settings.chunk_size = chunk_size
    if overlap is not None:
        settings.chunk_overlap = overlap

    console.print(f"[dim]profile: {settings.profile.value}[/dim]")
    with console.status("Ingesting..."):
        stats = RagPipeline(settings).ingest()

    table = Table(show_header=False, box=None)
    for key, value in stats.items():
        table.add_row(f"[dim]{key}[/dim]", str(value))
    console.print(Panel(table, title="Ingested", border_style="green"))


@app.command()
def search(
    question: str = typer.Argument(..., help="What to search for."),
    top_k: int = typer.Option(5, "--top-k", "-k"),
) -> None:
    """Retrieve chunks without generating an answer."""
    results = RagPipeline().retrieve(question, top_k=top_k)
    if not results:
        console.print("[yellow]Nothing retrieved. Has the corpus been ingested?[/yellow]")
        raise typer.Exit(1)

    for result in results:
        chunk = result.chunk
        scores = "  ".join(f"{k}={v:.4f}" for k, v in result.scores.items())
        console.print(
            Panel(
                chunk.text.strip(),
                title=f"[bold]{result.rank}. {chunk.title}[/bold] [dim]{chunk.chunk_id}[/dim]",
                subtitle=f"[dim]{scores}[/dim]",
                border_style="blue",
            )
        )


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    show_context: bool = typer.Option(False, "--context", help="Also print the passages used."),
) -> None:
    """Retrieve context and generate a grounded answer."""
    with console.status("Thinking..."):
        answer = RagPipeline().ask(question, top_k=top_k)

    console.print(Panel(answer.text, title="Answer", border_style="green"))

    if answer.citations:
        console.print("\n[bold]Cited[/bold]")
        for citation in answer.citations:
            console.print(f"  [dim]-[/dim] {citation.title} [dim]({citation.doc_id})[/dim]")

    if show_context:
        for index, result in enumerate(answer.contexts, start=1):
            console.print(
                Panel(
                    result.chunk.text.strip(),
                    title=f"[{index}] {result.chunk.chunk_id}",
                    border_style="blue",
                )
            )

    trace = answer.trace
    usage = answer.usage
    console.print(
        f"\n[dim]{trace['retriever']} · {trace['generator']} · "
        f"retrieval {trace['retrieval_ms']}ms · total {trace['total_ms']}ms · "
        f"{usage.input_tokens}+{usage.output_tokens} tokens"
        + (f" · ${usage.cost_usd:.5f}" if usage.cost_usd else "")
        + "[/dim]"
    )


@app.command()
def status() -> None:
    """Show the active profile and what is currently indexed."""
    settings = get_settings()
    pipeline = RagPipeline(settings)

    table = Table(show_header=False, box=None)
    table.add_row("[dim]profile[/dim]", settings.profile.value)
    table.add_row("[dim]store[/dim]", str(settings.chroma_path))
    table.add_row("[dim]chunks indexed[/dim]", str(pipeline.count()))
    table.add_row(
        "[dim]chunk size / overlap[/dim]", f"{settings.chunk_size} / {settings.chunk_overlap}"
    )
    if settings.profile.value == "local":
        table.add_row("[dim]embedding model[/dim]", settings.local_embedding_model)
        table.add_row("[dim]generation model[/dim]", settings.local_generation_model)
    else:
        table.add_row("[dim]embedding model[/dim]", settings.cloud_embedding_model)
        table.add_row("[dim]generation model[/dim]", settings.cloud_generation_model)
    console.print(Panel(table, title="ragkit", border_style="cyan"))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
