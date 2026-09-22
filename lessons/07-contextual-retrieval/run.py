"""Lesson 07 - generate situating prefixes and index them.

    python lessons/07-contextual-retrieval/run.py --preview        # a few examples, cheap
    python lessons/07-contextual-retrieval/run.py --build          # the whole corpus

The full pass is ~25s per chunk locally, so `--preview` exists to let you see what the
prefixes look like before committing 45 minutes. Results are cached on disk, keyed on
document content, so a second run costs nothing.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from ragkit.config import REPO_ROOT, get_settings
from ragkit.ingest import RecursiveChunker, chunk_documents, load_documents
from ragkit.ingest.contextual import ContextualiseCache, contextualise
from ragkit.pipeline import RagPipeline
from ragkit.providers import get_generator

console = Console()
CACHE_PATH = REPO_ROOT / ".cache" / "contextual.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Contextualise a few chunks only.")
    parser.add_argument("--build", action="store_true", help="Contextualise and re-index all.")
    parser.add_argument("--limit", type=int, default=4, help="How many chunks to preview.")
    args = parser.parse_args()

    settings = get_settings()
    documents = load_documents()
    chunks = chunk_documents(
        documents, RecursiveChunker(settings.chunk_size, settings.chunk_overlap)
    )
    cache = ContextualiseCache(CACHE_PATH)
    console.print(f"[dim]{len(chunks)} chunks, {len(cache)} already cached[/dim]\n")

    # The cheap model on purpose: situating a chunk is an easy task, and this is the
    # pass that runs once per chunk in the whole corpus.
    generator = get_generator(settings, cheap=True)

    if args.preview or not args.build:
        # Pick chunks that do NOT start at a document boundary — those are the ones
        # most likely to be missing their context, which is the whole point.
        candidates = [c for c in chunks if c.start_char > 0][: args.limit]
        contextualise(candidates, documents, generator, cache=cache, progress=False)
        for chunk in candidates:
            console.print(
                Panel(
                    f"[dim italic]{chunk.context}[/dim italic]\n\n{chunk.text[:320]}",
                    title=f"{chunk.chunk_id}",
                    border_style="cyan",
                )
            )
        console.print(
            "\n[dim]The italic line is generated and gets embedded with the chunk.\n"
            "The chunk text below it is untouched, so citations still point at the\n"
            "source rather than at model output.[/dim]"
        )
        if not args.build:
            return

    console.print("[yellow]Contextualising the full corpus. This is slow offline.[/yellow]")
    contextualise(chunks, documents, generator, cache=cache)

    pipeline = RagPipeline(settings)
    pipeline._store.reset()
    vectors = pipeline.embedder.embed_documents([c.embed_text for c in chunks])
    pipeline._store.add(chunks, vectors)
    pipeline._bm25.build(chunks)  # the prefix helps lexical search too
    console.print(f"[green]Re-indexed {len(chunks)} contextualised chunks.[/green]")
    console.print(
        "[dim]Now run: python -m ragkit.eval.run --retrieval-only --strategy rerank[/dim]"
    )


if __name__ == "__main__":
    main()
