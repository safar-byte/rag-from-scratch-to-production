"""Lesson 06 - see what each transformation does to a query, and what it costs.

    python lessons/06-query-transformation/run.py "why do I get charged twice"

Prints the variants each technique produces and times the transform separately from
the retrieval, because the whole question in this lesson is whether the extra model
call earns its place.
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table

from ragkit.config import get_settings
from ragkit.pipeline import RagPipeline
from ragkit.query import TRANSFORMERS

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="Why do I get charged twice?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    pipeline = RagPipeline(settings, strategy="rerank")
    console.rule(f"[bold]{args.question}")

    table = Table(title="What each transformation produces")
    for column in ("technique", "transform ms", "variants"):
        table.add_column(column)

    for name, cls in TRANSFORMERS.items():
        transformer = cls() if name == "identity" else cls(pipeline.generator)
        started = time.perf_counter()
        variants = transformer.transform(args.question)
        elapsed = (time.perf_counter() - started) * 1000

        rendered = "\n".join(
            f"[dim]{i}.[/dim] {v[:78]}{'...' if len(v) > 78 else ''}"
            for i, v in enumerate(variants)
        )
        table.add_row(name, f"{elapsed:.0f}", rendered)

    console.print(table)
    console.print(
        "\n[dim]Variant 0 is always the original query - every technique here keeps it,\n"
        "because a transformation that drops the one rare term in the question is a\n"
        "common and silent failure.\n\n"
        "Read the transform-ms column against the ~1s that retrieval itself takes. That\n"
        "ratio, not the cleverness of the variants, decides whether any of this ships.[/dim]"
    )


if __name__ == "__main__":
    main()
