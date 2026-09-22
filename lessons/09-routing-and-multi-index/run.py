"""Lesson 09 - see how the router classifies the golden set.

    python lessons/09-routing-and-multi-index/run.py

Prints the routing decision for every golden question, grouped by question kind, so you
can see whether the heuristic agrees with what you would have chosen.
"""

from __future__ import annotations

import argparse
from collections import Counter

from rich.console import Console
from rich.table import Table

from ragkit.eval.golden import load_golden
from ragkit.retrieve.router import looks_lexical

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=None, help="Classify one query instead.")
    args = parser.parse_args()

    if args.query:
        route = "lexical" if looks_lexical(args.query) else "semantic"
        console.print(f"[bold]{args.query}[/bold] -> [cyan]{route}[/cyan]")
        return

    questions = load_golden()
    table = Table(title="Routing decisions over the golden set")
    for column in ("kind", "route", "question"):
        table.add_column(column)

    counts: Counter[tuple[str, str]] = Counter()
    for question in sorted(questions, key=lambda q: q.kind):
        route = "lexical" if looks_lexical(question.question) else "semantic"
        counts[(question.kind, route)] += 1
        table.add_row(
            question.kind,
            f"[cyan]{route}[/cyan]" if route == "lexical" else f"[dim]{route}[/dim]",
            question.question[:66],
        )
    console.print(table)

    summary = Table(title="Summary")
    summary.add_column("kind")
    summary.add_column("lexical", justify="right")
    summary.add_column("semantic", justify="right")
    for kind in sorted({q.kind for q in questions}):
        summary.add_row(kind, str(counts[(kind, "lexical")]), str(counts[(kind, "semantic")]))
    console.print(summary)

    console.print(
        "\n[dim]The router is conservative about choosing lexical on purpose. The two\n"
        "mistakes are not equally bad: BM25 cannot bridge vocabulary mismatch at all,\n"
        "while dense retrieval on an identifier is weak but not blind. When unsure,\n"
        "take the failure that degrades gracefully.[/dim]"
    )


if __name__ == "__main__":
    main()
