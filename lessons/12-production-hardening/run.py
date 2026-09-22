"""Lesson 12 - the production surface, checked.

    python lessons/12-production-hardening/run.py            # run the checks
    python lessons/12-production-hardening/run.py --serve    # how to start the service

Verifies the things that fail silently: the abstention gate, cost accounting, and (on
the cloud profile) that prompt caching is actually being hit.
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from ragkit.config import Profile, get_settings
from ragkit.generate import DEFAULT_SCORE_FLOOR, abstain
from ragkit.pipeline import RagPipeline

console = Console()

SERVE = """
Terminal 1:  uvicorn ragkit.api.main:app --port 8000
Terminal 2:  cd ui && npm install && npm run dev
Then open:   http://localhost:5173

Use "Retrieve only" while debugging - it skips the model, so it returns in about a
second once the embedder is warm, instead of ~30s for a full answer.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Print how to run the service.")
    args = parser.parse_args()

    if args.serve:
        console.print(SERVE)
        return

    settings = get_settings()
    pipeline = RagPipeline(settings, strategy="rerank", score_floor=DEFAULT_SCORE_FLOOR)

    table = Table(title="Abstention gate")
    for column in ("question", "best dense", "decision"):
        table.add_column(column)

    probes = [
        ("What does exit status 75 mean?", "answerable"),
        ("What is the default value of RETRIEVAL_FANOUT?", "answerable"),
        ("What is the capital of France?", "unanswerable"),
        ("What is the population of Tokyo?", "unanswerable"),
        ("What is the default value of SHARD_REPLICATION_FACTOR?", "unanswerable, hard"),
    ]
    for question, label in probes:
        results = pipeline.retrieve(question, top_k=5)
        decision = abstain.decide(results, floor=DEFAULT_SCORE_FLOOR)
        verdict = "[yellow]refuse[/yellow]" if decision.abstain else "[green]answer[/green]"
        table.add_row(f"{question[:46]}\n[dim]{label}[/dim]", f"{decision.best_score:.4f}", verdict)
    console.print(table)

    console.print(
        "\n[dim]The last probe is the one to look at. It is unanswerable and it scores\n"
        "ABOVE the floor, because a question about a setting that does not exist looks\n"
        "exactly like one about a setting that does. No threshold catches it - that is\n"
        "lesson 10's grader, which reads the passages instead of scoring them.[/dim]"
    )

    if settings.profile is Profile.CLOUD:
        console.print("\n[bold]Prompt cache check[/bold]")
        results = pipeline.retrieve("What is reciprocal rank fusion?", top_k=3)
        for attempt in (1, 2):
            answer = pipeline.ask("What is reciprocal rank fusion?", top_k=3)
            console.print(
                f"  call {attempt}: cache_read={answer.usage.cache_read_tokens} "
                f"cost=${answer.usage.cost_usd:.5f}"
            )
        _ = results
        console.print(
            "[dim]Call 2 must show cache_read > 0. A zero means something in the prefix\n"
            "varies between requests and you are paying full price silently.[/dim]"
        )
    else:
        console.print(
            "\n[dim]Prompt caching and the Citations API are cloud-profile features.\n"
            "Set RAG_PROFILE=cloud with keys to exercise them.[/dim]"
        )


if __name__ == "__main__":
    main()
