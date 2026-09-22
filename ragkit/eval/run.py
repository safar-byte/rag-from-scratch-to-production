"""The evaluation runner.

    python -m ragkit.eval.run --label "naive dense, 512/64"
    python -m ragkit.eval.run --retrieval-only   # no generation, no judge

Reports retrieval and generation metrics separately, because that separation is the
whole diagnostic value of the harness. One end-to-end score tells you something is
wrong; two halves tell you where to work.

It also checks for a ceiling. If recall is already near 1.0 at every k, the corpus is
too small to distinguish techniques and every comparison below it is theatre. The
harness says so rather than letting you tune against noise.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from ragkit.config import Settings, get_settings
from ragkit.eval.golden import GoldenQuestion, load_golden, validate_golden
from ragkit.eval.judge import DeterministicJudge, Grade, LlmJudge
from ragkit.eval.metrics import (
    ndcg_at_k,
    precision_at_k,
    rank_of_first_relevant,
    recall_at_k,
    reciprocal_rank,
)
from ragkit.pipeline import RagPipeline

console = Console()

# Report at several depths. k=1 and k=3 discriminate far better than k=5 on a small
# corpus, and reporting only k=5 is how a ceiling effect goes unnoticed.
K_VALUES = (1, 3, 5)
CEILING_THRESHOLD = 0.95


@dataclass(slots=True)
class QuestionResult:
    id: str
    kind: str
    question: str
    retrieved: list[str]
    first_relevant_rank: int | None
    n_relevant: int = 0
    retrieval: dict[str, float] = field(default_factory=dict)
    answer: str = ""
    grade: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0


def evaluate_retrieval(question: GoldenQuestion, retrieved_docs: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for k in K_VALUES:
        scores[f"recall@{k}"] = recall_at_k(retrieved_docs, question.relevant, k)
        scores[f"precision@{k}"] = precision_at_k(retrieved_docs, question.relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(retrieved_docs, question.relevant, k)
    scores["mrr"] = reciprocal_rank(retrieved_docs, question.relevant)
    return scores


def dedupe_to_documents(chunk_doc_ids: list[str]) -> list[str]:
    """Collapse a chunk ranking into a document ranking, keeping each doc's best rank.

    Several chunks of one document routinely occupy the top of a ranking. Counting that
    document once, at its best position, is what makes document-level metrics mean
    anything.
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in chunk_doc_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def run_eval(
    settings: Settings | None = None,
    *,
    pipeline: RagPipeline | None = None,
    top_k: int = 5,
    retrieval_only: bool = False,
    use_llm_judge: bool = True,
    limit: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    settings = settings or get_settings()
    pipeline = pipeline or RagPipeline(settings)

    questions = load_golden()
    problems = validate_golden(questions)
    if problems:
        # A golden set naming a document that no longer exists creates a question that
        # can never be answered, which shows up as a permanent unexplained dent in
        # recall. Refuse to run rather than report a number built on it.
        raise ValueError("Golden set is invalid:\n  " + "\n  ".join(problems))
    if limit:
        questions = questions[:limit]

    judge: Any = DeterministicJudge()
    if not retrieval_only and use_llm_judge:
        try:
            from ragkit.providers import get_generator

            judge = LlmJudge(get_generator(settings, cheap=True))
        except Exception as exc:  # noqa: BLE001 - degrade rather than abort the run
            console.print(f"[yellow]LLM judge unavailable ({exc}); using deterministic.[/yellow]")

    results: list[QuestionResult] = []
    for question in questions:
        started = time.perf_counter()

        # Retrieve deeper than top_k so the rank diagnostic can tell "ranked 12" from
        # "absent entirely" — those need opposite responses.
        deep = pipeline.retrieve(question.question, top_k=max(top_k, 20))
        ordered_docs = dedupe_to_documents([s.chunk.doc_id for s in deep])
        contexts = deep[:top_k]

        result = QuestionResult(
            id=question.id,
            kind=question.kind,
            question=question.question,
            retrieved=ordered_docs[:top_k],
            first_relevant_rank=rank_of_first_relevant(ordered_docs, question.relevant),
            n_relevant=len(question.relevant),
            retrieval=evaluate_retrieval(question, ordered_docs),
        )

        if not retrieval_only:
            from ragkit.generate import generate_answer

            answer = generate_answer(question.question, contexts, pipeline.generator)
            grade: Grade = judge.grade(question, answer.text, contexts)
            result.answer = answer.text
            result.grade = asdict(grade)
            result.cost_usd = answer.usage.cost_usd

        result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        results.append(result)
        if not quiet:
            console.print(
                f"[dim]{result.id} {question.kind:13s} rank={result.first_relevant_rank}[/dim]"
            )

    judge_name = judge.name if not retrieval_only else "none"
    return summarise(results, settings, pipeline, top_k, judge_name)


def _mean(values: list[float]) -> float:
    # Drop NaN, which marks a judge that failed to return usable output. Averaging a
    # failed grade in as 0 would be indistinguishable from a genuinely bad answer.
    usable = [v for v in values if v == v]
    return round(statistics.fmean(usable), 4) if usable else float("nan")


def summarise(
    results: list[QuestionResult],
    settings: Settings,
    pipeline: RagPipeline,
    top_k: int,
    judge_name: str,
) -> dict[str, Any]:
    metric_names = list(results[0].retrieval)
    overall = {name: _mean([r.retrieval[name] for r in results]) for name in metric_names}

    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({r.kind for r in results}):
        subset = [r for r in results if r.kind == kind]
        by_kind[kind] = {name: _mean([r.retrieval[name] for r in subset]) for name in metric_names}
        by_kind[kind]["n"] = len(subset)
        # The highest recall@1 this kind could possibly score. A question with two
        # relevant documents caps at 0.5 by arithmetic, so a multi-hop row reading
        # "recall@1 = 0.500" can be a perfect score rather than a failure. Without
        # this column that row gets misread as broken retrieval every single time.
        by_kind[kind]["max_recall@1"] = _mean(
            [1.0 if r.n_relevant == 0 else min(1.0, 1.0 / r.n_relevant) for r in subset]
        )

    generation: dict[str, float] = {}
    graded = [r for r in results if r.grade]
    if graded:
        generation["groundedness"] = _mean([r.grade["groundedness"] for r in graded])
        generation["relevance"] = _mean([r.grade["relevance"] for r in graded])
        refusals = [
            r.grade["correct_refusal"] for r in graded if r.grade.get("correct_refusal") is not None
        ]
        if refusals:
            generation["correct_refusal"] = _mean(refusals)

    # Only answerable questions can show a ceiling: unanswerable ones score 1.0 by
    # construction and would otherwise drag the check into always firing.
    answerable = [r for r in results if r.kind != "unanswerable"]
    ceiling = bool(answerable) and all(
        _mean([r.retrieval[f"recall@{k}"] for r in answerable]) >= CEILING_THRESHOLD
        for k in K_VALUES
    )

    return {
        "config": {
            "profile": settings.profile.value,
            "top_k": top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunks_indexed": pipeline.count(),
            "judge": judge_name,
            "retriever": "dense",
        },
        "retrieval": overall,
        "by_kind": by_kind,
        "generation": generation,
        "ceiling_reached": ceiling,
        "mean_latency_ms": round(statistics.fmean([r.latency_ms for r in results]), 1),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 6),
        "results": [asdict(r) for r in results],
    }


def print_report(summary: dict[str, Any]) -> None:
    config = summary["config"]
    console.rule("[bold]Evaluation")
    console.print(
        f"[dim]profile={config['profile']} top_k={config['top_k']} "
        f"chunks={config['chunks_indexed']} judge={config['judge']}[/dim]\n"
    )

    table = Table(title="Retrieval (document level)")
    table.add_column("question kind")
    table.add_column("n", justify="right")
    for k in K_VALUES:
        table.add_column(f"recall@{k}", justify="right")
    table.add_column("max R@1", justify="right")
    table.add_column("mrr", justify="right")
    table.add_column("ndcg@5", justify="right")

    saturated: list[str] = []
    for kind, scores in summary["by_kind"].items():
        if kind != "unanswerable" and scores["recall@3"] >= CEILING_THRESHOLD:
            saturated.append(kind)
        # Show recall@1 against the best it could arithmetically be, so a capped score
        # reads as capped rather than as a failure.
        cap = scores["max_recall@1"]
        at_cap = scores["recall@1"] >= cap - 1e-9
        table.add_row(
            kind,
            str(int(scores["n"])),
            (f"[green]{scores['recall@1']:.3f}[/green]" if at_cap else f"{scores['recall@1']:.3f}"),
            *[f"{scores[f'recall@{k}']:.3f}" for k in K_VALUES[1:]],
            f"[dim]{cap:.3f}[/dim]",
            f"{scores['mrr']:.3f}",
            f"{scores['ndcg@5']:.3f}",
        )

    overall = summary["retrieval"]
    table.add_row(
        "[bold]all[/bold]",
        str(sum(int(s["n"]) for s in summary["by_kind"].values())),
        *[f"[bold]{overall[f'recall@{k}']:.3f}[/bold]" for k in K_VALUES],
        "",
        f"[bold]{overall['mrr']:.3f}[/bold]",
        f"[bold]{overall['ndcg@5']:.3f}[/bold]",
    )
    console.print(table)
    console.print(
        "[dim]max R@1 is the highest recall@1 arithmetically reachable: a question with\n"
        "two relevant documents caps at 0.500. Green means the kind is at its cap.[/dim]"
    )

    if saturated:
        console.print(
            f"\n[yellow]Saturated at recall@3: {', '.join(saturated)}.[/yellow] "
            "[dim]No technique can show a gain on these questions; only the\n"
            "unsaturated kinds can discriminate.[/dim]"
        )

    if summary["generation"]:
        gen = Table(title="Generation")
        gen.add_column("metric")
        gen.add_column("score", justify="right")
        for name, value in summary["generation"].items():
            gen.add_row(name, f"{value:.3f}")
        console.print(gen)

    near_misses = [
        r
        for r in summary["results"]
        if r["first_relevant_rank"] is not None and r["first_relevant_rank"] > 3
    ]
    if near_misses:
        console.print("\n[bold]Found, but ranked below 3[/bold] [dim]-> deepen the shortlist[/dim]")
        for r in near_misses:
            console.print(
                f"  [dim]{r['id']}[/dim] rank {r['first_relevant_rank']}: {r['question'][:62]}"
            )

    absent = [
        r
        for r in summary["results"]
        if r["first_relevant_rank"] is None and r["kind"] != "unanswerable"
    ]
    if absent:
        console.print(
            "\n[bold red]Never retrieved at any depth[/bold red] [dim]-> fix retrieval[/dim]"
        )
        for r in absent:
            console.print(f"  [dim]{r['id']}[/dim] {r['question'][:62]}")

    if summary["ceiling_reached"]:
        console.print(
            f"\n[bold yellow]Ceiling reached.[/bold yellow] Recall is at or above "
            f"{CEILING_THRESHOLD} at every k on answerable questions.\n"
            "[yellow]This corpus cannot distinguish retrieval techniques. Gains measured\n"
            "from here are not measurable at all - grow the corpus before concluding\n"
            "anything from lessons 04 onward.[/yellow]"
        )

    console.print(
        f"\n[dim]mean latency {summary['mean_latency_ms']}ms"
        + (f" - total cost ${summary['total_cost_usd']:.4f}" if summary["total_cost_usd"] else "")
        + "[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-only", action="store_true", help="Skip generation and grading."
    )
    parser.add_argument("--no-llm-judge", action="store_true", help="Use the deterministic grader.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions.")
    parser.add_argument("--label", default="", help="Label for the results.md row.")
    parser.add_argument(
        "--write", action="store_true", help="Append a row to benchmarks/results.md."
    )
    args = parser.parse_args()

    summary = run_eval(
        top_k=args.top_k,
        retrieval_only=args.retrieval_only,
        use_llm_judge=not args.no_llm_judge,
        limit=args.limit,
    )
    print_report(summary)

    if args.write:
        from ragkit.eval.report import append_row

        path = append_row(summary, label=args.label or "unlabelled")
        console.print(f"\n[green]Appended to {path}[/green]")


if __name__ == "__main__":
    main()
