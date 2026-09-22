"""Writing evaluation results into `benchmarks/results.md`.

The table is the spine of the repo, so appending to it is a code path rather than a
manual step. A result that requires someone to remember to write it down is a result
that will not be written down, and the whole eval-first premise collapses quietly.

Each run also writes its full per-question JSON next to the table. The aggregate says
whether something improved; the JSON says which questions moved, which is the part you
actually need when a change helps on average and breaks one question badly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ragkit.config import BENCHMARKS_DIR

RESULTS_PATH = BENCHMARKS_DIR / "results.md"
RUNS_DIR = BENCHMARKS_DIR / "runs"

LOCAL_MARKER = "<!-- local-results -->"
CLOUD_MARKER = "<!-- cloud-results -->"
PLACEHOLDER = "_awaiting"


def _fmt(value: float | None) -> str:
    if value is None or value != value:  # None or NaN
        return "-"
    return f"{value:.3f}"


def build_row(summary: dict[str, Any], label: str) -> str:
    config = summary["config"]
    retrieval = summary["retrieval"]
    generation = summary["generation"]

    # A ceiling flag on the row itself, so a number that cannot be trusted is never read
    # out of this table without its caveat attached.
    caveat = " ⚠️ ceiling" if summary["ceiling_reached"] else ""

    cells = [
        label + caveat,
        config["retriever"],
        str(config["chunks_indexed"]),
        _fmt(retrieval.get("recall@1")),
        _fmt(retrieval.get("recall@3")),
        _fmt(retrieval.get("recall@5")),
        _fmt(retrieval.get("mrr")),
        _fmt(retrieval.get("ndcg@5")),
        _fmt(generation.get("groundedness")),
        _fmt(generation.get("relevance")),
        _fmt(generation.get("correct_refusal")),
        f"{summary['mean_latency_ms']:.0f}",
    ]
    if config["profile"] == "cloud":
        cells.append(f"${summary['total_cost_usd']:.4f}")
    return "| " + " | ".join(cells) + " |"


def append_row(summary: dict[str, Any], label: str, path: Path | None = None) -> Path:
    """Insert a row under the marker for this run's profile."""
    target = path or RESULTS_PATH
    marker = CLOUD_MARKER if summary["config"]["profile"] == "cloud" else LOCAL_MARKER

    lines = target.read_text(encoding="utf-8").split("\n")
    try:
        index = next(i for i, line in enumerate(lines) if marker in line)
    except StopIteration as exc:
        raise ValueError(f"{target} has no {marker} marker to append under.") from exc

    # Drop the placeholder row the first time a real result lands.
    if index + 1 < len(lines) and PLACEHOLDER in lines[index + 1]:
        del lines[index + 1]

    lines.insert(index + 1, build_row(summary, label))
    target.write_text("\n".join(lines), encoding="utf-8")

    write_run(summary, label)
    return target


def write_run(summary: dict[str, Any], label: str) -> Path:
    """Persist the full per-question detail for this run."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(c if c.isalnum() else "-" for c in label.lower())[:48].strip("-")
    path = RUNS_DIR / f"{stamp}-{slug or 'run'}.json"
    path.write_text(
        json.dumps({"label": label, **summary}, indent=2, default=str), encoding="utf-8"
    )
    return path
