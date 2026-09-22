"""Lesson 00 - check the environment before anything else.

    python lessons/00-setup/run.py

Verifies the things that produce confusing errors later: the Python version, which
optional extras are installed, whether Ollama is reachable, and whether an index
exists. Everything is checked independently, so a missing piece reports itself rather
than failing the whole script.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from ragkit.config import DATA_DIR, get_settings

console = Console()


def _probe(name: str, check: object) -> tuple[str, str, str]:
    try:
        ok, detail = check()  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - a failed probe is a result, not a crash
        return name, "[red]fail[/red]", f"{type(exc).__name__}: {exc}"
    return name, "[green]ok[/green]" if ok else "[yellow]missing[/yellow]", detail


def main() -> None:
    settings = get_settings()

    def python() -> tuple[bool, str]:
        v = sys.version_info
        # 3.13+ has no torch wheels at the time of writing, which is the single most
        # confusing install failure in this repo.
        return (v.major, v.minor) == (3, 12), f"{v.major}.{v.minor}.{v.micro} (want 3.12)"

    def core() -> tuple[bool, str]:
        import chromadb  # noqa: F401

        return True, "chromadb, bm25s"

    def local_models() -> tuple[bool, str]:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False, 'not installed - pip install -e ".[local]"'
        return True, "sentence-transformers"

    def ollama() -> tuple[bool, str]:
        import httpx

        try:
            r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
            models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return False, f"not reachable at {settings.ollama_host} - run `ollama serve`"
        want = settings.local_generation_model
        return (
            want in models
        ), f"{len(models)} model(s); {want} {'present' if want in models else 'MISSING'}"

    def corpus() -> tuple[bool, str]:
        docs = [p for p in DATA_DIR.glob("*.md") if p.name.lower() != "readme.md"]
        return bool(docs), f"{len(docs)} documents in data/"

    def index() -> tuple[bool, str]:
        from ragkit.pipeline import RagPipeline

        n = RagPipeline(settings).count()
        return n > 0, f"{n} chunks indexed" + ("" if n else " - run `ragkit ingest`")

    table = Table(title="Environment")
    for column in ("check", "status", "detail"):
        table.add_column(column)
    for probe in (python, core, local_models, ollama, corpus, index):
        table.add_row(*_probe(probe.__name__.replace("_", " "), probe))
    console.print(table)

    console.print(
        f"\n[dim]profile={settings.profile.value} · store={settings.vector_store} · "
        f"chunk={settings.chunk_size}/{settings.chunk_overlap} · "
        f"rerank_candidates={settings.rerank_candidates}[/dim]"
    )
    console.print(
        "[dim]Only 'python', 'core' and 'corpus' must pass to read the code and run the\n"
        "tests. Ollama is needed for generation; local models for real embeddings.[/dim]"
    )


if __name__ == "__main__":
    main()
