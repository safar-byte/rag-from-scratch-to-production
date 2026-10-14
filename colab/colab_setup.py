"""Get a fresh Google Colab runtime ready for the lesson notebooks.

Every notebook in `lessons/*/colab.ipynb` clones this repo and then calls `setup()`.
The few display helpers at the bottom keep the notebooks' own cells about the lesson
rather than about pandas formatting.
It is idempotent: re-running a notebook's setup cell after a crash, or opening a second
notebook in the same runtime, skips whatever is already done.

Colab is not the author's machine, and the differences matter:

* **Python is whatever Colab ships**, not the 3.12 that `pyproject.toml` pins. Torch
  and sentence-transformers come preinstalled there, so the pin (which exists because
  of a missing torch wheel elsewhere) does not apply. The dependencies are installed
  directly and `ragkit` is imported from the clone, so the pin never blocks install.
* **There may be a GPU.** Pick Runtime -> Change runtime type -> T4 GPU for the lessons
  that call the generator a lot (03, 06, 07, 10). Ollama and sentence-transformers both
  use it automatically; nothing in `ragkit` changes.
* **Ollama is not installed**, so it is installed and started here, in the background.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_LOG = Path("/tmp/ollama.log")


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def _run(cmd: list[str] | str, *, shell: bool = False) -> None:
    """Run quietly, but show the tail of the output if it fails - a silent failure in a
    setup cell is the worst possible start to a lesson."""
    done = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if done.returncode != 0:
        print((done.stdout + done.stderr)[-3000:])
        raise RuntimeError(f"setup step failed: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")


def _requirements() -> list[str]:
    """Base + the extras the notebooks use, read from pyproject so they never drift."""
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    return [*project["dependencies"], *extras["local"], *extras["graph"], "pandas", "matplotlib"]


def install_python_deps() -> None:
    try:
        import bm25s  # noqa: F401
        import chromadb  # noqa: F401
        import networkx  # noqa: F401
        import sentence_transformers  # noqa: F401
        import Stemmer  # noqa: F401

        print("✓ python dependencies already installed")
        return
    except ImportError:
        pass
    print("… installing python dependencies (about a minute; torch is already on Colab)")
    _run([sys.executable, "-m", "pip", "install", "-q", *_requirements()])
    print("✓ python dependencies installed")


def _ollama_up() -> bool:
    import httpx

    try:
        return httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=2.0).status_code == 200
    except Exception:  # noqa: BLE001 - not running is an answer, not an error
        return False


def _ollama_models() -> list[str]:
    import httpx

    return [m["name"] for m in httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0).json()["models"]]


def start_ollama(model: str) -> bool:
    """Install, serve and pull. Returns False (and says why) rather than raising.

    Generation is optional in most lessons - retrieval is the half worth watching - so
    a failure here must not stop the notebook from running its retrieval cells.
    """
    if not shutil.which("ollama"):
        if not in_colab():
            print("⚠ ollama is not installed on this machine - generation cells will skip.")
            return False
        print("… installing Ollama (about a minute)")
        # The installer extracts a .tar.zst and probes the GPU with lspci/lshw.
        _run("apt-get -qq update && apt-get -qq install -y zstd pciutils lshw", shell=True)
        _run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)

    if not _ollama_up():
        log = OLLAMA_LOG.open("w")
        # Popen, not `!ollama serve &`: a process started from Python survives between
        # cells for as long as the runtime does.
        subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, start_new_session=True)
        for _ in range(60):
            if _ollama_up():
                break
            time.sleep(1)
        else:
            print(f"⚠ Ollama did not start - see {OLLAMA_LOG}. Generation cells will skip.")
            return False

    if not any(name == model or name == f"{model}:latest" for name in _ollama_models()):
        print(f"… pulling {model} (about 1GB, one-off per runtime)")
        _run(["ollama", "pull", model])
    print(f"✓ Ollama serving {model} at {OLLAMA_HOST}")
    return True


def gpu_name() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - no nvidia-smi means no GPU
        return None


def setup(*, llm: bool = True, model: str | None = None) -> bool:
    """Prepare the runtime. Returns whether a generator is available.

    `llm=False` skips Ollama entirely, for lessons (02, 04, 05, 08, 09, 11) whose core
    runs on retrieval alone.
    """
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    # So `!python lessons/...` and `!python -m ragkit...` subprocesses find ragkit too.
    os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    # Quieter model downloads; HF warns about a missing token otherwise.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    install_python_deps()

    gpu = gpu_name()
    print(f"✓ python {sys.version.split()[0]} · GPU: {gpu or 'none (CPU runtime)'}")

    model = model or os.environ.get("LOCAL_GENERATION_MODEL", "qwen2.5:1.5b")
    llm_ready = start_ollama(model) if llm else False
    if not llm:
        print("· Ollama skipped - this lesson runs on retrieval alone")
    print(f"✓ repo at {REPO_ROOT}")
    return llm_ready


# ---- display helpers ---------------------------------------------------------------


def results_frame(results: list, *, chars: int = 90, scores: bool = True):  # noqa: ANN201
    """A ranked result list as a DataFrame: rank, document, every per-stage score, text."""
    import pandas as pd

    rows = []
    for item in results:
        row = {"rank": item.rank, "doc_id": item.chunk.doc_id, "score": round(item.score, 4)}
        if scores:
            row.update({k: round(v, 4) for k, v in item.scores.items()})
        row["text"] = " ".join(item.chunk.text.split())[:chars] + "…"
        rows.append(row)
    return pd.DataFrame(rows).set_index("rank") if rows else pd.DataFrame()


def eval_frames(summary: dict):  # noqa: ANN201
    """`run_eval`'s summary as two DataFrames: overall, and broken down by question kind."""
    import pandas as pd

    cols = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5"]
    overall = pd.DataFrame([{c: summary["retrieval"][c] for c in cols}], index=["all"])
    by_kind = pd.DataFrame(summary["by_kind"]).T
    keep = ["n", *cols, "max_recall@1"]
    by_kind = by_kind[[c for c in keep if c in by_kind.columns]]
    by_kind["n"] = by_kind["n"].astype(int)
    return overall, by_kind


def compare_runs(summaries: dict[str, dict], kind: str | None = None):  # noqa: ANN201
    """One row per labelled run, so a lesson's comparison reads as a single table."""
    import pandas as pd

    cols = ["recall@1", "recall@3", "recall@5", "ndcg@5"]
    rows = {}
    for label, summary in summaries.items():
        source = summary["by_kind"].get(kind, {}) if kind else summary["retrieval"]
        rows[label] = {c: source.get(c, float("nan")) for c in cols}
        if not kind:
            vocab = summary["by_kind"].get("vocab_mismatch", {})
            rows[label]["vocab R@5"] = vocab.get("recall@5", float("nan"))
            rows[label]["ms/query"] = summary["mean_latency_ms"]
    return pd.DataFrame(rows).T
