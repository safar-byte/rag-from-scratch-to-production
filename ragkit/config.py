"""Configuration and provider selection.

One idea carries the whole repo: nothing outside this module and `providers/` knows
which backend is in use. Lessons import `get_settings()` and the provider protocols,
never a vendor SDK. That is what lets the same code run offline on a laptop with no
API keys and in the cloud against Claude + Voyage.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"


class Profile(StrEnum):
    """Which family of providers to use.

    LOCAL is the default on purpose: a clean clone must produce a real answer with no
    API keys and no network. CLOUD is the opt-in production-quality path.
    """

    LOCAL = "local"
    CLOUD = "cloud"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    profile: Profile = Field(default=Profile.LOCAL, alias="RAG_PROFILE")

    # ---- local backend ------------------------------------------------------
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    # qwen2.5:1.5b, not a larger or a reasoning model. Measured on this CPU-only
    # machine: qwen3:4b (a reasoning model) took over 10 minutes for one RAG query,
    # because it spends ~1000 tokens thinking before it writes anything. A 1.5B
    # non-reasoning model answers the same query in seconds. Retrieval quality is
    # what this repo teaches; the generator only has to read the context it is given,
    # and a small instruct model does that well enough to measure groundedness.
    local_generation_model: str = Field(default="qwen2.5:1.5b", alias="LOCAL_GENERATION_MODEL")
    local_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="LOCAL_EMBEDDING_MODEL"
    )
    # MiniLM rather than a BGE reranker: ~90MB and fast on CPU, against ~2.3GB for
    # bge-reranker-v2-m3. A teaching repo people actually run beats a marginally
    # stronger one they abandon at the download. Swap it in .env for the stronger
    # model and measure whether the size is worth it on your corpus.
    local_rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="LOCAL_RERANK_MODEL"
    )

    # ---- cloud backend ------------------------------------------------------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    voyage_api_key: str | None = Field(default=None, alias="VOYAGE_API_KEY")

    # Model IDs live here rather than inline so a stale default is a one-line fix.
    # Never append a date suffix to a Claude model ID — the bare ID is complete.
    cloud_generation_model: str = Field(default="claude-opus-5", alias="CLOUD_GENERATION_MODEL")
    cloud_cheap_model: str = Field(default="claude-haiku-4-5", alias="CLOUD_CHEAP_MODEL")
    cloud_embedding_model: str = Field(default="voyage-3-large", alias="CLOUD_EMBEDDING_MODEL")
    cloud_rerank_model: str = Field(default="rerank-2.5", alias="CLOUD_RERANK_MODEL")

    # ---- storage ------------------------------------------------------------
    # "chroma" (embedded, the default - no server, no Docker) or "qdrant".
    vector_store: str = Field(default="chroma", alias="VECTOR_STORE")
    # Qdrant Cloud has a free tier; leaving the URL unset uses Qdrant's in-process
    # local mode, which also needs no server.
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    chroma_path: Path = Field(default=REPO_ROOT / ".chroma", alias="CHROMA_PATH")

    # ---- retrieval defaults -------------------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    # Over-fetch depth before reranking. 10 rather than the conventional 25 because
    # lesson 05's sweep measured it on this corpus: depth 10 matches depth 25 exactly
    # (R@5 0.987, vocab R@5 1.000) at a quarter of the latency, and depth 50 is
    # actively WORSE (R@5 0.960) - a small cross-encoder given more distractors makes
    # more mistakes. More candidates is not monotonically better. Re-sweep this on
    # your own corpus; it is the single most corpus-dependent number in this file.
    rerank_candidates: int = 10

    def require_cloud_keys(self) -> None:
        """Fail loudly and early rather than deep inside an HTTP call."""
        missing = [
            name
            for name, value in (
                ("ANTHROPIC_API_KEY", self.anthropic_api_key),
                ("VOYAGE_API_KEY", self.voyage_api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"RAG_PROFILE=cloud needs {', '.join(missing)}. "
                "Set them in .env, or use the default RAG_PROFILE=local."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
