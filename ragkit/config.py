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
    local_generation_model: str = Field(default="qwen2.5:7b", alias="LOCAL_GENERATION_MODEL")
    local_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="LOCAL_EMBEDDING_MODEL"
    )
    local_rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="LOCAL_RERANK_MODEL")

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
    vector_store: str = Field(default="chroma", alias="VECTOR_STORE")
    chroma_path: Path = Field(default=REPO_ROOT / ".chroma", alias="CHROMA_PATH")

    # ---- retrieval defaults -------------------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    # Over-fetch before reranking: the reranker can only reorder what retrieval found,
    # so recall at this depth is the ceiling on final quality (see lesson 05).
    rerank_candidates: int = 25

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
