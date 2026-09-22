"""Shared fixtures."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from ragkit.config import Settings
from ragkit.pipeline import RagPipeline
from tests.fakes import FakeEmbedder, FakeGenerator


@pytest.fixture
def temp_settings() -> Iterator[Settings]:
    """Settings pointing at a throwaway Chroma directory.

    `ignore_cleanup_errors=True` is not laziness: on Windows, Chroma keeps its HNSW
    segment file open for the life of the process, so the directory cannot be removed
    while the test session is still running. The temp files are cleaned up by the OS.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(_env_file=None)
        settings.chroma_path = Path(tmp) / "chroma"
        yield settings


@pytest.fixture
def pipeline(temp_settings: Settings) -> RagPipeline:
    """A pipeline wired to deterministic fakes — no models, no network."""
    built = RagPipeline(temp_settings)
    built._embedder = FakeEmbedder(dimension=32)
    built._generator = FakeGenerator()
    return built
