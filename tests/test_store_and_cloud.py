"""The store factory and the cloud generator's request shapes.

No network. The cloud tests assert the *shape* of what would be sent, because those
request bodies encode several API rules that are easy to get wrong from memory and
impossible to check without either a key or a test like this.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragkit.config import Settings
from ragkit.store import get_store

# ---- store factory ---------------------------------------------------------


def test_chroma_is_the_default(tmp_path) -> None:
    settings = Settings(_env_file=None)
    settings.chroma_path = tmp_path / "chroma"
    assert "chroma" in get_store(settings).name


def test_qdrant_requires_the_dimension_up_front() -> None:
    # Qdrant fixes the vector size when the collection is created, unlike Chroma which
    # infers it from the first insert. Failing loudly here beats creating a collection
    # with the wrong size and discovering it on the first query.
    settings = Settings(_env_file=None, VECTOR_STORE="qdrant")
    with pytest.raises(ValueError, match="dimension"):
        get_store(settings)


def test_unknown_store_names_the_options() -> None:
    settings = Settings(_env_file=None, VECTOR_STORE="pinecone")
    with pytest.raises(ValueError, match="chroma.*qdrant"):
        get_store(settings)


# ---- cloud request shapes --------------------------------------------------


class _RecordingMessages:
    def __init__(self) -> None:
        self.last: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> Any:
        self.last = kwargs

        class _Block:
            type = "text"
            text = "an answer"
            citations = []

        class _Usage:
            input_tokens = 100
            output_tokens = 10
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

        class _Response:
            stop_reason = "end_turn"
            content = [_Block()]
            usage = _Usage()

        return _Response()


def _generator() -> tuple[Any, _RecordingMessages]:
    from ragkit.providers.cloud import ClaudeGenerator

    settings = Settings(_env_file=None, ANTHROPIC_API_KEY="k", VOYAGE_API_KEY="k")
    generator = ClaudeGenerator.__new__(ClaudeGenerator)
    generator._settings = settings
    generator._model = settings.cloud_generation_model
    generator._effort = "low"
    messages = _RecordingMessages()
    generator._client = type("C", (), {"messages": messages})()
    return generator, messages


def test_thinking_is_adaptive_and_budget_tokens_is_never_sent() -> None:
    """`budget_tokens` returns a 400 on Opus 5. Depth is tuned with effort instead."""
    generator, messages = _generator()
    generator.generate(system="s", prompt="p")

    assert messages.last is not None
    assert messages.last["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in str(messages.last)
    assert messages.last["output_config"]["effort"] == "low"


def test_model_id_carries_no_date_suffix() -> None:
    generator, messages = _generator()
    generator.generate(system="s", prompt="p")
    assert messages.last is not None
    assert messages.last["model"] == "claude-opus-5"


def test_cache_control_goes_on_the_system_block_only_when_asked() -> None:
    generator, messages = _generator()

    generator.generate(system="s", prompt="p")
    assert messages.last is not None
    assert "cache_control" not in messages.last["system"][0]

    generator.generate(system="s", prompt="p", cache_system=True)
    assert messages.last["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_refusal_raises_rather_than_returning_empty_text() -> None:
    generator, messages = _generator()

    class _Refused:
        stop_reason = "refusal"
        stop_details = type("D", (), {"category": "cyber"})()
        content = []
        usage = type("U", (), {"input_tokens": 1, "output_tokens": 0})()

    messages.create = lambda **kw: _Refused()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="declined"):
        generator.generate(system="s", prompt="p")


def test_citations_are_enabled_on_every_document_block() -> None:
    """The API requires citations on all document blocks or none."""
    generator, messages = _generator()
    generator.generate_with_citations(
        system="s",
        question="q",
        documents=[("Doc A", "text a"), ("Doc B", "text b")],
    )

    assert messages.last is not None
    blocks = messages.last["messages"][0]["content"]
    documents = [b for b in blocks if b["type"] == "document"]
    assert len(documents) == 2
    assert all(d["citations"] == {"enabled": True} for d in documents)
    # The question goes last, after the documents — both so the model reads the
    # sources first and so the documents form a cacheable prefix.
    assert blocks[-1]["type"] == "text"


def test_citation_caching_marks_the_last_document_not_the_question() -> None:
    # The documents are the stable prefix; the question changes every request. A
    # breakpoint after the question would never hit.
    generator, messages = _generator()
    generator.generate_with_citations(
        system="s", question="q", documents=[("A", "a"), ("B", "b")], cache_documents=True
    )
    assert messages.last is not None
    blocks = messages.last["messages"][0]["content"]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}  # last document
    assert "cache_control" not in blocks[-1]  # the question


def test_citations_are_not_combined_with_structured_output() -> None:
    # The pair returns a 400, so the two paths must stay separate. This pins that the
    # citations request never grows an output_config.format.
    generator, messages = _generator()
    generator.generate_with_citations(system="s", question="q", documents=[("A", "a")])
    assert messages.last is not None
    assert "format" not in messages.last.get("output_config", {})
