"""The Ollama generator's handling of reasoning models.

No network: the HTTP client is replaced with a stub. What is being tested is the
response *parsing*, which is where the interesting bug was.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragkit.config import Settings
from ragkit.providers.local import OllamaGenerator


class _StubResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _StubClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.last_json: dict[str, Any] | None = None

    def post(self, _path: str, json: dict[str, Any]) -> _StubResponse:
        self.last_json = json
        return _StubResponse(self._payload)


def _generator(payload: dict[str, Any]) -> tuple[OllamaGenerator, _StubClient]:
    generator = OllamaGenerator(Settings(_env_file=None))
    client = _StubClient(payload)
    generator._client = client  # type: ignore[assignment]
    return generator, client


def test_a_normal_response_is_returned_with_usage() -> None:
    generator, _ = _generator(
        {"response": "  The port is 8788. [1]  ", "prompt_eval_count": 120, "eval_count": 9}
    )
    text, usage = generator.generate(system="s", prompt="p")
    assert text == "The port is 8788. [1]"
    assert usage.input_tokens == 120
    assert usage.output_tokens == 9
    # Local inference is free, and the cost column has to stay honest about that.
    assert usage.cost_usd == 0.0


def test_thinking_only_response_raises_instead_of_returning_empty() -> None:
    """Regression: a reasoning model that runs out of budget mid-thought.

    Ollama returns the chain of thought in a separate `thinking` field. If the token
    budget is exhausted before the model starts writing, `response` comes back as an
    empty string with no error. Returning that "" would be scored by the eval harness
    as a refusal — so a truncation bug would show up as a model that declines to
    answer, which is about the most misleading failure available.
    """
    generator, _ = _generator(
        {"response": "", "thinking": "Let me consider the context... " * 40, "eval_count": 1024}
    )
    with pytest.raises(RuntimeError, match="whole token budget"):
        generator.generate(system="s", prompt="p", max_tokens=16)


def test_a_genuinely_empty_response_is_not_mistaken_for_truncation() -> None:
    # No thinking field means the model simply said nothing; that is not the
    # truncation case and must not raise.
    generator, _ = _generator({"response": "", "eval_count": 0})
    text, _usage = generator.generate(system="s", prompt="p")
    assert text == ""


def test_headroom_is_added_for_thinking_tokens() -> None:
    generator, client = _generator({"response": "ok"})
    generator.generate(system="s", prompt="p", max_tokens=256)
    assert client.last_json is not None
    expected = 256 + OllamaGenerator.THINKING_HEADROOM
    assert client.last_json["options"]["num_predict"] == expected


def test_temperature_is_zero_for_reproducibility() -> None:
    # The eval harness is meaningless if identical input gives different output.
    generator, client = _generator({"response": "ok"})
    generator.generate(system="s", prompt="p")
    assert client.last_json is not None
    assert client.last_json["options"]["temperature"] == 0.0
