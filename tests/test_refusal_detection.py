"""Refusal detection.

This is measurement code, and it was wrong once in a way that produced a headline
number. The first version of REFUSAL_MARKERS had "not provided" but not "does not
provide", so a genuine refusal —

    "The passage does not provide any information about the population of Tokyo."

— was scored as a confabulation, and the reported refusal rate read 0.000 when the true
rate was 0.300. A gap in the detector is indistinguishable from a failure in the model
once it reaches the results table.

Hence these tests, which use the actual answers the local model produced rather than
invented ones.
"""

from __future__ import annotations

import pytest

from ragkit.eval.judge import looks_like_refusal

# Verbatim from a real run.
REAL_REFUSALS = [
    "The passage does not provide any information about the population of Tokyo.",
    "The passage does not provide information about who wrote the documentation.",
    "The passage does not provide a specific target for the p99 query latency.",
    "The provided context does not contain this information.",
    "The context does not contain enough information to answer this question.",
    "The corpus does not mention this setting anywhere.",
    "I cannot answer this from the provided passages.",
    "This is not specified in the retrieved context.",
]

REAL_CONFABULATIONS = [
    "Paris is the capital of France.",
    "Paris.",
    "The default value of the SHARD_REPLICATION_FACTOR setting is 1.",
    "The default value of `MAX_CHUNK_BYTES` is 1024 bytes.",
    "AWS (Amazon Web Services) is the cloud provider used for running this service.",
    "This service runs on Acme Corp's cloud infrastructure segment.",
    "The system uses a bi-encoder model in production.",
]

REAL_ANSWERS = [
    "The query service listens on port 8788 by default.",
    "RRF uses a constant k conventionally set to 60.",
    "Exit status 75 means the lock was held; check for a stale lock file.",
]


@pytest.mark.parametrize("text", REAL_REFUSALS)
def test_genuine_refusals_are_detected(text: str) -> None:
    assert looks_like_refusal(text), f"missed a refusal: {text!r}"


@pytest.mark.parametrize("text", REAL_CONFABULATIONS)
def test_confabulations_are_not_mistaken_for_refusals(text: str) -> None:
    assert not looks_like_refusal(text), f"confabulation scored as refusal: {text!r}"


@pytest.mark.parametrize("text", REAL_ANSWERS)
def test_correct_answers_are_not_mistaken_for_refusals(text: str) -> None:
    # A false positive here would inflate the refusal rate, which is the mirror of the
    # bug that prompted this file. Both directions have to be checked.
    assert not looks_like_refusal(text), f"real answer scored as refusal: {text!r}"


def test_the_specific_phrase_that_was_missed() -> None:
    """Regression, named so it cannot be quietly deleted."""
    assert looks_like_refusal("The passage does not provide any information about that.")
