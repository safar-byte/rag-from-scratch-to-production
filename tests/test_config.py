"""A clean clone with no .env must land on the offline profile.

This is the test that protects the repo's central promise: clone it, run it, get an
answer, no API keys. If the default ever flips to cloud, someone's first experience
of this repo becomes a stack trace.
"""

from __future__ import annotations

import pytest

from ragkit.config import Profile, Settings


def test_default_profile_is_local() -> None:
    settings = Settings(_env_file=None)
    assert settings.profile is Profile.LOCAL


def test_local_profile_needs_no_keys() -> None:
    settings = Settings(_env_file=None)
    assert settings.anthropic_api_key is None
    assert settings.voyage_api_key is None


def test_cloud_profile_fails_fast_with_a_useful_message() -> None:
    settings = Settings(_env_file=None, RAG_PROFILE="cloud")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        settings.require_cloud_keys()


def test_claude_model_ids_carry_no_date_suffix() -> None:
    # A date-suffixed ID is the single most common stale-prior bug against this API.
    settings = Settings(_env_file=None)
    assert settings.cloud_generation_model == "claude-opus-5"
    assert not settings.cloud_generation_model[-1].isdigit() or "-20" not in (
        settings.cloud_generation_model
    )
