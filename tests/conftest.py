"""Shared fixtures.

Tests never touch a real Qdrant server or a paid API. The in-memory Qdrant
mode is the same client class used in production, so these fixtures exercise
real store behaviour rather than a mock of it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from vidyarag.settings import IN_MEMORY, QdrantMode, Settings

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real .env from leaking into tests.

    Without this, a populated .env would make tests pass locally and fail in
    CI (or worse, spend money).
    """
    for var in (
        "OPENAI_API_KEY",
        "QDRANT_MODE",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_PATH",
        "QDRANT_COLLECTION",
        "VIDYARAG_PROFILE",
        "LOG_LEVEL",
        "LOG_FORMAT",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def memory_settings() -> Settings:
    """Settings pointed at an ephemeral in-process Qdrant."""
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        OPENAI_API_KEY="test-key-not-real",
        QDRANT_MODE=QdrantMode.EMBEDDED,
        QDRANT_PATH=IN_MEMORY,
        QDRANT_COLLECTION="test_collection",
        VIDYARAG_PROFILE="baseline",
    )


@pytest.fixture
def config_dir() -> Iterator[Path]:
    """The repository's real config directory.

    Profiles are validated against the shipped configs on purpose: a broken
    profile should fail the test suite, not just runtime.
    """
    yield REPO_ROOT / "config"
