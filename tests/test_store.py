"""Vector store client construction.

These tests exercise the real ``QdrantClient`` in in-memory mode rather than a
mock, so they verify the same class production uses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vidyarag.settings import IN_MEMORY, QdrantMode, Settings
from vidyarag.store import build_client, describe_target


class TestBuildClient:
    def test_in_memory_client_is_usable(self, memory_settings: Settings) -> None:
        client = build_client(memory_settings)
        assert client.get_collections().collections == []

    def test_embedded_mode_creates_the_index_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "index"
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            QDRANT_MODE=QdrantMode.EMBEDDED,
            QDRANT_PATH=str(target),
        )
        client = build_client(settings)
        assert target.exists()
        client.close()

    def test_embedded_index_persists_across_clients(self, tmp_path: Path) -> None:
        """The shipped demo depends on this: the index must survive a restart."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            QDRANT_MODE=QdrantMode.EMBEDDED,
            QDRANT_PATH=str(tmp_path / "index"),
        )
        from qdrant_client.models import Distance, VectorParams

        client = build_client(settings)
        client.create_collection(
            collection_name="persisted",
            vectors_config=VectorParams(size=8, distance=Distance.COSINE),
        )
        client.close()

        reopened = build_client(settings)
        names = [c.name for c in reopened.get_collections().collections]
        assert "persisted" in names
        reopened.close()


class TestDescribeTarget:
    def test_embedded_target_reports_path(self, memory_settings: Settings) -> None:
        assert describe_target(memory_settings) == f"embedded ({IN_MEMORY})"

    def test_cloud_target_redacts_the_api_key(self) -> None:
        """describe_target() output reaches logs, so it must never leak a key."""
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            QDRANT_MODE=QdrantMode.CLOUD,
            QDRANT_URL="https://example.qdrant.io:6333",
            QDRANT_API_KEY="super-secret-key",
        )
        described = describe_target(settings)
        assert "super-secret-key" not in described
        assert "***" in described

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [(QdrantMode.EMBEDDED, "embedded"), (QdrantMode.SERVER, "server")],
    )
    def test_mode_is_named_in_the_description(self, mode: QdrantMode, expected: str) -> None:
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            QDRANT_MODE=mode,
            QDRANT_URL="http://localhost:6333",
        )
        assert describe_target(settings).startswith(expected)
