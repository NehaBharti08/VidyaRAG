"""Qdrant client construction.

This module is the *only* place in the codebase that knows how Qdrant is
reached. Everything downstream receives a plain ``QdrantClient`` and cannot
tell whether it is talking to an in-process index, a local container, or
Qdrant Cloud.

That property is what makes the deployment story work: evaluation runs against
Qdrant Cloud, tests run against an in-memory instance, and the published demo
ships a self-contained on-disk index -- with no conditional logic anywhere in
the retrieval or ingestion code.
"""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from vidyarag.settings import IN_MEMORY, QdrantMode, Settings

# qdrant-client's local mode warns past this many points and degrades to a
# linear scan. The Biology 2e + Microbiology corpus is expected to land near
# 6k chunks, so embedded mode is comfortable -- but adding titles is not free.
LOCAL_MODE_POINT_ADVISORY = 20_000


def build_client(settings: Settings) -> QdrantClient:
    """Construct a client for the configured target.

    Args:
        settings: Resolved environment settings.

    Returns:
        A connected ``QdrantClient``. For embedded mode the parent directory of
        the index is created if absent.
    """
    mode = settings.qdrant_mode

    if mode is QdrantMode.EMBEDDED:
        path = settings.resolved_qdrant_path
        if path == IN_MEMORY:
            return QdrantClient(location=IN_MEMORY)
        Path(path).mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=path)

    if mode is QdrantMode.SERVER:
        return QdrantClient(url=settings.qdrant_url)

    if mode is QdrantMode.CLOUD:
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        return QdrantClient(url=settings.qdrant_url, api_key=api_key)

    raise ValueError(f"Unsupported Qdrant mode: {mode!r}")  # pragma: no cover


def describe_target(settings: Settings) -> str:
    """Human-readable description of the active target, with secrets redacted.

    Used in startup logs and the ``vidyarag health`` command so it is always
    obvious which index a run actually touched -- a benchmark attributed to the
    wrong index is worse than no benchmark.
    """
    mode = settings.qdrant_mode
    if mode is QdrantMode.EMBEDDED:
        return f"embedded ({settings.resolved_qdrant_path})"
    if mode is QdrantMode.SERVER:
        return f"server ({settings.qdrant_url})"
    return f"cloud ({settings.qdrant_url}, api_key=***)"
