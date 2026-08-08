"""Collection schema and writes.

Point ids are derived deterministically from ``chunk_id`` rather than assigned
by the store. That single choice is what makes ingestion idempotent: re-running
overwrites each point in place instead of appending a second copy, so a run
interrupted halfway can simply be run again. Random ids would silently double
the corpus and quietly wreck every retrieval metric that follows.

The payload carries attribution -- licence, title, source URL -- on every point.
Attribution that lives only in a README stops being true the moment a chunk is
returned by an API; carrying it on the point means it survives retrieval.
"""

from __future__ import annotations

import uuid
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from vidyarag.ingest.chunk import Chunk

# Fixed namespace so a chunk_id maps to the same point id on every machine and
# every run. Regenerating this value would orphan every previously written point.
POINT_NAMESPACE = uuid.UUID("6f3c1e58-9a2b-4d77-9f0e-2c5a8b1d4e63")

DENSE_VECTOR = "dense"
"""Named vector. Sparse BM25 joins it under its own name in Phase 4."""


def point_id(chunk_id: str) -> str:
    """Deterministic point id for a chunk."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def ensure_collection(
    client: QdrantClient,
    name: str,
    dimension: int,
    *,
    recreate: bool = False,
) -> bool:
    """Create the collection if absent.

    Args:
        client: Connected Qdrant client.
        name: Collection name.
        dimension: Dense vector width; must match the embedding model.
        recreate: Drop and rebuild an existing collection.

    Returns:
        ``True`` if a collection was created, ``False`` if a usable one existed.

    Raises:
        ValueError: If an existing collection was built with a different vector
            width. Writing 768-dim vectors into a 1536-dim collection fails
            deep inside an upsert; catching it here says what actually happened.
    """
    exists = client.collection_exists(name)

    if exists and recreate:
        client.delete_collection(name)
        exists = False

    if exists:
        info = client.get_collection(name)
        configured = info.config.params.vectors
        current = configured.get(DENSE_VECTOR) if isinstance(configured, dict) else configured
        if current is not None and current.size != dimension:
            raise ValueError(
                f"Collection {name!r} has {current.size}-dim vectors but the configured "
                f"embedding model produces {dimension}. Re-ingest with --recreate, or "
                f"point QDRANT_COLLECTION at a different name."
            )
        return False

    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=dimension,
                distance=models.Distance.COSINE,
            )
        },
    )
    # Filtering by book is how a "search only Anatomy" query stays cheap, and
    # how per-book recall can be measured separately during evaluation.
    #
    # Local mode ignores payload indexes and says so via a UserWarning. That is
    # expected here rather than a problem -- the same code must produce a
    # correctly indexed collection on server and Cloud, where it does matter --
    # so the notice is silenced instead of being printed on every ingest.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Pp]ayload indexes.*", category=UserWarning)
        client.create_payload_index(
            collection_name=name,
            field_name="book_slug",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    return True


def build_payload(chunk: Chunk, license_name: str, source_url: str) -> dict[str, Any]:
    """Everything a retrieved chunk needs to be cited and attributed."""
    return {
        "chunk_id": chunk.chunk_id,
        "book_slug": chunk.book_slug,
        "book_title": chunk.book_title,
        "chapter": chunk.chapter,
        "section": chunk.section,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "printed_page": chunk.printed_page,
        "citation": chunk.citation,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "license": license_name,
        "source_url": source_url,
    }


def upsert_points(
    client: QdrantClient,
    name: str,
    points: Sequence[models.PointStruct],
) -> int:
    """Write a batch of points, returning how many were written."""
    if not points:
        return 0
    client.upsert(collection_name=name, points=list(points), wait=True)
    return len(points)


def make_points(
    pairs: Iterable[tuple[Chunk, list[float]]],
    *,
    license_name: str,
    source_url: str,
) -> Iterator[models.PointStruct]:
    """Pair chunks with their vectors into writable points."""
    for chunk, vector in pairs:
        yield models.PointStruct(
            id=point_id(chunk.chunk_id),
            vector={DENSE_VECTOR: vector},
            payload=build_payload(chunk, license_name, source_url),
        )


def count_points(client: QdrantClient, name: str) -> int:
    """Number of points currently in the collection, or 0 if it does not exist."""
    if not client.collection_exists(name):
        return 0
    return int(client.count(name, exact=True).count)


def existing_chunk_ids(client: QdrantClient, name: str) -> set[str]:
    """Chunk ids already written, for resuming an interrupted ingest.

    Scrolls payloads rather than trusting a count: a partial run leaves a
    partial collection, and only the ids say which chunks still need work.
    """
    if not client.collection_exists(name):
        return set()

    found: set[str] = set()
    offset: Any = None
    while True:
        records, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=["chunk_id"],
            with_vectors=False,
        )
        for record in records:
            if record.payload and (chunk_id := record.payload.get("chunk_id")):
                found.add(str(chunk_id))
        if offset is None:
            break
    return found
