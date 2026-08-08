"""Vector store access."""

from vidyarag.store.client import build_client, describe_target
from vidyarag.store.collection import (
    DENSE_VECTOR,
    count_points,
    ensure_collection,
    existing_chunk_ids,
    make_points,
    point_id,
    upsert_points,
)

__all__ = [
    "DENSE_VECTOR",
    "build_client",
    "count_points",
    "describe_target",
    "ensure_collection",
    "existing_chunk_ids",
    "make_points",
    "point_id",
    "upsert_points",
]
