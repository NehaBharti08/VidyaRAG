"""Collection schema and write tests.

These run against a real in-process Qdrant rather than a mock, so they exercise
actual store behaviour. Vectors are synthetic -- the embedding model is not
under test here, and downloading 210 MB of ONNX weights in CI would be a poor
trade for testing dictionary plumbing.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from vidyarag.ingest.chunk import Chunk
from vidyarag.store.collection import (
    DENSE_VECTOR,
    build_payload,
    count_points,
    ensure_collection,
    existing_chunk_ids,
    make_points,
    point_id,
    upsert_points,
)

DIM = 8


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(location=":memory:")


def _chunk(chunk_id: str = "biology-p0133-05", page: int = 133) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        book_slug="biology",
        book_title="Biology",
        chapter="Chapter 4. Cell Structure",
        section="4.4. The Endomembrane System",
        page_start=page,
        page_end=page,
        printed_page="121",
        text="The vesicle fuses with a lysosome.",
        token_count=9,
    )


def _vector(seed: float = 0.1) -> list[float]:
    return [seed] * DIM


class TestPointId:
    def test_is_deterministic(self) -> None:
        """Idempotent ingestion depends entirely on this."""
        assert point_id("biology-p0133-05") == point_id("biology-p0133-05")

    def test_differs_between_chunks(self) -> None:
        assert point_id("biology-p0133-05") != point_id("biology-p0133-06")


class TestEnsureCollection:
    def test_creates_when_absent(self, client: QdrantClient) -> None:
        assert ensure_collection(client, "c", DIM) is True
        assert client.collection_exists("c")

    def test_is_a_no_op_when_present(self, client: QdrantClient) -> None:
        ensure_collection(client, "c", DIM)
        assert ensure_collection(client, "c", DIM) is False

    def test_recreate_drops_existing_points(self, client: QdrantClient) -> None:
        ensure_collection(client, "c", DIM)
        upsert_points(
            client,
            "c",
            list(make_points([(_chunk(), _vector())], license_name="CC BY 4.0", source_url="u")),
        )
        assert count_points(client, "c") == 1

        ensure_collection(client, "c", DIM, recreate=True)
        assert count_points(client, "c") == 0

    def test_rejects_a_dimension_mismatch(self, client: QdrantClient) -> None:
        """Writing 768-dim vectors into a 384-dim collection fails deep inside
        an upsert; catching it here says what actually went wrong."""
        ensure_collection(client, "c", DIM)
        with pytest.raises(ValueError, match="Re-ingest with --recreate"):
            ensure_collection(client, "c", DIM * 2)


class TestPayload:
    def test_carries_attribution_and_citation(self) -> None:
        """Attribution must survive retrieval, not live only in a README."""
        payload = build_payload(_chunk(), "CC BY 4.0", "https://openstax.org/details/books/biology")
        assert payload["license"] == "CC BY 4.0"
        assert payload["source_url"].startswith("https://openstax.org/")
        assert payload["citation"] == "Biology, 4.4. The Endomembrane System, p.121"
        assert payload["printed_page"] == "121"
        assert payload["text"]


class TestUpsert:
    def test_writes_points(self, client: QdrantClient) -> None:
        ensure_collection(client, "c", DIM)
        points = list(
            make_points([(_chunk(), _vector())], license_name="CC BY 4.0", source_url="u")
        )
        assert upsert_points(client, "c", points) == 1
        assert count_points(client, "c") == 1

    def test_rewriting_the_same_chunk_does_not_duplicate(self, client: QdrantClient) -> None:
        """The property that makes re-running ingestion safe."""
        ensure_collection(client, "c", DIM)
        for _ in range(3):
            points = list(
                make_points([(_chunk(), _vector())], license_name="CC BY 4.0", source_url="u")
            )
            upsert_points(client, "c", points)
        assert count_points(client, "c") == 1

    def test_empty_batch_is_a_no_op(self, client: QdrantClient) -> None:
        ensure_collection(client, "c", DIM)
        assert upsert_points(client, "c", []) == 0

    def test_vectors_are_searchable_under_the_named_vector(self, client: QdrantClient) -> None:
        ensure_collection(client, "c", DIM)
        upsert_points(
            client,
            "c",
            list(make_points([(_chunk(), _vector())], license_name="CC BY 4.0", source_url="u")),
        )
        hits = client.query_points("c", query=_vector(), using=DENSE_VECTOR, limit=1).points
        assert hits
        assert hits[0].payload is not None
        assert hits[0].payload["chunk_id"] == "biology-p0133-05"


class TestExistingChunkIds:
    def test_returns_written_ids(self, client: QdrantClient) -> None:
        ensure_collection(client, "c", DIM)
        pairs = [(_chunk(f"biology-p{i:04d}-00", i), _vector(i / 100)) for i in range(5)]
        upsert_points(
            client, "c", list(make_points(pairs, license_name="CC BY 4.0", source_url="u"))
        )
        assert len(existing_chunk_ids(client, "c")) == 5

    def test_is_empty_for_a_missing_collection(self, client: QdrantClient) -> None:
        assert existing_chunk_ids(client, "never-created") == set()

    def test_count_is_zero_for_a_missing_collection(self, client: QdrantClient) -> None:
        assert count_points(client, "never-created") == 0
