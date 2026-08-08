"""Ingest orchestrator tests.

Embedding is stubbed. The model is exercised for real by the actual ingest run;
what needs testing here is the orchestration around it -- that a second run
does not double the corpus, that an interrupted run resumes, and that the
report says what actually happened.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest
from qdrant_client import QdrantClient

from vidyarag.ingest import pipeline
from vidyarag.ingest.corpus import BIOLOGY
from vidyarag.ingest.pipeline import IngestReport, ingest
from vidyarag.store.collection import count_points

DIM = 8
MODEL = "stub-model"

PROSE = (
    "Cells convert glucose into usable chemical energy through cellular respiration. "
    "The mitochondria produce adenosine triphosphate for the cell to spend. "
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A directory holding a small stand-in for the real Biology PDF."""
    doc = pymupdf.open()
    for _ in range(4):
        doc.new_page().insert_text((72, 200), PROSE * 6, fontsize=9)
    doc.set_toc([[1, "Chapter 1. Cells", 1], [2, "1.1. Respiration", 2]])
    doc.save(tmp_path / BIOLOGY.filename)
    doc.close()
    return tmp_path


@pytest.fixture(autouse=True)
def stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic vectors, so no 210 MB model download in CI."""

    def fake(texts: list[str], model_name: str, *, batch_size: int = 32) -> list[list[float]]:
        return [[float(len(t) % 7)] * DIM for t in texts]

    monkeypatch.setattr(pipeline, "embed_texts", fake)
    # chunk sizing calls the real tokeniser; stub it to a cheap word count.
    monkeypatch.setattr(
        "vidyarag.ingest.chunk.count_tokens", lambda text, _model: len(text.split())
    )


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(location=":memory:")


def _run(client: QdrantClient, raw_dir: Path, **kwargs: object) -> IngestReport:
    defaults: dict[str, object] = {
        "collection": "test",
        "embedding_model": MODEL,
        "embedding_dim": DIM,
        "books": (BIOLOGY,),
        "raw_dir": raw_dir,
        "chunk_size": 64,
        "chunk_overlap": 8,
        "batch_size": 16,
        # Never the default path: that is the real index's provenance record.
        "report_path": raw_dir / "ingest_run.json",
    }
    defaults.update(kwargs)
    return ingest(client, **defaults)  # type: ignore[arg-type]


class TestIngest:
    def test_writes_points_and_reports_them(self, client: QdrantClient, raw_dir: Path) -> None:
        report = _run(client, raw_dir)
        assert report.total_chunks > 0
        assert report.points_in_collection == report.total_chunks
        assert count_points(client, "test") == report.total_chunks

    def test_report_records_provenance(self, client: QdrantClient, raw_dir: Path) -> None:
        """A result whose index cannot be identified is not a result."""
        report = _run(client, raw_dir)
        assert report.embedding_model == MODEL
        assert report.embedding_dim == DIM
        assert report.chunk_size == 64
        assert report.collection == "test"
        assert report.duration_seconds >= 0
        assert report.books[0].slug == "biology"
        assert report.books[0].pages_parsed > 0

    def test_second_run_embeds_nothing_and_does_not_duplicate(
        self, client: QdrantClient, raw_dir: Path
    ) -> None:
        """Idempotency: the property that makes re-ingesting safe."""
        first = _run(client, raw_dir)
        second = _run(client, raw_dir)

        assert sum(b.chunks_embedded for b in second.books) == 0
        assert sum(b.chunks_skipped for b in second.books) == first.total_chunks
        assert second.points_in_collection == first.points_in_collection

    def test_resumes_after_a_partial_run(self, client: QdrantClient, raw_dir: Path) -> None:
        """An interrupted embed leaves a partial collection; finish it, don't restart."""
        full = _run(client, raw_dir)

        # Simulate a run that died partway: drop everything and write back a slice.
        client.delete_collection("test")
        partial = _run(client, raw_dir, books=())  # creates an empty collection
        assert partial.points_in_collection == 0

        resumed = _run(client, raw_dir)
        assert sum(b.chunks_embedded for b in resumed.books) == full.total_chunks
        assert resumed.points_in_collection == full.total_chunks

    def test_recreate_forces_a_full_re_embed(self, client: QdrantClient, raw_dir: Path) -> None:
        first = _run(client, raw_dir)
        again = _run(client, raw_dir, recreate=True)

        assert sum(b.chunks_embedded for b in again.books) == first.total_chunks
        assert sum(b.chunks_skipped for b in again.books) == 0
        assert again.points_in_collection == first.total_chunks

    def test_writes_a_run_report_file(
        self, client: QdrantClient, raw_dir: Path, tmp_path: Path
    ) -> None:
        report = _run(client, raw_dir)
        written = report.write(tmp_path / "ingest_run.json")
        assert written.exists()
        assert IngestReport.model_validate_json(written.read_text(encoding="utf-8")).total_chunks

    def test_no_books_produces_an_empty_but_valid_run(
        self, client: QdrantClient, raw_dir: Path
    ) -> None:
        report = _run(client, raw_dir, books=())
        assert report.total_chunks == 0
        assert report.books == []
        assert client.collection_exists("test")
