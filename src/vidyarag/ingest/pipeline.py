"""Ingestion orchestrator: PDFs in, searchable index out.

Two properties this is built around, both learned from the shape of the job
rather than added for neatness:

* **Idempotent.** Point ids derive from ``chunk_id``, so running twice writes
  the same points twice rather than doubling the corpus. Re-ingesting after a
  parser change is therefore safe and needs no manual cleanup.
* **Resumable.** Embedding 3,600 chunks on CPU takes ~15 minutes. An
  interrupted run leaves a partial collection, so the next run reads back which
  chunk ids already exist and embeds only the remainder. Nothing is recomputed
  that survived.

The run report written at the end is the provenance record for every number
that follows: which books, which embedding model, how many chunks, how long.
An evaluation result whose index cannot be identified is not a result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from vidyarag.ingest.chunk import Chunk, chunk_pages
from vidyarag.ingest.corpus import CORPUS, BookSpec
from vidyarag.ingest.download import RAW_DIR
from vidyarag.ingest.parse import extract_pages
from vidyarag.llm.provider import embed_texts
from vidyarag.settings import REPO_ROOT
from vidyarag.store.collection import (
    count_points,
    ensure_collection,
    existing_chunk_ids,
    make_points,
    upsert_points,
)

RUN_REPORT_PATH = REPO_ROOT / "data" / "ingest_run.json"

T = TypeVar("T")

ProgressFn = Callable[[str, int, int], None]
"""Called as ``(stage, done, total)``."""


class BookIngestStats(BaseModel):
    """Per-book outcome of one ingest run."""

    slug: str
    title: str
    pages_parsed: int
    chunks_created: int
    chunks_embedded: int
    chunks_skipped: int
    """Already present from an earlier run, so not re-embedded."""


class IngestReport(BaseModel):
    """Provenance for one ingest run."""

    started_at: str
    finished_at: str
    duration_seconds: float
    collection: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    total_chunks: int
    points_in_collection: int
    books: list[BookIngestStats] = Field(default_factory=list)

    def write(self, path: Path | None = None) -> Path:
        target = path or RUN_REPORT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return target


def _batched(iterable: Iterator[T], size: int) -> Iterator[list[T]]:
    """Yield successive lists of at most ``size`` items."""
    while batch := list(islice(iterable, size)):
        yield batch


def chunks_for_book(
    spec: BookSpec,
    raw_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> tuple[list[Chunk], int]:
    """Parse and chunk one book.

    Returns:
        ``(chunks, pages_parsed)``.
    """
    pages = list(extract_pages(spec, raw_dir / spec.filename))
    chunks = list(
        chunk_pages(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )
    )
    return chunks, len(pages)


def ingest(
    client: QdrantClient,
    *,
    collection: str,
    embedding_model: str,
    embedding_dim: int,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    books: tuple[BookSpec, ...] = CORPUS,
    raw_dir: Path | None = None,
    batch_size: int = 64,
    recreate: bool = False,
    report_path: Path | None = None,
    progress: ProgressFn | None = None,
) -> IngestReport:
    """Build the searchable index from downloaded PDFs.

    Args:
        client: Connected Qdrant client.
        collection: Target collection name.
        embedding_model: fastembed model id. Runs locally; no API key needed.
        embedding_dim: Vector width, must match the model.
        chunk_size: Target tokens per chunk, in the embedding model's tokens.
        chunk_overlap: Tokens of context repeated between chunks.
        books: Which books to ingest.
        raw_dir: Where the PDFs live. Defaults to ``data/raw``.
        batch_size: Chunks per embedding + upsert batch.
        recreate: Drop the collection first, forcing a full re-embed.
        report_path: Where to write the run report. Defaults to
            ``data/ingest_run.json``. Tests must pass their own path -- writing
            to the default is a side effect on the developer's real corpus
            metadata, and a test run would otherwise erase the provenance of
            the actual index.
        progress: Optional ``(stage, done, total)`` callback.

    Returns:
        An :class:`IngestReport`, also written to ``report_path``.
    """
    started = datetime.now(UTC)
    directory = raw_dir or RAW_DIR

    ensure_collection(client, collection, embedding_dim, recreate=recreate)
    already = set() if recreate else existing_chunk_ids(client, collection)

    stats: list[BookIngestStats] = []
    total_chunks = 0

    for spec in books:
        chunks, pages_parsed = chunks_for_book(
            spec,
            directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )
        total_chunks += len(chunks)

        pending = [c for c in chunks if c.chunk_id not in already]
        embedded = 0

        for batch in _batched(iter(pending), batch_size):
            vectors = embed_texts([c.text for c in batch], embedding_model, batch_size=batch_size)
            points = list(
                make_points(
                    zip(batch, vectors, strict=True),
                    license_name=spec.license_name,
                    source_url=spec.source_url,
                )
            )
            embedded += upsert_points(client, collection, points)
            if progress is not None:
                progress(spec.slug, embedded, len(pending))

        stats.append(
            BookIngestStats(
                slug=spec.slug,
                title=spec.title,
                pages_parsed=pages_parsed,
                chunks_created=len(chunks),
                chunks_embedded=embedded,
                chunks_skipped=len(chunks) - len(pending),
            )
        )

    finished = datetime.now(UTC)
    report = IngestReport(
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        duration_seconds=round((finished - started).total_seconds(), 1),
        collection=collection,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        total_chunks=total_chunks,
        points_in_collection=count_points(client, collection),
        books=stats,
    )
    report.write(report_path)
    return report
