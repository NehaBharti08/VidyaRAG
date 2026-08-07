"""Corpus acquisition.

Fetches the source PDFs and records enough provenance that any later result can
be traced back to exact bytes. The manifest this writes is the reason an
evaluation number is reproducible: it pins *which* file produced *which* index.

Three properties the implementation cares about:

* **Idempotent.** Re-running with the corpus already present hashes what is on
  disk and skips the network entirely.
* **Resumable.** The books total ~415 MB; a connection that drops at 90%
  resumes via HTTP Range rather than starting over.
* **Atomic.** Bytes land in ``<name>.pdf.part`` and are renamed only once the
  full body has been read, so an interrupted run can never leave a truncated
  file that looks complete to the next stage.

Checksums are *recorded*, not verified against a published digest -- OpenStax
does not publish one. Their value is detecting drift: if a re-download produces
a different hash, the upstream asset changed and every downstream number was
computed against different text.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pymupdf
from pydantic import BaseModel, Field

from vidyarag.ingest.corpus import CORPUS, BookSpec
from vidyarag.settings import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
MANIFEST_NAME = "manifest.json"

_STREAM_CHUNK_BYTES = 1 << 20  # 1 MiB
_HASH_BLOCK_BYTES = 1 << 20
_TIMEOUT = httpx.Timeout(30.0, read=120.0)

ProgressFn = Callable[[str, int, int], None]
"""Called as ``(slug, bytes_so_far, total_bytes)``. ``total`` is 0 if unknown."""


class DownloadRecord(BaseModel):
    """Provenance for one downloaded book."""

    slug: str
    title: str
    edition: str
    filename: str
    source_url: str
    pdf_url: str
    sha256: str
    size_bytes: int
    page_count: int
    license_name: str
    license_url: str
    book_uuid: str
    print_isbn_13: str
    downloaded_at: str

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1_000_000


class Manifest(BaseModel):
    """All provenance for one ingest run."""

    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    books: list[DownloadRecord] = Field(default_factory=list)

    def get(self, slug: str) -> DownloadRecord | None:
        return next((b for b in self.books if b.slug == slug), None)


def _hash_file(path: Path) -> tuple[str, int]:
    """Return ``(sha256_hex, size_bytes)`` for a file, read in blocks."""
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            hasher.update(block)
            size += len(block)
    return hasher.hexdigest(), size


def _page_count(path: Path) -> int:
    """Open the PDF and count pages.

    Doubles as an integrity check: a checksum proves the bytes are unchanged,
    but only opening the document proves they are a readable PDF rather than an
    error page that happened to transfer cleanly.
    """
    with pymupdf.open(path) as doc:
        return int(doc.page_count)


def _stream_to_part(
    spec: BookSpec,
    part: Path,
    progress: ProgressFn | None,
) -> None:
    """Download ``spec.pdf_url`` into ``part``, resuming if it already exists."""
    part.parent.mkdir(parents=True, exist_ok=True)

    # Two passes at most: the second only happens when a stale .part made the
    # first request unsatisfiable, and it runs without a Range header, so it
    # cannot 416 again.
    for _ in range(2):
        resume_from = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

        with httpx.stream(
            "GET",
            spec.pdf_url,
            headers=headers,
            follow_redirects=True,
            timeout=_TIMEOUT,
        ) as response:
            # 416 means the .part is already >= the asset: stale, not partial.
            if response.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE:
                part.unlink(missing_ok=True)
                continue

            # A 200 in reply to a Range request means the server ignored it and
            # is sending the whole body -- so overwrite rather than append.
            if resume_from and response.status_code == httpx.codes.OK:
                resume_from = 0

            response.raise_for_status()

            total = resume_from + int(response.headers.get("content-length", 0))
            written = resume_from

            with part.open("ab" if resume_from else "wb") as handle:
                for block in response.iter_bytes(_STREAM_CHUNK_BYTES):
                    handle.write(block)
                    written += len(block)
                    if progress is not None:
                        progress(spec.slug, written, total)
            return

    raise RuntimeError(f"Could not download {spec.slug}: server rejected both ranged and full GET")


def download_book(
    spec: BookSpec,
    dest_dir: Path | None = None,
    *,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> DownloadRecord:
    """Fetch one book, or verify the copy already on disk.

    Args:
        spec: Which book to fetch.
        dest_dir: Directory for raw PDFs. Defaults to ``data/raw``.
        force: Re-download even if the file is already present.
        progress: Optional ``(slug, done, total)`` callback for long transfers.

    Returns:
        A :class:`DownloadRecord` with the checksum, size, and page count of the
        file now on disk.
    """
    directory = dest_dir or RAW_DIR
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.filename

    if force or not target.exists():
        part = target.with_suffix(target.suffix + ".part")
        _stream_to_part(spec, part, progress)
        part.replace(target)

    digest, size = _hash_file(target)
    return DownloadRecord(
        slug=spec.slug,
        title=spec.title,
        edition=spec.edition,
        filename=spec.filename,
        source_url=spec.source_url,
        pdf_url=spec.pdf_url,
        sha256=digest,
        size_bytes=size,
        page_count=_page_count(target),
        license_name=spec.license_name,
        license_url=spec.license_url,
        book_uuid=spec.book_uuid,
        print_isbn_13=spec.print_isbn_13,
        downloaded_at=datetime.now(UTC).isoformat(),
    )


def download_corpus(
    dest_dir: Path | None = None,
    *,
    books: tuple[BookSpec, ...] = CORPUS,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> Manifest:
    """Fetch every book and write ``manifest.json`` beside them.

    Args:
        dest_dir: Directory for raw PDFs. Defaults to ``data/raw``.
        books: Which books to fetch. Defaults to the full corpus.
        force: Re-download even where files already exist.
        progress: Optional ``(slug, done, total)`` callback.

    Returns:
        The written :class:`Manifest`.
    """
    directory = dest_dir or RAW_DIR
    records = [download_book(spec, directory, force=force, progress=progress) for spec in books]
    manifest = Manifest(books=records)
    write_manifest(manifest, directory)
    return manifest


def write_manifest(manifest: Manifest, dest_dir: Path | None = None) -> Path:
    """Serialise the manifest to ``<dest_dir>/manifest.json``."""
    directory = dest_dir or RAW_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(dest_dir: Path | None = None) -> Manifest | None:
    """Read the manifest, or ``None`` if the corpus has not been fetched."""
    path = (dest_dir or RAW_DIR) / MANIFEST_NAME
    if not path.exists():
        return None
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
