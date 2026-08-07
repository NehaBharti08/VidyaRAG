"""Download tests.

No network: every request is mocked with respx. What is actually being tested
is the behaviour that protects a 415 MB fetch -- idempotency, resume, and the
atomic rename that stops an interrupted run leaving a truncated file the next
stage would happily parse.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pymupdf
import pytest
import respx

from vidyarag.ingest.corpus import BIOLOGY
from vidyarag.ingest.download import (
    Manifest,
    download_book,
    download_corpus,
    load_manifest,
    write_manifest,
)


@pytest.fixture
def pdf_bytes() -> bytes:
    """A tiny but genuinely valid three-page PDF."""
    doc = pymupdf.open()
    for n in range(3):
        doc.new_page().insert_text((72, 300), f"page {n}", fontsize=11)
    data: bytes = doc.tobytes()
    doc.close()
    return data


@respx.mock
def test_downloads_and_records_provenance(tmp_path: Path, pdf_bytes: bytes) -> None:
    respx.get(BIOLOGY.pdf_url).mock(return_value=httpx.Response(200, content=pdf_bytes))

    record = download_book(BIOLOGY, tmp_path)

    assert (tmp_path / "biology.pdf").read_bytes() == pdf_bytes
    assert record.size_bytes == len(pdf_bytes)
    assert record.page_count == 3
    assert len(record.sha256) == 64
    assert record.license_name == "CC BY 4.0"
    assert record.book_uuid == BIOLOGY.book_uuid


@respx.mock
def test_existing_file_is_not_refetched(tmp_path: Path, pdf_bytes: bytes) -> None:
    """Re-running a completed ingest must not touch the network."""
    route = respx.get(BIOLOGY.pdf_url).mock(return_value=httpx.Response(200, content=pdf_bytes))
    (tmp_path / "biology.pdf").write_bytes(pdf_bytes)

    record = download_book(BIOLOGY, tmp_path)

    assert not route.called
    assert record.page_count == 3


@respx.mock
def test_force_refetches_an_existing_file(tmp_path: Path, pdf_bytes: bytes) -> None:
    route = respx.get(BIOLOGY.pdf_url).mock(return_value=httpx.Response(200, content=pdf_bytes))
    (tmp_path / "biology.pdf").write_bytes(b"stale")

    download_book(BIOLOGY, tmp_path, force=True)

    assert route.called
    assert (tmp_path / "biology.pdf").read_bytes() == pdf_bytes


@respx.mock
def test_resumes_from_a_partial_file(tmp_path: Path, pdf_bytes: bytes) -> None:
    """A dropped connection must not restart 279 MB from zero."""
    head, tail = pdf_bytes[:40], pdf_bytes[40:]
    (tmp_path / "biology.pdf.part").write_bytes(head)

    def responder(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == f"bytes={len(head)}-"
        return httpx.Response(206, content=tail)

    respx.get(BIOLOGY.pdf_url).mock(side_effect=responder)

    record = download_book(BIOLOGY, tmp_path)

    assert (tmp_path / "biology.pdf").read_bytes() == pdf_bytes
    assert record.size_bytes == len(pdf_bytes)


@respx.mock
def test_ignored_range_header_restarts_cleanly(tmp_path: Path, pdf_bytes: bytes) -> None:
    """A server answering 200 to a Range request is sending the whole body.

    Appending it to the partial file would silently corrupt the download.
    """
    (tmp_path / "biology.pdf.part").write_bytes(pdf_bytes[:40])
    respx.get(BIOLOGY.pdf_url).mock(return_value=httpx.Response(200, content=pdf_bytes))

    download_book(BIOLOGY, tmp_path)

    assert (tmp_path / "biology.pdf").read_bytes() == pdf_bytes


@respx.mock
def test_stale_partial_triggers_a_full_refetch(tmp_path: Path, pdf_bytes: bytes) -> None:
    """416 means the .part is at least as large as the asset -- it is junk."""
    (tmp_path / "biology.pdf.part").write_bytes(b"x" * (len(pdf_bytes) + 500))
    calls: list[str | None] = []

    def responder(request: httpx.Request) -> httpx.Response:
        calls.append(request.headers.get("Range"))
        if len(calls) == 1:
            return httpx.Response(416)
        return httpx.Response(200, content=pdf_bytes)

    respx.get(BIOLOGY.pdf_url).mock(side_effect=responder)

    download_book(BIOLOGY, tmp_path)

    assert calls[1] is None  # second attempt drops the Range header
    assert (tmp_path / "biology.pdf").read_bytes() == pdf_bytes


@respx.mock
def test_failed_download_leaves_no_usable_file(tmp_path: Path) -> None:
    """A 404 must not produce a .pdf that the parser would then try to open."""
    respx.get(BIOLOGY.pdf_url).mock(return_value=httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        download_book(BIOLOGY, tmp_path)

    assert not (tmp_path / "biology.pdf").exists()


@respx.mock
def test_progress_callback_reports_transfer(tmp_path: Path, pdf_bytes: bytes) -> None:
    respx.get(BIOLOGY.pdf_url).mock(return_value=httpx.Response(200, content=pdf_bytes))
    seen: list[tuple[str, int, int]] = []

    download_book(BIOLOGY, tmp_path, progress=lambda s, d, t: seen.append((s, d, t)))

    assert seen
    assert seen[-1][0] == "biology"
    assert seen[-1][1] == len(pdf_bytes)


@respx.mock
def test_corpus_download_writes_a_manifest(tmp_path: Path, pdf_bytes: bytes) -> None:
    respx.get(BIOLOGY.pdf_url).mock(return_value=httpx.Response(200, content=pdf_bytes))

    manifest = download_corpus(tmp_path, books=(BIOLOGY,))

    assert (tmp_path / "manifest.json").exists()
    reloaded = load_manifest(tmp_path)
    assert reloaded is not None
    assert reloaded.books[0].sha256 == manifest.books[0].sha256
    assert reloaded.get("biology") is not None
    assert reloaded.get("nope") is None


def test_load_manifest_returns_none_before_ingestion(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) is None


def test_manifest_round_trips(tmp_path: Path) -> None:
    written = write_manifest(Manifest(), tmp_path)
    assert written.name == "manifest.json"
    assert load_manifest(tmp_path) is not None
