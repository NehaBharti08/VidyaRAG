"""Corpus registry tests.

The licence assertions here exist because this repository once published the
wrong one: ATTRIBUTION.md claimed CC BY 4.0 for Biology 2e and Microbiology,
both of which are actually CC BY-NC-SA 4.0. OpenStax relicensed much of its
catalogue at the second edition, so the licence cannot be inferred from a
title. These tests make the corpus fail the build rather than fail quietly.
"""

from __future__ import annotations

import pytest

from vidyarag.ingest.corpus import CC_BY_4_0, CORPUS, get_book


def test_every_indexed_title_is_cc_by() -> None:
    """A NonCommercial or ShareAlike corpus would restrict reuse of an MIT repo."""
    for book in CORPUS:
        assert book.license_name == CC_BY_4_0, f"{book.slug} is not CC BY 4.0"
        assert book.license_url == "https://creativecommons.org/licenses/by/4.0/"


def test_only_first_editions_are_indexed() -> None:
    """Second editions of these titles are NC-SA; pinning the edition matters."""
    for book in CORPUS:
        assert book.edition == "1st", f"{book.slug} must be the CC BY first edition"


def test_slugs_are_unique() -> None:
    slugs = [book.slug for book in CORPUS]
    assert len(set(slugs)) == len(slugs)


def test_every_book_carries_provenance() -> None:
    for book in CORPUS:
        assert book.book_uuid
        assert book.print_isbn_13
        assert book.pdf_url.endswith(".pdf")
        assert book.source_url.startswith("https://openstax.org/")


def test_filename_derives_from_slug() -> None:
    assert get_book("biology").filename == "biology.pdf"


def test_attribution_names_title_licence_and_link() -> None:
    """CC BY requires credit, title, and a link to the free version."""
    attribution = get_book("biology").attribution()
    assert "Biology" in attribution
    assert "CC BY 4.0" in attribution
    assert "https://openstax.org/details/books/biology" in attribution


def test_get_book_rejects_unknown_slugs_helpfully() -> None:
    with pytest.raises(KeyError, match="Available:"):
        get_book("microbiology")  # NC-SA, deliberately not in the corpus
