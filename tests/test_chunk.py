"""Chunking tests.

The two properties that matter downstream are asserted here: chunks never cut
a sentence in half, and every chunk carries metadata true enough to cite. A
chunk that loses its section or lands on the wrong page produces a citation
that looks authoritative and points nowhere.
"""

from __future__ import annotations

import pytest

from vidyarag.ingest.chunk import chunk_pages, split_sentences
from vidyarag.ingest.parse import PageText


def _page(
    text: str,
    *,
    page: int = 10,
    label: str | None = "1",
    section: str | None = "1.1. Membranes",
) -> PageText:
    return PageText(
        book_slug="biology",
        book_title="Biology",
        page=page,
        label=label,
        chapter="Chapter 1. Cells",
        section=section,
        text=text,
    )


SENTENCE = "Cells convert glucose into usable chemical energy through respiration. "


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self) -> None:
        assert split_sentences("One thing. Two things! Three?") == [
            "One thing.",
            "Two things!",
            "Three?",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "See Fig. 3 for the pathway.",
            "Enzymes, e.g. amylase, act quickly.",
            "Compare with prokaryotes, i.e. bacteria.",
            "Reported by Chen et al. 2019 in detail.",
        ],
    )
    def test_does_not_split_at_abbreviations(self, text: str) -> None:
        """A fragment ending at 'Fig.' would embed as a meaningless stub."""
        assert split_sentences(text) == [text]

    def test_splits_on_paragraph_breaks(self) -> None:
        assert len(split_sentences("First para\n\nSecond para")) == 2

    def test_handles_typographic_quotes(self) -> None:
        # OpenStax sets curly quotes; an ASCII-only rule would never split here.
        result = split_sentences("He said “yes.” Then she left.")
        assert len(result) == 2


class TestChunkPages:
    def test_respects_the_token_budget(self) -> None:
        chunks = list(chunk_pages([_page(SENTENCE * 200)], chunk_size=256, chunk_overlap=32))
        assert len(chunks) > 1
        # Packing is sentence-aligned, so a chunk may overshoot by less than one
        # sentence, but never by a whole one.
        assert all(c.token_count <= 256 + 32 for c in chunks)

    def test_never_cuts_mid_sentence(self) -> None:
        chunks = list(chunk_pages([_page(SENTENCE * 100)], chunk_size=128, chunk_overlap=16))
        for chunk in chunks:
            assert chunk.text.endswith((".", "!", "?"))

    def test_preserves_metadata_on_every_chunk(self) -> None:
        chunks = list(chunk_pages([_page(SENTENCE * 60)], chunk_size=128, chunk_overlap=16))
        assert chunks
        for chunk in chunks:
            assert chunk.book_slug == "biology"
            assert chunk.book_title == "Biology"
            assert chunk.chapter == "Chapter 1. Cells"
            assert chunk.section == "1.1. Membranes"
            assert chunk.page_start == 10

    def test_chunks_do_not_span_sections(self) -> None:
        """A chunk straddling two sections would carry a false citation."""
        pages = [
            _page(SENTENCE * 3, page=1, section="1.1. Membranes"),
            _page(SENTENCE * 3, page=2, section="1.2. Organelles"),
        ]
        chunks = list(chunk_pages(pages, chunk_size=4096, chunk_overlap=64))
        assert {c.section for c in chunks} == {"1.1. Membranes", "1.2. Organelles"}

    def test_records_the_page_range_it_covers(self) -> None:
        pages = [
            _page(SENTENCE * 30, page=41, label="29"),
            _page(SENTENCE * 30, page=42, label="30"),
        ]
        chunks = list(chunk_pages(pages, chunk_size=4096, chunk_overlap=0))
        assert len(chunks) == 1
        assert (chunks[0].page_start, chunks[0].page_end) == (41, 42)

    def test_cites_the_printed_page_not_the_pdf_index(self) -> None:
        """PDF page 133 of Biology is printed page 121; citing 133 misleads."""
        chunks = list(chunk_pages([_page(SENTENCE * 5, page=133, label="121")]))
        assert chunks[0].printed_page == "121"
        assert chunks[0].citation == "Biology, 1.1. Membranes, p.121"

    def test_falls_back_to_chapter_when_section_is_missing(self) -> None:
        chunks = list(chunk_pages([_page(SENTENCE * 5, section=None)]))
        assert chunks[0].citation == "Biology, Chapter 1. Cells, p.1"

    def test_chunk_ids_are_unique_and_stable(self) -> None:
        pages = [_page(SENTENCE * 80)]
        first = [c.chunk_id for c in chunk_pages(pages, chunk_size=128, chunk_overlap=16)]
        second = [c.chunk_id for c in chunk_pages(pages, chunk_size=128, chunk_overlap=16)]
        assert first == second
        assert len(set(first)) == len(first)

    def test_overlap_repeats_context_between_chunks(self) -> None:
        chunks = list(chunk_pages([_page(SENTENCE * 40)], chunk_size=128, chunk_overlap=48))
        assert len(chunks) > 1
        tail = chunks[0].text.split(".")[-2].strip()
        assert tail in chunks[1].text

    def test_oversized_sentence_is_emitted_rather_than_dropped(self) -> None:
        giant = "word " * 400 + "."
        chunks = list(chunk_pages([_page(giant)], chunk_size=64, chunk_overlap=8))
        assert len(chunks) == 1
        assert chunks[0].token_count > 64

    def test_rejects_overlap_that_would_stall_the_window(self) -> None:
        with pytest.raises(ValueError, match="must be <"):
            list(chunk_pages([_page(SENTENCE)], chunk_size=64, chunk_overlap=64))

    def test_empty_input_yields_nothing(self) -> None:
        assert list(chunk_pages([])) == []
