"""Parsing tests.

These build small PDFs in-process rather than shipping fixture files or
depending on the real corpus, so CI never needs the 415 MB of textbooks. The
structures exercised here are the ones the real books actually contain --
including the malformed outline entry that Biology really has.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from vidyarag.ingest.corpus import BIOLOGY
from vidyarag.ingest.parse import (
    OutlineEntry,
    build_structure_map,
    clean_text,
    extract_pages,
    is_question_page,
    load_outline,
    normalise_title,
    printed_page_number,
    strip_end_matter,
)


def _make_pdf(path: Path, pages: list[str], toc: list[list[object]]) -> Path:
    """Write a PDF with the given page bodies and outline."""
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 300), body, fontsize=11)
    doc.set_toc(toc)
    doc.save(path)
    doc.close()
    return path


class TestNormaliseTitle:
    def test_collapses_non_breaking_spaces(self) -> None:
        # Exactly what OpenStax outlines contain.
        assert (
            normalise_title("Chapter\xa01.\xa0The Study of Life") == "Chapter 1. The Study of Life"
        )

    def test_collapses_runs_of_whitespace(self) -> None:
        assert normalise_title("  A   B \n C ") == "A B C"


class TestOutlineOrdering:
    def test_entries_are_sorted_by_page(self, tmp_path: Path) -> None:
        """A backwards entry must not survive in document order.

        Biology ends with a `Blank Page` entry pointing at page 6, ~1,450 pages
        before its neighbour. Left unsorted it makes the final range span the
        whole book.
        """
        pdf = _make_pdf(
            tmp_path / "b.pdf",
            ["one", "two", "three", "four"],
            [[1, "Chapter 1. Start", 1], [1, "Chapter 2. Next", 3], [1, "Blank Page", 2]],
        )
        with pymupdf.open(pdf) as doc:
            outline = load_outline(doc)
        assert [e.page for e in outline] == sorted(e.page for e in outline)


class TestStructureMap:
    def test_pages_inherit_their_chapter_and_section(self) -> None:
        outline = [
            OutlineEntry(1, "Chapter 1. Cells", 1),
            OutlineEntry(2, "1.1. Membranes", 2),
            OutlineEntry(2, "1.2. Organelles", 4),
        ]
        mapping = build_structure_map(outline, page_count=5)
        assert mapping[1] == ("Chapter 1. Cells", None)
        assert mapping[2] == ("Chapter 1. Cells", "1.1. Membranes")
        assert mapping[3] == ("Chapter 1. Cells", "1.1. Membranes")
        assert mapping[5] == ("Chapter 1. Cells", "1.2. Organelles")

    def test_excluded_chapter_drops_its_pages(self) -> None:
        outline = [
            OutlineEntry(1, "Preface", 1),
            OutlineEntry(1, "Chapter 1. Cells", 3),
        ]
        mapping = build_structure_map(outline, page_count=4)
        assert 1 not in mapping
        assert 2 not in mapping
        assert mapping[3][0] == "Chapter 1. Cells"

    def test_solutions_is_excluded(self) -> None:
        """`Solutions` is OpenStax's answer key -- the most important exclusion."""
        outline = [OutlineEntry(1, "Chapter 1. Cells", 1), OutlineEntry(1, "Solutions", 3)]
        mapping = build_structure_map(outline, page_count=4)
        assert set(mapping) == {1, 2}

    def test_excluded_section_does_not_exclude_its_siblings(self) -> None:
        """Exclusion inherits downwards, never sideways.

        A single running flag would let `Index` swallow every later section in
        the same chapter.
        """
        outline = [
            OutlineEntry(1, "Chapter 9. Ecology", 1),
            OutlineEntry(2, "Index", 2),
            OutlineEntry(2, "9.2. Food Webs", 3),
        ]
        mapping = build_structure_map(outline, page_count=4)
        assert 2 not in mapping
        assert mapping[3] == ("Chapter 9. Ecology", "9.2. Food Webs")

    def test_empty_outline_yields_empty_map(self) -> None:
        assert build_structure_map([], page_count=10) == {}


class TestCleanText:
    def test_rejoins_hyphenated_line_breaks(self) -> None:
        assert clean_text("photosyn-\nthesis occurs") == "photosynthesis occurs"

    def test_keeps_genuine_compounds_intact(self) -> None:
        # Uppercase tail means a real hyphenated term, not a line break.
        assert "Wnt-\nSignal" in clean_text("Wnt-\nSignal")

    def test_collapses_blank_line_runs(self) -> None:
        assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


class TestEndMatter:
    def test_truncates_at_review_questions(self) -> None:
        text = "Real explanation here.\nREVIEW QUESTIONS\n1. What is a cell?\na. thing"
        assert strip_end_matter(text) == "Real explanation here."

    def test_keeps_chapter_summary_and_key_terms(self) -> None:
        """Summaries and definitions answer questions; question sets do not."""
        text = "CHAPTER SUMMARY\nCells are the unit of life.\nKEY TERMS\ncell basic unit"
        assert strip_end_matter(text) == text

    def test_detects_question_continuation_pages(self) -> None:
        assert is_question_page("1. Which? a. one b. two c. three d. four")

    def test_prose_is_not_a_question_page(self) -> None:
        prose = "Vitamins A, D, E, and K are fat-soluble. Glucose enters via facilitated diffusion."
        assert not is_question_page(prose)


class TestExtractPages:
    def test_raises_without_an_outline(self, tmp_path: Path) -> None:
        """No outline means no derivable citation, so fail loudly."""
        pdf = _make_pdf(tmp_path / "no_toc.pdf", ["body text"], [])
        with pytest.raises(ValueError, match="no outline"):
            list(extract_pages(BIOLOGY, pdf))

    def test_attaches_structure_and_book_metadata(self, tmp_path: Path) -> None:
        pdf = _make_pdf(
            tmp_path / "ok.pdf",
            ["Cells are small.", "Membranes are thin."],
            [[1, "Chapter 1. Cells", 1], [2, "1.1. Membranes", 2]],
        )
        pages = list(extract_pages(BIOLOGY, pdf))
        assert [p.page for p in pages] == [1, 2]
        assert pages[1].section == "1.1. Membranes"
        assert pages[1].book_slug == "biology"
        assert pages[1].book_title == "Biology"

    def test_skips_excluded_and_empty_pages(self, tmp_path: Path) -> None:
        pdf = _make_pdf(
            tmp_path / "mixed.pdf",
            ["front matter", "", "Real content."],
            [[1, "Preface", 1], [1, "Chapter 1. Cells", 2]],
        )
        pages = list(extract_pages(BIOLOGY, pdf))
        assert [p.page for p in pages] == [3]


class TestPrintedPageNumber:
    @pytest.mark.parametrize(
        ("head", "expected"),
        [
            ("188 Chapter 6 | Metabolism", "188"),  # verso: number leads
            ("Chapter 28 | Invertebrates 789", "789"),  # recto: number trails
            ("Chapter 6 | Metabolism", None),  # no number at all
        ],
    )
    def test_reads_number_from_running_head(
        self, tmp_path: Path, head: str, expected: str | None
    ) -> None:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 24), head, fontsize=9)  # inside the header band
        page.insert_text((72, 400), "Body text.", fontsize=11)
        path = tmp_path / f"head_{abs(hash(head))}.pdf"
        doc.save(path)
        doc.close()

        with pymupdf.open(path) as reopened:
            assert printed_page_number(reopened[0]) == expected
