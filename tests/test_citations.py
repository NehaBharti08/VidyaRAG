"""Citation tests -- load-bearing.

The claim this project makes is that every answer can be traced to a real page
in a real section. That claim is only as good as this file. Two failure modes
matter equally:

* **Inventing** a citation the model never had context for. Worse than no
  citation, because it is more convincing.
* **Losing** a real one. An answer resting on five passages that displays two
  references looks less grounded than it is, and the reference list stops
  matching the text.
"""

from __future__ import annotations

import pytest

from vidyarag.generate.citations import (
    extract_markers,
    format_context,
    render_references,
    resolve_citations,
    strip_invalid_markers,
)
from vidyarag.retrieve.dense import RetrievedChunk


def _chunk(n: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"biology-p{n:04d}-00",
        score=0.9 - n / 100,
        text=f"Body text of passage {n}.",
        citation=f"Biology, {n}.1. A Section, p.{100 + n}",
        book_slug="biology",
        book_title="Biology",
        chapter=f"Chapter {n}",
        section=f"{n}.1. A Section",
        page_start=100 + n,
        page_end=100 + n,
        printed_page=str(88 + n),
        license_name="CC BY 4.0",
        source_url="https://openstax.org/details/books/biology",
    )


CHUNKS = [_chunk(i) for i in range(1, 6)]


class TestExtractMarkers:
    def test_finds_single_markers(self) -> None:
        assert extract_markers("Glucose is polar [1] and large [3].") == [1, 3]

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("Supported by [1, 2].", [1, 2]),
            ("Supported by [1,2,5].", [1, 2, 5]),
            ("Supported by [1; 3].", [1, 3]),
            ("Supported by [ 2 , 4 ].", [2, 4]),
            ("Supported by [1][2].", [1, 2]),
        ],
    )
    def test_finds_grouped_markers(self, answer: str, expected: list[int]) -> None:
        """Gemini writes "[1, 2]" at least as often as "[1][2]".

        A pattern accepting only the single form silently dropped every grouped
        citation -- observed live, where an answer citing five passages
        resolved two.
        """
        assert extract_markers(answer) == expected

    def test_deduplicates_preserving_first_appearance(self) -> None:
        assert extract_markers("[3] then [1] then [3] again") == [3, 1]

    def test_ignores_non_citation_brackets(self) -> None:
        assert extract_markers("The range [a] and [] and [] are not citations.") == []

    def test_no_markers_is_empty(self) -> None:
        assert extract_markers("An answer with no citations at all.") == []


class TestResolveCitations:
    def test_resolves_markers_to_their_chunks(self) -> None:
        citations = resolve_citations("Claim [1] and claim [3].", CHUNKS)
        assert [c.marker for c in citations] == [1, 3]
        assert citations[0].chunk_id == "biology-p0001-00"
        assert citations[1].chunk_id == "biology-p0003-00"

    def test_every_citation_resolves_to_a_real_page_and_section(self) -> None:
        """The assertion the whole project's credibility rests on."""
        citations = resolve_citations("[1] [2] [3] [4] [5]", CHUNKS)
        assert len(citations) == 5
        for citation in citations:
            assert citation.section
            assert citation.printed_page
            assert citation.page_start > 0
            assert citation.book_title
            assert citation.source_url.startswith("https://openstax.org/")

    def test_drops_markers_beyond_the_supplied_context(self) -> None:
        """[9] against five chunks is hallucinated; mapping it anywhere would
        manufacture false provenance."""
        citations = resolve_citations("Real [2] and invented [9].", CHUNKS)
        assert [c.marker for c in citations] == [2]

    def test_drops_zero_and_negative_markers(self) -> None:
        assert resolve_citations("Bad [0] marker.", CHUNKS) == []

    def test_grouped_markers_all_resolve(self) -> None:
        citations = resolve_citations("Supported by [1, 3, 5].", CHUNKS)
        assert [c.marker for c in citations] == [1, 3, 5]

    def test_empty_context_resolves_nothing(self) -> None:
        assert resolve_citations("Claim [1].", []) == []

    def test_citations_are_ordered_by_marker(self) -> None:
        citations = resolve_citations("[4] came before [2].", CHUNKS)
        assert [c.marker for c in citations] == [2, 4]


class TestStripInvalidMarkers:
    def test_keeps_valid_markers(self) -> None:
        assert "[2]" in strip_invalid_markers("A claim [2].", CHUNKS)

    def test_removes_out_of_range_markers(self) -> None:
        """A reader must never be shown a reference that resolves to nothing."""
        cleaned = strip_invalid_markers("Real [1] and invented [9].", CHUNKS)
        assert "[9]" not in cleaned
        assert "[1]" in cleaned

    def test_prunes_invalid_numbers_out_of_a_group(self) -> None:
        """ "[2, 9]" should become "[2]" -- the claim really is supported by 2."""
        cleaned = strip_invalid_markers("Claim [2, 9].", CHUNKS)
        assert "[2]" in cleaned
        assert "9" not in cleaned

    def test_does_not_leave_dangling_punctuation(self) -> None:
        cleaned = strip_invalid_markers("A claim [9].", CHUNKS)
        assert cleaned == "A claim."

    def test_answer_without_markers_is_unchanged(self) -> None:
        text = "An answer with no citations."
        assert strip_invalid_markers(text, CHUNKS) == text


class TestFormatContext:
    def test_numbers_passages_from_one(self) -> None:
        rendered = format_context(CHUNKS[:3])
        assert rendered.startswith("[1] Biology, 1.1. A Section, p.101")
        assert "[3] Biology, 3.1. A Section, p.103" in rendered

    def test_includes_the_body_text(self) -> None:
        assert "Body text of passage 1." in format_context(CHUNKS[:1])

    def test_empty_context_renders_empty(self) -> None:
        assert format_context([]) == ""


class TestRenderReferences:
    def test_carries_licence_and_source(self) -> None:
        """CC BY requires attribution to travel with the content."""
        rendered = render_references(resolve_citations("[1]", CHUNKS))
        assert "CC BY 4.0" in rendered
        assert "https://openstax.org/details/books/biology" in rendered

    def test_no_citations_renders_nothing(self) -> None:
        assert render_references([]) == ""
