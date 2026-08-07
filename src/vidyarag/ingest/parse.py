"""PDF text extraction with structural metadata.

The job here is not "get the text out" -- it is "get the text out *and* know
where it came from". A citation like ``Biology, S7.3, p.214`` is only
trustworthy if the section and page attached to a chunk were derived from the
document's own structure rather than guessed.

Structure comes from the PDF outline (``doc.get_toc()``), which OpenStax PDFs
carry. That gives real (chapter, section, page) triples. The alternative --
inferring headings from font size -- misfires on pull quotes, figure captions,
and section headers that happen to wrap, and it fails silently.

Layout handling, in order of how much damage each prevents:

* **Header/footer stripping.** Running heads repeat the book title and page
  number on every page. Left in, they appear in the middle of chunk text and
  embed as noise on every single chunk.
* **Column ordering.** OpenStax lays out some pages in two columns. Naive
  top-to-bottom extraction interleaves them mid-sentence, which produces text
  that looks plausible and is actually shredded.
* **De-hyphenation.** Line-broken words ("photosyn-\\nthesis") otherwise survive
  into chunks as two tokens and never match a query.

Front matter, indices, and answer keys are excluded outright: they are dense in
keywords and empty of explanation, so they win retrieval and answer nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from vidyarag.ingest.corpus import BookSpec

# Fraction of page height treated as running header / footer. OpenStax running
# heads sit well inside these bands; body text does not reach them.
_HEADER_BAND = 0.06
_FOOTER_BAND = 0.94

# A page is treated as two-column when the widest horizontal gap between block
# edges exceeds this fraction of page width and falls near the middle.
_COLUMN_GAP_RATIO = 0.04
_COLUMN_CENTRE_TOLERANCE = 0.18

# Outline entries whose title matches are dropped along with their pages.
#
# "Solutions" is OpenStax's name for the answer key -- it is the single most
# important exclusion here. It is ~35 pages of bare answers with no
# explanation, phrased in the same vocabulary as the questions, so it competes
# for exactly the queries the book should answer and satisfies none of them.
_EXCLUDED_TITLE = re.compile(
    r"^\s*(preface|index|solutions?|answer key|the answer key|references"
    r"|appendix|about openstax|table of contents|contents|blank page)\b",
    re.IGNORECASE,
)

# The running head carries the printed page number, on the outer edge -- so it
# leads on verso pages ("188 Chapter 6 | Metabolism") and trails on recto ones
# ("Chapter 28 | Invertebrates 789").
_PAGE_NUMBER_LEADING = re.compile(r"^(\d{1,4})\b")
_PAGE_NUMBER_TRAILING = re.compile(r"\b(\d{1,4})$")

# End-of-chapter question sets. The outline gives these no entry of their own --
# they fall under the trailing "Glossary" heading along with the chapter summary
# and key terms -- so they have to be recognised from the text.
#
# Only the question blocks are dropped. CHAPTER SUMMARY and KEY TERMS stay:
# summaries are condensed explanation and key terms are definitions, both of
# which answer real questions. Review questions answer none -- they are written
# in the same vocabulary as the queries a student would ask, so they compete
# strongly in retrieval and then contain nothing but "a. ... b. ... c. ...".
_END_MATTER_HEADING = re.compile(
    r"(REVIEW QUESTIONS|CRITICAL THINKING QUESTIONS|ART CONNECTION QUESTIONS"
    r"|VISUAL CONNECTION QUESTIONS|INTERACTIVE LINK QUESTIONS)"
)

# Multiple-choice options, for pages that continue a question block with no
# heading of their own.
_CHOICE_OPTION = re.compile(r"\b[a-d]\.\s")
_MIN_OPTIONS_FOR_QUESTION_PAGE = 4

_NBSP = "\xa0"

# "photosyn-\nthesis" -> "photosynthesis", but never "well-\nknown" -> "wellknown"
# for a genuinely hyphenated compound: only rejoin when the tail is lowercase.
_HYPHEN_BREAK = re.compile(r"([A-Za-z])-\n([a-z])")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """One entry from the PDF's own table of contents."""

    level: int
    title: str
    page: int
    """1-based PDF page index where the entry starts."""

    @property
    def is_excluded(self) -> bool:
        return bool(_EXCLUDED_TITLE.match(self.title))


@dataclass(frozen=True, slots=True)
class PageText:
    """Extracted text for one page, with the structure it belongs to."""

    book_slug: str
    book_title: str
    page: int
    """1-based PDF page index -- what a reader gets from a PDF viewer."""

    label: str | None
    """Printed page label, when the PDF declares one. May differ from ``page``."""

    chapter: str | None
    section: str | None
    text: str

    @property
    def citation(self) -> str:
        """Rendered as it would appear beneath an answer."""
        where = self.section or self.chapter or ""
        page = self.label or str(self.page)
        return f"{self.book_title}, {where}, p.{page}".replace(", ,", ",")


def normalise_title(raw: str) -> str:
    """Collapse the non-breaking spaces PDF outlines are full of.

    OpenStax outline titles arrive as ``"Chapter\\xa01.\\xa0The Study of Life"``.
    Left alone, that U+00A0 reaches citations and chunk metadata, where it
    renders as a mojibake box and breaks any string comparison against a title
    typed by a human.
    """
    return " ".join(raw.replace(_NBSP, " ").split())


def load_outline(doc: pymupdf.Document) -> list[OutlineEntry]:
    """Read the PDF outline as ``OutlineEntry`` records.

    Entries are sorted by page. Real books contain stray outline entries that
    point backwards -- Biology ends with a ``Blank Page`` entry pointing at
    page 6, more than 1,400 pages before its neighbour. Left in document order
    that single entry swallows the whole book into one bogus range, so ordering
    is enforced here rather than assumed.

    Args:
        doc: An open PyMuPDF document.

    Returns:
        Entries in page order. Empty if the PDF carries no outline, which is
        the signal that this book needs a different extraction strategy.
    """
    entries: list[OutlineEntry] = []
    for level, title, page in doc.get_toc(simple=True):
        if page < 1:  # PyMuPDF uses -1 for entries with no destination
            continue
        entries.append(
            OutlineEntry(level=int(level), title=normalise_title(str(title)), page=int(page))
        )
    entries.sort(key=lambda entry: (entry.page, entry.level))
    return entries


def build_structure_map(
    outline: list[OutlineEntry],
    page_count: int,
) -> dict[int, tuple[str | None, str | None]]:
    """Map every page to the ``(chapter, section)`` in force on it.

    A page belongs to the most recent outline entry at or before it, which is
    how a reader would attribute it. Level 1 entries are chapters, deeper
    entries are sections; a deeper entry does not clear the chapter above it.

    Args:
        outline: Entries from :func:`load_outline`.
        page_count: Total pages in the document.

    Returns:
        ``{page_number: (chapter, section)}`` for pages that survive exclusion.
        Excluded ranges (front matter, index, answer keys) are absent entirely.
    """
    if not outline:
        return {}

    top_level = min(entry.level for entry in outline)
    boundaries: list[tuple[int, str | None, str | None, bool]] = []
    chapter: str | None = None
    section: str | None = None
    chapter_excluded = False

    for entry in outline:
        if entry.level == top_level:
            chapter = entry.title
            section = None
            chapter_excluded = entry.is_excluded
            excluded = chapter_excluded
        else:
            section = entry.title
            # Exclusion inherits downwards but never sideways: an excluded
            # section must not drag its siblings out with it, which is what
            # tracking a single running flag would do.
            excluded = chapter_excluded or entry.is_excluded
        boundaries.append((entry.page, chapter, section, excluded))

    mapping: dict[int, tuple[str | None, str | None]] = {}
    for index, (start, chap, sect, skip) in enumerate(boundaries):
        end = boundaries[index + 1][0] - 1 if index + 1 < len(boundaries) else page_count
        if skip:
            continue
        for page in range(start, min(end, page_count) + 1):
            mapping[page] = (chap, sect)
    return mapping


def _ordered_blocks(page: pymupdf.Page) -> list[str]:
    """Return the page's text blocks in reading order, headers/footers removed."""
    rect = page.rect
    height = rect.height or 1.0
    width = rect.width or 1.0

    blocks = []
    for x0, y0, x1, y1, text, _block_no, block_type in page.get_text("blocks"):
        if block_type != 0 or not text.strip():  # 0 == text, 1 == image
            continue
        relative_top = (y0 - rect.y0) / height
        relative_bottom = (y1 - rect.y0) / height
        if relative_bottom < _HEADER_BAND or relative_top > _FOOTER_BAND:
            continue
        blocks.append((x0, y0, x1, text))

    if not blocks:
        return []

    split = _column_split(blocks, rect.x0, width)
    if split is None:
        blocks.sort(key=lambda b: (b[1], b[0]))
        return [b[3] for b in blocks]

    left = sorted((b for b in blocks if b[0] < split), key=lambda b: (b[1], b[0]))
    right = sorted((b for b in blocks if b[0] >= split), key=lambda b: (b[1], b[0]))
    return [b[3] for b in left + right]


def _column_split(
    blocks: list[tuple[float, float, float, str]],
    page_left: float,
    width: float,
) -> float | None:
    """Find the x coordinate separating two columns, or ``None`` if single-column.

    Looks for a vertical band that no block crosses, wide enough to be a gutter
    and near enough to the middle to be *the* gutter rather than an indent.
    """
    spans = sorted((x0, x1) for x0, _y0, x1, _text in blocks)
    centre = page_left + width / 2
    reach = spans[0][1]
    best: tuple[float, float] | None = None

    for x0, x1 in spans[1:]:
        gap = x0 - reach
        if gap > 0:
            candidate = reach + gap / 2
            offset = abs(candidate - centre) / width
            wide_enough = gap / width >= _COLUMN_GAP_RATIO
            central_enough = offset <= _COLUMN_CENTRE_TOLERANCE
            if wide_enough and central_enough and (best is None or gap > best[1]):
                best = (candidate, gap)
        reach = max(reach, x1)

    return best[0] if best else None


def clean_text(raw: str) -> str:
    """Normalise extracted text without altering its wording.

    Rejoins hyphenated line breaks, collapses runs of blank lines, and strips
    trailing whitespace. Deliberately conservative: anything cleverer risks
    changing the text a citation claims to quote.
    """
    text = _HYPHEN_BREAK.sub(r"\1\2", raw)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def strip_end_matter(text: str) -> str:
    """Cut a page's text at the first end-of-chapter question heading."""
    match = _END_MATTER_HEADING.search(text)
    return text[: match.start()].strip() if match else text


def is_question_page(text: str) -> bool:
    """True if a page looks like the continuation of a question block.

    Question sets run over page breaks, and the later pages carry no heading --
    just numbered stems and lettered options. Counting options is a blunt test,
    but four or more ``a.``/``b.``/``c.``/``d.`` markers on one page essentially
    never occurs in textbook prose.
    """
    return len(_CHOICE_OPTION.findall(text)) >= _MIN_OPTIONS_FOR_QUESTION_PAGE


def extract_pages(spec: BookSpec, pdf_path: Path) -> Iterator[PageText]:
    """Yield structured text for every in-scope page of a book.

    Args:
        spec: The book being parsed, for attribution metadata.
        pdf_path: Path to the downloaded PDF.

    Yields:
        One :class:`PageText` per page that carries body content and belongs to
        a non-excluded section. Blank pages are skipped.

    Raises:
        ValueError: If the PDF has no usable outline, since every downstream
            citation depends on one.
    """
    with pymupdf.open(pdf_path) as doc:
        outline = load_outline(doc)
        if not outline:
            raise ValueError(
                f"{spec.slug}: PDF carries no outline, so (chapter, section) cannot be "
                "derived. Use the OpenStax structured HTML for this title instead."
            )

        structure = build_structure_map(outline, doc.page_count)

        for index in range(doc.page_count):
            page_number = index + 1
            if page_number not in structure:
                continue
            page = doc[index]
            text = strip_end_matter(clean_text("\n".join(_ordered_blocks(page))))
            if not text or is_question_page(text):
                continue
            chapter, section = structure[page_number]
            yield PageText(
                book_slug=spec.slug,
                book_title=spec.title,
                page=page_number,
                label=printed_page_number(page),
                chapter=chapter,
                section=section,
                text=text,
            )


def printed_page_number(page: pymupdf.Page) -> str | None:
    """Read the printed page number off the running head.

    These PDFs declare no page labels -- ``get_label()`` returns ``""`` on every
    page -- but the number a student would look up is printed in the running
    head, which is stripped from the body text anyway. Recovering it there is
    what makes a citation checkable against a paper copy: PDF page 200 of
    Biology is printed page 188, and citing 200 would send a reader to the
    wrong place.

    Returns:
        The printed number as a string, or ``None`` if the head carries none.
    """
    rect = page.rect
    height = rect.height or 1.0
    for _x0, _y0, _x1, y1, text, _no, block_type in page.get_text("blocks"):
        if block_type != 0 or (y1 - rect.y0) / height >= _HEADER_BAND:
            continue
        head = " ".join(text.split())
        match = _PAGE_NUMBER_LEADING.match(head) or _PAGE_NUMBER_TRAILING.search(head)
        if match:
            return match.group(1)
    return None
