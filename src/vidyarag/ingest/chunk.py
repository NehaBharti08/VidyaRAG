"""Chunking: pages in, retrievable units out.

Chunks are built *within a section*, not across the whole book. A section is
the smallest unit the outline guarantees is about one topic, so a chunk never
straddles a topic boundary and its ``(chapter, section)`` metadata is exactly
true rather than approximately true.

Chunks may still span a page break, which is deliberate: a definition split
across pages 213 and 214 should stay whole, so a chunk records the page range
it covers and cites the page where it starts.

Size is 512 tokens with 64 of overlap, measured with **the embedding model's own
tokeniser** -- which is load-bearing, not pedantry. ``bge-base-en-v1.5`` uses
BERT WordPiece and truncates hard at 512 tokens, while ``cl100k_base`` counts
the same English prose about 7% shorter. Sizing to 512 tiktoken tokens would
therefore hand the embedder ~548 of its own tokens and have the tail of every
full-size chunk silently dropped -- text still shown to the reader and still
named in the citation, but contributing nothing to the vector that retrieved
it. No error, no warning, just quietly worse retrieval.

The trade-off behind 512 is recorded in ``config/default.yaml``; the number
lives there, not here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby

from vidyarag.ingest.parse import PageText
from vidyarag.llm.provider import count_tokens

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Abbreviations that end in a period without ending a sentence. Splitting after
# "e.g." or "Fig." would cut a sentence in half and hand the fragment to the
# embedder as if it were a complete thought.
_ABBREVIATIONS = (
    "e.g",
    "i.e",
    "cf",
    "vs",
    "etc",
    "al",
    "Fig",
    "fig",
    "Eq",
    "No",
    "Dr",
    "Mr",
    "Mrs",
    "Ms",
    "Prof",
    "St",
    "approx",
    "ca",
    "spp",
    "sp",
)
_ABBREV_ALTERNATION = "|".join(re.escape(a) for a in _ABBREVIATIONS)

# Split after . ! ? (optionally closed by a quote or bracket) when followed by
# whitespace and something that can begin a sentence.
#
# The curly quotes are intentional and load-bearing: OpenStax sets typographic
# quotation marks, so a class containing only ASCII ones would fail to split
# after any quoted sentence. RUF001 flags them as visually ambiguous, which is
# true of the character but not a problem inside a character class.
#
# Abbreviations are handled by re-joining afterwards rather than by a negative
# lookbehind here: the alternation has entries of differing length, and Python's
# `re` only accepts fixed-width lookbehind.
# Each lookbehind branch is a fixed width -- one for bare terminal punctuation,
# one for punctuation closed by a quote or bracket -- because `[...]?` inside a
# single lookbehind makes it variable-width, which `re` rejects outright.
_SENTENCE_BOUNDARY = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"'”’)\]]))"  # noqa: RUF001
    r"\s+"
    r"(?=[A-Z“‘(\[\d])"  # noqa: RUF001
)
_ENDS_WITH_ABBREVIATION = re.compile(rf"(?:^|\s)(?:{_ABBREV_ALTERNATION})\.$")

_PAGE_MARKER = re.compile(r"\x00PAGE:(\d+)\x00")
_MARKER_TEMPLATE = "\x00PAGE:{page}\x00"


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit, with everything a citation needs."""

    chunk_id: str
    book_slug: str
    book_title: str
    chapter: str | None
    section: str | None
    page_start: int
    page_end: int
    """Inclusive PDF page range the chunk's text was drawn from."""

    printed_page: str | None
    """Printed page number at ``page_start`` -- what a citation should show."""

    text: str
    token_count: int

    @property
    def citation(self) -> str:
        """Rendered as it appears beneath an answer."""
        where = self.section or self.chapter
        page = self.printed_page or str(self.page_start)
        return f"{self.book_title}, {where}, p.{page}" if where else f"{self.book_title}, p.{page}"


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, conservatively.

    Paragraph breaks are always boundaries; within a paragraph the split is
    regex-based on terminal punctuation, guarded against common abbreviations.

    Args:
        text: Cleaned page text.

    Returns:
        Non-empty sentence strings in order.
    """
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        stripped = paragraph.strip()
        if not stripped:
            continue
        for part in _SENTENCE_BOUNDARY.split(stripped):
            candidate = part.strip()
            if not candidate:
                continue
            # "...shown in Fig. | 3 the pathway..." was split at an abbreviation,
            # not a sentence end; glue it back onto what it belongs to.
            if sentences and _ENDS_WITH_ABBREVIATION.search(sentences[-1]):
                sentences[-1] = f"{sentences[-1]} {candidate}"
            else:
                sentences.append(candidate)
    return sentences


def _section_key(page: PageText) -> tuple[str, str | None, str | None]:
    return (page.book_slug, page.chapter, page.section)


def _tag_pages(pages: Iterable[PageText]) -> str:
    """Join a section's pages, marking where each one starts.

    The markers are stripped before a chunk's text is emitted; they exist only
    so the chunker can report which pages a chunk actually covers.
    """
    return "\n\n".join(_MARKER_TEMPLATE.format(page=p.page) + p.text for p in pages)


def chunk_pages(
    pages: Iterable[PageText],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> Iterator[Chunk]:
    """Turn parsed pages into overlapping, sentence-aligned chunks.

    Args:
        pages: Output of :func:`vidyarag.ingest.parse.extract_pages`, in order.
        chunk_size: Target maximum tokens per chunk, in ``embedding_model``'s
            own tokens -- the model truncates at its sequence limit, so this
            must be measured the way the model measures.
        chunk_overlap: Tokens of trailing context repeated into the next chunk.
        embedding_model: The model whose tokeniser defines a token here.

    Yields:
        :class:`Chunk` records in document order.

    Raises:
        ValueError: If ``chunk_overlap`` is not smaller than ``chunk_size``,
            which would make the window fail to advance.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})")

    for (slug, chapter, section), group in groupby(pages, key=_section_key):
        section_pages = list(group)
        printed = {p.page: p.label for p in section_pages}
        title = section_pages[0].book_title

        sentences = split_sentences(_tag_pages(section_pages))
        yield from _pack(
            sentences,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            slug=slug,
            title=title,
            chapter=chapter,
            section=section,
            printed=printed,
            first_page=section_pages[0].page,
        )


def _pack(
    sentences: list[str],
    *,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    slug: str,
    title: str,
    chapter: str | None,
    section: str | None,
    printed: dict[int, str | None],
    first_page: int,
) -> Iterator[Chunk]:
    """Greedily pack sentences to ``chunk_size``, carrying overlap forward."""
    buffer: list[tuple[str, int, int]] = []  # (sentence, tokens, page it starts on)
    tokens = 0
    page = first_page
    sequence = 0

    def emit(items: list[tuple[str, int, int]], seq: int) -> Chunk:
        pages = [p for _s, _t, p in items]
        text = _PAGE_MARKER.sub("", " ".join(s for s, _t, _p in items)).strip()
        start = min(pages)
        return Chunk(
            chunk_id=f"{slug}-p{start:04d}-{seq:02d}",
            book_slug=slug,
            book_title=title,
            chapter=chapter,
            section=section,
            page_start=start,
            page_end=max(pages),
            printed_page=printed.get(start),
            text=text,
            token_count=count_tokens(text, embedding_model),
        )

    for sentence in sentences:
        for marker in _PAGE_MARKER.finditer(sentence):
            page = int(marker.group(1))
        clean = _PAGE_MARKER.sub("", sentence).strip()
        if not clean:
            continue

        cost = count_tokens(clean, embedding_model)

        # A single sentence longer than the budget cannot be packed; emit it
        # alone rather than dropping it or splitting mid-clause.
        if cost >= chunk_size:
            if buffer:
                yield emit(buffer, sequence)
                sequence += 1
                buffer, tokens = [], 0
            yield emit([(clean, cost, page)], sequence)
            sequence += 1
            continue

        if tokens + cost > chunk_size and buffer:
            yield emit(buffer, sequence)
            sequence += 1
            buffer, tokens = _carry_overlap(buffer, chunk_overlap)

        buffer.append((clean, cost, page))
        tokens += cost

    if buffer:
        yield emit(buffer, sequence)


def _carry_overlap(
    buffer: list[tuple[str, int, int]],
    chunk_overlap: int,
) -> tuple[list[tuple[str, int, int]], int]:
    """Take whole trailing sentences worth up to ``chunk_overlap`` tokens.

    Overlap is measured in whole sentences so the repeated context is readable
    on its own; a token-exact overlap would start the next chunk mid-clause.
    """
    carried: list[tuple[str, int, int]] = []
    total = 0
    for item in reversed(buffer):
        if total + item[1] > chunk_overlap:
            break
        carried.insert(0, item)
        total += item[1]
    return carried, total
