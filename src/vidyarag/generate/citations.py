"""Citations, and proving they are real.

A citation is only worth anything if it resolves. This module turns the model's
inline markers into references backed by chunks that were actually retrieved,
and drops any the model invented.

That last part is the point. A language model asked to cite its sources will
occasionally produce a plausible-looking marker for a passage it never saw --
and a fabricated citation is worse than none, because it is more convincing.
Validating here means the answer a reader sees can only point at text the
system actually retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vidyarag.retrieve.dense import RetrievedChunk

# The model is asked to cite as [1], [2] ... indexing the numbered context it
# was given. Numbers are far more reliable than asking it to echo chunk ids,
# which it mangles.
#
# Grouped markers are matched too. Told to cite several passages for one claim,
# the model writes "[1, 2]" at least as often as "[1][2]" -- and a pattern that
# only accepted the single form silently dropped every grouped citation, so an
# answer resting on five passages displayed two references and looked less
# grounded than it was. Losing a real citation is the same class of failure as
# inventing one: the reference list stops matching the answer.
_MARKER_GROUP = re.compile(r"\[\s*(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\s*\]")
_NUMBER = re.compile(r"\d{1,2}")


@dataclass(frozen=True, slots=True)
class Citation:
    """A validated reference from an answer back to a retrieved chunk."""

    marker: int
    chunk_id: str
    citation: str
    book_title: str
    section: str | None
    printed_page: str | None
    page_start: int
    source_url: str
    license_name: str

    @property
    def label(self) -> str:
        return f"[{self.marker}] {self.citation}"


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as the numbered context block the model sees.

    Each chunk is labelled with the marker the model should use to cite it and
    with its source, so the model can attribute a claim to a specific passage
    rather than to the context as a whole.
    """
    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(f"[{index}] {chunk.citation}\n{chunk.text}")
    return "\n\n".join(blocks)


def extract_markers(answer: str) -> list[int]:
    """Every citation marker in the answer, in order of first appearance.

    Handles both ``[1]`` and grouped forms like ``[1, 2]`` or ``[1; 3]``.
    """
    seen: list[int] = []
    for match in _MARKER_GROUP.finditer(answer):
        for number in _NUMBER.findall(match.group(1)):
            marker = int(number)
            if marker not in seen:
                seen.append(marker)
    return seen


def resolve_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Resolve markers in an answer to the chunks they refer to.

    Markers outside the range of supplied context are dropped rather than
    guessed at: ``[7]`` against five chunks is a hallucinated reference, and
    mapping it to something nearby would manufacture false provenance.

    Args:
        answer: Generated answer text containing ``[n]`` markers.
        chunks: The context that was given to the model, in the same order.

    Returns:
        Validated citations, ordered by marker.
    """
    resolved: list[Citation] = []
    for marker in sorted(extract_markers(answer)):
        if not 1 <= marker <= len(chunks):
            continue
        chunk = chunks[marker - 1]
        resolved.append(
            Citation(
                marker=marker,
                chunk_id=chunk.chunk_id,
                citation=chunk.citation,
                book_title=chunk.book_title,
                section=chunk.section,
                printed_page=chunk.printed_page,
                page_start=chunk.page_start,
                source_url=chunk.source_url,
                license_name=chunk.license_name,
            )
        )
    return resolved


def strip_invalid_markers(answer: str, chunks: list[RetrievedChunk]) -> str:
    """Remove citation markers that point outside the supplied context.

    Leaving them in would show a reader a reference that resolves to nothing.
    """
    limit = len(chunks)

    def keep(match: re.Match[str]) -> str:
        # Prune invalid numbers out of a group rather than dropping the whole
        # marker: "[2, 9]" with three chunks should become "[2]", because the
        # claim really is supported by passage 2.
        valid = [n for n in _NUMBER.findall(match.group(1)) if 1 <= int(n) <= limit]
        return f"[{', '.join(valid)}]" if valid else ""

    cleaned = _MARKER_GROUP.sub(keep, answer)
    # Tidy the double spaces a removed marker leaves behind.
    return re.sub(r" {2,}", " ", cleaned).replace(" .", ".").replace(" ,", ",").strip()


def render_references(citations: list[Citation]) -> str:
    """Render the reference list shown beneath an answer.

    Attribution is repeated here because CC BY requires it to travel with the
    content -- a reader who copies the answer should be able to see where it
    came from and under what licence.
    """
    if not citations:
        return ""
    lines = [
        f"{c.label} ({c.license_name}) {c.source_url}" if c.source_url else c.label
        for c in citations
    ]
    return "\n".join(lines)
