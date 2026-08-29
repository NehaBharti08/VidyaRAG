"""Screening of retrieved passages.

The attack most RAG demos miss, and -- stated plainly -- **not a live threat to
this system**. The corpus is two OpenStax PDFs fetched over HTTPS and verified by
SHA-256 at ingest. Nothing user-supplied reaches the index, so no attacker can
place a directive in a passage.

It is built because that property is a fact about today's configuration rather
than a guarantee. The moment the corpus includes anything user-submitted,
scraped, or shared, a retrieved passage becomes an untrusted channel that the
model reads with the same authority as its own instructions -- and a guard added
after the first bad document is added too late.

**Quarantine, not refusal.** A poisoned passage is a property of one chunk, not
of the user's question. Dropping the offending chunk and answering from the rest
is strictly better than refusing: the student asked a legitimate question and
usually four other passages can answer it. Refusing would let anyone who can
write one document deny service to every question that retrieves it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vidyarag.guard.patterns import Category, Detection, scan

if TYPE_CHECKING:  # pragma: no cover
    from vidyarag.retrieve.dense import RetrievedChunk

CONTEXT_CATEGORIES = (
    Category.EMBEDDED_DIRECTIVE,
    Category.INSTRUCTION_OVERRIDE,
)


@dataclass(frozen=True, slots=True)
class ContextVerdict:
    """The surviving passages, and what was removed."""

    kept: list[RetrievedChunk]
    quarantined: list[tuple[str, Detection]] = field(default_factory=list)
    """``(chunk_id, detection)`` for each dropped passage."""

    @property
    def any_quarantined(self) -> bool:
        return bool(self.quarantined)

    def as_dict(self) -> dict[str, object]:
        """Trace-friendly summary. Empty dict when nothing was dropped, so a
        clean query does not carry noise through every log line."""
        if not self.quarantined:
            return {}
        return {
            "quarantined": len(self.quarantined),
            "chunk_ids": [cid for cid, _ in self.quarantined],
            "categories": sorted({d.category.value for _, d in self.quarantined}),
        }


def screen_context(chunks: list[RetrievedChunk]) -> ContextVerdict:
    """Drop retrieved passages that address the assistant instead of the reader.

    Args:
        chunks: Passages selected for the prompt.

    Returns:
        A verdict carrying the passages that survived, in their original order,
        and a record of anything removed.
    """
    kept: list[RetrievedChunk] = []
    quarantined: list[tuple[str, Detection]] = []
    for chunk in chunks:
        detections = scan(chunk.text, CONTEXT_CATEGORIES)
        if detections:
            quarantined.append((chunk.chunk_id, detections[0]))
        else:
            kept.append(chunk)
    return ContextVerdict(kept=kept, quarantined=quarantined)
