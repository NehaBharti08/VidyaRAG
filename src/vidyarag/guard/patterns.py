"""Injection detection primitives.

Two threat surfaces, and they are not equally real for this system.

**User input** is genuinely untrusted. Anyone can type anything into the demo,
and the obvious attack is to override the system prompt -- get the assistant to
drop its grounding rules, reveal its instructions, or answer outside the corpus
without citations.

**Retrieved context** is the more interesting attack in general and the *less*
real one here, which is worth being straight about. This corpus is two OpenStax
PDFs fetched over HTTPS and verified by SHA-256 at ingest; nothing user-supplied
reaches the index. Retrieved-content injection matters for systems indexing user
submissions, scraped pages or shared drives, where a passage can carry
instructions the model reads as if they came from its operator.

So the context guard is defence in depth against a corpus that is currently
trusted, not mitigation of a live hole. It is built and tested anyway because
that property is a fact about today's configuration rather than a guarantee, and
a guard added after the first bad document is added too late.

**The design constraint is false positives.** A textbook is full of imperative
prose -- "Note that", "Consider the following", "Recall from Chapter 3". A guard
that fires on ordinary pedagogy is worse than no guard: it suppresses real
answers and trains whoever maintains it to ignore the alarm. Every pattern here
is anchored to something a textbook does not say to a reader, and the
false-positive rate is measured against all 3,608 real corpus chunks rather than
asserted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Category(StrEnum):
    """What kind of manipulation was detected."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    """Attempts to discard the operator's instructions."""

    ROLE_HIJACK = "role_hijack"
    """Attempts to reassign the assistant's identity or rules."""

    PROMPT_EXTRACTION = "prompt_extraction"
    """Attempts to make the system reveal its own instructions."""

    EMBEDDED_DIRECTIVE = "embedded_directive"
    """Retrieved text addressing the assistant rather than the reader.

    The signature of retrieved-content injection: a passage that stops
    describing biology and starts issuing orders."""


@dataclass(frozen=True, slots=True)
class Detection:
    """One matched pattern."""

    category: Category
    matched: str
    """The offending span, truncated. Kept so a block can be explained rather
    than merely asserted -- an unexplained refusal is indistinguishable from a
    bug to the person who hit it."""


# Anchored on phrasing that means "disregard your instructions". "ignore" and
# "previous" are common in prose; "ignore ... previous instructions" is not.
_INSTRUCTION_OVERRIDE = (
    r"\b(?:ignore|disregard|forget|override|bypass)\b[^.?!\n]{0,40}"
    r"\b(?:previous|prior|above|earlier|initial|original|all)\b[^.?!\n]{0,20}"
    r"\b(?:instruction|prompt|rule|direction|constraint|guideline)s?\b",
    r"\b(?:ignore|disregard)\b[^.?!\n]{0,20}\b(?:the\s+)?(?:system|developer)\s+"
    r"(?:prompt|message|instruction)s?\b",
)

_ROLE_HIJACK = (
    r"\byou\s+are\s+now\b[^.?!\n]{0,60}",
    r"\b(?:act|behave|respond)\s+as\s+(?:if\s+you\s+are\s+|though\s+you\s+are\s+|a\s+|an\s+)"
    r"[^.?!\n]{0,50}\b(?:unrestricted|unfiltered|jailbroken|dan|developer\s+mode)\b",
    r"\bpretend\s+(?:that\s+)?you\s+(?:are|have|can)\b[^.?!\n]{0,50}",
    r"\bfrom\s+now\s+on[^.?!\n]{0,20}\byou\s+(?:will|must|should)\b",
    r"\b(?:enter|enable|activate)\s+(?:developer|debug|god|dan)\s+mode\b",
)

_PROMPT_EXTRACTION = (
    r"\b(?:repeat|print|show|reveal|output|display|reproduce)\b[^.?!\n]{0,30}"
    r"\b(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|rules?)\b",
    r"\bwhat\s+(?:are|were)\s+your\s+(?:original\s+|initial\s+|system\s+)?"
    r"(?:instructions?|rules?|prompt)\b",
    r"\brepeat\s+(?:everything|all)\s+(?:above|before)\b",
)

# For retrieved text only. A textbook explains; it does not address an assistant.
# Each pattern needs an explicit assistant-directed marker, because imperatives
# alone ("Note that mitosis...") are ordinary pedagogy.
_EMBEDDED_DIRECTIVE = (
    # A bare role marker is NOT the signal. Measured against the real corpus,
    # `^SYSTEM:` matched three chapter headings -- "THE CARDIOVASCULAR SYSTEM:
    # BLOOD" wrapped so SYSTEM landed at a line start. Quarantining the
    # cardiovascular chapters to catch a hypothetical injection is a bad trade,
    # particularly for a guard protecting a corpus nobody can write to.
    #
    # So the marker must be followed by directive language on the same line. An
    # injection that merely asserts a fact under a SYSTEM: label is not caught;
    # one that issues an order is, and the order is what makes it dangerous.
    r"^\s*(?:system|assistant)\s*:\s*(?=[^\n]{0,80}\b(?:ignore|disregard|instead"
    r"|you\s+must|you\s+should|do\s+not|reply|respond|output|say|answer)\b)",
    r"\b(?:AI|assistant|model|chatbot|LLM)\b[^.?!\n]{0,20}"
    r"\b(?:you\s+must|you\s+should|please)\s+(?:ignore|disregard|instead|now)\b",
    r"\bwhen\s+(?:answering|asked|responding)[^.?!\n]{0,40}\b(?:say|reply|respond|output|tell)\b"
    r"[^.?!\n]{0,40}\binstead\b",
    r"\bdo\s+not\s+(?:cite|mention|reference)\s+(?:this|the)\s+(?:source|passage|document)\b",
    # Crosses a sentence boundary deliberately: the canonical phrasing is
    # "End of context. New instructions:", and excluding "." made this
    # pattern unable to match its own motivating example.
    r"\b(?:end|ignore)\s+of?\s*(?:context|document|passage)\b[^\n]{0,40}"
    r"\b(?:new|following)\s+instructions?\b",
)

_COMPILED: dict[Category, tuple[re.Pattern[str], ...]] = {
    Category.INSTRUCTION_OVERRIDE: tuple(re.compile(p, re.I | re.M) for p in _INSTRUCTION_OVERRIDE),
    Category.ROLE_HIJACK: tuple(re.compile(p, re.I | re.M) for p in _ROLE_HIJACK),
    Category.PROMPT_EXTRACTION: tuple(re.compile(p, re.I | re.M) for p in _PROMPT_EXTRACTION),
    Category.EMBEDDED_DIRECTIVE: tuple(re.compile(p, re.I | re.M) for p in _EMBEDDED_DIRECTIVE),
}

MAX_MATCH_CHARS = 120


def scan(text: str, categories: tuple[Category, ...]) -> list[Detection]:
    """Find every matching category in ``text``.

    Returns one detection per category rather than the first overall: a report
    saying "blocked" is far less useful than one saying which of three distinct
    things was attempted.
    """
    found: list[Detection] = []
    for category in categories:
        for pattern in _COMPILED[category]:
            match = pattern.search(text)
            if match:
                span = " ".join(match.group(0).split())[:MAX_MATCH_CHARS]
                found.append(Detection(category=category, matched=span))
                break
    return found
