"""Screening of user input.

Genuinely untrusted: anyone can type anything into the demo. The attack worth
stopping is an attempt to discard the grounding rules -- answer without
citations, ignore the corpus, or disclose the system prompt.

Blocking is the right response here, rather than sanitising. A question
containing "ignore your previous instructions" is not a biology question with an
unfortunate phrase in it; stripping the phrase and answering the remainder would
be doing an attacker's editing for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vidyarag.guard.patterns import Category, Detection, scan

INPUT_CATEGORIES = (
    Category.INSTRUCTION_OVERRIDE,
    Category.ROLE_HIJACK,
    Category.PROMPT_EXTRACTION,
)

REFUSAL = (
    "That request looks like an attempt to change how I work rather than a "
    "question about the textbooks. Ask me something from the biology or "
    "anatomy material and I'll answer it with citations."
)


@dataclass(frozen=True, slots=True)
class InputVerdict:
    """Whether a question may proceed, and why not if it may not."""

    blocked: bool
    detections: list[Detection] = field(default_factory=list)

    @property
    def categories(self) -> list[str]:
        return sorted({d.category.value for d in self.detections})

    @property
    def reason(self) -> str:
        if not self.blocked:
            return ""
        return "matched: " + ", ".join(self.categories)


def screen_input(question: str) -> InputVerdict:
    """Screen a user question before it reaches retrieval or generation.

    Runs before retrieval on purpose: a blocked question should cost nothing.
    Embedding and searching first would spend the work anyway, and on a rate
    limited free tier that is quota an attacker can burn for free.
    """
    detections = scan(question, INPUT_CATEGORIES)
    return InputVerdict(blocked=bool(detections), detections=detections)
