"""Claim-level groundedness grading.

The grader answers a narrower question than "is this answer good?": for each
atomic claim the answer makes, is that claim supported by the passages the
system actually retrieved?

**Why claim-level rather than answer-level.** Asking "is this answer grounded?"
returns one number and nothing to act on. A score of 0.6 does not say which part
was unsupported, so a retry has nothing to aim at and can only re-run the same
query and hope. Claim-level grading names the specific claim that failed, and
that claim becomes the query for the next retrieval attempt. The decomposition
is not decoration -- it is the entire corrective signal.

**The grader is a different model from the generator.** A model asked whether
its own output is grounded rates it favourably, which would inflate exactly the
metric this project claims to measure honestly. Generation runs on
``gemini-3.5-flash-lite``, grading on ``gemini-3.1-flash-lite``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class ClaimVerdict(enum.StrEnum):
    """Whether one claim is carried by the retrieved passages."""

    SUPPORTED = "supported"
    """Stated in the context, or a direct paraphrase of something stated."""

    PARTIAL = "partial"
    """Partly carried: the topic is present but a specific detail is not.

    Kept as a third category rather than folded into either neighbour. Forcing a
    binary makes the grader guess, and the guess is not random -- a plausible
    half-supported claim reads as supported far more often than as unsupported,
    which biases groundedness upward precisely where the answer is weakest."""

    UNSUPPORTED = "unsupported"
    """Absent from the context. True in the world is not the question."""


class Claim(BaseModel):
    """One atomic factual assertion extracted from an answer."""

    text: str = Field(description="The claim, as a single self-contained sentence.")
    verdict: ClaimVerdict = Field(description="Whether the passages support it.")
    evidence: str = Field(
        default="",
        description="Short quote from the passages that supports it, empty if none.",
    )
    reason: str = Field(default="", description="One sentence justifying the verdict.")


class GradedAnswer(BaseModel):
    """The grader's full reading of one answer."""

    refuses: bool = Field(
        default=False,
        description=(
            "True if the answer declines to answer -- stating the passages do not "
            "contain the information -- rather than answering the question."
        ),
    )
    claims: list[Claim] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Groundedness:
    """A scored answer, and what to do about it."""

    claims: list[Claim]
    error: str = ""
    refuses: bool = False
    """Whether the draft declines to answer rather than answering.

    Load-bearing, and easy to miss. The generator can already produce a refusal
    on its own -- "the provided passages do not contain..." -- and a refusal is
    *perfectly grounded*, because a statement that the passages lack something
    is supported by passages that lack it. It scores 1.0 and sails through the
    accept threshold.

    Without this flag the loop reports `abstained=False` on exactly the
    questions it exists to decline, its abstention path never executes, and the
    project would claim a corrective loop produced refusals the generator
    produced by itself."""

    @property
    def measured(self) -> bool:
        """Whether grading actually happened.

        An answer whose grading failed is neither grounded nor ungrounded.
        Treating a failed grader call as a score of 0.0 would make the loop
        abstain because of an API error -- a worse failure than the one it
        exists to prevent, and an invisible one.
        """
        return not self.error and bool(self.claims)

    @property
    def score(self) -> float:
        """Supported fraction, counting PARTIAL as half.

        Half is a deliberate choice rather than a tuned one. A partially
        supported claim is genuinely between the two, and any other weight would
        be a knob fitted against the same gold set used to report the result.
        """
        if not self.claims:
            return 0.0
        weights = {
            ClaimVerdict.SUPPORTED: 1.0,
            ClaimVerdict.PARTIAL: 0.5,
            ClaimVerdict.UNSUPPORTED: 0.0,
        }
        return sum(weights[claim.verdict] for claim in self.claims) / len(self.claims)

    @property
    def failed_claims(self) -> list[Claim]:
        """Claims the context did not carry, worst first.

        These are what a retry searches for. An answer-level score could not
        produce this list, which is the reason grading is claim-level at all.
        """
        order = {ClaimVerdict.UNSUPPORTED: 0, ClaimVerdict.PARTIAL: 1}
        return sorted(
            (c for c in self.claims if c.verdict is not ClaimVerdict.SUPPORTED),
            key=lambda c: order[c.verdict],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "claims": len(self.claims),
            "supported": sum(1 for c in self.claims if c.verdict is ClaimVerdict.SUPPORTED),
            "partial": sum(1 for c in self.claims if c.verdict is ClaimVerdict.PARTIAL),
            "unsupported": sum(1 for c in self.claims if c.verdict is ClaimVerdict.UNSUPPORTED),
            "refuses": self.refuses,
            "measured": self.measured,
            "error": self.error,
        }


GRADER_PROMPT = """\
You are checking whether an answer is supported by the passages it was given.

PASSAGES
{context}

ANSWER
{answer}

Break the answer into its atomic factual claims -- one assertion each, phrased so \
it stands alone. Ignore hedges, restatements of the question, and citation \
markers.

For each claim, decide whether THE PASSAGES ABOVE support it:

- supported: stated in the passages, or a direct paraphrase of something stated.
- partial: the topic is present but a specific detail of the claim is not.
- unsupported: absent from the passages.

Judge support, not truth. A claim that is correct in the world but not present \
in these passages is unsupported. That distinction is the reason this check \
exists and the most important thing to get right.

Also set `refuses`: true if the answer DECLINES to answer -- saying the passages \
do not contain the information, or that it cannot be determined from them -- \
rather than answering the question.

A refusal is trivially well supported, because noting an absence is carried by \
passages that lack the thing. The claim scores alone therefore cannot tell a \
refusal apart from a good answer, and this flag is the only thing that can.
"""


def grade_answer(
    llm: Any,
    *,
    answer: str,
    contexts: list[str],
    model: str,
) -> Groundedness:
    """Grade an answer claim by claim against its retrieved context.

    Args:
        llm: Gemini client.
        answer: The draft answer.
        contexts: Passage texts placed in the prompt that produced it.
        model: Grader model id. Must differ from the generation model.

    Returns:
        A :class:`Groundedness` whose ``measured`` flag is False when grading
        could not be performed. Callers must not read that as an ungrounded
        answer.
    """
    if not answer.strip():
        return Groundedness(claims=[], error="empty answer")
    if not contexts:
        return Groundedness(claims=[], error="no context to grade against")

    joined = "\n\n".join(f"--- passage {i} ---\n{c}" for i, c in enumerate(contexts, 1))
    try:
        response = llm.models.generate_content(
            model=model,
            contents=GRADER_PROMPT.format(context=joined, answer=answer),
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": GradedAnswer,
            },
        )
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never fatal
        return Groundedness(claims=[], error=f"{type(exc).__name__}: {exc}"[:200])

    parsed = getattr(response, "parsed", None)
    if not isinstance(parsed, GradedAnswer):
        text = getattr(response, "text", None)
        if not text:
            return Groundedness(claims=[], error="empty grader response")
        try:
            parsed = GradedAnswer.model_validate_json(text)
        except Exception as exc:  # noqa: BLE001
            return Groundedness(
                claims=[], error=f"unparseable grader response: {type(exc).__name__}"
            )

    if not parsed.claims:
        return Groundedness(claims=[], error="grader extracted no claims")
    return Groundedness(claims=parsed.claims, refuses=parsed.refuses)
