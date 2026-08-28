"""The bounded corrective loop: generate, check, retry or refuse.

Baseline abstention recall is 0.000 -- the pipeline answered all twelve
unanswerable questions rather than declining any, because nothing in it could
decline. This module is what has to move that number, and it is the project's
headline claim made testable.

The loop is deliberately small:

1. Generate an answer from the retrieved context.
2. Grade it claim by claim.
3. Accept it, retry against the claims that failed, or abstain.

**Retry re-retrieves; it does not just re-ask.** Regenerating from the same
passages with the same prompt is close to a no-op at temperature 0, and where it
does differ it differs by sampling noise rather than by evidence. The failed
claims are used as the new query, so the second attempt searches for the thing
the first attempt could not support.

**The bound is hard.** Every attempt costs a generation and a grading call, so
an unbounded loop is an unbounded bill and an unbounded latency. Exhausting the
budget without reaching the accept threshold produces an abstention, never a
draft the system already knows is under the bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vidyarag.correct.grader import Groundedness, grade_answer
from vidyarag.correct.policy import CorrectivePolicy, Decision

ABSTENTION_TEXT = (
    "I could not find this in the source material. The retrieved passages from "
    "the textbooks do not contain enough information to answer this question, "
    "and answering from outside them would not be grounded in the sources."
)
"""What the system says when it declines.

Phrased to name the reason rather than merely refuse. "I don't know" invites a
rephrase; saying the corpus does not cover it tells a student to look elsewhere,
which is the actually useful response."""


@dataclass(slots=True)
class Attempt:
    """One pass through generate-and-check."""

    query: str
    answer: str
    groundedness: Groundedness
    decision: Decision

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "decision": self.decision.value,
            **self.groundedness.as_dict(),
        }


@dataclass(slots=True)
class LoopOutcome:
    """The result of running the loop, with everything it did on the way."""

    answer: str
    abstained: bool
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def fired(self) -> bool:
        """Whether the loop did anything beyond a single accepted generation.

        The number worth reporting. A corrective loop that never fires has not
        been shown to work; one that fires constantly is papering over bad
        retrieval rather than correcting it.
        """
        return self.abstained or self.attempt_count > 1

    @property
    def final_score(self) -> float | None:
        if not self.attempts:
            return None
        last = self.attempts[-1].groundedness
        return last.score if last.measured else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempt_count,
            "abstained": self.abstained,
            "fired": self.fired,
            "final_score": self.final_score,
            "trace": [a.as_dict() for a in self.attempts],
        }


def reformulate(question: str, grounded: Groundedness, *, max_claims: int = 2) -> str:
    """Build the next retrieval query from the claims that failed.

    The unsupported claims are exactly what the corpus was not asked for, so
    they are what the next search should look for. The original question is
    kept alongside them: dropping it lets the query drift onto a detail and lose
    the thing actually being asked about.
    """
    failed = grounded.failed_claims[:max_claims]
    if not failed:
        return question
    return " ".join([question, *(claim.text for claim in failed)])


def run_corrective_loop(
    *,
    question: str,
    generate: Any,
    retrieve: Any,
    llm: Any,
    grader_model: str,
    policy: CorrectivePolicy,
) -> LoopOutcome:
    """Generate, check, and retry or abstain.

    Args:
        question: The user's question.
        generate: ``(question, context) -> (answer_text, context_texts)``.
        retrieve: ``(query) -> context`` returning whatever ``generate`` accepts.
        llm: Gemini client for grading.
        grader_model: Grader model id; must differ from the generation model.
        policy: Thresholds and attempt budget.

    Returns:
        A :class:`LoopOutcome` carrying the final answer and every attempt made.
    """
    query = question
    attempts: list[Attempt] = []

    for attempt_number in range(1, policy.max_attempts + 1):
        context = retrieve(query)
        answer_text, context_texts = generate(question, context)

        grounded = grade_answer(
            llm,
            answer=answer_text,
            contexts=context_texts,
            model=grader_model,
        )
        decision = policy.decide(grounded, attempt=attempt_number)
        attempts.append(
            Attempt(
                query=query,
                answer=answer_text,
                groundedness=grounded,
                decision=decision,
            )
        )

        if decision is Decision.ACCEPT:
            return LoopOutcome(answer=answer_text, abstained=False, attempts=attempts)
        if decision is Decision.ABSTAIN:
            return LoopOutcome(answer=ABSTENTION_TEXT, abstained=True, attempts=attempts)

        query = reformulate(question, grounded)

    # Falls through only if the budget ran out on a RETRY decision, which the
    # policy should already have converted to ABSTAIN. Kept explicit rather than
    # trusted: returning a draft here would silently ship an answer known to be
    # below the accept threshold.
    return LoopOutcome(answer=ABSTENTION_TEXT, abstained=True, attempts=attempts)
