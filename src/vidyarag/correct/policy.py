"""What to do with a groundedness score.

Kept apart from both the grader and the loop so the thresholds live in one
readable place. They are the most consequential numbers in the system: they
decide when it answers, when it tries again, and when it refuses. Burying them
inside a control-flow function would make them look like implementation detail
rather than the policy choice they are.

**These are starting points, tuned against the gold set in Phase 5, not
constants asserted from intuition.** The sweep and its results are in
docs/EVALUATION.md.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from vidyarag.correct.grader import Groundedness


class Decision(enum.StrEnum):
    """What the loop should do next."""

    ACCEPT = "accept"
    """Grounded enough to return."""

    RETRY = "retry"
    """Partly grounded, and the failed claims say what to search for.

    Worth another attempt precisely because the failure is specific: a claim the
    context did not carry is a query the retriever has not yet been asked."""

    ABSTAIN = "abstain"
    """Too little support to return anything, and no reason to expect a retry to
    help. Saying so is the correct answer, not a failure to produce one."""


@dataclass(frozen=True, slots=True)
class CorrectivePolicy:
    """Thresholds governing the self-check loop."""

    accept_threshold: float = 0.8
    abstain_threshold: float = 0.5
    max_attempts: int = 2

    def decide(self, grounded: Groundedness, *, attempt: int) -> Decision:
        """Choose accept, retry or abstain.

        Args:
            grounded: The grader's verdict on the current draft.
            attempt: 1-based attempt number.

        Returns:
            The next action.
        """
        # An ungraded answer is not an ungrounded one. If grading failed, the
        # loop has no evidence either way, and refusing on the strength of an
        # API error would be a silent, invisible failure -- the system would
        # look appropriately cautious while actually being broken.
        if not grounded.measured:
            return Decision.ACCEPT

        # A refusal is an abstention, whoever produced it. The generator can
        # decline on its own, and such a draft scores ~1.0 because noting an
        # absence is supported by passages that lack the thing. Reading that as
        # "accept" would report abstained=False on precisely the questions the
        # loop exists to decline.
        #
        # It abstains rather than retrying: the generator has already read the
        # context and concluded the answer is not in it, and with no failed
        # claim there is nothing for a reformulated query to search for -- a
        # retry would retrieve the same passages and reach the same conclusion.
        if grounded.refuses:
            return Decision.ABSTAIN

        score = grounded.score
        if score >= self.accept_threshold:
            return Decision.ACCEPT
        if score < self.abstain_threshold:
            return Decision.ABSTAIN
        # Between the thresholds: recoverable in principle, but only while
        # attempts remain. Exhausting them without reaching accept means the
        # evidence is not there, and returning a draft known to be under the
        # bar would defeat the whole check.
        if attempt >= self.max_attempts:
            return Decision.ABSTAIN
        return Decision.RETRY

    def __post_init__(self) -> None:
        if not 0.0 <= self.abstain_threshold <= self.accept_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= abstain <= accept <= 1; "
                f"got abstain={self.abstain_threshold}, accept={self.accept_threshold}"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be at least 1, got {self.max_attempts}")
