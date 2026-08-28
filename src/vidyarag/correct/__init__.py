"""The corrective self-check loop.

Baseline abstention recall is 0.000: the pipeline answers every question,
including the twelve it cannot support, because nothing in it can decline. This
package is what has to move that number.
"""

from vidyarag.correct.grader import (
    Claim,
    ClaimVerdict,
    Groundedness,
    grade_answer,
)
from vidyarag.correct.loop import (
    ABSTENTION_TEXT,
    Attempt,
    LoopOutcome,
    reformulate,
    run_corrective_loop,
)
from vidyarag.correct.policy import CorrectivePolicy, Decision

__all__ = [
    "ABSTENTION_TEXT",
    "Attempt",
    "Claim",
    "ClaimVerdict",
    "CorrectivePolicy",
    "Decision",
    "Groundedness",
    "LoopOutcome",
    "grade_answer",
    "reformulate",
    "run_corrective_loop",
]
