"""The corrective self-check loop.

Grades a draft claim by claim, then accepts it, retries against the claims that
failed, or withholds it. The baseline already declines unanswerable questions
in prose; this package turns those refusals, and drafts the passages do not
support, into an explicit, citation-free abstention a caller can detect without
parsing text. See ``correct/loop.py`` for why an earlier version of this
docstring said otherwise.
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
