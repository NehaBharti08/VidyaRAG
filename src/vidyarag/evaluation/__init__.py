"""Evaluation harness.

Code lives here rather than in a top-level ``eval/`` package, which would
shadow the ``eval`` builtin. ``eval/`` on disk holds only data: the gold set
and committed run results.
"""

from vidyarag.evaluation.goldset import (
    GoldQuestion,
    QuestionType,
    load_goldset,
    summarise_goldset,
)

__all__ = [
    "GoldQuestion",
    "QuestionType",
    "load_goldset",
    "summarise_goldset",
]
