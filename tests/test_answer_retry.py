"""Transient failures during answering must not discard a whole run.

A decompose ablation lost 11 of 58 questions and was correctly ruled invalid.
Every failure was transient -- 8 rate limits and 3 server 503s -- and none was
retried, because the answering loop recorded a failure on the first exception.
An hour of work was thrown away by blips that would have succeeded on a second
attempt.
"""

from __future__ import annotations

from typing import Any

import pytest

from vidyarag.evaluation import runner as R
from vidyarag.evaluation.goldset import GoldQuestion, Provenance, QuestionType


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(R.time, "sleep", lambda _s: None)


def _question() -> GoldQuestion:
    return GoldQuestion(
        id="fact-001",
        question="q?",
        type=QuestionType.FACTUAL,
        provenance=Provenance.LLM_DRAFTED_HUMAN_VERIFIED,
        reference="ref",
        gold_chunk_ids=["c1"],
    )


class _Answer:
    def __init__(self) -> None:
        from vidyarag.observe.trace import QueryTrace

        self.text = "an answer"
        self.citations: list[Any] = []
        self.retrieved: list[Any] = []
        self.trace = QueryTrace(query="q?")


class _Pipeline:
    """Fails with the given errors, then succeeds."""

    def __init__(self, *errors: Exception) -> None:
        self._errors = list(errors)
        self.calls = 0

    def answer(self, _question: str) -> Any:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return _Answer()


class TestTransientRetry:
    def test_a_rate_limit_is_retried_and_succeeds(self) -> None:
        pipeline = _Pipeline(RuntimeError("429 RESOURCE_EXHAUSTED"))
        result, _ = R._answer_one(pipeline, _question())  # type: ignore[arg-type]
        assert result.error is None
        assert result.answer == "an answer"
        assert pipeline.calls == 2
        assert result.retries == 1

    def test_a_server_fault_is_retried(self) -> None:
        pipeline = _Pipeline(RuntimeError("503 UNAVAILABLE"))
        result, _ = R._answer_one(pipeline, _question())  # type: ignore[arg-type]
        assert result.error is None
        assert pipeline.calls == 2

    def test_retries_are_bounded(self) -> None:
        """Persistent failure must still end, and end as a recorded failure."""
        pipeline = _Pipeline(*[RuntimeError("503 UNAVAILABLE")] * 10)
        result, _ = R._answer_one(pipeline, _question())  # type: ignore[arg-type]
        assert result.error is not None
        assert pipeline.calls == R.ANSWER_ATTEMPTS

    def test_a_permanent_error_is_not_retried(self) -> None:
        """Retrying a malformed request just spends quota to fail again."""
        pipeline = _Pipeline(*[ValueError("400 INVALID_ARGUMENT: bad schema")] * 5)
        result, _ = R._answer_one(pipeline, _question())  # type: ignore[arg-type]
        assert result.error is not None
        assert pipeline.calls == 1

    def test_success_first_time_does_not_retry_or_report_retries(self) -> None:
        pipeline = _Pipeline()
        result, _ = R._answer_one(pipeline, _question())  # type: ignore[arg-type]
        assert pipeline.calls == 1
        assert result.retries == 0
        assert result.error is None

    def test_error_is_cleared_after_a_successful_retry(self) -> None:
        """A stale error field would count a recovered question as failed."""
        pipeline = _Pipeline(RuntimeError("429"))
        result, _ = R._answer_one(pipeline, _question())  # type: ignore[arg-type]
        assert result.error is None
