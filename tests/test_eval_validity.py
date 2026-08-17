"""Run validity: when a partial run may not be quoted as a measurement.

This exists because of a real incident. A 58-question run lost 39 questions to
quota exhaustion and still printed a tidy metrics table reading faithfulness
0.949. The gold set is ordered factual -> multi-hop -> unanswerable, so the
survivors were 17 factual, 0 multi-hop and 2 unanswerable: the score was high
*because* of what was missing.
"""

from __future__ import annotations

from vidyarag.evaluation.goldset import QuestionType
from vidyarag.evaluation.report import render_report
from vidyarag.evaluation.runner import MAX_FAILURE_RATE, EvalRun, SampleResult


def _run(*, ok: int, failed_by_type: dict[QuestionType, int]) -> EvalRun:
    samples = [
        SampleResult(id=f"ok-{i}", question="q", type=QuestionType.FACTUAL, answer="a")
        for i in range(ok)
    ]
    for kind, count in failed_by_type.items():
        samples.extend(
            SampleResult(
                id=f"bad-{kind.value}-{i}",
                question="q",
                type=kind,
                error="ClientError: 429 RESOURCE_EXHAUSTED",
            )
            for i in range(count)
        )
    return EvalRun(
        run_id="20260101T000000Z",
        created_at="2026-01-01T00:00:00+00:00",
        profile="baseline",
        config={"retrieval": {}, "corrective": {}},
        goldset_path="goldset_v1.jsonl",
        goldset_sha256="deadbeef",
        goldset_counts={"factual": ok},
        generation_model="m",
        grader_model="g",
        embedding_model="e",
        python_version="3.11.0",
        samples=samples,
        aggregates={"faithfulness": 0.949, "graded_samples": float(ok)},
    )


class TestValidity:
    def test_complete_run_is_valid(self) -> None:
        assert _run(ok=58, failed_by_type={}).is_valid is True

    def test_the_incident_run_is_invalid(self) -> None:
        run = _run(ok=19, failed_by_type={QuestionType.MULTI_HOP: 18, QuestionType.FACTUAL: 21})
        assert run.failure_rate > MAX_FAILURE_RATE
        assert run.is_valid is False

    def test_a_couple_of_failures_still_counts(self) -> None:
        """One flaky question must not discard an otherwise complete run."""
        run = _run(ok=57, failed_by_type={QuestionType.FACTUAL: 1})
        assert run.is_valid is True

    def test_failures_are_broken_down_by_type(self) -> None:
        """Which questions were lost matters more than how many."""
        run = _run(
            ok=10, failed_by_type={QuestionType.MULTI_HOP: 18, QuestionType.UNANSWERABLE: 10}
        )
        assert run.failures_by_type() == {"multi_hop": 18, "unanswerable": 10}


class TestInvalidReport:
    def test_report_withholds_every_metric(self) -> None:
        """A number that must not be used should not be sitting in a table."""
        run = _run(ok=19, failed_by_type={QuestionType.MULTI_HOP: 18, QuestionType.FACTUAL: 21})
        report = render_report(run)
        assert "INVALID RUN" in report
        assert "0.949" not in report
        assert "## RAGAS metrics" not in report
        assert "Faithfulness" not in report

    def test_report_names_the_lost_categories(self) -> None:
        run = _run(ok=19, failed_by_type={QuestionType.MULTI_HOP: 18, QuestionType.FACTUAL: 21})
        report = render_report(run)
        assert "multi_hop" in report
        assert "18" in report

    def test_valid_report_still_shows_metrics(self) -> None:
        report = render_report(_run(ok=58, failed_by_type={}))
        assert "INVALID RUN" not in report
        assert "## RAGAS metrics" in report
