"""Run validity: when a partial run may not be quoted as a measurement.

This exists because of a real incident. A 58-question run lost 39 questions to
quota exhaustion and still printed a tidy metrics table reading faithfulness
0.949. The gold set is ordered factual -> multi-hop -> unanswerable, so the
survivors were 17 factual, 0 multi-hop and 2 unanswerable: the score was high
*because* of what was missing.
"""

from __future__ import annotations

from pathlib import Path

from vidyarag.evaluation.goldset import QuestionType
from vidyarag.evaluation.report import render_report
from vidyarag.evaluation.runner import (
    MAX_FAILURE_RATE,
    REPORT_ORDER,
    EvalRun,
    SampleResult,
    profiles_with_runs,
)


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


class TestReportProfileDiscovery:
    """`vidyarag report` with no arguments must compare, not self-compare.

    It defaulted to ["baseline"], so the headline command for reproducing the
    README results table rendered a single column of the control group against
    itself -- the one configuration whose numbers the table exists to contrast.
    """

    def test_finds_every_profile_that_has_a_run(self, tmp_path: Path) -> None:
        for name in ("guarded", "baseline", "rerank"):
            (tmp_path / f"{name}__20260101T000000Z.json").write_text("{}", encoding="utf-8")
        assert profiles_with_runs(tmp_path) == ["baseline", "rerank", "guarded"]

    def test_orders_the_control_first_not_alphabetically(self, tmp_path: Path) -> None:
        """Alphabetical order reverses the sequence the ablations were run in."""
        for name in REPORT_ORDER:
            (tmp_path / f"{name}__20260101T000000Z.json").write_text("{}", encoding="utf-8")
        assert profiles_with_runs(tmp_path) == list(REPORT_ORDER)

    def test_unknown_profiles_still_appear(self, tmp_path: Path) -> None:
        """A profile added later must not silently vanish from the comparison."""
        (tmp_path / "baseline__20260101T000000Z.json").write_text("{}", encoding="utf-8")
        (tmp_path / "experimental__20260101T000000Z.json").write_text("{}", encoding="utf-8")
        assert profiles_with_runs(tmp_path) == ["baseline", "experimental"]

    def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert profiles_with_runs(tmp_path / "nope") == []

    def test_the_committed_runs_cover_every_column_in_the_readme(self) -> None:
        """The README table has five columns; all five must be reproducible."""
        assert profiles_with_runs() == list(REPORT_ORDER)
