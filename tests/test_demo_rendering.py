"""What the demo tells a visitor about an answer.

The panel under every answer is the project's most-read output: it is where a
reader decides how much to trust what they just read. It said "Self-check
passed" for any non-abstaining answer in a corrective profile -- including one
whose grading call had returned 503 and had therefore been checked by nothing.
Observed live, not reasoned about.

These tests import the demo module directly, so the strings a visitor sees are
covered by the suite rather than by having looked at the page once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from vidyarag.correct.grader import Claim, ClaimVerdict, Groundedness
from vidyarag.correct.loop import Attempt, LoopOutcome
from vidyarag.correct.policy import Decision
from vidyarag.observe.trace import QueryTrace
from vidyarag.pipeline import Answer, SelfCheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
app = pytest.importorskip("app", reason="the demo module needs gradio installed")


def _answer(self_check: SelfCheck, *, graded: bool = True, blocked: bool = False) -> Answer:
    trace = QueryTrace(query="q", profile="guarded")
    trace.wall_ms = 1234.0
    trace.record("retrieve", 100.0)
    trace.record("generate", 900.0, depth=1)
    trace.add_usage("gemini-3.5-flash-lite", 2000, 60, "generation")
    trace.add_usage("gemini-3.1-flash-lite", 2500, 600, "grading")
    outcome = LoopOutcome(
        answer="text",
        abstained=self_check is SelfCheck.ABSTAINED,
        attempts=[
            Attempt(
                query="q",
                answer="text",
                groundedness=(
                    Groundedness([Claim(text="c", verdict=ClaimVerdict.SUPPORTED)])
                    if graded
                    else Groundedness([], error="503 UNAVAILABLE")
                ),
                decision=Decision.ACCEPT,
            )
        ],
    )
    trace.corrective = outcome.as_dict()
    trace.abstained = outcome.abstained
    return Answer(
        question="q",
        text="An answer.",
        citations=[],
        retrieved=[],
        trace=trace,
        grounded=not blocked,
        self_check=self_check,
        blocked=blocked,
    )


class TestSelfCheckIsReportedHonestly:
    def test_a_failed_grading_call_is_not_reported_as_a_pass(self) -> None:
        panel = app._render_trace(_answer(SelfCheck.UNAVAILABLE, graded=False))
        assert "Self-check passed" not in panel
        assert "Not verified" in panel

    def test_a_real_pass_says_so(self) -> None:
        panel = app._render_trace(_answer(SelfCheck.PASSED))
        assert "Self-check passed" in panel

    def test_an_abstention_says_so(self) -> None:
        panel = app._render_trace(_answer(SelfCheck.ABSTAINED))
        assert "Abstained" in panel
        assert "Self-check passed" not in panel

    def test_a_profile_without_a_self_check_claims_nothing(self) -> None:
        panel = app._render_trace(_answer(SelfCheck.NOT_RUN))
        assert "Self-check" not in panel
        assert "Not verified" not in panel


class TestPanelNumbers:
    def test_total_is_wall_clock_and_nested_stages_are_marked(self) -> None:
        panel = app._render_trace(_answer(SelfCheck.PASSED))
        assert "total (wall clock)" in panel
        assert "1,234" in panel
        assert "↳ generate" in panel

    def test_grading_tokens_are_shown_separately(self) -> None:
        panel = app._render_trace(_answer(SelfCheck.PASSED))
        assert "↳ grading" in panel
        assert "2,500 in" in panel


class TestErrorsDoNotLeakInternals:
    def test_quota_failure_gets_its_own_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Boom:
            def answer(self, question: str) -> Answer:
                raise RuntimeError("429 RESOURCE_EXHAUSTED for model gemini-3.5-flash-lite")

        monkeypatch.setattr(app, "get_pipeline", Boom)
        text, _, _ = app.ask("a question")
        assert "out of quota" in text
        assert "RESOURCE_EXHAUSTED" not in text

    def test_other_failures_show_no_exception_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Boom:
            def answer(self, question: str) -> Answer:
                raise RuntimeError("connection to internal-host-9 failed")

        monkeypatch.setattr(app, "get_pipeline", Boom)
        text, _, _ = app.ask("a question")
        assert "internal-host-9" not in text
        assert "RuntimeError" not in text
        assert "went wrong" in text
