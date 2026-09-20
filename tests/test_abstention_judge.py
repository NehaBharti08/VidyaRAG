"""The abstention judge, and the truncation that made it lie.

Every committed evaluation ran this judge with ``max_tokens=5``. Measured
against the live endpoint, that returns ``'REF'`` with ``finish_reason='length'``
-- the label truncated mid-word. The old parser tested ``startswith("REFUSED")``,
so every refusal was recorded as an answer, abstention recall read 0.000 for the
profiles with no abstention mechanism, and the corrective loop appeared to
invent the capability from nothing.

Nothing here needs a network. That is the point: the bug lived in a function
whose output was never once compared against a known label, because the only way
to reach it was to spend quota.
"""

from __future__ import annotations

from typing import Any

import pytest

from vidyarag.evaluation.abstention import (
    JUDGE_MAX_TOKENS,
    AbstentionStats,
    JudgeVerdict,
    judge_abstention,
    parse_judge_label,
    summarise_abstention,
)


class FakeChoice:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.message = type("Msg", (), {"content": content})()
        self.finish_reason = finish_reason


class FakeCompletions:
    """Records the request and replays one canned response."""

    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self._content = content
        self._finish_reason = finish_reason
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return type("Resp", (), {"choices": [FakeChoice(self._content, self._finish_reason)]})()


class FakeClient:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.completions = FakeCompletions(content, finish_reason)
        self.chat = type("Chat", (), {"completions": self.completions})()


class ExplodingClient:
    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("503 UNAVAILABLE")

    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": self._Completions()})()


class TestParsing:
    def test_a_truncated_label_is_not_a_verdict(self) -> None:
        """The shipped bug, pinned. 'REF' must never count as 'answered'."""
        verdict = parse_judge_label("REF", finish_reason="length")
        assert verdict.measured is False
        assert verdict.refused is False
        assert "truncated" in verdict.error

    def test_refused(self) -> None:
        verdict = parse_judge_label("REFUSED", finish_reason="stop")
        assert (verdict.refused, verdict.measured) == (True, True)

    def test_answered(self) -> None:
        verdict = parse_judge_label("ANSWERED", finish_reason="stop")
        assert (verdict.refused, verdict.measured) == (False, True)

    def test_case_and_whitespace_are_tolerated(self) -> None:
        assert parse_judge_label("  refused\n").refused is True

    def test_a_label_after_reasoning_is_read(self) -> None:
        """A model that thinks aloud ends on its conclusion."""
        assert parse_judge_label("The answer supplies the fact, so: ANSWERED").refused is False
        assert parse_judge_label("This is not ANSWERED. REFUSED").refused is True

    def test_prose_containing_neither_label_is_unmeasured(self) -> None:
        assert parse_judge_label("I cannot classify this.").measured is False

    @pytest.mark.parametrize("content", ["", None, "   "])
    def test_empty_is_unmeasured(self, content: str | None) -> None:
        assert parse_judge_label(content).measured is False

    def test_a_verdict_has_no_truth_value(self) -> None:
        """``if verdict:`` is the mistake this type exists to prevent."""
        with pytest.raises(TypeError):
            bool(JudgeVerdict(refused=True, measured=True))


class TestJudgeCall:
    async def test_budget_is_large_enough_for_the_label(self) -> None:
        client = FakeClient("REFUSED")
        await judge_abstention(client, model="m", question="q", answer="a")
        assert client.completions.kwargs["max_tokens"] == JUDGE_MAX_TOKENS
        assert JUDGE_MAX_TOKENS > 5

    async def test_truncation_reports_unmeasured_rather_than_answered(self) -> None:
        client = FakeClient("REF", finish_reason="length")
        verdict = await judge_abstention(client, model="m", question="q", answer="a")
        assert verdict.measured is False

    async def test_api_failure_is_unmeasured_not_answered(self) -> None:
        verdict = await judge_abstention(ExplodingClient(), model="m", question="q", answer="a")
        assert verdict.measured is False
        assert "RuntimeError" in verdict.error


class TestStats:
    def test_unmeasured_verdicts_are_reported(self) -> None:
        stats = summarise_abstention([(False, True), (True, False)], unmeasured=3)
        assert stats.as_dict()["unmeasured"] == 3

    def test_recall_over_the_unanswerable_questions(self) -> None:
        stats = AbstentionStats(
            unanswerable_total=12,
            unanswerable_abstained=11,
            answerable_total=46,
            answerable_abstained=2,
        )
        assert stats.recall == pytest.approx(11 / 12)
        assert stats.precision == pytest.approx(11 / 13)
        assert stats.false_abstention_rate == pytest.approx(2 / 46)
