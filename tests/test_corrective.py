"""The corrective self-check loop.

Baseline abstention recall is 0.000 across every Phase 4 profile: the pipeline
answered all twelve unanswerable questions because nothing in it could decline.
These tests pin the behaviour that has to change that, and in particular the
failure modes where a self-check makes things *worse* than having none.
"""

from __future__ import annotations

import pytest

from vidyarag.correct.grader import Claim, ClaimVerdict, Groundedness
from vidyarag.correct.loop import (
    ABSTENTION_TEXT,
    LoopOutcome,
    reformulate,
    run_corrective_loop,
)
from vidyarag.correct.policy import CorrectivePolicy, Decision


def _claims(*verdicts: ClaimVerdict) -> list[Claim]:
    return [Claim(text=f"claim {i}", verdict=v) for i, v in enumerate(verdicts)]


SUP, PAR, UNS = ClaimVerdict.SUPPORTED, ClaimVerdict.PARTIAL, ClaimVerdict.UNSUPPORTED


class TestScoring:
    def test_all_supported_scores_one(self) -> None:
        assert Groundedness(_claims(SUP, SUP, SUP)).score == pytest.approx(1.0)

    def test_all_unsupported_scores_zero(self) -> None:
        assert Groundedness(_claims(UNS, UNS)).score == pytest.approx(0.0)

    def test_partial_counts_as_half(self) -> None:
        assert Groundedness(_claims(SUP, PAR)).score == pytest.approx(0.75)

    def test_failed_claims_put_unsupported_first(self) -> None:
        """A retry should search for what is missing before what is vague."""
        g = Groundedness(
            [
                Claim(text="vague", verdict=PAR),
                Claim(text="missing", verdict=UNS),
            ]
        )
        assert [c.text for c in g.failed_claims] == ["missing", "vague"]

    def test_supported_claims_are_not_retry_targets(self) -> None:
        assert Groundedness(_claims(SUP, SUP)).failed_claims == []


class TestGradingFailureIsNotUngrounded:
    """The most dangerous confusion in this module.

    A failed grader call means the loop has no evidence either way. Scoring it
    as 0.0 would make the system abstain because of an API error while looking
    appropriately cautious -- a silent failure worse than the one it prevents.
    """

    def test_error_is_not_measured(self) -> None:
        assert Groundedness([], error="429 quota").measured is False

    def test_no_claims_is_not_measured(self) -> None:
        assert Groundedness([]).measured is False

    def test_policy_accepts_rather_than_abstains_when_grading_failed(self) -> None:
        policy = CorrectivePolicy()
        assert policy.decide(Groundedness([], error="boom"), attempt=1) is Decision.ACCEPT


class TestGeneratorRefusalIsAnAbstention:
    """Found by smoke-testing, not by reasoning, and it would have been fatal.

    The generator can decline on its own -- "the provided passages do not
    contain information comparing..." -- and such a draft is *perfectly
    grounded*: a statement that the passages lack something is supported by
    passages that lack it. Measured on a real unanswerable question, it scored
    1.0 and was accepted, and the loop reported abstained=False on exactly the
    question it exists to decline.

    Left unfixed, the corrective loop's abstention path would be dead code and
    the project would credit a loop for refusals the generator produced alone.
    """

    def test_a_refusal_scores_well_on_claims_alone(self) -> None:
        """The reason claim scores cannot detect this by themselves."""
        refusal = Groundedness(_claims(SUP), refuses=True)
        assert refusal.score == pytest.approx(1.0)
        assert refusal.measured is True

    def test_policy_abstains_on_a_refusal_despite_a_perfect_score(self) -> None:
        policy = CorrectivePolicy(accept_threshold=0.8)
        assert (
            policy.decide(Groundedness(_claims(SUP), refuses=True), attempt=1) is Decision.ABSTAIN
        )

    def test_a_refusal_does_not_burn_a_retry(self) -> None:
        """No failed claim means nothing for a reformulated query to search for."""
        policy = CorrectivePolicy(max_attempts=3)
        assert (
            policy.decide(Groundedness(_claims(SUP), refuses=True), attempt=1) is Decision.ABSTAIN
        )

    def test_a_normal_answer_is_unaffected(self) -> None:
        policy = CorrectivePolicy(accept_threshold=0.8)
        assert (
            policy.decide(Groundedness(_claims(SUP, SUP), refuses=False), attempt=1)
            is Decision.ACCEPT
        )

    def test_loop_reports_the_abstention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        h = _Harness()
        monkeypatch.setattr(
            "vidyarag.correct.loop.grade_answer",
            lambda *a, **k: Groundedness(_claims(SUP), refuses=True),
        )
        out = run_corrective_loop(
            question="q?",
            generate=h.generate,
            retrieve=h.retrieve,
            llm=object(),
            grader_model="grader",
            policy=CorrectivePolicy(),
        )
        assert out.abstained is True
        assert out.fired is True
        assert out.answer == ABSTENTION_TEXT


class TestPolicy:
    def test_well_grounded_answer_is_accepted(self) -> None:
        policy = CorrectivePolicy(accept_threshold=0.8, abstain_threshold=0.5)
        assert (
            policy.decide(Groundedness(_claims(SUP, SUP, SUP, SUP)), attempt=1) is Decision.ACCEPT
        )

    def test_middling_answer_retries_while_budget_remains(self) -> None:
        policy = CorrectivePolicy(accept_threshold=0.8, abstain_threshold=0.5, max_attempts=2)
        # 2 supported of 3 = 0.667, between the thresholds.
        assert policy.decide(Groundedness(_claims(SUP, SUP, UNS)), attempt=1) is Decision.RETRY

    def test_middling_answer_abstains_once_budget_is_spent(self) -> None:
        """Returning a draft known to be under the bar would defeat the check."""
        policy = CorrectivePolicy(accept_threshold=0.8, abstain_threshold=0.5, max_attempts=2)
        assert policy.decide(Groundedness(_claims(SUP, SUP, UNS)), attempt=2) is Decision.ABSTAIN

    def test_badly_grounded_answer_abstains_immediately(self) -> None:
        """No point spending a retry when the evidence is plainly absent."""
        policy = CorrectivePolicy(accept_threshold=0.8, abstain_threshold=0.5, max_attempts=3)
        assert policy.decide(Groundedness(_claims(UNS, UNS, UNS)), attempt=1) is Decision.ABSTAIN

    @pytest.mark.parametrize(
        ("accept", "abstain"),
        [(0.5, 0.8), (1.5, 0.5), (0.8, -0.1)],
    )
    def test_incoherent_thresholds_are_rejected(self, accept: float, abstain: float) -> None:
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            CorrectivePolicy(accept_threshold=accept, abstain_threshold=abstain)

    def test_zero_attempts_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            CorrectivePolicy(max_attempts=0)


class TestReformulation:
    def test_failed_claims_become_the_next_query(self) -> None:
        g = Groundedness([Claim(text="glucose uses GLUT4", verdict=UNS)])
        assert "glucose uses GLUT4" in reformulate("How is glucose moved?", g)

    def test_the_original_question_is_kept(self) -> None:
        """Dropping it lets the query drift onto a detail and lose the subject."""
        g = Groundedness([Claim(text="a detail", verdict=UNS)])
        assert "How is glucose moved?" in reformulate("How is glucose moved?", g)

    def test_nothing_failed_leaves_the_question_alone(self) -> None:
        g = Groundedness(_claims(SUP, SUP))
        assert reformulate("original", g) == "original"


class _Harness:
    """Drives the loop with scripted groundedness verdicts."""

    def __init__(self, *verdict_sets: list[Claim]) -> None:
        self._sets = list(verdict_sets)
        self.queries: list[str] = []
        self.retrievals = 0

    def retrieve(self, query: str) -> list[str]:
        self.queries.append(query)
        self.retrievals += 1
        return ["a passage"]

    def generate(self, _question: str, context: list[str]) -> tuple[str, list[str]]:
        return f"draft {len(self.queries)}", context

    def grade(self, *_a: object, **_k: object) -> Groundedness:
        return Groundedness(self._sets.pop(0))


def _run(
    harness: _Harness, policy: CorrectivePolicy, monkeypatch: pytest.MonkeyPatch
) -> LoopOutcome:
    monkeypatch.setattr("vidyarag.correct.loop.grade_answer", harness.grade)
    return run_corrective_loop(
        question="q?",
        generate=harness.generate,
        retrieve=harness.retrieve,
        llm=object(),
        grader_model="grader",
        policy=policy,
    )


class TestLoop:
    def test_a_grounded_first_draft_costs_one_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = _Harness(_claims(SUP, SUP))
        out = _run(h, CorrectivePolicy(), monkeypatch)
        assert out.abstained is False
        assert out.attempt_count == 1
        assert out.fired is False

    def test_a_retry_re_retrieves_with_a_new_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regenerating from the same passages would be a no-op at temperature 0."""
        h = _Harness(_claims(SUP, SUP, UNS), _claims(SUP, SUP, SUP))
        out = _run(h, CorrectivePolicy(max_attempts=2), monkeypatch)
        assert out.attempt_count == 2
        assert h.retrievals == 2
        assert h.queries[0] != h.queries[1]
        assert out.abstained is False
        assert out.fired is True

    def test_exhausting_the_budget_abstains_rather_than_shipping_the_draft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = _Harness(_claims(SUP, SUP, UNS), _claims(SUP, SUP, UNS))
        out = _run(h, CorrectivePolicy(max_attempts=2), monkeypatch)
        assert out.abstained is True
        assert out.answer == ABSTENTION_TEXT
        assert "draft" not in out.answer

    def test_hopeless_answer_abstains_without_spending_a_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = _Harness(_claims(UNS, UNS))
        out = _run(h, CorrectivePolicy(max_attempts=3), monkeypatch)
        assert out.abstained is True
        assert out.attempt_count == 1
        assert h.retrievals == 1

    def test_attempts_are_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unbounded loop is an unbounded bill and an unbounded latency."""
        h = _Harness(*[_claims(SUP, SUP, UNS) for _ in range(10)])
        out = _run(h, CorrectivePolicy(max_attempts=3), monkeypatch)
        assert out.attempt_count == 3
        assert h.retrievals == 3

    def test_the_abstention_names_its_reason(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'I don't know' invites a rephrase; naming the corpus tells a student
        to look elsewhere."""
        h = _Harness(_claims(UNS, UNS))
        out = _run(h, CorrectivePolicy(), monkeypatch)
        assert "source material" in out.answer

    def test_every_attempt_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        h = _Harness(_claims(SUP, SUP, UNS), _claims(SUP, SUP, SUP))
        out = _run(h, CorrectivePolicy(max_attempts=2), monkeypatch)
        payload = out.as_dict()
        assert payload["attempts"] == 2
        assert len(payload["trace"]) == 2
        assert payload["trace"][0]["decision"] == "retry"
        assert payload["trace"][1]["decision"] == "accept"
