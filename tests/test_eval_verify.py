"""Gold-set candidate verification.

The logic under test is small but load-bearing: it decides which questions are
allowed to become evidence. A permissive bug here would not crash anything, it
would quietly produce an evaluation that flatters the system.
"""

from __future__ import annotations

import pytest

from vidyarag.evaluation.goldset import GoldQuestion, Provenance, QuestionType
from vidyarag.evaluation.verify import (
    TriageFinding,
    UnanswerableCheck,
    in_domain_threshold,
)


def _check(**overrides: object) -> UnanswerableCheck:
    base: dict[str, object] = {
        "question": "q",
        "rationale": "r",
        "topic": "t",
        "top_score": 0.80,
        "retrieved": ["Biology, 1.1, p.15"],
        "in_domain": True,
        "answerable": False,
        "grader_reason": "not covered",
    }
    base.update(overrides)
    return UnanswerableCheck(**base)  # type: ignore[arg-type]


class TestAcceptance:
    """Both checks must pass. Either alone admits the wrong questions."""

    def test_in_domain_and_unanswered_is_accepted(self) -> None:
        assert _check().accepted is True

    def test_off_topic_is_rejected_even_when_unanswered(self) -> None:
        """The trivial case: a question about French history is also unanswered."""
        candidate = _check(in_domain=False, answerable=None)
        assert candidate.accepted is False
        assert "off-topic" in candidate.verdict

    def test_in_domain_but_answered_is_rejected(self) -> None:
        """The corpus covers it, so refusing it would be wrong, not correct."""
        candidate = _check(answerable=True)
        assert candidate.accepted is False
        assert candidate.verdict == "reject: corpus answers it"

    def test_grader_failure_is_not_silently_an_acceptance(self) -> None:
        """A failed grader call must never read as 'the corpus does not answer it'."""
        candidate = _check(answerable=None)
        assert candidate.accepted is False
        assert "error" in candidate.verdict

    def test_verdict_reports_the_score_that_caused_rejection(self) -> None:
        assert "0.310" in _check(in_domain=False, top_score=0.31, answerable=None).verdict


class TestDiversity:
    """A gold set of twelve rephrasings of one question measures one question.

    Worse, a set that shares a distinctive phrasing is gameable: a system could
    pattern-match the wording and refuse, scoring perfect abstention with no
    groundedness reasoning at all. An unguided run produced exactly that --
    eleven of twelve questions contained the word "exact".
    """

    def test_near_duplicate_is_rejected_despite_passing_both_checks(self) -> None:
        candidate = _check(similarity_to_accepted=0.95)
        assert candidate.in_domain is True
        assert candidate.answerable is False
        assert candidate.accepted is False
        assert "near-duplicate" in candidate.verdict

    def test_distinct_question_is_kept(self) -> None:
        assert _check(similarity_to_accepted=0.42).accepted is True

    def test_duplicate_is_reported_before_a_grader_error(self) -> None:
        """A duplicate is skipped before grading, so it has no grader verdict."""
        candidate = _check(similarity_to_accepted=0.95, answerable=None)
        assert "near-duplicate" in candidate.verdict

    def test_shapes_are_rotated_not_repeated(self) -> None:
        from vidyarag.evaluation.verify import QUESTION_SHAPES

        assert len(QUESTION_SHAPES) == len(set(QUESTION_SHAPES))
        assert len(QUESTION_SHAPES) >= 6

    def test_no_similarity_when_nothing_accepted_yet(self) -> None:
        from vidyarag.evaluation.verify import most_similar_accepted

        assert most_similar_accepted("anything", [], "unused-model") == 0.0


class TestThresholdCalibration:
    """The cutoff is derived from known in-domain questions, not chosen by feel."""

    class _FakeClient:
        def __init__(self, scores: dict[str, float]) -> None:
            self._scores = scores

    @staticmethod
    def _questions(texts: list[str]) -> list[GoldQuestion]:
        return [
            GoldQuestion(
                id=f"q-{i}",
                question=text,
                type=QuestionType.FACTUAL,
                provenance=Provenance.LLM_DRAFTED_HUMAN_VERIFIED,
                reference="ref",
                gold_chunk_ids=["c1"],
            )
            for i, text in enumerate(texts)
        ]

    def test_picks_the_requested_percentile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        scores = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45]
        ordered = iter(scores)

        class _Hit:
            def __init__(self, score: float) -> None:
                self.score = score

        monkeypatch.setattr(
            "vidyarag.evaluation.verify.retrieve_dense",
            lambda *a, **k: [_Hit(next(ordered))],
        )
        threshold = in_domain_threshold(
            None,  # type: ignore[arg-type]
            self._questions([f"q{i}" for i in range(10)]),
            collection="c",
            embedding_model="m",
            percentile=0.10,
        )
        # Sorted ascending, index int(10 * 0.10) == 1 -> the second lowest.
        assert threshold == pytest.approx(0.50)

    def test_raises_when_nothing_retrieves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Silently returning 0.0 would disable the in-domain check entirely."""
        monkeypatch.setattr("vidyarag.evaluation.verify.retrieve_dense", lambda *a, **k: [])
        with pytest.raises(ValueError, match="calibrate"):
            in_domain_threshold(
                None,  # type: ignore[arg-type]
                self._questions(["q"]),
                collection="c",
                embedding_model="m",
            )


class TestTriage:
    def test_supported_question_needs_no_attention(self) -> None:
        assert TriageFinding(id="a", question="q", supported=True).needs_attention is False

    def test_unsupported_question_is_flagged(self) -> None:
        assert TriageFinding(id="a", question="q", supported=False).needs_attention is True

    def test_grader_failure_is_flagged_rather_than_assumed_good(self) -> None:
        """An unknown verdict must surface for a human, not pass by default."""
        assert TriageFinding(id="a", question="q", supported=None).needs_attention is True


class TestProvenanceHonesty:
    def test_retrieval_verified_is_distinct_from_human_written(self) -> None:
        """Machine-proposed questions must never be recorded as hand-authored."""
        assert Provenance.LLM_DRAFTED_RETRIEVAL_VERIFIED != Provenance.HUMAN_WRITTEN
        assert Provenance.LLM_DRAFTED_RETRIEVAL_VERIFIED.value == "llm_drafted_retrieval_verified"

    def test_unanswerable_still_rejects_ground_truth(self) -> None:
        """The schema guard holds regardless of how the question was produced."""
        with pytest.raises(ValueError, match="must not carry a reference"):
            GoldQuestion(
                id="unans-001",
                question="q",
                type=QuestionType.UNANSWERABLE,
                provenance=Provenance.LLM_DRAFTED_RETRIEVAL_VERIFIED,
                reference="an answer",
            )
