"""Retrieval ranking, abstention arithmetic, and report rendering.

None of this touches a model or the network. These are the parts of the harness
that must be exactly right, because every headline number is computed from them.
"""

from __future__ import annotations

import pytest

from vidyarag.evaluation.abstention import (
    AbstentionStats,
    is_structural_abstention,
    summarise_abstention,
)
from vidyarag.evaluation.metrics import METRIC_NAMES, SampleScores, _clean
from vidyarag.evaluation.retrieval import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)
from vidyarag.generate.prompts import NO_CONTEXT_ANSWER

GOLD = ["g1", "g2"]


class TestRecallAtK:
    def test_all_gold_within_k(self) -> None:
        assert recall_at_k(["g1", "x", "g2"], GOLD, 3) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k(["g1", "x", "y"], GOLD, 3) == 0.5

    def test_k_truncates_the_pool(self) -> None:
        """A gold chunk beyond k has not been retrieved for scoring purposes."""
        assert recall_at_k(["x", "y", "g1"], GOLD, 2) == 0.0

    def test_no_gold_chunks_is_undefined_not_zero(self) -> None:
        """Averaging '0 out of 0' as zero would penalise a question unfairly."""
        assert recall_at_k(["a"], [], 5) is None

    def test_duplicate_gold_ids_do_not_inflate(self) -> None:
        assert recall_at_k(["g1", "g1"], ["g1", "g1"], 5) == 1.0


class TestHitAtK:
    def test_hit(self) -> None:
        assert hit_at_k(["x", "g2"], GOLD, 2) is True

    def test_miss(self) -> None:
        assert hit_at_k(["x", "y"], GOLD, 2) is False

    def test_undefined_without_gold(self) -> None:
        assert hit_at_k(["x"], [], 2) is None


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("retrieved", "expected"),
        [
            (["g1", "x", "y"], 1.0),
            (["x", "g1", "y"], 0.5),
            (["x", "y", "g2"], pytest.approx(1 / 3)),
            (["x", "y", "z"], 0.0),
        ],
    )
    def test_rank_of_first_gold_chunk(self, retrieved: list[str], expected: float) -> None:
        assert reciprocal_rank(retrieved, GOLD) == expected

    def test_undefined_without_gold(self) -> None:
        assert reciprocal_rank(["a"], []) is None


class TestScoreRetrieval:
    def test_separates_pool_recall_from_context_recall(self) -> None:
        """The gap between these two is precisely what reranking exists to close.

        Here both gold chunks are retrieved, but only one survives into the
        prompt -- a retrieval success and a pipeline failure at the same time.
        """
        scores = score_retrieval(["g1", "a", "b", "g2"], GOLD, k=4, context_k=2)
        assert scores.recall_at_k == 1.0
        assert scores.context_recall_at_context == 0.5
        assert scores.hit_at_k is True
        assert scores.reciprocal_rank == 1.0

    def test_as_dict_exposes_every_field(self) -> None:
        scores = score_retrieval(["g1"], GOLD, k=1, context_k=1)
        assert set(scores.as_dict()) == {
            "recall_at_k",
            "hit_at_k",
            "reciprocal_rank",
            "context_recall_at_context",
        }


class TestClean:
    def test_nan_becomes_none(self) -> None:
        """NaN is a missing score, not a zero, and would poison any mean."""
        assert _clean(float("nan")) is None

    def test_none_stays_none(self) -> None:
        assert _clean(None) is None

    def test_unparseable_becomes_none(self) -> None:
        assert _clean("not a number") is None

    def test_zero_is_preserved(self) -> None:
        """Zero is a real score and must survive."""
        assert _clean(0.0) == 0.0


class TestSampleScores:
    def test_incomplete_when_a_metric_failed(self) -> None:
        scores = SampleScores(faithfulness=1.0, errors={"context_recall": "boom"})
        assert not scores.complete
        assert set(scores.as_dict()) == set(METRIC_NAMES)

    def test_complete_when_all_present(self) -> None:
        scores = SampleScores(
            faithfulness=1.0,
            answer_relevancy=0.9,
            context_precision=0.8,
            context_recall=0.7,
        )
        assert scores.complete


class TestStructuralAbstention:
    def test_detects_the_no_context_sentinel(self) -> None:
        assert is_structural_abstention(NO_CONTEXT_ANSWER)

    def test_detects_the_trace_flag(self) -> None:
        assert is_structural_abstention("anything", trace_abstained=True)

    def test_ordinary_answer_is_not_an_abstention(self) -> None:
        assert not is_structural_abstention("Mitochondria produce ATP [1].")


class TestAbstentionStats:
    def test_perfect_behaviour(self) -> None:
        stats = summarise_abstention([(False, True), (False, True), (True, False)])
        assert stats.recall == 1.0
        assert stats.precision == 1.0
        assert stats.false_abstention_rate == 0.0
        assert stats.f1 == 1.0

    def test_refusing_everything_is_caught_by_false_abstention_rate(self) -> None:
        """Precision alone would score this 0.5 and hide the real problem."""
        stats = summarise_abstention([(False, True), (True, True), (True, True)])
        assert stats.recall == 1.0
        assert stats.false_abstention_rate == 1.0

    def test_never_abstaining(self) -> None:
        """The Phase 2 baseline's expected shape: it answers everything."""
        stats = summarise_abstention([(False, False), (True, False)])
        assert stats.recall == 0.0
        assert stats.precision is None  # no refusals at all, so undefined
        assert stats.false_abstention_rate == 0.0

    def test_undefined_without_unanswerable_questions(self) -> None:
        stats = AbstentionStats(0, 0, 3, 0)
        assert stats.recall is None
        assert stats.f1 is None

    def test_as_dict_is_json_ready(self) -> None:
        stats = summarise_abstention([(False, True), (True, False)])
        payload = stats.as_dict()
        assert payload["unanswerable_total"] == 1
        assert payload["precision"] == 1.0
