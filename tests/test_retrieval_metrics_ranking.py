"""Retrieval metrics must read the list that actually decided the outcome.

Regression tests for a real measurement bug. `score_retrieval` was fed only the
first-stage candidate pool, so context recall was computed from the first N of
the *pre-rerank* ordering and MRR from the *pre-rerank* ranking. Both were
structurally unable to move when a reranker reordered anything.

The rerank ablation duly produced a context recall and an MRR bit-identical to
the baseline -- 0.880 and 0.770 in both -- while RAGAS context precision, which
scores the passages actually handed to the model, moved six points. The
reranker was working. The instrument was reading the wrong list.
"""

from __future__ import annotations

import pytest

from vidyarag.evaluation.retrieval import score_retrieval

POOL = ["a", "b", "c", "d", "e", "f", "g", "h"]
GOLD = ["h"]  # last in the pool: outside a top-5 context window


class TestRerankIsVisible:
    def test_context_recall_follows_the_reranked_context(self) -> None:
        """Promoting the gold chunk into the prompt must show up."""
        before = score_retrieval(POOL, GOLD, k=8, context_k=5)
        after = score_retrieval(
            POOL,
            GOLD,
            k=8,
            context_k=5,
            ranked_ids=["h", "a", "b", "c", "d", "e", "f", "g"],
            context_ids=["h", "a", "b", "c", "d"],
        )
        assert before.context_recall_at_context == pytest.approx(0.0)
        assert after.context_recall_at_context == pytest.approx(1.0)

    def test_mrr_follows_the_reranked_ordering(self) -> None:
        before = score_retrieval(POOL, GOLD, k=8, context_k=5)
        after = score_retrieval(POOL, GOLD, k=8, context_k=5, ranked_ids=["h", *POOL[:-1]])
        assert before.reciprocal_rank == pytest.approx(1 / 8)
        assert after.reciprocal_rank == pytest.approx(1.0)

    def test_pool_recall_is_unchanged_by_reordering(self) -> None:
        """Reranking finds nothing new, so pool-level metrics must not move.

        If recall @k changes between baseline and rerank, something other than
        the reranker moved and the ablation is not measuring what it claims.
        """
        before = score_retrieval(POOL, GOLD, k=8, context_k=5)
        after = score_retrieval(POOL, GOLD, k=8, context_k=5, ranked_ids=["h", *POOL[:-1]])
        assert before.recall_at_k == after.recall_at_k == pytest.approx(1.0)
        assert before.hit_at_k == after.hit_at_k is True


class TestDefaults:
    def test_absent_ranking_falls_back_to_the_pool(self) -> None:
        """Correct when no reordering stage ran, which is the baseline."""
        scores = score_retrieval(POOL, GOLD, k=8, context_k=5)
        assert scores.reciprocal_rank == pytest.approx(1 / 8)
        assert scores.context_recall_at_context == pytest.approx(0.0)

    def test_context_defaults_to_the_head_of_the_final_ranking(self) -> None:
        scores = score_retrieval(POOL, GOLD, k=8, context_k=5, ranked_ids=["h", *POOL[:-1]])
        assert scores.context_recall_at_context == pytest.approx(1.0)

    def test_no_gold_chunks_scores_none_not_zero(self) -> None:
        scores = score_retrieval(POOL, [], k=8, context_k=5)
        assert scores.recall_at_k is None
        assert scores.context_recall_at_context is None
