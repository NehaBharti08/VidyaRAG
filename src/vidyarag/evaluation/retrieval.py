"""Retrieval quality, measured without a model.

These are ordinary ranking metrics computed from chunk ids. They cost nothing,
never rate-limit, and are perfectly reproducible -- which makes them the part
of the harness worth trusting most.

They also isolate blame. RAGAS context precision and recall are judged by an
LLM against a reference answer, so a low score is ambiguous: retrieval may have
missed the passage, or the grader may have disagreed about relevance. Recall
measured against known gold chunk ids has no such ambiguity. When the two
disagree, the deterministic one is the one that says where the fault is.
"""

from __future__ import annotations

from dataclasses import dataclass


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float | None:
    """Fraction of gold chunks appearing in the top ``k``.

    Returns ``None`` when there are no gold chunks, since a recall of "0 out of
    0" is undefined and averaging it as zero would penalise the run for a
    question that had nothing to find.
    """
    if not gold_ids:
        return None
    top = set(retrieved_ids[:k])
    return sum(1 for gold in set(gold_ids) if gold in top) / len(set(gold_ids))


def hit_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> bool | None:
    """Whether *any* gold chunk made the top ``k``.

    The loosest useful signal: below this, generation cannot succeed for the
    right reason even if the answer happens to read well.
    """
    if not gold_ids:
        return None
    return bool(set(retrieved_ids[:k]) & set(gold_ids))


def reciprocal_rank(retrieved_ids: list[str], gold_ids: list[str]) -> float | None:
    """Reciprocal of the rank of the first gold chunk; 0.0 if none was found.

    Sensitive to ordering in a way recall is not, which is what makes it the
    metric to watch when reranking is added in Phase 4: a reranker that changes
    nothing else should still move this.
    """
    if not gold_ids:
        return None
    wanted = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in wanted:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Deterministic retrieval quality for one question."""

    recall_at_k: float | None
    hit_at_k: bool | None
    reciprocal_rank: float | None
    context_recall_at_context: float | None
    """Recall restricted to the chunks actually placed in the prompt.

    Distinct from ``recall_at_k``, which covers the whole retrieved candidate
    pool. The gap between them is precisely what reranking exists to close: a
    gold chunk retrieved at rank 18 but cut before generation is a retrieval
    success and a pipeline failure, and one number cannot show both.
    """

    def as_dict(self) -> dict[str, float | bool | None]:
        return {
            "recall_at_k": self.recall_at_k,
            "hit_at_k": self.hit_at_k,
            "reciprocal_rank": self.reciprocal_rank,
            "context_recall_at_context": self.context_recall_at_context,
        }


def score_retrieval(
    retrieved_ids: list[str],
    gold_ids: list[str],
    *,
    k: int,
    context_k: int,
) -> RetrievalScores:
    """Score one question's retrieval against its gold chunks.

    Args:
        retrieved_ids: Candidate chunk ids, best first.
        gold_ids: Chunk ids known to support the answer.
        k: Size of the candidate pool.
        context_k: How many of those actually reached the prompt.
    """
    return RetrievalScores(
        recall_at_k=recall_at_k(retrieved_ids, gold_ids, k),
        hit_at_k=hit_at_k(retrieved_ids, gold_ids, k),
        reciprocal_rank=reciprocal_rank(retrieved_ids, gold_ids),
        context_recall_at_context=recall_at_k(retrieved_ids, gold_ids, context_k),
    )
