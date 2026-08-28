"""Cross-encoder reranking.

Dense retrieval scores a query and a passage independently and compares the two
vectors. A cross-encoder reads them together, which is slower but far better at
telling a passage that is *about* the right topic from one that actually
*answers* the question.

The baseline says exactly where this should help. Recall @k is 0.967 and recall
@context is 0.880: the gold passage is in the retrieved pool for almost every
question, and is then ranked out of the top 5 before generation for roughly one
question in eight. Reranking does not need to find anything new. It needs to
reorder what dense retrieval already found.

The model runs locally through fastembed -- ONNX on CPU, ~80 MB, no torch. That
keeps retrieval free of any external dependency, which is the same property that
lets the published demo survive an expired account.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from vidyarag.retrieve.dense import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from fastembed.rerank.cross_encoder import TextCrossEncoder


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> TextCrossEncoder:
    """Load a cross-encoder, cached per process.

    The first call downloads ONNX weights; every later call is free. Caching
    matters because constructing the model costs far more than scoring a batch
    with it.
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name)


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    *,
    model_name: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Reorder candidates by cross-encoder relevance.

    Args:
        query: The user's question.
        chunks: Candidates from first-stage retrieval, any order.
        model_name: A fastembed cross-encoder id.
        top_k: Keep only this many. ``None`` keeps all, reordered.

    Returns:
        Chunks sorted by descending relevance. Each carries its cross-encoder
        score in ``score``, with the first-stage score preserved in
        ``prior_score`` so a rank change can be attributed rather than guessed
        at.
    """
    if not chunks:
        return []

    scores = list(get_reranker(model_name).rerank(query, [c.text for c in chunks]))
    rescored = [
        dataclasses.replace(chunk, score=float(score), prior_score=chunk.score)
        for chunk, score in zip(chunks, scores, strict=True)
    ]
    rescored.sort(key=lambda c: c.score, reverse=True)
    return rescored if top_k is None else rescored[:top_k]


def rank_movement(before: list[RetrievedChunk], after: list[RetrievedChunk]) -> dict[str, Any]:
    """Summarise what reranking actually changed.

    Reported per query so an ablation can say *how* a score moved, not only
    that it did. A reranker that improves a metric while never changing the top
    5 has not earned the credit, and this is what would reveal that.
    """
    before_rank = {chunk.chunk_id: index for index, chunk in enumerate(before)}
    moved = sum(
        1 for index, chunk in enumerate(after) if before_rank.get(chunk.chunk_id, index) != index
    )
    promoted_into_top = [
        chunk.chunk_id
        for index, chunk in enumerate(after[:5])
        if before_rank.get(chunk.chunk_id, 0) >= 5
    ]
    return {
        "reordered": moved,
        "promoted_into_context": len(promoted_into_top),
        "top1_changed": bool(before and after and before[0].chunk_id != after[0].chunk_id),
    }
