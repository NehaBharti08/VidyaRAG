"""Query decomposition for multi-hop questions.

The rerank ablation is the argument for this module. Cross-encoder reranking
lifted recall @context on factual questions by 7.1 points and *lowered* it on
multi-hop by 2.8. The mechanism: a cross-encoder scores each passage
independently against the query and has no notion of what else was selected, so
it promotes whatever most resembles the question -- typically several
near-duplicates of the strongest match. A factual question has one right
passage and that is exactly correct. A multi-hop question needs two
*complementary* passages, and the duplicates crowd the second one out.

That is a diversity failure, and no better pointwise scorer fixes a pointwise
objective. Decomposition attacks it from the other side: ask two narrower
questions, retrieve for each, and fuse. Each sub-question gets its own shot at
the passage it needs rather than competing for slots against the other hop.

Fusion is Reciprocal Rank Fusion, which combines ranked lists using only rank
position. It is deliberately parameter-light -- scores from two different
sub-queries are not on a comparable scale, and normalising them would introduce
a knob that would then need tuning against the same gold set used to report the
result.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from vidyarag.retrieve.dense import RetrievedChunk, retrieve_dense

RRF_K = 60
"""Reciprocal Rank Fusion damping constant.

The value from the original paper. It flattens the difference between top ranks
so one sub-question cannot dominate the fused list purely by having ranked its
best hit first. Left at the published default rather than tuned, because tuning
it on the gold set would be fitting the evaluation."""

MAX_SUB_QUESTIONS = 3
"""Cap on the split. Beyond three the sub-questions start restating each other,
and each one costs a retrieval pass."""


class Decomposition(BaseModel):
    """How a question was split, and whether it needed splitting at all."""

    is_multi_hop: bool = Field(
        description="True only if answering requires facts from two or more distinct topics."
    )
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Two or three self-contained questions. Empty if not multi-hop.",
    )


DECOMPOSE_PROMPT = """\
You are preparing a search over two introductory textbooks: OpenStax *Biology* \
and *Anatomy and Physiology*.

Question: {question}

Decide whether answering it requires facts from two or more DISTINCT topics that \
would live in different sections of a textbook.

If it does, split it into two or three self-contained questions, each of which \
could be looked up on its own. Each must make sense without the others -- resolve \
pronouns and carry over any needed context.

If the question is a single lookup, however wordy, say so and return no \
sub-questions. Splitting an atomic question produces near-duplicates that \
retrieve the same passage repeatedly and crowd out everything else, which is \
worse than not splitting at all.
"""


def decompose(llm: Any, question: str, *, model: str) -> Decomposition:
    """Split a question into sub-questions, or report that it is atomic.

    A failed or unparseable call returns "not multi-hop" rather than raising.
    Falling back to the undecomposed query degrades to baseline behaviour, which
    is the safe direction: the alternative is failing a question that plain
    retrieval would have answered.
    """
    try:
        response = llm.models.generate_content(
            model=model,
            contents=DECOMPOSE_PROMPT.format(question=question),
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": Decomposition,
            },
        )
    except Exception:  # noqa: BLE001 - degrade to baseline, never fail the query
        return Decomposition(is_multi_hop=False)

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, Decomposition):
        return _clean(parsed)
    text = getattr(response, "text", None)
    if not text:
        return Decomposition(is_multi_hop=False)
    try:
        return _clean(Decomposition.model_validate_json(text))
    except Exception:  # noqa: BLE001
        return Decomposition(is_multi_hop=False)


def _clean(decomposition: Decomposition) -> Decomposition:
    """Drop blanks, cap the count, and reject a one-way split.

    A single sub-question is not a decomposition -- it is a paraphrase, and
    running it costs a retrieval pass to arrive back where we started.
    """
    subs = [s.strip() for s in decomposition.sub_questions if s and s.strip()]
    subs = subs[:MAX_SUB_QUESTIONS]
    if len(subs) < 2:
        return Decomposition(is_multi_hop=False)
    return Decomposition(is_multi_hop=True, sub_questions=subs)


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], *, k: int = RRF_K
) -> list[RetrievedChunk]:
    """Fuse ranked lists by rank position alone.

    Each chunk scores ``sum(1 / (k + rank))`` across the lists it appears in, so
    a passage retrieved by several sub-questions outranks one retrieved by only
    the strongest. That is the property worth having here: a chunk both hops
    agree on is more likely to be the bridge between them.
    """
    scores: dict[str, float] = {}
    seen: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            seen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [seen[chunk_id] for chunk_id, _ in ordered]


def retrieve_decomposed(
    client: QdrantClient,
    sub_questions: list[str],
    *,
    collection: str,
    embedding_model: str,
    limit: int,
) -> list[RetrievedChunk]:
    """Retrieve for each sub-question and fuse the results.

    Each sub-question gets the full ``limit``, not a share of it. Splitting the
    budget would hand a multi-hop question a smaller pool per hop than a factual
    question gets for its single one, and any drop in recall would then be an
    artefact of the budget rather than a property of decomposition.
    """
    ranked_lists = [
        retrieve_dense(
            client,
            sub_question,
            collection=collection,
            embedding_model=embedding_model,
            limit=limit,
        )
        for sub_question in sub_questions
    ]
    return reciprocal_rank_fusion(ranked_lists)
