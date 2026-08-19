"""Cross-encoder reranking.

The baseline says what this must do: recall @k 0.967 against recall @context
0.880 means the gold passage is nearly always retrieved and then ranked out of
the prompt about one question in eight. Reranking finds nothing new -- it
reorders what dense retrieval already returned -- so the candidate pool must
come through unchanged in content and changed only in order.
"""

from __future__ import annotations

import pytest

from vidyarag.retrieve.dense import RetrievedChunk
from vidyarag.retrieve.rerank import rank_movement, rerank


def _chunk(chunk_id: str, score: float, text: str = "passage") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        text=text,
        citation=f"Biology, 1.1, p.{chunk_id}",
        book_slug="biology",
        book_title="Biology",
        chapter="1",
        section="1.1",
        page_start=1,
        page_end=1,
        printed_page="1",
        license_name="CC BY 4.0",
        source_url="https://openstax.org",
    )


@pytest.fixture
def reversing_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in that scores later passages higher, guaranteeing reordering."""

    class _Fake:
        def rerank(self, query: str, documents: list[str]) -> list[float]:
            return [float(i) for i in range(len(documents))]

    monkeypatch.setattr("vidyarag.retrieve.rerank.get_reranker", lambda _name: _Fake())


class TestRerank:
    def test_reorders_by_cross_encoder_score(self, reversing_reranker: None) -> None:
        chunks = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]
        out = rerank("q", chunks, model_name="fake")
        assert [c.chunk_id for c in out] == ["c", "b", "a"]

    def test_preserves_the_first_stage_score(self, reversing_reranker: None) -> None:
        """A rank change has to be attributable, not inferred."""
        out = rerank("q", [_chunk("a", 0.91), _chunk("b", 0.42)], model_name="fake")
        by_id = {c.chunk_id: c for c in out}
        assert by_id["a"].prior_score == pytest.approx(0.91)
        assert by_id["b"].prior_score == pytest.approx(0.42)

    def test_does_not_invent_or_drop_candidates(self, reversing_reranker: None) -> None:
        """Reranking reorders the pool; it must not change what is in it."""
        chunks = [_chunk(str(i), 1.0 / (i + 1)) for i in range(8)]
        out = rerank("q", chunks, model_name="fake")
        assert {c.chunk_id for c in out} == {c.chunk_id for c in chunks}
        assert len(out) == len(chunks)

    def test_top_k_truncates_after_sorting(self, reversing_reranker: None) -> None:
        chunks = [_chunk(str(i), 0.5) for i in range(6)]
        out = rerank("q", chunks, model_name="fake", top_k=2)
        assert [c.chunk_id for c in out] == ["5", "4"]

    def test_empty_input_is_not_an_error(self) -> None:
        assert rerank("q", [], model_name="fake") == []


class TestRankMovement:
    """A reranker that changes a metric without changing the top 5 has not
    earned the credit. This is what would show that."""

    def test_no_change_is_reported_as_no_change(self) -> None:
        chunks = [_chunk(str(i), 1.0) for i in range(6)]
        movement = rank_movement(chunks, chunks)
        assert movement["reordered"] == 0
        assert movement["promoted_into_context"] == 0
        assert movement["top1_changed"] is False

    def test_counts_promotions_into_the_context_window(self) -> None:
        before = [_chunk(str(i), 1.0) for i in range(8)]
        after = [before[7], *before[:7]]  # rank 8 -> rank 1
        movement = rank_movement(before, after)
        assert movement["promoted_into_context"] == 1
        assert movement["top1_changed"] is True

    def test_reordering_within_the_pool_is_counted(self) -> None:
        before = [_chunk(str(i), 1.0) for i in range(4)]
        after = [before[1], before[0], before[2], before[3]]
        assert rank_movement(before, after)["reordered"] == 2
