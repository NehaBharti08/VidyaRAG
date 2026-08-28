"""Query decomposition and rank fusion.

Motivated by a measured failure rather than a hunch: cross-encoder reranking
lifted recall @context on factual questions by 7.1 points and lowered it on
multi-hop by 2.8, because independent per-passage scoring promotes
near-duplicates of the strongest match and crowds out the second passage a hop
requires.
"""

from __future__ import annotations

from typing import Any

from vidyarag.retrieve.decompose import (
    Decomposition,
    _clean,
    decompose,
    reciprocal_rank_fusion,
)
from vidyarag.retrieve.dense import RetrievedChunk


def _chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        text="passage",
        citation="Biology, 1.1, p.1",
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


class _FakeLLM:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    class _Models:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def generate_content(self, **_: Any) -> Any:
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    @property
    def models(self) -> Any:
        return self._Models(self._payload)


class _Response:
    def __init__(self, parsed: Any = None, text: str | None = None) -> None:
        self.parsed = parsed
        self.text = text


class TestCleaning:
    def test_a_single_sub_question_is_not_a_decomposition(self) -> None:
        """One sub-question is a paraphrase; running it costs a pass to go nowhere."""
        out = _clean(Decomposition(is_multi_hop=True, sub_questions=["only one"]))
        assert out.is_multi_hop is False
        assert out.sub_questions == []

    def test_blank_sub_questions_are_dropped(self) -> None:
        out = _clean(Decomposition(is_multi_hop=True, sub_questions=["a", "  ", "b"]))
        assert out.sub_questions == ["a", "b"]

    def test_split_is_capped(self) -> None:
        out = _clean(Decomposition(is_multi_hop=True, sub_questions=[f"q{i}" for i in range(9)]))
        assert len(out.sub_questions) == 3


class TestDecomposeFailureModes:
    def test_api_failure_degrades_to_baseline(self) -> None:
        """Falling back to the whole question is the safe direction."""
        out = decompose(_FakeLLM(RuntimeError("429")), "q", model="m")
        assert out.is_multi_hop is False

    def test_unparseable_response_degrades_to_baseline(self) -> None:
        out = decompose(_FakeLLM(_Response(text="not json")), "q", model="m")
        assert out.is_multi_hop is False

    def test_empty_response_degrades_to_baseline(self) -> None:
        out = decompose(_FakeLLM(_Response()), "q", model="m")
        assert out.is_multi_hop is False

    def test_a_real_split_survives(self) -> None:
        parsed = Decomposition(is_multi_hop=True, sub_questions=["hop one", "hop two"])
        out = decompose(_FakeLLM(_Response(parsed=parsed)), "q", model="m")
        assert out.sub_questions == ["hop one", "hop two"]


class TestFusion:
    def test_agreement_across_sub_questions_outranks_one_strong_hit(self) -> None:
        """A chunk both hops retrieve is likelier to be the bridge between them."""
        first = [_chunk("solo"), _chunk("shared")]
        second = [_chunk("other"), _chunk("shared")]
        fused = reciprocal_rank_fusion([first, second])
        assert fused[0].chunk_id == "shared"

    def test_every_chunk_survives_fusion(self) -> None:
        fused = reciprocal_rank_fusion([[_chunk("a"), _chunk("b")], [_chunk("c")]])
        assert {c.chunk_id for c in fused} == {"a", "b", "c"}

    def test_duplicates_are_not_repeated(self) -> None:
        fused = reciprocal_rank_fusion([[_chunk("a")], [_chunk("a")], [_chunk("a")]])
        assert [c.chunk_id for c in fused] == ["a"]

    def test_single_list_is_passed_through_in_order(self) -> None:
        fused = reciprocal_rank_fusion([[_chunk("a"), _chunk("b"), _chunk("c")]])
        assert [c.chunk_id for c in fused] == ["a", "b", "c"]

    def test_empty_input(self) -> None:
        assert reciprocal_rank_fusion([]) == []
