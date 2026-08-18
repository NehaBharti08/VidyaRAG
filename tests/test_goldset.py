"""Gold set schema and its load-time guarantees.

The validation rules here are the ones that stop a quietly broken gold set from
producing confident, meaningless numbers, so they are tested as behaviour
rather than trusted as code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vidyarag.evaluation.goldset import (
    GoldQuestion,
    Provenance,
    QuestionType,
    load_goldset,
    summarise_goldset,
)


def _record(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "fact-001",
        "question": "How does facilitated diffusion move glucose into a cell?",
        "type": "factual",
        "provenance": "llm_drafted_human_verified",
        "reference": "Through selective carrier proteins, down the gradient.",
        "gold_chunk_ids": ["biology::5.2::0007"],
        "gold_pages": ["Biology, 5.2, p.147"],
        "books": ["biology"],
        "notes": "",
    }
    base.update(overrides)
    return base


def _write(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    target = tmp_path / "goldset.jsonl"
    target.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    return target


class TestGoldQuestion:
    def test_valid_factual_question(self) -> None:
        question = GoldQuestion.model_validate(_record())
        assert question.type is QuestionType.FACTUAL
        assert question.is_answerable

    def test_answerable_requires_a_reference(self) -> None:
        with pytest.raises(ValidationError, match="require a reference answer"):
            GoldQuestion.model_validate(_record(reference=None))

    def test_answerable_requires_gold_chunks(self) -> None:
        """Without gold chunks, retrieval metrics score against nothing."""
        with pytest.raises(ValidationError, match="at least one gold chunk"):
            GoldQuestion.model_validate(_record(gold_chunk_ids=[]))

    def test_multi_hop_requires_two_chunks(self) -> None:
        """A one-chunk 'multi-hop' question is a factual lookup in disguise."""
        with pytest.raises(ValidationError, match="at least two gold chunks"):
            GoldQuestion.model_validate(
                _record(type="multi_hop", gold_chunk_ids=["only::one::0001"])
            )

    def test_multi_hop_with_two_chunks_is_valid(self) -> None:
        question = GoldQuestion.model_validate(
            _record(type="multi_hop", gold_chunk_ids=["a::1::0001", "b::2::0002"])
        )
        assert question.type is QuestionType.MULTI_HOP

    def test_unanswerable_must_not_carry_a_reference(self) -> None:
        """A reference answer on an 'unanswerable' question means it is mislabelled."""
        with pytest.raises(ValidationError, match="must not carry a reference"):
            GoldQuestion.model_validate(
                _record(
                    id="unans-001",
                    type="unanswerable",
                    provenance="human_written",
                    gold_chunk_ids=[],
                )
            )

    def test_unanswerable_must_not_cite_chunks(self) -> None:
        with pytest.raises(ValidationError, match="must not cite gold chunks"):
            GoldQuestion.model_validate(
                _record(
                    id="unans-001",
                    type="unanswerable",
                    provenance="human_written",
                    reference=None,
                )
            )

    def test_valid_unanswerable_question(self) -> None:
        question = GoldQuestion.model_validate(
            _record(
                id="unans-001",
                type="unanswerable",
                provenance="human_written",
                reference=None,
                gold_chunk_ids=[],
                gold_pages=[],
            )
        )
        assert not question.is_answerable
        assert question.provenance is Provenance.HUMAN_WRITTEN

    def test_ambiguous_counts_as_answerable(self) -> None:
        """Ambiguous questions are underspecified, not absent."""
        question = GoldQuestion.model_validate(_record(type="ambiguous"))
        assert question.is_answerable

    def test_unknown_field_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GoldQuestion.model_validate(_record(difficulty="hard"))


class TestLoadGoldset:
    def test_loads_records_in_order(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [_record(id="fact-001"), _record(id="fact-002")])
        assert [q.id for q in load_goldset(path)] == ["fact-001", "fact-002"]

    def test_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        """A duplicate id would double-weight one question in every aggregate."""
        path = _write(tmp_path, [_record(), _record()])
        with pytest.raises(ValueError, match="duplicate question id"):
            load_goldset(path)

    @pytest.mark.parametrize("marker", ["//", "#"])
    def test_skips_blank_and_comment_lines(self, tmp_path: Path, marker: str) -> None:
        """The drafting tool writes a `#` header; hand-edits tend to use `//`."""
        target = tmp_path / "goldset.jsonl"
        target.write_text(
            f"{marker} a header comment\n\n" + json.dumps(_record()) + "\n\n",
            encoding="utf-8",
        )
        assert len(load_goldset(target)) == 1

    def test_reports_line_number_on_bad_json(self, tmp_path: Path) -> None:
        target = tmp_path / "goldset.jsonl"
        target.write_text(json.dumps(_record()) + "\n{not json}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":2: invalid JSON"):
            load_goldset(target)

    def test_missing_file_names_the_builder(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="goldset draft"):
            load_goldset(tmp_path / "absent.jsonl")

    def test_empty_file_is_an_error(self, tmp_path: Path) -> None:
        target = tmp_path / "goldset.jsonl"
        target.write_text("// only a comment\n", encoding="utf-8")
        with pytest.raises(ValueError, match="contains no questions"):
            load_goldset(target)


class TestSummarise:
    def test_includes_types_with_no_entries(self) -> None:
        """A missing category is exactly what a reader needs to see."""
        questions = [GoldQuestion.model_validate(_record())]
        summary = summarise_goldset(questions)
        assert summary["factual"] == 1
        assert summary["unanswerable"] == 0
        assert set(summary) == {t.value for t in QuestionType}
