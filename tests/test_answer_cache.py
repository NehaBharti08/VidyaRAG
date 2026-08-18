"""The answer cache, and the key that makes it safe.

Caching answers is what lets a run stopped by a daily quota resume the next day
instead of restarting. The risk it introduces is misattribution: serving an
answer generated under one configuration while the report claims another. The
key is the only thing preventing that, so most of these tests are about the key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vidyarag.evaluation.answer_cache import AnswerCache, answer_key

BASE = {
    "profile": "baseline",
    "generation_model": "gemini-3.5-flash-lite",
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "temperature": 0.0,
    "top_k_retrieve": 20,
    "top_k_context": 5,
    "prompt_version": "answer-v1",
    "collection": "vidyarag_biology_v1",
    "question_id": "fact-001",
    "question": "What is facilitated diffusion?",
}


class TestKey:
    def test_is_deterministic(self) -> None:
        assert answer_key(**BASE) == answer_key(**BASE)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("profile", "rerank"),
            ("generation_model", "gemini-3.1-flash-lite"),
            ("embedding_model", "BAAI/bge-small-en-v1.5"),
            ("temperature", 0.7),
            ("top_k_retrieve", 50),
            ("top_k_context", 8),
            ("prompt_version", "answer-v2"),
            ("collection", "other_collection"),
            ("question", "a different question"),
        ],
    )
    def test_changes_when_anything_that_affects_the_answer_changes(
        self, field: str, value: object
    ) -> None:
        """Each of these would produce a different answer, so each must miss.

        prompt_version is the subtle one: editing a template without bumping it
        would let the cache serve answers from the old prompt while the report
        names the new one.
        """
        changed = {**BASE, field: value}
        assert answer_key(**changed) != answer_key(**BASE)  # type: ignore[arg-type]

    def test_question_id_alone_distinguishes_identical_text(self) -> None:
        """Two ids with the same text are still separate gold-set entries."""
        other = {**BASE, "question_id": "fact-002"}
        assert answer_key(**other) != answer_key(**BASE)  # type: ignore[arg-type]


class TestCache:
    def test_round_trip(self, tmp_path: Path) -> None:
        cache = AnswerCache(tmp_path)
        payload = {"result": {"id": "fact-001"}, "contexts": ["passage one"]}
        cache.put("k", payload)
        assert cache.get("k") == payload

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        assert AnswerCache(tmp_path).get("absent") is None

    def test_disabled_cache_never_stores(self, tmp_path: Path) -> None:
        cache = AnswerCache(None)
        cache.put("k", {"result": {}})
        assert cache.enabled is False
        assert cache.get("k") is None

    def test_corrupt_entry_is_a_miss_not_a_crash(self, tmp_path: Path) -> None:
        """A half-written file from an interrupted run costs one recomputation."""
        cache = AnswerCache(tmp_path)
        (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
        assert cache.get("broken") is None

    def test_entry_carries_the_contexts_grading_needs(self, tmp_path: Path) -> None:
        """Without contexts a resumed run could not grade the reused answers."""
        cache = AnswerCache(tmp_path)
        cache.put("k", {"result": {"id": "x"}, "contexts": ["a", "b"]})
        entry = cache.get("k")
        assert entry is not None
        assert entry["contexts"] == ["a", "b"]
