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
from vidyarag.settings import PipelineConfig

CONFIG = PipelineConfig(name="baseline")

BASE: dict[str, object] = {
    "config": CONFIG,
    "prompt_version": "answer-v1",
    "collection": "vidyarag_biology_v1",
    "question_id": "fact-001",
    "question": "What is facilitated diffusion?",
}


def _with(**overrides: object) -> dict[str, object]:
    """The base key arguments, with a configuration field changed."""
    config = CONFIG.model_copy(deep=True, update={}) if not overrides else None
    if overrides:
        data = CONFIG.model_dump()
        for dotted, value in overrides.items():
            target = data
            *path, leaf = dotted.split(".")
            for part in path:
                target = target[part]
            target[leaf] = value
        config = PipelineConfig.model_validate(data)
    return {**BASE, "config": config}


class TestKey:
    def test_is_deterministic(self) -> None:
        assert answer_key(**BASE) == answer_key(**BASE)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("name", "rerank"),
            ("generation_model", "gemini-3.1-flash-lite"),
            ("embedding_model", "BAAI/bge-small-en-v1.5"),
            ("temperature", 0.7),
            ("retrieval.top_k_retrieve", 50),
            ("retrieval.top_k_context", 8),
            # Every one of these was missing from the key. Each changes what the
            # pipeline answers, and a profile edited in place would otherwise
            # have been served the previous behaviour's answers under its name.
            ("grader_model", "gemini-3.6-flash"),
            ("retrieval.use_reranker", True),
            ("retrieval.use_decomposition", True),
            ("retrieval.reranker_model", "BAAI/bge-reranker-base"),
            ("corrective.enabled", True),
            ("corrective.accept_threshold", 0.6),
            ("corrective.abstain_threshold", 0.4),
            ("corrective.max_attempts", 3),
            ("guardrails.check_user_input", True),
            ("guardrails.check_retrieved_context", True),
            ("chunking.chunk_size", 256),
        ],
    )
    def test_changes_when_any_configuration_field_changes(self, field: str, value: object) -> None:
        """Each of these would produce a different answer, so each must miss."""
        assert answer_key(**_with(**{field: value})) != answer_key(**BASE)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("prompt_version", "answer-v2"),
            ("collection", "other_collection"),
            ("question", "a different question"),
            ("question_id", "fact-002"),
        ],
    )
    def test_changes_when_the_request_changes(self, field: str, value: object) -> None:
        """prompt_version is the subtle one: editing a template without bumping
        it would let the cache serve answers from the old prompt while the
        report names the new one."""
        changed = {**BASE, field: value}
        assert answer_key(**changed) != answer_key(**BASE)  # type: ignore[arg-type]


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
