"""Configuration loading and validation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vidyarag.settings import (
    IN_MEMORY,
    PipelineConfig,
    QdrantMode,
    Settings,
    _deep_merge,
    load_pipeline_config,
)


class TestSettings:
    def test_defaults_to_embedded_mode(self) -> None:
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.qdrant_mode is QdrantMode.EMBEDDED

    def test_server_mode_requires_url(self) -> None:
        with pytest.raises(ValueError, match="QDRANT_URL is required"):
            Settings(_env_file=None, QDRANT_MODE="server")  # type: ignore[call-arg]

    def test_cloud_mode_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="QDRANT_API_KEY is required"):
            Settings(  # type: ignore[call-arg]
                _env_file=None,
                QDRANT_MODE="cloud",
                QDRANT_URL="https://example.qdrant.io:6333",
            )

    def test_cloud_mode_accepts_full_config(self) -> None:
        settings = Settings(  # type: ignore[call-arg]
            _env_file=None,
            QDRANT_MODE="cloud",
            QDRANT_URL="https://example.qdrant.io:6333",
            QDRANT_API_KEY="secret",
        )
        assert settings.qdrant_mode is QdrantMode.CLOUD

    def test_secrets_are_not_exposed_by_repr(self) -> None:
        """A settings object must be safe to log."""
        settings = Settings(_env_file=None, GOOGLE_API_KEY="AQ.super-secret")  # type: ignore[call-arg]
        assert "AQ.super-secret" not in repr(settings)
        assert settings.google_api_key.get_secret_value() == "AQ.super-secret"

    def test_memory_sentinel_is_preserved(self) -> None:
        settings = Settings(_env_file=None, QDRANT_PATH=IN_MEMORY)  # type: ignore[call-arg]
        assert settings.resolved_qdrant_path == IN_MEMORY

    def test_relative_path_resolves_to_absolute(self) -> None:
        settings = Settings(_env_file=None, QDRANT_PATH="data/index")  # type: ignore[call-arg]
        assert Path(settings.resolved_qdrant_path).is_absolute()


class TestDeepMerge:
    def test_overlay_replaces_scalars(self) -> None:
        assert _deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    def test_overlay_recurses_into_mappings(self) -> None:
        base = {"retrieval": {"top_k_retrieve": 20, "use_hybrid": False}}
        override = {"retrieval": {"use_hybrid": True}}
        assert _deep_merge(base, override) == {
            "retrieval": {"top_k_retrieve": 20, "use_hybrid": True}
        }

    def test_inputs_are_not_mutated(self) -> None:
        base = {"a": {"b": 1}}
        _deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestPipelineConfig:
    def test_baseline_profile_loads(self, config_dir: Path) -> None:
        cfg = load_pipeline_config("baseline", config_dir=config_dir)
        assert cfg.name == "baseline"

    def test_baseline_is_a_true_control_group(self, config_dir: Path) -> None:
        """The baseline must have every enhancement disabled.

        This is the frozen control for every reported delta. If a later phase
        accidentally enables a feature here, every number in the README becomes
        wrong -- so assert it rather than trust it.
        """
        cfg = load_pipeline_config("baseline", config_dir=config_dir)
        assert cfg.retrieval.use_hybrid is False
        assert cfg.retrieval.use_reranker is False
        assert cfg.retrieval.use_decomposition is False
        assert cfg.corrective.enabled is False

    def test_defaults_inherit_from_base_config(self, config_dir: Path) -> None:
        """baseline.yaml does not set chunk_size; it must come from default.yaml."""
        cfg = load_pipeline_config("baseline", config_dir=config_dir)
        assert cfg.chunking.chunk_size == 512
        assert cfg.retrieval.top_k_retrieve == 20

    def test_retrieval_pool_exceeds_context_window(self, config_dir: Path) -> None:
        """Reranking is pointless unless the candidate pool is wider than the selection."""
        cfg = load_pipeline_config("baseline", config_dir=config_dir)
        assert cfg.retrieval.top_k_retrieve > cfg.retrieval.top_k_context

    def test_unknown_profile_lists_available_ones(self, config_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Available: baseline"):
            load_pipeline_config("does-not-exist", config_dir=config_dir)

    def test_unknown_key_is_rejected(self) -> None:
        """A typo in a profile must fail loudly, not silently invalidate a run."""
        with pytest.raises(ValueError, match=r"extra_inputs_not_permitted|Extra inputs"):
            PipelineConfig.model_validate({"retrieval": {"top_k_retreive": 20}})

    def test_env_example_names_a_profile_that_exists(self, config_dir: Path) -> None:
        """`.env.example` is the quickstart. It shipped `VIDYARAG_PROFILE=corrective`
        while only `baseline.yaml` existed, so a fresh clone following the README
        crashed on the first command that read a profile."""
        example = (config_dir.parent / ".env.example").read_text(encoding="utf-8")
        match = re.search(r"^VIDYARAG_PROFILE=(.+)$", example, re.MULTILINE)
        assert match, ".env.example must set VIDYARAG_PROFILE"
        load_pipeline_config(match.group(1).strip(), config_dir=config_dir)

    def test_every_shipped_profile_loads(self, config_dir: Path) -> None:
        """A broken profile should fail the build, not only a run that selects it."""
        profiles = sorted(p.stem for p in (config_dir / "profiles").glob("*.yaml"))
        assert profiles
        for name in profiles:
            assert load_pipeline_config(name, config_dir=config_dir).name == name
