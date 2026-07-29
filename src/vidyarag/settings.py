"""Runtime configuration.

Two layers, deliberately separated:

* :class:`Settings` -- *deployment* concerns that vary by environment and
  include secrets. Sourced from environment variables / ``.env``.
* :class:`PipelineConfig` -- *behavioural* concerns that define what the RAG
  pipeline actually does. Sourced from ``config/default.yaml`` overlaid with
  ``config/profiles/<profile>.yaml``.

The split matters for evaluation: a profile is a reproducible description of a
pipeline variant that can be committed, diffed, and cited in a results table.
Secrets must never end up in one.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"

IN_MEMORY = ":memory:"


class QdrantMode(enum.StrEnum):
    """How to reach Qdrant.

    All three resolve to the same ``QdrantClient`` API, so calling code never
    branches on this -- only :func:`vidyarag.store.client.build_client` does.
    """

    EMBEDDED = "embedded"
    """In-process, no server. Persists to ``qdrant_path`` (or ``:memory:``).

    This is what ships with the deployed demo. A free Qdrant Cloud cluster is
    suspended after ~1 week idle and deleted after ~4 weeks, which would break
    a portfolio link months after it was last touched; an embedded index has no
    such failure mode.
    """

    SERVER = "server"
    """Local Qdrant over HTTP (docker-compose)."""

    CLOUD = "cloud"
    """Qdrant Cloud. Used for development and evaluation runs."""


class Settings(BaseSettings):
    """Environment-sourced configuration. Contains secrets; never serialise it."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="OPENAI_API_KEY",
    )

    qdrant_mode: QdrantMode = Field(default=QdrantMode.EMBEDDED, validation_alias="QDRANT_MODE")
    qdrant_url: str | None = Field(default=None, validation_alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_path: str = Field(default="data/index", validation_alias="QDRANT_PATH")
    qdrant_collection: str = Field(
        default="vidyarag_biology_v1", validation_alias="QDRANT_COLLECTION"
    )

    profile: str = Field(default="baseline", validation_alias="VIDYARAG_PROFILE")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: Literal["json", "console"] = Field(default="json", validation_alias="LOG_FORMAT")

    @model_validator(mode="after")
    def _check_mode_requirements(self) -> Settings:
        """Fail at startup rather than at first query.

        A missing Qdrant URL surfaces as a confusing connection error deep in a
        retrieval call; catching it here keeps the failure legible.
        """
        if self.qdrant_mode in (QdrantMode.SERVER, QdrantMode.CLOUD) and not self.qdrant_url:
            raise ValueError(f"QDRANT_URL is required when QDRANT_MODE={self.qdrant_mode.value}")
        if self.qdrant_mode is QdrantMode.CLOUD and not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY is required when QDRANT_MODE=cloud")
        return self

    @property
    def resolved_qdrant_path(self) -> str:
        """Absolute on-disk index path, or the ``:memory:`` sentinel unchanged."""
        if self.qdrant_path == IN_MEMORY:
            return IN_MEMORY
        path = Path(self.qdrant_path)
        return str(path if path.is_absolute() else REPO_ROOT / path)


class ChunkingConfig(BaseModel):
    """How source documents become retrievable units."""

    model_config = ConfigDict(extra="forbid")

    chunk_size: int = 512
    chunk_overlap: int = 64
    respect_sentence_boundaries: bool = True


class RetrievalConfig(BaseModel):
    """Candidate generation and narrowing.

    ``top_k_retrieve`` > ``top_k_context`` on purpose: reranking can only help
    if it is given a pool larger than the final selection.
    """

    model_config = ConfigDict(extra="forbid")

    top_k_retrieve: int = 20
    top_k_context: int = 5
    use_hybrid: bool = False
    use_reranker: bool = False
    use_decomposition: bool = False
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    sparse_model: str = "Qdrant/bm25"


class CorrectiveConfig(BaseModel):
    """Self-check loop policy.

    Thresholds are starting points to be tuned against the gold set in Phase 5,
    not asserted constants. See docs/EVALUATION.md for the sweep.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    accept_threshold: float = 0.8
    abstain_threshold: float = 0.5
    max_attempts: int = 2


class GuardrailConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_user_input: bool = False
    check_retrieved_context: bool = False


class PipelineConfig(BaseModel):
    """A complete, reproducible description of one pipeline variant."""

    model_config = ConfigDict(extra="forbid")

    name: str = "baseline"
    description: str = ""
    generation_model: str = "gpt-4o-mini"
    grader_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    corrective: CorrectiveConfig = Field(default_factory=CorrectiveConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``override`` onto ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def load_pipeline_config(profile: str, config_dir: Path | None = None) -> PipelineConfig:
    """Load ``default.yaml`` overlaid with ``profiles/<profile>.yaml``.

    ``extra="forbid"`` on the models means a typo in a profile key is a startup
    error, not a silently ignored setting that quietly invalidates a benchmark.
    """
    directory = config_dir or CONFIG_DIR
    base = _read_yaml(directory / "default.yaml")
    profile_path = directory / "profiles" / f"{profile}.yaml"
    if not profile_path.exists():
        available = sorted(p.stem for p in (directory / "profiles").glob("*.yaml"))
        raise FileNotFoundError(
            f"Unknown profile {profile!r}. Available: {', '.join(available) or '(none)'}"
        )
    merged = _deep_merge(base, _read_yaml(profile_path))
    merged.setdefault("name", profile)
    return PipelineConfig.model_validate(merged)
