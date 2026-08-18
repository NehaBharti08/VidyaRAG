"""Disk cache for generated answers, so an interrupted run can resume.

Grader responses were already cached. Generated answers were not, which made a
run all-or-nothing: a 58-question evaluation that exhausted its daily quota at
question 19 threw away those 19 answers and started over the next day, spending
the same quota to recompute what it already knew.

That is not a theoretical concern on a free tier. It is the difference between
a benchmark that can be completed across two days and one that cannot be
completed at all.

**The cache key is the safety property here.** Anything that changes what the
model would answer must be part of it, or a cached answer gets attributed to a
configuration that never produced it -- the same class of silent
misattribution that ``extra="forbid"`` prevents in profile loading, and just as
hard to notice afterwards, because the number looks perfectly reasonable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_VERSION = "1"
"""Bump to invalidate every entry at once, if the cached shape changes."""


def answer_key(
    *,
    profile: str,
    generation_model: str,
    embedding_model: str,
    temperature: float,
    top_k_retrieve: int,
    top_k_context: int,
    prompt_version: str,
    collection: str,
    question_id: str,
    question: str,
) -> str:
    """Stable identity for one (configuration, question) pair.

    Every argument is load-bearing. ``prompt_version`` in particular: editing a
    template without bumping its version would let this cache serve answers
    from the old prompt while the report claims the new one.
    """
    payload = "\x1f".join(
        (
            CACHE_VERSION,
            profile,
            generation_model,
            embedding_model,
            f"{temperature:.4f}",
            str(top_k_retrieve),
            str(top_k_context),
            prompt_version,
            collection,
            question_id,
            question,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def abstention_key(*, judge_model: str, question: str, answer: str) -> str:
    """Stable identity for one abstention verdict.

    The judgement is "did this answer decline to answer?", which depends only on
    the answer text, the question, and the model asked. Caching it matters
    because it is the one grading call RAGAS does not cache: with 58 of them
    paced against a free-tier quota, they alone put a ~12 minute floor under a
    re-run whose every other call was already served from disk.
    """
    payload = "\x1f".join((CACHE_VERSION, "abstention", judge_model, question, answer))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class AnswerCache:
    """JSON-per-entry cache. Absent directory disables it entirely."""

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a cached entry, or ``None``.

        A corrupt entry is treated as a miss rather than an error: a half-written
        file from an interrupted run should cost one recomputation, not the run.
        """
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        """Store an entry. Failures are ignored -- a cache miss is never fatal."""
        if not self.enabled:
            return
        try:
            self._path(key).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            return
