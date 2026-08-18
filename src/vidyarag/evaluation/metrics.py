"""RAGAS metrics, and the two workarounds needed to drive them with Gemini.

**This is the only module that imports RAGAS.** Everything else speaks
:class:`SampleScores`. RAGAS moves fast and has broken twice already during
this project, so the blast radius of the next change is one file.

Two upstream problems are handled here, both worth knowing about:

1. ``ragas`` imports ``langchain_community.chat_models.vertexai``, which
   langchain-community deleted in 0.4.0, so ``import ragas`` fails outright
   (ragas #2741, #2745). Pinned in ``pyproject.toml``, not here.
2. ragas decides whether an LLM client is async by looking for
   ``chat.completions.create`` -- an OpenAI shape. A ``google.genai.Client``
   has no such attribute, so ragas concludes "synchronous", while the
   collections metrics only expose an async path. Gemini therefore cannot
   drive them at all as shipped.

The fix for (2) is Google's own OpenAI-compatibility endpoint: an
``AsyncOpenAI`` client pointed at ``generativelanguage.googleapis.com``
satisfies the detection and still calls Gemini. The ``openai`` package is a
*client library* here, not a model provider -- no OpenAI account or spend is
involved anywhere in this project.

Embeddings deliberately reuse the same local model that built the index. Using
a different one would score answer relevancy in a vector space unrelated to the
one retrieval actually searched, which is both misleading and needlessly paid.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vidyarag.llm.provider import embed_texts

if TYPE_CHECKING:  # pragma: no cover
    from openai import AsyncOpenAI

# Google's OpenAI-compatible surface. Documented and stable; see
# https://ai.google.dev/gemini-api/docs/openai
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

# Free-tier Gemini is rate limited per minute, and grading is far more
# request-hungry than it looks: one `ascore` is several HTTP calls, because
# faithfulness extracts statements and then runs an entailment check over them.
#
# Measured on a 6-question smoke run at concurrency 3 with no pacing: ~60
# requests in 163s (~22/min), and 7 of 24 metric calls died on HTTP 429. A
# semaphore alone cannot fix that -- it bounds how many calls are in flight,
# not how many happen per minute.
#
# So requests are paced by a rate limiter instead. The default assumes ~2.5
# HTTP calls per ascore against a 15/min quota.
DEFAULT_CONCURRENCY = 3
DEFAULT_SCORES_PER_MINUTE = 6.0

# Quotas reset on a clock, so a retry that waits out the window succeeds where
# a fast exponential retry just burns attempts against the same exhausted minute.
MAX_ATTEMPTS = 5
BACKOFF_SECONDS = 30.0


CACHE_HIT_SECONDS = 0.25
"""Below this, a call cannot have crossed the network and must have been served
from RAGAS's disk cache. Used to decide whether pacing is needed at all."""


class RateLimiter:
    """Paces async operations to a fixed rate, skipping the wait when it buys nothing.

    Each acquirer reserves the next slot and sleeps until it arrives, so callers
    are spread evenly rather than released in a burst that immediately trips the
    quota the pacing exists to respect.

    **Pacing is suspended while calls are being served from cache.** Grader
    responses are cached on disk, so re-running a profile after a code change is
    mostly cache hits -- and a cache hit consumes no quota. Pacing those was
    pure waiting: a fully cached 58-question run took the same ~50 minutes as
    the uncached one, which makes the Phase 4 ablations far more expensive than
    they need to be, and discourages exactly the re-running the harness exists
    to enable.

    The limiter therefore watches how long calls take. It starts unpaced; the
    first call that takes long enough to have crossed the network turns pacing
    on, and a run that goes back to cache hits turns it off again. Up to
    `concurrency` calls can slip through at the transition, which the retry and
    backoff already handle.
    """

    def __init__(self, per_minute: float) -> None:
        self._interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_slot = 0.0
        self._pacing = False
        """Off until a call proves slow enough to have been a real request."""

    @property
    def pacing(self) -> bool:
        return self._pacing

    def observe(self, elapsed_seconds: float) -> None:
        """Record how long a call took, to decide whether the next needs pacing."""
        self._pacing = elapsed_seconds >= CACHE_HIT_SECONDS

    async def acquire(self) -> None:
        if self._interval <= 0 or not self._pacing:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:
            now = loop.time()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        delay = slot - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)


@dataclass(frozen=True, slots=True)
class SampleScores:
    """RAGAS scores for one question.

    Every field is optional. A metric that fails is recorded as ``None`` with
    its reason in :attr:`errors` rather than as ``0.0`` -- scoring a crashed
    metric as zero would silently drag an average down and misattribute a
    harness bug to the pipeline being measured.
    """

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    errors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | None]:
        return {name: getattr(self, name) for name in METRIC_NAMES}

    @property
    def complete(self) -> bool:
        """Whether every metric produced a value."""
        return all(getattr(self, name) is not None for name in METRIC_NAMES)


def _clean(value: Any) -> float | None:
    """Coerce a RAGAS result to a float, mapping NaN to ``None``.

    RAGAS returns NaN when it cannot score a sample (an empty statement list,
    an unparseable response). NaN would poison any later mean, and it is not a
    score -- it is a missing one.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


class LocalEmbedder:
    """The project's local fastembed model, in the shape RAGAS wants.

    Deliberately inherits from nothing. RAGAS lives in the optional ``eval``
    dependency group, so a base class from it would make this module -- and the
    tests covering it -- unimportable in a normal install.
    :func:`_build_ragas_embeddings` adapts it by delegation instead.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        return embed_texts([text], self.model_name)[0]

    async def aembed_text(self, text: str) -> list[float]:
        # fastembed is synchronous CPU work. Handing it to a thread keeps it
        # from blocking the event loop while other samples are in flight.
        return await asyncio.to_thread(self.embed_text, text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts, self.model_name)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


def _build_ragas_embeddings(model_name: str) -> Any:
    """Wrap :class:`LocalEmbedder` in RAGAS's embedding base class.

    Delegation rather than multiple inheritance: RAGAS's base defines its own
    ``__init__`` taking a cache, so inheriting from both would put the wrong
    initialiser first in the MRO.
    """
    from ragas.embeddings.base import BaseRagasEmbedding

    impl = LocalEmbedder(model_name)

    class _Adapter(BaseRagasEmbedding):
        def embed_text(self, text: str, **_: Any) -> list[float]:
            return impl.embed_text(text)

        async def aembed_text(self, text: str, **_: Any) -> list[float]:
            return await impl.aembed_text(text)

        def embed_texts(self, texts: list[str], **_: Any) -> list[list[float]]:
            return impl.embed_documents(texts)

        async def aembed_texts(self, texts: list[str], **_: Any) -> list[list[float]]:
            return await impl.aembed_documents(texts)

    return _Adapter()


class MetricSuite:
    """Scores samples with the four RAGAS metrics.

    Args:
        api_key: Gemini API key.
        grader_model: Model used for grading. Distinct from the generation
            model so the system is not the sole judge of its own output.
        embedding_model: fastembed model id, matching the one used at ingest.
        cache_dir: Enables RAGAS's on-disk LLM cache. Re-running after a code
            change then costs almost nothing, which is what makes iterating on
            the harness affordable on a free tier.
        concurrency: Simultaneous in-flight samples.
    """

    def __init__(
        self,
        api_key: str,
        *,
        grader_model: str,
        embedding_model: str,
        cache_dir: Path | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        scores_per_minute: float = DEFAULT_SCORES_PER_MINUTE,
    ) -> None:
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey"
            )
        self.grader_model = grader_model
        self.embedding_model = embedding_model
        self._semaphore = asyncio.Semaphore(concurrency)
        self.limiter = RateLimiter(scores_per_minute)
        self._client = self._build_client(api_key)
        self._llm = self._build_llm(cache_dir)
        self._metrics = self._build_metrics()

    @property
    def client(self) -> AsyncOpenAI:
        """The underlying async client, shared with the abstention judge.

        Reusing one client keeps connection pooling and the configured base URL
        in a single place, rather than opening a second one that could drift.
        """
        return self._client

    @staticmethod
    def _build_client(api_key: str) -> AsyncOpenAI:
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL)

    def _build_llm(self, cache_dir: Path | None) -> Any:
        from ragas.llms import llm_factory

        cache = None
        if cache_dir is not None:
            from ragas.cache import DiskCacheBackend

            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = DiskCacheBackend(cache_dir=str(cache_dir))

        # provider="openai" describes the wire protocol, not the vendor: the
        # client points at Gemini.
        return llm_factory(
            model=self.grader_model,
            provider="openai",
            client=self._client,
            cache=cache,
        )

    def _build_metrics(self) -> dict[str, Any]:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )

        embeddings = _build_ragas_embeddings(self.embedding_model)
        return {
            "faithfulness": Faithfulness(llm=self._llm),
            "answer_relevancy": AnswerRelevancy(llm=self._llm, embeddings=embeddings),
            "context_precision": ContextPrecisionWithReference(llm=self._llm),
            "context_recall": ContextRecall(llm=self._llm),
        }

    async def _score_one(self, name: str, **kwargs: Any) -> tuple[float | None, str | None]:
        """Run a single metric, retrying transient failures.

        Rate limits and truncated streams are the common failures on a free
        tier and are worth retrying. A run that dies two thirds of the way
        through wastes every call it already paid for.
        """
        metric = self._metrics[name]
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                await self.limiter.acquire()
                started = asyncio.get_running_loop().time()
                async with self._semaphore:
                    result = await metric.ascore(**kwargs)
                # Feed the duration back so the limiter can tell a cached call
                # from a real one and stop pacing work that costs no quota.
                self.limiter.observe(asyncio.get_running_loop().time() - started)
                return _clean(getattr(result, "value", result)), None
            except Exception as exc:  # noqa: BLE001 - any failure is recorded, never fatal
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt == MAX_ATTEMPTS:
                    break
                # Linear, not exponential. Per-minute quotas recover on a clock,
                # so the useful thing is to wait out the current window rather
                # than to back off ever further from it.
                await asyncio.sleep(BACKOFF_SECONDS * attempt)
        return None, last_error

    async def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference: str,
    ) -> SampleScores:
        """Score one answered question against its reference.

        Only meaningful for answerable questions. Unanswerable ones are scored
        by abstention behaviour instead -- a refusal has no faithfulness to
        measure, and forcing it through these metrics would produce a confident
        number about nothing.
        """
        jobs: dict[str, dict[str, Any]] = {
            "faithfulness": {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
            },
            "answer_relevancy": {"user_input": question, "response": answer},
            "context_precision": {
                "user_input": question,
                "reference": reference,
                "retrieved_contexts": contexts,
            },
            "context_recall": {
                "user_input": question,
                "retrieved_contexts": contexts,
                "reference": reference,
            },
        }

        results = await asyncio.gather(
            *(self._score_one(name, **kwargs) for name, kwargs in jobs.items())
        )

        values: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for name, (value, error) in zip(jobs, results, strict=True):
            values[name] = value
            if error:
                errors[name] = error

        return SampleScores(
            faithfulness=values["faithfulness"],
            answer_relevancy=values["answer_relevancy"],
            context_precision=values["context_precision"],
            context_recall=values["context_recall"],
            errors=errors,
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
