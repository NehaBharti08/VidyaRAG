"""Run a profile against the gold set and record what happened.

The run is deliberately split into two phases:

* **Generate**, sequentially. Embedded Qdrant holds a lock on its index
  directory and cannot be opened concurrently, and answering is the expensive,
  stateful half.
* **Grade**, concurrently. Grading is pure I/O against a rate-limited API, and
  it is the half most likely to fail partway through.

Keeping them apart means a rate limit hit during grading never destroys the
generation work already paid for.

Every run writes a JSON file carrying the full configuration, the model ids,
the gold set digest, and per-sample detail. A number in the README that cannot
be traced back to one of these files is not evidence.

Retrieved passage *text* is held only for the duration of the run and never
serialised: it would add roughly a megabyte of copyrighted textbook prose to
every committed result file. Chunk ids are recorded instead, which are enough
to re-fetch the exact context from the index.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vidyarag.evaluation.abstention import (
    AbstentionStats,
    is_structural_abstention,
    judge_abstention,
    summarise_abstention,
)
from vidyarag.evaluation.answer_cache import AnswerCache, abstention_key, answer_key
from vidyarag.evaluation.goldset import (
    DEFAULT_GOLDSET,
    GoldQuestion,
    QuestionType,
    load_goldset,
    summarise_goldset,
)
from vidyarag.evaluation.metrics import (
    DEFAULT_SCORES_PER_MINUTE,
    METRIC_NAMES,
    MetricSuite,
)
from vidyarag.evaluation.retrieval import score_retrieval
from vidyarag.generate.prompts import ANSWER_PROMPT_VERSION
from vidyarag.pipeline import Pipeline, build_pipeline
from vidyarag.settings import REPO_ROOT, PipelineConfig, Settings, load_pipeline_config

RESULTS_DIR = REPO_ROOT / "eval" / "results"
CACHE_DIR = REPO_ROOT / ".eval_cache"
ANSWER_CACHE_DIR = REPO_ROOT / ".eval_cache" / "answers"
ABSTENTION_CACHE_DIR = REPO_ROOT / ".eval_cache" / "abstention"

ANSWER_GAP_SECONDS = 4.0
"""Minimum spacing between generation calls.

Grading was rate limited from the start; answering was not, which was an
inconsistency rather than a decision. The answering loop fired as fast as
retrieval allowed, tripped a per-minute limit partway through, and then hung
inside the SDK's internal retry -- 18 of 58 questions answered, no output, no
error. Pacing the cheaper half costs a few minutes and removes that failure.

Cache hits skip the wait entirely, so a resumed run is not slowed by questions
it already answered."""

MAX_FAILURE_RATE = 0.10
"""Above this share of failed questions, a run reports no aggregate metrics.

Not a stylistic preference. A run that lost 39 of 58 questions to quota
exhaustion still printed a metrics table, and because the gold set is ordered
factual -> multi-hop -> unanswerable, the survivors were 17 factual, 0
multi-hop and 2 unanswerable. Faithfulness read 0.949 -- measured almost
entirely on the easiest slice, and high *because* of what was missing. A
partial run is not a weaker measurement of the same thing; it is a confident
measurement of a different, easier thing."""


class SampleResult(BaseModel):
    """Everything observed for one gold question."""

    id: str
    question: str
    type: QuestionType
    answer: str = ""
    abstained: bool = False

    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_chunk_ids: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)

    ragas: dict[str, float | None] = Field(default_factory=dict)
    ragas_errors: dict[str, str] = Field(default_factory=dict)
    retrieval: dict[str, float | bool | None] = Field(default_factory=dict)

    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    list_price_usd: float = 0.0

    error: str | None = None
    """Set when answering raised. The sample is kept rather than dropped -- a
    run that silently shrinks when questions fail reports an average over
    whatever happened to succeed."""

    cached: bool = False
    """Whether the answer was reused from a previous run rather than generated.
    Recorded so a report can never imply a fresh measurement it did not make."""

    ranked_chunk_ids: list[str] = Field(default_factory=list)
    """Final ordering after reranking; equal to ``retrieved_chunk_ids`` when
    none ran. Stored separately so MRR is computed from the ranking that
    actually decided what reached the prompt."""

    rerank: dict[str, Any] = Field(default_factory=dict)
    """What reranking changed for this question, when it ran.

    Persisted because a metric moving is not evidence that the component
    credited for it did anything. A reranker that never alters the top 5 cannot
    be the reason context precision improved, and only this would show it."""


class EvalRun(BaseModel):
    """One complete evaluation, with enough context to reproduce it."""

    run_id: str
    created_at: str
    profile: str
    config: dict[str, Any]
    goldset_path: str
    goldset_sha256: str
    goldset_counts: dict[str, int]
    generation_model: str
    grader_model: str
    embedding_model: str
    python_version: str

    samples: list[SampleResult] = Field(default_factory=list)
    aggregates: dict[str, float | None] = Field(default_factory=dict)
    retrieval_aggregates: dict[str, float | None] = Field(default_factory=dict)
    abstention: dict[str, float | int | None] = Field(default_factory=dict)
    totals: dict[str, float] = Field(default_factory=dict)

    @property
    def failed(self) -> list[SampleResult]:
        return [s for s in self.samples if s.error is not None]

    @property
    def failure_rate(self) -> float:
        return len(self.failed) / len(self.samples) if self.samples else 0.0

    @property
    def is_valid(self) -> bool:
        """Whether the aggregates may be quoted as a measurement.

        False means some questions never got an answer, so the averages are
        over whichever subset happened to survive -- and that subset is rarely
        random. See :data:`MAX_FAILURE_RATE`.
        """
        return self.failure_rate <= MAX_FAILURE_RATE

    def failures_by_type(self) -> dict[str, int]:
        """Failure counts per question type.

        Reported because *which* questions were lost matters more than how
        many. Losing every multi-hop question is not a smaller version of
        losing a tenth of each.
        """
        counts: dict[str, int] = {}
        for sample in self.failed:
            counts[sample.type.value] = counts.get(sample.type.value, 0) + 1
        return counts

    def path(self, directory: Path | None = None) -> Path:
        return (directory or RESULTS_DIR) / f"{self.profile}__{self.run_id}.json"

    def save(self, directory: Path | None = None) -> Path:
        target = self.path(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return target


def _mean(values: list[float | None]) -> float | None:
    """Mean of the values that exist.

    ``None`` means "not measured" and is skipped. Treating it as zero would let
    a grader failure masquerade as a bad score.
    """
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _answer_one(pipeline: Pipeline, question: GoldQuestion) -> tuple[SampleResult, list[str]]:
    """Answer one gold question.

    Returns:
        The result, and the context passages given to the model. The passages
        are needed for grading but are not part of the persisted result.
    """
    result = SampleResult(
        id=question.id,
        question=question.question,
        type=question.type,
        gold_chunk_ids=list(question.gold_chunk_ids),
    )
    try:
        answer = pipeline.answer(question.question)
    except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
        result.error = f"{type(exc).__name__}: {exc}"
        return result, []

    trace = answer.trace
    result.answer = answer.text
    result.retrieved_chunk_ids = list(trace.retrieved_chunk_ids)
    result.context_chunk_ids = [c.chunk_id for c in answer.retrieved]
    result.cited_chunk_ids = [c.chunk_id for c in answer.citations]
    result.latency_ms = trace.total_ms
    result.input_tokens = trace.input_tokens
    result.output_tokens = trace.output_tokens
    result.list_price_usd = trace.list_price_usd
    result.abstained = is_structural_abstention(answer.text, trace_abstained=trace.abstained)
    result.rerank = dict(trace.rerank)
    result.ranked_chunk_ids = list(trace.ranked_chunk_ids or trace.retrieved_chunk_ids)

    return result, [c.text for c in answer.retrieved]


def _apply_retrieval_scores(
    result: SampleResult, question: GoldQuestion, config: PipelineConfig
) -> None:
    """Score retrieval from the chunk ids already recorded on the result.

    Deliberately outside the answer cache. These metrics are pure functions of
    ids the result already carries, so computing them here means a bug in them
    is fixed by re-running the report rather than by regenerating every answer
    -- which is exactly what went wrong once: a wrong context-recall formula was
    baked into cached results and could only be corrected by paying the whole
    generation cost again.
    """
    if not question.gold_chunk_ids:
        return
    result.retrieval = score_retrieval(
        result.retrieved_chunk_ids,
        question.gold_chunk_ids,
        k=config.retrieval.top_k_retrieve,
        context_k=config.retrieval.top_k_context,
        ranked_ids=result.ranked_chunk_ids or None,
        context_ids=result.context_chunk_ids or None,
    ).as_dict()


async def _grade_all(
    suite: MetricSuite,
    results: list[SampleResult],
    questions: dict[str, GoldQuestion],
    contexts: dict[str, list[str]],
    verdicts: AnswerCache,
    *,
    judge_model: str,
    on_graded: Callable[[SampleResult], None] | None = None,
) -> None:
    """Grade every sample: abstention everywhere, RAGAS where it applies."""

    async def grade_one(result: SampleResult) -> None:
        question = questions[result.id]

        # Abstention is checked on every question. On answerable ones it
        # measures the cost of over-refusing, which precision alone hides.
        if not result.abstained and result.answer:
            key = abstention_key(
                judge_model=judge_model,
                question=result.question,
                answer=result.answer,
            )
            entry = verdicts.get(key)
            if entry is not None:
                result.abstained = bool(entry["abstained"])
            else:
                # Shares the grader's quota, so it shares the grader's pacing.
                await suite.limiter.acquire()
                started = asyncio.get_running_loop().time()
                result.abstained = await judge_abstention(
                    suite.client,
                    model=judge_model,
                    question=result.question,
                    answer=result.answer,
                )
                suite.limiter.observe(asyncio.get_running_loop().time() - started)
                verdicts.put(key, {"abstained": result.abstained})

        # RAGAS needs a reference and a genuine attempt. A refusal has no
        # faithfulness to measure; scoring it would invent a number.
        passages = contexts.get(result.id, [])
        if question.is_answerable and question.reference and not result.abstained and passages:
            scores = await suite.score(
                question=result.question,
                answer=result.answer,
                contexts=passages,
                reference=question.reference,
            )
            result.ragas = scores.as_dict()
            result.ragas_errors = scores.errors

        if on_graded is not None:
            on_graded(result)

    await asyncio.gather(*(grade_one(r) for r in results if r.error is None))


def run_evaluation(
    *,
    profile: str | None = None,
    goldset_path: Path | None = None,
    limit: int | None = None,
    concurrency: int = 3,
    scores_per_minute: float = DEFAULT_SCORES_PER_MINUTE,
    use_cache: bool = True,
    settings: Settings | None = None,
    on_answered: Callable[[SampleResult], None] | None = None,
    on_graded: Callable[[SampleResult], None] | None = None,
) -> EvalRun:
    """Run one profile over the gold set.

    Args:
        profile: Profile name. Defaults to the configured one.
        goldset_path: Gold set JSONL. Defaults to the committed v1 set.
        limit: Use only the first N questions, for smoke runs during
            development. The full set is for phase boundaries.
        concurrency: Simultaneous grading requests.
        use_cache: Reuse cached grader responses, which makes re-running after
            a code change close to free.
        settings: Overrides environment settings.
        on_answered: Called after each question is answered.
        on_graded: Called after each question is graded.

    Returns:
        The completed run, not yet written to disk.
    """
    resolved_settings = settings or Settings()
    resolved_profile = profile or resolved_settings.profile
    config = load_pipeline_config(resolved_profile)

    resolved_goldset = goldset_path or DEFAULT_GOLDSET
    questions = load_goldset(resolved_goldset)
    if limit is not None:
        questions = questions[:limit]

    run = EvalRun(
        run_id=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        created_at=datetime.now(UTC).isoformat(),
        profile=config.name,
        config=config.model_dump(mode="json"),
        goldset_path=resolved_goldset.name,
        goldset_sha256=_digest(resolved_goldset),
        goldset_counts=summarise_goldset(questions),
        generation_model=config.generation_model,
        grader_model=config.grader_model,
        embedding_model=config.embedding_model,
        python_version=platform.python_version(),
    )

    # --- Phase 1: generate, sequentially -----------------------------------
    #
    # Answers are cached so a run stopped by a daily quota resumes tomorrow
    # instead of restarting. Without it a 58-question run that died at
    # question 19 discarded those 19 and spent the same quota recomputing them.
    contexts: dict[str, list[str]] = {}
    answers = AnswerCache(ANSWER_CACHE_DIR if use_cache else None)
    answered_live = 0
    pipeline = build_pipeline(resolved_settings, config)
    try:
        for question in questions:
            key = answer_key(
                profile=config.name,
                generation_model=config.generation_model,
                embedding_model=config.embedding_model,
                temperature=config.temperature,
                top_k_retrieve=config.retrieval.top_k_retrieve,
                top_k_context=config.retrieval.top_k_context,
                prompt_version=ANSWER_PROMPT_VERSION,
                collection=resolved_settings.qdrant_collection,
                question_id=question.id,
                question=question.question,
            )
            cached = answers.get(key)
            if cached is not None:
                result = SampleResult.model_validate(cached["result"])
                passages = list(cached.get("contexts", []))
                result.cached = True
            else:
                # Space out real calls only. A cache hit costs nothing and must
                # not be slowed down, or resuming a long run would be as slow as
                # the original.
                if answered_live:
                    time.sleep(ANSWER_GAP_SECONDS)
                answered_live += 1
                result, passages = _answer_one(pipeline, question)
                # Only successful answers are cached. Caching a quota failure
                # would make the failure permanent and silently shrink every
                # later run to the same truncated subset.
                if result.error is None:
                    answers.put(
                        key,
                        {"result": result.model_dump(mode="json"), "contexts": passages},
                    )
            # Scored here, outside the cache, so a metric fix does not require
            # regenerating answers that were already paid for.
            _apply_retrieval_scores(result, question, config)
            contexts[result.id] = passages
            run.samples.append(result)
            if on_answered is not None:
                on_answered(result)
    finally:
        pipeline.close()

    # --- Phase 2: grade, concurrently --------------------------------------
    suite = MetricSuite(
        resolved_settings.google_api_key.get_secret_value(),
        grader_model=config.grader_model,
        embedding_model=config.embedding_model,
        cache_dir=CACHE_DIR if use_cache else None,
        concurrency=concurrency,
        scores_per_minute=scores_per_minute,
    )
    by_id = {q.id: q for q in questions}
    # Abstention verdicts are the one grading call RAGAS does not cache. With
    # one per question, paced against a free-tier quota, they alone put a ~12
    # minute floor under a re-run whose every other call came from disk.
    verdicts = AnswerCache(ABSTENTION_CACHE_DIR if use_cache else None)

    async def grade() -> None:
        try:
            await _grade_all(
                suite,
                run.samples,
                by_id,
                contexts,
                verdicts,
                judge_model=config.grader_model,
                on_graded=on_graded,
            )
        finally:
            await suite.aclose()

    asyncio.run(grade())

    _aggregate(run, by_id)
    return run


def _aggregate(run: EvalRun, questions: dict[str, GoldQuestion]) -> None:
    """Compute run-level aggregates from per-sample results."""
    graded = [s for s in run.samples if s.ragas]

    run.aggregates = {name: _mean([s.ragas.get(name) for s in graded]) for name in METRIC_NAMES}
    run.aggregates["graded_samples"] = float(len(graded))

    scored = [s for s in run.samples if s.retrieval]
    for key in ("recall_at_k", "reciprocal_rank", "context_recall_at_context"):
        run.retrieval_aggregates[key] = _mean([_as_float(s.retrieval.get(key)) for s in scored])
    hits = [s.retrieval.get("hit_at_k") for s in scored]
    present = [bool(h) for h in hits if h is not None]
    run.retrieval_aggregates["hit_rate"] = sum(present) / len(present) if present else None

    stats: AbstentionStats = summarise_abstention(
        [
            (questions[s.id].is_answerable, s.abstained)
            for s in run.samples
            if s.id in questions and s.error is None
        ]
    )
    run.abstention = stats.as_dict()

    answered = [s for s in run.samples if s.error is None]
    run.totals = {
        "questions": float(len(run.samples)),
        "failed": float(sum(1 for s in run.samples if s.error)),
        "input_tokens": float(sum(s.input_tokens for s in answered)),
        "output_tokens": float(sum(s.output_tokens for s in answered)),
        "list_price_usd": sum(s.list_price_usd for s in answered),
        "mean_latency_ms": (
            sum(s.latency_ms for s in answered) / len(answered) if answered else 0.0
        ),
    }


def _as_float(value: float | bool | None) -> float | None:
    if value is None:
        return None
    return float(value)


def load_run(path: Path) -> EvalRun:
    """Load a previously written run."""
    return EvalRun.model_validate(json.loads(path.read_text(encoding="utf-8")))


def latest_run(profile: str, directory: Path | None = None) -> EvalRun | None:
    """Most recent committed run for a profile, if any."""
    folder = directory or RESULTS_DIR
    if not folder.exists():
        return None
    candidates = sorted(folder.glob(f"{profile}__*.json"))
    return load_run(candidates[-1]) if candidates else None
