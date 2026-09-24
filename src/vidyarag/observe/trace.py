"""Per-query tracing: where the time and the tokens went.

Every query carries a :class:`QueryTrace` from start to finish. It records how
long each stage took, how many tokens each model call consumed, and what that
would cost at list price. The trace is returned with the answer rather than
only logged, so latency and cost are inspectable from the API and the UI, not
just from a log file nobody opens.

On cost: generation runs on Gemini's free tier, so the real spend is zero. The
USD figure is what the same traffic would cost at published rates, and it is
labelled that way everywhere it surfaces. Reporting "$0.00/query" and stopping
there would hide the thing that actually matters -- whether the design is
economical enough to run for real -- and a reader who assumes the number means
"this is free forever" has been misled by a true statement.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

# Published Gemini rates, USD per million tokens, verified August 2026.
# Used only to price a trace; nothing here is billed on the free tier.
GEMINI_PRICING: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-3.6-flash": (0.30, 2.50),
}
_FALLBACK_PRICING = (0.30, 2.50)


def price_call(model: str, input_tokens: int, output_tokens: int) -> float:
    """List-price cost of one model call, in USD.

    Unknown models fall back to Flash rates rather than reporting zero: a
    silently free-looking call is worse than a slightly wrong estimate.
    """
    rate_in, rate_out = GEMINI_PRICING.get(model, _FALLBACK_PRICING)
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


class StageTiming(BaseModel):
    """How long one pipeline stage took."""

    name: str
    duration_ms: float
    depth: int = 0
    """Nesting level: 0 for a stage the pipeline entered directly.

    The corrective loop wraps retrieval, reranking and generation, so its stage
    contains theirs. Summing every recorded stage therefore counted the inner
    work twice and reported roughly double the real latency. Depth is what makes
    the outer and inner measurements distinguishable rather than additive."""

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.name}={self.duration_ms:.0f}ms"


class Usage(BaseModel):
    """Token usage for one model call."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    purpose: str = "generation"
    """Which job this call did -- generation, grading, decomposition."""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def list_price_usd(self) -> float:
        return price_call(self.model, self.input_tokens, self.output_tokens)


class QueryTrace(BaseModel):
    """Everything measurable about answering one question."""

    query: str
    profile: str = "baseline"
    prompt_version: str = ""
    """Which prompt template produced the answer. Enters every trace so a
    result can never be attributed to a prompt that did not generate it."""

    stages: list[StageTiming] = Field(default_factory=list)
    usage: list[Usage] = Field(default_factory=list)
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    """First-stage candidates, before any reordering."""

    ranked_chunk_ids: list[str] = Field(default_factory=list)
    """Final ordering after reranking. Equal to the above when none ran.

    Kept separate because MRR is a statement about rank: computing it from the
    pre-rerank list makes a reranker structurally unable to move it."""

    cited_chunk_ids: list[str] = Field(default_factory=list)
    attempts: int = 1
    abstained: bool = False

    sub_questions: list[str] = Field(default_factory=list)
    """How a multi-hop question was split, when decomposition ran.

    Empty both when decomposition is off and when it judged the question
    atomic -- the report distinguishes those by profile, and the split rate
    is itself a result worth seeing."""

    corrective: dict[str, object] = Field(default_factory=dict)
    """What the self-check loop did: attempts, decisions, scores per attempt.

    Empty when the loop is off. Recorded per query because how often the loop
    fires, and what it changed when it did, is the result -- a loop that never
    fires has not been shown to work, and one that fires constantly is
    papering over bad retrieval rather than correcting it."""

    guard_input: dict[str, object] = Field(default_factory=dict)
    """Set when the input guard blocked a question. Empty otherwise."""

    guard_context: dict[str, object] = Field(default_factory=dict)
    """Set when retrieved passages were quarantined. Empty otherwise, so a
    clean query carries no guard noise through the logs."""

    rerank: dict[str, object] = Field(default_factory=dict)
    """What reranking changed, when it ran. Empty when it did not.

    Kept per query so an ablation can report *how* a score moved rather than
    only that it did -- a reranker that improves a metric while never altering
    the top 5 has not earned the credit."""

    wall_ms: float = 0.0
    """Measured wall-clock time for the whole query, set by the pipeline.

    The authoritative latency. Stage timings are diagnostics and cannot be
    added up into a total: they nest, and they exclude whatever happens between
    them."""

    _depth: int = PrivateAttr(default=0)

    @property
    def total_ms(self) -> float:
        """How long the query took.

        Wall clock when the pipeline measured it, otherwise the sum of the
        outermost stages. Never the sum of every stage: the corrective loop
        contains the stages it runs, and adding those to their own container
        reported ~22s for queries that took ~11s."""
        if self.wall_ms:
            return self.wall_ms
        return sum(stage.duration_ms for stage in self.stages if stage.depth == 0)

    @property
    def stage_sum_ms(self) -> float:
        """Sum of every recorded stage, nested ones included.

        Kept for diagnostics only. It is larger than :attr:`total_ms` whenever a
        stage contains another, and quoting it as latency is the bug this
        property exists to keep visible rather than accidental."""
        return sum(stage.duration_ms for stage in self.stages)

    @property
    def input_tokens(self) -> int:
        return sum(u.input_tokens for u in self.usage)

    @property
    def output_tokens(self) -> int:
        return sum(u.output_tokens for u in self.usage)

    @property
    def list_price_usd(self) -> float:
        """What this query would cost at published rates. Actual spend is $0."""
        return sum(u.list_price_usd for u in self.usage)

    def stage_ms(self, name: str) -> float:
        """Total time attributed to one stage name."""
        return sum(s.duration_ms for s in self.stages if s.name == name)

    def tokens_by_purpose(self) -> dict[str, dict[str, float]]:
        """Tokens and list price per job: generation, grading, decomposition.

        Grading and decomposition went unrecorded until this was added, so a
        self-check profile reported the generator's tokens as the whole cost of
        a query. One grading call is larger than the generation it checks -- it
        reads the same passages plus the draft -- so the understatement was
        roughly a factor of two, in the direction that flatters."""
        totals: dict[str, dict[str, float]] = {}
        for entry in self.usage:
            bucket = totals.setdefault(
                entry.purpose, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
            )
            bucket["calls"] += 1
            bucket["input_tokens"] += entry.input_tokens
            bucket["output_tokens"] += entry.output_tokens
            bucket["usd"] += entry.list_price_usd
        return totals

    def record(self, name: str, duration_ms: float, *, depth: int = 0) -> None:
        self.stages.append(StageTiming(name=name, duration_ms=duration_ms, depth=depth))

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str = "generation",
    ) -> None:
        self.usage.append(
            Usage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                purpose=purpose,
            )
        )

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block and attach it to the trace.

        Records on the way out even if the block raises, so a failed query
        still shows where the time went -- which is exactly when that matters.
        """
        started = time.perf_counter()
        depth = self._depth
        self._depth = depth + 1
        try:
            yield
        finally:
            self._depth = depth
            self.record(name, (time.perf_counter() - started) * 1000, depth=depth)

    @contextmanager
    def measure_wall(self) -> Iterator[None]:
        """Time the whole query and record it as the authoritative total."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.wall_ms = (time.perf_counter() - started) * 1000

    def summary(self) -> str:
        """One-line human summary, for logs and the UI footer."""
        stages = " ".join(str(s) for s in self.stages if s.depth == 0)
        return (
            f"{self.total_ms:.0f}ms [{stages}] "
            f"{self.input_tokens}+{self.output_tokens} tok "
            f"(~${self.list_price_usd:.5f} at list price)"
        )


def record_gemini_usage(
    response: Any,
    trace: QueryTrace | None,
    model: str,
    purpose: str = "generation",
) -> None:
    """Attach token usage from a Gemini response to a trace, if there is one.

    One helper for every call site so that adding a model call cannot quietly
    add an unmeasured one: grading and decomposition were invisible in the cost
    figures for exactly as long as each had its own ad-hoc recording.
    """
    if trace is None:
        return
    usage = getattr(response, "usage_metadata", None)
    trace.add_usage(
        model=model,
        input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        purpose=purpose,
    )


@dataclass
class Timer:
    """Standalone stopwatch for code that has no trace to write to."""

    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000
