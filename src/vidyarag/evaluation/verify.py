"""Verify gold-set candidates against the corpus.

Two jobs, both aimed at making the human review pass short enough to actually
happen -- an unreviewed gold set is worth nothing, and a review that takes a
full day tends to be skipped or rushed.

**Proposing unanswerable questions.** These prove abstention works, so they are
the most valuable questions in the set and the easiest to get wrong. Asked
plainly for questions a corpus cannot answer, a model produces obviously
out-of-domain ones; refusing those is trivial and the resulting metric is
meaningless. So a candidate here must clear two independent checks:

1. **In domain**, verified by retrieval similarity against the real index. A
   question that retrieves nothing similar is off-topic and trivially refused.
2. **Genuinely absent**, verified by a grader reading the passages that were
   actually retrieved and judging whether they answer it.

The hard, useful case is precisely a question that scores *high* on the first
and fails the second: topically adjacent, plausibly in scope, and not in the
book. Neither check alone finds those.

This still produces *candidates*. A person approves them. But approving twelve
verified questions is a ten-minute job where authoring twelve was an afternoon.

**Triaging drafted answerable questions.** Checks one thing only: does the gold
chunk genuinely support the reference answer? That is a data-quality property.

It deliberately does **not** check whether retrieval finds the gold chunk. That
would be measuring the system with the instrument it is supposed to calibrate:
dropping questions the pipeline currently misses would leave a gold set the
baseline already succeeds on, and every later "improvement" would be measured
against a target quietly moved to meet it. A question whose gold chunk ranks
nowhere is a *result*, not a defect.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from vidyarag.evaluation.goldset import GoldQuestion
from vidyarag.llm.provider import embed_texts
from vidyarag.retrieve.dense import RetrievedChunk, retrieve_dense

# Free-tier pacing. Verification is only a couple of calls per candidate, so a
# fixed gap is enough; the evaluation runner's rate limiter is for the
# hundreds-of-calls case.
#
# Measured at a 4s gap: 13 of 30 drafting calls and most grader calls returned
# nothing, which is the free-tier quota being hit rather than the model
# declining to answer. Two calls per candidate at 7s is ~8.5 requests/minute,
# comfortably under a 15/minute allowance.
REQUEST_GAP_SECONDS = 7.0

RETRY_ATTEMPTS = 2
RETRY_WAIT_SECONDS = 45.0
"""Long enough to clear a per-minute window. A quota resets on a clock, so a
short retry just spends the second attempt inside the same exhausted minute."""

CONSECUTIVE_FAILURE_LIMIT = 3
"""Give up after this many proposals fail in a row.

A per-minute limit recovers within a retry; a *daily* one does not. Without this
the loop grinds through every remaining seed against a dead quota -- measured at
forty minutes to produce no candidates and no explanation. Three consecutive
failures already carry all the information the next forty will."""


class ProposalAborted(RuntimeError):
    """Raised when proposals fail repeatedly enough that continuing is pointless.

    Carries the underlying error so the caller can show *why* rather than
    reporting an empty result and leaving the cause to guesswork.
    """


DEFAULT_CONTEXT_K = 5
"""Passages shown to the grader. Matches the pipeline's context budget so a
question judged unanswerable here is unanswerable for the same reason there."""


class UnanswerableCandidate(BaseModel):
    """A proposed question the corpus is expected not to answer."""

    question: str = Field(description="The question, phrased as a student would ask it.")
    rationale: str = Field(description="Why an introductory textbook would not cover this.")
    topic: str = Field(description="The in-corpus topic it sits adjacent to.")


class SupportVerdict(BaseModel):
    """Whether a set of passages answers a question."""

    answerable: bool = Field(
        description="True only if the passages contain enough to answer the question."
    )
    reason: str = Field(description="One sentence justifying the verdict.")


@dataclass(slots=True)
class UnanswerableCheck:
    """The evidence behind accepting or rejecting one candidate."""

    question: str
    rationale: str
    topic: str
    top_score: float
    retrieved: list[str] = field(default_factory=list)
    in_domain: bool = False
    answerable: bool | None = None
    grader_reason: str = ""
    similarity_to_accepted: float = 0.0

    @property
    def duplicate(self) -> bool:
        return self.similarity_to_accepted >= DIVERSITY_THRESHOLD

    @property
    def accepted(self) -> bool:
        """In domain, unanswered by what that adjacency retrieves, and not a repeat."""
        return self.in_domain and self.answerable is False and not self.duplicate

    @property
    def verdict(self) -> str:
        # Order matters. An off-topic candidate is never graded, so it also has
        # `answerable is None` -- checking that first would report every
        # off-topic rejection as a grader failure and send someone debugging
        # the wrong thing.
        if not self.in_domain:
            return f"reject: off-topic (top score {self.top_score:.3f})"
        if self.duplicate:
            return f"reject: near-duplicate ({self.similarity_to_accepted:.3f})"
        if self.answerable is None:
            return "error: grader failed"
        if self.answerable:
            return "reject: corpus answers it"
        return "accept"


@dataclass(slots=True)
class TriageFinding:
    """The result of checking one drafted answerable question."""

    id: str
    question: str
    supported: bool | None
    reason: str = ""

    @property
    def needs_attention(self) -> bool:
        return self.supported is not True


QUESTION_SHAPES: tuple[str, ...] = (
    "a specific quantitative value the textbook never states -- a rate, a "
    "concentration, a threshold, a duration",
    "a named regulatory mechanism or signalling pathway more advanced than an "
    "introductory text covers",
    "a clinical management detail: a treatment protocol, a drug's action, a "
    "diagnostic criterion",
    "a named researcher, landmark experiment, or the history of how something " "was discovered",
    "a comparison with another species or system that the textbook does not draw",
    "a developmental detail: what happens at a particular stage, and when",
    "an evolutionary origin question -- where a structure or process came from",
    "the specific molecular basis of a named disease or disorder",
)
"""Rotated across candidates to force variety.

Without this the model converges hard on one template: an unguided run produced
twelve questions of which eleven contained the word "exact" and nine opened
"What is the exact...". A gold set shaped like that makes abstention trivially
gameable -- a system could pattern-match the phrasing and refuse, scoring
perfectly with no groundedness reasoning at all, which is the failure this whole
verification exists to prevent."""

DIVERSITY_THRESHOLD = 0.88
"""Reject a candidate this similar to one already accepted.

The same unguided run produced two near-identical questions about the
actin-myosin cross-bridge. Duplicates waste gold-set slots and quietly
over-weight whatever topic they land on."""

UNANSWERABLE_PROMPT = """\
You are helping build an evaluation set for a study assistant whose corpus is \
two OpenStax textbooks: *Biology* (1st ed) and *Anatomy and Physiology* (1st ed), \
both introductory undergraduate level.

Here is a passage from that corpus, to show you the topic and depth:

--- {citation} ---
{text}
---

Write ONE question that a student might plausibly ask about this topic, but which \
an INTRODUCTORY textbook would NOT answer.

For this one, ask about **{shape}**.

It must be:
- Clearly biology or human anatomy/physiology. Never another field, and never \
about laboratory instruments or imaging hardware.
- Phrased the way a curious student would actually say it, out loud.
- Genuinely beyond introductory scope.

IMPORTANT -- avoid these, they have been overused:
- The words "exact", "precise", or "atomic-level".
- Asking for a crystal structure, an amino-acid sequence, or atomic coordinates.
- Opening with "What is the exact...".

A question that telegraphs its own unanswerability through stiff phrasing is \
useless here: it could be refused on style alone, without reading the textbook. \
Make it sound answerable. The difficulty must be that the book does not cover it, \
not that the question reads like a trick.

Return the question, a one-sentence rationale for why an intro text omits it, and \
the in-corpus topic it sits next to.
"""

SUPPORT_PROMPT = """\
Below are passages retrieved from two introductory textbooks, followed by a question.

{passages}

Question: {question}

Do these passages contain enough information to answer that question?

Answer true ONLY if the passages genuinely contain the answer. Related or \
adjacent material is not enough -- a passage about the same topic that does not \
state the specific fact asked for should be judged false.
"""

REFERENCE_SUPPORT_PROMPT = """\
Below is a single passage from a textbook, followed by a question and a proposed \
reference answer.

--- {citation} ---
{text}
---

Question: {question}
Proposed answer: {reference}

Is the proposed answer supported by this passage?

Answer true only if the passage genuinely contains what the answer asserts. Judge \
support, not whether the answer is correct in general -- an answer that is true in \
the world but absent from this passage is NOT supported.
"""


def _structured(
    client: Any,
    model: str,
    prompt: str,
    schema: type[BaseModel],
    *,
    temperature: float = 0.0,
) -> tuple[Any | None, str]:
    """One structured-output call, retried past a rate-limit window.

    Returns ``(value, error)``. The error is handed back rather than swallowed
    because the alternative was measured and is genuinely bad: an earlier
    version returned a bare ``None``, so a run in which every call failed on an
    exhausted quota looked identical to one where the model declined every
    prompt. It reported "accepted 0 of 0" after forty minutes and explained
    nothing.

    A ``None`` value always means "unknown", never a negative verdict.
    """
    last_error = ""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )
        except Exception as exc:  # noqa: BLE001 - reported to the caller, never fatal
            last_error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:180]}"
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_WAIT_SECONDS)
                continue
            return None, last_error

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed, ""
        text = getattr(response, "text", None)
        if text:
            try:
                return schema.model_validate_json(text), ""
            except Exception as exc:  # noqa: BLE001
                last_error = f"unparseable response: {type(exc).__name__}"
        else:
            last_error = "empty response (possibly a safety block or truncation)"
        if attempt < RETRY_ATTEMPTS:
            time.sleep(RETRY_WAIT_SECONDS)
    return None, last_error


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"--- {c.citation} ---\n{c.text}" for c in chunks)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    return 0.0 if norm_left == 0 or norm_right == 0 else dot / (norm_left * norm_right)


def most_similar_accepted(
    question: str,
    accepted: list[str],
    embedding_model: str,
) -> float:
    """Highest similarity between ``question`` and anything already accepted.

    Returns 0.0 when nothing has been accepted yet.
    """
    if not accepted:
        return 0.0
    vectors = embed_texts([question, *accepted], embedding_model)
    return max(_cosine(vectors[0], other) for other in vectors[1:])


def in_domain_threshold(
    client: QdrantClient,
    questions: list[GoldQuestion],
    *,
    collection: str,
    embedding_model: str,
    percentile: float = 0.10,
) -> float:
    """Calibrate "in domain" from questions already known to be in domain.

    Picking a similarity cutoff by intuition would be arbitrary and unfalsifiable.
    Instead the drafted answerable questions -- which are in domain by
    construction, having been written *from* corpus passages -- supply the
    reference distribution, and the cutoff is a low percentile of their top-1
    scores.

    Args:
        client: Connected Qdrant client.
        questions: Answerable questions to calibrate against.
        collection: Collection to search.
        embedding_model: Must match the index.
        percentile: Fraction of known-in-domain questions allowed to fall below
            the cutoff. A low value keeps the bar permissive, which is correct
            here: the expensive mistake is rejecting a good hard candidate, not
            admitting one the grader will reject anyway.

    Returns:
        The similarity cutoff.
    """
    scores: list[float] = []
    for question in questions:
        hits = retrieve_dense(
            client,
            question.question,
            collection=collection,
            embedding_model=embedding_model,
            limit=1,
        )
        if hits:
            scores.append(hits[0].score)
    if not scores:
        raise ValueError("No retrieval scores available to calibrate a threshold")
    scores.sort()
    index = min(int(len(scores) * percentile), len(scores) - 1)
    return scores[index]


def check_unanswerable(
    llm: Any,
    client: QdrantClient,
    candidate: UnanswerableCandidate,
    *,
    grader_model: str,
    collection: str,
    embedding_model: str,
    threshold: float,
    context_k: int = DEFAULT_CONTEXT_K,
    already_accepted: list[str] | None = None,
) -> UnanswerableCheck:
    """Test one candidate against the corpus.

    Retrieves as the pipeline would, then asks a grader whether those passages
    answer the question. Both signals are recorded even when the first already
    decides the outcome, so a rejection can be read and argued with.
    """
    hits = retrieve_dense(
        client,
        candidate.question,
        collection=collection,
        embedding_model=embedding_model,
        limit=context_k,
    )
    check = UnanswerableCheck(
        question=candidate.question,
        rationale=candidate.rationale,
        topic=candidate.topic,
        top_score=hits[0].score if hits else 0.0,
        retrieved=[c.citation for c in hits],
    )
    check.in_domain = bool(hits) and check.top_score >= threshold
    check.similarity_to_accepted = most_similar_accepted(
        candidate.question, already_accepted or [], embedding_model
    )
    if not check.in_domain:
        check.answerable = None
        check.grader_reason = "not graded: failed the in-domain check"
        return check
    if check.duplicate:
        # No point spending a grader call on a question that cannot be accepted.
        check.answerable = None
        check.grader_reason = "not graded: near-duplicate of an accepted question"
        return check

    verdict, error = _structured(
        llm,
        grader_model,
        SUPPORT_PROMPT.format(passages=_format_passages(hits), question=candidate.question),
        SupportVerdict,
    )
    if verdict is None:
        check.answerable = None
        check.grader_reason = error or "grader call failed"
    else:
        check.answerable = verdict.answerable
        check.grader_reason = verdict.reason
    return check


def propose_unanswerable(
    llm: Any,
    client: QdrantClient,
    seeds: list[Any],
    *,
    draft_model: str,
    grader_model: str,
    collection: str,
    embedding_model: str,
    threshold: float,
    wanted: int,
    on_result: Any = None,
) -> list[UnanswerableCheck]:
    """Propose and verify candidates until ``wanted`` are accepted.

    Args:
        llm: Gemini client.
        client: Connected Qdrant client.
        seeds: Corpus chunks used as topic seeds, one per attempt.
        draft_model: Model that writes candidates.
        grader_model: Model that judges whether the corpus answers them.
        collection: Collection to search.
        embedding_model: Must match the index.
        threshold: In-domain similarity cutoff, from :func:`in_domain_threshold`.
        wanted: How many accepted candidates to stop at.
        on_result: Optional callback invoked with each :class:`UnanswerableCheck`.

    Returns:
        Every check performed, accepted or not. Rejections are returned rather
        than discarded so the acceptance rate is visible -- a run that accepted
        12 of 13 was not being selective enough to trust.
    """
    checks: list[UnanswerableCheck] = []
    accepted_questions: list[str] = []
    consecutive_failures = 0
    for index, seed in enumerate(seeds):
        if len(accepted_questions) >= wanted:
            break
        candidate, error = _structured(
            llm,
            draft_model,
            UNANSWERABLE_PROMPT.format(
                citation=seed.citation,
                text=seed.text,
                # Rotate the requested shape so the set does not collapse onto
                # one template. Indexed rather than random so a seed reproduces
                # the same run.
                shape=QUESTION_SHAPES[index % len(QUESTION_SHAPES)],
            ),
            UnanswerableCandidate,
            # Drafting wants variety across seeds; grading below stays at 0.0
            # because a verdict that changes between runs is not a verdict.
            temperature=0.9,
        )
        time.sleep(REQUEST_GAP_SECONDS)
        if candidate is None:
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                raise ProposalAborted(
                    f"{consecutive_failures} proposals failed in a row; stopping rather "
                    f"than working through {len(seeds)} seeds against an unavailable "
                    f"model. Last error: {error or 'unknown'}"
                )
            continue
        consecutive_failures = 0

        check = check_unanswerable(
            llm,
            client,
            candidate,
            grader_model=grader_model,
            collection=collection,
            embedding_model=embedding_model,
            threshold=threshold,
            context_k=DEFAULT_CONTEXT_K,
            already_accepted=accepted_questions,
        )
        time.sleep(REQUEST_GAP_SECONDS)
        checks.append(check)
        if on_result is not None:
            on_result(check)
        if check.accepted:
            accepted_questions.append(check.question)
    return checks


def triage_answerable(
    llm: Any,
    questions: list[GoldQuestion],
    chunk_text: dict[str, tuple[str, str]],
    *,
    grader_model: str,
    on_result: Any = None,
) -> list[TriageFinding]:
    """Check that each question's gold chunk supports its reference answer.

    Args:
        llm: Gemini client.
        questions: Answerable questions to check.
        chunk_text: ``chunk_id -> (citation, text)`` for the gold chunks.
        grader_model: Model that judges support.
        on_result: Optional callback invoked with each :class:`TriageFinding`.

    Returns:
        One finding per question, in input order.
    """
    findings: list[TriageFinding] = []
    for question in questions:
        reference = question.reference or ""
        missing = [cid for cid in question.gold_chunk_ids if cid not in chunk_text]
        if missing:
            finding = TriageFinding(
                id=question.id,
                question=question.question,
                supported=None,
                reason=f"gold chunk(s) not found in the index: {', '.join(missing)}",
            )
            findings.append(finding)
            if on_result is not None:
                on_result(finding)
            continue

        citation, text = chunk_text[question.gold_chunk_ids[0]]
        if len(question.gold_chunk_ids) > 1:
            # Multi-hop answers span passages, so judge them against the union;
            # checking only the first would flag correct questions as unsupported.
            joined = "\n\n".join(
                f"--- {chunk_text[cid][0]} ---\n{chunk_text[cid][1]}"
                for cid in question.gold_chunk_ids
            )
            citation, text = "combined passages", joined

        verdict, error = _structured(
            llm,
            grader_model,
            REFERENCE_SUPPORT_PROMPT.format(
                citation=citation,
                text=text,
                question=question.question,
                reference=reference,
            ),
            SupportVerdict,
        )
        time.sleep(REQUEST_GAP_SECONDS)
        finding = TriageFinding(
            id=question.id,
            question=question.question,
            supported=None if verdict is None else verdict.answerable,
            reason=(error or "grader call failed") if verdict is None else verdict.reason,
        )
        findings.append(finding)
        if on_result is not None:
            on_result(finding)
    return findings
