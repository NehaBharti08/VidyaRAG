"""Did the system refuse, and should it have?

RAGAS has nothing to say about this. Faithfulness asks whether an answer is
supported by its context; it cannot express "there was no answer to give, and
saying so was correct". Yet refusing well is the capability this project is
built around, so it needs a measurement of its own.

Abstention is detected in two ways, cheapest first:

1. **Structural.** The pipeline emits a known sentinel when retrieval returns
   nothing, and the corrective loop (Phase 5) sets ``trace.abstained``. Both are
   free and exact.
2. **Judged.** Everything else is free text. The Phase 2 baseline has no
   abstention mechanism at all, so when it declines it does so in prose the
   prompt encouraged -- "the passages do not describe...". Matching that with
   keywords would be guesswork, so a small model classifies it.

Reporting precision alone would be easy to game: a system that refuses
everything scores 1.0. The false abstention rate is reported beside it for
exactly that reason -- together they describe a trade-off, separately they
flatter.

**The judge's output budget is load-bearing, which is not obvious.** It was
``max_tokens=5`` through every committed run. Measured against the live
endpoint, that returns ``'REF'`` with ``finish_reason='length'``: the label is
truncated mid-word, the prefix test fails, and every refusal is silently
recorded as an answer. Abstention recall read 0.000 for the three profiles that
had no abstention mechanism -- which looked like the expected result and was
therefore never questioned -- and the corrective loop appeared to lift it from
nothing. A truncated judge cannot be distinguished from a confident one by its
verdict alone, so this module no longer tries: an unparseable response is
reported as *unmeasured* rather than folded into "answered".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from vidyarag.generate.prompts import NO_CONTEXT_ANSWER

JUDGE_MAX_TOKENS = 64
"""Output budget for one label.

The label itself is two tokens. The margin is there because the failure mode of
too small a budget is silent and biased -- it produces a plausible verdict, not
an error -- and because a model that prefixes its label with a courtesy word
should cost a retry's worth of tokens, not a wrong number in a results table.
"""

_LABEL_RE = re.compile(r"\b(REFUSED|ANSWERED)\b")

ABSTENTION_JUDGE_PROMPT = """\
You are labelling one answer produced by a textbook question-answering system.

Decide whether the answer REFUSES to answer -- that is, whether it states the \
source material does not contain the information, rather than actually \
answering the question.

Label REFUSED if the answer:
- says the passages/textbook do not cover it, or
- says it cannot find or determine the answer from the material, or
- only describes what is missing without supplying the requested information.

Label ANSWERED if the answer:
- provides the requested information, even partially, even hedged, or
- answers and separately notes some detail is missing.

Reply with exactly one word: REFUSED or ANSWERED.

Question: {question}

Answer: {answer}"""


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One abstention judgement, and whether it can be believed.

    ``measured`` is the point of this type. A judge call that was truncated,
    refused or unparseable tells us nothing about the answer, and recording it
    as "did not refuse" is how a broken judge produces a clean-looking metric.
    """

    refused: bool = False
    measured: bool = False
    raw: str = ""
    error: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - guards accidental truthiness
        raise TypeError("use .refused; a verdict's truth value is ambiguous when unmeasured")


@dataclass(frozen=True, slots=True)
class AbstentionStats:
    """How well refusal behaviour matches what the corpus can support."""

    unanswerable_total: int
    unanswerable_abstained: int
    answerable_total: int
    answerable_abstained: int
    unmeasured: int = 0
    """Questions whose abstention verdict could not be established.

    Reported rather than absorbed: every rate below is computed over the
    questions that were judged, so a number derived from half a run should not
    look like one derived from all of it."""

    @property
    def recall(self) -> float | None:
        """Of the genuinely unanswerable questions, how many were refused."""
        if not self.unanswerable_total:
            return None
        return self.unanswerable_abstained / self.unanswerable_total

    @property
    def precision(self) -> float | None:
        """Of all refusals, how many were correct."""
        total = self.unanswerable_abstained + self.answerable_abstained
        if not total:
            return None
        return self.unanswerable_abstained / total

    @property
    def false_abstention_rate(self) -> float | None:
        """Answerable questions wrongly refused.

        The cost of over-abstaining. Precision alone hides it, and a system
        that refuses everything would otherwise look perfect.
        """
        if not self.answerable_total:
            return None
        return self.answerable_abstained / self.answerable_total

    @property
    def f1(self) -> float | None:
        """Harmonic mean of abstention precision and recall."""
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "unanswerable_total": self.unanswerable_total,
            "unanswerable_abstained": self.unanswerable_abstained,
            "answerable_total": self.answerable_total,
            "answerable_abstained": self.answerable_abstained,
            "unmeasured": self.unmeasured,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_abstention_rate": self.false_abstention_rate,
        }


def is_structural_abstention(answer: str, *, trace_abstained: bool = False) -> bool:
    """Detect a refusal the pipeline signalled explicitly. Free and exact.

    Both sentinels count. ``NO_CONTEXT_ANSWER`` is emitted when retrieval
    returned nothing; ``ABSTENTION_TEXT`` is what the corrective loop returns
    when it declines. Only the first was recognised here, so re-reading a
    committed run file -- where the loop's flag lives in the trace rather than
    in the text -- would send a known, fixed refusal to a model to be
    classified, paying for a verdict the repository already knows.
    """
    from vidyarag.correct.loop import ABSTENTION_TEXT

    stripped = answer.strip()
    return (
        trace_abstained
        or stripped == NO_CONTEXT_ANSWER.strip()
        or stripped == ABSTENTION_TEXT.strip()
    )


def parse_judge_label(content: str | None, *, finish_reason: str | None = None) -> JudgeVerdict:
    """Turn a judge response into a verdict, or say it could not be read.

    Kept separate from the call so the parsing can be tested without a network,
    which is what was missing when the truncation shipped.

    The label is matched as a whole word anywhere in the response, and the last
    match wins: a model that reasons before answering ends on its conclusion,
    and a model that restates the options ("not ANSWERED, so REFUSED") means the
    last one. A truncated response is rejected outright -- ``'REF'`` is not a
    label, and treating it as one is precisely the bug this replaces.
    """
    raw = (content or "").strip()
    if finish_reason == "length":
        return JudgeVerdict(raw=raw, error="truncated: raise JUDGE_MAX_TOKENS")
    if not raw:
        return JudgeVerdict(raw=raw, error="empty judge response")
    matches = _LABEL_RE.findall(raw.upper())
    if not matches:
        return JudgeVerdict(raw=raw, error=f"no label in response: {raw[:60]!r}")
    return JudgeVerdict(refused=matches[-1] == "REFUSED", measured=True, raw=raw)


async def judge_abstention(
    client: Any,
    *,
    model: str,
    question: str,
    answer: str,
) -> JudgeVerdict:
    """Ask a small model whether an answer refused.

    Returns an unmeasured verdict on any failure. It deliberately does not fall
    back to "answered": that is what the previous version did, and a judge whose
    failures all land on one label does not produce a noisy metric, it produces
    a wrong one that looks plausible.
    """
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": ABSTENTION_JUDGE_PROMPT.format(question=question, answer=answer),
                }
            ],
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never fatal
        return JudgeVerdict(error=f"{type(exc).__name__}: {exc}"[:200])

    choice = response.choices[0]
    return parse_judge_label(
        choice.message.content, finish_reason=getattr(choice, "finish_reason", None)
    )


def summarise_abstention(
    records: list[tuple[bool, bool]],
    *,
    unmeasured: int = 0,
) -> AbstentionStats:
    """Aggregate ``(is_answerable, abstained)`` pairs into stats."""
    unanswerable = [abstained for answerable, abstained in records if not answerable]
    answerable = [abstained for answerable, abstained in records if answerable]
    return AbstentionStats(
        unanswerable_total=len(unanswerable),
        unanswerable_abstained=sum(unanswerable),
        answerable_total=len(answerable),
        answerable_abstained=sum(answerable),
        unmeasured=unmeasured,
    )
