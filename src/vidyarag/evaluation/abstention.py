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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vidyarag.generate.prompts import NO_CONTEXT_ANSWER

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
class AbstentionStats:
    """How well refusal behaviour matches what the corpus can support."""

    unanswerable_total: int
    unanswerable_abstained: int
    answerable_total: int
    answerable_abstained: int

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
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_abstention_rate": self.false_abstention_rate,
        }


def is_structural_abstention(answer: str, *, trace_abstained: bool = False) -> bool:
    """Detect a refusal the pipeline signalled explicitly. Free and exact."""
    return trace_abstained or answer.strip() == NO_CONTEXT_ANSWER.strip()


async def judge_abstention(
    client: Any,
    *,
    model: str,
    question: str,
    answer: str,
) -> bool:
    """Ask a small model whether an answer refused.

    Falls back to ``False`` on any error. A failed classification should not be
    recorded as a refusal -- inventing abstentions would inflate the headline
    number this project is trying to establish honestly.
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
            max_tokens=5,
        )
    except Exception:  # noqa: BLE001 - classification is best-effort
        return False

    content = (response.choices[0].message.content or "").strip().upper()
    return content.startswith("REFUSED")


def summarise_abstention(
    records: list[tuple[bool, bool]],
) -> AbstentionStats:
    """Aggregate ``(is_answerable, abstained)`` pairs into stats."""
    unanswerable = [abstained for answerable, abstained in records if not answerable]
    answerable = [abstained for answerable, abstained in records if answerable]
    return AbstentionStats(
        unanswerable_total=len(unanswerable),
        unanswerable_abstained=sum(unanswerable),
        answerable_total=len(answerable),
        answerable_abstained=sum(answerable),
    )
