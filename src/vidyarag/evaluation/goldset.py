"""The gold set: questions, ground truth, and provenance.

A gold set is only as good as the discipline behind it, so the schema records
*how* each question came to exist alongside the question itself. That field is
not decoration -- when a reviewer asks whether the evaluation is real, the
honest answer has to be in the data rather than in a claim about it.

Two properties matter enough to be enforced at load time rather than trusted:

* An **answerable** question must carry a reference answer and at least one
  gold chunk. Without them, retrieval metrics silently score against nothing.
* An **unanswerable** question must carry neither. A reference answer on a
  question the corpus cannot answer means it was mislabelled, and it would
  quietly convert the abstention measurement into noise.
"""

from __future__ import annotations

import enum
import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vidyarag.settings import REPO_ROOT

DEFAULT_GOLDSET = REPO_ROOT / "eval" / "goldset" / "goldset_v1.jsonl"


class QuestionType(enum.StrEnum):
    """What a question is designed to test."""

    FACTUAL = "factual"
    """Answerable from a single section. Baseline competence and regression canary."""

    MULTI_HOP = "multi_hop"
    """Requires joining two or more sections. What retrieval work must move."""

    UNANSWERABLE = "unanswerable"
    """In-domain and plausible, but genuinely absent from the corpus.

    These prove abstention. They must be written by hand: an LLM asked for
    unanswerable questions produces obviously out-of-domain ones, which makes
    refusing them trivial and the resulting metric meaningless.
    """

    AMBIGUOUS = "ambiguous"
    """Underspecified, or resting on a false presupposition. Tests graceful degradation."""


class Provenance(enum.StrEnum):
    """How a question was produced. Reported verbatim in docs/EVALUATION.md."""

    HUMAN_WRITTEN = "human_written"
    LLM_DRAFTED_HUMAN_VERIFIED = "llm_drafted_human_verified"


class GoldQuestion(BaseModel):
    """One evaluation question with its ground truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    type: QuestionType
    provenance: Provenance

    reference: str | None = None
    """The correct answer. ``None`` for unanswerable questions."""

    gold_chunk_ids: list[str] = Field(default_factory=list)
    """Chunks that genuinely support the answer. Empty for unanswerable questions."""

    gold_pages: list[str] = Field(default_factory=list)
    """Human-readable provenance, e.g. ``"Biology p.121"``. For manual checking."""

    books: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def is_answerable(self) -> bool:
        """Whether the corpus should be able to answer this at all.

        Ambiguous questions count as answerable: they are underspecified rather
        than absent, and the system is expected to engage with them.
        """
        return self.type is not QuestionType.UNANSWERABLE

    @model_validator(mode="after")
    def _check_ground_truth_matches_type(self) -> GoldQuestion:
        if self.type is QuestionType.UNANSWERABLE:
            if self.reference is not None:
                raise ValueError(
                    f"{self.id}: unanswerable questions must not carry a reference answer; "
                    "if one exists the question is mislabelled and abstention scoring breaks"
                )
            if self.gold_chunk_ids:
                raise ValueError(f"{self.id}: unanswerable questions must not cite gold chunks")
            return self

        if not self.reference:
            raise ValueError(f"{self.id}: answerable questions require a reference answer")
        if not self.gold_chunk_ids:
            raise ValueError(
                f"{self.id}: answerable questions require at least one gold chunk id, "
                "otherwise context recall and precision score against nothing"
            )
        if self.type is QuestionType.MULTI_HOP and len(self.gold_chunk_ids) < 2:
            raise ValueError(
                f"{self.id}: a multi-hop question needs at least two gold chunks; "
                "with one it is a factual lookup and would inflate the multi-hop score"
            )
        return self


def load_goldset(path: Path | None = None) -> list[GoldQuestion]:
    """Load and validate the gold set from JSONL.

    Args:
        path: JSONL file. Defaults to ``eval/goldset/goldset_v1.jsonl``.

    Returns:
        Validated questions in file order.

    Raises:
        FileNotFoundError: If the gold set is missing.
        ValueError: On a malformed record, a duplicate id, or ground truth that
            contradicts the question type.
    """
    target = path or DEFAULT_GOLDSET
    if not target.exists():
        raise FileNotFoundError(
            f"No gold set at {target}. Build one with `vidyarag goldset draft`."
        )

    questions: list[GoldQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        # Both comment styles are accepted: the drafting tool writes a `#`
        # header explaining what still needs doing, and hand-edited gold sets
        # tend to acquire `//` notes.
        if not stripped or stripped.startswith(("#", "//")):
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{number}: invalid JSON - {exc}") from exc

        question = GoldQuestion.model_validate(record)
        if question.id in seen:
            # Duplicate ids would double-weight a question and silently skew
            # every aggregate it appears in.
            raise ValueError(f"{target}:{number}: duplicate question id {question.id!r}")
        seen.add(question.id)
        questions.append(question)

    if not questions:
        raise ValueError(f"{target} contains no questions")
    return questions


def summarise_goldset(questions: list[GoldQuestion]) -> dict[str, int]:
    """Count questions by type, including types with no entries.

    Zeros are kept deliberately: a missing category is exactly what a reader
    needs to see, and omitting it hides an incomplete gold set.
    """
    counts = Counter(q.type.value for q in questions)
    return {t.value: counts.get(t.value, 0) for t in QuestionType}
