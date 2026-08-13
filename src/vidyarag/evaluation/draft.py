"""Draft gold-set candidates from the indexed corpus.

This tool produces *candidates*, never a finished gold set. Every draft it
writes is marked ``needs_review`` and must be checked by a human before it can
be used, and the review step is where most of the value is.

What is drafted, and what deliberately is not:

* **Factual and multi-hop questions are drafted here.** They are grounded in
  sampled passages, so a model can write a reasonable question and the human
  pass is a check rather than an authoring job.
* **Unanswerable questions are not.** Asked for questions a corpus cannot
  answer, a model reliably produces obviously out-of-domain ones -- "what is
  the capital of France" against a biology textbook. Refusing those is trivial,
  so the abstention metric they produce would be meaningless. They must be
  in-domain and plausible, which takes a person who knows the corpus. A stub
  file is emitted for them instead.

The parametric-knowledge check is the other reason this is not just prompting.
Each draft is also asked whether it could be answered from general knowledge
without the passage. Those are dropped: a question a model can answer from
memory measures nothing about retrieval, and a gold set full of them would show
a healthy score for a system whose retrieval was broken.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

MIN_CHUNK_TOKENS = 120
"""Skip very short chunks. Headers and stubs make poor question sources."""

# Reference material is kept in the index on purpose -- a glossary definition is
# a legitimate thing to retrieve -- but it makes a poor question *source*. A
# question drafted from a glossary entry tends to be a definition lookup whose
# answer appears verbatim in dozens of places, which measures string matching
# rather than comprehension.
EXCLUDED_SECTIONS = ("glossary", "index", "answer key", "solutions", "preface", "references")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A corpus chunk, as far as drafting is concerned."""

    chunk_id: str
    text: str
    citation: str
    book_slug: str
    book_title: str
    section: str | None
    printed_page: str | None
    token_count: int


class DraftedQuestion(BaseModel):
    """What the model returns for one draft."""

    question: str
    reference: str
    answerable_from_general_knowledge: bool = Field(
        description="True if a well-read person could answer without the passage."
    )
    uses_both_passages: bool = Field(
        default=True, description="Multi-hop only: whether both passages are required."
    )


FACTUAL_PROMPT = """\
You are helping build an evaluation set for a textbook question-answering system.

Below is one passage from an OpenStax biology textbook.

Write ONE exam-style question that:
- is answerable ENTIRELY and ONLY from this passage,
- a student revising this topic would plausibly ask,
- is specific rather than broad ("how does X do Y", not "explain X"),
- does not refer to "the passage" or "the text" -- it should read as a natural question.

Also write the correct answer, in one to three sentences, using only the passage.

Then judge honestly: could a well-read person answer your question correctly \
WITHOUT this passage, from general knowledge? Be strict. If the fact is common \
textbook knowledge, say true.

Passage ({citation}):
{text}"""

MULTIHOP_PROMPT = """\
You are helping build an evaluation set for a textbook question-answering system.

Below are TWO passages from OpenStax biology textbooks, from different sections.

Write ONE question that genuinely REQUIRES BOTH passages to answer.

It must be a CHAINED question: a fact in one passage is needed to reach the \
answer that the other passage supplies. Answering step two should depend on \
having done step one.

Do NOT write a compound question -- two separate questions joined by "and", of \
the form "what is X, and what is Y". Those are two lookups in one sentence, not \
multi-hop reasoning, and they are not acceptable. A good multi-hop question \
usually has a single question mark and a single thing being asked.

Also write the correct answer in two to four sentences, using only these passages.

Then judge honestly:
- could a well-read person answer without the passages, from general knowledge?
- does answering really require BOTH passages, not just one? Say false if \
either passage alone is enough, or if you wrote a compound question.

Passage A ({citation_a}):
{text_a}

Passage B ({citation_b}):
{text_b}"""


UNANSWERABLE_STUB_HEADER = """\
# Unanswerable questions -- WRITE THESE BY HAND
#
# These are the questions that prove abstention works, and they are the reason
# the evaluation is worth anything. They must be:
#
#   1. IN DOMAIN. Biology or human anatomy/physiology. A question about French
#      history is refused trivially and measures nothing.
#   2. PLAUSIBLE. It should sound like something these textbooks *might* cover.
#   3. GENUINELY ABSENT. Verify by searching the corpus:
#          uv run vidyarag ask "<your question>"
#      and confirming the retrieved passages do not contain the answer.
#
# Good shapes:
#   - a named mechanism or pathway more advanced than an intro text covers
#   - a specific numeric value the book states qualitatively instead
#   - recent research a first-edition textbook predates
#   - a clinical detail an introductory course omits
#
# Fill in `question` and `notes` (say WHY it is absent). Leave `reference` null
# and `gold_chunk_ids` empty -- the loader rejects the file otherwise.
"""


def scroll_chunks(
    client: QdrantClient,
    collection: str,
    *,
    limit: int | None = None,
) -> list[ChunkRecord]:
    """Read chunk payloads out of the collection.

    Args:
        client: Connected Qdrant client.
        collection: Collection to read.
        limit: Stop after this many chunks. ``None`` reads all of them.

    Returns:
        Chunks in scroll order.
    """
    records: list[ChunkRecord] = []
    offset: Any = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in batch:
            payload = record.payload or {}
            records.append(
                ChunkRecord(
                    chunk_id=str(payload.get("chunk_id", "")),
                    text=str(payload.get("text", "")),
                    citation=str(payload.get("citation", "")),
                    book_slug=str(payload.get("book_slug", "")),
                    book_title=str(payload.get("book_title", "")),
                    section=payload.get("section"),
                    printed_page=payload.get("printed_page"),
                    token_count=int(payload.get("token_count", 0) or 0),
                )
            )
            if limit is not None and len(records) >= limit:
                return records
        if offset is None:
            return records


def _is_reference_section(section: str) -> bool:
    """Whether a section is reference material rather than explanation."""
    lowered = section.lower()
    return any(marker in lowered for marker in EXCLUDED_SECTIONS)


def find_related_chunk(
    client: QdrantClient,
    collection: str,
    *,
    seed: ChunkRecord,
    embedding_model: str,
    pool: int = 12,
) -> ChunkRecord | None:
    """Find a topically related chunk from a *different* section.

    Multi-hop questions were originally drafted from two randomly paired
    chunks, and the results were unusable: asked to connect skeletal muscle
    tone to plant water potential, the model produced a question joining them
    on "both rely on continuous processes". That is not a multi-hop question,
    it is a non sequitur with a question mark, and a gold set full of them
    would measure nothing except the model's willingness to comply.

    Genuine multi-hop needs passages that really connect. Searching the index
    with the seed passage as the query finds its semantic neighbours; taking
    the best one from a different section gives a pair that shares a topic but
    not a location -- membrane structure and membrane transport, rather than
    muscles and plants.

    Returns:
        The closest chunk from another section, or ``None`` if there is none.
    """
    from vidyarag.retrieve.dense import retrieve_dense

    # The seed's own text is the query; its opening is the most topical part
    # and keeps the query inside the embedding model's window.
    hits = retrieve_dense(
        client,
        " ".join(seed.text.split()[:120]),
        collection=collection,
        embedding_model=embedding_model,
        limit=pool,
    )
    for hit in hits:
        if hit.chunk_id == seed.chunk_id or not hit.section:
            continue
        if hit.section == seed.section or _is_reference_section(hit.section):
            continue
        return ChunkRecord(
            chunk_id=hit.chunk_id,
            text=hit.text,
            citation=hit.citation,
            book_slug=hit.book_slug,
            book_title=hit.book_title,
            section=hit.section,
            printed_page=hit.printed_page,
            token_count=0,
        )
    return None


def sample_chunks(
    chunks: list[ChunkRecord],
    count: int,
    *,
    seed: int = 20260813,
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[ChunkRecord]:
    """Pick a spread of chunks to draft from.

    Sampling is stratified by book and seeded. Stratifying stops a gold set
    from over-representing whichever book happens to have more chunks; seeding
    means the same corpus yields the same candidates, so a regenerated gold set
    is a reviewable diff rather than an entirely new set of questions.
    """
    usable = [
        c
        for c in chunks
        if c.token_count >= min_tokens and c.section and not _is_reference_section(c.section)
    ]
    if not usable:
        return []

    by_book: dict[str, list[ChunkRecord]] = {}
    for chunk in usable:
        by_book.setdefault(chunk.book_slug, []).append(chunk)

    rng = random.Random(seed)
    spread = {book: _spread_by_section(items, rng) for book, items in by_book.items()}

    # Round-robin across books so neither title dominates, regardless of which
    # one has more chunks.
    picked: list[ChunkRecord] = []
    books = sorted(spread)
    depth = 0
    while len(picked) < count:
        added = False
        for book in books:
            if depth < len(spread[book]):
                picked.append(spread[book][depth])
                added = True
                if len(picked) >= count:
                    break
        if not added:
            break  # corpus exhausted
        depth += 1

    return picked


def _spread_by_section(chunks: list[ChunkRecord], rng: random.Random) -> list[ChunkRecord]:
    """Order chunks so every section is used once before any is used twice.

    A plain shuffle clusters picks in whichever chapter happens to be longest.
    Taking one chunk per section in rounds spreads questions across the whole
    book first, then goes deeper only when a larger sample is asked for -- so
    the sampler can still satisfy a big request without abandoning the spread.
    """
    by_section: dict[str, list[ChunkRecord]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk.section or "", []).append(chunk)
    for items in by_section.values():
        items.sort(key=lambda c: c.chunk_id)
        rng.shuffle(items)

    ordered: list[ChunkRecord] = []
    for depth in range(max(len(v) for v in by_section.values())):
        for section in sorted(by_section):
            if depth < len(by_section[section]):
                ordered.append(by_section[section][depth])
    return ordered


def _generate(client: Any, model: str, prompt: str) -> DraftedQuestion | None:
    """Ask the model for one draft, as structured JSON."""
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": 0.7,
                "response_mime_type": "application/json",
                "response_schema": DraftedQuestion,
            },
        )
    except Exception:  # noqa: BLE001 - a failed draft is skipped, never fatal
        return None

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, DraftedQuestion):
        return parsed
    text = getattr(response, "text", None)
    if not text:
        return None
    try:
        return DraftedQuestion.model_validate_json(text)
    except Exception:  # noqa: BLE001
        return None


def draft_factual(client: Any, model: str, chunk: ChunkRecord) -> dict[str, Any] | None:
    """Draft one single-passage question. Returns ``None`` if it should be dropped."""
    drafted = _generate(
        client, model, FACTUAL_PROMPT.format(citation=chunk.citation, text=chunk.text)
    )
    if drafted is None or drafted.answerable_from_general_knowledge:
        return None
    return {
        "question": drafted.question,
        "type": "factual",
        "provenance": "llm_drafted_human_verified",
        "reference": drafted.reference,
        "gold_chunk_ids": [chunk.chunk_id],
        "gold_pages": [chunk.citation],
        "books": [chunk.book_slug],
        "notes": "NEEDS REVIEW: verify the passage answers this and that retrieval is required.",
    }


def draft_multihop(
    client: Any, model: str, first: ChunkRecord, second: ChunkRecord
) -> dict[str, Any] | None:
    """Draft one two-passage question. Returns ``None`` if it should be dropped."""
    drafted = _generate(
        client,
        model,
        MULTIHOP_PROMPT.format(
            citation_a=first.citation,
            text_a=first.text,
            citation_b=second.citation,
            text_b=second.text,
        ),
    )
    if drafted is None or drafted.answerable_from_general_knowledge:
        return None
    if not drafted.uses_both_passages:
        # A "multi-hop" question answerable from one passage is a factual
        # lookup wearing a costume, and would inflate the multi-hop score.
        return None
    return {
        "question": drafted.question,
        "type": "multi_hop",
        "provenance": "llm_drafted_human_verified",
        "reference": drafted.reference,
        "gold_chunk_ids": [first.chunk_id, second.chunk_id],
        "gold_pages": [first.citation, second.citation],
        "books": sorted({first.book_slug, second.book_slug}),
        "notes": "NEEDS REVIEW: verify BOTH passages are genuinely required.",
    }


def unanswerable_stubs(count: int) -> list[dict[str, Any]]:
    """Blank entries for the human-written unanswerable questions."""
    return [
        {
            "id": f"unans-{index:03d}",
            "question": "TODO: write an in-domain, plausible, genuinely absent question",
            "type": "unanswerable",
            "provenance": "human_written",
            "reference": None,
            "gold_chunk_ids": [],
            "gold_pages": [],
            "books": [],
            "notes": "TODO: say why the corpus cannot answer this.",
        }
        for index in range(1, count + 1)
    ]


def assign_ids(drafts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Give each draft a stable id derived from its type and position."""
    prefixes = {"factual": "fact", "multi_hop": "multi", "ambiguous": "ambig"}
    counters: dict[str, int] = {}
    for draft in drafts:
        kind = str(draft["type"])
        prefix = prefixes.get(kind, kind[:5])
        counters[prefix] = counters.get(prefix, 0) + 1
        draft["id"] = f"{prefix}-{counters[prefix]:03d}"
    return drafts


def write_jsonl(records: list[dict[str, Any]], path: Path, header: str = "") -> Path:
    """Write records as JSONL, with an optional comment header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header] if header else []
    lines.extend(json.dumps(record, ensure_ascii=False) for record in records)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_review_sheet(records: list[dict[str, Any]], path: Path) -> Path:
    """Write a markdown checklist for the human verification pass.

    Reviewing raw JSONL is unpleasant enough that it invites skimming, and a
    skimmed gold set silently undermines every number measured against it.
    """
    lines = [
        "# Gold set review",
        "",
        "Check each question against its cited passage. Tick only what you have",
        "actually verified. Delete any question that fails a check rather than",
        "trying to repair it -- a doubtful question is worse than a missing one.",
        "",
        "For each: **(a)** does the passage really answer it, **(b)** does it read",
        "naturally, **(c)** does answering genuinely require retrieval?",
        "",
    ]
    for record in records:
        lines.append(f"### `{record['id']}` · {record['type']}")
        lines.append("")
        lines.append(f"**Q.** {record['question']}")
        lines.append("")
        lines.append(f"**Reference.** {record['reference']}")
        lines.append("")
        lines.append(f"**Source.** {'; '.join(record['gold_pages'])}")
        lines.append("")
        lines.append("- [ ] passage answers it  - [ ] reads naturally  - [ ] needs retrieval")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
