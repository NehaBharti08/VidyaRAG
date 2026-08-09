"""Answer generation.

Assembles retrieved context into a prompt, calls Gemini, validates the
citations that come back, and records the whole thing in the trace.

Two decisions worth stating:

* **Empty retrieval never reaches the model.** If nothing was retrieved there is
  nothing to ground an answer in, and asking a model to answer anyway is asking
  it to hallucinate. The system says it found nothing instead.
* **Citations are validated, not trusted.** Markers pointing outside the
  supplied context are stripped from the answer before a reader sees them. A
  fabricated citation is worse than no citation, because it is more convincing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vidyarag.generate.citations import Citation, resolve_citations, strip_invalid_markers
from vidyarag.generate.prompts import NO_CONTEXT_ANSWER, build_answer_prompt
from vidyarag.observe.trace import QueryTrace
from vidyarag.retrieve.dense import RetrievedChunk


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """A generated answer with its validated provenance."""

    text: str
    citations: list[Citation]
    grounded: bool
    """False when the model produced no resolvable citation at all."""

    @property
    def cited_chunk_ids(self) -> list[str]:
        return [c.chunk_id for c in self.citations]


def _extract_text(response: Any) -> str:
    """Pull the text out of a Gemini response, tolerating shape differences.

    The SDK's convenience accessor returns ``None`` when a response was cut
    short, so the parts are walked directly as a fallback rather than letting a
    truncated answer surface as the string "None".
    """
    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    parts: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            piece = getattr(part, "text", None)
            if piece:
                parts.append(str(piece))
    return "".join(parts).strip()


def _record_usage(response: Any, trace: QueryTrace, model: str) -> None:
    """Attach token usage from a Gemini response to the trace."""
    usage = getattr(response, "usage_metadata", None)
    trace.add_usage(
        model=model,
        input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
        output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
        purpose="generation",
    )


def generate_answer(
    client: Any,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    model: str,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
    trace: QueryTrace | None = None,
) -> GeneratedAnswer:
    """Answer a question from retrieved context.

    Args:
        client: A ``google.genai.Client``.
        question: The user's question.
        chunks: Retrieved context, in marker order.
        model: Pinned Gemini model id.
        temperature: 0 for reproducibility -- a benchmark that moves between
            runs cannot support a claim about a change.
        max_output_tokens: Cap on the generated answer.
        trace: Trace to record timing, tokens and citations into.

    Returns:
        A :class:`GeneratedAnswer` whose citations all resolve to real chunks.
    """
    trace = trace or QueryTrace(query=question)

    if not chunks:
        return GeneratedAnswer(text=NO_CONTEXT_ANSWER, citations=[], grounded=False)

    prompt = build_answer_prompt(question, chunks)
    trace.prompt_version = prompt.version

    with trace.stage("generate"):
        response = client.models.generate_content(
            model=model,
            contents=prompt.user,
            config={
                "system_instruction": prompt.system,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
        )

    _record_usage(response, trace, model)
    raw = _extract_text(response)

    if not raw:
        # An empty completion is a failure to answer, not an answer.
        return GeneratedAnswer(text=NO_CONTEXT_ANSWER, citations=[], grounded=False)

    citations = resolve_citations(raw, chunks)
    text = strip_invalid_markers(raw, chunks)
    trace.cited_chunk_ids = [c.chunk_id for c in citations]

    return GeneratedAnswer(text=text, citations=citations, grounded=bool(citations))
