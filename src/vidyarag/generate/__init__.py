"""Answer generation and citation handling."""

from vidyarag.generate.answer import GeneratedAnswer, generate_answer
from vidyarag.generate.citations import (
    Citation,
    extract_markers,
    format_context,
    render_references,
    resolve_citations,
    strip_invalid_markers,
)
from vidyarag.generate.prompts import (
    ANSWER_PROMPT_VERSION,
    NO_CONTEXT_ANSWER,
    Prompt,
    build_answer_prompt,
)

__all__ = [
    "ANSWER_PROMPT_VERSION",
    "NO_CONTEXT_ANSWER",
    "Citation",
    "GeneratedAnswer",
    "Prompt",
    "build_answer_prompt",
    "extract_markers",
    "format_context",
    "generate_answer",
    "render_references",
    "resolve_citations",
    "strip_invalid_markers",
]
