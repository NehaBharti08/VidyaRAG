"""Prompt templates, versioned.

Every template carries a version string that enters the query trace. A results
table is only meaningful if each row can be traced to the prompt that produced
it, and prompts change more often than code during tuning -- an unversioned
prompt makes an improvement impossible to attribute and impossible to repeat.

Changing a template's text means bumping its version. Editing in place silently
invalidates every number measured with the previous wording.
"""

from __future__ import annotations

from dataclasses import dataclass

from vidyarag.generate.citations import format_context
from vidyarag.retrieve.dense import RetrievedChunk


@dataclass(frozen=True, slots=True)
class Prompt:
    """A rendered prompt plus the template version that produced it."""

    system: str
    user: str
    version: str


ANSWER_V1_SYSTEM = """\
You are a study assistant that answers questions strictly from supplied \
textbook passages.

Rules:
1. Use ONLY the numbered passages provided. Do not add facts from memory, even \
if you are confident they are correct.
2. Cite every factual claim with the marker of the passage supporting it, like \
[1] or [3]. Cite the specific passage, not the whole context.
3. If the passages do not contain the answer, say exactly what is missing \
rather than guessing. A partial answer that names its gap is more useful than \
a complete-sounding one that invents the rest.
4. Never cite a number that was not provided to you.
5. Answer in clear prose for a student revising the topic. Be direct; do not \
restate the question or describe what you are about to do.\
"""

ANSWER_V1_USER = """\
Passages:

{context}

Question: {question}

Answer using only the passages above, citing each claim with its marker."""

ANSWER_PROMPT_VERSION = "answer-v1"


def build_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> Prompt:
    """Render the answering prompt for a question and its retrieved context.

    Args:
        question: The user's question.
        chunks: Retrieved context, in the order the markers will refer to.

    Returns:
        A :class:`Prompt` carrying its template version.
    """
    return Prompt(
        system=ANSWER_V1_SYSTEM,
        user=ANSWER_V1_USER.format(context=format_context(chunks), question=question),
        version=ANSWER_PROMPT_VERSION,
    )


NO_CONTEXT_ANSWER = (
    "I could not find anything about that in Biology or Anatomy and Physiology, "
    "so I have nothing to answer from. Try rephrasing, or ask about a topic these "
    "textbooks cover."
)
"""Returned when retrieval finds nothing. Not sent to the model at all --
asking a model to answer with no context is asking it to hallucinate."""
