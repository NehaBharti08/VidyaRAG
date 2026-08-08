"""Model access, behind one interface.

Nothing outside this package names a model or constructs a client. Swapping
providers is meant to be a change to :mod:`vidyarag.llm.provider` and nothing
else -- a property this project has already had to use once, when the corpus
work moved from OpenAI to local embeddings plus Gemini.
"""

from vidyarag.llm.provider import (
    count_tokens,
    embed_texts,
    get_embedder,
    get_gemini_client,
    get_tokenizer,
)

__all__ = [
    "count_tokens",
    "embed_texts",
    "get_embedder",
    "get_gemini_client",
    "get_tokenizer",
]
