"""The model swap point.

Embeddings and generation come from deliberately different places:

* **Embeddings run locally** through fastembed -- ONNX on CPU, no torch, no API
  key, no per-call cost. This is what lets the published demo answer a question
  without depending on any external service for retrieval, which matters more
  than it sounds: the whole deployment design exists so a portfolio link still
  works months after it was last touched.
* **Generation and grading go to Gemini.** These genuinely need a large model,
  and the free tier covers a project of this size.

The practical consequence is that **ingestion needs no credentials at all** --
the index can be built offline, and only answering a question requires a key.

Model names are never defaulted here. They come from ``PipelineConfig`` so that
a profile fully describes a run, and so a benchmark can never be attributed to
a model other than the one that produced it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from fastembed import TextEmbedding

if TYPE_CHECKING:  # pragma: no cover - import cost avoided at runtime
    from tokenizers import Tokenizer


@lru_cache(maxsize=4)
def get_embedder(model_name: str) -> TextEmbedding:
    """Load a local embedding model, cached per process.

    The first call downloads the ONNX weights (~210 MB for bge-base) and takes
    around a minute; every later call is free. Caching matters because building
    the model is far more expensive than embedding a batch with it.

    Args:
        model_name: A fastembed model id, e.g. ``"BAAI/bge-base-en-v1.5"``.

    Returns:
        A ready ``TextEmbedding``.
    """
    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> Tokenizer:
    """Return the embedding model's *own* tokeniser.

    This is not interchangeable with tiktoken. ``bge-base-en-v1.5`` uses BERT
    WordPiece and produces roughly 7% more tokens than ``cl100k_base`` on the
    same English prose, and it truncates hard at 512 tokens. Sizing chunks with
    the wrong tokeniser therefore silently drops the tail of every full-size
    chunk at embed time -- text that still appears in the answer and in the
    citation, but contributed nothing to the vector that retrieved it.

    Raises:
        RuntimeError: If fastembed exposes no tokeniser, since sizing chunks by
            guesswork would reintroduce exactly that failure.
    """
    embedder = get_embedder(model_name)
    for holder in (embedder, getattr(embedder, "model", None)):
        tokenizer = getattr(holder, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            return cast("Tokenizer", tokenizer)
    raise RuntimeError(
        f"fastembed exposes no tokenizer for {model_name!r}; chunk sizing cannot be verified"
    )


def count_tokens(text: str, model_name: str) -> int:
    """Count tokens as the embedding model itself will count them."""
    return len(get_tokenizer(model_name).encode(text).ids)


def embed_texts(
    texts: list[str],
    model_name: str,
    *,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of texts, preserving order.

    Args:
        texts: Texts to embed.
        model_name: A fastembed model id.
        batch_size: Documents per ONNX forward pass.

    Returns:
        One vector per input, in the same order.
    """
    if not texts:
        return []
    embedder = get_embedder(model_name)
    return [vector.tolist() for vector in embedder.embed(texts, batch_size=batch_size)]


REQUEST_TIMEOUT_MS = 90_000
"""Hard bound on one model call.

Generous enough for a long answer, short enough that a rate-limited call fails
visibly instead of hanging the run inside the SDK's own retry loop."""


def get_gemini_client(api_key: str) -> Any:
    """Construct a Gemini client.

    Imported lazily so that ingestion -- which never generates text -- does not
    pay for the SDK import or require a key to exist.

    Args:
        api_key: Gemini API key.

    Returns:
        A ``google.genai.Client``.

    Raises:
        ValueError: If no key was supplied, caught here rather than as an
            opaque 401 midway through an evaluation run.
    """
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey"
        )
    from google import genai
    from google.genai import types

    # The SDK also reads this from the environment; setting it explicitly keeps
    # a caller-supplied key authoritative over whatever the shell happens to have.
    os.environ.setdefault("GOOGLE_API_KEY", api_key)

    # An explicit timeout matters more than it looks. The SDK retries 429s
    # internally with backoff, so without a bound a single rate-limited call
    # blocks indefinitely -- an evaluation run was observed hung for minutes on
    # one question, producing no output and no error, which is far harder to
    # diagnose than a failure would have been.
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )
