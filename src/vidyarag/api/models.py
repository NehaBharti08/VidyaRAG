"""Request and response schemas for the HTTP API.

The response carries the trace alongside the answer on purpose. A RAG service
that returns only prose asks to be trusted; returning the passages it used, the
citations it validated, and where the time and tokens went lets a caller check.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A question to answer."""

    question: str = Field(min_length=1, max_length=2000)
    book_slug: str | None = Field(
        default=None,
        description="Restrict retrieval to one book, e.g. 'biology'.",
    )


class CitationOut(BaseModel):
    """A validated reference back to a source passage."""

    marker: int
    chunk_id: str
    citation: str
    book_title: str
    section: str | None
    printed_page: str | None
    source_url: str
    license: str


class RetrievedOut(BaseModel):
    """A passage placed in the model's context."""

    chunk_id: str
    score: float
    citation: str
    text: str


class StageOut(BaseModel):
    name: str
    duration_ms: float


class TraceOut(BaseModel):
    """Per-query timing and token accounting."""

    profile: str
    prompt_version: str
    total_ms: float
    stages: list[StageOut]
    input_tokens: int
    output_tokens: int
    list_price_usd: float = Field(
        description=(
            "What this query would cost at published Gemini rates. Actual spend "
            "is zero on the free tier."
        )
    )
    retrieved: int
    cited: int


class QueryResponse(BaseModel):
    """An answer, its provenance, and how it was produced."""

    question: str
    answer: str
    grounded: bool = Field(
        description="False when no citation resolved -- treat the answer as unsupported."
    )
    citations: list[CitationOut]
    context: list[RetrievedOut]
    trace: TraceOut


class HealthResponse(BaseModel):
    status: str
    collection: str
    points: int
    embedding_model: str
    generation_model: str
    profile: str


class ConfigResponse(BaseModel):
    """The active pipeline configuration.

    Exposed so a result can always be traced to the configuration that produced
    it, without reading the server's environment.
    """

    profile: str
    description: str
    generation_model: str
    grader_model: str
    embedding_model: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    top_k_retrieve: int
    top_k_context: int
    use_hybrid: bool
    use_reranker: bool
    use_decomposition: bool
    corrective_enabled: bool
