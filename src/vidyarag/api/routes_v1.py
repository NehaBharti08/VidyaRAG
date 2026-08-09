"""v1 routes.

The pipeline is created once at startup and shared, not built per request.
Embedded Qdrant holds an exclusive lock on the index directory and the local
embedding model costs ~60 s to load, so per-request construction would be both
slow and, on the second concurrent request, broken.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from vidyarag.api.models import (
    CitationOut,
    ConfigResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedOut,
    StageOut,
    TraceOut,
)
from vidyarag.pipeline import Pipeline
from vidyarag.store.collection import count_points

router = APIRouter(prefix="/v1", tags=["v1"])


def get_pipeline(request: Request) -> Pipeline:
    """Resolve the shared pipeline from application state."""
    pipeline: Pipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:  # pragma: no cover - only if startup failed
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline is not initialised.",
        )
    return pipeline


PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, pipeline: PipelineDep) -> QueryResponse:
    """Answer a question from the indexed corpus."""
    try:
        result = pipeline.answer(payload.question)
    except ValueError as exc:
        # Raised when no Gemini key is configured. A 503 with the cause beats a
        # 500 that makes the caller guess.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    trace = result.trace
    return QueryResponse(
        question=result.question,
        answer=result.text,
        grounded=result.grounded,
        citations=[
            CitationOut(
                marker=c.marker,
                chunk_id=c.chunk_id,
                citation=c.citation,
                book_title=c.book_title,
                section=c.section,
                printed_page=c.printed_page,
                source_url=c.source_url,
                license=c.license_name,
            )
            for c in result.citations
        ],
        context=[
            RetrievedOut(
                chunk_id=c.chunk_id,
                score=c.score,
                citation=c.citation,
                text=c.text,
            )
            for c in result.retrieved
        ],
        trace=TraceOut(
            profile=trace.profile,
            prompt_version=trace.prompt_version,
            total_ms=round(trace.total_ms, 1),
            stages=[
                StageOut(name=s.name, duration_ms=round(s.duration_ms, 1)) for s in trace.stages
            ],
            input_tokens=trace.input_tokens,
            output_tokens=trace.output_tokens,
            list_price_usd=round(trace.list_price_usd, 6),
            retrieved=len(trace.retrieved_chunk_ids),
            cited=len(result.citations),
        ),
    )


@router.get("/health", response_model=HealthResponse)
def health(pipeline: PipelineDep) -> HealthResponse:
    """Report readiness and what is actually indexed.

    Returns the point count rather than a bare "ok": a service pointed at an
    empty collection is up and useless, and that should be visible here.
    """
    settings = pipeline.settings
    config = pipeline.config
    return HealthResponse(
        status="ok",
        collection=settings.qdrant_collection,
        points=count_points(pipeline.client, settings.qdrant_collection),
        embedding_model=config.embedding_model,
        generation_model=config.generation_model,
        profile=config.name,
    )


@router.get("/config", response_model=ConfigResponse)
def config(pipeline: PipelineDep) -> ConfigResponse:
    """Expose the active pipeline configuration."""
    cfg = pipeline.config
    return ConfigResponse(
        profile=cfg.name,
        description=cfg.description,
        generation_model=cfg.generation_model,
        grader_model=cfg.grader_model,
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        chunk_size=cfg.chunking.chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
        top_k_retrieve=cfg.retrieval.top_k_retrieve,
        top_k_context=cfg.retrieval.top_k_context,
        use_hybrid=cfg.retrieval.use_hybrid,
        use_reranker=cfg.retrieval.use_reranker,
        use_decomposition=cfg.retrieval.use_decomposition,
        corrective_enabled=cfg.corrective.enabled,
    )
