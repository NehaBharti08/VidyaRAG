"""FastAPI application factory.

The pipeline is built during startup and closed on shutdown. That ordering
matters for the embedded index: Qdrant holds an exclusive lock on the directory
while open, so a process that exits without closing leaves a stale lock that
blocks the next run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vidyarag.api.routes_v1 import router as v1_router
from vidyarag.observe.logging import configure_logging, get_logger
from vidyarag.pipeline import Pipeline, build_pipeline
from vidyarag.settings import Settings

DESCRIPTION = """\
Agentic, self-correcting RAG study assistant over open-license OpenStax
textbooks. Answers are grounded in retrieved passages and cite the page they
came from; the response includes the context used and a per-stage trace so a
caller can check rather than trust.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the pipeline once, tear it down cleanly."""
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)
    logger = get_logger("vidyarag.api")

    pipeline: Pipeline = build_pipeline(settings)
    app.state.pipeline = pipeline
    logger.info(
        "api.startup",
        profile=pipeline.config.name,
        collection=settings.qdrant_collection,
        embedding_model=pipeline.config.embedding_model,
        generation_model=pipeline.config.generation_model,
    )
    try:
        yield
    finally:
        pipeline.close()
        logger.info("api.shutdown")


def create_app() -> FastAPI:
    """Construct the application."""
    app = FastAPI(
        title="VidyaRAG",
        description=DESCRIPTION,
        version="0.3.0",
        lifespan=lifespan,
    )
    app.include_router(v1_router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": "vidyarag", "docs": "/docs", "query": "POST /v1/query"}

    return app


app = create_app()
