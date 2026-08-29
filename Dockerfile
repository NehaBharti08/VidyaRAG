# Multi-stage build for the VidyaRAG API.
#
# The image deliberately excludes torch. Embeddings and reranking both run
# through fastembed on ONNX Runtime, which keeps this near 400 MB rather than
# the ~2.5 GB a torch-based stack would need -- the difference between an image
# that deploys on a free tier and one that does not.
#
# The evaluation dependency group is also excluded: ragas pulls langchain, about
# 200 MB of code that never executes while serving a query.

# --- builder ---------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv

WORKDIR /build

# Dependencies resolve from the lockfile before the source is copied, so a code
# change does not invalidate the dependency layer.
# LICENSE is required at build time, not just at runtime: pyproject declares
# license = { file = "LICENSE" } and hatchling reads it while building the
# project's own wheel. Omitting it fails the build with an error that never
# reproduces locally, where the file is always present.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
# --no-editable matters. The default editable install writes a path pointing
# at /build/src, which does not exist in the runtime stage, so the copied venv
# resolves to nothing and `import vidyarag` fails with ModuleNotFoundError.
# Installing a real wheel puts the package inside site-packages, where it
# travels with the venv.
RUN uv sync --frozen --no-dev --no-editable

# --- runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runs unprivileged. Nothing here needs root, and a container that does not
# need it should not have it.
RUN useradd --create-home --uid 1000 vidyarag

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QDRANT_MODE=embedded \
    QDRANT_PATH=/app/data/index \
    VIDYARAG_PROFILE=guarded \
    FASTEMBED_CACHE_PATH=/home/vidyarag/.cache/fastembed

WORKDIR /app

COPY --from=builder --chown=vidyarag:vidyarag /build/.venv /app/.venv
# No src/ here: --no-editable put the package inside the venv, so shipping
# the sources again would only add a second, shadowing copy.
COPY --chown=vidyarag:vidyarag config/ /app/config/

# The index is NOT baked in. It is ~35 MB of derived data, rebuildable offline
# with `vidyarag ingest` and mounted at runtime, so the image stays a build
# artefact rather than a data artefact:
#   docker run -v $(pwd)/data/index:/app/data/index:ro vidyarag
VOLUME ["/app/data/index"]

USER vidyarag
EXPOSE 8000

# Reuses the same health command CI and the CLI use, so a container that reports
# healthy has passed exactly the checks a developer would run by hand.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ["python", "-m", "vidyarag.cli", "health"]

CMD ["uvicorn", "vidyarag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
