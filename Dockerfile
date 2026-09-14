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

# The venv must be built at the path it runs from. Console scripts get an
# absolute shebang naming the venv's interpreter, so a venv built in /build and
# copied to /app left `uvicorn` pointing at /build/.venv/bin/python -- a path
# that does not exist at runtime. `exec` then fails with "no such file or
# directory" on a file that is plainly there, and CMD could never start the
# server. `python -c` does not go through a shebang, which is why importing
# the package inside the image kept passing. Building in /app makes it true.
WORKDIR /app

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
# at /app/src, which is not copied into the runtime stage, so the copied venv
# resolves to nothing and `import vidyarag` fails with ModuleNotFoundError.
# Installing a real wheel puts the package inside site-packages, where it
# travels with the venv.
RUN uv sync --frozen --no-dev --no-editable

# --- runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runs unprivileged. Nothing here needs root, and a container that does not
# need it should not have it.
RUN useradd --create-home --uid 1000 vidyarag

# VIDYARAG_CONFIG_DIR is required, not optional. The package locates config/
# relative to its own file, which is right in a checkout and on the Space --
# both run from src/ -- and wrong here, where --no-editable installs it into
# site-packages and "the repo root" becomes /app/.venv/lib/python3.11. The
# server then refused to start: "Unknown profile 'guarded'. Available: (none)".
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    QDRANT_MODE=embedded \
    QDRANT_PATH=/app/data/index \
    VIDYARAG_PROFILE=guarded \
    VIDYARAG_CONFIG_DIR=/app/config \
    FASTEMBED_CACHE_PATH=/home/vidyarag/.cache/fastembed

WORKDIR /app

COPY --from=builder --chown=vidyarag:vidyarag /app/.venv /app/.venv
# No src/ here: --no-editable put the package inside the venv, so shipping
# the sources again would only add a second, shadowing copy.
COPY --chown=vidyarag:vidyarag config/ /app/config/

# The index is NOT baked in. It is ~35 MB of derived data, rebuildable offline
# with `vidyarag ingest` and mounted at runtime, so the image stays a build
# artefact rather than a data artefact:
#   docker run -p 8000:8000 -v $(pwd)/data/index:/app/data/index vidyarag
#
# The mount must be writable. It was documented as `:ro`, which cannot work:
# Qdrant's embedded mode takes an exclusive lock by opening `<index>/.lock` with
# mode "r+", and that needs write access even when the file already exists, so a
# read-only mount fails at startup before serving a single request. The
# directory is created and handed to the runtime user so a named volume
# inherits ownership that user can actually write to.
RUN mkdir -p /app/data/index && chown -R vidyarag:vidyarag /app/data
VOLUME ["/app/data/index"]

USER vidyarag
EXPOSE 8000

# Probes the running server over HTTP. This used to run `vidyarag health`, which
# opens the embedded index itself -- but uvicorn takes that index's exclusive
# lock at startup, so the check failed with "already accessed by another
# instance of Qdrant client" and the container would have reported unhealthy
# for its entire life while serving correctly. Reproduced outside Docker: exit 1
# beside a server answering /v1/health with 3,608 points. Asking the server is
# also the truer check -- it proves the process serving traffic is up, not that
# a second process could start.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=8)"]

CMD ["uvicorn", "vidyarag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
