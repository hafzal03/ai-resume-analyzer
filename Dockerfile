# Multi-stage build: the runtime image carries the application and its serving
# dependencies, but not the compiler toolchain, the training extras (pandas,
# matplotlib) or the corpus.
#
# Note what is *not* here: no model artifact is baked in. The model is produced
# by training and mounted at runtime, so an image can never ship a stale model
# that silently disagrees with the dataset in the repository.

# ---------------------------------------------------------------------------
# Stage 1: build a virtual environment
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependency layer first so source edits do not invalidate the install cache.
COPY pyproject.toml README.md ./
COPY src/resume_classifier/__init__.py src/resume_classifier/__init__.py
RUN pip install --upgrade pip setuptools wheel && pip install .

COPY src/ src/
RUN pip install --no-deps .

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Run as an unprivileged user. Root in a container is still root on a kernel.
RUN groupadd --system --gid 1001 appuser \
 && useradd --system --uid 1001 --gid appuser --create-home appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser data/taxonomy/ /app/data/taxonomy/

# Mount points for artifacts produced outside the image.
RUN mkdir -p /app/models /app/data/processed /app/data/raw \
 && chown -R appuser:appuser /app

USER appuser

ENV RC_ENVIRONMENT=production \
    RC_DEBUG=false \
    RC_HOST=0.0.0.0 \
    RC_PORT=5000 \
    RC_DATA_DIR=/app/data \
    RC_MODELS_DIR=/app/models

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/v1/health', timeout=4).status==200 else 1)"

CMD ["rc-serve"]
