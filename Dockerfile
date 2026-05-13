# ---------------------------------------------------------------------------
# Stage 1 — builder: install Python dependencies into an isolated prefix.
# Build tools stay here and never reach the runtime image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — runtime: lean image with only what the app needs at run time.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# curl is needed for the HEALTHCHECK directive below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed packages from the builder stage.
COPY --from=builder /install /usr/local

# Run as a non-root user — no home directory, no login shell.
RUN useradd --no-create-home --shell /bin/false appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# Application source and pre-built artifacts.
#
# chroma_db/ contains the pre-built vector index — the app will fail to serve
# queries at startup if this directory is absent.
# Build prerequisite: run `python -m app.retrieval.ingest` locally first so
# that chroma_db/ exists and is fully populated before `docker build`.
# ---------------------------------------------------------------------------
COPY app/                    ./app/
COPY data/catalogs/          ./data/catalogs/
COPY data/user_profiles.json ./data/user_profiles.json
COPY chroma_db/              ./chroma_db/

# Transfer ownership to the non-root user.
RUN chown -R appuser:appuser /app

USER appuser

# ---------------------------------------------------------------------------
# Environment defaults.
# Secrets (UNIFIED_ENDPOINT_KEY, UNIFIED_ENDPOINT_BASE_URL_ANTHROPIC) are
# intentionally absent — they are injected at runtime via Cloud Run Secret
# Manager bindings or `docker run -e`.
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2 \
    VECTOR_STORE=local \
    CHROMA_PERSIST_DIR=/app/chroma_db \
    DEFAULT_LLM_PROVIDER=claude

EXPOSE 8080

# Cloud Run probes /health after the container reports RUNNING.
# The HEALTHCHECK is also useful for local docker-compose verification.
# start-period=60s allows the embedding model to load on the first request.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f "http://localhost:${PORT:-8080}/health" || exit 1

# Cloud Run injects $PORT (usually 8080). sh -c is required so the shell
# expands ${PORT:-8080} — exec-form CMD would treat it as a literal string.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
