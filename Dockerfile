# syntax=docker/dockerfile:1
# ── EduMind AI — Application Dockerfile ────────────────────────────────────────
# Base: python:3.11-slim
FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages required by python-docx, sentence-transformers, chromadb,
# and antiword (legacy .doc text extraction — python-docx only reads .docx)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    antiword \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── App layer ────────────────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

# Copy application source
COPY . .

# Create persistent directories
RUN mkdir -p data/staging vector_store/chroma_db

# Bake seed_documents/ into the image as an already-embedded corpus. This
# service (Render free plan) has no persistent disk, so a fresh container
# after every spin-down starts from this image layer, not from whatever
# was uploaded at runtime -- baking it in at build time is the only way
# the corpus survives that reset. Uses BuildKit secret mounts (Render
# injects env vars as plain build ARGs otherwise, which would leave the
# keys sitting in image layers -- see Render's own Docker-secrets docs)
# so GROQ_API_KEY/HF_API_TOKEN never land in the final image.
RUN --mount=type=secret,id=groq_api_key \
    --mount=type=secret,id=hf_api_token \
    GROQ_API_KEY="$(cat /run/secrets/groq_api_key)" \
    HF_API_TOKEN="$(cat /run/secrets/hf_api_token)" \
    LLM_BACKEND=groq \
    python scripts/bake_corpus.py

# Expose FastAPI port
EXPOSE 8000

# Render (and most PaaS Docker runtimes) inject PORT and expect the
# container to bind to it; shell form so ${PORT} actually expands. Falls
# back to 8000 for plain `docker run`/local use where PORT isn't set.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
