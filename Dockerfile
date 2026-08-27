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

# Bake the pre-embedded corpus into the image. This service (Render free
# plan) has no persistent disk, so a fresh container after every spin-down
# starts from this image layer, not from anything uploaded at runtime --
# the corpus has to be part of the image or the app comes up empty.
#
# seed_ledger.db already contains all 18 documents chunked AND embedded
# (built offline, see scripts/bake_corpus.py for how it was produced).
# Shipping the finished DB rather than embedding during `docker build`
# avoids needing HF/Groq credentials at build time and avoids ~600
# embedding API calls on every deploy -- the earlier build-time approach
# silently shipped 0 vectors whenever the build secret wasn't present,
# which is exactly the "no answers" failure that motivated this switch.
# ingestion_ledger.db holds only documents/chunks/embeddings; user
# accounts live in a separate institutional.db, so nothing here exposes
# credentials. source_file paths in the DB point at data/staging/<name>,
# so the seed files are copied there too for the document viewer.
COPY seed_ledger.db /app/ingestion_ledger.db
RUN cp seed_documents/* data/staging/

# Expose FastAPI port
EXPOSE 8000

# Render (and most PaaS Docker runtimes) inject PORT and expect the
# container to bind to it; shell form so ${PORT} actually expands. Falls
# back to 8000 for plain `docker run`/local use where PORT isn't set.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
