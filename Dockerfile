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

# Expose FastAPI port
EXPOSE 8000

# Render (and most PaaS Docker runtimes) inject PORT and expect the
# container to bind to it; shell form so ${PORT} actually expands. Falls
# back to 8000 for plain `docker run`/local use where PORT isn't set.
CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
