"""
scripts/bake_corpus.py
------------------------
Runs at Docker build time (see Dockerfile) to ingest every file under
seed_documents/ into the default ledger DB, so the resulting image
already contains an embedded corpus.

Only exists because this deployment (Render free plan) has no persistent
disk -- a fresh container starts from the image's writable-layer snapshot
on every spin-down, so the only way for the corpus to survive that is to
be part of the image itself rather than uploaded at runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.document_manager import ingest_uploaded_file

SEED_DIR = Path(__file__).resolve().parent.parent / "seed_documents"

if __name__ == "__main__":
    files = sorted(SEED_DIR.glob("*"))
    if not files:
        print(f"No files found in {SEED_DIR}", file=sys.stderr)
        sys.exit(1)

    for path in files:
        result = ingest_uploaded_file(path.name, path.read_bytes(), uploaded_by="build-time-seed")
        print(f"{result.status}: {path.name} ({result.chunks_created} chunks)")
        if result.status == "failed":
            print(f"  error: {result.error}", file=sys.stderr)
            sys.exit(1)
