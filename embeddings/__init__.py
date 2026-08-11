"""
embeddings/
-----------
Phase 4: Embedding Generation

Exports:
    BGEEmbedder  — singleton wrapper around BAAI/bge-base-en-v1.5
    EmbedPipeline — reads chunks from SQLite, embeds, returns embedding payloads
"""

from embeddings.embedder import BGEEmbedder, get_embedder
from embeddings.embed_pipeline import EmbedPipeline, run_embedding

__all__ = ["BGEEmbedder", "get_embedder", "EmbedPipeline", "run_embedding"]
