"""
backend/rag_integration.py
---------------------------
Thin adapter between FastAPI endpoints and the RAGPipeline.

Responsibility:
  - Convert RAGPipelineResponse → API-compatible dict
  - Map HTTP request fields (query, user_role) → RAGPipeline inputs
  - Expose streaming helper for SSE endpoints
  - Keep all AI logic inside rag_pipeline.py — this module only translates

Public API:
  query(question, role)             → dict  (non-streaming)
  stream_structured(question, role) → Iterator[tuple]  (streaming, with citations)
"""

from __future__ import annotations

import logging
import sys
import os

# Ensure project root is on sys.path when FastAPI loads from backend/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Iterator, Optional

logger = logging.getLogger(__name__)


def query(question: str, role: str) -> dict:
    """
    Run the full RAG pipeline and return an API-compatible response dict.

    Returns:
        {
            answer:           str,
            answer_with_refs: str,
            source_documents: list[str],   # display names for frontend
            confidence:       str,         # e.g. "87%"
            confidence_score: float,
            retrieval_mode:   str,
            processing_time_ms: float,
            retrieval_time_ms:  float,
            generation_time_ms: float,
            chunks_in_context: int,
            has_conflicts:    bool,
            model_name:       str,
        }
    """
    try:
        from rag_pipeline import get_pipeline
        pipeline = get_pipeline()
        resp = pipeline.run(question, role)

        source_documents = [c.display_name for c in resp.citations]

        citations_list = [
            {
                "rank":         c.rank,
                "doc_id":       c.doc_id,
                "display_name": c.display_name,
                "department":   c.department,
                "version":      c.version,
                "access_level": c.access_level,
                "score":        round(c.score, 4),
                "page_number":  c.page_number,
                "chunk_index":  c.chunk_index,
                "total_chunks": c.total_chunks,
                "source_file":  c.source_file,
            }
            for c in resp.citations
        ]

        return {
            "answer":             resp.answer,
            "answer_with_refs":   resp.answer_with_refs,
            "formatted_answer":   resp.formatted_answer,
            "source_documents":   source_documents,
            "citations":          citations_list,
            "citations_inferred": resp.citations_inferred,
            "confidence":         resp.confidence,
            "confidence_score":   round(resp.confidence_score, 4),
            "retrieval_mode":     resp.retrieval_mode,
            "rerank_method":      resp.rerank_method,
            "processing_time_ms": round(resp.processing_time_ms, 1),
            "retrieval_time_ms":  round(resp.retrieval_time_ms, 1),
            "generation_time_ms": round(resp.generation_time_ms, 1),
            "chunks_in_context":  resp.chunks_in_context,
            "has_conflicts":      resp.has_conflicts,
            "model_name":         resp.model_name,
            "template_used":      resp.template_used,
        }

    except Exception as exc:
        logger.error("[RAG_INTEGRATION] pipeline.run() failed: %s", exc, exc_info=True)
        return {
            "answer":             "The knowledge base is temporarily unavailable. Please try again shortly.",
            "answer_with_refs":   "The knowledge base is temporarily unavailable. Please try again shortly.",
            "formatted_answer":   "The knowledge base is temporarily unavailable. Please try again shortly.",
            "source_documents":   [],
            "citations":          [],
            "citations_inferred": False,
            "confidence":         "0%",
            "confidence_score":   0.0,
            "retrieval_mode":     "error",
            "rerank_method":      None,
            "processing_time_ms": 0.0,
            "retrieval_time_ms":  0.0,
            "generation_time_ms": 0.0,
            "chunks_in_context":  0,
            "has_conflicts":      False,
            "model_name":         "unavailable",
            "template_used":      "none",
        }


def stream_structured(question: str, role: str) -> Iterator[tuple]:
    """
    Live token stream that also surfaces a final metadata payload.

    Yields:
        ("token", str)  -- repeated, as tokens arrive
        ("meta", dict)  -- once, with answer / citations / confidence / timing

    Used by the authenticated SSE endpoint so it can paint tokens live and
    still save history + emit citations.
    """
    try:
        from rag_pipeline import get_pipeline
        yield from get_pipeline().run_stream_structured(question, role)
    except Exception as exc:
        logger.error("[RAG_INTEGRATION] run_stream_structured() failed: %s", exc, exc_info=True)
        error_text = "The knowledge base is temporarily unavailable. Please try again shortly."
        yield ("token", error_text)
        yield ("meta", {
            "answer":             error_text,
            "answer_with_refs":   error_text,
            "formatted_answer":   error_text,
            "source_documents":   [],
            "citations":          [],
            "citations_inferred": False,
            "confidence":         "0%",
            "confidence_score":   0.0,
            "retrieval_mode":     "error",
            "rerank_method":      None,
            "processing_time_ms": 0.0,
        })
