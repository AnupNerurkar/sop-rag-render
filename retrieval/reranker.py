"""
retrieval/reranker.py
----------------------
Reranker singleton. The actual reranking is a Groq listwise call
(retrieval/rerank_client.GroqListwiseReranker) -- this module is just the
process-level entry point that retriever.py imports.

Previously this wrapped a local BAAI/bge-reranker-base cross-encoder via
sentence-transformers, which is not installed on this box and never has
been. get_reranker() assigned the singleton *before* calling load(), so a
failed load (ImportError, every time) still cached a "reranker" whose
rerank() silently returned candidates unchanged -- and the caller in
retriever.py had no way to know that, so every response labeled its mode
"hybrid+rerank" for a rerank that had never actually run. See
retrieval/rerank_client.py's rerank() contract for how that's fixed now:
it returns a (results, reranked) pair, and the caller derives the mode
label from that boolean instead of assuming success.
"""

from __future__ import annotations

from typing import Optional

from retrieval.rerank_client import GroqListwiseReranker

_reranker_instance: Optional[GroqListwiseReranker] = None


def get_reranker() -> GroqListwiseReranker:
    """Returns the process-level reranker singleton.

    Unlike the old cross-encoder path, there is no model download or other
    load step that can fail at construction time -- failure is always
    per-request (a bad API call, an invalid response), which
    GroqListwiseReranker.rerank() already reports honestly via its
    `reranked` return value. There is nothing here to get wrong the way
    the old assign-before-load bug did.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = GroqListwiseReranker()
    return _reranker_instance
