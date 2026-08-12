#!/usr/bin/env python3
"""
verify_rag.py
--------------
Phase 9: System Verification Script.

Checks every component of the RAG pipeline independently, then runs
four end-to-end sample queries and prints detailed results.

READ-ONLY: Never modifies SQLite, the vector store, or any document.

Run: .venv/Scripts/python.exe verify_rag.py
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from colorama import Fore, Style, init as _cinit
    _cinit(autoreset=True)
except ImportError:
    class _Stub(str):
        def __getattr__(self, _): return ""
    Fore = Style = _Stub()

W = 60

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    print()
    print(Fore.CYAN + Style.BRIGHT + "=" * W)
    print(Fore.CYAN + Style.BRIGHT + title.center(W))
    print(Fore.CYAN + Style.BRIGHT + "=" * W)

def _section(title: str) -> None:
    print()
    print(Fore.MAGENTA + Style.BRIGHT + f"  {title}")
    print(Fore.MAGENTA + "  " + "-" * (W - 2))

def _check(label: str, passed: bool, detail: str = "") -> None:
    icon  = Fore.GREEN + "[PASS]" if passed else Fore.RED + "[FAIL]"
    extra = f"  {Fore.WHITE + Style.DIM}{detail}" if detail else ""
    print(f"  {icon}  {Fore.WHITE}{label}{extra}{Style.RESET_ALL}")

def _info(label: str, value) -> None:
    dots = "." * max(1, 34 - len(label))
    print(f"      {Fore.WHITE}{label}{dots} {Fore.WHITE + Style.BRIGHT}{value}{Style.RESET_ALL}")


# ---------------------------------------------------------------------------
# Component checks
# ---------------------------------------------------------------------------

_RESULTS: dict[str, bool] = {}


def _run_check(name: str, fn) -> bool:
    try:
        fn()
        _check(name, True)
        _RESULTS[name] = True
        return True
    except Exception as exc:
        _check(name, False, str(exc)[:80])
        _RESULTS[name] = False
        return False


def check_generation_backend() -> None:
    """
    Check 1/2: the LLM backend that LLM_BACKEND actually selects generates
    text. Previously two separate checks hardcoded Ollama regardless of
    which backend was configured -- on a box running LLM_BACKEND=groq (this
    one), that meant two permanently-failing checks for a backend that was
    never supposed to be active, while the backend actually in use went
    unverified. See rag/rag_engine.py's get_rag_engine() for the same
    backend-selection logic this mirrors.
    """
    backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
    _section(f"Check 1/2 -- Generation Backend (LLM_BACKEND={backend})")
    try:
        from rag.rag_engine import get_rag_engine
        engine = get_rag_engine()
        result = engine.client.chat(
            [{"role": "user", "content": "Reply with the single word: OK"}],
            temperature=0.0,
            max_tokens=5,
        )
        answer = result.get("answer", "").strip()
        model = getattr(engine.client, "model", backend)
        _check(f"{backend} generates text", True, f"model={model} reply={answer!r}")
        _RESULTS["Generation backend"] = True
    except Exception as exc:
        _check(f"{backend} generates text", False, str(exc)[:80])
        _RESULTS["Generation backend"] = False


def check_embedding_model() -> None:
    """Check 3: BGE embedding model loads and embeds a query."""
    _section("Check 3 -- BGE Embedding Model")
    try:
        from embeddings.embedder import MODEL_NAME, EMBEDDING_DIM, get_embedder
        t0      = time.perf_counter()
        emb     = get_embedder()
        elapsed = time.perf_counter() - t0
        vec     = emb.embed_query("test query")
        assert len(vec) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM} dims, got {len(vec)}"
        _check("Embedding model loaded", True, f"load={elapsed:.1f}s  dim={len(vec)}")
        _RESULTS["Embedding model"] = True
    except Exception as exc:
        _check("Embedding model loaded", False, str(exc)[:80])
        _RESULTS["Embedding model"] = False


def check_reranker() -> None:
    """Check 4: reranker actually reorders results (a live Groq call, not just construction).

    There is no local model to "load" any more -- the old is_loaded check
    would have passed unconditionally the moment the class was instantiated,
    which is exactly the kind of check that let the previous reranker report
    success for a rerank that had never run. Only a real rerank() call that
    returns reranked=True is evidence this actually works.
    """
    _section("Check 4 -- Reranker (Groq listwise)")
    try:
        from retrieval.hybrid_search import RawSearchResult
        from retrieval.reranker import get_reranker
        candidates = [
            RawSearchResult(chunk_id=f"c{i}", content=f"sample passage {i} about admissions", score=1.0 - i * 0.1, distance=None)
            for i in range(5)
        ]
        t0 = time.perf_counter()
        _, reranked = get_reranker().rerank("admissions process", candidates, top_k=5)
        elapsed = time.perf_counter() - t0
        _check("Reranker call succeeded", reranked, f"reranked={reranked}  latency={elapsed:.1f}s")
        _RESULTS["Reranker"] = reranked
    except Exception as exc:
        _check("Reranker call succeeded", False, str(exc)[:80])
        _RESULTS["Reranker"] = False


def check_sqlite() -> None:
    """Check 5: SQLite ingestion ledger accessible."""
    _section("Check 5 -- SQLite Ledger")
    try:
        import ledger
        docs   = ledger.get_all_documents()
        chunks = ledger.get_chunk_count()
        emb    = ledger.get_embedded_chunk_count()
        _check("SQLite ledger readable", True,
               f"docs={len(docs)}  chunks={chunks}  embedded={emb}")
        _info("Document count",  len(docs))
        _info("Chunk count",     chunks)
        _info("Embedded chunks", emb)
        _info("DB path",         ledger.DB_PATH)
        _RESULTS["SQLite"] = True
    except Exception as exc:
        _check("SQLite ledger readable", False, str(exc)[:80])
        _RESULTS["SQLite"] = False


def check_vector_store() -> None:
    """Check 6: vector collection accessible and non-empty."""
    _section("Check 6 -- the vector store")
    try:
        from vector_store.sqlite_store import get_vector_store, COLLECTION_NAME
        store = get_vector_store()
        stats = store.get_collection_stats()
        count = stats["vector_count"]
        assert count > 0, f"Collection '{COLLECTION_NAME}' is empty"
        _check("vector collection non-empty", True,
               f"collection={COLLECTION_NAME}  vectors={count}")
        _info("Collection",    COLLECTION_NAME)
        _info("Vector count",  count)
        _info("DB path",       stats["db_path"])
        _RESULTS["the vector store"] = True
    except Exception as exc:
        _check("vector collection non-empty", False, str(exc)[:80])
        _RESULTS["the vector store"] = False


def check_retrieval() -> None:
    """Check 7: Dense retrieval returns results."""
    _section("Check 7 -- Dense Retrieval")
    try:
        from retrieval.retriever import get_retriever
        t0   = time.perf_counter()
        resp = get_retriever().retrieve_by_text(
            "attendance requirement",
            role="Student",
            use_bm25=False,
            use_reranker=False,
            top_k_final=3,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        assert resp.total_results > 0, "Dense retrieval returned 0 results"
        _check("Dense retrieval returns results", True,
               f"results={resp.total_results}  latency={elapsed:.0f}ms")
        for r in resp.results:
            _info(f"  [{r.rank}] {r.citation.display_name[:30]}", f"score={r.score:.4f}")
        _RESULTS["Dense retrieval"] = True
    except Exception as exc:
        _check("Dense retrieval returns results", False, str(exc)[:80])
        _RESULTS["Dense retrieval"] = False


def check_hybrid_reranking() -> None:
    """Check 8: Full hybrid+reranker pipeline returns results."""
    _section("Check 8 -- Hybrid Retrieval + Reranking")
    try:
        from retrieval.retriever import get_retriever
        t0   = time.perf_counter()
        resp = get_retriever().retrieve_by_text(
            "attendance requirement",
            role="Student",
            use_bm25=True,
            use_reranker=True,
            top_k_dense=10,
            top_k_bm25=10,
            top_k_fusion=10,
            top_k_final=3,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        assert resp.total_results > 0, "Hybrid+reranker returned 0 results"
        has_rerank = any(r.rerank_score is not None for r in resp.results)
        _check("Hybrid + reranker returns results", True,
               f"results={resp.total_results}  reranked={has_rerank}  latency={elapsed:.0f}ms")
        for r in resp.results:
            rerank = f"{r.rerank_score:.4f}" if r.rerank_score else "N/A"
            _info(f"  [{r.rank}] {r.citation.display_name[:28]}",
                  f"score={r.score:.4f}  rerank={rerank}")
        _RESULTS["Hybrid + reranking"] = True
    except Exception as exc:
        _check("Hybrid + reranker returns results", False, str(exc)[:80])
        _RESULTS["Hybrid + reranking"] = False


def check_prompt_builder() -> None:
    """Check 9: Prompt builder assembles a valid prompt from live results."""
    _section("Check 9 -- Prompt Builder")
    try:
        from retrieval.retriever import get_retriever
        from rag.prompt_builder  import build_prompt
        from rag.prompt_schema   import PromptConfig

        results = get_retriever().retrieve_by_text(
            "attendance requirement", role="Student",
            use_reranker=False, top_k_final=3,
        ).results
        assert results, "No retrieval results -- cannot test prompt builder"

        bp = build_prompt("What is the attendance requirement?", results)
        assert bp.chunks_included > 0, "PromptBuilder included 0 chunks"
        assert "[SOURCE 1]" in bp.context_block, "No [SOURCE 1] in context block"
        assert bp.messages[0]["role"] == "system", "First message is not 'system'"

        _check("Prompt builder assembles prompt", True,
               f"chunks={bp.chunks_included}  chars={bp.context_chars}")
        _info("Template",         bp.template_used.value)
        _info("Chunks included",  bp.chunks_included)
        _info("Context chars",    bp.context_chars)
        _info("Conflicts",        len(bp.conflicts))
        _RESULTS["Prompt builder"] = True
    except Exception as exc:
        _check("Prompt builder assembles prompt", False, str(exc)[:80])
        _RESULTS["Prompt builder"] = False


def check_citation_engine() -> None:
    """Check 10: Citation engine deduplicates and ranks live results."""
    _section("Check 10 -- Citation Engine")
    try:
        from retrieval.retriever   import get_retriever
        from rag.citation_engine   import get_citation_engine

        results = get_retriever().retrieve_by_text(
            "attendance requirement", role="Student",
            use_reranker=False, top_k_final=5,
        ).results
        assert results, "No retrieval results -- cannot test citation engine"

        PLACEHOLDER = "Students must attend classes [SOURCE 1] [SOURCE 2]."
        cl = get_citation_engine().build(results, PLACEHOLDER)

        assert cl.total_citations > 0, "Citation engine returned 0 citations"
        assert cl.answer_with_refs != "", "answer_with_refs is empty"

        _check("Citation engine produces citations", True,
               f"citations={cl.total_citations}  conflicts={cl.has_version_conflicts}")
        for c in cl.citations:
            _info(f"  {c.inline_ref} {c.display_name[:28]}",
                  f"v{c.version}  score={c.score:.4f}")
        _RESULTS["Citation engine"] = True
    except Exception as exc:
        _check("Citation engine produces citations", False, str(exc)[:80])
        _RESULTS["Citation engine"] = False


# ---------------------------------------------------------------------------
# End-to-end sample queries
# ---------------------------------------------------------------------------

_SAMPLE_QUERIES = [
    ("What is the attendance requirement for semester examinations?", "Student"),
    ("Summarize the examination moderation process.",                 "Faculty"),
    ("Compare fee policy versions.",                                  "Admin"),
    ("Show latest examination circular.",                            "Student"),
]


def run_e2e_queries() -> None:
    _section("Check 11 -- End-to-End RAG Pipeline")

    if not _RESULTS.get("Generation backend"):
        _check("End-to-end RAG", False, "skipped -- generation backend not available")
        _RESULTS["End-to-end RAG"] = False
        return

    try:
        from rag_pipeline import get_pipeline, reset_pipeline, PipelineConfig
        reset_pipeline()
        pipeline = get_pipeline(PipelineConfig(
            max_tokens           = 512,
            temperature          = 0.3,
            use_reranker         = _RESULTS.get("Reranker", False),
        ))
    except Exception as exc:
        _check("End-to-end RAG", False, f"Pipeline init failed: {exc}")
        _RESULTS["End-to-end RAG"] = False
        return

    all_passed = True

    for i, (query, role) in enumerate(_SAMPLE_QUERIES, 1):
        print()
        print(Fore.YELLOW + Style.BRIGHT + f"  Query {i}: {query}" + Style.RESET_ALL)
        print(f"  Role   : {role}")
        t0 = time.perf_counter()
        try:
            resp    = pipeline.run(query, role=role)
            elapsed = (time.perf_counter() - t0) * 1000

            # ---- Metrics ---------------------------------------------------
            _info("Mode",            resp.retrieval_mode)
            _info("Chunks retrieved", resp.retrieved_chunks)
            _info("Chunks in ctx",   resp.chunks_in_context)
            _info("Top score",       f"{resp.confidence_score:.4f}")
            _info("Confidence",      resp.confidence)
            _info("Tokens",          resp.total_tokens)
            _info("Retrieval ms",    f"{resp.retrieval_time_ms:.0f}")
            _info("Generation ms",   f"{resp.generation_time_ms:.0f}")
            _info("Total ms",        f"{resp.processing_time_ms:.0f}")

            # ---- Answer preview -------------------------------------------
            preview = resp.answer[:300].replace("\n", " ").strip()
            print(f"      {Fore.WHITE + Style.DIM}Answer: {preview}...{Style.RESET_ALL}")

            # ---- Citations ------------------------------------------------
            print(f"      {Fore.CYAN}Citations:{Style.RESET_ALL}")
            for c in resp.citations:
                rerank = f"  rerank={c.rerank_score:.4f}" if c.rerank_score else ""
                print(f"        {Fore.YELLOW}{c.inline_ref}{Fore.WHITE} "
                      f"{c.display_name[:35]}  "
                      f"v{c.version}{Fore.WHITE + Style.DIM}{rerank}"
                      f"{Style.RESET_ALL}")

            passed = len(resp.answer) > 0 and not resp.answer.startswith("Error")
            _check(f"Query {i} completed", passed)
            if not passed:
                all_passed = False

        except Exception as exc:
            _check(f"Query {i} completed", False, str(exc)[:80])
            all_passed = False

    _RESULTS["End-to-end RAG"] = all_passed


# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

def print_summary() -> int:
    print()
    print(Fore.CYAN + Style.BRIGHT + "=" * W)
    print(Fore.CYAN + Style.BRIGHT + "VERIFICATION SUMMARY".center(W))
    print(Fore.CYAN + Style.BRIGHT + "=" * W)
    print()

    passed = sum(1 for v in _RESULTS.values() if v)
    total  = len(_RESULTS)

    for name, ok in _RESULTS.items():
        icon = Fore.GREEN + "[PASS]" if ok else Fore.RED + "[FAIL]"
        print(f"  {icon}  {Fore.WHITE}{name}{Style.RESET_ALL}")

    print()
    ratio = f"{passed}/{total}"
    clr   = Fore.GREEN if passed == total else (Fore.YELLOW if passed >= total // 2 else Fore.RED)
    print(Fore.WHITE + f"  Checks passed: {clr + Style.BRIGHT}{ratio}{Style.RESET_ALL}")

    print()
    if passed == total:
        print(Fore.GREEN + Style.BRIGHT + "  ALL CHECKS PASSED".center(W))
        print(Fore.GREEN + "  System is ready for FastAPI integration.".center(W))
    elif _RESULTS.get("Generation backend") is False:
        backend = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
        print(Fore.YELLOW + Style.BRIGHT + "  Core pipeline READY".center(W))
        print(Fore.YELLOW + f"  Fix the {backend} backend and run again for end-to-end verification.".center(W))
    else:
        print(Fore.RED + Style.BRIGHT + f"  {total - passed} check(s) failed.".center(W))
        print(Fore.RED + "  Review errors above before proceeding.".center(W))

    print(Fore.CYAN + Style.BRIGHT + "=" * W)
    print()
    return 0 if passed == total else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    _header("VIT AGENTIC AI -- RAG PIPELINE VERIFICATION")

    check_generation_backend()
    check_embedding_model()
    check_reranker()
    check_sqlite()
    check_vector_store()
    check_retrieval()
    check_hybrid_reranking()
    check_prompt_builder()
    check_citation_engine()
    run_e2e_queries()

    return print_summary()


if __name__ == "__main__":
    sys.exit(main())
