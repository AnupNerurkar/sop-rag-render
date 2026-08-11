#!/usr/bin/env python3
"""
demo.py
--------
VIT Agentic AI Assistant -- Project Status Demonstration

READ-ONLY: Never writes to SQLite, the vector store, or any document.
Every value is read from existing modules; no logic is duplicated.

Run: .venv/Scripts/python.exe demo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# -- ensure project root is importable ----------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# -- colorama -----------------------------------------------------------------
try:
    from colorama import Fore, Style, init as _cinit
    _cinit(autoreset=True)
except ImportError:
    class _Stub(str):          # fallback: colors are empty strings
        def __getattr__(self, _): return ""
    Fore = Style = _Stub()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

W  = 60   # section separator width

def _banner(text: str) -> None:
    print()
    print(Fore.CYAN + Style.BRIGHT + "=" * W)
    for line in text.strip().splitlines():
        print(Fore.CYAN + Style.BRIGHT + line.center(W))
    print(Fore.CYAN + Style.BRIGHT + "=" * W)

def _section(n: int, title: str) -> None:
    print()
    print(Fore.MAGENTA + Style.BRIGHT + f"  SECTION {n}  --  {title}")
    print(Fore.MAGENTA + "  " + "-" * (W - 2))

def _row(label: str, value, *, color=None, width: int = 36) -> None:
    dots = "." * max(1, width - len(label))
    clr  = color or (Fore.WHITE + Style.BRIGHT)
    print(f"  {Fore.WHITE}{label}{Fore.RESET}{dots} {clr}{value}{Style.RESET_ALL}")

def _check(label: str, passed: bool, detail: str = "") -> None:
    icon  = Fore.GREEN + "  [OK]" if passed else Fore.RED + "  [X]"
    state = Fore.GREEN + "PASS" if passed else Fore.RED + "FAIL"
    extra = f"  {Fore.YELLOW}{detail}" if detail else ""
    print(f"{icon}  {Fore.WHITE}{label:<36}{state}{extra}{Style.RESET_ALL}")

def _phase(label: str, done: bool, note: str = "") -> None:
    state = Fore.GREEN + "COMPLETE" if done else Fore.YELLOW + note
    dots  = "." * max(1, 46 - len(label))
    print(f"  {Fore.WHITE}{label}{dots} {state}{Style.RESET_ALL}")

def _fmt_bytes(n: int) -> str:
    if n >= 1_048_576: return f"{n / 1_048_576:.1f} MB"
    if n >= 1_024:     return f"{n / 1_024:.1f} KB"
    return f"{n} B"

def _dir_size(path: str) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


# -----------------------------------------------------------------------------
# Section 1 -- Repository Statistics
# -----------------------------------------------------------------------------

def section_1() -> dict:
    _section(1, "Repository Statistics")

    try:
        import ledger
        from chunk_validation import (
            _load_all_chunks,
            _load_document_doc_ids,
            validate_chunks,
        )

        docs   = ledger.get_all_documents()
        chunks = _load_all_chunks()
        known  = _load_document_doc_ids()
        vr     = validate_chunks(chunks, known, sample_size=0)

        total_docs     = len(docs)
        ingested       = sum(1 for d in docs if d.get("status") in
                             ("ingested", "chunked", "embedded", "assessed"))
        total_chunks   = vr.total_chunks
        avg_size       = vr.avg_chunk_size
        largest_chars  = vr.largest_chunk["char_count"]  if vr.largest_chunk  else 0
        smallest_chars = vr.smallest_chunk["char_count"] if vr.smallest_chunk else 0

        from collections import Counter
        hash_counts = Counter(d.get("sha256_hash") for d in docs if d.get("sha256_hash"))
        duplicates  = sum(1 for c in hash_counts.values() if c > 1)

        corrupted  = sum(1 for d in docs if d.get("status") == "corrupted")
        superseded = sum(1 for d in docs if d.get("status") == "superseded")

        depts = {d.get("department") for d in docs if d.get("department")}
        cats  = {d.get("category")   for d in docs if d.get("category")}

        _row("Total documents discovered",  total_docs)
        _row("Total documents ingested",    ingested)
        _row("Total chunks generated",      total_chunks)
        _row("Average chunk size (chars)",  f"{avg_size:.0f}")
        _row("Largest chunk (chars)",       largest_chars)
        _row("Smallest chunk (chars)",      smallest_chars)
        _row("Duplicate files",             duplicates,
             color=Fore.YELLOW if duplicates else Fore.GREEN)
        _row("Corrupted files",             corrupted,
             color=Fore.RED if corrupted else Fore.GREEN)
        _row("Superseded versions",         superseded,
             color=Fore.YELLOW if superseded else Fore.GREEN)
        _row("Departments",                 len(depts))
        _row("Categories",                  len(cats))

        return {
            "total_docs": total_docs, "ingested": ingested,
            "total_chunks": total_chunks, "avg_size": avg_size,
            "depts": len(depts), "cats": len(cats),
        }

    except Exception as exc:
        print(f"  {Fore.RED}ERROR: {exc}{Style.RESET_ALL}")
        return {}


# -----------------------------------------------------------------------------
# Section 2 -- Chunk Validation
# -----------------------------------------------------------------------------

def section_2() -> None:
    _section(2, "Chunk Validation")

    try:
        from chunk_validation import (
            _load_all_chunks,
            _load_document_doc_ids,
            validate_chunks,
        )
        from collections import defaultdict

        chunks = _load_all_chunks()
        known  = _load_document_doc_ids()
        vr     = validate_chunks(chunks, known, sample_size=0)

        _check("Empty chunks",
               len(vr.empty_chunks) == 0,
               f"({len(vr.empty_chunks)} found)" if vr.empty_chunks else "")

        _check("Duplicate chunk IDs",
               len(vr.duplicate_chunk_ids) == 0,
               f"({len(vr.duplicate_chunk_ids)} found)" if vr.duplicate_chunk_ids else "")

        _check("Missing metadata",
               len(vr.missing_metadata_chunks) == 0,
               f"({len(vr.missing_metadata_chunks)} chunks)" if vr.missing_metadata_chunks else "")

        _check("Invalid access levels",
               len(vr.access_level_issues) == 0,
               f"({len(vr.access_level_issues)} issues)" if vr.access_level_issues else "")

        by_doc: dict = defaultdict(list)
        for c in chunks:
            by_doc[c.get("doc_id", "")].append(c.get("chunk_index", -1))
        order_ok = all(
            sorted(idxs) == list(range(len(idxs)))
            for idxs in by_doc.values() if idxs
        )
        _check("Chunk ordering",  order_ok)

        _check("Parent document consistency",
               len(vr.doc_id_issues) == 0,
               f"({len(vr.doc_id_issues)} issues)" if vr.doc_id_issues else "")

    except Exception as exc:
        print(f"  {Fore.RED}ERROR: {exc}{Style.RESET_ALL}")


# -----------------------------------------------------------------------------
# Section 3 -- Embedding Status
# -----------------------------------------------------------------------------

def section_3() -> dict:
    _section(3, "Embedding Status")

    try:
        from embeddings.embedder import (
            MODEL_NAME, EMBEDDING_DIM, get_embedder,
        )
        import ledger

        embedded = ledger.get_embedded_chunk_count()
        pending  = len(ledger.get_chunks_pending_embedding())

        print(f"  {Fore.CYAN}Loading embedding model (may take ~30 s on CPU)...{Style.RESET_ALL}")
        t0       = time.perf_counter()
        embedder = get_embedder()
        elapsed  = time.perf_counter() - t0

        try:
            device = str(embedder._model.device)
        except AttributeError:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        _row("Embedding model",     MODEL_NAME)
        _row("Device",              device.upper(),
             color=Fore.GREEN if "cuda" in device.lower() else Fore.YELLOW)
        _row("Embedding dimension", EMBEDDING_DIM)
        _row("Model load time",     f"{elapsed:.1f} s")
        _row("Embedded chunks",     embedded, color=Fore.GREEN)
        _row("Pending chunks",      pending,
             color=Fore.GREEN if pending == 0 else Fore.YELLOW)

        query = "What is attendance requirement?"
        print()
        print(f"  {Fore.CYAN}Sample query: \"{query}\"{Style.RESET_ALL}")
        vec = embedder.embed_query(query)
        print(f"  {Fore.WHITE}Embedding dimension : {Style.BRIGHT}{len(vec)}{Style.RESET_ALL}")
        vals = "  ".join(f"{v:+.4f}" for v in vec[:8])
        print(f"  {Fore.WHITE}First 8 values      : {Style.BRIGHT}{vals}{Style.RESET_ALL}")

        return {"embedded": embedded, "pending": pending, "model": MODEL_NAME}

    except Exception as exc:
        print(f"  {Fore.RED}ERROR loading embedding model: {exc}{Style.RESET_ALL}")
        return {}


# -----------------------------------------------------------------------------
# Section 4 -- Vector Store Status
# -----------------------------------------------------------------------------

def section_4() -> dict:
    _section(4, "Vector Store Status")

    try:
        from vector_store.sqlite_store import get_vector_store, COLLECTION_NAME

        store = get_vector_store()
        stats = store.get_collection_stats()
        count = stats["vector_count"]

        # Vectors live in the ledger database, so the "store size" is that file.
        db_path = stats.get("db_path", "")
        db_size = os.path.getsize(db_path) if db_path and os.path.isfile(db_path) else 0

        _row("Collection name",  COLLECTION_NAME)
        _row("Vectors stored",   count, color=Fore.GREEN if count > 0 else Fore.RED)
        _row("Documents covered", stats.get("document_count", 0))
        _row("Distance metric",  "cosine (normalized dot product)")
        _row("Backend",          stats.get("backend", "sqlite"))
        _row("Persistence path", db_path)
        _row("Store size",       _fmt_bytes(db_size))

        return {"vectors": count, "db_path": db_path, "db_size": db_size}

    except Exception as exc:
        print(f"  {Fore.RED}ERROR accessing vector store: {exc}{Style.RESET_ALL}")
        return {}


# -----------------------------------------------------------------------------
# Section 5 -- Retrieval Demonstration
# -----------------------------------------------------------------------------

def section_5() -> list:
    _section(5, "Retrieval Demonstration")

    QUERY = "What is the attendance requirement for semester examinations?"
    print(f"  {Fore.CYAN}Query: \"{QUERY}\"{Style.RESET_ALL}")
    print()

    try:
        from retrieval.retriever import get_retriever

        retriever = get_retriever()

        try:
            response = retriever.retrieve_by_text(
                QUERY,
                role         = "Student",
                top_k        = 5,
                use_bm25     = True,
                use_reranker = True,
                top_k_dense  = 25,
                top_k_bm25   = 25,
                top_k_fusion = 25,
                top_k_final  = 5,
            )
        except Exception:
            response = retriever.retrieve_by_text(
                QUERY,
                role         = "Student",
                top_k        = 5,
                use_bm25     = True,
                use_reranker = False,
            )

        mode = response.retrieval_mode
        print(f"  {Fore.WHITE}Mode: {Fore.CYAN + Style.BRIGHT}{mode}"
              f"  {Fore.WHITE}Latency: {Fore.CYAN + Style.BRIGHT}{response.latency_ms:.0f} ms"
              f"{Style.RESET_ALL}")
        print()

        for r in response.results:
            meta  = r.metadata
            score = r.rerank_score if r.rerank_score is not None else r.score
            print(f"  {Fore.YELLOW + Style.BRIGHT}[{r.rank}] {r.citation.display_name}"
                  f"{Style.RESET_ALL}")
            print(f"      {Fore.WHITE}Dept    : {Fore.CYAN}{r.citation.department}"
                  f"  {Fore.WHITE}Category: {Fore.CYAN}{r.citation.category}"
                  f"  {Fore.WHITE}Version : {Fore.CYAN}v{r.citation.version}"
                  f"{Style.RESET_ALL}")
            rerank_str = f"{r.rerank_score:.4f}" if r.rerank_score else "N/A"
            print(f"      {Fore.WHITE}Score   : {Fore.GREEN + Style.BRIGHT}{score:.4f}"
                  f"  {Fore.WHITE}Rerank  : {Fore.GREEN}{rerank_str}"
                  f"  {Fore.WHITE}Access  : {Fore.CYAN}{meta.get('access_level', '?')}"
                  f"{Style.RESET_ALL}")
            preview = r.content[:200].replace("\n", " ").strip()
            print(f"      {Fore.WHITE + Style.DIM}\"{preview}...\"{Style.RESET_ALL}")
            print()

        return response.results

    except Exception as exc:
        print(f"  {Fore.RED}ERROR during retrieval: {exc}{Style.RESET_ALL}")
        return []


# -----------------------------------------------------------------------------
# Section 6 -- Prompt Builder
# -----------------------------------------------------------------------------

def section_6(results: list) -> object:
    _section(6, "Prompt Builder")

    QUESTION = "What is the attendance requirement for semester examinations?"

    try:
        from rag.prompt_builder import build_prompt
        from rag.prompt_schema  import PromptConfig, PromptTemplate

        if not results:
            print(f"  {Fore.YELLOW}No retrieval results -- skipping prompt generation.{Style.RESET_ALL}")
            return None

        cfg    = PromptConfig(template=PromptTemplate.DEFAULT, max_chunks=5, include_metadata=True)
        prompt = build_prompt(QUESTION, results, config=cfg)

        sys_preview = prompt.system_prompt[:280].replace("\n", " ").strip()
        usr_preview = prompt.user_message[:200].replace("\n", " ").strip()

        _row("Template used",        prompt.template_used.value)
        _row("Chunks in context",    prompt.chunks_included)
        _row("Chunks dropped",       prompt.chunks_dropped)
        _row("Context size (chars)", prompt.context_chars)
        _row("Prompt char count",    len(prompt.system_prompt) + len(prompt.user_message))
        _row("Conflict warnings",    len(prompt.conflicts),
             color=Fore.YELLOW if prompt.conflicts else Fore.GREEN)

        print()
        print(f"  {Fore.CYAN}-- System prompt (first 280 chars) --{Style.RESET_ALL}")
        print(f"  {Fore.WHITE + Style.DIM}{sys_preview}...{Style.RESET_ALL}")
        print()
        print(f"  {Fore.CYAN}-- User message (first 200 chars) --{Style.RESET_ALL}")
        print(f"  {Fore.WHITE + Style.DIM}{usr_preview}...{Style.RESET_ALL}")
        print()
        print(f"  {Fore.GREEN}NOTE: LLM not called. Prompt is ready for Qwen2.5:7B.{Style.RESET_ALL}")

        return prompt

    except Exception as exc:
        print(f"  {Fore.RED}ERROR in prompt builder: {exc}{Style.RESET_ALL}")
        return None


# -----------------------------------------------------------------------------
# Section 7 -- Citation Engine
# -----------------------------------------------------------------------------

def section_7(results: list) -> None:
    _section(7, "Citation Engine")

    PLACEHOLDER_ANSWER = (
        "According to the institutional regulations [SOURCE 1], students must maintain "
        "a minimum of 75% attendance in each subject [SOURCE 2] to be eligible for "
        "semester examinations [SOURCE 3]."
    )

    try:
        from rag.citation_engine import get_citation_engine

        if not results:
            print(f"  {Fore.YELLOW}No retrieval results -- skipping citation generation.{Style.RESET_ALL}")
            return

        engine = get_citation_engine()
        cl     = engine.build(results, PLACEHOLDER_ANSWER)

        _row("Total citations",    cl.total_citations)
        _row("Version conflicts",
             "YES" if cl.has_version_conflicts else "None detected",
             color=Fore.YELLOW if cl.has_version_conflicts else Fore.GREEN)
        print()

        for c in cl.citations:
            latest_str = "YES" if c.is_latest_version else "NO (superseded)"
            latest_clr = Fore.GREEN if c.is_latest_version else Fore.YELLOW
            eff        = c.rerank_score if c.rerank_score is not None else c.score
            print(f"  {Fore.YELLOW + Style.BRIGHT}{c.inline_ref}  "
                  f"{Fore.WHITE + Style.BRIGHT}{c.display_name}{Style.RESET_ALL}")
            print(f"      Dept: {Fore.CYAN}{c.department:<22}"
                  f"{Fore.WHITE}  Version: {Fore.CYAN}v{c.version:<8}"
                  f"{Fore.WHITE}  Latest: {latest_clr}{latest_str}{Style.RESET_ALL}")
            print(f"      Score: {Fore.GREEN + Style.BRIGHT}{eff:.4f}"
                  f"  {Fore.WHITE}Access: {Fore.CYAN}{c.access_level}"
                  f"  {Fore.WHITE}Page: {Fore.CYAN}{c.page_number}{Style.RESET_ALL}")
            print()

    except Exception as exc:
        print(f"  {Fore.RED}ERROR in citation engine: {exc}{Style.RESET_ALL}")


# -----------------------------------------------------------------------------
# Section 8 -- RAG Readiness
# -----------------------------------------------------------------------------

def section_8() -> None:
    _section(8, "RAG Readiness Check")

    _phase("Phase 1  Repository Assessment",      True)
    _phase("Phase 2  Document Ingestion",         True)
    _phase("Phase 3  SOP-Aware Chunking",         True)
    _phase("Phase 4  BGE Embeddings",             True)
    _phase("Phase 5  Vector Store",      True)
    _phase("Phase 6  Hybrid Retrieval",           True)
    _phase("Phase 6.5 Cross-Encoder Reranker",    True)
    _phase("Phase 7  Prompt Builder",             True)
    _phase("Phase 7.5 Citation Engine",           True)
    _phase("Phase 7B  Ollama Client/RAG Engine",  True)
    _phase("Phase 8  End-to-End RAG",             False, "NOT CONNECTED YET")
    _phase("Phase 9  FastAPI REST API",           False, "NOT STARTED")
    _phase("Phase 10 LangGraph Agents",           False, "NOT STARTED")


# -----------------------------------------------------------------------------
# Section 9 -- Full System Architecture
# -----------------------------------------------------------------------------

def section_9() -> None:
    _section(9, "Full System Architecture")

    pipeline = [
        ("Admin / Faculty Upload",      "document intake portal"),
        ("Metadata Extraction",         "title, dept, category, version"),
        ("Chunking",                    "SOP-aware recursive splitter"),
        ("BGE Embeddings",              "BAAI/bge-base-en-v1.5  (768-dim)"),
        ("the vector store",                    "persistent cosine-distance store"),
        ("Hybrid Retrieval",            "dense + SQLite FTS5 + RRF"),
        ("Reranker",                    "Groq listwise (one extra chat call)"),
        ("Prompt Builder",              "system + context + question"),
        ("Qwen2.5:7B  (Ollama)",        "local LLM, no data leaves campus"),
        ("Citation Engine",             "dedup, version-prefer, inline refs"),
        ("FastAPI  (Phase 9)",          "REST endpoints + RBAC"),
        ("LangGraph Agents (Phase 10)", "multi-step agentic workflows"),
        ("Final Answer",                "grounded, cited, role-filtered"),
    ]

    print()
    for i, (name, note) in enumerate(pipeline):
        print(f"  {Fore.WHITE + Style.BRIGHT}{name:<34}"
              f"{Fore.WHITE + Style.DIM}{note}{Style.RESET_ALL}")
        if i < len(pipeline) - 1:
            print(f"      {Fore.CYAN}|{Style.RESET_ALL}")
    print()


# -----------------------------------------------------------------------------
# Section 10 -- Performance Summary
# -----------------------------------------------------------------------------

def section_10(s1: dict, s3: dict, s4: dict) -> None:
    _section(10, "Performance Summary")

    try:
        import ledger
        db_size = os.path.getsize(ledger.DB_PATH) if os.path.exists(ledger.DB_PATH) else 0
    except Exception:
        db_size = 0

    total_docs   = s1.get("total_docs",   "N/A")
    total_chunks = s1.get("total_chunks", "N/A")
    avg_size     = s1.get("avg_size",     0)
    depts        = s1.get("depts",        "N/A")
    cats         = s1.get("cats",         "N/A")
    embedded     = s3.get("embedded",     "N/A")
    model        = s3.get("model",        "BAAI/bge-base-en-v1.5")
    vectors      = s4.get("vectors",      "N/A")
    store_size   = s4.get("db_size",      0)

    avg_per_doc = (
        f"{total_chunks / total_docs:.1f}"
        if isinstance(total_docs, int) and total_docs > 0
           and isinstance(total_chunks, int)
        else "N/A"
    )

    _row("Documents",               total_docs)
    _row("Chunks",                  total_chunks)
    _row("Embedded vectors",        embedded,     color=Fore.GREEN)
    _row("vectors",        vectors,      color=Fore.GREEN)
    _row("Departments",             depts)
    _row("Categories",              cats)
    _row("Average chunks/document", avg_per_doc)
    _row("Average chunk size",      f"{avg_size:.0f} chars" if avg_size else "N/A")
    _row("Embedding model",         model)
    _row("LLM model",               "Qwen2.5:7B  (via Ollama)")
    _row("Vector database",         "the vector store  (PersistentClient)")
    _row("SQLite ledger size",      _fmt_bytes(db_size))
    _row("Vector store size",         _fmt_bytes(store_size))


# -----------------------------------------------------------------------------
# Final Banner
# -----------------------------------------------------------------------------

def _final_banner() -> None:
    print()
    print(Fore.CYAN  + Style.BRIGHT + "=" * W)
    print(Fore.CYAN  + Style.BRIGHT + "CURRENT PROJECT COMPLETION".center(W))
    print(Fore.GREEN + Style.BRIGHT + "~80 %".center(W))
    print(Fore.WHITE + Style.BRIGHT + "Core AI Pipeline Complete".center(W))
    print(Fore.WHITE +                "Ready for FastAPI Integration".center(W))
    print(Fore.CYAN  + Style.BRIGHT + "=" * W)
    print()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    _banner(
        "VIT AGENTIC AI ASSISTANT\n"
        "PROJECT STATUS DEMONSTRATION\n"
        "============================\n"
        "READ-ONLY  .  No data is modified"
    )

    t_start = time.perf_counter()

    s1 = section_1()
    section_2()
    s3 = section_3()
    s4 = section_4()

    results = section_5()
    section_6(results)
    section_7(results)

    section_8()
    section_9()
    section_10(s1, s3, s4)

    elapsed = time.perf_counter() - t_start
    print()
    print(f"  {Fore.WHITE + Style.DIM}Demo completed in {elapsed:.1f} s{Style.RESET_ALL}")

    _final_banner()


if __name__ == "__main__":
    main()
