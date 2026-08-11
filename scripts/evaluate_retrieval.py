"""
scripts/evaluate_retrieval.py
-------------------------------
Comparative retrieval quality evaluation.

Compares three retrieval modes on a 20-query test set:
    Mode A: Dense Only         (BGE embedding + the vector store)
    Mode B: Hybrid             (Dense + BM25 + RRF, no reranker)
    Mode C: Hybrid + Reranker  (Dense + BM25 + RRF + bge-reranker-base)

Relevance proxy:
    Since we lack human-labeled relevance judgments, we use department
    matching as a proxy: a result is "relevant" if its department matches
    the query's target department.

    Limitation: this penalizes cross-departmental results that may genuinely
    answer the query (e.g., "fee collection" appears in both Finance and
    Fees and Billing). Treat scores as directional, not absolute.

Metrics:
    Precision@K  — fraction of top-K results from target department
    Recall@K     — fraction of all target-dept chunks in top-K
    MRR          — Mean Reciprocal Rank (1/rank of first relevant result)
    Hit@1        — fraction of queries where rank-1 result is relevant
    Hit@3        — fraction of queries where top-3 contains a relevant result
    Latency      — mean and p95 retrieval time in milliseconds

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --mode dense      # dense only
    python scripts/evaluate_retrieval.py --k 5             # evaluate at K=5
    python scripts/evaluate_retrieval.py --quiet           # no per-query output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.WARNING)  # suppress INFO during evaluation


# ---------------------------------------------------------------------------
# Test queries (department-tagged ground truth)
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    {
        "query":   "What is the student admission process at VIT?",
        "dept":    "Admissions",
        "notes":   "Core admissions workflow",
    },
    {
        "query":   "How are fees collected and billed to students?",
        "dept":    "Fees and Billing",
        "notes":   "Fee billing process",
    },
    {
        "query":   "Describe the faculty recruitment and evaluation procedure.",
        "dept":    "Faculty Recruitment & Evaluation",
        "notes":   "HR faculty process",
    },
    {
        "query":   "How is the theory examination paper set and evaluated?",
        "dept":    "Examination",
        "notes":   "Core exam workflow",
    },
    {
        "query":   "What is the library book issue and return process?",
        "dept":    "Library Management",
        "notes":   "Library SOP",
    },
    {
        "query":   "How does the training and placement cell operate?",
        "dept":    "Training & Placement",
        "notes":   "T&P SOP",
    },
    {
        "query":   "What is the budget preparation and approval process?",
        "dept":    "Budgeting",
        "notes":   "Finance/budget SOP",
    },
    {
        "query":   "How is campus security managed at VIT?",
        "dept":    "Security Management",
        "notes":   "Security SOP",
    },
    {
        "query":   "What are the research and development project procedures?",
        "dept":    "Research & Development",
        "notes":   "R&D SOP",
    },
    {
        "query":   "How does the alumni association function?",
        "dept":    "Alumni",
        "notes":   "Alumni SOP",
    },
    {
        "query":   "How is student enquiry and front office handled?",
        "dept":    "Enquiry Handling and Front Office",
        "notes":   "Front office SOP",
    },
    {
        "query":   "What is the affiliation management process with the university?",
        "dept":    "Affiliation Management",
        "notes":   "Affiliation SOP",
    },
    {
        "query":   "How are student activities and clubs managed?",
        "dept":    "Student Activities",
        "notes":   "Student activities SOP",
    },
    {
        "query":   "What is the repair and maintenance process for equipment?",
        "dept":    "Repair & Maintenance",
        "notes":   "Maintenance SOP",
    },
    {
        "query":   "How is the stores and purchase process managed?",
        "dept":    "Stores & Purchase",
        "notes":   "Procurement SOP",
    },
    {
        "query":   "What is the training and development process for HR staff?",
        "dept":    "Training & Development",
        "notes":   "HR T&D SOP",
    },
    {
        "query":   "How are committees formed and managed at VIT?",
        "dept":    "Other Committees",
        "notes":   "Committees SOP",
    },
    {
        "query":   "How does the finance and accounts department operate?",
        "dept":    "Finance & Accounts",
        "notes":   "Finance SOP",
    },
    {
        "query":   "What is the academics department process?",
        "dept":    "Academics",
        "notes":   "Academics SOP",
    },
    {
        "query":   "How are MMS admissions and academics managed?",
        "dept":    "Master of Management Studies (M.M.S)",
        "notes":   "MMS SOP",
    },
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query:          str
    target_dept:    str
    results_dense:  list[dict] = field(default_factory=list)
    results_hybrid: list[dict] = field(default_factory=list)
    results_rerank: list[dict] = field(default_factory=list)
    latency_dense:  float = 0.0
    latency_hybrid: float = 0.0
    latency_rerank: float = 0.0


@dataclass
class EvalMetrics:
    mode:       str
    precision:  float = 0.0   # Precision@K
    recall:     float = 0.0   # Recall@K
    mrr:        float = 0.0   # Mean Reciprocal Rank
    ndcg:       float = 0.0   # nDCG@K
    hit_at_1:   float = 0.0   # Hit@1
    hit_at_3:   float = 0.0   # Hit@3
    latency_mean: float = 0.0
    latency_p95:  float = 0.0
    k:          int   = 5
    n_queries:  int   = 0


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _is_relevant(result: dict, target_dept: str) -> bool:
    return result.get("metadata", {}).get("department", "") == target_dept


def _ndcg_at_k(results: list[dict], target_dept: str, k: int, total_relevant: int) -> float:
    """nDCG@K with binary relevance (department match)."""
    import math
    dcg = sum(
        (1.0 if _is_relevant(r, target_dept) else 0.0) / math.log2(rank + 1)
        for rank, r in enumerate(results[:k], start=1)
    )
    ideal_hits = min(k, max(total_relevant, 0))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    query_results: list[QueryResult],
    mode: str,
    k: int,
) -> EvalMetrics:
    """Compute retrieval metrics for a given mode across all queries."""
    precisions, recalls, rrs, ndcgs = [], [], [], []
    hit1s, hit3s, latencies  = [], [], []

    # Get total chunks per department for recall computation
    dept_totals = _get_dept_totals()

    for qr in query_results:
        if   mode == "dense":  results, lat = qr.results_dense,  qr.latency_dense
        elif mode == "hybrid": results, lat = qr.results_hybrid, qr.latency_hybrid
        else:                  results, lat = qr.results_rerank, qr.latency_rerank

        topk = results[:k]
        relevant_in_topk = [r for r in topk if _is_relevant(r, qr.target_dept)]

        # Precision@K
        precisions.append(len(relevant_in_topk) / k if k > 0 else 0.0)

        # Recall@K
        total_relevant = dept_totals.get(qr.target_dept, 1)
        recalls.append(len(relevant_in_topk) / total_relevant)

        # nDCG@K
        ndcgs.append(_ndcg_at_k(results, qr.target_dept, k, total_relevant))

        # MRR
        rr = 0.0
        for rank, r in enumerate(results[:k], start=1):
            if _is_relevant(r, qr.target_dept):
                rr = 1.0 / rank
                break
        rrs.append(rr)

        # Hit@1
        hit1s.append(1.0 if topk and _is_relevant(topk[0], qr.target_dept) else 0.0)

        # Hit@3
        top3 = results[:3]
        hit3s.append(1.0 if any(_is_relevant(r, qr.target_dept) for r in top3) else 0.0)

        latencies.append(lat)

    latencies_sorted = sorted(latencies)
    p95_idx = int(len(latencies_sorted) * 0.95)

    return EvalMetrics(
        mode         = mode,
        precision    = sum(precisions) / len(precisions) if precisions else 0.0,
        recall       = sum(recalls)    / len(recalls)    if recalls    else 0.0,
        mrr          = sum(rrs)        / len(rrs)        if rrs        else 0.0,
        ndcg         = sum(ndcgs)      / len(ndcgs)      if ndcgs      else 0.0,
        hit_at_1     = sum(hit1s)      / len(hit1s)      if hit1s      else 0.0,
        hit_at_3     = sum(hit3s)      / len(hit3s)      if hit3s      else 0.0,
        latency_mean = sum(latencies)  / len(latencies)  if latencies  else 0.0,
        latency_p95  = latencies_sorted[p95_idx]         if latencies_sorted else 0.0,
        k            = k,
        n_queries    = len(query_results),
    )


def _get_dept_totals() -> dict:
    """Count chunks per department from SQLite for recall computation.

    Uses LEDGER_DB_PATH when set so this measures the right database during
    a re-index migration (Phase 7) rather than silently falling back to
    whatever ingestion_ledger.db happens to be on disk.
    """
    import sqlite3
    db_path = os.environ.get("LEDGER_DB_PATH") or os.path.join(
        os.path.dirname(__file__), "..", "ingestion_ledger.db"
    )
    conn = sqlite3.connect(os.path.abspath(db_path))
    cur  = conn.cursor()
    cur.execute("SELECT department, COUNT(*) FROM chunks GROUP BY department")
    totals = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return totals


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    k:     int  = 5,
    modes: list = None,
    quiet: bool = False,
) -> dict[str, EvalMetrics]:
    """
    Run all three retrieval modes on all test queries.

    Args:
        k:     Evaluate metrics at top-K.
        modes: List of modes to run: ["dense", "hybrid", "rerank"]. Default: all.
        quiet: If True, suppress per-query output.

    Returns:
        Dict mapping mode name to EvalMetrics.
    """
    if modes is None:
        modes = ["dense", "hybrid", "rerank"]

    from retrieval.retriever import get_retriever

    retriever = get_retriever()
    query_results: list[QueryResult] = []

    total = len(TEST_QUERIES)
    print(f"\nRunning evaluation: {total} queries × {len(modes)} modes (K={k})")
    print("=" * 65)

    for i, tq in enumerate(TEST_QUERIES, start=1):
        query_text  = tq["query"]
        target_dept = tq["dept"]
        qr = QueryResult(query=query_text, target_dept=target_dept)

        if not quiet:
            print(f"\n[{i:02d}/{total}] {query_text[:55]}...")
            print(f"        Target dept: {target_dept}")

        # ---- Mode A: Dense only ----
        if "dense" in modes:
            t0 = time.perf_counter()
            resp = retriever.retrieve_by_text(
                query_text,
                use_bm25=False, use_reranker=False,
                top_k=k, top_k_dense=k,
            )
            qr.latency_dense  = (time.perf_counter() - t0) * 1000
            qr.results_dense  = [
                {"chunk_id": r.chunk_id, "metadata": r.metadata, "score": r.score}
                for r in resp.results
            ]
            if not quiet:
                top1_dept = qr.results_dense[0]["metadata"].get("department","?") if qr.results_dense else "—"
                hit = "✓" if qr.results_dense and _is_relevant(qr.results_dense[0], target_dept) else "✗"
                print(f"  Dense     [{hit}]: top dept={top1_dept} | {qr.latency_dense:.0f}ms")

        # ---- Mode B: Hybrid (no reranker) ----
        if "hybrid" in modes:
            t0 = time.perf_counter()
            resp = retriever.retrieve_by_text(
                query_text,
                use_bm25=True, use_reranker=False,
                top_k=k, top_k_dense=25, top_k_bm25=25, top_k_fusion=k,
            )
            qr.latency_hybrid  = (time.perf_counter() - t0) * 1000
            qr.results_hybrid  = [
                {"chunk_id": r.chunk_id, "metadata": r.metadata, "score": r.score}
                for r in resp.results
            ]
            if not quiet:
                top1_dept = qr.results_hybrid[0]["metadata"].get("department","?") if qr.results_hybrid else "—"
                hit = "✓" if qr.results_hybrid and _is_relevant(qr.results_hybrid[0], target_dept) else "✗"
                print(f"  Hybrid    [{hit}]: top dept={top1_dept} | {qr.latency_hybrid:.0f}ms")

        # ---- Mode C: Hybrid + Reranker ----
        if "rerank" in modes:
            t0 = time.perf_counter()
            resp = retriever.retrieve_by_text(
                query_text,
                use_bm25=True, use_reranker=True,
                top_k_dense=25, top_k_bm25=25, top_k_fusion=25, top_k_final=k,
            )
            qr.latency_rerank  = (time.perf_counter() - t0) * 1000
            qr.results_rerank  = [
                {"chunk_id": r.chunk_id, "metadata": r.metadata,
                 "score": r.score, "rerank_score": r.rerank_score}
                for r in resp.results
            ]
            if not quiet:
                top1_dept = qr.results_rerank[0]["metadata"].get("department","?") if qr.results_rerank else "—"
                hit = "✓" if qr.results_rerank and _is_relevant(qr.results_rerank[0], target_dept) else "✗"
                rr_s = f"{qr.results_rerank[0]['rerank_score']:.4f}" if qr.results_rerank else "—"
                print(f"  Hybrid+RR [{hit}]: top dept={top1_dept} | rerank_score={rr_s} | {qr.latency_rerank:.0f}ms")

        query_results.append(qr)

    # --- Compute metrics ---
    metrics: dict[str, EvalMetrics] = {}
    if "dense"  in modes: metrics["dense"]  = compute_metrics(query_results, "dense",  k)
    if "hybrid" in modes: metrics["hybrid"] = compute_metrics(query_results, "hybrid", k)
    if "rerank" in modes: metrics["rerank"] = compute_metrics(query_results, "rerank", k)

    return metrics


def print_metrics_table(metrics: dict[str, EvalMetrics], k: int) -> None:
    """Print a formatted comparison table."""
    print()
    print("=" * 65)
    print(f"  RETRIEVAL QUALITY COMPARISON  (K={k}, N={list(metrics.values())[0].n_queries} queries)")
    print("=" * 65)
    header = f"  {'Mode':<18} {'P@K':>6} {'Recall@K':>9} {'MRR':>7} {'nDCG@K':>7} {'Hit@1':>7} {'Hit@3':>7} {'Lat(ms)':>8} {'P95(ms)':>8}"
    print(header)
    print("-" * 65)

    mode_labels = {
        "dense":  "Dense Only",
        "hybrid": "Hybrid (D+B+RRF)",
        "rerank": "Hybrid + Reranker",
    }
    for mode, m in metrics.items():
        label = mode_labels.get(mode, mode)
        print(
            f"  {label:<18} "
            f"{m.precision:>6.3f} "
            f"{m.recall:>9.3f} "
            f"{m.mrr:>7.3f} "
            f"{m.ndcg:>7.3f} "
            f"{m.hit_at_1:>7.3f} "
            f"{m.hit_at_3:>7.3f} "
            f"{m.latency_mean:>8.0f} "
            f"{m.latency_p95:>8.0f}"
        )

    print("=" * 65)
    print()
    print("  Metrics explanation:")
    print(f"  P@K      — Precision@{k}: fraction of top-{k} from target department")
    print(f"  Recall@{k} — fraction of all target-dept chunks in top-{k}")
    print("  MRR      — Mean Reciprocal Rank (1/rank of first relevant result)")
    print(f"  nDCG@{k}   — normalized Discounted Cumulative Gain (binary relevance)")
    print("  Hit@1    — fraction of queries where rank-1 is from target dept")
    print("  Hit@3    — fraction of queries where top-3 contains a relevant result")
    print("  Lat/P95  — mean and 95th-percentile latency in ms")
    print()
    print("  Note: relevance proxy = department match. Cross-dept relevant")
    print("        results (e.g. fees in Finance + Fees and Billing) may")
    print("        be marked as irrelevant. Treat scores as directional.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality across modes.")
    parser.add_argument("--k",     type=int, default=5,   help="Evaluate at top-K (default: 5)")
    parser.add_argument("--mode",  type=str, default=None,
                        help="Comma-separated modes: dense,hybrid,rerank (default: all)")
    parser.add_argument("--quiet", action="store_true",   help="Suppress per-query output")
    parser.add_argument("--json",  type=str, default=None,
                        help="Write metrics as JSON to this path (for before/after diffing)")
    parser.add_argument("--label", type=str, default=None,
                        help="Label stored in the JSON output (e.g. 'baseline', 'phase2')")
    args = parser.parse_args()

    modes = args.mode.split(",") if args.mode else ["dense", "hybrid", "rerank"]
    modes = [m.strip() for m in modes]

    metrics = run_evaluation(k=args.k, modes=modes, quiet=args.quiet)
    print_metrics_table(metrics, k=args.k)

    if args.json:
        import datetime
        payload = {
            "label":     args.label,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "k":         args.k,
            "modes":     {mode: asdict(m) for mode, m in metrics.items()},
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  Wrote metrics to {args.json}")
