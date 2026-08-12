"""
scripts/evaluate_answers.py
-----------------------------
Answer-quality evaluation harness (Harness B).

Complements scripts/evaluate_retrieval.py, which only measures whether the
right chunks come back -- it can't tell you whether the final answer is
grounded, correctly cited, or correctly refuses to answer. This harness
drives the real pipeline end to end (retrieval -> generation -> citation)
over four query buckets:

    (a) answerable, single document  -> expect a grounded answer, >=1 citation
    (b) answerable, cross-document   -> expect >=2 citations
    (c) unanswerable / out-of-corpus -> MUST return the fallback, zero citations
    (d) deleted-document probe       -> upload, query, delete, query again;
                                         the deleted content must not resurface

Bucket (c) is the calibration target for RELEVANCE_FLOOR and the metric
that actually proves hallucination reduction -- it is where the recorded
baseline bug lived ("What is the capital of France?" got a correctly-worded
refusal that still shipped four fabricated citations).

Bucket (d) is the regression test for the originally reported bug (deleted
documents still cited). It mutates the corpus (uploads and deletes one
throwaway document) -- safe to run against a live instance, but not
something to run in a tight loop against production without knowing that.

Rate limits: this calls the real Groq backend once (sometimes twice, with
reranking) per query. Calls are spaced out to stay under free-tier TPM;
expect this to take a few minutes for the default query set.

Usage:
    python scripts/evaluate_answers.py
    python scripts/evaluate_answers.py --json results.json
    python scripts/evaluate_answers.py --skip-delete-probe
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logging.basicConfig(level=logging.WARNING)

_SOURCE_MARKER_RE = re.compile(r"\[SOURCE\s+\d+(?:\s*,\s*SOURCE\s+\d+)*\]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Query buckets
# ---------------------------------------------------------------------------

BUCKET_A_SINGLE_DOC = [
    "What is the library book issue and return process?",
    "What is the student admission process at VIT?",
    "How is campus security managed at VIT?",
    "What is the budget preparation and approval process?",
    "How does the alumni association function?",
    "What is the MMS admission process?",
]

BUCKET_B_CROSS_DOC = [
    "What steps are involved in both admissions and fee payment for a new student?",
    "How do examinations and academics departments coordinate on results?",
]

BUCKET_C_UNANSWERABLE = [
    "What is the capital of France?",
    "How many days of annual leave does a Google employee get?",
    "What is the boiling point of mercury?",
    "Who won the 2022 FIFA World Cup?",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryOutcome:
    bucket:              str
    query:                str
    answer_preview:       str = ""
    is_fallback:          bool = False
    citation_count:       int = 0
    marker_emitted:       bool = False
    hallucinated_markers: int = 0
    citations_inferred:   bool = False
    confidence_score:     float = 0.0
    latency_ms:           float = 0.0
    error:                Optional[str] = None


@dataclass
class HarnessResult:
    label:      str = ""
    timestamp:  str = ""
    outcomes:   list = field(default_factory=list)
    metrics:    dict = field(default_factory=dict)
    delete_probe: Optional[dict] = None


# ---------------------------------------------------------------------------
# Query runner
# ---------------------------------------------------------------------------

def _run_one(pipeline, query: str, bucket: str, role: str = "Public") -> QueryOutcome:
    from response_schema import FALLBACK_ANSWER

    t0 = time.perf_counter()
    try:
        resp = pipeline.run(query, role=role)
    except Exception as exc:
        return QueryOutcome(bucket=bucket, query=query, error=str(exc)[:200])
    latency_ms = (time.perf_counter() - t0) * 1000

    is_fallback = resp.answer.startswith(FALLBACK_ANSWER[:40])
    markers = _SOURCE_MARKER_RE.findall(resp.answer)

    return QueryOutcome(
        bucket              = bucket,
        query               = query,
        answer_preview      = resp.answer[:150].replace("\n", " "),
        is_fallback         = is_fallback,
        citation_count      = len(resp.citations),
        marker_emitted      = len(markers) > 0,
        citations_inferred  = resp.citations_inferred,
        confidence_score    = resp.confidence_score,
        latency_ms          = round(latency_ms, 1),
    )


def _pace():
    """Spacing between Groq calls to stay under free-tier TPM."""
    time.sleep(float(os.environ.get("EVAL_PACE_SECONDS", "8")))


# ---------------------------------------------------------------------------
# Bucket (d): deleted-document probe
# ---------------------------------------------------------------------------

def run_delete_probe(pipeline, quiet: bool) -> dict:
    """
    Uploads a throwaway document with a unique term, confirms it's
    retrievable, deletes it, confirms it is NOT retrievable and does not
    appear in citations -- in the same process, no restart. This is the
    direct regression test for the originally reported bug.
    """
    import ledger
    from backend.document_manager import ingest_uploaded_file, delete_document
    from response_schema import FALLBACK_ANSWER

    unique_term = f"Zylophant{int(time.time())}"
    doc_text = (
        f"[PROCESS: {unique_term} Equipment Loan]\n"
        f"The {unique_term} equipment loan scheme requires students to submit "
        f"form ZY-{int(time.time()) % 1000} co-signed by a faculty sponsor."
    )

    try:
        import docx
    except ImportError:
        return {"skipped": "python-docx not available in this environment"}

    tmp_path = os.path.join(os.path.dirname(__file__), f"_probe_{unique_term}.docx")
    d = docx.Document()
    for line in doc_text.split("\n"):
        d.add_paragraph(line)
    d.save(tmp_path)

    result = {"unique_term": unique_term}
    doc_id = None
    try:
        with open(tmp_path, "rb") as f:
            content = f.read()
        ingest_result = ingest_uploaded_file(f"probe_{unique_term}.docx", content, uploaded_by="eval-harness")
        doc_id = ingest_result.doc_id
        result["ingested"] = ingest_result.status == "ingested"

        query = f"What is the {unique_term} equipment loan process and form number?"

        before = _run_one(pipeline, query, "delete_probe_before")
        result["before_found"] = unique_term.lower() in before.answer_preview.lower() or before.citation_count > 0
        if not quiet:
            print(f"  before delete: found={result['before_found']}  citations={before.citation_count}")

        if doc_id:
            delete_document(doc_id)

        _pace()
        after = _run_one(pipeline, query, "delete_probe_after")
        result["after_is_fallback"] = after.is_fallback
        result["after_citation_count"] = after.citation_count
        result["after_mentions_term"] = unique_term.lower() in after.answer_preview.lower()
        if not quiet:
            print(f"  after delete:  fallback={after.is_fallback}  citations={after.citation_count}  "
                  f"mentions_term={result['after_mentions_term']}")

        conn = ledger.get_connection()
        try:
            for table in ("documents", "chunks", "embeddings"):
                n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE doc_id = ?", (doc_id,)).fetchone()[0]
                result[f"db_rows_remaining_{table}"] = n
        finally:
            conn.close()

        result["passed"] = (
            result.get("after_citation_count", 1) == 0
            and all(result.get(f"db_rows_remaining_{t}", 1) == 0 for t in ("documents", "chunks", "embeddings"))
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    return result


# ---------------------------------------------------------------------------
# Run + report
# ---------------------------------------------------------------------------

def run_evaluation(quiet: bool = False, skip_delete_probe: bool = False) -> HarnessResult:
    import datetime
    from rag_pipeline import get_pipeline

    pipeline = get_pipeline()
    outcomes: list[QueryOutcome] = []

    all_queries = (
        [(q, "single_doc") for q in BUCKET_A_SINGLE_DOC]
        + [(q, "cross_doc") for q in BUCKET_B_CROSS_DOC]
        + [(q, "unanswerable") for q in BUCKET_C_UNANSWERABLE]
    )

    for i, (query, bucket) in enumerate(all_queries, start=1):
        if not quiet:
            print(f"[{i}/{len(all_queries)}] ({bucket}) {query[:60]}")
        outcome = _run_one(pipeline, query, bucket)
        outcomes.append(outcome)
        if not quiet:
            if outcome.error:
                print(f"    ERROR: {outcome.error}")
            else:
                print(f"    fallback={outcome.is_fallback} citations={outcome.citation_count} "
                      f"marker={outcome.marker_emitted} inferred={outcome.citations_inferred} "
                      f"conf={outcome.confidence_score:.2f} lat={outcome.latency_ms:.0f}ms")
        if i < len(all_queries):
            _pace()

    delete_probe = None
    if not skip_delete_probe:
        if not quiet:
            print("\n[delete probe] uploading, querying, deleting, querying again...")
        _pace()
        delete_probe = run_delete_probe(pipeline, quiet)

    metrics = _compute_metrics(outcomes, delete_probe)

    return HarnessResult(
        label       = "answer-quality",
        timestamp   = datetime.datetime.now(datetime.timezone.utc).isoformat(),
        outcomes    = [asdict(o) for o in outcomes],
        metrics     = metrics,
        delete_probe = delete_probe,
    )


def _compute_metrics(outcomes: list[QueryOutcome], delete_probe: Optional[dict]) -> dict:
    def _bucket(name):
        return [o for o in outcomes if o.bucket == name and o.error is None]

    single_doc = _bucket("single_doc")
    cross_doc  = _bucket("cross_doc")
    unanswerable = _bucket("unanswerable")

    metrics = {
        "n_errors": sum(1 for o in outcomes if o.error),
        "single_doc": {
            "n": len(single_doc),
            "answered_with_citation_rate": _rate(single_doc, lambda o: not o.is_fallback and o.citation_count >= 1),
            "marker_emission_rate": _rate(single_doc, lambda o: o.marker_emitted),
        },
        "cross_doc": {
            "n": len(cross_doc),
            "cited_2plus_rate": _rate(cross_doc, lambda o: o.citation_count >= 2),
        },
        "unanswerable": {
            "n": len(unanswerable),
            # THE key hallucination-reduction metric.
            "fallback_rate": _rate(unanswerable, lambda o: o.is_fallback),
            "zero_citation_rate": _rate(unanswerable, lambda o: o.citation_count == 0),
        },
        "latency_ms_mean": round(sum(o.latency_ms for o in outcomes if not o.error) / max(1, len(outcomes) - sum(1 for o in outcomes if o.error)), 1),
    }
    if delete_probe is not None:
        metrics["delete_probe_passed"] = delete_probe.get("passed", False)
    return metrics


def _rate(items: list, predicate) -> float:
    if not items:
        return 0.0
    return round(sum(1 for i in items if predicate(i)) / len(items), 3)


def print_report(result: HarnessResult) -> None:
    print()
    print("=" * 65)
    print("  ANSWER QUALITY REPORT")
    print("=" * 65)
    m = result.metrics
    print(f"  Single-doc answered+cited : {m['single_doc']['answered_with_citation_rate']:.0%}  (n={m['single_doc']['n']})")
    print(f"  Single-doc marker emitted : {m['single_doc']['marker_emission_rate']:.0%}")
    print(f"  Cross-doc cited >=2       : {m['cross_doc']['cited_2plus_rate']:.0%}  (n={m['cross_doc']['n']})")
    print(f"  Unanswerable -> fallback  : {m['unanswerable']['fallback_rate']:.0%}  (n={m['unanswerable']['n']})  <-- hallucination signal")
    print(f"  Unanswerable -> 0 citations: {m['unanswerable']['zero_citation_rate']:.0%}")
    print(f"  Mean latency              : {m['latency_ms_mean']:.0f}ms")
    if "delete_probe_passed" in m:
        print(f"  Delete probe passed       : {m['delete_probe_passed']}")
    if m["n_errors"]:
        print(f"  Errors                    : {m['n_errors']}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answer quality end-to-end.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--skip-delete-probe", action="store_true")
    args = parser.parse_args()

    result = run_evaluation(quiet=args.quiet, skip_delete_probe=args.skip_delete_probe)
    if args.label:
        result.label = args.label
    print_report(result)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"  Wrote results to {args.json}")
