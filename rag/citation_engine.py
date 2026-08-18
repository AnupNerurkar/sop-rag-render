"""
rag/citation_engine.py
-----------------------
Phase 8: Source Citation Engine.

Pipeline:
    list[RetrievalResult] + answer str
        ↓  _group_by_doc()          — merge chunks from the same document
        ↓  _resolve_versions()      — flag superseded versions
        ↓  _sort()                  — score desc, then display_name asc (deterministic)
        ↓  _assign_ranks()          — 1-based rank + inline_ref "[N]"
        ↓  _inject_inline_refs()    — replace [SOURCE N] with [N] in answer text
    CitationList

Also supports ContextChunk input via build_from_chunks() for callers
that only have BuiltPrompt.context_chunks available (limited metadata).
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Optional, TYPE_CHECKING

from rag.citation_schema import Citation, CitationList

if TYPE_CHECKING:
    from retrieval.retrieval_schema import RetrievalResult
    from rag.prompt_schema import ContextChunk

logger = logging.getLogger(__name__)

# Matches [SOURCE 1], [SOURCE 12], [source 3], AND the bundled form the
# model sometimes emits instead of one marker per citation --
# [SOURCE 1, SOURCE 2, SOURCE 3]. A pattern matching only a single number
# per bracket (the original version of this regex) doesn't recognize the
# bundled form at all, so it passes through unresolved and unsuppressed --
# caught live: a real answer ended with "...late returns
# [SOURCE 1, SOURCE 2, SOURCE 3]." and that raw text reached the client
# verbatim, both mid-stream and in the final formatted_answer.
#
# The separator between bundled entries is "," for most models but the
# gpt-oss-120b swap (2026-08-18) showed "[SOURCE 3 ; SOURCE 4]" -- caught
# live the same way, so the separator itself is tolerant rather than a
# fixed comma. That same model also sometimes wraps the marker in markdown
# emphasis -- "[**SOURCE 1**]" -- hence the optional \*{0,2} around each
# SOURCE/number pair.
_SOURCE_GROUP_RE = re.compile(
    r"\[\s*\*{0,2}\s*SOURCE\s+\d+\s*\*{0,2}\s*"
    r"(?:[,;]\s*\*{0,2}\s*SOURCE\s+\d+\s*\*{0,2}\s*)*\]",
    re.IGNORECASE,
)
_SOURCE_NUM_RE = re.compile(r"\d+")


def _extract_source_numbers(marker_text: str) -> list[int]:
    """All source numbers inside one [SOURCE N] or [SOURCE N, SOURCE M, ...] marker."""
    return [int(n) for n in _SOURCE_NUM_RE.findall(marker_text)]


# When the answer emits no valid [SOURCE N] markers at all, cite this many
# top-ranked chunks that were actually placed in the prompt, flagged via
# CitationList.citations_inferred so the UI can label them accordingly
# rather than claiming the answer cited them.
CITATION_FALLBACK_TOP_N = 2


# ---------------------------------------------------------------------------
# Version comparison (no external dependencies)
# ---------------------------------------------------------------------------

def _version_key(v: str) -> tuple:
    """Sortable key for version strings: '2.1' > '1.5' > '1.0' > 'Final' > ''."""
    nums = re.findall(r"\d+", v)
    return tuple(int(n) for n in nums) if nums else (0,)


# ---------------------------------------------------------------------------
# CitationEngine
# ---------------------------------------------------------------------------

class CitationEngine:
    """
    Stateless citation builder. Thread-safe — all state is local to build().

    Swap note: replace with a subclass or alternate implementation to change
    deduplication strategy without modifying any other module.
    """

    # ------------------------------------------------------------------
    # Primary API — accepts RetrievalResult (full metadata)
    # ------------------------------------------------------------------

    def build(
        self,
        results: "list[RetrievalResult]",
        answer:  str,
        source_index: "Optional[dict[int, str]]" = None,
    ) -> CitationList:
        """
        Build a CitationList from retrieval results and the generated answer.

        Strict by default: citations are built only from documents the
        answer actually referenced via [SOURCE N], ranked by first
        appearance in the text -- not from every retrieved chunk regardless
        of use. Previously this grouped and cited *all* of `results`
        unconditionally, so "Sources:" was effectively a retrieval dump; an
        answer that referenced nothing (or referenced the wrong thing)
        still shipped a full citation list, which is how the recorded
        baseline case shipped four citations to unrelated departments on a
        question the model correctly said it couldn't answer.

        Args:
            results: list[RetrievalResult] from Retriever.retrieve().
            answer:  Generated answer string from RAGEngine.generate().
            source_index: {source_number: chunk_id} exactly as shown to the
                model in the prompt (BuiltPrompt.context_chunks). Without
                this, [SOURCE N] is assumed to number `results` positionally
                from 1 -- correct only when nothing was filtered or
                reordered between retrieval and the prompt. With
                `max_chunks`/budget trimming or confidence_threshold
                filtering in play, that assumption silently desyncs
                (prompt_builder renumbers *after* filtering), so callers
                that have a BuiltPrompt should always pass this.

        Returns:
            CitationList with citations in first-referenced order (or, if
            the answer emitted no valid markers, the top
            CITATION_FALLBACK_TOP_N chunks actually shown to the model,
            flagged via citations_inferred) and an annotated answer where
            [SOURCE N] → [N] for real citations and hallucinated markers
            (referencing a source number never shown to the model) are
            stripped rather than left as raw text.
        """
        if not results:
            return CitationList(
                citations        = [],
                answer_with_refs = answer,
                original_answer  = answer,
                total_citations  = 0,
            )

        chunk_to_doc = {r.chunk_id: r.citation.doc_id for r in results}

        if source_index:
            source_to_doc = {n: chunk_to_doc.get(cid) for n, cid in source_index.items()}
        else:
            # No prompt-derived mapping available -- fall back to
            # positional numbering in the order results were passed in.
            # This is only correct when the caller's [SOURCE N] numbering
            # matches this exact list, which is the assumption a
            # source_index argument exists to remove.
            source_to_doc = {i: r.citation.doc_id for i, r in enumerate(results, start=1)}

        referenced: list[int] = []
        for m in _SOURCE_GROUP_RE.finditer(answer):
            referenced.extend(_extract_source_numbers(m.group(0)))
        cited_doc_ids: list[str] = []
        hallucinated_numbers: set[int] = set()
        for n in referenced:
            doc_id = source_to_doc.get(n)
            if doc_id is None:
                hallucinated_numbers.add(n)
                continue
            if doc_id not in cited_doc_ids:
                cited_doc_ids.append(doc_id)

        citations_inferred = False
        if cited_doc_ids:
            # Strict mode: exactly what the answer referenced, ranked by
            # first appearance in the text -- reading order, not score
            # order, since that's what a reader actually encounters.
            selected_results = [r for r in results if r.citation.doc_id in cited_doc_ids]
            order_key = {doc_id: i for i, doc_id in enumerate(cited_doc_ids)}
        else:
            # No valid markers. Cite the top CITATION_FALLBACK_TOP_N chunks
            # that were actually in the prompt (source_index's values), not
            # the full retrieval set -- results can hold more candidates
            # than made it past max_chunks/the char budget, and citing one
            # the model never saw would misrepresent what it read.
            pool = results
            if source_index:
                shown_chunk_ids = set(source_index.values())
                pool = [r for r in results if r.chunk_id in shown_chunk_ids]
            # Sorted by effective score (display_name as a deterministic
            # tiebreak) rather than trusting input order -- `results` is
            # score-ranked coming from the retriever in production, but
            # this keeps "top" meaning top regardless of what order a
            # caller passes, and ties resolve the same way every run.
            pool = sorted(
                pool,
                key=lambda r: (
                    -(r.rerank_score if r.rerank_score is not None else r.score),
                    r.citation.display_name.lower(),
                ),
            )
            top_doc_ids: list[str] = []
            for r in pool:
                if r.citation.doc_id not in top_doc_ids:
                    top_doc_ids.append(r.citation.doc_id)
                if len(top_doc_ids) >= CITATION_FALLBACK_TOP_N:
                    break
            selected_results = [r for r in pool if r.citation.doc_id in top_doc_ids]
            order_key = {doc_id: i for i, doc_id in enumerate(top_doc_ids)}
            citations_inferred = True

        # 1. Merge chunks that belong to the same document
        groups = self._group_by_doc(selected_results)

        # 2. Build a Citation per unique doc_id
        raw_citations = [
            self._build_citation_from_results(doc_id, doc_results)
            for doc_id, doc_results in groups.items()
        ]

        # 3. Flag superseded versions (same SOP, different version)
        raw_citations = self._resolve_versions(raw_citations)

        # 4. Order by first-appearance (strict) or top-score (fallback) --
        # see order_key above. Not _sort()'s score-descending order: in
        # strict mode, citation order should match reading order.
        raw_citations.sort(key=lambda c: order_key.get(c.doc_id, len(order_key)))

        # 5. Assign 1-based ranks and inline refs
        citations = self._assign_ranks(raw_citations)

        # Log each generated citation
        for c in citations:
            logger.info(
                f"[CITATION] [GEN] citation_id={c.citation_id} doc_id={c.doc_id} filepath={c.filepath}"
            )

        # 6. Build source_number → rank map for inline ref injection
        doc_rank = {c.doc_id: c.rank for c in citations}
        source_map = {
            n: doc_rank[doc_id]
            for n, doc_id in source_to_doc.items()
            if doc_id is not None and doc_id in doc_rank
        }

        # 7. Replace [SOURCE N] in answer; strip hallucinated markers
        answer_with_refs = _inject_inline_refs(answer, source_map)

        has_conflicts = any(not c.is_latest_version for c in citations)

        if hallucinated_numbers:
            logger.warning(
                "[CITATION] %d hallucinated [SOURCE N] marker(s) in answer "
                "(referenced a source number never shown to the model): %s",
                len(hallucinated_numbers), sorted(hallucinated_numbers),
            )

        logger.debug(
            "[CITATION] %d results -> %d citations | inferred=%s | conflicts=%s",
            len(results), len(citations), citations_inferred, has_conflicts,
        )

        return CitationList(
            citations                 = citations,
            answer_with_refs          = answer_with_refs,
            original_answer           = answer,
            total_citations           = len(citations),
            has_version_conflicts     = has_conflicts,
            source_number_map         = {str(k): v for k, v in source_map.items()},
            citations_inferred        = citations_inferred,
            hallucinated_marker_count = len(hallucinated_numbers),
        )

    # ------------------------------------------------------------------
    # Secondary API — accepts ContextChunk (limited metadata)
    # ------------------------------------------------------------------

    def build_from_chunks(
        self,
        chunks: "list[ContextChunk]",
        answer: str,
    ) -> CitationList:
        """
        Build CitationList from BuiltPrompt.context_chunks.

        Metadata available from ContextChunk is sufficient for most fields;
        source_file and access_level default to '' and 'Public' respectively.
        """
        if not chunks:
            return CitationList(
                citations        = [],
                answer_with_refs = answer,
                original_answer  = answer,
                total_citations  = 0,
            )

        # Group by doc_id (same dedup logic)
        groups: dict[str, list] = defaultdict(list)
        for chunk in chunks:
            groups[chunk.doc_id].append(chunk)

        raw_citations = [
            self._build_citation_from_chunks(doc_id, doc_chunks)
            for doc_id, doc_chunks in groups.items()
        ]

        raw_citations = self._resolve_versions(raw_citations)
        raw_citations = self._sort(raw_citations)
        citations     = self._assign_ranks(raw_citations)

        # source_number comes directly from ContextChunk.source_number
        source_map: dict[int, int] = {}
        for chunk in chunks:
            for c in citations:
                if chunk.doc_id == c.doc_id:
                    source_map[chunk.source_number] = c.rank
                    break

        answer_with_refs = _inject_inline_refs(answer, source_map)
        has_conflicts    = any(not c.is_latest_version for c in citations)

        return CitationList(
            citations             = citations,
            answer_with_refs      = answer_with_refs,
            original_answer       = answer,
            total_citations       = len(citations),
            has_version_conflicts = has_conflicts,
            source_number_map     = {str(k): v for k, v in source_map.items()},
        )

    # ------------------------------------------------------------------
    # Step 1: Group by doc_id
    # ------------------------------------------------------------------

    def _group_by_doc(
        self, results: "list[RetrievalResult]"
    ) -> dict[str, list]:
        groups: dict[str, list] = defaultdict(list)
        for r in results:
            groups[r.citation.doc_id].append(r)
        return dict(groups)

    # ------------------------------------------------------------------
    # Step 2a: Build Citation from RetrievalResult group
    # ------------------------------------------------------------------

    def _build_citation_from_results(
        self, doc_id: str, results: "list[RetrievalResult]"
    ) -> Citation:
        # Best result by effective score
        best = max(results, key=lambda r: (
            r.rerank_score if r.rerank_score is not None else r.score
        ))
        cit = best.citation

        # Earliest chunk in the document (lowest chunk_index)
        earliest = min(results, key=lambda r: r.citation.chunk_index)

        eff_score    = best.rerank_score if best.rerank_score is not None else best.score
        access_level = best.metadata.get("access_level", "Public")
        chunk_index  = earliest.citation.chunk_index
        
        filepath = best.metadata.get("source_file") or best.metadata.get("filepath") or ""

        return Citation(
            citation_id      = f"cite_{doc_id[:12]}",
            rank             = 0,   # assigned later
            inline_ref       = "",  # assigned later
            doc_id           = doc_id,
            display_name     = cit.display_name,
            title            = best.metadata.get("title") or cit.display_name,
            department       = cit.department,
            category         = cit.category,
            version          = cit.version,
            source_file      = cit.source_file,
            filepath         = filepath,
            section_heading  = best.citation.section_heading,
            chunk_ids        = [r.chunk_id for r in results],
            chunk_id         = best.chunk_id,
            chunk_index      = chunk_index,
            total_chunks     = cit.total_chunks,
            page_number      = chunk_index + 1 if chunk_index >= 0 else 0,
            score            = eff_score,
            rerank_score     = best.rerank_score,
            retrieval_score  = best.score,
            access_level     = access_level,
            is_latest_version = True,  # resolved later
        )

    # ------------------------------------------------------------------
    # Step 2b: Build Citation from ContextChunk group
    # ------------------------------------------------------------------

    def _build_citation_from_chunks(
        self, doc_id: str, chunks: "list[ContextChunk]"
    ) -> Citation:
        best = max(chunks, key=lambda c: (
            c.rerank_score if c.rerank_score is not None else c.score
        ))
        eff_score = best.rerank_score if best.rerank_score is not None else best.score

        # Best chunk_index: derive from rank (rank - 1 is a rough proxy)
        best_rank_chunk = min(chunks, key=lambda c: c.rank)
        chunk_index     = best_rank_chunk.rank - 1

        return Citation(
            citation_id      = f"cite_{doc_id[:12]}" if doc_id else f"cite_{id(best):x}",
            rank             = 0,
            inline_ref       = "",
            doc_id           = doc_id,
            display_name     = _extract_display_name(best.display_citation),
            title            = _extract_display_name(best.display_citation),
            department       = best.department,
            category         = best.category,
            version          = best.version,
            source_file      = "",
            filepath         = "",
            section_heading  = best.section_heading,
            chunk_ids        = [c.chunk_id for c in chunks],
            chunk_id         = best.chunk_id,
            chunk_index      = chunk_index,
            total_chunks     = 0,
            page_number      = chunk_index + 1,
            score            = eff_score,
            rerank_score     = best.rerank_score,
            retrieval_score  = best.score,
            access_level     = "Public",
            is_latest_version = True,
        )

    # ------------------------------------------------------------------
    # Step 3: Flag superseded versions
    # ------------------------------------------------------------------

    def _resolve_versions(self, citations: list[Citation]) -> list[Citation]:
        """
        Groups citations by (display_name, department, category).
        Within each group, marks all but the highest-version citation as
        is_latest_version=False.
        """
        # Group by identity key (ignoring version)
        groups: dict[tuple, list[Citation]] = defaultdict(list)
        for c in citations:
            key = (c.display_name.strip().lower(), c.department.strip().lower(), c.category.strip().lower())
            groups[key].append(c)

        result: list[Citation] = []
        for group in groups.values():
            if len(group) == 1:
                result.append(group[0])
                continue
            # Find the latest version
            latest = max(group, key=lambda c: _version_key(c.version))
            for c in group:
                is_latest = (c.doc_id == latest.doc_id)
                # model_copy creates a new instance with updated fields (Pydantic v2)
                result.append(c.model_copy(update={"is_latest_version": is_latest}))

        return result

    # ------------------------------------------------------------------
    # Step 4: Sort
    # ------------------------------------------------------------------

    def _sort(self, citations: list[Citation]) -> list[Citation]:
        """
        Deterministic sort:
            Primary  : effective score descending
            Secondary: display_name ascending (stable alphabetic tiebreak)
        """
        return sorted(
            citations,
            key=lambda c: (-c.score, c.display_name.lower()),
        )

    # ------------------------------------------------------------------
    # Step 5: Assign ranks and inline refs
    # ------------------------------------------------------------------

    def _assign_ranks(self, citations: list[Citation]) -> list[Citation]:
        ranked = []
        for i, c in enumerate(citations, start=1):
            ranked.append(c.model_copy(update={"rank": i, "inline_ref": f"[{i}]"}))
        return ranked


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _inject_inline_refs(answer: str, source_map: dict[int, int]) -> str:
    """
    Replaces each [SOURCE N] (or bundled [SOURCE N, SOURCE M, ...]) marker
    with [rank] / [rank1, rank2, ...] in the answer text.

    Any number not in source_map -- a hallucinated marker referencing a
    source that was never shown to the model, or a real source that wasn't
    among the ones ultimately selected -- is dropped from its group rather
    than left as raw "SOURCE 9" text. If every number in a bundled marker
    is unresolved, the whole bracket is stripped. Previously every
    unresolved number was left verbatim, and a bundled marker wasn't even
    recognized by the single-number pattern this replaced.
    """
    def _replace(m: re.Match) -> str:
        nums = _extract_source_numbers(m.group(0))
        ranks: list[int] = []
        for n in nums:
            rank = source_map.get(n)
            if rank is not None and rank not in ranks:
                ranks.append(rank)
        return f"[{', '.join(str(r) for r in ranks)}]" if ranks else ""

    return _SOURCE_GROUP_RE.sub(_replace, answer)


def _extract_display_name(display_citation: str) -> str:
    """
    Extracts display_name from a display_citation string.
    e.g. 'Fee SOP (v2.0) — Section — chunk 1 of 10'  →  'Fee SOP'
    """
    # Strip " (vX.Y)" suffix
    s = re.sub(r"\s*\(v[^)]*\).*", "", display_citation).strip()
    return s or display_citation


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_engine_instance: Optional[CitationEngine] = None


def get_citation_engine() -> CitationEngine:
    """Returns the process-level CitationEngine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CitationEngine()
    return _engine_instance


def reset_citation_engine() -> None:
    """Clears the singleton — for testing."""
    global _engine_instance
    _engine_instance = None
