"""
rag/prompt_builder.py
----------------------
Prompt Builder — Phase 7 (pre-LLM assembly, no Ollama call).

Pipeline (all pure Python — no network, no model load):

    RetrievalResult list (from Phase 6 retriever)
        ↓  _prepare_chunks()
    list[ContextChunk]  (scored, ranked, budget-trimmed)
        ↓  _detect_conflicts()
    list[ConflictGroup] (version conflicts)
        ↓  _format_context_block()
    context_block str
        ↓  _render_system_prompt()  +  _build_user_message()
    BuiltPrompt
        .messages  →  Phase 9 Ollama call
        .messages  →  Phase 10 LangGraph ChatOllama

Usage
-----
    from rag.prompt_builder import PromptBuilder, get_prompt_builder
    from rag.prompt_schema import PromptConfig, PromptTemplate

    builder = get_prompt_builder()
    prompt  = builder.build(
        question = "What is the fee payment procedure?",
        results  = retrieval_response.results,
    )
    # Phase 9 will do: ollama_client.chat(messages=prompt.messages)
"""

from __future__ import annotations

import re
import logging
from typing import Optional, TYPE_CHECKING

from rag.prompt_schema import (
    BuiltPrompt,
    ContextChunk,
    ConflictGroup,
    PromptConfig,
    PromptTemplate,
    confidence_from_score,
)
from response_schema import FALLBACK_ANSWER

if TYPE_CHECKING:
    from retrieval.retrieval_schema import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt templates
# ---------------------------------------------------------------------------

# All three templates instruct the model to emit FALLBACK_ANSWER's exact text
# for "I don't know" -- not their own wording, as they each used to. Before
# this, an in-scope-but-unanswerable question could produce a *correctly
# worded* refusal that response_schema.is_fallback still failed to
# recognize, because it only matched response_schema.FALLBACK_ANSWER's own
# text. Now every path that says "I don't know" says it identically, so
# is_fallback (and any UI logic keyed on it) sees all of them.
#
# Confidence self-reporting ("[Confidence: N%]") is gone from every
# template. It was never a measurement -- see response_schema.
# compute_confidence's docstring -- and the pipeline no longer reads or
# trusts anything the model writes about its own confidence.

_SYSTEM_PROMPTS: dict[PromptTemplate, str] = {

    PromptTemplate.DEFAULT: f"""\
You are an institutional knowledge assistant. \
You help students, faculty, and administrators find accurate information about \
institutional policies, Standard Operating Procedures (SOPs), circulars, and academic procedures.

STRICT RULES — follow these without exception:

1. CONTEXT ONLY: Answer exclusively from the retrieved context provided below. \
Do not use any knowledge from your training data or any source outside the supplied context.

2. UNAVAILABLE INFORMATION: If the answer cannot be found in the provided context, respond with exactly:
   "{FALLBACK_ANSWER}"

3. PREFER LATEST VERSION: When multiple document versions are present, prefer information \
from the most recent version. Always note the version you are citing.

4. CONFLICTING SOPs: If different retrieved sources describe the same process differently, \
explicitly state the conflict, name the sources involved, and indicate which is the more recent authority.

5. CITATIONS: Use [SOURCE N] inline in your answer wherever you reference a specific source. \
Do not invent, guess, alter, or embellish any source details.

6. NO FABRICATION: Never fabricate facts, procedures, dates, names, roles, or document references. \
If information is not explicitly stated in the context, do not include it in your answer.

Respond in clear, professional English suitable for a university institutional context.\
""",

    PromptTemplate.CONCISE: f"""\
You are an institutional knowledge assistant. Answer only from the supplied context.

Rules:
1. Use only the provided [SOURCE N] blocks. No outside knowledge.
2. If unavailable, respond with exactly: "{FALLBACK_ANSWER}"
3. Prefer the latest document version when versions conflict.
4. State any SOP conflicts explicitly.
5. Cite inline using [SOURCE N]. Never fabricate citations.
6. No fabrication of any kind.\
""",

    PromptTemplate.STRICT_CITATION: f"""\
You are an institutional knowledge assistant operating under strict citation rules.

ANSWER RULES:
1. Answer exclusively from the numbered [SOURCE N] blocks in the context.
2. Every factual claim MUST be followed immediately by [SOURCE N].
3. If the answer is not present, respond with exactly: "{FALLBACK_ANSWER}"
4. Prefer the highest-version source for any given document.
5. Explicitly name conflicting sources when they disagree.
6. NEVER fabricate, infer, or extend beyond what is written in the source text.

Citation rule: if you cannot point to a specific [SOURCE N] for a claim, do not make the claim.\
""",
}


# ---------------------------------------------------------------------------
# Version comparison helper
# ---------------------------------------------------------------------------

def _version_key(v: str) -> tuple:
    """
    Returns a sortable tuple for version strings so that numeric-style versions
    (1.0, 1.5, 2.0, 3.0) sort correctly, and non-numeric strings ("Final", "Draft")
    sort after all numeric versions.

    Examples:
        "2.0"  → (2, 0, "2.0")   >  "1.5" → (1, 5, "1.5")
        "Final"→ ()  — treated as non-numeric, sorts last
    """
    nums = re.findall(r"\d+", v)
    if nums:
        return tuple(int(n) for n in nums)
    return (0,)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Stateless assembler — no models, no I/O, no singletons needed.
    All heavy work (retrieval, reranking) is already done upstream.

    Thread-safe: all state is local to each build() call.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        question: str,
        results: "list[RetrievalResult]",
        config: Optional[PromptConfig] = None,
    ) -> BuiltPrompt:
        """
        Assemble a BuiltPrompt from a user question and a ranked result list.

        Args:
            question: The raw user question string.
            results:  list[RetrievalResult] from Retriever.retrieve().
            config:   Optional PromptConfig. Defaults to PromptConfig() if None.

        Returns:
            BuiltPrompt with .messages ready for the Ollama chat endpoint.
        """
        if config is None:
            config = PromptConfig()

        if not question.strip():
            raise ValueError("question must be a non-empty string.")

        # 1. Convert RetrievalResult → ContextChunk, apply budget + threshold
        chunks, dropped = self._prepare_chunks(results, config)

        # 2. Detect version conflicts across included chunks
        conflicts = self._detect_conflicts(chunks)

        # 3. Assemble the numbered context block
        context_block = self._format_context_block(chunks, config.include_metadata)

        # 4. Render the system prompt (template + conflict warnings injected)
        system_prompt = self._render_system_prompt(config.template, conflicts)

        # 5. Build the user turn (context block + question)
        user_message = self._build_user_message(context_block, question)

        # 6. Assemble Ollama/OpenAI chat messages
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        source_citations = [c.display_citation for c in chunks]

        logger.debug(
            "[PROMPT_BUILDER] question=%r | chunks=%d dropped=%d | "
            "conflicts=%d | chars=%d | template=%s",
            question[:60], len(chunks), dropped,
            len(conflicts), len(context_block), config.template.value,
        )

        return BuiltPrompt(
            user_question    = question,
            system_prompt    = system_prompt,
            context_block    = context_block,
            user_message     = user_message,
            messages         = messages,
            chunks_included  = len(chunks),
            chunks_dropped   = dropped,
            context_chars    = len(context_block),
            template_used    = config.template,
            conflicts        = conflicts,
            has_conflicts    = len(conflicts) > 0,
            source_citations = source_citations,
            context_chunks   = chunks,
        )

    # ------------------------------------------------------------------
    # Step 1: Prepare ContextChunk list
    # ------------------------------------------------------------------

    def _prepare_chunks(
        self,
        results: "list[RetrievalResult]",
        config: PromptConfig,
    ) -> tuple[list[ContextChunk], int]:
        """
        Converts RetrievalResult objects to ContextChunk objects, then:
        - Filters by confidence_threshold
        - Caps at max_chunks
        - Trims to max_context_chars (truncating the last chunk if needed)

        Returns (included_chunks, dropped_count).
        """
        total_input = len(results)
        candidates: list[ContextChunk] = []

        for rank0, result in enumerate(results):
            eff_score = (
                result.rerank_score
                if result.rerank_score is not None
                else result.score
            )

            if eff_score < config.confidence_threshold:
                continue

            chunk = ContextChunk(
                source_number    = len(candidates) + 1,  # renumbered after filter
                chunk_id         = result.chunk_id,
                content          = result.content,
                inline_citation  = result.citation.to_inline_citation(),
                display_citation = result.citation.to_display_citation(),
                department       = result.citation.department,
                category         = result.citation.category,
                version          = result.citation.version,
                doc_id           = result.citation.doc_id,
                section_heading  = result.citation.section_heading,
                rank             = result.rank,
                score            = result.score,
                rerank_score     = result.rerank_score,
                confidence       = confidence_from_score(eff_score),
            )
            candidates.append(chunk)

        # Cap at max_chunks
        candidates = candidates[: config.max_chunks]

        # Apply character budget — keep whole chunks; truncate the last one to fit
        included: list[ContextChunk] = []
        chars_used = 0

        for chunk in candidates:
            block_len = len(chunk.to_prompt_block(config.include_metadata))
            if chars_used + block_len <= config.max_context_chars:
                included.append(chunk)
                chars_used += block_len
            else:
                remaining = config.max_context_chars - chars_used
                # Minimum useful block: header (≈120 chars) + some content (≥100 chars)
                if remaining >= 220:
                    header_overhead = block_len - len(chunk.content)
                    content_budget  = remaining - header_overhead
                    if content_budget > 80:
                        truncated_content = chunk.content[:content_budget] + "…"
                        truncated = chunk.model_copy(update={"content": truncated_content})
                        included.append(truncated)
                break  # budget exhausted

        dropped = total_input - len(included)
        return included, dropped

    # ------------------------------------------------------------------
    # Step 2: Conflict detection
    # ------------------------------------------------------------------

    def _detect_conflicts(
        self, chunks: list[ContextChunk]
    ) -> list[ConflictGroup]:
        """
        Detects version conflicts: same (display_name, department, category)
        appearing with different version strings in the same result set.

        Returns a list of ConflictGroup objects (one per conflicting document name).
        Uses display_citation prefix (display_name) as the grouping key because
        that is the human-readable document identifier.
        """
        from collections import defaultdict

        # Key: (display_name, department, category)
        groups: dict[tuple, list[ContextChunk]] = defaultdict(list)
        for chunk in chunks:
            key = (chunk.display_citation.split(" (v")[0].strip(),
                   chunk.department, chunk.category)
            groups[key].append(chunk)

        conflicts: list[ConflictGroup] = []
        for (display_name, dept, cat), group_chunks in groups.items():
            versions = list({c.version for c in group_chunks})
            if len(versions) <= 1:
                continue

            # Sort versions newest-first
            versions_sorted = sorted(versions, key=_version_key, reverse=True)
            latest = versions_sorted[0]

            source_nums = sorted(c.source_number for c in group_chunks)
            conflicts.append(ConflictGroup(
                display_name   = display_name,
                department     = dept,
                category       = cat,
                versions       = versions_sorted,
                source_numbers = source_nums,
                latest_version = latest,
            ))

        return conflicts

    # ------------------------------------------------------------------
    # Step 3: Format context block
    # ------------------------------------------------------------------

    def _format_context_block(
        self,
        chunks: list[ContextChunk],
        include_metadata: bool,
    ) -> str:
        """
        Assembles all ContextChunk blocks into a single string separated by
        blank lines. Returns an empty-context notice when no chunks are available.
        """
        if not chunks:
            return "(No relevant context was retrieved.)"

        header = f"RETRIEVED CONTEXT ({len(chunks)} source{'s' if len(chunks) != 1 else ''}):"
        separator = "=" * 60
        blocks = [block.to_prompt_block(include_metadata) for block in chunks]
        return f"{header}\n{separator}\n\n" + f"\n\n{'-'*40}\n\n".join(blocks) + f"\n\n{separator}"

    # ------------------------------------------------------------------
    # Step 4: Render system prompt
    # ------------------------------------------------------------------

    def _render_system_prompt(
        self,
        template: PromptTemplate,
        conflicts: list[ConflictGroup],
    ) -> str:
        """
        Returns the base system prompt for the chosen template, with any
        conflict warnings appended as an additional section.
        """
        base = _SYSTEM_PROMPTS[template]
        if not conflicts:
            return base

        warnings = "\n".join(f"  • {cg.to_warning()}" for cg in conflicts)
        conflict_block = (
            "\n\nCONFLICT NOTICES — apply rules 3 and 4 to these sources:\n"
            + warnings
        )
        return base + conflict_block

    # ------------------------------------------------------------------
    # Step 5: Build user message
    # ------------------------------------------------------------------

    def _build_user_message(self, context_block: str, question: str) -> str:
        """
        Constructs the user turn: context block followed by the question.

        The LLM sees context and question in the same turn so it can
        reference them jointly. The assistant turn is left empty for the
        model to complete.
        """
        return (
            f"{context_block}\n\n"
            f"{'=' * 60}\n"
            f"QUESTION: {question}\n\n"
            f"Answer using only the sources above. Cite each source inline as [SOURCE N]."
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lightweight — no model load)
# ---------------------------------------------------------------------------

_builder_instance: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Returns the process-level PromptBuilder singleton."""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = PromptBuilder()
    return _builder_instance


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_prompt(
    question: str,
    results:  "list[RetrievalResult]",
    config:   Optional[PromptConfig] = None,
) -> BuiltPrompt:
    """
    Module-level shortcut. Equivalent to get_prompt_builder().build(...).

    Example
    -------
        from rag.prompt_builder import build_prompt
        prompt = build_prompt("What is the fee payment deadline?", retrieval_response.results)
        # prompt.messages → pass to Ollama Phase 9
    """
    return get_prompt_builder().build(question, results, config)
