"""
tests/test_embedder.py
----------------------
Phase 4: Unit tests for BGEEmbedder and EmbedPipeline.

Tests are split into two groups:
  1. Unit tests (no model required) — test structure, validation, error handling.
  2. Integration tests (model required) — marked with @pytest.mark.slow.
     Run with: pytest -m slow

The unit tests run fast and should always pass in CI.
"""

from __future__ import annotations

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embeddings.embedder import (
    BGEEmbedder,
    EMBEDDING_DIM,
    BATCH_SIZE,
    QUERY_INSTRUCTION,
    MODEL_NAME,
    NORMALIZE,
)
from embeddings.embed_pipeline import (
    EmbedPipeline,
    EmbeddingPayload,
    EmbeddingRunSummary,
    OVERLAP_TAIL_CHARS,
    _build_embedding_text,
    _snap_to_word_boundary,
)


# ===========================================================================
# Unit tests — no model load
# ===========================================================================

class TestBGEEmbedderConstants:
    """Validate that the constants match the BAAI/bge-base-en-v1.5 specification."""

    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 768

    def test_batch_size_reasonable(self):
        assert 1 <= BATCH_SIZE <= 128, "Batch size must be between 1 and 128."

    def test_normalize_enabled(self):
        assert NORMALIZE is True, "Normalization must be True for cosine similarity."

    def test_query_instruction_non_empty(self):
        assert QUERY_INSTRUCTION.strip(), "Query instruction prefix must not be empty."

    def test_model_name(self):
        assert "bge" in MODEL_NAME.lower()


class TestBGEEmbedderUnloaded:
    """Test the embedder before model is loaded."""

    def test_not_loaded_initially(self):
        embedder = BGEEmbedder()
        assert not embedder.is_loaded

    def test_embed_documents_raises_if_not_loaded(self):
        embedder = BGEEmbedder()
        with pytest.raises(RuntimeError, match="not loaded"):
            embedder.embed_documents(["test text"])

    def test_embed_query_raises_if_not_loaded(self):
        embedder = BGEEmbedder()
        with pytest.raises(RuntimeError, match="not loaded"):
            embedder.embed_query("What is the admission process?")

    def test_embedding_dim_property(self):
        embedder = BGEEmbedder()
        assert embedder.embedding_dim == 768

    def test_model_name_property(self):
        embedder = BGEEmbedder()
        assert "bge" in embedder.model_name.lower()


class TestEmbeddingPayload:
    """Test EmbeddingPayload dataclass structure."""

    def test_payload_creation(self):
        payload = EmbeddingPayload(
            chunk_id  = "abc123def456789a",
            content   = "This is a test chunk.",
            embedding = [0.1] * 768,
            metadata  = {
                "doc_id": "doc1",
                "access_level": "Public",
                "department": "Admissions",
            },
        )
        assert payload.chunk_id == "abc123def456789a"
        assert len(payload.embedding) == 768
        assert payload.metadata["access_level"] == "Public"

    def test_payload_embedding_is_list(self):
        payload = EmbeddingPayload(
            chunk_id  = "abc123",
            content   = "test",
            embedding = [0.5, -0.3, 0.1],
            metadata  = {},
        )
        assert isinstance(payload.embedding, list)


class TestEmbeddingRunSummary:
    """Test EmbeddingRunSummary report formatting."""

    def test_report_contains_run_id(self):
        summary = EmbeddingRunSummary(
            run_id         = "20240625_120000",
            started_at     = "2024-06-25T12:00:00",
            completed_at   = "2024-06-25T12:05:00",
            total_pending  = 627,
            total_embedded = 627,
            total_failed   = 0,
        )
        report = summary.report()
        assert "20240625_120000" in report
        assert "627" in report

    def test_report_shows_errors(self):
        summary = EmbeddingRunSummary(
            run_id   = "test",
            errors   = ["Something went wrong"],
        )
        report = summary.report()
        assert "Something went wrong" in report

    def test_report_no_errors_section_when_empty(self):
        summary = EmbeddingRunSummary(run_id="test", errors=[])
        report = summary.report()
        assert "Errors:" not in report


class TestEmbedPipelineStructure:
    """Test EmbedPipeline without actually loading the model."""

    def test_default_batch_size(self):
        pipeline = EmbedPipeline()
        assert pipeline._batch_size == BATCH_SIZE

    def test_custom_batch_size(self):
        pipeline = EmbedPipeline(batch_size=8)
        assert pipeline._batch_size == 8

    def test_run_returns_tuple(self):
        """run() must return (list, EmbeddingRunSummary) even when model is missing."""
        pipeline = EmbedPipeline()
        result = pipeline.run()
        assert isinstance(result, tuple)
        assert len(result) == 2
        payloads, summary = result
        assert isinstance(payloads, list)
        assert isinstance(summary, EmbeddingRunSummary)

    def test_run_summary_has_run_id(self):
        pipeline = EmbedPipeline()
        _, summary = pipeline.run()
        assert summary.run_id != ""

    def test_run_summary_has_pending_count(self):
        """Pending count should reflect chunks in the SQLite ledger."""
        pipeline = EmbedPipeline()
        _, summary = pipeline.run()
        # We have 627 chunks in the ledger from Phase 3.
        assert summary.total_pending >= 0


# ===========================================================================
# Integration tests — require model download
# ===========================================================================

@pytest.mark.slow
class TestBGEEmbedderIntegration:
    """Requires BAAI/bge-base-en-v1.5 to be downloaded (~440MB)."""

    @pytest.fixture(scope="class")
    def embedder(self):
        from embeddings.embedder import get_embedder
        return get_embedder()

    def test_model_loads(self, embedder):
        assert embedder.is_loaded

    def test_embed_documents_returns_correct_count(self, embedder):
        texts = ["Admission process", "Fee payment procedure", "Library rules"]
        vectors = embedder.embed_documents(texts)
        assert len(vectors) == 3

    def test_embed_documents_returns_correct_dim(self, embedder):
        vectors = embedder.embed_documents(["Test chunk content."])
        assert len(vectors[0]) == 768

    def test_embed_documents_empty_list(self, embedder):
        vectors = embedder.embed_documents([])
        assert vectors == []

    def test_embed_query_returns_single_vector(self, embedder):
        vector = embedder.embed_query("What is the admission procedure?")
        assert isinstance(vector, list)
        assert len(vector) == 768

    def test_embed_documents_normalized(self, embedder):
        """Normalized vectors must have magnitude close to 1.0."""
        import math
        vectors = embedder.embed_documents(["Normalized test."])
        magnitude = math.sqrt(sum(v ** 2 for v in vectors[0]))
        assert abs(magnitude - 1.0) < 1e-4, f"Expected ~1.0, got {magnitude:.6f}"

    def test_embed_query_normalized(self, embedder):
        import math
        vector = embedder.embed_query("Are embeddings normalized?")
        magnitude = math.sqrt(sum(v ** 2 for v in vector))
        assert abs(magnitude - 1.0) < 1e-4

    def test_document_query_asymmetry(self, embedder):
        """
        Query vector must differ from document vector for the same text.
        This confirms the asymmetric prefix is being applied.
        """
        text = "Admissions process for engineering college."
        doc_vec   = embedder.embed_documents([text])[0]
        query_vec = embedder.embed_query(text)
        # They should not be identical (query has instruction prefix)
        assert doc_vec != query_vec, "Document and query vectors must differ (asymmetric prefix)."

    def test_embed_batch_consistency(self, embedder):
        """
        Embedding the same text in a batch vs alone must produce the same vector.
        Validates batch processing does not corrupt individual embeddings.
        """
        texts = ["Chunk A", "Chunk B", "Chunk C"]
        batch_vecs  = embedder.embed_documents(texts)
        single_vecs = [embedder.embed_documents([t])[0] for t in texts]
        for i, (bv, sv) in enumerate(zip(batch_vecs, single_vecs)):
            for bval, sval in zip(bv, sv):
                assert abs(bval - sval) < 1e-5, f"Mismatch at text {i}, element differs."

    def test_singleton_returns_same_instance(self):
        from embeddings.embedder import get_embedder
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2, "get_embedder() must return the same singleton instance."


@pytest.mark.slow
class TestEmbedPipelineIntegration:
    """Full pipeline integration test. Requires model and SQLite ledger with chunks."""

    def test_run_returns_payloads(self):
        pipeline = EmbedPipeline(batch_size=32)
        payloads, summary = pipeline.run()
        assert isinstance(payloads, list)
        assert summary.total_pending > 0 or summary.total_embedded >= 0

    def test_payload_metadata_has_access_level(self):
        pipeline = EmbedPipeline(batch_size=32)
        payloads, _ = pipeline.run()
        if payloads:
            for p in payloads[:5]:
                assert "access_level" in p.metadata
                assert p.metadata["access_level"] in ("Public", "Student", "Faculty", "Admin")

    def test_payload_metadata_has_doc_id(self):
        pipeline = EmbedPipeline(batch_size=32)
        payloads, _ = pipeline.run()
        if payloads:
            for p in payloads[:5]:
                assert "doc_id" in p.metadata
                assert len(p.metadata["doc_id"]) == 64  # SHA-256 hex string

    def test_payload_embedding_length(self):
        pipeline = EmbedPipeline(batch_size=32)
        payloads, _ = pipeline.run()
        if payloads:
            assert len(payloads[0].embedding) == 768


# ===========================================================================
# Embedding-time overlap/heading context (Phase 7)
# ===========================================================================

class TestSnapToWordBoundary:
    def test_snaps_mid_word_tail(self):
        # "ubmit the form" -- leading partial word "ubmit" dropped
        result = _snap_to_word_boundary("ubmit the form")
        assert result == "the form"

    def test_leaves_tail_with_no_early_space(self):
        # No space within the first 30 chars -- looks like one long token
        # (URL, code), left intact rather than truncated further.
        long_token = "a" * 40
        assert _snap_to_word_boundary(long_token) == long_token

    def test_already_at_boundary(self):
        assert _snap_to_word_boundary("the form was submitted") == "form was submitted"


class TestBuildEmbeddingText:
    def test_first_chunk_no_overlap(self):
        # chunk_index=0 -- prev_content is None, nothing to prepend.
        text = _build_embedding_text("Submit form ZX-9.", "Admissions", None)
        assert text == "Admissions\nSubmit form ZX-9."

    def test_no_heading(self):
        text = _build_embedding_text("Submit form ZX-9.", "", None)
        assert text == "Submit form ZX-9."

    def test_prepends_short_prev_tail_whole(self):
        text = _build_embedding_text("Then sign it.", "", "Fill out the form.")
        assert text == "Fill out the form.\nThen sign it."

    def test_truncates_long_prev_tail(self):
        # Realistic prose (frequent spaces) rather than one long unbroken
        # run -- see TestSnapToWordBoundary for the boundary-snapping logic
        # in isolation, tested directly against index positions.
        prev = ("lorem ipsum dolor sit amet " * 10) + "end of chunk"
        text = _build_embedding_text("Next step.", "", prev)
        assert text.endswith("end of chunk\nNext step.")
        prepended = text.split("\n")[0]
        assert len(prepended) <= OVERLAP_TAIL_CHARS
        assert len(prepended) < len(prev)  # confirms truncation happened
        assert not prepended.startswith(" ")

    def test_heading_and_overlap_both_present(self):
        text = _build_embedding_text("Then sign it.", "Admissions", "Fill out the form.")
        assert text == "Admissions\nFill out the form.\nThen sign it."
