# File: tests/test_reranker.py
"""
Day 8 — Unit tests for CrossEncoderReranker.

Tests:
  1. test_rerank_reorders_correctly     — relevant doc rises to top
  2. test_rerank_cache_hit              — second call uses cache, skips model
  3. test_rerank_deterministic          — same input → same output every time
  4. test_rerank_keeps_k                — output length always == keep_k
  5. test_rerank_empty_input            — empty list returns empty list
  6. test_score_texts_returns_floats    — score_texts() returns correct shape
  7. test_cache_disabled                — cache_dir=None never writes to disk
  8. test_rerank_attaches_score         — reranker_score attribute is present
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ira.retrieval.hybrid_retriever import EvidencePack
from ira.retrieval.reranker import CrossEncoderReranker, _cache_key, _cache_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pack(chunk_id: str, child_text: str, parent_text: str = "") -> EvidencePack:
    """Build a minimal EvidencePack for testing."""
    return EvidencePack(
        chunk_id=chunk_id,
        parent_id=f"parent_{chunk_id}",
        doc_id="test_doc",
        title="Test Doc",
        url=None,
        section="Test Section",
        child_text=child_text,
        parent_text=parent_text or child_text,
        parent_token_count=len(child_text.split()),
        rrf_score=0.5,
        dense_score=0.5,
        bm25_score=0.5,
        dense_rank=1,
        bm25_rank=1,
        final_rank=1,
        is_code=False,
        is_table=False,
        source_type="internal",
        in_dense=True,
        in_bm25=True,
        in_both=True,
        reranker_score=None,
    )


def _make_reranker_with_mock_model(
    scores: list[float],
    cache_dir: Path | None = None,
) -> CrossEncoderReranker:
    """
    Build a CrossEncoderReranker whose model is pre-mocked.
    The mock model's predict() returns the given scores list.
    """
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-base",
        cache_dir=cache_dir,
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = scores
    reranker._model = mock_model   # inject mock — skips download
    return reranker


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRerankReordersCorrectly:
    """Relevant document should rise to rank 1 regardless of input order."""

    def test_relevant_doc_rises_to_top(self):
        # Arrange: 3 packs, second one is most relevant per mock scores
        packs = [
            _make_pack("chunk_a", "GPU memory bandwidth utilization"),
            _make_pack("chunk_b", "FlashAttention reduces memory reads via tiling"),
            _make_pack("chunk_c", "Python packaging with uv"),
        ]
        # Mock scores: chunk_b gets highest score
        reranker = _make_reranker_with_mock_model(scores=[0.2, 0.9, 0.1])

        result = reranker.rerank(
            query="How does FlashAttention save memory?",
            evidence_packs=packs,
            keep_k=3,
        )

        assert result[0].chunk_id == "chunk_b", (
            "chunk_b had the highest mock score and should be ranked first"
        )
        assert result[1].chunk_id == "chunk_a"
        assert result[2].chunk_id == "chunk_c"

    def test_worst_doc_goes_to_bottom(self):
        packs = [
            _make_pack("chunk_x", "speculative decoding draft model"),
            _make_pack("chunk_y", "completely unrelated content about cooking"),
        ]
        reranker = _make_reranker_with_mock_model(scores=[0.8, 0.05])

        result = reranker.rerank(
            query="speculative decoding latency",
            evidence_packs=packs,
            keep_k=2,
        )

        assert result[0].chunk_id == "chunk_x"
        assert result[1].chunk_id == "chunk_y"


class TestRerankCacheHit:
    """Second call with same query+chunks must use cache and skip the model."""

    def test_model_called_once_on_second_identical_query(self, tmp_path):
        packs = [
            _make_pack("chunk_1", "KV cache memory layout"),
            _make_pack("chunk_2", "LoRA fine-tuning adapters"),
        ]
        reranker = _make_reranker_with_mock_model(
            scores=[0.7, 0.3],
            cache_dir=tmp_path,
        )

        query = "KV cache compression techniques"

        # First call — model should be invoked
        reranker.rerank(query=query, evidence_packs=packs, keep_k=2)
        assert reranker._model.predict.call_count == 1

        # Second call — should be a cache hit, model NOT invoked again
        reranker.rerank(query=query, evidence_packs=packs, keep_k=2)
        assert reranker._model.predict.call_count == 1, (
            "Model was called again on second identical query — cache miss"
        )

    def test_cache_file_written_after_first_call(self, tmp_path):
        packs = [_make_pack("chunk_a", "quantization INT4")]
        reranker = _make_reranker_with_mock_model(scores=[0.6], cache_dir=tmp_path)

        reranker.rerank(
            query="INT4 quantization accuracy",
            evidence_packs=packs,
            keep_k=1,
        )

        key = _cache_key("INT4 quantization accuracy", ["chunk_a"])
        cache_file = _cache_path(tmp_path, key)
        assert cache_file.exists(), "Cache file should exist after first rerank call"

        content = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "chunk_a" in content, "Cache should contain the chunk_id as a key"

    def test_different_query_is_cache_miss(self, tmp_path):
        packs = [_make_pack("chunk_a", "paged attention memory")]
        reranker = _make_reranker_with_mock_model(scores=[0.5], cache_dir=tmp_path)

        reranker.rerank(query="paged attention", evidence_packs=packs, keep_k=1)
        reranker.rerank(query="different query", evidence_packs=packs, keep_k=1)

        # Different query → different cache key → model called twice
        assert reranker._model.predict.call_count == 2


class TestRerankDeterministic:
    """Same input must always produce the same output."""

    def test_same_scores_same_order(self, tmp_path):
        packs = [
            _make_pack("c1", "flash attention tiling algorithm"),
            _make_pack("c2", "weight quantization calibration"),
            _make_pack("c3", "speculative decoding acceptance rate"),
        ]
        reranker = _make_reranker_with_mock_model(
            scores=[0.3, 0.8, 0.5],
            cache_dir=tmp_path,
        )
        query = "quantization calibration methods"

        result_1 = reranker.rerank(query=query, evidence_packs=packs, keep_k=3)
        # Reset mock so second call hits cache (not model)
        result_2 = reranker.rerank(query=query, evidence_packs=packs, keep_k=3)

        ids_1 = [p.chunk_id for p in result_1]
        ids_2 = [p.chunk_id for p in result_2]
        assert ids_1 == ids_2, f"Non-deterministic: {ids_1} != {ids_2}"


class TestRerankKeepsK:
    """Output length must always equal min(keep_k, len(input))."""

    @pytest.mark.parametrize("n_packs,keep_k,expected_len", [
        (10, 5, 5),   # standard case
        (3,  5, 3),   # fewer packs than keep_k
        (5,  5, 5),   # exactly keep_k
        (1,  5, 1),   # single pack
        (20, 5, 5),   # large pool
    ])
    def test_output_length(self, n_packs, keep_k, expected_len):
        packs = [_make_pack(f"c{i}", f"text {i}") for i in range(n_packs)]
        scores = [float(i) / n_packs for i in range(n_packs)]
        reranker = _make_reranker_with_mock_model(scores=scores)

        result = reranker.rerank(
            query="test query",
            evidence_packs=packs,
            keep_k=keep_k,
        )
        assert len(result) == expected_len


class TestRerankEmptyInput:
    """Empty input must return empty output without errors."""

    def test_empty_packs_returns_empty(self):
        reranker = _make_reranker_with_mock_model(scores=[])
        result = reranker.rerank(
            query="any query",
            evidence_packs=[],
            keep_k=5,
        )
        assert result == []
        reranker._model.predict.assert_not_called()


class TestScoreTexts:
    """score_texts() must return one float per input text."""

    def test_returns_correct_number_of_scores(self):
        reranker = _make_reranker_with_mock_model(scores=[0.9, 0.4, 0.1])
        texts = ["doc one", "doc two", "doc three"]

        scores = reranker.score_texts(query="test", texts=texts)

        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_empty_texts_returns_empty(self):
        reranker = _make_reranker_with_mock_model(scores=[])
        scores = reranker.score_texts(query="test", texts=[])
        assert scores == []


class TestCacheDisabled:
    """When cache_dir=None, no files are written anywhere."""

    def test_no_cache_written_when_disabled(self, tmp_path):
        packs = [_make_pack("chunk_z", "tensor parallelism sharding")]
        # cache_dir=None → caching disabled
        reranker = _make_reranker_with_mock_model(scores=[0.7], cache_dir=None)

        reranker.rerank(query="tensor parallelism", evidence_packs=packs, keep_k=1)

        # No files should have been written anywhere in tmp_path
        all_files = list(tmp_path.rglob("*"))
        assert all_files == [], f"Unexpected files written: {all_files}"

    def test_model_called_every_time_when_cache_disabled(self):
        packs = [_make_pack("chunk_z", "pipeline parallelism stages")]
        reranker = _make_reranker_with_mock_model(scores=[0.7], cache_dir=None)

        reranker.rerank(query="pipeline parallelism", evidence_packs=packs, keep_k=1)
        reranker.rerank(query="pipeline parallelism", evidence_packs=packs, keep_k=1)

        # No cache → model called both times
        assert reranker._model.predict.call_count == 2


class TestRerankerScoreAttached:
    """reranker_score must be present and correct on returned packs."""

    def test_reranker_score_attached(self):
        packs = [
            _make_pack("c1", "attention sink phenomenon"),
            _make_pack("c2", "grouped query attention"),
        ]
        reranker = _make_reranker_with_mock_model(scores=[0.3, 0.85])

        result = reranker.rerank(
            query="attention mechanism variants",
            evidence_packs=packs,
            keep_k=2,
        )

        for pack in result:
            assert hasattr(pack, "reranker_score"), (
                f"Pack {pack.chunk_id} missing reranker_score"
            )
            assert isinstance(pack.reranker_score, float)

    def test_top_ranked_has_highest_score(self):
        packs = [
            _make_pack("low",  "unrelated content"),
            _make_pack("high", "quantization aware training calibration dataset"),
        ]
        reranker = _make_reranker_with_mock_model(scores=[0.1, 0.95])

        result = reranker.rerank(
            query="quantization calibration",
            evidence_packs=packs,
            keep_k=2,
        )

        assert result[0].reranker_score > result[1].reranker_score