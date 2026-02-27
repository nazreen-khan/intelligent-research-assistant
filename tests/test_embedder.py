# File: tests/test_embedder.py
"""
Day 5 — Tests for the Embedder class.

These tests use a tiny mock model to avoid downloading real weights in CI.
The mock produces deterministic vectors by hashing input text.
Only test_embedder_loads_real_model requires network access and is marked slow.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Sequence
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ira.retrieval.embedder import (
    Embedder,
    _cache_path,
    _model_slug,
    _sha256,
    reset_embedder,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_mock_model(dim: int = 384) -> MagicMock:
    """
    Create a mock SentenceTransformer that produces deterministic vectors
    by hashing each input text. No network, no disk, no GPU needed.
    """
    mock = MagicMock()
    mock.max_seq_length = 512
    mock.get_sentence_embedding_dimension.return_value = dim

    def fake_encode(texts, **kwargs):
        vecs = []
        for t in texts:
            sha = hashlib.sha256(t.encode()).digest()
            # Repeat sha bytes to fill dim floats, then normalize
            raw = np.frombuffer((sha * ((dim * 4 // len(sha)) + 1))[:dim * 4], dtype=np.float32).copy()
            norm = np.linalg.norm(raw)
            if norm > 0:
                raw /= norm
            vecs.append(raw)
        return np.array(vecs, dtype=np.float32)

    mock.encode.side_effect = fake_encode
    return mock


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture()
def embedder_with_cache(tmp_cache: Path) -> Embedder:
    """Embedder wired to a temp cache dir, with mocked model."""
    e = Embedder(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
        batch_size=8,
        cache_dir=tmp_cache,
    )
    e._model = _make_mock_model(dim=384)
    return e


@pytest.fixture()
def embedder_no_cache() -> Embedder:
    """Embedder without cache, with mocked model."""
    e = Embedder(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
        batch_size=8,
        cache_dir=None,
    )
    e._model = _make_mock_model(dim=384)
    return e


# ── Helper utilities ───────────────────────────────────────────────────────────

class TestHelpers:
    def test_sha256_is_deterministic(self):
        assert _sha256("hello") == _sha256("hello")

    def test_sha256_differs_for_different_text(self):
        assert _sha256("hello") != _sha256("world")

    def test_model_slug_replaces_slash(self):
        assert _model_slug("BAAI/bge-small-en-v1.5") == "BAAI_bge-small-en-v1.5"

    def test_model_slug_safe_for_filesystem(self):
        slug = _model_slug("BAAI/bge-small-en-v1.5")
        assert "/" not in slug
        assert "\\" not in slug

    def test_cache_path_structure(self, tmp_path: Path):
        p = _cache_path(tmp_path, "BAAI_bge-small", "abcdef1234567890")
        # Should have 2-char sub-directory
        assert p.parent.name == "ab"
        assert p.name.endswith(".npy")


# ── Embedding shape & values ───────────────────────────────────────────────────

class TestEmbeddingShape:
    def test_embed_passages_single_text(self, embedder_no_cache: Embedder):
        vecs = embedder_no_cache.embed_passages(["hello world"])
        assert vecs.shape == (1, 384)
        assert vecs.dtype == np.float32

    def test_embed_passages_multiple_texts(self, embedder_no_cache: Embedder):
        texts = ["hello", "world", "foo bar baz"]
        vecs = embedder_no_cache.embed_passages(texts)
        assert vecs.shape == (3, 384)

    def test_embed_passages_empty_list(self, embedder_no_cache: Embedder):
        vecs = embedder_no_cache.embed_passages([])
        assert vecs.shape == (0, 384)

    def test_embed_query_returns_1d(self, embedder_no_cache: Embedder):
        vec = embedder_no_cache.embed_query("What is FlashAttention?")
        assert vec.shape == (384,)
        assert vec.dtype == np.float32

    def test_embed_queries_batch(self, embedder_no_cache: Embedder):
        queries = ["query 1", "query 2"]
        vecs = embedder_no_cache.embed_queries(queries)
        assert vecs.shape == (2, 384)

    def test_embedding_dim_property_known_model(self, embedder_no_cache: Embedder):
        # Should return 384 without loading the model for bge-small
        assert embedder_no_cache.embedding_dim == 384


# ── Determinism ───────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_text_same_vector(self, embedder_no_cache: Embedder):
        text = "FlashAttention reduces memory from O(N^2) to O(N)"
        v1 = embedder_no_cache.embed_passages([text])
        v2 = embedder_no_cache.embed_passages([text])
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self, embedder_no_cache: Embedder):
        v1 = embedder_no_cache.embed_passages(["attention is all you need"])
        v2 = embedder_no_cache.embed_passages(["quantization reduces memory"])
        assert not np.allclose(v1, v2)

    def test_query_differs_from_passage(self, embedder_no_cache: Embedder):
        """Query should produce different vector from passage (due to prefix)."""
        text = "FlashAttention IO-awareness"
        passage_vec = embedder_no_cache.embed_passages([text])[0]
        query_vec = embedder_no_cache.embed_query(text)
        # With real BGE model these would differ; with our hash mock they do too
        assert not np.array_equal(passage_vec, query_vec)


# ── Disk cache ────────────────────────────────────────────────────────────────

class TestDiskCache:
    def test_cache_miss_then_hit(self, embedder_with_cache: Embedder):
        text = "KV cache quantization"
        sha = _sha256(text)

        # First call: cache miss, calls model
        call_count_before = embedder_with_cache._model.encode.call_count
        vecs1 = embedder_with_cache.embed_passages([text])
        call_count_after = embedder_with_cache._model.encode.call_count
        assert call_count_after > call_count_before, "Model should have been called on cache miss"

        # Second call: cache hit, model NOT called again
        call_count_before = embedder_with_cache._model.encode.call_count
        vecs2 = embedder_with_cache.embed_passages([text])
        call_count_after = embedder_with_cache._model.encode.call_count
        assert call_count_after == call_count_before, "Model should NOT be called on cache hit"

        # Vectors must be equal
        np.testing.assert_array_almost_equal(vecs1, vecs2)

    def test_cache_file_created(self, embedder_with_cache: Embedder, tmp_cache: Path):
        text = "speculative decoding"
        embedder_with_cache.embed_passages([text])

        sha = _sha256(text)
        expected_path = _cache_path(tmp_cache, "BAAI_bge-small-en-v1.5", sha)
        assert expected_path.exists(), f"Cache file not found: {expected_path}"

    def test_cache_stats_returns_correct_count(self, embedder_with_cache: Embedder):
        texts = ["text one", "text two", "text three"]
        embedder_with_cache.embed_passages(texts)

        stats = embedder_with_cache.cache_stats()
        assert stats["enabled"] is True
        assert stats["files"] == 3

    def test_clear_cache_removes_files(self, embedder_with_cache: Embedder):
        embedder_with_cache.embed_passages(["hello", "world"])
        deleted = embedder_with_cache.clear_cache()
        assert deleted == 2
        stats = embedder_with_cache.cache_stats()
        assert stats["files"] == 0

    def test_no_cache_embedder_still_works(self, embedder_no_cache: Embedder):
        """Cache=None should not raise errors."""
        vecs = embedder_no_cache.embed_passages(["test"])
        assert vecs.shape == (1, 384)
        assert embedder_no_cache.cache_stats() == {"enabled": False}


# ── Batching ──────────────────────────────────────────────────────────────────

class TestBatching:
    def test_large_batch_split_correctly(self, embedder_no_cache: Embedder):
        """20 texts with batch_size=8 should call encode in ceil(20/8)=3 batches."""
        embedder_no_cache.batch_size = 8
        texts = [f"text number {i}" for i in range(20)]
        vecs = embedder_no_cache.embed_passages(texts)
        assert vecs.shape == (20, 384)
        # Model.encode should have been called 3 times (8 + 8 + 4)
        assert embedder_no_cache._model.encode.call_count == 3

    def test_exact_batch_size_boundary(self, embedder_no_cache: Embedder):
        """Exactly batch_size texts should call encode once."""
        embedder_no_cache.batch_size = 4
        embedder_no_cache._model.encode.reset_mock()
        texts = [f"text {i}" for i in range(4)]
        embedder_no_cache.embed_passages(texts)
        assert embedder_no_cache._model.encode.call_count == 1


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_reset_clears_singleton(self):
        from ira.retrieval.embedder import get_embedder
        reset_embedder()
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2  # same instance
        reset_embedder()

    def test_override_creates_new_instance(self, tmp_path: Path):
        from ira.retrieval.embedder import get_embedder
        reset_embedder()
        e_default = get_embedder()
        e_override = get_embedder(cache_dir=tmp_path)
        assert e_default is not e_override
        reset_embedder()


# ── Slow/real-model test (skipped by default) ─────────────────────────────────

@pytest.mark.slow
def test_embedder_loads_real_model():
    """
    Requires network access to download BAAI/bge-small-en-v1.5 (~130 MB).
    Run with: pytest -m slow tests/test_embedder.py::test_embedder_loads_real_model
    """
    with tempfile.TemporaryDirectory() as tmp:
        e = Embedder(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
            cache_dir=Path(tmp),
        )
        vec = e.embed_query("What is FlashAttention?")
        assert vec.shape == (384,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5, "Vector should be unit-normalised"