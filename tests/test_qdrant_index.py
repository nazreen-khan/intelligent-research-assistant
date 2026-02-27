# File: tests/test_qdrant_index.py
"""
Day 5 — Integration tests for QdrantIndex.

Uses Qdrant in embedded mode with a temp directory so tests are:
  - Self-contained (no external services needed)
  - Isolated (each test gets a fresh collection)
  - Fast (in-memory embedded Qdrant)
"""

from __future__ import annotations

import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from ira.retrieval.qdrant_index import (
    PayloadField,
    QdrantIndex,
    SearchResult,
    _content_sha,
    reset_index,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

DIM = 8   # tiny vectors for fast tests


@pytest.fixture()
def tmp_qdrant(tmp_path: Path) -> Generator[QdrantIndex, None, None]:
    """Fresh QdrantIndex with temp storage and small dim for each test."""
    idx = QdrantIndex(
        mode="embedded",
        path=str(tmp_path / "qdrant"),
        collection_name="test_chunks",
        vector_dim=DIM,
        hnsw_m=4,
        hnsw_ef_construct=16,
    )
    idx.ensure_collection()
    yield idx
    # Cleanup: client will release files when garbage collected


def _make_chunk(text: str = "hello world", doc_id: str = "doc1") -> dict:
    """Create a minimal chunk dict matching ChildChunk schema."""
    return {
        "chunk_id": str(uuid.uuid4()),
        "parent_id": str(uuid.uuid4()),
        "doc_id": doc_id,
        "title": "Test Document",
        "url": "https://example.com/doc",
        "section": "Introduction",
        "text": text,
        "token_count": len(text.split()),
        "chunk_index": 0,
        "parent_chunk_index": 0,
        "is_table": False,
        "is_code": False,
    }


def _random_vec(dim: int = DIM) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_vectors(n: int) -> np.ndarray:
    return np.array([_random_vec() for _ in range(n)], dtype=np.float32)


# ── Collection management ─────────────────────────────────────────────────────

class TestCollectionManagement:
    def test_ensure_collection_creates_collection(self, tmp_qdrant: QdrantIndex):
        info = tmp_qdrant.collection_info()
        assert info["name"] == "test_chunks"

    def test_ensure_collection_is_idempotent(self, tmp_qdrant: QdrantIndex):
        # Calling ensure_collection twice should not raise
        created = tmp_qdrant.ensure_collection()
        assert created is False  # already exists

    def test_collection_info_shows_zero_points_initially(self, tmp_qdrant: QdrantIndex):
        info = tmp_qdrant.collection_info()
        assert info.get("points_count", 0) == 0

    def test_delete_collection(self, tmp_qdrant: QdrantIndex):
        tmp_qdrant.delete_collection()
        info = tmp_qdrant.collection_info()
        assert "error" in info


# ── Upsert ────────────────────────────────────────────────────────────────────

class TestUpsert:
    def test_upsert_single_chunk(self, tmp_qdrant: QdrantIndex):
        chunk = _make_chunk("FlashAttention reduces HBM access")
        vecs = _make_vectors(1)
        counts = tmp_qdrant.upsert_chunks([chunk], vecs)
        assert counts["upserted"] == 1
        assert counts["skipped"] == 0
        assert counts["total"] == 1

    def test_upsert_multiple_chunks(self, tmp_qdrant: QdrantIndex):
        chunks = [_make_chunk(f"chunk text {i}") for i in range(5)]
        vecs = _make_vectors(5)
        counts = tmp_qdrant.upsert_chunks(chunks, vecs)
        assert counts["upserted"] == 5
        assert counts["total"] == 5

    def test_upsert_empty_list(self, tmp_qdrant: QdrantIndex):
        counts = tmp_qdrant.upsert_chunks([], np.zeros((0, DIM), dtype=np.float32))
        assert counts == {"upserted": 0, "skipped": 0, "total": 0}

    def test_points_appear_in_collection_after_upsert(self, tmp_qdrant: QdrantIndex):
        chunks = [_make_chunk(f"text {i}") for i in range(3)]
        vecs = _make_vectors(3)
        tmp_qdrant.upsert_chunks(chunks, vecs)
        info = tmp_qdrant.collection_info()
        assert info.get("points_count", 0) == 3

    def test_payload_stored_correctly(self, tmp_qdrant: QdrantIndex):
        chunk = _make_chunk("KV cache paging strategy")
        chunk["doc_id"] = "arxiv_test"
        chunk["section"] = "2.1 Memory Management"
        vecs = _make_vectors(1)
        tmp_qdrant.upsert_chunks([chunk], vecs)

        # Retrieve the point and check payload
        results = tmp_qdrant._get_client().retrieve(
            collection_name="test_chunks",
            ids=[chunk["chunk_id"]],
            with_payload=True,
            with_vectors=False,
        )
        assert len(results) == 1
        payload = results[0].payload
        assert payload[PayloadField.DOC_ID] == "arxiv_test"
        assert payload[PayloadField.SECTION] == "2.1 Memory Management"
        assert payload[PayloadField.TEXT] == "KV cache paging strategy"
        assert PayloadField.CONTENT_SHA in payload


# ── Deduplication ─────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_second_upsert_of_same_chunk_is_skipped(self, tmp_qdrant: QdrantIndex):
        chunk = _make_chunk("speculative decoding draft model")
        vecs = _make_vectors(1)

        counts1 = tmp_qdrant.upsert_chunks([chunk], vecs)
        assert counts1["upserted"] == 1

        # Same chunk_id + same text → should skip
        counts2 = tmp_qdrant.upsert_chunks([chunk], vecs)
        assert counts2["skipped"] == 1
        assert counts2["upserted"] == 0

    def test_changed_text_forces_reupsert(self, tmp_qdrant: QdrantIndex):
        """Same chunk_id but different text → must re-upsert."""
        chunk_id = str(uuid.uuid4())
        chunk_v1 = {**_make_chunk("original text"), "chunk_id": chunk_id}
        chunk_v2 = {**_make_chunk("updated text"), "chunk_id": chunk_id}
        vecs = _make_vectors(1)

        tmp_qdrant.upsert_chunks([chunk_v1], vecs)
        counts = tmp_qdrant.upsert_chunks([chunk_v2], vecs)

        assert counts["upserted"] == 1   # re-upserted because SHA changed
        assert counts["skipped"] == 0

    def test_mixed_new_and_existing_chunks(self, tmp_qdrant: QdrantIndex):
        chunks_first = [_make_chunk(f"text {i}") for i in range(3)]
        vecs = _make_vectors(3)
        tmp_qdrant.upsert_chunks(chunks_first, vecs)

        # 3 new + 3 existing
        chunks_second = chunks_first + [_make_chunk(f"new text {i}") for i in range(3)]
        vecs_second = _make_vectors(6)
        counts = tmp_qdrant.upsert_chunks(chunks_second, vecs_second)

        assert counts["skipped"] == 3
        assert counts["upserted"] == 3
        assert counts["total"] == 6


# ── Search ────────────────────────────────────────────────────────────────────

class TestSearch:
    def _setup_chunks(self, idx: QdrantIndex, n: int = 5) -> list[dict]:
        chunks = [_make_chunk(f"document chunk number {i}") for i in range(n)]
        vecs = _make_vectors(n)
        idx.upsert_chunks(chunks, vecs)
        return chunks, vecs

    def test_search_returns_top_k_results(self, tmp_qdrant: QdrantIndex):
        chunks, vecs = self._setup_chunks(tmp_qdrant, n=5)
        query_vec = _random_vec()
        results = tmp_qdrant.search(query_vec, top_k=3)
        assert len(results) == 3

    def test_search_results_are_search_result_objects(self, tmp_qdrant: QdrantIndex):
        self._setup_chunks(tmp_qdrant)
        results = tmp_qdrant.search(_random_vec(), top_k=1)
        assert isinstance(results[0], SearchResult)

    def test_search_results_have_valid_fields(self, tmp_qdrant: QdrantIndex):
        self._setup_chunks(tmp_qdrant)
        results = tmp_qdrant.search(_random_vec(), top_k=2)
        for r in results:
            assert r.chunk_id != ""
            assert r.doc_id == "doc1"
            assert isinstance(r.score, float)
            assert r.text != ""

    def test_search_returns_fewer_when_collection_smaller(self, tmp_qdrant: QdrantIndex):
        """Asking for top_k=10 when only 3 chunks exist should return 3."""
        self._setup_chunks(tmp_qdrant, n=3)
        results = tmp_qdrant.search(_random_vec(), top_k=10)
        assert len(results) == 3

    def test_search_with_exact_query_vector_scores_highest(self, tmp_qdrant: QdrantIndex):
        """
        If we search with a vector we just indexed, it should be the top result.
        (Works because L2-normalised cosine similarity: same vector = score ~1.0)
        """
        chunk = _make_chunk("FlashAttention IO-awareness paper")
        vec = _random_vec().reshape(1, DIM)
        tmp_qdrant.upsert_chunks([chunk], vec)

        results = tmp_qdrant.search(vec[0], top_k=1)
        assert len(results) == 1
        assert results[0].chunk_id == chunk["chunk_id"]
        # cosine similarity of a vector with itself = 1.0
        assert abs(results[0].score - 1.0) < 1e-3

    def test_filter_by_doc_id(self, tmp_qdrant: QdrantIndex):
        chunks_a = [_make_chunk(f"doc A text {i}", doc_id="doc_A") for i in range(3)]
        chunks_b = [_make_chunk(f"doc B text {i}", doc_id="doc_B") for i in range(3)]
        vecs = _make_vectors(6)
        tmp_qdrant.upsert_chunks(chunks_a + chunks_b, vecs)

        results = tmp_qdrant.search(_random_vec(), top_k=10, filter_doc_ids=["doc_A"])
        assert all(r.doc_id == "doc_A" for r in results)
        assert len(results) == 3

    def test_exclude_code_chunks(self, tmp_qdrant: QdrantIndex):
        code_chunk = {**_make_chunk("def main(): pass"), "is_code": True}
        prose_chunk = _make_chunk("This is prose text")
        vecs = _make_vectors(2)
        tmp_qdrant.upsert_chunks([code_chunk, prose_chunk], vecs)

        results = tmp_qdrant.search(_random_vec(), top_k=10, exclude_code=True)
        assert all(not r.is_code for r in results)

    def test_search_empty_collection_returns_empty(self, tmp_qdrant: QdrantIndex):
        results = tmp_qdrant.search(_random_vec(), top_k=5)
        assert results == []


# ── Content SHA helper ─────────────────────────────────────────────────────────

class TestContentSha:
    def test_sha_is_deterministic(self):
        assert _content_sha("hello") == _content_sha("hello")

    def test_sha_differs_for_different_text(self):
        assert _content_sha("hello") != _content_sha("world")

    def test_sha_is_hex_string(self):
        sha = _content_sha("test")
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)


# ── Index runner integration ───────────────────────────────────────────────────

class TestIndexRunnerIntegration:
    """
    Tests that combine the index_runner with QdrantIndex and a mock embedder.
    These simulate the full build pipeline without network or real model.
    """

    def _make_mock_embedder(self, dim: int = DIM):
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.embedding_dim = dim
        mock.batch_size = 8

        def fake_embed_passages(texts):
            n = len(texts)
            vecs = np.random.randn(n, dim).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        mock.embed_passages.side_effect = fake_embed_passages
        return mock

    def test_build_index_from_chunks_dir(self, tmp_path: Path):
        """Build index from a temp chunks directory."""
        import json

        # Create fake chunks dir
        chunks_root = tmp_path / "chunks"
        doc_dir = chunks_root / "test_doc_001"
        doc_dir.mkdir(parents=True)

        chunks = [
            {
                "chunk_id": str(uuid.uuid4()),
                "parent_id": str(uuid.uuid4()),
                "doc_id": "test_doc_001",
                "title": "Test Paper",
                "url": "https://example.com",
                "section": "Introduction",
                "text": f"This is chunk number {i} about attention mechanisms.",
                "token_count": 10,
                "chunk_index": i,
                "parent_chunk_index": 0,
                "is_table": False,
                "is_code": False,
            }
            for i in range(5)
        ]
        (doc_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chunks), encoding="utf-8"
        )

        # Create index and embedder
        idx = QdrantIndex(
            mode="embedded",
            path=str(tmp_path / "qdrant"),
            collection_name="test_run",
            vector_dim=DIM,
        )
        embedder = self._make_mock_embedder()

        from ira.retrieval.index_runner import build_index
        result = build_index(
            chunks_root=chunks_root,
            embedder=embedder,
            index=idx,
        )

        assert len(result.docs) == 1
        assert result.docs[0].ok
        assert result.docs[0].chunks_read == 5
        assert result.total_upserted == 5

    def test_build_index_incremental(self, tmp_path: Path):
        """Second build should skip all already-indexed chunks."""
        import json

        chunks_root = tmp_path / "chunks"
        doc_dir = chunks_root / "doc_incr"
        doc_dir.mkdir(parents=True)

        chunks = [
            {
                "chunk_id": str(uuid.uuid4()),
                "parent_id": str(uuid.uuid4()),
                "doc_id": "doc_incr",
                "title": "Incremental Test",
                "url": None,
                "section": "Body",
                "text": f"stable text {i}",
                "token_count": 3,
                "chunk_index": i,
                "parent_chunk_index": 0,
                "is_table": False,
                "is_code": False,
            }
            for i in range(3)
        ]
        (doc_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chunks), encoding="utf-8"
        )

        idx = QdrantIndex(
            mode="embedded",
            path=str(tmp_path / "qdrant"),
            collection_name="test_incr",
            vector_dim=DIM,
        )
        embedder = self._make_mock_embedder()

        from ira.retrieval.index_runner import build_index

        # First build
        r1 = build_index(chunks_root=chunks_root, embedder=embedder, index=idx)
        assert r1.total_upserted == 3

        # Second build — all skipped
        r2 = build_index(chunks_root=chunks_root, embedder=embedder, index=idx)
        assert r2.total_upserted == 0
        assert r2.total_skipped == 3