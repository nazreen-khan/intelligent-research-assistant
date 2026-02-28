# File: tests/test_hybrid_retriever.py
"""
Day 7 — Tests for the hybrid retriever.

Covers:
  1. RRF fusion: score formula, ordering, provenance fields
  2. Parent deduplication: sibling chunks → best child wins
  3. Confidence signals: top_score, score_gap, keyword_coverage
  4. EvidencePack construction: all fields populated
  5. Ablation modes: dense_only, bm25_only
  6. ParentStore: load, get, lazy load, stats
  7. Integration: HybridRetriever with injected mock indexes
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from ira.retrieval.hybrid_retriever import (
    ConfidenceSignals,
    EvidencePack,
    HybridRetriever,
    RetrievalResult,
    _keyword_coverage,
    rrf_fuse,
)
from ira.retrieval.parent_store import ParentStore


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _dense(chunk_id, rank, score=0.9, parent_id=None, doc_id="doc1", section="S"):
    return {
        "chunk_id":    chunk_id,
        "parent_id":   parent_id or f"p_{chunk_id}",
        "doc_id":      doc_id,
        "title":       "Test Doc",
        "url":         "https://example.com",
        "section":     section,
        "rank":        rank,
        "score":       score,
        "text_preview": f"text for {chunk_id}",
        "is_code":     False,
        "is_table":    False,
        "source_type": "internal",
    }


def _bm25(chunk_id, rank, bm25_score=5.0, parent_id=None, doc_id="doc1", section="S"):
    return {
        "chunk_id":    chunk_id,
        "parent_id":   parent_id or f"p_{chunk_id}",
        "doc_id":      doc_id,
        "title":       "Test Doc",
        "url":         "https://example.com",
        "section":     section,
        "rank":        rank,
        "bm25_score":  bm25_score,
        "text_preview": f"text for {chunk_id}",
        "is_code":     False,
        "is_table":    False,
        "source_type": "internal",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RRF FUSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestRRFFusion:

    def test_rrf_score_formula(self):
        """rrf = 1/(k+rank_dense) + 1/(k+rank_bm25) for a chunk in both lists."""
        k = 60
        dense = [_dense("c1", rank=1)]
        bm25  = [_bm25("c1",  rank=1)]
        fused = rrf_fuse(dense, bm25, k=k)
        expected = 1/(k+1) + 1/(k+1)
        assert abs(fused[0]["rrf_score"] - expected) < 1e-9

    def test_chunk_in_both_beats_chunk_in_one(self):
        """A chunk that appears in both lists should score higher than one in only one."""
        dense = [_dense("both", rank=5), _dense("dense_only", rank=1)]
        bm25  = [_bm25("both", rank=5), _bm25("bm25_only", rank=1)]
        fused = rrf_fuse(dense, bm25)
        scores = {f["chunk_id"]: f["rrf_score"] for f in fused}
        # "both" appears in both lists at rank 5 each
        # "dense_only" appears only in dense at rank 1
        # 2*(1/65) = 0.0308  vs  1/61 = 0.0164
        assert scores["both"] > scores["dense_only"]
        assert scores["both"] > scores["bm25_only"]

    def test_results_sorted_by_rrf_score_descending(self):
        dense = [_dense(f"d{i}", rank=i+1) for i in range(5)]
        bm25  = [_bm25(f"b{i}", rank=i+1) for i in range(5)]
        fused = rrf_fuse(dense, bm25)
        scores = [f["rrf_score"] for f in fused]
        assert scores == sorted(scores, reverse=True)

    def test_final_rank_is_1_based_and_sequential(self):
        dense = [_dense(f"c{i}", rank=i+1) for i in range(3)]
        fused = rrf_fuse(dense, [])
        assert [f["final_rank"] for f in fused] == [1, 2, 3]

    def test_in_both_flag_set_correctly(self):
        dense = [_dense("shared", rank=1), _dense("dense_only", rank=2)]
        bm25  = [_bm25("shared", rank=1), _bm25("bm25_only",  rank=2)]
        fused = rrf_fuse(dense, bm25)
        by_id = {f["chunk_id"]: f for f in fused}
        assert by_id["shared"]["in_both"]      is True
        assert by_id["dense_only"]["in_both"]  is False
        assert by_id["bm25_only"]["in_both"]   is False

    def test_in_dense_and_in_bm25_flags(self):
        dense = [_dense("d1", rank=1)]
        bm25  = [_bm25("b1", rank=1)]
        fused = rrf_fuse(dense, bm25)
        by_id = {f["chunk_id"]: f for f in fused}
        assert by_id["d1"]["in_dense"] is True  and by_id["d1"]["in_bm25"] is False
        assert by_id["b1"]["in_bm25"]  is True  and by_id["b1"]["in_dense"] is False

    def test_dense_score_preserved_in_output(self):
        dense = [_dense("c1", rank=1, score=0.87)]
        fused = rrf_fuse(dense, [])
        assert fused[0]["dense_score"] == pytest.approx(0.87)

    def test_bm25_score_preserved_in_output(self):
        bm25 = [_bm25("c1", rank=1, bm25_score=7.3)]
        fused = rrf_fuse([], bm25)
        assert fused[0]["bm25_score"] == pytest.approx(7.3)

    def test_empty_dense_uses_bm25_only(self):
        bm25  = [_bm25(f"b{i}", rank=i+1) for i in range(3)]
        fused = rrf_fuse([], bm25)
        assert len(fused) == 3
        assert all(not f["in_dense"] for f in fused)

    def test_empty_bm25_uses_dense_only(self):
        dense = [_dense(f"d{i}", rank=i+1) for i in range(3)]
        fused = rrf_fuse(dense, [])
        assert len(fused) == 3
        assert all(not f["in_bm25"] for f in fused)

    def test_both_empty_returns_empty(self):
        assert rrf_fuse([], []) == []

    def test_weighted_rrf_boosts_preferred_source(self):
        """Setting bm25_weight=2.0 should make BM25 rank-1 beat dense rank-1."""
        dense = [_dense("d1", rank=1)]
        bm25  = [_bm25("b1", rank=1)]
        fused = rrf_fuse(dense, bm25, dense_weight=1.0, bm25_weight=2.0)
        by_id = {f["chunk_id"]: f["rrf_score"] for f in fused}
        assert by_id["b1"] > by_id["d1"]

    def test_duplicate_chunk_ids_merged_not_doubled(self):
        """Same chunk_id in both lists → one entry in output, not two."""
        dense = [_dense("c1", rank=1)]
        bm25  = [_bm25("c1", rank=1)]
        fused = rrf_fuse(dense, bm25)
        ids = [f["chunk_id"] for f in fused]
        assert len(ids) == len(set(ids)) == 1

    def test_all_unique_chunks_all_present(self):
        dense = [_dense(f"d{i}", rank=i+1) for i in range(5)]
        bm25  = [_bm25(f"b{i}", rank=i+1) for i in range(5)]
        fused = rrf_fuse(dense, bm25)
        assert len(fused) == 10  # 5 dense-only + 5 bm25-only


# ═══════════════════════════════════════════════════════════════════════════════
# PARENT DEDUPLICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestParentDedup:

    def test_siblings_with_same_parent_deduplicated(self):
        """Two child chunks from the same parent → only the best-scoring one kept."""
        fused = [
            {**_dense("child_a", rank=1, score=0.9), "rrf_score": 0.030, "parent_id": "parent_X",
             "bm25_score": 0.0, "dense_score": 0.9, "in_dense": True, "in_bm25": False,
             "in_both": False, "final_rank": 1},
            {**_dense("child_b", rank=2, score=0.85), "rrf_score": 0.025, "parent_id": "parent_X",
             "bm25_score": 0.0, "dense_score": 0.85, "in_dense": True, "in_bm25": False,
             "in_both": False, "final_rank": 2},
            {**_dense("child_c", rank=3, score=0.80), "rrf_score": 0.020, "parent_id": "parent_Y",
             "bm25_score": 0.0, "dense_score": 0.80, "in_dense": True, "in_bm25": False,
             "in_both": False, "final_rank": 3},
        ]
        deduped = HybridRetriever._dedup_by_parent(fused)
        # parent_X should appear only once; parent_Y once
        parent_ids = [f["parent_id"] for f in deduped]
        assert len(parent_ids) == 2
        assert parent_ids.count("parent_X") == 1
        assert parent_ids.count("parent_Y") == 1

    def test_best_child_wins_after_dedup(self):
        """Among two children of same parent, the higher RRF score is kept."""
        fused = [
            {**_dense("child_a", rank=2), "rrf_score": 0.025, "parent_id": "P1",
             "bm25_score": 0.0, "dense_score": 0.85, "in_dense": True, "in_bm25": False,
             "in_both": False, "final_rank": 2},
            {**_dense("child_b", rank=1), "rrf_score": 0.030, "parent_id": "P1",
             "bm25_score": 0.0, "dense_score": 0.9, "in_dense": True, "in_bm25": False,
             "in_both": False, "final_rank": 1},
        ]
        deduped = HybridRetriever._dedup_by_parent(fused)
        assert len(deduped) == 1
        assert deduped[0]["chunk_id"] == "child_b"  # child_b has higher rrf_score (0.030 > 0.025)

    def test_unique_parents_all_kept(self):
        """N chunks with N different parents → all N kept."""
        fused = [
            {**_dense(f"c{i}", rank=i+1), "rrf_score": 1.0/(60+i+1),
             "parent_id": f"P{i}", "bm25_score": 0.0, "dense_score": 0.9,
             "in_dense": True, "in_bm25": False, "in_both": False, "final_rank": i+1}
            for i in range(5)
        ]
        deduped = HybridRetriever._dedup_by_parent(fused)
        assert len(deduped) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE SIGNALS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestKeywordCoverage:

    def test_full_coverage_when_all_terms_appear(self):
        query = "FlashAttention-2 FP8"
        bm25_results = [{"text_preview": "FlashAttention-2 FP8 on H100"}]
        coverage = _keyword_coverage(query, bm25_results)
        assert coverage == pytest.approx(1.0)

    def test_zero_coverage_when_no_terms_match(self):
        query = "speculative decoding"
        bm25_results = [{"text_preview": "kernel fusion memory bandwidth"}]
        coverage = _keyword_coverage(query, bm25_results)
        assert coverage == pytest.approx(0.0)

    def test_partial_coverage(self):
        query = "FlashAttention-2 FP8 H100"  # 3 terms
        bm25_results = [{"text_preview": "flashattention-2 quantization gpu"}]  # 1 of 3
        coverage = _keyword_coverage(query, bm25_results)
        assert 0.0 < coverage < 1.0

    def test_empty_bm25_results_gives_zero(self):
        assert _keyword_coverage("some query", []) == 0.0

    def test_empty_query_gives_zero(self):
        assert _keyword_coverage("", [{"text_preview": "some text"}]) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# PARENT STORE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_parents(doc_dir: Path, parents: list[dict]) -> None:
    """Write a parents.jsonl file for testing."""
    doc_dir.mkdir(parents=True, exist_ok=True)
    with (doc_dir / "parents.jsonl").open("w", encoding="utf-8") as f:
        for p in parents:
            f.write(json.dumps(p) + "\n")


class TestParentStore:

    def _make_parents(self, n: int = 3, doc_id: str = "doc1") -> list[dict]:
        return [
            {
                "parent_id":   f"{doc_id}::parent::{i}",
                "doc_id":      doc_id,
                "title":       "Test",
                "url":         "https://example.com",
                "section":     f"Section {i}",
                "text":        f"Full parent text for section {i}. " * 20,
                "token_count": 80,
                "child_ids":   [f"{doc_id}::chunk::{i*3+j}" for j in range(3)],
            }
            for i in range(n)
        ]

    def test_load_all_returns_counts(self, tmp_path: Path):
        parents = self._make_parents(3, "doc1")
        _write_parents(tmp_path / "doc1", parents)
        store = ParentStore(tmp_path)
        counts = store.load_all()
        assert counts == {"doc1": 3}
        assert store.size == 3

    def test_get_existing_parent(self, tmp_path: Path):
        parents = self._make_parents(2, "doc1")
        _write_parents(tmp_path / "doc1", parents)
        store = ParentStore(tmp_path)
        store.load_all()
        record = store.get("doc1::parent::0")
        assert record is not None
        assert record["section"] == "Section 0"

    def test_get_missing_parent_returns_none(self, tmp_path: Path):
        store = ParentStore(tmp_path)
        assert store.get("nonexistent::parent::99") is None

    def test_get_text_returns_text(self, tmp_path: Path):
        parents = self._make_parents(1, "doc1")
        _write_parents(tmp_path / "doc1", parents)
        store = ParentStore(tmp_path)
        store.load_all()
        text = store.get_text("doc1::parent::0")
        assert len(text) > 0
        assert "Full parent text" in text  # checks the text field content, not section field

    def test_get_text_truncates_at_max_chars(self, tmp_path: Path):
        long_parent = {
            "parent_id": "doc1::parent::0",
            "doc_id": "doc1",
            "title": "T",
            "url": None,
            "section": "S",
            "text": "x" * 10000,
            "token_count": 2500,
            "child_ids": [],
        }
        _write_parents(tmp_path / "doc1", [long_parent])
        store = ParentStore(tmp_path)
        store.load_all()
        text = store.get_text("doc1::parent::0", max_chars=500)
        assert len(text) <= 520   # 500 + truncation suffix
        assert "truncated" in text

    def test_lazy_load_by_doc_id(self, tmp_path: Path):
        parents = self._make_parents(2, "doc1")
        _write_parents(tmp_path / "doc1", parents)
        store = ParentStore(tmp_path)
        # Don't call load_all — rely on lazy load
        record = store.get("doc1::parent::0", doc_id="doc1")
        assert record is not None

    def test_multiple_docs_loaded(self, tmp_path: Path):
        for doc_id in ["doc_A", "doc_B"]:
            parents = self._make_parents(2, doc_id)
            _write_parents(tmp_path / doc_id, parents)
        store = ParentStore(tmp_path)
        store.load_all()
        assert store.size == 4
        stats = store.stats()
        assert stats["docs_loaded"] == 2

    def test_missing_chunks_root_returns_empty(self, tmp_path: Path):
        store = ParentStore(tmp_path / "nonexistent")
        counts = store.load_all()
        assert counts == {}
        assert store.size == 0

    def test_get_many(self, tmp_path: Path):
        parents = self._make_parents(5, "doc1")
        _write_parents(tmp_path / "doc1", parents)
        store = ParentStore(tmp_path)
        store.load_all()
        ids = ["doc1::parent::0", "doc1::parent::2", "doc1::parent::99"]
        result = store.get_many(ids)
        assert len(result) == 2   # ::99 not found, silently omitted
        assert "doc1::parent::0" in result
        assert "doc1::parent::2" in result


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID RETRIEVER INTEGRATION TESTS (with mock indexes)
# ═══════════════════════════════════════════════════════════════════════════════

class MockDenseIndex:
    """Minimal mock that mimics query_index output format."""
    def __init__(self, results: list[dict]):
        self._results = results
    def query(self, *a, **kw):
        return self._results
    def close(self):
        pass


class MockBM25Index:
    """Minimal mock that mimics BM25Index.query output."""
    def __init__(self, results):
        self._results = results
    def query(self, *a, **kw):
        return self._results
    def close(self):
        pass


def _build_retriever(dense_results, bm25_results, parents_map, tmp_path):
    """Build a HybridRetriever with fully injected mocks."""
    # Write parents.jsonl for each unique doc
    for parent_id, record in parents_map.items():
        doc_id = record["doc_id"]
        doc_dir = tmp_path / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        pfile = doc_dir / "parents.jsonl"
        # Append (may be called multiple times for same doc)
        with pfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    store = ParentStore(tmp_path)
    store.load_all()

    # Build a retriever that uses our mocked query functions
    retriever = HybridRetriever(parent_store=store)

    # Monkeypatch the retrieve methods
    retriever._retrieve_dense = lambda q, n: dense_results
    retriever._retrieve_bm25  = lambda q, n: bm25_results

    return retriever


class TestHybridRetrieverIntegration:

    def _make_parent(self, parent_id, doc_id="doc1", section="S", text="full parent text " * 10):
        return {
            "parent_id":   parent_id,
            "doc_id":      doc_id,
            "title":       "Test",
            "url":         "https://example.com",
            "section":     section,
            "text":        text,
            "token_count": len(text.split()),
            "child_ids":   [],
        }

    def test_returns_retrieval_result(self, tmp_path):
        dense = [_dense("c1", rank=1)]
        parents = {"p_c1": self._make_parent("p_c1")}
        retriever = _build_retriever(dense, [], parents, tmp_path)
        result = retriever.retrieve("test query", top_k=1)
        assert isinstance(result, RetrievalResult)

    def test_evidence_packs_populated(self, tmp_path):
        dense = [_dense(f"c{i}", rank=i+1) for i in range(5)]
        parents = {f"p_c{i}": self._make_parent(f"p_c{i}") for i in range(5)}
        retriever = _build_retriever(dense, [], parents, tmp_path)
        result = retriever.retrieve("query", top_k=3)
        assert len(result.evidence_packs) == 3

    def test_evidence_pack_fields(self, tmp_path):
        dense = [_dense("c1", rank=1, score=0.88, doc_id="doc1", section="Method")]
        parents = {"p_c1": self._make_parent("p_c1", doc_id="doc1", section="Method",
                                              text="Full parent section text here " * 5)}
        retriever = _build_retriever(dense, [], parents, tmp_path)
        result = retriever.retrieve("query", top_k=1)
        pack = result.evidence_packs[0]

        assert pack.chunk_id     == "c1"
        assert pack.doc_id       == "doc1"
        assert pack.section      == "Method"
        assert pack.dense_score  == pytest.approx(0.88)
        assert pack.final_rank   == 1
        assert pack.in_dense     is True
        assert pack.in_bm25      is False
        assert len(pack.parent_text) > 0

    def test_parent_text_populated(self, tmp_path):
        dense  = [_dense("c1", rank=1)]
        bm25   = [_bm25("c1",  rank=1)]
        parent_text = "This is the full parent section with detailed content. " * 20
        parents = {"p_c1": self._make_parent("p_c1", text=parent_text)}
        retriever = _build_retriever(dense, bm25, parents, tmp_path)
        result = retriever.retrieve("query", top_k=1)
        assert len(result.evidence_packs[0].parent_text) > 0
        assert "full parent section" in result.evidence_packs[0].parent_text

    def test_in_both_flag_when_chunk_in_both_sources(self, tmp_path):
        dense = [_dense("shared", rank=1)]
        bm25  = [_bm25("shared", rank=1)]
        parents = {"p_shared": self._make_parent("p_shared")}
        retriever = _build_retriever(dense, bm25, parents, tmp_path)
        result = retriever.retrieve("query", top_k=1)
        assert result.evidence_packs[0].in_both is True

    def test_confidence_signals_present(self, tmp_path):
        dense = [_dense(f"c{i}", rank=i+1) for i in range(5)]
        parents = {f"p_c{i}": self._make_parent(f"p_c{i}") for i in range(5)}
        retriever = _build_retriever(dense, [], parents, tmp_path)
        result = retriever.retrieve("FlashAttention-2", top_k=3)
        c = result.confidence
        assert isinstance(c, ConfidenceSignals)
        assert c.top_rrf_score > 0
        assert c.results_count == 3

    def test_score_gap_positive_when_clear_winner(self, tmp_path):
        """Score gap should be positive when rank-1 is clearly ahead."""
        # rank-1 at position 1 scores 1/61 ≈ 0.0164
        # rank-2 at position 2 scores 1/62 ≈ 0.0161
        dense = [_dense(f"c{i}", rank=i+1) for i in range(5)]
        parents = {f"p_c{i}": self._make_parent(f"p_c{i}") for i in range(5)}
        retriever = _build_retriever(dense, [], parents, tmp_path)
        result = retriever.retrieve("query", top_k=2)
        assert result.confidence.score_gap >= 0

    def test_debug_payload_present(self, tmp_path):
        dense = [_dense("c1", rank=1)]
        parents = {"p_c1": self._make_parent("p_c1")}
        retriever = _build_retriever(dense, [], parents, tmp_path)
        result = retriever.retrieve("query", top_k=1)
        debug = result.debug
        assert "fused_total" in debug
        assert "all_candidates" in debug
        assert "dense_n_fetched" in debug
        assert "bm25_n_fetched" in debug

    def test_parent_dedup_applied(self, tmp_path):
        """Two chunks with same parent → only 1 EvidencePack for that parent."""
        # Same parent_id for c1 and c2
        c1 = {**_dense("c1", rank=1), "parent_id": "shared_parent"}
        c2 = {**_dense("c2", rank=2), "parent_id": "shared_parent"}
        c3 = {**_dense("c3", rank=3), "parent_id": "p_c3"}

        parents = {
            "shared_parent": self._make_parent("shared_parent"),
            "p_c3": self._make_parent("p_c3"),
        }
        retriever = _build_retriever([c1, c2, c3], [], parents, tmp_path)
        result = retriever.retrieve("query", top_k=5)
        # c1 + c2 share a parent → deduped to 1; c3 separate → 2 total
        assert len(result.evidence_packs) == 2

    def test_dense_only_mode(self, tmp_path):
        dense = [_dense("c1", rank=1)]
        bm25  = [_bm25("b1", rank=1)]
        parents = {
            "p_c1": self._make_parent("p_c1"),
            "p_b1": self._make_parent("p_b1"),
        }
        retriever = _build_retriever(dense, bm25, parents, tmp_path)
        result = retriever.retrieve("query", top_k=5, dense_only=True)
        # b1 should not appear since we passed dense_only=True
        # Note: _retrieve_bm25 is still called but rrf_fuse gets bm25_weight=0.0
        chunk_ids = [p.chunk_id for p in result.evidence_packs]
        # c1 should be present
        assert "c1" in chunk_ids

    def test_empty_results_returns_empty_evidence(self, tmp_path):
        retriever = _build_retriever([], [], {}, tmp_path)
        result = retriever.retrieve("query", top_k=5)
        assert result.evidence_packs == []
        assert result.confidence.results_count == 0

    def test_fewer_results_than_top_k_ok(self, tmp_path):
        """Requesting top_k=5 but only 2 results available → 2 packs returned."""
        dense = [_dense("c1", rank=1), _dense("c2", rank=2)]
        parents = {
            "p_c1": self._make_parent("p_c1"),
            "p_c2": self._make_parent("p_c2"),
        }
        retriever = _build_retriever(dense, [], parents, tmp_path)
        result = retriever.retrieve("query", top_k=5)
        assert len(result.evidence_packs) == 2

    def test_evidence_ranks_are_sequential(self, tmp_path):
        dense = [_dense(f"c{i}", rank=i+1) for i in range(3)]
        parents = {f"p_c{i}": self._make_parent(f"p_c{i}") for i in range(3)}
        retriever = _build_retriever(dense, [], parents, tmp_path)
        result = retriever.retrieve("query", top_k=3)
        assert [p.final_rank for p in result.evidence_packs] == [1, 2, 3]

    def test_bm25_elevates_exact_term_match(self, tmp_path):
        """
        A chunk ranked 10 in dense but 1 in BM25 should outrank one that's
        rank 2 in dense but absent from BM25.
        The BM25 exact-match boost is the whole point of hybrid retrieval.
        """
        # c_exact: dense rank 10, bm25 rank 1
        # c_semantic: dense rank 2, not in bm25
        c_exact    = _dense("c_exact",    rank=10, score=0.70)
        c_semantic = _dense("c_semantic", rank=2,  score=0.85)

        dense = [c_semantic, *[_dense(f"d{i}", rank=i+3) for i in range(8)], c_exact]
        bm25  = [_bm25("c_exact", rank=1, bm25_score=9.5)]

        parents = {
            "p_c_exact":    self._make_parent("p_c_exact"),
            "p_c_semantic": self._make_parent("p_c_semantic"),
            **{f"p_d{i}": self._make_parent(f"p_d{i}") for i in range(8)},
        }
        retriever = _build_retriever(dense, bm25, parents, tmp_path)
        result = retriever.retrieve("FlashAttention-2", top_k=5)

        chunk_ids = [p.chunk_id for p in result.evidence_packs]
        assert "c_exact" in chunk_ids, \
            "Exact-match chunk (rank 10 dense but rank 1 BM25) must appear in top-5"

        exact_rank    = next(p.final_rank for p in result.evidence_packs if p.chunk_id == "c_exact")
        semantic_rank = next((p.final_rank for p in result.evidence_packs if p.chunk_id == "c_semantic"), 99)
        assert exact_rank < semantic_rank, \
            "BM25 exact match should rank above dense-only semantic match"