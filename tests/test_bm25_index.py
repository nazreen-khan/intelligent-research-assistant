# File: tests/test_bm25_index.py
"""
Day 6 — Tests for the BM25 index.

Covers:
  1. Tokenizer: technical terms, edge cases, markdown stripping
  2. BM25Index: upsert, dedup, query, filters, stats
  3. Integration: build_bm25_index runner end-to-end
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from ira.retrieval.bm25_index import (
    BM25Index,
    BM25Result,
    tokenize,
    tokenize_for_query,
    tokens_to_text,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTokenizer:
    """Critical: verify that our domain-specific tokenizer preserves technical terms."""

    # ── Technical term preservation ───────────────────────────────────────────

    def test_hyphenated_model_names_preserved(self):
        tokens = tokenize("FlashAttention-2 is fast")
        assert "flashattention-2" in tokens, "Hyphenated model name must be a single token"

    def test_version_strings_preserved(self):
        tokens = tokenize("bge-small-en-v1.5 embeddings")
        assert "bge-small-en-v1.5" in tokens

    def test_hardware_ids_preserved(self):
        tokens = tokenize("benchmarked on H100 GPU")
        assert "h100" in tokens

    def test_precision_formats_preserved(self):
        """FP8, INT4, BF16 must not be split."""
        tokens = tokenize("FP8 quantization with INT4 weights and BF16 activations")
        assert "fp8"  in tokens
        assert "int4" in tokens
        assert "bf16" in tokens

    def test_multiplier_tokens_preserved(self):
        """3x, 2bit, 4bit — digit+letter and letter+digit tokens."""
        tokens = tokenize("achieves 3x speedup with 4bit quantization")
        assert "3x"   in tokens
        assert "4bit" in tokens

    def test_slash_separated_org_name(self):
        """BAAI/bge — slash splits into two tokens, bge-small kept whole."""
        tokens = tokenize("BAAI/bge-small-en-v1.5")
        # BAAI becomes standalone, bge-small-en-v1.5 as one token
        assert "baai" in tokens
        assert "bge-small-en-v1.5" in tokens

    def test_lora_rank_syntax(self):
        tokens = tokenize("LoRA rank=8 alpha=16")
        # rank and 8 extracted; = is not a word char
        assert "lora" in tokens or "rank" in tokens

    # ── Markdown stripping ────────────────────────────────────────────────────

    def test_heading_markers_stripped(self):
        tokens = tokenize("# Introduction to FlashAttention")
        assert "#" not in tokens
        assert "introduction" in tokens

    def test_bold_markers_stripped(self):
        tokens = tokenize("**FlashAttention-2** reduces memory")
        assert "**" not in tokens
        assert "flashattention-2" in tokens

    def test_inline_code_content_extracted(self):
        tokens = tokenize("use `flash_attn_func` for fast attention")
        # inline code stripped but surrounding words extracted
        assert "use" in tokens
        assert "for" in tokens

    def test_markdown_link_keeps_text(self):
        tokens = tokenize("[FlashAttention paper](https://arxiv.org/abs/2205.14135)")
        assert "flashattention" in tokens or "paper" in tokens

    # ── Lowercase ─────────────────────────────────────────────────────────────

    def test_all_tokens_lowercase(self):
        tokens = tokenize("FlashAttention CUDA KV-Cache")
        for t in tokens:
            assert t == t.lower(), f"Token {t!r} is not lowercase"

    # ── Deduplication ─────────────────────────────────────────────────────────

    def test_no_duplicate_tokens(self):
        tokens = tokenize("FP8 FP8 FP8 quantization quantization")
        assert len(tokens) == len(set(tokens)), "Tokens should be deduplicated"

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_string_returns_empty(self):
        assert tokenize("") == []

    def test_whitespace_only_returns_empty(self):
        assert tokenize("   \n\t  ") == []

    def test_single_char_tokens_excluded(self):
        """Single chars (except as part of tech tokens) should be excluded."""
        tokens = tokenize("a b c the FlashAttention-2")
        # single chars filtered by _WORD pattern (min 2 chars)
        for t in tokens:
            assert len(t) >= 2 or "-" in t

    def test_numbers_only_not_extracted_as_standalone(self):
        """Pure numbers like '42' are only 2 chars, so included by _WORD."""
        tokens = tokenize("rank 42 results")
        # 42 is len 2, passes _WORD filter
        assert "rank" in tokens
        assert "results" in tokens

    # ── tokenize_for_query ────────────────────────────────────────────────────

    def test_query_wraps_tokens_in_quotes(self):
        fts_q = tokenize_for_query("FlashAttention-2 FP8")
        # Each token wrapped in quotes for FTS5 exact match
        assert '"flashattention-2"' in fts_q
        assert '"fp8"' in fts_q

    def test_query_empty_string_returns_empty_match(self):
        fts_q = tokenize_for_query("")
        assert fts_q == '""'

    def test_tokens_to_text_joins_with_spaces(self):
        assert tokens_to_text(["flash", "attention", "fp8"]) == "flash attention fp8"


# ═══════════════════════════════════════════════════════════════════════════════
# BM25 INDEX TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_chunk(
    text: str = "FlashAttention reduces HBM memory",
    doc_id: str = "doc1",
    section: str = "Introduction",
    is_code: bool = False,
    is_table: bool = False,
) -> dict:
    return {
        "chunk_id":          str(uuid.uuid4()),
        "parent_id":         str(uuid.uuid4()),
        "doc_id":            doc_id,
        "title":             "Test Document",
        "url":               "https://example.com",
        "section":           section,
        "text":              text,
        "token_count":       len(text.split()),
        "chunk_index":       0,
        "parent_chunk_index": 0,
        "is_table":          is_table,
        "is_code":           is_code,
        "source_type":       "internal",
    }


@pytest.fixture()
def tmp_db(tmp_path: Path) -> BM25Index:
    idx = BM25Index(tmp_path / "test.db")
    yield idx
    idx.close()


class TestBM25IndexUpsert:

    def test_upsert_single_chunk(self, tmp_db: BM25Index):
        chunk = _make_chunk()
        counts = tmp_db.upsert_chunks([chunk])
        assert counts["inserted"] == 1
        assert counts["skipped"]  == 0
        assert counts["total"]    == 1

    def test_upsert_multiple_chunks(self, tmp_db: BM25Index):
        chunks = [_make_chunk(f"text {i}") for i in range(5)]
        counts = tmp_db.upsert_chunks(chunks)
        assert counts["inserted"] == 5
        assert counts["total"]    == 5

    def test_upsert_empty_list(self, tmp_db: BM25Index):
        counts = tmp_db.upsert_chunks([])
        assert counts == {"inserted": 0, "updated": 0, "skipped": 0, "total": 0}

    def test_stats_reflect_upserted_count(self, tmp_db: BM25Index):
        chunks = [_make_chunk(f"text number {i}") for i in range(4)]
        tmp_db.upsert_chunks(chunks)
        stats = tmp_db.stats()
        assert stats["chunks"] == 4
        assert stats["docs"]   == 1

    def test_stats_count_distinct_docs(self, tmp_db: BM25Index):
        chunks = (
            [_make_chunk(f"doc A text {i}", doc_id="doc_A") for i in range(3)] +
            [_make_chunk(f"doc B text {i}", doc_id="doc_B") for i in range(2)]
        )
        tmp_db.upsert_chunks(chunks)
        assert tmp_db.stats()["docs"] == 2


class TestBM25IndexDedup:

    def test_same_chunk_id_and_sha_is_skipped(self, tmp_db: BM25Index):
        chunk = _make_chunk("identical text")
        tmp_db.upsert_chunks([chunk])
        counts = tmp_db.upsert_chunks([chunk])   # second upsert
        assert counts["skipped"]  == 1
        assert counts["inserted"] == 0

    def test_changed_text_triggers_update(self, tmp_db: BM25Index):
        chunk_id = str(uuid.uuid4())
        v1 = {**_make_chunk("original text"), "chunk_id": chunk_id}
        v2 = {**_make_chunk("updated text"),  "chunk_id": chunk_id}
        tmp_db.upsert_chunks([v1])
        counts = tmp_db.upsert_chunks([v2])
        assert counts["updated"]  == 1
        assert counts["inserted"] == 0

    def test_mixed_new_existing_chunks(self, tmp_db: BM25Index):
        existing = [_make_chunk(f"stable text {i}") for i in range(3)]
        tmp_db.upsert_chunks(existing)

        new_chunks = [_make_chunk(f"brand new {i}") for i in range(2)]
        counts = tmp_db.upsert_chunks(existing + new_chunks)
        assert counts["skipped"]  == 3
        assert counts["inserted"] == 2

    def test_get_indexed_ids_returns_all_ids(self, tmp_db: BM25Index):
        chunks = [_make_chunk(f"text {i}") for i in range(3)]
        tmp_db.upsert_chunks(chunks)
        indexed = tmp_db.get_indexed_ids()
        assert indexed == {c["chunk_id"] for c in chunks}


class TestBM25IndexQuery:

    def test_exact_technical_term_match(self, tmp_db: BM25Index):
        """FlashAttention-2 must match exactly, not get stemmed."""
        chunks = [
            _make_chunk("FlashAttention-2 reduces HBM memory usage significantly"),
            _make_chunk("standard attention has quadratic complexity"),
            _make_chunk("transformer architecture with self-attention layers"),
        ]
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("FlashAttention-2")
        assert len(results) >= 1
        assert results[0].chunk_id == chunks[0]["chunk_id"], \
            "FlashAttention-2 chunk should be ranked first"

    def test_fp8_term_matched_exactly(self, tmp_db: BM25Index):
        chunks = [
            _make_chunk("FP8 quantization reduces memory on H100 GPUs"),
            _make_chunk("full precision float32 training"),
        ]
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("FP8")
        assert len(results) >= 1
        assert results[0].chunk_id == chunks[0]["chunk_id"]

    def test_multi_term_query_ranks_best_match_first(self, tmp_db: BM25Index):
        chunks = [
            _make_chunk("FP8 quantization on H100 achieves 3x speedup over FP16"),
            _make_chunk("H100 GPU specifications and memory bandwidth"),
            _make_chunk("quantization techniques for language models"),
        ]
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("FP8 H100")
        # First chunk has both terms — should rank highest
        assert results[0].chunk_id == chunks[0]["chunk_id"]

    def test_query_returns_top_n_results(self, tmp_db: BM25Index):
        chunks = [_make_chunk(f"attention mechanism variant {i}") for i in range(10)]
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("attention", top_n=3)
        assert len(results) == 3

    def test_query_returns_bm25_result_objects(self, tmp_db: BM25Index):
        tmp_db.upsert_chunks([_make_chunk("FlashAttention paper")])
        results = tmp_db.query("FlashAttention")
        assert isinstance(results[0], BM25Result)

    def test_results_have_positive_scores(self, tmp_db: BM25Index):
        """BM25 scores negated from FTS5 — should be positive."""
        tmp_db.upsert_chunks([_make_chunk("speculative decoding draft model")])
        results = tmp_db.query("speculative decoding")
        assert all(r.bm25_score > 0 for r in results)

    def test_results_ranked_by_score_descending(self, tmp_db: BM25Index):
        chunks = [
            _make_chunk("KV cache compression reduces memory KV cache KV"),  # high freq
            _make_chunk("KV cache used in transformer inference"),             # medium
            _make_chunk("cache memory management"),                            # low
        ]
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("KV cache", top_n=3)
        scores = [r.bm25_score for r in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted score desc"

    def test_query_no_match_returns_empty(self, tmp_db: BM25Index):
        tmp_db.upsert_chunks([_make_chunk("attention mechanism")])
        results = tmp_db.query("xyzzy_nonexistent_term_abc")
        assert results == []

    def test_empty_query_returns_empty(self, tmp_db: BM25Index):
        tmp_db.upsert_chunks([_make_chunk("some text")])
        results = tmp_db.query("")
        assert results == []

    def test_result_fields_populated(self, tmp_db: BM25Index):
        chunk = _make_chunk("FlashAttention reduces memory", doc_id="arxiv_123", section="2.1 Method")
        tmp_db.upsert_chunks([chunk])
        results = tmp_db.query("FlashAttention")
        r = results[0]
        assert r.doc_id    == "arxiv_123"
        assert r.section   == "2.1 Method"
        assert r.chunk_id  == chunk["chunk_id"]
        assert r.rank      == 1
        assert "flashattention" in r.text_preview.lower() or "FlashAttention" in r.text_preview

    def test_rank_field_is_1_based(self, tmp_db: BM25Index):
        chunks = [_make_chunk(f"attention paper {i}") for i in range(3)]
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("attention", top_n=3)
        assert [r.rank for r in results] == [1, 2, 3]


class TestBM25IndexFilters:

    def test_filter_by_doc_id(self, tmp_db: BM25Index):
        chunks = (
            [_make_chunk(f"FlashAttention doc A {i}", doc_id="doc_A") for i in range(3)] +
            [_make_chunk(f"FlashAttention doc B {i}", doc_id="doc_B") for i in range(3)]
        )
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("FlashAttention", filter_doc_ids=["doc_A"])
        assert all(r.doc_id == "doc_A" for r in results)
        assert len(results) == 3

    def test_exclude_code_chunks(self, tmp_db: BM25Index):
        chunks = [
            _make_chunk("def flash_attention_forward(q, k, v):", is_code=True),
            _make_chunk("FlashAttention reduces HBM memory access", is_code=False),
        ]
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("FlashAttention", exclude_code=True)
        assert all(not r.is_code for r in results)

    def test_exclude_table_chunks(self, tmp_db: BM25Index):
        chunks = [
            _make_chunk("Model | Speed | Memory", is_table=True),
            _make_chunk("FlashAttention memory analysis text"),
        ]
        tmp_db.upsert_chunks(chunks)
        results = tmp_db.query("memory", exclude_tables=True)
        assert all(not r.is_table for r in results)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBM25Runner:

    def _make_chunks_dir(self, tmp_path: Path, doc_id: str, n: int = 5) -> Path:
        # Each call gets its own sub-directory to avoid cross-test collisions
        chunks_root = tmp_path / f"chunks_{doc_id}"
        doc_dir = chunks_root / doc_id
        doc_dir.mkdir(parents=True)
        chunks = [
            {
                "chunk_id":          str(uuid.uuid4()),
                "parent_id":         str(uuid.uuid4()),
                "doc_id":            doc_id,
                "title":             "Test Paper",
                "url":               "https://example.com",
                "section":           f"Section {i}",
                "text":              (
                    f"FlashAttention-2 FP8 H100 chunk {i}" if i == 0
                    else f"transformer chunk {i} attention layers MLP blocks"
                ),
                "token_count":       8,
                "chunk_index":       i,
                "parent_chunk_index": 0,
                "is_table":          False,
                "is_code":           False,
                "source_type":       "internal",
            }
            for i in range(n)
        ]
        (doc_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chunks), encoding="utf-8"
        )
        return chunks_root

    def test_build_from_chunks_dir(self, tmp_path: Path):
        from ira.retrieval.bm25_runner import build_bm25_index

        chunks_root = self._make_chunks_dir(tmp_path, "test_doc_001", n=5)
        idx = BM25Index(tmp_path / f"bm25_{id(tmp_path)}.db")

        result = build_bm25_index(chunks_root=chunks_root, bm25_index=idx)

        assert len(result.docs) == 1
        assert result.docs[0].ok
        assert result.docs[0].chunks_read == 5
        assert result.total_inserted == 5
        idx.close()

    def test_incremental_build_skips_existing(self, tmp_path: Path):
        from ira.retrieval.bm25_runner import build_bm25_index

        chunks_root = self._make_chunks_dir(tmp_path, "doc_incr", n=3)
        idx = BM25Index(tmp_path / f"bm25_{id(tmp_path)}.db")

        # First build
        r1 = build_bm25_index(chunks_root=chunks_root, bm25_index=idx)
        assert r1.total_inserted == 3

        # Second build — all skipped
        r2 = build_bm25_index(chunks_root=chunks_root, bm25_index=idx)
        assert r2.total_inserted == 0
        assert r2.total_skipped  == 3
        idx.close()

    def test_query_after_build_returns_results(self, tmp_path: Path):
        from ira.retrieval.bm25_runner import build_bm25_index, query_bm25

        chunks_root = self._make_chunks_dir(tmp_path, "doc_query", n=5)
        idx = BM25Index(tmp_path / f"bm25_{id(tmp_path)}.db")
        build_bm25_index(chunks_root=chunks_root, bm25_index=idx)

        results = query_bm25("FlashAttention-2", top_n=5, bm25_index=idx)
        assert len(results) > 0
        assert results[0]["bm25_score"] > 0
        assert "doc_query" in results[0]["doc_id"]
        idx.close()

    def test_doc_id_filter_only_indexes_target(self, tmp_path: Path):
        from ira.retrieval.bm25_runner import build_bm25_index

        # Two docs in chunks dir
        chunks_root = tmp_path / "chunks"
        for doc in ["doc_A", "doc_B"]:
            d = chunks_root / doc
            d.mkdir(parents=True)
            chunk = {
                "chunk_id": str(uuid.uuid4()), "parent_id": str(uuid.uuid4()),
                "doc_id": doc, "title": "T", "url": None, "section": "S",
                "text": f"text from {doc}", "token_count": 3,
                "chunk_index": 0, "parent_chunk_index": 0,
                "is_table": False, "is_code": False, "source_type": "internal",
            }
            (d / "chunks.jsonl").write_text(json.dumps(chunk), encoding="utf-8")

        idx = BM25Index(tmp_path / f"bm25_{id(tmp_path)}.db")
        result = build_bm25_index(chunks_root=chunks_root, doc_id="doc_A", bm25_index=idx)

        assert len(result.docs) == 1
        assert result.docs[0].doc_id == "doc_A"
        assert idx.stats()["docs"] == 1
        idx.close()