# File: src/ira/retrieval/rerank_runner.py
"""
Day 8 — Rerank runner: orchestrates hybrid retrieval → cross-encoder reranking.

Mirrors the pattern of bm25_runner.py and index_runner.py:
  - Thin orchestration layer
  - Wires together HybridRetriever + CrossEncoderReranker
  - Used by CLI commands and the agent layer (Day 9)
  - All heavy objects passed in or lazy-loaded via singletons

Pipeline:
    query
      └── HybridRetriever.retrieve(top_k=RERANKER_TOP_N)   → up to 20 candidates
            └── CrossEncoderReranker.rerank(keep_k=KEEP_K) → top-5 final results
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RerankResult:
    """
    Full output of a hybrid-retrieve + rerank call.

    Designed to be the drop-in replacement for RetrievalResult once
    the reranker is enabled — same evidence_packs interface, richer metadata.
    """
    query:               str
    evidence_packs:      list          # EvidencePack objects with reranker_score attached
    retrieval_count:     int           # how many hybrid candidates were fed to reranker
    rerank_count:        int           # how many were returned after reranking
    retrieval_ms:        int           # wall-clock time for hybrid step
    rerank_ms:           int           # wall-clock time for rerank step
    total_ms:            int           # end-to-end latency
    cache_hit:           bool          # True if reranker result came from disk cache
    debug:               dict[str, Any]


# ── Core function ─────────────────────────────────────────────────────────────

def query_with_rerank(
    query_text: str,
    keep_k: int = 5,
    retrieval_top_n: int = 20,
    hybrid_retriever=None,
    reranker=None,
    dense_n: int = 20,
    bm25_n: int = 20,
) -> RerankResult:
    """
    Run the full pipeline: hybrid retrieval → cross-encoder rerank.

    Args:
        query_text:       user query string
        keep_k:           final number of results to return after reranking
        retrieval_top_n:  how many hybrid candidates to feed into reranker
        hybrid_retriever: injected HybridRetriever (lazy-loads if None)
        reranker:         injected CrossEncoderReranker (lazy-loads if None)
        dense_n:          dense candidates to fetch (passed to HybridRetriever)
        bm25_n:           BM25 candidates to fetch (passed to HybridRetriever)

    Returns:
        RerankResult with final evidence_packs + full timing + debug info
    """
    # ── Lazy load dependencies ────────────────────────────────────────────────
    if hybrid_retriever is None:
        from ira.retrieval.hybrid_retriever import get_hybrid_retriever
        hybrid_retriever = get_hybrid_retriever()

    if reranker is None:
        from ira.retrieval.reranker import get_reranker
        reranker = get_reranker()

    # ── Step 1: Hybrid retrieval ──────────────────────────────────────────────
    t0 = time.perf_counter()
    retrieval_result = hybrid_retriever.retrieve(
        query=query_text,
        top_k=retrieval_top_n,
        dense_n=dense_n,
        bm25_n=bm25_n,
        exclude_sections=["references", "bibliography", "acknowledgements",
                          "acknowledgments"],
    )
    retrieval_ms = int((time.perf_counter() - t0) * 1000)

    candidates = retrieval_result.evidence_packs
    logger.debug(
        "Hybrid retrieval: %d candidates in %dms", len(candidates), retrieval_ms
    )

    # ── Step 2: Cross-encoder rerank ──────────────────────────────────────────
    t1 = time.perf_counter()

    # Detect cache hit by checking if scores already exist before calling rerank
    from ira.retrieval.reranker import _cache_key, _cache_path
    chunk_ids = [p.chunk_id for p in candidates]
    key = _cache_key(query_text, chunk_ids)
    cache_hit = False
    if reranker.cache_dir is not None:
        cache_hit = _cache_path(reranker.cache_dir, key).exists()

    reranked = reranker.rerank(
        query=query_text,
        evidence_packs=candidates,
        keep_k=keep_k,
    )
    rerank_ms = int((time.perf_counter() - t1) * 1000)

    total_ms = int((time.perf_counter() - t0) * 1000)

    logger.info(
        "query_with_rerank | candidates=%d kept=%d retrieval_ms=%d "
        "rerank_ms=%d cache_hit=%s",
        len(candidates),
        len(reranked),
        retrieval_ms,
        rerank_ms,
        cache_hit,
    )

    # ── Debug payload ─────────────────────────────────────────────────────────
    debug: dict[str, Any] = {
        "query":              query_text,
        "retrieval_top_n":    retrieval_top_n,
        "keep_k":             keep_k,
        "retrieval_ms":       retrieval_ms,
        "rerank_ms":          rerank_ms,
        "total_ms":           total_ms,
        "cache_hit":          cache_hit,
        "retrieval_confidence": {
            "top_rrf_score":      retrieval_result.confidence.top_rrf_score,
            "score_gap":          retrieval_result.confidence.score_gap,
            "keyword_coverage":   retrieval_result.confidence.keyword_coverage,
            "both_sources_agree": retrieval_result.confidence.both_sources_agree,
        },
        # Pre-rerank order vs post-rerank order for comparison
        "pre_rerank_order": [
            {
                "rank":        i + 1,
                "chunk_id":    p.chunk_id,
                "doc_id":      p.doc_id,
                "section":     p.section,
                "rrf_score":   p.rrf_score,
                "in_both":     p.in_both,
            }
            for i, p in enumerate(candidates[:retrieval_top_n])
        ],
        "post_rerank_order": [
            {
                "rank":            i + 1,
                "chunk_id":        p.chunk_id,
                "doc_id":          p.doc_id,
                "section":         p.section,
                "reranker_score":  getattr(p, "reranker_score", None),
                "rrf_score":       p.rrf_score,
            }
            for i, p in enumerate(reranked)
        ],
    }

    return RerankResult(
        query=query_text,
        evidence_packs=reranked,
        retrieval_count=len(candidates),
        rerank_count=len(reranked),
        retrieval_ms=retrieval_ms,
        rerank_ms=rerank_ms,
        total_ms=total_ms,
        cache_hit=cache_hit,
        debug=debug,
    )


# ── Cache warm-up utility ─────────────────────────────────────────────────────

def warm_rerank_cache(
    queries: list[str],
    keep_k: int = 5,
    retrieval_top_n: int = 20,
    hybrid_retriever=None,
    reranker=None,
) -> dict[str, Any]:
    """
    Pre-warm the rerank cache for a list of known queries.

    Useful before a demo: run this once so every demo query is a cache hit
    and latency is <5ms instead of ~200ms.

    Returns a summary dict with timing and cache stats.
    """
    results = []
    total_start = time.perf_counter()

    for i, q in enumerate(queries):
        logger.info("Warming cache %d/%d: %.60s", i + 1, len(queries), q)
        result = query_with_rerank(
            query_text=q,
            keep_k=keep_k,
            retrieval_top_n=retrieval_top_n,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
        )
        results.append({
            "query":        q[:60],
            "cache_hit":    result.cache_hit,
            "rerank_ms":    result.rerank_ms,
            "total_ms":     result.total_ms,
        })

    total_ms = int((time.perf_counter() - total_start) * 1000)

    # Get cache stats from reranker singleton
    if reranker is None:
        from ira.retrieval.reranker import get_reranker
        reranker = get_reranker()

    return {
        "queries_processed": len(queries),
        "total_ms":          total_ms,
        "per_query":         results,
        "cache_stats":       reranker.cache_stats(),
    }