# File: src/ira/retrieval/hybrid_retriever.py
"""
Day 7 — Hybrid retriever: Dense + BM25 → RRF fusion → Parent context expansion.

Design rationale
────────────────
Dense retrieval (Qdrant):
  - Finds semantically related content even without exact term overlap
  - Weak on proper nouns, version strings, and technical identifiers

BM25 retrieval (SQLite FTS5):
  - Exact token match: "FlashAttention-2", "FP8", "H100" hit reliably
  - Weak on paraphrase, context, and conceptual similarity

RRF fusion:
  - Uses rank position, not raw scores — no normalisation needed
  - Score: rrf(chunk) = Σ 1/(k + rank_in_list)  where k=60 (empirical standard)
  - Stable across corpus changes; used in Elasticsearch, Weaviate, Cohere
  - Optional weighted mode for experimentation: rrf = α/dense_rank + (1-α)/bm25_rank

Parent context expansion:
  - Each child chunk (~400 tokens) maps to a parent section (~1500 tokens)
  - Multiple children may match the same parent — we dedup to the best child
  - Parent text is what gets sent to the LLM, giving full synthesis context

Confidence signals:
  - top_score:          rrf score of rank-1 result (absolute quality)
  - score_gap:          rrf[0] - rrf[1]  (is #1 clearly better, or a toss-up?)
  - keyword_coverage:   fraction of query tokens that appeared in BM25 results
  - both_sources_agree: True if top result appeared in BOTH dense AND BM25 lists
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RRF_K = 60          # standard constant; keeps low-rank items from dominating
DEFAULT_DENSE_N  = 20
DEFAULT_BM25_N   = 20
DEFAULT_TOP_K    = 5   # final EvidencePacks returned


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class EvidencePack:
    """
    A single retrieved evidence unit returned to the caller.

    Contains both the matched child chunk and its full parent context,
    plus all scoring metadata for transparency and reranking.
    """
    # Identity
    chunk_id:   str
    parent_id:  str
    doc_id:     str
    title:      str
    url:        Optional[str]
    section:    str

    # Content
    child_text:    str   # matched child chunk (~400 tokens)
    parent_text:   str   # full parent section (~1500 tokens); "" if not found
    parent_token_count: int

    # Scores
    rrf_score:     float
    dense_score:   float   # cosine similarity; 0.0 if not in dense results
    bm25_score:    float   # BM25 score (positive); 0.0 if not in BM25 results

    # Rank provenance (for debug / evaluation)
    dense_rank:    Optional[int]   # rank in dense list (1-based); None if absent
    bm25_rank:     Optional[int]   # rank in BM25 list (1-based); None if absent
    final_rank:    int             # 1-based rank after RRF fusion

    # Metadata
    is_code:   bool
    is_table:  bool
    source_type: str

    # Signals
    in_dense:   bool  # appeared in dense results
    in_bm25:    bool  # appeared in BM25 results
    in_both:    bool  # appeared in both (strong signal)


@dataclass
class ConfidenceSignals:
    """
    Retrieval confidence signals for the calling agent to reason about.

    These are designed to answer: "Should I trust this retrieval result,
    or should I widen the search / tell the user I'm uncertain?"
    """
    top_rrf_score:         float   # absolute quality of best result
    score_gap:             float   # rrf[0] - rrf[1]; large = clear winner
    keyword_coverage:      float   # 0.0–1.0; fraction of query terms in BM25 hits
    both_sources_agree:    bool    # top result came from both dense AND BM25
    results_count:         int     # actual number of EvidencePacks returned
    dense_contributed:     int     # how many top-K came from dense only
    bm25_contributed:      int     # how many top-K came from BM25 only
    both_contributed:      int     # how many top-K came from both


@dataclass
class RetrievalResult:
    """
    The complete output of a hybrid retrieval call.

    evidence_packs: ordered list of top-K results, ready for LLM synthesis
    confidence:     signals for the agent to assess retrieval quality
    debug:          full fusion details for inspection / logging
    """
    query:           str
    evidence_packs:  list[EvidencePack]
    confidence:      ConfidenceSignals
    debug:           dict[str, Any]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def rrf_fuse(
    dense_results:  list[dict[str, Any]],
    bm25_results:   list[dict[str, Any]],
    k:              int   = RRF_K,
    dense_weight:   float = 1.0,
    bm25_weight:    float = 1.0,
) -> list[dict[str, Any]]:
    """
    Fuse dense and BM25 ranked lists using Reciprocal Rank Fusion.

    Each result must have a "chunk_id" field.
    dense_results must have "score" field (cosine similarity).
    bm25_results must have "bm25_score" field.

    Returns merged list sorted by rrf_score descending, with full provenance.
    """
    # Build lookup maps: chunk_id → result dict
    dense_by_id: dict[str, dict] = {r["chunk_id"]: r for r in dense_results}
    bm25_by_id:  dict[str, dict] = {r["chunk_id"]: r for r in bm25_results}

    all_ids = set(dense_by_id) | set(bm25_by_id)

    fused: list[dict[str, Any]] = []
    for cid in all_ids:
        in_dense = cid in dense_by_id
        in_bm25  = cid in bm25_by_id

        dense_rank = dense_by_id[cid]["rank"] if in_dense else None
        bm25_rank  = bm25_by_id[cid]["rank"]  if in_bm25  else None

        rrf = 0.0
        if dense_rank is not None:
            rrf += dense_weight / (k + dense_rank)
        if bm25_rank is not None:
            rrf += bm25_weight / (k + bm25_rank)

        # Merge all metadata — prefer dense record (richer payload) then bm25
        base = (dense_by_id.get(cid) or bm25_by_id.get(cid)).copy()
        base.update({
            "chunk_id":    cid,
            "rrf_score":   rrf,
            "dense_score": dense_by_id[cid].get("score", 0.0) if in_dense else 0.0,
            "bm25_score":  bm25_by_id[cid].get("bm25_score", 0.0) if in_bm25 else 0.0,
            "dense_rank":  dense_rank,
            "bm25_rank":   bm25_rank,
            "in_dense":    in_dense,
            "in_bm25":     in_bm25,
            "in_both":     in_dense and in_bm25,
        })
        fused.append(base)

    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    # Assign final rank after sort
    for i, item in enumerate(fused):
        item["final_rank"] = i + 1

    return fused


# ── Keyword coverage ──────────────────────────────────────────────────────────

def _keyword_coverage(query: str, bm25_results: list[dict[str, Any]]) -> float:
    """
    Fraction of query tokens that appeared in at least one BM25 result.

    Uses the same tokenizer as the BM25 index for consistency.
    """
    if not bm25_results:
        return 0.0
    try:
        from ira.retrieval.bm25_index import tokenize
        query_tokens = set(tokenize(query))
    except ImportError:
        # Fallback: simple lowercase word split
        query_tokens = set(query.lower().split())

    if not query_tokens:
        return 0.0

    # Collect all text from BM25 results to check term presence
    matched_tokens: set[str] = set()
    for result in bm25_results:
        preview = (result.get("text_preview") or "").lower()
        for tok in query_tokens:
            if tok in preview:
                matched_tokens.add(tok)

    return len(matched_tokens) / len(query_tokens)


# ── Main retriever class ──────────────────────────────────────────────────────

class HybridRetriever:
    """
    Hybrid retriever combining dense (Qdrant) + keyword (BM25) search
    with RRF fusion and parent context expansion.

    Designed to be instantiated once and reused across queries.
    All heavy objects (embedder, indexes, parent store) are injected
    so this class is testable without disk access.
    """

    def __init__(
        self,
        embedder=None,
        dense_index=None,
        bm25_index=None,
        parent_store=None,
        chunks_root=None,
        rrf_k: int = RRF_K,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        self._embedder     = embedder
        self._dense_index  = dense_index
        self._bm25_index   = bm25_index
        self._parent_store = parent_store
        self._chunks_root  = chunks_root
        self.rrf_k         = rrf_k
        self.dense_weight  = dense_weight
        self.bm25_weight   = bm25_weight

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_embedder(self):
        if self._embedder is None:
            from ira.retrieval.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder

    def _get_dense_index(self):
        if self._dense_index is None:
            from ira.retrieval.qdrant_index import get_index
            embedder = self._get_embedder()
            self._dense_index = get_index(vector_dim=embedder.embedding_dim)
        return self._dense_index

    def _get_bm25_index(self):
        if self._bm25_index is None:
            from ira.retrieval.bm25_index import get_bm25_index
            self._bm25_index = get_bm25_index()
        return self._bm25_index

    def _get_parent_store(self):
        if self._parent_store is None:
            from ira.retrieval.parent_store import get_parent_store
            self._parent_store = get_parent_store(self._chunks_root)
        return self._parent_store

    def close(self) -> None:
        """Release dense index connection (BM25 and parent store need no cleanup)."""
        if self._dense_index is not None:
            try:
                self._dense_index.close()
            except Exception:
                pass

    # ── Dense retrieval ───────────────────────────────────────────────────────

    def _retrieve_dense(
        self, query: str, top_n: int
    ) -> list[dict[str, Any]]:
        """Run dense query and return normalised result dicts with 'rank' field."""
        try:
            from ira.retrieval.index_runner import query_index
            raw = query_index(
                query_text=query,
                top_k=top_n,
                score_threshold=0.0,
            )
            # query_index already returns rank field; ensure it's present
            for i, r in enumerate(raw):
                r.setdefault("rank", i + 1)
            return raw
        except Exception as e:
            logger.warning("Dense retrieval failed: %s", e)
            return []

    # ── BM25 retrieval ────────────────────────────────────────────────────────

    def _retrieve_bm25(
        self, query: str, top_n: int
    ) -> list[dict[str, Any]]:
        """Run BM25 query and return result dicts with 'rank' field."""
        try:
            from ira.retrieval.bm25_runner import query_bm25
            raw = query_bm25(
                query_text=query,
                top_n=top_n,
                bm25_index=self._get_bm25_index(),
            )
            for i, r in enumerate(raw):
                r.setdefault("rank", i + 1)
            return raw
        except Exception as e:
            logger.warning("BM25 retrieval failed: %s", e)
            return []

    # ── Parent expansion ──────────────────────────────────────────────────────

    def _expand_to_parent(
        self,
        fused_item: dict[str, Any],
        parent_store,
        max_parent_chars: int = 6000,
    ) -> dict[str, Any]:
        """
        Add parent_text and parent_token_count to a fused result item.
        Uses lazy load if the doc hasn't been loaded yet.
        """
        parent_id = fused_item.get("parent_id", "")
        doc_id    = fused_item.get("doc_id", "")

        parent_record = parent_store.get(parent_id, doc_id=doc_id)

        if parent_record:
            raw_text = parent_record.get("text", "")
            parent_text = raw_text[:max_parent_chars]
            if len(raw_text) > max_parent_chars:
                parent_text += "\n…[truncated]"
            parent_token_count = parent_record.get("token_count", 0)
        else:
            parent_text        = ""
            parent_token_count = 0
            logger.debug(
                "Parent not found: parent_id=%s doc_id=%s", parent_id, doc_id
            )

        return {
            **fused_item,
            "parent_text":        parent_text,
            "parent_token_count": parent_token_count,
        }

    # ── Parent deduplication ──────────────────────────────────────────────────

    @staticmethod
    def _dedup_by_parent(
        fused: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        When multiple child chunks map to the same parent, keep only the
        highest-rrf_score child per parent.

        This prevents the LLM from receiving the same parent section 3 times
        with slightly different child excerpts.
        """
        seen_parents: dict[str, dict[str, Any]] = {}
        for item in fused:
            pid = item.get("parent_id", item["chunk_id"])
            if pid not in seen_parents:
                seen_parents[pid] = item
            else:
                # Keep whichever child has the higher RRF score
                if item["rrf_score"] > seen_parents[pid]["rrf_score"]:
                    seen_parents[pid] = item

        # Return in original order (dict preserves insertion order in Python 3.7+)
        return list(seen_parents.values())

    # ── Main query ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k:          int   = DEFAULT_TOP_K,
        dense_n:        int   = DEFAULT_DENSE_N,
        bm25_n:         int   = DEFAULT_BM25_N,
        max_parent_chars: int = 6000,
        dense_only:     bool  = False,
        bm25_only:      bool  = False,
    ) -> RetrievalResult:
        """
        Run hybrid retrieval for the given query.

        Args:
            query:            natural language or keyword query
            top_k:            number of EvidencePacks to return (default 5)
            dense_n:          candidates to fetch from dense index (default 20)
            bm25_n:           candidates to fetch from BM25 index (default 20)
            max_parent_chars: truncate parent text at this length (default 6000 ≈ 1500 tok)
            dense_only:       skip BM25 (useful for ablation)
            bm25_only:        skip dense (useful for ablation)

        Returns:
            RetrievalResult with evidence_packs, confidence, debug
        """
        # ── 1. Retrieve from each source ──────────────────────────────────────
        dense_results: list[dict] = []
        bm25_results:  list[dict] = []

        if not bm25_only:
            dense_results = self._retrieve_dense(query, dense_n)
            logger.debug("Dense returned %d results", len(dense_results))

        if not dense_only:
            bm25_results = self._retrieve_bm25(query, bm25_n)
            logger.debug("BM25 returned %d results", len(bm25_results))

        # ── 2. RRF fusion ─────────────────────────────────────────────────────
        fused = rrf_fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
            k=self.rrf_k,
            dense_weight=self.dense_weight if not bm25_only else 0.0,
            bm25_weight=self.bm25_weight   if not dense_only else 0.0,
        )

        # ── 3. Deduplicate by parent, then take top candidates ────────────────
        # Take more than top_k before dedup to ensure we have enough after dedup
        prefetch = min(len(fused), max(top_k * 3, 15))
        fused_top = fused[:prefetch]
        deduped = self._dedup_by_parent(fused_top)

        # ── 4. Parent expansion ───────────────────────────────────────────────
        parent_store = self._get_parent_store()
        expanded = [
            self._expand_to_parent(item, parent_store, max_parent_chars)
            for item in deduped
        ]

        # ── 5. Final top-K selection ──────────────────────────────────────────
        final = expanded[:top_k]

        # ── 6. Build EvidencePack objects ─────────────────────────────────────
        evidence_packs: list[EvidencePack] = []
        for rank, item in enumerate(final, 1):
            pack = EvidencePack(
                chunk_id     = item.get("chunk_id", ""),
                parent_id    = item.get("parent_id", ""),
                doc_id       = item.get("doc_id", ""),
                title        = item.get("title", ""),
                url          = item.get("url"),
                section      = item.get("section", ""),
                child_text   = item.get("text_preview", ""),
                parent_text  = item.get("parent_text", ""),
                parent_token_count = item.get("parent_token_count", 0),
                rrf_score    = round(item.get("rrf_score", 0.0), 6),
                dense_score  = round(item.get("dense_score", 0.0), 4),
                bm25_score   = round(item.get("bm25_score", 0.0), 4),
                dense_rank   = item.get("dense_rank"),
                bm25_rank    = item.get("bm25_rank"),
                final_rank   = rank,
                is_code      = bool(item.get("is_code", False)),
                is_table     = bool(item.get("is_table", False)),
                source_type  = item.get("source_type", "internal"),
                in_dense     = bool(item.get("in_dense", False)),
                in_bm25      = bool(item.get("in_bm25", False)),
                in_both      = bool(item.get("in_both", False)),
            )
            evidence_packs.append(pack)

        # ── 7. Confidence signals ─────────────────────────────────────────────
        rrf_scores = [item["rrf_score"] for item in fused]
        top_score  = rrf_scores[0] if rrf_scores else 0.0
        score_gap  = (rrf_scores[0] - rrf_scores[1]) if len(rrf_scores) >= 2 else top_score

        kw_coverage = _keyword_coverage(query, bm25_results)

        dense_only_count = sum(1 for p in evidence_packs if p.in_dense and not p.in_bm25)
        bm25_only_count  = sum(1 for p in evidence_packs if p.in_bm25 and not p.in_dense)
        both_count       = sum(1 for p in evidence_packs if p.in_both)

        confidence = ConfidenceSignals(
            top_rrf_score      = round(top_score, 6),
            score_gap          = round(score_gap, 6),
            keyword_coverage   = round(kw_coverage, 3),
            both_sources_agree = bool(evidence_packs and evidence_packs[0].in_both),
            results_count      = len(evidence_packs),
            dense_contributed  = dense_only_count,
            bm25_contributed   = bm25_only_count,
            both_contributed   = both_count,
        )

        # ── 8. Debug payload ─────────────────────────────────────────────────
        debug: dict[str, Any] = {
            "query":          query,
            "fusion_mode":    "rrf_only" if (self.dense_weight == self.bm25_weight == 1.0)
                              else f"weighted(dense={self.dense_weight}, bm25={self.bm25_weight})",
            "rrf_k":          self.rrf_k,
            "dense_n_fetched": len(dense_results),
            "bm25_n_fetched":  len(bm25_results),
            "fused_total":    len(fused),
            "after_dedup":    len(deduped),
            "final_k":        len(evidence_packs),
            "all_candidates": [
                {
                    "rank":        c["final_rank"],
                    "chunk_id":    c["chunk_id"],
                    "doc_id":      c.get("doc_id", ""),
                    "section":     c.get("section", ""),
                    "rrf_score":   round(c["rrf_score"], 6),
                    "dense_score": round(c.get("dense_score", 0.0), 4),
                    "bm25_score":  round(c.get("bm25_score", 0.0), 4),
                    "dense_rank":  c.get("dense_rank"),
                    "bm25_rank":   c.get("bm25_rank"),
                    "in_both":     c.get("in_both", False),
                }
                for c in fused[:20]   # show top-20 in debug
            ],
        }

        return RetrievalResult(
            query          = query,
            evidence_packs = evidence_packs,
            confidence     = confidence,
            debug          = debug,
        )

    def close(self) -> None:
        """Release dense index connection."""
        if self._dense_index is not None:
            try:
                self._dense_index.close()
            except Exception:
                pass
        if self._bm25_index is not None:
            try:
                self._bm25_index.close()
            except Exception:
                pass


# ── Module-level convenience function ────────────────────────────────────────

def get_hybrid_retriever(
    chunks_root=None,
    rrf_k: int = RRF_K,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> HybridRetriever:
    """Build a HybridRetriever using default settings."""
    from ira.settings import settings
    if chunks_root is None:
        chunks_root = settings.data_dir / "chunks"
    return HybridRetriever(
        chunks_root=chunks_root,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )