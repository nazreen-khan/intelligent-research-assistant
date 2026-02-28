# File: src/ira/retrieval/reranker.py
"""
Day 8 — Cross-encoder reranker with disk cache.

Design rationale
────────────────
Bi-encoder (BGE embedder used in Days 5-7):
  - Encodes query and document INDEPENDENTLY into vectors
  - Fast: embed once, search with dot product
  - Weakness: misses fine-grained query-document interaction

Cross-encoder (this file):
  - Sees (query, document) TOGETHER as a single input
  - Produces one relevance score per pair
  - Slower: O(N) forward passes for N candidates
  - Stronger: captures exact term overlap, negation, context

Pipeline position:
  hybrid retrieval (top-20) → cross-encoder rerank → top-5

Why bge-reranker-base:
  - 278MB, CPU-workable (~50-200ms for 20 pairs)
  - Free, no API key
  - Uses sentence_transformers.CrossEncoder — already in deps
  - Upgrade path: set RERANKER_MODEL=BAAI/bge-reranker-v2-m3 in .env

Disk cache:
  - Key: sha256(query + sorted(chunk_ids))
  - Value: {chunk_id: score} JSON
  - Atomic write: .tmp → rename (safe on Windows NTFS)
  - Prevents re-running the model on repeated identical queries
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(query: str, chunk_ids: list[str]) -> str:
    """
    Deterministic cache key for a (query, candidate_set) pair.

    Sorting chunk_ids ensures the key is order-independent —
    the same candidates in a different order hit the same cache entry.
    """
    raw = query + "|" + ",".join(sorted(chunk_ids))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, key: str) -> Path:
    """Two-level directory to avoid huge flat dirs: <cache_dir>/<key[:2]>/<key>.json"""
    return cache_dir / key[:2] / f"{key}.json"


# ── Main class ────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Wraps sentence_transformers.CrossEncoder with:
      - Lazy model loading (downloads on first use, ~278MB)
      - Batched scoring
      - Disk cache keyed by sha256(query + chunk_ids)
      - Deterministic output for evaluation

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, evidence_packs, keep_k=5)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        device: str = "cpu",
        batch_size: int = 16,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self._model = None  # lazy-loaded

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load CrossEncoder on first use. Downloads model if not cached."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. Run: uv add sentence-transformers"
            ) from e

        logger.info(
            "Loading reranker model: %s on device=%s (downloading if needed)",
            self.model_name,
            self.device,
        )
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=512,
        )
        logger.info("Reranker model loaded: %s", self.model_name)

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_get(self, key: str) -> Optional[dict[str, float]]:
        """Return cached {chunk_id: score} dict, or None on miss."""
        if self.cache_dir is None:
            return None
        p = _cache_path(self.cache_dir, key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Rerank cache read failed: %s — will re-score", p)
            return None

    def _cache_put(self, key: str, scores: dict[str, float]) -> None:
        """Atomically write {chunk_id: score} dict to disk."""
        if self.cache_dir is None:
            return
        p = _cache_path(self.cache_dir, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        try:
            tmp.write_text(
                json.dumps(scores, ensure_ascii=False), encoding="utf-8"
            )
            tmp.replace(p)
        except Exception:
            logger.warning("Rerank cache write failed: %s", p)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_pairs(
        self,
        query: str,
        texts: list[str],
    ) -> list[float]:
        """
        Run the cross-encoder on (query, text) pairs.
        Returns a float score per text. Higher = more relevant.
        """
        self._load_model()
        if not texts:
            return []

        pairs = [[query, t] for t in texts]

        scores = self._model.predict(  # type: ignore[union-attr]
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        evidence_packs: list,
        keep_k: int = 5,
    ) -> list:
        """
        Rerank a list of EvidencePack objects and return the top-keep_k.

        Steps:
          1. Check disk cache (key = sha256 of query + chunk_ids)
          2. On miss: score all (query, parent_text) pairs via cross-encoder
          3. Sort by score descending, keep top-keep_k
          4. Attach reranker_score to each pack (mutates a copy)
          5. Write scores to cache

        Args:
            query:          the user query string
            evidence_packs: list of EvidencePack dataclass objects
            keep_k:         how many to return (default 5)

        Returns:
            Sorted list of EvidencePack, length = min(keep_k, len(evidence_packs))
            Each pack has a reranker_score attribute attached.
        """
        if not evidence_packs:
            return []

        keep_k = min(keep_k, len(evidence_packs))

        chunk_ids = [p.chunk_id for p in evidence_packs]
        cache_key = _cache_key(query, chunk_ids)

        # ── Cache lookup ──────────────────────────────────────────────────────
        cached_scores = self._cache_get(cache_key)
        if cached_scores is not None:
            logger.debug("Rerank cache hit for query: %.60s…", query)
            scores_by_id = cached_scores
        else:
            # ── Score all candidates ──────────────────────────────────────────
            # Use parent_text as the document — gives the cross-encoder full context
            # Fall back to child_text if parent_text is empty
            texts = [
                p.parent_text if p.parent_text else p.child_text
                for p in evidence_packs
            ]
            raw_scores = self._score_pairs(query, texts)
            scores_by_id = {
                p.chunk_id: score
                for p, score in zip(evidence_packs, raw_scores)
            }
            self._cache_put(cache_key, scores_by_id)
            logger.debug(
                "Reranked %d candidates → keeping %d | top score=%.4f",
                len(evidence_packs),
                keep_k,
                max(raw_scores) if raw_scores else 0.0,
            )

        # ── Sort and attach scores ────────────────────────────────────────────
        # We use dataclasses.replace if available, else monkey-patch a copy
        import dataclasses

        scored: list[tuple[float, object]] = []
        for pack in evidence_packs:
            score = scores_by_id.get(pack.chunk_id, 0.0)
            # Attach reranker_score as a new field on a shallow copy
            pack_copy = dataclasses.replace(pack, **{})  # shallow copy
            object.__setattr__(pack_copy, "reranker_score", score)
            scored.append((score, pack_copy))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [pack for _, pack in scored[:keep_k]]

    def score_texts(
        self,
        query: str,
        texts: Sequence[str],
    ) -> list[float]:
        """
        Score raw (query, text) pairs without EvidencePack wrapping.
        Used directly by the eval harness to measure reranker quality.

        Returns list of floats, same length as texts. Higher = more relevant.
        """
        return self._score_pairs(query, list(texts))

    # ── Cache management ──────────────────────────────────────────────────────

    def cache_stats(self) -> dict:
        """Return cache statistics."""
        if self.cache_dir is None:
            return {"enabled": False}
        if not self.cache_dir.exists():
            return {"enabled": True, "files": 0, "size_kb": 0.0}
        files = list(self.cache_dir.rglob("*.json"))
        size = sum(f.stat().st_size for f in files)
        return {
            "enabled": True,
            "model": self.model_name,
            "files": len(files),
            "size_kb": round(size / 1024, 2),
        }

    def clear_cache(self) -> int:
        """Delete all cached rerank results. Returns number of files deleted."""
        if self.cache_dir is None:
            return 0
        if not self.cache_dir.exists():
            return 0
        files = list(self.cache_dir.rglob("*.json"))
        for f in files:
            f.unlink(missing_ok=True)
        return len(files)


# ── Module-level singleton ─────────────────────────────────────────────────────

_default_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> CrossEncoderReranker:
    """
    Return the module-level singleton CrossEncoderReranker.
    Accepts overrides for testing.
    """
    global _default_reranker

    if any(v is not None for v in (model_name, device, cache_dir)):
        from ira.settings import settings
        return CrossEncoderReranker(
            model_name=model_name or settings.RERANKER_MODEL,
            device=device or settings.RERANKER_DEVICE,
            batch_size=settings.RERANKER_BATCH_SIZE,
            cache_dir=cache_dir,
        )

    if _default_reranker is None:
        from ira.settings import settings
        _default_reranker = CrossEncoderReranker(
            model_name=settings.RERANKER_MODEL,
            device=settings.RERANKER_DEVICE,
            batch_size=settings.RERANKER_BATCH_SIZE,
            cache_dir=settings.reranker_cache_dir,
        )
    return _default_reranker


def reset_reranker() -> None:
    """Reset singleton — useful in tests."""
    global _default_reranker
    _default_reranker = None