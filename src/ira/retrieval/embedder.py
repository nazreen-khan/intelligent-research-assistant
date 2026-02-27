# File: src/ira/retrieval/embedder.py
"""
Day 5 — Batched, cached, deterministic text embedder.

Design decisions:
  - Model: BAAI/bge-small-en-v1.5 (default) — 33M params, 384-dim, CPU-friendly
  - Batching: configurable batch size, progress logging for large corpora
  - Disk cache: SHA256(text) → .npy file so re-indexing never re-embeds known text
  - Deterministic: same text → same vector every time (model is frozen at load)
  - Normalisation: L2-normalised so cosine_similarity == dot_product (faster Qdrant query)

BGE prompt prefix:
  BGE models are trained with an instruction prefix for queries but NOT for passages.
  Passages (chunks) are embedded WITHOUT prefix.
  Queries SHOULD use prefix: "Represent this sentence for searching relevant passages: "
  We expose embed_query() and embed_passages() to enforce this distinction.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# BGE instruction prefix used ONLY for query embedding (not passage embedding)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(cache_dir: Path, model_slug: str, text_sha: str) -> Path:
    """
    Two-level directory: <cache_dir>/<model_slug>/<first2>/<sha256>.npy
    The first-2-char sub-directory prevents huge flat directories.
    """
    return cache_dir / model_slug / text_sha[:2] / f"{text_sha}.npy"


def _model_slug(model_name: str) -> str:
    """Convert 'BAAI/bge-small-en-v1.5' → 'BAAI_bge-small-en-v1.5'"""
    return model_name.replace("/", "_")


class Embedder:
    """
    Wraps a SentenceTransformer model with:
      - Lazy loading (model loads on first call, not on import)
      - Batched encoding with configurable batch size
      - Optional disk cache keyed by SHA256(text)
      - Separate embed_query() / embed_passages() for BGE prefix handling
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 512,
        normalize: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.cache_dir = cache_dir
        self._model = None   # lazy-loaded
        self._model_slug = _model_slug(model_name)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Run: uv add sentence-transformers"
            ) from e

        logger.info("Loading embedding model: %s on device=%s", self.model_name, self.device)
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )
        # Override max_seq_length to our config value
        self._model.max_seq_length = self.max_length
        logger.info("Embedding model loaded. Vector dim=%d", self.embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Return vector dimension without loading the model if possible."""
        # Well-known dims for common BGE models — avoids loading just for dim check
        _known_dims = {
            "BAAI/bge-small-en-v1.5": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
        }
        if self.model_name in _known_dims:
            return _known_dims[self.model_name]
        # Fallback: load model and ask
        self._load_model()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_get(self, text_sha: str) -> Optional[np.ndarray]:
        if self.cache_dir is None:
            return None
        p = _cache_path(self.cache_dir, self._model_slug, text_sha)
        if p.exists():
            try:
                return np.load(str(p))
            except Exception:
                logger.warning("Cache read failed for %s, will re-embed", p)
                return None
        return None

    def _cache_put(self, text_sha: str, vec: np.ndarray) -> None:
        if self.cache_dir is None:
            return
        p = _cache_path(self.cache_dir, self._model_slug, text_sha)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".npy.tmp")
        try:
            np.save(str(tmp), vec)
            tmp.replace(p)
        except Exception:
            logger.warning("Cache write failed for %s", p)
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # ── Core encode ───────────────────────────────────────────────────────────

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts. Returns float32 ndarray of shape (N, dim).
        Uses disk cache per text; only sends cache-misses to the model.
        """
        self._load_model()

        n = len(texts)
        dim = self.embedding_dim
        result = np.zeros((n, dim), dtype=np.float32)

        # Split into cached vs uncached
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        shas: list[str] = [_sha256(t) for t in texts]

        for i, (text, sha) in enumerate(zip(texts, shas)):
            cached = self._cache_get(sha)
            if cached is not None:
                result[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        cache_hits = n - len(miss_indices)
        if cache_hits > 0:
            logger.debug("Embedding cache: %d hits, %d misses out of %d", cache_hits, len(miss_indices), n)

        # Encode cache misses in sub-batches
        if miss_texts:
            for batch_start in range(0, len(miss_texts), self.batch_size):
                batch_texts = miss_texts[batch_start: batch_start + self.batch_size]
                batch_indices = miss_indices[batch_start: batch_start + self.batch_size]

                vecs: np.ndarray = self._model.encode(  # type: ignore[union-attr]
                    batch_texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

                for local_i, (global_i, sha, vec) in enumerate(
                    zip(batch_indices, [shas[j] for j in batch_indices], vecs)
                ):
                    result[global_i] = vec
                    self._cache_put(sha, vec)

        return result

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed document passages (chunks). No instruction prefix.
        Returns float32 ndarray of shape (N, dim).

        Use this for indexing chunks into Qdrant.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        return self._encode_batch(list(texts))

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a user query with BGE instruction prefix.
        Returns float32 ndarray of shape (dim,).

        Use this for search/retrieval queries.
        """
        prefixed = _BGE_QUERY_PREFIX + query
        result = self._encode_batch([prefixed])
        return result[0]

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        """
        Embed multiple queries with BGE instruction prefix.
        Returns float32 ndarray of shape (N, dim).
        """
        prefixed = [_BGE_QUERY_PREFIX + q for q in queries]
        return self._encode_batch(prefixed)

    # ── Cache management ──────────────────────────────────────────────────────

    def cache_stats(self) -> dict:
        """Return cache statistics (file count, total size)."""
        if self.cache_dir is None:
            return {"enabled": False}
        model_cache = self.cache_dir / self._model_slug
        if not model_cache.exists():
            return {"enabled": True, "files": 0, "size_mb": 0.0}
        files = list(model_cache.rglob("*.npy"))
        size = sum(f.stat().st_size for f in files)
        return {
            "enabled": True,
            "model": self.model_name,
            "files": len(files),
            "size_mb": round(size / 1024 / 1024, 2),
        }

    def clear_cache(self) -> int:
        """Delete all cached embeddings for this model. Returns number of files deleted."""
        if self.cache_dir is None:
            return 0
        model_cache = self.cache_dir / self._model_slug
        if not model_cache.exists():
            return 0
        files = list(model_cache.rglob("*.npy"))
        for f in files:
            f.unlink(missing_ok=True)
        return len(files)


# ── Module-level singleton (lazy) ─────────────────────────────────────────────

_default_embedder: Optional[Embedder] = None


def get_embedder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> Embedder:
    """
    Return the module-level singleton Embedder, constructed from settings if not provided.
    Accepts overrides for testing.
    """
    global _default_embedder

    # If any override is provided, always create a new instance
    if any(v is not None for v in (model_name, device, batch_size, cache_dir)):
        from ira.settings import settings
        return Embedder(
            model_name=model_name or settings.EMBED_MODEL,
            device=device or settings.EMBED_DEVICE,
            batch_size=batch_size or settings.EMBED_BATCH_SIZE,
            max_length=settings.EMBED_MAX_LENGTH,
            normalize=settings.EMBED_NORMALIZE,
            cache_dir=cache_dir,
        )

    if _default_embedder is None:
        from ira.settings import settings
        _default_embedder = Embedder(
            model_name=settings.EMBED_MODEL,
            device=settings.EMBED_DEVICE,
            batch_size=settings.EMBED_BATCH_SIZE,
            max_length=settings.EMBED_MAX_LENGTH,
            normalize=settings.EMBED_NORMALIZE,
            cache_dir=settings.embed_cache_dir,
        )
    return _default_embedder


def reset_embedder() -> None:
    """Reset the singleton (useful in tests)."""
    global _default_embedder
    _default_embedder = None