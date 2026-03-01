"""
agent/web_cache.py — TTL disk cache for web search results.

Keyed by sha256(provider + query) → JSON file on disk.
Expiry checked on read — stale entries treated as cache miss.

Design:
  - Windows-safe: pathlib.Path throughout, atomic write via tmp + rename
  - Thread-safe for single-process use (no file locking needed for our scale)
  - Returns None on any read error (cache miss) — never raises
  - Writes are best-effort — failure logged but not raised

Usage:
    cache = WebCache()                          # uses settings defaults
    results = cache.get("vLLM benchmarks")      # None = miss
    if results is None:
        results = call_api(...)
        cache.set("vLLM benchmarks", results)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ira.settings import settings

logger = logging.getLogger(__name__)


class WebCache:
    """
    Disk-backed TTL cache for web search results.

    Each entry is a JSON file: {cached_at: ISO str, results: list[dict]}
    File path: <cache_dir>/<xx>/<sha256>.json  (two-char prefix for dir spread)
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_hours: int | None = None,
        provider: str | None = None,
    ) -> None:
        self._cache_dir = cache_dir or settings.web_search_cache_dir or Path("data/cache/web")
        self._ttl_hours = ttl_hours if ttl_hours is not None else settings.WEB_SEARCH_CACHE_TTL_HOURS
        self._provider = provider or settings.WEB_SEARCH_PROVIDER

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, query: str) -> list[dict[str, Any]] | None:
        """
        Retrieve cached results for a query.

        Returns:
            List of result dicts if cache hit and not expired, else None.
        """
        path = self._cache_path(query)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data["cached_at"])

            # Check TTL
            age_hours = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
            if age_hours >= self._ttl_hours:
                logger.debug("web_cache: expired entry for query=%r (age=%.1fh)", query[:60], age_hours)
                return None

            results = data.get("results", [])
            logger.debug(
                "web_cache: hit query=%r results=%d age=%.1fh",
                query[:60], len(results), age_hours,
            )
            return results

        except Exception as exc:
            logger.debug("web_cache: read error %s for path=%s", exc, path)
            return None

    def set(self, query: str, results: list[dict[str, Any]]) -> None:
        """
        Write results to cache.

        Uses atomic write (temp file + rename) to avoid partial reads
        on Windows if the process is interrupted mid-write.
        """
        path = self._cache_path(query)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cached_at": datetime.now(timezone.utc).isoformat(),
                "provider": self._provider,
                "query": query,
                "results": results,
            }
            # Atomic write: write to temp file in same dir, then rename
            tmp_fd, tmp_path_str = tempfile.mkstemp(
                dir=path.parent, suffix=".tmp.json"
            )
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                Path(tmp_path_str).replace(path)  # atomic on same filesystem
            except Exception:
                # Clean up temp file if rename failed
                try:
                    Path(tmp_path_str).unlink(missing_ok=True)
                except Exception:
                    pass
                raise

            logger.debug(
                "web_cache: wrote query=%r results=%d path=%s",
                query[:60], len(results), path,
            )
        except Exception as exc:
            logger.warning("web_cache: write failed for query=%r: %s", query[:60], exc)

    def delete(self, query: str) -> None:
        """Remove a single cached entry. No-op if not found."""
        path = self._cache_path(query)
        try:
            path.unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("web_cache: delete failed: %s", exc)

    def clear(self) -> int:
        """
        Delete all cache files. Returns count of files deleted.
        Used in tests and manual cache busting.
        """
        count = 0
        if not self._cache_dir.exists():
            return 0
        for json_file in self._cache_dir.rglob("*.json"):
            try:
                json_file.unlink()
                count += 1
            except Exception:
                pass
        logger.info("web_cache: cleared %d entries", count)
        return count

    # ── Internal ──────────────────────────────────────────────────────────────

    def _cache_key(self, query: str) -> str:
        """sha256(provider + query) → hex string."""
        raw = f"{self._provider}:{query}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_path(self, query: str) -> Path:
        """Two-char prefix subdirectory to avoid giant flat directories."""
        key = self._cache_key(query)
        return self._cache_dir / key[:2] / f"{key}.json"


# ── Module-level singleton ────────────────────────────────────────────────────

_cache: WebCache | None = None


def get_web_cache() -> WebCache:
    """Return module-level WebCache singleton."""
    global _cache
    if _cache is None:
        _cache = WebCache()
    return _cache


def reset_web_cache() -> None:
    """Force singleton rebuild. Used in tests."""
    global _cache
    _cache = None