"""
agent/web_search_tool.py — Web search tool with provider abstraction.

Supports three providers:
  "mock"   — loads from data/fixtures/web_mock.jsonl (no API key, CI-safe)
import hashlib
  "tavily" — Tavily Search API (free tier: 1000/month, best for RAG)
  "exa"    — Exa Search API (semantic search, good for technical content)

Public interface:
    results = search_web_tool("vLLM throughput benchmarks")
    # → list[WebResult]

Each WebResult has:
    title, url, snippet, fetched_at, source, score

Full pipeline per call:
    sanitize_query
      → check cache (hit → return)
      → call provider API
      → sanitize each result
      → filter by domain allowlist
      → filter by age
      → write cache
      → return list[WebResult]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib

from ira.agent.sanitizer import sanitize_query, sanitize_web_result, is_allowed_domain
from ira.agent.web_cache import get_web_cache
from ira.settings import settings

logger = logging.getLogger(__name__)

# ── Fixtures path for mock provider ──────────────────────────────────────────
_MOCK_FIXTURE_PATH = Path("data/fixtures/web_mock.jsonl")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    fetched_at: str          # ISO 8601 UTC timestamp
    source: str              # "tavily" | "exa" | "mock"
    score: float = 0.0       # relevance score from provider (0.0 if not available)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WebResult":
        return cls(
            title=d.get("title", ""),
            url=d.get("url", ""),
            snippet=d.get("snippet", ""),
            fetched_at=d.get("fetched_at", ""),
            source=d.get("source", "unknown"),
            score=float(d.get("score", 0.0)),
        )


# ── Main entry point ──────────────────────────────────────────────────────────

def search_web_tool(
    query: str,
    max_results: int | None = None,
    *,
    provider: str | None = None,
    use_cache: bool = True,
) -> list[WebResult]:
    """
    Search the web and return sanitized, cached results.

    Args:
        query:       Raw research question. Will be sanitized before sending.
        max_results: Max results to return. Defaults to settings.WEB_SEARCH_MAX_RESULTS.
        provider:    Override settings.WEB_SEARCH_PROVIDER for this call.
        use_cache:   Set False to bypass cache (e.g. force-refresh).

    Returns:
        List of WebResult objects, possibly empty if nothing found or all filtered.
        Never raises — errors are logged and empty list returned.
    """
    _provider = provider or settings.WEB_SEARCH_PROVIDER
    _max = max_results if max_results is not None else settings.WEB_SEARCH_MAX_RESULTS

    # ── 1. Sanitize query ─────────────────────────────────────────────────────
    clean_query = sanitize_query(query)
    if not clean_query:
        logger.warning("search_web_tool: query empty after sanitization, original=%r", query[:60])
        return []

    logger.info(
        "search_web_tool",
        extra={"provider": _provider, "query": clean_query[:80], "max_results": _max},
    )

    # ── 2. Cache check ────────────────────────────────────────────────────────
    cache = get_web_cache()
    if use_cache:
        cached = cache.get(clean_query)
        if cached is not None:
            results = [WebResult.from_dict(r) for r in cached]
            logger.info("search_web_tool: cache hit, results=%d", len(results))
            return results[:_max]

    # ── 3. Call provider ──────────────────────────────────────────────────────
    try:
        raw_results = _call_provider(_provider, clean_query, _max)
    except Exception as exc:
        logger.error("search_web_tool: provider %r failed: %s", _provider, exc, exc_info=True)
        return []

    # ── 4. Sanitize + filter each result ─────────────────────────────────────
    allowlist = settings.web_domain_allowlist
    now_iso = datetime.now(timezone.utc).isoformat()
    results: list[WebResult] = []

    for raw in raw_results:
        url = raw.get("url", "")

        # Domain allowlist filter
        if not is_allowed_domain(url, allowlist):
            logger.debug("search_web_tool: domain blocked url=%s", url)
            continue

        # Sanitize title + snippet
        cleaned = sanitize_web_result(
            title=raw.get("title", ""),
            snippet=raw.get("snippet", ""),
            url=url,
        )
        if cleaned is None:
            logger.debug("search_web_tool: result dropped after sanitization url=%s", url)
            continue

        results.append(WebResult(
            title=cleaned["title"],
            url=url,
            snippet=cleaned["snippet"],
            fetched_at=now_iso,
            source=_provider,
            score=float(raw.get("score", 0.0)),
        ))

    logger.info(
        "search_web_tool: returning results=%d (raw=%d filtered=%d)",
        len(results), len(raw_results), len(raw_results) - len(results),
    )

    # ── 5. Write cache ────────────────────────────────────────────────────────
    if use_cache and results:
        cache.set(clean_query, [r.to_dict() for r in results])

    return results[:_max]


# ── Provider implementations ──────────────────────────────────────────────────

def _call_provider(provider: str, query: str, max_results: int) -> list[dict[str, Any]]:
    """Dispatch to the correct provider. Returns list of raw result dicts."""
    if provider == "mock":
        return _call_mock(query, max_results)
    elif provider == "tavily":
        return _call_tavily(query, max_results)
    elif provider == "exa":
        return _call_exa(query, max_results)
    else:
        raise ValueError(f"Unknown web search provider: {provider!r}")


def _call_mock(query: str, max_results: int) -> list[dict[str, Any]]:
    """
    Load results from data/fixtures/web_mock.jsonl.
    Returns all fixture entries (up to max_results) — no actual filtering by query.
    This is intentional: mock mode is for structural testing, not relevance testing.
    """
    fixture_path = _MOCK_FIXTURE_PATH
    if not fixture_path.exists():
        logger.warning("search_web_tool: mock fixture not found at %s", fixture_path)
        return []

    results: list[dict[str, Any]] = []
    try:
        for line in fixture_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                results.append(json.loads(line))
    except Exception as exc:
        logger.error("search_web_tool: failed to load mock fixture: %s", exc)
        return []

    logger.debug("search_web_tool: mock loaded %d fixtures", len(results))
    return results[:max_results]


def _call_tavily(query: str, max_results: int) -> list[dict[str, Any]]:
    """
    Call Tavily Search API.
    Requires: uv add tavily-python  +  WEB_SEARCH_API_KEY in .env

    Tavily response schema:
        {"results": [{"title", "url", "content", "score"}, ...]}
    """
    if not settings.WEB_SEARCH_API_KEY:
        raise RuntimeError(
            "WEB_SEARCH_API_KEY is not set. "
            "Add WEB_SEARCH_API_KEY=tvly-... to your .env file."
        )

    try:
        from tavily import TavilyClient  # lazy import
    except ImportError:
        raise RuntimeError(
            "tavily-python is not installed. Run: uv add tavily-python"
        )

    client = TavilyClient(api_key=settings.WEB_SEARCH_API_KEY)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",       # "basic" = faster + cheaper; "advanced" = deeper
        include_answer=False,       # we do our own synthesis
        include_raw_content=False,  # snippet only — raw content is too large
    )

    # Normalise to common schema: title, url, snippet, score
    return [
        {
            "title":   r.get("title", ""),
            "url":     r.get("url", ""),
            "snippet": r.get("content", ""),   # Tavily calls it "content"
            "score":   r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]


def _call_exa(query: str, max_results: int) -> list[dict[str, Any]]:
    """
    Call Exa Search API (semantic neural search).
    Requires: uv add exa-py  +  WEB_SEARCH_API_KEY in .env

    Exa response schema:
        result.results → [Result(title, url, text, score), ...]
    """
    if not settings.WEB_SEARCH_API_KEY:
        raise RuntimeError(
            "WEB_SEARCH_API_KEY is not set. "
            "Add WEB_SEARCH_API_KEY=your_exa_key to your .env file."
        )

    try:
        from exa_py import Exa  # lazy import
    except ImportError:
        raise RuntimeError(
            "exa-py is not installed. Run: uv add exa-py"
        )

    client = Exa(api_key=settings.WEB_SEARCH_API_KEY)
    response = client.search_and_contents(
        query=query,
        num_results=max_results,
        text={"max_characters": 1000},   # snippet length
        use_autoprompt=True,             # Exa rewrites query for better recall
    )

    return [
        {
            "title":   r.title or "",
            "url":     r.url or "",
            "snippet": r.text or "",
            "score":   r.score or 0.0,
        }
        for r in response.results
    ]


# ── Evidence pack converter ───────────────────────────────────────────────────

def web_results_to_evidence_packs(results: list[WebResult]) -> list[dict[str, Any]]:
    """
    Convert WebResults to evidence_pack dicts compatible with AgentState.

    Web results use the same dict shape as internal EvidencePacks so
    synthesize_answer handles them identically — no special cases.
    source_type="web" distinguishes them in the final AnswerResponse.
    """
    packs = []
    for i, r in enumerate(results):
        packs.append({
            # Use URL as chunk_id for web results (stable, unique)
            "chunk_id":      f"web_{hashlib.sha256(r.url.encode()).hexdigest()[:12]}",
            "doc_id":        f"web_{i}",
            "title":         r.title,
            "url":           r.url,
            "section":       None,
            "child_text":    r.snippet,
            "parent_text":   r.snippet,   # web has no parent — use snippet for both
            "rrf_score":     r.score,
            "reranker_score": r.score,    # no reranker for web; use provider score
            "in_both":       False,
            "source_type":   "web",       # distinguishes from internal in AnswerResponse
        })
    return packs


# Need hashlib for chunk_id generation