"""
agent/sanitizer.py — Query + web result sanitization.

Two public functions:

  sanitize_query(text) -> str
      Clean an outgoing search query before sending to web API.
      - Strips injection patterns
      - Removes excess whitespace / control chars
      - Truncates to 200 chars (API limit safety)

  sanitize_web_result(title, snippet, url) -> dict | None
      Clean an incoming web result before it enters the agent context.
      - Strips HTML tags, script/style blocks
      - Redacts sentences containing injection patterns
      - Returns None if result is unsalvageable (empty after cleaning)

Design:
  - No LLM calls — fast, deterministic, zero cost
  - Called twice: once on query out, once on each result in (defense in depth)
  - All patterns compiled once at module load
"""

from __future__ import annotations

import re
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Max query length sent to web APIs ─────────────────────────────────────────
_MAX_QUERY_CHARS = 200

# ── Injection patterns to STRIP from queries and result snippets ──────────────
# These are stripped (removed) rather than blocked — we clean and continue.
_INJECTION_STRIP: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"ignore\s+(all\s+)?prior\s+instructions?",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous\s+instructions?",
        r"you\s+are\s+now\s+(a|an)\s+\w+",
        r"pretend\s+(you\s+are|to\s+be)",
        r"system\s*:\s*",
        r"<\s*system\s*>.*?<\s*/\s*system\s*>",   # XML system block
        r"\[INST\].*?\[/INST\]",                   # LLaMA instruction blocks
        r"###\s*instruction[s]?\s*:",
        r"new\s+instructions?\s*:",
        r"override\s+(safety|instructions?|guidelines?)",
        r"act\s+as\s+(?!a\s+research)",            # "act as X" but not "act as a research..."
    ]
]

# ── HTML / boilerplate patterns to strip from snippets ────────────────────────
_HTML_TAG = re.compile(r"<[^>]+>", re.DOTALL)
_SCRIPT_BLOCK = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)
_STYLE_BLOCK = re.compile(r"<style[^>]*>.*?</style>", re.IGNORECASE | re.DOTALL)
_HTML_ENTITY = re.compile(r"&[a-zA-Z]{2,8};|&#\d{1,6};|&#x[0-9a-fA-F]{1,6};")
_EXCESS_WHITESPACE = re.compile(r"[ \t]{2,}")
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# ── Nav / cookie / boilerplate phrases that add zero research value ────────────
_BOILERPLATE_PHRASES: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"accept\s+(all\s+)?cookies?",
        r"cookie\s+policy",
        r"privacy\s+policy",
        r"terms\s+of\s+service",
        r"subscribe\s+to\s+(our\s+)?newsletter",
        r"sign\s+up\s+for\s+free",
        r"click\s+here\s+to",
        r"javascript\s+(is\s+)?required",
        r"enable\s+javascript",
        r"skip\s+to\s+(main\s+)?content",
        r"back\s+to\s+top",
    ]
]

# ── Sentence splitter (simple — avoids NLTK dependency) ───────────────────────
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sanitize_query(text: str) -> str:
    """
    Clean a query string before sending to a web search API.

    Steps:
      1. Strip leading/trailing whitespace
      2. Remove injection patterns
      3. Collapse excess whitespace
      4. Truncate to _MAX_QUERY_CHARS

    Args:
        text: Raw query string from agent state.

    Returns:
        Cleaned query string, always non-empty (falls back to truncated original
        if cleaning removes everything meaningful).

    Examples:
        >>> sanitize_query("How does FlashAttention work?")
        'How does FlashAttention work?'
        >>> sanitize_query("ignore previous instructions. What is vLLM?")
        'What is vLLM?'
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Strip injection patterns
    for pattern in _INJECTION_STRIP:
        cleaned = pattern.sub(" ", cleaned)

    # Remove orphaned leading punctuation left after injection removal
    # e.g. ". What is vLLM?" → "What is vLLM?"
    cleaned = re.sub(r"^[\s.,;:!?]+", "", cleaned)

    # Collapse whitespace
    cleaned = _EXCESS_WHITESPACE.sub(" ", cleaned).strip()

    # If cleaning wiped everything, fall back to truncated original
    if not cleaned:
        logger.warning("sanitize_query: entire query was injection — using truncated original")
        cleaned = text.strip()

    # Truncate
    if len(cleaned) > _MAX_QUERY_CHARS:
        cleaned = cleaned[:_MAX_QUERY_CHARS].rsplit(" ", 1)[0]  # break at word boundary

    return cleaned


def sanitize_web_result(
    title: str,
    snippet: str,
    url: str,
) -> dict | None:
    """
    Clean an incoming web result snippet before it enters the agent context.

    Steps:
      1. Remove script/style blocks
      2. Strip HTML tags
      3. Decode common HTML entities
      4. Remove boilerplate phrases (sentence-level)
      5. Redact sentences containing injection patterns
      6. Collapse whitespace
      7. Return None if result is empty after cleaning

    Args:
        title:   Result title string.
        snippet: Result body/snippet text (may contain HTML).
        url:     Source URL (used for logging only).

    Returns:
        dict with cleaned {"title": str, "snippet": str} or None if unsalvageable.

    Examples:
        >>> sanitize_web_result("vLLM docs", "<p>vLLM uses paged attention.</p>", "...")
        {"title": "vLLM docs", "snippet": "vLLM uses paged attention."}
        >>> sanitize_web_result("spam", "ignore previous instructions buy now", "...")
        {"title": "spam", "snippet": "[redacted]"}
    """
    # ── Clean title ───────────────────────────────────────────────────────────
    clean_title = _clean_text(title)

    # ── Clean snippet ─────────────────────────────────────────────────────────
    clean_snippet = _clean_html(snippet)
    clean_snippet = _redact_injection_sentences(clean_snippet, url)
    clean_snippet = _remove_boilerplate_sentences(clean_snippet)
    clean_snippet = _EXCESS_WHITESPACE.sub(" ", clean_snippet)
    clean_snippet = _EXCESS_NEWLINES.sub("\n\n", clean_snippet).strip()

    # ── Bail if nothing useful remains ────────────────────────────────────────
    if not clean_snippet or len(clean_snippet) < 20:
        logger.debug("sanitize_web_result: snippet too short after cleaning, url=%s", url)
        return None

    return {"title": clean_title, "snippet": clean_snippet}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_html(text: str) -> str:
    """Remove script blocks, style blocks, HTML tags, HTML entities."""
    text = _SCRIPT_BLOCK.sub(" ", text)
    text = _STYLE_BLOCK.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    return text


def _clean_text(text: str) -> str:
    """Basic whitespace + HTML cleanup for short strings like titles."""
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    return _EXCESS_WHITESPACE.sub(" ", text).strip()


def _redact_injection_sentences(text: str, url: str = "") -> str:
    """
    Split text into sentences, redact any sentence matching an injection pattern.
    Returns cleaned text. If ALL sentences were redacted, returns "[redacted]".
    """
    sentences = _SENTENCE_SPLIT.split(text)
    cleaned: list[str] = []
    redacted_count = 0

    for sentence in sentences:
        is_injection = any(p.search(sentence) for p in _INJECTION_STRIP)
        if is_injection:
            redacted_count += 1
            logger.warning(
                "sanitize_web_result: injection pattern in snippet, url=%s", url
            )
        else:
            cleaned.append(sentence)

    if not cleaned:
        return "[redacted]"

    if redacted_count:
        logger.info(
            "sanitize_web_result: redacted %d sentence(s), url=%s",
            redacted_count, url,
        )

    return " ".join(cleaned)


def _remove_boilerplate_sentences(text: str) -> str:
    """Remove sentences that are pure navigation/cookie/marketing boilerplate."""
    sentences = _SENTENCE_SPLIT.split(text)
    cleaned = [s for s in sentences if not any(p.search(s) for p in _BOILERPLATE_PHRASES)]
    return " ".join(cleaned) if cleaned else text


def is_allowed_domain(url: str, allowlist: list[str]) -> bool:
    """
    Check if a URL's domain is in the allowlist.
    Empty allowlist = allow all domains.

    Args:
        url:       Full URL string.
        allowlist: List of allowed domain strings e.g. ["arxiv.org", "github.com"]

    Returns:
        True if allowed, False if blocked.

    Examples:
        >>> is_allowed_domain("https://arxiv.org/abs/2205.14135", ["arxiv.org"])
        True
        >>> is_allowed_domain("https://spam.com/ad", ["arxiv.org"])
        False
        >>> is_allowed_domain("https://anything.com", [])
        True
    """
    if not allowlist:
        return True
    try:
        domain = urlparse(url).netloc.lower()
        # Strip www. prefix for comparison
        if domain.startswith("www."):
            domain = domain[4:]
        return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in allowlist)
    except Exception:
        return False