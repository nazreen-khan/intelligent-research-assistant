# File: src/ira/retrieval/bm25_index.py
"""
Day 6 — BM25 keyword index using SQLite FTS5.

Design rationale:
  SQLite FTS5 was chosen over rank_bm25 for three reasons:
    1. Persistence   — single .db file, no pickle, no RAM overhead at startup
    2. Speed         — C extension, not pure Python; handles 100K+ chunks easily
    3. Zero deps     — sqlite3 is Python stdlib, works identically on Windows/Linux/Mac

  The default FTS5 tokenizer (Porter stemmer) destroys technical terms:
    "FlashAttention-2" → "flashatteint"  WRONG
    "FP8"              → "fp"            WRONG
    "H100"             → "h"             WRONG

  Solution: a custom technical-aware pre-tokenizer that:
    - Lowercases everything
    - Preserves hyphenated terms as single tokens  (bge-small-en-v1.5, FlashAttention-2)
    - Preserves alphanumeric units as single tokens (FP8, H100, INT4, 2bit, 3x)
    - Strips Markdown syntax before tokenizing
    - Uses FTS5 with `tokenize="unicode61"` (no stemming) so stored tokens match exactly

Schema (two tables):
  fts_chunks   — FTS5 virtual table for BM25 full-text search
  chunk_meta   — regular table with payload fields, joined on chunk_id for results

Usage:
    from ira.retrieval.bm25_index import BM25Index
    idx = BM25Index("data/bm25/bm25_index.db")
    idx.build_from_chunks([{"chunk_id": ..., "text": ..., ...}, ...])
    results = idx.query("FlashAttention-2 FP8", top_n=10)
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Schema constants ──────────────────────────────────────────────────────────

_DDL_META = """
CREATE TABLE IF NOT EXISTS chunk_meta (
    chunk_id            TEXT PRIMARY KEY,
    parent_id           TEXT NOT NULL,
    doc_id              TEXT NOT NULL,
    title               TEXT NOT NULL,
    url                 TEXT,
    section             TEXT NOT NULL,
    token_count         INTEGER NOT NULL DEFAULT 0,
    is_table            INTEGER NOT NULL DEFAULT 0,
    is_code             INTEGER NOT NULL DEFAULT 0,
    source_type         TEXT NOT NULL DEFAULT 'internal',
    content_sha         TEXT NOT NULL DEFAULT ''
);
"""

# FTS5 virtual table.
# tokenize="unicode61 remove_diacritics 2" — unicode-aware, no stemming,
# so "FlashAttention-2" stored as "flashattention-2" and matched exactly.
# The `text_tokens` column stores our pre-tokenized string (space-separated).
# The `text_raw` column is stored for snippet/preview retrieval.
_DDL_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_id    UNINDEXED,
    text_tokens,
    text_raw    UNINDEXED,
    tokenize    = "unicode61 remove_diacritics 2"
);
"""

_DDL_META_IDX = """
CREATE INDEX IF NOT EXISTS idx_chunk_meta_doc_id ON chunk_meta(doc_id);
"""


# ── Technical tokenizer ───────────────────────────────────────────────────────

# Markdown noise — strip markers but PRESERVE inner text.
# Key fix: bold/italic keeps inner text so **FlashAttention-2** → "FlashAttention-2"
_MD_NOISE = re.compile(
    r"```.*?```"                  # fenced code blocks  — remove entirely
    r"|`([^`\n]+)`"               # inline code         — keep inner text (group 1)
    r"|\*{1,3}([^*\n]+)\*{1,3}"  # bold/italic         — keep inner text (group 2)
    r"|#{1,6}\s"                  # heading markers     — remove marker only
    r"|\[([^\]]+)\]\([^)]+\)"  # markdown links      — keep link text (group 3)
    r"|!\[[^\]]*\]\([^)]+\)"   # images              — remove entirely
    r"|\|[-:]+\|"                # table separators
    r"|---+|===+",
    re.DOTALL,
)


def _md_sub(m: re.Match) -> str:
    """Return first non-None capture group, or a single space."""
    for g in m.groups():
        if g is not None:
            return f" {g} "
    return " "

# Technical tokens: hyphenated terms, version strings, units like FP8/H100/INT4
# Must come BEFORE generic word splitting
_TECH_TOKEN = re.compile(
    r"[A-Za-z0-9]+"           # base word/number
    r"(?:[-_.][A-Za-z0-9]+)+" # one or more hyphen/dot/underscore segments
    r"|[A-Za-z]+\d+"          # letters followed by digits: FP8, H100, INT4, GPT2
    r"|\d+[A-Za-z]+"          # digits followed by letters: 2bit, 3x, 4bit
    r"|\d+\.\d+"              # decimal numbers: 1.5, 3.14
)

# Generic word tokenizer (after tech tokens extracted)
_WORD = re.compile(r"[a-z0-9]{2,}")  # at least 2 chars, lowercase


def tokenize(text: str) -> list[str]:
    """
    Technical-aware tokenizer for LLM efficiency domain text.

    Strategy:
      1. Strip markdown syntax
      2. Lowercase
      3. Extract technical tokens first (hyphenated, versioned, unit terms)
      4. Extract remaining generic words from the residual text
      5. Deduplicate while preserving order

    Examples:
      "FlashAttention-2 on H100 with FP8"
        → ["flashattention-2", "h100", "fp8", "on", "with"]

      "BAAI/bge-small-en-v1.5 achieves 384-dim"
        → ["baai", "bge-small-en-v1.5", "achieves", "384-dim"]

      "3x speedup, 2bit quantization"
        → ["3x", "speedup", "2bit", "quantization"]
    """
    if not text:
        return []

    # Step 1: strip markdown noise, replace link text with just the text
    cleaned = _MD_NOISE.sub(_md_sub, text)

    # Step 2: lowercase
    cleaned = cleaned.lower()

    # Step 3: extract tech tokens and track their positions
    tokens: list[str] = []
    seen: set[str] = set()
    residual = cleaned

    tech_spans: list[tuple[int, int, str]] = []
    for m in _TECH_TOKEN.finditer(cleaned):
        tok = m.group(0)
        tech_spans.append((m.start(), m.end(), tok))

    # Build residual text with tech token positions blanked out
    residual_chars = list(cleaned)
    for start, end, tok in tech_spans:
        for i in range(start, end):
            residual_chars[i] = " "
        if tok not in seen:
            tokens.append(tok)
            seen.add(tok)

    residual = "".join(residual_chars)

    # Step 4: generic words from residual
    for m in _WORD.finditer(residual):
        tok = m.group(0)
        if tok not in seen:
            tokens.append(tok)
            seen.add(tok)

    return tokens


def tokens_to_text(tokens: list[str]) -> str:
    """Join tokens into a space-separated string for FTS5 storage."""
    return " ".join(tokens)


def tokenize_for_query(query: str) -> str:
    """
    Tokenize a query and join for FTS5 MATCH syntax.
    Each token becomes a prefix search term (token*) to catch plurals/variants.
    Multi-token query uses implicit AND.
    """
    tokens = tokenize(query)
    if not tokens:
        return '""'  # empty FTS5 match
    # Wrap each token in quotes to handle special chars (hyphens etc.)
    return " ".join(f'"{t}"' for t in tokens)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BM25Result:
    chunk_id: str
    parent_id: str
    doc_id: str
    title: str
    url: Optional[str]
    section: str
    text_preview: str       # first 300 chars of raw text
    bm25_score: float       # raw FTS5 bm25() score (negative; more negative = better)
    rank: int               # 1-based rank in result set
    token_count: int
    is_table: bool
    is_code: bool
    source_type: str


# ── BM25Index class ───────────────────────────────────────────────────────────

class BM25Index:
    """
    SQLite FTS5-backed BM25 index for chunk retrieval.

    Thread safety: SQLite in WAL mode is safe for concurrent reads,
    but writes should be serialised (fine for our batch-build use case).
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    # ── Connection ────────────────────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,   # safe: we serialise writes ourselves
        )
        conn.row_factory = sqlite3.Row

        # Performance pragmas — safe on Windows, survive crashes gracefully
        conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads during writes
        conn.execute("PRAGMA synchronous=NORMAL")  # fsync on WAL checkpoint only
        conn.execute("PRAGMA cache_size=-32000")   # 32 MB page cache
        conn.execute("PRAGMA temp_store=MEMORY")   # temp tables in RAM

        # Create tables if not present
        conn.execute(_DDL_META)
        conn.execute(_DDL_FTS)
        conn.execute(_DDL_META_IDX)
        conn.commit()

        self._conn = conn
        logger.debug("BM25Index opened: %s", self.db_path)
        return conn

    def close(self) -> None:
        """Explicitly close the SQLite connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 500,
    ) -> dict[str, int]:
        """
        Insert or replace chunks into the BM25 index.

        Dedup strategy: chunk_id is the PRIMARY KEY in chunk_meta.
        We check content_sha — skip if unchanged, replace if text changed.

        Returns: {"inserted": N, "updated": N, "skipped": N, "total": N}
        """
        if not chunks:
            return {"inserted": 0, "updated": 0, "skipped": 0, "total": 0}

        conn = self._get_conn()

        # Fetch existing content_shas in one query
        all_ids = [c["chunk_id"] for c in chunks]
        placeholders = ",".join("?" * len(all_ids))
        existing = {
            row["chunk_id"]: row["content_sha"]
            for row in conn.execute(
                f"SELECT chunk_id, content_sha FROM chunk_meta WHERE chunk_id IN ({placeholders})",
                all_ids,
            )
        }

        inserted = updated = skipped = 0
        meta_rows: list[tuple] = []
        fts_rows: list[tuple] = []

        for chunk in chunks:
            cid = chunk["chunk_id"]
            text = chunk.get("text", "")
            import hashlib
            sha = hashlib.sha256(text.encode()).hexdigest()

            if cid in existing:
                if existing[cid] == sha:
                    skipped += 1
                    continue
                else:
                    # Will be replaced via INSERT OR REPLACE
                    updated += 1
            else:
                inserted += 1

            tokens = tokenize(text)
            token_text = tokens_to_text(tokens)

            meta_rows.append((
                cid,
                chunk.get("parent_id", ""),
                chunk.get("doc_id", ""),
                chunk.get("title", ""),
                chunk.get("url"),
                chunk.get("section", ""),
                chunk.get("token_count", 0),
                int(chunk.get("is_table", False)),
                int(chunk.get("is_code", False)),
                chunk.get("source_type", "internal"),
                sha,
            ))
            fts_rows.append((cid, token_text, text[:500]))  # cap raw text at 500 chars

        if not meta_rows:
            return {"inserted": 0, "updated": 0, "skipped": skipped, "total": len(chunks)}

        # Delete old FTS rows for chunks being updated (FTS5 doesn't support UPDATE)
        update_ids = [c["chunk_id"] for c in chunks
                      if c["chunk_id"] in existing
                      and existing[c["chunk_id"]] != hashlib.sha256(c.get("text","").encode()).hexdigest()]
        if update_ids:
            placeholders_u = ",".join("?" * len(update_ids))
            conn.execute(
                f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders_u})",
                update_ids,
            )

        # Batch upsert meta + fts rows
        for i in range(0, len(meta_rows), batch_size):
            conn.executemany(
                """INSERT OR REPLACE INTO chunk_meta
                   (chunk_id, parent_id, doc_id, title, url, section,
                    token_count, is_table, is_code, source_type, content_sha)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                meta_rows[i: i + batch_size],
            )
            conn.executemany(
                "INSERT INTO fts_chunks (chunk_id, text_tokens, text_raw) VALUES (?,?,?)",
                fts_rows[i: i + batch_size],
            )
            conn.commit()

        logger.info(
            "BM25 upsert: inserted=%d updated=%d skipped=%d total=%d",
            inserted, updated, skipped, len(chunks),
        )
        return {"inserted": inserted, "updated": updated, "skipped": skipped, "total": len(chunks)}

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_n: int = 20,
        filter_doc_ids: Optional[list[str]] = None,
        exclude_code: bool = False,
        exclude_tables: bool = False,
    ) -> list[BM25Result]:
        """
        Query the FTS5 index using BM25 ranking.

        FTS5's bm25() auxiliary function returns negative scores
        (more negative = better match). We negate to get positive scores
        for consistent comparison with Qdrant cosine scores in Day 7.

        Multi-term queries use implicit AND (all terms must appear).
        Technical terms are preserved exactly as indexed.
        """
        conn = self._get_conn()

        fts_query = tokenize_for_query(query_text)
        if fts_query == '""':
            return []

        # Build optional WHERE clause for filters on chunk_meta
        where_clauses: list[str] = []
        params: list[Any] = [fts_query]

        if filter_doc_ids:
            placeholders = ",".join("?" * len(filter_doc_ids))
            where_clauses.append(f"m.doc_id IN ({placeholders})")
            params.extend(filter_doc_ids)

        if exclude_code:
            where_clauses.append("m.is_code = 0")

        if exclude_tables:
            where_clauses.append("m.is_table = 0")

        where_sql = ("AND " + " AND ".join(where_clauses)) if where_clauses else ""

        sql = f"""
            SELECT
                f.chunk_id,
                f.text_raw,
                -bm25(fts_chunks) AS bm25_score,
                m.parent_id,
                m.doc_id,
                m.title,
                m.url,
                m.section,
                m.token_count,
                m.is_table,
                m.is_code,
                m.source_type
            FROM fts_chunks f
            JOIN chunk_meta m ON f.chunk_id = m.chunk_id
            WHERE fts_chunks MATCH ?
            {where_sql}
            ORDER BY bm25(fts_chunks)
            LIMIT ?
        """
        params.append(top_n)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            # Common cause: FTS5 query syntax error (e.g. empty after tokenizing)
            logger.warning("BM25 query error for %r: %s", query_text, e)
            return []

        return [
            BM25Result(
                chunk_id=row["chunk_id"],
                parent_id=row["parent_id"],
                doc_id=row["doc_id"],
                title=row["title"],
                url=row["url"],
                section=row["section"],
                text_preview=row["text_raw"][:300],
                bm25_score=float(row["bm25_score"]),
                rank=rank + 1,
                token_count=row["token_count"],
                is_table=bool(row["is_table"]),
                is_code=bool(row["is_code"]),
                source_type=row["source_type"],
            )
            for rank, row in enumerate(rows)
        ]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return index statistics."""
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) FROM chunk_meta").fetchone()[0]
            docs = conn.execute("SELECT COUNT(DISTINCT doc_id) FROM chunk_meta").fetchone()[0]
            db_size_mb = round(self.db_path.stat().st_size / 1024 / 1024, 2) if self.db_path.exists() else 0
            return {
                "chunks": total,
                "docs": docs,
                "db_path": str(self.db_path),
                "db_size_mb": db_size_mb,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_indexed_ids(self) -> set[str]:
        """Return all chunk_ids currently in the index."""
        conn = self._get_conn()
        rows = conn.execute("SELECT chunk_id FROM chunk_meta").fetchall()
        return {row[0] for row in rows}


# ── Module-level singleton ─────────────────────────────────────────────────────

_default_bm25: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Return the module-level singleton BM25Index, built from settings."""
    global _default_bm25
    if _default_bm25 is None:
        from ira.settings import settings
        _default_bm25 = BM25Index(settings.bm25_db_path)
    return _default_bm25


def reset_bm25_index() -> None:
    """Reset singleton — useful in tests."""
    global _default_bm25
    if _default_bm25 is not None:
        _default_bm25.close()
    _default_bm25 = None