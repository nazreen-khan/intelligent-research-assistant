# File: src/ira/retrieval/bm25_runner.py
"""
Day 6 — BM25 index runner: chunks/ → BM25Index build pipeline.

Mirrors the structure of index_runner.py (Day 5) so the two indexes
are built and queried with identical call patterns — important for
Day 7 hybrid fusion where both are called together.

Usage (CLI):
    uv run python -m ira index bm25-build --chunks data/chunks
    uv run python -m ira index bm25-build --chunks data/chunks --doc-id arxiv_2205.14135v1

Usage (Python):
    from ira.retrieval.bm25_runner import build_bm25_index
    result = build_bm25_index(chunks_root=Path("data/chunks"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class BM25DocResult:
    doc_id: str
    ok: bool
    chunks_read: int = 0
    inserted: int = 0
    updated: int = 0
    skipped: int = 0
    error: Optional[str] = None


@dataclass
class BM25BuildResult:
    docs: list[BM25DocResult] = field(default_factory=list)
    index_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_inserted(self) -> int:
        return sum(d.inserted for d in self.docs)

    @property
    def total_updated(self) -> int:
        return sum(d.updated for d in self.docs)

    @property
    def total_skipped(self) -> int:
        return sum(d.skipped for d in self.docs)

    @property
    def total_chunks(self) -> int:
        return sum(d.chunks_read for d in self.docs)

    @property
    def failed_docs(self) -> list[str]:
        return [d.doc_id for d in self.docs if not d.ok]


# ── Chunk loading (shared with index_runner pattern) ─────────────────────────

def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON at %s line %d: %s", chunks_path, line_no, e)
    return chunks


def _iter_doc_dirs(
    chunks_root: Path,
    doc_id_filter: Optional[str] = None,
) -> Generator[Path, None, None]:
    if not chunks_root.exists():
        logger.warning("Chunks root does not exist: %s", chunks_root)
        return
    for doc_dir in sorted(chunks_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        if doc_id_filter and doc_dir.name != doc_id_filter:
            continue
        if (doc_dir / "chunks.jsonl").exists():
            yield doc_dir


# ── Core build function ───────────────────────────────────────────────────────

def build_bm25_index(
    chunks_root: Path,
    *,
    doc_id: Optional[str] = None,
    batch_size: int = 500,
    bm25_index=None,
) -> BM25BuildResult:
    """
    Build or incrementally update the BM25 index from data/chunks/.

    chunks_root  : path to data/chunks/
    doc_id       : only index this specific doc (incremental add)
    batch_size   : SQLite executemany batch size
    bm25_index   : BM25Index instance (uses singleton if None)

    Returns BM25BuildResult with per-doc stats.
    """
    if bm25_index is None:
        from ira.retrieval.bm25_index import get_bm25_index
        bm25_index = get_bm25_index()

    result = BM25BuildResult()

    for doc_dir in _iter_doc_dirs(chunks_root, doc_id_filter=doc_id):
        doc_result = _index_one_doc(
            doc_dir=doc_dir,
            bm25_index=bm25_index,
            batch_size=batch_size,
        )
        result.docs.append(doc_result)
        status = "ok" if doc_result.ok else f"FAIL: {doc_result.error}"
        logger.info(
            "BM25 indexed %s: %s | chunks=%d inserted=%d updated=%d skipped=%d",
            doc_result.doc_id, status,
            doc_result.chunks_read,
            doc_result.inserted,
            doc_result.updated,
            doc_result.skipped,
        )

    result.index_stats = bm25_index.stats()
    logger.info(
        "BM25 build complete: docs=%d total_chunks=%d inserted=%d updated=%d skipped=%d",
        len(result.docs),
        result.total_chunks,
        result.total_inserted,
        result.total_updated,
        result.total_skipped,
    )
    return result


def _index_one_doc(
    doc_dir: Path,
    bm25_index,
    batch_size: int,
) -> BM25DocResult:
    doc_id = doc_dir.name
    try:
        chunks = _load_chunks(doc_dir / "chunks.jsonl")
        if not chunks:
            return BM25DocResult(doc_id=doc_id, ok=True, chunks_read=0)

        counts = bm25_index.upsert_chunks(chunks, batch_size=batch_size)
        return BM25DocResult(
            doc_id=doc_id,
            ok=True,
            chunks_read=len(chunks),
            inserted=counts["inserted"],
            updated=counts["updated"],
            skipped=counts["skipped"],
        )
    except Exception as e:
        logger.exception("Failed to BM25-index doc %s", doc_id)
        return BM25DocResult(doc_id=doc_id, ok=False, error=str(e))


# ── Query helper (mirrors index_runner.query_index) ───────────────────────────

def query_bm25(
    query_text: str,
    top_n: int = 10,
    bm25_index=None,
    filter_doc_ids: Optional[list[str]] = None,
    exclude_code: bool = False,
    exclude_tables: bool = False,
) -> list[dict[str, Any]]:
    """
    Query the BM25 index. Returns list of result dicts for CLI display.
    """
    if bm25_index is None:
        from ira.retrieval.bm25_index import get_bm25_index
        bm25_index = get_bm25_index()

    results = bm25_index.query(
        query_text,
        top_n=top_n,
        filter_doc_ids=filter_doc_ids,
        exclude_code=exclude_code,
        exclude_tables=exclude_tables,
    )

    return [
        {
            "rank": r.rank,
            "bm25_score": round(r.bm25_score, 4),
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "title": r.title,
            "section": r.section,
            "url": r.url,
            "is_code": r.is_code,
            "is_table": r.is_table,
            "text_preview": r.text_preview,
        }
        for r in results
    ]