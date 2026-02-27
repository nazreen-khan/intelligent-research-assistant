# File: src/ira/retrieval/index_runner.py
"""
Day 5 — Index runner: chunks/ → Embedder → QdrantIndex

Reads all chunks.jsonl files from data/chunks/<doc_id>/chunks.jsonl,
embeds text in batches, upserts to Qdrant with dedup.

Usage (CLI):
    uv run python -m ira index build --chunks data/chunks
    uv run python -m ira index build --chunks data/chunks --doc-id arxiv_2205.14135v1

Usage (Python):
    from ira.retrieval.index_runner import build_index
    result = build_index(chunks_root=Path("data/chunks"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class DocIndexResult:
    doc_id: str
    ok: bool
    chunks_read: int = 0
    upserted: int = 0
    skipped: int = 0
    error: Optional[str] = None


@dataclass
class IndexBuildResult:
    docs: list[DocIndexResult]
    collection_info: dict[str, Any]

    @property
    def total_upserted(self) -> int:
        return sum(d.upserted for d in self.docs)

    @property
    def total_skipped(self) -> int:
        return sum(d.skipped for d in self.docs)

    @property
    def total_chunks(self) -> int:
        return sum(d.chunks_read for d in self.docs)

    @property
    def failed_docs(self) -> list[str]:
        return [d.doc_id for d in self.docs if not d.ok]


# ── Chunk loading ─────────────────────────────────────────────────────────────

def _load_chunks_from_file(chunks_path: Path) -> list[dict[str, Any]]:
    """Load all chunks from a chunks.jsonl file."""
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


def _iter_doc_dirs(chunks_root: Path, doc_id_filter: Optional[str] = None) -> Generator[Path, None, None]:
    """Yield doc directories from chunks_root, optionally filtered by doc_id."""
    if not chunks_root.exists():
        logger.warning("Chunks root does not exist: %s", chunks_root)
        return

    for doc_dir in sorted(chunks_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        if doc_id_filter and doc_dir.name != doc_id_filter:
            continue
        chunks_file = doc_dir / "chunks.jsonl"
        if not chunks_file.exists():
            logger.debug("No chunks.jsonl in %s, skipping", doc_dir)
            continue
        yield doc_dir


# ── Core build function ───────────────────────────────────────────────────────

def build_index(
    chunks_root: Path,
    *,
    doc_id: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    upsert_batch_size: int = 256,
    embedder=None,
    index=None,
) -> IndexBuildResult:
    """
    Build (or incrementally update) the Qdrant index from chunks/.

    chunks_root     : path to data/chunks/
    doc_id          : if set, only index this specific doc (useful for incremental adds)
    embed_batch_size: override embedder batch size
    upsert_batch_size: how many points per Qdrant upsert call
    embedder        : Embedder instance (uses singleton if None)
    index           : QdrantIndex instance (uses singleton if None)

    Returns IndexBuildResult with per-doc stats.
    """
    # ── Resolve dependencies ──────────────────────────────────────────────────
    if embedder is None:
        from ira.retrieval.embedder import get_embedder
        embedder = get_embedder()

    if index is None:
        from ira.retrieval.qdrant_index import get_index
        index = get_index(vector_dim=embedder.embedding_dim)

    # ── Ensure collection exists ──────────────────────────────────────────────
    index.ensure_collection()

    # ── Process each doc ──────────────────────────────────────────────────────
    doc_results: list[DocIndexResult] = []

    for doc_dir in _iter_doc_dirs(chunks_root, doc_id_filter=doc_id):
        result = _index_one_doc(
            doc_dir=doc_dir,
            embedder=embedder,
            index=index,
            embed_batch_size=embed_batch_size,
            upsert_batch_size=upsert_batch_size,
        )
        doc_results.append(result)

        status = "ok" if result.ok else f"FAIL: {result.error}"
        logger.info(
            "Indexed %s: %s | chunks=%d upserted=%d skipped=%d",
            result.doc_id, status, result.chunks_read, result.upserted, result.skipped,
        )

    # ── Collection stats ──────────────────────────────────────────────────────
    collection_info = index.collection_info()

    build_result = IndexBuildResult(docs=doc_results, collection_info=collection_info)

    logger.info(
        "Index build complete: docs=%d total_chunks=%d upserted=%d skipped=%d failed=%s",
        len(doc_results),
        build_result.total_chunks,
        build_result.total_upserted,
        build_result.total_skipped,
        build_result.failed_docs or "none",
    )

    return build_result


def _index_one_doc(
    doc_dir: Path,
    embedder,
    index,
    embed_batch_size: Optional[int],
    upsert_batch_size: int,
) -> DocIndexResult:
    """Index a single doc's chunks.jsonl into Qdrant."""
    doc_id = doc_dir.name
    chunks_path = doc_dir / "chunks.jsonl"

    try:
        chunks = _load_chunks_from_file(chunks_path)
        if not chunks:
            logger.warning("No chunks found in %s", chunks_path)
            return DocIndexResult(doc_id=doc_id, ok=True, chunks_read=0)

        logger.info("Embedding %d chunks for doc: %s", len(chunks), doc_id)

        # Extract texts for embedding
        texts = [c.get("text", "") for c in chunks]

        # Override batch size if provided
        original_batch_size = embedder.batch_size
        if embed_batch_size is not None:
            embedder.batch_size = embed_batch_size

        try:
            vectors: np.ndarray = embedder.embed_passages(texts)
        finally:
            embedder.batch_size = original_batch_size

        # Upsert to Qdrant
        counts = index.upsert_chunks(
            chunks=chunks,
            vectors=vectors,
            batch_size=upsert_batch_size,
        )

        return DocIndexResult(
            doc_id=doc_id,
            ok=True,
            chunks_read=len(chunks),
            upserted=counts["upserted"],
            skipped=counts["skipped"],
        )

    except Exception as e:
        logger.exception("Failed to index doc %s", doc_id)
        return DocIndexResult(doc_id=doc_id, ok=False, error=str(e))


# ── Query helper ──────────────────────────────────────────────────────────────

def query_index(
    query_text: str,
    top_k: int = 5,
    embedder=None,
    index=None,
    score_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Query the index with a text query. Returns a list of result dicts.
    Suitable for CLI display and API response building.
    """
    if embedder is None:
        from ira.retrieval.embedder import get_embedder
        embedder = get_embedder()

    if index is None:
        from ira.retrieval.qdrant_index import get_index
        index = get_index(vector_dim=embedder.embedding_dim)

    results = index.search_by_text(
        query=query_text,
        embedder=embedder,
        top_k=top_k,
        score_threshold=score_threshold,
    )

    return [
        {
            "rank": rank + 1,
            "score": round(r.score, 4),
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "title": r.title,
            "section": r.section,
            "url": r.url,
            "is_code": r.is_code,
            "is_table": r.is_table,
            "text_preview": r.text[:200] + "..." if len(r.text) > 200 else r.text,
        }
        for rank, r in enumerate(results)
    ]