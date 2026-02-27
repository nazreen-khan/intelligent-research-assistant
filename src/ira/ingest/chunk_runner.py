# File: src/ira/ingest/chunk_runner.py
"""
Day 4 — Chunk runner: processed/ → chunks/

Reads:  data/processed/<doc_id>/content.md  +  meta.json
Writes: data/chunks/<doc_id>/chunks.jsonl

Each line of chunks.jsonl is a ChildChunk record (the retrieval unit).
A companion parents.jsonl is also written for synthesis / context injection.

Usage (CLI):
    uv run python -m ira ingest chunk --processed data/processed --out data/chunks

Usage (Python):
    from ira.ingest.chunk_runner import chunk_all
    results = chunk_all(processed_root=Path("data/processed"), out_root=Path("data/chunks"))
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ira.ingest.chunker import (
    ChildChunk,
    ParentChunk,
    chunk_document,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkResult:
    doc_id: str
    ok: bool
    out_dir: Optional[str] = None
    parent_count: int = 0
    child_count: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _chunk_to_dict(chunk: ChildChunk) -> Dict[str, Any]:
    d = asdict(chunk)
    # char_span is a tuple → convert to list for JSON
    d["char_span"] = list(d["char_span"])
    return d


def _parent_to_dict(parent: ParentChunk) -> Dict[str, Any]:
    d = asdict(parent)
    d["char_span"] = list(d["char_span"])
    return d


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Single-doc pipeline
# ---------------------------------------------------------------------------

def chunk_one_doc(
    processed_doc_dir: Path,
    out_root: Path,
    *,
    force: bool = False,
) -> ChunkResult:
    """
    Chunk a single processed doc folder.

    processed_doc_dir : data/processed/<doc_id>/
    out_root          : data/chunks/
    """
    doc_id = processed_doc_dir.name
    out_dir = out_root / doc_id
    chunks_path = out_dir / "chunks.jsonl"
    parents_path = out_dir / "parents.jsonl"

    # Skip if already chunked
    if not force and chunks_path.exists() and parents_path.exists():
        try:
            existing = chunks_path.read_text(encoding="utf-8").splitlines()
            child_count = sum(1 for ln in existing if ln.strip())
            existing_p = parents_path.read_text(encoding="utf-8").splitlines()
            parent_count = sum(1 for ln in existing_p if ln.strip())
            return ChunkResult(
                doc_id=doc_id, ok=True, out_dir=str(out_dir),
                parent_count=parent_count, child_count=child_count,
            )
        except Exception:
            pass  # re-process on any read error

    content_path = processed_doc_dir / "content.md"
    meta_path = processed_doc_dir / "meta.json"

    if not content_path.exists():
        return ChunkResult(doc_id=doc_id, ok=False, error="missing content.md")
    if not meta_path.exists():
        return ChunkResult(doc_id=doc_id, ok=False, error="missing meta.json")

    try:
        content_md = content_path.read_text(encoding="utf-8")
        meta = _read_json(meta_path)

        title: str = meta.get("title") or doc_id
        url: Optional[str] = meta.get("source_url")

        parents, children = chunk_document(
            doc_id=doc_id,
            title=title,
            url=url,
            content_md=content_md,
        )

        if not children:
            logger.warning("no children produced for %s", doc_id)

        # Write chunks.jsonl (child chunks — retrieval units)
        child_records = [_chunk_to_dict(c) for c in children]
        _write_jsonl(chunks_path, child_records)

        # Write parents.jsonl (section-level context for synthesis)
        parent_records = [_parent_to_dict(p) for p in parents]
        _write_jsonl(parents_path, parent_records)

        # Write a summary meta.json alongside
        summary = {
            "doc_id": doc_id,
            "title": title,
            "url": url,
            "chunked_at": datetime.now(timezone.utc).isoformat(),
            "parent_count": len(parents),
            "child_count": len(children),
            "content_chars": len(content_md),
        }
        (out_dir / "chunk_meta.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info(
            "chunked doc",
            extra={"doc_id": doc_id, "parents": len(parents), "children": len(children)},
        )

        return ChunkResult(
            doc_id=doc_id,
            ok=True,
            out_dir=str(out_dir),
            parent_count=len(parents),
            child_count=len(children),
        )

    except Exception as exc:
        logger.exception("failed to chunk doc %s", doc_id)
        return ChunkResult(doc_id=doc_id, ok=False, error=str(exc))


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def chunk_all(
    processed_root: Path,
    out_root: Path,
    *,
    force: bool = False,
    limit: Optional[int] = None,
) -> List[ChunkResult]:
    """
    Chunk all processed docs under *processed_root*.

    Continues on per-doc errors and returns a full report list.
    """
    results: List[ChunkResult] = []
    count = 0

    for doc_dir in sorted(processed_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        if limit is not None and count >= limit:
            break

        result = chunk_one_doc(doc_dir, out_root, force=force)
        results.append(result)
        count += 1

        status = "ok" if result.ok else f"FAIL: {result.error}"
        logger.info(
            "chunk_all progress",
            extra={
                "doc_id": result.doc_id,
                "status": status,
                "parents": result.parent_count,
                "children": result.child_count,
            },
        )

    return results