# File: src/ira/ingest/processor.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ira.ingest.parse_code_snippets import parse_github_doc
from ira.ingest.parse_html_to_md import parse_html_doc
from ira.ingest.parse_pdf_to_md import parse_pdf_doc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessResult:
    doc_id: str
    kind: str
    ok: bool
    out_dir: Optional[str] = None
    error: Optional[str] = None
    content_chars: Optional[int] = None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _detect_kind(raw_doc_dir: Path, meta: Dict[str, Any]) -> str:
    """
    Detect type using meta['kind'] if present, else infer from artifacts.
    Returns one of: 'docs', 'github', 'pdf'
    """
    k = (meta.get("kind") or meta.get("source_type") or "").strip().lower()
    if k in {"docs", "html"}:
        return "docs"
    if k in {"github", "code"}:
        return "github"
    if k in {"arxiv", "pdf", "paper"}:
        return "pdf"

    # Infer from files
    if any(raw_doc_dir.glob("*.pdf")) or (raw_doc_dir / "paper.pdf").exists():
        return "pdf"
    if any(raw_doc_dir.glob("*.html")) or (raw_doc_dir / "page.html").exists():
        return "docs"

    return "github"


def process_one_doc(
    raw_doc_dir: Path,
    out_root: Path,
    *,
    force: bool = False,
    keep_pdf_page_breaks: bool = True,
) -> ProcessResult:
    """
    Process a single raw doc folder:
      data/raw/<doc_id> -> data/processed/<doc_id>/{content.md, meta.json}
    """
    doc_id = raw_doc_dir.name
    meta_path = raw_doc_dir / "meta.json"
    if not meta_path.exists():
        return ProcessResult(doc_id=doc_id, kind="unknown", ok=False, error="missing meta.json")

    out_dir = out_root / doc_id
    out_md = out_dir / "content.md"
    out_meta = out_dir / "meta.json"

    if not force and out_md.exists() and out_meta.exists():
        # Already processed
        try:
            m = _read_json(out_meta)
            content_chars = m.get("content_chars")
        except Exception:
            content_chars = None
        return ProcessResult(doc_id=doc_id, kind="cached", ok=True, out_dir=str(out_dir), content_chars=content_chars)

    try:
        raw_meta = _read_json(meta_path)
        kind = _detect_kind(raw_doc_dir, raw_meta)

        if kind == "pdf":
            content_md, processed_meta = parse_pdf_doc(raw_doc_dir, keep_page_breaks=keep_pdf_page_breaks)
        elif kind == "docs":
            content_md, processed_meta = parse_html_doc(raw_doc_dir)
        else:
            content_md, processed_meta = parse_github_doc(raw_doc_dir)

        # Add processing envelope
        processed_meta = dict(processed_meta)  # copy
        processed_meta["doc_id"] = processed_meta.get("doc_id") or doc_id
        processed_meta["processed_at"] = datetime.now(timezone.utc).isoformat()
        processed_meta["raw_dir"] = str(raw_doc_dir.as_posix())
        processed_meta["content_chars"] = len(content_md)

        _write_text_atomic(out_md, content_md)
        _write_json_atomic(out_meta, processed_meta)

        logger.info("processed doc", extra={"doc_id": doc_id, "kind": kind, "out_dir": str(out_dir)})
        return ProcessResult(doc_id=doc_id, kind=kind, ok=True, out_dir=str(out_dir), content_chars=len(content_md))

    except Exception as e:
        logger.exception("failed to process doc", extra={"doc_id": doc_id})
        return ProcessResult(doc_id=doc_id, kind="unknown", ok=False, error=str(e))


def iter_raw_doc_dirs(raw_root: Path) -> Iterable[Path]:
    for p in sorted(raw_root.iterdir()):
        if p.is_dir():
            yield p


def process_all(
    raw_root: Path,
    out_root: Path,
    *,
    force: bool = False,
    limit: Optional[int] = None,
    only_kind: Optional[str] = None,
    keep_pdf_page_breaks: bool = True,
) -> List[ProcessResult]:
    """
    Process all docs under raw_root. Continues on errors and returns a report list.
    """
    results: List[ProcessResult] = []
    n = 0

    for d in iter_raw_doc_dirs(raw_root):
        if limit is not None and n >= limit:
            break

        # Optional pre-filter by kind (cheap detect)
        if only_kind:
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    raw_meta = _read_json(meta_path)
                    k = _detect_kind(d, raw_meta)
                    if k != only_kind:
                        continue
                except Exception:
                    continue
            else:
                continue

        r = process_one_doc(d, out_root, force=force, keep_pdf_page_breaks=keep_pdf_page_breaks)
        results.append(r)
        n += 1

    return results
