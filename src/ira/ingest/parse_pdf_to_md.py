# File: src/ira/ingest/parse_pdf_to_md.py

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from llama_parse import LlamaParse

from pdfminer.high_level import extract_text

from ira.ingest.normalize import normalize_markdown


def _pick(meta: Dict[str, Any], paths: List[List[str]]) -> Optional[str]:
    for path in paths:
        cur: Any = meta
        ok = True
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok and isinstance(cur, str) and cur.strip():
            return cur.strip()
    return None


def _find_pdf_file(raw_dir: Path) -> Optional[Path]:
    # Common naming
    for name in ("paper.pdf", "document.pdf", "main.pdf"):
        p = raw_dir / name
        if p.exists():
            return p
    # Fallback: first pdf
    pdfs = sorted(raw_dir.glob("*.pdf"))
    return pdfs[0] if pdfs else None


def _split_pages(text: str) -> List[str]:
    # pdfminer often uses \f as page breaks
    pages = [p.strip("\n") for p in text.split("\f")]
    return [p for p in pages if p and p.strip()]


def _remove_repeated_header_footer_lines(pages: List[str]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Remove short lines that repeat across many pages (likely headers/footers).
    """
    if len(pages) < 2:
        return pages, {"removed_lines": 0, "candidates": 0}

    # Collect per-page unique lines
    per_page_lines: List[List[str]] = []
    freq: Dict[str, int] = {}

    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        uniq = list(dict.fromkeys(lines))  # stable unique
        per_page_lines.append(uniq)
        for ln in uniq:
            if 0 < len(ln) <= 80:
                # Exclude lines that are mostly punctuation
                alnum = sum(ch.isalnum() for ch in ln)
                if alnum >= max(6, int(0.4 * len(ln))):
                    freq[ln] = freq.get(ln, 0) + 1

    # Candidate if appears in >= 60% pages (min 2)
    threshold = max(2, int(0.6 * len(pages)))
    candidates = {ln for ln, c in freq.items() if c >= threshold}

    if not candidates:
        return pages, {"removed_lines": 0, "candidates": 0}

    cleaned_pages: List[str] = []
    removed = 0
    for page in pages:
        out_lines = []
        for ln in page.splitlines():
            s = ln.strip()
            if s in candidates:
                removed += 1
                continue
            out_lines.append(ln)
        cleaned_pages.append("\n".join(out_lines).strip("\n"))

    return cleaned_pages, {"removed_lines": removed, "candidates": len(candidates)}


_HEADING_EXACT = {
    "abstract",
    "introduction",
    "conclusion",
    "references",
    "acknowledgments",
    "acknowledgements",
    "appendix",
}


def _heading_level_from_number_prefix(s: str) -> int:
    """
    Map "1", "2.1", "3.2.1" -> heading levels.
    We keep:
      1      -> ##
      2.1    -> ###
      2.1.3  -> ####
    """
    m = re.match(r"^\s*(\d+(?:\.\d+)*)\s+.+$", s)
    if not m:
        return 2
    depth = m.group(1).count(".") + 1
    return min(2 + (depth - 1), 4)


def _is_heading_line(s: str) -> bool:
    if not s:
        return False
    if len(s) > 120:
        return False

    low = s.strip().lower().rstrip(":")
    if low in _HEADING_EXACT:
        return True

    # Numbered headings: "1 Introduction", "2.1 Method"
    if re.match(r"^\s*\d+(?:\.\d+)*\s+[A-Z].+$", s):
        # avoid sentences
        if not s.strip().endswith("."):
            return True

    # ALL CAPS headings (short-ish)
    alpha = sum(ch.isalpha() for ch in s)
    upper = sum(ch.isupper() for ch in s)
    if alpha >= 6 and upper / max(1, alpha) > 0.85 and len(s.split()) <= 10:
        return True

    return False


def _text_to_markdown(pages: List[str], keep_page_breaks: bool = True) -> str:
    out: List[str] = []

    for pi, page in enumerate(pages):
        lines = page.splitlines()
        for ln in lines:
            s = ln.strip()
            if not s:
                out.append("")
                continue

            if _is_heading_line(s):
                level = _heading_level_from_number_prefix(s)
                out.append("")
                out.append("#" * level + " " + s.strip())
                out.append("")
            else:
                out.append(s)

        if keep_page_breaks and pi < len(pages) - 1:
            out.append("")
            out.append("---")
            out.append("")

    return "\n".join(out)


def parse_pdf_doc(raw_doc_dir: Path, keep_page_breaks: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a PDF snapshot folder (data/raw/<doc_id>/) into Markdown + processed meta.
    Windows-safe default: pdfminer.six.
    """
    meta_path = raw_doc_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {raw_doc_dir}")

    raw_meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))

    doc_id = raw_doc_dir.name
    title = _pick(raw_meta, [["seed", "title"], ["title"]]) or doc_id
    source_url = _pick(
        raw_meta,
        [
            ["source", "pdf_url"],
            ["source", "resolved_url"],
            ["source", "url"],
            ["seed", "url"],
        ],
    )
    version = _pick(raw_meta, [["source", "version"], ["seed", "version"]])

    pdf_path = _find_pdf_file(raw_doc_dir)
    if pdf_path is None:
        raise RuntimeError(f"No PDF found in {raw_doc_dir}")

    # Extract text with pdfminer (layout-aware-ish default)
    extracted = extract_text(str(pdf_path)) or ""
    pages = _split_pages(extracted)

    # Remove repeated headers/footers if possible
    pages_clean, stats = _remove_repeated_header_footer_lines(pages)

    md_body = _text_to_markdown(pages_clean, keep_page_breaks=keep_page_breaks)

    # Canonical header (stable citations)
    header_lines = [f"# {title}"]
    header_lines.append(f"**Doc ID:** {doc_id}")
    if source_url:
        header_lines.append(f"**Source:** {source_url}")
    if version:
        header_lines.append(f"**Version:** {version}")
    header_lines.append("")

    markdown = "\n".join(header_lines) + "\n" + md_body
    markdown = normalize_markdown(markdown)

    # Non-whitespace content measure (useful for choosing candidates later)
    content_non_ws_chars = len(re.sub(r"\s+", "", md_body)) if md_body else 0

    processed_meta: Dict[str, Any] = {
        "doc_id": doc_id,
        "title": title,
        "source_type": (raw_meta.get("kind") or "pdf"),
        "source_url": source_url,
        "version": version,
        "parser": {"name": "parse_pdf_to_md", "version": 1, "method": "pdfminer"},
        "keep_page_breaks": keep_page_breaks,
        "pages": len(pages_clean),
        "boilerplate": stats,
        "content_non_ws_chars": content_non_ws_chars,
        "raw_meta": raw_meta,
        "artifacts": {"pdf": pdf_path.name},
    }

    return markdown, processed_meta

def parse_pdf_doc_v2(raw_doc_dir: Path):
    meta_path = raw_doc_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {raw_doc_dir}")

    raw_meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    doc_id = raw_doc_dir.name
    title = _pick(raw_meta, [["seed", "title"], ["title"]]) or doc_id
    source_url = _pick(
        raw_meta,
        [
            ["source", "pdf_url"],
            ["source", "resolved_url"],
            ["source", "url"],
            ["seed", "url"],
        ],
    )
    version = _pick(raw_meta, [["source", "version"], ["seed", "version"]])

    pdf_path = _find_pdf_file(raw_doc_dir)
    if pdf_path is None:
        raise RuntimeError(f"No PDF found in {raw_doc_dir}")

    parser = LlamaParse(
        api_key="demo",  # Replace with your actual API key, or set as env var
        result_type="markdown"
    )
    # Load the data from your PDF file
    documents = parser.load_data(pdf_path)
    markdown_content = "\n\n".join([doc.text for doc in documents])

    processed_meta: Dict[str, Any] = {
        "doc_id": doc_id,
        "title": title,
        "source_type": (raw_meta.get("kind") or "pdf"),
        "source_url": source_url,
        "version": version,
        "parser": {"name": "parse_pdf_to_md", "version": 1, "method": "llama_parse"},
        "raw_meta": raw_meta,
        "artifacts": {"pdf": pdf_path.name},
    }

    return markdown_content, processed_meta