from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify as md

from ira.ingest.normalize import normalize_markdown


def _pick(meta: Dict[str, Any], paths: list[list[str]]) -> Optional[str]:
    """
    Pick the first non-empty string value from meta using dotted paths.
    Example paths: [["seed", "title"], ["source", "url"]]
    """
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


def _find_html_file(raw_dir: Path) -> Optional[Path]:
    if (raw_dir / "page.html").exists():
        return raw_dir / "page.html"
    htmls = sorted(raw_dir.glob("*.html"))
    return htmls[0] if htmls else None


def _find_md_file(raw_dir: Path) -> Optional[Path]:
    if (raw_dir / "page.md").exists():
        return raw_dir / "page.md"
    mds = sorted(raw_dir.glob("*.md"))
    # avoid meta-like markdown files if any; keep first reasonable
    for p in mds:
        if p.name.lower() not in {"readme.md"}:
            return p
    return mds[0] if mds else None


def _strip_unwanted(soup: BeautifulSoup) -> None:
    # Remove noise
    for tag_name in ("script", "style", "nav", "footer", "header", "aside", "form", "button", "svg"):
        for t in soup.find_all(tag_name):
            t.decompose()

    # Remove comments
    for c in soup.find_all(string=lambda s: isinstance(s, type(soup.comment))):
        try:
            c.extract()
        except Exception:
            pass


def _select_main(soup: BeautifulSoup) -> Tag:
    # Prefer semantic main containers
    for selector in ("article", "main"):
        t = soup.find(selector)
        if isinstance(t, Tag):
            return t

    # Common patterns
    t = soup.find(attrs={"role": "main"})
    if isinstance(t, Tag):
        return t

    body = soup.body
    if isinstance(body, Tag):
        return body

    # Fallback: entire document
    return soup


def _detect_code_lang(code_tag: Tag) -> str:
    # Look for common language class patterns: language-python, lang-python, python
    classes = code_tag.get("class") or []
    if isinstance(classes, str):
        classes = [classes]

    for c in classes:
        c = str(c).lower()
        m = re.search(r"(language|lang)[-_]([a-z0-9_+-]+)", c)
        if m:
            return m.group(2)
        # sometimes class is literally "python"
        if c in {"python", "cpp", "c", "javascript", "typescript", "bash", "shell", "json", "yaml"}:
            return c
    return ""


def _replace_pre_with_fenced(main: Tag) -> None:
    """
    Replace <pre><code> blocks with literal fenced markdown text nodes.
    This avoids converters turning code into indented blocks.
    """
    for pre in main.find_all("pre"):
        code = pre.find("code")
        if isinstance(code, Tag):
            lang = _detect_code_lang(code)
            code_text = code.get_text("\n", strip=False)
        else:
            lang = ""
            code_text = pre.get_text("\n", strip=False)

        code_text = code_text.rstrip("\n")
        fence = f"\n\n```{lang}\n{code_text}\n```\n\n"
        pre.replace_with(NavigableString(fence))


def _table_to_md(table: Tag, max_rows: int = 30) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""

    grid: list[list[str]] = []
    for r in rows[:max_rows]:
        cells = r.find_all(["th", "td"])
        row_vals = []
        for c in cells:
            txt = c.get_text(" ", strip=True)
            txt = re.sub(r"\s+", " ", txt).strip()
            row_vals.append(txt)
        if row_vals:
            grid.append(row_vals)

    if not grid:
        return ""

    # Determine column count
    ncols = max(len(r) for r in grid)
    for r in grid:
        if len(r) < ncols:
            r.extend([""] * (ncols - len(r)))

    # Header detection: first row has any <th> OR looks header-like
    first_tr = rows[0]
    has_th = bool(first_tr.find_all("th"))
    header = grid[0] if has_th else [f"col_{i+1}" for i in range(ncols)]
    body = grid[1:] if has_th else grid

    def esc(cell: str) -> str:
        return cell.replace("|", "\\|")

    out = []
    out.append("| " + " | ".join(esc(c) for c in header) + " |")
    out.append("| " + " | ".join(["---"] * ncols) + " |")
    for r in body:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")

    if len(rows) > max_rows:
        out.append("")
        out.append(f"> Note: table truncated to first {max_rows} rows.")

    return "\n".join(out)


def _replace_tables(main: Tag) -> None:
    for table in main.find_all("table"):
        md_table = _table_to_md(table)
        if md_table.strip():
            table.replace_with(NavigableString("\n\n" + md_table + "\n\n"))
        else:
            # Fallback: plain text
            txt = table.get_text("\n", strip=True)
            table.replace_with(NavigableString("\n\n" + txt + "\n\n"))


def parse_html_doc(raw_doc_dir: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a docs HTML snapshot folder (data/raw/<doc_id>/) into Markdown + processed meta.

    Uses content-length based fallback instead of file-size thresholds.
    """
    meta_path = raw_doc_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {raw_doc_dir}")

    raw_meta: Dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))

    doc_id = raw_doc_dir.name
    title = _pick(raw_meta, [["seed", "title"], ["title"]]) or doc_id
    source_url = _pick(raw_meta, [["source", "resolved_url"], ["source", "url"], ["seed", "url"]])
    version = _pick(raw_meta, [["source", "version"], ["seed", "version"]])

    html_path = _find_html_file(raw_doc_dir)
    md_path = _find_md_file(raw_doc_dir)

    def non_ws_len(s: Optional[str]) -> int:
        if not s:
            return 0
        return len(re.sub(r"\s+", "", s))

    markdown_from_html: Optional[str] = None
    markdown_from_md: Optional[str] = None

    # 1) Try HTML if present (regardless of file size)
    if html_path is not None and html_path.exists():
        try:
            html = html_path.read_text(encoding="utf-8", errors="replace")
            soup = BeautifulSoup(html, "html.parser")
            _strip_unwanted(soup)
            main = _select_main(soup)

            # Improve fidelity for code + tables before markdown conversion
            _replace_pre_with_fenced(main)
            _replace_tables(main)

            html_clean = str(main)
            markdown_from_html = md(
                html_clean,
                heading_style="ATX",
                bullets="-",
            )
        except Exception:
            markdown_from_html = None

    # 2) Try existing markdown snapshot if present
    if md_path is not None and md_path.exists():
        markdown_from_md = md_path.read_text(encoding="utf-8", errors="replace")

    # 3) Choose best candidate (prefer the one with more extracted content)
    used_fallback_md = False
    if markdown_from_html and markdown_from_md:
        # Prefer the richer one. If HTML is too thin, use MD.
        if non_ws_len(markdown_from_html) >= max(200, int(0.8 * non_ws_len(markdown_from_md))):
            markdown_body = markdown_from_html
        else:
            markdown_body = markdown_from_md
            used_fallback_md = True
    elif markdown_from_html:
        markdown_body = markdown_from_html
    elif markdown_from_md:
        markdown_body = markdown_from_md
        used_fallback_md = True
    else:
        raise RuntimeError(f"No usable page.html or page.md found in {raw_doc_dir}")

    # Build canonical header (stable citations)
    header_lines = [f"# {title}"]
    if source_url:
        header_lines.append(f"**Source:** {source_url}")
    if version:
        header_lines.append(f"**Version:** {version}")
    header_lines.append("")  # spacer

    markdown = "\n".join(header_lines) + "\n" + markdown_body
    markdown = normalize_markdown(markdown)

    processed_meta: Dict[str, Any] = {
        "doc_id": doc_id,
        "title": title,
        "source_type": "docs",
        "source_url": source_url,
        "version": version,
        "parser": {"name": "parse_html_to_md", "version": 2},
        "used_fallback_md": used_fallback_md,
        "content_non_ws_chars": non_ws_len(markdown_body),
        "raw_meta": raw_meta,
        "artifacts": {
            "page_html": html_path.name if html_path else None,
            "page_md": md_path.name if md_path else None,
        },
    }

    return markdown, processed_meta
