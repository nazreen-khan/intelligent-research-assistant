# File: src/ira/ingest/chunker.py
"""
Day 4 — Hybrid-semantic chunker for technical Markdown documents.

Real-data fixes applied (from data inspection):
  FIX 1 — Flat-H1 arXiv PDFs: LlamaParse emits ALL headings as `#` (h1).
           Numbered headings like "2.1 Hardware" are detected and assigned
           logical depth so the section stack nests them correctly.

  FIX 2 — Docs HTML anchor noise: markdownify preserves Sphinx/MkDocs anchors
           as `## Heading[#](#anchor "Link to this heading")`.
           Stripped from heading text before use.

  FIX 3 — Code-block comment lines: GitHub README files contain lines like
           `# JIT cache comment` inside fenced bash blocks.
           Heading detection now skips content inside fenced code blocks.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODING_NAME = "cl100k_base"
CHILD_MAX_TOKENS: int = 400
CHILD_OVERLAP_TOKENS: int = 50
PARENT_MAX_TOKENS: int = 1500
SCHEMA_VERSION: int = 1

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
# Prefix must start with a digit (1, 2.1, 3.2.1) or be a letter WITH dots (A.1, B.2)
# Plain words like GPU, BERT, Abstract are NOT section numbers
_NUMBERED_SECTION_RE = re.compile(r"^([0-9][A-Z0-9]*(?:\.[A-Z0-9]+)*|[A-Z](?:\.[A-Z0-9]+)+)\s+\S")
_ANCHOR_NOISE_RE = re.compile(r'\s*\[(?:#|¶)\]\([^)]*\)(?:\s*"[^"]*")?')
_FENCE_RE = re.compile(r"^(`{3,}|~{3,})")
_TABLE_ROW_RE = re.compile(r"^\s*\|")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ParentChunk:
    parent_id: str
    doc_id: str
    title: str
    url: Optional[str]
    section: str
    heading_level: int
    text: str
    char_span: Tuple[int, int]
    token_count: int
    schema_version: int = SCHEMA_VERSION


@dataclass
class ChildChunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    title: str
    url: Optional[str]
    section: str
    text: str
    char_span: Tuple[int, int]
    token_count: int
    chunk_index: int
    parent_chunk_index: int
    is_table: bool = False
    is_code: bool = False
    schema_version: int = SCHEMA_VERSION


@dataclass
class _Block:
    kind: str   # "paragraph" | "table" | "code"
    text: str
    char_start: int
    char_end: int


@dataclass
class _Section:
    heading: str
    level: int
    section_path: str
    body: str
    char_start: int
    char_end: int


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def count_tokens(text: str, encoder: Optional[tiktoken.Encoding] = None) -> int:
    enc = encoder or _get_encoder()
    return len(enc.encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# FIX 2 — Strip anchor noise from heading text
# ---------------------------------------------------------------------------

def _clean_heading_text(raw: str) -> str:
    """
    Remove docs anchor patterns from heading text.
    '## Intro[#](#intro "Link to this heading")' → 'Intro'
    """
    return _ANCHOR_NOISE_RE.sub("", raw).strip()


# ---------------------------------------------------------------------------
# FIX 3 — Mask fenced code before heading scan
# ---------------------------------------------------------------------------

def _mask_fenced_code(content: str) -> str:
    """
    Replace interior lines of fenced code blocks with spaces of equal length.
    Fence open/close lines are kept. This stops '# comment' inside bash blocks
    from being detected as Markdown headings.
    """
    lines = content.split("\n")
    out: List[str] = []
    in_fence = False
    fence_marker = ""

    for line in lines:
        stripped = line.strip()
        m = _FENCE_RE.match(stripped)
        if m and not in_fence:
            in_fence = True
            fence_marker = m.group(1)[0] * 3
            out.append(line)
        elif in_fence:
            if stripped.startswith(fence_marker):
                in_fence = False
                out.append(line)
            else:
                out.append(" " * len(line))   # blank out interior, preserve length
        else:
            out.append(line)

    return "\n".join(out)


# ---------------------------------------------------------------------------
# FIX 1 — Infer logical depth for flat-H1 numbered headings
# ---------------------------------------------------------------------------

def _infer_logical_level(heading_text: str) -> int:
    """
    Recover logical depth from numbering prefix when all headings are flat H1.
      "1 Introduction"  → 1
      "2.1 Hardware"    → 2
      "3.2.1 Detail"    → 3
      "A Related Work"  → 1
      "B.1 Sub"         → 2
      "Abstract"        → 1  (no number)
    """
    m = _NUMBERED_SECTION_RE.match(heading_text.strip())
    if not m:
        return 1
    dot_count = m.group(1).count(".")
    return min(dot_count + 1, 4)


def _build_section_path(stack: List[Tuple[int, str]]) -> str:
    return " > ".join(text for _, text in stack)


# ---------------------------------------------------------------------------
# Heading finder — fence-aware + anchor-clean
# ---------------------------------------------------------------------------

def _find_headings(content: str) -> List[Tuple[int, int, int, str]]:
    """
    Return (match_start, match_end, logical_level, clean_heading_text)
    for every Markdown heading, skipping lines inside fenced code blocks.

    For flat-H1 documents (arXiv PDFs from LlamaParse):
      - Numbered headings get logical depth from prefix: '2.1' -> level 2
      - Unnumbered headings (e.g. 'GPU Memory Hierarchy') between numbered
        siblings inherit the level of the most recent numbered heading.
        Without this, 'GPU Memory Hierarchy' (level 1) would incorrectly
        become the parent of the following '2.2 ...' section.
    """
    masked = _mask_fenced_code(content)
    raw_headings = []

    for m in _HEADING_RE.finditer(masked):
        raw_level = len(m.group(1))
        clean_text = _clean_heading_text(m.group(2))
        if not clean_text:
            continue
        raw_headings.append((m.start(), m.end(), raw_level, clean_text))

    # Only apply flat-H1 logic when doc uses exclusively H1 headings
    levels = {h[2] for h in raw_headings}
    all_flat_h1 = (levels == {1}) and len(raw_headings) > 3

    if not all_flat_h1:
        return [(h_start, h_end, raw_level, clean_text)
                for h_start, h_end, raw_level, clean_text in raw_headings]

    # First pass: assign numbered levels, mark unnumbered as placeholder
    provisional = []  # (h_start, h_end, level, clean_text, is_numbered)
    for h_start, h_end, _, clean_text in raw_headings:
        m = _NUMBERED_SECTION_RE.match(clean_text.strip())
        if m:
            dot_count = m.group(1).count('.')
            logical = min(dot_count + 1, 4)
            provisional.append((h_start, h_end, logical, clean_text, True))
        else:
            provisional.append((h_start, h_end, 1, clean_text, False))

    # Second pass: unnumbered headings inherit the last numbered heading's level
    # so they sit as siblings, not parents, of the next numbered section.
    last_numbered_level = 1
    result = []
    for h_start, h_end, level, clean_text, is_numbered in provisional:
        if is_numbered:
            last_numbered_level = level
            result.append((h_start, h_end, level, clean_text))
        else:
            result.append((h_start, h_end, last_numbered_level, clean_text))

    return result


# ---------------------------------------------------------------------------
# Section parser
# ---------------------------------------------------------------------------

def parse_markdown_sections(content: str) -> List[_Section]:
    """
    Split *content* into sections, one per heading.
    Handles flat-H1 PDFs, anchor-noisy docs, and code-comment false positives.
    """
    sections: List[_Section] = []
    headings = _find_headings(content)

    # Preamble before first heading
    preamble_end = headings[0][0] if headings else len(content)
    preamble_body = content[:preamble_end].strip()
    if preamble_body:
        sections.append(_Section(
            heading="Document",
            level=0,
            section_path="Document",
            body=preamble_body,
            char_start=0,
            char_end=preamble_end,
        ))

    heading_stack: List[Tuple[int, str]] = []

    for i, (h_start, h_end, level, h_text) in enumerate(headings):
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, h_text))

        body_start = h_end
        body_end = headings[i + 1][0] if i + 1 < len(headings) else len(content)
        body = content[body_start:body_end].strip()

        sections.append(_Section(
            heading=h_text,
            level=level,
            section_path=_build_section_path(heading_stack),
            body=body,
            char_start=h_start,
            char_end=body_end,
        ))

    return sections


# ---------------------------------------------------------------------------
# Block iterator — atomic units within a section body
# ---------------------------------------------------------------------------

def _iter_blocks(body: str, body_offset: int = 0) -> Generator[_Block, None, None]:
    """
    Yield _Block items (code | table | paragraph) from a section body.
    Code and table blocks are yielded as single atomic units.
    """
    lines = body.split("\n")
    buf: List[str] = []
    buf_start: int = 0
    cursor: int = 0
    in_fence = False
    fence_marker = ""

    def flush_paragraph() -> Optional[_Block]:
        nonlocal buf, buf_start
        text = "\n".join(buf).strip()
        start = buf_start
        buf = []
        if text:
            return _Block("paragraph", text, body_offset + start, body_offset + cursor)
        return None

    i = 0
    while i < len(lines):
        line = lines[i]
        line_len = len(line) + 1
        stripped = line.strip()

        # Fenced code block
        m = _FENCE_RE.match(stripped)
        if m and not in_fence:
            blk = flush_paragraph()
            if blk:
                yield blk
            in_fence = True
            fence_marker = m.group(1)[0] * 3
            fence_lines = [line]
            fence_start = cursor
            i += 1
            cursor += line_len
            while i < len(lines):
                fl = lines[i]
                fence_lines.append(fl)
                cursor += len(fl) + 1
                if fl.strip().startswith(fence_marker):
                    in_fence = False
                    i += 1
                    break
                i += 1
            yield _Block("code", "\n".join(fence_lines), body_offset + fence_start, body_offset + cursor)
            buf_start = cursor
            continue

        # Table block
        if _TABLE_ROW_RE.match(line):
            blk = flush_paragraph()
            if blk:
                yield blk
            table_lines = []
            table_start = cursor
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                cursor += len(lines[i]) + 1
                i += 1
            yield _Block("table", "\n".join(table_lines), body_offset + table_start, body_offset + cursor)
            buf_start = cursor
            continue

        # Normal paragraph line
        if not buf:
            buf_start = cursor
        buf.append(line)
        cursor += line_len
        i += 1

    blk = flush_paragraph()
    if blk:
        yield blk


# ---------------------------------------------------------------------------
# Sliding-window token splitter
# ---------------------------------------------------------------------------

def _split_paragraph_by_tokens(
    text: str,
    char_start: int,
    max_tokens: int,
    overlap_tokens: int,
    encoder: tiktoken.Encoding,
) -> List[Tuple[str, int, int]]:
    tokens = encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return [(text, char_start, char_start + len(text))]

    results: List[Tuple[str, int, int]] = []
    step = max(1, max_tokens - overlap_tokens)
    pos = 0
    while pos < len(tokens):
        window = tokens[pos: pos + max_tokens]
        sub_text = encoder.decode(window)
        results.append((sub_text, char_start, char_start + len(sub_text)))
        if pos + max_tokens >= len(tokens):
            break
        pos += step
    return results


# ---------------------------------------------------------------------------
# Section → ChildChunks
# ---------------------------------------------------------------------------

def split_section_into_children(
    section: _Section,
    doc_id: str,
    title: str,
    url: Optional[str],
    parent_id: str,
    parent_chunk_index: int,
    encoder: tiktoken.Encoding,
    max_tokens: int = CHILD_MAX_TOKENS,
    overlap_tokens: int = CHILD_OVERLAP_TOKENS,
) -> List[ChildChunk]:
    if not section.body.strip():
        return []

    children: List[ChildChunk] = []
    chunk_index = 0

    if section.level > 0:
        heading_line = "#" * section.level + " " + section.heading
        body_offset = section.char_start + len(heading_line) + 1
    else:
        body_offset = 0

    for block in _iter_blocks(section.body, body_offset=body_offset):
        if block.kind in ("code", "table"):
            children.append(ChildChunk(
                chunk_id=str(uuid.uuid4()),
                parent_id=parent_id,
                doc_id=doc_id,
                title=title,
                url=url,
                section=section.section_path,
                text=block.text,
                char_span=(block.char_start, block.char_end),
                token_count=count_tokens(block.text, encoder),
                chunk_index=chunk_index,
                parent_chunk_index=parent_chunk_index,
                is_table=(block.kind == "table"),
                is_code=(block.kind == "code"),
            ))
            chunk_index += 1
        else:
            for sub_text, cs, ce in _split_paragraph_by_tokens(
                block.text, block.char_start, max_tokens, overlap_tokens, encoder
            ):
                if not sub_text.strip():
                    continue
                children.append(ChildChunk(
                    chunk_id=str(uuid.uuid4()),
                    parent_id=parent_id,
                    doc_id=doc_id,
                    title=title,
                    url=url,
                    section=section.section_path,
                    text=sub_text,
                    char_span=(cs, ce),
                    token_count=count_tokens(sub_text, encoder),
                    chunk_index=chunk_index,
                    parent_chunk_index=parent_chunk_index,
                ))
                chunk_index += 1

    return children


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def chunk_document(
    doc_id: str,
    title: str,
    url: Optional[str],
    content_md: str,
    max_child_tokens: int = CHILD_MAX_TOKENS,
    overlap_tokens: int = CHILD_OVERLAP_TOKENS,
    max_parent_tokens: int = PARENT_MAX_TOKENS,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """
    Main entry point. Returns (parents, children) from a processed content.md.
    """
    encoder = _get_encoder()
    sections = parse_markdown_sections(content_md)
    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    for parent_idx, section in enumerate(sections):
        if not section.body.strip():
            continue

        parent_id = str(uuid.uuid4())
        enc_tokens = encoder.encode(section.body, disallowed_special=())
        parent_text = (
            encoder.decode(enc_tokens[:max_parent_tokens])
            if len(enc_tokens) > max_parent_tokens
            else section.body
        )

        parents.append(ParentChunk(
            parent_id=parent_id,
            doc_id=doc_id,
            title=title,
            url=url,
            section=section.section_path,
            heading_level=section.level,
            text=parent_text,
            char_span=(section.char_start, section.char_end),
            token_count=count_tokens(parent_text, encoder),
        ))

        children.extend(split_section_into_children(
            section=section,
            doc_id=doc_id,
            title=title,
            url=url,
            parent_id=parent_id,
            parent_chunk_index=parent_idx,
            encoder=encoder,
            max_tokens=max_child_tokens,
            overlap_tokens=overlap_tokens,
        ))

    return parents, children